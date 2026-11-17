"""Bounded CPU preparation of visually selected, continuous narrative streams.

JPEGs keep the original decoder PTS; a select filter samples original frames
without an fps filter or synthesized timestamps. Preparation produces no QA,
negative labels or video-quality judgments. Source media are never rewritten.
"""

from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import time

from PIL import Image

from .io import code_revision, sha256_file, write_json
from .narrative_visual import probe_source


PROTOCOL = "narrative-dense-stream-preparation-v1"
POLICY = "overlap_allowed_by_user"
CLOCK_DEFINITION = "source_seconds = original_pts * time_base - origin_pts_seconds"


def _number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be an explicit finite number")
    return float(value)


def _digest(value, name):
    if not isinstance(value, str) or not re.fullmatch(r"[a-fA-F0-9]{64}", value):
        raise ValueError(f"{name} must be an explicit SHA256")
    return value.lower()


def _text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be explicit text")
    return value


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def selected_streams(manifest, *, raw_manifest=None):
    """Read selected visual stream records and verify grouping, clocks and bytes.

    Supports nested ``source``/``interval``/``probe`` visual-screening records or
    explicit flat source fields plus stream_start/stream_end/source_clock. The
    caller must record its decision to proceed to dense preparation; a screening
    recommendation alone is never interpreted as acceptance.
    """
    manifest = Path(manifest).resolve(strict=True)
    raw_manifest = manifest.read_bytes() if raw_manifest is None else raw_manifest
    rows = [json.loads(line) for line in raw_manifest.splitlines() if line.strip()]
    if not rows:
        raise ValueError("Selected stream manifest is empty")
    result, groups, hashes, identities, intervals, ids = [], {}, {}, {}, {}, set()
    for row in rows:
        if not isinstance(row, dict) or row.get("accepted_for_dense_preparation") is not True:
            raise ValueError("Explicit accepted_for_dense_preparation=true is required")
        visual_hash = _digest(row.get("visual_result_sha256"), "visual_result_sha256")
        source = row.get("source", row)
        if not isinstance(source, dict):
            raise ValueError("Invalid source record")
        sid = _text(source.get("source_id"), "source_id")
        group = _text(source.get("source_group"), "source_group")
        split = source.get("split", row.get("split"))
        if split not in {"train", "validation"}:
            raise ValueError("An explicit frozen train/validation split is required")
        if row.get("split", split) != split or row.get("source_group", group) != group:
            raise ValueError("Conflicting nested and top-level source group/split")
        if groups.setdefault(group, split) != split:
            raise ValueError("A source group crosses the frozen split")
        if source.get("benchmark_source_policy", row.get("benchmark_source_policy")) != POLICY:
            raise ValueError("Preserve the user's benchmark overlap authorization")
        expected = _digest(source.get("source_sha256", source.get("sha256")), "source_sha256")
        for value in [source.get("sha256"), row.get("source_sha256")]:
            if value is not None and _digest(value, "source_sha256") != expected:
                raise ValueError("Conflicting source hashes")
        path = Path(_text(source.get("local_path"), "local_path"))
        path = (manifest.parent / path if not path.is_absolute() else path).resolve(strict=True)
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"Raw source hash mismatch: {sid}")
        if hashes.setdefault(expected, group) != group:
            raise ValueError("Identical raw source assigned different source groups")
        if identities.setdefault(sid, (group, expected)) != (group, expected):
            raise ValueError("Source identity maps to conflicting group/content")
        interval = row.get("interval", {"start": row.get("stream_start"), "end": row.get("stream_end")})
        start, end = _number(interval.get("start"), "stream start"), _number(interval.get("end"), "stream end")
        duration = end - start
        if start < 0 or not 600 <= duration <= 1200 or abs(duration - round(duration)) > 1e-7:
            raise ValueError("Continuous streams must contain 600–1200 complete one-second bins")
        for previous in intervals.setdefault(expected, []):
            if max(start, previous[0]) < min(end, previous[1]):
                raise ValueError("Selected intervals overlap in the same raw source")
        intervals[expected].append((start, end))
        probe = row.get("probe", {})
        clock = row.get("source_clock", source.get("source_clock"))
        if clock is None:
            clock = {"origin_pts_seconds": probe.get("origin_pts_seconds")}
        if not isinstance(clock, dict):
            raise ValueError("Explicit source_clock origin is required")
        origin = _number(clock.get("origin_pts_seconds"), "source_clock origin_pts_seconds")
        if "origin_pts_seconds" in probe and abs(origin - _number(probe["origin_pts_seconds"], "probe origin")) > 1e-7:
            raise ValueError("Source clock disagrees with frozen visual probe")
        overlap = source.get("benchmark_overlap", {"status": "unknown"})
        if not isinstance(overlap, dict) or overlap.get("status") not in {"unknown", "known_shared", "known_disjoint"}:
            raise ValueError("Invalid benchmark overlap provenance")
        if row.get("visual_result_path"):
            visual_path = Path(row["visual_result_path"])
            visual_path = (manifest.parent / visual_path if not visual_path.is_absolute() else visual_path).resolve(strict=True)
            if sha256_file(visual_path) != visual_hash:
                raise ValueError("Visual selection artifact hash changed")
        stream_id = row.get("stream_id") or _hash([sid, group, expected, start, end])[:20]
        if not re.fullmatch(r"[a-f0-9]{20}", str(stream_id)) or stream_id in ids:
            raise ValueError("Stream IDs must be unique, safe SHA256 prefixes")
        ids.add(stream_id)
        result.append({"stream_id": stream_id, "source_id": sid, "source_group": group, "split": split,
                       "local_path": str(path), "source_sha256": expected, "sha256": expected,
                       "interval": {"start": start, "end": end},
                       "source_clock": {"origin_pts_seconds": origin, "definition": CLOCK_DEFINITION},
                       "benchmark_source_policy": POLICY, "benchmark_overlap": overlap,
                       "visual_result_sha256": visual_hash, "input_stream": row})
    return sorted(result, key=lambda row: (row["source_group"], row["source_id"], row["interval"]["start"]))


def measured_frames(stderr, paths, *, origin, start, end, long_side):
    """Match each emitted JPEG to showinfo's preserved integer decoder PTS."""
    base = re.findall(r"config in time_base:\s*(\d+/\d+)", stderr)
    stamps = re.findall(r"\bn:\s*(\d+)\s+pts:\s*(-?\d+)\s+pts_time:", stderr)
    expected = round(end - start)
    if len(base) != 1 or len(paths) != expected or len(stamps) != expected:
        raise ValueError("JPEG/showinfo count does not cover every dense one-second bin")
    time_base = Fraction(base[0])
    if time_base <= 0:
        raise ValueError("Decoder time base must be positive")
    origin_fraction, start_fraction = Fraction(str(origin)), Fraction(str(start))
    frames, previous = [], None
    for index, (path, stamp) in enumerate(zip(paths, stamps)):
        n, pts = map(int, stamp)
        seconds = pts * time_base - origin_fraction
        offset = seconds - start_fraction
        if n != index or not start_fraction + index <= seconds < start_fraction + index + 1:
            raise ValueError("Measured PTS is outside its actual one-second source bin")
        if seconds >= Fraction(str(end)) or previous is not None and seconds <= previous:
            raise ValueError("Measured PTS is future or nonmonotonic")
        previous = seconds
        with Image.open(path) as image:
            width, height = image.size
            if image.format != "JPEG" or min(width, height) < 1 or max(width, height) > long_side:
                raise ValueError("Invalid bounded JPEG output")
            image.verify()
        frames.append({"index": index, "path": str(Path(path).resolve()), "timestamp": float(seconds),
                       "relative_time": float(offset), "requested_timestamp": start + index,
                       "original_pts": pts, "time_base": base[0], "source_time_fraction": str(seconds),
                       "sample_bin": index, "sha256": sha256_file(path), "width": width, "height": height,
                       "timestamp_validation": "original_copyts_select_showinfo_pts"})
    return frames


def choose_dense_seek(source, ffmpeg, *, start, origin, timeout=120):
    """Prove first-bin coverage before a full decode, including unindexed TS.

    MPEG-TS input seeking can land after the preceding random-access frame, so
    the decoder may emit only a later I-frame. Earlier seeks plus an original-PTS
    select keep the clock and sample guard intact. A source-start fallback is
    bounded by the same CPU/thread and timeout constraints, not assumed accurate.
    """
    start, origin = _number(start, "start"), _number(origin, "origin")
    if start < 0 or type(timeout) is not int or not 1 <= timeout <= 7200:
        raise ValueError("Invalid bounded seek probe settings")
    candidates = list(dict.fromkeys([max(0.0, start-10), max(0.0, start-60), 0.0]))
    left, right = format(start+origin, ".17g"), format(start+origin+1, ".17g")
    environment = {**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    attempts = []
    for seek in candidates:
        seek_options = ["-ss", f"{seek:.9f}"] if seek else []
        command = [str(ffmpeg), "-hide_banner", "-nostdin", "-loglevel", "info", "-threads", "1",
                   "-filter_threads", "1", "-filter_complex_threads", "1", *seek_options,
                   "-t", f"{start-seek+2:.9f}", "-copyts", "-hwaccel", "none", "-i", source["local_path"],
                   "-map", "0:v:0", "-an", "-sn", "-dn", "-vf", f"select='gte(t,{left})*lt(t,{right})',showinfo",
                   "-frames:v", "1", "-fps_mode", "passthrough", "-threads", "1", "-f", "null", "-"]
        attempt = {"seek_source_seconds": seek, "command": command}
        began = time.monotonic()
        try:
            result = subprocess.run(command, check=True, capture_output=True, text=True,
                                    timeout=min(timeout, 120), env=environment)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
            attempt.update(status="probe_failed", error_type=type(error).__name__, elapsed_seconds=time.monotonic()-began)
            attempts.append(attempt)
            continue
        attempt.update(stderr_sha256=hashlib.sha256(result.stderr.encode()).hexdigest(), elapsed_seconds=time.monotonic()-began)
        base = re.search(r"config in time_base:\s*(\d+/\d+)", result.stderr)
        pts = re.search(r"\bn:\s*0\s+pts:\s*(-?\d+)\s+pts_time:", result.stderr)
        if base and pts and Fraction(base[1]) > 0:
            actual = int(pts[1])*Fraction(base[1])-Fraction(str(origin))
            attempt.update(original_pts=int(pts[1]), time_base=base[1], timestamp=float(actual))
            if Fraction(str(start)) <= actual < Fraction(str(start))+1:
                attempt["status"] = "first_bin_measured"
                attempts.append(attempt)
                return seek, attempts
        attempt["status"] = "first_bin_not_observed"
        attempts.append(attempt)
    raise ValueError("No seek established measured first-bin coverage; source clock and guard were not relaxed")


def extract_dense_frames(source, output, ffmpeg, *, start, end, origin, long_side=640, timeout=3600):
    """One CPU process/one thread; select real frames, never relabel fps output."""
    if type(long_side) is not int or not 64 <= long_side <= 640:
        raise ValueError("Dense JPEG long_side must be 64–640")
    start, end, origin = (_number(value, name) for value, name in [(start, "start"), (end, "end"), (origin, "origin")])
    if start < 0 or not 1 <= end - start <= 1200 or abs(end - start - round(end - start)) > 1e-7:
        raise ValueError("Decode interval must contain complete one-second bins")
    if type(timeout) is not int or not 1 <= timeout <= 7200:
        raise ValueError("Decode timeout must be 1–7200 seconds")
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Preserve frozen decoded frames; use an empty directory")
    output.mkdir(parents=True, exist_ok=True)
    seek, attempts = choose_dense_seek(source, ffmpeg, start=start, origin=origin, timeout=timeout)
    write_json(output / "seek_validation.json", {"start": start, "origin_pts_seconds": origin,
        "selected_seek_source_seconds": seek, "attempts": attempts,
        "guard": "original source PTS in [start,start+1); no clock reset or timestamp relabeling"})
    left, right = format(start + origin, ".17g"), format(end + origin, ".17g")
    select = (f"select='gte(t,{left})*lt(t,{right})*"
              f"(isnan(prev_selected_t)+gt(floor(t-({left})),floor(prev_selected_t-({left}))))'")
    filters = (select + ",scale=w=trunc(iw*sar):h=ih,setsar=1,"
               f"scale=w='min({long_side},iw)':h='min({long_side},ih)':force_original_aspect_ratio=decrease,showinfo")
    seek_options = ["-ss", f"{seek:.9f}"] if seek else []
    command = [str(ffmpeg), "-hide_banner", "-nostdin", "-loglevel", "info", "-threads", "1",
               "-filter_threads", "1", "-filter_complex_threads", "1", *seek_options,
               "-t", f"{end-seek+2:.9f}", "-copyts", "-hwaccel", "none", "-i", source["local_path"], "-map", "0:v:0",
               "-an", "-sn", "-dn", "-vf", filters, "-fps_mode", "passthrough", "-threads", "1",
               "-q:v", "3", "-start_number", "0", "-n", str(output / "frame_%06d.jpg")]
    environment = {**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    result = subprocess.run(command, check=True, capture_output=True, text=True, timeout=timeout, env=environment)
    (output / "ffmpeg_showinfo.log").write_text(result.stderr)
    paths = sorted(output.glob("frame_*.jpg"))
    frames = measured_frames(result.stderr, paths, origin=origin, start=start, end=end, long_side=long_side)
    if frames[0]["original_pts"]*Fraction(frames[0]["time_base"]) != attempts[-1]["original_pts"]*Fraction(attempts[-1]["time_base"]):
        raise ValueError("Full decode disagrees with measured first-bin seek probe")
    return frames, command


def clips_and_windows(source, manifest_path, *, stride=64):
    """Build eight-second clips and causal 64-second annotation/query windows."""
    if type(stride) is not int or not 8 <= stride <= 64 or stride % 8:
        raise ValueError("Window stride must be an 8-second multiple in [8,64]")
    start, end = source["stream_start"], source["stream_end"]
    frames = source["frames"]
    if len(frames) != round(end - start):
        raise ValueError("Dense source does not contain every one-second bin")
    for index, frame in enumerate(frames):
        timestamp = _number(frame["timestamp"], "measured frame timestamp")
        if not start + index <= timestamp < start + index + 1:
            raise ValueError("Missing, duplicated or future dense source frame")
    if len({frame["path"] for frame in frames}) != len(frames):
        raise ValueError("Dense frame paths are not unique")
    clip_count = len(frames) // 8
    clips = []
    for index in range(clip_count):
        a, b = start + index * 8, start + (index + 1) * 8
        selected = frames[index * 8:(index + 1) * 8]
        clips.append({"id": f"{source['source_id']}:{a:.3f}-{b:.3f}", "start": a, "end": b,
                      "frames": [frame["path"] for frame in selected], "sample_fps": 1.0})
    windows = []
    for offset in range(0, clip_count * 8 - 64 + 1, stride):
        left, query = start + offset, start + offset + 64
        selected = frames[offset:offset + 64]
        local = clips[offset // 8:offset // 8 + 8]
        prefix = [clip for clip in clips if clip["end"] <= query]
        window = {"window_id": f"{source['stream_id']}_{offset:06d}", "stream_id": source["stream_id"],
                  "source_id": source["source_id"], "source_group": source["source_group"], "split": source["split"],
                  "sha256": source["sha256"], "source_sha256": source["sha256"], "source_clock": source["source_clock"],
                  "visible_start": start, "start": left, "duration": 64, "question_time": query,
                  "source_manifest": str(Path(manifest_path).resolve()),
                  "frames": [{**frame, "index": i, "relative_time": frame["timestamp"] - left} for i, frame in enumerate(selected)],
                  "clips": local, "frame_timestamps": {frame["path"]: frame["timestamp"] for frame in selected},
                  "causal_prefix_clip_ids": [clip["id"] for clip in prefix],
                  "history_clip_ids": [clip["id"] for clip in prefix[:-1]], "current_clip_id": local[-1]["id"],
                  "benchmark_source_policy": POLICY, "benchmark_overlap": source["benchmark_overlap"],
                  "training_eligible": False, "action_concepts_excluded": True,
                  "plan_origin": "visually selected continuous source; dense measured-PTS causal window"}
        if len(local) != 8 or local[-1]["end"] != query or any(not left <= frame["timestamp"] < query for frame in window["frames"]):
            raise ValueError("A query window contains incomplete or future frames")
        windows.append(window)
    return clips, windows


def prepare_streams(manifest, output, ffmpeg="ffmpeg", ffprobe="ffprobe", *, long_side=640, stride=64, timeout=3600):
    """Freeze a new output directory, then decode selected streams serially."""
    if type(long_side) is not int or not 64 <= long_side <= 640:
        raise ValueError("Dense JPEG long_side must be 64–640")
    if type(stride) is not int or not 8 <= stride <= 64 or stride % 8:
        raise ValueError("Window stride must be an 8-second multiple in [8,64]")
    if type(timeout) is not int or not 1 <= timeout <= 7200:
        raise ValueError("Decode timeout must be 1–7200 seconds")
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Dense preparation is frozen; choose a new empty output directory")
    raw_manifest = Path(manifest).read_bytes()
    streams = selected_streams(manifest, raw_manifest=raw_manifest)
    output.mkdir(parents=True, exist_ok=True)
    versions = {}
    for name, binary in [("ffmpeg", ffmpeg), ("ffprobe", ffprobe)]:
        versions[name] = subprocess.run([str(binary), "-version"], check=True, capture_output=True,
                                        text=True, timeout=30).stdout.splitlines()[0]
    config = {"protocol": PROTOCOL, "input_manifest_sha256": hashlib.sha256(raw_manifest).hexdigest(),
              "long_side": long_side, "sample_fps": 1.0, "clip_seconds": 8, "window_seconds": 64,
              "window_stride": stride, "ffmpeg_processes": 1, "decoder_threads": 1, "filter_threads": 1,
              "encoder_threads": 1, "decode_timeout_seconds": timeout, "tools": versions,
              "benchmark_source_policy": POLICY, "training_eligible": False, "action_concepts_excluded": True,
              "revision": code_revision()}
    write_json(output / "prepare_config.json", config)
    (output / "selected_streams.input.jsonl").write_bytes(raw_manifest)
    sources, window_paths = [], []
    for stream in streams:
        probe = probe_source(stream, ffprobe)
        start, end = stream["interval"]["start"], stream["interval"]["end"]
        origin = stream["source_clock"]["origin_pts_seconds"]
        if abs(probe["origin_pts_seconds"] - origin) > 1e-7:
            raise ValueError("Measured container origin differs from frozen source clock")
        if start < probe["video_start_seconds"] or end > probe["duration_seconds"] + 1e-6:
            raise ValueError("Selected interval extends outside actual video coverage")
        directory = output / "sources" / stream["stream_id"]
        frames, command = extract_dense_frames(stream, directory / "frames", ffmpeg,
            start=start, end=end, origin=origin, long_side=long_side, timeout=timeout)
        if sha256_file(stream["local_path"]) != stream["sha256"]:
            raise ValueError("Raw source bytes changed during dense decode")
        manifest_path = directory / "source.json"
        source = {"protocol": PROTOCOL, **{key: stream[key] for key in (
            "stream_id", "source_id", "source_group", "split", "source_sha256", "sha256", "source_clock",
            "benchmark_source_policy", "benchmark_overlap", "visual_result_sha256")},
            "path": stream["local_path"], "local_path": stream["local_path"],
            "stream_start": start, "stream_end": end, "visible_start": start, "duration": end - start,
            "frames": frames, "frame_timestamps": {frame["path"]: frame["timestamp"] for frame in frames},
            "frame_pts_manifest": str((directory / "frame_pts.json").resolve()),
            "seek_validation_manifest": str((directory / "frames/seek_validation.json").resolve()),
            "probe": probe, "decode_command": command, "training_eligible": False,
            "action_concepts_excluded": True, "raw_source_modified": False}
        clips, windows = clips_and_windows(source, manifest_path, stride=stride)
        source.update(clips=clips, clipped_interval={"start": start, "end": clips[-1]["end"]},
                      unsliced_tail_seconds=end-clips[-1]["end"])
        write_json(directory / "frame_pts.json", {frame["path"]: {
            key: frame[key] for key in ("timestamp", "original_pts", "time_base", "source_time_fraction", "sha256")
        } for frame in frames})
        write_json(manifest_path, source)
        local_paths = []
        for window in windows:
            path = output / "windows" / f"{window['window_id']}.json"
            write_json(path, window)
            local_paths.append(str(path.resolve()))
        write_json(directory / "windows.json", local_paths)
        sources.append(str(manifest_path.resolve()))
        window_paths.extend(local_paths)
    write_json(output / "sources.json", sources)
    write_json(output / "windows.json", window_paths)
    summary = {**config, "sources": len(sources), "query_windows": len(window_paths),
               "dense_frames": sum(round(row["interval"]["end"]-row["interval"]["start"]) for row in streams),
               "source_manifests": sources, "interpretation": "Preparation only. No QA or exhaustive absence/event claims; Action concepts remain excluded during later annotation."}
    write_json(output / "prepare_summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--streams", type=Path, required=True, help="Explicitly accepted visual/source JSONL records")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--ffprobe", default="ffprobe")
    parser.add_argument("--long-side", type=int, default=640)
    parser.add_argument("--window-stride", type=int, default=64)
    parser.add_argument("--decode-timeout", type=int, default=3600)
    args = parser.parse_args()
    print(json.dumps(prepare_streams(args.streams, args.output, args.ffmpeg, args.ffprobe,
        long_side=args.long_side, stride=args.window_stride, timeout=args.decode_timeout), ensure_ascii=False))
