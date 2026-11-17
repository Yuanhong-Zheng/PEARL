#!/usr/bin/env python3
"""CPU-only movie pixels for fresh legacy 64-second window annotation.

Input intervals are candidates, not visually approved streams. This entry point
does not call selected_streams, a teacher, or a network service. Visual usability
is left to the original per-window ANNOTATE stage.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from fractions import Fraction
import json
import math
from pathlib import Path
import re
import subprocess
import traceback

from pearl_training.io import read_json, read_jsonl, sha256_file, write_json
from pearl_training.narrative_prepare import CLOCK_DEFINITION, clips_and_windows, extract_dense_frames
from pearl_training.narrative_visual import probe_source


REPO = Path(__file__).resolve().parents[1]
HELPERS = [REPO / "src/pearl_training" / (name + ".py") for name in
           ["narrative_prepare", "narrative_visual", "io"]]


def helper_hashes():
    return {str(p): sha256_file(p) for p in HELPERS}


def candidate_inputs(streams, split_registry):
    """Check local identities and clocks without claiming visual acceptance."""
    records = read_jsonl(streams)
    if not records:
        raise ValueError("Candidate stream manifest is empty")
    registry = read_json(split_registry)
    identifiers, groups, hashes, result = set(), {}, {}, []
    for row in records:
        source = row["source"]
        stream_id, source_id = row["stream_id"], source["source_id"]
        if not re.fullmatch(r"[0-9a-f]{20}", stream_id) or stream_id in identifiers:
            raise ValueError("Candidate stream IDs must be safe and unique")
        identifiers.add(stream_id)
        group, split, raw_hash = source["source_group"], source["split"], source["sha256"]
        if split not in {"train", "validation"} or registry[group]["split"] != split:
            raise ValueError("Candidate changes the frozen source-work split")
        if groups.setdefault(group, split) != split or hashes.setdefault(raw_hash, split) != split:
            raise ValueError("Source group or raw content crosses the frozen split")
        if (not isinstance(source_id, str) or not source_id or not re.fullmatch(r"[0-9a-f]{64}", raw_hash)
                or source.get("benchmark_source_policy") != "overlap_allowed_by_user"):
            raise ValueError("Explicit local source identity/hash/provenance is required")
        overlap = source.get("benchmark_overlap", {"status": "unknown"})
        if overlap.get("status") not in {"unknown", "known_shared", "known_disjoint"}:
            raise ValueError("Invalid existing benchmark overlap provenance")
        start, end = row["interval"]["start"], row["interval"]["end"]
        origin = row["probe"]["origin_pts_seconds"]
        if (any(isinstance(t, bool) or not isinstance(t, (int, float)) or not math.isfinite(t)
                for t in [start, end, origin]) or start < 0 or end - start != 900):
            raise ValueError("Preserve the existing finite 900-second candidate interval")
        path = Path(source["local_path"]).resolve(strict=True)
        if sha256_file(path) != raw_hash:
            raise ValueError("Candidate raw source hash changed")
        result.append({"stream_id": stream_id, "source_id": source_id, "source_group": group,
            "split": split, "sha256": raw_hash, "source_sha256": raw_hash, "local_path": str(path),
            "source_clock": {"origin_pts_seconds": origin, "definition": CLOCK_DEFINITION},
            "benchmark_source_policy": "overlap_allowed_by_user", "benchmark_overlap": overlap,
            "interval": {"start": start, "end": end}})
    return result


def prepare_one(source, output, args):
    directory = output / "sources" / source["stream_id"]
    directory.mkdir(parents=True, exist_ok=False)
    start, end = source["interval"]["start"], source["interval"]["end"]
    try:
        measured_probe = probe_source(source, args.ffprobe)
        origin = source["source_clock"]["origin_pts_seconds"]
        if (abs(measured_probe["origin_pts_seconds"] - origin) > 1e-7
                or start < measured_probe["video_start_seconds"]
                or end > measured_probe["duration_seconds"] + 1e-6):
            raise ValueError("Candidate interval/clock differs from the measured raw video")
        frames, command = extract_dense_frames(source, directory / "frames", args.ffmpeg,
            start=start, end=end, origin=origin, long_side=args.long_side, timeout=args.timeout)
        if sha256_file(source["local_path"]) != source["sha256"]:
            raise ValueError("Raw source content changed during CPU preparation")
        manifest_path = directory / "source.json"
        dense = {**source, "path": source["local_path"], "stream_start": start, "stream_end": end,
            "visible_start": start, "duration": end-start, "frames": frames,
            "frame_timestamps": {f["path"]: f["timestamp"] for f in frames},
            "probe": measured_probe, "decode_command": command,
            "frame_pts_manifest": str((directory / "frame_pts.json").resolve()),
            "seek_validation_manifest": str((directory / "frames/seek_validation.json").resolve()),
            "preparation_kind": "CPU-only candidate pixels for legacy 64-second ANNOTATE",
            "visual_screening_status": "not_run", "legacy_annotation_status": "pending",
            "training_eligible": False, "action_concepts_excluded": True, "raw_source_modified": False}
        clips, windows = clips_and_windows(dense, manifest_path, stride=args.window_stride)
        dense.update(clips=clips, clipped_interval={"start": start, "end": clips[-1]["end"]},
                     unsliced_tail_seconds=end-clips[-1]["end"])
        write_json(directory / "frame_pts.json", {f["path"]: {k: f[k] for k in
            ["timestamp", "original_pts", "time_base", "source_time_fraction", "sha256"]} for f in frames})
        write_json(manifest_path, dense)
        paths = []
        for window in windows:
            window.update(plan_origin="CPU-prepared candidate; usability awaits legacy per-window ANNOTATE",
                          visual_screening_status="not_run", legacy_annotation_status="pending")
            path = output / "windows" / (window["window_id"] + ".json")
            write_json(path, window)
            paths.append(str(path.resolve()))
        write_json(directory / "windows.json", paths)
        status = {"stream_id": source["stream_id"], "source_id": source["source_id"],
            "source_group": source["source_group"], "split": source["split"], "status": "cpu_ready",
            "source_manifest": str(manifest_path.resolve()), "source_manifest_sha256": sha256_file(manifest_path),
            "window_paths": paths, "frames": len(frames), "clips": len(clips), "windows": len(paths),
            "unsliced_tail_seconds": dense["unsliced_tail_seconds"],
            "max_source_bin_offset_seconds": max(float(Fraction(f["source_time_fraction"])) - (start+i)
                for i, f in enumerate(frames)),
            "visual_usability_decided": False, "model_calls": 0}
    except Exception as error:
        stderr = getattr(error, "stderr", None)
        if stderr:
            (directory / "ffmpeg_failure.stderr.log").write_text(
                stderr.decode(errors="replace") if isinstance(stderr, bytes) else stderr)
        status = {"stream_id": source["stream_id"], "source_id": source["source_id"],
            "source_group": source["source_group"], "split": source["split"], "status": "held_cpu_error",
            "error_type": type(error).__name__, "error": str(error), "traceback": traceback.format_exc(),
            "failed_command": getattr(error, "cmd", None),
            "preserved_frames": len(list((directory / "frames").glob("frame_*.jpg"))),
            "original_guards_unchanged": True, "label_assigned": False, "model_calls": 0}
    write_json(directory / "preparation_status.json", status)
    return status


def publish(output, statuses, config):
    ordered = sorted(statuses.values(), key=lambda r: r["stream_id"])
    ready = [r for r in ordered if r["status"] == "cpu_ready"]
    write_json(output / "sources.json", [r["source_manifest"] for r in ready])
    write_json(output / "windows.json", [p for r in ready for p in r["window_paths"]])
    summary = {"candidate_sources": config["candidate_sources"], "cpu_ready_sources": len(ready),
        "held_sources": sum(r["status"] == "held_cpu_error" for r in ordered),
        "pending_sources": config["candidate_sources"]-len(ordered),
        "dense_frames": sum(r["frames"] for r in ready), "query_windows": sum(r["windows"] for r in ready),
        "statuses": ordered, "model_calls": 0, "visual_usability_decided": False,
        "next_stage": "Original legacy64s ANNOTATE usable/concepts/events then original generation/V4",
        "source_clock_reset": False, "benchmark_overlap_fabricated": False}
    write_json(output / "prepare_summary.json", summary)
    return summary


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--streams", required=True, type=Path)
    p.add_argument("--split-registry", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--ffmpeg", required=True, type=Path)
    p.add_argument("--ffprobe", required=True, type=Path)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--long-side", type=int, default=640)
    p.add_argument("--window-stride", type=int, default=64)
    p.add_argument("--timeout", type=int, default=3600)
    args = p.parse_args(argv)
    if not 1 <= args.workers <= 2 or args.window_stride != 64:
        p.error("Use at most two CPU ffmpeg processes and the existing 64-second window stride")
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Preserve existing artifacts; choose a new empty output directory")
    frozen = helper_hashes()
    candidates = candidate_inputs(args.streams, args.split_registry)
    output.mkdir(parents=True, exist_ok=False)
    config = {"candidate_sources": len(candidates), "streams_path": str(args.streams.resolve()),
        "streams_sha256": sha256_file(args.streams), "split_registry_path": str(args.split_registry.resolve()),
        "split_registry_sha256": sha256_file(args.split_registry), "script_sha256": sha256_file(__file__),
        "helper_sha256": frozen, "workers": args.workers, "ffmpeg_threads_per_process": 1,
        "long_side": args.long_side, "window_stride": args.window_stride, "timeout_seconds": args.timeout,
        "visual_usability_decided": False, "model_calls": 0}
    write_json(output / "prepare_config.json", config)
    (output / "candidate_streams.input.jsonl").write_bytes(args.streams.read_bytes())
    statuses = {}
    publish(output, statuses, config)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(prepare_one, row, output, args): row for row in candidates}
        for future in as_completed(futures):
            status = future.result()
            statuses[status["stream_id"]] = status
            summary = publish(output, statuses, config)
            print(json.dumps({"stream_id": status["stream_id"], "status": status["status"],
                "cpu_ready_sources": summary["cpu_ready_sources"], "held_sources": summary["held_sources"],
                "pending_sources": summary["pending_sources"]}), flush=True)
    assert helper_hashes() == frozen, "Original CPU helpers changed during preparation"
    write_json(output / "completion.json", {"helpers_unchanged": True,
        "summary_sha256": sha256_file(output / "prepare_summary.json"), "model_calls": 0})
    return 2 if summary["held_sources"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
