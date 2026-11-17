"""The successful Flash500 V4 sequence, with measured source-PTS rendering.

Time and Event questions use the same original V4 checks. There is no additional
temporal discovery, anchor-cardinality test, or exhaustive support-group review.
The successful V4 modules remain unchanged. This adapter makes no teacher calls
unless qualify_candidate is explicitly invoked.
"""
from __future__ import annotations

import fcntl
import hashlib
import inspect
import json
import math
from pathlib import Path
from types import SimpleNamespace

from . import narrative_qualification as measured
from .io import read_json, sha256_file, write_json
from .multimodal import VERIFIER_INPUT_PROTOCOL as READER_INPUT_PROTOCOL, stream_clock_text
from .narrative_qa import causal_clips
from .narrative_temporal import _call_receipt
from .replenishment import NEGATIVE_CHECK, choose_negative_candidates
from .schema import validate_sample
from .stream_audit import (
    PROTOCOL_VERSION, ablation_passed, aggregate_blind, history_chunks,
    require_repaired_sample, response_validator,
)
from .teacher import image_part

PROTOCOL = "restored-flash500-v4-measured-pts-v1"


def _digest(value):
    return measured._digest(value)


def _implementations():
    functions = (measured.measured_review_call, measured._finish_quality,
                 require_repaired_sample, validate_sample, image_part, causal_clips,
                 choose_negative_candidates, stream_clock_text)
    paths = {Path(__file__).resolve()}
    paths.update(Path(inspect.getsourcefile(function)).resolve() for function in functions)
    return {str(path): sha256_file(path) for path in sorted(paths)}


def _time(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError("An actual finite nonnegative source PTS is required")
    return float(value)


def _source(sample):
    return {key: sample[key] for key in ("source_id", "source_sha256", "source_group", "split",
                                        "benchmark_source_policy", "benchmark_overlap")}


def _check_input(sample, *, check_files):
    """Clock/provenance contract only; do not add a Time-specific QA gate."""
    # The old compiler sets a nominal accepted marker before V4; other callers
    # may correctly leave a proposal unqualified. Validate its structure here,
    # never use that marker as qualification, and never mutate the proposal.
    validate_sample({**sample, "verification": {"accepted": True}}, check_files=check_files)
    if sample.get("task") not in {"current", "past"}:
        raise ValueError("Unknown original V4 task")
    if (sample.get("split") not in {"train", "validation"}
            or sample.get("benchmark_source_policy") != "overlap_allowed_by_user"
            or not isinstance(sample.get("benchmark_overlap"), dict)
            or sample["benchmark_overlap"].get("status") not in {"unknown", "known_shared", "known_disjoint"}):
        raise ValueError("Frozen source split and explicit benchmark-overlap provenance are required")
    try:
        int(sample["source_sha256"], 16)
    except (ValueError, TypeError):
        raise ValueError("Explicit source SHA256 is required") from None
    times = sample.get("frame_timestamps")
    if not isinstance(times, dict) or not times:
        raise ValueError("Actual frame_timestamps are required; nominal FPS is not a fallback")
    query = _time(sample["question_time"])
    history, current = causal_clips(sample["candidates"]+[sample["current_clip"]], query,
                                    visible_start=_time(sample["visible_start"]))
    if history != sample["candidates"] or current != sample["current_clip"]:
        raise ValueError("V4 must inspect the complete observed causal prefix")
    seen = set()
    for clip in history+[current]:
        previous = None
        for path in clip["frames"]:
            if path not in times or path in seen:
                raise ValueError("Every distinct observed frame needs an actual PTS")
            timestamp = _time(times[path])
            if (not clip["start"] <= timestamp < clip["end"] or timestamp >= query
                    or previous is not None and timestamp <= previous):
                raise ValueError("Observed frame PTS is unordered, future or outside its clip")
            previous = timestamp
            seen.add(path)
    for concept in sample["concepts"]:
        if concept.get("kind") not in {None, "person", "object"}:
            raise ValueError("Action concepts are excluded")
        path = concept["image"]
        if (path not in seen or not concept.get("name") or not concept.get("description")
                or not sample["visible_start"] <= _time(concept["known_at"]) < query
                or abs(_time(concept["known_at"])-_time(times[path])) > 1e-6):
            raise ValueError("A runtime reference must be an actual causal source frame")


def _expected_call(sample, history, mode, suffix, negative, encoder):
    # Exactly the existing measured_review_call seed/prompt contract. IDs,
    # generation annotations, labels and proposed evidence are not model input.
    visible = {key: sample[key] for key in ("task", "question", "options", "visible_start", "question_time")}
    visible["concepts"] = [{key: concept[key] for key in ("name", "description", "image", "known_at")}
                           for concept in sample["concepts"]]
    seed = _digest(visible)+":"+measured.PROTOCOL+":"+mode+":"+str(suffix)+(":negative" if negative else "")
    content, order, mapping, times = measured.measured_audit_content(sample, history, mode, seed,
                                                                    image_encoder=encoder)
    if negative:
        if mode != "blind" or sample["task"] != "past":
            raise ValueError("Explicit historical negative checks require past blind input")
        content[0]["text"] = NEGATIVE_CHECK+"\n"+stream_clock_text(sample)
    metadata = {"option_permutation": order, "expected_display_index": order.index(sample["answer_index"]),
        "clip_mapping": mapping, "protocol_version": PROTOCOL_VERSION, "reader_input_protocol": READER_INPUT_PROTOCOL,
        "narrative_rendering_protocol": measured.PROTOCOL, "measured_frame_timestamps": times,
        "request_content_sha256": hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest(),
        "proposed_evidence_locations_visible": False,
        "annotation_window_visible": False, "context_isolation": "single_question"}
    return content, metadata


def _check_call(sample, row, call, history, mode, suffix, negative, encoder):
    content, expected = _expected_call(sample, history, mode, suffix, negative, encoder)
    if (any(call.get(key) != value for key, value in expected.items())
            or bool(call.get("negative_confirmation")) != negative or call.get("supplemental_protocol") is not None):
        raise ValueError("V4 call interpretation differs from the actual measured request")
    _call_receipt(call, SimpleNamespace(content=content))
    response_validator(mode, expected["clip_mapping"], sample["task"])( {"questions": [row]} )


def require_qualified(sample, *, image_encoder=None, check_files=True):
    """Recompute the original V4 proof plus actual request/PTS bindings."""
    require_repaired_sample(sample)
    _check_input(sample, check_files=check_files)
    gate = sample.get("restored_v4_qualification", {})
    if (gate.get("protocol") != PROTOCOL or gate.get("source_provenance") != _source(sample)
            or gate.get("candidate_sha256") != sample["repair_provenance"]["original_sample_sha256"]):
        raise ValueError("Restored V4 qualification provenance is missing or changed")
    encoder = image_encoder or image_part
    by_id = {clip["id"]: clip for clip in sample["candidates"]}
    records = gate.get("blind_receipts", [])
    if not isinstance(records, list) or not records:
        raise ValueError("Original full-prefix blind response receipts are missing")
    reviews = []
    for record in records:
        ids = record.get("history_clip_ids")
        if (not isinstance(ids, list) or len(set(ids)) != len(ids) or any(cid not in by_id for cid in ids)
                or type(record.get("negative")) is not bool):
            raise ValueError("A blind receipt has invalid historical context")
        history = [by_id[cid] for cid in ids]
        _check_call(sample, record["response"], record["call"], history, "blind", record["suffix"],
                    record["negative"], encoder)
        reviews.append((record["response"], record["call"]))
    fresh = aggregate_blind(reviews, sample)
    screens = sample["quality_screens"]
    if (not fresh["accepted"] or fresh != screens["global_blind"]
            or [call for _, call in reviews] != screens["global_blind_calls"]):
        raise ValueError("Stored V4 blind approval differs from actual independent responses")
    if sample["task"] == "past" and sample["split"] == "train" and not sample["negative_clip_ids"]:
        raise ValueError("Training Past QA still needs an explicitly verified retrieval negative")
    for mode in ("evidence_only", "training_context", "reference_only", "no_concept"):
        call = screens["calls"][mode] if mode in {"reference_only", "no_concept"} else screens[mode+"_call"]
        ids = [cid for label, cid in call.get("clip_mapping", {}).items() if label != "CURRENT"]
        if len(set(ids)) != len(ids) or any(cid not in by_id for cid in ids):
            raise ValueError("A quality check has invalid historical context")
        history = sorted((by_id[cid] for cid in ids), key=lambda clip: clip["start"])
        _check_call(sample, screens[mode], call, history, mode, "", False, encoder)
    context = sample.get("context_positive_clip_ids", [])
    if not context or len(set(context)) != len(context) or len(context) > 3:
        raise ValueError("The original reader context must use one to three support clips")
    if not any(event["answer_index"] == sample["answer_index"] and set(event["clip_ids"]) == set(context)
               for event in fresh["occurrences"]):
        raise ValueError("The selected context is not an original blind support occurrence")


def qualify_candidate(teacher, directory, sample, *, max_frames=224, max_bytes=12*1024*1024,
                      image_encoder=None, check_files=True):
    """Original V4 sequence. Quality returns reasons; API/budget errors propagate."""
    _check_input(sample, check_files=check_files)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".restored_v4.lock").open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        config = {"protocol": PROTOCOL, "candidate_sha256": _digest(sample), "max_frames": max_frames,
                  "max_bytes": max_bytes, "implementation_sha256": _implementations()}
        path = directory / "restored_v4_input.json"
        if path.exists() and read_json(path) != config:
            raise ValueError("Frozen restored V4 input/code changed; use another directory")
        write_json(path, config)
        encoder = image_encoder or image_part

        def finish(accepted, reasons):
            write_json(directory / "qualification_result.json", {"protocol": PROTOCOL, "sample": accepted, "reasons": reasons})
            return accepted, reasons

        reduced, _ = measured.measured_review_call(teacher, directory, sample, [], "reference_only", image_encoder=encoder)
        if not ablation_passed(reduced):
            return finish(None, ["reference_only_dependence_not_established"])
        reviews, records = [], []

        def review(history, suffix, negative=False):
            row, call = measured.measured_review_call(teacher, directory, sample, history, "blind", suffix,
                                                     image_encoder=encoder, negative=negative)
            reviews.append((row, call))
            records.append({"response": row, "call": call, "history_clip_ids": [clip["id"] for clip in history],
                            "suffix": str(suffix), "negative": negative})

        for index, chunk in enumerate(history_chunks(sample, max_frames, 2, max_bytes)):
            review(chunk, str(index))
        blind = aggregate_blind(reviews, sample)
        if not blind["accepted"]:
            return finish(None, blind["reasons"])
        if sample["task"] == "past":
            used = set()
            for attempt in range(2):
                if blind["negative_clip_ids"]:
                    break
                history = choose_negative_candidates(sample, blind["positive_clip_ids"], used)
                if not history:
                    break
                used.update(clip["id"] for clip in history)
                review(history, str(attempt), True)
                blind = aggregate_blind(reviews, sample)
                if not blind["accepted"]:
                    return finish(None, blind["reasons"])
            if sample["split"] == "train" and not blind["negative_clip_ids"]:
                return finish(None, ["no_verified_retrieval_negative"])
        possible = [event for event in blind["occurrences"] if len(event["clip_ids"]) <= 3]
        if not possible:
            return finish(None, ["support_exceeds_original_media_limit"])
        context = min(possible, key=lambda event: (len(event["clip_ids"]), event["clip_ids"]))["clip_ids"]
        accepted, reasons = measured._finish_quality(teacher, directory, sample, blind, [call for _, call in reviews],
            context, blind["negative_clip_ids"][:2], image_encoder=encoder, check_files=check_files)
        if accepted is not None:
            accepted["restored_v4_qualification"] = {"protocol": PROTOCOL,
                "candidate_sha256": config["candidate_sha256"], "source_provenance": _source(sample),
                "blind_receipts": records, "gates": "original_flash500_v4", "temporal_extra_gate": False}
            require_qualified(accepted, image_encoder=encoder, check_files=check_files)
        return finish(accepted, reasons)
