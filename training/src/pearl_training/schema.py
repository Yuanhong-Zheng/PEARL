"""Fail-closed checks for causal, evidence-labelled training examples."""

from __future__ import annotations

import math
import re
from pathlib import Path


def interval(value, name):
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"{name}: expected [start, end]")
    a, b = value
    if isinstance(a, bool) or isinstance(b, bool) or not all(isinstance(t, (int, float)) and math.isfinite(t) for t in (a, b)):
        raise ValueError(f"{name}: invalid time")
    if not 0 <= a < b:
        raise ValueError(f"{name}: invalid interval")
    return float(a), float(b)


def validate_proposal(proposal, frame_count, duration, clip_seconds=8, max_questions=4, max_concepts=3, allow_reference_until_query=False):
    if set(proposal) != {"concepts", "questions"}:
        raise ValueError("Expected concepts and questions only")
    concepts = proposal["concepts"]
    if not 1 <= len(concepts) <= max_concepts:
        raise ValueError(f"Expected 1–{max_concepts} concepts")
    ids = set()
    for concept in concepts:
        cid = concept["id"]
        if not isinstance(cid, str) or not re.fullmatch(r"C[0-9]", cid) or cid in ids:
            raise ValueError("Invalid or repeated concept ID")
        ids.add(cid)
        index = concept["reference_frame"]
        reference_limit = frame_count if allow_reference_until_query else min(frame_count, duration - clip_seconds)
        if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < reference_limit:
            raise ValueError("Reference frame is outside the permitted causal prefix")
        if not isinstance(concept["description"], str) or not concept["description"].strip():
            raise ValueError("Missing appearance-only description")
    questions = proposal["questions"]
    if not 1 <= len(questions) <= max_questions:
        raise ValueError(f"Expected 1–{max_questions} original questions")
    qids = set()
    for question in questions:
        if not isinstance(question["id"], str) or not re.fullmatch(r"q[0-9]+", question["id"]) or question["id"] in qids:
            raise ValueError("Repeated question ID")
        qids.add(question["id"])
        if question["task"] not in {"current", "past"}:
            raise ValueError("Unknown QA task")
        if not question["concept_ids"] or not set(question["concept_ids"]) <= ids:
            raise ValueError("Undefined concept in question")
        options = question["options"]
        if len(options) != 4 or len({str(s).strip().casefold() for s in options}) != 4 or not all(isinstance(s, str) and s.strip() for s in options):
            raise ValueError("Expected four distinct, nonempty options")
        question_and_options = question["question"] + "\n" + "\n".join(options)
        for cid in question["concept_ids"]:
            if "{" + cid + "}" not in question_and_options:
                raise ValueError("Concept must be explicit in question or options")
        answer = question["answer_index"]
        if isinstance(answer, bool) or not isinstance(answer, int) or not 0 <= answer < 4:
            raise ValueError("Invalid answer index")
        if not question["evidence"]:
            raise ValueError("Missing evidence")
        for evidence in question["evidence"]:
            start, end = interval(evidence, "evidence")
            if end > duration:
                raise ValueError("Evidence after the question")
            if question["task"] == "past" and end > duration - clip_seconds:
                raise ValueError("Past evidence must be in a completed historical clip")
            if question["task"] == "current" and start < duration - clip_seconds:
                raise ValueError("Current evidence must be in the current clip")
        for clip in question.get("negative_clip_indices", []):
            if isinstance(clip, bool) or not isinstance(clip, int) or not 0 <= clip < duration // clip_seconds - 1:
                raise ValueError("Negative must be an eligible historical clip")
            a, b = clip * clip_seconds, (clip + 1) * clip_seconds
            if any(max(a, x) < min(b, y) for x, y in question["evidence"]):
                raise ValueError("A positive clip cannot be a negative")


def validate_sample(sample, check_files=True):
    if sample["split"] not in {"train", "validation", "test"}:
        raise ValueError("Invalid split")
    if not sample.get("source_group") or len(sample.get("source_sha256", "")) != 64:
        raise ValueError("Missing source provenance")
    time = sample["question_time"]
    for concept in sample["concepts"]:
        if concept["known_at"] > time:
            raise ValueError("Concept defined after question")
        if check_files and not Path(concept["image"]).is_file():
            raise ValueError("Missing concept image")
    all_clips = sample["candidates"] + [sample["current_clip"]]
    ids = [x["id"] for x in all_clips]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate clip IDs")
    for clip in all_clips:
        if clip["end"] > time or clip["start"] >= clip["end"]:
            raise ValueError("Invalid or noncausal clip")
        if not clip["frames"]:
            raise ValueError("Empty clip")
        if check_files and not all(Path(p).is_file() for p in clip["frames"]):
            raise ValueError("Missing clip frames")
    history = {x["id"] for x in sample["candidates"] if x["end"] <= sample["current_clip"]["start"]}
    if not set(sample["positive_clip_ids"]) <= set(ids):
        raise ValueError("Unknown positive")
    if sample["task"] == "past" and not set(sample["positive_clip_ids"]) <= history:
        raise ValueError("Past positive is not completed history")
    if not set(sample["negative_clip_ids"]) <= history:
        raise ValueError("Negative is not eligible history")
    if set(sample["positive_clip_ids"]) & set(sample["negative_clip_ids"]):
        raise ValueError("Conflicting retrieval labels")
    if len(sample["options"]) != 4 or not 0 <= sample["answer_index"] < 4:
        raise ValueError("Invalid answer")
    if sample.get("verification", {}).get("accepted") is not True:
        raise ValueError("Unverified sample")


def check_split_isolation(samples):
    groups, hashes, base_ids = {}, {}, {}
    for sample in samples:
        for registry, key in ((groups, sample["source_group"]), (hashes, sample["source_sha256"]), (base_ids, sample["base_question_id"])):
            previous = registry.setdefault(key, sample["split"])
            if previous != sample["split"]:
                raise ValueError(f"Cross-split leakage for {key}")
