"""V4 data checks with causal source time and separated annotation verification."""

from __future__ import annotations

import hashlib
import json
import math
import random
from functools import lru_cache
from pathlib import Path

from .batch_generate import stage_call
from .multimodal import VERIFIER_INPUT_PROTOCOL as READER_INPUT_PROTOCOL, stream_clock_text
from .teacher import image_part


PROTOCOL_VERSION = "stream-v4-no-location-hints"

BLIND = """Independently inspect one personalized video question. No proposed answer,
evidence locations, generation annotations or annotation-window boundaries are
provided. Use pixels and the arbitrary runtime names only; ignore subtitles,
audio, real character names and movie knowledge.

The supplied historical clips are one part of the entire causal stream, not a
new episode. Identify ALL visible occurrences satisfying the question, including
occurrences giving different answer choices. For past questions, occurrences
must be in historical clips; for current questions, in CURRENT. Respect temporal
words relative to the stated stream/query time. Missing history is not itself a
reason to declare the question unclear. Set question_clear false for intrinsically
ambiguous actor, action, relation, or temporal wording. A later aggregation checks
other history chunks. Never assume an unannotated clip is negative.

Return JSON only: {"questions":[{"id":"item","concept_bindings_valid":true,
"question_clear":true,"occurrences":[{"answer_index":0,"clip_labels":["H0"]}],
"confirmed_negative_labels":[],"reason":"short visible facts"}]}.
For current questions, confirmed_negative_labels must be empty. For past
questions, list only historical H labels as retrieval negatives, never CURRENT.
List an occurrence only if the supplied clips actually support it; include all
clips needed for a multi-clip event. Use only displayed labels. A confirmed negative
requires visible evidence that the requested actor/event/attribute is absent or
incompatible. Empty occurrence/negative lists are allowed. Use false or null when
uncertain. Do not guess an answer to pass the question.
"""

ANSWER = """Answer the ONLY question from the explicitly supplied runtime references
and video clips. You have no proposed answer or evidence interval. All clips are
the entire visual input available for this check. Do not use subtitles, real names,
movie knowledge or unseen frames. Decide whether these inputs suffice for one
unique answer with the correct runtime concept binding. If not, abstain.
Return JSON only: {"questions":[{"id":"item","answer_index":0,
"sufficient_information":true,"unique_answer":true,"concept_bindings_valid":true,
"reason":"short visible facts"}]}. Use answer_index null when insufficient.
"""

ABLATION = """Answer the ONLY question using ONLY the reduced input supplied here.
Some information has deliberately been removed. Arbitrary names have no meaning
without their supplied runtime definitions. Do not infer mappings from appearance
order. Do not use movie knowledge or guess. If these pixels/definitions do not
determine a unique answer, set answer_index null and sufficient_information false.
Return JSON only: {"questions":[{"id":"item","answer_index":null,
"sufficient_information":false,"reason":"what information is missing"}]}.
"""

REWRITE = """Repair a newly generated visual question for a continuous video stream.
The pictures show the original local source episode only, but the eventual reader
may retrieve ANY completed history since source time zero. The episode boundary
is annotation metadata and will NOT be supplied to the reader.

Keep the exact four options, their order, the intended correct option, the runtime
concept names and the queried fact. Rewrite only the question if necessary. Use
a short, visibly supported scene/event anchor that is meaningful in the full
stream. Do not use numeric timestamps, frame/clip IDs, the 64-second annotation
window, or unqualified 'beginning/start of the video/scene'. Do not reveal the
answer through the anchor or remove the need to bind arbitrary concept names.
The phrases 'now', 'in the CURRENT clip', and 'just before the CURRENT clip' are
legitimate: the reader knows the current clip and its source time. Do not replace
them merely because they mention a clip. Use only visible motion, objects and
spatial relations. Ignore subtitles/dialogue and do not anchor a question on what
someone says, demands, thinks or their inferred relationship. An existing runtime
name may be added for another visibly grounded actor in the anchor, but never
add the correct answer option itself as a hint or invent a new name.
Do not invent details. If the old question is already clear, keep it unchanged.
If no faithful unambiguous repair is supported, set repairable false.
Return JSON only: {"repairable":true,"question":"repaired question",
"reason":"short explanation"}. This is generation, not blind verification.
"""


def shuffled_display(sample, seed):
    order = list(range(4))
    random.Random(seed).shuffle(order)
    # A whitelist is essential: neither labels nor locating hints are exposed.
    return {"id": "item", "task": sample["task"], "question": sample["question"],
            "options": [sample["options"][i] for i in order]}, order


@lru_cache(maxsize=None)
def encoded_image_size(path):
    return 4 * math.ceil(Path(path).stat().st_size / 3) + 200


def history_chunks(sample, max_frames=224, overlap_clips=2, max_bytes=12 * 1024 * 1024):
    if sample["task"] == "current":
        return [[]]
    clips = sorted(sample["candidates"], key=lambda c: c["start"])
    if not clips or any(c["end"] > sample["current_clip"]["start"] for c in clips):
        raise ValueError("Past questions require a complete causal historical pool")
    cursor = sample.get("visible_start", 0)
    for clip in clips:
        if abs(clip["start"] - cursor) > 1e-6:
            raise ValueError("Causal history has a gap or overlap; do not silently narrow the reviewed stream")
        cursor = clip["end"]
    if abs(cursor - sample["current_clip"]["start"]) > 1e-6:
        raise ValueError("Causal history does not reach the current clip")
    fixed = [c["image"] for c in sample["concepts"]] + sample["current_clip"]["frames"]
    fixed_bytes = sum(encoded_image_size(p) for p in fixed) + 65536 if max_bytes else 0
    result, start = [], 0
    while start < len(clips):
        end, count, size = start, 0, fixed_bytes
        while end < len(clips):
            clip = clips[end]
            added = sum(encoded_image_size(p) for p in clip["frames"]) + 300 if max_bytes else 0
            if count + len(clip["frames"]) > max_frames or max_bytes and size + added > max_bytes:
                break
            count += len(clip["frames"])
            size += added
            end += 1
        if end == start or end < len(clips) and end - start <= overlap_clips:
            raise ValueError("A chunk cannot fit sufficient overlapping clips; review media limits")
        result.append(clips[start:end])
        if end == len(clips):
            break
        start = end - overlap_clips
    if {c["id"] for chunk in result for c in chunk} != {c["id"] for c in clips}:
        raise AssertionError("The global blind review omitted historical clips")
    return result


def reference_parts(sample):
    result = []
    for concept in sample["concepts"]:
        result += [{"type": "text", "text": "Runtime concept {" + concept["name"] + "}: " + concept["description"]},
                   image_part(concept["image"])]
    return result


def clip_parts(clip, label):
    result = [{"type": "text", "text": f"BEGIN CLIP {label}: source [{clip['start']:.3f}, {clip['end']:.3f}) seconds."}]
    for i, frame in enumerate(clip["frames"]):
        result += [{"type": "text", "text": f"Sample at source {clip['start'] + i / clip['sample_fps']:.3f} seconds."}, image_part(frame)]
    result.append({"type": "text", "text": "END CLIP " + label + "."})
    return result


def audit_content(sample, history, mode, seed):
    if mode not in {"blind", "evidence_only", "training_context", "reference_only", "no_concept"}:
        raise ValueError("Unknown V4 audit mode")
    if any(c["end"] > sample["current_clip"]["start"] for c in history):
        raise ValueError("Audit contains current or future historical pixels")
    prompt = BLIND if mode == "blind" else ABLATION if mode in {"reference_only", "no_concept"} else ANSWER
    content = [{"type": "text", "text": prompt + "\n" + stream_clock_text(sample)}]
    if mode != "no_concept":
        content += reference_parts(sample)
    mapping = {}
    if mode != "reference_only":
        for i, clip in enumerate(history):
            label = "H" + str(i)
            mapping[label] = clip["id"]
            content += clip_parts(clip, label)
    if not (mode == "reference_only" and sample["task"] == "current"):
        content += clip_parts(sample["current_clip"], "CURRENT")
        mapping["CURRENT"] = sample["current_clip"]["id"]
    display, order = shuffled_display(sample, seed)
    content.append({"type": "text", "text": "The ONLY question; use id item:\n" + json.dumps(display, ensure_ascii=False)})
    return content, order, mapping


def response_validator(mode, mapping, task):
    def validate(response):
        if not isinstance(response, dict):
            raise ValueError("Audit response must be an object")
        rows = response.get("questions")
        if not isinstance(rows, list) or len(rows) != 1 or not isinstance(rows[0], dict) or rows[0].get("id") != "item":
            raise ValueError("One blind question/result is required")
        row = rows[0]
        if mode == "blind":
            if not isinstance(row.get("occurrences"), list) or not isinstance(row.get("confirmed_negative_labels"), list):
                raise ValueError("Blind review needs occurrence and confirmed-negative lists")
            for field in ["concept_bindings_valid", "question_clear"]:
                if row.get(field) is not None and type(row[field]) is not bool:
                    raise ValueError("Review confidence flags must be boolean or null")
            for occurrence in row["occurrences"]:
                if not isinstance(occurrence, dict):
                    raise ValueError("Each occurrence must be an object, not a list or scalar")
                labels = occurrence.get("clip_labels")
                answer = occurrence.get("answer_index")
                if type(answer) is not int or not 0 <= answer < 4 or not labels or not isinstance(labels, list):
                    raise ValueError("Malformed occurrence")
                if any(label not in mapping or task == "past" and label == "CURRENT" for label in labels):
                    raise ValueError("Occurrence references unshown or nonhistorical pixels")
            if any(label not in mapping for label in row["confirmed_negative_labels"]):
                raise ValueError("Negative references unshown pixels")
        else:
            answer = row.get("answer_index")
            if "answer_index" not in row or not (answer is None or type(answer) is int and 0 <= answer < 4):
                raise ValueError("Answer must be null or an option index")
            if row.get("sufficient_information") is not None and type(row["sufficient_information"]) is not bool:
                raise ValueError("Sufficiency must be boolean or null")
    return validate


def review_call(teacher, directory, sample, history, mode, suffix=""):
    seed = sample["sample_id"] + ":" + PROTOCOL_VERSION + ":" + mode + ":" + suffix
    content, order, mapping = audit_content(sample, history, mode, seed)
    fingerprint = hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()
    stage = mode + "_" + suffix + "_" + PROTOCOL_VERSION + "_" + fingerprint[:12]
    task = "visual_ablation" if mode in {"reference_only", "no_concept"} else "visual_verification"
    response, call = stage_call(teacher, directory, stage, task, content, response_validator(mode, mapping, sample["task"]))
    return response["questions"][0], {**call, "option_permutation": order,
        "expected_display_index": order.index(sample["answer_index"]), "clip_mapping": mapping,
        "protocol_version": PROTOCOL_VERSION, "reader_input_protocol": READER_INPUT_PROTOCOL,
        "request_content_sha256": fingerprint, "proposed_evidence_locations_visible": False,
        "annotation_window_visible": False, "context_isolation": "single_question"}


def aggregate_blind(reviews, sample):
    answers, occurrences, negatives, reasons = set(), [], set(), []
    reviewed = set()
    for row, call in reviews:
        if row.get("concept_bindings_valid") is not True or row.get("question_clear") is not True:
            reasons.append("blind_grounding_or_question_scope_uncertain")
        order, mapping = call["option_permutation"], call["clip_mapping"]
        reviewed.update(cid for label, cid in mapping.items() if label != "CURRENT")
        for event in row["occurrences"]:
            answer = order[event["answer_index"]]
            answers.add(answer)
            occurrences.append({"answer_index": answer, "clip_ids": sorted({mapping[label] for label in event["clip_labels"]})})
        # A valid observation that CURRENT lacks the event is not a historical
        # retrieval negative, nor a JSON/schema error to retry for another answer.
        negatives.update(mapping[label] for label in row["confirmed_negative_labels"] if label != "CURRENT")
    expected_history = {c["id"] for c in sample["candidates"]} if sample["task"] == "past" else set()
    if reviewed != expected_history:
        raise ValueError("Blind review did not cover the entire required causal prefix")
    if answers != {sample["answer_index"]}:
        reasons.append("global_blind_answer_missing_conflicting_or_disagrees")
    positives = {cid for event in occurrences for cid in event["clip_ids"]}
    if positives & negatives:
        reasons.append("conflicting_blind_evidence_and_negative")
    return {"accepted": not reasons, "reasons": sorted(set(reasons)), "observed_answer_indices": sorted(answers),
            "occurrences": occurrences, "positive_clip_ids": sorted(positives),
            "negative_clip_ids": sorted(negatives - positives), "reviewed_history_clip_ids": sorted(reviewed),
            "whole_causal_prefix_reviewed": True}


def answer_passed(row, call):
    return (row.get("sufficient_information") is True and row.get("unique_answer") is True
            and row.get("concept_bindings_valid") is True and type(row.get("answer_index")) is int
            and row["answer_index"] == call["expected_display_index"])


def ablation_passed(row):
    return row.get("sufficient_information") is False and "answer_index" in row and row["answer_index"] is None


def require_repaired_sample(sample):
    """A V3 approval or a true flag alone cannot authorize new V4 training."""
    screens = sample.get("quality_screens", {})
    if (screens.get("protocol") != PROTOCOL_VERSION
            or screens.get("reader_input_protocol") != READER_INPUT_PROTOCOL):
        raise ValueError("Data must pass the repaired V4 protocol before new training")
    blind = screens.get("global_blind", {})
    if (blind.get("accepted") is not True or blind.get("whole_causal_prefix_reviewed") is not True
            or blind.get("observed_answer_indices") != [sample["answer_index"]]
            or set(blind.get("positive_clip_ids", [])) != set(sample["positive_clip_ids"])
            or not set(sample["negative_clip_ids"]) <= set(blind.get("negative_clip_ids", []))):
        raise ValueError("Incomplete or inconsistent global blind review")
    calls = screens.get("global_blind_calls", [])
    if not calls:
        raise ValueError("Missing independent blind calls")
    historical = {cid for call in calls for label, cid in call.get("clip_mapping", {}).items() if label != "CURRENT"}
    expected = {c["id"] for c in sample["candidates"]} if sample["task"] == "past" else set()
    if historical != expected:
        raise ValueError("The accepted audit did not cover the complete causal history")
    extra_calls = [screens.get("evidence_only_call", {}), screens.get("training_context_call", {})]
    extra_calls += list(screens.get("calls", {}).values())
    for call in calls + extra_calls:
        if (call.get("protocol_version") != PROTOCOL_VERSION
                or call.get("reader_input_protocol") != READER_INPUT_PROTOCOL
                or call.get("proposed_evidence_locations_visible") is not False
                or call.get("annotation_window_visible") is not False
                or call.get("context_isolation") != "single_question"):
            raise ValueError("A repaired check exposes privileged locating information or uses another protocol")
    for stage in ["evidence_only", "training_context"]:
        if not answer_passed(screens.get(stage, {}), screens.get(stage + "_call", {})):
            raise ValueError("Independent supporting evidence or training context failed")
    support = {cid for label, cid in screens["evidence_only_call"]["clip_mapping"].items() if label != "CURRENT"}
    if not support <= set(sample["positive_clip_ids"]) or sample["task"] == "past" and not support:
        raise ValueError("The independent evidence check included other historical context")
    for stage in ["reference_only", "no_concept"]:
        if not ablation_passed(screens.get(stage, {})) or stage not in screens.get("calls", {}):
            raise ValueError("Missing repaired information-dependence check")
    context = set(sample.get("context_positive_clip_ids", sample["positive_clip_ids"])) | set(sample["negative_clip_ids"])
    context.discard(sample["current_clip"]["id"])
    for call in [screens["training_context_call"], screens["calls"]["no_concept"]]:
        mapped = {cid for label, cid in call["clip_mapping"].items() if label != "CURRENT"}
        if mapped != context:
            raise ValueError("Teacher checks and actual supervised historical context differ")
