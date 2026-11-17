"""New-question generation and explicit negative checks under the V4 protocol."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import re
import runpy

from .batch_generate import compile_global, normalize_references, stage_call
from .io import read_json
from .schema import validate_sample
from .stream_audit import (PROTOCOL_VERSION, ablation_passed, aggregate_blind,
    answer_passed, audit_content, history_chunks, require_repaired_sample,
    response_validator, review_call)


# Reuse the reviewed implementation without changing its code/cache signature.
_repair = runpy.run_path(str(Path(__file__).resolve().parents[2] / "scripts/repair_dataset.py"),
                        run_name="v4_repair_library")
finish_quality = _repair["finish_quality"]
rewrite_is_compatible = _repair["rewrite_is_compatible"]
GENERATION_VERSION = "replenish-v4-new-facts-v2"

NEW_QUESTIONS = """Generate NEW personalized visual questions from the supplied
pixel-derived annotation. The same source is a continuous stream beginning at
source time zero, not at this annotation window. No window boundary is shown to
the eventual reader. Do not copy or paraphrase any original benchmark questions.

Return at most SIX questions: prefer four historical and two current, only when
visually supported. Vary the event and queried attribute; each must be a different
fact. The excluded event/attribute/task keys below are already retained facts;
do not generate them again. No swap pairs or option-rotation duplicates.

Use only the fixed annotated concepts {C0}, {C1}, etc. No real character names,
dialogue, subtitles, inferred intentions or story knowledge. The query is at
relative time 64, current is [56,64), past evidence is entirely before 56.
Every question must need arbitrary concept binding AND video beyond the reference
picture. References/appearance descriptions must not already answer it. A past
answer must not remain visible in the current clip. Prefer transient motion,
changed objects and interactions after the concept's reference frame.

At least FOUR questions, when possible, should query identity/recipient/relation
with at least TWO different runtime-name answer options. Ask which registered
actor performs a transient visible action or receives something; references must
not already show the answer. Without definitions the option names are arbitrary.
Use three/four registered-name options where available; with only two registered
names, 'An unregistered person' and 'No one' are distinct identity distractors.
Do not invent extra concept IDs. Use fewer questions if these are unsupported.

An attribute question without runtime-name options is allowed ONLY if another
registered actor has a DIFFERENT option answer to exactly the same neutral
question in the same scene/moment. Supply binding_witness with target_id,
alternative_id, alternative_answer_index and alternative_event_ids. These are
generation provenance, not answer hints. No witness: do not generate the question.
The target's action must not be presupposed. 'What does {C0} pour?' identifies the
only pouring actor even with no definitions; 'What is {C0} holding while the group
stands at the counter?' can work if different actors hold different objects.

Use neutral target-conditioned questions with more than one plausible actor in
the shared scene, or identity/recipient questions with runtime-name options.
An action presupposition must not uniquely identify the target without a concept
definition. A scene/event anchor may identify the time, but must not reveal the
queried fact or the answer. Do not ask about standing versus looking: those can
both be true. Four options must be mutually exclusive, precise, and have the same
semantic role. Do not use None/All-of-the-above or unanswerable distractor roles.

Historical wording needs a short, distinct visible scene/event anchor meaningful
in the ENTIRE stream. Never use numeric times, frame/clip IDs, the annotation
window, or unqualified 'beginning/start of the video/scene'. 'Now', 'in the current
clip', and 'just before the current clip' are legitimate when the event matches.
Current questions explicitly say 'now' or 'in the current clip'.

Use only visually_clear annotated events, and supporting relative intervals
intersecting at most THREE 8-second clips. For historical questions propose up to
two DIFFERENT local historical clip indices in negative_clip_indices as candidates
for independent checking; do not assume they are verified negatives.
Attribute is one of object,container,action,location,order,recipient,relation.
Correct answer positions should vary. Use fewer questions if support is limited.

Return JSON only: {"questions":[{"id":"q0","task":"past",
"concept_ids":["C0","C1","C2"],"question":"Who performs the observed action at the visually anchored moment?",
"options":["{C0}","{C1}","{C2}","An unregistered person"],
"answer_index":0,"evidence":[[24,27]],"event_ids":["E0"],
"attribute":"relation","negative_clip_indices":[1,5]}]}.
Attribute-only questions additionally need this provenance field:
"binding_witness":{"target_id":"C0","alternative_id":"C1",
"alternative_answer_index":1,"alternative_event_ids":["E1"]}.
The example is a schema, never copy nonexistent concept/event IDs or actions.
"""

NEGATIVE_CHECK = """Independently inspect candidate historical clips for the one
question below. No candidate is a presumed negative. Decide from the displayed
sampled pixels which historical H clips definitely do NOT support the requested
actor/event/attribute, for ANY answer option. An absent requested actor, a clearly
incompatible action/object, or a different event can establish a negative in the
sampled clip. Judge the shown frames; do not demand proof about unsampled moments.

Explicitly list EVERY H clip with such visible non-support in confirmed_negative_labels.
Do not leave the list empty when clear non-support is visible. Uncertain clips,
possible partial evidence, and clips containing another valid occurrence must NOT
be negatives. Report any actual supported occurrence instead. Temporal words use
the continuous source clock. Missing context alone does not make a question unclear.
No proposed answer, evidence time or annotation-window boundary is provided.
Ignore subtitles/dialogue and real character/story knowledge.

Use the same JSON schema: {"questions":[{"id":"item",
"concept_bindings_valid":true,"question_clear":true,
"occurrences":[],"confirmed_negative_labels":["H0"],
"reason":"specific visible support/non-support for each candidate"}]}.
If a supported occurrence is visible, each occurrences entry must use
{"clip_labels":["H1"],"answer_index":0}: clip_labels is a nonempty list of
displayed historical H labels, and answer_index is the integer 0, 1, 2, or 3
for the DISPLAYED option order. Do not use clip/answer text fields instead.
This is a format example, not a proposed occurrence or answer. If no occurrence
is supported, use an empty occurrences list; never invent one to fill the schema.
Only displayed historical H labels may be negatives. Use empty lists or false/null
when uncertain; never invent a negative to meet a requested count.
"""


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def negative_check(teacher, directory, sample, history, suffix):
    seed = sample["sample_id"] + ":" + GENERATION_VERSION + ":negative:" + str(suffix)
    content, order, mapping = audit_content(sample, history, "blind", seed)
    from .multimodal import VERIFIER_INPUT_PROTOCOL as READER_INPUT_PROTOCOL, stream_clock_text
    content[0]["text"] = NEGATIVE_CHECK + "\n" + stream_clock_text(sample)
    fingerprint = digest(content)
    stage = "negative_" + str(suffix) + "_" + GENERATION_VERSION + "_" + fingerprint[:12]
    response, call = stage_call(teacher, directory, stage, "visual_verification", content,
                                response_validator("blind", mapping, sample["task"]))
    return response["questions"][0], {**call, "option_permutation": order,
        "expected_display_index": order.index(sample["answer_index"]), "clip_mapping": mapping,
        "protocol_version": PROTOCOL_VERSION, "reader_input_protocol": READER_INPUT_PROTOCOL,
        "request_content_sha256": fingerprint, "proposed_evidence_locations_visible": False,
        "annotation_window_visible": False, "context_isolation": "single_question",
        "negative_confirmation": True}


def choose_negative_candidates(sample, positive_ids, used=(), limit=2):
    proposed = set(sample.get("candidate_negative_clip_ids", []))
    excluded = set(positive_ids) | set(used)
    positive_clips = [c for c in sample["candidates"] if c["id"] in positive_ids]
    centers = [(c["start"] + c["end"]) / 2 for c in positive_clips]
    def priority(clip):
        distance = min((abs((clip["start"] + clip["end"]) / 2 - p) for p in centers), default=0)
        return (clip["id"] not in proposed, distance, clip["start"])
    return sorted((c for c in sample["candidates"] if c["id"] not in excluded), key=priority)[:limit]


def make_candidates(window, annotation, generation, generation_call, annotation_provenance):
    proposal, invalid, corrections = normalize_references({"concepts": annotation["concepts"],
                                                          "questions": generation["questions"]})
    source = read_json(window["source_manifest"])
    valid_events = {e["id"] for e in annotation["events"] if e.get("visually_clear") is True}
    result = []
    known = {c["id"] for c in annotation["concepts"]}
    for question in proposal["questions"]:
        if question.get("swap_pair_id") or question.get("counterfactual_group"):
            invalid.append({"id": question["id"], "reasons": ["swap_augmentation_not_a_new_fact_family"]})
            continue
        if not question.get("event_ids") or not set(question["event_ids"]) <= valid_events:
            invalid.append({"id": question["id"], "reasons": ["unsupported_annotated_event"]})
            continue
        if question.get("attribute") not in {"object", "container", "action", "location", "order", "recipient", "relation"}:
            invalid.append({"id": question["id"], "reasons": ["unsupported_attribute"]})
            continue
        option_concepts = set(re.findall(r"\{(C[0-9])\}", " ".join(question["options"])))
        if len(option_concepts) < 2:
            witness = question.get("binding_witness", {})
            if not isinstance(witness, dict):
                witness = {}
            if (witness.get("target_id") not in known or witness.get("alternative_id") not in known
                    or witness.get("target_id") == witness.get("alternative_id")
                    or type(witness.get("alternative_answer_index")) is not int
                    or not 0 <= witness["alternative_answer_index"] < 4
                    or witness["alternative_answer_index"] == question["answer_index"]
                    or not witness.get("alternative_event_ids")
                    or not set(witness["alternative_event_ids"]) <= valid_events
                    or "{" + str(witness.get("target_id")) + "}" not in question["question"]
                    or not any(e.get("id") in witness["alternative_event_ids"] and e.get("actor") == witness["alternative_id"]
                               for e in annotation["events"])):
                invalid.append({"id": question["id"], "reasons": ["missing_concept_binding_witness"]})
                continue
        sample = compile_global(window, proposal, question, {}, generation_call, source, [])
        sample["sample_id"] = GENERATION_VERSION + ":" + window["window_id"] + ":g" + str(generation_call["request_id"]) + ":" + question["id"]
        if not rewrite_is_compatible(sample, sample["question"]) or len(sample["positive_clip_ids"]) > 3:
            invalid.append({"id": question["id"], "reasons": ["invalid_scope_or_media_limit"]})
            continue
        sample["candidate_negative_clip_ids"] = [window["clips"][i]["id"] for i in question.get("negative_clip_indices", [])]
        sample["new_generation_provenance"] = {"protocol": GENERATION_VERSION,
            "annotation": annotation_provenance, "generation_call": generation_call,
            "new_question_not_relabelled_old_sample": True,
            "proposed_question": copy.deepcopy(question)}
        validate_sample(sample)
        result.append(sample)
    return result, invalid, corrections


def verify_new_candidate(teacher, directory, sample, max_frames=224, max_bytes=12 * 1024 * 1024):
    # This exact reference-only input is reused by the final check: history and
    # its negatives are absent in this mode, so moving it earlier saves calls.
    reduced, _ = review_call(teacher, directory, sample, [], "reference_only")
    if not ablation_passed(reduced):
        return None, ["reference_only_dependence_not_established"]
    reviews = [review_call(teacher, directory, sample, chunk, "blind", str(i))
               for i, chunk in enumerate(history_chunks(sample, max_frames, 2, max_bytes))]
    blind = aggregate_blind(reviews, sample)
    if not blind["accepted"]:
        return None, blind["reasons"]
    used = set()
    if sample["task"] == "past":
        for attempt in range(2):
            if blind["negative_clip_ids"]:
                break
            candidates = choose_negative_candidates(sample, blind["positive_clip_ids"], used)
            if not candidates:
                break
            used.update(c["id"] for c in candidates)
            reviews.append(negative_check(teacher, directory, sample, candidates, str(attempt)))
            blind = aggregate_blind(reviews, sample)
            if not blind["accepted"]:
                return None, blind["reasons"]
        if sample["split"] == "train" and not blind["negative_clip_ids"]:
            return None, ["no_verified_retrieval_negative"]
    occurrences = [o for o in blind["occurrences"] if len(o["clip_ids"]) <= 3]
    if not occurrences:
        return None, ["support_exceeds_original_media_limit"]
    context = min(occurrences, key=lambda o: (len(o["clip_ids"]), o["clip_ids"]))["clip_ids"]
    return finish_quality(teacher, directory, sample, blind, [c for _, c in reviews], context,
        blind["negative_clip_ids"][:2], {"kind": "new_generation_already_scoped", "api_request_made": False}, sample)


def augment_seed_group(teacher, directory, group, originals):
    """Preserve seed approvals; add only explicitly verified common negatives."""
    if group[0]["task"] != "past" or all(s["negative_clip_ids"] for s in group):
        return group, "unchanged"
    positive_ids = set().union(*(set(s["positive_clip_ids"]) for s in group))
    used = set()
    for attempt in range(2):
        probe = copy.deepcopy(group[0])
        probe["candidate_negative_clip_ids"] = list(set().union(*(set(originals[s["sample_id"]]["negative_clip_ids"]) for s in group)))
        candidates = choose_negative_candidates(probe, positive_ids, used)
        if not candidates:
            break
        used.update(c["id"] for c in candidates)
        responses, common = [], {c["id"] for c in candidates}
        for sample in group:
            subdir = directory / digest(sample["sample_id"])[:16]
            subdir.mkdir(parents=True, exist_ok=True)
            row, call = negative_check(teacher, subdir, sample, candidates, "seed" + str(attempt))
            # A conflicting or uncertain observation never authorizes negatives.
            if (row.get("question_clear") is not True or row.get("concept_bindings_valid") is not True
                    or row["occurrences"]):
                common.clear()
            else:
                common &= {call["clip_mapping"][label] for label in row["confirmed_negative_labels"] if label != "CURRENT"}
            responses.append((row, call))
        if not common:
            continue
        negatives = sorted(common)[:2]
        augmented = []
        for sample, (row, call) in zip(group, responses):
            updated = copy.deepcopy(sample)
            chosen = set(sample.get("context_positive_clip_ids", sample["positive_clip_ids"])) | set(negatives)
            history = [c for c in sample["candidates"] if c["id"] in chosen]
            subdir = directory / digest(sample["sample_id"])[:16]
            context, context_call = review_call(teacher, subdir, sample, history, "training_context", "negative_aug")
            if not answer_passed(context, context_call):
                return group, "context_recheck_failed_seed_preserved"
            reduced, reduced_call = review_call(teacher, subdir, sample, history, "no_concept", "negative_aug")
            if not ablation_passed(reduced):
                return group, "concept_recheck_failed_seed_preserved"
            updated["negative_clip_ids"] = negatives
            screens = updated["quality_screens"]
            screens["global_blind"]["negative_clip_ids"] = sorted(set(screens["global_blind"]["negative_clip_ids"]) | common)
            screens["global_blind_calls"].append(call)
            screens["training_context"], screens["training_context_call"] = context, context_call
            screens["no_concept"], screens["calls"]["no_concept"] = reduced, reduced_call
            updated["verification"]["global_blind_calls"] = copy.deepcopy(screens["global_blind_calls"])
            updated["verification"]["training_context_call"] = context_call
            updated["negative_augmentation"] = {"source_sample_sha256": digest(sample),
                "negative_response": row, "negative_call": call, "new_context_checked": True}
            validate_sample(updated)
            require_repaired_sample(updated)
            augmented.append(updated)
        return augmented, "augmented"
    return group, "no_confirmed_negatives_seed_preserved"
