"""Versioned answer-unique NarrativeMovies qualification.

This isolated wrapper retains the original V4 prefix, negative, evidence,
training-context and information-dependence checks. Independent blind refinement
distinguishes alternative sufficient groups from jointly necessary clips. A
positive union is a relevance summary, never an asserted conjunction. Nothing in
this module calls a teacher until ``qualify_candidate`` is explicitly invoked.
"""
from __future__ import annotations

import copy
import fcntl
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

from . import narrative_qualification as v1
from . import narrative_temporal as temporal_v1
from . import narrative_temporal_v2 as temporal
from .io import read_json, sha256_file, write_json
from .multimodal import VERIFIER_INPUT_PROTOCOL as READER_INPUT_PROTOCOL, stream_clock_text
from .replenishment import NEGATIVE_CHECK, choose_negative_candidates
from .schema import validate_sample
from .stream_audit import (
    PROTOCOL_VERSION, ablation_passed, aggregate_blind, answer_passed,
    history_chunks, require_repaired_sample, response_validator,
)
from .teacher import image_part

PROTOCOL = "narrative-answer-unique-v4-qualification-v2"
GROUP_PROTOCOL = "narrative-blind-sufficient-groups-v2"

SCOPE = """Respect EVERY restriction in the complete question: the scene,
subject domain, and temporal qualification. A question restricted to registered
entities does not make an unregistered person a competing answer. A locating clue
does not broaden the question's domain. Multiple occurrences or temporal anchors
are allowed when all reasonable interpretations yield the same displayed answer;
their number alone does not make the question ambiguous. Inspect other choices
as well. Conflicting answers or incomplete verification cannot establish a unique
answer. Use only the supplied question, runtime references, pixels and clocks.
"""

GROUP_PROMPT = """Independently inspect the ONLY question and the supplied pixels.
These historical clips were independently observed; no proposed answer, original
annotation, or proposed support grouping is supplied. Ignore movie knowledge,
subtitles and dialogue. Determine sufficient evidence groups from the pixels.
A support group is an AND: ALL of its listed clips are jointly needed to answer.
Different groups are OR alternatives: EACH group must suffice on its own with the
runtime references and CURRENT. A long list of clips is not automatically an AND,
and independent repeated occurrences must not be merged into one long AND. Do not
automatically split a list either. Inspect all reasonable supported interpretations
and options. Use a minimal necessary observed clip set for each occurrence, with
no claim about unsampled absence/continuity. Report genuine required groups longer
than three clips unchanged; never trim support to fit a reader budget. Report all
groups you can establish from this input. Mark incomplete or uncertain instead of
guessing or omitting a conflicting answer. CURRENT is support only for a current
question; historical H clips are support for a past question.
Return JSON only with exactly question_clear (boolean), concept_bindings_valid
(boolean), review_complete (boolean), uncertain (boolean), support_groups (list).
Each group has exactly answer_index (displayed option index 0..3) and clip_labels
(nonempty list of distinct actual displayed support labels). Empty groups list is
allowed and does not by itself assert a conflicting answer.
"""


def _digest(value):
    return v1._digest(value)


def _implementations():
    paths = {Path(__file__).resolve()}
    for function in (v1.persistent_stage_call, v1.measured_audit_content,
                     temporal.plan_joint_requests, temporal_v1.plan_joint_requests,
                     v1.stage_call, image_part, stream_clock_text, choose_negative_candidates,
                     require_repaired_sample, validate_sample):
        paths.add(Path(inspect.getsourcefile(function)).resolve())
    return {str(path): sha256_file(path) for path in sorted(paths)}


def _visible_seed(sample, mode, suffix, negative=False):
    visible = {key: sample[key] for key in ("task", "question", "options", "visible_start", "question_time")}
    visible["concepts"] = [{key: c[key] for key in ("name", "description", "image", "known_at")}
                           for c in sample["concepts"]]
    return _digest(visible)+":"+PROTOCOL+":"+mode+":"+str(suffix)+("extra-negative" if negative else "")


def measured_audit_content(sample, history, mode, seed, *, image_encoder=None, negative=False, refinement=False):
    """The original measured renderer with the new, explicit scope policy."""
    content, order, mapping, measured = v1.measured_audit_content(sample, history, mode, seed,
                                                               image_encoder=image_encoder)
    if negative:
        if mode != "blind" or sample["task"] != "past" or refinement:
            raise ValueError("Only past blind calls can explicitly confirm negatives")
        content[0]["text"] = NEGATIVE_CHECK+"\n"+stream_clock_text(sample)
    if refinement:
        if mode != "blind":
            raise ValueError("Support-group refinement is an independent blind check")
        content[0]["text"] = GROUP_PROMPT+"\n"+stream_clock_text(sample)
    content[0]["text"] += "\n"+SCOPE
    return content, order, mapping, measured


def _group_validator(mapping, task):
    def validate(response):
        if not isinstance(response, dict) or set(response) != {
                "question_clear", "concept_bindings_valid", "review_complete", "uncertain", "support_groups"}:
            raise ValueError("Support refinement needs its complete versioned schema")
        if any(type(response[key]) is not bool for key in
               ("question_clear", "concept_bindings_valid", "review_complete", "uncertain")):
            raise ValueError("Support refinement flags must be boolean")
        if not isinstance(response["support_groups"], list):
            raise ValueError("Support groups must be a list")
        for group in response["support_groups"]:
            if not isinstance(group, dict) or set(group) != {"answer_index", "clip_labels"}:
                raise ValueError("A support group needs only answer and actual labels")
            labels, answer = group["clip_labels"], group["answer_index"]
            if (type(answer) is not int or not 0 <= answer < 4 or not isinstance(labels, list)
                    or not labels or any(not isinstance(label, str) for label in labels)
                    or len(set(labels)) != len(labels)
                    or any(label not in mapping or task == "past" and label == "CURRENT"
                           or task == "current" and label != "CURRENT" for label in labels)):
                raise ValueError("A support group cites invalid answers or unshown support")
        return response
    return validate


def _source_binding(sample):
    return {key: copy.deepcopy(sample[key]) for key in
            ("source_id", "source_sha256", "source_group", "split", "benchmark_source_policy", "benchmark_overlap")}


def _request(sample, history, mode, suffix, encoder, *, negative=False, refinement=False):
    if type(negative) is not bool or type(refinement) is not bool:
        raise ValueError("Audit request flags must be explicit booleans")
    seed = _visible_seed(sample, mode, suffix, negative)
    content, order, mapping, measured = measured_audit_content(sample, history, mode, seed,
        image_encoder=encoder, negative=negative, refinement=refinement)
    metadata = {"option_permutation": order, "expected_display_index": order.index(sample["answer_index"]),
        "clip_mapping": mapping, "protocol_version": PROTOCOL_VERSION,
        "reader_input_protocol": READER_INPUT_PROTOCOL, "narrative_rendering_protocol": PROTOCOL,
        "measured_frame_timestamps": measured, "request_content_sha256": _digest(content),
        "proposed_evidence_locations_visible": False, "annotation_window_visible": False,
        "context_isolation": "single_question", "qualification_mode": mode, "qualification_suffix": str(suffix),
        "shown_history_ids": [clip["id"] for clip in history], "source_provenance": _source_binding(sample),
        "negative_confirmation": negative, "support_refinement": refinement}
    return content, metadata


def _review(teacher, directory, sample, history, mode, suffix, *, encoder, negative=False, refinement=False):
    content, metadata = _request(sample, history, mode, suffix, encoder, negative=negative, refinement=refinement)
    validator = _group_validator(metadata["clip_mapping"], sample["task"]) if refinement else response_validator(mode, metadata["clip_mapping"], sample["task"])
    task = "visual_ablation" if mode in {"reference_only", "no_concept"} else "visual_verification"
    name = ("groups" if refinement else "negative" if negative else mode)+"_"+str(suffix)+"_"+metadata["request_content_sha256"][:20]
    response, call = v1.persistent_stage_call(teacher, directory, name, task, content, validator,
        binding_context={"protocol": PROTOCOL, "implementation_sha256": _implementations(),
                         "source_provenance": _source_binding(sample)})
    temporal_v1._call_receipt(call, SimpleNamespace(content=content))
    row = response if refinement else response["questions"][0]
    return {"response": row, "call": {**call, **metadata}}


def _validate_record(sample, record, encoder):
    if not isinstance(record, dict) or set(record) != {"response", "call"}:
        raise ValueError("A complete independent response and actual call are required")
    call = record["call"]
    by_id = {clip["id"]: clip for clip in sample["candidates"]}
    ids = call.get("shown_history_ids", [])
    if not isinstance(ids, list) or len(set(ids)) != len(ids) or any(cid not in by_id for cid in ids):
        raise ValueError("A qualified record has invalid source clips")
    content, metadata = _request(sample, [by_id[cid] for cid in ids], call.get("qualification_mode"),
        call.get("qualification_suffix"), encoder, negative=call.get("negative_confirmation"),
        refinement=call.get("support_refinement"))
    if any(call.get(key) != value for key, value in metadata.items()):
        raise ValueError("Qualified interpretation metadata does not match its actual request")
    temporal_v1._call_receipt(call, SimpleNamespace(content=content))
    if call["support_refinement"]:
        _group_validator(call["clip_mapping"], sample["task"])(record["response"])
    else:
        response_validator(call["qualification_mode"], call["clip_mapping"], sample["task"])(
            {"questions": [record["response"]]})
    return record


def _wrong_answer(sample, record, groups=False):
    row, call = record["response"], record["call"]
    values = row["support_groups"] if groups else row["occurrences"]
    return any(call["option_permutation"][event["answer_index"]] != sample["answer_index"] for event in values)


def _temporal_reviews(teacher, directory, sample, chunks, *, max_joint_requests, encoder):
    requests = temporal.build_discovery_requests(sample, [[c["id"] for c in chunk] for chunk in chunks],
        frame_timestamps=sample["frame_timestamps"], seed=42, image_encoder=encoder)
    discoveries = []
    for index, request in enumerate(requests):
        result = v1.persistent_stage_call(teacher, directory, "temporal_v2_discovery_"+request.binding_sha256,
            request.task, request.content, lambda value, req=request: temporal.validate_discovery_response(req, value),
            binding_context={"protocol": PROTOCOL, "implementation_sha256": _implementations()})
        receipt = temporal.record_discovery(request, result)
        discoveries.append(receipt)
        write_json(directory / f"temporal_v2_discovery_{index}.json", receipt)
        flags = receipt["flags"]
        if (any(flags.get(key) is not True for key in
                ("question_clear", "concept_bindings_valid", "discovery_complete_for_shown_pixels"))
                or flags.get("uncertain") is not False):
            return None, [], {"protocol": temporal.PROTOCOL, "status": "deferred",
                "reason": "uncertain_or_incomplete_temporal_discovery"}, None
    plan = temporal.plan_joint_requests(sample, discoveries, max_joint_requests=max_joint_requests,
        frame_timestamps=sample["frame_timestamps"], seed=42, image_encoder=encoder)
    joints = []
    if plan["status"] == "ready":
        for index, request in enumerate(plan["requests"]):
            result = v1.persistent_stage_call(teacher, directory, "temporal_v2_joint_"+request.binding_sha256,
                request.task, request.content, lambda value, req=request: temporal.validate_joint_response(req, value),
                binding_context={"protocol": PROTOCOL, "implementation_sha256": _implementations()})
            receipt = temporal.record_joint(request, result)
            joints.append(receipt)
            write_json(directory / f"temporal_v2_joint_{index}.json", receipt)
            if any(event["answer_index"] != sample["answer_index"] for event in receipt["occurrences"]):
                return plan, joints, {"protocol": temporal.PROTOCOL, "status": "rejected",
                    "reason": "temporal_blind_answer_disagrees"}, None
    verdict = temporal.assess_temporal_gate(plan, joints)
    snapshot = {key: value for key, value in plan.items() if key != "requests"}
    snapshot["request_bindings"] = [request.binding_sha256 for request in plan["requests"]]
    saved = {"verdict": verdict, "plan": snapshot, "discovery_receipts": discoveries, "joint_receipts": joints}
    write_json(directory / "temporal_v2_gate.json", saved)
    return plan, joints, verdict, saved


def _refine_groups(teacher, directory, sample, blind, *, encoder, max_support_groups):
    clips = {c["id"]: c for c in sample["candidates"]+[sample["current_clip"]]}
    candidate_sets = sorted({tuple(sorted(event["clip_ids"])) for event in blind["occurrences"]})
    records, groups = [], {}
    for ids in candidate_sets:
        history = sorted((clips[cid] for cid in ids if cid != sample["current_clip"]["id"]), key=lambda c: c["start"])
        record = _review(teacher, directory, sample, history, "blind", "refinement_"+_digest(ids)[:20],
                         encoder=encoder, refinement=True)
        records.append(record)
        row = record["response"]
        if _wrong_answer(sample, record, groups=True):
            return [], records, ["support_refinement_answer_disagrees"]
        if (not all(row[key] for key in ("question_clear", "concept_bindings_valid", "review_complete"))
                or row["uncertain"]):
            return [], records, ["uncertain_or_incomplete_support_refinement"]
        for group in row["support_groups"]:
            ids = sorted(record["call"]["clip_mapping"][label] for label in group["clip_labels"])
            groups.setdefault(tuple(ids), {"required_clip_ids": ids, "answer_index": sample["answer_index"],
                "origin": GROUP_PROTOCOL, "discovery_request_sha256": record["call"]["request_content_sha256"]})
        if len(groups) > max_support_groups:
            return [], records, ["support_group_overflow_deferred"]
    return list(groups.values()), records, []


def _temporal_candidates(temporal_gate):
    unique = {}
    for group in temporal_gate["verdict"]["alternative_support_sets"]:
        key = tuple(sorted(group["required_clip_ids"]))
        unique.setdefault(key, {**group, "origin": temporal.PROTOCOL, "required_clip_ids": list(key),
                                "interpretations": []})["interpretations"].append(group)
    return list(unique.values())


def _refined_candidates(sample, records):
    groups = {}
    for record in records:
        for group in record["response"]["support_groups"]:
            ids = sorted(record["call"]["clip_mapping"][label] for label in group["clip_labels"])
            groups.setdefault(tuple(ids), {"required_clip_ids": ids, "answer_index": sample["answer_index"],
                "origin": GROUP_PROTOCOL, "discovery_request_sha256": record["call"]["request_content_sha256"]})
    return list(groups.values())


def _check_groups(teacher, directory, sample, groups, negatives, reference, *, encoder):
    """Each group is tested once. Semantic failures never trigger a new vote."""
    by_id = {c["id"]: c for c in sample["candidates"]+[sample["current_clip"]]}
    verified, failed = [], []
    for group in sorted(groups, key=lambda item: (len(item["required_clip_ids"]), item["required_clip_ids"])):
        ids = group["required_clip_ids"]
        if len(ids) > 3:
            failed.append({**group, "reason": "support_exceeds_original_media_limit"})
            continue
        suffix = "group_"+_digest(ids)[:20]
        evidence_history = sorted((by_id[cid] for cid in ids if cid != sample["current_clip"]["id"]), key=lambda c: c["start"])
        context_history = sorted((by_id[cid] for cid in set(ids)|set(negatives) if cid != sample["current_clip"]["id"]), key=lambda c: c["start"])
        checks = {"reference_only": reference}
        failure = None
        for mode, history in (("evidence_only", evidence_history), ("training_context", context_history),
                              ("no_concept", context_history)):
            record = _review(teacher, directory, sample, history, mode, suffix, encoder=encoder)
            checks[mode] = record
            passed = ablation_passed(record["response"]) if mode == "no_concept" else answer_passed(record["response"], record["call"])
            if not passed:
                failure = {"evidence_only": "independent_evidence_sufficiency_failed",
                           "training_context": "actual_supervised_context_answer_failed",
                           "no_concept": "no_concept_dependence_not_established"}[mode]
                answer = record["response"].get("answer_index")
                if mode != "no_concept" and type(answer) is int and record["call"]["option_permutation"][answer] != sample["answer_index"]:
                    # A contrary independent answer is not a reason to shop
                    # other evidence groups for a preferred vote.
                    failed.append({**group, "reason": failure, "checks": checks,
                                   "contrary_independent_answer": True})
                    return [], failed
                break
        if failure:
            failed.append({**group, "reason": failure, "checks": checks})
        else:
            verified.append({**group, "negative_clip_ids": list(negatives), "checks": checks,
                             "all_original_independent_quality_gates_passed": True})
    return verified, failed


def _assemble(sample, blind, reviews, group, verified_groups, *, check_files):
    """The V4 schema/conditions are identical to v1._finish_quality."""
    repaired = copy.deepcopy(sample)
    clips = {clip["id"]: clip for clip in sample["candidates"]+[sample["current_clip"]]}
    checks, calls = group["checks"], [call for _, call in reviews]
    repaired["positive_clip_ids"] = blind["positive_clip_ids"]
    repaired["verified_retrieval_positive_clip_ids"] = sorted({cid for alternative in verified_groups
                                                              for cid in alternative["required_clip_ids"]})
    repaired["evidence"] = [[clips[cid]["start"], clips[cid]["end"]] for cid in blind["positive_clip_ids"]]
    repaired["context_positive_clip_ids"] = list(group["required_clip_ids"])
    repaired["negative_clip_ids"] = list(group["negative_clip_ids"])
    repaired["verification"] = {"accepted": True, "unique_answer": True, "concept_bindings_valid": True,
        "evidence_sufficient": True, "whole_causal_prefix_reviewed": True, "global_blind_calls": calls,
        "evidence_only": checks["evidence_only"]["response"], "evidence_only_call": checks["evidence_only"]["call"],
        "training_context_call": checks["training_context"]["call"]}
    screens = {"protocol": PROTOCOL_VERSION, "reader_input_protocol": READER_INPUT_PROTOCOL,
        "context_isolation": "single_question", "global_blind": blind, "global_blind_calls": calls,
        "calls": {mode: checks[mode]["call"] for mode in ("reference_only", "no_concept")}}
    for mode in ("reference_only", "no_concept", "evidence_only", "training_context"):
        screens[mode] = checks[mode]["response"]
        if mode in {"evidence_only", "training_context"}:
            screens[mode+"_call"] = checks[mode]["call"]
    repaired["quality_screens"] = screens
    repaired["repair_provenance"] = {"protocol": PROTOCOL_VERSION,
        "rewrite_call": {"kind": "new_generation_already_scoped", "api_request_made": False},
        "original_question": sample["question"], "original_answer_index": sample["answer_index"],
        "original_evidence": copy.deepcopy(sample["evidence"]), "original_positive_clip_ids": list(sample["positive_clip_ids"]),
        "answer_label_changed": False, "original_sample_sha256": _digest(sample)}
    validate_sample(repaired, check_files=check_files)
    require_repaired_sample(repaired)
    return repaired


def _quality_record(sample, group, record, mode, encoder):
    """Bind both successful and failed quality checks to this group's context."""
    _validate_record(sample, record, encoder)
    call = record["call"]
    if (call["qualification_mode"] != mode or call["support_refinement"] is not False
            or call["negative_confirmation"] is not False):
        raise ValueError("A sufficient group is missing an original quality gate")
    wanted = set() if mode == "reference_only" else set(group["required_clip_ids"])
    if mode in {"training_context", "no_concept"}:
        wanted |= set(sample["negative_clip_ids"])
    wanted.discard(sample["current_clip"]["id"])
    if set(call["shown_history_ids"]) != wanted:
        raise ValueError("An alternative quality check used another historical context")
    return record


def require_narrative_qualified(sample, *, image_encoder=None, check_files=True):
    """Dispatch old approvals to their original validator; strictly audit v2."""
    gate = sample.get("narrative_qualification", {})
    if gate.get("protocol") == v1.PROTOCOL:
        return v1.require_narrative_qualified(sample, image_encoder=image_encoder, check_files=check_files)
    if gate.get("protocol") != PROTOCOL or gate.get("accepted") is not True:
        raise ValueError("A known, accepted Narrative qualification protocol is required")
    require_repaired_sample(sample)
    validate_sample(sample, check_files=check_files)
    v1._check_candidate(sample, check_files=check_files, check_proposal=False)
    encoder = image_encoder or image_part
    records = gate.get("original_blind_receipts", [])
    if not records:
        raise ValueError("Original full-prefix response receipts are missing")
    for record in records:
        _validate_record(sample, record, encoder)
        if record["call"].get("qualification_mode") != "blind" or record["call"].get("support_refinement"):
            raise ValueError("Full-prefix review must be original blind input")
    reviews = [(record["response"], record["call"]) for record in records]
    temporal_gate = gate.get("temporal")
    if sample["qa_category"] == "Time-based":
        if not isinstance(temporal_gate, dict):
            raise ValueError("Time-based qualification needs versioned temporal receipts")
        verdict = temporal_gate.get("verdict", {})
        modules = {temporal.PROTOCOL: temporal, temporal_v1.PROTOCOL: temporal_v1}
        module = modules.get(verdict.get("protocol"))
        if module is None:
            raise ValueError("Unknown temporal receipt protocol")
        joints = temporal_gate.get("joint_receipts", [])
        plan = module.plan_joint_requests(sample, temporal_gate.get("discovery_receipts", []),
            max_joint_requests=max(1, len(joints)), frame_timestamps=sample["frame_timestamps"], seed=42,
            image_encoder=encoder)
        fresh = module.assess_temporal_gate(plan, joints)
        if fresh != verdict or fresh.get("status") != "passed" or fresh.get("observed_answer_indices") != [sample["answer_index"]]:
            raise ValueError("Temporal approval does not match reconstructed versioned requests")
        reviews = module.combine_with_v4_reviews(reviews, joints, plan, [c["id"] for c in sample["candidates"]],
            v4_protocol_version=PROTOCOL_VERSION, reader_input_protocol=READER_INPUT_PROTOCOL)
    elif temporal_gate is not None:
        raise ValueError("Unexpected temporal approval on a non-temporal question")
    blind = aggregate_blind(reviews, sample)
    if blind != sample["quality_screens"]["global_blind"] or [call for _, call in reviews] != sample["quality_screens"]["global_blind_calls"]:
        raise ValueError("Original full-prefix approval differs from stored independent responses")
    if not blind["accepted"]:
        raise ValueError("Original full-prefix review did not pass")
    refinement = gate.get("support_refinement_receipts", [])
    for record in refinement:
        _validate_record(sample, record, encoder)
        if not record["call"].get("support_refinement") or _wrong_answer(sample, record, groups=True):
            raise ValueError("Support refinement is not a matching independent blind review")
        row = record["response"]
        if not all(row[key] for key in ("question_clear", "concept_bindings_valid", "review_complete")) or row["uncertain"]:
            raise ValueError("Support refinement is incomplete")
    if gate.get("support_refinement_protocol") != GROUP_PROTOCOL:
        raise ValueError("The independent grouping protocol is missing")
    if not temporal_gate:
        expected_sets = {tuple(sorted(event["clip_ids"])) for event in blind["occurrences"]}
        actual_sets = [tuple(sorted(set(record["call"]["shown_history_ids"]) |
                        ({sample["current_clip"]["id"]} if sample["task"] == "current" else set())))
                       for record in refinement]
        if len(actual_sets) != len(set(actual_sets)) or set(actual_sets) != expected_sets:
            raise ValueError("Independent grouping did not cover every discovered support candidate")
    elif refinement:
        raise ValueError("Temporal groups must retain their independently verified joint interpretation")
    groups = gate.get("alternative_support_sets", [])
    if not groups or gate.get("support_union_is_required_conjunction") is not False:
        raise ValueError("Verified alternative support sets are missing")
    candidates = _temporal_candidates(temporal_gate) if temporal_gate else _refined_candidates(sample, refinement)
    independently_grouped = {tuple(group["required_clip_ids"]): group for group in candidates}
    unadopted = gate.get("unadopted_support_sets", [])
    attempted = [tuple(group.get("required_clip_ids", [])) for group in groups+unadopted]
    if len(attempted) != len(set(attempted)) or set(attempted) != set(independently_grouped):
        raise ValueError("Every independently grouped support must retain its verification outcome")
    for group in groups+unadopted:
        expected = independently_grouped[tuple(group["required_clip_ids"])]
        if any(group.get(key) != value for key, value in expected.items()):
            raise ValueError("A persisted support interpretation differs from its independent receipt")
    for group in groups:
        ids = group.get("required_clip_ids", [])
        if (not ids or len(ids) > 3 or len(set(ids)) != len(ids) or tuple(sorted(ids)) not in independently_grouped
                or not set(ids) <= set(sample["positive_clip_ids"]) or group.get("answer_index") != sample["answer_index"]
                or group.get("all_original_independent_quality_gates_passed") is not True):
            raise ValueError("An alternative group lacks independently established bounded support")
        if group.get("negative_clip_ids") != sample["negative_clip_ids"]:
            raise ValueError("Alternative groups must use the actual verified negatives")
        checks = group.get("checks", {})
        if set(checks) != {"evidence_only", "training_context", "reference_only", "no_concept"}:
            raise ValueError("An adopted group needs exactly the four original quality gates")
        for mode in ("evidence_only", "training_context", "reference_only", "no_concept"):
            record = checks.get(mode, {})
            _quality_record(sample, group, record, mode, encoder)
            call = record["call"]
            passed = ablation_passed(record["response"]) if mode in {"reference_only", "no_concept"} else answer_passed(record["response"], call)
            if not passed:
                raise ValueError("An alternative group did not pass every original quality gate")
    # Failed groups remain auditable; a contrary answer cannot be hidden among
    # unadopted groups while another group's preferred answer authorizes QA.
    for group in unadopted:
        if len(group["required_clip_ids"]) > 3:
            if group.get("reason") != "support_exceeds_original_media_limit" or group.get("checks"):
                raise ValueError("An oversized group must remain an explicit technical defer")
            continue
        checks = group.get("checks", {})
        failure = None
        order = ("reference_only", "evidence_only", "training_context", "no_concept")
        for index, mode in enumerate(order):
            if mode not in checks:
                raise ValueError("An unadopted group is missing its actual failure receipt")
            record = _quality_record(sample, group, checks[mode], mode, encoder)
            row, call = record["response"], record["call"]
            if mode in {"evidence_only", "training_context"}:
                answer = row.get("answer_index")
                if type(answer) is int and call["option_permutation"][answer] != sample["answer_index"]:
                    raise ValueError("An independent contrary answer invalidates answer uniqueness")
            passed = ablation_passed(row) if mode in {"reference_only", "no_concept"} else answer_passed(row, call)
            if not passed:
                failure = {"reference_only": "reference_only_dependence_not_established",
                           "evidence_only": "independent_evidence_sufficiency_failed",
                           "training_context": "actual_supervised_context_answer_failed",
                           "no_concept": "no_concept_dependence_not_established"}[mode]
                if set(checks) != set(order[:index+1]):
                    raise ValueError("An unadopted group must retain exactly its attempted gates")
                break
        if failure is None or failure != group.get("reason"):
            raise ValueError("An unadopted group's recorded failure is not supported by its actual result")
    verified_retrieval = sorted({cid for group in groups for cid in group["required_clip_ids"]})
    if sample.get("verified_retrieval_positive_clip_ids") != verified_retrieval:
        raise ValueError("Training retrieval positives must equal the verified sufficient-group union")
    selected = gate.get("selected_support_set_index")
    if type(selected) is not int or not 0 <= selected < len(groups):
        raise ValueError("The actual supervised support group is missing")
    chosen = groups[selected]
    if chosen["required_clip_ids"] != sample["context_positive_clip_ids"]:
        raise ValueError("The chosen sufficient set differs from actual supervised context")
    screens = sample["quality_screens"]
    for mode, record in chosen["checks"].items():
        call = screens["calls"].get(mode) if mode in {"reference_only", "no_concept"} else screens.get(mode+"_call")
        if screens.get(mode) != record["response"] or call != record["call"]:
            raise ValueError("Selected group receipts differ from the original V4 fields")


def qualify_candidate(teacher, directory, sample, *, max_frames=224, max_bytes=12*1024*1024,
                      max_joint_requests=32, max_support_groups=64, image_encoder=None, check_files=True):
    """Return a strict V4-qualified sample or reasons; transport/budget errors propagate."""
    v1._check_candidate(sample, check_files=check_files)
    if type(max_support_groups) is not int or max_support_groups < 1:
        raise ValueError("An explicit positive support-group overflow bound is required")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".qualification_v2.lock").open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        config = {"protocol": PROTOCOL, "sample_sha256": _digest(sample), "max_frames": max_frames,
                  "max_bytes": max_bytes, "max_joint_requests": max_joint_requests,
                  "max_support_groups": max_support_groups, "implementation_sha256": _implementations()}
        path = directory / "qualification_v2_input.json"
        if path.exists() and read_json(path) != config:
            raise ValueError("Frozen v2 qualification input/code changed; use a new directory")
        write_json(path, config)
        encoder = image_encoder or image_part

        def finish(accepted, reasons):
            write_json(directory / "qualification_v2_result.json", {"protocol": PROTOCOL, "sample": accepted, "reasons": reasons})
            return accepted, reasons

        reference = _review(teacher, directory, sample, [], "reference_only", "shared", encoder=encoder)
        if not ablation_passed(reference["response"]):
            return finish(None, ["reference_only_dependence_not_established"])
        chunks = history_chunks(sample, max_frames, 2, max_bytes)
        records = []
        for index, chunk in enumerate(chunks):
            record = _review(teacher, directory, sample, chunk, "blind", str(index), encoder=encoder)
            records.append(record)
            # Empty rows are legitimate. Only a parsed non-null contrary choice
            # can stop before remaining history; same-gold repeats never do.
            if _wrong_answer(sample, record):
                return finish(None, ["global_blind_answer_missing_conflicting_or_disagrees"])
        reviews = [(record["response"], record["call"]) for record in records]
        temporal_gate = None
        if sample["qa_category"] == "Time-based":
            plan, joints, verdict, temporal_gate = _temporal_reviews(teacher, directory, sample, chunks,
                max_joint_requests=max_joint_requests, encoder=encoder)
            if verdict.get("status") != "passed":
                return finish(None, ["temporal_"+str(verdict.get("reason", verdict.get("status")))])
            if verdict.get("observed_answer_indices") != [sample["answer_index"]]:
                return finish(None, ["temporal_blind_answer_disagrees"])
            reviews = temporal.combine_with_v4_reviews(reviews, joints, plan, [c["id"] for c in sample["candidates"]],
                v4_protocol_version=PROTOCOL_VERSION, reader_input_protocol=READER_INPUT_PROTOCOL)
        blind = aggregate_blind(reviews, sample)
        if not blind["accepted"]:
            return finish(None, blind["reasons"])
        if sample["task"] == "past":
            used = set()
            for attempt in range(2):
                if blind["negative_clip_ids"]:
                    break
                candidates = choose_negative_candidates(sample, blind["positive_clip_ids"], used)
                if not candidates:
                    break
                used.update(c["id"] for c in candidates)
                record = _review(teacher, directory, sample, candidates, "blind", str(attempt), encoder=encoder, negative=True)
                records.append(record)
                reviews = [(item["response"], item["call"]) for item in records]
                if temporal_gate:
                    reviews = temporal.combine_with_v4_reviews(reviews, joints, plan,
                        [c["id"] for c in sample["candidates"]], v4_protocol_version=PROTOCOL_VERSION,
                        reader_input_protocol=READER_INPUT_PROTOCOL)
                blind = aggregate_blind(reviews, sample)
                if not blind["accepted"]:
                    return finish(None, blind["reasons"])
            if sample["split"] == "train" and not blind["negative_clip_ids"]:
                return finish(None, ["no_verified_retrieval_negative"])
        if temporal_gate:
            groups = _temporal_candidates(temporal_gate)
            refinement = []
            # Identical support with distinct temporal interpretations retains
            # its semantic metadata but is checked once on identical pixels.
            reasons = ["support_group_overflow_deferred"] if len(groups) > max_support_groups else []
        else:
            groups, refinement, reasons = _refine_groups(teacher, directory, sample, blind, encoder=encoder,
                                                        max_support_groups=max_support_groups)
        if reasons:
            return finish(None, reasons)
        verified, failed = _check_groups(teacher, directory, sample, groups, blind["negative_clip_ids"][:2],
                                        reference, encoder=encoder)
        if not verified:
            reasons = sorted({group["reason"] for group in failed}) or ["no_independently_verified_sufficient_group"]
            return finish(None, reasons)
        accepted = _assemble(sample, blind, reviews, verified[0], verified, check_files=check_files)
        accepted["narrative_qualification"] = {"protocol": PROTOCOL, "accepted": True,
            "all_original_v4_gates_preserved": True, "measured_pts_rendering": True,
            "temporal": temporal_gate, "original_blind_receipts": records,
            "support_refinement_protocol": GROUP_PROTOCOL, "support_refinement_receipts": refinement,
            "alternative_support_sets": verified, "unadopted_support_sets": failed,
            "selected_support_set_index": 0, "support_union_is_required_conjunction": False,
            "positive_union_semantics": "relevance_summary_not_joint_requirement"}
        require_narrative_qualified(accepted, image_encoder=encoder, check_files=check_files)
        return finish(accepted, [])
