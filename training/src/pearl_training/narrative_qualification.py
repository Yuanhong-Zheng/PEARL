"""Measured-PTS NarrativeMovies qualification retaining every original V4 gate.

No original module globals are replaced. Candidate labels/evidence remain local
to the aggregator; independent payloads contain only causal pixels, runtime
references, observable clocks and the shuffled question. Time-based candidates
add full-prefix discovery and every planned joint temporal check before V4
aggregation. Neither timing checks nor a supplemental temporal pass accepts QA.
"""

from __future__ import annotations

import copy
import fcntl
import hashlib
import inspect
import json
import math
from pathlib import Path
import re
import sqlite3
import time

from .batch_generate import stage_call
from .io import read_json, sha256_file, write_json
from .multimodal import VERIFIER_INPUT_PROTOCOL as READER_INPUT_PROTOCOL, stream_clock_text
from .narrative_qa import causal_clips, evidence_clip_ids, validate_category
from .narrative_temporal import (
    PROTOCOL as TEMPORAL_PROTOCOL, assess_temporal_gate, build_discovery_requests,
    combine_with_v4_reviews, plan_joint_requests, record_discovery, record_joint,
    strict_direction, validate_discovery_response, validate_joint_response,
)
from .replenishment import NEGATIVE_CHECK, choose_negative_candidates
from .schema import validate_sample
from .stream_audit import (
    ABLATION, ANSWER, BLIND, PROTOCOL_VERSION, ablation_passed, aggregate_blind,
    answer_passed, history_chunks, require_repaired_sample, response_validator,
    shuffled_display,
)
from .teacher import TeacherBudgetExhausted, TeacherRetryExhausted, image_part


PROTOCOL = "narrative-measured-pts-v4-qualification-v1"
POLICY = "overlap_allowed_by_user"


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def _implementation_hashes(validator=None):
    functions = [stage_call, require_repaired_sample, plan_joint_requests, validate_sample, evidence_clip_ids]
    if validator is not None:
        functions.append(validator)
    paths = {Path(__file__).resolve()}
    for function in functions:
        source = inspect.getsourcefile(function)
        if source and Path(source).is_file():
            paths.add(Path(source).resolve())
    return {str(path): sha256_file(path) for path in sorted(paths)}


def _ledger_rows(teacher, task, prompt_hash):
    if not hasattr(teacher, "ledger"):
        return None
    path = Path(teacher.ledger)
    if not path.is_file():
        raise ValueError("The teacher's isolated SQL ledger is missing")
    with sqlite3.connect(path, timeout=30) as database:
        return database.execute("SELECT id,model,task,prompt_sha256,status FROM calls WHERE task=? AND prompt_sha256=?",
                                (task, prompt_hash)).fetchall()


def _checked_stage_result(result, teacher, task, content, prompt_hash, validator):
    response, call = result
    if validator:
        validator(response)
    if not isinstance(call, dict) or type(call.get("request_id")) is not int:
        raise ValueError("A stage result needs its actual request receipt")
    if hasattr(teacher, "model") and call.get("model") != teacher.model:
        raise ValueError("A stage receipt belongs to another teacher model")
    if call.get("finish_reason") != "stop":
        raise ValueError("A stage response did not finish normally")
    settings = call.get("generation_settings")
    if not isinstance(settings, dict):
        raise ValueError("A stage receipt lacks actual generation settings")
    payload = {"model": call.get("model"), "messages": [{"role": "user", "content": content}], **settings}
    actual = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    if call.get("request_payload_sha256") != actual:
        raise ValueError("A stage receipt does not bind this exact request body")
    ledger = _ledger_rows(teacher, task, prompt_hash)
    if ledger is not None and (call["request_id"], call.get("model"), task, prompt_hash, "ok") not in ledger:
        raise ValueError("A cached stage has no matching successful call in the isolated SQL ledger")
    return response, call


def _retry_pause(seconds):
    time.sleep(seconds)


class _OneAttemptTeacher:
    """Private adapter: never change retry settings on the shared real client."""
    max_retries = 0

    def __init__(self, teacher):
        self.teacher = teacher

    def chat_json(self, task, content):
        return self.teacher.chat_json(task, content)


def persistent_stage_call(teacher, directory, name, task, content, validator=None, *, max_attempts=6, binding_context=None):
    """Reuse V4 stage_call with a persistent, at-most-six dispatch budget.

    Reservations are persisted BEFORE dispatch and remain consumed after process
    interruption. Existing SQL rows (including reserved/failed calls) and error
    receipts also count. A per-prompt ledger-side lock prevents simultaneous
    helpers from overrunning the same prompt budget. Successful raw responses
    are revalidated against request/code hashes and their actual SQL receipt.
    """
    approved = getattr(teacher, "max_retries", 5)
    if type(max_attempts) is not int or not 1 <= max_attempts <= 6 or type(approved) is not int or not 0 <= approved <= 5:
        raise ValueError("Stage retries must preserve the approved one-to-six-attempt limit")
    limit = min(max_attempts, approved+1)
    directory = Path(directory)
    stage = directory / "persistent_stages" / _digest([name, task])[:32]
    stage.mkdir(parents=True, exist_ok=True)
    prompt_hash = hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()
    binding = {"protocol": PROTOCOL, "name": name, "task": task, "prompt_sha256": prompt_hash,
               "max_attempts": limit, "implementation_sha256": _implementation_hashes(validator),
               "client": {"ledger": str(Path(teacher.ledger).resolve()) if hasattr(teacher, "ledger") else None,
                          "model": getattr(teacher, "model", None),
                          "approval_sha256": _digest(teacher.approval) if hasattr(teacher, "approval") else None,
                          "endpoint_sha256": _digest(teacher.url) if hasattr(teacher, "url") else None},
               "binding_context": binding_context}
    locks = Path(teacher.ledger).parent / "narrative_stage_locks" if hasattr(teacher, "ledger") else stage
    locks.mkdir(parents=True, exist_ok=True)
    with (locks / (_digest([task, prompt_hash])+".lock")).open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        binding_path = stage / "binding.json"
        if binding_path.exists() and read_json(binding_path) != binding:
            raise ValueError("Frozen stage input/code binding changed")
        write_json(binding_path, binding)
        state_path = stage / "attempts.json"
        state = read_json(state_path) if state_path.exists() else {"binding_sha256": _digest(binding), "attempts": []}
        if state.get("binding_sha256") != _digest(binding):
            raise ValueError("Persistent attempt ledger belongs to another stage binding")
        completed = stage / "completed.json"
        if completed.exists():
            saved = read_json(completed)
            if saved.get("binding_sha256") != _digest(binding):
                raise ValueError("Completed stage belongs to a different input/code binding")
            return _checked_stage_result((saved["response"], saved["call"]), teacher, task, content, prompt_hash, validator)
        # A process may have ended after V4 saved its validated response but
        # before the outer completion marker. Recover that exact tuple for free.
        for attempt in reversed(state["attempts"]):
            raw = stage / f"attempt_{attempt['number']:02d}" / "call.json"
            if raw.exists():
                saved = read_json(raw)
                result = _checked_stage_result((saved["response"], saved["call"]), teacher, task, content, prompt_hash, validator)
                write_json(completed, {"binding_sha256": _digest(binding), "response": result[0], "call": result[1]})
                return result
        last_failure = {"category": "persistent_attempt_cap", "error": "Existing or interrupted reservations exhausted this stage"}
        while True:
            ledger = _ledger_rows(teacher, task, prompt_hash)
            previous_errors = list(directory.glob(name+"_error_*.json"))
            used = max(max((attempt["number"] for attempt in state["attempts"]), default=0),
                       len(ledger) if ledger is not None else 0, len(previous_errors))
            if used >= limit:
                raise TeacherRetryExhausted(name, limit, state.get("last_failure", last_failure))
            number = used+1
            reservation = {"number": number, "status": "reserved", "reserved_at": time.time(),
                           "observed_sql_attempts_before_dispatch": len(ledger) if ledger is not None else None}
            state["attempts"].append(reservation)
            write_json(state_path, state)
            attempt_dir = stage / f"attempt_{number:02d}"
            attempt_dir.mkdir(exist_ok=True)
            try:
                result = stage_call(_OneAttemptTeacher(teacher), attempt_dir, "call", task, content, validator)
                result = _checked_stage_result(result, teacher, task, content, prompt_hash, validator)
            except TeacherBudgetExhausted:
                reservation.update(status="global_budget_exhausted", finished_at=time.time())
                write_json(state_path, state)
                raise
            except (TeacherRetryExhausted, ValueError, KeyError, TypeError, IndexError) as error:
                last_failure = getattr(error, "last_failure", {"type": type(error).__name__, "error": str(error)})
                reservation.update(status="failed", finished_at=time.time(), failure=last_failure)
                state["last_failure"] = last_failure
                write_json(state_path, state)
                if number >= limit:
                    raise TeacherRetryExhausted(name, limit, last_failure) from error
                _retry_pause(max(min(2**number, 32), last_failure.get("retry_after_seconds") or 0))
            else:
                reservation.update(status="ok", finished_at=time.time(), request_id=result[1]["request_id"])
                write_json(state_path, state)
                write_json(completed, {"binding_sha256": _digest(binding), "response": result[0], "call": result[1]})
                return result


def _time(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be a measured finite nonnegative source time")
    return float(value)


def _check_candidate(sample, *, check_files=True, check_proposal=True):
    category = sample.get("qa_category")
    validate_category(category, sample.get("task"))
    if category == "Time-based":
        strict_direction(sample["question"])
    if sample.get("benchmark_source_policy") != POLICY:
        raise ValueError("Benchmark overlap authorization must be explicit")
    if sample.get("split") not in {"train", "validation"} or not sample.get("source_group"):
        raise ValueError("Frozen source group/split is required")
    if not re.fullmatch(r"[a-fA-F0-9]{64}", str(sample.get("source_sha256", ""))):
        raise ValueError("Explicit source SHA256 is required")
    overlap = sample.get("benchmark_overlap")
    if not isinstance(overlap, dict) or overlap.get("status") not in {"unknown", "known_shared", "known_disjoint"}:
        raise ValueError("Benchmark overlap status is required")
    start = _time(sample["visible_start"], "visible_start")
    query = _time(sample["question_time"], "question_time")
    times = sample.get("frame_timestamps")
    if not isinstance(times, dict) or not times:
        raise ValueError("Measured frame_timestamps are required; nominal FPS is not a fallback")
    clips = sample["candidates"] + [sample["current_clip"]]
    history, current = causal_clips(clips, query, visible_start=start)
    if [clip["id"] for clip in history] != [clip["id"] for clip in sample["candidates"]] or current != sample["current_clip"]:
        raise ValueError("Candidate pool must be exactly the complete causal prefix")
    known_paths = set()
    for clip in clips:
        if clip["id"] != f"{sample['source_id']}:{clip['start']:.3f}-{clip['end']:.3f}":
            raise ValueError("Clip ID does not preserve its actual source clock")
        previous = None
        if not clip["frames"]:
            raise ValueError("A causal clip has no observed pixels")
        for path in clip["frames"]:
            if not isinstance(path, str) or path not in times:
                raise ValueError("Every clip path needs an explicit measured PTS")
            timestamp = _time(times[path], "frame timestamp")
            if not clip["start"] <= timestamp < clip["end"] or timestamp >= query or previous is not None and timestamp <= previous:
                raise ValueError("Frame PTS is noncausal, unordered or outside its clip")
            if path in known_paths:
                raise ValueError("A frame path occurs in multiple causal clips")
            if check_files and not Path(path).is_file():
                raise ValueError("Missing observed source pixels")
            known_paths.add(path)
            previous = timestamp
    reference_frames = {}
    for frame in sample.get("reference_frames", []):
        path = frame.get("path")
        timestamp = _time(frame.get("timestamp"), "exact observed reference PTS")
        if (not isinstance(path, str) or path in reference_frames or path not in times
                or abs(_time(times[path], "reference frame PTS")-timestamp) > 1e-6
                or not start <= timestamp < query
                or any(frame.get(key) != sample[key] for key in ("source_id", "source_sha256", "source_group"))
                or not re.fullmatch(r"[a-fA-F0-9]{64}", str(frame.get("sha256", "")))):
            raise ValueError("Exact reference pixels need matching causal clock and source provenance")
        if check_files and sha256_file(path) != frame["sha256"].lower():
            raise ValueError("Exact observed reference pixel hash changed")
        reference_frames[path] = frame
    concepts = sample["concepts"]
    if not 1 <= len(concepts) <= 4 or len({concept["name"] for concept in concepts}) != len(concepts):
        raise ValueError("One to four distinct runtime concepts are required")
    for concept in concepts:
        if concept.get("kind") not in {"person", "object"}:
            raise ValueError("Only explicit person/object concepts are allowed; Action concepts are excluded")
        path = concept["image"]
        known = _time(concept["known_at"], "reference timestamp")
        if (path not in known_paths and path not in reference_frames or not start <= known < query
                or abs(_time(times[path], "reference PTS")-known) > 1e-6):
            raise ValueError("Reference must use an actual causal measured source frame")
        if not concept.get("name") or not concept.get("description"):
            raise ValueError("Runtime concept name/description is missing")
    options = sample["options"]
    if len(options) != 4 or len(set(options)) != 4 or any(not isinstance(option, str) or not option.strip() for option in options):
        raise ValueError("Four distinct answer options are required")
    if type(sample["answer_index"]) is not int or not 0 <= sample["answer_index"] < 4:
        raise ValueError("Invalid proposed answer index")
    if check_proposal:
        proposed = evidence_clip_ids(sample["evidence"], clips, query, category, visible_start=start)
        if set(proposed) != set(sample["positive_clip_ids"]):
            raise ValueError("Proposed support IDs disagree with source-clock evidence")
    else:
        proposed = sample["positive_clip_ids"]
    history_ids = {clip["id"] for clip in history}
    if not set(sample.get("negative_clip_ids", [])) <= history_ids or set(sample.get("negative_clip_ids", [])) & set(proposed):
        raise ValueError("Invalid proposed historical negatives")


def measured_audit_content(sample, history, mode, seed, *, image_encoder=None):
    """Original V4 input modes rendered using explicit measured sample PTS."""
    if mode not in {"blind", "evidence_only", "training_context", "reference_only", "no_concept"}:
        raise ValueError("Unknown V4 audit mode")
    by_id = {clip["id"]: clip for clip in sample["candidates"]}
    if any(clip["id"] not in by_id or clip != by_id[clip["id"]] or clip["end"] > sample["current_clip"]["start"] for clip in history):
        raise ValueError("Audit history must use actual causal source clips")
    encode = image_encoder or image_part
    prompt = BLIND if mode == "blind" else ABLATION if mode in {"reference_only", "no_concept"} else ANSWER
    content = [{"type": "text", "text": prompt+"\n"+stream_clock_text(sample)}]
    measured, mapping = {}, {}
    if mode != "no_concept":
        for concept in sample["concepts"]:
            content += [{"type": "text", "text": "Runtime concept {"+concept["name"]+"}: "+concept["description"]}, encode(concept["image"])]
            measured[concept["image"]] = _time(sample["frame_timestamps"][concept["image"]], "reference PTS")

    def add_clip(clip, label):
        content.append({"type": "text", "text": f"BEGIN CLIP {label}: source [{clip['start']:.3f}, {clip['end']:.3f}) seconds."})
        previous = None
        for path in clip["frames"]:
            timestamp = _time(sample["frame_timestamps"][path], "frame PTS")
            if not clip["start"] <= timestamp < clip["end"] or timestamp >= sample["question_time"] or previous is not None and timestamp <= previous:
                raise ValueError("Measured renderer cannot show future or misclocked pixels")
            previous = timestamp
            measured[path] = timestamp
            content.extend([{"type": "text", "text": f"Sample at measured source {timestamp:.9f} seconds."}, encode(path)])
        content.append({"type": "text", "text": "END CLIP "+label})
        mapping[label] = clip["id"]

    if mode != "reference_only":
        for i, clip in enumerate(history):
            add_clip(clip, "H"+str(i))
    if not (mode == "reference_only" and sample["task"] == "current"):
        add_clip(sample["current_clip"], "CURRENT")
    display, order = shuffled_display(sample, seed)
    content.append({"type": "text", "text": "The ONLY question; use id item:\n"+json.dumps(display, ensure_ascii=False)})
    return content, order, mapping, measured


def measured_review_call(teacher, directory, sample, history, mode, suffix="", *, image_encoder=None, negative=False):
    # Candidate IDs can encode generator-side labels. Seed only from fields the
    # verifier is legitimately allowed to observe, never IDs/evidence/annotations.
    visible = {key: sample[key] for key in ("task", "question", "options", "visible_start", "question_time")}
    visible["concepts"] = [{key: concept[key] for key in ("name", "description", "image", "known_at")} for concept in sample["concepts"]]
    seed = _digest(visible)+":"+PROTOCOL+":"+mode+":"+str(suffix)+(":negative" if negative else "")
    content, order, mapping, measured = measured_audit_content(sample, history, mode, seed, image_encoder=image_encoder)
    if negative:
        if mode != "blind" or sample["task"] != "past":
            raise ValueError("Explicit negative checks require historical blind input")
        content[0]["text"] = NEGATIVE_CHECK+"\n"+stream_clock_text(sample)
    fingerprint = hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()
    stage = ("negative" if negative else mode)+"_"+str(suffix)+"_"+PROTOCOL+"_"+fingerprint[:16]
    task = "visual_ablation" if mode in {"reference_only", "no_concept"} else "visual_verification"
    response, call = persistent_stage_call(teacher, Path(directory), stage, task, content, response_validator(mode, mapping, sample["task"]))
    return response["questions"][0], {**call, "option_permutation": order, "expected_display_index": order.index(sample["answer_index"]),
        "clip_mapping": mapping, "protocol_version": PROTOCOL_VERSION, "reader_input_protocol": READER_INPUT_PROTOCOL,
        "narrative_rendering_protocol": PROTOCOL, "measured_frame_timestamps": measured,
        "request_content_sha256": fingerprint, "proposed_evidence_locations_visible": False,
        "annotation_window_visible": False, "context_isolation": "single_question", **({"negative_confirmation": True} if negative else {})}


def _temporal_reviews(teacher, directory, sample, chunks, *, max_joint_requests, image_encoder):
    requests = build_discovery_requests(sample, [[clip["id"] for clip in chunk] for chunk in chunks],
        frame_timestamps=sample["frame_timestamps"], seed=42, image_encoder=image_encoder)
    discoveries = []
    for index, request in enumerate(requests):
        response = persistent_stage_call(teacher, directory, "temporal_discovery_"+str(index)+"_"+request.binding_sha256[:20],
            request.task, request.content, lambda value, req=request: validate_discovery_response(req, value))
        receipt = record_discovery(request, response)
        write_json(directory / f"temporal_discovery_receipt_{index}.json", receipt)
        discoveries.append(receipt)
    plan = plan_joint_requests(sample, discoveries, max_joint_requests=max_joint_requests,
        frame_timestamps=sample["frame_timestamps"], seed=42, image_encoder=image_encoder)
    snapshot = {key: value for key, value in plan.items() if key != "requests"}
    snapshot["request_bindings"] = [request.binding_sha256 for request in plan["requests"]]
    write_json(directory / "temporal_plan.json", snapshot)
    joints = []
    if plan["status"] == "ready":
        for index, request in enumerate(plan["requests"]):
            response = persistent_stage_call(teacher, directory, "temporal_joint_"+str(index)+"_"+request.binding_sha256[:20],
                request.task, request.content, lambda value, req=request: validate_joint_response(req, value))
            receipt = record_joint(request, response)
            write_json(directory / f"temporal_joint_receipt_{index}.json", receipt)
            joints.append(receipt)
    verdict = assess_temporal_gate(plan, joints)
    write_json(directory / "temporal_verdict.json", verdict)
    return plan, joints, verdict, {"verdict": verdict, "plan": snapshot,
                                  "discovery_receipts": discoveries, "joint_receipts": joints}


def _finish_quality(teacher, directory, sample, blind, calls, context_ids, negatives, *, image_encoder, check_files):
    """Same evidence/context/ablation gates and fields as V4 finish_quality."""
    clips = {clip["id"]: clip for clip in sample["candidates"]+[sample["current_clip"]]}
    possible = [event["clip_ids"] for event in blind["occurrences"] if event["answer_index"] == sample["answer_index"] and len(event["clip_ids"]) <= 3]
    if not possible:
        return None, ["no_independently_discovered_support_within_media_limit"]
    original_positive = set(sample["positive_clip_ids"])
    possible.sort(key=lambda ids: (-len(set(ids) & original_positive), len(ids), ids))
    support = possible[0]
    evidence_history = sorted((clips[cid] for cid in support if cid != sample["current_clip"]["id"]), key=lambda clip: clip["start"])
    evidence, evidence_call = measured_review_call(teacher, directory, sample, evidence_history, "evidence_only", image_encoder=image_encoder)
    if not answer_passed(evidence, evidence_call):
        return None, ["independent_evidence_sufficiency_failed"]
    context_history = sorted((clips[cid] for cid in set(context_ids) | set(negatives) if cid != sample["current_clip"]["id"]), key=lambda clip: clip["start"])
    context, context_call = measured_review_call(teacher, directory, sample, context_history, "training_context", image_encoder=image_encoder)
    if not answer_passed(context, context_call):
        return None, ["actual_supervised_context_answer_failed"]
    screens, screen_calls = {}, {}
    for mode in ["reference_only", "no_concept"]:
        screen, call = measured_review_call(teacher, directory, sample, context_history, mode, image_encoder=image_encoder)
        screens[mode], screen_calls[mode] = screen, call
        if not ablation_passed(screen):
            return None, [mode+"_dependence_not_established"]
    repaired = copy.deepcopy(sample)
    repaired["positive_clip_ids"] = blind["positive_clip_ids"]
    repaired["evidence"] = [[clips[cid]["start"], clips[cid]["end"]] for cid in blind["positive_clip_ids"]]
    repaired["context_positive_clip_ids"] = list(context_ids)
    repaired["negative_clip_ids"] = list(negatives)
    repaired["verification"] = {"accepted": True, "unique_answer": True, "concept_bindings_valid": True,
        "evidence_sufficient": True, "whole_causal_prefix_reviewed": True, "global_blind_calls": calls,
        "evidence_only": evidence, "evidence_only_call": evidence_call, "training_context_call": context_call}
    repaired["quality_screens"] = {"protocol": PROTOCOL_VERSION, "reader_input_protocol": READER_INPUT_PROTOCOL,
        "context_isolation": "single_question", "reference_only": screens["reference_only"], "no_concept": screens["no_concept"],
        "calls": screen_calls, "global_blind": blind, "global_blind_calls": calls,
        "evidence_only": evidence, "evidence_only_call": evidence_call, "training_context": context, "training_context_call": context_call}
    repaired["repair_provenance"] = {"protocol": PROTOCOL_VERSION,
        "rewrite_call": {"kind": "new_generation_already_scoped", "api_request_made": False},
        "original_question": sample["question"], "original_answer_index": sample["answer_index"],
        "original_evidence": copy.deepcopy(sample["evidence"]), "original_positive_clip_ids": list(sample["positive_clip_ids"]),
        "answer_label_changed": False, "original_sample_sha256": _digest(sample)}
    validate_sample(repaired, check_files=check_files)
    require_repaired_sample(repaired)
    return repaired, []


def require_narrative_qualified(sample, *, image_encoder=None, check_files=True):
    """Original V4 approval plus measured rendering and supplemental gating."""
    require_repaired_sample(sample)
    _check_candidate(sample, check_files=check_files, check_proposal=False)
    gate = sample.get("narrative_qualification", {})
    if gate.get("protocol") != PROTOCOL or gate.get("accepted") is not True:
        raise ValueError("Narrative qualification is missing")
    calls = sample["quality_screens"]["global_blind_calls"] + [sample["quality_screens"][key] for key in ["evidence_only_call", "training_context_call"]]
    calls += list(sample["quality_screens"]["calls"].values())
    for call in calls:
        if call.get("supplemental_protocol") == TEMPORAL_PROTOCOL:
            continue
        if call.get("narrative_rendering_protocol") != PROTOCOL or not call.get("measured_frame_timestamps"):
            raise ValueError("An original V4 gate did not use measured PTS rendering")
        for path, timestamp in call["measured_frame_timestamps"].items():
            if path not in sample["frame_timestamps"] or timestamp != sample["frame_timestamps"][path]:
                raise ValueError("A qualified call uses a different measured frame clock")
    if sample["qa_category"] == "Time-based":
        temporal = gate.get("temporal", {})
        verdict = temporal.get("verdict", {})
        if verdict.get("status") != "passed" or verdict.get("observed_answer_indices") != [sample["answer_index"]]:
            raise ValueError("Strict-order temporal qualification did not pass")
        discoveries, joints = temporal.get("discovery_receipts", []), temporal.get("joint_receipts", [])
        fresh_plan = plan_joint_requests(sample, discoveries, max_joint_requests=max(1, len(joints)),
            frame_timestamps=sample["frame_timestamps"], seed=42, image_encoder=image_encoder or image_part)
        fresh = assess_temporal_gate(fresh_plan, joints)
        if fresh != verdict or fresh.get("status") != "passed" or fresh.get("observed_answer_indices") != [sample["answer_index"]]:
            raise ValueError("Stored temporal approval does not match reconstructed requests and receipts")


def qualify_candidate(teacher, directory, sample, *, max_frames=224, max_bytes=12*1024*1024,
                      max_joint_requests=32, image_encoder=None, check_files=True):
    """Return (accepted sample or None, reasons); API/budget failures propagate.

    The directory belongs to one frozen candidate. Stage caches remain resumable;
    model tuples are revalidated against reconstructed temporal requests each run.
    Input-contract failures raise ValueError before any teacher dispatch.
    """
    _check_candidate(sample, check_files=check_files)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".qualification.lock").open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        config = {"protocol": PROTOCOL, "sample_sha256": _digest(sample), "max_frames": max_frames,
                  "max_bytes": max_bytes, "max_joint_requests": max_joint_requests,
                  "implementation_sha256": _implementation_hashes()}
        path = directory / "qualification_input.json"
        if path.exists() and read_json(path) != config:
            raise ValueError("Frozen candidate qualification changed; choose a new directory")
        write_json(path, config)
        encoder = image_encoder or image_part
        reduced, _ = measured_review_call(teacher, directory, sample, [], "reference_only", image_encoder=encoder)
        if not ablation_passed(reduced):
            return None, ["reference_only_dependence_not_established"]
        chunks = history_chunks(sample, max_frames, 2, max_bytes)
        reviews = [measured_review_call(teacher, directory, sample, chunk, "blind", str(index), image_encoder=encoder)
                   for index, chunk in enumerate(chunks)]
        temporal = None
        if sample["qa_category"] == "Time-based":
            plan, joints, verdict, temporal = _temporal_reviews(teacher, directory, sample, chunks,
                max_joint_requests=max_joint_requests, image_encoder=encoder)
            if verdict.get("status") != "passed":
                return None, ["temporal_"+str(verdict.get("reason", verdict.get("status")))]
            if verdict.get("observed_answer_indices") != [sample["answer_index"]]:
                return None, ["temporal_blind_answer_disagrees"]
            reviews = combine_with_v4_reviews(reviews, joints, plan, [clip["id"] for clip in sample["candidates"]],
                v4_protocol_version=PROTOCOL_VERSION, reader_input_protocol=READER_INPUT_PROTOCOL)
        blind = aggregate_blind(reviews, sample)
        if not blind["accepted"]:
            return None, blind["reasons"]
        if sample["task"] == "past":
            used = set()
            for attempt in range(2):
                if blind["negative_clip_ids"]:
                    break
                candidates = choose_negative_candidates(sample, blind["positive_clip_ids"], used)
                if not candidates:
                    break
                used.update(clip["id"] for clip in candidates)
                reviews.append(measured_review_call(teacher, directory, sample, candidates, "blind", str(attempt),
                    image_encoder=encoder, negative=True))
                blind = aggregate_blind(reviews, sample)
                if not blind["accepted"]:
                    return None, blind["reasons"]
            if sample["split"] == "train" and not blind["negative_clip_ids"]:
                return None, ["no_verified_retrieval_negative"]
        occurrences = [event for event in blind["occurrences"] if len(event["clip_ids"]) <= 3]
        if not occurrences:
            return None, ["support_exceeds_original_media_limit"]
        context = min(occurrences, key=lambda event: (len(event["clip_ids"]), event["clip_ids"]))["clip_ids"]
        accepted, reasons = _finish_quality(teacher, directory, sample, blind, [call for _, call in reviews], context,
            blind["negative_clip_ids"][:2], image_encoder=encoder, check_files=check_files)
        if accepted is not None:
            accepted["narrative_qualification"] = {"protocol": PROTOCOL, "accepted": True,
                "all_original_v4_gates_preserved": True, "measured_pts_rendering": True, "temporal": temporal}
            require_narrative_qualified(accepted, image_encoder=encoder, check_files=check_files)
        write_json(directory / "qualification_result.json", {"protocol": PROTOCOL, "sample": accepted, "reasons": reasons})
        return accepted, reasons
