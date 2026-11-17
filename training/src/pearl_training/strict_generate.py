"""Individually verified generation, with a persistent shared round call cap."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import random
import re
import signal
import threading
from datetime import datetime, timezone
from pathlib import Path

from .batch_generate import (alias_names, alias_text, compile_global, export,
                             family_counts, normalize_references, stage_call)
from .batch_prompts import ANNOTATE, GENERATE
from .generate import video_content
from .io import code_revision, read_json, write_json
from .isolated_audit import PROTOCOL_VERSION, single_question_audit
from .local_embeddings import LocalEmbeddings
from .schema import validate_sample
from .teacher import Teacher, TeacherRetryExhausted


COMPLETED = "completed_isolated.json"


def may_defer(teacher, error):
    # Credentials, payment and persistent quota errors need account-level help.
    return (teacher.approval.get("on_api_retry_exhausted") == "defer_candidate_and_continue"
            and error.last_failure.get("http_status") not in {401, 402, 403, 429})


def deferred_record(window, scope, error, question_id=None):
    return {"window_id": window["window_id"], "scope": scope, "question_id": question_id,
            "stage": error.stage, "attempts": error.attempts, "last_failure": error.last_failure,
            "status": "api_exhausted_unlabelled", "label_assigned": False,
            "handling": "user-approved deferral; continue other candidates"}


def audit_with_policy(teacher, directory, window, concepts, question, mode, extras):
    try:
        return single_question_audit(teacher, directory, window, concepts, question, mode, extras)
    except TeacherRetryExhausted as error:
        if not may_defer(teacher, error):
            raise
        record = deferred_record(window, mode, error, question["id"])
        write_json(directory / (question["id"] + "_isolated_deferred.json"), record)
        print("DEFER_API", window["window_id"], question["id"], mode, str(error)[:200], flush=True)
        return None


def annotation_format(response):
    if not isinstance(response, dict):
        raise ValueError("Annotation must be an object")
    if type(response.get("usable")) is not bool:
        raise ValueError("Annotation usable must be boolean")
    if response["usable"] and (not isinstance(response.get("concepts"), list) or not isinstance(response.get("events"), list)):
        raise ValueError("Usable annotations need concept and event arrays")


def generation_format(response):
    if not isinstance(response, dict):
        raise ValueError("Generation must be an object")
    if not isinstance(response.get("questions"), list):
        raise ValueError("Generation must return a questions array")


def full_check_failures(check, call, question):
    required = ["concept_bindings_valid", "visually_answerable", "unique_answer",
                "evidence_sufficient", "requires_concept_binding", "no_reference_shortcut"]
    reasons = []
    if not all(check.get(key) is True for key in required):
        reasons.append("visual_verification_failed")
    if question["task"] == "past" and check.get("requires_history_for_past") is not True:
        reasons.append("history_not_required")
    if type(check.get("answer_index")) is not int or check["answer_index"] != call["expected_display_index"]:
        reasons.append("blind_answer_disagrees")
    return reasons


def ablation_passed(row):
    # Confident wrong answers and malformed results do not establish dependence.
    return row.get("sufficient_information") is False and "answer_index" in row and row["answer_index"] is None


def pair_counterfactuals(window, questions, accepted, names):
    """Keep the same QA and pixels, changing only the runtime name mapping."""
    pairs = {}
    for question in questions:
        if question.get("swap_pair_id"):
            pairs.setdefault(question["swap_pair_id"], []).append(question)
    by_id = {s["sample_id"].rsplit(":", 1)[-1]: s for s in accepted}
    for pair_id, members in pairs.items():
        if len(members) != 2 or any(q["id"] not in by_id for q in members):
            continue
        first, second = members
        if len(first["concept_ids"]) != 1 or len(second["concept_ids"]) != 1:
            continue
        a, b = first["concept_ids"][0], second["concept_ids"][0]
        if a == b or first["options"] != second["options"] or first["answer_index"] == second["answer_index"]:
            continue
        if first["task"] != second["task"] or any(re.search(r"\{C[0-9]\}", x) for x in first["options"]):
            continue
        if first["question"].replace("{" + a + "}", "{TARGET}") != second["question"].replace("{" + b + "}", "{TARGET}"):
            continue
        left, right = by_id[first["id"]], by_id[second["id"]]
        positives = sorted(set(left["positive_clip_ids"]) | set(right["positive_clip_ids"]))
        historical = {c["id"] for c in left["candidates"]}
        if len(set(positives) & historical) > 3:
            continue
        shared_negatives = sorted((set(left["negative_clip_ids"]) & set(right["negative_clip_ids"])) - set(positives))
        swapped = {**names, a: names[b], b: names[a]}
        right["question"] = alias_text(second["question"], swapped)
        right["options"] = [alias_text(option, swapped) for option in second["options"]]
        for concept in right["concepts"]:
            concept["name"] = swapped[concept["id"]]
        right["base_question_id"] = left["base_question_id"]
        for i, sample in enumerate((left, right)):
            sample["counterfactual_group"] = window["window_id"] + ":" + pair_id
            sample["counterfactual_variant"] = i
            sample["context_positive_clip_ids"] = positives
            sample["negative_clip_ids"] = shared_negatives
            sample["counterfactual_context"] = "same clips and QA; only alias binding changes"
        if left["question"] != right["question"] or left["options"] != right["options"]:
            raise ValueError("Counterfactual pair changed the visible question or options")


def process_window(path, args):
    window = read_json(path)
    directory = Path(args.output) / "items" / window["window_id"]
    directory.mkdir(parents=True, exist_ok=True)
    completed = directory / COMPLETED
    if completed.exists():
        return read_json(completed)
    teacher = Teacher(args.credential_readme, args.approval, Path(args.output) / "calls.sqlite3", proxy=args.proxy)
    # Original annotation/generation requests may be reused with their original
    # call IDs; pooled acceptance and ablation decisions are never reused.
    result = {"window_id": window["window_id"], "split": window["split"], "accepted": [], "rejected": [],
              "protocol": PROTOCOL_VERSION, "proposed_questions": 0,
              "generation_run_id": args.run_id, "deferred": []}
    def window_stage(name, task, content, validator):
        try:
            return stage_call(teacher, directory, name, task, content, validator)
        except TeacherRetryExhausted as error:
            if not may_defer(teacher, error):
                raise
            record = deferred_record(window, name, error)
            result["deferred"].append(record)
            result["reason"] = "api_deferred_before_questions"
            write_json(directory / "deferred_window.json", record)
            write_json(completed, result)
            print("DEFER_API", window["window_id"], name, str(error)[:200], flush=True)
            return None
    annotation_result = window_stage("annotation", "visual_annotation", [{"type": "text", "text": ANNOTATE}] + video_content(window), annotation_format)
    if annotation_result is None:
        return result
    annotation, annotation_call = annotation_result
    if annotation.get("usable") is not True or not annotation.get("concepts") or not annotation.get("events"):
        result["reason"] = "no_supported_events"
        write_json(completed, result)
        return result
    generation_result = window_stage("generation", "visual_generation", [{"type": "text", "text": GENERATE + "\nAnnotation:\n" + json.dumps(annotation, ensure_ascii=False)}], generation_format)
    if generation_result is None:
        return result
    raw, generation_call = generation_result
    proposal, invalid, corrections = normalize_references({"concepts": annotation["concepts"], "questions": raw["questions"]})
    write_json(directory / "normalized_proposal_isolated.json", {"proposal": proposal, "metadata_normalizations": corrections, "rejected_schema_items": invalid})
    result["rejected"].extend(invalid)
    result["proposed_questions"] = len(raw["questions"])
    names = alias_names(window, proposal["concepts"])
    event_ids = {e["id"] for e in annotation["events"] if e.get("visually_clear") is True}
    source = read_json(window["source_manifest"])
    prefix = [clip for clip in source["clips"] if clip["end"] <= window["start"]]
    local = LocalEmbeddings(args.embeddings)
    for question in proposal["questions"]:
        qid = question["id"]
        question_file = directory / (qid + "_isolated_decision.json")
        deferred_file = directory / (qid + "_isolated_deferred.json")
        if deferred_file.exists():
            result["deferred"].append(read_json(deferred_file))
            continue
        if question_file.exists():
            saved = read_json(question_file)
            result["accepted" if saved["accepted"] else "rejected"].append(saved["sample"] if saved["accepted"] else saved["rejection"])
            continue
        reasons = []
        if not question.get("event_ids") or not set(question["event_ids"]) <= event_ids:
            reasons.append("unsupported_event_id")
        if question.get("attribute") not in {"object", "container", "action", "location", "order", "recipient", "relation"}:
            reasons.append("unsupported_attribute")
        positive_ids = {clip["id"] for clip in window["clips"] if any(max(clip["start"] - window["start"], a) < min(clip["end"] - window["start"], b) for a, b in question["evidence"])}
        if len(positive_ids) > 3:
            reasons.append("evidence_exceeds_training_media_budget")
        if reasons:
            rejection = {"id": qid, "reasons": reasons}
            result["rejected"].append(rejection)
            write_json(question_file, {"accepted": False, "rejection": rejection})
            continue
        query = {"concepts": [{"name": names[c["id"]], "description": c["description"], "image": window["frames"][c["reference_frame"]]["path"]} for c in proposal["concepts"]], "question": alias_text(question["question"], names)}
        extras = local.retrieve(query, prefix, k=3)
        labels = {"L" + str(i): clip for i, clip in enumerate(window["clips"][:-1])}
        labels.update({"H" + str(i): clip for i, clip in enumerate(extras)})
        full_result = audit_with_policy(teacher, directory, window, proposal["concepts"], question, "full", extras)
        if full_result is None:
            result["deferred"].append(read_json(deferred_file))
            continue
        check, check_call = full_result
        reasons += full_check_failures(check, check_call, question)
        negatives = []
        for label in check.get("confirmed_negative_labels", []):
            if label not in labels:
                reasons.append("unknown_negative_label")
            elif labels[label]["id"] in positive_ids:
                reasons.append("conflicting_negative_label")
            else:
                negatives.append(labels[label]["id"])
        screens, screen_calls = {}, {}
        api_deferred = False
        if not reasons:
            for mode in ["reference_only", "no_concept"]:
                reduced = audit_with_policy(teacher, directory, window, proposal["concepts"], question, mode, extras)
                if reduced is None:
                    api_deferred = True
                    break
                screen, screen_call = reduced
                screens[mode], screen_calls[mode] = screen, screen_call
                if not ablation_passed(screen):
                    reasons.append(mode + "_not_demonstrated_missing_information")
                    break
        if api_deferred:
            result["deferred"].append(read_json(deferred_file))
            continue
        if reasons:
            rejection = {"id": qid, "reasons": sorted(set(reasons)), "verification": check, "screens": screens}
            result["rejected"].append(rejection)
            write_json(question_file, {"accepted": False, "rejection": rejection})
            print("REJECT", window["window_id"], qid, ",".join(rejection["reasons"]), flush=True)
            continue
        negatives = sorted(set(negatives), key=lambda cid: (cid not in {c["id"] for c in extras}, cid))
        sample = compile_global(window, proposal, question, check, check_call, source, negatives)
        sample.update({"data_round": "batch500", "teacher_annotation_call": annotation_call, "teacher_generation_call": generation_call,
                       "generation_run_id": args.run_id,
                       "quality_screens": {**screens, "calls": screen_calls, "context_isolation": "single_question", "protocol": PROTOCOL_VERSION,
                                           "independent_option_permutations": True, "teacher_rotations_per_screen": 1}})
        validate_sample(sample)
        result["accepted"].append(sample)
        write_json(question_file, {"accepted": True, "sample": sample})
        print("ACCEPT", window["window_id"], qid, question["task"], "negatives", len(negatives), flush=True)
    pair_counterfactuals(window, proposal["questions"], result["accepted"], names)
    for sample in result["accepted"]:
        validate_sample(sample)
    write_json(completed, result)
    return result


def planned_windows(args):
    excluded = read_json(args.pilot_windows)
    plan = []
    seen = set()
    # Preserve the original plan order; append the additional pool afterwards.
    for folder, seed in [("windows", 42), ("reserve_windows", 43)]:
        windows = []
        for path in sorted((Path(args.prepared) / folder).glob("*.json")):
            window = read_json(path)
            if window["window_id"] in seen:
                raise ValueError("Duplicate annotation window across plans")
            seen.add(window["window_id"])
            if any(window["source_id"] == old["source_id"] and max(window["start"], old["start"]) < min(window["start"] + 64, old["start"] + 64) for old in excluded):
                continue
            windows.append((path, window))
        random.Random(seed).shuffle(windows)
        validation = [x for x in windows if x[1]["split"] == "validation"]
        training = [x for x in windows if x[1]["split"] == "train"]
        while training or validation:
            if validation:
                plan.append(validation.pop())
            for _ in range(5):
                if training:
                    plan.append(training.pop())
    return plan[:args.max_windows] if args.max_windows else plan


def main():
    parser = argparse.ArgumentParser()
    for name in ["prepared", "output", "credential-readme", "approval", "pilot-windows"]:
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--embeddings", default="http://127.0.0.1:28780")
    parser.add_argument("--proxy", default="http://your-proxy-host:8080")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-windows", type=int)
    parser.add_argument("--stop-after-drain-file", help="Stop dispatching windows when this local control file exists")
    args = parser.parse_args()
    parser.error("V3 generation was retired after audit information leakage was found. "
                 "Use scripts/repair_dataset.py; new freeze/training requires the V4 audit protocol.")
    args.run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    teacher = Teacher(args.credential_readme, args.approval, output / "calls.sqlite3", proxy=args.proxy)
    approval = teacher.approval
    if (approval.get("target_original_questions") != 500 or not isinstance(approval.get("max_vision_requests"), int)
            or approval["max_vision_requests"] <= 0 or not approval.get("approval_source")
            or "individually" not in approval.get("quality_policy", "")):
        raise ValueError("Strict round requires a recorded user-approved cap and individual-audit policy")
    write_json(output / "runs" / (args.run_id + ".json"), {"arguments": vars(args), "code": code_revision(),
               "approval": approval, "protocol": PROTOCOL_VERSION, "initial_reserved_call_count": len(teacher.usage())})
    iterator = iter(planned_windows(args))
    pending, failures = {}, []
    drain = threading.Event()
    def request_drain(signum, frame):
        drain.set()
        print("DRAIN_REQUESTED: finish active windows, dispatch no new windows", flush=True)
    signal.signal(signal.SIGINT, request_drain)
    signal.signal(signal.SIGTERM, request_drain)
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        stopped = False
        while pending or not stopped:
            if drain.is_set() or (args.stop_after_drain_file and Path(args.stop_after_drain_file).exists()):
                stopped = True
            while not stopped and len(pending) < args.workers:
                counts = family_counts(output, COMPLETED)
                if sum(n for (s, t), n in counts.items() if s == "train") >= 450 and sum(n for (s, t), n in counts.items() if s == "validation") >= 50:
                    stopped = True
                    break
                try:
                    path, window = next(iterator)
                except StopIteration:
                    stopped = True
                    break
                target = 450 if window["split"] == "train" else 50
                if sum(n for (s, t), n in counts.items() if s == window["split"]) >= target:
                    continue
                pending[pool.submit(process_window, path, args)] = window["window_id"]
                print("START", window["window_id"], window["split"], flush=True)
            if not pending:
                break
            done, _ = concurrent.futures.wait(pending, return_when=concurrent.futures.FIRST_COMPLETED)
            for future in done:
                wid = pending.pop(future)
                try:
                    row = future.result()
                    print("DONE", wid, "accepted", len(row["accepted"]), "rejected", len(row["rejected"]),
                          "deferred", len(row.get("deferred", [])), flush=True)
                except Exception as error:
                    failures.append({"window": wid, "type": type(error).__name__, "error": str(error)[:300]})
                    write_json(output / "failures_isolated.json", failures)
                    write_json(output / "runs" / (args.run_id + "_failures.json"), failures)
                    stopped = True
                    print("ERROR", wid, type(error).__name__, str(error)[:150], flush=True)
                summary = export(output, COMPLETED, "samples_isolated.jsonl")
                usage = teacher.usage()
                write_json(output / "usage.json", usage)
                print("PROGRESS", json.dumps(summary), "requests", len(usage), flush=True)
    summary = export(output, COMPLETED, "samples_isolated.jsonl")
    print("FINISHED", json.dumps(summary), "requests", len(teacher.usage()), flush=True)
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
