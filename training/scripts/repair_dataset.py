#!/usr/bin/env python3
"""Versioned revalidation of generated QA; no old dataset/result is overwritten."""

import argparse
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import copy
import fcntl
import hashlib
import json
import os
import random
import re
import sqlite3
from pathlib import Path

from pearl_training.batch_generate import stage_call
from pearl_training.io import code_revision, read_json, read_jsonl, sha256_file, write_json, write_jsonl
from pearl_training.multimodal import VERIFIER_INPUT_PROTOCOL as READER_INPUT_PROTOCOL
from pearl_training.schema import check_split_isolation, validate_sample
from pearl_training.stream_audit import (ABLATION, ANSWER, BLIND, REWRITE, PROTOCOL_VERSION,
    ablation_passed, aggregate_blind, answer_passed, clip_parts, history_chunks,
    reference_parts, require_repaired_sample, review_call)
from pearl_training.teacher import Teacher, TeacherBudgetExhausted, TeacherRetryExhausted


def groups_for_review(rows, limit, seed):
    groups = {}
    for row in rows:
        groups.setdefault(row["base_question_id"], []).append(row)
    if limit is None:
        return list(groups.values())
    if limit < 1:
        raise ValueError("Family limit must be positive")
    parts = {split: [v for v in groups.values() if v[0]["split"] == split] for split in ["train", "validation"]}
    for split in parts:
        random.Random(str(seed) + ":" + split).shuffle(parts[split])
    chosen = []
    # A source-blind pilot includes validation but is never chosen by student errors.
    while len(chosen) < limit and any(parts.values()):
        if parts["validation"]:
            chosen.append(parts["validation"].pop())
        for _ in range(3):
            if len(chosen) < limit and parts["train"]:
                chosen.append(parts["train"].pop())
    return chosen


def rewrite_sample(teacher, directory, sample):
    if sample["task"] == "current":
        question = sample["question"]
        if not re.search(r"\b(?:current|now)\b", question, re.I):
            question = "In the current clip, " + question[0].lower() + question[1:]
        return ({"repairable": True, "question": question,
                 "reason": "Current-clip scope is already defined by the source clock; preserve wording"},
                {"kind": "unchanged_current_question", "api_request_made": False})
    episode = [c for c in sample["candidates"] if c["start"] >= sample["source_window_start"]]
    text = REWRITE + f"\nThe query is at source time {sample['question_time']} seconds."
    content = [{"type": "text", "text": text}] + reference_parts(sample)
    for i, clip in enumerate(episode + [sample["current_clip"]]):
        content += clip_parts(clip, "G" + str(i))
    original = {k: sample[k] for k in ["task", "question", "options", "answer_index"]}
    content.append({"type": "text", "text": "Original item, used ONLY for generation:\n" + json.dumps(original, ensure_ascii=False)})
    fingerprint = hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()
    def validate(response):
        if type(response.get("repairable")) is not bool:
            raise ValueError("Rewriter needs a repairable boolean")
        if response["repairable"] and (not isinstance(response.get("question"), str) or not response["question"].strip()):
            raise ValueError("Repairable item needs a question")
    response, call = stage_call(teacher, directory, "rewrite_" + PROTOCOL_VERSION + "_" + fingerprint[:12],
                                "visual_generation", content, validate)
    return response, call


def rewrite_is_compatible(before, question):
    before_refs = set(re.findall(r"\{[^{}]+\}", before["question"]))
    after_refs = set(re.findall(r"\{[^{}]+\}", question))
    known = {"{" + c["name"] + "}" for c in before["concepts"]}
    if not before_refs <= after_refs or not after_refs <= known:
        return False
    answer = before["options"][before["answer_index"]].casefold()
    if answer in question.casefold() and answer not in before["question"].casefold():
        return False
    # Do not turn annotation metadata into a location hint to make a question pass.
    forbidden = r"\b(?:source\s+time|frame\s*\d|64[- ]second|(?:last|past)\s+64|(?:beginning|start)\s+of\s+(?:the\s+)?(?:video|scene))\b"
    return (not re.search(forbidden, question, re.I)
            and not re.search(r"\b\d+(?:\.\d+)?\s*(?:seconds?|minutes?)\b|\b\d{1,2}:\d{2}(?::\d{2})?\b", question, re.I))


def acquire_output_lock(output):
    handle = (output / ".repair_v4.lock").open("a+")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        raise RuntimeError("This repair output already has an active coordinator") from None
    return handle


def finish_quality(teacher, directory, sample, blind, calls, context_ids, negatives, rewrite_call, original):
    clips = {c["id"]: c for c in sample["candidates"] + [sample["current_clip"]]}
    possible = [event["clip_ids"] for event in blind["occurrences"]
                if event["answer_index"] == sample["answer_index"] and len(event["clip_ids"]) <= 3]
    if not possible:
        return None, ["no_independently_discovered_support_within_media_limit"]
    original_positive = set(original["positive_clip_ids"])
    possible.sort(key=lambda ids: (-len(set(ids) & original_positive), len(ids), ids))
    support_ids = possible[0]
    evidence_history = sorted((clips[cid] for cid in support_ids if cid != sample["current_clip"]["id"]), key=lambda c: c["start"])
    evidence, evidence_call = review_call(teacher, directory, sample, evidence_history, "evidence_only")
    if not answer_passed(evidence, evidence_call):
        return None, ["independent_evidence_sufficiency_failed"]
    context_history = sorted((clips[cid] for cid in set(context_ids) | set(negatives)
                              if cid != sample["current_clip"]["id"]), key=lambda c: c["start"])
    context, context_call = review_call(teacher, directory, sample, context_history, "training_context")
    if not answer_passed(context, context_call):
        return None, ["actual_supervised_context_answer_failed"]
    screens, screen_calls = {}, {}
    for mode in ["reference_only", "no_concept"]:
        screen, call = review_call(teacher, directory, sample, context_history, mode)
        screens[mode], screen_calls[mode] = screen, call
        if not ablation_passed(screen):
            return None, [mode + "_dependence_not_established"]
    repaired = copy.deepcopy(sample)
    repaired["positive_clip_ids"] = blind["positive_clip_ids"]
    repaired["evidence"] = [[clips[cid]["start"], clips[cid]["end"]] for cid in blind["positive_clip_ids"]]
    repaired["context_positive_clip_ids"] = list(context_ids)
    repaired["negative_clip_ids"] = list(negatives)
    repaired["verification"] = {"accepted": True, "unique_answer": True, "concept_bindings_valid": True,
        "evidence_sufficient": True, "whole_causal_prefix_reviewed": True, "global_blind_calls": calls,
        "evidence_only": evidence, "evidence_only_call": evidence_call, "training_context_call": context_call}
    repaired["quality_screens"] = {"protocol": PROTOCOL_VERSION, "reader_input_protocol": READER_INPUT_PROTOCOL,
        "context_isolation": "single_question", "reference_only": screens["reference_only"],
        "no_concept": screens["no_concept"], "calls": screen_calls,
        "global_blind": blind, "global_blind_calls": calls,
        "evidence_only": evidence, "evidence_only_call": evidence_call,
        "training_context": context, "training_context_call": context_call}
    repaired["repair_provenance"] = {"protocol": PROTOCOL_VERSION, "rewrite_call": rewrite_call,
        "original_question": original["question"], "original_answer_index": original["answer_index"],
        "original_evidence": original["evidence"], "original_positive_clip_ids": original["positive_clip_ids"],
        "answer_label_changed": False, "original_sample_sha256": hashlib.sha256(json.dumps(original, sort_keys=True).encode()).hexdigest()}
    validate_sample(repaired)
    require_repaired_sample(repaired)
    return repaired, []


def process_family(group, args, signature):
    fid = group[0]["base_question_id"]
    directory = Path(args.output) / "items" / hashlib.sha256(fid.encode()).hexdigest()[:24]
    directory.mkdir(parents=True, exist_ok=True)
    completion = directory / "completed.json"
    if completion.exists():
        saved = read_json(completion)
        if saved.get("signature") == signature and saved.get("protocol") == PROTOCOL_VERSION:
            return saved
        raise ValueError("Repair cache protocol changed; use a new output directory")
    teacher = Teacher(args.credential_readme, args.effective_approval, args.ledger, proxy=args.proxy)
    result = {"base_question_id": fid, "signature": signature, "protocol": PROTOCOL_VERSION,
              "accepted": [], "reasons": [], "status": "rejected"}
    try:
        reviewed = []
        shared_rewrite = None
        counterfactual = bool(group[0].get("counterfactual_group"))
        for original in group:
            subdir = directory / hashlib.sha256(original["sample_id"].encode()).hexdigest()[:16]
            subdir.mkdir(exist_ok=True)
            sample = copy.deepcopy(original)
            if shared_rewrite is None or not counterfactual:
                rewrite, rewrite_call = rewrite_sample(teacher, subdir, original)
                if counterfactual:
                    shared_rewrite = rewrite, rewrite_call
            else:
                rewrite, rewrite_call = shared_rewrite
            if not rewrite["repairable"] or not rewrite_is_compatible(original, rewrite["question"]):
                result["reasons"] = ["question_cannot_be_faithfully_repaired"]
                break
            sample["question"] = rewrite["question"].strip()
            reviews = []
            for index, history in enumerate(history_chunks(sample, args.max_frames, 2, args.max_bytes)):
                reviews.append(review_call(teacher, subdir, sample, history, "blind", str(index)))
            blind = aggregate_blind(reviews, sample)
            write_json(subdir / "global_blind.json", blind)
            if not blind["accepted"]:
                result["reasons"] = blind["reasons"]
                break
            reviewed.append((original, sample, subdir, rewrite_call, blind, [call for _, call in reviews]))
        if len(reviewed) == len(group):
            shared_context = None
            shared_negatives = None
            if counterfactual:
                if any(row[1]["question"] != reviewed[0][1]["question"] or row[1]["options"] != reviewed[0][1]["options"] for row in reviewed):
                    raise ValueError("Counterfactual wording/options changed across variants")
                shared_context = set(group[0].get("context_positive_clip_ids", group[0]["positive_clip_ids"]))
                if any(not shared_context & set(row[4]["positive_clip_ids"]) for row in reviewed):
                    result["reasons"] = ["shared_counterfactual_context_not_globally_supported"]
                shared_negatives = set.intersection(*(set(row[4]["negative_clip_ids"]) for row in reviewed))
                shared_negatives -= set.union(*(set(row[4]["positive_clip_ids"]) for row in reviewed))
            if not result["reasons"]:
                for original, sample, subdir, rewrite_call, blind, calls in reviewed:
                    events = [e for e in blind["occurrences"] if e["answer_index"] == sample["answer_index"] and len(e["clip_ids"]) <= 3]
                    if not events:
                        result["reasons"] = ["support_exceeds_original_media_limit"]
                        break
                    context = sorted(shared_context) if counterfactual else sorted(min(events, key=lambda e: (len(e["clip_ids"]), e["clip_ids"]))["clip_ids"])
                    confirmed = shared_negatives if counterfactual else set(blind["negative_clip_ids"])
                    old_neg = set(original["negative_clip_ids"])
                    negatives = sorted(confirmed, key=lambda cid: (cid not in old_neg, cid))[:2]
                    repaired, reasons = finish_quality(teacher, subdir, sample, blind, calls, context, negatives, rewrite_call, original)
                    if reasons:
                        result["reasons"] = reasons
                        break
                    result["accepted"].append(repaired)
        if result["reasons"] or len(result["accepted"]) != len(group):
            result["accepted"] = []
            result["reasons"] = result["reasons"] or ["family_incomplete"]
        else:
            result["status"] = "accepted"
    except TeacherRetryExhausted as error:
        if (teacher.approval.get("on_api_retry_exhausted") != "defer_candidate_and_continue"
                or error.last_failure.get("http_status") in {401, 402, 403, 429}):
            raise
        result.update(status="deferred_api", accepted=[], reasons=[str(error)], last_failure=error.last_failure)
    write_json(completion, result)
    return result


def export_results(output):
    results = [read_json(p) for p in output.glob("items/*/completed.json")]
    if any(row.get("protocol") != PROTOCOL_VERSION for row in results):
        raise ValueError("Legacy completed records cannot be exported as repaired data")
    samples = [s for row in results for s in row["accepted"]]
    for sample in samples:
        require_repaired_sample(sample)
    check_split_isolation(samples)
    write_jsonl(output / "samples.jsonl", samples)
    summary = {"protocol": PROTOCOL_VERSION, "completed_families": len(results),
               "accepted_families": len({s["base_question_id"] for s in samples}), "accepted_variants": len(samples),
               "statuses": dict(Counter(r["status"] for r in results)),
               "reasons": dict(Counter(reason for r in results for reason in r["reasons"])),
               "not_a_replacement_for_the_fixed_500_until_target_reached": True}
    write_json(output / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", default="data/batch500/snapshots/strict500-seed42-v1/samples.jsonl")
    parser.add_argument("--output", required=True)
    parser.add_argument("--approval", default="data/approvals/batch500.json")
    parser.add_argument("--credential-readme", default="../README.md")
    parser.add_argument("--ledger", default="data/batch500/generation/calls.sqlite3")
    parser.add_argument("--proxy", default="http://your-proxy-host:8080")
    parser.add_argument("--workers", type=int, default=100)
    parser.add_argument("--limit-families", type=int)
    parser.add_argument("--max-new-calls", type=int)
    parser.add_argument("--max-frames", type=int, default=224)
    parser.add_argument("--max-bytes", type=int, default=12 * 1024 * 1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    output = Path(args.output)
    if output.resolve() == Path(args.samples).parent.resolve():
        raise ValueError("Repair output must not overwrite the frozen source")
    output.mkdir(parents=True, exist_ok=True)
    marker = output / ".repair_v4_output"
    if not marker.exists():
        if any(output.iterdir()):
            raise FileExistsError("Use an empty output directory; old data/results must not be overwritten")
        marker.write_text(PROTOCOL_VERSION + "\n")
    output_lock = acquire_output_lock(output)
    source_sha = sha256_file(args.samples)
    rows = read_jsonl(args.samples)
    check_split_isolation(rows)
    groups = groups_for_review(rows, args.limit_families, args.seed)
    chunks = sum(len(history_chunks(s, args.max_frames, 2, args.max_bytes)) for group in groups for s in group)
    selected_count = sum(map(len, groups))
    approval = read_json(args.approval)
    if not 1 <= args.workers <= approval["max_concurrent_requests"]:
        raise ValueError("Worker count exceeds user approval")
    with sqlite3.connect(args.ledger) as db:
        initial_calls = db.execute("select count(*) from calls").fetchone()[0]
    old_plan = output / "plan.json"
    previous_plan = read_json(old_plan) if old_plan.exists() else None
    local_budget_origin = previous_plan["initial_calls"] if previous_plan else initial_calls
    capped = {**approval}
    if args.max_new_calls is not None:
        if args.max_new_calls <= 0:
            raise ValueError("Local call subcap must be positive")
        capped["max_vision_requests"] = min(approval["max_vision_requests"], local_budget_origin + args.max_new_calls)
    args.effective_approval = str(output / "effective_approval.json")
    write_json(args.effective_approval, capped)
    # Completed families may be cached only under exactly the same repair logic.
    sources = [Path(__file__), Path(__file__).parents[1] / "src/pearl_training/stream_audit.py",
               Path(__file__).parents[1] / "src/pearl_training/multimodal.py"]
    signature = hashlib.sha256(json.dumps({"source_sha": source_sha, "code_sha": [sha256_file(p) for p in sources],
        "max_frames": args.max_frames, "max_bytes": args.max_bytes}, sort_keys=True).encode()).hexdigest()
    if old_plan.exists() and read_json(old_plan)["signature"] != signature:
        raise ValueError("Repair protocol changed; choose a new output directory")
    write_json(old_plan, {"arguments": vars(args), "code": code_revision(), "signature": signature,
        "protocol": PROTOCOL_VERSION, "reader_input_protocol": READER_INPUT_PROTOCOL,
        "source_samples_sha256": source_sha, "question_families": len(groups), "variants": selected_count,
        "blind_chunks": chunks, "maximum_other_checks_if_all_pass": selected_count * 4,
        "maximum_rewrites": selected_count, "initial_calls": local_budget_origin,
        "calls_at_this_invocation": initial_calls,
        "user_total_cap": approval["max_vision_requests"], "effective_total_cap": capped["max_vision_requests"],
        "selected_base_question_ids": [g[0]["base_question_id"] for g in groups]})
    if args.plan_only:
        print(json.dumps(read_json(old_plan), ensure_ascii=False), flush=True)
        output_lock.close()
        return
    if initial_calls >= capped["max_vision_requests"]:
        raise TeacherBudgetExhausted("No approved calls remain; request additional budget")
    write_json(output / "state.json", {"status": "running", "phase": "repair", "signature": signature, "pid": os.getpid()})
    iterator, pending, failures = iter(groups), {}, []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        stopped = False
        while pending or not stopped:
            while not stopped and len(pending) < args.workers:
                try:
                    group = next(iterator)
                except StopIteration:
                    stopped = True
                    break
                pending[pool.submit(process_family, group, args, signature)] = group[0]["base_question_id"]
            if not pending:
                break
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                fid = pending.pop(future)
                try:
                    row = future.result()
                    print("FAMILY", fid, row["status"], row["reasons"], flush=True)
                except Exception as error:
                    failures.append({"base_question_id": fid, "type": type(error).__name__, "error": str(error)[:1000]})
                    stopped = True
                print("PROGRESS", json.dumps(export_results(output), ensure_ascii=False), flush=True)
    if sha256_file(args.samples) != source_sha:
        raise ValueError("Frozen source changed during repair")
    summary = export_results(output)
    write_json(output / "state.json", {"status": "needs_review" if failures else "complete",
        "phase": "repair", "signature": signature, "failures": failures, "summary": summary, "pid": os.getpid()})
    output_lock.close()
    if failures:
        raise SystemExit(2)
    print("COMPLETE", json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
