#!/usr/bin/env python3
"""Replenish V4 to exactly 450/50 families using the original shared API ledger."""

import argparse
from collections import Counter, defaultdict, deque
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import copy
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import random
import sqlite3
import threading
import traceback

from pearl_training.io import code_revision, read_json, read_jsonl, sha256_file, write_json, write_jsonl
from pearl_training.replenishment import (GENERATION_VERSION, NEW_QUESTIONS, augment_seed_group,
    digest, make_candidates, verify_new_candidate)
from pearl_training.batch_generate import stage_call
from pearl_training.schema import check_split_isolation
from pearl_training.stream_audit import require_repaired_sample
from pearl_training.teacher import Teacher, TeacherBudgetExhausted, TeacherRequestError, TeacherRetryExhausted


STOP = threading.Event()


class DispatchStopped(Exception):
    pass


class AccountReviewRequired(Exception):
    pass


class GuardedTeacher(Teacher):
    def chat_json(self, task, content):
        if STOP.is_set():
            raise DispatchStopped("Dispatch stopped; saved successful stages remain resumable")
        try:
            return super().chat_json(task, content)
        except TeacherBudgetExhausted:
            STOP.set()
            raise
        except TeacherRequestError as error:
            if error.details.get("http_status") in {401, 402, 403, 429}:
                STOP.set()
                raise AccountReviewRequired(str(error)) from None
            raise


def grouped(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row["base_question_id"]].append(row)
    return dict(groups)


def reserve_fact(sample, registry):
    """Conservatively reject paraphrases with overlapping supporting event IDs."""
    family = sample["question_family"]
    key = (family["source"], family["window"], family["attribute"], family["task"])
    events = set(family["events"])
    if not events or events & registry.get(key, set()):
        return False
    registry.setdefault(key, set()).update(events)
    return True


def window_plan(prepared, ledger, seed):
    windows = {}
    for folder in ["windows", "reserve_windows"]:
        for path in sorted((prepared / folder).glob("*.json")):
            window = read_json(path)
            if window["window_id"] in windows:
                raise ValueError("Duplicate frozen source window")
            annotation_path = prepared / "generation/items" / window["window_id"] / "annotation.json"
            if not annotation_path.exists():
                continue
            record = read_json(annotation_path)
            annotation = record["response"]
            if not annotation.get("usable") or not annotation.get("events"):
                continue
            call = ledger.execute("SELECT status,model FROM calls WHERE id=?", (record["call"]["request_id"],)).fetchone()
            if call != ("ok", "gemini-3.1-pro-preview"):
                raise ValueError("Cached annotation has no successful approved ledger record")
            windows[window["window_id"]] = {"window_path": str(path), "annotation_path": str(annotation_path),
                "window_sha256": sha256_file(path), "annotation_sha256": sha256_file(annotation_path),
                "window_id": window["window_id"], "source_id": window["source_id"], "split": window["split"], "start": window["start"],
                "binding_rich": len(annotation.get("concepts", [])) >= 2 and
                    len({e.get("actor") for e in annotation["events"] if e.get("visually_clear") is True}) >= 2}
    # Fair rotation over sources; shorter prefixes first reduce duplicate review
    # cost without omitting any part of a question's causal historical pool.
    sources = defaultdict(list)
    for window in windows.values():
        sources[window["source_id"]].append(window)
    for values in sources.values():
        values.sort(key=lambda w: (not w["binding_rich"], w["start"]))
    source_ids = sorted(sources)
    random.Random(seed).shuffle(source_ids)
    result = []
    while any(sources.values()):
        for sid in source_ids:
            if sources[sid]:
                result.append(sources[sid].pop(0))
    return result


def load_seeds(args):
    rows = read_jsonl(args.seed_samples)
    for row in rows:
        require_repaired_sample(row)
    check_split_isolation(rows)
    return rows


def source_signature(args, windows):
    paths = [Path(__file__), Path("src/pearl_training/replenishment.py"), Path("src/pearl_training/stream_audit.py"),
             Path("src/pearl_training/multimodal.py"), Path("scripts/repair_dataset.py"), Path("src/pearl_training/teacher.py")]
    output_limits = read_json(args.approval).get("max_output_tokens_by_task", {})
    return digest({"seed": sha256_file(args.seed_samples), "original": sha256_file(args.original_samples),
                   "sources": windows, "code": [sha256_file(p) for p in paths],
                   "max_frames": args.max_frames, "max_bytes": args.max_bytes,
                   "teacher_output_limits": output_limits})


def generation_format(response):
    questions = response.get("questions")
    if not isinstance(questions, list) or len(questions) > 6:
        raise ValueError("Expected at most six new questions")
    fields = {"id", "task", "concept_ids", "question", "options", "answer_index", "evidence", "event_ids", "attribute"}
    ids = set()
    for q in questions:
        if not isinstance(q, dict) or not fields <= set(q):
            raise ValueError("Question schema is incomplete")
        if not isinstance(q["question"], str) or not isinstance(q["options"], list):
            raise ValueError("Invalid question/options types")
        if not isinstance(q["id"], str) or q["id"] in ids:
            raise ValueError("Question IDs must be unique within a generation request")
        ids.add(q["id"])
        if (not all(isinstance(option, str) for option in q["options"])
                or not all(isinstance(q[key], list) for key in ["concept_ids", "evidence", "event_ids"])):
            raise ValueError("Invalid concept/options/evidence types")


def execute(kind, payload, args, signature, original_rows, exclusions):
    key = payload[0]["base_question_id"] if kind == "seed" else payload["sample_id"] if kind == "question" else payload["window_id"]
    folder = Path(args.output) / {"seed": "seed_updates", "question": "questions", "window": "windows"}[kind] / digest(key)[:24]
    folder.mkdir(parents=True, exist_ok=True)
    completion = folder / "completed.json"
    if completion.exists():
        saved = read_json(completion)
        if saved["signature"] != signature:
            raise ValueError("Replenishment code/data changed; choose a new output directory")
        return saved
    result = {"kind": kind, "key": key, "signature": signature, "accepted": [], "candidates": [], "reasons": []}
    teacher = GuardedTeacher(args.credential_readme, args.approval, args.ledger, proxy=args.proxy)
    try:
        if kind == "seed":
            rows, status = augment_seed_group(teacher, folder, payload, original_rows)
            result.update(status=status, accepted=rows)
        elif kind == "question":
            sample, reasons = verify_new_candidate(teacher, folder, payload, args.max_frames, args.max_bytes)
            result.update(status="accepted" if sample else "rejected", accepted=[sample] if sample else [], reasons=reasons)
        else:
            if sha256_file(payload["window_path"]) != payload["window_sha256"] or sha256_file(payload["annotation_path"]) != payload["annotation_sha256"]:
                raise ValueError("Cached source/annotation changed")
            window, record = read_json(payload["window_path"]), read_json(payload["annotation_path"])
            content = [{"type": "text", "text": NEW_QUESTIONS + "\nAlready retained fact keys:\n" + json.dumps(exclusions.get(window["window_id"], []))
                        + "\nFixed annotation (relative times):\n" + json.dumps(record["response"], ensure_ascii=False)}]
            generation, call = stage_call(teacher, folder, "new_questions_" + GENERATION_VERSION + "_" + digest(content)[:12],
                                          "visual_generation", content, generation_format)
            provenance = {"path": payload["annotation_path"], "sha256": payload["annotation_sha256"], "call": record["call"]}
            candidates, invalid, corrections = make_candidates(window, record["response"], generation, call, provenance)
            result.update(status="generated", candidates=candidates, reasons=invalid, metadata_normalizations=corrections)
    except TeacherRetryExhausted as error:
        if teacher.approval.get("on_api_retry_exhausted") != "defer_candidate_and_continue":
            raise
        # An exhausted augmentation never removes an already approved seed.
        result.update(status="deferred_api", accepted=payload if kind == "seed" else [],
                      reasons=[str(error)], last_failure=error.last_failure)
    write_json(completion, result)
    return result


def export(args, seeds):
    groups = grouped(seeds)
    output = Path(args.output)
    statuses = Counter()
    reasons = Counter()
    for path in output.glob("seed_updates/*/completed.json"):
        record = read_json(path)
        groups[record["key"]] = record["accepted"]
        statuses["seed:" + record["status"]] += 1
    for path in output.glob("questions/*/completed.json"):
        record = read_json(path)
        statuses["question:" + record["status"]] += 1
        reasons.update(record["reasons"])
        if record["accepted"]:
            sample = record["accepted"][0]
            fid = sample["base_question_id"]
            if fid in groups:
                raise ValueError("Duplicate fact family in retained/new data")
            groups[fid] = record["accepted"]
    rows = [s for fid in sorted(groups) for s in groups[fid]]
    for s in rows:
        require_repaired_sample(s)
    check_split_isolation(rows)
    counts = Counter(g[0]["split"] for g in groups.values())
    if counts["train"] > 450 or counts["validation"] > 50:
        raise ValueError("Output exceeded the approved split targets")
    write_jsonl(output / "samples.jsonl", rows)
    summary = {"families": len(groups), "variants": len(rows), "family_splits": dict(counts),
        "remaining": {"train": 450 - counts["train"], "validation": 50 - counts["validation"]},
        "variant_tasks": dict(Counter(s["task"] for s in rows)),
        "train_past_with_negatives": sum(s["split"] == "train" and s["task"] == "past" and bool(s["negative_clip_ids"]) for s in rows),
        "job_statuses": dict(statuses), "rejection_reasons": dict(reasons),
        "target_met": (counts["train"], counts["validation"]) == (450, 50)}
    write_json(output / "summary.json", summary)
    return groups, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed-samples", default="data/batch500/repair-v4-full-v1/samples.jsonl")
    parser.add_argument("--original-samples", default="data/batch500/snapshots/strict500-seed42-v1/samples.jsonl")
    parser.add_argument("--prepared", type=Path, default=Path("data/batch500"))
    parser.add_argument("--approval", default="data/approvals/batch500.json")
    parser.add_argument("--credential-readme", default="../README.md")
    parser.add_argument("--ledger", default="data/batch500/generation/calls.sqlite3")
    parser.add_argument("--proxy", default="http://your-proxy-host:8080")
    parser.add_argument("--workers", type=int, default=100)
    parser.add_argument("--max-frames", type=int, default=224)
    parser.add_argument("--max-bytes", type=int, default=12 * 1024 * 1024)
    parser.add_argument("--limit-windows", type=int)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    marker = output / ".replenishment_owned"
    if not marker.exists():
        if any(output.iterdir()):
            raise ValueError("Choose a new empty replenishment output directory")
        marker.write_text(GENERATION_VERSION + "\n")
    lock = (output / ".lock").open("a+")
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    approval = read_json(args.approval)
    if not 1 <= args.workers <= approval["max_concurrent_requests"]:
        raise ValueError("Worker count exceeds approval")
    seeds = load_seeds(args)
    with sqlite3.connect(args.ledger) as db:
        windows = window_plan(args.prepared, db, 42)
        initial_calls = db.execute("SELECT COUNT(*) FROM calls").fetchone()[0]
    if args.limit_windows:
        windows = windows[:args.limit_windows]
    signature = source_signature(args, windows)
    previous_plan = read_json(output / "plan.json") if (output / "plan.json").exists() else None
    if previous_plan and previous_plan["signature"] != signature:
        raise ValueError("Source/code changed; use a new output directory")
    write_json(output / "plan.json", {"arguments": {**vars(args), "prepared": str(args.prepared)},
        "signature": signature, "code": code_revision(), "protocol": GENERATION_VERSION,
        "seed_samples_sha256": sha256_file(args.seed_samples), "cached_windows": len(windows),
        "initial_calls": previous_plan["initial_calls"] if previous_plan else initial_calls,
        "calls_at_this_invocation": initial_calls, "approved_cumulative_cap": approval["max_vision_requests"],
        "target": {"train": 450, "validation": 50}, "window_plan": windows})
    seed_groups = grouped(seeds)
    if args.plan_only:
        print(json.dumps({"cached_windows": len(windows), "seed_families": len(seed_groups), "calls": initial_calls,
                          "remaining_approved_calls": approval["max_vision_requests"] - initial_calls, "signature": signature}))
        return
    originals = {s["sample_id"]: s for s in read_jsonl(args.original_samples)}
    exclusions = defaultdict(list)
    for s in seeds:
        family = s.get("question_family", {})
        window_id = s["source_id"] + "_" + f"{int(s['source_window_start']):06d}"
        exclusions[window_id].append({k: family[k] for k in ["events", "attribute", "task"] if k in family})
    queue = deque(("seed", g) for g in seed_groups.values() if g[0]["task"] == "past" and g[0]["split"] == "train"
                  and not all(s["negative_clip_ids"] for s in g))
    queue.extend(("window", w) for w in windows)
    groups, summary = export(args, seeds)
    known = set(groups)
    known_facts = {}
    for group in groups.values():
        for sample in group:
            reserve_fact(sample, known_facts)
    pending, failures = {}, []
    write_json(output / "state.json", {"status": "running", "pid": os.getpid(), "signature": signature})
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        while queue or pending:
            deferred = deque()
            while queue and len(pending) < args.workers and not STOP.is_set():
                kind, payload = queue.popleft()
                split = payload[0]["split"] if kind == "seed" else payload["split"]
                if kind != "seed" and summary["remaining"][split] <= 0:
                    continue
                active = sum(k == "question" and p["split"] == split for k, p in pending.values())
                if kind != "seed" and active >= summary["remaining"][split]:
                    deferred.append((kind, payload))
                    continue
                future = pool.submit(execute, kind, payload, args, signature, originals, exclusions)
                pending[future] = kind, payload
            queue.extendleft(reversed(deferred))
            if not pending:
                break
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                kind, payload = pending.pop(future)
                try:
                    result = future.result()
                    if kind == "window":
                        for sample in reversed(result["candidates"]):
                            fid = sample["base_question_id"]
                            if fid not in known and reserve_fact(sample, known_facts):
                                known.add(fid)
                                queue.appendleft(("question", sample))
                    print("JOB", kind, result["key"], result["status"], result["reasons"][:1], flush=True)
                except DispatchStopped:
                    pass
                except Exception as error:
                    STOP.set()
                    failure = {"kind": kind, "key": payload[0]["base_question_id"] if kind == "seed" else payload.get("sample_id", payload.get("window_id")),
                               "type": type(error).__name__, "error": str(error)[:1200], "traceback": traceback.format_exc()[-5000:]}
                    failures.append(failure)
                    write_json(output / "latest_failure.json", failure)
                    write_json(output / "state.json", {"status": "draining_after_failure", "pid": os.getpid(),
                                                      "signature": signature, "failures": failures})
                    print("STOP_DISPATCH", json.dumps(failure), flush=True)
                groups, summary = export(args, seeds)
                print("PROGRESS", json.dumps({k:summary[k] for k in ["families", "variants", "remaining", "train_past_with_negatives"]}), flush=True)
            if STOP.is_set():
                queue.clear()
    groups, summary = export(args, seeds)
    status = "complete" if summary["target_met"] and not failures else "needs_review" if failures else "needs_more_candidates"
    write_json(output / "state.json", {"status": status, "signature": signature, "failures": failures,
        "summary": summary, "pid": os.getpid(), "updated_utc": datetime.now(timezone.utc).isoformat()})
    print("FINISHED", status, json.dumps(summary), flush=True)
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
