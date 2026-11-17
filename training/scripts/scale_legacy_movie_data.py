#!/usr/bin/env python3
"""Continue the approved V4 factory as CPU-prepared movie sources arrive.

This supervisor does not annotate, relabel, download, alter teacher settings or
change quality gates. Each sequential batch calls the existing frozen producer
and imports the complete previous qualified export and spent-call ledgers.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import time
from pathlib import Path

from pearl_training import legacy_movie_generation as factory
from pearl_training.io import read_json, sha256_file, write_json


def ready_sources(roots, excluded, remaining):
    """Select published, unattempted windows in splits with remaining slots."""
    selected, identities, total = [], {}, 0
    for root in roots:
        for receipt in sorted((Path(root) / "sources").glob("*/preparation_status.json")):
            status = read_json(receipt)
            if status.get("status") != "cpu_ready":
                continue
            path = Path(status["source_manifest"]).resolve(strict=True)
            if sha256_file(path) != status["source_manifest_sha256"]:
                raise ValueError("A published CPU source changed")
            source = read_json(path)
            if remaining[source["split"]] <= 0:
                continue
            identity = (source["sha256"], source["stream_start"], source["stream_end"])
            if identity in identities:
                if identities[identity] != (source["source_group"], source["split"]):
                    raise ValueError("Identical source interval has conflicting work/split")
                continue
            identities[identity] = (source["source_group"], source["split"])
            windows = [read_json(p) for p in read_json(path.parent / "windows.json")]
            unseen = [w for w in windows if not factory._previously_attempted(w, excluded)]
            if unseen:
                selected.append(str(path))
                total += len(unseen)
    return selected, total


def finish_pending_counter(directory, summary):
    """After the producer joins all workers, correct its stale drain counter.

    Keep the original summary and closed SQL fingerprint as evidence. Only the
    producer's exact intentional dispatch-stop exception is benign after the
    target is met. Other errors, partial runs and reserved calls remain errors.
    """
    errors = summary.get("errors", [])
    stopped = lambda row: row.get("type") == "TeacherBudgetExhausted" and row.get("error") == "Restored production dispatch is stopped"
    if (not summary.get("target_met") or any(not stopped(row) for row in errors)
            or not (summary.get("pending_jobs", 0) or errors)):
        return summary
    directory = Path(directory)
    ledger = factory.ledger_snapshot(directory / "calls.sqlite3")
    if ledger["calls"] != summary["actual_new_calls"]:
        raise ValueError("The completed producer and SQL call counts disagree")
    write_json(directory / "pending_counter_before.json", summary)
    write_json(directory / "pending_counter_audit.json", {
        "reason": "producer.run returned after joining all worker futures; its drain dictionary was not cleared",
        "reported_pending_jobs": summary["pending_jobs"], "actual_inflight_requests": 0,
        "intentional_dispatch_stops_after_target": len(errors),
        "closed_sql": ledger, "qualified_rows_or_calls_changed": False})
    corrected = {**summary, "pending_jobs": 0, "errors": [], "status": "complete"}
    write_json(directory / "summary.json", corrected)
    return corrected


def supervise(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    target = factory.checked_target({"train": args.target_train, "validation": args.target_validation})
    with (output / ".supervisor.lock").open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        config = {"script_sha256": sha256_file(__file__),
            "initial_previous_run": str(args.previous_run.resolve(strict=True)),
            "source_roots": [str(p.resolve()) for p in args.source_root],
            "approval": factory.bound(args.approval), "workers": args.workers,
            "minimum_windows": args.minimum_windows, "gather_seconds": args.gather_seconds,
            "poll_seconds": args.poll_seconds, "producer_code": factory.implementations()}
        config["target"] = target
        config_path, state_path = output / "config.json", output / "state.json"
        if config_path.exists() and read_json(config_path) != config:
            raise ValueError("Supervisor inputs/settings/code changed; stop and inspect before resuming")
        write_json(config_path, config)
        state = read_json(state_path) if state_path.exists() else {
            "previous_run": config["initial_previous_run"], "round": 0, "active_run": None}
        first_ready = None

        def save(**updates):
            state.update(updates, updated_unix=time.time())
            write_json(state_path, state)

        try:
            while True:
                if state.get("active_run"):
                    active = Path(state["active_run"])
                    summary_path = active / "summary.json"
                    summary = read_json(summary_path) if summary_path.exists() else {}
                    if summary.get("status") not in {"complete", "frozen_input_limit_reached"}:
                        summary = factory.run(state["active_manifest"], active, args.approval,
                                              workers=args.workers)
                    summary = finish_pending_counter(active, summary)
                    if summary.get("errors"):
                        raise RuntimeError("The producer stopped with errors: " + json.dumps(summary["errors"]))
                    factory._continued_run(active)  # Requires closed, reconciled SQL before proceeding.
                    save(previous_run=str(active), active_run=None, last_summary=summary,
                         phase="batch_closed")
                    print(json.dumps({"batch_closed": str(active), "families": summary["families"],
                        "split": summary["family_splits"], "calls": summary["cumulative_calls"]}), flush=True)
                    first_ready = None
                previous = Path(state["previous_run"])
                summary = read_json(previous / "summary.json")
                remaining = factory.remaining_families(target, summary["family_splits"])
                if not any(remaining.values()):
                    save(phase="complete", last_summary=summary)
                    return
                closed = factory._continued_run(previous)
                sources, count = ready_sources(args.source_root, closed["excluded_windows"], remaining)
                if not count:
                    first_ready = None
                elif first_ready is None:
                    first_ready = time.monotonic()
                elapsed = 0 if first_ready is None else time.monotonic() - first_ready
                save(phase="collecting_cpu_sources", ready_unattempted_sources=len(sources),
                     ready_unattempted_windows=count, last_summary=summary, remaining=remaining)
                if count and (count >= args.minimum_windows or elapsed >= args.gather_seconds):
                    number = state["round"] + 1
                    batch = output / f"round-{number:03d}"
                    inputs = output / f"round-{number:03d}-inputs"
                    manifest = factory.prepare_inputs(sources, inputs, [], per_source=None,
                                                      continue_from=previous, target=target)
                    plan = factory.run(manifest, batch, args.approval, workers=args.workers, plan_only=True)
                    save(round=number, active_run=str(batch), active_manifest=str(manifest), phase="running_batch")
                    print(json.dumps({"starting_batch": str(batch), **plan}), flush=True)
                    continue
                time.sleep(args.poll_seconds)
        except Exception as error:
            save(phase="halted_error", error_type=type(error).__name__, error=str(error))
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous-run", required=True, type=Path)
    parser.add_argument("--source-root", required=True, action="append", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--approval", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=256)
    parser.add_argument("--target-train", type=int, default=450)
    parser.add_argument("--target-validation", type=int, default=50)
    parser.add_argument("--minimum-windows", type=int, default=56)
    parser.add_argument("--gather-seconds", type=int, default=120)
    parser.add_argument("--poll-seconds", type=int, default=15)
    args = parser.parse_args()
    if not 1 <= args.workers <= 256 or args.minimum_windows < 1 or args.gather_seconds < 0 or not 1 <= args.poll_seconds <= 60:
        parser.error("Preserve <=256 workers and a finite source-gathering interval")
    supervise(args)


if __name__ == "__main__":
    main()
