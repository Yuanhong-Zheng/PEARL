"""Continuous scheduling around unchanged Flash500 V4 generation functions."""
from __future__ import annotations

from collections import Counter, deque
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import argparse
import fcntl
import json
from pathlib import Path
import shutil
import signal
import sqlite3
import threading
import time

from . import legacy_movie_generation as factory
from . import restored_v4_qualification as qualification
from .fleet_continuous_inputs import InputFeeder
from .fleet_elastic import ElasticRouter, MAX_WORKERS, checked_cluster_approval
from .fleet_proof_cache import ClosedLedgerCache, QualifiedProofCache
from .fleet_transport import RoutedSessions
from .io import read_json, read_jsonl, sha256_file, write_json, write_jsonl
from .schema import check_split_isolation
from .teacher import TeacherBudgetExhausted

PROTOCOL = "continuous-restored-v4-fleet-v1"


def implementation_bindings():
    result = factory.implementations()
    for name in ("fleet_continuous.py", "fleet_continuous_inputs.py", "fleet_proof_cache.py",
                 "fleet_elastic.py", "fleet_transport.py"):
        path = Path(__file__).with_name(name)
        result[str(path.resolve())] = sha256_file(path)
    return result


def load_seed_cache(path, samples_path, rows):
    receipt = read_json(path)
    cache_path = Path(__file__).with_name("fleet_proof_cache.py")
    if (receipt.get("status") != "original_v4_proofs_verified"
            or receipt["samples_sha256"] != sha256_file(samples_path)
            or receipt["count"] != len(rows)
            or receipt["original_verifier_sha256"] != qualification._implementations()
            or receipt["cache_code_sha256"] != sha256_file(cache_path)):
        raise ValueError("The parallel seed proof receipt no longer matches the frozen input/code")
    cache = QualifiedProofCache()
    for key, media in receipt["entries"]:
        cache.prime(key, tuple(tuple(part) for part in media))
    for sample in rows:
        cache.verify(sample)
    return cache


def drain_scheduler(queue, *, take_inputs, source_status, set_pressure, capacity,
                    remaining, dispatch, process_result, snapshot, stop, workers=MAX_WORKERS):
    """A slow future never prevents admission/dispatch of unrelated new inputs.

    The original per-split pending-question reservation predicate remains the
    sole admission condition, in addition to the approved resource capacity.
    """
    pending = {}
    failure = None
    with ThreadPoolExecutor(max_workers=workers) as pool:
        try:
            while not stop.is_set():
                status = source_status()
                if status.get("error"):
                    raise RuntimeError("Input producer failed: "+json.dumps(status["error"]))
                queue.extend(("window", item) for item in take_inputs())
                available = capacity()
                set_pressure(sum(kind == "window" for kind, _ in queue)
                             +sum(kind == "window" for kind, _ in pending.values()), available)
                held = deque()
                while queue and not stop.is_set() and len(pending) < min(workers, available):
                    kind, payload = queue.popleft()
                    if not factory.can_dispatch(kind, payload["split"], remaining(), pending):
                        held.append((kind, payload))
                        continue
                    pending[pool.submit(dispatch, kind, payload)] = (kind, payload)
                queue.extendleft(reversed(held))
                if not any(remaining().values()):
                    stop.set()
                    break
                if pending:
                    done, _ = wait(pending, timeout=.5, return_when=FIRST_COMPLETED)
                    for future in done:
                        kind, payload = pending.pop(future)
                        process_result(kind, payload, future.result(), queue)
                else:
                    stop.wait(.2)
                snapshot(pending, queue)
        except BaseException as error:
            failure = error
            stop.set()
        # Stop only new reservations. Existing calls finish and retain receipts.
        for future, (kind, payload) in list(pending.items()):
            try:
                result = future.result()
            except TeacherBudgetExhausted as error:
                if str(error) != "Restored production dispatch is stopped":
                    failure = failure or error
            except Exception as error:
                failure = failure or error
            else:
                try:
                    process_result(kind, payload, result, queue)
                except Exception as error:
                    failure = failure or error
            pending.pop(future, None)
        snapshot(pending, queue, force=True)
    if failure:
        raise failure


def run(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    lock = (output/".continuous.lock").open("a+")
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    stop = threading.Event()
    feeder = None
    append_file = None
    reason = {"value": None}
    for number in (signal.SIGINT, signal.SIGTERM):
        signal.signal(number, lambda signum, frame: (reason.update(value="operator_stop"), stop.set()))
    try:
        target = factory.checked_target({"train": args.target_train, "validation": args.target_validation})
        settings = checked_cluster_approval(args.approval, factory.checked_approval, target=target)
        if "fleet_resources" not in settings:
            raise ValueError("Use the separately approved per-node256 cluster profile")
        ledger_cache = ClosedLedgerCache(factory.ledger_snapshot)
        factory.ledger_snapshot = ledger_cache
        closed = factory._continued_run(args.previous_run)
        retained = read_jsonl(closed["samples"]["path"])
        proof_cache = load_seed_cache(args.seed_validation, closed["samples"]["path"], retained)
        qualification.require_qualified = proof_cache.verify
        check_split_isolation(retained)
        inputs = factory.checked_inputs(args.initial_manifest)
        priors = closed["prior_ledgers"]+[closed["ledger"]]
        if inputs["prior_ledgers"] != priors and {row["path"]: row for row in inputs["prior_ledgers"]} != {row["path"]: row for row in priors}:
            raise ValueError("Initial inputs must preserve all completed call ledgers")
        plan = {"protocol": PROTOCOL, "v4_protocol": factory.PROTOCOL,
            "previous_run": str(args.previous_run.resolve()), "manifest": factory.bound(args.initial_manifest),
            "inputs": {"prior_ledgers": priors}, "approval": factory.bound(args.approval),
            "settings": settings, "target": target, "prior_calls": sum(row["calls"] for row in priors),
            "seed_families": len(retained), "implementation_sha256": implementation_bindings(),
            "source_roots": [str(root.resolve()) for root in args.source_root],
            "seed_validation": factory.bound(args.seed_validation), "chunk_windows": args.chunk_windows,
            "resource_policy": "256 requests per ready teacher; continuously replenish ready windows"}
        path = output/"plan.json"
        if path.exists() and read_json(path) != plan:
            raise ValueError("Frozen continuous coordinator configuration/code changed")
        write_json(path, plan)
        rows = {sample["base_question_id"]: sample for sample in retained}
        registry, seen, finished = {}, set(rows), {sample["sample_id"] for sample in retained}
        for sample in retained:
            if not factory.reserve_fact(sample, registry):
                raise ValueError("Previously qualified fact families collide")
        for path in sorted((output/"questions").glob("*/completed.json")):
            result = read_json(path)
            finished.add(result["key"])
            if result["accepted"] is not None:
                sample = result["accepted"]
                proof_cache.verify(sample)
                if sample["base_question_id"] in rows or not factory.reserve_fact(sample, registry):
                    raise ValueError("A resumed qualified fact family collides")
                rows[sample["base_question_id"]] = sample
                seen.add(sample["base_question_id"])
        check_split_isolation(rows.values())
        counts = Counter(sample["split"] for sample in rows.values())
        factory.remaining_families(target, counts)
        samples_path = output/"samples.jsonl"
        if samples_path.exists():
            write_jsonl(samples_path, sorted(rows.values(), key=lambda row: row["base_question_id"]))
        else:
            shutil.copyfile(closed["samples"]["path"], samples_path)
        append_file = samples_path.open("a", encoding="utf-8")
        router = ElasticRouter(args.pool)
        teacher = factory.create_teacher(args.approval, output, plan, stop)
        teacher._thread = RoutedSessions(router, teacher.ledger)
        feeder = InputFeeder(output/"input-catalog", args.source_root, args.initial_manifest,
                             closed["excluded_windows"], [row["path"] for row in priors], target, retained,
                             chunk_windows=args.chunk_windows, poll_seconds=args.poll_seconds)
        feeder.initialize()
        feeder.start()
        statuses, errors = Counter(), []
        group_splits, sha_splits, base_splits = {}, {}, {}
        for sample in rows.values():
            for mapping, key in ((group_splits, sample["source_group"]),
                                 (sha_splits, sample["source_sha256"]),
                                 (base_splits, sample["base_question_id"])):
                mapping[key] = sample["split"]

        def remaining():
            return factory.remaining_families(target, counts)

        def dispatch(kind, payload):
            return (factory.run_window if kind == "window" else factory.run_question)(payload, output, teacher)

        def process_result(kind, payload, result, queue):
            statuses[kind+":"+result["status"]] += 1
            if kind == "window":
                for sample in reversed(result.get("candidates", [])):
                    fid = sample["base_question_id"]
                    if fid not in seen and factory.reserve_fact(sample, registry):
                        seen.add(fid)
                        if sample["sample_id"] not in finished:
                            queue.appendleft(("question", sample))
            elif result["accepted"] is not None:
                sample = result["accepted"]
                proof_cache.verify(sample)
                fid = sample["base_question_id"]
                if fid in rows:
                    raise ValueError("Duplicate accepted family")
                for mapping, key in ((group_splits, sample["source_group"]),
                                     (sha_splits, sample["source_sha256"]), (base_splits, fid)):
                    if mapping.setdefault(key, sample["split"]) != sample["split"]:
                        raise ValueError("Cross-split leakage for "+str(key))
                rows[fid] = sample
                counts[sample["split"]] += 1
                remaining()  # The unchanged exact target guard must still pass.
                append_file.write(json.dumps(sample, ensure_ascii=False)+"\n")
                append_file.flush()
                finished.add(sample["sample_id"])

        last_snapshot = [0.0]

        def snapshot(pending, queue, force=False):
            now = time.monotonic()
            if not force and now-last_snapshot[0] < 1:
                return
            last_snapshot[0] = now
            with sqlite3.connect(teacher.ledger, timeout=30) as database:
                own = database.execute("SELECT count(*) FROM calls").fetchone()[0]
            phase = "halted_error" if errors else ("running" if pending or queue else "waiting_for_cpu_inputs")
            summary = {"protocol": PROTOCOL, "v4_protocol": factory.PROTOCOL, "families": len(rows),
                "family_splits": dict(counts), "remaining": remaining(), "target_met": not any(remaining().values()),
                "pending_jobs": len(pending), "queued_jobs": len(queue), "job_statuses": dict(statuses),
                "errors": errors, "prior_calls": plan["prior_calls"], "actual_new_calls": own,
                "cumulative_calls": plan["prior_calls"]+own, "retained_qualified_imports": len(retained),
                "new_qualified_families": len(rows)-len(retained), "status": phase,
                "input_prefetch": feeder.status(), "proof_cache_hits": proof_cache.hits,
                "proof_cache_misses": proof_cache.misses, "updated_unix": time.time()}
            write_json(output/"summary.json", summary)
            write_json(output/"state.json", {"previous_run": str(args.previous_run.resolve()),
                "active_run": str(output), "phase": phase, "last_summary": summary,
                "updated_unix": time.time()})

        snapshot({}, deque(), force=True)
        try:
            drain_scheduler(deque(), take_inputs=feeder.take, source_status=feeder.status,
                set_pressure=feeder.set_pressure, capacity=router.capacity, remaining=remaining,
                dispatch=dispatch, process_result=process_result, snapshot=snapshot, stop=stop)
        except Exception as error:
            stop.set()
            errors.append({"type": type(error).__name__, "error": str(error)})
            snapshot({}, deque(), force=True)
            raise
        finally:
            feeder.close()
        append_file.close()
        append_file = None
        check_split_isolation(rows.values())
        write_jsonl(samples_path, sorted(rows.values(), key=lambda row: row["base_question_id"]))
        summary = read_json(output/"summary.json")
        with sqlite3.connect(teacher.ledger, timeout=30) as database:
            if database.execute("SELECT count(*) FROM calls WHERE status='reserved'").fetchone()[0]:
                raise ValueError("Never close while model calls remain reserved")
        summary.update(status="complete" if not any(remaining().values()) else "paused",
                       pending_jobs=0, stopped_reason=reason["value"])
        write_json(output/"summary.json", summary)
        write_json(output/"state.json", {"previous_run": str(output), "active_run": None,
            "phase": summary["status"], "last_summary": summary, "updated_unix": time.time()})
    finally:
        if feeder:
            feeder.close()
        if append_file:
            append_file.close()
        lock.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("pool", "previous-run", "initial-manifest", "seed-validation", "output", "approval"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--source-root", action="append", type=Path, required=True)
    parser.add_argument("--target-train", type=int, default=9000)
    parser.add_argument("--target-validation", type=int, default=1000)
    parser.add_argument("--chunk-windows", type=int, default=512)
    parser.add_argument("--poll-seconds", type=float, default=15)
    args = parser.parse_args()
    if not 1 <= args.chunk_windows <= 4096 or not 0 < args.poll_seconds <= 60:
        parser.error("Use bounded input chunks and a positive polling interval")
    run(args)


if __name__ == "__main__":
    main()
