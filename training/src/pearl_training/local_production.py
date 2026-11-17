"""Persistence and quota guards for local-only V4 production."""

from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3

from .io import read_json, write_json


def utc():
    return datetime.now(timezone.utc).isoformat()


def split_pending(pending):
    return Counter(payload["split"] for kind, payload in pending.values() if kind == "question")


def can_dispatch(kind, split, remaining, pending):
    # In-flight questions can all succeed. Reserve that many target slots before
    # starting more checks; completed futures remain reserved until exported.
    return remaining[split] > split_pending(pending)[split]


def bootstrap_ledger(source, target, model):
    target = Path(target)
    if target.exists():
        raise FileExistsError("Refusing to overwrite a local production ledger")
    target.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(source) as previous:
        rows = previous.execute("SELECT model,status FROM calls").fetchall()
        if not rows or any(name != model or status == "reserved" for name, status in rows):
            raise ValueError("Seed ledger is incomplete or contains another teacher")
        with sqlite3.connect(target) as destination:
            previous.backup(destination)
    return len(rows)


def reconcile_interrupted_calls(root, ledger, initial_call_id):
    """Count calls interrupted before their success/error receipt was saved.

    A JSON/schema response may be charged and recorded in SQLite before the stage
    cache is written. Such attempts must not disappear from the six-attempt cap.
    """
    root = Path(root)
    stages = defaultdict(list)
    for directory in [root / "windows", root / "questions"]:
        for path in directory.glob("*/*_request_summary.json"):
            summary = read_json(path)
            name = path.name.removesuffix("_request_summary.json")
            success = path.with_name(name + ".json")
            success_id = read_json(success).get("call", {}).get("request_id") if success.exists() else None
            stages[summary["content_sha256"]].append((path.parent, name, success_id))
    repaired = []
    with sqlite3.connect(ledger, timeout=30) as db:
        rows = db.execute("SELECT id,prompt_sha256,status FROM calls WHERE id>?", (initial_call_id,)).fetchall()
        for request_id, fingerprint, status in rows:
            if fingerprint not in stages:
                raise ValueError(f"Cannot attribute local request {request_id} to a saved stage; review before resuming")
            for directory, name, success_id in stages.get(fingerprint, []):
                receipt = directory / f"{name}_error_request_{request_id}.json"
                if success_id == request_id or receipt.exists():
                    continue
                previous = list(directory.glob(name + "_error_*.json"))
                record = {"type": "InterruptedRequestUnknown", "request_id": request_id,
                    "category": "interrupted_conservatively_spent", "attempt": len(previous) + 1,
                    "max_attempts": 6, "next_retry_seconds": None,
                    "error": "The ledger records this attempt but no final stage receipt was saved before interruption",
                    "previous_ledger_status": status, "reconciled_utc": utc()}
                write_json(receipt, record)
                repaired.append({"request_id": request_id, "receipt": str(receipt)})
            if status == "reserved":
                db.execute("UPDATE calls SET status='interrupted_unknown' WHERE id=?", (request_id,))
    return repaired
