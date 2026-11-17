"""Bulk local source preparation for the approved post-pilot10k continuation.

Reuse validated Luna rankings and the original S3-download/CPU-PTS helpers.
No text/visual API calls and no QA labels are produced here. Original V4
annotation determines visual usability later, on unchanged64-second windows.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
import fcntl
import hashlib
import importlib.util
import os
import random
from pathlib import Path
import time
from types import SimpleNamespace

from .io import read_json, read_jsonl, sha256_file, write_json, write_jsonl
from .narrative_prepare import CLOCK_DEFINITION
from .narrative_sources import (POLICY, DownloadRateLimit, download_one, group_works, s3_location)
from .narrative_visual import choose_intervals, probe_source


ROOT = Path(__file__).resolve().parents[2]


def binding(path):
    path = Path(path).resolve(strict=True)
    return {"path": str(path), "sha256": sha256_file(path)}


def choose_works(works, resources, rankings, registry, output, *, limit=90, seed=42):
    """Original adequate-resource rule; preserve every prior work assignment."""
    by_work = {row["work_id"]: row for row in works}
    if len(by_work) != len(works) or len({row["work_id"] for row in rankings}) != len(rankings):
        raise ValueError("Duplicate metadata identities")
    groups, by_resource = group_works(works), {row["resource_id"]: row for row in resources}
    candidates = []
    for rank in rankings:
        work = by_work[rank["work_id"]]
        group = groups[work["work_id"]]
        if rank["priority"] not in {"high", "medium"} or group["source_group"] in registry:
            continue
        valid = [by_resource[rid] for rid in work["resource_ids"] if rid in by_resource
            and not by_resource[rid].get("initial_rule_flags")
            and by_resource[rid].get("duration", 0) >= 900
            and min(by_resource[rid].get("width", 0), by_resource[rid].get("height", 0)) >= 480
            and rid.startswith("upload:videoBucket:")]
        if valid:
            valid.sort(key=lambda row: (min(row["width"], row["height"]), -row["duration"], row["resource_id"]))
            candidates.append((work, rank, group, valid[0], hashlib.sha256(f"{seed}\0{work['work_id']}".encode()).hexdigest()))
    selected, used, genres = [], set(registry), set()
    while len(selected) < limit:
        remaining = [row for row in candidates if row[2]["source_group"] not in used]
        if not remaining:
            break
        item = min(remaining, key=lambda row: (row[1]["priority"] != "high",
                   (row[0]["genres"] or ["Unknown"])[0] in genres, row[4]))
        selected.append(item)
        used.add(item[2]["source_group"])
        genres.add((item[0]["genres"] or ["Unknown"])[0])
    indices = list(range(len(selected)))
    random.Random(seed).shuffle(indices)
    validation = set(indices[:max(1, round(len(selected) / 10))]) if selected else set()
    registry, rows = copy.deepcopy(registry), []
    for index, (work, rank, group, resource, _) in enumerate(selected):
        split = "validation" if index in validation else "train"
        registry[group["source_group"]] = {**group, "split": split}
        sid = resource["resource_id"]
        name = hashlib.sha256(sid.encode()).hexdigest()[:20]
        rows.append({"source_id": sid, "source_group": group["source_group"], "work_id": work["work_id"],
            "split": split, "titles": work["titles"], "genres": work["genres"], "years": work.get("years", []),
            "metadata_ranking": rank, "group_provenance": group, "resource_metadata": resource,
            "s3": s3_location(sid), "local_path": str(Path(output).resolve() / "videos" / f"{name}.source"),
            "benchmark_source_policy": POLICY, "benchmark_overlap": {"status": "unknown"},
            "screening_only": True, "training_eligible": False})
    return rows, registry


def candidate_intervals(source, probe, occupied):
    intervals = choose_intervals(source, probe, seconds=900, count=1000000, seed=42)
    return [row for row in intervals if not any(source["sha256"] == old["sha256"]
        and max(row["start"], old["start"]) < min(row["end"], old["end"]) for old in occupied)]


def make_plan(args):
    output = args.output.resolve()
    paths = [args.resources, args.works, args.rankings, args.split_registry, *args.reuse_manifest]
    contract = {"inputs": [binding(path) for path in paths], "source_roots": [str(p.resolve()) for p in args.source_root],
                "new_works_limit": args.new_works, "seed": 42, "code": binding(__file__)}
    contract["prior_cpu_sources"] = [binding(path) for root in args.source_root
                                    for path in sorted((root / "sources").glob("*/source.json"))]
    dependencies = ["narrative_sources", "narrative_visual", "narrative_prepare", "io"]
    contract["helpers"] = {name: binding(Path(__file__).with_name(name + ".py")) for name in dependencies}
    contract["prepare_script"] = binding(ROOT / "scripts/prepare_legacy_movie_sources.py")
    path = output / "selection.json"
    if path.exists():
        value = read_json(path)
        if value["contract"] != contract:
            raise ValueError("Bulk source preparation inputs or code changed; preserve the existing version")
        if read_json(output / "split_registry.json") != value["split_registry"]:
            raise ValueError("Bulk source-work split registry changed")
        return value
    registry, reuse = read_json(args.split_registry), {}
    for manifest in args.reuse_manifest:
        for row in read_jsonl(manifest):
            if (row.get("sha256") and row.get("local_path") and Path(row["local_path"]).is_file()):
                if registry[row["source_group"]]["split"] != row["split"]:
                    raise ValueError("Existing source-work split conflict")
                if row["sha256"] in reuse and reuse[row["sha256"]]["source_id"] != row["source_id"]:
                    raise ValueError("Different existing source IDs have identical content; inspect provenance")
                reuse[row["sha256"]] = row
    rows, registry = choose_works(read_jsonl(args.works), read_jsonl(args.resources), read_jsonl(args.rankings),
                                  registry, output, limit=args.new_works)
    occupied = []
    for root in args.source_root:
        for source in sorted((root / "sources").glob("*/source.json")):
            item = read_json(source)
            occupied.append({"sha256": item["sha256"], "start": item["stream_start"], "end": item["stream_end"]})
    plan = {"contract": contract, "selected_new": rows, "reused_sources": list(reuse.values()),
            "split_registry": registry, "occupied_intervals": occupied,
            "original_workflow": "64-second per-window ANNOTATE followed by original V4 generation/qualification",
            "stream_seconds": 900, "cpu_workers": 2, "download_workers": 1,
            "get_spacing_seconds": 2, "retry_limit": 5, "model_calls": 0, "training_eligible": False}
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "split_registry.json", registry)
    write_json(path, plan)
    return plan


def require_pilot_complete(directory):
    if directory is None:
        raise ValueError("Provide the completed four-arm pilot before bulk execution")
    state = read_json(Path(directory) / "execution_state.json")
    paired = read_json(Path(directory) / "paired_comparison.json")
    expected = {"baseline", "embedding_only", "reader_only", "joint"}
    if (state.get("status") != "complete" or set(state.get("completed_arms", [])) != expected
            or paired.get("status") != "passed" or set(paired.get("summaries", {})) != expected):
        raise ValueError("Wait for completed pilot scores and matched-input checks before expansion")


def execute(plan, args):
    require_pilot_complete(args.pilot_output)
    output = args.output.resolve()
    spec = importlib.util.spec_from_file_location("pearl_scale_cpu_preparation", ROOT / "scripts/prepare_legacy_movie_sources.py")
    prepare = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prepare)
    dense = output / "dense"
    cpu_args = SimpleNamespace(ffmpeg=args.ffmpeg.resolve(strict=True), ffprobe=args.ffprobe.resolve(strict=True),
                               long_side=640, window_stride=64, timeout=3600)
    prepared, pending, raw_ids, completed, downloads = {}, {}, {}, {}, {}
    state = {"status": "running", "pid": os.getpid(), "model_calls": 0, "qualified_qa_created": 0, "cpu_workers": 2,
             "download_workers": 1, "selected_new": len(plan["selected_new"]), "reused": len(plan["reused_sources"])}
    def save():
        write_jsonl(output / "source_manifest.jsonl", sorted(completed.values(), key=lambda row: row["source_id"]))
        write_json(output / "progress.json", {**state, "updated_unix": time.time(), "sources_ready": len(completed),
            "downloads": downloads, "cpu_submitted": len(pending),
            "cpu_ready": sum(row["status"] == "cpu_ready" for row in prepared.values()),
            "cpu_held": sum(row["status"] != "cpu_ready" for row in prepared.values())})
    def collect():
        for future in list(pending):
            if future.done():
                stream_id = pending.pop(future)
                prepared[stream_id] = future.result()
    def enqueue(source, pool):
        previous = raw_ids.setdefault(source["sha256"], source["source_id"])
        if previous != source["source_id"]:
            raise ValueError("Different source IDs have identical downloaded content; inspect groups/splits before use")
        if plan["split_registry"][source["source_group"]]["split"] != source["split"]:
            raise ValueError("Source split changed after planning")
        completed[source["source_id"]] = source
        probe = probe_source(source, cpu_args.ffprobe)
        for interval in candidate_intervals(source, probe, plan["occupied_intervals"]):
            identity = f"{source['source_id']}\0{source['sha256']}\0{interval['start']}\0{interval['end']}"
            sid = hashlib.sha256(identity.encode()).hexdigest()[:20]
            directory = dense / "sources" / sid
            receipt = directory / "preparation_status.json"
            if receipt.exists():
                status = read_json(receipt)
                if status["status"] == "cpu_ready" and sha256_file(status["source_manifest"]) != status["source_manifest_sha256"]:
                    raise ValueError("Published CPU source changed")
                prepared[sid] = status  # Preserve held attempts; do not silently reset them.
                continue
            if directory.exists():
                raise RuntimeError(f"Interrupted CPU source has no completion receipt; preserve and review {directory}")
            item = {"stream_id": sid, "source_id": source["source_id"], "source_group": source["source_group"],
                "split": source["split"], "sha256": source["sha256"], "source_sha256": source["sha256"],
                "local_path": source["local_path"], "source_clock": {"origin_pts_seconds": probe["origin_pts_seconds"],
                    "definition": CLOCK_DEFINITION}, "interval": interval,
                "benchmark_source_policy": POLICY, "benchmark_overlap": source.get("benchmark_overlap", {"status": "unknown"})}
            pending[pool.submit(prepare.prepare_one, item, dense, cpu_args)] = sid
    with (output / ".source-pipeline.lock").open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            with ThreadPoolExecutor(max_workers=2) as pool:
                for source in plan["reused_sources"]:
                    enqueue(source, pool)
                    collect()
                    save()
                limiter = DownloadRateLimit()
                for source in plan["selected_new"]:
                    result = download_one(source, output, cpu_args.ffprobe, limiter=limiter, max_retries=5)
                    downloads[source["source_id"]] = result
                    if result["status"] == "complete":
                        enqueue({**source, "sha256": result["sha256"], "bytes": result["bytes"],
                                 "download_receipt": result}, pool)
                    collect()
                    save()
                    print({"source": source["source_id"], "download": result["status"],
                           "sources_ready": len(completed), "cpu_ready": sum(r["status"] == "cpu_ready" for r in prepared.values())}, flush=True)
                for future in as_completed(list(pending)):
                    sid = pending.pop(future)
                    prepared[sid] = future.result()
                    save()
            state["status"] = "complete_with_held" if any(row["status"] != "complete" for row in downloads.values()) or any(
                row["status"] != "cpu_ready" for row in prepared.values()) else "complete"
            save()
        except BaseException as error:
            state.update(status="needs_review", error={"type": type(error).__name__, "message": str(error)})
            save()
            raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("resources", "works", "rankings", "split-registry", "output", "ffmpeg", "ffprobe"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--reuse-manifest", type=Path, action="append", required=True)
    p.add_argument("--source-root", type=Path, action="append", required=True)
    p.add_argument("--new-works", type=int, default=90)
    p.add_argument("--pilot-output", type=Path)
    p.add_argument("--execute", action="store_true")
    args = p.parse_args()
    if not 1 <= args.new_works <= 200:
        p.error("Use a bounded selection from the existing200-work metadata pilot")
    plan = make_plan(args)
    print({"selected_new": len(plan["selected_new"]), "reused_sources": len(plan["reused_sources"]), "model_calls": 0}, flush=True)
    if args.execute:
        execute(plan, args)


if __name__ == "__main__":
    main()
