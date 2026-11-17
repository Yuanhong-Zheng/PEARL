"""Freeze six metadata-prioritized works, then bounded resumable download of their source videos.

The source videos are YouTube videos that users download themselves and serve from
an S3-compatible store; set ENDPOINT below to that store. No teacher calls, QA
labels, training, or edits to original datasets. Only one unsigned endpoint is
supported. Benchmark overlap is authorized;
source-work groups still receive one persistent train/validation assignment.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import math
from pathlib import Path
import random
import re
import shutil
import subprocess
import threading
import time
import unicodedata

from .io import code_revision, read_json, read_jsonl, sha256_file, write_json, write_jsonl


PROTOCOL = "narrative-source-preparation-v1"
ENDPOINT = "http://your-s3-endpoint"  # placeholder: S3-compatible store holding the downloaded YouTube videos
POLICY = "overlap_allowed_by_user"
TRANSPORT_REVISION = "single-get-with-resume-byte-overlap-v2"


def title_family(title):
    """Conservative alias grouping: remakes/seasons/numbered installments join.

    This can overgroup unrelated names. It is a split-leakage precaution, not an
    assertion that names prove identity. Preserve both IMDb and all raw aliases.
    """
    value = unicodedata.normalize("NFKC", title).casefold().strip()
    value = re.sub(r"[\[(（][^\])）]*[\])）]", " ", value)
    value = re.split(r"[:：]", value, maxsplit=1)[0]
    value = re.sub(r"(?:第[0-9一二三四五六七八九十百]+[季集部章]|\b(?:season|episode|part|chapter)\s*[0-9ivx]+\b|\bs\d+(?:e\d+)?\b).*$", "", value)
    value = re.sub(r"(?:\s+|[-_])(?:\d{1,4}|[ivx]{1,6})\s*$", "", value)
    return "".join(ch for ch in value if ch.isalnum())


def group_works(works):
    """Union across all catalog aliases before ranking/split assignment."""
    parent = {w["work_id"]: w["work_id"] for w in works}

    def find(value):
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(a, b):
        a, b = find(a), find(b)
        if a != b:
            parent[max(a, b)] = min(a, b)

    aliases, normalized = {}, {}
    for work in sorted(works, key=lambda x: x["work_id"]):
        wid = work["work_id"]
        if not re.fullmatch(r"imdb:tt\d+", wid):
            raise ValueError("Explicit IMDb work IDs are required")
        normalized[wid] = sorted({title_family(t) for t in work["titles"] if len(title_family(t)) >= 2})
        for alias in normalized[wid]:
            if alias in aliases:
                union(wid, aliases[alias])
            else:
                aliases[alias] = wid
    members = {}
    for work in works:
        members.setdefault(find(work["work_id"]), []).append(work["work_id"])
    groups = {}
    for canonical, ids in members.items():
        group = {"source_group": canonical, "member_work_ids": sorted(ids),
                 "normalized_title_families": sorted({a for wid in ids for a in normalized[wid]}),
                 "grouping_policy": "IMDb plus transitive conservative normalized title-family equality; overgrouping possible"}
        for wid in ids:
            groups[wid] = group
    return groups


def s3_location(resource_id):
    match = re.fullmatch(r"upload:videoBucket:([^:\s]+)", resource_id)
    if not match:
        raise ValueError("Only the proved upload:videoBucket namespace is supported")
    return {"endpoint": ENDPOINT, "bucket": "upload-videoBucket", "key": match[1],
            "authentication": "UNSIGNED", "addressing_style": "path"}


def freeze_selection(resources, works, rankings, output, *, pilot=None, seed=42):
    """Write a frozen 5/1 split and resource plan before any remote request."""
    output = Path(output).resolve()
    inputs = {name: {"path": str(Path(path).resolve()), "sha256": sha256_file(path)}
              for name, path in (("resources", resources), ("works", works), ("rankings", rankings))}
    if pilot is not None:
        inputs["pilot"] = {"path": str(Path(pilot).resolve()), "sha256": sha256_file(pilot)}
    plan_path = output / "selection.json"
    if plan_path.exists():
        plan = read_json(plan_path)
        if plan["inputs"] != inputs or plan["seed"] != seed:
            raise ValueError("Frozen input hashes or seed changed; use a new version directory")
        registry = read_json(output / "split_registry.json")
        if registry != plan["split_registry"]:
            raise ValueError("Frozen source-work split registry changed")
        return plan
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use an empty version directory for a new source selection")
    all_works, ranked = read_jsonl(works), read_jsonl(rankings)
    by_work = {w["work_id"]: w for w in all_works}
    if len(by_work) != len(all_works) or len({r["work_id"] for r in ranked}) != len(ranked):
        raise ValueError("Duplicate work IDs in catalog or recovered rankings")
    groups = group_works(all_works)
    needed = {rid for row in ranked if row["priority"] in {"high", "medium"}
              for rid in by_work[row["work_id"]]["resource_ids"]}
    resource_map = {}
    with Path(resources).open() as stream:
        for line in stream:
            row = json.loads(line)
            if row["resource_id"] in needed:
                resource_map[row["resource_id"]] = row
    candidates, review = [], []
    for rank in ranked:
        work = by_work[rank["work_id"]]
        valid = [resource_map[rid] for rid in work["resource_ids"] if rid in resource_map
                 and not resource_map[rid].get("initial_rule_flags")
                 and resource_map[rid].get("duration", 0) >= 900
                 and min(resource_map[rid].get("width", 0), resource_map[rid].get("height", 0)) >= 480
                 and rid.startswith("upload:videoBucket:")]
        if rank["priority"] not in {"high", "medium"} or not valid:
            review.append({"work_id": work["work_id"], "ranking": rank,
                           "status": "retained_for_review", "hard_rejected": False,
                           "reason": "metadata_not_high_medium" if rank["priority"] not in {"high", "medium"} else "no_initial_900s_resource"})
            continue
        # Lowest adequate dimensions limits this initial screening download; this
        # is a choice among supplied raw resources, never a transcode.
        valid.sort(key=lambda r: (min(r["width"], r["height"]), -r["duration"], r["resource_id"]))
        candidates.append({"work": work, "ranking": rank, "resource": valid[0],
                           "group": groups[work["work_id"]],
                           "tie": hashlib.sha256(f"{seed}\0{work['work_id']}".encode()).hexdigest()})
    selected, used_groups, used_genres = [], set(), set()
    while len(selected) < 6:
        available = [c for c in candidates if c["group"]["source_group"] not in used_groups]
        if not available:
            raise ValueError("At least six different eligible source-work groups are required")
        choice = min(available, key=lambda c: (0 if c["ranking"]["priority"] == "high" else 1,
                     (c["work"]["genres"] or ["Unknown"])[0] in used_genres, c["tie"]))
        selected.append(choice)
        used_groups.add(choice["group"]["source_group"])
        used_genres.add((choice["work"]["genres"] or ["Unknown"])[0])
    validation_index = random.Random(seed).randrange(6)
    rows, registry = [], {}
    for index, choice in enumerate(selected):
        work, resource, group = choice["work"], choice["resource"], choice["group"]
        split = "validation" if index == validation_index else "train"
        registry[group["source_group"]] = {**group, "split": split}
        source_id = resource["resource_id"]
        basename = hashlib.sha256(source_id.encode()).hexdigest()[:20]
        rows.append({"source_id": source_id, "source_group": group["source_group"], "work_id": work["work_id"],
                     "split": split, "titles": work["titles"], "genres": work["genres"], "years": work.get("years", []),
                     "metadata_ranking": choice["ranking"], "group_provenance": group,
                     "resource_metadata": resource, "s3": s3_location(source_id),
                     "local_path": str(output / "videos" / f"{basename}.source"),
                     "benchmark_source_policy": POLICY, "benchmark_overlap": {"status": "unknown"},
                     "screening_only": True, "training_eligible": False})
    chosen_work_ids = {r["work_id"] for r in rows}
    for choice in candidates:
        if choice["work"]["work_id"] not in chosen_work_ids:
            review.append({"work_id": choice["work"]["work_id"], "ranking": choice["ranking"],
                           "status": "retained_for_review", "hard_rejected": False,
                           "reason": "not_selected_in_bounded_initial_six"})
    if pilot is not None:
        ranked_ids = {r["work_id"] for r in ranked}
        for work in read_jsonl(pilot):
            if work["work_id"] not in ranked_ids:
                review.append({"work_id": work["work_id"], "status": "retained_for_review", "hard_rejected": False,
                               "reason": "unresolved_metadata_ranking", "metadata": work})
    plan = {"protocol": PROTOCOL, "inputs": inputs, "seed": seed, "selected": rows, "split_registry": registry,
            "selection_policy": "Luna metadata high before medium; prefer unused primary genre; SHA256(seed,work_id) tie order; six distinct conservative work groups; seed42 one validation group",
            "resources_policy": "one lowest-adequate-resolution raw resource per work, duration>=900s; no transcode",
            "benchmark_source_policy": POLICY, "source_independent_from_benchmark_claimed": False,
            "revision": code_revision(), "screening_only": True, "training_eligible": False}
    output.mkdir(parents=True, exist_ok=True)
    # Registry is physically persisted before plan completion and any downloads.
    write_json(output / "split_registry.json", registry)
    write_jsonl(output / "review_pool.jsonl", sorted(review, key=lambda r: r["work_id"]))
    write_json(plan_path, plan)
    return plan


def create_s3_client():
    import boto3
    from botocore import UNSIGNED
    from botocore.config import Config
    session = boto3.session.Session()
    return session.client("s3", endpoint_url=ENDPOINT, config=Config(
        signature_version=UNSIGNED, s3={"addressing_style": "path"},
        retries={"total_max_attempts": 1}, connect_timeout=15, read_timeout=90,
        proxies={}, max_pool_connections=2))


def freeze_preflight_replacement(output):
    """Version the tiny preflight coverage choice; never rewrite v1 or its split."""
    output = Path(output).resolve()
    parent = read_json(output / "selection.json")
    parent_hash = sha256_file(output / "selection.json")
    destination = output / "selection-v2.json"
    if destination.exists():
        frozen = read_json(destination)
        if frozen["parent_selection_sha256"] != parent_hash:
            raise ValueError("Parent frozen selection changed")
        return frozen
    if read_json(output / "split_registry.json") != parent["split_registry"]:
        raise ValueError("Original persistent split registry changed")
    for record in parent["inputs"].values():
        if sha256_file(record["path"]) != record["sha256"]:
            raise ValueError("Frozen metadata inputs changed before preflight replacement")
    removed = [r for r in parent["selected"] if "Adult" in r["genres"]]
    if len(removed) != 1 or removed[0]["split"] != "train":
        raise ValueError("This explicit versioning step expects one Adult-tagged training work")
    works = read_jsonl(parent["inputs"]["works"]["path"])
    by_work = {w["work_id"]: w for w in works}
    groups = group_works(works)
    registry = dict(parent["split_registry"])
    # The initial registry only assigned selected groups. New groups receive
    # train now, before any downloads/QA; existing assignments are never changed.
    ranks = [r for r in read_jsonl(parent["inputs"]["rankings"]["path"])
             if r["priority"] == "high" and (by_work[r["work_id"]]["genres"] or [None])[0] in {"Drama", "Comedy", "Animation"}
             and "Adult" not in by_work[r["work_id"]]["genres"]
             and groups[r["work_id"]]["source_group"] not in registry]
    needed = {rid for rank in ranks for rid in by_work[rank["work_id"]]["resource_ids"]}
    resources = {}
    with Path(parent["inputs"]["resources"]["path"]).open() as stream:
        for line in stream:
            row = json.loads(line)
            if row["resource_id"] in needed and not row.get("initial_rule_flags") and row.get("duration", 0) >= 900 and min(row.get("width", 0), row.get("height", 0)) >= 480:
                resources[row["resource_id"]] = row
    ranks.sort(key=lambda rank: hashlib.sha256(f"{parent['seed']}\0{rank['work_id']}".encode()).hexdigest())
    replacement = None
    for rank in ranks:
        work = by_work[rank["work_id"]]
        options = [resources[rid] for rid in work["resource_ids"] if rid in resources and rid.startswith("upload:videoBucket:")]
        if not options:
            continue
        resource = min(options, key=lambda r: (min(r["width"], r["height"]), -r["duration"], r["resource_id"]))
        group = groups[work["work_id"]]
        source_id = resource["resource_id"]
        replacement = {"source_id": source_id, "source_group": group["source_group"], "work_id": work["work_id"],
                       "split": "train", "titles": work["titles"], "genres": work["genres"], "years": work.get("years", []),
                       "metadata_ranking": rank, "group_provenance": group, "resource_metadata": resource,
                       "s3": s3_location(source_id), "local_path": str(output / "videos" / f"{hashlib.sha256(source_id.encode()).hexdigest()[:20]}.source"),
                       "benchmark_source_policy": POLICY, "benchmark_overlap": {"status": "unknown"},
                       "screening_only": True, "training_eligible": False}
        registry[group["source_group"]] = {**group, "split": "train", "assignment_provenance": "new group assigned train before QA for v2 tiny-preflight domain coverage; original registry only assigned selected groups"}
        break
    if replacement is None:
        raise ValueError("No distinct high-priority mainstream-domain replacement is available")
    reason = "Tiny initial preflight covers mainstream Drama/Comedy/Animation; Adult-tagged work retained for review, not a global genre exclusion or visual-quality judgment"
    rows = [replacement if r["source_id"] == removed[0]["source_id"] else r for r in parent["selected"]]
    review = [r for r in read_jsonl(output / "review_pool.jsonl") if r["work_id"] != replacement["work_id"]]
    review.append({"work_id": removed[0]["work_id"], "ranking": removed[0]["metadata_ranking"],
                   "status": "retained_for_review", "hard_rejected": False, "reason": reason})
    frozen = {**parent, "selected": rows, "split_registry": registry, "selection_version": "v2",
              "parent_selection_sha256": parent_hash, "pilot_coverage_replacement": {
                  "removed_work_id": removed[0]["work_id"], "replacement_work_id": replacement["work_id"], "reason": reason}}
    write_json(output / "split_registry-v2.json", registry)
    write_jsonl(output / "review_pool-v2.jsonl", sorted(review, key=lambda r: r["work_id"]))
    write_json(destination, frozen)
    return frozen


def freeze_six_work_extension(output):
    """Append six new groups without touching old files, assignments or budgets."""
    output = Path(output).resolve()
    parent_path = output / "selection-v2.json"
    parent = read_json(parent_path)
    parent_hash = sha256_file(parent_path)
    destination = output / "selection-v3.json"
    if destination.exists():
        frozen = read_json(destination)
        if (frozen["parent_selection_sha256"] != parent_hash
                or read_json(output / "split_registry-v3.json") != frozen["split_registry"]):
            raise ValueError("Frozen extension parent or registry changed")
        return frozen
    if read_json(output / "split_registry-v2.json") != parent["split_registry"]:
        raise ValueError("Existing global source-work registry changed")
    for record in parent["inputs"].values():
        if sha256_file(record["path"]) != record["sha256"]:
            raise ValueError("Extension must use the same frozen metadata/rankings")
    works = read_jsonl(parent["inputs"]["works"]["path"])
    by_work = {w["work_id"]: w for w in works}
    groups = group_works(works)
    registry = dict(parent["split_registry"])
    for gid, registered in registry.items():
        if any(groups[wid]["source_group"] != gid for wid in registered["member_work_ids"]):
            raise ValueError("Catalog title-family aliases conflict with an existing group")
    ranks = [r for r in read_jsonl(parent["inputs"]["rankings"]["path"])
             if r["priority"] in {"high", "medium"} and groups[r["work_id"]]["source_group"] not in registry]
    needed = {rid for rank in ranks for rid in by_work[rank["work_id"]]["resource_ids"]}
    resources = {}
    with Path(parent["inputs"]["resources"]["path"]).open() as stream:
        for line in stream:
            row = json.loads(line)
            if (row["resource_id"] in needed and not row.get("initial_rule_flags")
                    and row.get("duration", 0) >= 900 and min(row.get("width", 0), row.get("height", 0)) >= 480
                    and row["resource_id"].startswith("upload:videoBucket:")):
                resources[row["resource_id"]] = row
    candidates = []
    existing_sources = {r["source_id"] for r in parent["selected"]}
    for rank in ranks:
        work = by_work[rank["work_id"]]
        options = [resources[rid] for rid in work["resource_ids"] if rid in resources and rid not in existing_sources]
        if options:
            candidates.append({"work": work, "rank": rank, "group": groups[work["work_id"]],
                "resource": min(options, key=lambda r: (min(r["width"], r["height"]), -r["duration"], r["resource_id"])),
                "tie": hashlib.sha256(f"{parent['seed']}\0{work['work_id']}".encode()).hexdigest()})
    selected, used_groups = [], set(registry)
    used_genres = {(r["genres"] or ["Unknown"])[0] for r in parent["selected"]}
    while len(selected) < 6:
        eligible = [c for c in candidates if c["group"]["source_group"] not in used_groups]
        if not eligible:
            raise ValueError("Six new eligible work groups are required for the bounded extension")
        # Carry forward tiny-preflight coverage preference, never a quality
        # verdict or global exclusion of Adult-tagged catalog works.
        chosen = min(eligible, key=lambda c: (0 if c["rank"]["priority"] == "high" else 1,
            "Adult" in c["work"]["genres"], (c["work"]["genres"] or ["Unknown"])[0] in used_genres, c["tie"]))
        selected.append(chosen)
        used_groups.add(chosen["group"]["source_group"])
        used_genres.add((chosen["work"]["genres"] or ["Unknown"])[0])
    validation_index = random.Random(f"{parent['seed']}:source-extension-v3").randrange(6)
    new_rows = []
    for index, chosen in enumerate(selected):
        work, resource, group = chosen["work"], chosen["resource"], chosen["group"]
        split = "validation" if index == validation_index else "train"
        sid = resource["resource_id"]
        row = {"source_id": sid, "source_group": group["source_group"], "work_id": work["work_id"],
            "split": split, "titles": work["titles"], "genres": work["genres"], "years": work.get("years", []),
            "metadata_ranking": chosen["rank"], "group_provenance": group, "resource_metadata": resource,
            "s3": s3_location(sid), "local_path": str(output / "videos" / f"{hashlib.sha256(sid.encode()).hexdigest()[:20]}.source"),
            "benchmark_source_policy": POLICY, "benchmark_overlap": {"status": "unknown"},
            "screening_only": True, "training_eligible": False}
        new_rows.append(row)
        registry[group["source_group"]] = {**group, "split": split,
            "assignment_provenance": "new source-work family assigned before downloads/QA in seed42 bounded six-work v3 extension"}
    states = []
    for row in parent["selected"]:
        state_path = output / "downloads" / f"{Path(row['local_path']).stem}.state.json"
        if state_path.exists():
            state = read_json(state_path)
            states.append({"source_id": row["source_id"], "state_sha256": sha256_file(state_path),
                "attempts": state["attempts"], "status": state["status"], "bytes": state.get("bytes"), "sha256": state.get("sha256")})
    new_ids = {r["work_id"] for r in new_rows}
    review = [r for r in read_jsonl(output / "review_pool-v2.jsonl") if r["work_id"] not in new_ids]
    extension = {"reason": "Additional source coverage for approved 500-QA stage before possible 10k expansion; metadata prioritization only",
        "previous_count": 6, "added_count": 6, "total_count": 12,
        "new_source_ids": [r["source_id"] for r in new_rows], "new_work_ids": [r["work_id"] for r in new_rows],
        "new_split_counts": {"train": 5, "validation": 1},
        "preserved_previous_download_states": states, "old_retry_budgets_reset": False,
        "failed_previous_objects_hard_rejected": False,
        "grouping_catalog_sha256": parent["inputs"]["works"]["sha256"],
        "selection_policy": "high before medium; retain tiny-preflight mainstream coverage preference; prefer primary genres unused in prior/added works; seed42 SHA256 tie order"}
    frozen = {**parent, "selected": parent["selected"] + new_rows, "split_registry": registry,
        "selection_version": "v3", "parent_selection_sha256": parent_hash, "extension": extension}
    write_json(output / "split_registry-v3.json", registry)
    write_jsonl(output / "review_pool-v3.jsonl", sorted(review, key=lambda r: r["work_id"]))
    write_json(output / "extension_audit-v3.json", extension)
    write_json(destination, frozen)
    return frozen


def selection_suffix(version):
    if version not in {"v1", "v2", "v3"}:
        raise ValueError("Unknown frozen selection version")
    return "" if version == "v1" else "-" + version


def probe_download(path, ffprobe):
    command = [str(ffprobe), "-v", "error", "-show_entries",
               "format=duration,size,format_name,start_time:stream=codec_type,width,height,duration,time_base,start_time",
               "-of", "json", str(path)]
    raw = json.loads(subprocess.run(command, capture_output=True, check=True, text=True, timeout=120).stdout)
    videos = [s for s in raw.get("streams", []) if s.get("codec_type") == "video"]
    duration = float(raw.get("format", {}).get("duration", 0))
    if not videos or not math.isfinite(duration) or duration < 900 or min(videos[0].get("width", 0), videos[0].get("height", 0)) < 480:
        raise ValueError("Downloaded raw resource is not an adequate >=900s, >=480p video")
    if int(raw["format"]["size"]) != Path(path).stat().st_size:
        raise ValueError("ffprobe file size disagrees with the local download")
    return {"command": command, "result": raw}


class DownloadRateLimit:
    """Share server-request start spacing and Retry-After across two workers."""

    def __init__(self, minimum_interval=2.0, *, sleep=time.sleep, clock=time.monotonic):
        self.lock = threading.Lock()
        self.next_request = 0.0
        self.interval, self.sleep, self.clock = minimum_interval, sleep, clock

    def wait(self):
        while True:
            with self.lock:
                now = self.clock()
                delay = max(0.0, self.next_request - now)
                if delay <= 0:
                    self.next_request = now + self.interval
                    return
            # Recheck after sleeping: another worker may have extended a shared
            # Retry-After embargo while this request was waiting for its turn.
            self.sleep(min(delay, 30))

    def defer(self, seconds):
        with self.lock:
            self.next_request = max(self.next_request, self.clock() + seconds)


def _error_details(error):
    response = getattr(error, "response", {}) or {}
    metadata = response.get("ResponseMetadata", {})
    status = metadata.get("HTTPStatusCode")
    headers = metadata.get("HTTPHeaders", {})
    retry = headers.get("retry-after")
    try:
        seconds = max(0.0, float(retry)) if retry is not None else 0.0
    except (ValueError, TypeError):
        from email.utils import parsedate_to_datetime
        try:
            seconds = max(0.0, parsedate_to_datetime(retry).timestamp() - time.time())
        except (ValueError, TypeError, OverflowError):
            seconds = 0.0
    return {"type": type(error).__name__, "message": str(error)[:1000], "http_status": status,
            "code": response.get("Error", {}).get("Code"), "retry_after_seconds": seconds}


def download_one(row, output, ffprobe, *, client_factory=create_s3_client, limiter=None,
                 max_retries=5, reuse=None, sleep=time.sleep):
    """ContentLength-checked resumable stream; keep partial/state/error history."""
    if type(max_retries) is not int or not 0 <= max_retries <= 5:
        raise ValueError("At most five retries beyond the initial attempt are allowed")
    output = Path(output).resolve()
    path = Path(row["local_path"]).resolve()
    if not path.is_relative_to(output / "videos") or row["s3"] != s3_location(row["source_id"]):
        raise ValueError("Frozen local path or proved S3 location was changed")
    path.parent.mkdir(parents=True, exist_ok=True)
    name = path.stem
    state_dir = output / "downloads"
    state_dir.mkdir(parents=True, exist_ok=True)
    state_path = state_dir / f"{name}.state.json"
    part = path.with_suffix(path.suffix + ".part")
    state = read_json(state_path) if state_path.exists() else {"source_id": row["source_id"], "attempts": 0, "errors": []}
    if state["source_id"] != row["source_id"]:
        raise ValueError("Download state belongs to another source")
    if state.get("status") == "verified" and not path.exists() and part.exists():
        if (not state.get("ffprobe") or part.stat().st_size != state["bytes"]
                or sha256_file(part) != state["sha256"]):
            raise ValueError("Verified preserved partial changed before finalization")
        part.replace(path)
        state.update(status="complete", local_path=str(path), downloaded_bytes=state["bytes"])
        write_json(state_path, state)
        return state
    if state.get("status") in {"complete", "verified"} and path.exists():
        if path.stat().st_size != state["bytes"] or sha256_file(path) != state["sha256"]:
            raise ValueError("Frozen completed source content changed")
        state["status"] = "complete"
        write_json(state_path, state)
        return state
    if path.exists():
        raise ValueError("Existing final file has no verified completion receipt")
    if reuse is not None and not part.exists() and not path.exists():
        reused = Path(reuse["local_path"])
        if reuse["source_id"] != row["source_id"] or sha256_file(reused) != reuse["sha256"]:
            raise ValueError("Reusable source identity/hash mismatch")
        shutil.copyfile(reused, part)
        state["reuse_provenance"] = reuse
    limiter = limiter or DownloadRateLimit()
    client = client_factory()
    while state["attempts"] < max_retries + 1:
        state.update(attempts=state["attempts"] + 1, status="requesting", local_partial=str(part))
        write_json(state_path, state)
        body = None
        request_record = None
        try:
            offset = part.stat().st_size if part.exists() else 0
            # Fresh objects need one full GET. Resume one byte before the local
            # offset: validate the overlap before appending, and allow an already
            # complete preserved .part to finish without a redundant probe/416.
            request_start = max(0, offset - 1)
            parameters = {"Bucket": row["s3"]["bucket"], "Key": row["s3"]["key"]}
            if offset:
                parameters["Range"] = f"bytes={request_start}-"
            limiter.wait()
            request_record = {"attempt": state["attempts"], "operation": "GetObject",
                "transport_revision": TRANSPORT_REVISION,
                "started_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "parameters": parameters, "local_offset": offset,
                "overlap_bytes": 1 if offset else 0, "result": "requesting"}
            state.setdefault("requests", []).append(request_record)
            request_log = state_dir / f"{name}.get.{state['attempts']}.json"
            write_json(request_log, request_record)
            write_json(state_path, state)
            response = client.get_object(**parameters)
            body = response["Body"]
            request_record.update(result="headers_received",
                headers_received_at_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                http_status=response.get("ResponseMetadata", {}).get("HTTPStatusCode"),
                request_id=response.get("ResponseMetadata", {}).get("RequestId"),
                content_length=response.get("ContentLength"), content_range=response.get("ContentRange"),
                etag=response.get("ETag"), version_id=response.get("VersionId"))
            write_json(request_log, request_record)
            content_range = response.get("ContentRange")
            match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", content_range or "")
            if content_range and not match:
                raise ValueError("S3 returned a malformed ContentRange")
            if offset and not match:
                raise ValueError("Server ignored resume Range; preserving partial")
            if not match:
                expected = int(response["ContentLength"])
            else:
                start, stop, expected = map(int, match.groups())
                if start != request_start or stop != expected - 1:
                    raise ValueError("S3 ContentRange disagrees with measured resume offset")
            if (expected < max(offset, 1) or int(response["ContentLength"]) != expected - request_start
                    or not isinstance(response.get("ETag"), str) or not response["ETag"]):
                raise ValueError("S3 ContentLength/ETag/ContentRange is invalid")
            identity = {"etag": response.get("ETag"), "version_id": response.get("VersionId"), "content_length": expected}
            if state.get("remote") and state["remote"] != identity:
                raise ValueError("Remote object identity changed; preserving partial for review")
            state.update(remote=identity, status="downloading", expected_bytes=expected, downloaded_bytes=offset)
            write_json(state_path, state)
            received = 0
            if offset:
                with part.open("rb") as stream:
                    stream.seek(offset - 1)
                    original_byte = stream.read(1)
                overlap = body.read(1)
                if not overlap:
                    raise ValueError("Response ended before the resume overlap byte")
                if overlap != original_byte:
                    raise ValueError("Resume overlap byte differs; preserving partial for review")
                received = 1
            last_report = time.monotonic()
            with part.open("ab") as stream:
                while True:
                    chunk = body.read(8 * 1024 * 1024)
                    if not chunk:
                        break
                    received += len(chunk)
                    if received > int(response["ContentLength"]):
                        raise ValueError("Response exceeded declared ContentLength")
                    stream.write(chunk)
                    state["downloaded_bytes"] += len(chunk)
                    if time.monotonic() - last_report >= 5:
                        stream.flush()
                        write_json(state_path, state)
                        last_report = time.monotonic()
            if part.stat().st_size != expected or received != int(response["ContentLength"]):
                raise ValueError("Response ended before declared ContentLength")
            request_record.update(result="stream_complete", received_bytes=received,
                finished_at_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
            write_json(request_log, request_record)
            digest = sha256_file(part)
            probe = probe_download(part, ffprobe)
            state.update(status="verified", bytes=expected, sha256=digest, ffprobe=probe)
            write_json(state_path, state)
            part.replace(path)
            state.update(status="complete", bytes=expected, sha256=digest, ffprobe=probe,
                         local_path=str(path), downloaded_bytes=expected)
            write_json(state_path, state)
            return state
        except Exception as error:
            details = _error_details(error)
            details["attempt"] = state["attempts"]
            details["recorded_at_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            details["partial_bytes"] = part.stat().st_size if part.exists() else 0
            if request_record is not None:
                if request_record["result"] != "stream_complete":
                    request_record["result"] = "error"
                request_record.update(finished_at_utc=details["recorded_at_utc"],
                    http_status=details["http_status"] or request_record.get("http_status"),
                    error=details)
                write_json(request_log, request_record)
            state["errors"].append(details)
            state.update(status="failed_attempt", downloaded_bytes=details["partial_bytes"])
            write_json(state_path, state)
            write_json(state_dir / f"{name}.error.{state['attempts']}.json", details)
            terminal = details["http_status"] in {400, 401, 403, 404} or isinstance(error, (ValueError, subprocess.CalledProcessError)) and "ended before" not in str(error)
            if terminal or state["attempts"] >= max_retries + 1:
                state["status"] = "error"
                write_json(state_path, state)
                return state
            delay = max(min(2 ** state["attempts"], 32), details["retry_after_seconds"])
            if details["http_status"] == 429 or details["code"] in {"SlowDown", "RequestLimitExceeded"}:
                delay = max(delay, 60)
            limiter.defer(delay)
            sleep(delay)
        finally:
            if body is not None:
                body.close()
    state.update(status="error", retry_budget_exhausted=True)
    write_json(state_path, state)
    return state


def download_selection(output, ffprobe, *, workers=2, max_retries=5, reuse_manifest=None, selection_version="v1"):
    if type(workers) is not int or not 1 <= workers <= 2:
        raise ValueError("Initial preparation permits at most two concurrent downloads")
    output = Path(output).resolve()
    suffix = selection_suffix(selection_version)
    if selection_version == "v3" and workers != 1:
        raise ValueError("The v3 extension permits one download worker")
    selection_path = output / f"selection{suffix}.json"
    plan = read_json(selection_path)
    expected_count = 12 if selection_version == "v3" else 6
    if read_json(output / f"split_registry{suffix}.json") != plan["split_registry"] or len(plan["selected"]) != expected_count:
        raise ValueError("Frozen source-work split registry or bounded selection count changed")
    rows = plan["selected"]
    if len({r["source_group"] for r in rows}) != expected_count or sum(r["split"] == "validation" for r in rows) != expected_count // 6:
        raise ValueError("Distinct work groups and one validation work per six are required")
    for row in rows:
        frozen = plan["split_registry"][row["source_group"]]
        if (row["split"] != frozen["split"] or row["work_id"] not in frozen["member_work_ids"]
                or row["benchmark_source_policy"] != POLICY):
            raise ValueError("A selected work no longer matches its frozen group/split/policy")
    reuse = {r["source_id"]: r for r in read_jsonl(reuse_manifest)} if reuse_manifest else {}
    limiter = DownloadRateLimit()
    results = {}
    download_rows = rows
    if selection_version == "v3":
        added_ids = set(plan["extension"]["new_source_ids"])
        if len(added_ids) != 6 or not added_ids <= {r["source_id"] for r in rows}:
            raise ValueError("Only the frozen six new source objects may be downloaded")
        download_rows = [r for r in rows if r["source_id"] in added_ids]
        for row in rows:
            if row["source_id"] not in added_ids:
                state_path = output / "downloads" / f"{Path(row['local_path']).stem}.state.json"
                results[row["source_id"]] = read_json(state_path) if state_path.exists() else {"status": "deferred_unchanged"}
        pending = []
        for row in download_rows:
            state_path = output / "downloads" / f"{Path(row['local_path']).stem}.state.json"
            state = read_json(state_path) if state_path.exists() else None
            if state and (state.get("status") in {"complete", "error"}
                          or state.get("attempts", 0) >= max_retries + 1 and state.get("status") != "verified"):
                results[row["source_id"]] = state
            else:
                pending.append(row)
        download_rows = pending
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(download_one, row, output, ffprobe, limiter=limiter,
                              max_retries=max_retries, reuse=reuse.get(row["source_id"])): row for row in download_rows}
        for future in as_completed(futures):
            row = futures[future]
            try:
                results[row["source_id"]] = future.result()
            except Exception as error:
                results[row["source_id"]] = {"status": "error", "errors": [_error_details(error)]}
            print("SOURCE", row["work_id"], row["split"], results[row["source_id"]]["status"], flush=True)
            write_json(output / f"download_progress{suffix}.json", results)
            export_completed_manifest(output, selection_version=selection_version)
    manifest = []
    for row in rows:
        state = results[row["source_id"]]
        if state["status"] == "complete":
            manifest.append({**row, "sha256": state["sha256"], "bytes": state["bytes"],
                             "download_receipt": state, "selection_sha256": sha256_file(selection_path)})
    write_jsonl(output / f"source_manifest{suffix}.jsonl", manifest)
    summary = {"selected": len(rows), "completed": len(manifest), "errors": len(rows) - len(manifest),
               "completed_bytes": sum(r["bytes"] for r in manifest), "workers": workers,
               "network_scheduled_sources": len(download_rows),
               "benchmark_source_policy": POLICY, "screening_only": True, "training_eligible": False,
               "source_independent_from_benchmark_claimed": False}
    write_json(output / f"download_summary{suffix}.json", summary)
    return summary


def export_completed_manifest(output, *, selection_version="v2"):
    """Export only verified completed objects while other downloads continue."""
    output = Path(output).resolve()
    suffix = selection_suffix(selection_version)
    selection_path = output / f"selection{suffix}.json"
    plan = read_json(selection_path)
    if read_json(output / f"split_registry{suffix}.json") != plan["split_registry"]:
        raise ValueError("Frozen split registry changed")
    records = []
    for row in plan["selected"]:
        frozen = plan["split_registry"][row["source_group"]]
        if row["split"] != frozen["split"] or row["work_id"] not in frozen["member_work_ids"]:
            raise ValueError("Source work/split changed")
        path = Path(row["local_path"])
        state_path = output / "downloads" / f"{path.stem}.state.json"
        if not state_path.exists():
            continue
        state = read_json(state_path)
        if state.get("status") != "complete":
            continue
        if (state["source_id"] != row["source_id"] or not state.get("ffprobe")
                or path.stat().st_size != state["bytes"] or sha256_file(path) != state["sha256"]):
            raise ValueError("Completed source no longer matches its hash/probe receipt")
        records.append({**row, "sha256": state["sha256"], "bytes": state["bytes"],
                        "download_receipt": state, "selection_sha256": sha256_file(selection_path)})
    write_jsonl(output / f"source_manifest{suffix}.jsonl", records)
    write_jsonl(output / "source_manifest.jsonl", records)
    return {"manifest": str(output / "source_manifest.jsonl"), "selected": len(plan["selected"]),
            "completed": len(records), "selection_version": selection_version,
            "completed_bytes": sum(r["bytes"] for r in records)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    select = sub.add_parser("select", help="Freeze the bounded six-work plan; no network")
    select.add_argument("--resources", type=Path, required=True)
    select.add_argument("--works", type=Path, required=True)
    select.add_argument("--rankings", type=Path, required=True)
    select.add_argument("--pilot", type=Path)
    select.add_argument("--output", type=Path, required=True)
    select.add_argument("--seed", type=int, default=42)
    replace = sub.add_parser("preflight-v2", help="Version only the explicit mainstream-domain tiny-preflight replacement")
    replace.add_argument("--output", type=Path, required=True)
    extend = sub.add_parser("extend-v3", help="Freeze six additional ranked works while preserving existing splits/files/retry budgets")
    extend.add_argument("--output", type=Path, required=True)
    export = sub.add_parser("export-manifest", help="Export hash/probe-verified completed sources without network or interrupting active downloads")
    export.add_argument("--output", type=Path, required=True)
    export.add_argument("--selection-version", choices=["v1", "v2", "v3"], default="v2")
    download = sub.add_parser("download", help="Download only the frozen six source videos from the configured S3-compatible ENDPOINT")
    download.add_argument("--output", type=Path, required=True)
    download.add_argument("--ffprobe", type=Path, required=True)
    download.add_argument("--workers", type=int, default=2)
    download.add_argument("--max-retries", type=int, default=5)
    download.add_argument("--reuse-manifest", type=Path)
    download.add_argument("--selection-version", choices=["v1", "v2", "v3"], default="v1")
    args = parser.parse_args(argv)
    if args.command in {"select", "preflight-v2", "extend-v3"}:
        plan = (freeze_selection(args.resources, args.works, args.rankings, args.output, pilot=args.pilot, seed=args.seed)
                if args.command == "select" else freeze_preflight_replacement(args.output)
                if args.command == "preflight-v2" else freeze_six_work_extension(args.output))
        summary = {"selected": [{"work_id": r["work_id"], "titles": r["titles"], "genres": r["genres"],
                    "source_group": r["source_group"], "split": r["split"]} for r in plan["selected"]]}
    elif args.command == "export-manifest":
        summary = export_completed_manifest(args.output, selection_version=args.selection_version)
    else:
        summary = download_selection(args.output, args.ffprobe, workers=args.workers,
                                     max_retries=args.max_retries, reuse_manifest=args.reuse_manifest,
                                     selection_version=args.selection_version)
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
