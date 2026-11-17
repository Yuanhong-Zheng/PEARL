#!/usr/bin/env python3
"""Freeze qualified data and export reusable frozen-document vectors."""

import argparse
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
from pathlib import Path

import requests
import torch

from pearl_training.io import code_revision, read_jsonl, sha256_file, write_json, write_jsonl
from pearl_training.local_embeddings import LocalEmbeddings
from pearl_training.schema import check_split_isolation, validate_sample
from pearl_training.stream_audit import PROTOCOL_VERSION, require_repaired_sample


def source_overlap_metadata(rows):
    """Preserve known/unknown provenance; legacy benchmark rows remain known shared."""
    statuses = {}
    for sample in rows:
        record = sample.get("benchmark_overlap")
        status = record.get("status") if isinstance(record, dict) else None
        legacy = sample.get("source_overlap_with_original_benchmark")
        if status is None:
            status = "known_shared" if legacy is True else "known_disjoint" if legacy is False else "unknown"
        if status not in {"known_shared", "known_disjoint", "unknown"}:
            raise ValueError("Invalid source-overlap provenance")
        if status != "unknown" and legacy is not None and bool(legacy) != (status == "known_shared"):
            raise ValueError("Contradictory source-overlap provenance")
        source = sample["source_id"]
        if statuses.setdefault(source, status) != status:
            raise ValueError("One source has conflicting overlap records")
    observed = set(statuses.values())
    overlap = True if "known_shared" in observed else False if observed == {"known_disjoint"} else None
    return {"source_overlap_with_original_benchmark": overlap,
            "benchmark_overlap_by_source": statuses,
            "benchmark_overlap_source_counts": dict(Counter(statuses.values()))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--embeddings", nargs="+", default=["http://127.0.0.1:28780"])
    parser.add_argument("--expected-families", type=int, default=500)
    args = parser.parse_args()
    rows = read_jsonl(args.samples)
    for sample in rows:
        validate_sample(sample)
        require_repaired_sample(sample)
        if sample.get("restored_v4_qualification"):
            from pearl_training.restored_v4_qualification import require_qualified
            require_qualified(sample)
    check_split_isolation(rows)
    families = {s["base_question_id"] for s in rows}
    if len(families) != args.expected_families:
        raise ValueError(f"Expected {args.expected_families} families, found {len(families)}; do not silently change the target")
    output = Path(args.output)
    if (output / "manifest.json").exists():
        raise FileExistsError("Refusing to overwrite a frozen dataset snapshot")
    output.mkdir(parents=True, exist_ok=True)
    destination = output / "samples.jsonl"
    write_jsonl(destination, rows)
    groups = {s["source_group"]: s["split"] for s in rows}
    family_counts = Counter((s["split"], s["task"]) for s in {r["base_question_id"]: r for r in rows}.values())
    write_json(output / "dataset_manifest.json", {"code": code_revision(), "samples_sha256": sha256_file(destination),
        "source_samples_sha256": sha256_file(args.samples), "question_families": len(families), "variants": len(rows),
        "by_split_task": {s + ":" + t: n for (s, t), n in family_counts.items()}, "source_group_splits": groups,
        **source_overlap_metadata(rows), "qualification": PROTOCOL_VERSION})
    session = requests.Session()
    session.trust_env = False
    identities = []
    for endpoint in args.embeddings:
        response = session.get(endpoint + "/health", timeout=30)
        response.raise_for_status()
        identities.append(response.json())
    if any(identity != identities[0] for identity in identities):
        raise ValueError("Parallel document encoders have different checkpoint identities")
    identity = identities[0]
    clips = {c["id"]: c for s in rows if s["task"] == "past" for c in s["candidates"]}
    chunks = [[] for _ in args.embeddings]
    for cid, clip in clips.items():
        worker = int(hashlib.sha256(cid.encode()).hexdigest(), 16) % len(chunks)
        chunks[worker].append((cid, clip))

    def encode_chunk(worker):
        local = LocalEmbeddings(args.embeddings[worker])
        result = {}
        for i, (cid, clip) in enumerate(chunks[worker]):
            result[cid] = torch.from_numpy(local.document(clip)).float()
            if i % 50 == 0:
                print("DOCUMENTS", "worker", worker, i + 1, "/", len(chunks[worker]), flush=True)
        return result

    vectors = {}
    with ThreadPoolExecutor(max_workers=len(chunks)) as pool:
        for future in as_completed([pool.submit(encode_chunk, i) for i in range(len(chunks))]):
            vectors.update(future.result())
    if set(vectors) != set(clips):
        raise ValueError("Parallel document encoding lost clips")
    cache_path = output / "document_embeddings.pt"
    torch.save(vectors, cache_path)
    write_json(output / "manifest.json", {"identity": identity, "document_count": len(vectors),
        "embedding_workers": len(chunks), "worker_assignment": "sha256_clip_id_mod_workers",
        "samples_sha256": sha256_file(destination), "cache_sha256": sha256_file(cache_path),
        "source_video_sha256": {s["source_id"]: s["source_sha256"] for s in rows},
        "pool": "all completed historical clips before each query's current clip"})
    print("FROZEN", json.dumps({"families": len(families), "variants": len(rows), "documents": len(vectors)}), flush=True)


if __name__ == "__main__":
    main()
