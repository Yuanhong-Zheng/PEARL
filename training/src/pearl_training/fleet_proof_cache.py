"""Reuse identical, fully checked V4 proofs without changing their predicates.

No model calls or labels are produced here. Content and media fingerprints must
match before a previous verification can be reused. Modified inputs go through
the original validator again. Every process keeps a bounded image-byte cache.
"""
from __future__ import annotations

from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor
import argparse
import copy
import hashlib
import json
import multiprocessing
from pathlib import Path
import threading
import time

from . import restored_v4_qualification as qualification
from .teacher import image_part

ORIGINAL_REQUIRE = qualification.require_qualified


def file_fingerprint(path):
    # stat follows symlinks; resolving every ancestor would add many Ceph RPCs.
    path = Path(path).absolute()
    stat = path.stat()
    return (str(path), stat.st_dev, stat.st_ino, stat.st_size,
            stat.st_mtime_ns, stat.st_ctime_ns)


def sample_fingerprint(sample):
    paths = set(sample["frame_timestamps"])
    return tuple(file_fingerprint(path) for path in sorted(paths))


def sample_digest(sample):
    return hashlib.sha256(json.dumps(sample, sort_keys=True).encode()).hexdigest()


class ImageBytesCache:
    """Return exactly teacher.image_part bytes, including fresh mutable wrappers."""

    def __init__(self, max_bytes=512*1024*1024, encoder=image_part):
        self.max_bytes, self.encoder = max_bytes, encoder
        self.entries, self.bytes = OrderedDict(), 0
        self.lock = threading.RLock()

    def __call__(self, path):
        key = file_fingerprint(path)
        with self.lock:
            row = self.entries.pop(key, None)
            if row is not None:
                self.entries[key] = row
                return copy.deepcopy(row)
        value = self.encoder(path)
        size = len(value["image_url"]["url"])
        # Concurrent edits never become a cached representation of newer media.
        if file_fingerprint(path) == key and size <= self.max_bytes:
            with self.lock:
                previous = self.entries.pop(key, None)
                if previous is not None:
                    self.bytes -= len(previous["image_url"]["url"])
                self.entries[key] = copy.deepcopy(value)
                self.bytes += size
                while self.bytes > self.max_bytes:
                    _, old = self.entries.popitem(last=False)
                    self.bytes -= len(old["image_url"]["url"])
        return value


class QualifiedProofCache:
    """A cache hit requires the same complete sample and unchanged media files."""

    def __init__(self, max_samples=20000, original=ORIGINAL_REQUIRE, image_bytes=512*1024*1024):
        self.original, self.max_samples = original, max_samples
        self.encoder = ImageBytesCache(image_bytes)
        self.entries = OrderedDict()
        self.lock = threading.RLock()
        self.hits = self.misses = 0

    def verify(self, sample, *, image_encoder=None, check_files=True):
        # Nonstandard callers retain their original semantics; no cache hit.
        if image_encoder is not None or not check_files:
            return self.original(sample, image_encoder=image_encoder, check_files=check_files)
        try:
            key, media = sample_digest(sample), sample_fingerprint(sample)
        except (KeyError, TypeError, ValueError, OSError):
            return self.original(sample, check_files=check_files)
        with self.lock:
            previous = self.entries.pop(key, None)
            if previous == media:
                self.entries[key] = media
                self.hits += 1
                return
            self.misses += 1
        result = self.original(sample, image_encoder=self.encoder, check_files=True)
        if sample_fingerprint(sample) == media:
            self.prime(key, media)
        return result

    def prime(self, digest, media):
        with self.lock:
            self.entries.pop(digest, None)
            self.entries[digest] = media
            while len(self.entries) > self.max_samples:
                self.entries.popitem(last=False)


class ClosedLedgerCache:
    """Original SQL snapshot values, reused only while DB/WAL/journal are unchanged."""

    def __init__(self, original):
        self.original, self.entries = original, {}
        self.lock = threading.RLock()

    @staticmethod
    def fingerprint(path):
        path = Path(path).resolve(strict=True)
        return tuple(file_fingerprint(p) if p.exists() else None
                     for p in [path, Path(str(path)+"-wal"), Path(str(path)+"-journal")])

    def __call__(self, path):
        key, before = str(Path(path).resolve(strict=True)), self.fingerprint(path)
        with self.lock:
            previous = self.entries.get(key)
            if previous is not None and previous[0] == before:
                return copy.deepcopy(previous[1])
        value = self.original(path)
        if self.fingerprint(path) == before:
            with self.lock:
                self.entries[key] = (before, copy.deepcopy(value))
        return value


def _validate_group(rows):
    cache = QualifiedProofCache()
    result = []
    for sample in rows:
        cache.verify(sample)
        key, media = sample_digest(sample), sample_fingerprint(sample)
        # Publish only entries produced by the actual original verifier.
        if cache.entries.get(key) != media:
            cache.verify(sample)
        result.append((key, media))
    return result


def validate_seed_rows(rows, cache, workers=8):
    """CPU-only original checks, grouped by source for media locality."""
    groups = OrderedDict()
    for sample in rows:
        groups.setdefault(sample["source_sha256"], []).append(sample)
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
        for checked in pool.map(_validate_group, groups.values()):
            for key, media in checked:
                cache.prime(key, media)
    # Re-check current fingerprints in the parent before any cached reuse.
    for sample in rows:
        cache.verify(sample)


def main():
    from .io import read_jsonl, sha256_file, write_json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if not 1 <= args.workers <= 16:
        parser.error("Use one to sixteen CPU verification workers")
    started = time.time()
    before = file_fingerprint(args.samples)
    digest = sha256_file(args.samples)
    rows = read_jsonl(args.samples)
    cache = QualifiedProofCache()
    validate_seed_rows(rows, cache, args.workers)
    if file_fingerprint(args.samples) != before or sha256_file(args.samples) != digest:
        raise ValueError("The closed seed export changed during validation")
    receipt = {"status": "original_v4_proofs_verified", "samples": str(args.samples.resolve()),
        "samples_sha256": digest, "count": len(rows), "model_calls": 0,
        "original_verifier_sha256": qualification._implementations(),
        "cache_code_sha256": sha256_file(Path(__file__)),
        "entries": list(cache.entries.items()), "workers": args.workers,
        "elapsed_seconds": time.time()-started, "finished_at": time.time()}
    write_json(args.output, receipt)
    print(json.dumps({key: receipt[key] for key in
        ["status", "count", "workers", "model_calls", "elapsed_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
