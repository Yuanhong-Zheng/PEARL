"""Prepare and publish frozen V4 inputs while model requests keep running.

One head-owned producer publishes immutable manifests. Generation, temporal
checks and source rotation are delegated to the existing V4 implementation.
"""
from __future__ import annotations

from collections import deque
import threading
import time
from pathlib import Path

from . import legacy_movie_generation as factory
from .io import read_json, sha256_file, write_json


class InputFeeder:
    def __init__(self, directory, roots, initial_manifest, excluded, prior_ledgers,
                 target, seed_rows, *, chunk_windows=512, poll_seconds=15):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.roots = [Path(p) for p in roots]
        self.initial_manifest = Path(initial_manifest)
        self.excluded = list(excluded)
        self.prior_ledgers, self.target = prior_ledgers, target
        self.chunk_windows, self.poll_seconds = chunk_windows, poll_seconds
        self.claimed, self.groups, self.raw_splits = set(), {}, {}
        for sample in seed_rows:
            self.check_split(sample["source_group"], sample["source_sha256"], sample["split"])
        self.available = deque()
        self.lock = threading.RLock()
        self.stop = threading.Event()
        self.thread = None
        self.error = None
        self.pressure, self.low_water = 0, 512
        self.catalog_path = self.directory/"catalog.json"
        self.catalog = read_json(self.catalog_path) if self.catalog_path.exists() else {
            "protocol": "continuous-frozen-v4-inputs-v1", "manifests": []}
        for record in self.catalog["manifests"]:
            self.load_manifest(record, publish=False)

    def check_split(self, group, raw_sha, split):
        if (self.groups.setdefault(group, split) != split
                or self.raw_splits.setdefault(raw_sha, split) != split):
            raise ValueError("A source work or raw video crosses train/validation")

    def load_manifest(self, record, *, publish):
        if factory.bound(record["path"]) != record:
            raise ValueError("A published streaming input manifest changed")
        value = factory.checked_inputs(record["path"])
        for row in value["sources"]:
            self.check_split(row["source_group"], row["source_sha256"], row["split"])
        items = []
        for item in value["windows"]:
            wid = item["window_id"]
            if wid in self.claimed:
                raise ValueError("A window was admitted twice")
            window = read_json(item["path"])
            if factory._previously_attempted(window, self.excluded):
                raise ValueError("A newly admitted window overlaps previous inputs")
            self.claimed.add(wid)
            items.append(item)
        with self.lock:
            if publish:
                updated = {**self.catalog, "manifests": [*self.catalog["manifests"], record]}
                write_json(self.catalog_path, updated)
                self.catalog = updated
            self.available.extend(items)
        return len(items)

    def initialize(self):
        if not self.catalog["manifests"]:
            return self.load_manifest(factory.bound(self.initial_manifest), publish=True)
        return len(self.claimed)

    def take(self):
        with self.lock:
            result = list(self.available)
            self.available.clear()
        return result

    def set_pressure(self, windows, capacity):
        with self.lock:
            self.pressure, self.low_water = windows, max(512, capacity*2)

    def status(self):
        with self.lock:
            return {"admitted_windows": len(self.claimed), "prepared_manifests": len(self.catalog["manifests"]),
                    "prepared_not_taken": len(self.available), "error": self.error}

    def prepare_more(self):
        selected, identities, count = [], {}, 0
        # Preserve the original root/source ordering and interval deduplication.
        for root in self.roots:
            for path in sorted((root/"sources").glob("*/preparation_status.json")):
                status = read_json(path)
                if status.get("status") != "cpu_ready":
                    continue
                source_path = Path(status["source_manifest"])
                if sha256_file(source_path) != status["source_manifest_sha256"]:
                    raise ValueError("A published CPU source changed")
                source = read_json(source_path)
                identity = (source["sha256"], source["stream_start"], source["stream_end"])
                if identity in identities:
                    if identities[identity] != (source["source_group"], source["split"]):
                        raise ValueError("An identical source interval has conflicting splits")
                    continue
                identities[identity] = (source["source_group"], source["split"])
                paths = read_json(source_path.parent/"windows.json")
                remaining = []
                for window_path in paths:
                    window = read_json(window_path)
                    if (window["window_id"] not in self.claimed
                            and not factory._previously_attempted(window, self.excluded)):
                        remaining.append(window_path)
                if remaining:
                    self.check_split(source["source_group"], source["sha256"], source["split"])
                    selected.append((source_path, source, remaining[:self.chunk_windows-count]))
                    count += len(selected[-1][2])
                if count >= self.chunk_windows:
                    break
            if count >= self.chunk_windows:
                break
        if not count:
            return 0
        number = len(self.catalog["manifests"])
        # A crash cannot overwrite an earlier partial preparation directory.
        attempt = self.directory/"prepared"/f"chunk-{number:06d}-{time.time_ns()}"
        views = []
        for index, (source_path, source, windows) in enumerate(selected):
            view = attempt/"views"/f"source-{index:04d}"/"source.json"
            write_json(view, source)
            write_json(view.parent/"windows.json", windows)
            write_json(view.parent/"published-source.json", factory.bound(source_path))
            views.append(view)
        # No seed import is repeated: this manifest supplies work units only.
        # Global seed proofs, quotas and prior calls remain in the one coordinator.
        manifest = factory.prepare_inputs(views, attempt/"inputs", self.prior_ledgers,
                                          per_source=None, target=self.target)
        return self.load_manifest(factory.bound(manifest), publish=True)

    def start(self):
        def produce():
            try:
                while not self.stop.is_set():
                    with self.lock:
                        need = self.pressure+len(self.available) < self.low_water
                    if need and self.prepare_more():
                        continue
                    self.stop.wait(self.poll_seconds)
            except Exception as error:
                with self.lock:
                    self.error = {"type": type(error).__name__, "error": str(error)}
        self.thread = threading.Thread(target=produce, name="v4-input-prefetch", daemon=True)
        self.thread.start()

    def close(self):
        self.stop.set()
        if self.thread:
            self.thread.join()
