"""Versioned resource adapter; V4 labels, requests and budget stay unchanged."""
from __future__ import annotations

import copy
import time
from pathlib import Path

from .fleet_transport import Router, fleet
from .io import read_json, sha256_file


PER_NODE = 256
MAX_WORKERS = 4096  # Lazy thread pool, supporting up to sixteen ready teachers.


def checked_cluster_approval(path, original_check, *, target=None):
    value = read_json(path)
    policy = value.get("fleet_resources")
    if policy is None:
        return original_check(path, target=target)
    if set(policy) != {"node_approval", "per_node_requests", "worker_pool_limit", "dynamic_ready_nodes"}:
        raise ValueError("Unknown cluster resource policy")
    if (policy["per_node_requests"] != PER_NODE or policy["worker_pool_limit"] != MAX_WORKERS
            or policy["dynamic_ready_nodes"] is not True
            or value.get("max_concurrent_requests") != MAX_WORKERS):
        raise ValueError("Preserve the approved per-node256 dynamic resource policy")
    binding = policy["node_approval"]
    node_path = Path(binding["path"])
    if not node_path.is_absolute():
        node_path = Path(__file__).resolve().parents[2]/node_path
    if sha256_file(node_path) != binding["sha256"]:
        raise ValueError("The original per-node approval changed")
    node = original_check(node_path, target=target)
    comparable = copy.deepcopy(value)
    comparable.pop("fleet_resources")
    comparable["max_concurrent_requests"] = node["max_concurrent_requests"]
    if node["max_concurrent_requests"] != PER_NODE or comparable != node:
        raise ValueError("Cluster resources must not change model, quality, retry or budget settings")
    return value


class ElasticRouter(Router):
    """Backpressure per teacher, including the original local H200 endpoint."""

    def capacity(self):
        with self.lock:
            return min(MAX_WORKERS, PER_NODE*(1+len(self.backends())))

    def choose(self):
        while True:
            with self.lock:
                nodes = ["local", *self.backends()]
                free = [node for node in nodes if self.active.get(node, 0) < PER_NODE]
                if free:
                    node = min(free, key=lambda name:
                        (self.active.get(name, 0)+1)/(2 if name == "local" else 1))
                    self.active[node] = self.active.get(node, 0)+1
                    return node
            time.sleep(.05)


def dispatch_with_capacity(router, original, kind, split, remaining, pending):
    return len(pending) < router.capacity() and original(kind, split, remaining, pending)


def boundary_ready(state, expected_run):
    """Only a naturally closed batch is a valid coordinator replacement point."""
    return (state.get("active_run") is None
            and state.get("previous_run") == str(Path(expected_run).resolve())
            and state.get("phase") in {"batch_closed", "collecting_cpu_sources", "complete"})
