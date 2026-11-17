"""Shared-disk transport for the existing local teacher, with head-only ledgers.

Workers forward an unchanged OpenAI payload to their own loopback vLLM. Every
lost/uncertain call returns a transport error to the original bounded retry
logic; this layer never silently replays a model call.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import sys
import threading
import time
import urllib.error
import urllib.request
from urllib.parse import urlparse

FLEET_ROOT = Path(__file__).resolve().parents[3] / "shared-fleet"
sys.path.insert(0, str(FLEET_ROOT))
import engine as fleet


def payload_hash(payload):
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def local_endpoint(value):
    parsed = urlparse(value)
    if parsed.scheme != "http" or parsed.hostname not in {"localhost", "127.0.0.1", "::1"} or parsed.username or parsed.password:
        raise ValueError("Teacher workers only call loopback HTTP endpoints")
    return value.rstrip("/")


class Reply:
    def __init__(self, status, body):
        self.status_code, self.text = status, body

    def json(self):
        return json.loads(self.text)


class Router:
    def __init__(self, pool, *, lease=15, clock=time.monotonic):
        self.pool = Path(pool).resolve()
        self.pool.mkdir(parents=True, exist_ok=True)
        self.lease, self.clock = lease, clock
        self.lock, self.active, self.observations = threading.Lock(), {}, {}
        self.cached, self.alive, self.refreshed = [], [], float("-inf")
        config = fleet.read(self.pool/"config.json", {})
        self.fleet_state = Path(config.get("fleet_state", FLEET_ROOT/"state"))

    def backends(self):
        now = self.clock()
        if now-self.refreshed < 1:
            return self.cached
        ready, alive = [], []
        availability = fleet.read(self.fleet_state/"availability.json", {})
        for path in sorted((self.pool/"nodes").glob("*/health.json")):
            try:
                health = fleet.read(path)
                token = (health["instance"], health["sequence"])
                old, observed = self.observations.get(path.parent.name, (None, now))
                if token != old:
                    observed = now
                self.observations[path.parent.name] = (token, observed)
                if health["ready"] and now-observed < self.lease:
                    node = path.parent.name
                    alive.append(node)
                    control = fleet.read(self.pool/"nodes"/node/"control.json", {"enabled": True})
                    if control["enabled"] and availability.get(node, "online") == "online":
                        ready.append(node)
            except fleet.RECORD_ERRORS:
                continue
        self.cached, self.alive, self.refreshed = ready, alive, now
        return ready

    def choose(self):
        with self.lock:
            candidates = ["local", *self.backends()]
            node = min(candidates, key=lambda name: (self.active.get(name, 0)+1)/(2 if name == "local" else 1))
            self.active[node] = self.active.get(node, 0)+1
            return node

    def post(self, url, payload, timeout, session, ledger):
        import requests
        node, request = self.choose(), fleet.identifier()
        directory = self.pool/"calls"/request
        directory.mkdir(parents=True)
        digest = payload_hash(payload)
        trace = {"request": request, "node": node, "ledger": str(Path(ledger).resolve()),
                 "ledger_request_id": payload.get("seed", 42)-42, "payload_sha256": digest,
                 "submitted_at": time.time(), "status": "dispatched"}
        fleet.atomic(directory/"trace.json", trace)
        try:
            if node == "local":
                response = session.post(url, json=payload, timeout=timeout)
            else:
                fleet.atomic(directory/"request.json", {"id": request, "node": node,
                    "payload": payload, "payload_sha256": digest})
                fleet.atomic(self.pool/"queues"/node/(request+".json"), {"id": request})
                end = self.clock() + (timeout[-1] if isinstance(timeout, tuple) else timeout)
                while True:
                    result = fleet.read(directory/"response.json")
                    if result is not None:
                        if result.get("id") != request or result.get("node") != node or result.get("payload_sha256") != digest:
                            raise requests.ConnectionError("Shared teacher response owner/hash mismatch")
                        if result.get("transport_error"):
                            raise requests.ConnectionError(result["transport_error"])
                        response = Reply(result["http_status"], result["body"])
                        break
                    with self.lock:
                        self.backends()
                        online = node in self.alive
                    if not online or self.clock() >= end:
                        fleet.atomic(directory/"cancel.json", {"reason": "worker_unavailable_or_timeout"})
                        raise requests.ConnectionError("Shared teacher worker unavailable/timeout: "+node)
                    time.sleep(.2)
            trace.update(status="returned", http_status=response.status_code)
            return response
        except Exception as error:
            trace.update(status="transport_error", error_type=type(error).__name__)
            if node != "local":
                fleet.atomic(directory/"cancel.json", {"reason": "transport_error"})
            if isinstance(error, (OSError, ValueError, KeyError, TypeError)):
                raise requests.ConnectionError("Shared teacher transport: "+type(error).__name__) from error
            raise
        finally:
            trace["finished_at"] = time.time()
            fleet.atomic(directory/"trace.json", trace)
            # Images already exist in the frozen source inputs. Retain their
            # request hash and receipt, rather than accumulating TBs of repeated
            # base64 pixels after the worker has returned or been cancelled.
            if node != "local" and ((directory/"response.json").exists() or (directory/"cancel.json").exists()):
                fleet.atomic(directory/"request.json", {"id": request, "node": node,
                    "payload_sha256": digest, "payload_retired": True})
            with self.lock:
                self.active[node] -= 1


class RoutedSessions(threading.local):
    def __init__(self, router, ledger):
        import requests
        session = requests.Session()
        session.trust_env = False
        class Routed:
            def post(self, url, *, json, timeout):
                return router.post(url, json, timeout, session, ledger)
        self.session = Routed()


class Worker:
    def __init__(self, pool, node, endpoint, *, workers=128):
        self.pool, self.node = Path(pool).resolve(), fleet.safe_name(node)
        self.endpoint, self.instance = local_endpoint(endpoint), fleet.identifier()
        self.executor = ThreadPoolExecutor(max_workers=workers)
        self.limit, self.pending, self.sequence = workers, {}, 0
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        self.ready, self.last_health = False, float("-inf")

    def execute(self, request):
        directory = self.pool/"calls"/fleet.safe_name(request)
        try:
            guard = fleet.local_lock(self.pool, "teacher-call:"+request)
        except RuntimeError:
            return
        publish = False
        result = {"id": request, "node": self.node, "payload_sha256": None}
        try:
            if (directory/"response.json").exists() or (directory/"cancel.json").exists():
                return
            publish = True
            record = fleet.read(directory/"request.json")
            result["payload_sha256"] = record["payload_sha256"]
            if record["id"] != request or record["node"] != self.node or payload_hash(record["payload"]) != record["payload_sha256"]:
                raise ValueError("Invalid teacher request owner/hash")
            if (directory/"started.json").exists():
                raise RuntimeError("A previously dispatched model call was lost; replay requires a new budgeted attempt")
            fleet.atomic(directory/"started.json", {"node": self.node, "instance": self.instance, "at": time.time()})
            data = json.dumps(record["payload"]).encode()
            message = urllib.request.Request(self.endpoint+"/chat/completions", data=data,
                                             headers={"Content-Type": "application/json"})
            try:
                with self.opener.open(message, timeout=1800) as response:
                    result.update(http_status=response.status, body=response.read().decode())
            except urllib.error.HTTPError as error:
                result.update(http_status=error.code, body=error.read().decode(errors="replace"))
        except Exception as error:
            result["transport_error"] = type(error).__name__+": "+str(error)
        finally:
            if publish and not (directory/"cancel.json").exists():
                fleet.atomic(directory/"response.json", result)
            guard.close()

    def tick(self):
        now = time.monotonic()
        if now-self.last_health >= 5:
            try:
                with self.opener.open(self.endpoint+"/models", timeout=3) as response:
                    self.ready = any(row["id"] == "Qwen3.8-Flash-Next" for row in json.load(response)["data"])
            except (OSError, ValueError, KeyError, TypeError):
                self.ready = False
            self.last_health = now
        self.sequence += 1
        fleet.atomic(self.pool/"nodes"/self.node/"health.json", {"instance": self.instance,
            "sequence": self.sequence, "node": self.node, "hostname": socket.gethostname(),
            "ready": self.ready, "active": len(self.pending), "at": time.time()})
        for request, future in list(self.pending.items()):
            if future.done():
                future.result()
                self.pending.pop(request)
                (self.pool/"queues"/self.node/(request+".json")).unlink(missing_ok=True)
        if self.ready:
            for path in sorted((self.pool/"queues"/self.node).glob("*.json")):
                if len(self.pending) >= self.limit:
                    break
                request = fleet.read(path)["id"]
                if request not in self.pending:
                    self.pending[request] = self.executor.submit(self.execute, request)


def run_worker(pool, node, endpoint, workers=128):
    node = fleet.safe_name(node)
    guard = fleet.local_lock(pool, "teacher-worker:"+node)
    worker, stopping = Worker(pool, node, endpoint, workers=workers), False
    def stop(signum, frame):
        nonlocal stopping
        stopping = True
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        while not stopping:
            try:
                worker.tick()
            except fleet.RECORD_ERRORS as error:
                fleet.local_fault(pool, "teacher-worker:"+node, error)
            time.sleep(1)
    finally:
        fleet.atomic(Path(pool)/"nodes"/node/"health.json", {"instance": worker.instance,
            "sequence": worker.sequence+1, "node": node, "ready": False, "at": time.time()})
        worker.executor.shutdown(wait=True)
        guard.close()
