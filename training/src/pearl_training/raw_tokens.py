"""Durable records from actual inference tensors and engine token outputs."""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import socket
import threading
import time
import uuid


PROTOCOL = "pearl-actual-inference-tokens-v1"
_WRITERS = {}


class TokenCaptureError(RuntimeError):
    pass


def capture_mode():
    mode = os.environ.get("PEARL_RAW_TOKEN_MODE", "record")
    if mode not in {"record", "disabled_by_user"}:
        raise TokenCaptureError("Unknown PEARL_RAW_TOKEN_MODE")
    return mode


def capture_enabled():
    return capture_mode() == "record"


def disabled_capture_record():
    return {"protocol": PROTOCOL, "mode": "disabled_by_user", "enabled": False,
            "complete": False, "reason": "user_requested_no_full_benchmark_token_recording"}


def capture_report(directory):
    if not capture_enabled():
        return disabled_capture_record()
    return summarize_directory(directory)


def stamp():
    return {"wall_time_ns": time.time_ns(), "monotonic_ns": time.monotonic_ns()}


def plain(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if hasattr(value, "detach"):
        return value.detach().cpu().tolist()
    return str(value)


def modality_status():
    return {"audio_feature_positions": [], "audio_status": "not_applicable_no_audio_input",
            "executed_control_tokens": [], "control_status": "not_applicable_no_interactive_control_policy",
            "tts_speech_codes": [], "returned_audio_chunks": [],
            "tts_status": "not_applicable_no_tts_backend"}


class TokenWriter:
    def __init__(self, directory, kind):
        self.directory = Path(directory).resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.kind = kind
        self.path = self.directory / f"{kind}-{socket.gethostname()}-{os.getpid()}-{uuid.uuid4().hex[:8]}.jsonl"
        self.fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_APPEND, 0o600)
        self.lock = threading.RLock()
        self.broken = False
        self.write({"event": "capture_ready", "protocol": PROTOCOL, "kind": kind,
                    "host": socket.gethostname(), "pid": os.getpid(), **stamp()})

    def write(self, event):
        if self.broken:
            raise TokenCaptureError(f"Raw-token writer is unavailable: {self.path}")
        event = plain(event)
        encoded = (json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n").encode()
        try:
            with self.lock:
                view = memoryview(encoded)
                while view:
                    written = os.write(self.fd, view)
                    if written <= 0:
                        raise OSError("Raw-token write made no progress")
                    view = view[written:]
                os.fsync(self.fd)
        except BaseException as error:
            self.broken = True
            fallback = Path("/tmp/pearl-raw-token-write-failures")
            fallback.mkdir(mode=0o700, parents=True, exist_ok=True)
            recovery = {"complete": False, "target": str(self.path), "error": repr(error),
                        "unpublished_event": event, **stamp()}
            (fallback / f"{os.getpid()}-{time.time_ns()}.json").write_text(json.dumps(recovery))
            raise TokenCaptureError(f"Raw-token write failed; local recovery retained in {fallback}") from error

    def begin(self, metadata, *, new_inference=True):
        return TokenCall(self, metadata, new_inference=new_inference)


def writer(kind):
    if not capture_enabled():
        raise TokenCaptureError("Raw-token recording was explicitly disabled for this benchmark")
    root = os.environ.get("PEARL_RAW_TOKEN_DIR")
    if not root:
        raise TokenCaptureError("Set PEARL_RAW_TOKEN_DIR before any model inference or validation")
    key = (str(Path(root).resolve()), kind, os.getpid())
    if key not in _WRITERS:
        _WRITERS[key] = TokenWriter(root, kind)
    return _WRITERS[key]


class TokenCall:
    def __init__(self, output, metadata, *, new_inference):
        self.writer = output
        self.record_id = uuid.uuid4().hex
        self.has_input = False
        self.units = 0
        self.new_inference = new_inference
        self.closed = False
        self.event("begin", metadata=metadata, new_model_inference=new_inference,
                   applicability=modality_status())

    def event(self, event, **values):
        self.writer.write({"protocol": PROTOCOL, "record_id": self.record_id,
                           "event": event, **stamp(), **values})

    def inputs(self, *, prefix, token_ids, **metadata):
        if token_ids is None:
            raise TokenCaptureError("Backend did not expose actual model input token IDs")
        self.event("actual_input", initial_prefix=prefix, actual_input_token_ids=token_ids, **metadata)
        self.has_input = True

    def unit(self, token_ids, *, unit_type, **metadata):
        self.event("unit", unit_index=self.units, unit_type=unit_type,
                   raw_sampled_token_ids=token_ids, **metadata)
        self.units += 1

    def finish(self, **values):
        if self.new_inference and not self.has_input:
            raise TokenCaptureError("Refusing to mark an inference complete without actual inputs")
        self.event("complete", complete=True, unit_count=self.units, **values)
        self.closed = True
        return {"protocol": PROTOCOL, "record_id": self.record_id,
                "path": str(self.writer.path), "complete": True}

    def fail(self, error):
        if not self.closed and not self.writer.broken:
            self.event("failed", complete=False, error_type=type(error).__name__, error=str(error))
        self.closed = True


@contextmanager
def recorded(kind, metadata, *, new_inference=True):
    call = writer(kind).begin(metadata, new_inference=new_inference)
    try:
        yield call
    except BaseException as error:
        call.fail(error)
        raise
    else:
        if not call.closed:
            error = TokenCaptureError("Inference record was not explicitly completed")
            call.fail(error)
            raise error


def tensor_inputs(call, inputs, processor, *, messages=None, prefix=None):
    ids = inputs.get("input_ids")
    if ids is None:
        raise TokenCaptureError("Model inputs do not contain input_ids")
    values = plain(ids)
    if prefix is None:
        prefix = {"messages": messages,
                  "decoded_actual_input_ids": processor.tokenizer.batch_decode(values, skip_special_tokens=False)}
    structural = {key: plain(inputs[key]) for key in (
        "attention_mask", "position_ids", "image_grid_thw", "video_grid_thw", "second_per_grid_ts") if key in inputs}
    features = {key: {"shape": list(value.shape), "dtype": str(value.dtype)}
                for key, value in inputs.items() if key.startswith("pixel_values")}
    call.inputs(prefix=prefix, token_ids=values, processor_tensors=structural,
                multimodal_feature_tensors=features, **modality_status())


def summarize_directory(directory):
    """Verify event completeness without inventing records for cached inputs."""
    root = Path(directory)
    states = {}
    files = []
    for path in sorted(root.rglob("*.jsonl")):
        count = 0
        with path.open() as stream:
            for line in stream:
                row = json.loads(line)
                rid = row.get("record_id")
                if not rid:
                    continue
                key = (str(path), rid)
                item = states.setdefault(key, {"begin": False, "input": False, "complete": False,
                                               "new_inference": True, "failed": False})
                event = row["event"]
                if event == "begin":
                    item.update(begin=True, new_inference=row["new_model_inference"])
                elif event == "actual_input":
                    item["input"] = True
                elif event == "complete":
                    item["complete"] = row.get("complete") is True
                elif event == "failed":
                    item["failed"] = True
                count += 1
        files.append({"path": str(path), "events": count, "bytes": path.stat().st_size})
    incomplete = [key for key, item in states.items() if not item["begin"] or not item["complete"]
                  or item["failed"] or (item["new_inference"] and not item["input"])]
    failures = []
    for path in Path("/tmp/pearl-raw-token-write-failures").glob("*.json"):
        failure = json.loads(path.read_text())
        if Path(failure["target"]).is_relative_to(root.resolve()):
            failures.append({"recovery_path": str(path), "target": failure["target"], "error": failure["error"]})
    return {"protocol": PROTOCOL, "complete": bool(states) and not incomplete and not failures,
            "records": len(states), "new_model_inferences": sum(x["new_inference"] for x in states.values()),
            "cached_input_events": sum(not x["new_inference"] for x in states.values()),
            "incomplete": [list(x) for x in incomplete], "write_failures": failures, "files": files}
