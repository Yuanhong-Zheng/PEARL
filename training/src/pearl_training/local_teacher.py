"""Loopback-only teacher client; never reads paid API credentials or ledgers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sqlite3
import threading
import time
from urllib.parse import urlparse

import requests

from .io import read_json
from .teacher import Teacher, TeacherRequestError, json_response


class LocalTeacher(Teacher):
    """Reuse request caps and stage retry bookkeeping with a separate local ledger."""

    def __init__(self, approval, ledger):
        self.approval = read_json(approval)
        if self.approval.get("provider") != "local_vllm":
            raise ValueError("Local teacher requires its own local_vllm configuration")
        base = self.approval["base_url"].rstrip("/")
        parsed = urlparse(base)
        if (parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}
                or parsed.username or parsed.password or parsed.query or parsed.fragment
                or parsed.path != "/v1"):
            raise ValueError("The local teacher endpoint must be a loopback HTTP /v1 URL")
        self.model = self.approval["model"]
        if self.model != "Qwen3.8-Flash-Next":
            raise ValueError("This pilot was authorized for Qwen3.8-Flash-Next only")
        self.url = base + "/chat/completions"
        self.max_retries = int(self.approval.get("request_retries", 5))
        if not 0 <= self.max_retries <= 5:
            raise ValueError("At most five retries are allowed")
        self.ledger = Path(ledger)
        self.ledger.parent.mkdir(parents=True, exist_ok=True)
        if self.ledger.resolve() == (Path(__file__).resolve().parents[2] / "data/batch500/generation/calls.sqlite3"):
            raise ValueError("Local calls must not use the existing Gemini ledger")
        self._thread = threading.local()
        with sqlite3.connect(self.ledger, timeout=30) as database:
            database.execute("CREATE TABLE IF NOT EXISTS calls (id INTEGER PRIMARY KEY, model TEXT, task TEXT, prompt_sha256 TEXT, status TEXT, elapsed REAL, usage TEXT)")

    def chat_json(self, task, content):
        max_output = self.approval.get("max_output_tokens_by_task", {}).get(task, 6000)
        if type(max_output) is not int or max_output < 1:
            raise ValueError("Output token limit must be a positive integer")
        effort = self.approval.get("reasoning_effort_by_task", {}).get(
            task, self.approval.get("reasoning_effort"))
        if effort is not None and effort not in {"xhigh", "medium", "low"}:
            raise ValueError("Unsupported local reasoning effort")
        request_id = self.reserve(task, content)
        if not hasattr(self._thread, "session"):
            self._thread.session = requests.Session()
            self._thread.session.trust_env = False
        payload = {
            "model": self.model, "messages": [{"role": "user", "content": content}],
            # Keep every attempt reproducible without making all retries return
            # the same malformed sample under local deterministic decoding.
            "temperature": self.approval.get("temperature", 1.0), "seed": 42 + request_id,
            "max_tokens": max_output,
            "chat_template_kwargs": {"enable_thinking": self.approval.get("enable_thinking", True)},
        }
        if effort:
            payload["reasoning_effort"] = effort
        started = time.monotonic()
        status, usage = "failed", {}
        finish_reason = None
        try:
            response = self._thread.session.post(self.url, json=payload, timeout=(10, 1800))
            if response.status_code != 200:
                status = "http_" + str(response.status_code)
                raise TeacherRequestError("Local vLLM returned HTTP " + str(response.status_code) + ": " + response.text[:1000],
                                          request_id=request_id, category="http", http_status=response.status_code)
            body = response.json()
            usage = body.get("usage") or {}
            choice = body["choices"][0]
            finish_reason = choice.get("finish_reason")
            if body.get("model") != self.model:
                raise ValueError("Response model does not match the authorized local model")
            result = json_response(choice["message"]["content"])
            if not isinstance(result, dict):
                raise ValueError("Teacher JSON root is not an object")
            status = "ok"
            return result, {
                "request_id": request_id, "model": self.model, "provider": "local_vllm",
                "usage": usage, "elapsed_seconds": time.monotonic() - started,
                "response_model": body["model"], "requested_max_output_tokens": max_output,
                "finish_reason": finish_reason,
                "generation_settings": {k: v for k, v in payload.items() if k not in {"model", "messages"}},
                "request_payload_sha256": hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest(),
            }
        except requests.RequestException as error:
            raise TeacherRequestError(f"Local vLLM transport failure: {type(error).__name__}: {error}",
                                      request_id=request_id, category="transport") from None
        except (ValueError, KeyError, IndexError, TypeError) as error:
            status = "invalid_response"
            raise TeacherRequestError(f"Local vLLM response failure: {type(error).__name__}: {error}; finish_reason={finish_reason}",
                                      request_id=request_id, category="response_format", finish_reason=finish_reason) from None
        finally:
            with sqlite3.connect(self.ledger, timeout=30) as database:
                database.execute("UPDATE calls SET status=?,elapsed=?,usage=? WHERE id=?",
                                 (status, time.monotonic() - started, json.dumps(usage), request_id))
