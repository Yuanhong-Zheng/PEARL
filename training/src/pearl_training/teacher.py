"""Approved teacher calls with a persistent request cap and no credential logs."""

from __future__ import annotations

import base64
import hashlib
import json
import math
import re
import sqlite3
import time
from pathlib import Path

import requests

from .io import read_json


class TeacherRequestError(RuntimeError):
    def __init__(self, message, **details):
        super().__init__(message)
        self.details = details


class TeacherBudgetExhausted(RuntimeError):
    pass


class TeacherRetryExhausted(RuntimeError):
    def __init__(self, stage, attempts, last_failure):
        self.stage = stage
        self.attempts = attempts
        self.last_failure = last_failure
        super().__init__(f"{stage} failed after {attempts} attempts; last failure: {last_failure.get('error')}")


def safe_error_text(value, secret, limit=800):
    if value is None:
        return None
    value = str(value).replace(secret, "[REDACTED]") if secret else str(value)
    value = re.sub(r"(?i)Bearer\s+[^\s,'\"}]+", "Bearer [REDACTED]", value)
    value = re.sub(r"(?i)(api[_-]?key|token|secret)=([^&\s]+)", r"\1=[REDACTED]", value)
    return value[:limit]


def retry_after_seconds(value):
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        return None
    return seconds if math.isfinite(seconds) and seconds >= 0 else None


def credential_account(readme, alias):
    content = Path(readme).read_text()
    match = re.search(r"<!-- BEGIN_API_CONFIG -->\s*```json\s*(.*?)\s*```", content, re.S)
    if match is None:
        raise ValueError("Structured credential configuration was not found")
    config = json.loads(match.group(1))
    matches = [item for item in config["accounts"] if item["alias"] == alias]
    if len(matches) != 1:
        raise ValueError("Account alias is not unique")
    return matches[0]


def image_part(path):
    encoded = base64.b64encode(Path(path).read_bytes()).decode("ascii")
    return {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + encoded}}


def json_response(content):
    if not isinstance(content, str):
        raise ValueError("Teacher response is not text")
    content = content.strip()
    if content.startswith("```"):
        match = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", content, re.S)
        if match is None:
            raise ValueError("Malformed fenced JSON response")
        content = match.group(1)
    return json.loads(content)


class Teacher:
    def __init__(self, credential_readme, approval, ledger, proxy=None):
        self.approval = read_json(approval)
        self.account = credential_account(credential_readme, self.approval["account_alias"])
        if self.account["protocol"] != "openai":
            raise ValueError("This client requires the approved OpenAI-compatible endpoint")
        if self.approval["model"] != "gemini-3.1-pro-preview":
            raise ValueError("Obtain explicit approval before using another model")
        self.model = self.approval["model"]
        self.max_retries = int(self.approval.get("request_retries", 1))
        if self.max_retries < 0:
            raise ValueError("Retry count must be nonnegative")
        self.url = self.account["base"].rstrip("/") + self.account["generate_path"]
        self.ledger = Path(ledger)
        self.ledger.parent.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.headers["Authorization"] = "Bearer " + self.account["key"]
        if proxy:
            self.session.proxies.update({"http": proxy, "https": proxy})
            self.session.trust_env = False
        with sqlite3.connect(self.ledger) as database:
            database.execute("CREATE TABLE IF NOT EXISTS calls (id INTEGER PRIMARY KEY, model TEXT, task TEXT, prompt_sha256 TEXT, status TEXT, elapsed REAL, usage TEXT)")

    def reserve(self, task, content):
        if task not in self.approval["allowed_tasks"]:
            raise ValueError("Task is not approved")
        digest = hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()
        with sqlite3.connect(self.ledger, timeout=30) as database:
            database.execute("BEGIN IMMEDIATE")
            count = database.execute("SELECT COUNT(*) FROM calls").fetchone()[0]
            if count >= self.approval["max_vision_requests"]:
                raise TeacherBudgetExhausted("Approved teacher request budget has been exhausted")
            cursor = database.execute("INSERT INTO calls(model, task, prompt_sha256, status) VALUES (?, ?, ?, 'reserved')", (self.model, task, digest))
            return cursor.lastrowid

    def chat_json(self, task, content):
        max_output = self.approval.get("max_output_tokens_by_task", {}).get(task, 6000)
        if type(max_output) is not int or max_output < 1:
            raise ValueError("Approved output token limit must be a positive integer")
        request_id = self.reserve(task, content)
        started = time.monotonic()
        status, usage = "failed", {}
        response = None
        finish_reason = None
        provider_id = None
        body, choice = {}, {}
        try:
            response = self.session.post(self.url, json={
                "model": self.model,
                "messages": [{"role": "user", "content": content}],
                "temperature": 0,
                "max_tokens": max_output,
            }, timeout=(30, 240))
            provider_id = safe_error_text(response.headers.get("http_x_reqid") or response.headers.get("x-request-id"), self.account["key"], 160)
            if response.status_code != 200:
                status = f"http_{response.status_code}"
                try:
                    error = response.json().get("error", {})
                except (ValueError, AttributeError):
                    error = {}
                error = error if isinstance(error, dict) else {"message": str(error)}
                message = safe_error_text(error.get("message"), self.account["key"])
                raise TeacherRequestError(f"Approved teacher returned HTTP {response.status_code}: {message or 'no error message returned'}",
                    request_id=request_id, category="http", http_status=response.status_code,
                    provider_request_id=provider_id, provider_error_message=message,
                    provider_error_type=safe_error_text(error.get("type"), self.account["key"]),
                    provider_error_code=safe_error_text(error.get("code"), self.account["key"]),
                    retry_after_seconds=retry_after_seconds(response.headers.get("Retry-After")),
                    elapsed_seconds=time.monotonic() - started)
            try:
                body = response.json()
            except ValueError:
                raise ValueError("HTTP 200 response body is not valid JSON") from None
            usage = body.get("usage") or {}
            if not isinstance(usage, dict):
                usage = {}
            choice = body["choices"][0]
            provider_id = provider_id or safe_error_text(body.get("id"), self.account["key"], 160)
            finish_reason = safe_error_text(choice.get("finish_reason"), self.account["key"], 80)
            parsed = json_response(choice["message"]["content"])
            if not isinstance(parsed, dict):
                raise ValueError("Teacher JSON root is not an object")
            status = "ok"
            return parsed, {"request_id": request_id, "model": self.model, "usage": usage,
                            "requested_max_output_tokens": max_output,
                            "response_model": safe_error_text(body.get("model"), self.account["key"], 160),
                            "provider_request_id": provider_id,
                            "elapsed_seconds": time.monotonic() - started}
        except requests.RequestException as error:
            detail = safe_error_text(error, self.account["key"])
            raise TeacherRequestError(f"Teacher transport failure: {type(error).__name__}: {detail}",
                request_id=request_id, category="transport", transport_class=type(error).__name__,
                transport_detail=detail, provider_request_id=provider_id,
                elapsed_seconds=time.monotonic() - started) from None
        except (ValueError, KeyError, IndexError, TypeError, AttributeError) as error:
            status = "invalid_response"
            message = safe_error_text(error, self.account["key"])
            reply = choice.get("message", {}) if isinstance(choice, dict) else {}
            reply = reply if isinstance(reply, dict) else {}
            content = reply.get("content")
            raise TeacherRequestError(f"Teacher response format failure: {type(error).__name__}: {message}; finish_reason={finish_reason}",
                request_id=request_id, category="response_format", format_error_type=type(error).__name__,
                format_error=message, provider_request_id=provider_id, finish_reason=finish_reason,
                elapsed_seconds=time.monotonic() - started,
                response_content_characters=len(content) if isinstance(content, str) else None,
                response_content_preview=safe_error_text(content, self.account["key"], 1600),
                response_refusal=safe_error_text(reply.get("refusal"), self.account["key"], 400),
                response_model=safe_error_text(body.get("model"), self.account["key"], 160) if isinstance(body, dict) else None) from None
        finally:
            with sqlite3.connect(self.ledger, timeout=30) as database:
                database.execute("UPDATE calls SET status=?, elapsed=?, usage=? WHERE id=?", (status, time.monotonic() - started, json.dumps(usage), request_id))

    def usage(self):
        with sqlite3.connect(self.ledger) as database:
            rows = database.execute("SELECT id, model, task, status, elapsed, usage FROM calls ORDER BY id").fetchall()
        return [{"request_id": i, "model": m, "task": t, "status": s, "seconds": e, "usage": json.loads(u or "{}")} for i, m, t, s, e, u in rows]
