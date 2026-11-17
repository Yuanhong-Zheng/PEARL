#!/usr/bin/env python3
"""Plan or explicitly build fixed release text-only retrieval queries."""
import argparse
from datetime import datetime, timezone
import ipaddress
import json
from pathlib import Path
import time
from urllib.parse import urlparse

from pearl_training.io import write_json
from pearl_training.released_text_query import digest
from pearl_training.text_query_cache import build_text_query_cache, cache_plan


def loopback_executor(base_url, identity, timeout):
    parsed = urlparse(base_url)
    if parsed.scheme not in {"http", "https"} or parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("Use an explicit loopback OpenAI-compatible base-helper URL")
    host = parsed.hostname
    try:
        loopback = host == "localhost" or ipaddress.ip_address(host).is_loopback
    except ValueError:
        loopback = False
    if not loopback:
        raise ValueError("Query-cache generation permits only a local base helper, never an external paid API")
    import requests
    session = requests.Session()
    session.trust_env = False
    url = base_url.rstrip("/") + "/chat/completions"

    def execute(payload):
        if payload.get("model") != identity["model_path"]:
            raise ValueError("Helper request model differs from the frozen base checkpoint")
        began = time.monotonic()
        response = session.post(url, json=payload, headers={"Authorization": "Bearer EMPTY"}, timeout=timeout,
                                allow_redirects=False)
        if 300 <= response.status_code < 400:
            raise ValueError("The local helper must not redirect query-cache requests")
        response.raise_for_status()
        body = response.json()
        request_id = body.get("id") or response.headers.get("x-request-id")
        if not request_id or body.get("model") != payload["model"]:
            raise ValueError("Base helper response lacks its actual request ID or reports a different model")
        return body, {"request_id": request_id, "http_status": response.status_code,
            "model": body["model"], "usage": body.get("usage"), "response_sha256": digest(body),
            "base_helper_sha256": digest(identity), "elapsed_seconds": time.monotonic()-began,
            "endpoint": url, "received_utc": datetime.now(timezone.utc).isoformat()}

    return execute


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", required=True, type=Path)
    parser.add_argument("--helper-model", required=True, type=Path, help="Local base reader checkpoint, without any LoRA adapter")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--release-root", type=Path)
    parser.add_argument("--execute", action="store_true", help="Explicitly call the local base helper; default only writes a CPU plan")
    parser.add_argument("--base-url", help="Loopback OpenAI-compatible /v1 URL; required only with --execute")
    parser.add_argument("--timeout", type=float, default=3600)
    parser.add_argument("--work-dir", type=Path, help="Persistent per-operation receipts, used to resume without repeating successes")
    args = parser.parse_args(argv)
    identity, samples = cache_plan(args.samples, args.helper_model, release_root=args.release_root)
    if not args.execute:
        plan_path = args.output.with_name(args.output.name + ".plan.json")
        if plan_path.exists():
            raise FileExistsError("Preserve an existing CPU cache plan")
        write_json(plan_path, {"status": "plan_only", **identity, "queries": len(samples), "model_calls": 0})
        print(json.dumps({"status": "plan_only", "queries": len(samples), "plan": str(plan_path), "model_calls": 0}))
        return 0
    if not args.base_url or args.timeout <= 0:
        parser.error("--execute requires an explicit --base-url and positive --timeout")
    if args.output.exists():
        raise FileExistsError("Preserve the completed cache; choose another output")
    result = build_text_query_cache(args.samples, args.output, args.helper_model,
        loopback_executor(args.base_url, identity["base_helper"], args.timeout), release_root=args.release_root,
        execution_kind="loopback_base_helper", work_dir=args.work_dir or args.output.with_name(args.output.name + ".work"))
    print(json.dumps({"status": "complete", "queries": result["manifest"]["queries"], "cache": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
