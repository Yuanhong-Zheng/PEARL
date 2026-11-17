"""Immutable, oracle-free release-text queries for retrieval training.

No model or client is imported here. An explicit executor is needed to build a
cache. Training loads a completed cache once, before GPU initialization, and
uses its same fixed strings for optimization and both validation passes.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time
from types import MappingProxyType

from .io import read_json, read_jsonl, sha256_file, write_json
from .released_text_query import (contract_identity, description_request, digest,
                                 execute_request, rewrite_request, text_query_messages)

PROTOCOL = "pearl-released-text-query-cache-v1"
QUERY_MODE = "release_text_only_rewritten_query"
_ORACLE_KEYS = {"answer", "answer_index", "correct_answer", "gold", "options", "evidence",
                "positive_clip_ids", "negative_clip_ids", "candidate_negative_clip_ids", "candidates",
                "current_clip", "verification", "narrative_qualification"}


def _no_oracle_keys(value):
    if isinstance(value, dict):
        if set(value) & _ORACLE_KEYS:
            raise ValueError("Query-cache records must not contain answer/options/evidence or retrieval labels")
        for item in value.values():
            _no_oracle_keys(item)
    elif isinstance(value, list):
        for item in value:
            _no_oracle_keys(item)


def implementation_identity():
    return {name: sha256_file(Path(__file__).with_name(name))
            for name in ["text_query_cache.py", "released_text_query.py"]}


def checkpoint_identity(model_path):
    """Bind the actual local base-helper weights and processor, never an adapter."""
    root = Path(model_path).resolve(strict=True)
    if not root.is_dir() or (root / "adapter_config.json").exists() or list(root.glob("adapter_model.*")):
        raise ValueError("The query helper must be an explicit base checkpoint without a PEFT adapter")
    config = read_json(root / "config.json")
    if "peft_type" in config:
        raise ValueError("A PEFT checkpoint cannot generate fixed base-helper queries")
    weights = sorted([*root.glob("*.safetensors"), *root.glob("pytorch_model*.bin")])
    if not weights:
        raise ValueError("Base-helper checkpoint has no local model weights")
    suffixes = {".json", ".jinja", ".model", ".txt", ".tiktoken"}
    files = sorted(set(weights + [p for p in root.iterdir() if p.is_file() and p.suffix in suffixes]))
    return {"model_path": str(root), "adapter_attached": False,
            "files": {str(p.relative_to(root)): {"bytes": p.stat().st_size, "sha256": sha256_file(p)} for p in files}}


def _sample_map(samples):
    result = {}
    for sample in samples:
        sid = sample.get("sample_id")
        if not isinstance(sid, str) or not sid or sid in result:
            raise ValueError("Query caching requires distinct explicit sample IDs")
        result[sid] = sample
    return result


def _light_binding(sample):
    """An in-memory mutation guard; labels/options are never read."""
    return digest({"question": sample["question"], "concepts": [
        {key: concept.get(key, "frame" if key == "concept_type" else None)
         for key in ["name", "image", "description", "concept_type"]}
        for concept in sample["concepts"]]})


def visible_input(sample, *, model_path, release_root=None):
    # With no replacement descriptions this exact released request is a no-call
    # fallback: its text is the question stem after the original options parser.
    stem = rewrite_request(sample, {}, model_path=model_path, release_root=release_root)
    if stem.payload is not None or not isinstance(stem.fallback_text, str) or not stem.fallback_text.strip():
        raise ValueError("Could not obtain the released option-free question stem")
    concepts, names = [], set()
    for concept in sample["concepts"]:
        name, image, description = concept["name"], concept["image"], concept["description"]
        if not all(isinstance(x, str) and x for x in (name, image, description)) or name in names:
            raise ValueError("Registered concepts need unique names and explicit frame/description inputs")
        names.add(name)
        path = Path(image).resolve(strict=True)
        concepts.append({"name": name, "image": str(path), "pixel_sha256": sha256_file(path),
                         "description": description, "concept_type": concept.get("concept_type", "frame")})
    return {"question_stem": stem.fallback_text, "concepts": concepts}


def cache_plan(samples_path, helper_model, *, release_root=None):
    samples_path = Path(samples_path).resolve(strict=True)
    sample_bytes = samples_path.read_bytes()
    samples = [json.loads(line) for line in sample_bytes.splitlines() if line.strip()]
    indexed = _sample_map(samples)
    chosen = {sid: sample for sid, sample in indexed.items() if sample["task"] == "past"}
    if not chosen:
        raise ValueError("A retrieval query cache needs past training/validation samples")
    model = checkpoint_identity(helper_model)
    contract = contract_identity(release_root)
    bindings = {sid: digest(visible_input(sample, model_path=model["model_path"], release_root=release_root))
                for sid, sample in chosen.items()}
    identity = {"protocol": PROTOCOL, "samples_sha256": hashlib.sha256(sample_bytes).hexdigest(),
        "base_helper": model, "helper_contract": contract, "implementation_sha256": implementation_identity(),
        "query_mode": QUERY_MODE, "sample_ids": list(chosen), "input_sha256": bindings,
        "helper_contract_sha256": digest(contract), "base_helper_sha256": digest(model)}
    _no_oracle_keys(identity)
    return identity, chosen


def _operation(request, executor, *, identity, memo, work_dir):
    key = digest({"base_helper": identity["base_helper_sha256"],
                  "visible_request": request.identity_sha256,
                  "wire_request": request.metadata["request_payload_sha256"]})
    path = work_dir / (key + ".json") if work_dir is not None else None
    if key not in memo and path is not None and path.exists():
        memo[key] = read_json(path)
    if key not in memo:
        try:
            text, receipt = execute_request(request, executor)
        except Exception as error:
            if work_dir is not None:
                write_json(work_dir / (key + f".error.{time.time_ns()}.json"), {
                    "payload": request.payload, "metadata": request.metadata,
                    "error_type": type(error).__name__, "error": str(error)})
            raise
        operation = {"payload": request.payload, "metadata": request.metadata,
                     "identity_sha256": request.identity_sha256, "text": text, "receipt": receipt}
        _no_oracle_keys(operation)
        memo[key] = operation
        if path is not None:
            write_json(path, operation)
    operation = memo[key]
    _check_operation(operation, request, execution_kind=identity["execution_kind"])
    return json.loads(json.dumps(operation))


def _check_operation(operation, request, *, execution_kind):
    if (set(operation) != {"payload", "metadata", "identity_sha256", "text", "receipt"}
            or operation["payload"] != request.payload or operation["metadata"] != request.metadata
            or operation["identity_sha256"] != request.identity_sha256):
        raise ValueError("Cached helper operation differs from the original released request")
    text, receipt = operation["text"], operation["receipt"]
    if not isinstance(receipt, dict):
        raise ValueError("A helper operation needs its structured receipt")
    if not isinstance(text, str) or not text.strip() or text != text.strip():
        raise ValueError("Cached query helper returned invalid text")
    expected = {**request.metadata, "kind": request.kind, "api_request_made": request.payload is not None,
                "response_text_sha256": digest(text), "call": receipt.get("call")}
    if receipt != expected:
        raise ValueError("Query-cache response/receipt binding changed")
    if request.payload is None:
        if text != request.fallback_text or receipt["call"] is not None:
            raise ValueError("No-call rewrite must use the exact released fallback")
    elif execution_kind == "loopback_base_helper":
        call = receipt["call"]
        if (not isinstance(call, dict) or call.get("http_status") != 200
                or call.get("model") != request.payload["model"] or not call.get("request_id")):
            raise ValueError("A real cache needs actual successful base-helper transport receipts")
    _no_oracle_keys(operation)


def build_text_query_cache(samples_path, output, helper_model, executor, *, release_root=None,
                          execution_kind="injected_cpu_test", work_dir=None):
    """Materialize exact description+rewrite outputs with explicit execution.

    The default test marker cannot be used for training. The CLI's explicit
    --execute path supplies a loopback executor and the real execution marker.
    """
    output = Path(output)
    if output.exists():
        raise FileExistsError("Preserve a completed query cache; choose a new path")
    if execution_kind not in {"loopback_base_helper", "injected_cpu_test"}:
        raise ValueError("Unknown helper execution kind")
    identity, samples = cache_plan(samples_path, helper_model, release_root=release_root)
    identity["execution_kind"] = execution_kind
    work_dir = Path(work_dir) if work_dir is not None else None
    if work_dir is not None:
        work_dir.mkdir(parents=True, exist_ok=True)
        state_path = work_dir / "identity.json"
        if state_path.exists() and read_json(state_path) != identity:
            raise ValueError("Partial helper outputs belong to different samples/model/prompts")
        write_json(state_path, identity)
    memo, records = {}, []
    model = identity["base_helper"]["model_path"]
    for sid, sample in samples.items():
        descriptions, by_name = [], {}
        for concept in sample["concepts"]:
            request = description_request(concept, model_path=model, release_root=release_root)
            operation = _operation(request, executor, identity=identity, memo=memo, work_dir=work_dir)
            by_name[concept["name"]] = operation["text"]
            descriptions.append({"name": concept["name"], "operation": operation})
        request = rewrite_request(sample, by_name, model_path=model, release_root=release_root)
        rewrite = _operation(request, executor, identity=identity, memo=memo, work_dir=work_dir)
        binding = visible_input(sample, model_path=model, release_root=release_root)
        if digest(binding) != identity["input_sha256"][sid]:
            raise ValueError("Registered reference pixels/input changed during query generation")
        row = {"sample_id": sid, "input_binding": binding, "input_sha256": digest(binding),
               "descriptions": descriptions, "rewrite": rewrite}
        row["record_sha256"] = digest(row)
        records.append(row)
    if sha256_file(samples_path) != identity["samples_sha256"]:
        raise ValueError("Qualified samples changed while helper queries were being generated")
    if (checkpoint_identity(helper_model) != identity["base_helper"]
            or contract_identity(release_root) != identity["helper_contract"]
            or implementation_identity() != identity["implementation_sha256"]):
        raise ValueError("Base helper, implementation or release prompts changed during query generation")
    if any(digest(visible_input(sample, model_path=model, release_root=release_root)) != identity["input_sha256"][sid]
           for sid, sample in samples.items()):
        raise ValueError("Registered reference pixels/input changed during query generation")
    artifact = {"manifest": {**identity, "status": "complete", "queries": len(records)}, "records": records}
    _no_oracle_keys(artifact)
    artifact["cache_binding_sha256"] = digest(artifact)
    write_json(output, artifact)
    return artifact


@dataclass(frozen=True)
class TextQueryCache:
    queries: object
    sample_bindings: object
    metadata: object

    def messages(self, sample):
        sid = sample["sample_id"]
        if sid not in self.queries or self.sample_bindings[sid] != _light_binding(sample):
            raise ValueError("Training sample differs from the validated query cache input")
        return text_query_messages(self.queries[sid])


def load_text_query_cache(path, samples_path, *, samples=None, release_root=None, allow_test_cache=False):
    """Rebuild all visible bindings once; no model/client/GPU is involved."""
    path = Path(path).resolve(strict=True)
    cache_bytes = path.read_bytes()
    artifact = json.loads(cache_bytes)
    _no_oracle_keys(artifact)
    if set(artifact) != {"manifest", "records", "cache_binding_sha256"}:
        raise ValueError("Invalid query cache root fields")
    if artifact["cache_binding_sha256"] != digest({k: v for k, v in artifact.items() if k != "cache_binding_sha256"}):
        raise ValueError("Query cache content/hash changed")
    manifest = artifact["manifest"]
    if manifest.get("status") != "complete" or manifest.get("protocol") != PROTOCOL:
        raise ValueError("Training requires a completed known query cache")
    if manifest.get("execution_kind") != "loopback_base_helper" and not (allow_test_cache and manifest.get("execution_kind") == "injected_cpu_test"):
        raise ValueError("CPU-test query outputs cannot be used for actual training")
    expected, file_samples = cache_plan(samples_path, manifest["base_helper"]["model_path"], release_root=release_root)
    if set(manifest) != set(expected) | {"execution_kind", "status", "queries"}:
        raise ValueError("Invalid query cache manifest fields")
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"Query cache has a different {key}")
    if samples is not None:
        supplied = _sample_map(samples)
        actual = _sample_map(read_jsonl(samples_path))
        if set(supplied) != set(actual) or any(digest(supplied[sid]) != digest(row) for sid, row in actual.items()):
            raise ValueError("In-memory training rows differ from the frozen samples file")
    indexed = {}
    model = manifest["base_helper"]["model_path"]
    for row in artifact["records"]:
        if set(row) != {"sample_id", "input_binding", "input_sha256", "descriptions", "rewrite", "record_sha256"}:
            raise ValueError("Invalid query cache row fields")
        sid = row["sample_id"]
        if sid in indexed or sid not in file_samples:
            raise ValueError("Duplicate or foreign cached sample ID")
        sample = file_samples[sid]
        binding = visible_input(sample, model_path=model, release_root=release_root)
        if (row["input_binding"] != binding or row["input_sha256"] != digest(binding)
                or row["record_sha256"] != digest({k: v for k, v in row.items() if k != "record_sha256"})):
            raise ValueError("Cached sample/reference pixels changed")
        if len(row["descriptions"]) != len(sample["concepts"]):
            raise ValueError("Incomplete concept description cache")
        descriptions = {}
        for concept, saved in zip(sample["concepts"], row["descriptions"]):
            if set(saved) != {"name", "operation"} or saved["name"] != concept["name"]:
                raise ValueError("Cached concept binding changed")
            request = description_request(concept, model_path=model, release_root=release_root)
            _check_operation(saved["operation"], request, execution_kind=manifest["execution_kind"])
            descriptions[concept["name"]] = saved["operation"]["text"]
        request = rewrite_request(sample, descriptions, model_path=model, release_root=release_root)
        _check_operation(row["rewrite"], request, execution_kind=manifest["execution_kind"])
        indexed[sid] = row["rewrite"]["text"]
    if set(indexed) != set(file_samples) or manifest.get("queries") != len(indexed):
        raise ValueError("Query cache must cover every past train/validation question exactly once")
    cache_sha = hashlib.sha256(cache_bytes).hexdigest()
    if sha256_file(path) != cache_sha or sha256_file(samples_path) != expected["samples_sha256"]:
        raise ValueError("Query cache or samples changed during CPU validation")
    metadata = {"query_mode": QUERY_MODE, "cache_path": str(path), "cache_sha256": cache_sha,
        "samples_sha256": expected["samples_sha256"], "base_helper_sha256": expected["base_helper_sha256"],
        "helper_contract_sha256": expected["helper_contract_sha256"], "queries": len(indexed)}
    return TextQueryCache(MappingProxyType(indexed), MappingProxyType({sid: _light_binding(s) for sid, s in file_samples.items()}),
                          MappingProxyType(metadata))
