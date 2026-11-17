"""Exact released PEARL text-query helpers, without importing a model/client.

Only selected prompt-building functions are compiled from the unchanged release.
A capture-only client obtains their request payloads; no OpenAI client, network,
torch, processor or GPU is constructed here. Callers own checkpoint identity,
persistent request budgets, transport receipts and immutable cache publication.
"""
from __future__ import annotations

import ast
from contextlib import redirect_stdout
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import io
import json
from pathlib import Path
import re
from types import SimpleNamespace
from typing import List

PROTOCOL = "pearl-released-text-query-v1"
# Root of the released inference code, i.e. the repository root (training/src/pearl_training/ -> ../../..).
DEFAULT_RELEASE_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INSTRUCTION = "Represent the user's input."


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _root(value):
    return Path(value or DEFAULT_RELEASE_ROOT).resolve(strict=True)


def _definition(path, name, owner=None):
    tree = ast.parse(path.read_text())
    body = tree.body
    if owner is not None:
        body = next(node for node in body if isinstance(node, ast.ClassDef) and node.name == owner).body
    return next(node for node in body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name)


def _compile(nodes, filename):
    namespace = {"Path": Path, "re": re, "List": List}
    tree = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *nodes], type_ignores=[])
    exec(compile(ast.fix_missing_locations(tree), filename, "exec"), namespace)
    return namespace


@lru_cache(maxsize=8)
def _builders(root, source_bindings):
    # The hashes are part of the cache key, so a changed release never reuses old
    # compiled builders. Selected release code receives only a capture client.
    root = Path(root)
    nodes = [
        _definition(root / "utils.py", "extract_concepts"),
        _definition(root / "utils.py", "extract_question_without_options"),
        _definition(root / "concept_desc.py", "_to_file_url"),
        _definition(root / "concept_desc.py", "_build_frame_level_prompt"),
        _definition(root / "concept_desc.py", "generate_distinctive_description"),
        _definition(root / "video_qa_inference.py", "replace_concepts_with_descriptions", "VideoQAInference"),
    ]
    return _compile(nodes, "<released-PEARL-prompt-builders>")


def contract_identity(release_root=None):
    root = _root(release_root)
    files = ("utils.py", "concept_desc.py", "video_qa_inference.py",
             "third_party/Qwen3-VL-Embedding/src/models/qwen3_vl_embedding.py")
    sources = {name: _sha(root / name) for name in files}
    prompts = {
        "description": _definition(root / "concept_desc.py", "_build_frame_level_prompt"),
        "rewrite": _definition(root / "video_qa_inference.py", "replace_concepts_with_descriptions", "VideoQAInference"),
        "strip_options": _definition(root / "utils.py", "extract_question_without_options"),
        "extract_concepts": _definition(root / "utils.py", "extract_concepts"),
    }
    embed = ast.parse((root / files[-1]).read_text())
    wrapper = next(node for node in embed.body if isinstance(node, ast.ClassDef) and node.name == "Qwen3VLEmbedder")
    init = next(node for node in wrapper.body if isinstance(node, ast.FunctionDef) and node.name == "__init__")
    defaults = dict(zip([arg.arg for arg in init.args.args][-len(init.args.defaults):], init.args.defaults))
    if ast.literal_eval(defaults["default_instruction"]) != DEFAULT_INSTRUCTION:
        raise ValueError("Released embedding instruction changed")
    constants = {node.targets[0].id: ast.literal_eval(node.value) for node in embed.body
                 if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
                 and isinstance(node.value, ast.Constant)}
    if constants.get("MAX_LENGTH") != 8192 or constants.get("FPS") != 1:
        raise ValueError("Released text length or sampling contract changed")
    return {"protocol": PROTOCOL, "source_sha256": sources, "helper_implementation_sha256": _sha(__file__),
        "prompt_sha256": {key: digest(ast.dump(node, include_attributes=False)) for key, node in prompts.items()},
        "default_instruction": DEFAULT_INSTRUCTION, "pooling": "last_nonpadding_token_then_l2",
        "padding_side": "right", "max_length": 8192, "released_processor_truncation": True,
        "training_input_budget_policy": "caller must reject over-budget inputs rather than silently truncate",
        "query_input": "rewritten_question_stem_text_only", "options_in_query": False,
        "concept_helpers": "base_reader_only", "reader_adapter_scope": "final_qa_only",
        "embedding_adapter_scope": "memory_query_only", "video_documents": "frozen_base"}


def _functions(root, identity):
    return _builders(str(root), tuple(sorted(identity["source_sha256"].items())))


class _Capture:
    def __init__(self):
        self.payload = None
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **payload):
        if self.payload is not None:
            raise ValueError("One helper operation may create at most one request")
        self.payload = payload
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="CAPTURE_ONLY"))])


def _visible_input(payload):
    """Canonical helper-cache identity uses actual pixels, never path spelling."""
    value = json.loads(json.dumps(payload))
    for message in value.get("messages", []):
        content = message.get("content", [])
        if isinstance(content, str):
            continue
        for part in content:
            if part.get("type") == "image_url":
                url = part["image_url"]["url"]
                if not url.startswith("file://"):
                    raise ValueError("Base concept helpers require actual local reference pixels")
                from urllib.parse import unquote, urlparse
                path = Path(unquote(urlparse(url).path)).resolve(strict=True)
                part["image_url"] = {"pixel_sha256": _sha(path)}
            elif part.get("type") != "text":
                raise ValueError("Only frame-level Entity references are supported")
    return value


@dataclass(frozen=True)
class HelperRequest:
    kind: str
    payload: dict | None
    fallback_text: str | None
    metadata: dict

    @property
    def request(self):
        return self.payload

    @property
    def identity_sha256(self):
        return self.metadata["identity_sha256"]


def _request(kind, capture, fallback, identity):
    payload = capture.payload
    metadata = {"protocol": PROTOCOL, "contract_sha256": digest(identity),
                "request_payload_sha256": digest(payload), "base_helper_only": True,
                "visible_input_sha256": digest(_visible_input(payload)) if payload is not None else digest(None),
                "fallback_text_sha256": digest(fallback)}
    metadata["identity_sha256"] = digest({"kind": kind, "contract": identity,
        "visible_input": _visible_input(payload) if payload is not None else None, "fallback_text": fallback})
    return HelperRequest(kind, payload, fallback, metadata)


def description_request(concept, *, model_path, release_root=None):
    """Capture the exact original frame-description request; no other QA seen."""
    root = _root(release_root)
    identity = contract_identity(root)
    name = concept.get("name", concept.get("concept_name"))
    image = concept.get("image", concept.get("frame_path"))
    description = concept.get("description")
    if (concept.get("concept_type", "frame") != "frame" or not all(isinstance(x, str) and x for x in (name, image, description, model_path))):
        raise ValueError("A fixed Entity frame name/image/description and base model are required")
    capture = _Capture()
    _functions(root, identity)["generate_distinctive_description"](
        capture, model_path, image, name, description, "frame")
    return _request("description", capture, None, identity)


def rewrite_request(sample_or_question, descriptions_by_name, *, model_path, release_root=None):
    """Strip options, then capture the released rewrite or its exact fallback.

    A sample is whitelisted to question alone. Its gold, options, evidence,
    annotation, IDs and temporal labels never participate in a helper request.
    """
    root = _root(release_root)
    identity = contract_identity(root)
    functions = _functions(root, identity)
    question = sample_or_question if isinstance(sample_or_question, str) else sample_or_question["question"]
    if not isinstance(question, str) or not question.strip() or not isinstance(model_path, str) or not model_path:
        raise ValueError("A visible question and explicit base model are required")
    stem = functions["extract_question_without_options"](question).strip()
    capture = _Capture()
    owner = SimpleNamespace(concept_retrieval_map=descriptions_by_name, model_path=model_path, client=capture)
    with redirect_stdout(io.StringIO()):
        fallback = functions["replace_concepts_with_descriptions"](owner, stem)
    return _request("rewrite", capture, fallback if capture.payload is None else None, identity)


def _response_text(response):
    if isinstance(response, str):
        text = response
    elif isinstance(response, dict):
        text = response["choices"][0]["message"]["content"]
    else:
        text = response.choices[0].message.content
    if not isinstance(text, str) or not text.strip():
        raise ValueError("The base helper did not return nonempty text")
    return text.strip()


def execute_request(request, executor):
    """Run one explicit injected executor; this helper owns no retry policy."""
    if not isinstance(request, HelperRequest) or request.metadata.get("protocol") != PROTOCOL:
        raise ValueError("A known released helper request is required")
    if digest(request.payload) != request.metadata.get("request_payload_sha256"):
        raise ValueError("Helper request changed after capture")
    if (digest(_visible_input(request.payload) if request.payload is not None else None)
            != request.metadata.get("visible_input_sha256")
            or digest(request.fallback_text) != request.metadata.get("fallback_text_sha256")):
        raise ValueError("Helper reference pixels or fallback changed after capture")
    call = None
    if request.payload is None:
        text = request.fallback_text
        if not isinstance(text, str):
            raise ValueError("A no-call helper needs its exact released fallback")
    else:
        result = executor(json.loads(json.dumps(request.payload)))
        response, call = result if isinstance(result, tuple) and len(result) == 2 else (result, None)
        text = _response_text(response)
    return text, {**request.metadata, "kind": request.kind, "api_request_made": request.payload is not None,
                  "response_text_sha256": digest(text), "call": call}


def text_query_messages(text):
    """Same instruction and text-only conversation as released format_model_input."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("A cached nonempty rewritten text query is required")
    return [{"role": "system", "content": [{"type": "text", "text": DEFAULT_INSTRUCTION}]},
            {"role": "user", "content": [{"type": "text", "text": text}]}]
