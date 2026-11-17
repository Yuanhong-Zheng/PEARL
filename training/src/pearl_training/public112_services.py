"""Explicitly started public112 backends; importing this file loads no models.

The reader uses the release's own message conversion and vision preprocessing.
Only final QA selects a vLLM LoRARequest. The embedding model is the unchanged
released class, with a PEFT adapter enabled only for memory-query text batches.
"""
from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path
import sys
import threading
import time
import uuid

from .public112_adapter import (PROTOCOL, BaseHelperCache, QueryOnlyEmbedder,
                                arm_settings, reader_adapter_enabled, reader_stage, sha)
from .public112_adapter import runtime_identity
from .released_text_query import digest
from .raw_tokens import PROTOCOL as TOKEN_PROTOCOL, capture_enabled, capture_mode, writer
from .raw_token_backends import RecordedHelperCache, VLLMTokenCapture, trace_embedder


def _adapter(path):
    if not isinstance(path, (str, Path)):
        raise ValueError("This arm needs an explicit trained adapter directory")
    root = Path(path).resolve(strict=True)
    config = json.loads((root/"adapter_config.json").read_text())
    allowed = {"q_proj", "k_proj", "v_proj", "o_proj"}
    if (config.get("peft_type") != "LORA" or config.get("bias", "none") != "none"
            or config.get("modules_to_save") or not isinstance(config.get("target_modules"), list)
            or not config["target_modules"]
            or any(name.rsplit(".", 1)[-1] not in allowed or
                   ("." in name and ".language_model." not in name) for name in config["target_modules"])):
        raise ValueError("Only the trained language-attention LoRA adapter is in scope")
    # PEFT can minimize a long target list to q_proj/k_proj/... on save. Inspect
    # actual tensor names as well; a suffix alone does not prove adapter scope.
    with (root/"adapter_model.safetensors").open("rb") as stream:
        length = int.from_bytes(stream.read(8), "little")
        if not 0 < length <= 16*1024*1024:
            raise ValueError("Invalid saved LoRA tensor header")
        tensors = json.loads(stream.read(length))
    names = [name for name in tensors if name != "__metadata__"]
    if not names or any(".language_model." not in name or name.split(".")[-3] not in allowed
                        or name.split(".")[-2] not in {"lora_A", "lora_B"}
                        or not name.endswith(".weight") for name in names):
        raise ValueError("Saved LoRA tensors extend beyond language attention")
    return {"path": str(root), "rank": config["r"],
            "files_sha256": {name: sha(root/name) for name in
                             ("adapter_config.json", "adapter_model.safetensors")}}


def service_identity(kind, arm, model, adapter, release_root):
    if kind not in {"reader", "embedding"}:
        raise ValueError("Unknown public112 backend kind")
    settings = arm_settings(arm)
    enabled = settings["final_qa_lora" if kind == "reader" else "memory_query_lora"]
    base = Path(model).resolve(strict=True)
    from .text_query_cache import checkpoint_identity
    root = Path(release_root).resolve(strict=True)
    files = ("video_qa_inference.py", "concept_desc.py", "utils.py",
             "server/qwenvl_flask_server.py", "third_party/Qwen3-VL-Embedding/src/models/qwen3_vl_embedding.py")
    return {"protocol": PROTOCOL, "kind": kind, **settings,
            "model": str(base), "config_sha256": sha(base/"config.json"),
            "base_checkpoint": checkpoint_identity(base), "runtime": runtime_identity(),
            "adapter": _adapter(adapter) if enabled else None,
            "raw_token_protocol": TOKEN_PROTOCOL,
            "raw_token_mode": capture_mode(),
            "raw_token_implementation_sha256": {name: sha(Path(__file__).with_name(name))
                                                 for name in ("raw_tokens.py", "raw_token_backends.py")},
            "release_sha256": {name: sha(root/name) for name in files},
            "implementation_sha256": sha(__file__)}


def create_reader_app(executor, identity, helper_cache):
    """CPU-testable Flask routes; executor(payload, adapter_enabled) is injected."""
    from flask import Flask, jsonify, request
    app = Flask(__name__+".reader")

    @app.get("/health")
    def health():
        return jsonify({**identity, "model_loaded": True})

    @app.post("/v1/chat/completions")
    def complete():
        payload = request.get_json()
        if not isinstance(payload, dict) or payload.get("model") != identity["model"]:
            return jsonify({"error": "Request must name the frozen base reader model"}), 400
        try:
            stage = reader_stage(payload)
            if stage == "qa":
                response = executor(payload, reader_adapter_enabled(identity["arm"], payload))
                cached = False
            else:
                response, cached = helper_cache.invoke(payload, lambda body: executor(body, False))
            return jsonify({**response, "pearl_adapter_route": {
                "protocol": PROTOCOL, "stage": stage, "base_helper_cache_hit": cached,
                "adapter_enabled": reader_adapter_enabled(identity["arm"], payload)}})
        except ValueError as error:
            return jsonify({"error": str(error)}), 400
    return app


def create_embedding_app(router, identity):
    from flask import Flask, jsonify, request
    app = Flask(__name__+".embedding")

    @app.get("/health")
    def health():
        return jsonify({**identity, "model_loaded": True})

    @app.post("/get_embeddings")
    def get_embeddings():
        try:
            return jsonify({"embeddings": router.process(request.get_json()["inputs"]).tolist()})
        except (ValueError, TypeError, KeyError) as error:
            return jsonify({"error": str(error)}), 400

    @app.post("/compute_similarity")
    def compute_similarity():
        try:
            body = request.get_json()
            queries = router.process(body["queries"])
            documents = router.documents(body["documents"])
            return jsonify({"similarity_scores": (queries @ documents.T).tolist()})
        except (ValueError, TypeError, KeyError) as error:
            return jsonify({"error": str(error)}), 400
    return app


def load_embedding_app(identity, release_root):
    """Called only by an explicit --execute service invocation."""
    if capture_enabled():
        writer("embedding")
    path = Path(release_root)/"third_party/Qwen3-VL-Embedding/src/models/qwen3_vl_embedding.py"
    spec = importlib.util.spec_from_file_location("pearl_public112_released_embedding", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    # Deliberately no dtype, attention, FPS, pixel or processor override.
    embedder = module.Qwen3VLEmbedder(model_name_or_path=identity["model"])
    if identity["adapter"] is not None:
        from peft import PeftModel
        embedder.model = PeftModel.from_pretrained(embedder.model, identity["adapter"]["path"],
                                                  is_trainable=False)
        embedder.model.eval()
    identity = {**identity, "actual_model_dtype": str(next(embedder.model.parameters()).dtype),
                "embedding_settings": {key: getattr(embedder, key) for key in
                    ("max_length", "min_pixels", "max_pixels", "total_pixels", "fps",
                     "num_frames", "max_frames", "default_instruction")}}
    trace_embedder(embedder, identity)
    router = QueryOnlyEmbedder(embedder, arm=identity["arm"], adapter_loaded=identity["adapter"] is not None)
    return create_embedding_app(router, identity)


def load_reader_app(identity, release_root, helper_cache_dir, tensor_parallel_size=1):
    """Same release preprocessing/LLM parameters, plus final-QA LoRA routing."""
    if capture_enabled():
        writer("vllm")
    import os
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    from qwen_vl_utils import process_vision_info
    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams

    path = Path(release_root)/"server/qwenvl_flask_server.py"
    names = {"prepare_inputs_for_vllm", "convert_openai_messages_to_qwen_format"}
    nodes = [node for node in ast.parse(path.read_text()).body
             if isinstance(node, ast.FunctionDef) and node.name in names]
    if {node.name for node in nodes} != names:
        raise ValueError("Released reader preprocessing functions changed")
    namespace = {"process_vision_info": process_vision_info}
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), str(path), "exec"), namespace)
    processor = AutoProcessor.from_pretrained(identity["model"])
    engine = {"model": identity["model"], "mm_encoder_tp_mode": "data",
              "tensor_parallel_size": tensor_parallel_size, "seed": 0,
              "gpu_memory_utilization": 0.8, "max_model_len": 200000}
    lora = None
    if identity["adapter"] is not None:
        from vllm.lora.request import LoRARequest
        engine.update(enable_lora=True, max_lora_rank=identity["adapter"]["rank"])
        lora = LoRARequest("narrative-final-qa", 1, identity["adapter"]["path"])
    llm = LLM(**engine)
    capture = VLLMTokenCapture(llm)
    lock = threading.Lock()

    def executor(payload, adapter_enabled):
        if adapter_enabled and lora is None:
            raise ValueError("The final-QA adapter was not loaded")
        with lock:
            messages = namespace["convert_openai_messages_to_qwen_format"](payload["messages"])
            inputs = namespace["prepare_inputs_for_vllm"](messages, processor)
            sampling = SamplingParams(temperature=payload.get("temperature", 0.0),
                max_tokens=payload.get("max_tokens", 1024), top_k=payload.get("top_k", -1), stop_token_ids=[])
            start = time.perf_counter()
            kwargs = {"lora_request": lora} if adapter_enabled else {}
            outputs, token_record = capture.generate(llm, inputs, sampling,
                metadata={"identity": identity, "payload": payload, "adapter_enabled": adapter_enabled}, **kwargs)
            text = outputs[0].outputs[0].text
            elapsed = round((time.perf_counter()-start)*1000)
        return {"id": "chatcmpl-"+uuid.uuid4().hex, "object": "chat.completion",
                "created": int(time.time()), "model": identity["model"],
                "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                             "finish_reason": "stop"}],
                "usage": {"prompt_tokens": -1, "completion_tokens": -1, "total_tokens": -1},
                "inference_time_ms": elapsed, "pearl_raw_token_trace": token_record}

    # Adapter identity is deliberately absent: every arm shares these base calls.
    base = {"checkpoint": identity["base_checkpoint"], "adapter": None,
            "release": identity["release_sha256"]}
    helpers = RecordedHelperCache(BaseHelperCache(helper_cache_dir, base), base)
    return create_reader_app(executor, {**identity, "engine_settings": engine,
                                       "helper_base_sha256": digest(base)}, helpers)
