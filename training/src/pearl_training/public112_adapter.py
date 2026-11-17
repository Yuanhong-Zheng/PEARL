"""Isolated public112 adapter routing and CPU-only protocol preflight.

The release is input data/code, never edited. Reader adapters affect final QA
only; description/rewrite helpers and all document embeddings stay at base.
No model, client, GPU or service is constructed when importing this module.
"""
from __future__ import annotations

from collections import Counter
import contextlib
import ctypes
import fcntl
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import threading
import sys

from .released_text_query import DEFAULT_RELEASE_ROOT, _visible_input, contract_identity, digest
from .raw_tokens import PROTOCOL as TOKEN_PROTOCOL, capture_enabled, capture_mode

PROTOCOL = "pearl-public112-text-query-adapters-v1"
ARMS = {"baseline": (False, False), "embedding_only": (True, False),
        "reader_only": (False, True), "joint": (True, True)}
QA_SUFFIX = "Please output your answer choice in <ans></ans> tags."
REWRITE_PREFIX = "Rewrite the following question by replacing the concept names (in curly braces)"
DESCRIPTION_PREFIX = "Based on the image and the original description provided, generate a concise visual description"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def runtime_identity():
    result = {"python": ".".join(map(str, sys.version_info[:3])),
              **{name: importlib.metadata.version(name) for name in
                 ("torch", "transformers", "peft", "vllm", "qwen-vl-utils", "triton")}}
    override = os.environ.get("PEARL_NVRTC_LIBRARY")
    if override:
        library = Path(override).resolve()
        preloads = os.environ.get("LD_PRELOAD", "").replace(":", " ").split()
        if not any(Path(value).resolve() == library for value in preloads):
            raise ValueError("NVRTC runtime identity requires its actual LD_PRELOAD override")
        major, minor = ctypes.c_int(), ctypes.c_int()
        if ctypes.CDLL(str(library)).nvrtcVersion(ctypes.byref(major), ctypes.byref(minor)):
            raise RuntimeError("Cannot query the selected NVRTC runtime")
        result["nvrtc_override"] = {"path": str(library), "sha256": sha(library),
                                    "version": [major.value, minor.value]}
    return result


def base_artifacts(directory):
    return {str(path): sha(path) for path in sorted(Path(directory).iterdir()) if path.is_file()}


def validated_completed_receipt(plan, arm, source):
    """Recheck a closed annotation before resume or reporting any score."""
    name = Path(source["annotation"]).stem
    directory = Path(plan["output"])/arm
    marker = directory/"completed"/(name+".json")
    receipt = json.loads(marker.read_text())
    shared = Path(plan["output"])/"shared-base-cache"/name
    documents = shared/(Path(source["clips_info"]).stem+"_embeddings_video.npy")
    outputs = {str(directory/"traces"/(name+".json")), str(directory/"results"/(name+"_result.json")),
               str(directory/"results"/(name+"_evaluation.json"))}
    if (receipt.get("binding_sha256") != digest({"plan": plan, "arm": arm, "source": source})
            or set(receipt.get("output_sha256", {})) != outputs
            or any(sha(path) != checksum for path, checksum in receipt["output_sha256"].items())
            or receipt.get("base_document_cache_sha256") != sha(documents)
            or receipt.get("shared_base_artifact_sha256") != base_artifacts(shared)):
        raise ValueError("A completed public112 annotation changed")
    return receipt


def reader_stage(payload):
    """Recognize the unchanged release requests, never infer stage from gold."""
    if not isinstance(payload, dict) or not isinstance(payload.get("messages"), list):
        raise ValueError("An original OpenAI-format reader request is required")
    messages = payload["messages"]
    if len(messages) != 1 or messages[0].get("role") != "user" or not isinstance(messages[0].get("content"), list):
        raise ValueError("Unrecognized released reader message layout")
    parts = messages[0]["content"]
    texts = [part.get("text", "") for part in parts if part.get("type") == "text"]
    if (len(parts) == 1 and len(texts) == 1 and texts[0].startswith(REWRITE_PREFIX)
            and payload.get("max_tokens") == 512 and payload.get("temperature") == 0):
        return "rewrite"
    if (len(parts) == 2 and len(texts) == 1 and texts[0].startswith(DESCRIPTION_PREFIX)
            and parts[0].get("type") == "image_url"
            and payload.get("max_tokens") == 512 and payload.get("temperature") == 0.7):
        return "description"
    if (texts and texts[-1].endswith(QA_SUFFIX) and payload.get("max_tokens") == 1024
            and payload.get("temperature") == 0):
        return "qa"
    raise ValueError("Unknown reader request; refusing to guess an adapter stage")


def arm_settings(arm):
    if arm not in ARMS:
        raise ValueError("Unknown public112 ablation arm")
    embedding, reader = ARMS[arm]
    return {"arm": arm, "memory_query_lora": embedding, "final_qa_lora": reader,
            "concept_description_lora": False, "rewrite_lora": False,
            "concept_retrieval_lora": False, "video_document_lora": False}


def reader_adapter_enabled(arm, payload):
    return arm_settings(arm)["final_qa_lora"] and reader_stage(payload) == "qa"


def embedding_role(item):
    if not isinstance(item, dict):
        raise ValueError("A released embedding input must be an object")
    if item.get("video") is not None or item.get("image") is not None:
        return "document"
    if isinstance(item.get("text"), str) and item["text"]:
        return "memory_query"
    raise ValueError("Empty or unknown released embedding input")


class QueryOnlyEmbedder:
    """Thread-safe routing around an actual released Qwen3VLEmbedder instance.

    Callers create/load the model only in an explicitly executed future service.
    The fake-object CPU tests exercise the same disable_adapter boundary.
    """
    def __init__(self, embedder, *, arm, adapter_loaded):
        self.embedder, self.settings = embedder, arm_settings(arm)
        self.adapter_loaded = adapter_loaded
        if self.settings["memory_query_lora"] and not adapter_loaded:
            raise ValueError("This arm needs its trained memory-query adapter")
        self.lock = threading.RLock()

    def process(self, inputs, normalize=True):
        if not isinstance(inputs, list) or not inputs:
            raise ValueError("A nonempty released embedding batch is required")
        roles = {embedding_role(item) for item in inputs}
        if len(roles) != 1:
            raise ValueError("Query/document batches must be separated before adapter routing")
        enabled = roles == {"memory_query"} and self.settings["memory_query_lora"]
        with self.lock:
            context = contextlib.nullcontext() if enabled or not self.adapter_loaded else self.embedder.model.disable_adapter()
            with context:
                return self.embedder.process(inputs, normalize=normalize)

    def documents(self, inputs, normalize=True):
        """Explicit document route also keeps text documents at the base model."""
        if not isinstance(inputs, list) or not inputs:
            raise ValueError("A nonempty document batch is required")
        with self.lock:
            context = self.embedder.model.disable_adapter() if self.adapter_loaded else contextlib.nullcontext()
            with context:
                return self.embedder.process(inputs, normalize=normalize)


class BaseHelperCache:
    """Cross-arm base helper responses, bound to exact text and reference pixels."""
    def __init__(self, directory, base_identity):
        if base_identity.get("adapter") is not None:
            raise ValueError("Description/rewrite helper identity must have no adapter")
        self.directory = Path(directory)
        self.base_identity = base_identity

    def invoke(self, payload, executor):
        stage = reader_stage(payload)
        if stage not in {"description", "rewrite"}:
            raise ValueError("Final QA responses must never enter the shared helper cache")
        identity = {"protocol": PROTOCOL, "base": self.base_identity, "stage": stage,
                    "visible_input": _visible_input(payload)}
        key = digest(identity)
        self.directory.mkdir(parents=True, exist_ok=True)
        target = self.directory / (key+".json")
        with (self.directory / (key+".lock")).open("a+") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            if target.exists():
                record = json.loads(target.read_text())
                if (record.get("identity") != identity or record.get("response_sha256") != digest(record.get("response"))):
                    raise ValueError("A shared base-helper cache record changed")
                return record["response"], True
            response = executor(payload)
            text = response["choices"][0]["message"]["content"]
            if not isinstance(text, str) or not text.strip():
                raise ValueError("A base helper returned no usable text")
            record = {"identity": identity, "response": response, "response_sha256": digest(response),
                      "first_wire_request_sha256": digest(payload)}
            temporary = target.with_suffix(".tmp")
            temporary.write_text(json.dumps(record, ensure_ascii=False, indent=2)+"\n")
            temporary.replace(target)
            return response, False


def _seconds(value):
    if isinstance(value, str):
        parts = [float(part) for part in value.split(":")]
        if not 1 <= len(parts) <= 3:
            raise ValueError("Malformed original timestamp")
        result = sum(part*60**index for index, part in enumerate(reversed(parts)))
    elif not isinstance(value, bool) and isinstance(value, (int, float)):
        result = float(value)
    else:
        raise ValueError("Malformed original timestamp")
    if not math.isfinite(result) or result < 0:
        raise ValueError("Invalid original timestamp")
    return result


def corpus_inventory(release_root=None, *, expected=(112, 418, 922, 394, 382)):
    """Read annotations/clip metadata only; preserve all QA and the old Recall mask."""
    root = Path(release_root or DEFAULT_RELEASE_ROOT).resolve(strict=True)
    directory = root / "data/frame-level/annotations"
    if not directory.is_dir():
        raise ValueError("Pass the released corpus root containing data/frame-level/annotations")
    counters, sources, excluded, eligible = Counter(), [], [], []
    for path in sorted(directory.glob("*.json")):
        annotations = json.loads(path.read_text())
        counters["annotation_files"] += 1
        keys = set()
        for annotation in annotations:
            video_name = Path(annotation["video_path"]).stem
            clips_path = root / "data/frame-level/output_clips" / video_name / (video_name+"_clips_info.json")
            clips = json.loads(clips_path.read_text())["clips"]
            for qa in annotation["timestamps"]:
                key = (path.name, qa["id"])
                if key in keys:
                    raise ValueError("Duplicate original question ID")
                keys.add(key)
                kind = qa["qa_type"]
                if kind not in {"concept definition", "current-time qa", "past-time qa"}:
                    raise ValueError("Only original frame-level Entity QA is in scope")
                counters[kind] += 1
                if kind != "past-time qa":
                    continue
                time, evidence = _seconds(qa.get("time") or qa.get("end_time")), _seconds(qa["evidence_time"])
                matches = [clip for clip in clips if clip["start_time"] <= evidence < clip["end_time"]]
                gold = [clip["clip_id"] for clip in matches if clip["end_time"] < time]
                item = {"annotation": path.name, "qa_id": qa["id"]}
                if gold:
                    eligible.append({**item, "gold_clip_ids": gold})
                else:
                    excluded.append({**item, "reason": "no_matching_clip" if not matches else "not_eligible_history"})
            sources.append({"annotation": str(path.resolve()), "annotation_sha256": sha(path),
                            "clips_info": str(clips_path.resolve()), "clips_info_sha256": sha(clips_path)})
    actual = (counters["annotation_files"], counters["concept definition"], counters["current-time qa"],
              counters["past-time qa"], len(eligible))
    if actual != tuple(expected):
        raise ValueError(f"Public corpus differs from the frozen protocol: {actual} != {expected}")
    return {"counts": dict(counters), "recall_eligible_count": len(eligible), "recall_excluded": excluded,
            "recall_eligible": eligible, "sources": sources,
            "qa_rows_excluded": 0, "future_definition_policy": "preserve_original_all-definition_preload",
            "recall_rule": "clip.start <= evidence < clip.end and clip.end < question_time",
            "action_evaluation": False}


def make_plan(*, release_root=None, embedding_base, reader_base, embedding_adapter, reader_adapter,
              output, arms=("baseline", "joint")):
    root = Path(release_root or DEFAULT_RELEASE_ROOT).resolve(strict=True)
    arms = tuple(arms)
    if len(set(arms)) != len(arms) or not {"baseline", "joint"} <= set(arms):
        raise ValueError("At least baseline and joint must be planned exactly once")
    if arms[0] != "baseline":
        raise ValueError("Run baseline first to freeze the common base document/helper caches")
    settings = [arm_settings(arm) for arm in arms]
    checkpoints = {}
    from .text_query_cache import checkpoint_identity
    for name, value in (("embedding_base", embedding_base), ("reader_base", reader_base)):
        path = Path(value).resolve(strict=True)
        checkpoints[name] = {"path": str(path), "config_sha256": sha(path / "config.json"),
                             "identity": checkpoint_identity(path)}
    adapters = {}
    for name, value in (("embedding", embedding_adapter), ("reader", reader_adapter)):
        path = Path(value).resolve()
        files = {name: sha(path / name) for name in ("adapter_config.json", "adapter_model.safetensors") if (path / name).is_file()}
        adapters[name] = {"path": str(path), "files_sha256": files, "ready": len(files) == 2}
    output = Path(output).resolve()
    if output.is_relative_to(root):
        raise ValueError("New evaluation outputs must stay outside the released baseline tree")
    release_files = ("video_qa_inference.py", "clip_memory.py", "concept_database.py", "utils.py",
                     "concept_desc.py", "server/qwenvl_flask_server.py")
    return {"protocol": PROTOCOL, "release_root": str(root), "output": str(output),
        "release_sha256": {name: sha(root/name) for name in release_files},
        "implementation_sha256": {name: sha(Path(__file__).with_name(name)) for name in
            ("public112_adapter.py", "public112_services.py", "public112_runner.py", "released_text_query.py", "text_query_cache.py",
             "raw_tokens.py", "raw_token_backends.py")},
        "runtime": runtime_identity(),
        "helper_contract": contract_identity(root), "corpus": corpus_inventory(root),
        "checkpoints": checkpoints, "adapters": adapters, "arms": settings,
        "retrieval_query": "base_rewritten_stem_text_only", "top_k": 4, "neighbors": 1, "embedding_fps": 1,
        "reader_rotations": 4, "qa_correct_rule": "all_four_rotations_correct",
        "use_video_embedding": True, "document_batch_size": 4,
        "embedding_dtype": "release_default_no_dtype_override",
        "training_preprocessing": {"precision": "bf16", "video_input": "eight_observed_frames_per_clip",
                                   "document_cache": "separate_bf16_training_encoder_cache",
                                   "public112_cache_reuse": False},
        "public112_documents": "shared_base_release_FP32_original_mp4_cache",
        "reader_engine": {"mm_encoder_tp_mode": "data", "seed": 0,
                          "gpu_memory_utilization": 0.8, "max_model_len": 200000,
                          "tensor_parallel_size": 1},
        "qa_max_new_tokens": 1024, "qa_temperature": 0,
        "raw_token_protocol": TOKEN_PROTOCOL,
        "raw_token_mode": capture_mode(),
        "raw_token_scope": ("all_new_inferences_with_separate_historical_cache_input_lineage"
                            if capture_enabled() else "disabled_by_user_for_benchmark"),
        "base_helpers_shared_across_arms": True, "released_inference_modified": False,
        "gpu_execution_validated": False, "model_calls_made_by_plan": 0,
        "runtime_status": "await500_snapshot_and_saved_adapters_then_GPU_smoke"}


def observe_retrieval(inference):
    """Capture the released ranking using its existing single query forward.

    Released search scores and sorts the complete history before its last slice.
    Requesting ten rows and returning the original requested prefix changes no
    scores, neighbors, prompt, or request count. Only this instance is wrapped.
    """
    original_search = inference.clip_memory.search
    original_expand = inference.expand_clips_with_neighbors
    original_process = inference.process_qa
    active, traces = {}, []

    def search(query, top_k=5):
        if not 0 < top_k <= 10:
            raise ValueError("The frozen evaluation uses retrieval K4 and Recall through K10")
        ranked = original_search(query, top_k=10)
        active["query"] = query
        active["eligible_clip_count"] = len(inference.clip_memory.clips_data)
        active["ranked_top10"] = [{key: row[key] for key in
            ("clip_id", "start_time", "end_time", "similarity_score")} for row in ranked]
        return ranked[:top_k]

    def expand(retrieved_clips, current_clip_start_time=None):
        result = original_expand(retrieved_clips, current_clip_start_time)
        active["expanded_top4"] = [{key: row[key] for key in
            ("clip_id", "start_time", "end_time")} for row in result]
        return result

    def process(qa_item, top_k_clips=1):
        if top_k_clips != 4:
            raise ValueError("The public112 protocol fixes retrieval K4")
        active.clear()
        active.update(qa_id=qa_item["id"], qa_type=qa_item["qa_type"],
                      query=None, eligible_clip_count=0, ranked_top10=[], expanded_top4=[])
        result = original_process(qa_item, top_k_clips=top_k_clips)
        traces.append(json.loads(json.dumps(active)))
        return result

    inference.clip_memory.search = search
    inference.expand_clips_with_neighbors = expand
    inference.process_qa = process
    return traces


def summarize_results(plan, arm):
    """Score unchanged release result files and the same 382-question Recall mask."""
    arm_settings(arm)
    directory = Path(plan["output"])/arm
    totals, traces = Counter(), {}
    for source in plan["corpus"]["sources"]:
        validated_completed_receipt(plan, arm, source)
        name = Path(source["annotation"]).stem
        evaluation = json.loads((directory/"results"/(name+"_evaluation.json")).read_text())
        for key in ("current_time_count", "current_time_correct", "past_time_count", "past_time_correct"):
            totals[key] += evaluation[key]
        for row in json.loads((directory/"traces"/(name+".json")).read_text()):
            key = (Path(source["annotation"]).name, row["qa_id"])
            if key in traces:
                raise ValueError("Repeated public112 question trace")
            traces[key] = row
    if (totals["current_time_count"], totals["past_time_count"], len(traces)) != (922, 394, 1316):
        raise ValueError("Complete public112 QA outputs are required before reporting scores")
    hits, expanded = Counter(), 0
    for item in plan["corpus"]["recall_eligible"]:
        trace = traces[(item["annotation"], item["qa_id"])]
        gold = set(item["gold_clip_ids"])
        ranked = [row["clip_id"] for row in trace["ranked_top10"]]
        if len(set(ranked)) != len(ranked) or len(ranked) != min(10, trace["eligible_clip_count"]):
            raise ValueError("Malformed captured released ranking")
        for k in (1, 2, 4, 8, 10):
            hits[k] += bool(gold & set(ranked[:k]))
        expanded += bool(gold & {row["clip_id"] for row in trace["expanded_top4"]})
    n = plan["corpus"]["recall_eligible_count"]
    return {"protocol": PROTOCOL, **arm_settings(arm), "qa": {
        task: {"count": totals[task+"_time_count"], "correct": totals[task+"_time_correct"],
               "accuracy": totals[task+"_time_correct"]/totals[task+"_time_count"]}
        for task in ("current", "past")},
        "recall": {str(k): {"count": n, "hits": hits[k], "rate": hits[k]/n} for k in (1, 2, 4, 8, 10)},
        "expanded_top4": {"count": n, "hits": expanded, "rate": expanded/n},
        "qa_rows_excluded": 0, "recall_only_excluded": len(plan["corpus"]["recall_excluded"])}
