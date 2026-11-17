"""Run the unchanged public112 inference class against isolated adapter services."""
from __future__ import annotations

import fcntl
import importlib
import json
from pathlib import Path
import sys
from urllib.parse import urlparse

from .io import read_json, write_json
from .public112_adapter import (PROTOCOL, base_artifacts, observe_retrieval, sha,
                                summarize_results, validated_completed_receipt)
from .released_text_query import digest


def _loopback(value):
    parsed = urlparse(value)
    if (parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}
            or parsed.username or parsed.password or parsed.query or parsed.fragment):
        raise ValueError("Explicit loopback service URLs are required")
    return value.rstrip("/")


def freeze_plan(plan, path):
    path = Path(path)
    if path.exists():
        if read_json(path) != plan:
            raise ValueError("Evaluation inputs/code changed; preserve the old plan and choose a new output")
    else:
        write_json(path, plan)


def _check_health(plan, arm, kind, identity):
    base = plan["checkpoints"][kind+"_base"]
    settings = next(row for row in plan["arms"] if row["arm"] == arm)
    if (identity.get("protocol") != PROTOCOL or identity.get("kind") != kind
            or identity.get("model_loaded") is not True
            or identity.get("model") != base["path"] or identity.get("config_sha256") != base["config_sha256"]
            or identity.get("base_checkpoint") != base["identity"] or identity.get("runtime") != plan["runtime"]
            or any(identity.get(key) != value for key, value in settings.items())):
        raise ValueError("A public112 service differs from the planned arm/checkpoint")
    if identity.get("implementation_sha256") != plan["implementation_sha256"]["public112_services.py"]:
        raise ValueError("The public112 service implementation differs from the frozen plan")
    if (identity.get("raw_token_protocol") != plan.get("raw_token_protocol")
            or identity.get("raw_token_mode", "record") != plan.get("raw_token_mode", "record")
            or any(identity.get("raw_token_implementation_sha256", {}).get(name) != plan["implementation_sha256"][name]
                   for name in ("raw_tokens.py", "raw_token_backends.py"))):
        raise ValueError("The public112 service is missing the planned raw-token capture implementation")
    enabled = settings["final_qa_lora" if kind == "reader" else "memory_query_lora"]
    expected = plan["adapters"][kind]
    adapter = identity.get("adapter")
    if enabled:
        if (not isinstance(adapter, dict) or adapter.get("path") != expected["path"]
                or adapter.get("files_sha256") != expected["files_sha256"]):
            raise ValueError("A public112 service loaded a different adapter")
    elif adapter is not None:
        raise ValueError("A base-only public112 route unexpectedly has an adapter")
    release_hashes = identity.get("release_sha256", {})
    required = {"video_qa_inference.py", "concept_desc.py", "utils.py", "server/qwenvl_flask_server.py",
                "third_party/Qwen3-VL-Embedding/src/models/qwen3_vl_embedding.py"}
    if set(release_hashes) != required:
        raise ValueError("Missing public112 service release identity")
    for name, checksum in release_hashes.items():
        expected_sha = plan["release_sha256"].get(name, plan["helper_contract"]["source_sha256"].get(name))
        if checksum != expected_sha:
            raise ValueError("A public112 service uses different release code")
    if kind == "embedding" and identity.get("actual_model_dtype") != "torch.float32":
        raise ValueError("Public112 evaluation must retain the release's FP32 embedding backend")
    if kind == "reader" and any(identity.get("engine_settings", {}).get(key) != value
                                 for key, value in plan["reader_engine"].items()):
        raise ValueError("Public112 reader engine settings differ from the release")


def _inference_class(root):
    root = Path(root).resolve()
    sys.path.insert(0, str(root))
    module = importlib.import_module("video_qa_inference")
    for name in ("video_qa_inference", "clip_memory", "concept_database", "concept_desc", "utils"):
        if Path(sys.modules[name].__file__).resolve() != root/(name+".py"):
            raise ValueError("Use a dedicated eval process; another release module is already imported")
    return module.VideoQAInference


def run_arm(plan, arm, reader_url, embedding_url, *, shard_index=0, shard_count=1):
    """Explicit execution only; this function makes the original inference calls."""
    if (type(shard_count) is not int or type(shard_index) is not int
            or not 1 <= shard_count <= 8 or not 0 <= shard_index < shard_count):
        raise ValueError("Use one to eight disjoint annotation shards")
    if plan.get("protocol") != PROTOCOL or arm not in {item["arm"] for item in plan["arms"]}:
        raise ValueError("Unknown public112 plan/arm")
    if any(not item["ready"] for item in plan["adapters"].values()):
        raise ValueError("Train and save both adapters before the matched public112 run")
    reader_url, embedding_url = _loopback(reader_url), _loopback(embedding_url)
    if urlparse(reader_url).path != "/v1" or urlparse(embedding_url).path:
        raise ValueError("Use reader URL ending /v1 and an embedding URL without a path")
    import requests
    session = requests.Session()
    session.trust_env = False
    health = {}
    for kind, url in (("reader", reader_url.removesuffix("/v1")), ("embedding", embedding_url)):
        response = session.get(url+"/health", timeout=30)
        response.raise_for_status()
        health[kind] = response.json()
        _check_health(plan, arm, kind, health[kind])
    output = Path(plan["output"])
    freeze_plan(plan, output/"plan.json")
    inference_class = _inference_class(plan["release_root"])
    done = []
    for source in plan["corpus"]["sources"][shard_index::shard_count]:
        if sha(source["annotation"]) != source["annotation_sha256"] or sha(source["clips_info"]) != source["clips_info_sha256"]:
            raise ValueError("A frozen public112 annotation or clip inventory changed")
        name = Path(source["annotation"]).stem
        arm_root = output/arm
        marker = arm_root/"completed"/(name+".json")
        marker.parent.mkdir(parents=True, exist_ok=True)
        binding = digest({"plan": plan, "arm": arm, "source": source})
        shared = output/"shared-base-cache"/name
        document_cache = shared/(Path(source["clips_info"]).stem+"_embeddings_video.npy")
        with marker.with_suffix(".lock").open("a+") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            if marker.exists():
                validated_completed_receipt(plan, arm, source)
                done.append(name)
                continue
            base_document_hash = None
            expected_base_artifacts = None
            if arm != "baseline":
                baseline_marker = output/"baseline"/"completed"/(name+".json")
                if not baseline_marker.is_file():
                    raise ValueError("Complete this annotation's baseline before another matched arm")
                baseline_receipt = validated_completed_receipt(plan, "baseline", source)
                base_document_hash = baseline_receipt.get("base_document_cache_sha256")
                expected_base_artifacts = baseline_receipt.get("shared_base_artifact_sha256")
                if (not base_document_hash or sha(document_cache) != base_document_hash
                        or not expected_base_artifacts or base_artifacts(shared) != expected_base_artifacts):
                    raise ValueError("The common base document cache changed after baseline")
            inference = inference_class(annotation_path=source["annotation"], clips_info_path=source["clips_info"],
                cache_dir=str(output/"shared-base-cache"), api_base_url=reader_url,
                model_path=plan["checkpoints"]["reader_base"]["path"], use_video_embedding=True,
                embedding_api_url=embedding_url, batch_size=4, output_dir=str(arm_root/"temporary"/name),
                clear_concept_db=False, num_neighbor=1, enable_rotation=True)
            if base_document_hash is not None and (sha(document_cache) != base_document_hash
                                                  or base_artifacts(shared) != expected_base_artifacts):
                raise ValueError("A matched arm recomputed the common base document cache")
            traces = observe_retrieval(inference)
            inference.process_annotation_file(annotation_path=source["annotation"],
                                               output_path=str(arm_root/"results"), top_k_clips=4)
            trace_file = arm_root/"traces"/(name+".json")
            write_json(trace_file, traces)
            files = [trace_file, arm_root/"results"/(name+"_result.json"),
                     arm_root/"results"/(name+"_evaluation.json")]
            write_json(marker, {"binding_sha256": binding, "services": health,
                                "base_document_cache_sha256": sha(document_cache),
                                "shared_base_artifact_sha256": base_artifacts(shared),
                                "output_sha256": {str(path): sha(path) for path in files}})
            done.append(name)
    return {"arm": arm, "shard_index": shard_index, "shard_count": shard_count,
            "completed_annotations_in_shard": len(done), "annotation_names": done}


def summarize(plan):
    result = {"protocol": PROTOCOL, "plan_sha256": digest(plan),
              "arms": [summarize_results(plan, item["arm"]) for item in plan["arms"]],
              "source_overlap_warning": "New work splits do not establish unseen-source benchmark generalization"}
    write_json(Path(plan["output"])/"summary.json", result)
    return result
