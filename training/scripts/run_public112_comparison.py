#!/usr/bin/env python3
"""Manage isolated public112 services and the four fixed-checkpoint ablations."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

from pearl_training.io import sha256_file, write_json
from pearl_training.public112_adapter import QA_SUFFIX, make_plan, summarize_results
from pearl_training.public112_runner import _check_health, _inference_class, freeze_plan, summarize
from pearl_training.raw_tokens import capture_enabled, capture_mode, capture_report


REPO = Path(__file__).resolve().parents[1]
ARMS = ["baseline", "embedding_only", "reader_only", "joint"]


def stop_owned(processes):
    for process in processes:
        # An exited/reaped child's PID can already belong to another job.
        if process.poll() is not None:
            continue
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    for process in processes:
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--release-root", type=Path, required=True)
    parser.add_argument("--embedding-base", type=Path, required=True)
    parser.add_argument("--reader-base", type=Path, required=True)
    parser.add_argument("--ffmpeg-dir", type=Path, required=True)
    parser.add_argument("--lora-pdl-patch-receipt", type=Path)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    training = args.training_run.resolve(strict=True)
    output = args.output.resolve()
    release = args.release_root.resolve(strict=True)
    embedding = args.embedding_base.resolve(strict=True)
    reader = args.reader_base.resolve(strict=True)
    tools_dir = args.ffmpeg_dir.resolve(strict=True)
    tools = {}
    for name in ("ffmpeg", "ffprobe"):
        binary = tools_dir / name
        if not binary.is_file() or not os.access(binary, os.X_OK):
            raise FileNotFoundError(f"Missing executable prerequisite: {binary}")
        version = subprocess.run([str(binary), "-version"], text=True, capture_output=True, check=True)
        tools[name] = {"path": str(binary.resolve()), "version": version.stdout.splitlines()[0]}
    os.environ["PATH"] = str(tools_dir) + os.pathsep + os.environ.get("PATH", "")
    runtime_workarounds = []
    if args.lora_pdl_patch_receipt:
        receipt_path = args.lora_pdl_patch_receipt.resolve(strict=True)
        receipt = json.loads(receipt_path.read_text())
        if (receipt.get("upstream") != "https://github.com/vllm-project/vllm/pull/32836"
                or receipt.get("status") != "applied" or len(receipt.get("files", [])) != 2):
            raise ValueError("Invalid LoRA PDL backport receipt")
        for item in receipt["files"]:
            if sha256_file(item["installed_path"]) != item["after_sha256"]:
                raise ValueError("Installed LoRA PDL backport changed after receipt")
        os.environ["VLLM_LORA_DISABLE_PDL"] = "1"
        runtime_workarounds.append({"receipt": str(receipt_path),
            "receipt_sha256": sha256_file(receipt_path), "upstream": receipt["upstream"],
            "environment": {"VLLM_LORA_DISABLE_PDL": "1"},
            "scope": "LoRA kernel launch scheduling only; no-LoRA arms unaffected"})
    training_state = json.loads((training / "state.json").read_text())
    if training_state.get("status") not in {"training_complete", "paused_after_vlm_training"}:
        raise ValueError("Finish the fixed training run before public112 evaluation")
    for role, updates in (("retrieval", 2244), ("reader", 9140)):
        if training_state.get("qualified_groups") == 10000:
            if json.loads((training / role / "summary.json").read_text())["optimizer_updates"] != updates:
                raise ValueError("Both fixed 10k training phases must be complete before evaluation")
    plan = make_plan(release_root=release, embedding_base=embedding, reader_base=reader,
                     embedding_adapter=training / "retrieval", reader_adapter=training / "reader",
                     output=output, arms=ARMS)
    if not all(item["ready"] for item in plan["adapters"].values()):
        raise ValueError("Both trained adapters must be present")
    freeze_plan(plan, output / "plan.json")
    if not args.execute:
        print(json.dumps({"status": "plan_only", "model_calls": 0, "arms": ARMS,
                          "counts": plan["corpus"]["counts"], "plan": str(output / "plan.json")}))
        return

    import numpy as np
    import requests
    session = requests.Session()
    session.trust_env = False
    state_path = output / "execution_state.json"
    if state_path.exists():
        old = json.loads(state_path.read_text())
        archive = output / "execution_attempts" / (str(time.time_ns()) + ".json")
        write_json(archive, old)
    state = {"status": "running", "pid": os.getpid(), "training_run": str(training),
             "qualified_groups": training_state["qualified_groups"], "gpu_allocation": list(range(8)),
             "dynamic_placeholders": "kept_running", "arms": ARMS, "completed_arms": [],
             "raw_token_mode": capture_mode(),
             "driver_sha256": sha256_file(__file__),
             "media_tools": tools,
             "runtime_workarounds": runtime_workarounds,
             "runtime_environment": {key: os.environ[key] for key in
                 ("PYTHONPATH", "TRITON_PTXAS_PATH", "TRITON_CACHE_DIR", "VLLM_CACHE_ROOT", "TORCHINDUCTOR_CACHE_DIR",
                  "LD_PRELOAD", "PEARL_NVRTC_LIBRARY", "PEARL_RAW_TOKEN_MODE")
                 if key in os.environ},
             "started_utc": datetime.now(timezone.utc).isoformat()}

    def save():
        state["updated_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(state_path, state)

    def launch(command, gpu, log_path):
        log_path.parent.mkdir(parents=True, exist_ok=True)
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "HF_HUB_OFFLINE": "1",
               "TRANSFORMERS_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false",
               "OMP_NUM_THREADS": "2", "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
               "PEARL_RAW_TOKEN_DIR": str(output / "raw-token-traces" / state["arm"] / f"gpu{gpu}")}
        with log_path.open("ab", buffering=0) as stream:
            process = subprocess.Popen(command, cwd=REPO, env=env, stdout=stream,
                                       stderr=subprocess.STDOUT, start_new_session=True)
        try:
            state.setdefault("processes", []).append({"pid": process.pid, "gpu": gpu,
                                                       "command": command, "log": str(log_path)})
            save()
        except BaseException:
            stop_owned([process])
            raise
        return process

    def start_pair(arm, gpu, processes):
        for kind, model, port in (("embedding", embedding, 28780 + gpu), ("reader", reader, 28880 + gpu)):
            command = [sys.executable, "-u", str(REPO / "scripts/serve_public112_adapters.py"),
                       "--kind", kind, "--arm", arm, "--model", str(model),
                       "--adapter", str(training / ("retrieval" if kind == "embedding" else "reader")),
                       "--release-root", str(release), "--port", str(port), "--execute"]
            if kind == "reader":
                command += ["--helper-cache", str(output / "shared-base-helper-cache")]
            processes.append(launch(command, gpu, output / "services" / arm / f"gpu{gpu}-{kind}.log"))

    def wait_ready(arm, gpus, processes):
        pending = {(gpu, kind) for gpu in gpus for kind in ("embedding", "reader")}
        deadline = time.monotonic() + 1800
        while pending:
            if any(process.poll() is not None for process in processes):
                raise RuntimeError("A public112 service exited during startup; inspect its recorded log")
            for gpu, kind in list(pending):
                port = (28780 if kind == "embedding" else 28880) + gpu
                try:
                    response = session.get(f"http://127.0.0.1:{port}/health", timeout=2)
                    response.raise_for_status()
                except requests.RequestException:
                    continue
                _check_health(plan, arm, kind, response.json())
                pending.remove((gpu, kind))
            state["waiting_for_services"] = [list(item) for item in sorted(pending)]
            save()
            if pending and time.monotonic() > deadline:
                raise TimeoutError("Public112 service startup timed out")
            if pending:
                time.sleep(2)

    def smoke(arm):
        config = json.loads((training / "reader/run_config.json").read_text())
        with Path(config["samples"]).open() as stream:
            sample = next(row for row in map(json.loads, stream)
                          if row["split"] == "train" and row["task"] == "current")
        source = plan["corpus"]["sources"][0]
        first_clip = json.loads(Path(source["clips_info"]).read_text())["clips"][0]
        clip = Path(first_clip["clip_path"])
        clip = clip.resolve() if clip.is_absolute() else (release / clip).resolve()
        if not clip.is_file():
            raise FileNotFoundError(clip)
        vectors = {}
        for kind, item in (("query", {"text": "a person holding an object"}), ("document", {"video": str(clip)})):
            response = session.post("http://127.0.0.1:28780/get_embeddings",
                                    json={"inputs": [item]}, timeout=180)
            response.raise_for_status()
            vector = np.asarray(response.json()["embeddings"], dtype=np.float32)
            if vector.ndim != 2 or vector.shape[0] != 1 or not np.isfinite(vector).all():
                raise ValueError("Invalid embedding smoke output")
            norm = float(np.linalg.norm(vector[0]))
            if abs(norm - 1.0) > 1e-3:
                raise ValueError("Embedding smoke output is not normalized")
            vectors[kind] = {"shape": list(vector.shape), "norm": norm}
        parts = []
        for concept in sample["concepts"]:
            parts += [{"type": "image_url", "image_url": {"url": Path(concept["image"]).resolve().as_uri()}},
                      {"type": "text", "text": f"Runtime concept {concept['name']}: {concept['description']}"}]
        parts.extend({"type": "image_url", "image_url": {"url": Path(frame).resolve().as_uri()}}
                     for frame in sample["current_clip"]["frames"])
        options = "\n".join(f"{chr(65 + i)}. {text}" for i, text in enumerate(sample["options"]))
        parts.append({"type": "text", "text": sample["question"] + "\n" + options + "\n" + QA_SUFFIX})
        response = session.post("http://127.0.0.1:28880/v1/chat/completions", json={
            "model": str(reader), "messages": [{"role": "user", "content": parts}],
            "max_tokens": 1024, "temperature": 0}, timeout=300)
        response.raise_for_status()
        body = response.json()
        route = body.get("pearl_adapter_route", {})
        if (route.get("stage") != "qa" or route.get("adapter_enabled") != (arm in {"reader_only", "joint"})
                or not body["choices"][0]["message"]["content"].strip()):
            raise ValueError("Final-QA adapter smoke route failed")
        write_json(output / "services" / arm / "smoke.json", {
            "purpose": "functional_loading_only_excluded_from_scores", "vectors": vectors,
            "reader_route": route, "response": body, "sample_id": sample["sample_id"]})

    def registration_smoke():
        source = plan["corpus"]["sources"][0]
        name = Path(source["annotation"]).stem
        inference = _inference_class(release)(
            annotation_path=source["annotation"], clips_info_path=source["clips_info"],
            cache_dir=str(output / "shared-base-cache"), api_base_url="http://127.0.0.1:28880/v1",
            model_path=str(reader), use_video_embedding=True,
            embedding_api_url="http://127.0.0.1:28780", batch_size=4,
            output_dir=str(output / "baseline" / "temporary" / name),
            clear_concept_db=False, num_neighbor=1, enable_rotation=True)
        write_json(output / "services" / "baseline" / "registration_smoke.json", {
            "purpose": "original_initialization_only_excluded_from_scores",
            "annotation": source["annotation"], "media_tools": tools, "status": "complete"})
        del inference

    save()
    def interrupted(signum, frame):
        raise SystemExit(128 + signum)
    signal.signal(signal.SIGTERM, interrupted)
    try:
        for arm in ARMS:
            if all((output / arm / "completed" / (Path(source["annotation"]).stem + ".json")).exists()
                   for source in plan["corpus"]["sources"]):
                summarize_results(plan, arm)
                state["completed_arms"].append(arm)
                save()
                continue
            services, workers = [], []
            try:
                state.update(phase="starting_first_pair", arm=arm)
                save()
                start_pair(arm, 0, services)
                wait_ready(arm, [0], services)
                state["phase"] = "functional_smoke"
                save()
                smoke(arm)
                if arm == "baseline":
                    state["phase"] = "original_registration_smoke"
                    save()
                    registration_smoke()
                state["phase"] = "starting_remaining_pairs"
                save()
                for gpu in range(1, 8):
                    start_pair(arm, gpu, services)
                wait_ready(arm, range(8), services)
                state["phase"] = "evaluating"
                save()
                for gpu in range(8):
                    command = [sys.executable, "-u", str(REPO / "scripts/evaluate_public112_adapters.py"),
                               "--release-root", str(release), "--embedding-base", str(embedding),
                               "--reader-base", str(reader), "--embedding-adapter", str(training / "retrieval"),
                               "--reader-adapter", str(training / "reader"), "--output", str(output),
                               "--arms", *ARMS, "--arm", arm, "--reader-url", f"http://127.0.0.1:{28880 + gpu}/v1",
                               "--embedding-url", f"http://127.0.0.1:{28780 + gpu}",
                               "--shard-index", str(gpu), "--shard-count", "8", "--execute"]
                    workers.append(launch(command, gpu, output / arm / "worker_logs" / f"gpu{gpu}.log"))
                while any(process.poll() is None for process in workers):
                    if any(process.poll() not in (None, 0) for process in workers):
                        raise RuntimeError("A public112 shard failed; preserve its log and completed receipts")
                    if any(process.poll() is not None for process in services):
                        raise RuntimeError("A public112 service exited during evaluation")
                    state["completed_annotations"] = len(list((output / arm / "completed").glob("*.json")))
                    save()
                    time.sleep(5)
                if any(process.returncode != 0 for process in workers):
                    raise RuntimeError("A public112 shard failed at completion")
                state["arm_result"] = summarize_results(plan, arm)
                token_report = capture_report(output / "raw-token-traces" / arm)
                write_json(output / arm / "raw-token-manifest.json", token_report)
                if capture_enabled() and not token_report["complete"]:
                    raise RuntimeError("Raw token capture is incomplete; do not publish this evaluation as complete")
                state.setdefault("raw_token_manifests", {})[arm] = str(output / arm / "raw-token-manifest.json")
                state["completed_arms"].append(arm)
                save()
            finally:
                stop_owned(workers)
                stop_owned(services)
        result = summarize(plan)
        state.update(status="complete", phase="complete", summary=str(output / "summary.json"))
        save()
        print(json.dumps(result, ensure_ascii=False), flush=True)
    except BaseException as error:
        state.update(status="needs_review", error={"type": type(error).__name__, "message": str(error)})
        save()
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
