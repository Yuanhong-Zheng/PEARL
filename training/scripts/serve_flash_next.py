#!/usr/bin/env python3
"""Wait for the approved checkpoint, then serve one eight-GPU local teacher."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--state-dir", required=True, type=Path)
    parser.add_argument("--environment", required=True, type=Path)
    parser.add_argument("--port", type=int, default=28790)
    args = parser.parse_args()
    args.state_dir.mkdir(parents=True, exist_ok=True)
    state_path = args.state_dir / "service_state.json"

    def save(value):
        value.update({"pid": os.getpid(), "hostname": socket.gethostname(),
                      "updated_utc": datetime.now(timezone.utc).isoformat()})
        temporary = state_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
        temporary.replace(state_path)

    metadata = json.loads((args.state_dir / "modelscope_files.json").read_text())
    expected = [f for f in metadata["Data"]["Files"]
                if f.get("Type") == "blob" and f.get("Name") not in {".gitignore", ".gitattributes"}]
    while True:
        missing = [f["Path"] for f in expected
                   if not (args.model / f["Path"]).is_file()
                   or (args.model / f["Path"]).stat().st_size != f["Size"]]
        if not missing:
            break
        download = json.loads((args.state_dir / "download_state.json").read_text())
        if download["status"] == "needs_review":
            raise RuntimeError("Checkpoint download needs review: " + str(download.get("error")))
        save({"status": "waiting_for_checkpoint", "missing_files": len(missing)})
        time.sleep(15)
    with socket.socket() as connection:
        connection.bind(("127.0.0.1", args.port))
    config = json.loads((args.model / "config.json").read_text())
    if config.get("architectures") != ["Qwen4ExpForConditionalGeneration"]:
        raise ValueError("Checkpoint architecture differs from the requested Flash-Next model")
    command = [str(args.environment / "bin/vllm"), "serve", str(args.model),
        "--host", "127.0.0.1", "--port", str(args.port), "--served-model-name", "Qwen3.8-Flash-Next",
        "--tensor-parallel-size", "8", "--enable-expert-parallel", "--dtype", "bfloat16",
        "--max-model-len", "262144", "--max-num-seqs", "256", "--max-num-batched-tokens", "32768",
        "--gpu-memory-utilization", "0.85", "--enable-prefix-caching", "--api-server-count", "4",
        "--reasoning-parser", "qwen3", "--limit-mm-per-prompt", '{"image":256,"video":0}',
        "--mm-processor-kwargs", '{"max_pixels":262144}']
    cache = Path(__file__).resolve().parents[2] / "cache/flash_next" / socket.gethostname()
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7", "OMP_NUM_THREADS": "4",
           "TOKENIZERS_PARALLELISM": "false", "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
           "VLLM_USE_RUST_FRONTEND": "0", "VLLM_ENGINE_READY_TIMEOUT_S": "7200",
           "LD_LIBRARY_PATH": str(args.environment / "lib/python3.12/site-packages/torch/lib")
               + ":/usr/local/cuda/compat/lib:/usr/local/nvidia/lib",
           "PATH": str(args.environment / "bin") + ":" + os.environ.get("PATH", ""),
           "HF_HUB_OFFLINE": "1", "VLLM_CACHE_ROOT": str(cache / "vllm"),
           "TORCHINDUCTOR_CACHE_DIR": str(cache / "torchinductor"), "TRITON_HOME": str(cache / "triton")}
    # Runtime kernel downloads are overseas resources. Override inherited proxy
    # variants consistently and keep local inference/collectives off the proxy.
    for key in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        env[key] = "http://your-proxy-host:8080"
    for key in ("no_proxy", "NO_PROXY"):
        env[key] = "localhost,127.0.0.1,localaddress,localdomain.com,internal"
    env.pop("TRITON_CACHE_DIR", None)
    # Populate the shared MoE cache once before eight workers prepare artifact
    # headers concurrently. Preserve FlashInfer/Ninja's full dependency checks;
    # never force-load a stale binary or change kernel math to bypass startup.
    warm_command = [str(args.environment / "bin/python"), "-u", "-c",
        "from flashinfer.jit.fused_moe import gen_trtllm_gen_fused_moe_sm100_module; "
        "gen_trtllm_gen_fused_moe_sm100_module().build_and_load(); "
        "print('FLASHINFER_CACHE_PREFLIGHT_COMPLETE', flush=True)"]
    save({"status": "warming_kernel_cache", "command": warm_command})
    subprocess.run(warm_command, env=env, check=True)
    child = subprocess.Popen(command, env=env, start_new_session=True)
    save({"status": "starting", "server_pid": child.pid, "command": command,
          "model": str(args.model), "allocated_gpus": list(range(8)), "dynamic_placeholders": "kept_running"})

    def stop(signum, frame):
        os.killpg(child.pid, signal.SIGTERM)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    result = child.wait()
    save({"status": "exited" if result == 0 else "needs_review", "server_pid": child.pid,
          "exit_code": result, "command": command})
    raise SystemExit(result)


if __name__ == "__main__":
    main()
