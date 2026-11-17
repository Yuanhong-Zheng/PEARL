#!/bin/bash
# ============================================================
# Multi-GPU servers for PEARL with the trained LoRA adapters
# (Qwen3-VL-8B+PEARL trained on PSVU-10K)
#
# Launches, on each GPU, the same two services as start_multi_gpu_servers.sh and on
# the same ports, so scripts/eval_qwen.sh runs unchanged against them:
#   - reader (vLLM, Qwen3-VL-8B-Instruct): the answering-VLM adapter is applied only to
#     the final question answering; concept descriptions and query rewrites use the
#     base model, as at training time;
#   - embedding (Qwen3-VL-Embedding-2B): the retriever adapter is applied only to text
#     queries; clip embeddings use the frozen base encoder.
# The services are training/scripts/serve_public112_adapters.py, also used by the
# training-side comparison (training/README.md). They serve concepts defined by a
# reference frame (the frame-level split).
#
# Usage: bash scripts/start_lora_servers.sh [NUM_GPUS] [CUDA_VISIBLE_DEVICES]
#   NUM_GPUS: Number of GPUs to use (default: 8)
#   CUDA_VISIBLE_DEVICES: Comma-separated physical GPU ids, e.g. 0,1,2,3
#
# Environment variables:
#   READER_ADAPTER: answering-VLM adapter (default: models/PEARL-Qwen3-VL-8B-LoRA)
#   EMBEDDING_ADAPTER: retriever adapter (default: models/PEARL-Retriever-Qwen3-VL-Embedding-2B-LoRA)
#   ARM: joint (default, both adapters), reader_only, embedding_only or baseline
#   HELPER_CACHE: cache of base-model concept descriptions and rewrites (default: .cache/lora_helper_cache)
#
# Requires: pip install -r requirements-train.txt (peft)
# ============================================================

set -e

NUM_GPUS=${1:-8}
VISIBLE_GPUS_ARG=${2:-${CUDA_VISIBLE_DEVICES:-}}

VLLM_BASE_PORT=22003
EMBEDDING_BASE_PORT=5000

# Project root (PEARL)
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# video_qa_inference.py names models/Qwen3-VL-8B-Instruct in its requests, and the
# reader service only accepts requests for its own base model.
READER_MODEL="${PROJECT_ROOT}/models/Qwen3-VL-8B-Instruct"
EMBEDDING_MODEL="${PROJECT_ROOT}/models/Qwen3-VL-Embedding-2B"
READER_ADAPTER=${READER_ADAPTER:-"${PROJECT_ROOT}/models/PEARL-Qwen3-VL-8B-LoRA"}
EMBEDDING_ADAPTER=${EMBEDDING_ADAPTER:-"${PROJECT_ROOT}/models/PEARL-Retriever-Qwen3-VL-Embedding-2B-LoRA"}
ARM=${ARM:-joint}
HELPER_CACHE=${HELPER_CACHE:-"${PROJECT_ROOT}/.cache/lora_helper_cache"}
SERVE_SCRIPT="${PROJECT_ROOT}/training/scripts/serve_public112_adapters.py"
LOG_DIR="${PROJECT_ROOT}/server/logs"

# The pearl_training package, without requiring pip install -e training
export PYTHONPATH="${PROJECT_ROOT}/training/src${PYTHONPATH:+:${PYTHONPATH}}"
# Per-token inference records are only needed for the training-side audit
export PEARL_RAW_TOKEN_MODE=${PEARL_RAW_TOKEN_MODE:-disabled_by_user}

for path in "${READER_MODEL}" "${EMBEDDING_MODEL}"; do
    if [ ! -d "${path}" ]; then
        echo "Error: base model not found: ${path}"
        exit 1
    fi
done
case "${ARM}" in
    joint) NEED_ADAPTERS=("${READER_ADAPTER}" "${EMBEDDING_ADAPTER}") ;;
    reader_only) NEED_ADAPTERS=("${READER_ADAPTER}") ;;
    embedding_only) NEED_ADAPTERS=("${EMBEDDING_ADAPTER}") ;;
    baseline) NEED_ADAPTERS=() ;;
    *) echo "Error: unknown ARM ${ARM}"; exit 1 ;;
esac
for path in "${NEED_ADAPTERS[@]}"; do
    if [ ! -f "${path}/adapter_config.json" ]; then
        echo "Error: LoRA adapter not found: ${path}"
        exit 1
    fi
done

GPU_ID_MAP=()
if [ -n "${VISIBLE_GPUS_ARG}" ]; then
    IFS=',' read -r -a GPU_ID_MAP <<< "${VISIBLE_GPUS_ARG}"
    if [ "${NUM_GPUS}" -gt "${#GPU_ID_MAP[@]}" ]; then
        echo "Error: NUM_GPUS (${NUM_GPUS}) exceeds the number of visible GPUs provided (${#GPU_ID_MAP[@]})"
        exit 1
    fi
else
    for ((gpu=0; gpu<NUM_GPUS; gpu++)); do
        GPU_ID_MAP+=("${gpu}")
    done
fi

echo "========================================"
echo "PEARL LoRA server startup"
echo "========================================"
echo "GPU count: ${NUM_GPUS}"
echo "Arm: ${ARM}"
echo "Reader: ${READER_MODEL} + ${READER_ADAPTER}"
echo "Embedding: ${EMBEDDING_MODEL} + ${EMBEDDING_ADAPTER}"
echo "Helper cache: ${HELPER_CACHE}"
echo "========================================"
echo ""

mkdir -p "${LOG_DIR}" "${HELPER_CACHE}"
PIDS=()

for ((gpu=0; gpu<NUM_GPUS; gpu++)); do
    PHYSICAL_GPU="${GPU_ID_MAP[$gpu]}"
    VLLM_PORT=$((VLLM_BASE_PORT + gpu))
    EMBEDDING_PORT=$((EMBEDDING_BASE_PORT + gpu))

    echo "-------- GPU ${gpu} (physical ${PHYSICAL_GPU}) --------"

    CUDA_VISIBLE_DEVICES=${PHYSICAL_GPU} python "${SERVE_SCRIPT}" \
        --kind reader --arm "${ARM}" --model "${READER_MODEL}" --adapter "${READER_ADAPTER}" \
        --release-root "${PROJECT_ROOT}" --port ${VLLM_PORT} --helper-cache "${HELPER_CACHE}" --execute \
        > "${LOG_DIR}/lora_reader_gpu${gpu}.log" 2>&1 &
    PIDS+=($!)
    echo "  Reader started on port ${VLLM_PORT} (PID: $!, log: ${LOG_DIR}/lora_reader_gpu${gpu}.log)"

    CUDA_VISIBLE_DEVICES=${PHYSICAL_GPU} python "${SERVE_SCRIPT}" \
        --kind embedding --arm "${ARM}" --model "${EMBEDDING_MODEL}" --adapter "${EMBEDDING_ADAPTER}" \
        --release-root "${PROJECT_ROOT}" --port ${EMBEDDING_PORT} --execute \
        > "${LOG_DIR}/lora_embedding_gpu${gpu}.log" 2>&1 &
    PIDS+=($!)
    echo "  Embedding started on port ${EMBEDDING_PORT} (PID: $!, log: ${LOG_DIR}/lora_embedding_gpu${gpu}.log)"
    echo ""
done

echo "All services have been started. Next: RUN_NAME=qwen3vl_lora_k4_n1_fps1 bash scripts/eval_qwen.sh ${NUM_GPUS}"
echo "Stop all: kill ${PIDS[*]}"
echo ""

trap "echo 'Stopping all services...'; kill ${PIDS[*]} 2>/dev/null; exit 0" SIGINT SIGTERM
wait
