#!/bin/bash
# ============================================================
# Multi-GPU server startup script
# Launch one pair of vLLM + Embedding servers on each GPU
#
# Usage: bash scripts/start_multi_gpu_servers.sh [NUM_GPUS] [CUDA_VISIBLE_DEVICES]
#   NUM_GPUS: Number of GPUs to use (default: 8)
#   CUDA_VISIBLE_DEVICES: Comma-separated physical GPU ids, e.g. 0,1,2,3
#
# Port assignment:
#   GPU 0: vLLM -> 22003, Embedding -> 5000
#   GPU 1: vLLM -> 22004, Embedding -> 5001
#   GPU 2: vLLM -> 22005, Embedding -> 5002
#   ...
# ============================================================

set -e

# ============ Configurable Parameters ============
NUM_GPUS=${1:-8}
VISIBLE_GPUS_ARG=${2:-${CUDA_VISIBLE_DEVICES:-}}

VLLM_BASE_PORT=22003
EMBEDDING_BASE_PORT=5000

QWENVL_MODEL_PATH="models/Qwen3-VL-8B-Instruct"
LLAVA_MODEL_PATH="models/llava-onevision-qwen2-7b-ov-hf"

# Project root (PEARL)
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# Script paths relative to the project root
VLLM_SCRIPT_REL="server/qwenvl_flask_server.py"
# VLLM_SCRIPT_REL="server/llava_ov_flask_server.py"
EMBEDDING_SCRIPT_REL="server/embedding_server.py"
LOG_DIR_REL="server/logs"

VLLM_SCRIPT="${PROJECT_ROOT}/${VLLM_SCRIPT_REL}"
EMBEDDING_SCRIPT="${PROJECT_ROOT}/${EMBEDDING_SCRIPT_REL}"
LOG_DIR="${PROJECT_ROOT}/${LOG_DIR_REL}"

case "$(basename "${VLLM_SCRIPT_REL}")" in
    "qwenvl_flask_server.py")
        VLLM_MODEL_PATH="${QWENVL_MODEL_PATH}"
        ;;
    "llava_ov_flask_server.py")
        VLLM_MODEL_PATH="${LLAVA_MODEL_PATH}"
        ;;
    *)
        echo "Error: Unknown vLLM script ${VLLM_SCRIPT_REL}; unable to determine model_path"
        exit 1
        ;;
esac

# ====================================

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
echo "Multi-GPU server startup script"
echo "========================================"
echo "GPU count: ${NUM_GPUS}"
echo "CUDA_VISIBLE_DEVICES: ${VISIBLE_GPUS_ARG:-<default sequential mapping>}"
echo "vLLM base port: ${VLLM_BASE_PORT}"
echo "Embedding base port: ${EMBEDDING_BASE_PORT}"
echo "vLLM script: ${VLLM_SCRIPT_REL}"
echo "vLLM model path: ${VLLM_MODEL_PATH}"
echo "========================================"
echo ""

# Create the log directory
mkdir -p "${LOG_DIR}"

# Store PIDs for all background processes
PIDS=()

for ((gpu=0; gpu<NUM_GPUS; gpu++)); do
    PHYSICAL_GPU="${GPU_ID_MAP[$gpu]}"
    VLLM_PORT=$((VLLM_BASE_PORT + gpu))
    EMBEDDING_PORT=$((EMBEDDING_BASE_PORT + gpu))

    echo "-------- GPU ${gpu} (physical ${PHYSICAL_GPU}) --------"
    echo "  vLLM port: ${VLLM_PORT}"
    echo "  Embedding port: ${EMBEDDING_PORT}"

    # Start the vLLM server
    CUDA_VISIBLE_DEVICES=${PHYSICAL_GPU} python "${VLLM_SCRIPT}" \
        --model_path "${VLLM_MODEL_PATH}" \
        --port ${VLLM_PORT} \
        > "${LOG_DIR}/vllm_gpu${gpu}.log" 2>&1 &
    PIDS+=($!)
    echo "  vLLM server started (PID: $!, log: ${LOG_DIR}/vllm_gpu${gpu}.log)"

    # Start the Embedding server
    CUDA_VISIBLE_DEVICES=${PHYSICAL_GPU} python "${EMBEDDING_SCRIPT}" \
        --port ${EMBEDDING_PORT} \
        > "${LOG_DIR}/embedding_gpu${gpu}.log" 2>&1 &
    PIDS+=($!)
    echo "  Embedding server started (PID: $!, log: ${LOG_DIR}/embedding_gpu${gpu}.log)"

    echo ""
done

echo "========================================"
echo "All services have been started!"
echo "========================================"
echo ""
echo "Port summary:"
for ((gpu=0; gpu<NUM_GPUS; gpu++)); do
    PHYSICAL_GPU="${GPU_ID_MAP[$gpu]}"
    VLLM_PORT=$((VLLM_BASE_PORT + gpu))
    EMBEDDING_PORT=$((EMBEDDING_BASE_PORT + gpu))
    echo "  GPU ${gpu} (physical ${PHYSICAL_GPU}): vLLM=http://127.0.0.1:${VLLM_PORT}/v1  Embedding=http://127.0.0.1:${EMBEDDING_PORT}"
done
echo ""
echo "All background process PIDs: ${PIDS[*]}"
echo ""
echo "View logs:    tail -f ${LOG_DIR}/vllm_gpu0.log"
echo "Stop all:     kill ${PIDS[*]}"
echo "    Or:       pkill -f vllm_flask_server.py && pkill -f embedding_server.py"
echo ""

# Wait for all background processes (Ctrl+C stops everything)
trap "echo 'Stopping all services...'; kill ${PIDS[*]} 2>/dev/null; exit 0" SIGINT SIGTERM
wait
