#!/bin/bash
# ============================================================
# Single-GPU debug evaluation script (Qwen)
# Usage: bash scripts/eval_qwen_debug.sh [GPU_ID] [MAX_FILES]
#   GPU_ID: GPU index to use (default: 0)
#   MAX_FILES: Maximum number of files to process (default: 0, meaning all)
# ============================================================

set -e

# ============ Configurable Parameters ============
GPU_ID=${1:-0}
MAX_FILES=${2:-0}

# Project root (PEARL)
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

ANNOTATION_DIR="${PROJECT_ROOT}/data/frame-level/annotations_short"
CLIPS_BASE_DIR="${PROJECT_ROOT}/data/frame-level/output_clips"
CACHE_DIR="${PROJECT_ROOT}/.cache"
OUTPUT_DIR="${PROJECT_ROOT}/output_results/renamed/debug_qwen_short_clip"
PYTHON_SCRIPT="${PROJECT_ROOT}/video_qa_inference.py"

VLLM_BASE_PORT=22003
EMBEDDING_BASE_PORT=5000

# Toggle: whether to skip existing output files (true/false)
SKIP_EXISTING=true
# ====================================

VLLM_PORT=$((VLLM_BASE_PORT + GPU_ID))
EMBEDDING_PORT=$((EMBEDDING_BASE_PORT + GPU_ID))
API_URL="http://127.0.0.1:${VLLM_PORT}/v1"
EMBEDDING_URL="http://127.0.0.1:${EMBEDDING_PORT}"

echo "========================================"
echo "Single-GPU debug evaluation script"
echo "========================================"
echo "GPU ID: ${GPU_ID}"
echo "vLLM API: ${API_URL}"
echo "Embedding API: ${EMBEDDING_URL}"
echo "Annotation directory: ${ANNOTATION_DIR}"
echo "Clips base directory: ${CLIPS_BASE_DIR}"
echo "Output directory: ${OUTPUT_DIR}"
echo "Maximum files to process: ${MAX_FILES} (0 means all)"
echo "Skip existing outputs: ${SKIP_EXISTING}"
echo "========================================"
echo ""

if [ ! -d "${ANNOTATION_DIR}" ]; then
    echo "Error: Annotation directory does not exist: ${ANNOTATION_DIR}"
    exit 1
fi

if [ ! -f "${PYTHON_SCRIPT}" ]; then
    echo "Error: Inference script does not exist: ${PYTHON_SCRIPT}"
    exit 1
fi

if [ ! -d "${CLIPS_BASE_DIR}" ]; then
    echo "Error: Clips base directory does not exist: ${CLIPS_BASE_DIR}"
    exit 1
fi

# Ensure the output directory exists
mkdir -p "${OUTPUT_DIR}"

# ============ Wait For The Single-GPU Servers ============
echo "Checking single-GPU service status..."
MAX_WAIT=600
WAIT_INTERVAL=10

echo -n "  Checking vLLM (port ${VLLM_PORT})..."
elapsed=0
while true; do
    if curl -s "http://127.0.0.1:${VLLM_PORT}/health" > /dev/null 2>&1; then
        echo " ✓ ready"
        break
    fi
    elapsed=$((elapsed + WAIT_INTERVAL))
    if [ $elapsed -ge $MAX_WAIT ]; then
        echo " ✗ timeout (${MAX_WAIT}s)"
        echo "Error: vLLM service is not ready"
        exit 1
    fi
    sleep ${WAIT_INTERVAL}
done

echo -n "  Checking Embedding (port ${EMBEDDING_PORT})..."
elapsed=0
while true; do
    if curl -s "http://127.0.0.1:${EMBEDDING_PORT}/health" > /dev/null 2>&1; then
        echo " ✓ ready"
        break
    fi
    elapsed=$((elapsed + WAIT_INTERVAL))
    if [ $elapsed -ge $MAX_WAIT ]; then
        echo " ✗ timeout (${MAX_WAIT}s)"
        echo "Error: Embedding service is not ready"
        exit 1
    fi
    sleep ${WAIT_INTERVAL}
done
echo ""

# ============ Collect JSON Files ============
JSON_FILES=()
for json_file in "${ANNOTATION_DIR}"/*.json; do
    if [ -f "${json_file}" ]; then
        if [ "${SKIP_EXISTING}" = true ]; then
            filename=$(basename "${json_file}")
            base_name="${filename%.json}"
            output_file="${OUTPUT_DIR}/${base_name}_evaluation.json"
            if [ -f "${output_file}" ]; then
                echo "⊘ Skipping existing file: ${filename}"
                continue
            fi
        fi
        JSON_FILES+=("${json_file}")
    fi
done

TOTAL_FILES=${#JSON_FILES[@]}
if [ ${TOTAL_FILES} -eq 0 ]; then
    echo "No files need processing (all skipped or directory is empty)"
    exit 0
fi

if [ ${MAX_FILES} -gt 0 ] && [ ${MAX_FILES} -lt ${TOTAL_FILES} ]; then
    TOTAL_FILES=${MAX_FILES}
fi

echo "Actual number of files to process: ${TOTAL_FILES}"
echo ""

success_count=0
fail_count=0

for ((i=0; i<TOTAL_FILES; i++)); do
    json_file="${JSON_FILES[$i]}"
    filename=$(basename "${json_file}")
    echo "[DEBUG][GPU ${GPU_ID}] Processing ($((i + 1))/${TOTAL_FILES}): ${filename}"

    if python "${PYTHON_SCRIPT}" \
        --annotation_path "${json_file}" \
        --clips_base_dir "${CLIPS_BASE_DIR}" \
        --cache_dir "${CACHE_DIR}" \
        --output_path "${OUTPUT_DIR}" \
        --api_base_url "${API_URL}" \
        --embedding_api_url "${EMBEDDING_URL}" \
        --gpu_id "${GPU_ID}" \
        --enable_rotation; then
        echo "[DEBUG][GPU ${GPU_ID}] ✓ Success: ${filename}"
        success_count=$((success_count + 1))
    else
        echo "[DEBUG][GPU ${GPU_ID}] ✗ Failed: ${filename}"
        fail_count=$((fail_count + 1))
        echo "[DEBUG][GPU ${GPU_ID}] Abnormal exit detected, stopping immediately"
        exit 1
    fi
done

echo ""
echo "========================================"
echo "[DEBUG] Finished"
echo "Success: ${success_count}, Failed: ${fail_count}"
echo "Output directory: ${OUTPUT_DIR}"
echo "========================================"
echo ""
