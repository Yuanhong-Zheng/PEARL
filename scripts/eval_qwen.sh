#!/bin/bash
# ============================================================
# Multi-GPU parallel evaluation script
# Evenly distribute annotation files across multiple GPUs
#
# Usage: bash scripts/eval_qwen.sh [NUM_GPUS] [TOP_K_CLIPS] [NUM_NEIGHBOR] [CUDA_VISIBLE_DEVICES]
#   NUM_GPUS: Number of GPUs to use (default: 8, should match start_multi_gpu_servers.sh)
#   TOP_K_CLIPS: Number of retrieved clips (default: 4)
#   NUM_NEIGHBOR: Number of neighboring clips on each side (default: 1)
#   CUDA_VISIBLE_DEVICES: Comma-separated physical GPU ids, e.g. 0,1,2,3
#
# Prerequisite: the corresponding servers have already been started with start_multi_gpu_servers.sh
# ============================================================

set -e

# ============ Configurable Parameters ============
NUM_GPUS=${1:-8}
TOP_K_CLIPS=${2:-4}
NUM_NEIGHBOR=${3:-1}
VISIBLE_GPUS_ARG=${4:-${CUDA_VISIBLE_DEVICES:-}}

# Project root (PEARL)
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

ANNOTATION_DIR="${PROJECT_ROOT}/data/frame-level/annotations"
CLIPS_BASE_DIR="${PROJECT_ROOT}/data/frame-level/output_clips"
CACHE_DIR="${PROJECT_ROOT}/.cache"
OUTPUT_DIR="${PROJECT_ROOT}/output_results/frame-level/qwen3vl_k4_n1_fps1"
PYTHON_SCRIPT="${PROJECT_ROOT}/video_qa_inference.py"

VLLM_BASE_PORT=22003
EMBEDDING_BASE_PORT=5000

# Toggle: whether to skip existing output files (true/false)
SKIP_EXISTING=true
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
echo "Multi-GPU parallel evaluation script"
echo "========================================"
echo "GPU count: ${NUM_GPUS}"
echo "CUDA_VISIBLE_DEVICES: ${VISIBLE_GPUS_ARG:-<default sequential mapping>}"
echo "Top-K clips: ${TOP_K_CLIPS}"
echo "Num neighbor: ${NUM_NEIGHBOR}"
echo "Annotation directory: ${ANNOTATION_DIR}"
echo "Clips base directory: ${CLIPS_BASE_DIR}"
echo "Output directory: ${OUTPUT_DIR}"
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

# ============ Wait For All Servers ============
echo "Checking server status..."
MAX_WAIT=600  # Wait up to 600 seconds
WAIT_INTERVAL=10

for ((gpu=0; gpu<NUM_GPUS; gpu++)); do
    VLLM_PORT=$((VLLM_BASE_PORT + gpu))
    EMBEDDING_PORT=$((EMBEDDING_BASE_PORT + gpu))

    echo -n "  Checking GPU ${gpu} vLLM (port ${VLLM_PORT})..."
    elapsed=0
    while true; do
        if curl -s "http://127.0.0.1:${VLLM_PORT}/health" > /dev/null 2>&1; then
            echo " ✓ ready"
            break
        fi
        elapsed=$((elapsed + WAIT_INTERVAL))
        if [ $elapsed -ge $MAX_WAIT ]; then
            echo " ✗ timeout (${MAX_WAIT}s)"
            echo "Error: vLLM service for GPU ${gpu} is not ready; please verify it has been started"
            exit 1
        fi
        sleep ${WAIT_INTERVAL}
    done

    echo -n "  Checking GPU ${gpu} Embedding (port ${EMBEDDING_PORT})..."
    elapsed=0
    while true; do
        if curl -s "http://127.0.0.1:${EMBEDDING_PORT}/health" > /dev/null 2>&1; then
            echo " ✓ ready"
            break
        fi
        elapsed=$((elapsed + WAIT_INTERVAL))
        if [ $elapsed -ge $MAX_WAIT ]; then
            echo " ✗ timeout (${MAX_WAIT}s)"
            echo "Error: Embedding service for GPU ${gpu} is not ready; please verify it has been started"
            exit 1
        fi
        sleep ${WAIT_INTERVAL}
    done
done
echo ""
echo "✓ All servers are ready!"
echo ""

# ============ Collect All JSON Files ============
JSON_FILES=()
for json_file in "${ANNOTATION_DIR}"/*.json; do
    if [ -f "$json_file" ]; then
        # If skipping is enabled, check whether the output file already exists
        if [ "$SKIP_EXISTING" = true ]; then
            filename=$(basename "$json_file")
            base_name="${filename%.json}"
            output_file="${OUTPUT_DIR}/${base_name}_evaluation.json"
            if [ -f "$output_file" ]; then
                echo "⊘ Skipping existing file: ${filename}"
                continue
            fi
        fi
        JSON_FILES+=("$json_file")
    fi
done

TOTAL_FILES=${#JSON_FILES[@]}

if [ $TOTAL_FILES -eq 0 ]; then
    echo ""
    echo "No files need processing (all skipped or directory is empty)"
    exit 0
fi

echo ""
echo "Files to process: ${TOTAL_FILES}"
echo "GPU count: ${NUM_GPUS}"
echo ""

# ============ Assign Files To GPUs ============
# Number of files per GPU (rounded up)
FILES_PER_GPU=$(( (TOTAL_FILES + NUM_GPUS - 1) / NUM_GPUS ))

echo "Each GPU will process about ${FILES_PER_GPU} file(s)"
echo ""

# Ensure the output directory exists
mkdir -p "${OUTPUT_DIR}"

# ============ Define The Single-GPU Worker ============
process_gpu() {
    local gpu_id=$1
    local start_idx=$2
    local end_idx=$3
    local physical_gpu="${GPU_ID_MAP[$gpu_id]}"
    local vllm_port=$((VLLM_BASE_PORT + gpu_id))
    local embedding_port=$((EMBEDDING_BASE_PORT + gpu_id))
    local api_url="http://127.0.0.1:${vllm_port}/v1"
    local embedding_url="http://127.0.0.1:${embedding_port}"

    local success_count=0
    local fail_count=0

    echo "[GPU ${gpu_id}] Processing files ${start_idx} ~ $((end_idx - 1)), total $((end_idx - start_idx))"
    echo "[GPU ${gpu_id}] Physical GPU: ${physical_gpu}"
    echo "[GPU ${gpu_id}] vLLM API: ${api_url}"
    echo "[GPU ${gpu_id}] Embedding API: ${embedding_url}"

    for ((i=start_idx; i<end_idx; i++)); do
        local json_file="${JSON_FILES[$i]}"
        local filename=$(basename "$json_file")

        echo "[GPU ${gpu_id}] Processing ($((i - start_idx + 1))/$((end_idx - start_idx))): ${filename}"

        CUDA_VISIBLE_DEVICES=${physical_gpu} python "$PYTHON_SCRIPT" \
            --annotation_path "$json_file" \
            --clips_base_dir "$CLIPS_BASE_DIR" \
            --cache_dir "$CACHE_DIR" \
            --output_path "$OUTPUT_DIR" \
            --top_k_clips "$TOP_K_CLIPS" \
            --num_neighbor "$NUM_NEIGHBOR" \
            --api_base_url "$api_url" \
            --embedding_api_url "$embedding_url" \
            --gpu_id "$gpu_id" \
            --enable_rotation
            

        EXIT_CODE=$?
        if [ $EXIT_CODE -eq 0 ]; then
            echo "[GPU ${gpu_id}] ✓ Success: ${filename}"
            success_count=$((success_count + 1))
        else
            echo "[GPU ${gpu_id}] ✗ Failed: ${filename} (exit code: ${EXIT_CODE})"
            fail_count=$((fail_count + 1))
            # If a file fails, you can choose to continue or stop
            # Uncomment the next line to stop this GPU worker on failure
            # return 1
        fi
    done

    echo ""
    echo "[GPU ${gpu_id}] ======== Finished ========"
    echo "[GPU ${gpu_id}] Success: ${success_count}, Failed: ${fail_count}"
    echo ""

    if [ $fail_count -gt 0 ]; then
        return 1
    fi
    return 0
}

# ============ Start Multi-GPU Parallel Processing ============
echo "========================================"
echo "Starting multi-GPU parallel processing..."
echo "========================================"
echo ""

GPU_PIDS=()

for ((gpu=0; gpu<NUM_GPUS; gpu++)); do
    START_IDX=$((gpu * FILES_PER_GPU))
    END_IDX=$((START_IDX + FILES_PER_GPU))

    # Make sure we do not exceed the total number of files
    if [ $END_IDX -gt $TOTAL_FILES ]; then
        END_IDX=$TOTAL_FILES
    fi

    # Skip GPUs that did not receive any files
    if [ $START_IDX -ge $TOTAL_FILES ]; then
        echo "[GPU ${gpu}] No files assigned, skipping"
        continue
    fi

    # Run in the background inside a subshell
    process_gpu ${gpu} ${START_IDX} ${END_IDX} &
    GPU_PIDS+=($!)
    echo "Started worker for GPU ${gpu} (PID: $!), processing files ${START_IDX} ~ $((END_IDX - 1))"
done

echo ""
echo "All GPU workers have been started. Waiting for completion..."
echo ""

# ============ Wait For All Processes ============
ALL_SUCCESS=true
for pid in "${GPU_PIDS[@]}"; do
    if ! wait $pid; then
        ALL_SUCCESS=false
    fi
done

echo ""
echo "========================================"
if [ "$ALL_SUCCESS" = true ]; then
    echo "✓ All GPU jobs completed!"
else
    echo "⚠ Some GPU jobs failed. Please check the logs"
fi
echo "========================================"
echo "Output directory: ${OUTPUT_DIR}"
echo ""