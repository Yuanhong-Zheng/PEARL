#!/bin/bash
# ============================================================
# 多 GPU 并行评估脚本
# 将 annotation 文件均匀分配到多个 GPU 上并行处理
#
# 用法: bash scripts/eval_qwen.sh [NUM_GPUS]
#   NUM_GPUS: 使用的 GPU 数量（默认：4，需与 start_multi_gpu_servers.sh 一致）
#
# 前置条件: 已通过 start_multi_gpu_servers.sh 启动了对应数量的服务器
# ============================================================

set -e

# ============ 可配置参数 ============
NUM_GPUS=${1:-8}

# 项目根目录（PEARL）
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

ANNOTATION_DIR="${PROJECT_ROOT}/data/frame-level/annotations_debug"
CLIPS_BASE_DIR="${PROJECT_ROOT}/data/frame-level/output_clips"
CACHE_DIR="${PROJECT_ROOT}/.cache"
OUTPUT_DIR="${PROJECT_ROOT}/output_results/test/qwen3vl_k4_n1_pre0_fps1_debug2"
PYTHON_SCRIPT="${PROJECT_ROOT}/video_qa_inference.py"

VLLM_BASE_PORT=22003
EMBEDDING_BASE_PORT=5000

# 开关：是否跳过已存在的输出文件 (true/false)
SKIP_EXISTING=false
# ====================================

echo "========================================"
echo "多 GPU 并行评估脚本"
echo "========================================"
echo "GPU 数量: ${NUM_GPUS}"
echo "Annotation 目录: ${ANNOTATION_DIR}"
echo "Clips 基础目录: ${CLIPS_BASE_DIR}"
echo "输出目录: ${OUTPUT_DIR}"
echo "跳过已存在: ${SKIP_EXISTING}"
echo "========================================"
echo ""

if [ ! -d "${ANNOTATION_DIR}" ]; then
    echo "错误: Annotation 目录不存在: ${ANNOTATION_DIR}"
    exit 1
fi

if [ ! -f "${PYTHON_SCRIPT}" ]; then
    echo "错误: 推理脚本不存在: ${PYTHON_SCRIPT}"
    exit 1
fi

if [ ! -d "${CLIPS_BASE_DIR}" ]; then
    echo "错误: Clips 基础目录不存在: ${CLIPS_BASE_DIR}"
    exit 1
fi

# ============ 等待所有服务器就绪 ============
echo "正在检查服务器状态..."
MAX_WAIT=600  # 最多等待 600 秒
WAIT_INTERVAL=10

for ((gpu=0; gpu<NUM_GPUS; gpu++)); do
    VLLM_PORT=$((VLLM_BASE_PORT + gpu))
    EMBEDDING_PORT=$((EMBEDDING_BASE_PORT + gpu))

    echo -n "  检查 GPU ${gpu} vLLM (端口 ${VLLM_PORT})..."
    elapsed=0
    while true; do
        if curl -s "http://127.0.0.1:${VLLM_PORT}/health" > /dev/null 2>&1; then
            echo " ✓ 就绪"
            break
        fi
        elapsed=$((elapsed + WAIT_INTERVAL))
        if [ $elapsed -ge $MAX_WAIT ]; then
            echo " ✗ 超时（${MAX_WAIT}s）"
            echo "错误: GPU ${gpu} 的 vLLM 服务未就绪，请检查服务是否已启动"
            exit 1
        fi
        sleep ${WAIT_INTERVAL}
    done

    echo -n "  检查 GPU ${gpu} Embedding (端口 ${EMBEDDING_PORT})..."
    elapsed=0
    while true; do
        if curl -s "http://127.0.0.1:${EMBEDDING_PORT}/health" > /dev/null 2>&1; then
            echo " ✓ 就绪"
            break
        fi
        elapsed=$((elapsed + WAIT_INTERVAL))
        if [ $elapsed -ge $MAX_WAIT ]; then
            echo " ✗ 超时（${MAX_WAIT}s）"
            echo "错误: GPU ${gpu} 的 Embedding 服务未就绪，请检查服务是否已启动"
            exit 1
        fi
        sleep ${WAIT_INTERVAL}
    done
done
echo ""
echo "✓ 所有服务器已就绪！"
echo ""

# ============ 收集所有 JSON 文件 ============
JSON_FILES=()
for json_file in "${ANNOTATION_DIR}"/*.json; do
    if [ -f "$json_file" ]; then
        # 如果启用了跳过，检查输出文件是否已存在
        if [ "$SKIP_EXISTING" = true ]; then
            filename=$(basename "$json_file")
            base_name="${filename%.json}"
            output_file="${OUTPUT_DIR}/${base_name}_evaluation.json"
            if [ -f "$output_file" ]; then
                echo "⊘ 跳过已存在: ${filename}"
                continue
            fi
        fi
        JSON_FILES+=("$json_file")
    fi
done

TOTAL_FILES=${#JSON_FILES[@]}

if [ $TOTAL_FILES -eq 0 ]; then
    echo ""
    echo "没有需要处理的文件（全部已跳过或目录为空）"
    exit 0
fi

echo ""
echo "需要处理的文件数: ${TOTAL_FILES}"
echo "GPU 数量: ${NUM_GPUS}"
echo ""

# ============ 分配文件到各 GPU ============
# 每个 GPU 分配的文件数（向上取整）
FILES_PER_GPU=$(( (TOTAL_FILES + NUM_GPUS - 1) / NUM_GPUS ))

echo "每个 GPU 约处理 ${FILES_PER_GPU} 个文件"
echo ""

# 确保输出目录存在
mkdir -p "${OUTPUT_DIR}"

# ============ 定义单 GPU 处理函数 ============
process_gpu() {
    local gpu_id=$1
    local start_idx=$2
    local end_idx=$3
    local vllm_port=$((VLLM_BASE_PORT + gpu_id))
    local embedding_port=$((EMBEDDING_BASE_PORT + gpu_id))
    local api_url="http://127.0.0.1:${vllm_port}/v1"
    local embedding_url="http://127.0.0.1:${embedding_port}"

    local success_count=0
    local fail_count=0

    echo "[GPU ${gpu_id}] 开始处理文件 ${start_idx} ~ $((end_idx - 1))，共 $((end_idx - start_idx)) 个"
    echo "[GPU ${gpu_id}] vLLM API: ${api_url}"
    echo "[GPU ${gpu_id}] Embedding API: ${embedding_url}"

    for ((i=start_idx; i<end_idx; i++)); do
        local json_file="${JSON_FILES[$i]}"
        local filename=$(basename "$json_file")

        echo "[GPU ${gpu_id}] 正在处理 ($((i - start_idx + 1))/$((end_idx - start_idx))): ${filename}"

        python "$PYTHON_SCRIPT" \
            --annotation_path "$json_file" \
            --clips_base_dir "$CLIPS_BASE_DIR" \
            --cache_dir "$CACHE_DIR" \
            --output_path "$OUTPUT_DIR" \
            --api_base_url "$api_url" \
            --embedding_api_url "$embedding_url" \
            --gpu_id "$gpu_id" \
            --enable_rotation
            

        EXIT_CODE=$?
        if [ $EXIT_CODE -eq 0 ]; then
            echo "[GPU ${gpu_id}] ✓ 成功: ${filename}"
            success_count=$((success_count + 1))
        else
            echo "[GPU ${gpu_id}] ✗ 失败: ${filename} (exit code: ${EXIT_CODE})"
            fail_count=$((fail_count + 1))
            # 如果某个文件失败，可以选择继续或终止
            # 取消注释下一行可在失败时终止该 GPU 的处理
            # return 1
        fi
    done

    echo ""
    echo "[GPU ${gpu_id}] ======== 处理完成 ========"
    echo "[GPU ${gpu_id}] 成功: ${success_count}, 失败: ${fail_count}"
    echo ""

    if [ $fail_count -gt 0 ]; then
        return 1
    fi
    return 0
}

# ============ 启动多 GPU 并行处理 ============
echo "========================================"
echo "开始多 GPU 并行处理..."
echo "========================================"
echo ""

GPU_PIDS=()

for ((gpu=0; gpu<NUM_GPUS; gpu++)); do
    START_IDX=$((gpu * FILES_PER_GPU))
    END_IDX=$((START_IDX + FILES_PER_GPU))

    # 确保不超过总文件数
    if [ $END_IDX -gt $TOTAL_FILES ]; then
        END_IDX=$TOTAL_FILES
    fi

    # 如果该 GPU 没有分配到文件，跳过
    if [ $START_IDX -ge $TOTAL_FILES ]; then
        echo "[GPU ${gpu}] 没有分配到文件，跳过"
        continue
    fi

    # 在子 shell 中后台运行
    process_gpu ${gpu} ${START_IDX} ${END_IDX} &
    GPU_PIDS+=($!)
    echo "已启动 GPU ${gpu} 的处理进程 (PID: $!), 处理文件 ${START_IDX} ~ $((END_IDX - 1))"
done

echo ""
echo "所有 GPU 处理进程已启动，等待完成..."
echo ""

# ============ 等待所有进程完成 ============
ALL_SUCCESS=true
for pid in "${GPU_PIDS[@]}"; do
    if ! wait $pid; then
        ALL_SUCCESS=false
    fi
done

echo ""
echo "========================================"
if [ "$ALL_SUCCESS" = true ]; then
    echo "✓ 所有 GPU 处理完成！"
else
    echo "⚠ 部分 GPU 处理存在失败，请检查日志"
fi
echo "========================================"
echo "输出目录: ${OUTPUT_DIR}"
echo ""
