#!/bin/bash
# ============================================================
# 单 GPU Debug 评估脚本（Qwen）
# 用法: bash scripts/eval_qwen_debug.sh [GPU_ID] [MAX_FILES]
#   GPU_ID: 使用的单卡编号（默认：0）
#   MAX_FILES: 最多处理文件数（默认：0，表示处理全部）
# ============================================================

set -e

# ============ 可配置参数 ============
GPU_ID=${1:-0}
MAX_FILES=${2:-0}

# 项目根目录（PEARL）
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

ANNOTATION_DIR="${PROJECT_ROOT}/data/frame-level/annotations_short"
CLIPS_BASE_DIR="${PROJECT_ROOT}/data/frame-level/output_clips"
CACHE_DIR="${PROJECT_ROOT}/.cache"
OUTPUT_DIR="${PROJECT_ROOT}/output_results/renamed/debug_qwen_short_clip"
PYTHON_SCRIPT="${PROJECT_ROOT}/video_qa_inference.py"

VLLM_BASE_PORT=22003
EMBEDDING_BASE_PORT=5000

# 开关：是否跳过已存在的输出文件 (true/false)
SKIP_EXISTING=true
# ====================================

VLLM_PORT=$((VLLM_BASE_PORT + GPU_ID))
EMBEDDING_PORT=$((EMBEDDING_BASE_PORT + GPU_ID))
API_URL="http://127.0.0.1:${VLLM_PORT}/v1"
EMBEDDING_URL="http://127.0.0.1:${EMBEDDING_PORT}"

echo "========================================"
echo "单 GPU Debug 评估脚本"
echo "========================================"
echo "GPU ID: ${GPU_ID}"
echo "vLLM API: ${API_URL}"
echo "Embedding API: ${EMBEDDING_URL}"
echo "Annotation 目录: ${ANNOTATION_DIR}"
echo "Clips 基础目录: ${CLIPS_BASE_DIR}"
echo "输出目录: ${OUTPUT_DIR}"
echo "最多处理文件数: ${MAX_FILES} (0 表示全部)"
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

# 确保输出目录存在
mkdir -p "${OUTPUT_DIR}"

# ============ 等待单卡服务器就绪 ============
echo "正在检查单卡服务状态..."
MAX_WAIT=600
WAIT_INTERVAL=10

echo -n "  检查 vLLM (端口 ${VLLM_PORT})..."
elapsed=0
while true; do
    if curl -s "http://127.0.0.1:${VLLM_PORT}/health" > /dev/null 2>&1; then
        echo " ✓ 就绪"
        break
    fi
    elapsed=$((elapsed + WAIT_INTERVAL))
    if [ $elapsed -ge $MAX_WAIT ]; then
        echo " ✗ 超时（${MAX_WAIT}s）"
        echo "错误: vLLM 服务未就绪"
        exit 1
    fi
    sleep ${WAIT_INTERVAL}
done

echo -n "  检查 Embedding (端口 ${EMBEDDING_PORT})..."
elapsed=0
while true; do
    if curl -s "http://127.0.0.1:${EMBEDDING_PORT}/health" > /dev/null 2>&1; then
        echo " ✓ 就绪"
        break
    fi
    elapsed=$((elapsed + WAIT_INTERVAL))
    if [ $elapsed -ge $MAX_WAIT ]; then
        echo " ✗ 超时（${MAX_WAIT}s）"
        echo "错误: Embedding 服务未就绪"
        exit 1
    fi
    sleep ${WAIT_INTERVAL}
done
echo ""

# ============ 收集 JSON 文件 ============
JSON_FILES=()
for json_file in "${ANNOTATION_DIR}"/*.json; do
    if [ -f "${json_file}" ]; then
        if [ "${SKIP_EXISTING}" = true ]; then
            filename=$(basename "${json_file}")
            base_name="${filename%.json}"
            output_file="${OUTPUT_DIR}/${base_name}_evaluation.json"
            if [ -f "${output_file}" ]; then
                echo "⊘ 跳过已存在: ${filename}"
                continue
            fi
        fi
        JSON_FILES+=("${json_file}")
    fi
done

TOTAL_FILES=${#JSON_FILES[@]}
if [ ${TOTAL_FILES} -eq 0 ]; then
    echo "没有需要处理的文件（全部已跳过或目录为空）"
    exit 0
fi

if [ ${MAX_FILES} -gt 0 ] && [ ${MAX_FILES} -lt ${TOTAL_FILES} ]; then
    TOTAL_FILES=${MAX_FILES}
fi

echo "实际处理文件数: ${TOTAL_FILES}"
echo ""

success_count=0
fail_count=0

for ((i=0; i<TOTAL_FILES; i++)); do
    json_file="${JSON_FILES[$i]}"
    filename=$(basename "${json_file}")
    echo "[DEBUG][GPU ${GPU_ID}] 正在处理 ($((i + 1))/${TOTAL_FILES}): ${filename}"

    if python "${PYTHON_SCRIPT}" \
        --annotation_path "${json_file}" \
        --clips_base_dir "${CLIPS_BASE_DIR}" \
        --cache_dir "${CACHE_DIR}" \
        --output_path "${OUTPUT_DIR}" \
        --api_base_url "${API_URL}" \
        --embedding_api_url "${EMBEDDING_URL}" \
        --gpu_id "${GPU_ID}" \
        --replace_concept_in_query \
        --enable_rotation; then
        echo "[DEBUG][GPU ${GPU_ID}] ✓ 成功: ${filename}"
        success_count=$((success_count + 1))
    else
        echo "[DEBUG][GPU ${GPU_ID}] ✗ 失败: ${filename}"
        fail_count=$((fail_count + 1))
        echo "[DEBUG][GPU ${GPU_ID}] 检测到异常退出，立即终止脚本"
        exit 1
    fi
done

echo ""
echo "========================================"
echo "[DEBUG] 处理完成"
echo "成功: ${success_count}, 失败: ${fail_count}"
echo "输出目录: ${OUTPUT_DIR}"
echo "========================================"
echo ""
