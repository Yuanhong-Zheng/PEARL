#!/bin/bash
# ============================================================
# 多 GPU 服务器启动脚本
# 每个 GPU 上启动一对 vLLM + Embedding 服务器
#
# 用法: bash scripts/start_multi_gpu_servers.sh [NUM_GPUS]
#   NUM_GPUS: 使用的 GPU 数量（默认：4）
#
# 端口分配:
#   GPU 0: vLLM -> 22003, Embedding -> 5000
#   GPU 1: vLLM -> 22004, Embedding -> 5001
#   GPU 2: vLLM -> 22005, Embedding -> 5002
#   ...
# ============================================================

set -e

# ============ 可配置参数 ============
NUM_GPUS=${1:-8}

VLLM_BASE_PORT=22003
EMBEDDING_BASE_PORT=5000

# 项目根目录（PEARL）
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# 相对项目根目录的脚本路径
VLLM_SCRIPT_REL="server/qwenvl_flask_server.py"
# VLLM_SCRIPT_REL="server/llava_ov_flask_server.py"
EMBEDDING_SCRIPT_REL="server/embedding_server.py"
LOG_DIR_REL="server/logs"

VLLM_SCRIPT="${PROJECT_ROOT}/${VLLM_SCRIPT_REL}"
EMBEDDING_SCRIPT="${PROJECT_ROOT}/${EMBEDDING_SCRIPT_REL}"
LOG_DIR="${PROJECT_ROOT}/${LOG_DIR_REL}"


# ====================================

echo "========================================"
echo "多 GPU 服务器启动脚本"
echo "========================================"
echo "GPU 数量: ${NUM_GPUS}"
echo "vLLM 基础端口: ${VLLM_BASE_PORT}"
echo "Embedding 基础端口: ${EMBEDDING_BASE_PORT}"
echo "========================================"
echo ""

# 创建日志目录
mkdir -p "${LOG_DIR}"

# 存储所有后台进程的 PID
PIDS=()

for ((gpu=0; gpu<NUM_GPUS; gpu++)); do
    VLLM_PORT=$((VLLM_BASE_PORT + gpu))
    EMBEDDING_PORT=$((EMBEDDING_BASE_PORT + gpu))

    echo "-------- GPU ${gpu} --------"
    echo "  vLLM 服务端口: ${VLLM_PORT}"
    echo "  Embedding 服务端口: ${EMBEDDING_PORT}"

    # 启动 vLLM 服务器
    CUDA_VISIBLE_DEVICES=${gpu} python "${VLLM_SCRIPT}" \
        --port ${VLLM_PORT} \
        > "${LOG_DIR}/vllm_gpu${gpu}.log" 2>&1 &
    PIDS+=($!)
    echo "  vLLM 服务已启动 (PID: $!, 日志: ${LOG_DIR}/vllm_gpu${gpu}.log)"

    # 启动 Embedding 服务器
    CUDA_VISIBLE_DEVICES=${gpu} python "${EMBEDDING_SCRIPT}" \
        --port ${EMBEDDING_PORT} \
        > "${LOG_DIR}/embedding_gpu${gpu}.log" 2>&1 &
    PIDS+=($!)
    echo "  Embedding 服务已启动 (PID: $!, 日志: ${LOG_DIR}/embedding_gpu${gpu}.log)"

    echo ""
done

echo "========================================"
echo "所有服务已启动！"
echo "========================================"
echo ""
echo "端口汇总:"
for ((gpu=0; gpu<NUM_GPUS; gpu++)); do
    VLLM_PORT=$((VLLM_BASE_PORT + gpu))
    EMBEDDING_PORT=$((EMBEDDING_BASE_PORT + gpu))
    echo "  GPU ${gpu}: vLLM=http://127.0.0.1:${VLLM_PORT}/v1  Embedding=http://127.0.0.1:${EMBEDDING_PORT}"
done
echo ""
echo "所有后台进程 PID: ${PIDS[*]}"
echo ""
echo "查看日志:  tail -f ${LOG_DIR}/vllm_gpu0.log"
echo "停止所有:  kill ${PIDS[*]}"
echo "    或者:  pkill -f vllm_flask_server.py && pkill -f embedding_server.py"
echo ""

# 等待所有后台进程（按 Ctrl+C 终止所有）
trap "echo '正在终止所有服务...'; kill ${PIDS[*]} 2>/dev/null; exit 0" SIGINT SIGTERM
wait

