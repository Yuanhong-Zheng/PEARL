#!/usr/bin/env bash
# Adapted from https://github.com/TIGER-AI-Lab/VLM2Vec/blob/main/experiments/public/eval/eval_8gpu.sh
# Original license: Apache-2.0
# Modified by Qwen3-VL-Embedding 2025

echo "==> Environment"
echo "Python location: $(which python)"
echo "Python version: $(python --version)"
echo ""

cd "$(dirname "${BASH_SOURCE[0]}")/../../.."

# ==============================================================================
# Multi-node Configuration
# ==============================================================================
MASTER_ADDR="${MASTER_ADDR:-localhost}"
MASTER_PORT="${MASTER_PORT:-2277}"
RANK="${RANK:-0}"
WORLD_SIZE="${WORLD_SIZE:-1}"

echo "==> Distributed Training Configuration"
echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"
echo "RANK: $RANK"
echo "WORLD_SIZE: $WORLD_SIZE"
echo ""

# ==============================================================================
# GPU Configuration
# ==============================================================================
if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
    CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((GPU_COUNT-1)))
else
    GPU_COUNT=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)
fi

echo "Using $GPU_COUNT GPUs per node: $CUDA_VISIBLE_DEVICES"
echo "Total GPUs across all nodes: $((GPU_COUNT * WORLD_SIZE))"
echo ""

# ==============================================================================
# Model Configuration
# ==============================================================================
if [ -z "$1" ]; then
    echo "Error: Model path is required"
    echo "Usage: $0 <model_path>"
    echo "Example: $0 qwen/qwen3-vl-embedding-2b"
    echo ""
    echo "Environment variables for multi-node setup:"
    echo "  MASTER_ADDR - Address of the master node (default: localhost)"
    echo "  MASTER_PORT - Port for communication (default: 2277)"
    echo "  RANK   - Rank of current node (default: 0)"
    echo "  WORLD_SIZE      - Total number of nodes (default: 1)"
    echo ""
    echo "Example multi-node usage:"
    echo "  # On master node (rank 0):"
    echo "  MASTER_ADDR=192.168.1.100 MASTER_PORT=2277 RANK=0 WORLD_SIZE=2 $0 qwen/qwen3-vl-embedding-2b"
    echo ""
    echo "  # On worker node (rank 1):"
    echo "  MASTER_ADDR=192.168.1.100 MASTER_PORT=2277 RANK=1 WORLD_SIZE=2 $0 qwen/qwen3-vl-embedding-2b"
    exit 1
fi

MODEL_NAME="$1"
MODEL_BASENAME=$(basename "$MODEL_NAME")

BATCH_SIZE=16
MODALITIES=("image" "video" "visdoc")
# MODALITIES=("tmp")
DATA_BASEDIR=data/evaluation/mmeb_v2
OUTPUT_BASEDIR=results/evaluation/mmeb_v2

BASE_OUTPUT_PATH="$OUTPUT_BASEDIR/$MODEL_BASENAME"

echo "================================================="
echo "🚀 Processing Model: $MODEL_NAME"
echo "   Output Base: $BASE_OUTPUT_PATH"
echo "================================================="
echo ""

# ==============================================================================
# Main Execution Loop
# ==============================================================================
for MODALITY in "${MODALITIES[@]}"; do
    DATA_CONFIG_PATH="scripts/evaluation/mmeb_v2/$MODALITY.yaml"
    OUTPUT_PATH="$BASE_OUTPUT_PATH/$MODALITY/"

    echo "-------------------------------------------------"
    echo "  - Modality: $MODALITY"
    echo "  - Output Path: $OUTPUT_PATH"

    # Ensure the output directory exists (only on master node)
    if [ "$RANK" -eq 0 ]; then
        mkdir -p "$OUTPUT_PATH"
    fi

    # wait for master node
    sleep 2

    cmd="CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES torchrun \
        --nproc_per_node=$GPU_COUNT \
        --nnodes=$WORLD_SIZE \
        --node_rank=$RANK \
        --master_addr=$MASTER_ADDR \
        --master_port=$MASTER_PORT \
        --max_restarts=0 \
        -m src.evaluation.mmeb_v2.eval_embedding \
        --normalize true \
        --per_device_eval_batch_size $BATCH_SIZE \
        --model_name_or_path \"$MODEL_NAME\" \
        --dataset_config \"$DATA_CONFIG_PATH\" \
        --encode_output_path \"$OUTPUT_PATH\" \
        --data_basedir \"$DATA_BASEDIR\""

    echo "  - Executing command on node $RANK..."
    eval "$cmd"
    
    if [ $? -eq 0 ]; then
        echo "  - ✅ Done on node $RANK."
    else
        echo "  - ❌ Failed on node $RANK."
        exit 1
    fi
    echo "-------------------------------------------------"
    echo ""
done

if [ "$RANK" -eq 0 ]; then
    echo "✅ All jobs completed on master node."
    
    # ==============================================================================
    # Gather Results (only on master node)
    # ==============================================================================
    echo ""
    echo "================================================="
    echo "📊 Gathering evaluation results..."
    echo "================================================="
    
    python -m src.evaluation.mmeb_v2.gather_results \
        "$BASE_OUTPUT_PATH" \
        --output_dir "$BASE_OUTPUT_PATH"
    
    if [ $? -eq 0 ]; then
        echo "✅ Results gathered successfully."
    else
        echo "❌ Failed to gather results."
        exit 1
    fi
else
    echo "✅ All jobs completed on worker node $RANK."
fi