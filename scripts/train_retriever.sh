#!/bin/bash
# ============================================================
# Train the query-side LoRA of the PEARL retriever (Qwen3-VL-Embedding-2B) on PSVU-10K
#
# Usage: bash scripts/train_retriever.sh FROZEN_DIR RUN_DIR
#   FROZEN_DIR: frozen PSVU-10K snapshot with samples.jsonl, document_embeddings.pt and
#               release_text_queries.json (training/README.md, section 1)
#   RUN_DIR: training run directory; the adapter is written to RUN_DIR/retrieval
#
# Environment variables:
#   EMBEDDING_MODEL: base checkpoint (default: models/Qwen3-VL-Embedding-2B)
# Hyperparameters: configs/train_retriever.env
# Requires: pip install -r requirements-train.txt
# ============================================================

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "${PROJECT_ROOT}/configs/train_retriever.env"

if [ $# -ne 2 ]; then
    echo "Usage: bash scripts/train_retriever.sh FROZEN_DIR RUN_DIR"
    exit 1
fi
FROZEN_DIR="$(cd "$1" && pwd)"
mkdir -p "$2"
RUN_DIR="$(cd "$2" && pwd)"
EMBEDDING_MODEL="$(cd "${EMBEDDING_MODEL:-${PROJECT_ROOT}/models/Qwen3-VL-Embedding-2B}" && pwd)"

export PYTHONPATH="${PROJECT_ROOT}/training/src${PYTHONPATH:+:${PYTHONPATH}}"
# Frame paths in samples.jsonl are relative to the snapshot root.
cd "${FROZEN_DIR}"

accelerate launch --multi_gpu --num_processes "${NUM_GPUS}" --num_machines 1 --mixed_precision bf16 \
    --dynamo_backend no --module pearl_training.train_retrieval \
    --samples "${FROZEN_DIR}/samples.jsonl" --model "${EMBEDDING_MODEL}" --output "${RUN_DIR}/retrieval" \
    --max-updates "${MAX_UPDATES}" --accumulation "${ACCUMULATION}" --lr "${LR}" --rank "${LORA_RANK}" \
    --max-length "${MAX_LENGTH}" --seed "${SEED}" --expected-gpus "${NUM_GPUS}" --temperature "${TEMPERATURE}" \
    --document-cache "${FROZEN_DIR}/document_embeddings.pt" \
    --query-cache "${FROZEN_DIR}/release_text_queries.json"
