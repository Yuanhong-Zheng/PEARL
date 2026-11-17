#!/bin/bash
# ============================================================
# Train the LoRA of the PEARL answering VLM (Qwen3-VL-8B-Instruct) on PSVU-10K
#
# Usage: bash scripts/train_reader.sh FROZEN_DIR RUN_DIR
#   FROZEN_DIR: frozen PSVU-10K snapshot with samples.jsonl (training/README.md, section 1)
#   RUN_DIR: training run directory; the adapter is written to RUN_DIR/reader
#
# Environment variables:
#   READER_MODEL: base checkpoint (default: models/Qwen3-VL-8B-Instruct)
# Hyperparameters: configs/train_reader.env
# Requires: pip install -r requirements-train.txt
# ============================================================

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "${PROJECT_ROOT}/configs/train_reader.env"

if [ $# -ne 2 ]; then
    echo "Usage: bash scripts/train_reader.sh FROZEN_DIR RUN_DIR"
    exit 1
fi
FROZEN_DIR="$(cd "$1" && pwd)"
mkdir -p "$2"
RUN_DIR="$(cd "$2" && pwd)"
READER_MODEL="$(cd "${READER_MODEL:-${PROJECT_ROOT}/models/Qwen3-VL-8B-Instruct}" && pwd)"

export PYTHONPATH="${PROJECT_ROOT}/training/src${PYTHONPATH:+:${PYTHONPATH}}"
# Frame paths in samples.jsonl are relative to the snapshot root.
cd "${FROZEN_DIR}"

accelerate launch --multi_gpu --num_processes "${NUM_GPUS}" --num_machines 1 --mixed_precision bf16 \
    --dynamo_backend no --module pearl_training.train_sft \
    --samples "${FROZEN_DIR}/samples.jsonl" --model "${READER_MODEL}" --output "${RUN_DIR}/reader" \
    --max-updates "${MAX_UPDATES}" --accumulation "${ACCUMULATION}" --lr "${LR}" --rank "${LORA_RANK}" \
    --max-length "${MAX_LENGTH}" --seed "${SEED}" --expected-gpus "${NUM_GPUS}" --save-every "${SAVE_EVERY}"
