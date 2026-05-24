#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${PROJECT_ROOT}/venv/bin/python}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="python"
fi

DATASET="${1:-${DATASET:-}}"
if [[ -z "${DATASET}" ]]; then
  echo "Usage: $0 <PATH-VQA|VQA-RAD|SLAKE|ALL_MED_VQA>"
  exit 1
fi

case "${DATASET}" in
  PATH-VQA|VQA-RAD|SLAKE|ALL_MED_VQA) ;;
  *)
    echo "Unsupported dataset: ${DATASET}"
    echo "Valid datasets: PATH-VQA, VQA-RAD, SLAKE, ALL_MED_VQA"
    exit 1
    ;;
esac

MODEL_ID="${MODEL_ID:-Qwen/Qwen2.5-VL-7B-Instruct}"
MODEL_STEM="${MODEL_STEM:-qwen25-vl-7b-instruct}"
METHOD="${METHOD:-full}"
GPU_ID="${GPU_ID:-0}"
DATASET_DIR="${DATASET_DIR:-${PROJECT_ROOT}/all_samples/${DATASET}}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/fine-tune/results/${DATASET}/${MODEL_STEM//\//__}-${METHOD}}"

EPOCHS="${EPOCHS:-1.0}"
MAX_LENGTH="${MAX_LENGTH:-4096}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-1}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-8}"
LEARNING_RATE="${LEARNING_RATE:-2e-5}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
WARMUP_RATIO="${WARMUP_RATIO:-0.03}"
LOGGING_STEPS="${LOGGING_STEPS:-10}"
EVAL_STEPS="${EVAL_STEPS:-200}"
SAVE_STEPS="${SAVE_STEPS:-200}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-2}"
EVAL_SPLIT="${EVAL_SPLIT:-none}"

LORA_R="${LORA_R:-64}"
LORA_ALPHA="${LORA_ALPHA:-128}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
SEED="${SEED:-42}"
BF16_FLAG="${BF16_FLAG:-1}"
FP16_FLAG="${FP16_FLAG:-0}"

extra_flags=()
if [[ "${BF16_FLAG}" == "1" ]]; then
  extra_flags+=("--bf16")
fi
if [[ "${FP16_FLAG}" == "1" ]]; then
  extra_flags+=("--fp16")
fi

mkdir -p "${PROJECT_ROOT}/fine-tune/results/${DATASET}"

"${PYTHON_BIN}" "${PROJECT_ROOT}/fine-tune/qwen25_vl_finetune.py" \
  --dataset_name "${DATASET}" \
  --dataset_dir "${DATASET_DIR}" \
  --model_id "${MODEL_ID}" \
  --model_stem "${MODEL_STEM}" \
  --method "${METHOD}" \
  --output_dir "${OUTPUT_DIR}" \
  --approach image_question \
  --reasoning direct \
  --options_mode with_options \
  --options_order default \
  --max_length "${MAX_LENGTH}" \
  --epochs "${EPOCHS}" \
  --train_batch_size "${TRAIN_BATCH_SIZE}" \
  --eval_batch_size "${EVAL_BATCH_SIZE}" \
  --grad_accum_steps "${GRAD_ACCUM_STEPS}" \
  --learning_rate "${LEARNING_RATE}" \
  --weight_decay "${WEIGHT_DECAY}" \
  --warmup_ratio "${WARMUP_RATIO}" \
  --logging_steps "${LOGGING_STEPS}" \
  --eval_steps "${EVAL_STEPS}" \
  --save_steps "${SAVE_STEPS}" \
  --save_total_limit "${SAVE_TOTAL_LIMIT}" \
  --eval_split "${EVAL_SPLIT}" \
  --lora_r "${LORA_R}" \
  --lora_alpha "${LORA_ALPHA}" \
  --lora_dropout "${LORA_DROPOUT}" \
  --seed "${SEED}" \
  --gpu_id "${GPU_ID}" \
  "${extra_flags[@]}"
