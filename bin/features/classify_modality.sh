#!/bin/bash

set -Eeuo pipefail

DATASET="PATH-VQA"
DATASET_DIR="./samples/PATH-VQA"
OUTPUT_DIR="./data/modality_labels"
DELAY=0.3

echo "=================================================================="
echo "[RUN] Classifying image modalities for BOTH splits"
echo "=================================================================="

python ./src/classify_modality.py \
    --dataset "${DATASET}" \
    --dataset_dir "${DATASET_DIR}" \
    --split both \
    --output_dir "${OUTPUT_DIR}" \
    --delay "${DELAY}"

echo "=================================================================="
echo "[DONE] Modality classification complete"
echo "=================================================================="
