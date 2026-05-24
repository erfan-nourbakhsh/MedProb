#!/bin/bash

set -Euo pipefail

TRAIN_DATASET_EMBEDDING="VQA-RAD"
TEST_DATASET_EMBEDDING="SLAKE"
TRAIN_DATASET_DIR="./samples/${TRAIN_DATASET_EMBEDDING}"
TEST_DATASET_DIR="./samples/${TEST_DATASET_EMBEDDING}"
NORMALIZE=1
CV_FOLDS=5
SELECTIVITY=0

APPROACH="image_question"
REASONING="direct"
OPTIONS_MODE="with_options"
OPTIONS_ORDER="default"

BIOMEDICAL_MODELS=(
    "llava-med-7b"
    "medix-r1-2b"
    "adapt-internVL3-1b"
)

GENERAL_MODELS=(
    "llava-v0-7b"
    "qwen3-vl-2b-instruct"
    "internvl3-1b"
)

NSHOTS=(
    0
)

PROBE_TASKS=(
    "answer_decoding"
)

PROBE_TYPES=(
    "logistic"
)

POSITIONS=(
    "last_input_token"
)

run_count=0
fail_count=0
skip_count=0

run_probe() {
    local model="$1"
    local n_shots="$2"
    local probe_task="$3"
    local probe_type="$4"
    local position="$5"

    local config_name="${APPROACH}_${REASONING}_${OPTIONS_MODE}"
    if [[ "${OPTIONS_ORDER}" != "default" ]]; then
        config_name="${config_name}_${OPTIONS_ORDER}"
    fi
    if [[ "$n_shots" -gt 0 ]]; then
        config_name="${config_name}_${n_shots}shots"
    fi

    local pos_suffix=""
    if [[ "$position" != "last_input_token" ]]; then
        pos_suffix="_${position}"
    fi

    local train_emb_path="./data/sample_features/${model}/${TRAIN_DATASET_EMBEDDING}/${config_name}${pos_suffix}"
    local test_emb_path="./data/sample_features/${model}/${TEST_DATASET_EMBEDDING}/${config_name}${pos_suffix}"
    local train_npz_path="${train_emb_path}/train_embeddings.npz"
    local test_npz_path="${test_emb_path}/test_embeddings.npz"

    if [[ ! -f "$train_npz_path" ]]; then
        echo "[SKIP] Missing train embeddings: ${train_npz_path}"
        ((skip_count+=1))
        return
    fi

    if [[ ! -f "$test_npz_path" ]]; then
        echo "[SKIP] Missing test embeddings: ${test_npz_path}"
        ((skip_count+=1))
        return
    fi

    echo "=================================================================="
    echo "[RUN] ${probe_task}/${probe_type}: model=${model} | train=${TRAIN_DATASET_EMBEDDING} | test=${TEST_DATASET_EMBEDDING} | ${config_name} | pos=${position}"
    echo "=================================================================="

    if python ./src/linear_probe_cross_dataset.py \
        --model_stem "${model}" \
        --train_dataset_embedding "${TRAIN_DATASET_EMBEDDING}" \
        --test_dataset_embedding "${TEST_DATASET_EMBEDDING}" \
        --train_dataset_dir "${TRAIN_DATASET_DIR}" \
        --test_dataset_dir "${TEST_DATASET_DIR}" \
        --approach "${APPROACH}" \
        --reasoning "${REASONING}" \
        --options_mode "${OPTIONS_MODE}" \
        --options_order "${OPTIONS_ORDER}" \
        --n_shots "${n_shots}" \
        --probe_task "${probe_task}" \
        --probe_type "${probe_type}" \
        --normalize "${NORMALIZE}" \
        --cv "${CV_FOLDS}" \
        --selectivity "${SELECTIVITY}" \
        --extraction_position "${position}"; then
        ((run_count+=1))
    else
        ((fail_count+=1))
        echo "[FAIL] ${model} | train=${TRAIN_DATASET_EMBEDDING} | test=${TEST_DATASET_EMBEDDING} | ${config_name} | ${probe_task}/${probe_type} | pos=${position}"
    fi
    echo ""
}

ALL_MODELS=("${BIOMEDICAL_MODELS[@]}" "${GENERAL_MODELS[@]}")

for model in "${ALL_MODELS[@]}"; do
    for n_shots in "${NSHOTS[@]}"; do
        for probe_task in "${PROBE_TASKS[@]}"; do
            for probe_type in "${PROBE_TYPES[@]}"; do
                for position in "${POSITIONS[@]}"; do
                    run_probe "${model}" "${n_shots}" "${probe_task}" "${probe_type}" "${position}"
                done
            done
        done
    done
done

echo "=================================================================="
echo "[DONE] Cross-dataset probe run complete"
echo "  Successful: ${run_count}"
echo "  Failed:     ${fail_count}"
echo "  Skipped:    ${skip_count}"
echo "=================================================================="
