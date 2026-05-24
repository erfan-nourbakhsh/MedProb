#!/bin/bash

set -Euo pipefail

DATASET="VQA-RAD"
DATASET_DIR="./samples/VQA-RAD"
NORMALIZE=1
CV_FOLDS=5
SELECTIVITY=0

TRAIN_OPTIONS_ORDER="default"

BIOMEDICAL_MODELS=(

    "medvlthinker-32b"
)

GENERAL_MODELS=(

)

APPROACHES=(
    "image_question"
)

REASONING=(
    "direct"
)

OPTIONS_MODES=(
    "with_options"
)

TEST_OPTIONS_ORDER=(
    "correct_first"
    "correct_end"
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
    local approach="$2"
    local reasoning="$3"
    local options_mode="$4"
    local test_options_order="$5"
    local n_shots="$6"
    local probe_task="$7"
    local probe_type="$8"
    local position="$9"

    local train_config="${approach}_${reasoning}_${options_mode}"
    local test_config="${approach}_${reasoning}_${options_mode}_${test_options_order}"
    if [[ "$n_shots" -gt 0 ]]; then
        train_config="${train_config}_${n_shots}shots"
        test_config="${test_config}_${n_shots}shots"
    fi

    local pos_suffix=""
    if [[ "$position" != "last_input_token" ]]; then
        pos_suffix="_${position}"
    fi

    local train_npz="./data/sample_features/${model}/${DATASET}/${train_config}${pos_suffix}/train_embeddings.npz"
    local test_npz="./data/sample_features/${model}/${DATASET}/${test_config}${pos_suffix}/test_embeddings.npz"

    if [[ ! -f "$train_npz" ]]; then
        echo "[SKIP] Missing default-train embeddings: ${train_npz}"
        ((skip_count+=1))
        return
    fi
    if [[ ! -f "$test_npz" ]]; then
        echo "[SKIP] Missing reordered-test embeddings: ${test_npz}"
        ((skip_count+=1))
        return
    fi

    echo "=================================================================="
    echo "[RUN] ${probe_task}/${probe_type}: model=${model} | train=${TRAIN_OPTIONS_ORDER} | test=${test_options_order} | pos=${position}"
    echo "=================================================================="

    if python ./src/linear_probe_default_train.py \
        --dataset "${DATASET}" \
        --dataset_dir "${DATASET_DIR}" \
        --model_stem "${model}" \
        --approach "${approach}" \
        --reasoning "${reasoning}" \
        --options_mode "${options_mode}" \
        --train_options_order "${TRAIN_OPTIONS_ORDER}" \
        --test_options_order "${test_options_order}" \
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
        echo "[FAIL] ${model} | train=${TRAIN_OPTIONS_ORDER} | test=${test_options_order} | ${probe_task}/${probe_type} | pos=${position}"
    fi
    echo ""
}

ALL_MODELS=("${BIOMEDICAL_MODELS[@]}" "${GENERAL_MODELS[@]}")

for model in "${ALL_MODELS[@]}"; do
    for approach in "${APPROACHES[@]}"; do
        for reasoning in "${REASONING[@]}"; do
            for options_mode in "${OPTIONS_MODES[@]}"; do
                for test_options_order in "${TEST_OPTIONS_ORDER[@]}"; do
                    for n_shots in "${NSHOTS[@]}"; do
                        for probe_task in "${PROBE_TASKS[@]}"; do
                            for probe_type in "${PROBE_TYPES[@]}"; do
                                for position in "${POSITIONS[@]}"; do
                                    run_probe "${model}" "${approach}" "${reasoning}" "${options_mode}" "${test_options_order}" "${n_shots}" "${probe_task}" "${probe_type}" "${position}"
                                done
                            done
                        done
                    done
                done
            done
        done
    done
done

echo "=================================================================="
echo "[DONE] Default-train probing complete"
echo "  Successful: ${run_count}"
echo "  Failed:     ${fail_count}"
echo "  Skipped:    ${skip_count}"
echo "=================================================================="

if [[ $fail_count -gt 0 ]]; then
    exit 1
fi
