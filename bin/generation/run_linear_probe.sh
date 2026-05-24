#!/bin/bash

set -Euo pipefail

DATASET="PATH-VQA"
DATASET_DIR="./samples/PATH-VQA"
NORMALIZE=1
CV_FOLDS=5
SELECTIVITY=0

BIOMEDICAL_MODELS=(
    "medgemma"

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

NSHOTS=(
    0

)

PROBE_TASKS=(
    "answer_decoding"

)

PROBE_TYPES=(

    "mlp"
)

POSITIONS=(
    "last_input_token"

)

run_count=0
fail_count=0

run_probe() {
    local model="$1"
    local approach="$2"
    local reasoning="$3"
    local options_mode="$4"
    local n_shots="$5"
    local probe_task="$6"
    local probe_type="$7"
    local position="$8"

    local config_name="${approach}_${reasoning}_${options_mode}"
    if [[ "$n_shots" -gt 0 ]]; then
        config_name="${config_name}_${n_shots}shots"
    fi

    local pos_suffix=""
    if [[ "$position" != "last_input_token" ]]; then
        pos_suffix="_${position}"
    fi

    local emb_path="./data/sample_features/${model}/${DATASET}/${config_name}${pos_suffix}"
    local npz_path="${emb_path}/train_embeddings.npz"

    if [[ ! -f "$npz_path" ]]; then
        echo "[SKIP] Missing embeddings: ${npz_path}"
        return
    fi

    echo "=================================================================="
    echo "[RUN] ${probe_task}/${probe_type}: model=${model} | ${config_name} | pos=${position}"
    echo "=================================================================="

    if python ./src/linear_probe.py \
        --dataset "${DATASET}" \
        --dataset_dir "${DATASET_DIR}" \
        --model_stem "${model}" \
        --approach "${approach}" \
        --reasoning "${reasoning}" \
        --options_mode "${options_mode}" \
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
        echo "[FAIL] ${model} | ${config_name} | ${probe_task}/${probe_type}"
    fi
    echo ""
}

ALL_MODELS=("${BIOMEDICAL_MODELS[@]}" "${GENERAL_MODELS[@]}")

for model in "${ALL_MODELS[@]}"; do
    for approach in "${APPROACHES[@]}"; do
        for reasoning in "${REASONING[@]}"; do
            for options_mode in "${OPTIONS_MODES[@]}"; do
                for n_shots in "${NSHOTS[@]}"; do
                    for probe_task in "${PROBE_TASKS[@]}"; do
                        for probe_type in "${PROBE_TYPES[@]}"; do
                            for position in "${POSITIONS[@]}"; do
                                run_probe "${model}" "${approach}" "${reasoning}" "${options_mode}" "${n_shots}" "${probe_task}" "${probe_type}" "${position}"
                            done
                        done
                    done
                done
            done
        done
    done
done

echo "=================================================================="
echo "[DONE] Probing complete"
echo "  Successful: ${run_count}"
echo "  Failed:     ${fail_count}"
echo "=================================================================="

if [[ $fail_count -gt 0 ]]; then
    exit 1
fi
