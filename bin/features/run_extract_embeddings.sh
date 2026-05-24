#!/bin/bash

set -Euo pipefail

DATASETS=(

    "SLAKE"
)

CUDA_DEVICES="1"
BATCH_SIZE=32

FINE_TUNED_QWEN_MODELS=(

)

FINE_TUNED_LLAMA_MODELS=(

    "meta-llama3.2-11b-vision-instruct-full-all-med-vqa"

)

BIOMEDICAL_MODELS=(

)

GENERAL_MODELS=(
    "${FINE_TUNED_QWEN_MODELS[@]}"
    "${FINE_TUNED_LLAMA_MODELS[@]}"

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

OPTIONS_ORDER=(
    "default"

)

NSHOTS=(
    0

)

POSITIONS=(
    "last_input_token"

)

run_count=0
fail_count=0

run_embedding() {
    local dataset="$1"
    local model="$2"
    local approach="$3"
    local reasoning="$4"
    local options_mode="$5"
    local options_order="$6"
    local n_shots="$7"
    local position="$8"
    local dataset_dir="./samples/${dataset}"

    echo "=================================================================="
    echo "[RUN] Embedding: dataset=${dataset} | model=${model} | ${approach} | ${reasoning} | ${options_mode} | order=${options_order} | shots=${n_shots} | pos=${position}"
    echo "=================================================================="

    if [[ ! -d "${dataset_dir}" ]]; then
        ((fail_count+=1))
        echo "[FAIL] Missing dataset dir: ${dataset_dir}"
        echo ""
        return
    fi

    if env CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" \
        python ./src/extract_embeddings.py \
            --dataset "${dataset}" \
            --dataset_dir "${dataset_dir}" \
            --model "${model}" \
            --approach "${approach}" \
            --reasoning "${reasoning}" \
            --options_mode "${options_mode}" \
            --options_order "${options_order}" \
            --n_shots "${n_shots}" \
            --batch_size "${BATCH_SIZE}" \
            --extraction_position "${position}"; then
        ((run_count+=1))
        echo "[OK] Completed: dataset=${dataset} | model=${model} | ${approach} | ${reasoning} | ${options_mode} | order=${options_order} | shots=${n_shots} | pos=${position}"
    else
        ((fail_count+=1))
        echo "[FAIL] dataset=${dataset} | model=${model} | ${approach} | ${reasoning} | ${options_mode} | order=${options_order} | shots=${n_shots} | pos=${position}"
    fi
    echo ""
}

ALL_MODELS=("${BIOMEDICAL_MODELS[@]}" "${GENERAL_MODELS[@]}")

for dataset in "${DATASETS[@]}"; do
    for model in "${ALL_MODELS[@]}"; do
        for approach in "${APPROACHES[@]}"; do
            for reasoning in "${REASONING[@]}"; do
                for options_mode in "${OPTIONS_MODES[@]}"; do
                    for options_order in "${OPTIONS_ORDER[@]}"; do
                        for n_shots in "${NSHOTS[@]}"; do
                            for position in "${POSITIONS[@]}"; do
                                run_embedding "${dataset}" "${model}" "${approach}" "${reasoning}" "${options_mode}" "${options_order}" "${n_shots}" "${position}"
                            done
                        done
                    done
                done
            done
        done
    done
done

echo "=================================================================="
echo "[DONE] Embedding extraction complete"
echo "  Successful: ${run_count}"
echo "  Failed:     ${fail_count}"
echo "=================================================================="

if [[ $fail_count -gt 0 ]]; then
    exit 1
fi
