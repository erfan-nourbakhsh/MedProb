#!/bin/bash

set -Euo pipefail

DATASETS=(
    "PATH-VQA"

)

CUDA_DEVICES="1"
BATCH_SIZE=32

FINE_TUNED_QWEN_MODELS=(

)

FINE_TUNED_LLAMA_MODELS=(
    "meta-llama3.2-11b-vision-instruct-full-path-vqa"

)

MAX_NEW_TOKENS=200
TEMPERATURE=0.0
TOP_P="null"

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

run_count=0
fail_count=0

run_experiment() {
    local dataset="$1"
    local model="$2"
    local approach="$3"
    local reasoning="$4"
    local options_mode="$5"
    local options_order="$6"
    local n_shots="$7"
    local dataset_dir="./samples/${dataset}"
    local max_new_tokens="${MAX_NEW_TOKENS}"
    local temperature="${TEMPERATURE}"
    local top_p="${TOP_P}"

    if [[ "${model}" == "med-flamingo-9b" || "${model}" == random_med-flamingo-9b_* || "${model}" == "open-flamingo-9b" || "${model}" == random_open-flamingo-9b_* ]]; then
        max_new_tokens=512
        temperature=0.0
        top_p="null"
    elif [[ "${model}" == "llava-med-7b" || "${model}" == random_llava-med-7b_* || "${model}" == "llava-v0-7b" || "${model}" == random_llava-v0-7b_* ]]; then
        max_new_tokens=200
        temperature=0.0
        top_p="null"
    elif [[ "${model}" == "medgemma" || "${model}" == random_medgemma_4b_* || "${model}" == "medgemma-27b" || "${model}" == random_medgemma_27b_* || "${model}" == "gemma" || "${model}" == random_gemma_4b_* || "${model}" == "gemma-27b" || "${model}" == random_gemma_27b_* ]]; then
        max_new_tokens=200
        temperature=0.0
        top_p="null"
    elif [[ "${model}" == medvlthinker-* ]]; then
        max_new_tokens=2048
        temperature=0.6
        top_p=0.95
    elif [[ "${model}" == "medix-r1-2b" || "${model}" == "medix-r1-30b" ]]; then
        max_new_tokens=2048
        temperature=0.0
        top_p=1.0
    elif [[ "${model}" == meta-llama3.2-11b-vision-instruct* ]]; then
        max_new_tokens=2048
        temperature=0.0
        top_p="null"
    elif [[ "${model}" == "adapt-llama3.2-11b" ]]; then
        max_new_tokens=2048
        temperature=0.0
        top_p="null"
    elif [[ "${model}" == "adapt-qwen2-2b" ]]; then
        max_new_tokens=2048
        temperature=0.7
        top_p=0.8
    elif [[ "${model}" == "adapt-internVL3-1b" ]]; then
        max_new_tokens=1024
        temperature=0.0
        top_p="null"
    elif [[ "${model}" == "internvl3-1b" ]]; then
        max_new_tokens=1024
        temperature=0.0
        top_p="null"
    elif [[ "${model}" == "medmo-4b-next" || "${model}" == "medmo-8b-next" ]]; then
        max_new_tokens=512
        temperature=0.0
        top_p="null"
    elif [[ "${model}" == "qwen2-vl-2b-instruct" ]]; then
        max_new_tokens=1024
        temperature=0.7
        top_p=0.8
    elif [[ "${model}" == qwen*-vl-*instruct* ]]; then
        max_new_tokens=128
        temperature=0.7
        top_p=0.8
    fi

    echo "=================================================================="
    echo "[RUN] dataset=${dataset} | model=${model} | approach=${approach} | reasoning=${reasoning} | options=${options_mode} | order=${options_order} | shots=${n_shots}"
    echo "=================================================================="

    if env CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" \
        python ./src/vlm_inference.py \
            --dataset "${dataset}" \
            --dataset_dir "${dataset_dir}" \
            --model "${model}" \
            --approach "${approach}" \
            --reasoning "${reasoning}" \
            --options_mode "${options_mode}" \
            --options_order "${options_order}" \
            --n_shots "${n_shots}" \
            --batch_size "${BATCH_SIZE}" \
            --max_new_tokens "${max_new_tokens}" \
            --temperature "${temperature}" \
            --top_p "${top_p}" \
            --use_gpt_judge "${USE_GPT_JUDGE}"; then
        ((run_count+=1))
        echo "[OK] Completed: ${dataset} | ${model} | ${approach} | ${reasoning} | ${options_mode} | order=${options_order} | shots=${n_shots}"
    else
        ((fail_count+=1))
        echo "[FAIL] ${dataset} | ${model} | ${approach} | ${reasoning} | ${options_mode} | order=${options_order} | shots=${n_shots}"
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
                            run_experiment "${dataset}" "${model}" "${approach}" "${reasoning}" "${options_mode}" "${options_order}" "${n_shots}"
                        done
                    done
                done
            done
        done
    done
done

echo "=================================================================="
echo "[DONE] Prompting complete"
echo "  Successful: ${run_count}"
echo "  Failed:     ${fail_count}"
echo "=================================================================="

if [[ $fail_count -gt 0 ]]; then
    exit 1
fi
