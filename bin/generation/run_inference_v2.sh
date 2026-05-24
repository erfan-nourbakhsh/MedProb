#!/bin/bash

set -Euo pipefail

DATASET="PATH-VQA"
DATASET_DIR="./samples/PATH-VQA"

CUDA_DEVICES="5"
BATCH_SIZE=1

MAX_NEW_TOKENS=200
TEMPERATURE=0.0

SC_SAMPLES=5
SC_TEMPERATURE=0.7
SC_MIN_AGREEMENT=0.6
RAG_TOP_K=3
RAG_MAX_CHARS=1200
RETRIEVAL_MODE="hybrid"
RRF_K=60
DISABLE_RAG=0

BIOMEDICAL_MODELS=(
    "medgemma"
)

GENERAL_MODELS=(
    "gemma"
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

USE_GPT_JUDGE=0

run_count=0
fail_count=0

run_experiment() {
    local model="$1"
    local approach="$2"
    local reasoning="$3"
    local options_mode="$4"
    local n_shots="$5"

    echo "=================================================================="
    echo "[RUN-V2] model=${model} | approach=${approach} | reasoning=${reasoning} | options=${options_mode} | shots=${n_shots}"
    echo "=================================================================="

    local disable_rag_flag=""
    if [[ "${DISABLE_RAG}" -eq 1 ]]; then
        disable_rag_flag="--disable_rag"
    fi

    if env CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" \
        python ./src/vlm_inference_v2.py \
            --dataset "${DATASET}" \
            --dataset_dir "${DATASET_DIR}" \
            --model "${model}" \
            --approach "${approach}" \
            --reasoning "${reasoning}" \
            --options_mode "${options_mode}" \
            --n_shots "${n_shots}" \
            --batch_size "${BATCH_SIZE}" \
            --max_new_tokens "${MAX_NEW_TOKENS}" \
            --temperature "${TEMPERATURE}" \
            --sc_samples "${SC_SAMPLES}" \
            --sc_temperature "${SC_TEMPERATURE}" \
            --sc_min_agreement "${SC_MIN_AGREEMENT}" \
            --rag_top_k "${RAG_TOP_K}" \
            --rag_max_chars "${RAG_MAX_CHARS}" \
            --retrieval_mode "${RETRIEVAL_MODE}" \
            --rrf_k "${RRF_K}" \
            --use_gpt_judge "${USE_GPT_JUDGE}" \
            ${disable_rag_flag}; then
        ((run_count+=1))
        echo "[OK] Completed: ${model} | ${approach} | ${reasoning} | ${options_mode} | shots=${n_shots}"
    else
        ((fail_count+=1))
        echo "[FAIL] ${model} | ${approach} | ${reasoning} | ${options_mode} | shots=${n_shots}"
    fi
    echo ""
}

ALL_MODELS=("${BIOMEDICAL_MODELS[@]}" "${GENERAL_MODELS[@]}")

for model in "${ALL_MODELS[@]}"; do
    for approach in "${APPROACHES[@]}"; do
        for reasoning in "${REASONING[@]}"; do
            for options_mode in "${OPTIONS_MODES[@]}"; do
                for n_shots in "${NSHOTS[@]}"; do
                    run_experiment "${model}" "${approach}" "${reasoning}" "${options_mode}" "${n_shots}"
                done
            done
        done
    done
done

echo "=================================================================="
echo "[DONE] Prompting v2 complete"
echo "  Successful: ${run_count}"
echo "  Failed:     ${fail_count}"
echo "=================================================================="

if [[ $fail_count -gt 0 ]]; then
    exit 1
fi
