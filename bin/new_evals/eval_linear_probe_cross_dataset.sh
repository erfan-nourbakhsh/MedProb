#!/bin/bash

set -Eeuo pipefail
shopt -s nullglob

TRAIN_DATASET_EMBEDDING=(
    "VQA-RAD"
    "PATH-VQA"
    "SLAKE"
)
TEST_DATASET_EMBEDDING=(
    "VQA-RAD"
    "PATH-VQA"
    "SLAKE"
)

EVAL_PY="./src/evaluate.py"
ROOT="./data/sample_generations"
RESULTS_PATH="./data/results/new_evals/raw_new_results.tsv"
REMOVE_MISSING="${REMOVE_MISSING:-1}"

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

eval_count=0
fail_count=0

eval_lr_dir() {
    local run_dir="$1"
    local desc="$2"
    local dataset_dir="$3"

    if [[ ! -d "$run_dir" ]]; then
        echo "[SKIP] missing dir: $run_dir"
        return
    fi

    mapfile -d '' LAYER_FILES < <(
        find "$run_dir" \
            -maxdepth 1 -type f -name 'layer*.jsonl' -print0 \
            | sort -z
    )

    if [[ ${#LAYER_FILES[@]} -eq 0 ]]; then
        echo "[SKIP] no layer files in: $run_dir"
        return
    fi

    echo "=================================================================="
    echo "[EVAL] ${desc} | ${#LAYER_FILES[@]} layers"

    for pred_path in "${LAYER_FILES[@]}"; do
        echo "  -> eval: $pred_path"
        if python "$EVAL_PY" \
            --pred_path "$pred_path" \
            --dataset_dir "$dataset_dir" \
            --results_path "$RESULTS_PATH" \
            --remove_missing "$REMOVE_MISSING"; then
            ((eval_count+=1))
        else
            echo "  [FAIL] $pred_path"
            ((fail_count+=1))
        fi
    done
}

print_dataset_skip_summary() {
    local dataset_dir="$1"
    python - "$dataset_dir" "$REMOVE_MISSING" <<'PY'
import json
import os
import sys

dataset_dir = sys.argv[1]
remove_missing = bool(int(sys.argv[2]))
test_path = os.path.join(dataset_dir, "test.json")
with open(test_path, "r", encoding="utf-8") as f:
    rows = json.load(f)

dataset_total = len(rows)
if remove_missing:
    skipped = sum(
        1 for row in rows
        if not row.get("image") or not os.path.exists(os.path.join(dataset_dir, row["image"]))
    )
else:
    skipped = 0

print(f"[DATASET] total={dataset_total} processed={dataset_total - skipped} skipped={skipped} remove_missing={int(remove_missing)}")
PY
}

mkdir -p "$(dirname "$RESULTS_PATH")"

ALL_MODELS=("${BIOMEDICAL_MODELS[@]}" "${GENERAL_MODELS[@]}")

for test_dataset in "${TEST_DATASET_EMBEDDING[@]}"; do
    print_dataset_skip_summary "./samples/${test_dataset}"
done

for train_dataset in "${TRAIN_DATASET_EMBEDDING[@]}"; do
    for test_dataset in "${TEST_DATASET_EMBEDDING[@]}"; do
        test_dataset_dir="./samples/${test_dataset}"

        for model in "${ALL_MODELS[@]}"; do
            for n_shots in "${NSHOTS[@]}"; do
                for probe_task in "${PROBE_TASKS[@]}"; do
                    for probe_type in "${PROBE_TYPES[@]}"; do
                        for position in "${POSITIONS[@]}"; do
                            config_name="${APPROACH}_${REASONING}_${OPTIONS_MODE}"
                            if [[ "${OPTIONS_ORDER}" != "default" ]]; then
                                config_name="${config_name}_${OPTIONS_ORDER}"
                            fi
                            if [[ "$n_shots" -gt 0 ]]; then
                                config_name="${config_name}_${n_shots}shots"
                            fi
                            pos_suffix=""
                            if [[ "$position" != "last_input_token" ]]; then
                                pos_suffix="_${position}"
                            fi
                            task_dir="${probe_task}_${probe_type}_cross_dataset"
                            transfer_name="train_${train_dataset}__test_${test_dataset}"
                            run_dir="${ROOT}/${model}/cross_dataset/${task_dir}/${transfer_name}/${config_name}${pos_suffix}"
                            desc="model=${model} | train=${train_dataset} | test=${test_dataset} | ${task_dir} | ${config_name} | pos=${position}"
                            eval_lr_dir "$run_dir" "$desc" "$test_dataset_dir"
                        done
                    done
                done
            done
        done
    done
done

echo "=================================================================="
echo "[DONE] Cross-dataset probing evaluation complete"
echo "  Evaluated: ${eval_count}"
echo "  Failed:    ${fail_count}"
echo "  Results:   ${RESULTS_PATH}"
echo "=================================================================="

if [[ $fail_count -gt 0 ]]; then
    exit 1
fi
