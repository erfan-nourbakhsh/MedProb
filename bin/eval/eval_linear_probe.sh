#!/bin/bash

set -Eeuo pipefail
shopt -s nullglob

DATASETS=(
    "SLAKE"
    "VQA-RAD"
    "PATH-VQA"
)
EVAL_PY="./src/evaluate.py"
ROOT="data/sample_generations"
REMOVE_MISSING="${REMOVE_MISSING:-1}"

FINE_TUNED_QWEN_MODELS=(
    "qwen25-vl-7b-instruct-full-path-vqa"
    "qwen25-vl-7b-instruct-full-all-med-vqa"
    "qwen25-vl-7b-instruct-full-slake"
    "qwen25-vl-7b-instruct-full-vqa-rad"
)

FINE_TUNED_LLAMA_MODELS=(
    "meta-llama3.2-11b-vision-instruct-full-path-vqa"
    "meta-llama3.2-11b-vision-instruct-full-all-med-vqa"
    "meta-llama3.2-11b-vision-instruct-full-slake"
    "meta-llama3.2-11b-vision-instruct-full-vqa-rad"
)

MODELS=(

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

for dataset in "${DATASETS[@]}"; do
    dataset_dir="./samples/${dataset}"
    print_dataset_skip_summary "$dataset_dir"

    for model in "${MODELS[@]}"; do
        for approach in "${APPROACHES[@]}"; do
            for reasoning in "${REASONING[@]}"; do
                for options_mode in "${OPTIONS_MODES[@]}"; do
                    for options_order in "${OPTIONS_ORDER[@]}"; do
                        for n_shots in "${NSHOTS[@]}"; do
                            for probe_task in "${PROBE_TASKS[@]}"; do
                                for probe_type in "${PROBE_TYPES[@]}"; do
                                    for position in "${POSITIONS[@]}"; do
                                        config_name="${approach}_${reasoning}_${options_mode}"
                                        if [[ "${options_order}" != "default" ]]; then
                                            config_name="${config_name}_${options_order}"
                                        fi
                                        if [[ "$n_shots" -gt 0 ]]; then
                                            config_name="${config_name}_${n_shots}shots"
                                        fi
                                        pos_suffix=""
                                        if [[ "$position" != "last_input_token" ]]; then
                                            pos_suffix="_${position}"
                                        fi
                                        task_dir="${probe_task}_${probe_type}"
                                        run_dir="${ROOT}/${model}/${dataset}/${task_dir}/${config_name}${pos_suffix}"
                                        desc="dataset=${dataset} | model=${model} | ${task_dir} | ${config_name} | order=${options_order} | pos=${position}"
                                        eval_lr_dir "$run_dir" "$desc" "$dataset_dir"
                                    done
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
echo "[DONE] Probing evaluation complete"
echo "  Evaluated: ${eval_count}"
echo "  Failed:    ${fail_count}"
echo "=================================================================="

if [[ $fail_count -gt 0 ]]; then
    exit 1
fi
