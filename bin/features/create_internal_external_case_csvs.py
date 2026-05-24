from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SAMPLES_ROOT = PROJECT_ROOT / "data" / "sample_generations"
DATASETS_ROOT = PROJECT_ROOT / "samples"
OUTPUT_ROOT = PROJECT_ROOT / "data" / "results" / "internal_external_cases"
CONFIGS = {
    "default": "image_question_direct_with_options",
    "correct_first": "image_question_direct_with_options_correct_first",
    "correct_end": "image_question_direct_with_options_correct_end",
}
DETAIL_HEADERS = [
    "model",
    "dataset",
    "options_order",
    "config_name",
    "best_layer",
    "idx",
    "gt_label",
    "gt_answer",
    "internal_pred_label",
    "internal_correct",
    "external_pred_label",
    "external_correct",
    "case_label",
]
SUMMARY_HEADERS = [
    "model",
    "dataset",
    "options_order",
    "config_name",
    "best_layer",
    "num_rows",
    "skipped_missing_images",
    "internal_correct_external_correct",
    "internal_correct_external_wrong",
    "internal_wrong_external_correct",
    "internal_wrong_external_wrong",
]
AGGREGATE_HEADERS = [
    "model",
    "options_order",
    "num_datasets",
    "num_rows",
    "skipped_missing_images",
    "internal_correct_external_correct",
    "internal_correct_external_wrong",
    "internal_wrong_external_correct",
    "internal_wrong_external_wrong",
]

def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                rows.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSONL in {path} at line {line_number}"
                ) from exc
    return rows

def load_best_layer(stability_path: Path) -> int:
    records = load_jsonl(stability_path)
    if not records:
        raise ValueError(f"No records found in {stability_path}")
    best_layer = records[0].get("best_layer")
    if best_layer is None:
        raise ValueError(f"Missing best_layer in {stability_path}")
    return int(best_layer)

def build_prompt_index(records: list[dict]) -> dict[int, dict]:
    indexed: dict[int, dict] = {}
    for record in records:
        idx = record.get("idx")
        if idx is None:
            continue
        indexed[int(idx)] = record
    return indexed

def bool_to_int(value: bool) -> int:
    return 1 if value else 0

def case_label(internal_correct: int, external_correct: int) -> str:
    if internal_correct and external_correct:
        return "internal_correct_external_correct"
    if internal_correct and not external_correct:
        return "internal_correct_external_wrong"
    if not internal_correct and external_correct:
        return "internal_wrong_external_correct"
    return "internal_wrong_external_wrong"

def write_tsv(path: Path, headers: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        import csv

        writer = csv.DictWriter(handle, fieldnames=headers, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

def get_missing_test_image_indices(dataset_name: str) -> set[int]:
    dataset_dir = DATASETS_ROOT / dataset_name
    test_path = dataset_dir / "test.json"
    if not test_path.exists():
        raise FileNotFoundError(
            f"test.json not found for dataset {dataset_name}: {test_path}"
        )
    with test_path.open("r", encoding="utf-8") as handle:
        test_rows = json.load(handle)
    missing_indices: set[int] = set()
    for idx, row in enumerate(test_rows):
        image_rel = row.get("image")
        image_path = dataset_dir / image_rel if image_rel else None
        if image_path is None or not image_path.exists():
            missing_indices.add(idx)
    return missing_indices

def collect_run_rows(
    model_dir: Path, dataset_dir: Path, options_order: str, config_name: str
) -> tuple[list[dict], dict] | None:
    prompt_path = dataset_dir / "prompting" / f"{config_name}.jsonl"
    probe_dir = dataset_dir / "answer_decoding_logistic" / config_name
    stability_path = probe_dir / "layer_stability.jsonl"
    if not prompt_path.exists() or not stability_path.exists():
        return None
    best_layer = load_best_layer(stability_path)
    probe_path = probe_dir / f"layer{best_layer}.jsonl"
    if not probe_path.exists():
        return None
    prompt_index = build_prompt_index(load_jsonl(prompt_path))
    probe_records = load_jsonl(probe_path)
    missing_indices = get_missing_test_image_indices(dataset_dir.name)
    detail_rows: list[dict] = []
    counts: Counter[str] = Counter()
    skipped_missing_images = 0
    for probe_record in probe_records:
        idx = probe_record.get("idx")
        if idx is None:
            continue
        idx = int(idx)
        if idx in missing_indices:
            skipped_missing_images += 1
            continue
        prompt_record = prompt_index.get(idx)
        if prompt_record is None:
            continue
        internal_correct = bool_to_int(
            probe_record.get("gt_label") == probe_record.get("pred_label")
        )
        external_correct = int(prompt_record.get("correct", 0))
        row_case = case_label(internal_correct, external_correct)
        counts[row_case] += 1
        detail_rows.append(
            {
                "model": model_dir.name,
                "dataset": dataset_dir.name,
                "options_order": options_order,
                "config_name": config_name,
                "best_layer": best_layer,
                "idx": idx,
                "gt_label": prompt_record.get(
                    "gt_label", probe_record.get("gt_label", "")
                ),
                "gt_answer": prompt_record.get("gt_answer", ""),
                "internal_pred_label": probe_record.get("pred_label", ""),
                "internal_correct": internal_correct,
                "external_pred_label": prompt_record.get("pred_label", ""),
                "external_correct": external_correct,
                "case_label": row_case,
            }
        )
    detail_rows.sort(key=lambda row: row["idx"])
    summary_row = {
        "model": model_dir.name,
        "dataset": dataset_dir.name,
        "options_order": options_order,
        "config_name": config_name,
        "best_layer": best_layer,
        "num_rows": len(detail_rows),
        "skipped_missing_images": skipped_missing_images,
        "internal_correct_external_correct": counts[
            "internal_correct_external_correct"
        ],
        "internal_correct_external_wrong": counts["internal_correct_external_wrong"],
        "internal_wrong_external_correct": counts["internal_wrong_external_correct"],
        "internal_wrong_external_wrong": counts["internal_wrong_external_wrong"],
    }
    return detail_rows, summary_row

def main() -> None:
    summary_rows: list[dict] = []
    written_files = 0
    for model_dir in sorted(path for path in SAMPLES_ROOT.iterdir() if path.is_dir()):
        for dataset_dir in sorted(
            path for path in model_dir.iterdir() if path.is_dir()
        ):
            for options_order, config_name in CONFIGS.items():
                collected = collect_run_rows(
                    model_dir, dataset_dir, options_order, config_name
                )
                if collected is None:
                    continue
                detail_rows, summary_row = collected
                if not detail_rows:
                    continue
                out_path = (
                    OUTPUT_ROOT
                    / model_dir.name
                    / dataset_dir.name
                    / f"{options_order}.tsv"
                )
                write_tsv(out_path, DETAIL_HEADERS, detail_rows)
                summary_rows.append(summary_row)
                written_files += 1
    summary_rows.sort(
        key=lambda row: (row["model"], row["dataset"], row["options_order"])
    )
    write_tsv(OUTPUT_ROOT / "summary.tsv", SUMMARY_HEADERS, summary_rows)
    aggregate_rows: list[dict] = []
    grouped: dict[tuple[str, str], Counter] = {}
    for row in summary_rows:
        key = (row["model"], row["options_order"])
        if key not in grouped:
            grouped[key] = Counter()
        grouped[key]["num_datasets"] += 1
        grouped[key]["num_rows"] += int(row["num_rows"])
        grouped[key]["skipped_missing_images"] += int(row["skipped_missing_images"])
        grouped[key]["internal_correct_external_correct"] += int(
            row["internal_correct_external_correct"]
        )
        grouped[key]["internal_correct_external_wrong"] += int(
            row["internal_correct_external_wrong"]
        )
        grouped[key]["internal_wrong_external_correct"] += int(
            row["internal_wrong_external_correct"]
        )
        grouped[key]["internal_wrong_external_wrong"] += int(
            row["internal_wrong_external_wrong"]
        )
    for (model, options_order), counts in sorted(grouped.items()):
        aggregate_rows.append(
            {
                "model": model,
                "options_order": options_order,
                "num_datasets": counts["num_datasets"],
                "num_rows": counts["num_rows"],
                "skipped_missing_images": counts["skipped_missing_images"],
                "internal_correct_external_correct": counts[
                    "internal_correct_external_correct"
                ],
                "internal_correct_external_wrong": counts[
                    "internal_correct_external_wrong"
                ],
                "internal_wrong_external_correct": counts[
                    "internal_wrong_external_correct"
                ],
                "internal_wrong_external_wrong": counts[
                    "internal_wrong_external_wrong"
                ],
            }
        )
    write_tsv(
        OUTPUT_ROOT / "summary_by_model_options_order.tsv",
        AGGREGATE_HEADERS,
        aggregate_rows,
    )
    print(f"Wrote {written_files} detailed TSV files to {OUTPUT_ROOT}")
    print(f"Wrote summary TSV to {OUTPUT_ROOT / 'summary.tsv'}")
    print(
        f"Wrote aggregate summary TSV to {OUTPUT_ROOT / 'summary_by_model_options_order.tsv'}"
    )

if __name__ == "__main__":
    main()
