import argparse
import json
import os
import sys
from datetime import datetime
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.file_io import load_jsonl, ensure_header, append_row
from utils.evaluation_helpers import compute_retrieval_failures, evaluate_single
from utils.loaders import load_dataset

SKIP_EVAL_BASENAMES = {
    "layer_stability.jsonl",
    "retrieval_failure_analysis.jsonl",
}

def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate predictions and append to leaderboard."
    )
    p.add_argument(
        "--pred_path", type=str, default=None, help="Path to JSONL predictions file"
    )
    p.add_argument(
        "--results_path",
        type=str,
        default="./data/results/raw_results.tsv",
        help="TSV leaderboard file to append results",
    )
    p.add_argument(
        "--retrieval_failure",
        action="store_true",
        help="Run retrieval failure analysis",
    )
    p.add_argument(
        "--probe_path",
        type=str,
        default=None,
        help="Probe predictions JSONL (for retrieval failure)",
    )
    p.add_argument(
        "--gen_path",
        type=str,
        default=None,
        help="Generation predictions JSONL (for retrieval failure)",
    )
    p.add_argument(
        "--gt_path",
        type=str,
        default=None,
        help="Ground truth JSON (test.json) for retrieval failure",
    )
    p.add_argument(
        "--rf_results_path",
        type=str,
        default=None,
        help="Optional output JSONL path for retrieval failure analysis",
    )
    p.add_argument(
        "--dataset_dir",
        type=str,
        default=None,
        help="Dataset directory containing test.json and images/",
    )
    p.add_argument(
        "--remove_missing",
        type=int,
        default=1,
        choices=[0, 1],
        help="If 1, skip evaluation rows whose test-set image file is missing",
    )
    p.add_argument(
        "--refresh_prompting_eval",
        type=int,
        default=1,
        choices=[0, 1],
        help="If 1, recompute prompting correctness from model_output using current evaluation logic",
    )
    p.add_argument(
        "--changed_results_dir",
        type=str,
        default="./data/results/changed/prompting",
        help="Directory where per-run JSONL files with changed prompting rows are written",
    )
    return p.parse_args()

def compute_metrics_prompting(results_list: list[dict]) -> dict:
    total = len(results_list)
    correct = sum(1 for r in results_list if r.get("correct", 0) == 1)
    acc = correct / max(total, 1)
    from collections import defaultdict

    gt_labels = set()
    per_class = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    for r in results_list:
        gt = r.get("gt_label", r.get("gt", ""))
        pred = r.get("pred_label", r.get("pred", ""))
        gt_labels.add(gt)
        if gt == pred or r.get("correct", 0) == 1:
            per_class[gt]["tp"] += 1
        else:
            per_class[gt]["fn"] += 1
            per_class[pred]["fp"] += 1
    f1_scores = []
    for label in gt_labels:
        tp = per_class[label]["tp"]
        fp = per_class[label]["fp"]
        fn = per_class[label]["fn"]
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0
        )
        f1_scores.append(f1)
    macro_f1 = sum(f1_scores) / max(len(f1_scores), 1)
    return {"accuracy": acc, "macro_f1": macro_f1, "total": total, "correct": correct}

def compute_metrics_probing(results_list: list[dict]) -> dict:
    from sklearn.metrics import accuracy_score, f1_score

    gts = [r["gt"] for r in results_list]
    preds = [r["pred"] for r in results_list]
    acc = float(accuracy_score(gts, preds))
    f1m = float(f1_score(gts, preds, average="macro", zero_division=0))
    return {"accuracy": acc, "macro_f1": f1m, "total": len(gts)}

def infer_dataset_dir(pred_path: str) -> str | None:
    parts = os.path.normpath(pred_path).split(os.sep)
    try:
        root_idx = parts.index("sample_generations")
    except ValueError:
        return None
    if root_idx + 2 >= len(parts):
        return None
    dataset_name = parts[root_idx + 2]
    candidate = os.path.join(".", "samples", dataset_name)
    return candidate if os.path.exists(os.path.join(candidate, "test.json")) else None

def infer_model_stem(pred_path: str) -> Optional[str]:
    parts = os.path.normpath(pred_path).split(os.sep)
    try:
        root_idx = parts.index("sample_generations")
    except ValueError:
        return None
    if root_idx + 1 >= len(parts):
        return None
    return parts[root_idx + 1]

def infer_dataset_name(pred_path: str) -> Optional[str]:
    parts = os.path.normpath(pred_path).split(os.sep)
    try:
        root_idx = parts.index("sample_generations")
    except ValueError:
        return None
    if root_idx + 2 >= len(parts):
        return None
    return parts[root_idx + 2]

def infer_config_name(pred_path: str) -> str:
    return os.path.splitext(os.path.basename(pred_path))[0]

def get_missing_test_image_indices(dataset_dir: str) -> tuple[set[int], int]:
    test_path = os.path.join(dataset_dir, "test.json")
    with open(test_path, "r", encoding="utf-8") as f:
        test_rows = json.load(f)
    missing_indices: set[int] = set()
    for idx, row in enumerate(test_rows):
        image_rel = row.get("image")
        image_path = os.path.join(dataset_dir, image_rel) if image_rel else None
        if not image_path or not os.path.exists(image_path):
            missing_indices.add(idx)
    return missing_indices, len(test_rows)

def filter_results_for_missing_images(
    results_list: list[dict],
    dataset_dir: str | None,
    remove_missing: bool,
) -> tuple[list[dict], dict]:
    stats = {
        "dataset_total": None,
        "processed": len(results_list),
        "skipped": 0,
        "remove_missing": bool(remove_missing),
        "dataset_dir": dataset_dir,
    }
    if not remove_missing:
        return results_list, stats
    if dataset_dir is None:
        raise ValueError(
            "--remove_missing=1 requires --dataset_dir or an inferable dataset path from --pred_path"
        )
    test_path = os.path.join(dataset_dir, "test.json")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"test.json not found for remove_missing: {test_path}")
    missing_indices, dataset_total = get_missing_test_image_indices(dataset_dir)
    filtered = [row for row in results_list if row.get("idx") not in missing_indices]
    stats.update(
        dataset_total=dataset_total,
        processed=len(filtered),
        skipped=len(results_list) - len(filtered),
    )
    return filtered, stats

def refresh_prompting_results(
    results_list: list[dict],
    pred_path: str,
    dataset_dir: str | None,
) -> tuple[list[dict], list[dict]]:
    if dataset_dir is None:
        raise ValueError("dataset_dir is required to refresh prompting evaluation")
    _, test_data = load_dataset(
        dataset_dir, dataset_name=os.path.basename(os.path.normpath(dataset_dir))
    )
    model_stem = infer_model_stem(pred_path)
    refreshed_results: list[dict] = []
    changed_rows: list[dict] = []
    for row in results_list:
        idx = row.get("idx")
        if not isinstance(idx, int):
            refreshed_results.append(row)
            continue
        if idx < 0 or idx >= len(test_data):
            refreshed_results.append(row)
            continue
        if "model_output" not in row:
            refreshed_results.append(row)
            continue
        options_mode = row.get("options_mode", "with_options")
        options_order = row.get("options_order", "default")
        refreshed = evaluate_single(
            model_output=row.get("model_output", ""),
            sample=test_data[idx],
            options_mode=options_mode,
            options_order=options_order,
            use_gpt_judge=False,
            model_stem=model_stem,
        )
        merged = dict(row)
        merged.update(
            {
                "gt_label": refreshed.get("gt_label", row.get("gt_label")),
                "gt_answer": refreshed.get("gt_answer", row.get("gt_answer")),
                "correct": refreshed.get("correct", row.get("correct", 0)),
                "pred_label": refreshed.get("pred_label", row.get("pred_label")),
                "pred_text": refreshed.get("pred_text", row.get("pred_text")),
                "method": refreshed.get("method", row.get("method")),
            }
        )
        refreshed_results.append(merged)
        old_correct = row.get("correct")
        new_correct = merged.get("correct")
        old_pred_label = row.get("pred_label")
        new_pred_label = merged.get("pred_label")
        old_pred_text = row.get("pred_text")
        new_pred_text = merged.get("pred_text")
        old_method = row.get("method")
        new_method = merged.get("method")
        if (
            old_correct != new_correct
            or old_pred_label != new_pred_label
            or old_pred_text != new_pred_text
            or old_method != new_method
        ):
            changed_rows.append(
                {
                    "idx": idx,
                    "model": model_stem,
                    "dataset": os.path.basename(os.path.normpath(dataset_dir)),
                    "pred_path": pred_path,
                    "options_mode": row.get("options_mode"),
                    "options_order": row.get("options_order", "default"),
                    "old_correct": old_correct,
                    "new_correct": new_correct,
                    "old_pred_label": old_pred_label,
                    "new_pred_label": new_pred_label,
                    "old_pred_text": old_pred_text,
                    "new_pred_text": new_pred_text,
                    "old_method": old_method,
                    "new_method": new_method,
                    "gt_label": merged.get("gt_label"),
                    "gt_answer": merged.get("gt_answer"),
                    "model_output": row.get("model_output"),
                }
            )
    return refreshed_results, changed_rows

def write_changed_prompting_rows(
    changed_rows: list[dict],
    pred_path: str,
    changed_results_dir: str,
) -> str | None:
    if not changed_rows:
        return None
    model_stem = infer_model_stem(pred_path) or "unknown_model"
    dataset_name = infer_dataset_name(pred_path) or "unknown_dataset"
    config_name = infer_config_name(pred_path)
    out_dir = os.path.join(changed_results_dir, dataset_name, config_name)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{model_stem}.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for row in changed_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return out_path

def run_retrieval_failure_analysis(probe_path, gen_path, gt_path, rf_results_path=None):
    probe_preds = load_jsonl(probe_path)
    gen_preds = load_jsonl(gen_path)
    with open(gt_path, "r") as f:
        gt_data = json.load(f)
    gt_labels = {i: entry["answer_label"] for i, entry in enumerate(gt_data)}
    results = compute_retrieval_failures(probe_preds, gen_preds, gt_labels)
    print("=" * 70)
    print("RETRIEVAL FAILURE ANALYSIS")
    print("=" * 70)
    print(f"Probe path: {probe_path}")
    print(f"Gen path:   {gen_path}")
    print(f"Total samples analyzed: {results['rates']['total_samples']}")
    print(
        f"  Both correct (probe+gen):   {results['rates']['both_correct_rate']:.2%} ({len(results['both_correct'])})"
    )
    print(
        f"  Retrieval failure (probe=ok, gen=wrong): {results['rates']['retrieval_failure_rate']:.2%} ({len(results['retrieval_failure'])})"
    )
    print(
        f"  Gen-only correct (probe=wrong, gen=ok):  {results['rates']['gen_only_correct_rate']:.2%} ({len(results['gen_only_correct'])})"
    )
    print(
        f"  Both wrong:                 {results['rates']['both_wrong_rate']:.2%} ({len(results['both_wrong'])})"
    )
    print("=" * 70)
    if rf_results_path:
        out_path = rf_results_path
    else:
        out_dir = os.path.dirname(probe_path)
        out_path = os.path.join(out_dir, "retrieval_failure_analysis.jsonl")
    out_parent = os.path.dirname(out_path)
    if out_parent:
        os.makedirs(out_parent, exist_ok=True)
    from utils.file_io import append_jsonl

    append_jsonl(
        out_path,
        {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "probe_path": probe_path,
            "gen_path": gen_path,
            **results["rates"],
            "retrieval_failure_idxs": results["retrieval_failure"],
            "gen_only_correct_idxs": results["gen_only_correct"],
        },
    )
    print(f"Saved to: {out_path}")
    return results

def main():
    args = parse_args()
    if args.retrieval_failure:
        if not all([args.probe_path, args.gen_path, args.gt_path]):
            print(
                "[ERROR] --retrieval_failure requires --probe_path, --gen_path, and --gt_path"
            )
            sys.exit(1)
        run_retrieval_failure_analysis(
            args.probe_path,
            args.gen_path,
            args.gt_path,
            args.rf_results_path,
        )
        return
    if not args.pred_path:
        print("[ERROR] --pred_path is required (or use --retrieval_failure mode)")
        sys.exit(1)
    pred_basename = os.path.basename(args.pred_path)
    if pred_basename in SKIP_EVAL_BASENAMES:
        print(f"[SKIP] Non-evaluation file: {args.pred_path}")
        return
    if not os.path.exists(args.pred_path):
        print(f"[ERROR] File not found: {args.pred_path}")
        sys.exit(1)
    results_list = load_jsonl(args.pred_path)
    if not results_list:
        print(f"[SKIP] Empty file: {args.pred_path}")
        return
    dataset_dir = args.dataset_dir or infer_dataset_dir(args.pred_path)
    try:
        filtered_results, filter_stats = filter_results_for_missing_images(
            results_list=results_list,
            dataset_dir=dataset_dir,
            remove_missing=bool(args.remove_missing),
        )
    except Exception as e:
        print(f"[ERROR] Failed to apply remove_missing filter: {e}")
        sys.exit(1)
    if not filtered_results:
        print(
            f"[SKIP] No evaluable rows remain after remove_missing filtering: {args.pred_path}"
        )
        if filter_stats["dataset_total"] is not None:
            print(f"Dataset Total : {filter_stats['dataset_total']}")
            print(f"Processed     : {filter_stats['processed']}")
            print(f"Skipped       : {filter_stats['skipped']}")
        return
    sample = filtered_results[0]
    changed_rows_path = None
    changed_rows_count = 0
    if "correct" in sample:
        if args.refresh_prompting_eval:
            try:
                filtered_results, changed_rows = refresh_prompting_results(
                    results_list=filtered_results,
                    pred_path=args.pred_path,
                    dataset_dir=dataset_dir,
                )
                changed_rows_count = len(changed_rows)
                changed_rows_path = write_changed_prompting_rows(
                    changed_rows=changed_rows,
                    pred_path=args.pred_path,
                    changed_results_dir=args.changed_results_dir,
                )
            except Exception as e:
                print(f"[ERROR] Failed to refresh prompting evaluation: {e}")
                sys.exit(1)
        metrics = compute_metrics_prompting(filtered_results)
    else:
        metrics = compute_metrics_probing(filtered_results)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    header = ["accuracy", "macro_f1", "total", "pred_path", "timestamp"]
    os.makedirs(os.path.dirname(args.results_path), exist_ok=True)
    ensure_header(args.results_path, header)
    row = [
        f"{metrics['accuracy']:.6f}",
        f"{metrics['macro_f1']:.6f}",
        str(metrics.get("total", 0)),
        args.pred_path,
        timestamp,
    ]
    append_row(args.results_path, row)
    print(f"File     : {args.pred_path}")
    print(f"Samples  : {metrics.get('total', 0)}")
    if "correct" in sample:
        print(f"Refreshed : {int(bool(args.refresh_prompting_eval))}")
        print(f"Changed Rows : {changed_rows_count}")
        if changed_rows_path:
            print(f"Changed File : {changed_rows_path}")
    if filter_stats["dataset_total"] is not None:
        print(f"Dataset Total : {filter_stats['dataset_total']}")
        print(f"Processed     : {filter_stats['processed']}")
        print(f"Skipped       : {filter_stats['skipped']}")
    print(f"Accuracy : {metrics['accuracy']:.4f} ({metrics['accuracy']:.2%})")
    print(f"Macro F1 : {metrics['macro_f1']:.4f} ({metrics['macro_f1']:.2%})")

if __name__ == "__main__":
    main()
