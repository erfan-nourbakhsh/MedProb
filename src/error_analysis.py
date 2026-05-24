import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.file_io import load_jsonl

OPTIONS = ["A", "B", "C", "D"]
MAX_ENTROPY_4 = np.log(4)

def analyze_errors(
    predictions: list[str],
    gold_labels: list[str],
    options: list[str] = OPTIONS,
) -> dict:
    from collections import Counter

    predictions = [p.strip().upper() if p else "" for p in predictions]
    gold_labels = [g.strip().upper() if g else "" for g in gold_labels]
    errors = [(p, g) for p, g in zip(predictions, gold_labels) if p != g and p and g]
    wrong_predictions = Counter(e[0] for e in errors)
    all_preds = Counter(p for p in predictions if p in options)
    total = len([p for p in predictions if p in options])
    if total == 0:
        entropy = 0.0
    else:
        probs = [all_preds[opt] / total for opt in options]
        entropy = -sum(p * np.log(p) for p in probs if p > 0)
    return {
        "wrong_predictions": dict(wrong_predictions),
        "all_predictions": dict(all_preds),
        "entropy": float(entropy),
        "max_entropy": float(MAX_ENTROPY_4),
        "entropy_ratio": float(entropy / MAX_ENTROPY_4) if MAX_ENTROPY_4 > 0 else 0,
        "n_errors": len(errors),
        "n_total": total,
        "error_rate": len(errors) / max(total, 1),
    }

def run_analysis(
    models: list[str],
    dataset: str,
    config: str,
    base_dir: str,
) -> dict:
    base_path = os.path.join(
        base_dir, "{model}", dataset, "prompting", f"{config}.jsonl"
    )
    results = {}
    for model in models:
        path = base_path.format(model=model)
        if not os.path.exists(path):
            results[model] = {"error": f"File not found: {path}"}
            continue
        rows = load_jsonl(path)
        predictions = [r.get("pred_label", "") for r in rows]
        gold_labels = [r.get("gt_label", "") for r in rows]
        results[model] = analyze_errors(predictions, gold_labels)
    return results

def main():
    p = argparse.ArgumentParser(
        description="Error distribution analysis (entropy, wrong-answer bias)"
    )
    p.add_argument("--models", type=str, default="biogemma-lora,gemma,medgemma")
    p.add_argument("--dataset", type=str, default="PATH-VQA")
    p.add_argument("--config", type=str, default="image_question_direct_with_options")
    p.add_argument("--base_dir", type=str, default="./data/sample_generations")
    p.add_argument(
        "--out_dir", type=str, default="./data/results/format_compliance_audit"
    )
    args = p.parse_args()
    models = [m.strip() for m in args.models.split(",")]
    results = run_analysis(
        models=models,
        dataset=args.dataset,
        config=args.config,
        base_dir=args.base_dir,
    )
    os.makedirs(args.out_dir, exist_ok=True)
    out_json = os.path.join(
        args.out_dir, f"error_distribution_{args.dataset}_{args.config}.json"
    )
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print("=" * 70)
    print("ERROR DISTRIBUTION ANALYSIS (format–content dissociation)")
    print("=" * 70)
    print(f"Dataset: {args.dataset} | Config: {args.config}")
    print(f"Max entropy (4 options): {MAX_ENTROPY_4:.3f}")
    print("=" * 70)
    for model, r in results.items():
        if "error" in r:
            print(f"\n{model}: {r['error']}")
            continue
        print(f"\n{model}:")
        print(
            f"  Wrong-answer distribution (when pred != gold): {r['wrong_predictions']}"
        )
        print(f"  All-prediction distribution: {r['all_predictions']}")
        print(f"  Prediction entropy: {r['entropy']:.3f} (max {r['max_entropy']:.3f})")
        print(f"  Entropy ratio (entropy/max): {r['entropy_ratio']:.3f}")
        print(f"  Errors: {r['n_errors']} / {r['n_total']} ({r['error_rate']:.1%})")
    print("\n" + "=" * 70)
    print(
        "Interpretation: Low entropy → output collapsed to few options (e.g. over-regularization)."
    )
    print(
        "Compare biogemma-lora vs gemma/medgemma; low entropy supports output distribution collapse."
    )
    print("=" * 70)
    print(f"Results saved to: {out_json}")

if __name__ == "__main__":
    main()
