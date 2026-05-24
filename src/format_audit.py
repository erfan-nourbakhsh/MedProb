import argparse
import json
import os
import random
import re
import sys
from collections import Counter, defaultdict
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.file_io import load_jsonl

def classify_output(raw_text: str) -> str:
    raw = (raw_text or "").strip().lower()
    if re.match(r"^(answer:|the answer is|option [a-d]|\b[a-d]\b)", raw):
        return "correct_format"
    if any(
        opt in raw
        for opt in [
            "option a",
            "option b",
            "option c",
            "option d",
            "choice a",
            "choice b",
            "choice c",
            "choice d",
        ]
    ):
        return "wrong_format_recoverable"
    if re.search(r"(?:the answer is|answer:)\s*\[?([a-d])\]?", raw):
        return "wrong_format_recoverable"
    words = raw.split()
    if len(words) < 3 or raw in ("", ".", "...", "i", "i don't", "i cannot"):
        return "degenerate"
    return "prose_answer"

def extract_answer_lenient(raw_text: str) -> Optional[str]:
    raw = (raw_text or "").strip()
    if not raw:
        return None
    patterns = [
        r"[Tt]he answer is\s*\[([A-E])\]",
        r"[Tt]he answer is\s*\(([A-E])\)",
        r"[Tt]he answer is\s*([A-E])\b",
        r"[Aa]nswer:\s*\[?([A-E])\]?",
        r"[Oo]ption\s+([A-E])\b",
        r"[Cc]hoice\s+([A-E])\b",
        r"\b([A-E])\b.*(?:answer|option|choice)",
        r"(?:answer|option|choice).*\b([A-E])\b",
    ]
    for pat in patterns:
        m = re.search(pat, raw, re.IGNORECASE)
        if m:
            return m.group(1).upper()
    for m in re.finditer(r"\b([A-D])\b", raw):
        return m.group(1).upper()
    return None

def compute_macro_f1(results: list[dict], pred_key: str = "pred_label") -> float:
    from collections import defaultdict

    per_class = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    gt_labels = set()
    for r in results:
        gt = r.get("gt_label", "")
        pred = r.get(pred_key, "")
        gt_labels.add(gt)
        if gt == pred:
            per_class[gt]["tp"] += 1
        else:
            per_class[gt]["fn"] += 1
            if pred:
                per_class[pred]["fp"] += 1
    f1_scores = []
    for label in sorted(gt_labels):
        tp = per_class[label]["tp"]
        fp = per_class[label]["fp"]
        fn = per_class[label]["fn"]
        p = tp / (tp + fp) if (tp + fp) > 0 else 0
        r_val = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * p * r_val / (p + r_val) if (p + r_val) > 0 else 0
        f1_scores.append(f1)
    return sum(f1_scores) / max(len(f1_scores), 1)

def run_audit(
    models: list[str],
    dataset: str,
    config: str,
    n_samples: int,
    base_dir: str,
    out_dir: str,
    seed: int = 42,
) -> dict:
    random.seed(seed)
    base_path = os.path.join(
        base_dir, "{model}", dataset, "prompting", f"{config}.jsonl"
    )
    all_results = {}
    raw_samples_dir = os.path.join(out_dir, "raw_samples")
    os.makedirs(raw_samples_dir, exist_ok=True)
    for model in models:
        path = base_path.format(model=model)
        if not os.path.exists(path):
            all_results[model] = {"error": f"File not found: {path}"}
            continue
        rows = load_jsonl(path)
        if len(rows) < n_samples:
            sampled = rows
        else:
            sampled = random.sample(rows, n_samples)
        raw_out = os.path.join(raw_samples_dir, f"{model}_{dataset}_{config}_raw.jsonl")
        with open(raw_out, "w", encoding="utf-8") as f:
            for r in sampled:
                f.write(
                    json.dumps(
                        {
                            "idx": r["idx"],
                            "gt_label": r.get("gt_label"),
                            "model_output": r.get("model_output", ""),
                            "format_category": classify_output(
                                r.get("model_output", "")
                            ),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        buckets = Counter()
        for r in sampled:
            cat = classify_output(r.get("model_output", ""))
            buckets[cat] += 1
        total = len(sampled)
        rates = {k: v / total for k, v in buckets.items()}
        results_strict = []
        results_lenient = []
        for r in sampled:
            raw = r.get("model_output", "")
            cat = classify_output(raw)
            gt = r.get("gt_label", "")
            pred_strict = r.get("pred_label", "")
            if not pred_strict:
                pred_strict = ""
            results_strict.append({"gt_label": gt, "pred_label": pred_strict})
            if cat == "correct_format" and pred_strict:
                pred_lenient = pred_strict
            else:
                pred_lenient = extract_answer_lenient(raw) or ""
            results_lenient.append({"gt_label": gt, "pred_label": pred_lenient})
        f1_strict = compute_macro_f1(results_strict)
        f1_lenient = compute_macro_f1(results_lenient)
        all_results[model] = {
            "n_samples": total,
            "format_breakdown": dict(buckets),
            "format_rates": rates,
            "correct_format_pct": rates.get("correct_format", 0),
            "wrong_format_recoverable_pct": rates.get("wrong_format_recoverable", 0),
            "degenerate_pct": rates.get("degenerate", 0),
            "prose_answer_pct": rates.get("prose_answer", 0),
            "f1_strict": f1_strict,
            "f1_lenient": f1_lenient,
            "raw_samples_path": raw_out,
        }
    return all_results

def main():
    p = argparse.ArgumentParser(
        description="Format compliance audit for biogemma-lora prompting"
    )
    p.add_argument(
        "--models",
        type=str,
        default="biogemma-lora,gemma,medgemma",
        help="Comma-separated model names",
    )
    p.add_argument("--dataset", type=str, default="PATH-VQA")
    p.add_argument("--config", type=str, default="image_question_direct_with_options")
    p.add_argument("--n_samples", type=int, default=200)
    p.add_argument("--base_dir", type=str, default="./data/sample_generations")
    p.add_argument(
        "--out_dir", type=str, default="./data/results/format_compliance_audit"
    )
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    models = [m.strip() for m in args.models.split(",")]
    results = run_audit(
        models=models,
        dataset=args.dataset,
        config=args.config,
        n_samples=args.n_samples,
        base_dir=args.base_dir,
        out_dir=args.out_dir,
        seed=args.seed,
    )
    os.makedirs(args.out_dir, exist_ok=True)
    out_json = os.path.join(
        args.out_dir, f"format_compliance_{args.dataset}_{args.config}.json"
    )
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print("=" * 70)
    print("FORMAT COMPLIANCE AUDIT")
    print("=" * 70)
    print(
        f"Dataset: {args.dataset} | Config: {args.config} | N samples: {args.n_samples}"
    )
    print("=" * 70)
    for model, r in results.items():
        if "error" in r:
            print(f"\n{model}: {r['error']}")
            continue
        print(f"\n{model}:")
        print(f"  Format breakdown: {r['format_breakdown']}")
        print(f"  Correct format:     {r['correct_format_pct']:.1%}")
        print(f"  Wrong (recoverable): {r['wrong_format_recoverable_pct']:.1%}")
        print(f"  Degenerate:          {r['degenerate_pct']:.1%}")
        print(f"  Prose answer:        {r['prose_answer_pct']:.1%}")
        print(f"  F1 (strict parser):  {r['f1_strict']:.4f}")
        print(f"  F1 (lenient parser): {r['f1_lenient']:.4f}")
    print("\n" + "=" * 70)
    print(f"Results saved to: {out_json}")
    print("=" * 70)

if __name__ == "__main__":
    main()
