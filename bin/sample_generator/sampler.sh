#!/bin/bash

set -euo pipefail

export BASE_DATA="/raid/rsq813/MedAG/FinalProject/Data"
export OUTPUT_BASE="/raid/rsq813/MedAG/FinalProject/all_samples"
export DATASETS="PATH-VQA,SLAKE,VQA-RAD"
export TRAIN_SIZE=-1
export TEST_SIZE=-1
export RANDOM_SEED=42

echo "============================================"
echo "  Medical VQA Full Export Pipeline"
echo "============================================"
echo "Train size : all"
echo "Test size  : all"
echo "Seed       : $RANDOM_SEED"
echo ""

python3 - <<'PYEOF'
import json
import os
import shutil
import random
import math
from collections import defaultdict
from datetime import datetime

BASE_DATA   = os.environ.get("BASE_DATA",   "/raid/rsq813/MedAG/FinalProject/Data")
OUTPUT_BASE = os.environ.get("OUTPUT_BASE", "/raid/rsq813/MedAG/FinalProject/samples")
DATASETS    = os.environ.get("DATASETS",    "PATH-VQA,SLAKE,VQA-RAD").split(",")
TRAIN_SIZE  = int(os.environ.get("TRAIN_SIZE", -1))
TEST_SIZE   = int(os.environ.get("TEST_SIZE",  -1))
SEED        = int(os.environ.get("RANDOM_SEED", 42))

random.seed(SEED)

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
RUN_TS  = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def read_jsonl(path):
    if not os.path.exists(path):
        print(f"  [WARN] File not found, skipping: {path}")
        return []
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"  [WARN] JSON parse error at line {line_no}: {e}")
    return records

def build_options_dict(options_list):
    return {LETTERS[i]: opt for i, opt in enumerate(options_list)}

def find_answer_label(options_dict, answer):
    answer_lower = str(answer).strip().lower()
    for letter, opt in options_dict.items():
        if str(opt).strip().lower() == answer_lower:
            return letter
    if answer.upper() in options_dict:
        return answer.upper()
    return "A"

def format_record(raw, source_split):
    image_filename = raw.get("image", "")
    if not image_filename.startswith("images/"):
        image_filename = f"images/{image_filename}"
    options_list = raw.get("options", [])
    options_dict = build_options_dict(options_list)
    answer       = str(raw.get("answer", "")).strip()
    answer_label = find_answer_label(options_dict, answer)
    return {
        "image":         image_filename,
        "question":      raw.get("question", ""),
        "options":       options_dict,
        "answer":        answer,
        "answer_label":  answer_label,
        "_source_split": source_split,
    }

def stratified_sample(records, n, key="answer"):
    """Sample n records from records, stratified by key. n<=0 means use all."""
    if n <= 0 or n >= len(records):
        requested = "all" if n <= 0 else str(n)
        print(f"  [INFO] Requested {requested}; using all {len(records):,} records.")
        result = records.copy()
        random.shuffle(result)
        return result
    strata    = defaultdict(list)
    for rec in records:
        strata[str(rec.get(key, "unknown")).strip().lower()].append(rec)
    total     = len(records)
    quotas    = {k: n * len(v) / total for k, v in strata.items()}
    floors    = {k: math.floor(q)      for k, q in quotas.items()}
    remainder = n - sum(floors.values())
    fracs     = sorted(quotas.keys(), key=lambda k: -(quotas[k] - floors[k]))
    for i in range(remainder):
        floors[fracs[i]] += 1
    sampled = []
    for stratum, quota in floors.items():
        pool = strata[stratum].copy()
        random.shuffle(pool)
        sampled.extend(pool[:quota])
    random.shuffle(sampled)
    return sampled

def answer_dist(data):
    dist = defaultdict(int)
    for r in data:
        dist[r["answer"].lower()] += 1
    return dict(sorted(dist.items()))

def build_dataset_stats(dataset, split_counts, train_data, test_data):
    W = 54
    L = []
    L.append("=" * W)
    L.append(f"  Statistics  :  {dataset}")
    L.append(f"  Generated   :  {RUN_TS}")
    L.append(f"  Random seed :  {SEED}")
    L.append("=" * W)
    L.append("")
    L.append("  ORIGINAL DATASET  (per split, NOT pooled)")
    L.append(f"  {'train.jsonl':<16}:  {split_counts.get('train', 0):>7,}")
    L.append(f"  {'val.jsonl':<16}:  {split_counts.get('val',   0):>7,}  (discarded)")
    L.append(f"  {'test.jsonl':<16}:  {split_counts.get('test',  0):>7,}")
    L.append(f"  {'─'*32}")
    orig_used = split_counts.get('train', 0) + split_counts.get('test', 0)
    L.append(f"  {'Used (train+test)':<16}:  {orig_used:>7,}")
    L.append("")
    L.append("  EXPORTED OUTPUT")
    L.append(f"  {'train.json':<16}:  {len(train_data):>7,}  (from train.jsonl)")
    L.append(f"  {'test.json':<16}:  {len(test_data):>7,}  (from test.jsonl)")
    L.append(f"  {'─'*32}")
    L.append(f"  {'Total':<16}:  {len(train_data) + len(test_data):>7,}")
    L.append("")
    for split_name, data in [("train", train_data), ("test", test_data)]:
        dist = answer_dist(data)
        n    = len(data)
        L.append(f"  Answer distribution  [{split_name}]  (n={n:,})")
        for ans, cnt in dist.items():
            pct = 100 * cnt / n if n else 0
            bar = "█" * int(pct / 2)
            L.append(f"    {ans:<22}:  {cnt:>5,}  ({pct:5.1f}%)  {bar}")
        L.append("")
    L.append("=" * W)
    return "\n".join(L) + "\n"

def build_global_summary(summary):
    W = 12
    L = []
    L.append(f"{'='*74}")
    L.append(f"  DATASET SUMMARY  —  {RUN_TS}  (seed={SEED})")
    L.append(f"{'='*74}")
    L.append(
        f"  {'Dataset':<14}"
        f"{'Orig Train':>{W}}"
        f"{'Orig Val':>{W}}"
        f"{'Orig Test':>{W}}"
        f"  │"
        f"{'Export Train':>{W}}"
        f"{'Export Test':>{W}}"
    )
    L.append(f"  {'─'*14}{'─'*W}{'─'*W}{'─'*W}  │{'─'*W}{'─'*W}")

    g_otr = g_oval = g_ote = g_str = g_ste = 0
    for ds, s in summary.items():
        otr  = s.get("train", 0);  oval = s.get("val", 0);  ote = s.get("test", 0)
        str_ = s.get("sampled_train", 0);  ste = s.get("sampled_test", 0)
        g_otr += otr;  g_oval += oval;  g_ote += ote
        g_str += str_; g_ste  += ste
        L.append(
            f"  {ds:<14}"
            f"{otr:>{W},}{oval:>{W},}{ote:>{W},}"
            f"  │{str_:>{W},}{ste:>{W},}"
        )

    L.append(f"  {'─'*14}{'─'*W}{'─'*W}{'─'*W}  │{'─'*W}{'─'*W}")
    L.append(
        f"  {'TOTAL':<14}"
        f"{g_otr:>{W},}{g_oval:>{W},}{g_ote:>{W},}"
        f"  │{g_str:>{W},}{g_ste:>{W},}"
    )
    L.append(f"{'='*74}")
    return "\n".join(L) + "\n"

summary       = {}
dataset_stats = {}

for dataset in DATASETS:
    print(f"\n{'─'*50}")
    print(f"  Dataset: {dataset}")
    print(f"{'─'*50}")

    preprocessed_dir = os.path.join(BASE_DATA, dataset, "Preprocessed")
    output_dir       = os.path.join(OUTPUT_BASE, dataset)
    os.makedirs(output_dir, exist_ok=True)

    split_counts = {}
    split_data   = {}

    for split_file, split_name in [("train.jsonl", "train"),
                                    ("val.jsonl",   "val"),
                                    ("test.jsonl",  "test")]:
        path = os.path.join(preprocessed_dir, split_file)
        recs = read_jsonl(path)
        for r in recs:
            r["_raw_source_split"] = split_name
        split_counts[split_name] = len(recs)
        split_data[split_name]   = recs
        status = "  (will be discarded)" if split_name == "val" else ""
        print(f"  Loaded {len(recs):>6,} records from {split_file}{status}")

    train_recs = split_data.get("train", [])
    test_recs  = split_data.get("test",  [])

    if not train_recs and not test_recs:
        print("  [ERROR] No records found — skipping dataset.")
        summary[dataset] = {**split_counts,
                            "sampled_train": 0, "sampled_test": 0}
        continue

    train_raw = stratified_sample(train_recs, TRAIN_SIZE, key="answer")
    test_raw  = stratified_sample(test_recs,  TEST_SIZE,  key="answer")

    print(f"  Exported → train: {len(train_raw):,}  (from {len(train_recs):,} originals)")
    print(f"  Exported → test:  {len(test_raw):,}  (from {len(test_recs):,} originals)")

    def fmt_list(recs):
        return [format_record(r, r.get("_raw_source_split", "unknown")) for r in recs]

    train_fmt = fmt_list(train_raw)
    test_fmt  = fmt_list(test_raw)

    for split_name, data in [("train", train_fmt), ("test", test_fmt)]:
        out_path = os.path.join(output_dir, f"{split_name}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"  Wrote  {out_path}  ({len(data):,} records)")

    stats_text = build_dataset_stats(dataset, split_counts, train_fmt, test_fmt)
    print(stats_text)

    stats_path = os.path.join(output_dir, "statistics.txt")
    with open(stats_path, "w", encoding="utf-8") as f:
        f.write(stats_text)
    print(f"  Saved  {stats_path}")

    src_images = os.path.join(preprocessed_dir, "images")
    dst_images = os.path.join(output_dir, "images")

    if os.path.isdir(src_images):
        if os.path.exists(dst_images):
            print(f"  [INFO] images/ already exists at destination — removing old copy.")
            shutil.rmtree(dst_images)
        print(f"  Copying images/  {src_images}  →  {dst_images}")
        shutil.copytree(src_images, dst_images)
        n_imgs = sum(len(files) for _, _, files in os.walk(dst_images))
        print(f"  Copied {n_imgs:,} image files.")
    else:
        print(f"  [WARN] No images/ folder found at: {src_images}")

    summary[dataset] = {
        **split_counts,
        "sampled_train": len(train_fmt),
        "sampled_test":  len(test_fmt),
    }
    dataset_stats[dataset] = stats_text

global_table = build_global_summary(summary)
print(global_table)

global_stats_path = os.path.join(OUTPUT_BASE, "statistics.txt")
os.makedirs(OUTPUT_BASE, exist_ok=True)
with open(global_stats_path, "w", encoding="utf-8") as f:
    f.write(global_table)
    f.write("\n\n")
    f.write("─" * 74 + "\n")
    f.write("  PER-DATASET DETAILS\n")
    f.write("─" * 74 + "\n\n")
    for ds in DATASETS:
        if ds in dataset_stats:
            f.write(dataset_stats[ds])
            f.write("\n")

print(f"✓ Global statistics saved  →  {global_stats_path}")
print(f"✓ All datasets processed successfully.")
print(f"  Output root: {OUTPUT_BASE}")

PYEOF

echo ""
echo "Done."
