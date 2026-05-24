import json
import os
from typing import Optional

from .loaders import load_dataset, VQADataset

def get_answer_label_counts_from_train(dataset_dir: str) -> dict[str, int]:
    train_path = os.path.join(dataset_dir, "train.json")
    if not os.path.exists(train_path):
        return {}
    with open(train_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    counts: dict[str, int] = {}
    for entry in data:
        label = entry.get("answer_label", "")
        counts[label] = counts.get(label, 0) + 1
    return counts

def get_test_stratum_by_answer_freq(
    dataset_dir: str,
    dataset_name: str = "PATH-VQA",
    n_buckets: int = 2,
) -> dict[int, str]:
    train_data, test_data = load_dataset(dataset_dir, dataset_name)
    label_counts = get_answer_label_counts_from_train(dataset_dir)
    if not label_counts:
        return {}
    test_counts: list[tuple[int, int]] = []
    for s in test_data.samples:
        count = label_counts.get(s.answer_label, 0)
        test_counts.append((s.idx, count))
    if not test_counts:
        return {}
    counts_only = [c for _, c in test_counts]
    counts_only.sort()
    n = len(counts_only)
    if n_buckets == 2:
        mid = n // 2
        threshold_lo = counts_only[mid - 1] if mid > 0 else 0
        threshold_hi = counts_only[mid] if mid < n else counts_only[-1]
        def bucket(c: int) -> str:
            if c <= threshold_lo:
                return "low_freq"
            return "high_freq"
    elif n_buckets == 3:
        t1 = n // 3
        t2 = 2 * (n // 3)
        thr_lo = counts_only[t1 - 1] if t1 > 0 else 0
        thr_mid = counts_only[t2 - 1] if t2 > 0 else 0
        def bucket(c: int) -> str:
            if c <= thr_lo:
                return "low_freq"
            if c <= thr_mid:
                return "mid_freq"
            return "high_freq"
    else:
        def bucket(c: int) -> str:
            return f"freq_{c}"
    return {idx: bucket(c) for idx, c in test_counts}

def get_answer_label_counts(dataset_dir: str) -> dict[str, int]:
    return get_answer_label_counts_from_train(dataset_dir)
