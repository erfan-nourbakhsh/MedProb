import os
import os.path as osp
import re
import json
import pickle

import pandas as pd

import warnings

warnings.filterwarnings("ignore")
PVQA_DIR = "/raid/rsq813/MedAG/FinalProject/Data/PATH-VQA/pvqa"
OUTPUT_DIR = "/raid/rsq813/MedAG/FinalProject/Data/PATH-VQA/Preprocessed"


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    pvqa_train_pkl = pickle.load(open(osp.join(PVQA_DIR, "qas/train/train_qa.pkl"), "rb"))
    pvqa_val_pkl = pickle.load(open(osp.join(PVQA_DIR, "qas/val/val_qa.pkl"), "rb"))
    pvqa_test_pkl = pickle.load(open(osp.join(PVQA_DIR, "qas/test/test_qa.pkl"), "rb"))
    train_df = pd.DataFrame.from_records(pvqa_train_pkl)
    val_df = pd.DataFrame.from_records(pvqa_val_pkl)
    test_df = pd.DataFrame.from_records(pvqa_test_pkl)

    for df in [train_df, val_df, test_df]:
        df["image"] = df["image"].apply(lambda x: f"{x}.jpg")
    _is_closed = lambda x: (
        len(re.findall(r"\byes\b|\bno\b", x)) > 0 and x.lower().strip() in ["yes", "no"]
    )
    train_df["answer_type"] = train_df["answer"].apply(
        lambda x: "closed" if _is_closed(x) else "open"
    )
    val_df["answer_type"] = val_df["answer"].apply(
        lambda x: "closed" if _is_closed(x) else "open"
    )
    test_df["answer_type"] = test_df["answer"].apply(
        lambda x: "closed" if _is_closed(x) else "open"
    )

    print("Before filtering:")

    print(f"  Total QA pairs : {train_df.shape[0] + val_df.shape[0] + test_df.shape[0]}")

    print(
        f"  Closed-ended   : "
        f"{(train_df['answer_type']=='closed').sum() + (val_df['answer_type']=='closed').sum() + (test_df['answer_type']=='closed').sum()}"
    )
    train_df = train_df[train_df["answer_type"] == "closed"].reset_index(drop=True)
    val_df = val_df[val_df["answer_type"] == "closed"].reset_index(drop=True)
    test_df = test_df[test_df["answer_type"] == "closed"].reset_index(drop=True)

    def convert(df: pd.DataFrame) -> list[dict]:
        records = []
        for _, row in df.iterrows():
            question = row["question"].capitalize()
            if not question.endswith("?"):
                continue
            answer = row["answer"].lower().strip()
            options = ["yes", "no"]
            records.append(
                dict(
                    id=osp.splitext(row["image"])[0],
                    image=row["image"],
                    question=question.strip(),
                    answer=answer.strip(),
                    options=options,
                )
            )
        return records
    train_records = convert(train_df)
    val_records = convert(val_df)
    test_records = convert(test_df)

    def write_jsonl(data: list[dict], path: str) -> None:
        with open(path, "w") as fh:
            for record in data:
                json.dump(record, fh)
                fh.write("\n")

    def read_jsonl(path: str) -> list[dict]:
        with open(path, "r") as fh:
            return [json.loads(line) for line in fh]

    write_jsonl(train_records, osp.join(OUTPUT_DIR, "train.jsonl"))

    write_jsonl(val_records, osp.join(OUTPUT_DIR, "val.jsonl"))

    write_jsonl(test_records, osp.join(OUTPUT_DIR, "test.jsonl"))

    print("\n" + "=" * 50)

    print("PathVQA  |  Closed-ended splits")

    print("=" * 50)

    print(f"  Train : {len(train_records):>5} samples")

    print(f"  Val   : {len(val_records):>5} samples")

    print(f"  Test  : {len(test_records):>5} samples")

    print(
        f"  Total : {len(train_records) + len(val_records) + len(test_records):>5} samples"
    )

    print("=" * 50)

    with open(osp.join(OUTPUT_DIR, "dataset_statistics.txt"), "w") as f:
        f.write(f"Train: {len(train_records)}\n")
        f.write(f"Val: {len(val_records)}\n")
        f.write(f"Test: {len(test_records)}\n")
        f.write(f"Total: {len(train_records) + len(val_records) + len(test_records)}\n")
    import shutil

    IMAGE_SPLITS = ["train", "val", "test"]
    IMAGE_BASE_DIR = "/raid/rsq813/MedAG/FinalProject/Data/PATH-VQA/pvqa/images"
    MERGED_IMAGE_DIR = OUTPUT_DIR + "/images"

    os.makedirs(MERGED_IMAGE_DIR, exist_ok=True)
    total_copied = 0

    for split in IMAGE_SPLITS:
        split_dir = osp.join(IMAGE_BASE_DIR, split)
        files = os.listdir(split_dir)
        for fname in files:
            src = osp.join(split_dir, fname)
            dst = osp.join(MERGED_IMAGE_DIR, fname)
            if osp.isfile(src):
                shutil.copy2(src, dst)
                total_copied += 1
        print(f"  Copied {len(files)} images from '{split}' split")

    print(f"\n  Total images merged : {total_copied}")

    print(f"  Saved to            : {MERGED_IMAGE_DIR}")

if __name__ == "__main__":
    main()
