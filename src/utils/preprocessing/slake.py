import os
import os.path as osp
import json

import pandas as pd

import warnings

warnings.filterwarnings("ignore")
SLAKE_DIR = "/raid/rsq813/MedAG/FinalProject/Data/SLAKE/Slake1.0"
OUTPUT_DIR = "/raid/rsq813/MedAG/FinalProject/Data/SLAKE/Preprocessed"


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    slake_train_df = pd.read_json(osp.join(SLAKE_DIR, "train.json"))
    slake_val_df = pd.read_json(osp.join(SLAKE_DIR, "validate.json"))
    slake_test_df = pd.read_json(osp.join(SLAKE_DIR, "test.json"))

    for df in [slake_train_df, slake_val_df, slake_test_df]:
        mask = (df["q_lang"] == "en") & (df["answer_type"] == "CLOSED")
        df.drop(df[~mask].index, inplace=True)
        df.reset_index(drop=True, inplace=True)
        df["answer_type"] = df["answer_type"].str.lower()

    print("After filtering (English + closed-ended):")

    print(f"  Train : {len(slake_train_df)}")

    print(f"  Val   : {len(slake_val_df)}")

    print(f"  Test  : {len(slake_test_df)}")

    print(f"  Total : {len(slake_train_df) + len(slake_val_df) + len(slake_test_df)}")
    QUESTION_OPTIONS: list[tuple[str, list[str]]] = [
        ("is brain tumor white or gray relative to other tissues?", ["white", "gray"]),
        ("is the abnormality hyperdense or hypodense?", ["hyperdense", "hypodense"]),
        (
            "is the brain enhancing tumor hyperdense or hypodense?",
            ["hyperdense", "hypodense"],
        ),
        (
            "is the brain non-enhancing tumor hyperdense or hypodense?",
            ["hyperdense", "hypodense"],
        ),
        ("is this a t1 weighted or t2 weighted mri image?", ["t1", "t2"]),
        (
            "which plane is the image scanned, transverse plane or coronal plane?",
            ["transverse plane", "coronal plane"],
        ),
        (
            "which is bigger in this image,left lung or left kidney?",
            ["left lung", "left kidney"],
        ),
        ("which is bigger in this image,small bowel or kidney?", ["small bowel", "kidney"]),
        ("which is smaller in this image,liver or spinal cord?", ["liver", "spinal cord"]),
        (
            "which is smaller in this image,liver or right kidney?",
            ["liver", "right kidney"],
        ),
        (
            "which is smaller in this image,kidney or spinal cord?",
            ["kidney", "spinal cord"],
        ),
        (
            "which is smaller in this image, small bowel or right kidney?",
            ["small bowel", "right kidney"],
        ),
        (
            "which is the smallest in this image,spleen,left kidney or liver?",
            ["left kidney", "liver"],
        ),
        (
            "which is the smallest in this image, colon, left lung or liver?",
            ["colon", "left lung", "liver"],
        ),
        ("which is bigger in this image, heart or right lung?", ["heart", "right lung"]),
        (
            "which is smaller in this image, bladder or small bowel?",
            ["bladder", "small bowel"],
        ),
        ("which is bigger in this image, colon or small bowel?", ["colon", "small bowel"]),
        ("which is bigger, left kidney or spleen ?", ["left kidney", "spleen"]),
        (
            "where is the spleen in this image, right or lower right?",
            ["right", "lower right"],
        ),
        ("which is smaller in this image,liver or left lung?", ["liver", "left lung"]),
        (
            "which is bigger in this image, small bowel or kidney?",
            ["small bowel", "kidney"],
        ),
        (
            "where is the left kidney in this image, right or lower right?",
            ["right", "lower right"],
        ),
        ("which is bigger in this image,small bowel or liver?", ["small bowel", "liver"]),
        (
            "which is smaller in this image,spleen or right kidney?",
            ["spleen", "right kidney"],
        ),
        (
            "which is bigger in this image, rectum or small bowel?",
            ["rectum", "small bowel"],
        ),
        ("which is bigger in this image, kidney or spleen?", ["kidney", "spleen"]),
        ("which is bigger in this image, small bowel or colon?", ["small bowel", "colon"]),
    ]

    def resolve_options(question: str, answer: str) -> list[str] | None:
        q_lower = question.lower().strip()
        if answer in ["yes", "no"]:
            return ["yes", "no"]
        for key, opts in QUESTION_OPTIONS:
            if key in q_lower:
                return opts
        try:
            after_comma = question.split(",", 1)[1]
            option_str = after_comma.rstrip("?").replace(",", " ").replace("or", " ")
            option_str = " ".join(option_str.split())
            options = [o.lower().strip() for o in option_str.split() if o.strip()]
            if options:
                return options
        except (IndexError, AttributeError):
            pass
        return None

    def convert(df: pd.DataFrame) -> list[dict]:
        records = []
        skipped_no_options = 0
        skipped_mismatch = 0
        for _, row in df.iterrows():
            question = row["question"].capitalize().strip()
            answer = row["answer"].lower().strip()
            options = resolve_options(question, answer)
            if options is None or len(options) == 0:
                skipped_no_options += 1
                print(f"[SKIP – no options]  Q: {question}  |  A: {answer}")
                continue
            if answer not in options:
                if answer in ["both", "almost the same"]:
                    options = options + [answer]
                else:
                    skipped_mismatch += 1
                    print(
                        f"[SKIP – mismatch]  Q: {question}  |  "
                        f"Options: {options}  |  A: {answer}"
                    )
                    continue
            records.append(
                dict(
                    id=osp.splitext(row["img_name"])[0],
                    image=row["img_name"],
                    question=question,
                    answer=answer,
                    options=options,
                )
            )
        if skipped_no_options:
            print(f"  → Skipped (no options resolved): {skipped_no_options}")
        if skipped_mismatch:
            print(f"  → Skipped (answer ∉ options):    {skipped_mismatch}")
        return records
    train_records = convert(slake_train_df)
    val_records = convert(slake_val_df)
    test_records = convert(slake_test_df)

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

    print("SLAKE  |  Closed-ended splits (English only)")

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

    SLAKE_IMAGE_SRC = "/raid/rsq813/MedAG/FinalProject/Data/SLAKE/Slake1.0/imgs"
    SLAKE_IMAGE_DST = osp.join(OUTPUT_DIR, "images")

    shutil.copytree(SLAKE_IMAGE_SRC, SLAKE_IMAGE_DST)

    print(f"  Copied images to : {SLAKE_IMAGE_DST}")

if __name__ == "__main__":
    main()
