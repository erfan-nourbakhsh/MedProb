#!/bin/bash
set -euo pipefail

HF_TOKEN="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"

: "${HF_TOKEN:?HF_TOKEN is not set. Please export HF_TOKEN before running.}"

rm -rf ./Data/VQA-RAD
mkdir -p ./Data/VQA-RAD

echo "Downloading parquet files..."
hf download flaviagiammarino/vqa-rad \
  --repo-type dataset \
  --token "$HF_TOKEN" \
  --include "data/*.parquet" \
  --local-dir ./Data/VQA-RAD

echo "Checking downloaded files..."
if ! ls ./Data/VQA-RAD/data/*.parquet >/dev/null 2>&1; then
  echo "ERROR: No parquet files found in ./Data/VQA-RAD/data/ after download."
  exit 1
fi

echo "Extracting parquet files..."

python3 - <<'EOF'
import pandas as pd
import json
from pathlib import Path

data_dir = Path("./Data/VQA-RAD/data")
output_dir = Path("./Data/VQA-RAD/Dataset")
images_dir = output_dir / "images"
images_dir.mkdir(parents=True, exist_ok=True)

parquets = list(data_dir.glob("*.parquet"))
print(f"Found {len(parquets)} parquet files.")
if not parquets:
    raise SystemExit("No parquet files found; aborting extraction.")

YES_NO_OPTIONS = {"A": "yes", "B": "no"}

for parquet_file in parquets:
    split = "train" if "train" in parquet_file.name else "test"
    print(f"Processing {parquet_file.name} ...")

    df = pd.read_parquet(parquet_file)

    original_len = len(df)
    df = df[df["answer"].str.strip().str.lower().isin(["yes", "no"])].reset_index(drop=True)
    print(f"  -> Filtered {original_len} -> {len(df)} yes/no records")

    records = []

    for idx, row in df.iterrows():
        record = {}
        img_filename = None

        for col in df.columns:
            if col == "image" and row[col] is not None:
                img_data = row[col]
                if isinstance(img_data, dict):
                    img_bytes = img_data.get("bytes")
                    original_path = img_data.get("path", f"{split}_{idx}.jpg")
                else:
                    img_bytes = img_data
                    original_path = f"{split}_{idx}.jpg"

                if img_bytes is not None:
                    img_filename = f"{split}_{idx}_{Path(original_path).name}"
                    img_path = images_dir / img_filename
                    img_path.write_bytes(img_bytes)
            else:
                record[col] = row[col]

        answer = str(record.pop("answer", "")).strip().lower()
        question = record.pop("question", None)

        answer_label = next((label for label, val in YES_NO_OPTIONS.items() if val == answer), None)

        records.append({
            "image": f"images/{img_filename}" if img_filename else None,
            "question": question,
            "options": YES_NO_OPTIONS,
            "answer": answer,
            "answer_label": answer_label,
            "_source_split": split,
            **record,
        })

    json_path = output_dir / f"{split}.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(records, indent=2, default=str))

    print(f"  -> Saved {len(records)} records to {json_path}")
    print(f"  -> Images saved to {images_dir}")

print("Extraction complete!")
print(f"Outputs in: {output_dir}")
EOF

echo "Done!"

rm -rf ./Data/VQA-RAD/data
rm -rf ./Data/VQA-RAD/.cache