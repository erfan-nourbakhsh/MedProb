#!/bin/bash

HF_TOKEN="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"

rm -rf ./Data/SLAKE

hf download BoKelvin/SLAKE \
  --repo-type dataset \
  --token $HF_TOKEN \
  --local-dir ./Data/SLAKE

unzip ./Data/SLAKE/imgs.zip -d ./Data/SLAKE/images

SRC_DIR="/raid/rsq813/MedAG/FinalProject/Data/SLAKE/images/imgs"
DST_DIR="/raid/rsq813/MedAG/FinalProject/Data/SLAKE/images"

mkdir -p "$DST_DIR"

for folder in "$SRC_DIR"/*/; do
    folder_name=$(basename "$folder")
    src_file="$folder/source.jpg"
    if [ -f "$src_file" ]; then
        cp "$src_file" "$DST_DIR/${folder_name}.jpg"
        echo "Copied: ${folder_name}.jpg"
    fi
done

echo "Done!"

rm -rf ./Data/SLAKE/images/imgs
rm -rf ./Data/SLAKE/images/__MACOSX
rm -rf ./Data/SLAKE/.cache
rm -rf ./Data/SLAKE/imgs.zip
rm -rf ./Data/SLAKE/KG.zip
rm -rf ./Data/SLAKE/mask.txt
rm -rf ./Data/SLAKE/README.md
rm -rf ./Data/SLAKE/validation.json
rm -rf ./Data/SLAKE/.gitattributes

python3 << 'EOF'
import json
import os

DATA_DIR = "/raid/rsq813/MedAG/FinalProject/Data/SLAKE"
FILES = {
    "train": os.path.join(DATA_DIR, "train.json"),
    "test":  os.path.join(DATA_DIR, "test.json"),
}

YES_NO_OPTIONS = {"A": "yes", "B": "no"}
ANSWER_LABEL_MAP = {"yes": "A", "no": "B"}

def transform_image_path(img_name):

    folder = img_name.split("/")[0]
    return f"images/{folder}.jpg"

def transform_entry(entry, split):
    answer_text = entry.get("answer", "").strip().lower()
    answer_label = ANSWER_LABEL_MAP.get(answer_text, "A")

    return {
        "image": transform_image_path(entry.get("img_name", "")),
        "question": entry.get("question", ""),
        "options": dict(YES_NO_OPTIONS),
        "answer": answer_text,
        "answer_label": answer_label,
        "question_type": "multiple_choice",
        "metadata": {
            "content_type": entry.get("content_type", ""),
            "correct_text": answer_text,
            "location": entry.get("location", ""),
            "modality": entry.get("modality", ""),
            "question_id": str(entry.get("qid", "")),
        },
        "_source_split": split,
    }

for split, path in FILES.items():
    print(f"Processing {split}: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"  Original entries: {len(data)}")

    filtered = [e for e in data if e.get("answer_type", "").upper() != "OPEN"]
    print(f"  After removing OPEN: {len(filtered)}")

    transformed = [transform_entry(e, split) for e in filtered]

    with open(path, "w", encoding="utf-8") as f:
        json.dump(transformed, f, indent=2, ensure_ascii=False)

    print(f"  Saved {len(transformed)} entries -> {path}")

print("Done.")
EOF

mkdir -p ./Data/SLAKE/Dataset
mv ./Data/SLAKE/train.json ./Data/SLAKE/Dataset/train.json
mv ./Data/SLAKE/test.json ./Data/SLAKE/Dataset/test.json
mv ./Data/SLAKE/images ./Data/SLAKE/Dataset/images