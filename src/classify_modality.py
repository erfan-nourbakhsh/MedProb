import argparse
import base64
import json
import os
import sys
import time

import dotenv
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.constants import MODALITY_CATEGORIES, MODALITY_CLASSIFICATION_PROMPT
from utils.loaders import load_dataset

def encode_image_base64(image_path: str) -> tuple[str, str]:
    ext = os.path.splitext(image_path)[1].lower()
    media_type_map = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".webp": "image/webp",
    }
    media_type = media_type_map.get(ext, "image/jpeg")
    with open(image_path, "rb") as f:
        data = base64.standard_b64encode(f.read()).decode("utf-8")
    return data, media_type

def classify_single(question: str, image_path: str, api_key: str) -> str:
    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    img_data, media_type = encode_image_base64(image_path)
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": MODALITY_CLASSIFICATION_PROMPT},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{media_type};base64,{img_data}",
                            "detail": "low",
                        },
                    },
                    {
                        "type": "text",
                        "text": f"Question about this image: {question}\n\nWhat is the imaging modality? Reply with ONLY the category name.",
                    },
                ],
            },
        ],
        temperature=0,
        max_tokens=20,
    )
    raw = response.choices[0].message.content.strip()
    for cat in MODALITY_CATEGORIES:
        if cat.lower() in raw.lower():
            return cat
    return raw

def main():
    parser = argparse.ArgumentParser(description="Classify image modality using GPT-4o")
    parser.add_argument("--dataset", type=str, default="PATH-VQA")
    parser.add_argument("--dataset_dir", type=str, default="./samples/PATH-VQA")
    parser.add_argument(
        "--split", type=str, default="test", choices=["train", "test", "both"]
    )
    parser.add_argument("--output_dir", type=str, default="./data/modality_labels")
    parser.add_argument("--delay", type=float, default=0.3)
    parser.add_argument("--max_samples", type=int, default=-1)
    args = parser.parse_args()
    dotenv.load_dotenv("./.env")
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not found in .env")
        sys.exit(1)
    train_data, test_data = load_dataset(args.dataset_dir, args.dataset)
    splits_to_process = []
    if args.split in ("test", "both"):
        splits_to_process.append(("test", test_data))
    if args.split in ("train", "both"):
        splits_to_process.append(("train", train_data))
    out_dir = os.path.join(args.output_dir, args.dataset)
    os.makedirs(out_dir, exist_ok=True)
    for split_name, split_data in splits_to_process:
        out_path = os.path.join(out_dir, f"{split_name}_modalities.json")
        existing = {}
        if os.path.exists(out_path):
            with open(out_path, "r") as f:
                existing = json.load(f)
        samples = split_data.samples
        if args.max_samples > 0:
            samples = samples[: args.max_samples]
        print(f"\nClassifying {split_name} split ({len(samples)} samples)")
        for sample in tqdm(samples, desc=f"Modality classification ({split_name})"):
            key = str(sample.idx)
            if key in existing:
                continue
            image_path = os.path.join(args.dataset_dir, sample.image_path)
            if not os.path.exists(image_path):
                existing[key] = "unknown"
                continue
            try:
                modality = classify_single(sample.question, image_path, api_key)
                existing[key] = modality
                time.sleep(args.delay)
            except Exception as e:
                print(f"\n  [FAIL] Sample {sample.idx}: {e}")
                existing[key] = "unknown"
                time.sleep(2)
        with open(out_path, "w") as f:
            json.dump(existing, f, indent=2)
        counts = {}
        for v in existing.values():
            counts[v] = counts.get(v, 0) + 1
        print(f"\nModality distribution ({split_name}):")
        for k, v in sorted(counts.items(), key=lambda x: -x[1]):
            print(f"  {k}: {v}")
        print(f"Saved to: {out_path}")

if __name__ == "__main__":
    main()
