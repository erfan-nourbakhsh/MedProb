import json
import os
from dataclasses import dataclass, field
from typing import Any, Optional

@dataclass
class VQASample:
    idx: int
    image_path: str
    question: str
    options: dict[str, str]
    answer: str
    answer_label: str
    source_split: str

@dataclass
class VQADataset:
    samples: list[VQASample]
    split: str
    dataset_name: str
    image_base_dir: str
    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx) -> VQASample:
        return self.samples[idx]

    def __iter__(self):
        return iter(self.samples)

    def get_image_path(self, sample: VQASample) -> str:
        return os.path.join(self.image_base_dir, sample.image_path)

    def get_label_set(self) -> list[str]:
        return sorted(set(s.answer_label for s in self.samples))

def load_vqa_json(json_path: str, split: str) -> list[VQASample]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    samples = []
    for idx, entry in enumerate(data):
        sample = VQASample(
            idx=idx,
            image_path=entry["image"],
            question=entry["question"].strip(),
            options=entry["options"],
            answer=entry["answer"],
            answer_label=entry["answer_label"],
            source_split=entry.get("_source_split", split),
        )
        samples.append(sample)
    return samples

def load_dataset(
    dataset_dir: str,
    dataset_name: str = "PATH-VQA",
) -> tuple[VQADataset, VQADataset]:
    train_path = os.path.join(dataset_dir, "train.json")
    test_path = os.path.join(dataset_dir, "test.json")
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Train file not found: {train_path}")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Test file not found: {test_path}")
    train_samples = load_vqa_json(train_path, "train")
    test_samples = load_vqa_json(test_path, "test")
    train_dataset = VQADataset(
        samples=train_samples,
        split="train",
        dataset_name=dataset_name,
        image_base_dir=dataset_dir,
    )
    test_dataset = VQADataset(
        samples=test_samples,
        split="test",
        dataset_name=dataset_name,
        image_base_dir=dataset_dir,
    )
    return train_dataset, test_dataset
