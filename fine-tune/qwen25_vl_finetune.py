from __future__ import annotations

import argparse
import inspect
import json
import os
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

def _bootstrap_single_gpu_from_argv(default_gpu: int = 0) -> int:
    gpu_id = default_gpu
    argv = sys.argv[1:]
    for i, token in enumerate(argv):
        if token == "--gpu_id" and i + 1 < len(argv):
            try:
                gpu_id = int(argv[i + 1])
            except ValueError:
                pass
        elif token.startswith("--gpu_id="):
            try:
                gpu_id = int(token.split("=", 1)[1])
            except ValueError:
                pass
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    return gpu_id
_BOOTSTRAP_GPU_ID = _bootstrap_single_gpu_from_argv(default_gpu=0)
import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen2_5_VLForConditionalGeneration,
    Trainer,
    TrainingArguments,
    set_seed,
)
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from src.utils.loaders import VQADataset, load_dataset
from src.utils.prompt_builders import build_few_shot_example, build_prompt

DEFAULT_MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
DEFAULT_MODEL_STEM = "qwen25-vl-7b-instruct"
VALID_DATASETS = ("PATH-VQA", "VQA-RAD", "SLAKE", "ALL_MED_VQA")
VALID_METHODS = ("full", "lora", "qlora")

def _collect_vision_inputs_from_messages(
    messages: list[dict],
) -> tuple[list[object] | None, list[object] | None]:
    images: list[object] = []
    videos: list[object] = []
    for message in messages:
        content = message.get("content", [])
        if not isinstance(content, list):
            continue
        for item in content:
            if not isinstance(item, dict):
                continue
            item_type = item.get("type")
            if item_type == "image" and "image" in item:
                image_value = item["image"]
                if isinstance(image_value, str):
                    with Image.open(image_value) as img:
                        images.append(img.convert("RGB"))
                else:
                    images.append(image_value)
            elif item_type == "video" and "video" in item:
                videos.append(item["video"])
    return (images or None), (videos or None)

def _sanitize_name(value: str) -> str:
    return value.replace("/", "__")

class TrainSplitDataset(Dataset):
    def __init__(self, dataset: VQADataset) -> None:
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset.samples)

    def __getitem__(self, idx: int) -> dict[str, int | str]:
        return {"sample_idx": idx, "split": self.dataset.split}

def _filter_dataset_missing_images(dataset: VQADataset) -> VQADataset:
    valid_samples = []
    missing_samples: list[tuple[int, str]] = []
    for sample in dataset.samples:
        image_path = dataset.get_image_path(sample)
        if os.path.exists(image_path):
            valid_samples.append(sample)
        else:
            missing_samples.append((sample.idx, sample.image_path))
    if missing_samples:
        preview = ", ".join(
            f"idx={sample_idx}:{image_rel}"
            for sample_idx, image_rel in missing_samples[:5]
        )
        if len(missing_samples) > 5:
            preview += ", ..."
        warnings.warn(
            f"Skipping {len(missing_samples)} {dataset.split} samples with missing images. "
            f"Examples: {preview}",
            stacklevel=2,
        )
    return VQADataset(
        samples=valid_samples,
        split=dataset.split,
        dataset_name=dataset.dataset_name,
        image_base_dir=dataset.image_base_dir,
    )

@dataclass
class QwenVisionSFTCollator:
    processor: Any
    datasets_by_split: dict[str, VQADataset]
    model_stem: str
    approach: str
    reasoning: str
    options_mode: str
    options_order: str
    specialist_dir: str
    max_length: int
    _warned_prompt_truncation: bool = False
    _warned_dropped_samples: bool = False
    _warned_sequence_truncation: bool = False
    def __post_init__(self) -> None:
        try:
            from qwen_vl_utils import process_vision_info

        except ImportError:
            process_vision_info = None
        self._process_vision_info = process_vision_info

    def _build_messages(
        self,
        split: str,
        sample_idx: int,
    ) -> tuple[list[dict], list[dict], str]:
        dataset = self.datasets_by_split[split]
        sample = dataset.samples[sample_idx]
        prompt = build_prompt(
            sample=sample,
            dataset=dataset,
            model_stem=self.model_stem,
            approach=self.approach,
            reasoning=self.reasoning,
            options_mode=self.options_mode,
            options_order=self.options_order,
            specialist_dir=self.specialist_dir,
        )
        target = build_few_shot_example(
            sample=sample,
            options_mode=self.options_mode,
            model_stem=self.model_stem,
            options_order=self.options_order,
        )["assistant_text"]
        if not prompt.image_path:
            raise FileNotFoundError(
                f"Could not resolve image path for sample idx={sample.idx}: {sample.image_path}"
            )
        prompt_messages = prompt.get_messages()
        full_messages = list(prompt_messages)
        full_messages.append(
            {
                "role": "assistant",
                "content": [{"type": "text", "text": target}],
            }
        )
        return prompt_messages, full_messages, prompt.image_path

    def _prepare_batch_inputs(
        self,
        messages_batch: list[list[dict]],
        *,
        add_generation_prompt: bool,
    ) -> dict[str, torch.Tensor]:
        texts: list[str] = []
        images: list[object] = []
        videos: list[object] = []
        for messages in messages_batch:
            texts.append(
                self.processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=add_generation_prompt,
                )
            )
            if self._process_vision_info is not None:
                image_inputs, video_inputs = self._process_vision_info(messages)
            else:
                image_inputs, video_inputs = _collect_vision_inputs_from_messages(
                    messages
                )
            if video_inputs:
                raise ValueError("This fine-tune script does not support video inputs.")
            if image_inputs is None or len(image_inputs) != 1:
                raise ValueError(
                    "Qwen2.5-VL fine-tuning expects exactly one image per training sample."
                )
            images.append(image_inputs[0])
        return self.processor(
            text=texts,
            images=images,
            padding=True,
            truncation=False,
            return_tensors="pt",
        )

    def _truncate_encoded_inputs(
        self,
        encoded: dict[str, torch.Tensor],
        *,
        batch_name: str,
    ) -> dict[str, torch.Tensor]:
        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]
        seq_len = input_ids.shape[1]
        if seq_len <= self.max_length:
            return encoded
        image_token_id = getattr(self.processor, "image_token_id", None)
        video_token_id = getattr(self.processor, "video_token_id", None)
        min_required_length = 0
        offending_rows: list[int] = []
        for row_idx in range(input_ids.shape[0]):
            sample_len = int(attention_mask[row_idx].sum().item())
            if sample_len <= self.max_length:
                continue
            sample_ids = input_ids[row_idx, :sample_len]
            mm_positions: list[torch.Tensor] = []
            if image_token_id is not None:
                mm_positions.append(
                    torch.nonzero(
                        sample_ids == image_token_id, as_tuple=False
                    ).flatten()
                )
            if video_token_id is not None:
                mm_positions.append(
                    torch.nonzero(
                        sample_ids == video_token_id, as_tuple=False
                    ).flatten()
                )
            mm_positions = [
                positions for positions in mm_positions if positions.numel() > 0
            ]
            if not mm_positions:
                continue
            last_mm_position = max(
                int(positions[-1].item()) for positions in mm_positions
            )
            min_required_length = max(min_required_length, last_mm_position + 1)
            if self.max_length <= last_mm_position:
                offending_rows.append(row_idx)
        if offending_rows:
            raise ValueError(
                f"`--max_length={self.max_length}` is too small for {batch_name}: at least one sample's "
                f"image/video token block extends past that boundary. Increase `--max_length` to at least "
                f"{min_required_length} (and usually higher to leave room for question/answer text)."
            )
        if not self._warned_sequence_truncation:
            warnings.warn(
                "Truncating tokenized text after multimodal expansion to satisfy `--max_length`. "
                "This preserves image/video token alignment, unlike processor-side truncation.",
                stacklevel=2,
            )
            self._warned_sequence_truncation = True
        truncated = dict(encoded)
        for key, value in encoded.items():
            if (
                isinstance(value, torch.Tensor)
                and value.ndim >= 2
                and value.shape[:2] == input_ids.shape[:2]
            ):
                truncated[key] = value[:, : self.max_length]
        return truncated

    def _select_visual_rows(
        self,
        pixel_values: torch.Tensor,
        grid_thw: torch.Tensor,
        keep_rows: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if grid_thw.ndim != 2 or grid_thw.shape[1] != 3:
            raise ValueError(
                f"Expected grid_thw to have shape [N, 3], got {tuple(grid_thw.shape)}"
            )
        if pixel_values.ndim == 0:
            raise ValueError("Expected pixel_values to have at least one dimension.")
        token_counts = (grid_thw[:, 0] * grid_thw[:, 1] * grid_thw[:, 2]).tolist()
        offsets = [0]
        for count in token_counts:
            offsets.append(offsets[-1] + int(count))
        selected_chunks: list[torch.Tensor] = []
        for row_idx in keep_rows.tolist():
            start = offsets[row_idx]
            end = offsets[row_idx + 1]
            selected_chunks.append(pixel_values[start:end])
        selected_pixel_values = (
            torch.cat(selected_chunks, dim=0) if selected_chunks else pixel_values[:0]
        )
        selected_grid_thw = grid_thw.index_select(0, keep_rows)
        return selected_pixel_values, selected_grid_thw

    def __call__(self, features: list[dict[str, int]]) -> dict[str, torch.Tensor]:
        prompt_messages_batch: list[list[dict]] = []
        full_messages_batch: list[list[dict]] = []
        skipped_missing_images = 0
        for feature in features:
            try:
                prompt_messages, full_messages, _ = self._build_messages(
                    str(feature["split"]),
                    int(feature["sample_idx"]),
                )
            except FileNotFoundError:
                skipped_missing_images += 1
                continue
            prompt_messages_batch.append(prompt_messages)
            full_messages_batch.append(full_messages)
        if skipped_missing_images:
            warnings.warn(
                f"Skipped {skipped_missing_images} samples in the current batch because their images "
                "could not be resolved.",
                stacklevel=2,
            )
        if not full_messages_batch:
            raise ValueError(
                "All samples in the batch were skipped because their images could not be resolved."
            )
        enc_full = self._prepare_batch_inputs(
            full_messages_batch,
            add_generation_prompt=False,
        )
        enc_full = self._truncate_encoded_inputs(enc_full, batch_name="full prompts")
        enc_prompt = self._prepare_batch_inputs(
            prompt_messages_batch,
            add_generation_prompt=True,
        )
        enc_prompt = self._truncate_encoded_inputs(
            enc_prompt, batch_name="prompt-only inputs"
        )
        input_ids = enc_full["input_ids"]
        attention_mask = enc_full["attention_mask"]
        prompt_lengths = enc_prompt["attention_mask"].sum(dim=1).tolist()
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        valid_rows: list[int] = []
        for row_idx, prompt_len in enumerate(prompt_lengths):
            if prompt_len >= input_ids.shape[1]:
                if not self._warned_prompt_truncation:
                    warnings.warn(
                        "At least one prompt consumes the full `max_length` budget after tokenization. "
                        "That sample contributes no answer tokens to the loss. Increase `--max_length` "
                        "if this happens often.",
                        stacklevel=2,
                    )
                    self._warned_prompt_truncation = True
                continue
            labels[row_idx, : int(prompt_len)] = -100
            if (labels[row_idx] != -100).any():
                valid_rows.append(row_idx)
        if len(valid_rows) != input_ids.shape[0]:
            if not valid_rows:
                raise ValueError(
                    "All samples in the batch were truncated before any answer tokens remained. "
                    "Increase --max_length."
                )
            if not self._warned_dropped_samples:
                warnings.warn(
                    "Dropping truncated samples that contain no assistant tokens in the loss. "
                    "Increase --max_length if this happens often.",
                    stacklevel=2,
                )
                self._warned_dropped_samples = True
            keep_rows = torch.tensor(valid_rows, dtype=torch.long)
        else:
            keep_rows = None
        batch = dict(enc_full)
        if keep_rows is not None:
            image_grid_thw = batch.get("image_grid_thw")
            pixel_values = batch.get("pixel_values")
            batch = {
                key: (
                    value.index_select(0, keep_rows)
                    if isinstance(value, torch.Tensor)
                    and value.shape[0] == input_ids.shape[0]
                    else value
                )
                for key, value in batch.items()
            }
            if isinstance(pixel_values, torch.Tensor) and isinstance(
                image_grid_thw, torch.Tensor
            ):
                batch["pixel_values"], batch["image_grid_thw"] = (
                    self._select_visual_rows(
                        pixel_values,
                        image_grid_thw,
                        keep_rows,
                    )
                )
            labels = labels.index_select(0, keep_rows)
        batch["labels"] = labels
        return batch

class BaseQwenPeftStrategy:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args

    def _get_dtype(self) -> torch.dtype:
        if self.args.bf16:
            return torch.bfloat16
        if self.args.fp16:
            return torch.float16
        return torch.float32

    def _base_model_kwargs(self) -> dict[str, Any]:
        return {
            "trust_remote_code": True,
            "attn_implementation": "sdpa",
            "torch_dtype": self._get_dtype(),
        }

    def build_model(self):
        raise NotImplementedError

class FullFineTuneStrategy(BaseQwenPeftStrategy):
    def build_model(self):
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.args.model_id,
            **self._base_model_kwargs(),
        )
        for param in model.parameters():
            param.requires_grad = True
        return model

class LoRAStrategy(BaseQwenPeftStrategy):
    def build_model(self):
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.args.model_id,
            **self._base_model_kwargs(),
        )
        target_modules = [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
        lora_cfg = LoraConfig(
            r=self.args.lora_r,
            lora_alpha=self.args.lora_alpha,
            lora_dropout=self.args.lora_dropout,
            task_type=TaskType.CAUSAL_LM,
            target_modules=target_modules,
        )
        model = get_peft_model(model, lora_cfg)
        return model

class QLoRAStrategy(BaseQwenPeftStrategy):
    def build_model(self):
        compute_dtype = (
            torch.bfloat16 if self.args.bf16 or not self.args.fp16 else torch.float16
        )
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
        )
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.args.model_id,
            trust_remote_code=True,
            quantization_config=quantization_config,
            device_map="auto",
        )
        model = prepare_model_for_kbit_training(model)
        target_modules = [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
        lora_cfg = LoraConfig(
            r=self.args.lora_r,
            lora_alpha=self.args.lora_alpha,
            lora_dropout=self.args.lora_dropout,
            task_type=TaskType.CAUSAL_LM,
            target_modules=target_modules,
        )
        model = get_peft_model(model, lora_cfg)
        return model
STRATEGIES = {
    "full": FullFineTuneStrategy,
    "lora": LoRAStrategy,
    "qlora": QLoRAStrategy,
}

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fine-tune Qwen2.5-VL-7B-Instruct on PATH-VQA, VQA-RAD, SLAKE, "
            "or ALL_MED_VQA"
        )
    )
    parser.add_argument(
        "--dataset_name", type=str, required=True, choices=VALID_DATASETS
    )
    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--model_id", type=str, default=DEFAULT_MODEL_ID)
    parser.add_argument("--model_stem", type=str, default=DEFAULT_MODEL_STEM)
    parser.add_argument("--method", type=str, default="full", choices=VALID_METHODS)
    parser.add_argument("--output_dir", type=str, default="")
    parser.add_argument("--approach", type=str, default="image_question")
    parser.add_argument("--reasoning", type=str, default="direct")
    parser.add_argument("--options_mode", type=str, default="with_options")
    parser.add_argument("--options_order", type=str, default="default")
    parser.add_argument(
        "--specialist_dir", type=str, default="./data/specialist_prompts"
    )
    parser.add_argument("--max_length", type=int, default=4096)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--eval_batch_size", type=int, default=1)
    parser.add_argument("--grad_accum_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--eval_steps", type=int, default=200)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument(
        "--eval_split", type=str, default="none", choices=("none", "test")
    )
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--optim", type=str, default="adamw_torch_fused")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--lora_r", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=128)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    return parser.parse_args()

def _write_prompt_preview(
    output_dir: str,
    dataset: VQADataset,
    model_stem: str,
    approach: str,
    reasoning: str,
    options_mode: str,
    options_order: str,
    specialist_dir: str,
) -> None:
    sample = dataset.samples[0]
    prompt = build_prompt(
        sample=sample,
        dataset=dataset,
        model_stem=model_stem,
        approach=approach,
        reasoning=reasoning,
        options_mode=options_mode,
        options_order=options_order,
        specialist_dir=specialist_dir,
    )
    assistant_text = build_few_shot_example(
        sample=sample,
        options_mode=options_mode,
        model_stem=model_stem,
        options_order=options_order,
    )["assistant_text"]
    preview = {
        "dataset": dataset.dataset_name,
        "sample_idx": sample.idx,
        "resolved_image_path": prompt.image_path,
        "system_prompt": prompt.system_prompt,
        "user_text": prompt.user_text,
        "assistant_target": assistant_text,
        "messages": prompt.get_messages(),
    }
    preview_path = os.path.join(output_dir, "prompt_preview.json")
    with open(preview_path, "w", encoding="utf-8") as f:
        json.dump(preview, f, indent=2, ensure_ascii=False)

def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print(
            f"[GPU] Requested physical GPU {args.gpu_id} | bootstrap {_BOOTSTRAP_GPU_ID} | "
            f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')} | "
            f"current_device={torch.cuda.current_device()}",
            flush=True,
        )
    dataset_dir = os.path.abspath(args.dataset_dir)
    if not os.path.isdir(dataset_dir):
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")
    if not args.output_dir:
        model_tag = _sanitize_name(args.model_stem)
        args.output_dir = os.path.join(
            PROJECT_ROOT,
            "fine-tune",
            "results",
            args.dataset_name,
            f"{model_tag}-{args.method}",
        )
    os.makedirs(args.output_dir, exist_ok=True)
    train_data, test_data = load_dataset(dataset_dir, args.dataset_name)
    train_data = _filter_dataset_missing_images(train_data)
    test_data = _filter_dataset_missing_images(test_data)
    if not train_data.samples:
        raise ValueError(
            "No training samples remain after filtering out missing images."
        )
    train_dataset = TrainSplitDataset(train_data)
    eval_dataset = None
    eval_vqa_dataset = None
    if args.eval_split == "test":
        if not test_data.samples:
            warnings.warn(
                "Evaluation split requested, but no test samples remain after filtering out missing images. "
                "Disabling evaluation.",
                stacklevel=2,
            )
        else:
            eval_vqa_dataset = test_data
            eval_dataset = TrainSplitDataset(test_data)
    processor = AutoProcessor.from_pretrained(
        args.model_id,
        trust_remote_code=True,
        use_fast=False,
    )
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is not None:
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
    strategy = STRATEGIES[args.method](args)
    model = strategy.build_model()
    model.config.use_cache = False
    if hasattr(model, "enable_input_require_grads") and args.method == "full":
        model.enable_input_require_grads()
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    if hasattr(model, "print_trainable_parameters"):
        model.print_trainable_parameters()
    collator = QwenVisionSFTCollator(
        processor=processor,
        datasets_by_split={
            "train": train_data,
            "test": test_data,
        },
        model_stem=args.model_stem,
        approach=args.approach,
        reasoning=args.reasoning,
        options_mode=args.options_mode,
        options_order=args.options_order,
        specialist_dir=args.specialist_dir,
        max_length=args.max_length,
    )
    _write_prompt_preview(
        output_dir=args.output_dir,
        dataset=train_data,
        model_stem=args.model_stem,
        approach=args.approach,
        reasoning=args.reasoning,
        options_mode=args.options_mode,
        options_order=args.options_order,
        specialist_dir=args.specialist_dir,
    )
    with open(
        os.path.join(args.output_dir, "run_config.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)
    ta_sig = inspect.signature(TrainingArguments.__init__)
    ta_params = ta_sig.parameters
    strategy_key = (
        "eval_strategy" if "eval_strategy" in ta_params else "evaluation_strategy"
    )
    ta_kwargs: dict[str, Any] = {
        "output_dir": args.output_dir,
        "num_train_epochs": args.epochs,
        "per_device_train_batch_size": args.train_batch_size,
        "per_device_eval_batch_size": args.eval_batch_size,
        "gradient_accumulation_steps": args.grad_accum_steps,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "logging_steps": args.logging_steps,
        "save_steps": args.save_steps,
        "save_total_limit": args.save_total_limit,
        "bf16": args.bf16,
        "fp16": args.fp16,
        "remove_unused_columns": False,
        "dataloader_num_workers": args.num_workers,
    }
    if "optim" in ta_params:
        ta_kwargs["optim"] = args.optim
    ta_kwargs[strategy_key] = "steps" if eval_dataset is not None else "no"
    if eval_dataset is not None:
        ta_kwargs["eval_steps"] = args.eval_steps
    if "warmup_steps" in ta_params:
        ta_kwargs["warmup_steps"] = args.warmup_steps
    elif "warmup_ratio" in ta_params:
        ta_kwargs["warmup_ratio"] = args.warmup_ratio
    if "report_to" in ta_params:
        ta_kwargs["report_to"] = "none"
    training_args = TrainingArguments(**ta_kwargs)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)

if __name__ == "__main__":
    main()
