from __future__ import annotations

import os
import json
import importlib
import warnings
import contextlib
import shutil
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import dotenv
import torch
import torch.nn as nn
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
from huggingface_hub import hf_hub_download, login, snapshot_download
from transformers import GenerationConfig, StoppingCriteria, StoppingCriteriaList

from .prompt_objects import (
    BasePrompt,
    VisionPrompt,
    FlamingoPrompt,
    MedFlamingoPrompt,
    OpenFlamingoPrompt,
    LlavaMedPrompt,
    LlavaV0Prompt,
    MedGemmaPrompt,
    MedIXPrompt,
    MedMOPrompt,
    MedVLThinkerPrompt,
    Qwen25VLPrompt,
    Qwen2VLPrompt,
    AdaptQwen2VLPrompt,
    AdaptInternVL3Prompt,
    InternVL3Prompt,
    BioMedLlamaPrompt,
    AdaptLlamaPrompt,
    MetaLlama32VisionPrompt,
    GemmaPrompt,
    BioGemmaLoraPrompt,
    build_medix_messages,
    MEDIX_PROMPT_PREFIX,
)
from .constants import MODEL_REGISTRY

class BasePrompter(ABC):
    def __init__(
        self,
        model_stem: str,
        model_id: str,
        max_new_tokens: int = 1024,
        temperature: float = 0.1,
    ):
        self.model_stem = model_stem
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

    @abstractmethod
    def get_completion(self, prompt: BasePrompt) -> str:
        raise NotImplementedError

    @abstractmethod
    def get_completion_batch(self, prompts: list[BasePrompt]) -> list[str]:
        raise NotImplementedError

    def get_completion_conversation(
        self,
        system_prompt: str,
        few_shot_examples: list[dict],
        query_prompt: BasePrompt,
    ) -> str:
        combined_text = ""
        if system_prompt:
            combined_text += system_prompt + "\n\n"
        for ex in few_shot_examples:
            combined_text += (
                f"User: {ex['user_text']}\nAssistant: {ex['assistant_text']}\n\n"
            )
        combined_text += f"User: {query_prompt.user_text}\nAssistant: "
        from .prompt_objects import TextOnlyPrompt as _TextOnly

        fallback = _TextOnly(user_text=combined_text, system_prompt=None)
        return self.get_completion(fallback)

    @abstractmethod
    def get_all_layer_embeddings(self, prompt: BasePrompt) -> dict[str, torch.Tensor]:
        raise NotImplementedError

    @abstractmethod
    def get_all_layer_embeddings_batch(
        self, prompts: list[BasePrompt]
    ) -> dict[str, torch.Tensor]:
        raise NotImplementedError

    @abstractmethod
    def get_first_generated_token_embeddings(
        self, prompt: BasePrompt
    ) -> dict[str, torch.Tensor]:
        raise NotImplementedError

    @abstractmethod
    def get_mean_all_tokens_embeddings(
        self, prompt: BasePrompt
    ) -> dict[str, torch.Tensor]:
        raise NotImplementedError

    @abstractmethod
    def get_mean_image_tokens_embeddings(
        self, prompt: BasePrompt
    ) -> dict[str, torch.Tensor]:
        raise NotImplementedError

    @abstractmethod
    def get_mean_text_tokens_embeddings(
        self, prompt: BasePrompt
    ) -> dict[str, torch.Tensor]:
        raise NotImplementedError

    @abstractmethod
    def get_concat_img_text_last_embeddings(
        self, prompt: BasePrompt
    ) -> dict[str, torch.Tensor]:
        raise NotImplementedError

    def _strip_assistant_header(self, text: str) -> str:
        stripped = text.lstrip()
        if stripped.lower().startswith("assistant"):
            parts = stripped.split("\n", 1)
            if len(parts) == 2:
                return parts[1].strip()
            import re

            remainder = re.sub(r"^[Aa]ssistant\s*:?\s*", "", stripped)
            return remainder.strip()
        return text.strip()

class _TokenSuffixStoppingCriteria(StoppingCriteria):
    def __init__(self, stop_sequences: list[list[int]]):
        super().__init__()
        self.stop_sequences = [seq for seq in stop_sequences if seq]

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        if input_ids.ndim != 2 or input_ids.shape[0] == 0:
            return False
        row = input_ids[0].tolist()
        for stop_ids in self.stop_sequences:
            if len(row) >= len(stop_ids) and row[-len(stop_ids) :] == stop_ids:
                return True
        return False

def _load_torch_cache(path: Path, *, label: str):
    try:
        return torch.load(path, map_location="cpu")
    except (OSError, RuntimeError, EOFError, ValueError, KeyError) as e:
        print(f"[cache] Cached {label} is unreadable: {e}")
        print(f"[cache] Removing corrupt cache file: {path}")
        path.unlink(missing_ok=True)
        return None

def _atomic_torch_save(obj, path: Path, *, label: str):
    tmp_path = path.with_name(f"{path.name}.tmp-{os.getpid()}-{time.time_ns()}")
    try:
        torch.save(obj, tmp_path)
        os.replace(tmp_path, path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise
    else:
        print(f"[save] Saved {label} to: {path}")

@contextlib.contextmanager
def _force_cpu_linspace():
    original_linspace = torch.linspace
    def cpu_linspace(*args, **kwargs):
        kwargs = dict(kwargs)
        kwargs.setdefault("device", "cpu")
        return original_linspace(*args, **kwargs)
    torch.linspace = cpu_linspace
    try:
        yield
    finally:
        torch.linspace = original_linspace

@contextlib.contextmanager
def _compat_mark_tied_weights():
    from transformers.modeling_utils import PreTrainedModel

    original_method = PreTrainedModel.mark_tied_weights_as_initialized
    def compat_mark_tied_weights_as_initialized(self):
        tied_keys = getattr(self, "all_tied_weights_keys", None)
        if tied_keys is None:
            if hasattr(self, "get_expanded_tied_weights_keys"):
                tied_keys = self.get_expanded_tied_weights_keys(all_submodels=False)
            else:
                tied_keys = getattr(self, "_tied_weights_keys", {}) or {}
            self.all_tied_weights_keys = tied_keys
        for tied_param in tied_keys.keys():
            try:
                param = self.get_parameter(tied_param)
            except Exception:
                continue
            param._is_hf_initialized = True
    PreTrainedModel.mark_tied_weights_as_initialized = (
        compat_mark_tied_weights_as_initialized
    )
    try:
        yield
    finally:
        PreTrainedModel.mark_tied_weights_as_initialized = original_method

def _get_image_token_mask(input_ids: torch.Tensor, image_token_id: int) -> torch.Tensor:
    return input_ids.squeeze(0) == image_token_id

def _get_any_token_mask(input_ids: torch.Tensor, token_ids: set[int]) -> torch.Tensor:
    seq = input_ids.squeeze(0)
    mask = torch.zeros_like(seq, dtype=torch.bool)
    for token_id in token_ids:
        mask |= seq == token_id
    return mask

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
                images.append(item["image"])
            elif item_type == "video" and "video" in item:
                videos.append(item["video"])
    return (images or None), (videos or None)

def _safe_mean_pool(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if hidden.dim() == 3:
        hidden = hidden.squeeze(0)
    if mask.sum() == 0:
        return hidden.mean(dim=0).float()
    return hidden[mask].mean(dim=0).float()

def _sanitize_repo_id(repo_id: str) -> str:
    return repo_id.replace("/", "--")

def _sharded_weights_complete(model_dir: Path, index_name: str) -> bool:
    index_path = model_dir / index_name
    if not index_path.exists():
        return False
    try:
        with index_path.open("r", encoding="utf-8") as f:
            index_data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return False
    weight_map = index_data.get("weight_map", {})
    if not weight_map:
        return False
    referenced_shards = {model_dir / shard_name for shard_name in weight_map.values()}
    if not referenced_shards:
        return False
    return all(shard_path.exists() for shard_path in referenced_shards)

def _model_weight_files_exist(model_dir: Path) -> bool:
    return (model_dir / "config.json").exists() and (
        (model_dir / "model.safetensors").exists()
        or _sharded_weights_complete(
            model_dir,
            "model.safetensors.index.json",
        )
        or (model_dir / "pytorch_model.bin").exists()
        or _sharded_weights_complete(
            model_dir,
            "pytorch_model.bin.index.json",
        )
    )

def _clear_directory_contents(path: Path) -> None:
    if not path.exists():
        return
    for child in path.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()

class OldLlavaRuntime(nn.Module):
    def __init__(
        self,
        language_model,
        vision_tower,
        mm_projector,
        image_patch_token_id: int,
        config,
        vision_select_layer: int = -2,
    ):
        super().__init__()
        self.language_model = language_model
        self.vision_tower = vision_tower
        self.mm_projector = mm_projector
        self.image_patch_token_id = image_patch_token_id
        self.config = config
        self.vision_select_layer = vision_select_layer

    @property
    def device(self):
        return self.language_model.device

    @property
    def dtype(self):
        return self.language_model.dtype

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def _encode_images(self, pixel_values: torch.Tensor) -> torch.Tensor:
        vision_outputs = self.vision_tower(pixel_values, output_hidden_states=True)
        hidden = vision_outputs.hidden_states[self.vision_select_layer]
        if hidden.shape[1] > 1:
            hidden = hidden[:, 1:, :]
        return self.mm_projector(hidden)

    def _build_inputs_embeds(
        self,
        input_ids: torch.Tensor,
        pixel_values: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        inputs_embeds = self.language_model.get_input_embeddings()(input_ids)
        if pixel_values is None:
            return inputs_embeds
        image_features = self._encode_images(pixel_values).to(inputs_embeds.dtype)
        patch_mask = input_ids == self.image_patch_token_id
        for batch_idx in range(input_ids.shape[0]):
            positions = patch_mask[batch_idx].nonzero(as_tuple=False).flatten()
            if positions.numel() == 0:
                continue
            n_img, n_patch, hidden_d = image_features.shape
            total_patches = n_img * n_patch
            if input_ids.shape[0] == 1 and positions.numel() == total_patches:
                features = image_features.reshape(-1, hidden_d)
            elif image_features.shape[0] == input_ids.shape[0]:
                features = image_features[batch_idx]
            else:
                if n_img == 1:
                    features = image_features[0]
                else:
                    raise ValueError(
                        f"LLaVA image batch mismatch: input batch={input_ids.shape[0]}, "
                        f"vision batch={n_img}, patch_tokens={positions.numel()}, "
                        f"patches_per_image={n_patch}"
                    )
            if features.shape[0] != positions.numel():
                if features.shape[0] > positions.numel():
                    features = features[: positions.numel()]
                else:
                    raise ValueError(
                        f"Image feature / patch-token mismatch: features={features.shape[0]} "
                        f"vs patch_tokens={positions.numel()}"
                    )
            inputs_embeds[batch_idx, positions, :] = features
        return inputs_embeds

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        inputs_embeds = self._build_inputs_embeds(input_ids, pixel_values=pixel_values)
        return self.language_model(
            input_ids=None,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            **kwargs,
        )

    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        inputs_embeds = self._build_inputs_embeds(input_ids, pixel_values=pixel_values)
        return self.language_model.generate(
            input_ids=None,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            **kwargs,
        )

class LlavaMedPrompter(BasePrompter):
    def __init__(
        self,
        model_id: str = "./checkpoints/llava-med-7b",
        max_new_tokens: int = 200,
        temperature: float = 0.0,
        model_stem: str = "llava-med-7b",
    ):
        super().__init__(model_stem, model_id, max_new_tokens, temperature)
        dotenv.load_dotenv("./.env")
        hf_token = os.environ.get("HF_TOKEN", "")
        if hf_token:
            login(token=hf_token)
        from safetensors.torch import load_file
        from transformers import (
            AutoTokenizer,
            CLIPImageProcessor,
            CLIPVisionModel,
            LlamaConfig,
            LlamaForCausalLM,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id, use_fast=False, trust_remote_code=True
        )
        with open(os.path.join(model_id, "config.json"), "r", encoding="utf-8") as f:
            raw_config = json.load(f)
        self.processor = CLIPImageProcessor.from_pretrained(
            raw_config["mm_vision_tower"]
        )
        self.processor.tokenizer = self.tokenizer
        text_config = LlamaConfig(
            **{
                "vocab_size": int(raw_config["vocab_size"]),
                "hidden_size": raw_config["hidden_size"],
                "intermediate_size": raw_config["intermediate_size"],
                "num_hidden_layers": raw_config["num_hidden_layers"],
                "num_attention_heads": raw_config["num_attention_heads"],
                "num_key_value_heads": raw_config.get(
                    "num_key_value_heads", raw_config["num_attention_heads"]
                ),
                "hidden_act": raw_config["hidden_act"],
                "max_position_embeddings": raw_config["max_position_embeddings"],
                "rms_norm_eps": raw_config["rms_norm_eps"],
                "tie_word_embeddings": raw_config.get("tie_word_embeddings", False),
                "bos_token_id": raw_config.get("bos_token_id", 1),
                "eos_token_id": raw_config.get("eos_token_id", 2),
                "rope_theta": raw_config.get("rope_theta", 10000.0),
                "attention_bias": raw_config.get("attention_bias", False),
                "mlp_bias": raw_config.get("mlp_bias", False),
                "torch_dtype": raw_config.get("torch_dtype", "float16"),
            }
        )
        language_model = LlamaForCausalLM(text_config)
        state_dict = {}
        index_path = os.path.join(model_id, "model.safetensors.index.json")
        with open(index_path, "r", encoding="utf-8") as f:
            index = json.load(f)
        shard_names = sorted(set(index["weight_map"].values()))
        for shard_name in shard_names:
            shard_path = os.path.join(model_id, shard_name)
            state_dict.update(load_file(shard_path))
        text_state_dict = {
            key: value
            for key, value in state_dict.items()
            if key.startswith("model.layers.")
            or key
            in {"model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"}
        }
        text_load = language_model.load_state_dict(text_state_dict, strict=False)
        if text_load.unexpected_keys:
            raise RuntimeError(
                f"Unexpected text keys when loading old LLaVA checkpoint: {text_load.unexpected_keys}"
            )
        non_rotary_missing = [
            k for k in text_load.missing_keys if "rotary_emb.inv_freq" not in k
        ]
        if non_rotary_missing:
            raise RuntimeError(
                f"Missing text keys when loading old LLaVA checkpoint: {non_rotary_missing}"
            )
        vision_tower = CLIPVisionModel.from_pretrained(raw_config["mm_vision_tower"])
        mm_projector = nn.Linear(
            raw_config["mm_hidden_size"], raw_config["hidden_size"], bias=True
        )
        mm_projector.weight.data.copy_(state_dict["model.mm_projector.weight"])
        mm_projector.bias.data.copy_(state_dict["model.mm_projector.bias"])
        target_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        language_model = language_model.to(
            dtype=torch.float16, device=target_device
        ).eval()
        vision_tower = vision_tower.to(dtype=torch.float16, device=target_device).eval()
        mm_projector = mm_projector.to(dtype=torch.float16, device=target_device).eval()
        image_patch_token_id = self.tokenizer.convert_tokens_to_ids("<im_patch>")
        runtime_config = language_model.config
        runtime_config.image_token_index = image_patch_token_id
        runtime_config.image_token_id = image_patch_token_id
        runtime_config.mm_use_im_start_end = raw_config.get("mm_use_im_start_end", True)
        runtime_config.mm_vision_select_layer = raw_config.get(
            "mm_vision_select_layer", -2
        )
        self.model = OldLlavaRuntime(
            language_model=language_model,
            vision_tower=vision_tower,
            mm_projector=mm_projector,
            image_patch_token_id=image_patch_token_id,
            config=runtime_config,
            vision_select_layer=raw_config.get("mm_vision_select_layer", -2),
        ).eval()
        self.mm_use_im_start_end = bool(raw_config.get("mm_use_im_start_end", True))
        self.image_token_len = 256
        self.image_patch_token = "<im_patch>"
        self.image_start_token = "<im_start>"
        self.image_end_token = "<im_end>"

    def _old_llava_few_shot_stopping_criteria(self) -> StoppingCriteriaList:
        stop_strings = ["\n### Human:", "\n### Assistant:"]
        stop_sequences = [
            self.tokenizer.encode(stop_text, add_special_tokens=False)
            for stop_text in stop_strings
        ]
        return StoppingCriteriaList([_TokenSuffixStoppingCriteria(stop_sequences)])

    @staticmethod
    def _strip_old_llava_separator(text: str) -> str:
        for marker in ("\n### Human:", "\n### Assistant:"):
            if marker in text:
                text = text.split(marker, 1)[0]
        return text.strip()

    def _collect_llava_images_pil(
        self,
        prompt: LlavaMedPrompt | LlavaV0Prompt,
        few_shot_examples: Optional[list[dict]] = None,
    ) -> list:
        images: list = []
        for ex in few_shot_examples or []:
            if not ex.get("has_image") or not ex.get("image_path"):
                continue
            path = ex["image_path"]
            if os.path.isfile(path):
                images.append(Image.open(path).convert("RGB"))
        if prompt.image is not None:
            images.append(prompt.image)
        return images

    def _prepare_inputs(
        self,
        prompt: LlavaMedPrompt | LlavaV0Prompt,
        few_shot_examples: Optional[list[dict]] = None,
    ):
        text = prompt.render_prompt_text(
            few_shot_examples=few_shot_examples,
            mm_use_im_start_end=self.mm_use_im_start_end,
            image_token_len=self.image_token_len,
        )
        tokenized = self.tokenizer(text, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in tokenized.items()}
        pil_images = self._collect_llava_images_pil(
            prompt, few_shot_examples=few_shot_examples
        )
        if pil_images:
            pixel_values = self.processor(images=pil_images, return_tensors="pt")[
                "pixel_values"
            ]
            inputs["pixel_values"] = pixel_values.to(
                self.model.device,
                dtype=getattr(self.model, "dtype", torch.float16),
            )
        return inputs, text

    def get_completion(self, prompt: LlavaMedPrompt | LlavaV0Prompt) -> str:
        inputs, text = self._prepare_inputs(prompt)
        with torch.inference_mode():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=False,
                num_beams=1,
                top_p=None,
            )
        prompt_len = inputs["input_ids"].shape[1]
        if generated_ids.shape[1] > prompt_len:
            gen_ids = generated_ids[0, prompt_len:]
        else:
            gen_ids = generated_ids[0]
        out = self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        return self._strip_assistant_header(out)

    def get_completion_conversation(
        self,
        system_prompt: str,
        few_shot_examples: list[dict],
        query_prompt: LlavaMedPrompt | LlavaV0Prompt,
    ) -> str:
        query_prompt.system_prompt = system_prompt
        inputs, text = self._prepare_inputs(
            query_prompt,
            few_shot_examples=few_shot_examples,
        )
        with torch.inference_mode():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=False,
                num_beams=1,
                top_p=None,
                stopping_criteria=self._old_llava_few_shot_stopping_criteria(),
            )
        prompt_len = inputs["input_ids"].shape[1]
        if generated_ids.shape[1] > prompt_len:
            gen_ids = generated_ids[0, prompt_len:]
        else:
            gen_ids = generated_ids[0]
        out = self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        out = self._strip_old_llava_separator(out)
        return self._strip_assistant_header(out)

    def get_completion_batch(
        self, prompts: list[LlavaMedPrompt | LlavaV0Prompt]
    ) -> list[str]:
        return [self.get_completion(p) for p in prompts]

    def get_all_layer_embeddings(
        self,
        prompt: LlavaMedPrompt | LlavaV0Prompt,
        few_shot_examples: Optional[list[dict]] = None,
    ) -> dict[str, torch.Tensor]:
        inputs, _ = self._prepare_inputs(prompt, few_shot_examples=few_shot_examples)
        with torch.inference_mode():
            outputs = self.model(**inputs, output_hidden_states=True)
            all_embs = {}
            for layer_idx, hidden in enumerate(outputs.hidden_states):
                embs = hidden[:, -1, :].squeeze(0).float()
                all_embs[str(layer_idx)] = embs
        return all_embs

    def get_first_generated_token_embeddings(
        self,
        prompt: LlavaMedPrompt | LlavaV0Prompt,
        few_shot_examples: Optional[list[dict]] = None,
    ) -> dict[str, torch.Tensor]:
        inputs, _ = self._prepare_inputs(prompt, few_shot_examples=few_shot_examples)
        with torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=2,
                min_new_tokens=2,
                do_sample=False,
                output_hidden_states=True,
                return_dict_in_generate=True,
            )
            all_embs = {}
            if len(generated.hidden_states) >= 2:
                gen_hidden = generated.hidden_states[1]
            else:
                gen_hidden = generated.hidden_states[0]
            for layer_idx, hidden in enumerate(gen_hidden):
                embs = hidden[:, -1, :].squeeze(0).float()
                all_embs[str(layer_idx)] = embs
        return all_embs

    def _get_image_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        image_token_ids = []
        for token in (
            self.image_patch_token,
            self.image_start_token,
            self.image_end_token,
        ):
            tok_id = self.tokenizer.convert_tokens_to_ids(token)
            if tok_id is not None and tok_id >= 0:
                image_token_ids.append(tok_id)
        if not image_token_ids:
            return torch.zeros(
                input_ids.shape[-1], dtype=torch.bool, device=input_ids.device
            )
        mask = torch.zeros(
            input_ids.shape[-1], dtype=torch.bool, device=input_ids.device
        )
        for tok_id in image_token_ids:
            mask |= input_ids.squeeze(0) == tok_id
        return mask

    def get_mean_all_tokens_embeddings(
        self,
        prompt: LlavaMedPrompt | LlavaV0Prompt,
        few_shot_examples: Optional[list[dict]] = None,
    ) -> dict[str, torch.Tensor]:
        inputs, _ = self._prepare_inputs(prompt, few_shot_examples=few_shot_examples)
        with torch.inference_mode():
            outputs = self.model(**inputs, output_hidden_states=True)
            all_embs = {}
            for layer_idx, hidden in enumerate(outputs.hidden_states):
                all_embs[str(layer_idx)] = hidden.squeeze(0).mean(dim=0).float()
        return all_embs

    def get_mean_image_tokens_embeddings(
        self,
        prompt: LlavaMedPrompt | LlavaV0Prompt,
        few_shot_examples: Optional[list[dict]] = None,
    ) -> dict[str, torch.Tensor]:
        inputs, _ = self._prepare_inputs(prompt, few_shot_examples=few_shot_examples)
        img_mask = self._get_image_mask(inputs["input_ids"])
        with torch.inference_mode():
            outputs = self.model(**inputs, output_hidden_states=True)
            all_embs = {}
            for layer_idx, hidden in enumerate(outputs.hidden_states):
                all_embs[str(layer_idx)] = _safe_mean_pool(hidden, img_mask)
        return all_embs

    def get_mean_text_tokens_embeddings(
        self,
        prompt: LlavaMedPrompt | LlavaV0Prompt,
        few_shot_examples: Optional[list[dict]] = None,
    ) -> dict[str, torch.Tensor]:
        inputs, _ = self._prepare_inputs(prompt, few_shot_examples=few_shot_examples)
        text_mask = ~self._get_image_mask(inputs["input_ids"])
        with torch.inference_mode():
            outputs = self.model(**inputs, output_hidden_states=True)
            all_embs = {}
            for layer_idx, hidden in enumerate(outputs.hidden_states):
                all_embs[str(layer_idx)] = _safe_mean_pool(hidden, text_mask)
        return all_embs

    def get_concat_img_text_last_embeddings(
        self,
        prompt: LlavaMedPrompt | LlavaV0Prompt,
        few_shot_examples: Optional[list[dict]] = None,
    ) -> dict[str, torch.Tensor]:
        inputs, _ = self._prepare_inputs(prompt, few_shot_examples=few_shot_examples)
        img_mask = self._get_image_mask(inputs["input_ids"])
        text_mask = ~img_mask
        with torch.inference_mode():
            outputs = self.model(**inputs, output_hidden_states=True)
            all_embs = {}
            for layer_idx, hidden in enumerate(outputs.hidden_states):
                img_mean = _safe_mean_pool(hidden, img_mask)
                txt_mean = _safe_mean_pool(hidden, text_mask)
                last_tok = hidden[:, -1, :].squeeze(0).float()
                all_embs[str(layer_idx)] = torch.cat(
                    [img_mean, txt_mean, last_tok], dim=0
                )
        return all_embs

    def get_all_layer_embeddings_batch(
        self, prompts: list[LlavaMedPrompt | LlavaV0Prompt]
    ) -> dict[str, torch.Tensor]:
        all_layer_embs: dict[str, list[torch.Tensor]] = {}
        for p in prompts:
            single_embs = self.get_all_layer_embeddings(p)
            for layer_key, emb in single_embs.items():
                if layer_key not in all_layer_embs:
                    all_layer_embs[layer_key] = []
                all_layer_embs[layer_key].append(emb)
        return {k: torch.stack(v, dim=0) for k, v in all_layer_embs.items()}

class RandomLlavaPrompter(LlavaMedPrompter):
    def __init__(
        self,
        model_id: str,
        max_new_tokens: int = 200,
        temperature: float = 0.0,
        model_stem: str = "random_llava-med-7b_1",
        device: str | None = None,
        dtype: torch.dtype = torch.float16,
        cache_root: str = "/raid/rsq813/random_init",
        force_reinit: int = 0,
        seed: int | None = None,
    ):
        BasePrompter.__init__(self, model_stem, model_id, max_new_tokens, temperature)
        dotenv.load_dotenv("./.env")
        hf_token = os.environ.get("HF_TOKEN", "")
        if hf_token:
            login(token=hf_token)
        from transformers import (
            AutoTokenizer,
            CLIPImageProcessor,
            CLIPVisionConfig,
            CLIPVisionModel,
            LlamaConfig,
            LlamaForCausalLM,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id, use_fast=False, trust_remote_code=True
        )
        with open(os.path.join(model_id, "config.json"), "r", encoding="utf-8") as f:
            raw_config = json.load(f)
        self.processor = CLIPImageProcessor.from_pretrained(
            raw_config["mm_vision_tower"]
        )
        self.processor.tokenizer = self.tokenizer
        text_config = LlamaConfig(
            **{
                "vocab_size": int(raw_config["vocab_size"]),
                "hidden_size": raw_config["hidden_size"],
                "intermediate_size": raw_config["intermediate_size"],
                "num_hidden_layers": raw_config["num_hidden_layers"],
                "num_attention_heads": raw_config["num_attention_heads"],
                "num_key_value_heads": raw_config.get(
                    "num_key_value_heads", raw_config["num_attention_heads"]
                ),
                "hidden_act": raw_config["hidden_act"],
                "max_position_embeddings": raw_config["max_position_embeddings"],
                "rms_norm_eps": raw_config["rms_norm_eps"],
                "tie_word_embeddings": raw_config.get("tie_word_embeddings", False),
                "bos_token_id": raw_config.get("bos_token_id", 1),
                "eos_token_id": raw_config.get("eos_token_id", 2),
                "rope_theta": raw_config.get("rope_theta", 10000.0),
                "attention_bias": raw_config.get("attention_bias", False),
                "mlp_bias": raw_config.get("mlp_bias", False),
                "torch_dtype": raw_config.get("torch_dtype", "float16"),
            }
        )
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        target_device = torch.device(device)
        dtype_tag = (
            "fp16" if dtype == torch.float16 else str(dtype).replace("torch.", "")
        )
        seed_tag = f"_seed{seed}" if seed is not None else ""
        save_dir = (
            Path(cache_root) / Path(model_id).name / f"dtype_{dtype_tag}{seed_tag}"
        )
        save_dir.mkdir(parents=True, exist_ok=True)
        cache_path = save_dir / "runtime.pt"
        payload = None
        if cache_path.exists() and not force_reinit:
            print(f"[cache] Loading cached random LLaVA model from: {cache_path}")
            payload = _load_torch_cache(cache_path, label="random LLaVA model")
        if payload is not None:
            language_model = LlamaForCausalLM(text_config)
            language_model.load_state_dict(payload["language_model"])
            vision_config = CLIPVisionConfig.from_pretrained(
                raw_config["mm_vision_tower"]
            )
            vision_tower = CLIPVisionModel(vision_config)
            vision_tower.load_state_dict(payload["vision_tower"])
            mm_projector = nn.Linear(
                raw_config["mm_hidden_size"], raw_config["hidden_size"], bias=True
            )
            mm_projector.load_state_dict(payload["mm_projector"])
        else:
            print(f"[build] Building random LLaVA model for {model_stem} on {device}")
            if seed is not None:
                print(f"[build] Setting random seed to {seed}")
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)
            old_default = torch.get_default_dtype()
            torch.set_default_dtype(dtype)
            try:
                language_model = LlamaForCausalLM(text_config)
                vision_config = CLIPVisionConfig.from_pretrained(
                    raw_config["mm_vision_tower"]
                )
                vision_tower = CLIPVisionModel(vision_config)
                mm_projector = nn.Linear(
                    raw_config["mm_hidden_size"], raw_config["hidden_size"], bias=True
                )
            finally:
                torch.set_default_dtype(old_default)
            payload = {
                "language_model": language_model.state_dict(),
                "vision_tower": vision_tower.state_dict(),
                "mm_projector": mm_projector.state_dict(),
            }
            _atomic_torch_save(payload, cache_path, label="random LLaVA model")
        target_device = torch.device(device)
        language_model = language_model.to(dtype=dtype, device=target_device).eval()
        vision_tower = vision_tower.to(dtype=dtype, device=target_device).eval()
        mm_projector = mm_projector.to(dtype=dtype, device=target_device).eval()
        image_patch_token_id = self.tokenizer.convert_tokens_to_ids("<im_patch>")
        runtime_config = language_model.config
        runtime_config.image_token_index = image_patch_token_id
        runtime_config.image_token_id = image_patch_token_id
        runtime_config.mm_use_im_start_end = raw_config.get("mm_use_im_start_end", True)
        runtime_config.mm_vision_select_layer = raw_config.get(
            "mm_vision_select_layer", -2
        )
        self.model = OldLlavaRuntime(
            language_model=language_model,
            vision_tower=vision_tower,
            mm_projector=mm_projector,
            image_patch_token_id=image_patch_token_id,
            config=runtime_config,
            vision_select_layer=raw_config.get("mm_vision_select_layer", -2),
        ).eval()
        self.mm_use_im_start_end = bool(raw_config.get("mm_use_im_start_end", True))
        self.image_token_len = 256
        self.image_patch_token = "<im_patch>"
        self.image_start_token = "<im_start>"
        self.image_end_token = "<im_end>"

class RandomLlavaPrompter1(RandomLlavaPrompter):
    def __init__(
        self,
        model_id: str,
        max_new_tokens: int = 200,
        temperature: float = 0.0,
        model_stem: str = "random_llava-med-7b_1",
        device: str | None = None,
        dtype: torch.dtype = torch.float16,
        cache_root: str = "/raid/rsq813/random_init",
        force_reinit: int = 0,
    ):
        super().__init__(
            model_id=model_id,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            model_stem=model_stem,
            device=device,
            dtype=dtype,
            cache_root=cache_root,
            force_reinit=force_reinit,
            seed=1,
        )

class RandomLlavaPrompter2(RandomLlavaPrompter):
    def __init__(
        self,
        model_id: str,
        max_new_tokens: int = 200,
        temperature: float = 0.0,
        model_stem: str = "random_llava-med-7b_2",
        device: str | None = None,
        dtype: torch.dtype = torch.float16,
        cache_root: str = "/raid/rsq813/random_init",
        force_reinit: int = 0,
    ):
        super().__init__(
            model_id=model_id,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            model_stem=model_stem,
            device=device,
            dtype=dtype,
            cache_root=cache_root,
            force_reinit=force_reinit,
            seed=2,
        )

class RandomLlavaPrompter3(RandomLlavaPrompter):
    def __init__(
        self,
        model_id: str,
        max_new_tokens: int = 200,
        temperature: float = 0.0,
        model_stem: str = "random_llava-med-7b_3",
        device: str | None = None,
        dtype: torch.dtype = torch.float16,
        cache_root: str = "/raid/rsq813/random_init",
        force_reinit: int = 0,
    ):
        super().__init__(
            model_id=model_id,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            model_stem=model_stem,
            device=device,
            dtype=dtype,
            cache_root=cache_root,
            force_reinit=force_reinit,
            seed=3,
        )

class FlamingoPrompter(BasePrompter):
    def __init__(
        self,
        model_id: str,
        model_stem: str,
        max_new_tokens: int = 200,
        temperature: float = 0.0,
        top_p: float | None = None,
        num_beams: int = 1,
        do_sample: bool = False,
    ):
        super().__init__(model_stem, model_id, max_new_tokens, temperature)
        self.top_p = top_p
        self.num_beams = num_beams
        self.do_sample = do_sample
        dotenv.load_dotenv("./.env")
        hf_token = os.environ.get("HF_TOKEN", "")
        if hf_token:
            login(token=hf_token)
        from open_flamingo import create_model_and_transforms

        lang_encoder_path = (
            os.environ.get("FLAMINGO_LLAMA_PATH")
            or os.environ.get("OPEN_FLAMINGO_LLAMA_PATH")
            or "huggyllama/llama-7b"
        )
        tokenizer_path = lang_encoder_path
        if model_stem == "med-flamingo-9b":
            tokenizer_path = snapshot_download(repo_id=model_id)
        self.model, self.image_processor, self.tokenizer = create_model_and_transforms(
            clip_vision_encoder_path="ViT-L-14",
            clip_vision_encoder_pretrained="openai",
            lang_encoder_path=lang_encoder_path,
            tokenizer_path=tokenizer_path,
            cross_attn_every_n_layers=4,
        )
        checkpoint_filename = (
            "model.pt" if model_stem == "med-flamingo-9b" else "checkpoint.pt"
        )
        checkpoint_path = hf_hub_download(
            repo_id=model_id, filename=checkpoint_filename
        )
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]
        load_result = self.model.load_state_dict(checkpoint, strict=False)
        unexpected = [
            key
            for key in load_result.unexpected_keys
            if not key.endswith("position_ids") and "rotary_emb.inv_freq" not in key
        ]
        missing = [
            key
            for key in load_result.missing_keys
            if not key.endswith("position_ids")
            and "rotary_emb.inv_freq" not in key
            and not key.startswith("vision_encoder.")
            and not key.startswith("lang_encoder.model.layers.")
            and not key.startswith("lang_encoder.model.norm.")
            and key
            not in {
                "lang_encoder.model.embed_tokens.weight",
                "lang_encoder.lm_head.weight",
            }
        ]
        if unexpected:
            raise RuntimeError(
                f"Unexpected Flamingo checkpoint keys: {unexpected[:20]}"
            )
        if missing:
            raise RuntimeError(f"Missing Flamingo checkpoint keys: {missing[:20]}")
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._dtype = torch.float16
        self.model = self.model.to(device=self._device, dtype=self._dtype).eval()
        self.image_token_id = self.tokenizer.convert_tokens_to_ids("<image>")

    def _load_images(
        self,
        prompt: FlamingoPrompt,
        few_shot_examples: Optional[list[dict]] = None,
    ) -> list[Image.Image]:
        images: list[Image.Image] = []
        for example in few_shot_examples or []:
            image_path = example.get("image_path")
            if image_path:
                images.append(Image.open(image_path).convert("RGB"))
        if prompt.image is not None:
            images.append(prompt.image)
        return images

    def _prepare_vision_x(self, images: list[Image.Image]) -> torch.Tensor:
        if images:
            processed = [self.image_processor(image) for image in images]
            vision_x = torch.stack(processed, dim=0).unsqueeze(0).unsqueeze(2)
        else:
            size = getattr(self.image_processor, "size", None) or 224
            if isinstance(size, dict):
                size = size.get("shortest_edge") or size.get("height") or 224
            vision_x = torch.zeros(
                (1, 1, 1, 3, int(size), int(size)), dtype=torch.float32
            )
        return vision_x.to(self._device, dtype=self._dtype)

    def _prepare_inputs(
        self,
        prompt: FlamingoPrompt,
        few_shot_examples: Optional[list[dict]] = None,
    ):
        text = prompt.render_prompt_text(few_shot_examples=few_shot_examples)
        tokenized = self.tokenizer(text, return_tensors="pt")
        images = self._load_images(prompt, few_shot_examples=few_shot_examples)
        vision_x = self._prepare_vision_x(images)
        lang_x = tokenized["input_ids"].to(self._device)
        attention_mask = tokenized["attention_mask"].to(self._device)
        return {
            "vision_x": vision_x,
            "lang_x": lang_x,
            "attention_mask": attention_mask,
        }, text

    def _decode_completion(self, generated_ids: torch.Tensor, prompt_len: int) -> str:
        gen_ids = generated_ids[0, prompt_len:]
        return self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

    def _run_conditioned_lm(
        self,
        vision_x: torch.Tensor,
        lang_x: torch.Tensor,
        attention_mask: torch.Tensor,
        *,
        use_cache: bool = False,
        past_key_values=None,
    ):
        self.model._encode_vision_x(vision_x=vision_x)
        try:
            return self.model.lang_encoder(
                input_ids=lang_x,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=use_cache,
                past_key_values=past_key_values,
            )
        finally:
            if not use_cache:
                self.model.lang_encoder.clear_conditioned_layers()

    def get_completion(self, prompt: MedFlamingoPrompt | OpenFlamingoPrompt) -> str:
        inputs, _ = self._prepare_inputs(prompt)
        with torch.inference_mode():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                num_beams=self.num_beams,
                do_sample=self.do_sample,
            )
        return self._decode_completion(generated_ids, inputs["lang_x"].shape[1])

    def get_completion_conversation(
        self,
        system_prompt: str,
        few_shot_examples: list[dict],
        query_prompt: MedFlamingoPrompt | OpenFlamingoPrompt,
    ) -> str:
        query_prompt.system_prompt = system_prompt
        inputs, _ = self._prepare_inputs(
            query_prompt, few_shot_examples=few_shot_examples
        )
        with torch.inference_mode():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                num_beams=self.num_beams,
                do_sample=self.do_sample,
            )
        return self._decode_completion(generated_ids, inputs["lang_x"].shape[1])

    def get_completion_batch(
        self, prompts: list[MedFlamingoPrompt | OpenFlamingoPrompt]
    ) -> list[str]:
        return [self.get_completion(p) for p in prompts]

    def get_all_layer_embeddings(
        self,
        prompt: MedFlamingoPrompt | OpenFlamingoPrompt,
        few_shot_examples: Optional[list[dict]] = None,
    ) -> dict[str, torch.Tensor]:
        inputs, _ = self._prepare_inputs(prompt, few_shot_examples=few_shot_examples)
        with torch.inference_mode():
            outputs = self._run_conditioned_lm(**inputs)
        return {
            str(layer_idx): hidden[:, -1, :].squeeze(0).float()
            for layer_idx, hidden in enumerate(outputs.hidden_states)
        }

    def get_first_generated_token_embeddings(
        self,
        prompt: MedFlamingoPrompt | OpenFlamingoPrompt,
        few_shot_examples: Optional[list[dict]] = None,
    ) -> dict[str, torch.Tensor]:
        inputs, _ = self._prepare_inputs(prompt, few_shot_examples=few_shot_examples)
        with torch.inference_mode():
            self.model._encode_vision_x(vision_x=inputs["vision_x"])
            try:
                prefill = self.model.lang_encoder(
                    input_ids=inputs["lang_x"],
                    attention_mask=inputs["attention_mask"],
                    output_hidden_states=True,
                    use_cache=True,
                )
                next_token = prefill.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                next_attention = torch.ones_like(next_token, device=next_token.device)
                decode = self.model.lang_encoder(
                    input_ids=next_token,
                    attention_mask=next_attention,
                    past_key_values=prefill.past_key_values,
                    output_hidden_states=True,
                    use_cache=True,
                )
            finally:
                self.model.lang_encoder.clear_conditioned_layers()
        return {
            str(layer_idx): hidden[:, -1, :].squeeze(0).float()
            for layer_idx, hidden in enumerate(decode.hidden_states)
        }

    def get_mean_all_tokens_embeddings(
        self,
        prompt: MedFlamingoPrompt | OpenFlamingoPrompt,
        few_shot_examples: Optional[list[dict]] = None,
    ) -> dict[str, torch.Tensor]:
        inputs, _ = self._prepare_inputs(prompt, few_shot_examples=few_shot_examples)
        with torch.inference_mode():
            outputs = self._run_conditioned_lm(**inputs)
        return {
            str(layer_idx): hidden.squeeze(0).mean(dim=0).float()
            for layer_idx, hidden in enumerate(outputs.hidden_states)
        }

    def get_mean_image_tokens_embeddings(
        self,
        prompt: MedFlamingoPrompt | OpenFlamingoPrompt,
        few_shot_examples: Optional[list[dict]] = None,
    ) -> dict[str, torch.Tensor]:
        inputs, _ = self._prepare_inputs(prompt, few_shot_examples=few_shot_examples)
        img_mask = _get_image_token_mask(inputs["lang_x"], self.image_token_id)
        with torch.inference_mode():
            outputs = self._run_conditioned_lm(**inputs)
        return {
            str(layer_idx): _safe_mean_pool(hidden, img_mask)
            for layer_idx, hidden in enumerate(outputs.hidden_states)
        }

    def get_mean_text_tokens_embeddings(
        self,
        prompt: MedFlamingoPrompt | OpenFlamingoPrompt,
        few_shot_examples: Optional[list[dict]] = None,
    ) -> dict[str, torch.Tensor]:
        inputs, _ = self._prepare_inputs(prompt, few_shot_examples=few_shot_examples)
        text_mask = ~_get_image_token_mask(inputs["lang_x"], self.image_token_id)
        with torch.inference_mode():
            outputs = self._run_conditioned_lm(**inputs)
        return {
            str(layer_idx): _safe_mean_pool(hidden, text_mask)
            for layer_idx, hidden in enumerate(outputs.hidden_states)
        }

    def get_concat_img_text_last_embeddings(
        self,
        prompt: MedFlamingoPrompt | OpenFlamingoPrompt,
        few_shot_examples: Optional[list[dict]] = None,
    ) -> dict[str, torch.Tensor]:
        inputs, _ = self._prepare_inputs(prompt, few_shot_examples=few_shot_examples)
        img_mask = _get_image_token_mask(inputs["lang_x"], self.image_token_id)
        text_mask = ~img_mask
        with torch.inference_mode():
            outputs = self._run_conditioned_lm(**inputs)
        all_embs = {}
        for layer_idx, hidden in enumerate(outputs.hidden_states):
            img_mean = _safe_mean_pool(hidden, img_mask)
            txt_mean = _safe_mean_pool(hidden, text_mask)
            last_tok = hidden[:, -1, :].squeeze(0).float()
            all_embs[str(layer_idx)] = torch.cat([img_mean, txt_mean, last_tok], dim=0)
        return all_embs

    def get_all_layer_embeddings_batch(
        self, prompts: list[MedFlamingoPrompt | OpenFlamingoPrompt]
    ) -> dict[str, torch.Tensor]:
        all_layer_embs: dict[str, list[torch.Tensor]] = {}
        for p in prompts:
            single_embs = self.get_all_layer_embeddings(p)
            for layer_key, emb in single_embs.items():
                all_layer_embs.setdefault(layer_key, []).append(emb)
        return {k: torch.stack(v, dim=0) for k, v in all_layer_embs.items()}

class RandomFlamingoPrompter(FlamingoPrompter):
    def __init__(
        self,
        model_id: str,
        model_stem: str,
        max_new_tokens: int = 200,
        temperature: float = 0.0,
        top_p: float | None = None,
        num_beams: int = 1,
        do_sample: bool = False,
        device: str | None = None,
        dtype: torch.dtype = torch.float16,
        cache_root: str = "/raid/rsq813/random_init",
        force_reinit: int = 0,
        seed: int | None = None,
    ):
        BasePrompter.__init__(self, model_stem, model_id, max_new_tokens, temperature)
        self.top_p = top_p
        self.num_beams = num_beams
        self.do_sample = do_sample
        dotenv.load_dotenv("./.env")
        hf_token = os.environ.get("HF_TOKEN", "")
        if hf_token:
            login(token=hf_token)
        from open_flamingo import create_model_and_transforms

        lang_encoder_path = (
            os.environ.get("FLAMINGO_LLAMA_PATH")
            or os.environ.get("OPEN_FLAMINGO_LLAMA_PATH")
            or "huggyllama/llama-7b"
        )
        tokenizer_path = lang_encoder_path
        if "med-flamingo-9b" in model_stem:
            tokenizer_path = snapshot_download(repo_id=model_id)
        dtype_tag = (
            "fp16" if dtype == torch.float16 else str(dtype).replace("torch.", "")
        )
        seed_tag = f"_seed{seed}" if seed is not None else ""
        save_dir = (
            Path(cache_root)
            / _sanitize_repo_id(model_id)
            / f"dtype_{dtype_tag}{seed_tag}"
        )
        save_dir.mkdir(parents=True, exist_ok=True)
        cache_path = save_dir / "flamingo_random.pt"
        if seed is not None:
            print(f"[build] Setting random seed to {seed}")
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        self.model, self.image_processor, self.tokenizer = create_model_and_transforms(
            clip_vision_encoder_path="ViT-L-14",
            clip_vision_encoder_pretrained="openai",
            lang_encoder_path=lang_encoder_path,
            tokenizer_path=tokenizer_path,
            cross_attn_every_n_layers=4,
        )
        if cache_path.exists() and not force_reinit:
            print(f"[cache] Loading cached random Flamingo model from: {cache_path}")
            state = _load_torch_cache(cache_path, label="random Flamingo model")
            if state is not None:
                self.model.load_state_dict(state, strict=False)
            else:
                _atomic_torch_save(
                    self.model.state_dict(),
                    cache_path,
                    label="random Flamingo model",
                )
        else:
            _atomic_torch_save(
                self.model.state_dict(),
                cache_path,
                label="random Flamingo model",
            )
        self._device = (
            torch.device(device)
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self._dtype = dtype
        self.model = self.model.to(device=self._device, dtype=self._dtype).eval()
        self.image_token_id = self.tokenizer.convert_tokens_to_ids("<image>")

class RandomFlamingoPrompter1(RandomFlamingoPrompter):
    def __init__(
        self,
        model_id: str,
        model_stem: str,
        max_new_tokens: int = 200,
        temperature: float = 0.0,
        top_p: float | None = None,
        num_beams: int = 1,
        do_sample: bool = False,
        device: str | None = None,
        dtype: torch.dtype = torch.float16,
        cache_root: str = "/raid/rsq813/random_init",
        force_reinit: int = 0,
    ):
        super().__init__(
            model_id=model_id,
            model_stem=model_stem,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            num_beams=num_beams,
            do_sample=do_sample,
            device=device,
            dtype=dtype,
            cache_root=cache_root,
            force_reinit=force_reinit,
            seed=1,
        )

class RandomFlamingoPrompter2(RandomFlamingoPrompter):
    def __init__(
        self,
        model_id: str,
        model_stem: str,
        max_new_tokens: int = 200,
        temperature: float = 0.0,
        top_p: float | None = None,
        num_beams: int = 1,
        do_sample: bool = False,
        device: str | None = None,
        dtype: torch.dtype = torch.float16,
        cache_root: str = "/raid/rsq813/random_init",
        force_reinit: int = 0,
    ):
        super().__init__(
            model_id=model_id,
            model_stem=model_stem,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            num_beams=num_beams,
            do_sample=do_sample,
            device=device,
            dtype=dtype,
            cache_root=cache_root,
            force_reinit=force_reinit,
            seed=2,
        )

class RandomFlamingoPrompter3(RandomFlamingoPrompter):
    def __init__(
        self,
        model_id: str,
        model_stem: str,
        max_new_tokens: int = 200,
        temperature: float = 0.0,
        top_p: float | None = None,
        num_beams: int = 1,
        do_sample: bool = False,
        device: str | None = None,
        dtype: torch.dtype = torch.float16,
        cache_root: str = "/raid/rsq813/random_init",
        force_reinit: int = 0,
    ):
        super().__init__(
            model_id=model_id,
            model_stem=model_stem,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            num_beams=num_beams,
            do_sample=do_sample,
            device=device,
            dtype=dtype,
            cache_root=cache_root,
            force_reinit=force_reinit,
            seed=3,
        )

class MedGemmaPrompter(BasePrompter):
    def __init__(
        self,
        model_id: str = "google/medgemma-4b-it",
        max_new_tokens: int = 200,
        temperature: float = 0.0,
    ):
        super().__init__("medgemma", model_id, max_new_tokens, temperature)
        dotenv.load_dotenv("./.env")
        hf_token = os.environ.get("HF_TOKEN", "")
        if hf_token:
            login(token=hf_token)
        from transformers import AutoProcessor, AutoModelForImageTextToText

        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        ).eval()

    def _prepare_inputs(self, prompt: MedGemmaPrompt):
        messages = prompt.get_messages()
        if hasattr(self.processor, "chat_template") and self.processor.chat_template:
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            warnings.warn(
                f"Processor for {self.model_id} has no chat_template. "
                "Falling back to plain-text formatting.",
                UserWarning,
            )
            text = prompt.user_text
        kw = dict(text=text, return_tensors="pt")
        if prompt.image is not None:
            kw["images"] = prompt.image
        inputs = self.processor(**kw)
        return inputs.to(self.model.device), text

    def _prepare_conversation_inputs(self, messages: list[dict]):
        if hasattr(self.processor, "chat_template") and self.processor.chat_template:
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            text = "\n\n".join(
                (
                    message.get("content", "")
                    if isinstance(message.get("content", ""), str)
                    else next(
                        (
                            part.get("text", "")
                            for part in message.get("content", [])
                            if isinstance(part, dict) and part.get("type") == "text"
                        ),
                        "",
                    )
                )
                for message in messages
                if message.get("role") == "user"
            )
        image_inputs, _ = _collect_vision_inputs_from_messages(messages)
        kw = dict(text=text, return_tensors="pt")
        if image_inputs:
            kw["images"] = image_inputs
        inputs = self.processor(**kw)
        return inputs.to(self.model.device), text

    def get_completion(self, prompt: MedGemmaPrompt) -> str:
        inputs, text = self._prepare_inputs(prompt)
        with torch.inference_mode():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=self.temperature > 0,
            )
        prompt_len = inputs["input_ids"].shape[1]
        gen_ids = generated_ids[0, prompt_len:]
        out = self.processor.decode(gen_ids, skip_special_tokens=True).strip()
        return self._strip_assistant_header(out)

    def get_completion_conversation(
        self,
        system_prompt: str,
        few_shot_examples: list[dict],
        query_prompt: MedGemmaPrompt,
    ) -> str:
        messages = []
        if few_shot_examples:
            first_user_text = few_shot_examples[0]["user_text"]
            if system_prompt:
                first_user_text = f"{system_prompt}\n\n{first_user_text}"
            first_content = [{"type": "text", "text": first_user_text}]
            if few_shot_examples[0].get("image_path"):
                first_content.append(
                    {"type": "image", "image": few_shot_examples[0]["image_path"]}
                )
            messages.append({"role": "user", "content": first_content})
            messages.append(
                {"role": "model", "content": few_shot_examples[0]["assistant_text"]}
            )
            for ex in few_shot_examples[1:]:
                content = [{"type": "text", "text": ex["user_text"]}]
                if ex.get("image_path"):
                    content.append({"type": "image", "image": ex["image_path"]})
                messages.append({"role": "user", "content": content})
                messages.append({"role": "model", "content": ex["assistant_text"]})
        elif system_prompt:
            messages.append({"role": "user", "content": system_prompt})
        content = [{"type": "text", "text": query_prompt.user_text}]
        if query_prompt.image_path:
            content.append({"type": "image", "image": query_prompt.image_path})
        messages.append({"role": "user", "content": content})
        inputs, text = self._prepare_conversation_inputs(messages)
        with torch.inference_mode():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=self.temperature > 0,
            )
        prompt_len = inputs["input_ids"].shape[1]
        gen_ids = generated_ids[0, prompt_len:]
        out = self.processor.decode(gen_ids, skip_special_tokens=True).strip()
        return self._strip_assistant_header(out)

    def get_completion_batch(self, prompts: list[MedGemmaPrompt]) -> list[str]:
        return [self.get_completion(p) for p in prompts]

    def get_all_layer_embeddings(
        self, prompt: MedGemmaPrompt
    ) -> dict[str, torch.Tensor]:
        inputs, _ = self._prepare_inputs(prompt)
        with torch.inference_mode():
            outputs = self.model(**inputs, output_hidden_states=True)
            all_embs = {}
            for layer_idx, hidden in enumerate(outputs.hidden_states):
                embs = hidden[:, -1, :].squeeze(0).float()
                all_embs[str(layer_idx)] = embs
        return all_embs

    def get_first_generated_token_embeddings(
        self, prompt: MedGemmaPrompt
    ) -> dict[str, torch.Tensor]:
        inputs, _ = self._prepare_inputs(prompt)
        with torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=2,
                min_new_tokens=2,
                do_sample=False,
                output_hidden_states=True,
                return_dict_in_generate=True,
            )
            all_embs = {}
            if len(generated.hidden_states) >= 2:
                gen_hidden = generated.hidden_states[1]
            else:
                gen_hidden = generated.hidden_states[0]
            for layer_idx, hidden in enumerate(gen_hidden):
                embs = hidden[:, -1, :].squeeze(0).float()
                all_embs[str(layer_idx)] = embs
        return all_embs

    def _get_image_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        img_tok_id = getattr(self.model.config, "image_token_index", None)
        if img_tok_id is None:
            img_tok_id = self.processor.tokenizer.convert_tokens_to_ids("<image>")
        if img_tok_id is None or img_tok_id < 0:
            img_tok_id = getattr(self.model.config, "image_token_id", None)
        if img_tok_id is None:
            return torch.zeros(
                input_ids.shape[-1], dtype=torch.bool, device=input_ids.device
            )
        return _get_image_token_mask(input_ids, img_tok_id)

    def get_mean_all_tokens_embeddings(
        self, prompt: MedGemmaPrompt
    ) -> dict[str, torch.Tensor]:
        inputs, _ = self._prepare_inputs(prompt)
        with torch.inference_mode():
            outputs = self.model(**inputs, output_hidden_states=True)
            all_embs = {}
            for layer_idx, hidden in enumerate(outputs.hidden_states):
                all_embs[str(layer_idx)] = hidden.squeeze(0).mean(dim=0).float()
        return all_embs

    def get_mean_image_tokens_embeddings(
        self, prompt: MedGemmaPrompt
    ) -> dict[str, torch.Tensor]:
        inputs, _ = self._prepare_inputs(prompt)
        img_mask = self._get_image_mask(inputs["input_ids"])
        with torch.inference_mode():
            outputs = self.model(**inputs, output_hidden_states=True)
            all_embs = {}
            for layer_idx, hidden in enumerate(outputs.hidden_states):
                all_embs[str(layer_idx)] = _safe_mean_pool(hidden, img_mask)
        return all_embs

    def get_mean_text_tokens_embeddings(
        self, prompt: MedGemmaPrompt
    ) -> dict[str, torch.Tensor]:
        inputs, _ = self._prepare_inputs(prompt)
        text_mask = ~self._get_image_mask(inputs["input_ids"])
        with torch.inference_mode():
            outputs = self.model(**inputs, output_hidden_states=True)
            all_embs = {}
            for layer_idx, hidden in enumerate(outputs.hidden_states):
                all_embs[str(layer_idx)] = _safe_mean_pool(hidden, text_mask)
        return all_embs

    def get_concat_img_text_last_embeddings(
        self, prompt: MedGemmaPrompt
    ) -> dict[str, torch.Tensor]:
        inputs, _ = self._prepare_inputs(prompt)
        img_mask = self._get_image_mask(inputs["input_ids"])
        text_mask = ~img_mask
        with torch.inference_mode():
            outputs = self.model(**inputs, output_hidden_states=True)
            all_embs = {}
            for layer_idx, hidden in enumerate(outputs.hidden_states):
                img_mean = _safe_mean_pool(hidden, img_mask)
                txt_mean = _safe_mean_pool(hidden, text_mask)
                last_tok = hidden[:, -1, :].squeeze(0).float()
                all_embs[str(layer_idx)] = torch.cat(
                    [img_mean, txt_mean, last_tok], dim=0
                )
        return all_embs

    def get_all_layer_embeddings_batch(
        self, prompts: list[MedGemmaPrompt]
    ) -> dict[str, torch.Tensor]:
        all_layer_embs: dict[str, list[torch.Tensor]] = {}
        for p in prompts:
            single_embs = self.get_all_layer_embeddings(p)
            for layer_key, emb in single_embs.items():
                if layer_key not in all_layer_embs:
                    all_layer_embs[layer_key] = []
                all_layer_embs[layer_key].append(emb)
        return {k: torch.stack(v, dim=0) for k, v in all_layer_embs.items()}

class RandomMedGemmaPrompter(MedGemmaPrompter):
    def __init__(
        self,
        model_stem: str,
        model_id: str = "google/medgemma-4b-it",
        max_new_tokens: int = 200,
        temperature: float = 0.0,
        device: str | None = None,
        dtype: torch.dtype = torch.bfloat16,
        cache_root: str = "/raid/rsq813/random_init",
        force_reinit: int = 0,
        seed: int | None = None,
    ):
        BasePrompter.__init__(self, model_stem, model_id, max_new_tokens, temperature)
        dotenv.load_dotenv("./.env")
        hf_token = os.environ.get("HF_TOKEN", "")
        if hf_token:
            login(token=hf_token)
        from transformers import AutoConfig, AutoProcessor, AutoModelForImageTextToText

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        target_device = torch.device(device)
        dtype_tag = (
            "bf16" if dtype == torch.bfloat16 else str(dtype).replace("torch.", "")
        )
        seed_tag = f"_seed{seed}" if seed is not None else ""
        save_dir = (
            Path(cache_root)
            / _sanitize_repo_id(model_id)
            / f"dtype_{dtype_tag}{seed_tag}"
        )
        save_dir.mkdir(parents=True, exist_ok=True)
        processor_src = (
            str(save_dir)
            if (save_dir / "preprocessor_config.json").exists()
            else model_id
        )
        self.processor = AutoProcessor.from_pretrained(processor_src)
        if _model_weight_files_exist(save_dir) and not force_reinit:
            print(f"[cache] Loading cached random MedGemma model from: {save_dir}")
            t0 = time.time()
            try:
                self.model = (
                    AutoModelForImageTextToText.from_pretrained(
                        str(save_dir),
                        dtype=dtype,
                        device_map=None,
                    )
                    .to(device)
                    .eval()
                )
                print(f"[cache] Loaded in {time.time() - t0:.1f}s on {device}")
                return
            except Exception as e:
                print(
                    f"[cache] Cached MedGemma checkpoint is incomplete or unreadable: {e}"
                )
                print(f"[cache] Rebuilding random MedGemma model in: {save_dir}")
                _clear_directory_contents(save_dir)
        print(f"[build] Building random MedGemma model for {model_stem} on {device}")
        config = AutoConfig.from_pretrained(model_id)
        if seed is not None:
            print(f"[build] Setting random seed to {seed}")
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        old_default = torch.get_default_dtype()
        torch.set_default_dtype(dtype)
        try:
            t0 = time.time()
            model = AutoModelForImageTextToText.from_config(config)
            model = model.to(device).eval()
            print(f"[build] Constructed + moved in {time.time() - t0:.1f}s")
        finally:
            torch.set_default_dtype(old_default)
        self.model = model
        print(f"[save] Saving random MedGemma model to: {save_dir}")
        t0 = time.time()
        _clear_directory_contents(save_dir)
        cpu_model = self.model.to("cpu")
        cpu_model.save_pretrained(str(save_dir), safe_serialization=True)
        self.processor.save_pretrained(str(save_dir))
        print(f"[save] Done in {time.time() - t0:.1f}s")
        self.model = (
            cpu_model.to(device).eval() if device != "cpu" else cpu_model.eval()
        )

class RandomMedGemmaPrompter1(RandomMedGemmaPrompter):
    def __init__(
        self,
        model_stem: str = "random_medgemma_4b_1",
        model_id: str = "google/medgemma-4b-it",
        max_new_tokens: int = 200,
        temperature: float = 0.0,
        device: str | None = None,
        dtype: torch.dtype = torch.bfloat16,
        cache_root: str = "/raid/rsq813/random_init",
        force_reinit: int = 0,
    ):
        super().__init__(
            model_stem=model_stem,
            model_id=model_id,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            device=device,
            dtype=dtype,
            cache_root=cache_root,
            force_reinit=force_reinit,
            seed=1,
        )

class RandomMedGemmaPrompter2(RandomMedGemmaPrompter):
    def __init__(
        self,
        model_stem: str = "random_medgemma_4b_2",
        model_id: str = "google/medgemma-4b-it",
        max_new_tokens: int = 200,
        temperature: float = 0.0,
        device: str | None = None,
        dtype: torch.dtype = torch.bfloat16,
        cache_root: str = "/raid/rsq813/random_init",
        force_reinit: int = 0,
    ):
        super().__init__(
            model_stem=model_stem,
            model_id=model_id,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            device=device,
            dtype=dtype,
            cache_root=cache_root,
            force_reinit=force_reinit,
            seed=2,
        )

class RandomMedGemmaPrompter3(RandomMedGemmaPrompter):
    def __init__(
        self,
        model_stem: str = "random_medgemma_4b_3",
        model_id: str = "google/medgemma-4b-it",
        max_new_tokens: int = 200,
        temperature: float = 0.0,
        device: str | None = None,
        dtype: torch.dtype = torch.bfloat16,
        cache_root: str = "/raid/rsq813/random_init",
        force_reinit: int = 0,
    ):
        super().__init__(
            model_stem=model_stem,
            model_id=model_id,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            device=device,
            dtype=dtype,
            cache_root=cache_root,
            force_reinit=force_reinit,
            seed=3,
        )

class MedVLThinkerPrompter(BasePrompter):
    def __init__(
        self,
        model_stem: str = "medvlthinker-32b",
        model_id: str = "UCSC-VLAA/MedVLThinker-32B-RL_m23k",
        max_new_tokens: int = 2048,
        temperature: float = 0.6,
        top_p: float | None = 0.95,
    ):
        super().__init__(model_stem, model_id, max_new_tokens, temperature)
        self.top_p = top_p
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        ).eval()

    def _prepare_inputs(self, messages: list[dict]):
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        image_inputs, video_inputs = _collect_vision_inputs_from_messages(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        return inputs.to(self.model.device), text

    def _generation_kwargs(self) -> dict:
        kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "do_sample": self.temperature > 0,
        }
        if self.top_p is not None and self.temperature > 0:
            kwargs["top_p"] = self.top_p
        return kwargs

    def get_completion(self, prompt: MedVLThinkerPrompt) -> str:
        inputs, _ = self._prepare_inputs(prompt.get_messages())
        with torch.inference_mode():
            generated_ids = self.model.generate(**inputs, **self._generation_kwargs())
        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return self._strip_assistant_header(output_text[0].strip())

    def get_completion_conversation(
        self,
        system_prompt: str,
        few_shot_examples: list[dict],
        query_prompt: MedVLThinkerPrompt,
    ) -> str:
        messages: list[dict] = []
        if system_prompt:
            messages.append(
                {
                    "role": "system",
                    "content": [{"type": "text", "text": system_prompt}],
                }
            )
        for ex in few_shot_examples:
            demo_content: list[dict] = []
            image_path = ex.get("image_path")
            if image_path and os.path.isfile(image_path):
                demo_content.append({"type": "image", "image": image_path})
            demo_content.append({"type": "text", "text": ex["user_text"]})
            messages.append({"role": "user", "content": demo_content})
            messages.append(
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": ex["assistant_text"]}],
                }
            )
        messages.extend(query_prompt.get_messages())
        inputs, _ = self._prepare_inputs(messages)
        with torch.inference_mode():
            generated_ids = self.model.generate(**inputs, **self._generation_kwargs())
        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return self._strip_assistant_header(output_text[0].strip())

    def get_completion_batch(self, prompts: list[MedVLThinkerPrompt]) -> list[str]:
        return [self.get_completion(p) for p in prompts]

    def get_all_layer_embeddings(
        self, prompt: MedVLThinkerPrompt
    ) -> dict[str, torch.Tensor]:
        inputs, _ = self._prepare_inputs(prompt.get_messages())
        with torch.inference_mode():
            outputs = self.model(**inputs, output_hidden_states=True)
        return {
            str(layer_idx): hidden[:, -1, :].squeeze(0).float()
            for layer_idx, hidden in enumerate(outputs.hidden_states)
        }

    def get_first_generated_token_embeddings(
        self, prompt: MedVLThinkerPrompt
    ) -> dict[str, torch.Tensor]:
        inputs, _ = self._prepare_inputs(prompt.get_messages())
        with torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=2,
                min_new_tokens=2,
                do_sample=False,
                output_hidden_states=True,
                return_dict_in_generate=True,
            )
        gen_hidden = (
            generated.hidden_states[1]
            if len(generated.hidden_states) >= 2
            else generated.hidden_states[0]
        )
        return {
            str(layer_idx): hidden[:, -1, :].squeeze(0).float()
            for layer_idx, hidden in enumerate(gen_hidden)
        }

    def _get_image_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        candidate_ids: set[int] = set()
        for attr in ("image_token_id", "image_token_index"):
            token_id = getattr(self.model.config, attr, None)
            if isinstance(token_id, int) and token_id >= 0:
                candidate_ids.add(token_id)
        tokenizer = getattr(self.processor, "tokenizer", None)
        if tokenizer is not None:
            for token in (
                "<|image_pad|>",
                "<image>",
                "<|vision_start|>",
                "<|vision_end|>",
            ):
                token_id = tokenizer.convert_tokens_to_ids(token)
                if (
                    isinstance(token_id, int)
                    and token_id >= 0
                    and token_id != tokenizer.unk_token_id
                ):
                    candidate_ids.add(token_id)
        if not candidate_ids:
            return torch.zeros(
                input_ids.shape[-1], dtype=torch.bool, device=input_ids.device
            )
        return _get_any_token_mask(input_ids, candidate_ids)

    def get_mean_all_tokens_embeddings(
        self, prompt: MedVLThinkerPrompt
    ) -> dict[str, torch.Tensor]:
        inputs, _ = self._prepare_inputs(prompt.get_messages())
        with torch.inference_mode():
            outputs = self.model(**inputs, output_hidden_states=True)
        return {
            str(layer_idx): hidden.squeeze(0).mean(dim=0).float()
            for layer_idx, hidden in enumerate(outputs.hidden_states)
        }

    def get_mean_image_tokens_embeddings(
        self, prompt: MedVLThinkerPrompt
    ) -> dict[str, torch.Tensor]:
        inputs, _ = self._prepare_inputs(prompt.get_messages())
        img_mask = self._get_image_mask(inputs["input_ids"])
        with torch.inference_mode():
            outputs = self.model(**inputs, output_hidden_states=True)
        return {
            str(layer_idx): _safe_mean_pool(hidden, img_mask)
            for layer_idx, hidden in enumerate(outputs.hidden_states)
        }

    def get_mean_text_tokens_embeddings(
        self, prompt: MedVLThinkerPrompt
    ) -> dict[str, torch.Tensor]:
        inputs, _ = self._prepare_inputs(prompt.get_messages())
        text_mask = ~self._get_image_mask(inputs["input_ids"])
        with torch.inference_mode():
            outputs = self.model(**inputs, output_hidden_states=True)
        return {
            str(layer_idx): _safe_mean_pool(hidden, text_mask)
            for layer_idx, hidden in enumerate(outputs.hidden_states)
        }

    def get_concat_img_text_last_embeddings(
        self, prompt: MedVLThinkerPrompt
    ) -> dict[str, torch.Tensor]:
        inputs, _ = self._prepare_inputs(prompt.get_messages())
        img_mask = self._get_image_mask(inputs["input_ids"])
        text_mask = ~img_mask
        with torch.inference_mode():
            outputs = self.model(**inputs, output_hidden_states=True)
        all_embs = {}
        for layer_idx, hidden in enumerate(outputs.hidden_states):
            img_mean = _safe_mean_pool(hidden, img_mask)
            txt_mean = _safe_mean_pool(hidden, text_mask)
            last_tok = hidden[:, -1, :].squeeze(0).float()
            all_embs[str(layer_idx)] = torch.cat([img_mean, txt_mean, last_tok], dim=0)
        return all_embs

    def get_all_layer_embeddings_batch(
        self, prompts: list[MedVLThinkerPrompt]
    ) -> dict[str, torch.Tensor]:
        all_layer_embs: dict[str, list[torch.Tensor]] = {}
        for p in prompts:
            single_embs = self.get_all_layer_embeddings(p)
            for layer_key, emb in single_embs.items():
                all_layer_embs.setdefault(layer_key, []).append(emb)
        return {k: torch.stack(v, dim=0) for k, v in all_layer_embs.items()}

class Qwen25VLPrompter(BasePrompter):
    def __init__(
        self,
        model_stem: str = "qwen25-vl-32b-instruct",
        model_id: str = "Qwen/Qwen2.5-VL-32B-Instruct",
        max_new_tokens: int = 128,
        temperature: float = 0.0,
        top_p: float | None = None,
    ):
        super().__init__(model_stem, model_id, max_new_tokens, temperature)
        self.top_p = top_p
        from transformers import AutoProcessor

        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        tokenizer = getattr(self.processor, "tokenizer", None)
        if tokenizer is not None:
            tokenizer.padding_side = "left"
        self.model = self._load_model(model_id).eval()
        try:
            from qwen_vl_utils import process_vision_info

        except ImportError:
            process_vision_info = None
        self._process_vision_info = process_vision_info

    def _load_model(self, model_id: str):
        import transformers

        model_type = None
        try:
            config = transformers.AutoConfig.from_pretrained(
                model_id, trust_remote_code=True
            )
            model_type = getattr(config, "model_type", None)
        except Exception:
            config = None
        flash_attn_usable = self._flash_attn_usable()
        if model_id in {"MBZUAI/MedMO-4B-Next", "MBZUAI/MedMO-8B-Next"}:
            class_name = "Qwen3VLForConditionalGeneration"
            if flash_attn_usable:
                preferred_load_kwargs = {
                    "torch_dtype": torch.bfloat16,
                    "attn_implementation": "flash_attention_2",
                    "device_map": "auto",
                }
                fallback_load_kwargs = {
                    "torch_dtype": torch.bfloat16,
                    "device_map": "auto",
                }
            else:
                preferred_load_kwargs = {
                    "torch_dtype": torch.bfloat16,
                    "device_map": "auto",
                }
                fallback_load_kwargs = None
        elif model_id == "MBZUAI/MediX-R1-2B":
            class_name = "Qwen3VLForConditionalGeneration"
            preferred_load_kwargs = {
                "torch_dtype": torch.bfloat16,
                "attn_implementation": "sdpa",
                "device_map": "auto",
            }
            second_fallback_load_kwargs = {
                "torch_dtype": torch.bfloat16,
                "device_map": "auto",
            }
            if flash_attn_usable:
                preferred_load_kwargs = {
                    "torch_dtype": torch.bfloat16,
                    "attn_implementation": "flash_attention_2",
                    "device_map": "auto",
                }
                fallback_load_kwargs = {
                    "torch_dtype": torch.bfloat16,
                    "attn_implementation": "sdpa",
                    "device_map": "auto",
                }
            else:
                fallback_load_kwargs = None
        elif model_id == "MBZUAI/MediX-R1-30B":
            self._patch_qwen3_vl_moe_experts()
            class_name = "Qwen3VLMoeForConditionalGeneration"
            preferred_load_kwargs = {
                "torch_dtype": torch.bfloat16,
                "attn_implementation": "sdpa",
                "device_map": "auto",
            }
            second_fallback_load_kwargs = {
                "torch_dtype": torch.bfloat16,
                "device_map": "auto",
            }
            if flash_attn_usable:
                preferred_load_kwargs = {
                    "torch_dtype": torch.bfloat16,
                    "attn_implementation": "flash_attention_2",
                    "device_map": "auto",
                }
                fallback_load_kwargs = {
                    "torch_dtype": torch.bfloat16,
                    "attn_implementation": "sdpa",
                    "device_map": "auto",
                }
            else:
                fallback_load_kwargs = None
        elif model_type == "qwen3_vl_moe" or "Qwen3-VL-30B-A3B" in model_id:
            self._patch_qwen3_vl_moe_experts()
            class_name = "Qwen3VLMoeForConditionalGeneration"
            preferred_load_kwargs = {
                "torch_dtype": "auto",
                "device_map": "auto",
            }
            fallback_load_kwargs = None
            second_fallback_load_kwargs = None
        elif model_type == "qwen3_vl" or "Qwen3-VL" in model_id:
            class_name = "Qwen3VLForConditionalGeneration"
            preferred_load_kwargs = {
                "torch_dtype": "auto",
                "device_map": "auto",
            }
            fallback_load_kwargs = None
            second_fallback_load_kwargs = None
        else:
            class_name = "Qwen2_5_VLForConditionalGeneration"
            preferred_load_kwargs = {
                "torch_dtype": "auto",
                "device_map": "auto",
            }
            fallback_load_kwargs = None
            second_fallback_load_kwargs = None
        if model_id in {"MBZUAI/MedMO-4B-Next", "MBZUAI/MedMO-8B-Next"}:
            second_fallback_load_kwargs = None
        model_cls = getattr(transformers, class_name, None)
        if model_cls is None:
            raise ImportError(
                f"{class_name} is not available in the installed transformers package. "
                f"Upgrade transformers to a version that supports {model_id}."
            )
        try:
            return model_cls.from_pretrained(model_id, **preferred_load_kwargs)
        except ImportError as exc:
            if fallback_load_kwargs is None or "flash_attn" not in str(exc):
                raise
            warnings.warn(
                f"FlashAttention2 is unavailable for {model_id}; "
                "falling back to a slower attention implementation.",
                RuntimeWarning,
            )
            try:
                return model_cls.from_pretrained(model_id, **fallback_load_kwargs)
            except Exception:
                if second_fallback_load_kwargs is None:
                    raise
                warnings.warn(
                    f"SDPA is unavailable for {model_id}; "
                    "falling back to the default attention implementation.",
                    RuntimeWarning,
                )
                return model_cls.from_pretrained(
                    model_id, **second_fallback_load_kwargs
                )

    def _flash_attn_usable(self) -> bool:
        try:
            importlib.import_module("flash_attn")
            return True
        except Exception:
            return False

    def _patch_qwen3_vl_moe_experts(self) -> None:
        import torch.nn as nn
        from transformers.activations import ACT2FN
        from transformers.models.qwen3_vl_moe import (
            modeling_qwen3_vl_moe as qwen3_vl_moe,
        )
        current_cls = qwen3_vl_moe.Qwen3VLMoeTextExperts
        if getattr(current_cls, "_medag_transposed_ckpt_patch", False):
            return
        class PatchedQwen3VLMoeTextExperts(nn.Module):
            _medag_transposed_ckpt_patch = True
            def __init__(self, config):
                super().__init__()
                self.num_experts = config.num_experts
                self.hidden_dim = config.hidden_size
                self.intermediate_dim = config.moe_intermediate_size
                self.gate_up_proj = nn.Parameter(
                    torch.empty(
                        self.num_experts, self.hidden_dim, 2 * self.intermediate_dim
                    )
                )
                self.down_proj = nn.Parameter(
                    torch.empty(
                        self.num_experts, self.intermediate_dim, self.hidden_dim
                    )
                )
                self.act_fn = ACT2FN[config.hidden_act]
            def forward(
                self,
                hidden_states: torch.Tensor,
                top_k_index: torch.Tensor,
                top_k_weights: torch.Tensor,
            ) -> torch.Tensor:
                final_hidden_states = torch.zeros_like(hidden_states)
                with torch.no_grad():
                    expert_mask = torch.nn.functional.one_hot(
                        top_k_index, num_classes=self.num_experts
                    )
                    expert_mask = expert_mask.permute(2, 1, 0)
                    expert_hit = torch.greater(
                        expert_mask.sum(dim=(-1, -2)), 0
                    ).nonzero()
                for expert_idx in expert_hit:
                    expert_idx = expert_idx[0]
                    if expert_idx == self.num_experts:
                        continue
                    top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
                    current_state = hidden_states[token_idx]
                    gate, up = nn.functional.linear(
                        current_state,
                        self.gate_up_proj[expert_idx].transpose(0, 1),
                    ).chunk(2, dim=-1)
                    current_hidden_states = self.act_fn(gate) * up
                    current_hidden_states = nn.functional.linear(
                        current_hidden_states,
                        self.down_proj[expert_idx].transpose(0, 1),
                    )
                    current_hidden_states = (
                        current_hidden_states
                        * top_k_weights[token_idx, top_k_pos, None]
                    )
                    final_hidden_states.index_add_(
                        0,
                        token_idx,
                        current_hidden_states.to(final_hidden_states.dtype),
                    )
                return final_hidden_states
        qwen3_vl_moe.Qwen3VLMoeTextExperts = PatchedQwen3VLMoeTextExperts

    def _prepare_inputs(self, messages: list[dict]):
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        if self._process_vision_info is not None:
            image_inputs, video_inputs = self._process_vision_info(messages)
        else:
            image_inputs, video_inputs = _collect_vision_inputs_from_messages(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        return inputs.to(self.model.device), text

    def _prepare_inputs_batch(self, prompts: list[BasePrompt]):
        texts: list[str] = []
        images: list[object] = []
        for prompt in prompts:
            messages = prompt.get_messages()
            texts.append(
                self.processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            )
            if self._process_vision_info is not None:
                image_inputs, video_inputs = self._process_vision_info(messages)
            else:
                image_inputs, video_inputs = _collect_vision_inputs_from_messages(
                    messages
                )
            if video_inputs:
                raise ValueError(
                    "Batch generation for Qwen/MediX does not support video inputs."
                )
            if image_inputs is None or len(image_inputs) != 1:
                raise ValueError(
                    "Batch generation for Qwen/MediX expects exactly one image per prompt."
                )
            images.append(image_inputs[0])
        inputs = self.processor(
            text=texts,
            images=images,
            padding=True,
            return_tensors="pt",
        )
        return inputs.to(self.model.device), texts

    def _generation_kwargs(self) -> dict:
        kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "do_sample": self.temperature > 0,
        }
        if self.top_p is not None and self.temperature > 0:
            kwargs["top_p"] = self.top_p
        return kwargs

    def _messages_with_qwen_vision_few_shot(
        self,
        few_shot_examples: list[dict],
        query_prompt: Qwen25VLPrompt | MedMOPrompt,
    ) -> list[dict]:
        messages: list[dict] = []
        for ex in few_shot_examples:
            content: list[dict] = []
            p = ex.get("image_path")
            if p:
                content.append({"type": "image", "image": p})
            content.append({"type": "text", "text": ex["user_text"]})
            messages.append({"role": "user", "content": content})
            messages.append(
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": ex["assistant_text"]}],
                }
            )
        messages.extend(query_prompt.get_messages())
        return messages

    def _qwen_messages_for_prompt(
        self,
        prompt: Qwen25VLPrompt | MedMOPrompt,
        few_shot_examples: Optional[list[dict]] = None,
    ) -> list[dict]:
        if few_shot_examples:
            return self._messages_with_qwen_vision_few_shot(few_shot_examples, prompt)
        return prompt.get_messages()

    def get_completion(self, prompt: Qwen25VLPrompt | MedMOPrompt) -> str:
        inputs, _ = self._prepare_inputs(prompt.get_messages())
        with torch.inference_mode():
            generated_ids = self.model.generate(**inputs, **self._generation_kwargs())
        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return self._strip_assistant_header(output_text[0].strip())

    def get_completion_conversation(
        self,
        system_prompt: str,
        few_shot_examples: list[dict],
        query_prompt: Qwen25VLPrompt | MedMOPrompt,
    ) -> str:
        messages = self._messages_with_qwen_vision_few_shot(
            few_shot_examples, query_prompt
        )
        inputs, _ = self._prepare_inputs(messages)
        with torch.inference_mode():
            generated_ids = self.model.generate(**inputs, **self._generation_kwargs())
        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return self._strip_assistant_header(output_text[0].strip())

    def get_completion_batch(
        self, prompts: list[Qwen25VLPrompt | MedMOPrompt | MedIXPrompt]
    ) -> list[str]:
        if not prompts:
            return []
        inputs, _ = self._prepare_inputs_batch(prompts)
        with torch.inference_mode():
            generated_ids = self.model.generate(**inputs, **self._generation_kwargs())
        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_texts = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return [self._strip_assistant_header(text.strip()) for text in output_texts]

    def get_all_layer_embeddings(
        self,
        prompt: Qwen25VLPrompt | MedMOPrompt,
        few_shot_examples: Optional[list[dict]] = None,
    ) -> dict[str, torch.Tensor]:
        inputs, _ = self._prepare_inputs(
            self._qwen_messages_for_prompt(prompt, few_shot_examples)
        )
        with torch.inference_mode():
            outputs = self.model(**inputs, output_hidden_states=True)
        return {
            str(layer_idx): hidden[:, -1, :].squeeze(0).float()
            for layer_idx, hidden in enumerate(outputs.hidden_states)
        }

    def get_first_generated_token_embeddings(
        self,
        prompt: Qwen25VLPrompt | MedMOPrompt,
        few_shot_examples: Optional[list[dict]] = None,
    ) -> dict[str, torch.Tensor]:
        inputs, _ = self._prepare_inputs(
            self._qwen_messages_for_prompt(prompt, few_shot_examples)
        )
        with torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=2,
                min_new_tokens=2,
                do_sample=False,
                output_hidden_states=True,
                return_dict_in_generate=True,
            )
        gen_hidden = (
            generated.hidden_states[1]
            if len(generated.hidden_states) >= 2
            else generated.hidden_states[0]
        )
        return {
            str(layer_idx): hidden[:, -1, :].squeeze(0).float()
            for layer_idx, hidden in enumerate(gen_hidden)
        }

    def _get_image_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        candidate_ids: set[int] = set()
        for attr in ("image_token_id", "image_token_index"):
            token_id = getattr(self.model.config, attr, None)
            if isinstance(token_id, int) and token_id >= 0:
                candidate_ids.add(token_id)
        tokenizer = getattr(self.processor, "tokenizer", None)
        if tokenizer is not None:
            for token in (
                "<|image_pad|>",
                "<image>",
                "<|vision_start|>",
                "<|vision_end|>",
            ):
                token_id = tokenizer.convert_tokens_to_ids(token)
                if (
                    isinstance(token_id, int)
                    and token_id >= 0
                    and token_id != tokenizer.unk_token_id
                ):
                    candidate_ids.add(token_id)
        if not candidate_ids:
            return torch.zeros(
                input_ids.shape[-1], dtype=torch.bool, device=input_ids.device
            )
        return _get_any_token_mask(input_ids, candidate_ids)

    def get_mean_all_tokens_embeddings(
        self,
        prompt: Qwen25VLPrompt | MedMOPrompt,
        few_shot_examples: Optional[list[dict]] = None,
    ) -> dict[str, torch.Tensor]:
        inputs, _ = self._prepare_inputs(
            self._qwen_messages_for_prompt(prompt, few_shot_examples)
        )
        with torch.inference_mode():
            outputs = self.model(**inputs, output_hidden_states=True)
        return {
            str(layer_idx): hidden.squeeze(0).mean(dim=0).float()
            for layer_idx, hidden in enumerate(outputs.hidden_states)
        }

    def get_mean_image_tokens_embeddings(
        self,
        prompt: Qwen25VLPrompt | MedMOPrompt,
        few_shot_examples: Optional[list[dict]] = None,
    ) -> dict[str, torch.Tensor]:
        inputs, _ = self._prepare_inputs(
            self._qwen_messages_for_prompt(prompt, few_shot_examples)
        )
        img_mask = self._get_image_mask(inputs["input_ids"])
        with torch.inference_mode():
            outputs = self.model(**inputs, output_hidden_states=True)
        return {
            str(layer_idx): _safe_mean_pool(hidden, img_mask)
            for layer_idx, hidden in enumerate(outputs.hidden_states)
        }

    def get_mean_text_tokens_embeddings(
        self,
        prompt: Qwen25VLPrompt | MedMOPrompt,
        few_shot_examples: Optional[list[dict]] = None,
    ) -> dict[str, torch.Tensor]:
        inputs, _ = self._prepare_inputs(
            self._qwen_messages_for_prompt(prompt, few_shot_examples)
        )
        text_mask = ~self._get_image_mask(inputs["input_ids"])
        with torch.inference_mode():
            outputs = self.model(**inputs, output_hidden_states=True)
        return {
            str(layer_idx): _safe_mean_pool(hidden, text_mask)
            for layer_idx, hidden in enumerate(outputs.hidden_states)
        }

    def get_concat_img_text_last_embeddings(
        self,
        prompt: Qwen25VLPrompt | MedMOPrompt,
        few_shot_examples: Optional[list[dict]] = None,
    ) -> dict[str, torch.Tensor]:
        inputs, _ = self._prepare_inputs(
            self._qwen_messages_for_prompt(prompt, few_shot_examples)
        )
        img_mask = self._get_image_mask(inputs["input_ids"])
        text_mask = ~img_mask
        with torch.inference_mode():
            outputs = self.model(**inputs, output_hidden_states=True)
        all_embs = {}
        for layer_idx, hidden in enumerate(outputs.hidden_states):
            img_mean = _safe_mean_pool(hidden, img_mask)
            txt_mean = _safe_mean_pool(hidden, text_mask)
            last_tok = hidden[:, -1, :].squeeze(0).float()
            all_embs[str(layer_idx)] = torch.cat([img_mean, txt_mean, last_tok], dim=0)
        return all_embs

    def get_all_layer_embeddings_batch(
        self, prompts: list[Qwen25VLPrompt | MedMOPrompt]
    ) -> dict[str, torch.Tensor]:
        all_layer_embs: dict[str, list[torch.Tensor]] = {}
        for p in prompts:
            single_embs = self.get_all_layer_embeddings(p)
            for layer_key, emb in single_embs.items():
                all_layer_embs.setdefault(layer_key, []).append(emb)
        return {k: torch.stack(v, dim=0) for k, v in all_layer_embs.items()}

class RandomQwen25VLPrompter(Qwen25VLPrompter):
    def __init__(
        self,
        model_stem: str,
        model_id: str = "Qwen/Qwen2.5-VL-3B-Instruct",
        max_new_tokens: int = 128,
        temperature: float = 0.0,
        top_p: float | None = None,
        device: str | None = None,
        dtype: torch.dtype = torch.bfloat16,
        cache_root: str = "/raid/rsq813/random_init",
        force_reinit: int = 0,
        seed: int | None = None,
    ):
        BasePrompter.__init__(self, model_stem, model_id, max_new_tokens, temperature)
        self.top_p = top_p
        from transformers import AutoConfig, AutoProcessor
        import transformers

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        target_device = torch.device(device)
        dtype_tag = (
            "bf16" if dtype == torch.bfloat16 else str(dtype).replace("torch.", "")
        )
        seed_tag = f"_seed{seed}" if seed is not None else ""
        save_dir = (
            Path(cache_root)
            / _sanitize_repo_id(model_id)
            / f"dtype_{dtype_tag}{seed_tag}"
        )
        save_dir.mkdir(parents=True, exist_ok=True)
        processor_src = (
            str(save_dir)
            if (save_dir / "preprocessor_config.json").exists()
            else model_id
        )
        self.processor = AutoProcessor.from_pretrained(
            processor_src, trust_remote_code=True
        )
        tokenizer = getattr(self.processor, "tokenizer", None)
        if tokenizer is not None:
            tokenizer.padding_side = "left"
        model_cls = getattr(transformers, "Qwen2_5_VLForConditionalGeneration", None)
        if model_cls is None:
            raise ImportError(
                "Qwen2_5_VLForConditionalGeneration is not available in the installed "
                "transformers package. Upgrade transformers to a version that supports "
                f"{model_id}."
            )
        if _model_weight_files_exist(save_dir) and not force_reinit:
            print(f"[cache] Loading cached random Qwen2.5-VL model from: {save_dir}")
            t0 = time.time()
            try:
                self.model = (
                    model_cls.from_pretrained(
                        str(save_dir),
                        torch_dtype=dtype,
                        device_map=None,
                    )
                    .to(target_device)
                    .eval()
                )
                print(f"[cache] Loaded in {time.time() - t0:.1f}s on {device}")
            except Exception as e:
                print(
                    f"[cache] Cached Qwen2.5-VL checkpoint is incomplete or unreadable: {e}"
                )
                print(f"[cache] Rebuilding random Qwen2.5-VL model in: {save_dir}")
                _clear_directory_contents(save_dir)
            else:
                try:
                    from qwen_vl_utils import process_vision_info

                except ImportError:
                    process_vision_info = None
                self._process_vision_info = process_vision_info
                return
        print(f"[build] Building random Qwen2.5-VL model for {model_stem} on {device}")
        config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        if seed is not None:
            print(f"[build] Setting random seed to {seed}")
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        old_default = torch.get_default_dtype()
        torch.set_default_dtype(dtype)
        try:
            t0 = time.time()
            model = model_cls(config).to(target_device).eval()
            print(f"[build] Constructed + moved in {time.time() - t0:.1f}s")
        finally:
            torch.set_default_dtype(old_default)
        self.model = model
        print(f"[save] Saving random Qwen2.5-VL model to: {save_dir}")
        t0 = time.time()
        _clear_directory_contents(save_dir)
        cpu_model = self.model.to("cpu")
        cpu_model.save_pretrained(str(save_dir), safe_serialization=True)
        self.processor.save_pretrained(str(save_dir))
        print(f"[save] Done in {time.time() - t0:.1f}s")
        self.model = cpu_model.to(target_device).eval()
        try:
            from qwen_vl_utils import process_vision_info

        except ImportError:
            process_vision_info = None
        self._process_vision_info = process_vision_info

class RandomQwen25VLPrompter1(RandomQwen25VLPrompter):
    def __init__(
        self,
        model_stem: str = "random_Qwen2.5-VL-3B_1",
        model_id: str = "Qwen/Qwen2.5-VL-3B-Instruct",
        max_new_tokens: int = 128,
        temperature: float = 0.0,
        top_p: float | None = None,
        device: str | None = None,
        dtype: torch.dtype = torch.bfloat16,
        cache_root: str = "/raid/rsq813/random_init",
        force_reinit: int = 0,
    ):
        super().__init__(
            model_stem=model_stem,
            model_id=model_id,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            device=device,
            dtype=dtype,
            cache_root=cache_root,
            force_reinit=force_reinit,
            seed=1,
        )

class RandomQwen25VLPrompter2(RandomQwen25VLPrompter):
    def __init__(
        self,
        model_stem: str = "random_Qwen2.5-VL-3B_2",
        model_id: str = "Qwen/Qwen2.5-VL-3B-Instruct",
        max_new_tokens: int = 128,
        temperature: float = 0.0,
        top_p: float | None = None,
        device: str | None = None,
        dtype: torch.dtype = torch.bfloat16,
        cache_root: str = "/raid/rsq813/random_init",
        force_reinit: int = 0,
    ):
        super().__init__(
            model_stem=model_stem,
            model_id=model_id,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            device=device,
            dtype=dtype,
            cache_root=cache_root,
            force_reinit=force_reinit,
            seed=2,
        )

class RandomQwen25VLPrompter3(RandomQwen25VLPrompter):
    def __init__(
        self,
        model_stem: str = "random_Qwen2.5-VL-3B_3",
        model_id: str = "Qwen/Qwen2.5-VL-3B-Instruct",
        max_new_tokens: int = 128,
        temperature: float = 0.0,
        top_p: float | None = None,
        device: str | None = None,
        dtype: torch.dtype = torch.bfloat16,
        cache_root: str = "/raid/rsq813/random_init",
        force_reinit: int = 0,
    ):
        super().__init__(
            model_stem=model_stem,
            model_id=model_id,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            device=device,
            dtype=dtype,
            cache_root=cache_root,
            force_reinit=force_reinit,
            seed=3,
        )

class RandomQwen3VLPrompter(RandomQwen25VLPrompter):
    def __init__(
        self,
        model_stem: str,
        model_id: str = "Qwen/Qwen3-VL-8B-Instruct",
        max_new_tokens: int = 128,
        temperature: float = 0.0,
        top_p: float | None = None,
        device: str | None = None,
        dtype: torch.dtype = torch.bfloat16,
        cache_root: str = "/raid/rsq813/random_init",
        force_reinit: int = 0,
        seed: int | None = None,
    ):
        BasePrompter.__init__(self, model_stem, model_id, max_new_tokens, temperature)
        self.top_p = top_p
        from transformers import AutoConfig, AutoProcessor
        import transformers

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        target_device = torch.device(device)
        dtype_tag = (
            "bf16" if dtype == torch.bfloat16 else str(dtype).replace("torch.", "")
        )
        seed_tag = f"_seed{seed}" if seed is not None else ""
        save_dir = (
            Path(cache_root)
            / _sanitize_repo_id(model_id)
            / f"dtype_{dtype_tag}{seed_tag}"
        )
        save_dir.mkdir(parents=True, exist_ok=True)
        processor_src = (
            str(save_dir)
            if (save_dir / "preprocessor_config.json").exists()
            else model_id
        )
        self.processor = AutoProcessor.from_pretrained(
            processor_src, trust_remote_code=True
        )
        tokenizer = getattr(self.processor, "tokenizer", None)
        if tokenizer is not None:
            tokenizer.padding_side = "left"
        model_cls = getattr(transformers, "Qwen3VLForConditionalGeneration", None)
        if model_cls is None:
            raise ImportError(
                "Qwen3VLForConditionalGeneration is not available in the installed "
                "transformers package. Upgrade transformers to a version that supports "
                f"{model_id}."
            )
        if _model_weight_files_exist(save_dir) and not force_reinit:
            print(f"[cache] Loading cached random Qwen3-VL model from: {save_dir}")
            t0 = time.time()
            try:
                self.model = (
                    model_cls.from_pretrained(
                        str(save_dir),
                        torch_dtype=dtype,
                        device_map=None,
                    )
                    .to(target_device)
                    .eval()
                )
                print(f"[cache] Loaded in {time.time() - t0:.1f}s on {device}")
            except Exception as e:
                print(
                    f"[cache] Cached Qwen3-VL checkpoint is incomplete or unreadable: {e}"
                )
                print(f"[cache] Rebuilding random Qwen3-VL model in: {save_dir}")
                _clear_directory_contents(save_dir)
            else:
                try:
                    from qwen_vl_utils import process_vision_info

                except ImportError:
                    process_vision_info = None
                self._process_vision_info = process_vision_info
                return
        print(f"[build] Building random Qwen3-VL model for {model_stem} on {device}")
        config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        if seed is not None:
            print(f"[build] Setting random seed to {seed}")
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        old_default = torch.get_default_dtype()
        torch.set_default_dtype(dtype)
        try:
            t0 = time.time()
            model = model_cls(config).to(target_device).eval()
            print(f"[build] Constructed + moved in {time.time() - t0:.1f}s")
        finally:
            torch.set_default_dtype(old_default)
        self.model = model
        print(f"[save] Saving random Qwen3-VL model to: {save_dir}")
        t0 = time.time()
        _clear_directory_contents(save_dir)
        cpu_model = self.model.to("cpu")
        cpu_model.save_pretrained(str(save_dir), safe_serialization=True)
        self.processor.save_pretrained(str(save_dir))
        print(f"[save] Done in {time.time() - t0:.1f}s")
        self.model = cpu_model.to(target_device).eval()
        try:
            from qwen_vl_utils import process_vision_info

        except ImportError:
            process_vision_info = None
        self._process_vision_info = process_vision_info

class RandomQwen3VLPrompter1(RandomQwen3VLPrompter):
    def __init__(
        self,
        model_stem: str = "random_Qwen3-VL-8B_1",
        model_id: str = "Qwen/Qwen3-VL-8B-Instruct",
        max_new_tokens: int = 128,
        temperature: float = 0.0,
        top_p: float | None = None,
        device: str | None = None,
        dtype: torch.dtype = torch.bfloat16,
        cache_root: str = "/raid/rsq813/random_init",
        force_reinit: int = 0,
    ):
        super().__init__(
            model_stem=model_stem,
            model_id=model_id,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            device=device,
            dtype=dtype,
            cache_root=cache_root,
            force_reinit=force_reinit,
            seed=1,
        )

class RandomQwen3VLPrompter2(RandomQwen3VLPrompter):
    def __init__(
        self,
        model_stem: str = "random_Qwen3-VL-8B_2",
        model_id: str = "Qwen/Qwen3-VL-8B-Instruct",
        max_new_tokens: int = 128,
        temperature: float = 0.0,
        top_p: float | None = None,
        device: str | None = None,
        dtype: torch.dtype = torch.bfloat16,
        cache_root: str = "/raid/rsq813/random_init",
        force_reinit: int = 0,
    ):
        super().__init__(
            model_stem=model_stem,
            model_id=model_id,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            device=device,
            dtype=dtype,
            cache_root=cache_root,
            force_reinit=force_reinit,
            seed=2,
        )

class RandomQwen3VLPrompter3(RandomQwen3VLPrompter):
    def __init__(
        self,
        model_stem: str = "random_Qwen3-VL-8B_3",
        model_id: str = "Qwen/Qwen3-VL-8B-Instruct",
        max_new_tokens: int = 128,
        temperature: float = 0.0,
        top_p: float | None = None,
        device: str | None = None,
        dtype: torch.dtype = torch.bfloat16,
        cache_root: str = "/raid/rsq813/random_init",
        force_reinit: int = 0,
    ):
        super().__init__(
            model_stem=model_stem,
            model_id=model_id,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            device=device,
            dtype=dtype,
            cache_root=cache_root,
            force_reinit=force_reinit,
            seed=3,
        )

class MedIXPrompter(Qwen25VLPrompter):
    def __init__(
        self,
        model_stem: str = "medix-r1-2b",
        model_id: str = "MBZUAI/MediX-R1-2B",
        max_new_tokens: int = 2048,
        temperature: float = 0.0,
        top_p: float | None = 1.0,
    ):
        super().__init__(
            model_stem=model_stem,
            model_id=model_id,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )

    def get_completion_conversation(
        self,
        system_prompt: str,
        few_shot_examples: list[dict],
        query_prompt: MedIXPrompt,
    ) -> str:
        query_messages = query_prompt.get_messages()
        system_messages = [m for m in query_messages if m.get("role") == "system"]
        user_messages = [m for m in query_messages if m.get("role") == "user"]
        messages: list[dict] = []
        if system_messages:
            messages.append(system_messages[0])
        for ex in few_shot_examples:
            demo_image = ex.get("image_path")
            messages.extend(
                build_medix_messages(
                    question=ex["user_text"],
                    options=ex.get("options"),
                    image=demo_image,
                )
            )
            messages.append(
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": ex["assistant_text"]}],
                }
            )
        messages.extend(user_messages)
        inputs, _ = self._prepare_inputs(messages)
        with torch.inference_mode():
            generated_ids = self.model.generate(**inputs, **self._generation_kwargs())
        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return self._strip_assistant_header(output_text[0].strip())

class AdaptQwen2VLPrompter(Qwen25VLPrompter):
    def __init__(
        self,
        model_stem: str = "adapt-qwen2-2b",
        model_id: str = "AdaptLLM/biomed-Qwen2-VL-2B-Instruct",
        max_new_tokens: int = 128,
        temperature: float = 0.0,
        top_p: float | None = None,
    ):
        super().__init__(
            model_stem=model_stem,
            model_id=model_id,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )

    def _load_model(self, model_id: str):
        import transformers

        model_cls = getattr(transformers, "Qwen2VLForConditionalGeneration", None)
        if model_cls is None:
            raise ImportError(
                "Qwen2VLForConditionalGeneration is not available in the installed "
                "transformers package. Upgrade transformers to a version that supports "
                f"{model_id}."
            )
        return model_cls.from_pretrained(
            model_id,
            torch_dtype="auto",
            device_map="auto",
        )

class Qwen2VLPrompter(Qwen25VLPrompter):
    def __init__(
        self,
        model_stem: str = "qwen2-vl-2b-instruct",
        model_id: str = "Qwen/Qwen2-VL-2B-Instruct",
        max_new_tokens: int = 128,
        temperature: float = 0.0,
        top_p: float | None = None,
    ):
        super().__init__(
            model_stem=model_stem,
            model_id=model_id,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )

    def _load_model(self, model_id: str):
        import transformers

        model_cls = getattr(transformers, "Qwen2VLForConditionalGeneration", None)
        if model_cls is None:
            raise ImportError(
                "Qwen2VLForConditionalGeneration is not available in the installed "
                "transformers package. Upgrade transformers to a version that supports "
                f"{model_id}."
            )
        return model_cls.from_pretrained(
            model_id,
            torch_dtype="auto",
            device_map="auto",
        )

class AdaptInternVL3Prompter(BasePrompter):
    def __init__(
        self,
        model_stem: str = "adapt-internVL3-1b",
        model_id: str = "AdaptLLM/biomed-InternVL3-1B",
        max_new_tokens: int = 1024,
        temperature: float = 0.0,
    ):
        super().__init__(model_stem, model_id, max_new_tokens, temperature)
        from transformers import AutoModel, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=True,
            use_fast=False,
        )
        self.model = self._load_model(AutoModel, model_id).eval()
        self._image_size = 448
        self._image_mean = (0.485, 0.456, 0.406)
        self._image_std = (0.229, 0.224, 0.225)

    def _load_model(self, auto_model_cls, model_id: str):
        from transformers import AutoConfig

        load_kwargs = {
            "torch_dtype": torch.bfloat16,
            "trust_remote_code": True,
        }
        if self._flash_attn_usable():
            load_kwargs["use_flash_attn"] = True
        world_size = torch.cuda.device_count()
        if world_size > 1:
            load_kwargs["device_map"] = self._split_model(AutoConfig, model_id)
        with _force_cpu_linspace(), _compat_mark_tied_weights():
            model = auto_model_cls.from_pretrained(model_id, **load_kwargs)
        if world_size == 1 and torch.cuda.is_available():
            model = model.cuda()
        return model

    def _flash_attn_usable(self) -> bool:
        try:
            importlib.import_module("flash_attn")
            return True
        except Exception:
            return False

    def _split_model(self, auto_config_cls, model_id: str) -> dict[str, int]:
        import math

        world_size = torch.cuda.device_count()
        if world_size <= 1:
            return {}
        config = auto_config_cls.from_pretrained(model_id, trust_remote_code=True)
        num_layers = config.llm_config.num_hidden_layers
        num_layers_per_gpu = math.ceil(num_layers / (world_size - 0.5))
        allocations = [num_layers_per_gpu] * world_size
        allocations[0] = math.ceil(allocations[0] * 0.5)
        device_map: dict[str, int] = {}
        layer_idx = 0
        for device_idx, num_device_layers in enumerate(allocations):
            for _ in range(num_device_layers):
                if layer_idx >= num_layers:
                    break
                device_map[f"language_model.model.layers.{layer_idx}"] = device_idx
                layer_idx += 1
        device_map["vision_model"] = 0
        device_map["mlp1"] = 0
        device_map["language_model.model.tok_embeddings"] = 0
        device_map["language_model.model.embed_tokens"] = 0
        device_map["language_model.output"] = 0
        device_map["language_model.model.norm"] = 0
        device_map["language_model.model.rotary_emb"] = 0
        device_map["language_model.lm_head"] = 0
        device_map[f"language_model.model.layers.{num_layers - 1}"] = 0
        return device_map

    def _get_model_device(self) -> torch.device:
        if hasattr(self.model, "device"):
            return self.model.device
        return next(self.model.parameters()).device

    def _build_transform(self):
        import torchvision.transforms as T
        from torchvision.transforms.functional import InterpolationMode

        return T.Compose(
            [
                T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
                T.Resize(
                    (self._image_size, self._image_size),
                    interpolation=InterpolationMode.BICUBIC,
                ),
                T.ToTensor(),
                T.Normalize(mean=self._image_mean, std=self._image_std),
            ]
        )

    def _find_closest_aspect_ratio(self, aspect_ratio, target_ratios, width, height):
        best_ratio_diff = float("inf")
        best_ratio = (1, 1)
        area = width * height
        for ratio in target_ratios:
            target_aspect_ratio = ratio[0] / ratio[1]
            ratio_diff = abs(aspect_ratio - target_aspect_ratio)
            if ratio_diff < best_ratio_diff:
                best_ratio_diff = ratio_diff
                best_ratio = ratio
            elif ratio_diff == best_ratio_diff:
                if (
                    area
                    > 0.5 * self._image_size * self._image_size * ratio[0] * ratio[1]
                ):
                    best_ratio = ratio
        return best_ratio

    def _dynamic_preprocess(self, image, min_num=1, max_num=12, use_thumbnail=True):
        orig_width, orig_height = image.size
        aspect_ratio = orig_width / orig_height
        target_ratios = sorted(
            {
                (i, j)
                for n in range(min_num, max_num + 1)
                for i in range(1, n + 1)
                for j in range(1, n + 1)
                if min_num <= i * j <= max_num
            },
            key=lambda x: x[0] * x[1],
        )
        target_aspect_ratio = self._find_closest_aspect_ratio(
            aspect_ratio,
            target_ratios,
            orig_width,
            orig_height,
        )
        target_width = self._image_size * target_aspect_ratio[0]
        target_height = self._image_size * target_aspect_ratio[1]
        blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
        resized_img = image.resize((target_width, target_height))
        processed_images = []
        for idx in range(blocks):
            box = (
                (idx % (target_width // self._image_size)) * self._image_size,
                (idx // (target_width // self._image_size)) * self._image_size,
                ((idx % (target_width // self._image_size)) + 1) * self._image_size,
                ((idx // (target_width // self._image_size)) + 1) * self._image_size,
            )
            processed_images.append(resized_img.crop(box))
        if use_thumbnail and len(processed_images) != 1:
            processed_images.append(image.resize((self._image_size, self._image_size)))
        return processed_images

    def _load_image_tensor(self, image_path: str | None) -> torch.Tensor | None:
        if not image_path:
            return None
        transform = self._build_transform()
        image = Image.open(image_path).convert("RGB")
        pixel_values = torch.stack(
            [
                transform(tile)
                for tile in self._dynamic_preprocess(
                    image, max_num=12, use_thumbnail=True
                )
            ]
        )
        return pixel_values.to(torch.bfloat16).to(self._get_model_device())

    def _build_messages(
        self,
        prompt: AdaptInternVL3Prompt,
        few_shot_examples: list[dict] | None = None,
        system_prompt: str | None = None,
    ) -> list[dict]:
        messages: list[dict] = []
        if system_prompt:
            messages.append(
                {"role": "system", "content": [{"type": "text", "text": system_prompt}]}
            )
        for ex in few_shot_examples or []:
            messages.append(
                {"role": "user", "content": [{"type": "text", "text": ex["user_text"]}]}
            )
            messages.append(
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": ex["assistant_text"]}],
                }
            )
        messages.extend(prompt.get_messages())
        return messages

    def _build_generation_config(self) -> dict:
        generation_config = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.temperature > 0,
        }
        if self.temperature > 0:
            generation_config["temperature"] = self.temperature
        return generation_config

    def _inputs_for_generate(
        self, inputs: dict[str, torch.Tensor | None]
    ) -> dict[str, torch.Tensor | None]:
        cleaned = dict(inputs)
        cleaned.pop("image_flags", None)
        return cleaned

    def _prepare_question(
        self,
        prompt: AdaptInternVL3Prompt,
        history: list[tuple[str, str]] | None = None,
    ) -> str:
        question = prompt.user_text
        if prompt.image_path and "<image>" not in question:
            question = f"<image>\n{question}"
        if history is None and prompt.image_path and "<image>" not in question:
            question = f"<image>\n{question}"
        return question

    def _build_query(
        self,
        question: str,
        pixel_values: torch.Tensor | None,
        history: list[tuple[str, str]] | None = None,
        num_patches_list: list[int] | None = None,
    ) -> tuple[dict[str, torch.Tensor], str, list[int], int]:
        if num_patches_list is None:
            num_patches_list = (
                [pixel_values.shape[0]] if pixel_values is not None else []
            )
        if pixel_values is not None:
            assert len(pixel_values) == sum(num_patches_list)
        img_context_token = "<IMG_CONTEXT>"
        img_start_token = "<img>"
        img_end_token = "</img>"
        img_context_token_id = self.tokenizer.convert_tokens_to_ids(img_context_token)
        self.model.img_context_token_id = img_context_token_id
        template = self.model.conv_template.copy()
        template.system_message = self.model.system_message
        eos_token_id = self.tokenizer.convert_tokens_to_ids(template.sep.strip())
        history = [] if history is None else history
        for old_question, old_answer in history:
            template.append_message(template.roles[0], old_question)
            template.append_message(template.roles[1], old_answer)
        template.append_message(template.roles[0], question)
        template.append_message(template.roles[1], None)
        query = template.get_prompt()
        for num_patches in num_patches_list:
            image_tokens = (
                img_start_token
                + img_context_token * self.model.num_image_token * num_patches
                + img_end_token
            )
            query = query.replace("<image>", image_tokens, 1)
        model_inputs = self.tokenizer(query, return_tensors="pt")
        input_ids = model_inputs["input_ids"].to(self._get_model_device())
        attention_mask = model_inputs["attention_mask"].to(self._get_model_device())
        prepared_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
        }
        if pixel_values is not None:
            prepared_inputs["image_flags"] = torch.ones(
                (pixel_values.shape[0], 1),
                dtype=torch.long,
                device=pixel_values.device,
            )
        return prepared_inputs, query, num_patches_list, eos_token_id

    def _prepare_inputs(
        self,
        prompt: AdaptInternVL3Prompt,
        few_shot_examples: list[dict] | None = None,
        system_prompt: str | None = None,
    ):
        few_shot_examples = few_shot_examples or []
        patch_groups: list[torch.Tensor] = []
        num_patches_list: list[int] = []
        history: list[tuple[str, str]] = []
        for ex in few_shot_examples:
            q = ex["user_text"]
            p = ex.get("image_path")
            if p and os.path.isfile(p):
                pv = self._load_image_tensor(p)
                if pv is not None:
                    patch_groups.append(pv)
                    num_patches_list.append(int(pv.shape[0]))
                    if "<image>" not in q:
                        q = f"<image>\n{q}"
            history.append((q, ex["assistant_text"]))
        question = prompt.user_text
        if prompt.image_path and os.path.isfile(str(prompt.image_path)):
            pv_q = self._load_image_tensor(prompt.image_path)
            if pv_q is not None:
                patch_groups.append(pv_q)
                num_patches_list.append(int(pv_q.shape[0]))
            if "<image>" not in question:
                question = f"<image>\n{question}"
        pixel_values = torch.cat(patch_groups, dim=0) if patch_groups else None
        inputs, query, _, eos_token_id = self._build_query(
            question=question,
            pixel_values=pixel_values,
            history=history if history else None,
            num_patches_list=num_patches_list if num_patches_list else None,
        )
        generation_config = self._build_generation_config()
        generation_config["eos_token_id"] = eos_token_id
        return inputs, query, generation_config

    def get_completion(self, prompt: AdaptInternVL3Prompt) -> str:
        pixel_values = self._load_image_tensor(prompt.image_path)
        question = self._prepare_question(prompt)
        response = self.model.chat(
            self.tokenizer,
            pixel_values,
            question,
            self._build_generation_config(),
        )
        return self._strip_assistant_header(response.strip())

    def get_completion_conversation(
        self,
        system_prompt: str,
        few_shot_examples: list[dict],
        query_prompt: AdaptInternVL3Prompt,
    ) -> str:
        inputs, _, generation_config = self._prepare_inputs(
            query_prompt,
            few_shot_examples=few_shot_examples,
        )
        gen_inputs = self._inputs_for_generate(inputs)
        with torch.inference_mode():
            gen_ids = self.model.generate(
                **gen_inputs,
                **generation_config,
            )
        template = self.model.conv_template.copy()
        sep = template.sep.strip()
        response = self.tokenizer.batch_decode(gen_ids, skip_special_tokens=True)[0]
        response = response.split(sep)[0].strip()
        return self._strip_assistant_header(response)

    def get_completion_batch(self, prompts: list[AdaptInternVL3Prompt]) -> list[str]:
        return [self.get_completion(prompt) for prompt in prompts]

    def _forward_hidden_states(self, inputs: dict[str, torch.Tensor | None]):
        if inputs.get("pixel_values") is None:
            return self.model.language_model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                output_hidden_states=True,
                return_dict=True,
            )
        return self.model(**inputs, output_hidden_states=True, return_dict=True)

    def _generate_hidden_states(
        self,
        inputs: dict[str, torch.Tensor | None],
        generation_config: dict,
    ):
        gen_kwargs = {**generation_config, "max_new_tokens": 2, "min_new_tokens": 2}
        if inputs.get("pixel_values") is None:
            return self.model.language_model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                **gen_kwargs,
                output_hidden_states=True,
                return_dict_in_generate=True,
            )
        return self.model.generate(
            **self._inputs_for_generate(inputs),
            **gen_kwargs,
            output_hidden_states=True,
            return_dict_in_generate=True,
        )

    def get_all_layer_embeddings(
        self,
        prompt: AdaptInternVL3Prompt,
        few_shot_examples: Optional[list[dict]] = None,
    ) -> dict[str, torch.Tensor]:
        inputs, _, _ = self._prepare_inputs(prompt, few_shot_examples=few_shot_examples)
        with torch.inference_mode():
            outputs = self._forward_hidden_states(inputs)
        return {
            str(layer_idx): hidden[:, -1, :].squeeze(0).float()
            for layer_idx, hidden in enumerate(outputs.hidden_states)
        }

    def get_first_generated_token_embeddings(
        self,
        prompt: AdaptInternVL3Prompt,
        few_shot_examples: Optional[list[dict]] = None,
    ) -> dict[str, torch.Tensor]:
        inputs, _, generation_config = self._prepare_inputs(
            prompt, few_shot_examples=few_shot_examples
        )
        with torch.inference_mode():
            generated = self._generate_hidden_states(inputs, generation_config)
        gen_hidden = (
            generated.hidden_states[1]
            if len(generated.hidden_states) >= 2
            else generated.hidden_states[0]
        )
        return {
            str(layer_idx): hidden[:, -1, :].squeeze(0).float()
            for layer_idx, hidden in enumerate(gen_hidden)
        }

    def _get_image_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        candidate_ids = {
            self.tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>"),
            self.tokenizer.convert_tokens_to_ids("<img>"),
            self.tokenizer.convert_tokens_to_ids("</img>"),
        }
        unk_token_id = getattr(self.tokenizer, "unk_token_id", None)
        candidate_ids = {
            token_id
            for token_id in candidate_ids
            if isinstance(token_id, int) and token_id >= 0 and token_id != unk_token_id
        }
        if not candidate_ids:
            return torch.zeros(
                input_ids.shape[-1], dtype=torch.bool, device=input_ids.device
            )
        return _get_any_token_mask(input_ids, candidate_ids)

    def get_mean_all_tokens_embeddings(
        self,
        prompt: AdaptInternVL3Prompt,
        few_shot_examples: Optional[list[dict]] = None,
    ) -> dict[str, torch.Tensor]:
        inputs, _, _ = self._prepare_inputs(prompt, few_shot_examples=few_shot_examples)
        with torch.inference_mode():
            outputs = self._forward_hidden_states(inputs)
        return {
            str(layer_idx): hidden.squeeze(0).mean(dim=0).float()
            for layer_idx, hidden in enumerate(outputs.hidden_states)
        }

    def get_mean_image_tokens_embeddings(
        self,
        prompt: AdaptInternVL3Prompt,
        few_shot_examples: Optional[list[dict]] = None,
    ) -> dict[str, torch.Tensor]:
        inputs, _, _ = self._prepare_inputs(prompt, few_shot_examples=few_shot_examples)
        img_mask = self._get_image_mask(inputs["input_ids"])
        with torch.inference_mode():
            outputs = self._forward_hidden_states(inputs)
        return {
            str(layer_idx): _safe_mean_pool(hidden, img_mask)
            for layer_idx, hidden in enumerate(outputs.hidden_states)
        }

    def get_mean_text_tokens_embeddings(
        self,
        prompt: AdaptInternVL3Prompt,
        few_shot_examples: Optional[list[dict]] = None,
    ) -> dict[str, torch.Tensor]:
        inputs, _, _ = self._prepare_inputs(prompt, few_shot_examples=few_shot_examples)
        text_mask = ~self._get_image_mask(inputs["input_ids"])
        with torch.inference_mode():
            outputs = self._forward_hidden_states(inputs)
        return {
            str(layer_idx): _safe_mean_pool(hidden, text_mask)
            for layer_idx, hidden in enumerate(outputs.hidden_states)
        }

    def get_concat_img_text_last_embeddings(
        self,
        prompt: AdaptInternVL3Prompt,
        few_shot_examples: Optional[list[dict]] = None,
    ) -> dict[str, torch.Tensor]:
        inputs, _, _ = self._prepare_inputs(prompt, few_shot_examples=few_shot_examples)
        img_mask = self._get_image_mask(inputs["input_ids"])
        text_mask = ~img_mask
        with torch.inference_mode():
            outputs = self._forward_hidden_states(inputs)
        all_embs = {}
        for layer_idx, hidden in enumerate(outputs.hidden_states):
            img_mean = _safe_mean_pool(hidden, img_mask)
            txt_mean = _safe_mean_pool(hidden, text_mask)
            last_tok = hidden[:, -1, :].squeeze(0).float()
            all_embs[str(layer_idx)] = torch.cat([img_mean, txt_mean, last_tok], dim=0)
        return all_embs

    def get_all_layer_embeddings_batch(
        self, prompts: list[AdaptInternVL3Prompt]
    ) -> dict[str, torch.Tensor]:
        all_layer_embs: dict[str, list[torch.Tensor]] = {}
        for prompt in prompts:
            single_embs = self.get_all_layer_embeddings(prompt)
            for layer_key, emb in single_embs.items():
                all_layer_embs.setdefault(layer_key, []).append(emb)
        return {k: torch.stack(v, dim=0) for k, v in all_layer_embs.items()}

class InternVL3Prompter(AdaptInternVL3Prompter):
    def __init__(
        self,
        model_stem: str = "internvl3-1b",
        model_id: str = "OpenGVLab/InternVL3-1B",
        max_new_tokens: int = 1024,
        temperature: float = 0.0,
    ):
        super().__init__(
            model_stem=model_stem,
            model_id=model_id,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )

    def _prepare_question(
        self,
        prompt: InternVL3Prompt,
        history: list[tuple[str, str]] | None = None,
    ) -> str:
        question = prompt.user_text
        if prompt.image_path and "<image>" not in question:
            question = f"<image>\n{question}"
        return question

class RandomInternVL3Prompter(AdaptInternVL3Prompter):
    def __init__(
        self,
        model_stem: str,
        model_id: str = "OpenGVLab/InternVL3-1B",
        max_new_tokens: int = 1024,
        temperature: float = 0.0,
        device: str | None = None,
        dtype: torch.dtype = torch.bfloat16,
        cache_root: str = "/raid/rsq813/random_init",
        force_reinit: int = 0,
        seed: int | None = None,
    ):
        BasePrompter.__init__(self, model_stem, model_id, max_new_tokens, temperature)
        from transformers import AutoConfig, AutoModel, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=True,
            use_fast=False,
        )
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        target_device = torch.device(device)
        dtype_tag = (
            "bf16" if dtype == torch.bfloat16 else str(dtype).replace("torch.", "")
        )
        seed_tag = f"_seed{seed}" if seed is not None else ""
        save_dir = (
            Path(cache_root)
            / _sanitize_repo_id(model_id)
            / f"dtype_{dtype_tag}{seed_tag}"
        )
        save_dir.mkdir(parents=True, exist_ok=True)
        tokenizer_files = (
            save_dir / "tokenizer_config.json",
            save_dir / "special_tokens_map.json",
        )
        if all(path.exists() for path in tokenizer_files):
            self.tokenizer = AutoTokenizer.from_pretrained(
                str(save_dir),
                trust_remote_code=True,
                use_fast=False,
            )
        if _model_weight_files_exist(save_dir) and not force_reinit:
            print(f"[cache] Loading cached random InternVL3 model from: {save_dir}")
            t0 = time.time()
            load_kwargs = {
                "torch_dtype": dtype,
                "trust_remote_code": True,
            }
            try:
                with _force_cpu_linspace(), _compat_mark_tied_weights():
                    self.model = AutoModel.from_pretrained(
                        str(save_dir),
                        **load_kwargs,
                    )
                self.model = self.model.to(target_device)
                print(f"[cache] Loaded in {time.time() - t0:.1f}s on {device}")
            except Exception as e:
                print(
                    f"[cache] Cached InternVL3 checkpoint is incomplete or unreadable: {e}"
                )
                print(f"[cache] Rebuilding random InternVL3 model in: {save_dir}")
                _clear_directory_contents(save_dir)
            else:
                self.model = self.model.eval()
                self._image_size = 448
                self._image_mean = (0.485, 0.456, 0.406)
                self._image_std = (0.229, 0.224, 0.225)
                return
        print(f"[build] Building random InternVL3 model for {model_stem} on {device}")
        config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        if seed is not None:
            print(f"[build] Setting random seed to {seed}")
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        old_default = torch.get_default_dtype()
        torch.set_default_dtype(dtype)
        try:
            t0 = time.time()
            with _force_cpu_linspace(), _compat_mark_tied_weights():
                model = AutoModel.from_config(
                    config,
                    trust_remote_code=True,
                )
            model = model.to(target_device)
            model = model.eval()
            print(f"[build] Constructed + moved in {time.time() - t0:.1f}s")
        finally:
            torch.set_default_dtype(old_default)
        self.model = model
        self._image_size = 448
        self._image_mean = (0.485, 0.456, 0.406)
        self._image_std = (0.229, 0.224, 0.225)
        print(f"[save] Saving random InternVL3 model to: {save_dir}")
        t0 = time.time()
        _clear_directory_contents(save_dir)
        cpu_model = self.model.to("cpu")
        cpu_model.save_pretrained(str(save_dir), safe_serialization=True)
        self.tokenizer.save_pretrained(str(save_dir))
        print(f"[save] Done in {time.time() - t0:.1f}s")
        self.model = cpu_model.to(target_device).eval()

    def _prepare_question(
        self,
        prompt: InternVL3Prompt,
        history: list[tuple[str, str]] | None = None,
    ) -> str:
        question = prompt.user_text
        if prompt.image_path and "<image>" not in question:
            question = f"<image>\n{question}"
        return question

class RandomInternVL3Prompter1(RandomInternVL3Prompter):
    def __init__(
        self,
        model_stem: str = "random_InternVL3-1B_1",
        model_id: str = "OpenGVLab/InternVL3-1B",
        max_new_tokens: int = 1024,
        temperature: float = 0.0,
        device: str | None = None,
        dtype: torch.dtype = torch.bfloat16,
        cache_root: str = "/raid/rsq813/random_init",
        force_reinit: int = 0,
    ):
        super().__init__(
            model_stem=model_stem,
            model_id=model_id,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            device=device,
            dtype=dtype,
            cache_root=cache_root,
            force_reinit=force_reinit,
            seed=1,
        )

class RandomInternVL3Prompter2(RandomInternVL3Prompter):
    def __init__(
        self,
        model_stem: str = "random_InternVL3-1B_2",
        model_id: str = "OpenGVLab/InternVL3-1B",
        max_new_tokens: int = 1024,
        temperature: float = 0.0,
        device: str | None = None,
        dtype: torch.dtype = torch.bfloat16,
        cache_root: str = "/raid/rsq813/random_init",
        force_reinit: int = 0,
    ):
        super().__init__(
            model_stem=model_stem,
            model_id=model_id,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            device=device,
            dtype=dtype,
            cache_root=cache_root,
            force_reinit=force_reinit,
            seed=2,
        )

class RandomInternVL3Prompter3(RandomInternVL3Prompter):
    def __init__(
        self,
        model_stem: str = "random_InternVL3-1B_3",
        model_id: str = "OpenGVLab/InternVL3-1B",
        max_new_tokens: int = 1024,
        temperature: float = 0.0,
        device: str | None = None,
        dtype: torch.dtype = torch.bfloat16,
        cache_root: str = "/raid/rsq813/random_init",
        force_reinit: int = 0,
    ):
        super().__init__(
            model_stem=model_stem,
            model_id=model_id,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            device=device,
            dtype=dtype,
            cache_root=cache_root,
            force_reinit=force_reinit,
            seed=3,
        )

class BioMedLlamaPrompter(BasePrompter):
    def __init__(
        self,
        model_id: str = "ContactDoctor/Bio-Medical-MultiModal-Llama-3-8B-V1",
        max_new_tokens: int = 1024,
        temperature: float = 0.95,
    ):
        super().__init__("biomedllama", model_id, max_new_tokens, temperature)
        from transformers import AutoModel, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        ).eval()

    def get_completion(self, prompt: BioMedLlamaPrompt) -> str:
        full_question = ""
        if prompt.system_prompt:
            full_question += prompt.system_prompt + "\n\n"
        full_question += prompt.user_text
        msgs = [
            {
                "role": "user",
                "content": (
                    [prompt.image, full_question] if prompt.image else [full_question]
                ),
            }
        ]
        res = self.model.chat(
            image=prompt.image,
            msgs=msgs,
            tokenizer=self.tokenizer,
            sampling=True,
            temperature=self.temperature,
        )
        if hasattr(res, "__iter__") and not isinstance(res, str):
            output = "".join(list(res))
        else:
            output = str(res)
        return self._strip_assistant_header(output)

    def get_completion_conversation(
        self,
        system_prompt: str,
        few_shot_examples: list[dict],
        query_prompt: BioMedLlamaPrompt,
    ) -> str:
        full_question = ""
        if system_prompt:
            full_question += system_prompt + "\n\n"
        for ex in few_shot_examples:
            full_question += (
                f"User: {ex['user_text']}\nAssistant: {ex['assistant_text']}\n\n"
            )
        full_question += query_prompt.user_text
        msgs = [
            {
                "role": "user",
                "content": (
                    [query_prompt.image, full_question]
                    if query_prompt.image
                    else [full_question]
                ),
            }
        ]
        res = self.model.chat(
            image=query_prompt.image,
            msgs=msgs,
            tokenizer=self.tokenizer,
            sampling=True,
            temperature=self.temperature,
        )
        if hasattr(res, "__iter__") and not isinstance(res, str):
            output = "".join(list(res))
        else:
            output = str(res)
        return self._strip_assistant_header(output)

    def get_completion_batch(self, prompts: list[BioMedLlamaPrompt]) -> list[str]:
        return [self.get_completion(p) for p in prompts]

    def get_all_layer_embeddings(
        self, prompt: BioMedLlamaPrompt
    ) -> dict[str, torch.Tensor]:
        full_text = ""
        if prompt.system_prompt:
            full_text += prompt.system_prompt + "\n\n"
        full_text += prompt.user_text
        inputs = self.tokenizer(full_text, return_tensors="pt").to(self.model.device)
        with torch.inference_mode():
            llm = self.model.llm if hasattr(self.model, "llm") else self.model
            outputs = llm(**inputs, output_hidden_states=True)
            all_embs = {}
            for layer_idx, hidden in enumerate(outputs.hidden_states):
                embs = hidden[:, -1, :].squeeze(0).float()
                all_embs[str(layer_idx)] = embs
        return all_embs

    def get_first_generated_token_embeddings(
        self, prompt: BioMedLlamaPrompt
    ) -> dict[str, torch.Tensor]:
        full_text = ""
        if prompt.system_prompt:
            full_text += prompt.system_prompt + "\n\n"
        full_text += prompt.user_text
        inputs = self.tokenizer(full_text, return_tensors="pt").to(self.model.device)
        with torch.inference_mode():
            llm = self.model.llm if hasattr(self.model, "llm") else self.model
            outputs = llm(**inputs, output_hidden_states=True)
            last_hidden = outputs.hidden_states[-1]
            logits = (
                outputs.logits
                if hasattr(outputs, "logits")
                else llm.lm_head(last_hidden)
            )
            next_token_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            new_inputs = {"input_ids": next_token_id}
            if hasattr(outputs, "past_key_values") and outputs.past_key_values:
                new_inputs["past_key_values"] = outputs.past_key_values
            gen_outputs = llm(**new_inputs, output_hidden_states=True)
            all_embs = {}
            for layer_idx, hidden in enumerate(gen_outputs.hidden_states):
                embs = hidden[:, -1, :].squeeze(0).float()
                all_embs[str(layer_idx)] = embs
        return all_embs

    def get_mean_all_tokens_embeddings(
        self, prompt: BioMedLlamaPrompt
    ) -> dict[str, torch.Tensor]:
        full_text = ""
        if prompt.system_prompt:
            full_text += prompt.system_prompt + "\n\n"
        full_text += prompt.user_text
        inputs = self.tokenizer(full_text, return_tensors="pt").to(self.model.device)
        with torch.inference_mode():
            llm = self.model.llm if hasattr(self.model, "llm") else self.model
            outputs = llm(**inputs, output_hidden_states=True)
            all_embs = {}
            for layer_idx, hidden in enumerate(outputs.hidden_states):
                all_embs[str(layer_idx)] = hidden.squeeze(0).mean(dim=0).float()
        return all_embs

    def get_mean_image_tokens_embeddings(
        self, prompt: BioMedLlamaPrompt
    ) -> dict[str, torch.Tensor]:
        return self.get_mean_all_tokens_embeddings(prompt)

    def get_mean_text_tokens_embeddings(
        self, prompt: BioMedLlamaPrompt
    ) -> dict[str, torch.Tensor]:
        return self.get_mean_all_tokens_embeddings(prompt)

    def get_concat_img_text_last_embeddings(
        self, prompt: BioMedLlamaPrompt
    ) -> dict[str, torch.Tensor]:
        full_text = ""
        if prompt.system_prompt:
            full_text += prompt.system_prompt + "\n\n"
        full_text += prompt.user_text
        inputs = self.tokenizer(full_text, return_tensors="pt").to(self.model.device)
        with torch.inference_mode():
            llm = self.model.llm if hasattr(self.model, "llm") else self.model
            outputs = llm(**inputs, output_hidden_states=True)
            all_embs = {}
            for layer_idx, hidden in enumerate(outputs.hidden_states):
                mean_all = hidden.squeeze(0).mean(dim=0).float()
                last_tok = hidden[:, -1, :].squeeze(0).float()
                all_embs[str(layer_idx)] = torch.cat(
                    [mean_all, mean_all, last_tok], dim=0
                )
        return all_embs

    def get_all_layer_embeddings_batch(
        self, prompts: list[BioMedLlamaPrompt]
    ) -> dict[str, torch.Tensor]:
        all_layer_embs: dict[str, list[torch.Tensor]] = {}
        for p in prompts:
            single_embs = self.get_all_layer_embeddings(p)
            for layer_key, emb in single_embs.items():
                if layer_key not in all_layer_embs:
                    all_layer_embs[layer_key] = []
                all_layer_embs[layer_key].append(emb)
        return {k: torch.stack(v, dim=0) for k, v in all_layer_embs.items()}

class AdaptLlamaPrompter(BasePrompter):
    def __init__(
        self,
        model_id: str = "AdaptLLM/biomed-Llama-3.2-11B-Vision-Instruct",
        max_new_tokens: int = 30,
        temperature: float = 0.0,
    ):
        super().__init__("adapt-llama3.2-11b", model_id, max_new_tokens, temperature)
        from transformers import AutoProcessor, MllamaForConditionalGeneration

        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = MllamaForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        ).eval()

    def _placeholder_image(self) -> Image.Image:
        image_size = 224
        processor_image = getattr(self.processor, "image_processor", None)
        if processor_image is not None:
            size_cfg = getattr(processor_image, "size", None)
            if isinstance(size_cfg, dict):
                image_size = (
                    size_cfg.get("height")
                    or size_cfg.get("shortest_edge")
                    or size_cfg.get("width")
                    or image_size
                )
            elif isinstance(size_cfg, int):
                image_size = size_cfg
        return Image.new("RGB", (image_size, image_size), color=(0, 0, 0))

    def _prepare_inputs(
        self,
        prompt: AdaptLlamaPrompt,
        few_shot_examples: list[dict] | None = None,
        system_prompt: str | None = None,
    ):
        messages = []
        if system_prompt:
            messages.append(
                {"role": "system", "content": [{"type": "text", "text": system_prompt}]}
            )
        pil_images: list[Image.Image] = []
        for ex in few_shot_examples or []:
            parts: list[dict] = []
            path = ex.get("image_path")
            if path and os.path.isfile(path):
                parts.append({"type": "image"})
                pil_images.append(Image.open(path).convert("RGB"))
            parts.append({"type": "text", "text": ex["user_text"]})
            messages.append({"role": "user", "content": parts})
            messages.append(
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": ex["assistant_text"]}],
                }
            )
        messages.extend(prompt.get_messages())
        image = prompt.image
        if image is None:
            user_msg = next(
                (m for m in reversed(messages) if m.get("role") == "user"), None
            )
            if user_msg is not None:
                content = user_msg.get("content", [])
                if isinstance(content, list) and not any(
                    c.get("type") == "image" for c in content if isinstance(c, dict)
                ):
                    user_msg["content"] = [{"type": "image"}] + content
            image = self._placeholder_image()
        if image is not None:
            pil_images.append(image)
        input_text = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
        )
        if pil_images:
            if len(pil_images) == 1:
                inputs = self.processor(
                    pil_images[0],
                    input_text,
                    add_special_tokens=False,
                    return_tensors="pt",
                )
            else:
                inputs = self.processor(
                    images=pil_images,
                    text=input_text,
                    add_special_tokens=False,
                    return_tensors="pt",
                )
        else:
            inputs = self.processor(
                text=input_text,
                add_special_tokens=False,
                return_tensors="pt",
            )
        return inputs.to(self.model.device), input_text

    def get_completion(self, prompt: AdaptLlamaPrompt) -> str:
        inputs, _ = self._prepare_inputs(prompt)
        with torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
            )
        prompt_len = inputs["input_ids"].shape[1]
        gen_ids = output[0, prompt_len:]
        out = self.processor.decode(gen_ids, skip_special_tokens=True).strip()
        return self._strip_assistant_header(out)

    def get_completion_conversation(
        self,
        system_prompt: str,
        few_shot_examples: list[dict],
        query_prompt: AdaptLlamaPrompt,
    ) -> str:
        inputs, _ = self._prepare_inputs(
            query_prompt,
            few_shot_examples=few_shot_examples,
            system_prompt=system_prompt or None,
        )
        with torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
            )
        prompt_len = inputs["input_ids"].shape[1]
        gen_ids = output[0, prompt_len:]
        out = self.processor.decode(gen_ids, skip_special_tokens=True).strip()
        return self._strip_assistant_header(out)

    def get_completion_batch(self, prompts: list[AdaptLlamaPrompt]) -> list[str]:
        return [self.get_completion(p) for p in prompts]

    def get_all_layer_embeddings(
        self,
        prompt: AdaptLlamaPrompt,
        few_shot_examples: Optional[list[dict]] = None,
    ) -> dict[str, torch.Tensor]:
        inputs, _ = self._prepare_inputs(prompt, few_shot_examples=few_shot_examples)
        with torch.inference_mode():
            outputs = self.model(**inputs, output_hidden_states=True)
        all_embs = {}
        for layer_idx, hidden in enumerate(outputs.hidden_states):
            all_embs[str(layer_idx)] = hidden[:, -1, :].squeeze(0).float()
        return all_embs

    def get_first_generated_token_embeddings(
        self,
        prompt: AdaptLlamaPrompt,
        few_shot_examples: Optional[list[dict]] = None,
    ) -> dict[str, torch.Tensor]:
        inputs, _ = self._prepare_inputs(prompt, few_shot_examples=few_shot_examples)
        with torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=2,
                output_hidden_states=True,
                return_dict_in_generate=True,
            )
        if len(generated.hidden_states) >= 2:
            gen_hidden = generated.hidden_states[1]
        else:
            gen_hidden = generated.hidden_states[0]
        all_embs = {}
        for layer_idx, hidden in enumerate(gen_hidden):
            all_embs[str(layer_idx)] = hidden[:, -1, :].squeeze(0).float()
        return all_embs

    def _get_image_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        img_tok_id = getattr(self.model.config, "image_token_index", None)
        if img_tok_id is None:
            img_tok_id = getattr(self.model.config, "image_token_id", None)
        if img_tok_id is None and hasattr(self.processor, "tokenizer"):
            img_tok_id = self.processor.tokenizer.convert_tokens_to_ids("<image>")
        if img_tok_id is None or img_tok_id < 0:
            return torch.zeros(
                input_ids.shape[-1], dtype=torch.bool, device=input_ids.device
            )
        return _get_image_token_mask(input_ids, img_tok_id)

    def get_mean_all_tokens_embeddings(
        self,
        prompt: AdaptLlamaPrompt,
        few_shot_examples: Optional[list[dict]] = None,
    ) -> dict[str, torch.Tensor]:
        inputs, _ = self._prepare_inputs(prompt, few_shot_examples=few_shot_examples)
        with torch.inference_mode():
            outputs = self.model(**inputs, output_hidden_states=True)
        all_embs = {}
        for layer_idx, hidden in enumerate(outputs.hidden_states):
            all_embs[str(layer_idx)] = hidden.squeeze(0).mean(dim=0).float()
        return all_embs

    def get_mean_image_tokens_embeddings(
        self,
        prompt: AdaptLlamaPrompt,
        few_shot_examples: Optional[list[dict]] = None,
    ) -> dict[str, torch.Tensor]:
        inputs, _ = self._prepare_inputs(prompt, few_shot_examples=few_shot_examples)
        img_mask = self._get_image_mask(inputs["input_ids"])
        with torch.inference_mode():
            outputs = self.model(**inputs, output_hidden_states=True)
        all_embs = {}
        for layer_idx, hidden in enumerate(outputs.hidden_states):
            all_embs[str(layer_idx)] = _safe_mean_pool(hidden, img_mask)
        return all_embs

    def get_mean_text_tokens_embeddings(
        self,
        prompt: AdaptLlamaPrompt,
        few_shot_examples: Optional[list[dict]] = None,
    ) -> dict[str, torch.Tensor]:
        inputs, _ = self._prepare_inputs(prompt, few_shot_examples=few_shot_examples)
        text_mask = ~self._get_image_mask(inputs["input_ids"])
        with torch.inference_mode():
            outputs = self.model(**inputs, output_hidden_states=True)
        all_embs = {}
        for layer_idx, hidden in enumerate(outputs.hidden_states):
            all_embs[str(layer_idx)] = _safe_mean_pool(hidden, text_mask)
        return all_embs

    def get_concat_img_text_last_embeddings(
        self,
        prompt: AdaptLlamaPrompt,
        few_shot_examples: Optional[list[dict]] = None,
    ) -> dict[str, torch.Tensor]:
        inputs, _ = self._prepare_inputs(prompt, few_shot_examples=few_shot_examples)
        img_mask = self._get_image_mask(inputs["input_ids"])
        text_mask = ~img_mask
        with torch.inference_mode():
            outputs = self.model(**inputs, output_hidden_states=True)
        all_embs = {}
        for layer_idx, hidden in enumerate(outputs.hidden_states):
            img_mean = _safe_mean_pool(hidden, img_mask)
            txt_mean = _safe_mean_pool(hidden, text_mask)
            last_tok = hidden[:, -1, :].squeeze(0).float()
            all_embs[str(layer_idx)] = torch.cat([img_mean, txt_mean, last_tok], dim=0)
        return all_embs

    def get_all_layer_embeddings_batch(
        self, prompts: list[AdaptLlamaPrompt]
    ) -> dict[str, torch.Tensor]:
        all_layer_embs: dict[str, list[torch.Tensor]] = {}
        for p in prompts:
            single_embs = self.get_all_layer_embeddings(p)
            for layer_key, emb in single_embs.items():
                all_layer_embs.setdefault(layer_key, []).append(emb)
        return {k: torch.stack(v, dim=0) for k, v in all_layer_embs.items()}

class MetaLlama32VisionPrompter(BasePrompter):
    def __init__(
        self,
        model_stem: str = "meta-llama3.2-11b-vision-instruct",
        model_id: str = "meta-llama/Llama-3.2-11B-Vision-Instruct",
        processor_src: str | None = None,
        max_new_tokens: int = 30,
        temperature: float = 0.0,
    ):
        super().__init__(model_stem, model_id, max_new_tokens, temperature)
        from transformers import AutoProcessor, MllamaForConditionalGeneration

        dotenv.load_dotenv("./.env")
        hf_token = (
            os.getenv("HF_TOKEN_LLAMA32_11B")
            or os.getenv("HF_TOKEN")
            or os.getenv("HUGGINGFACE_TOKEN")
            or os.getenv("HUGGINGFACE_HUB_TOKEN")
        )
        processor_kwargs = {}
        model_kwargs = {
            "torch_dtype": torch.bfloat16,
            "device_map": "auto",
        }
        if hf_token:
            processor_kwargs["token"] = hf_token
            model_kwargs["token"] = hf_token
        processor_src = processor_src or model_id
        self.processor = AutoProcessor.from_pretrained(
            processor_src, **processor_kwargs
        )
        self.model = MllamaForConditionalGeneration.from_pretrained(
            model_id,
            **model_kwargs,
        ).eval()

    def _placeholder_image(self) -> Image.Image:
        image_size = 224
        processor_image = getattr(self.processor, "image_processor", None)
        if processor_image is not None:
            size_cfg = getattr(processor_image, "size", None)
            if isinstance(size_cfg, dict):
                image_size = (
                    size_cfg.get("height")
                    or size_cfg.get("shortest_edge")
                    or size_cfg.get("width")
                    or image_size
                )
            elif isinstance(size_cfg, int):
                image_size = size_cfg
        return Image.new("RGB", (image_size, image_size), color=(0, 0, 0))

    def _prepare_inputs(
        self,
        prompt: MetaLlama32VisionPrompt,
        few_shot_examples: list[dict] | None = None,
        system_prompt: str | None = None,
    ):
        messages: list[dict] = []
        if system_prompt:
            messages.append(
                {"role": "system", "content": [{"type": "text", "text": system_prompt}]}
            )
        pil_images: list[Image.Image] = []
        for ex in few_shot_examples or []:
            image_path = ex.get("image_path")
            if image_path and os.path.isfile(image_path):
                pil_images.append(Image.open(image_path).convert("RGB"))
            else:
                pil_images.append(self._placeholder_image())
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": ex["user_text"]},
                    ],
                }
            )
            messages.append(
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": ex["assistant_text"]}],
                }
            )
        messages.extend(prompt.get_messages())
        query_image = prompt.image
        if query_image is None:
            user_msg = next(
                (m for m in reversed(messages) if m.get("role") == "user"), None
            )
            if user_msg is not None:
                content = user_msg.get("content", [])
                if isinstance(content, list) and not any(
                    c.get("type") == "image" for c in content if isinstance(c, dict)
                ):
                    user_msg["content"] = [{"type": "image"}] + content
            query_image = self._placeholder_image()
        pil_images.append(query_image)
        input_text = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
        )
        if len(pil_images) == 1:
            inputs = self.processor(
                pil_images[0],
                input_text,
                add_special_tokens=False,
                return_tensors="pt",
            )
        else:
            inputs = self.processor(
                images=pil_images,
                text=input_text,
                add_special_tokens=False,
                return_tensors="pt",
            )
        return inputs.to(self.model.device), input_text

    def get_completion(self, prompt: MetaLlama32VisionPrompt) -> str:
        inputs, _ = self._prepare_inputs(prompt)
        with torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
            )
        prompt_len = inputs["input_ids"].shape[1]
        gen_ids = output[0, prompt_len:]
        out = self.processor.decode(gen_ids, skip_special_tokens=True).strip()
        return self._strip_assistant_header(out)

    def get_completion_conversation(
        self,
        system_prompt: str,
        few_shot_examples: list[dict],
        query_prompt: MetaLlama32VisionPrompt,
    ) -> str:
        inputs, _ = self._prepare_inputs(
            query_prompt,
            few_shot_examples=few_shot_examples,
            system_prompt=system_prompt or None,
        )
        with torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
            )
        prompt_len = inputs["input_ids"].shape[1]
        gen_ids = output[0, prompt_len:]
        out = self.processor.decode(gen_ids, skip_special_tokens=True).strip()
        return self._strip_assistant_header(out)

    def get_completion_batch(self, prompts: list[MetaLlama32VisionPrompt]) -> list[str]:
        return [self.get_completion(p) for p in prompts]

    def get_all_layer_embeddings(
        self, prompt: MetaLlama32VisionPrompt
    ) -> dict[str, torch.Tensor]:
        inputs, _ = self._prepare_inputs(prompt)
        with torch.inference_mode():
            outputs = self.model(**inputs, output_hidden_states=True)
        all_embs = {}
        for layer_idx, hidden in enumerate(outputs.hidden_states):
            all_embs[str(layer_idx)] = hidden[:, -1, :].squeeze(0).float()
        return all_embs

    def get_first_generated_token_embeddings(
        self, prompt: MetaLlama32VisionPrompt
    ) -> dict[str, torch.Tensor]:
        inputs, _ = self._prepare_inputs(prompt)
        with torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=2,
                output_hidden_states=True,
                return_dict_in_generate=True,
            )
        if len(generated.hidden_states) >= 2:
            gen_hidden = generated.hidden_states[1]
        else:
            gen_hidden = generated.hidden_states[0]
        all_embs = {}
        for layer_idx, hidden in enumerate(gen_hidden):
            all_embs[str(layer_idx)] = hidden[:, -1, :].squeeze(0).float()
        return all_embs

    def _get_image_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        img_tok_id = getattr(self.model.config, "image_token_index", None)
        if img_tok_id is None:
            img_tok_id = getattr(self.model.config, "image_token_id", None)
        if img_tok_id is None and hasattr(self.processor, "tokenizer"):
            img_tok_id = self.processor.tokenizer.convert_tokens_to_ids("<image>")
        if img_tok_id is None or img_tok_id < 0:
            return torch.zeros(
                input_ids.shape[-1], dtype=torch.bool, device=input_ids.device
            )
        return _get_image_token_mask(input_ids, img_tok_id)

    def get_mean_all_tokens_embeddings(
        self, prompt: MetaLlama32VisionPrompt
    ) -> dict[str, torch.Tensor]:
        inputs, _ = self._prepare_inputs(prompt)
        with torch.inference_mode():
            outputs = self.model(**inputs, output_hidden_states=True)
        all_embs = {}
        for layer_idx, hidden in enumerate(outputs.hidden_states):
            all_embs[str(layer_idx)] = hidden.squeeze(0).mean(dim=0).float()
        return all_embs

    def get_mean_image_tokens_embeddings(
        self, prompt: MetaLlama32VisionPrompt
    ) -> dict[str, torch.Tensor]:
        inputs, _ = self._prepare_inputs(prompt)
        img_mask = self._get_image_mask(inputs["input_ids"])
        with torch.inference_mode():
            outputs = self.model(**inputs, output_hidden_states=True)
        all_embs = {}
        for layer_idx, hidden in enumerate(outputs.hidden_states):
            all_embs[str(layer_idx)] = _safe_mean_pool(hidden, img_mask)
        return all_embs

    def get_mean_text_tokens_embeddings(
        self, prompt: MetaLlama32VisionPrompt
    ) -> dict[str, torch.Tensor]:
        inputs, _ = self._prepare_inputs(prompt)
        text_mask = ~self._get_image_mask(inputs["input_ids"])
        with torch.inference_mode():
            outputs = self.model(**inputs, output_hidden_states=True)
        all_embs = {}
        for layer_idx, hidden in enumerate(outputs.hidden_states):
            all_embs[str(layer_idx)] = _safe_mean_pool(hidden, text_mask)
        return all_embs

    def get_concat_img_text_last_embeddings(
        self, prompt: MetaLlama32VisionPrompt
    ) -> dict[str, torch.Tensor]:
        inputs, _ = self._prepare_inputs(prompt)
        img_mask = self._get_image_mask(inputs["input_ids"])
        text_mask = ~img_mask
        with torch.inference_mode():
            outputs = self.model(**inputs, output_hidden_states=True)
        all_embs = {}
        for layer_idx, hidden in enumerate(outputs.hidden_states):
            img_mean = _safe_mean_pool(hidden, img_mask)
            txt_mean = _safe_mean_pool(hidden, text_mask)
            last_tok = hidden[:, -1, :].squeeze(0).float()
            all_embs[str(layer_idx)] = torch.cat([img_mean, txt_mean, last_tok], dim=0)
        return all_embs

    def get_all_layer_embeddings_batch(
        self, prompts: list[MetaLlama32VisionPrompt]
    ) -> dict[str, torch.Tensor]:
        all_layer_embs: dict[str, list[torch.Tensor]] = {}
        for p in prompts:
            single_embs = self.get_all_layer_embeddings(p)
            for layer_key, emb in single_embs.items():
                all_layer_embs.setdefault(layer_key, []).append(emb)
        return {k: torch.stack(v, dim=0) for k, v in all_layer_embs.items()}

class GemmaPrompter(BasePrompter):
    def __init__(
        self,
        model_id: str = "google/gemma-3-4b-it",
        max_new_tokens: int = 200,
        temperature: float = 0.0,
    ):
        super().__init__("gemma", model_id, max_new_tokens, temperature)
        from transformers import AutoProcessor, AutoModelForImageTextToText

        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        ).eval()

    def _prepare_inputs(self, prompt: GemmaPrompt):
        messages = prompt.get_messages()
        if hasattr(self.processor, "chat_template") and self.processor.chat_template:
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            warnings.warn(
                f"Processor for {self.model_id} has no chat_template. "
                "Falling back to plain-text formatting.",
                UserWarning,
            )
            user_content = ""
            if prompt.system_prompt:
                user_content += prompt.system_prompt + "\n\n"
            user_content += prompt.user_text
            text = user_content
        kw = dict(text=text, return_tensors="pt")
        if prompt.image is not None:
            kw["images"] = prompt.image
        inputs = self.processor(**kw)
        return inputs.to(self.model.device), text

    def _prepare_conversation_inputs(self, messages: list[dict]):
        if hasattr(self.processor, "chat_template") and self.processor.chat_template:
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            text = "\n\n".join(
                (
                    message.get("content", "")
                    if isinstance(message.get("content", ""), str)
                    else next(
                        (
                            part.get("text", "")
                            for part in message.get("content", [])
                            if isinstance(part, dict) and part.get("type") == "text"
                        ),
                        "",
                    )
                )
                for message in messages
                if message.get("role") == "user"
            )
        image_inputs, _ = _collect_vision_inputs_from_messages(messages)
        kw = dict(text=text, return_tensors="pt")
        if image_inputs:
            kw["images"] = image_inputs
        inputs = self.processor(**kw)
        return inputs.to(self.model.device), text

    def get_completion(self, prompt: GemmaPrompt) -> str:
        inputs, text = self._prepare_inputs(prompt)
        with torch.inference_mode():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=self.temperature > 0,
            )
        prompt_len = inputs["input_ids"].shape[1]
        gen_ids = generated_ids[0, prompt_len:]
        out = self.processor.decode(gen_ids, skip_special_tokens=True).strip()
        return self._strip_assistant_header(out)

    def get_completion_conversation(
        self,
        system_prompt: str,
        few_shot_examples: list[dict],
        query_prompt: GemmaPrompt,
    ) -> str:
        messages = []
        if few_shot_examples:
            first_user_text = few_shot_examples[0]["user_text"]
            if system_prompt:
                first_user_text = f"{system_prompt}\n\n{first_user_text}"
            first_content = [{"type": "text", "text": first_user_text}]
            if few_shot_examples[0].get("image_path"):
                first_content.append(
                    {"type": "image", "image": few_shot_examples[0]["image_path"]}
                )
            messages.append({"role": "user", "content": first_content})
            messages.append(
                {"role": "model", "content": few_shot_examples[0]["assistant_text"]}
            )
            for ex in few_shot_examples[1:]:
                content = [{"type": "text", "text": ex["user_text"]}]
                if ex.get("image_path"):
                    content.append({"type": "image", "image": ex["image_path"]})
                messages.append({"role": "user", "content": content})
                messages.append({"role": "model", "content": ex["assistant_text"]})
        elif system_prompt:
            messages.append({"role": "user", "content": system_prompt})
        content = [{"type": "text", "text": query_prompt.user_text}]
        if query_prompt.image_path:
            content.append({"type": "image", "image": query_prompt.image_path})
        messages.append({"role": "user", "content": content})
        inputs, text = self._prepare_conversation_inputs(messages)
        with torch.inference_mode():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=self.temperature > 0,
            )
        prompt_len = inputs["input_ids"].shape[1]
        gen_ids = generated_ids[0, prompt_len:]
        out = self.processor.decode(gen_ids, skip_special_tokens=True).strip()
        return self._strip_assistant_header(out)

    def get_completion_batch(self, prompts: list[GemmaPrompt]) -> list[str]:
        return [self.get_completion(p) for p in prompts]

    def get_all_layer_embeddings(self, prompt: GemmaPrompt) -> dict[str, torch.Tensor]:
        inputs, _ = self._prepare_inputs(prompt)
        with torch.inference_mode():
            outputs = self.model(**inputs, output_hidden_states=True)
            all_embs = {}
            for layer_idx, hidden in enumerate(outputs.hidden_states):
                embs = hidden[:, -1, :].squeeze(0).float()
                all_embs[str(layer_idx)] = embs
        return all_embs

    def get_first_generated_token_embeddings(
        self, prompt: GemmaPrompt
    ) -> dict[str, torch.Tensor]:
        inputs, _ = self._prepare_inputs(prompt)
        with torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=2,
                min_new_tokens=2,
                do_sample=False,
                output_hidden_states=True,
                return_dict_in_generate=True,
            )
            all_embs = {}
            if len(generated.hidden_states) >= 2:
                gen_hidden = generated.hidden_states[1]
            else:
                gen_hidden = generated.hidden_states[0]
            for layer_idx, hidden in enumerate(gen_hidden):
                embs = hidden[:, -1, :].squeeze(0).float()
                all_embs[str(layer_idx)] = embs
        return all_embs

    def _get_image_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        img_tok_id = getattr(self.model.config, "image_token_index", None)
        if img_tok_id is None:
            img_tok_id = getattr(self.model.config, "image_token_id", None)
        if img_tok_id is None:
            img_tok_id = self.processor.tokenizer.convert_tokens_to_ids("<image>")
        if img_tok_id is None or img_tok_id < 0:
            return torch.zeros(
                input_ids.shape[-1], dtype=torch.bool, device=input_ids.device
            )
        return _get_image_token_mask(input_ids, img_tok_id)

    def get_mean_all_tokens_embeddings(
        self, prompt: GemmaPrompt
    ) -> dict[str, torch.Tensor]:
        inputs, _ = self._prepare_inputs(prompt)
        with torch.inference_mode():
            outputs = self.model(**inputs, output_hidden_states=True)
            all_embs = {}
            for layer_idx, hidden in enumerate(outputs.hidden_states):
                all_embs[str(layer_idx)] = hidden.squeeze(0).mean(dim=0).float()
        return all_embs

    def get_mean_image_tokens_embeddings(
        self, prompt: GemmaPrompt
    ) -> dict[str, torch.Tensor]:
        inputs, _ = self._prepare_inputs(prompt)
        img_mask = self._get_image_mask(inputs["input_ids"])
        with torch.inference_mode():
            outputs = self.model(**inputs, output_hidden_states=True)
            all_embs = {}
            for layer_idx, hidden in enumerate(outputs.hidden_states):
                all_embs[str(layer_idx)] = _safe_mean_pool(hidden, img_mask)
        return all_embs

    def get_mean_text_tokens_embeddings(
        self, prompt: GemmaPrompt
    ) -> dict[str, torch.Tensor]:
        inputs, _ = self._prepare_inputs(prompt)
        text_mask = ~self._get_image_mask(inputs["input_ids"])
        with torch.inference_mode():
            outputs = self.model(**inputs, output_hidden_states=True)
            all_embs = {}
            for layer_idx, hidden in enumerate(outputs.hidden_states):
                all_embs[str(layer_idx)] = _safe_mean_pool(hidden, text_mask)
        return all_embs

    def get_concat_img_text_last_embeddings(
        self, prompt: GemmaPrompt
    ) -> dict[str, torch.Tensor]:
        inputs, _ = self._prepare_inputs(prompt)
        img_mask = self._get_image_mask(inputs["input_ids"])
        text_mask = ~img_mask
        with torch.inference_mode():
            outputs = self.model(**inputs, output_hidden_states=True)
            all_embs = {}
            for layer_idx, hidden in enumerate(outputs.hidden_states):
                img_mean = _safe_mean_pool(hidden, img_mask)
                txt_mean = _safe_mean_pool(hidden, text_mask)
                last_tok = hidden[:, -1, :].squeeze(0).float()
                all_embs[str(layer_idx)] = torch.cat(
                    [img_mean, txt_mean, last_tok], dim=0
                )
        return all_embs

    def get_all_layer_embeddings_batch(
        self, prompts: list[GemmaPrompt]
    ) -> dict[str, torch.Tensor]:
        all_layer_embs: dict[str, list[torch.Tensor]] = {}
        for p in prompts:
            single_embs = self.get_all_layer_embeddings(p)
            for layer_key, emb in single_embs.items():
                if layer_key not in all_layer_embs:
                    all_layer_embs[layer_key] = []
                all_layer_embs[layer_key].append(emb)
        return {k: torch.stack(v, dim=0) for k, v in all_layer_embs.items()}

class RandomGemmaPrompter(GemmaPrompter):
    def __init__(
        self,
        model_stem: str,
        model_id: str = "google/gemma-3-4b-it",
        max_new_tokens: int = 200,
        temperature: float = 0.0,
        device: str | None = None,
        dtype: torch.dtype = torch.bfloat16,
        cache_root: str = "/raid/rsq813/random_init",
        force_reinit: int = 0,
        seed: int | None = None,
    ):
        BasePrompter.__init__(self, model_stem, model_id, max_new_tokens, temperature)
        from transformers import AutoConfig, AutoProcessor, AutoModelForImageTextToText

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype_tag = (
            "bf16" if dtype == torch.bfloat16 else str(dtype).replace("torch.", "")
        )
        seed_tag = f"_seed{seed}" if seed is not None else ""
        save_dir = (
            Path(cache_root)
            / _sanitize_repo_id(model_id)
            / f"dtype_{dtype_tag}{seed_tag}"
        )
        save_dir.mkdir(parents=True, exist_ok=True)
        processor_src = (
            str(save_dir)
            if (save_dir / "preprocessor_config.json").exists()
            else model_id
        )
        self.processor = AutoProcessor.from_pretrained(processor_src)
        if _model_weight_files_exist(save_dir) and not force_reinit:
            print(f"[cache] Loading cached random Gemma model from: {save_dir}")
            t0 = time.time()
            try:
                self.model = (
                    AutoModelForImageTextToText.from_pretrained(
                        str(save_dir),
                        dtype=dtype,
                        device_map=None,
                    )
                    .to(device)
                    .eval()
                )
                print(f"[cache] Loaded in {time.time() - t0:.1f}s on {device}")
                return
            except Exception as e:
                print(
                    f"[cache] Cached Gemma checkpoint is incomplete or unreadable: {e}"
                )
                print(f"[cache] Rebuilding random Gemma model in: {save_dir}")
                _clear_directory_contents(save_dir)
        print(f"[build] Building random Gemma model for {model_stem} on {device}")
        config = AutoConfig.from_pretrained(model_id)
        if seed is not None:
            print(f"[build] Setting random seed to {seed}")
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        old_default = torch.get_default_dtype()
        torch.set_default_dtype(dtype)
        try:
            t0 = time.time()
            model = AutoModelForImageTextToText.from_config(config)
            model = model.to(device).eval()
            print(f"[build] Constructed + moved in {time.time() - t0:.1f}s")
        finally:
            torch.set_default_dtype(old_default)
        self.model = model
        print(f"[save] Saving random Gemma model to: {save_dir}")
        t0 = time.time()
        _clear_directory_contents(save_dir)
        cpu_model = self.model.to("cpu")
        cpu_model.save_pretrained(str(save_dir), safe_serialization=True)
        self.processor.save_pretrained(str(save_dir))
        print(f"[save] Done in {time.time() - t0:.1f}s")
        self.model = (
            cpu_model.to(device).eval() if device != "cpu" else cpu_model.eval()
        )

class RandomGemmaPrompter1(RandomGemmaPrompter):
    def __init__(
        self,
        model_stem: str = "random_gemma_4b_1",
        model_id: str = "google/gemma-3-4b-it",
        max_new_tokens: int = 200,
        temperature: float = 0.0,
        device: str | None = None,
        dtype: torch.dtype = torch.bfloat16,
        cache_root: str = "/raid/rsq813/random_init",
        force_reinit: int = 0,
    ):
        super().__init__(
            model_stem=model_stem,
            model_id=model_id,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            device=device,
            dtype=dtype,
            cache_root=cache_root,
            force_reinit=force_reinit,
            seed=1,
        )

class RandomGemmaPrompter2(RandomGemmaPrompter):
    def __init__(
        self,
        model_stem: str = "random_gemma_4b_2",
        model_id: str = "google/gemma-3-4b-it",
        max_new_tokens: int = 200,
        temperature: float = 0.0,
        device: str | None = None,
        dtype: torch.dtype = torch.bfloat16,
        cache_root: str = "/raid/rsq813/random_init",
        force_reinit: int = 0,
    ):
        super().__init__(
            model_stem=model_stem,
            model_id=model_id,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            device=device,
            dtype=dtype,
            cache_root=cache_root,
            force_reinit=force_reinit,
            seed=2,
        )

class RandomGemmaPrompter3(RandomGemmaPrompter):
    def __init__(
        self,
        model_stem: str = "random_gemma_4b_3",
        model_id: str = "google/gemma-3-4b-it",
        max_new_tokens: int = 200,
        temperature: float = 0.0,
        device: str | None = None,
        dtype: torch.dtype = torch.bfloat16,
        cache_root: str = "/raid/rsq813/random_init",
        force_reinit: int = 0,
    ):
        super().__init__(
            model_stem=model_stem,
            model_id=model_id,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            device=device,
            dtype=dtype,
            cache_root=cache_root,
            force_reinit=force_reinit,
            seed=3,
        )

class BioGemmaLoraPrompter(BasePrompter):
    def __init__(
        self,
        adapter_path: str = "./PEFT/checkpoints/gemma-3-4b-it-lora",
        base_model_id: str = "google/gemma-3-4b-it",
        max_new_tokens: int = 200,
        temperature: float = 0.0,
    ):
        super().__init__("biogemma-lora", adapter_path, max_new_tokens, temperature)
        from transformers import AutoProcessor, AutoModelForImageTextToText
        from peft import PeftModel

        adapter_path = os.path.abspath(adapter_path)
        self.processor = AutoProcessor.from_pretrained(
            adapter_path, trust_remote_code=True
        )
        base_model = AutoModelForImageTextToText.from_pretrained(
            base_model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        ).eval()
        self.model = PeftModel.from_pretrained(base_model, adapter_path).eval()

    def _prepare_inputs(self, prompt: BioGemmaLoraPrompt):
        messages = prompt.get_messages()
        if hasattr(self.processor, "chat_template") and self.processor.chat_template:
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            warnings.warn(
                f"Processor for {self.model_id} has no chat_template. "
                "Falling back to plain-text formatting.",
                UserWarning,
            )
            user_content = ""
            if prompt.system_prompt:
                user_content += prompt.system_prompt + "\n\n"
            user_content += prompt.user_text
            text = user_content
        kw = dict(text=text, return_tensors="pt")
        if prompt.image is not None:
            kw["images"] = prompt.image
        inputs = self.processor(**kw)
        return inputs.to(self.model.device), text

    def get_completion(self, prompt: BioGemmaLoraPrompt) -> str:
        inputs, text = self._prepare_inputs(prompt)
        with torch.inference_mode():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=self.temperature > 0,
            )
        prompt_len = inputs["input_ids"].shape[1]
        gen_ids = generated_ids[0, prompt_len:]
        out = self.processor.decode(gen_ids, skip_special_tokens=True).strip()
        return self._strip_assistant_header(out)

    def get_completion_conversation(
        self,
        system_prompt: str,
        few_shot_examples: list[dict],
        query_prompt: BioGemmaLoraPrompt,
    ) -> str:
        messages = []
        if system_prompt:
            messages.append(
                {"role": "system", "content": [{"type": "text", "text": system_prompt}]}
            )
        for ex in few_shot_examples:
            messages.append({"role": "user", "content": ex["user_text"]})
            messages.append({"role": "assistant", "content": ex["assistant_text"]})
        content = [{"type": "text", "text": query_prompt.user_text}]
        if query_prompt.image_path:
            content.append({"type": "image", "image": query_prompt.image_path})
        messages.append({"role": "user", "content": content})
        if hasattr(self.processor, "chat_template") and self.processor.chat_template:
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            text = query_text
        kw = dict(text=text, return_tensors="pt")
        if query_prompt.image is not None:
            kw["images"] = query_prompt.image
        inputs = self.processor(**kw).to(self.model.device)
        with torch.inference_mode():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=self.temperature > 0,
            )
        prompt_len = inputs["input_ids"].shape[1]
        gen_ids = generated_ids[0, prompt_len:]
        out = self.processor.decode(gen_ids, skip_special_tokens=True).strip()
        return self._strip_assistant_header(out)

    def get_completion_batch(self, prompts: list[BioGemmaLoraPrompt]) -> list[str]:
        return [self.get_completion(p) for p in prompts]

    def get_all_layer_embeddings(
        self, prompt: BioGemmaLoraPrompt
    ) -> dict[str, torch.Tensor]:
        inputs, _ = self._prepare_inputs(prompt)
        with torch.inference_mode():
            outputs = self.model(**inputs, output_hidden_states=True)
            all_embs = {}
            for layer_idx, hidden in enumerate(outputs.hidden_states):
                embs = hidden[:, -1, :].squeeze(0).float()
                all_embs[str(layer_idx)] = embs
        return all_embs

    def get_first_generated_token_embeddings(
        self, prompt: BioGemmaLoraPrompt
    ) -> dict[str, torch.Tensor]:
        inputs, _ = self._prepare_inputs(prompt)
        with torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=2,
                min_new_tokens=2,
                do_sample=False,
                output_hidden_states=True,
                return_dict_in_generate=True,
            )
            all_embs = {}
            if len(generated.hidden_states) >= 2:
                gen_hidden = generated.hidden_states[1]
            else:
                gen_hidden = generated.hidden_states[0]
            for layer_idx, hidden in enumerate(gen_hidden):
                embs = hidden[:, -1, :].squeeze(0).float()
                all_embs[str(layer_idx)] = embs
        return all_embs

    def _get_image_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        img_tok_id = getattr(self.model.config, "image_token_index", None)
        if img_tok_id is None:
            img_tok_id = getattr(self.model.config, "image_token_id", None)
        if img_tok_id is None:
            img_tok_id = self.processor.tokenizer.convert_tokens_to_ids("<image>")
        if img_tok_id is None or img_tok_id < 0:
            return torch.zeros(
                input_ids.shape[-1], dtype=torch.bool, device=input_ids.device
            )
        return _get_image_token_mask(input_ids, img_tok_id)

    def get_mean_all_tokens_embeddings(
        self, prompt: BioGemmaLoraPrompt
    ) -> dict[str, torch.Tensor]:
        inputs, _ = self._prepare_inputs(prompt)
        with torch.inference_mode():
            outputs = self.model(**inputs, output_hidden_states=True)
            all_embs = {}
            for layer_idx, hidden in enumerate(outputs.hidden_states):
                all_embs[str(layer_idx)] = hidden.squeeze(0).mean(dim=0).float()
        return all_embs

    def get_mean_image_tokens_embeddings(
        self, prompt: BioGemmaLoraPrompt
    ) -> dict[str, torch.Tensor]:
        inputs, _ = self._prepare_inputs(prompt)
        img_mask = self._get_image_mask(inputs["input_ids"])
        with torch.inference_mode():
            outputs = self.model(**inputs, output_hidden_states=True)
            all_embs = {}
            for layer_idx, hidden in enumerate(outputs.hidden_states):
                all_embs[str(layer_idx)] = _safe_mean_pool(hidden, img_mask)
        return all_embs

    def get_mean_text_tokens_embeddings(
        self, prompt: BioGemmaLoraPrompt
    ) -> dict[str, torch.Tensor]:
        inputs, _ = self._prepare_inputs(prompt)
        text_mask = ~self._get_image_mask(inputs["input_ids"])
        with torch.inference_mode():
            outputs = self.model(**inputs, output_hidden_states=True)
            all_embs = {}
            for layer_idx, hidden in enumerate(outputs.hidden_states):
                all_embs[str(layer_idx)] = _safe_mean_pool(hidden, text_mask)
        return all_embs

    def get_concat_img_text_last_embeddings(
        self, prompt: BioGemmaLoraPrompt
    ) -> dict[str, torch.Tensor]:
        inputs, _ = self._prepare_inputs(prompt)
        img_mask = self._get_image_mask(inputs["input_ids"])
        text_mask = ~img_mask
        with torch.inference_mode():
            outputs = self.model(**inputs, output_hidden_states=True)
            all_embs = {}
            for layer_idx, hidden in enumerate(outputs.hidden_states):
                img_mean = _safe_mean_pool(hidden, img_mask)
                txt_mean = _safe_mean_pool(hidden, text_mask)
                last_tok = hidden[:, -1, :].squeeze(0).float()
                all_embs[str(layer_idx)] = torch.cat(
                    [img_mean, txt_mean, last_tok], dim=0
                )
        return all_embs

    def get_all_layer_embeddings_batch(
        self, prompts: list[BioGemmaLoraPrompt]
    ) -> dict[str, torch.Tensor]:
        all_layer_embs: dict[str, list[torch.Tensor]] = {}
        for p in prompts:
            single_embs = self.get_all_layer_embeddings(p)
            for layer_key, emb in single_embs.items():
                if layer_key not in all_layer_embs:
                    all_layer_embs[layer_key] = []
                all_layer_embs[layer_key].append(emb)
        return {k: torch.stack(v, dim=0) for k, v in all_layer_embs.items()}
PROMPTER_MAP = {
    "med-flamingo-9b": lambda **kw: FlamingoPrompter(
        model_id=MODEL_REGISTRY["med-flamingo-9b"],
        model_stem="med-flamingo-9b",
        **kw,
    ),
    "random_med-flamingo-9b_1": lambda **kw: RandomFlamingoPrompter1(
        model_id=MODEL_REGISTRY["random_med-flamingo-9b_1"],
        model_stem="random_med-flamingo-9b_1",
        **kw,
    ),
    "random_med-flamingo-9b_2": lambda **kw: RandomFlamingoPrompter2(
        model_id=MODEL_REGISTRY["random_med-flamingo-9b_2"],
        model_stem="random_med-flamingo-9b_2",
        **kw,
    ),
    "random_med-flamingo-9b_3": lambda **kw: RandomFlamingoPrompter3(
        model_id=MODEL_REGISTRY["random_med-flamingo-9b_3"],
        model_stem="random_med-flamingo-9b_3",
        **kw,
    ),
    "open-flamingo-9b": lambda **kw: FlamingoPrompter(
        model_id=MODEL_REGISTRY["open-flamingo-9b"],
        model_stem="open-flamingo-9b",
        **kw,
    ),
    "random_open-flamingo-9b_1": lambda **kw: RandomFlamingoPrompter1(
        model_id=MODEL_REGISTRY["random_open-flamingo-9b_1"],
        model_stem="random_open-flamingo-9b_1",
        **kw,
    ),
    "random_open-flamingo-9b_2": lambda **kw: RandomFlamingoPrompter2(
        model_id=MODEL_REGISTRY["random_open-flamingo-9b_2"],
        model_stem="random_open-flamingo-9b_2",
        **kw,
    ),
    "random_open-flamingo-9b_3": lambda **kw: RandomFlamingoPrompter3(
        model_id=MODEL_REGISTRY["random_open-flamingo-9b_3"],
        model_stem="random_open-flamingo-9b_3",
        **kw,
    ),
    "llava-med-7b": lambda **kw: LlavaMedPrompter(
        model_id=MODEL_REGISTRY["llava-med-7b"],
        model_stem="llava-med-7b",
        **kw,
    ),
    "random_llava-med-7b_1": lambda **kw: RandomLlavaPrompter1(
        model_id=MODEL_REGISTRY["random_llava-med-7b_1"],
        model_stem="random_llava-med-7b_1",
        **kw,
    ),
    "random_llava-med-7b_2": lambda **kw: RandomLlavaPrompter2(
        model_id=MODEL_REGISTRY["random_llava-med-7b_2"],
        model_stem="random_llava-med-7b_2",
        **kw,
    ),
    "random_llava-med-7b_3": lambda **kw: RandomLlavaPrompter3(
        model_id=MODEL_REGISTRY["random_llava-med-7b_3"],
        model_stem="random_llava-med-7b_3",
        **kw,
    ),
    "llava-v0-7b": lambda **kw: LlavaMedPrompter(
        model_id=MODEL_REGISTRY["llava-v0-7b"],
        model_stem="llava-v0-7b",
        **kw,
    ),
    "random_llava-v0-7b_1": lambda **kw: RandomLlavaPrompter1(
        model_id=MODEL_REGISTRY["random_llava-v0-7b_1"],
        model_stem="random_llava-v0-7b_1",
        **kw,
    ),
    "random_llava-v0-7b_2": lambda **kw: RandomLlavaPrompter2(
        model_id=MODEL_REGISTRY["random_llava-v0-7b_2"],
        model_stem="random_llava-v0-7b_2",
        **kw,
    ),
    "random_llava-v0-7b_3": lambda **kw: RandomLlavaPrompter3(
        model_id=MODEL_REGISTRY["random_llava-v0-7b_3"],
        model_stem="random_llava-v0-7b_3",
        **kw,
    ),
    "medgemma": MedGemmaPrompter,
    "random_medgemma_4b_1": lambda **kw: RandomMedGemmaPrompter1(
        model_stem="random_medgemma_4b_1",
        model_id=MODEL_REGISTRY["random_medgemma_4b_1"],
        **kw,
    ),
    "random_medgemma_4b_2": lambda **kw: RandomMedGemmaPrompter2(
        model_stem="random_medgemma_4b_2",
        model_id=MODEL_REGISTRY["random_medgemma_4b_2"],
        **kw,
    ),
    "random_medgemma_4b_3": lambda **kw: RandomMedGemmaPrompter3(
        model_stem="random_medgemma_4b_3",
        model_id=MODEL_REGISTRY["random_medgemma_4b_3"],
        **kw,
    ),
    "medvlthinker-3b": lambda **kw: MedVLThinkerPrompter(
        model_stem="medvlthinker-3b",
        model_id=MODEL_REGISTRY["medvlthinker-3b"],
        **kw,
    ),
    "medvlthinker-7b": lambda **kw: MedVLThinkerPrompter(
        model_stem="medvlthinker-7b",
        model_id=MODEL_REGISTRY["medvlthinker-7b"],
        **kw,
    ),
    "medvlthinker-32b": lambda **kw: MedVLThinkerPrompter(
        model_stem="medvlthinker-32b",
        model_id=MODEL_REGISTRY["medvlthinker-32b"],
        **kw,
    ),
    "medmo-4b-next": lambda **kw: Qwen25VLPrompter(
        model_stem="medmo-4b-next",
        model_id=MODEL_REGISTRY["medmo-4b-next"],
        **kw,
    ),
    "medmo-8b-next": lambda **kw: Qwen25VLPrompter(
        model_stem="medmo-8b-next",
        model_id=MODEL_REGISTRY["medmo-8b-next"],
        **kw,
    ),
    "medix-r1-2b": lambda **kw: MedIXPrompter(
        model_stem="medix-r1-2b",
        model_id=MODEL_REGISTRY["medix-r1-2b"],
        **kw,
    ),
    "medix-r1-30b": lambda **kw: MedIXPrompter(
        model_stem="medix-r1-30b",
        model_id=MODEL_REGISTRY["medix-r1-30b"],
        **kw,
    ),
    "qwen25-vl-3b-instruct": lambda **kw: Qwen25VLPrompter(
        model_stem="qwen25-vl-3b-instruct",
        model_id=MODEL_REGISTRY["qwen25-vl-3b-instruct"],
        **kw,
    ),
    "random_Qwen2.5-VL-3B_1": lambda **kw: RandomQwen25VLPrompter1(
        model_stem="random_Qwen2.5-VL-3B_1",
        model_id=MODEL_REGISTRY["random_Qwen2.5-VL-3B_1"],
        **kw,
    ),
    "random_Qwen2.5-VL-3B_2": lambda **kw: RandomQwen25VLPrompter2(
        model_stem="random_Qwen2.5-VL-3B_2",
        model_id=MODEL_REGISTRY["random_Qwen2.5-VL-3B_2"],
        **kw,
    ),
    "random_Qwen2.5-VL-3B_3": lambda **kw: RandomQwen25VLPrompter3(
        model_stem="random_Qwen2.5-VL-3B_3",
        model_id=MODEL_REGISTRY["random_Qwen2.5-VL-3B_3"],
        **kw,
    ),
    "qwen25-vl-7b-instruct": lambda **kw: Qwen25VLPrompter(
        model_stem="qwen25-vl-7b-instruct",
        model_id=MODEL_REGISTRY["qwen25-vl-7b-instruct"],
        **kw,
    ),
    "qwen25-vl-7b-instruct-full-path-vqa": lambda **kw: Qwen25VLPrompter(
        model_stem="qwen25-vl-7b-instruct-full-path-vqa",
        model_id=MODEL_REGISTRY["qwen25-vl-7b-instruct-full-path-vqa"],
        **kw,
    ),
    "qwen25-vl-7b-instruct-full-all-med-vqa": lambda **kw: Qwen25VLPrompter(
        model_stem="qwen25-vl-7b-instruct-full-all-med-vqa",
        model_id=MODEL_REGISTRY["qwen25-vl-7b-instruct-full-all-med-vqa"],
        **kw,
    ),
    "qwen25-vl-7b-instruct-full-slake": lambda **kw: Qwen25VLPrompter(
        model_stem="qwen25-vl-7b-instruct-full-slake",
        model_id=MODEL_REGISTRY["qwen25-vl-7b-instruct-full-slake"],
        **kw,
    ),
    "qwen25-vl-7b-instruct-full-vqa-rad": lambda **kw: Qwen25VLPrompter(
        model_stem="qwen25-vl-7b-instruct-full-vqa-rad",
        model_id=MODEL_REGISTRY["qwen25-vl-7b-instruct-full-vqa-rad"],
        **kw,
    ),
    "qwen25-vl-32b-instruct": lambda **kw: Qwen25VLPrompter(
        model_stem="qwen25-vl-32b-instruct",
        model_id=MODEL_REGISTRY["qwen25-vl-32b-instruct"],
        **kw,
    ),
    "qwen2-vl-2b-instruct": lambda **kw: Qwen2VLPrompter(
        model_stem="qwen2-vl-2b-instruct",
        model_id=MODEL_REGISTRY["qwen2-vl-2b-instruct"],
        **kw,
    ),
    "qwen3-vl-2b-instruct": lambda **kw: Qwen25VLPrompter(
        model_stem="qwen3-vl-2b-instruct",
        model_id=MODEL_REGISTRY["qwen3-vl-2b-instruct"],
        **kw,
    ),
    "qwen3-vl-4b-instruct": lambda **kw: Qwen25VLPrompter(
        model_stem="qwen3-vl-4b-instruct",
        model_id=MODEL_REGISTRY["qwen3-vl-4b-instruct"],
        **kw,
    ),
    "qwen3-vl-8b-instruct": lambda **kw: Qwen25VLPrompter(
        model_stem="qwen3-vl-8b-instruct",
        model_id=MODEL_REGISTRY["qwen3-vl-8b-instruct"],
        **kw,
    ),
    "random_Qwen3-VL-8B_1": lambda **kw: RandomQwen3VLPrompter1(
        model_stem="random_Qwen3-VL-8B_1",
        model_id=MODEL_REGISTRY["random_Qwen3-VL-8B_1"],
        **kw,
    ),
    "random_Qwen3-VL-8B_2": lambda **kw: RandomQwen3VLPrompter2(
        model_stem="random_Qwen3-VL-8B_2",
        model_id=MODEL_REGISTRY["random_Qwen3-VL-8B_2"],
        **kw,
    ),
    "random_Qwen3-VL-8B_3": lambda **kw: RandomQwen3VLPrompter3(
        model_stem="random_Qwen3-VL-8B_3",
        model_id=MODEL_REGISTRY["random_Qwen3-VL-8B_3"],
        **kw,
    ),
    "qwen3-vl-30b-a3b-instruct": lambda **kw: Qwen25VLPrompter(
        model_stem="qwen3-vl-30b-a3b-instruct",
        model_id=MODEL_REGISTRY["qwen3-vl-30b-a3b-instruct"],
        **kw,
    ),
    "adapt-qwen2-2b": lambda **kw: AdaptQwen2VLPrompter(
        model_stem="adapt-qwen2-2b",
        model_id=MODEL_REGISTRY["adapt-qwen2-2b"],
        **kw,
    ),
    "adapt-internVL3-1b": lambda **kw: AdaptInternVL3Prompter(
        model_stem="adapt-internVL3-1b",
        model_id=MODEL_REGISTRY["adapt-internVL3-1b"],
        **kw,
    ),
    "internvl3-1b": lambda **kw: InternVL3Prompter(
        model_stem="internvl3-1b",
        model_id=MODEL_REGISTRY["internvl3-1b"],
        **kw,
    ),
    "random_InternVL3-1B_1": lambda **kw: RandomInternVL3Prompter1(
        model_stem="random_InternVL3-1B_1",
        model_id=MODEL_REGISTRY["random_InternVL3-1B_1"],
        **kw,
    ),
    "random_InternVL3-1B_2": lambda **kw: RandomInternVL3Prompter2(
        model_stem="random_InternVL3-1B_2",
        model_id=MODEL_REGISTRY["random_InternVL3-1B_2"],
        **kw,
    ),
    "random_InternVL3-1B_3": lambda **kw: RandomInternVL3Prompter3(
        model_stem="random_InternVL3-1B_3",
        model_id=MODEL_REGISTRY["random_InternVL3-1B_3"],
        **kw,
    ),
    "biomedllama": BioMedLlamaPrompter,
    "adapt-llama3.2-11b": AdaptLlamaPrompter,
    "meta-llama3.2-11b-vision-instruct": lambda **kw: MetaLlama32VisionPrompter(
        model_stem="meta-llama3.2-11b-vision-instruct",
        model_id=MODEL_REGISTRY["meta-llama3.2-11b-vision-instruct"],
        **kw,
    ),
    "meta-llama3.2-11b-vision-instruct-full-path-vqa": lambda **kw: MetaLlama32VisionPrompter(
        model_stem="meta-llama3.2-11b-vision-instruct-full-path-vqa",
        model_id=MODEL_REGISTRY["meta-llama3.2-11b-vision-instruct-full-path-vqa"],
        processor_src=MODEL_REGISTRY["meta-llama3.2-11b-vision-instruct"],
        **kw,
    ),
    "meta-llama3.2-11b-vision-instruct-full-all-med-vqa": lambda **kw: MetaLlama32VisionPrompter(
        model_stem="meta-llama3.2-11b-vision-instruct-full-all-med-vqa",
        model_id=MODEL_REGISTRY["meta-llama3.2-11b-vision-instruct-full-all-med-vqa"],
        processor_src=MODEL_REGISTRY["meta-llama3.2-11b-vision-instruct"],
        **kw,
    ),
    "meta-llama3.2-11b-vision-instruct-full-slake": lambda **kw: MetaLlama32VisionPrompter(
        model_stem="meta-llama3.2-11b-vision-instruct-full-slake",
        model_id=MODEL_REGISTRY["meta-llama3.2-11b-vision-instruct-full-slake"],
        processor_src=MODEL_REGISTRY["meta-llama3.2-11b-vision-instruct"],
        **kw,
    ),
    "meta-llama3.2-11b-vision-instruct-full-vqa-rad": lambda **kw: MetaLlama32VisionPrompter(
        model_stem="meta-llama3.2-11b-vision-instruct-full-vqa-rad",
        model_id=MODEL_REGISTRY["meta-llama3.2-11b-vision-instruct-full-vqa-rad"],
        processor_src=MODEL_REGISTRY["meta-llama3.2-11b-vision-instruct"],
        **kw,
    ),
    "gemma": GemmaPrompter,
    "random_gemma_4b_1": RandomGemmaPrompter1,
    "random_gemma_4b_2": RandomGemmaPrompter2,
    "random_gemma_4b_3": RandomGemmaPrompter3,
    "biogemma-lora": BioGemmaLoraPrompter,
    "medgemma-27b": lambda **kw: MedGemmaPrompter(
        model_id="google/medgemma-27b-it", **kw
    ),
    "random_medgemma_27b_1": lambda **kw: RandomMedGemmaPrompter1(
        model_stem="random_medgemma_27b_1",
        model_id=MODEL_REGISTRY["random_medgemma_27b_1"],
        **kw,
    ),
    "random_medgemma_27b_2": lambda **kw: RandomMedGemmaPrompter2(
        model_stem="random_medgemma_27b_2",
        model_id=MODEL_REGISTRY["random_medgemma_27b_2"],
        **kw,
    ),
    "random_medgemma_27b_3": lambda **kw: RandomMedGemmaPrompter3(
        model_stem="random_medgemma_27b_3",
        model_id=MODEL_REGISTRY["random_medgemma_27b_3"],
        **kw,
    ),
    "gemma-27b": lambda **kw: GemmaPrompter(model_id="google/gemma-3-27b-it", **kw),
    "random_gemma_27b_1": lambda **kw: RandomGemmaPrompter1(
        model_stem="random_gemma_27b_1",
        model_id=MODEL_REGISTRY["random_gemma_27b_1"],
        **kw,
    ),
    "random_gemma_27b_2": lambda **kw: RandomGemmaPrompter2(
        model_stem="random_gemma_27b_2",
        model_id=MODEL_REGISTRY["random_gemma_27b_2"],
        **kw,
    ),
    "random_gemma_27b_3": lambda **kw: RandomGemmaPrompter3(
        model_stem="random_gemma_27b_3",
        model_id=MODEL_REGISTRY["random_gemma_27b_3"],
        **kw,
    ),
}

def get_prompter(model_stem: str, **kwargs) -> BasePrompter:
    if model_stem not in PROMPTER_MAP:
        raise ValueError(
            f"Unknown model stem: '{model_stem}'. "
            f"Available models: {list(PROMPTER_MAP.keys())}"
        )
    return PROMPTER_MAP[model_stem](**kwargs)
