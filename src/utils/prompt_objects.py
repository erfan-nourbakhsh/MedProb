from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

@dataclass
class BasePrompt(ABC):
    user_text: str
    system_prompt: Optional[str] = None
    assistant_text: Optional[str] = None
    @abstractmethod
    def get_messages(self) -> list[dict]:
        raise NotImplementedError

@dataclass
class TextOnlyPrompt(BasePrompt):
    def get_messages(self) -> list[dict]:
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": self.user_text})
        if self.assistant_text:
            messages.append({"role": "assistant", "content": self.assistant_text})
        return messages

@dataclass
class OldLlavaConversation:
    system: str = ""
    roles: tuple[str, str] = ("Human", "Assistant")
    sep: str = "\n### "
    messages: list[tuple[str, Optional[str]]] = field(default_factory=list)
    def copy(self) -> "OldLlavaConversation":
        return OldLlavaConversation(
            system=self.system,
            roles=self.roles,
            sep=self.sep,
            messages=list(self.messages),
        )

    def append_message(self, role: str, message: Optional[str]) -> None:
        self.messages.append((role, message))

    def get_prompt(self) -> str:
        prompt = self.system.rstrip()
        for idx, (role, message) in enumerate(self.messages):
            if prompt:
                prompt += self.sep
            elif idx == 0 and self.sep.startswith("\n"):
                prompt += self.sep[1:]
            else:
                prompt += self.sep
            prompt += f"{role}:"
            if message:
                prompt += f" {message}"
        return prompt

@dataclass
class VisionPrompt(BasePrompt):
    image_path: Optional[str] = None
    image: Optional[Image.Image] = field(init=False, default=None)
    def __post_init__(self):
        if self.image_path:
            self.image = Image.open(self.image_path).convert("RGB")

    def get_messages(self) -> list[dict]:
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        content = []
        if self.image_path:
            content.append({"type": "image"})
        content.append({"type": "text", "text": self.user_text})
        messages.append({"role": "user", "content": content})
        if self.assistant_text:
            messages.append({"role": "assistant", "content": self.assistant_text})
        return messages

@dataclass
class LlavaPrompt(VisionPrompt):
    conv_mode: str = field(init=False, default="llava_v0")
    def _build_image_token_text(
        self,
        mm_use_im_start_end: bool,
        image_token_len: int,
    ) -> str:
        if mm_use_im_start_end:
            return "<im_start>" + ("<im_patch>" * image_token_len) + "<im_end>"
        return "<im_patch>" * image_token_len

    def _attach_image_tokens(
        self,
        question_text: str,
        has_image: bool,
        mm_use_im_start_end: bool,
        image_token_len: int,
    ) -> str:
        if not has_image:
            return question_text
        return f"{question_text}\n{self._build_image_token_text(mm_use_im_start_end, image_token_len)}"

    def _get_conversation_template(self) -> OldLlavaConversation:
        conv = OldLlavaConversation()
        if "v0" in self.conv_mode:
            conv.sep = "\n### "
        return conv

    def render_prompt_text(
        self,
        few_shot_examples: Optional[list[dict]] = None,
        mm_use_im_start_end: bool = True,
        image_token_len: int = 256,
    ) -> str:
        conv = self._get_conversation_template().copy()
        conv.system = f"{self.system_prompt.strip()}\n" if self.system_prompt else ""
        for example in few_shot_examples or []:
            example_question = self._attach_image_tokens(
                example["user_text"],
                bool(example.get("has_image", False)),
                mm_use_im_start_end,
                image_token_len,
            )
            conv.append_message(conv.roles[0], example_question)
            conv.append_message(conv.roles[1], example["assistant_text"])
        query_question = self._attach_image_tokens(
            self.user_text,
            self.image is not None,
            mm_use_im_start_end,
            image_token_len,
        )
        conv.append_message(conv.roles[0], query_question)
        conv.append_message(conv.roles[1], self.assistant_text or "")
        return conv.get_prompt()

    def get_messages(self) -> list[dict]:
        return [{"role": "user", "content": self.render_prompt_text()}]

@dataclass
class LlavaMedPrompt(LlavaPrompt):
    pass

@dataclass
class LlavaV0Prompt(LlavaPrompt):
    pass

@dataclass
class FlamingoPrompt(VisionPrompt):
    answer_prefix: str = "Answer:"
    def _render_single_turn(
        self,
        user_text: str,
        answer_text: Optional[str],
        image_count: int,
        add_endofchunk: bool,
    ) -> str:
        prefix = "<image>" * max(image_count, 0)
        rendered = f"{prefix} {user_text}".strip()
        rendered += f"\n{self.answer_prefix}"
        if answer_text:
            rendered += f" {answer_text}"
        if add_endofchunk:
            rendered += " <|endofchunk|>"
        return rendered

    def render_prompt_text(self, few_shot_examples: Optional[list[dict]] = None) -> str:
        parts: list[str] = []
        if self.system_prompt:
            parts.append(self.system_prompt.strip())
        for example in few_shot_examples or []:
            parts.append(
                self._render_single_turn(
                    user_text=example["user_text"],
                    answer_text=example["assistant_text"],
                    image_count=int(example.get("image_count", 0)),
                    add_endofchunk=True,
                )
            )
        parts.append(
            self._render_single_turn(
                user_text=self.user_text,
                answer_text=self.assistant_text,
                image_count=1 if self.image is not None else 0,
                add_endofchunk=False,
            )
        )
        return "\n".join(parts).strip()

    def get_messages(self) -> list[dict]:
        return [{"role": "user", "content": self.render_prompt_text()}]

@dataclass
class MedFlamingoPrompt(FlamingoPrompt):
    answer_prefix: str = "Answer:"

@dataclass
class OpenFlamingoPrompt(FlamingoPrompt):
    answer_prefix: str = "Answer:"

@dataclass
class MedGemmaPrompt(VisionPrompt):
    def get_messages(self) -> list[dict]:
        user_text = (
            f"{self.system_prompt}\n\n{self.user_text}"
            if self.system_prompt
            else self.user_text
        )
        content = [{"type": "text", "text": user_text}]
        if self.image_path:
            content.append({"type": "image", "image": self.image_path})
        messages = [{"role": "user", "content": content}]
        if self.assistant_text:
            messages.append({"role": "assistant", "content": self.assistant_text})
        return messages

@dataclass
class MedVLThinkerPrompt(VisionPrompt):
    def get_messages(self) -> list[dict]:
        content = []
        if self.image_path:
            content.append({"type": "image", "image": self.image_path})
        content.append({"type": "text", "text": self.user_text})
        messages: list[dict] = []
        if self.system_prompt:
            messages.append(
                {
                    "role": "system",
                    "content": [{"type": "text", "text": self.system_prompt}],
                }
            )
        messages.append({"role": "user", "content": content})
        if self.assistant_text:
            messages.append(
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": self.assistant_text}],
                }
            )
        return messages

@dataclass
class Qwen25VLPrompt(VisionPrompt):
    def get_messages(self) -> list[dict]:
        content = []
        if self.image_path:
            content.append({"type": "image", "image": self.image_path})
        content.append({"type": "text", "text": self.user_text})
        messages = [{"role": "user", "content": content}]
        if self.assistant_text:
            messages.append(
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": self.assistant_text}],
                }
            )
        return messages

@dataclass
class AdaptQwen2VLPrompt(VisionPrompt):
    def get_messages(self) -> list[dict]:
        content = []
        if self.image_path:
            content.append({"type": "image", "image": self.image_path})
        content.append({"type": "text", "text": self.user_text})
        messages = [{"role": "user", "content": content}]
        if self.assistant_text:
            messages.append(
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": self.assistant_text}],
                }
            )
        return messages

@dataclass
class Qwen2VLPrompt(VisionPrompt):
    def get_messages(self) -> list[dict]:
        content = []
        if self.image_path:
            content.append({"type": "image", "image": self.image_path})
        content.append({"type": "text", "text": self.user_text})
        messages = [{"role": "user", "content": content}]
        if self.assistant_text:
            messages.append(
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": self.assistant_text}],
                }
            )
        return messages

@dataclass
class AdaptInternVL3Prompt(VisionPrompt):
    def get_messages(self) -> list[dict]:
        content = []
        if self.image_path:
            content.append({"type": "image", "image": self.image_path})
        content.append({"type": "text", "text": self.user_text})
        messages = [{"role": "user", "content": content}]
        if self.assistant_text:
            messages.append(
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": self.assistant_text}],
                }
            )
        return messages

@dataclass
class InternVL3Prompt(VisionPrompt):
    def get_messages(self) -> list[dict]:
        content = []
        if self.image_path:
            content.append({"type": "image", "image": self.image_path})
        content.append({"type": "text", "text": self.user_text})
        messages = [{"role": "user", "content": content}]
        if self.assistant_text:
            messages.append(
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": self.assistant_text}],
                }
            )
        return messages

@dataclass
class MedMOPrompt(VisionPrompt):
    def get_messages(self) -> list[dict]:
        content = []
        if self.image_path:
            content.append({"type": "image", "image": self.image_path})
        content.append({"type": "text", "text": self.user_text})
        messages = [{"role": "user", "content": content}]
        if self.assistant_text:
            messages.append(
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": self.assistant_text}],
                }
            )
        return messages
MEDIX_PROMPT_PREFIX = """You are a Medical AI Assistant with advanced reasoning capabilities
Your task:
1. First output the image modality tag from this set:
   <X_RAY>, <MICROSCOPY>, <CLINICAL_PHOTOGRAPHY>, <CT_SCAN>, <GRAPHICS>,
   <ANGIOGRAPHY>, <PET_SCAN>, <ULTRASOUND>, <MRI_SCAN>, <FUNDUS_PHOTOGRAPHY>,
   <OCT_SCAN>, <ENDOSCOPY>, <MAMMOGRAPHY>, <FLUOROSCOPY>, <OTHER>, <SPECT>
   (Only output the tag, nothing else.)
2. Then output the thinking and medical reasoning process in <thinking>...</thinking>tags.
3. Finally, provide the correct answer inside <answer>...</answer> tags.
4. Do not include any extra information or text outside of these tags.
Question:
"""
OPTION_KEYS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

def build_question_text(question: str, options: Optional[List[str]] = None) -> str:
    question = question.strip()
    if not options:
        return question
    lines = [question, "", "Choices:"]
    for i, option in enumerate(options):
        lines.append(f"{OPTION_KEYS[i]}. {option}")
    return "\n".join(lines)

def build_medix_messages(
    question: str,
    options: Optional[List[str]] = None,
    image: Optional[str] = None,
) -> List[Dict[str, Any]]:
    question_text = build_question_text(question, options)
    content: List[Dict[str, Any]] = []
    if image:
        content.append({"type": "image", "image": image})
    content.append({"type": "text", "text": question_text})
    return [
        {
            "role": "user",
            "content": content,
        }
    ]

@dataclass
class MedIXPrompt(VisionPrompt):
    options: Optional[List[str]] = None
    def get_messages(self) -> list[dict]:
        messages: list[dict] = [
            {
                "role": "system",
                "content": [{"type": "text", "text": MEDIX_PROMPT_PREFIX}],
            }
        ]
        messages.extend(
            build_medix_messages(
                question=self.user_text,
                options=self.options,
                image=self.image_path,
            )
        )
        if self.assistant_text:
            messages.append(
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": self.assistant_text}],
                }
            )
        return messages

@dataclass
class BioMedLlamaPrompt(VisionPrompt):
    def get_messages(self) -> list[dict]:
        content = []
        if self.image:
            content.append(self.image)
        content.append(self.user_text)
        messages = [{"role": "user", "content": content}]
        if self.assistant_text:
            messages.append({"role": "assistant", "content": self.assistant_text})
        return messages

@dataclass
class AdaptLlamaPrompt(VisionPrompt):
    def get_messages(self) -> list[dict]:
        messages = []
        if self.system_prompt:
            messages.append(
                {
                    "role": "system",
                    "content": [{"type": "text", "text": self.system_prompt}],
                }
            )
        content = []
        if self.image_path:
            content.append({"type": "image"})
        content.append({"type": "text", "text": self.user_text})
        messages.append({"role": "user", "content": content})
        if self.assistant_text:
            messages.append(
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": self.assistant_text}],
                }
            )
        return messages

@dataclass
class MetaLlama32VisionPrompt(VisionPrompt):
    def get_messages(self) -> list[dict]:
        messages = []
        if self.system_prompt:
            messages.append(
                {
                    "role": "system",
                    "content": [{"type": "text", "text": self.system_prompt}],
                }
            )
        content = []
        if self.image_path:
            content.append({"type": "image"})
        content.append({"type": "text", "text": self.user_text})
        messages.append({"role": "user", "content": content})
        if self.assistant_text:
            messages.append(
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": self.assistant_text}],
                }
            )
        return messages

@dataclass
class GemmaPrompt(VisionPrompt):
    def get_messages(self) -> list[dict]:
        user_text = (
            f"{self.system_prompt}\n\n{self.user_text}"
            if self.system_prompt
            else self.user_text
        )
        content = [{"type": "text", "text": user_text}]
        if self.image_path:
            content.append({"type": "image", "image": self.image_path})
        messages = [{"role": "user", "content": content}]
        if self.assistant_text:
            messages.append({"role": "assistant", "content": self.assistant_text})
        return messages

@dataclass
class BioGemmaLoraPrompt(VisionPrompt):
    def get_messages(self) -> list[dict]:
        user_text = (
            f"{self.system_prompt}\n\n{self.user_text}"
            if self.system_prompt
            else self.user_text
        )
        content = [{"type": "text", "text": user_text}]
        if self.image_path:
            content.append({"type": "image", "image": self.image_path})
        messages = [{"role": "user", "content": content}]
        if self.assistant_text:
            messages.append({"role": "assistant", "content": self.assistant_text})
        return messages
PROMPT_CLASS_MAP = {
    "med-flamingo-9b": MedFlamingoPrompt,
    "random_med-flamingo-9b_1": MedFlamingoPrompt,
    "random_med-flamingo-9b_2": MedFlamingoPrompt,
    "random_med-flamingo-9b_3": MedFlamingoPrompt,
    "open-flamingo-9b": OpenFlamingoPrompt,
    "random_open-flamingo-9b_1": OpenFlamingoPrompt,
    "random_open-flamingo-9b_2": OpenFlamingoPrompt,
    "random_open-flamingo-9b_3": OpenFlamingoPrompt,
    "llava-med-7b": LlavaMedPrompt,
    "random_llava-med-7b_1": LlavaMedPrompt,
    "random_llava-med-7b_2": LlavaMedPrompt,
    "random_llava-med-7b_3": LlavaMedPrompt,
    "llava-v0-7b": LlavaV0Prompt,
    "random_llava-v0-7b_1": LlavaV0Prompt,
    "random_llava-v0-7b_2": LlavaV0Prompt,
    "random_llava-v0-7b_3": LlavaV0Prompt,
    "medgemma": MedGemmaPrompt,
    "random_medgemma_4b_1": MedGemmaPrompt,
    "random_medgemma_4b_2": MedGemmaPrompt,
    "random_medgemma_4b_3": MedGemmaPrompt,
    "medgemma-27b": MedGemmaPrompt,
    "random_medgemma_27b_1": MedGemmaPrompt,
    "random_medgemma_27b_2": MedGemmaPrompt,
    "random_medgemma_27b_3": MedGemmaPrompt,
    "medvlthinker-3b": MedVLThinkerPrompt,
    "medvlthinker-7b": MedVLThinkerPrompt,
    "medvlthinker-32b": MedVLThinkerPrompt,
    "medmo-4b-next": MedMOPrompt,
    "medmo-8b-next": MedMOPrompt,
    "medix-r1-2b": MedIXPrompt,
    "medix-r1-30b": MedIXPrompt,
    "qwen25-vl-3b-instruct": Qwen25VLPrompt,
    "random_Qwen2.5-VL-3B_1": Qwen25VLPrompt,
    "random_Qwen2.5-VL-3B_2": Qwen25VLPrompt,
    "random_Qwen2.5-VL-3B_3": Qwen25VLPrompt,
    "qwen25-vl-7b-instruct": Qwen25VLPrompt,
    "qwen25-vl-7b-instruct-full-path-vqa": Qwen25VLPrompt,
    "qwen25-vl-7b-instruct-full-all-med-vqa": Qwen25VLPrompt,
    "qwen25-vl-7b-instruct-full-slake": Qwen25VLPrompt,
    "qwen25-vl-7b-instruct-full-vqa-rad": Qwen25VLPrompt,
    "qwen25-vl-32b-instruct": Qwen25VLPrompt,
    "qwen2-vl-2b-instruct": Qwen2VLPrompt,
    "qwen3-vl-2b-instruct": Qwen25VLPrompt,
    "qwen3-vl-4b-instruct": Qwen25VLPrompt,
    "qwen3-vl-8b-instruct": Qwen25VLPrompt,
    "random_Qwen3-VL-8B_1": Qwen25VLPrompt,
    "random_Qwen3-VL-8B_2": Qwen25VLPrompt,
    "random_Qwen3-VL-8B_3": Qwen25VLPrompt,
    "qwen3-vl-30b-a3b-instruct": Qwen25VLPrompt,
    "adapt-qwen2-2b": AdaptQwen2VLPrompt,
    "adapt-internVL3-1b": AdaptInternVL3Prompt,
    "internvl3-1b": InternVL3Prompt,
    "random_InternVL3-1B_1": InternVL3Prompt,
    "random_InternVL3-1B_2": InternVL3Prompt,
    "random_InternVL3-1B_3": InternVL3Prompt,
    "biomedllama": BioMedLlamaPrompt,
    "adapt-llama3.2-11b": AdaptLlamaPrompt,
    "meta-llama3.2-11b-vision-instruct": MetaLlama32VisionPrompt,
    "meta-llama3.2-11b-vision-instruct-full-path-vqa": MetaLlama32VisionPrompt,
    "meta-llama3.2-11b-vision-instruct-full-all-med-vqa": MetaLlama32VisionPrompt,
    "meta-llama3.2-11b-vision-instruct-full-slake": MetaLlama32VisionPrompt,
    "meta-llama3.2-11b-vision-instruct-full-vqa-rad": MetaLlama32VisionPrompt,
    "gemma": GemmaPrompt,
    "random_gemma_4b_1": GemmaPrompt,
    "random_gemma_4b_2": GemmaPrompt,
    "random_gemma_4b_3": GemmaPrompt,
    "gemma-27b": GemmaPrompt,
    "random_gemma_27b_1": GemmaPrompt,
    "random_gemma_27b_2": GemmaPrompt,
    "random_gemma_27b_3": GemmaPrompt,
    "biogemma-lora": BioGemmaLoraPrompt,
}

def get_prompt_class(model_stem: str):
    if model_stem not in PROMPT_CLASS_MAP:
        raise ValueError(
            f"Unknown model stem: {model_stem}. Available: {list(PROMPT_CLASS_MAP.keys())}"
        )
    return PROMPT_CLASS_MAP[model_stem]
