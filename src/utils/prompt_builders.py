import json
import os
import random
from typing import Optional

from .constants import (
    SYSTEM_PROMPTS,
    SYSTEM_PROMPT_SPECIALIST_TEMPLATE,
    SYSTEM_PROMPT_GENERAL,
    REASONING_TEMPLATES,
    OUTPUT_FORMATS,
    VISION_MODELS,
    LLAVA_MODELS,
    LLAVA_MED_MODELS,
    LLAVA_SYSTEM_PROMPT_MED,
    LLAVA_SYSTEM_PROMPT_V0,
    FLAMINGO_MODELS,
    MED_FLAMINGO_MODELS,
    OPEN_FLAMINGO_MODELS,
    FLAMINGO_SYSTEM_PROMPT_MED,
    FLAMINGO_SYSTEM_PROMPT_OPEN,
    MEDVLTHINKER_MODELS,
    MEDMO_MODELS,
    MEDIX_MODELS,
    QWEN_VL_MODELS,
    QWEN2_VL_MODELS,
    ADAPT_QWEN_VL_MODELS,
    ADAPT_INTERNVL_MODELS,
    INTERNVL_MODELS,
    ADAPT_LLAMA_MODELS,
    META_LLAMA_VISION_MODELS,
)
from .loaders import VQASample, VQADataset
from .prompt_objects import (
    get_prompt_class,
    BasePrompt,
    VisionPrompt,
)
_GEMMA_SHOTS_DATASETS = frozenset({"SLAKE", "PATH-VQA", "VQA-RAD"})
_DEFAULT_GEMMA_SHOTS_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "Shots")
)
MEDVLTHINKER_INSTRUCTION_PROMPT = (
    "You will solve a problem/request. You should provide your thoughts within "
    "<think> </think> tags before providing the answer.\n"
    "Write your final answer within <answer> </answer> tags."
)

def _uses_simple_gemma_prompt(model_stem: str) -> bool:
    return (
        model_stem in {"gemma", "gemma-27b", "medgemma", "medgemma-27b"}
        or model_stem.startswith("random_gemma_4b_")
        or model_stem.startswith("random_gemma_27b_")
        or model_stem.startswith("random_medgemma_4b_")
        or model_stem.startswith("random_medgemma_27b_")
    )

def _eval_sample_is_yes_no_answer(sample: VQASample) -> bool:
    return sample.answer.strip().lower() in ("yes", "no")

def _resolve_gemma_shots_dir(
    dataset_name: str,
    sample: VQASample,
    shots_root: str,
) -> Optional[str]:
    if dataset_name not in _GEMMA_SHOTS_DATASETS:
        return None
    if dataset_name == "PATH-VQA":
        subset = "yes_no"
    else:
        subset = "yes_no" if _eval_sample_is_yes_no_answer(sample) else "multiplechoice"
    path = os.path.normpath(os.path.join(shots_root, dataset_name, subset))
    manifest_path = os.path.join(path, "manifest.json")
    samples_path = os.path.join(path, "samples.json")
    if os.path.isfile(manifest_path) or os.path.isfile(samples_path):
        return path
    return None

def _load_gemma_shots_from_manifest(shots_dir: str, n_shots: int) -> list[VQASample]:
    if n_shots <= 0:
        return []
    manifest_path = os.path.join(shots_dir, "manifest.json")
    with open(manifest_path, "r", encoding="utf-8") as f:
        entries = json.load(f)
    samples: list[VQASample] = []
    for entry in entries[:n_shots]:
        samples.append(
            VQASample(
                idx=int(entry.get("index", len(samples))),
                image_path=entry["copied_image"],
                question=entry["question"].strip(),
                options=dict(entry["options"]),
                answer=entry["correct_answer"],
                answer_label=entry["correct_label"],
                source_split="train",
            )
        )
    return samples

def _load_gemma_shots_from_samples(shots_dir: str, n_shots: int) -> list[VQASample]:
    if n_shots <= 0:
        return []
    samples_path = os.path.join(shots_dir, "samples.json")
    with open(samples_path, "r", encoding="utf-8") as f:
        entries = json.load(f)
    samples: list[VQASample] = []
    for entry in entries[:n_shots]:
        samples.append(
            VQASample(
                idx=len(samples),
                image_path=entry["image"],
                question=entry["question"].strip(),
                options=dict(entry["options"]),
                answer=entry["answer"],
                answer_label=entry["answer_label"],
                source_split=entry.get("_source_split", "train"),
            )
        )
    return samples

def _select_gemma_few_shot_samples(
    dataset: VQADataset,
    train_data: VQADataset,
    sample: VQASample,
    n_shots: int,
    seed: int,
    shots_root: str,
) -> tuple[list[VQASample], Optional[str]]:
    shots_dir = _resolve_gemma_shots_dir(dataset.dataset_name, sample, shots_root)
    if shots_dir is not None:
        manifest_path = os.path.join(shots_dir, "manifest.json")
        if os.path.isfile(manifest_path):
            loaded = _load_gemma_shots_from_manifest(shots_dir, n_shots)
        else:
            loaded = _load_gemma_shots_from_samples(shots_dir, n_shots)
        if loaded:
            return loaded, shots_dir
    return select_few_shot_samples(train_data, n_shots, seed=seed), None

def get_reordered_options(
    sample: VQASample,
    options_order: str = "default",
) -> tuple[dict[str, str], str]:
    original_items = sorted(sample.options.items())
    if not original_items:
        return {}, sample.answer_label
    original_labels = [label for label, _ in original_items]
    correct_text = sample.options[sample.answer_label]
    wrong_items = [
        (label, text) for label, text in original_items if label != sample.answer_label
    ]
    if options_order == "default":
        ordered_texts = [text for _, text in original_items]
    elif options_order == "correct_first":
        ordered_texts = [correct_text] + [text for _, text in wrong_items]
    elif options_order == "correct_end":
        ordered_texts = [text for _, text in wrong_items] + [correct_text]
    else:
        raise ValueError(f"Unknown options_order: {options_order}")
    reordered_options = dict(zip(original_labels, ordered_texts))
    effective_answer_label = next(
        label for label, text in reordered_options.items() if text == correct_text
    )
    return reordered_options, effective_answer_label

def _get_option_items(
    sample: VQASample,
    options_mode: str,
    options_order: str = "default",
) -> tuple[list[tuple[str, str]], str]:
    reordered_options, effective_answer_label = get_reordered_options(
        sample,
        options_order=options_order,
    )
    option_items = sorted(reordered_options.items())
    if options_mode in ("incorrect_options", "incorrect_options_blind"):
        option_items = [
            (label, text)
            for label, text in option_items
            if label != effective_answer_label
        ]
    return option_items, effective_answer_label

def _build_options_text(
    sample: VQASample,
    options_mode: str,
    options_order: str = "default",
) -> str:
    if options_mode == "no_options":
        return ""
    option_items, effective_answer_label = _get_option_items(
        sample,
        options_mode,
        options_order=options_order,
    )
    if options_mode == "with_options":
        lines = ["Options:"]
        for label, text in option_items:
            lines.append(f"  {label}) {text}")
        return "\n".join(lines)
    if options_mode in ("incorrect_options", "incorrect_options_blind"):
        lines = ["Options:"]
        for label, text in option_items:
            lines.append(f"  {label}) {text}")
        if options_mode == "incorrect_options":
            lines.append(
                "\nNote: If none of the provided options are correct, "
                "state 'None of the above' and provide what you believe is the correct answer."
            )
        return "\n".join(lines)
    raise ValueError(f"Unknown options_mode: {options_mode}")

def _build_simple_gemma_prompt_text(
    sample: VQASample,
    options_mode: str,
    options_order: str = "default",
) -> str:
    parts = [sample.question.strip()]
    option_items, _ = _get_option_items(
        sample,
        options_mode,
        options_order=options_order,
    )
    if option_items:
        parts.extend(f"{label}: {text}" for label, text in option_items)
    return "\n".join(parts)

def _build_simple_gemma_few_shot_user_text(
    sample: VQASample,
    options_mode: str,
    options_order: str = "default",
) -> str:
    parts = [f"Question: {sample.question.strip()}"]
    option_items, _ = _get_option_items(
        sample,
        options_mode,
        options_order=options_order,
    )
    if option_items:
        parts.extend(f"{label}: {text}" for label, text in option_items)
    return "\n".join(parts)

def _get_option_texts(
    sample: VQASample,
    options_mode: str,
    options_order: str = "default",
) -> list[str]:
    if options_mode == "no_options":
        return []
    option_items, _ = _get_option_items(
        sample,
        options_mode,
        options_order=options_order,
    )
    if options_mode in ("with_options", "incorrect_options", "incorrect_options_blind"):
        return [text for _, text in option_items]
    raise ValueError(f"Unknown options_mode: {options_mode}")

def _build_llava_user_text(
    sample: VQASample,
    reasoning: str,
    options_mode: str,
    options_order: str = "default",
) -> str:
    question_text = sample.question.strip()
    option_texts = _get_option_texts(sample, options_mode, options_order=options_order)
    if option_texts:
        question_text += (
            f" Please choose from the following options: [{', '.join(option_texts)}]."
        )
    parts = [question_text]
    if reasoning != "direct":
        parts.append(REASONING_TEMPLATES[reasoning])
    return "\n".join(parts)

def _build_llava_few_shot_example(
    sample: VQASample,
    options_mode: str,
    options_order: str = "default",
) -> dict:
    user_text = _build_llava_user_text(
        sample=sample,
        reasoning="direct",
        options_mode=options_mode,
        options_order=options_order,
    )
    assistant_text = sample.answer.strip()
    return {"user_text": user_text, "assistant_text": assistant_text}

def _build_adapt_train_few_shot_examples_with_images(
    selected_samples: list[VQASample],
    train_data: VQADataset,
    options_mode: str,
    model_stem: str,
    options_order: str = "default",
) -> list[dict]:
    examples: list[dict] = []
    for selected_sample in selected_samples:
        example = build_few_shot_example(
            selected_sample,
            options_mode,
            model_stem=model_stem,
            options_order=options_order,
        )
        image_path = os.path.join(train_data.image_base_dir, selected_sample.image_path)
        if os.path.exists(image_path):
            example["image_path"] = image_path
        examples.append(example)
    return examples

def _build_meta_llama_train_few_shot_examples_with_images(
    selected_samples: list[VQASample],
    train_data: VQADataset,
    options_mode: str,
    model_stem: str,
    options_order: str = "default",
) -> list[dict]:
    examples: list[dict] = []
    for selected_sample in selected_samples:
        example = build_few_shot_example(
            selected_sample,
            options_mode,
            model_stem=model_stem,
            options_order=options_order,
        )
        image_path = os.path.join(train_data.image_base_dir, selected_sample.image_path)
        if os.path.exists(image_path):
            example["image_path"] = image_path
        examples.append(example)
    return examples

def _build_internvl_train_few_shot_examples_with_images(
    selected_samples: list[VQASample],
    train_data: VQADataset,
    options_mode: str,
    model_stem: str,
    options_order: str = "default",
) -> list[dict]:
    examples: list[dict] = []
    for selected_sample in selected_samples:
        example = build_few_shot_example(
            selected_sample,
            options_mode,
            model_stem=model_stem,
            options_order=options_order,
        )
        image_path = os.path.join(train_data.image_base_dir, selected_sample.image_path)
        if os.path.exists(image_path):
            example["image_path"] = image_path
        examples.append(example)
    return examples

def _build_qwen_train_few_shot_examples_with_images(
    selected_samples: list[VQASample],
    train_data: VQADataset,
    options_mode: str,
    model_stem: str,
    options_order: str = "default",
) -> list[dict]:
    examples: list[dict] = []
    for selected_sample in selected_samples:
        example = build_few_shot_example(
            selected_sample,
            options_mode,
            model_stem=model_stem,
            options_order=options_order,
        )
        image_path = os.path.join(train_data.image_base_dir, selected_sample.image_path)
        if os.path.exists(image_path):
            example["image_path"] = image_path
        examples.append(example)
    return examples

def _build_medvlthinker_train_few_shot_examples_with_images(
    selected_samples: list[VQASample],
    train_data: VQADataset,
    options_mode: str,
    model_stem: str,
    options_order: str = "default",
) -> list[dict]:
    examples: list[dict] = []
    for selected_sample in selected_samples:
        example = build_few_shot_example(
            selected_sample,
            options_mode,
            model_stem=model_stem,
            options_order=options_order,
        )
        image_path = os.path.join(train_data.image_base_dir, selected_sample.image_path)
        if os.path.exists(image_path):
            example["image_path"] = image_path
        examples.append(example)
    return examples

def _build_medix_train_few_shot_examples_with_images(
    selected_samples: list[VQASample],
    train_data: VQADataset,
    options_mode: str,
    model_stem: str,
    options_order: str = "default",
) -> list[dict]:
    examples: list[dict] = []
    for selected_sample in selected_samples:
        example = build_few_shot_example(
            selected_sample,
            options_mode,
            model_stem=model_stem,
            options_order=options_order,
        )
        image_path = os.path.join(train_data.image_base_dir, selected_sample.image_path)
        if os.path.exists(image_path):
            example["image_path"] = image_path
        examples.append(example)
    return examples

def _build_llava_train_few_shot_examples_with_images(
    selected_samples: list[VQASample],
    train_data: VQADataset,
    options_mode: str,
    model_stem: str,
    options_order: str = "default",
    image_base_dir_override: Optional[str] = None,
) -> list[dict]:
    examples: list[dict] = []
    for selected_sample in selected_samples:
        example = build_few_shot_example(
            selected_sample,
            options_mode,
            model_stem=model_stem,
            options_order=options_order,
        )
        example["has_image"] = False
        examples.append(example)
    return examples

def _build_flamingo_option_lines(
    sample: VQASample,
    options_mode: str,
    options_order: str = "default",
) -> list[str]:
    option_items, _ = _get_option_items(
        sample,
        options_mode,
        options_order=options_order,
    )
    return [f"({label}) {text}" for label, text in option_items]

def _build_flamingo_user_text(
    sample: VQASample,
    reasoning: str,
    options_mode: str,
    options_order: str = "default",
) -> str:
    lines = [sample.question.replace("<image>", "").strip()]
    lines.extend(
        _build_flamingo_option_lines(sample, options_mode, options_order=options_order)
    )
    if reasoning != "direct":
        lines.append(REASONING_TEMPLATES[reasoning])
    return "\n".join(lines)

def _build_flamingo_few_shot_example(
    sample: VQASample,
    dataset: VQADataset,
    options_mode: str,
    options_order: str = "default",
) -> dict:
    image_path = os.path.join(dataset.image_base_dir, sample.image_path)
    if not os.path.exists(image_path):
        image_path = None
    reordered_options, effective_answer_label = get_reordered_options(
        sample, options_order=options_order
    )
    if options_mode == "no_options":
        assistant_text = sample.answer.strip()
    else:
        option_text = reordered_options[effective_answer_label]
        assistant_text = f"({effective_answer_label}) {option_text}"
    return {
        "user_text": _build_flamingo_user_text(
            sample,
            reasoning="direct",
            options_mode=options_mode,
            options_order=options_order,
        ),
        "assistant_text": assistant_text,
        "image_path": image_path,
        "image_count": 1 if image_path else 0,
    }

def _build_medvlthinker_prompt_text(
    sample: VQASample,
    options_mode: str,
    options_order: str = "default",
) -> str:
    lines = [f"Question: {sample.question.strip()}"]
    if options_mode != "no_options":
        lines.extend(["", "Options:"])
        option_items, _ = _get_option_items(
            sample,
            options_mode,
            options_order=options_order,
        )
        for label, option_text in option_items:
            lines.extend(["", f"{label}. {option_text}"])
        if options_mode == "incorrect_options":
            lines.extend(
                [
                    "",
                    "If none of the provided options are correct, answer with the correct answer text inside <answer> </answer>.",
                ]
            )
    return "\n".join(lines).strip()

def _build_medvlthinker_few_shot_example(
    sample: VQASample,
    options_mode: str,
    options_order: str = "default",
) -> dict:
    _, effective_answer_label = get_reordered_options(
        sample, options_order=options_order
    )
    if options_mode == "no_options":
        assistant_text = f"<answer> {sample.answer.strip()} </answer>"
    else:
        assistant_text = f"<answer> {effective_answer_label.strip()} </answer>"
    prompt_options_mode = options_mode
    if options_mode in ("incorrect_options", "incorrect_options_blind"):
        prompt_options_mode = "with_options"
    return {
        "user_text": _build_medvlthinker_prompt_text(
            sample, options_mode=prompt_options_mode, options_order=options_order
        ),
        "assistant_text": assistant_text,
    }

def _build_qwen25_vl_prompt_text(
    sample: VQASample,
    options_mode: str,
    options_order: str = "default",
) -> str:
    lines = [f"Question: {sample.question.strip()}"]
    if options_mode != "no_options":
        lines.append("Options:")
        option_items, _ = _get_option_items(
            sample,
            options_mode,
            options_order=options_order,
        )
        for label, option_text in option_items:
            lines.append(f"{label}. {option_text}")
        lines.append("Please select the correct answer from the options above.")
    return "\n".join(lines).strip()

def _build_qwen25_vl_few_shot_example(
    sample: VQASample,
    options_mode: str,
    options_order: str = "default",
) -> dict:
    prompt_options_mode = options_mode
    if options_mode in ("incorrect_options", "incorrect_options_blind"):
        prompt_options_mode = "with_options"
    if options_mode == "no_options":
        assistant_text = sample.answer.strip()
    else:
        reordered_options, effective_answer_label = get_reordered_options(
            sample, options_order=options_order
        )
        assistant_text = (
            f"{effective_answer_label.strip()}. "
            f"{reordered_options[effective_answer_label].strip()}"
        )
    return {
        "user_text": _build_qwen25_vl_prompt_text(
            sample, options_mode=prompt_options_mode, options_order=options_order
        ),
        "assistant_text": assistant_text,
    }

def _build_medmo_prompt_text(
    sample: VQASample,
    options_mode: str,
    options_order: str = "default",
) -> str:
    if options_mode == "no_options":
        return sample.question.strip()
    option_items, _ = _get_option_items(
        sample,
        options_mode,
        options_order=options_order,
    )
    option_lines = []
    for idx, (_, option_text) in enumerate(option_items):
        label = chr(ord("A") + idx)
        option_lines.append(f"({label}) {option_text}")
    return (
        f"{sample.question.strip()}\n" "Options:\n" + "\n".join(option_lines)
    ).strip()

def _build_medmo_few_shot_example(
    sample: VQASample,
    options_mode: str,
    options_order: str = "default",
) -> dict:
    prompt_options_mode = options_mode
    if options_mode in ("incorrect_options", "incorrect_options_blind"):
        prompt_options_mode = "with_options"
    if options_mode == "no_options":
        assistant_text = sample.answer.strip()
    else:
        reordered_options, effective_answer_label = get_reordered_options(
            sample, options_order=options_order
        )
        assistant_text = (
            f"({effective_answer_label.strip()}) "
            f"{reordered_options[effective_answer_label].strip()}"
        )
    return {
        "user_text": _build_medmo_prompt_text(
            sample, options_mode=prompt_options_mode, options_order=options_order
        ),
        "assistant_text": assistant_text,
    }

def _build_adapt_llama_prompt_text(
    sample: VQASample,
    options_mode: str,
    options_order: str = "default",
) -> str:
    if options_mode == "no_options":
        return f"Question: {sample.question.strip()}"
    options = _get_option_texts(sample, options_mode, options_order=options_order)
    return f"Question: {sample.question.strip()}\nThe choices are: {options}"

def _build_adapt_qwen_prompt_text(
    sample: VQASample,
    options_mode: str,
    options_order: str = "default",
) -> str:
    if options_mode == "no_options":
        return f"Question: {sample.question.strip()}"
    options = _get_option_texts(sample, options_mode, options_order=options_order)
    return f"Question: {sample.question.strip()}\nThe choices are: {options}"

def _build_qwen2_vl_prompt_text(
    sample: VQASample,
    options_mode: str,
    options_order: str = "default",
) -> str:
    if options_mode == "no_options":
        return f"Question: {sample.question.strip()}"
    option_items, _ = _get_option_items(
        sample,
        options_mode,
        options_order=options_order,
    )
    option_lines = [f"{label}. {option_text}" for label, option_text in option_items]
    return "\n".join(
        [
            f"Question: {sample.question.strip()}",
            "Options:",
            *option_lines,
            "Please select the correct answer from the options above.",
        ]
    )

def _build_adapt_internvl_prompt_text(
    sample: VQASample,
    options_mode: str,
    options_order: str = "default",
) -> str:
    if options_mode == "no_options":
        return f"Question: {sample.question.strip()}"
    options = _get_option_texts(sample, options_mode, options_order=options_order)
    return f"Question: {sample.question.strip()}\nThe choices are: {options}"

def _build_internvl_prompt_text(
    sample: VQASample,
    options_mode: str,
    options_order: str = "default",
) -> str:
    if options_mode == "no_options":
        return f"Question: {sample.question.strip()}"
    option_items, _ = _get_option_items(
        sample,
        options_mode,
        options_order=options_order,
    )
    option_lines = [f"{label}. {option_text}" for label, option_text in option_items]
    return "\n".join(
        [
            f"Question: {sample.question.strip()}",
            "Options:",
            *option_lines,
            "Please select the correct answer from the options above.",
        ]
    )

def _build_adapt_qwen_few_shot_example(
    sample: VQASample,
    options_mode: str,
    options_order: str = "default",
) -> dict:
    prompt_options_mode = options_mode
    if options_mode in ("incorrect_options", "incorrect_options_blind"):
        prompt_options_mode = "with_options"
    if options_mode == "no_options":
        assistant_text = sample.answer.strip()
    else:
        reordered_options, effective_answer_label = get_reordered_options(
            sample, options_order=options_order
        )
        assistant_text = reordered_options[effective_answer_label].strip()
    return {
        "user_text": _build_adapt_qwen_prompt_text(
            sample, options_mode=prompt_options_mode, options_order=options_order
        ),
        "assistant_text": assistant_text,
    }

def _build_qwen2_vl_few_shot_example(
    sample: VQASample,
    options_mode: str,
    options_order: str = "default",
) -> dict:
    prompt_options_mode = options_mode
    if options_mode in ("incorrect_options", "incorrect_options_blind"):
        prompt_options_mode = "with_options"
    if options_mode == "no_options":
        assistant_text = sample.answer.strip()
    else:
        reordered_options, effective_answer_label = get_reordered_options(
            sample, options_order=options_order
        )
        assistant_text = (
            f"{effective_answer_label.strip()}. "
            f"{reordered_options[effective_answer_label].strip()}"
        )
    return {
        "user_text": _build_qwen2_vl_prompt_text(
            sample, options_mode=prompt_options_mode, options_order=options_order
        ),
        "assistant_text": assistant_text,
    }

def _build_adapt_internvl_few_shot_example(
    sample: VQASample,
    options_mode: str,
    options_order: str = "default",
) -> dict:
    prompt_options_mode = options_mode
    if options_mode in ("incorrect_options", "incorrect_options_blind"):
        prompt_options_mode = "with_options"
    if options_mode == "no_options":
        assistant_text = sample.answer.strip()
    else:
        reordered_options, effective_answer_label = get_reordered_options(
            sample, options_order=options_order
        )
        assistant_text = (
            f"{effective_answer_label.strip()}. "
            f"{reordered_options[effective_answer_label].strip()}"
        )
    return {
        "user_text": _build_adapt_internvl_prompt_text(
            sample, options_mode=prompt_options_mode, options_order=options_order
        ),
        "assistant_text": assistant_text,
    }

def _build_internvl_few_shot_example(
    sample: VQASample,
    options_mode: str,
    options_order: str = "default",
) -> dict:
    prompt_options_mode = options_mode
    if options_mode in ("incorrect_options", "incorrect_options_blind"):
        prompt_options_mode = "with_options"
    if options_mode == "no_options":
        assistant_text = sample.answer.strip()
    else:
        reordered_options, effective_answer_label = get_reordered_options(
            sample, options_order=options_order
        )
        assistant_text = (
            f"{effective_answer_label.strip()}. "
            f"{reordered_options[effective_answer_label].strip()}"
        )
    return {
        "user_text": _build_internvl_prompt_text(
            sample, options_mode=prompt_options_mode, options_order=options_order
        ),
        "assistant_text": assistant_text,
    }

def _build_adapt_llama_few_shot_example(
    sample: VQASample,
    options_mode: str,
    options_order: str = "default",
) -> dict:
    prompt_options_mode = options_mode
    if options_mode in ("incorrect_options", "incorrect_options_blind"):
        prompt_options_mode = "with_options"
    if options_mode == "no_options":
        assistant_text = sample.answer.strip()
    else:
        reordered_options, effective_answer_label = get_reordered_options(
            sample, options_order=options_order
        )
        assistant_text = reordered_options[effective_answer_label].strip()
    return {
        "user_text": _build_adapt_llama_prompt_text(
            sample, options_mode=prompt_options_mode, options_order=options_order
        ),
        "assistant_text": assistant_text,
    }

def _build_meta_llama_prompt_text(
    sample: VQASample,
    options_mode: str,
    options_order: str = "default",
) -> str:
    lines = [sample.question.strip()]
    if options_mode != "no_options":
        lines.append("Options:")
        option_items, _ = _get_option_items(
            sample,
            options_mode,
            options_order=options_order,
        )
        for label, option_text in option_items:
            lines.append(f"{label}) {option_text}")
        if options_mode == "incorrect_options":
            lines.append(
                "If none of the options are correct, answer with 'None of the above' only."
            )
        else:
            lines.append("Answer with the letter only.")
    else:
        lines.append("Answer briefly.")
    return "\n".join(lines)

def _build_meta_llama_few_shot_example(
    sample: VQASample,
    options_mode: str,
    options_order: str = "default",
) -> dict:
    prompt_options_mode = options_mode
    if options_mode in ("incorrect_options", "incorrect_options_blind"):
        prompt_options_mode = "with_options"
    if options_mode == "no_options":
        assistant_text = sample.answer.strip()
    else:
        _, effective_answer_label = get_reordered_options(
            sample, options_order=options_order
        )
        assistant_text = effective_answer_label.strip()
    return {
        "user_text": _build_meta_llama_prompt_text(
            sample, options_mode=prompt_options_mode, options_order=options_order
        ),
        "assistant_text": assistant_text,
    }

def _build_medix_prompt_parts(
    sample: VQASample,
    options_mode: str,
    options_order: str = "default",
) -> tuple[str, list[str]]:
    option_texts = _get_option_texts(sample, options_mode, options_order=options_order)
    return sample.question.strip(), option_texts

def _build_medix_few_shot_example(
    sample: VQASample,
    options_mode: str,
    options_order: str = "default",
) -> dict:
    question_text, option_texts = _build_medix_prompt_parts(
        sample,
        (
            "with_options"
            if options_mode in ("incorrect_options", "incorrect_options_blind")
            else options_mode
        ),
        options_order=options_order,
    )
    if options_mode == "no_options":
        assistant_text = f"<answer>{sample.answer.strip()}</answer>"
    else:
        _, effective_answer_label = get_reordered_options(
            sample, options_order=options_order
        )
        assistant_text = f"<answer>{effective_answer_label.strip()}</answer>"
    return {
        "user_text": question_text,
        "assistant_text": assistant_text,
        "options": option_texts,
    }

def _load_specialist_prompt(
    sample: VQASample,
    dataset_name: str,
    specialist_dir: str = "./data/specialist_prompts",
) -> str:
    specialist_path = os.path.join(
        specialist_dir, dataset_name, f"sample_{sample.idx}.txt"
    )
    if os.path.exists(specialist_path):
        with open(specialist_path, "r", encoding="utf-8") as f:
            return f.read().strip()
    return SYSTEM_PROMPT_GENERAL

def resolve_system_prompt(
    approach: str,
    sample: Optional[VQASample] = None,
    dataset_name: str = "",
    specialist_dir: str = "./data/specialist_prompts",
) -> str:
    if approach == "image_question_specialist":
        if sample is None:
            raise ValueError("Specialist approach requires a sample to load the role.")
        specialist_role = _load_specialist_prompt(sample, dataset_name, specialist_dir)
        return SYSTEM_PROMPT_SPECIALIST_TEMPLATE.format(specialist_role=specialist_role)
    if approach not in SYSTEM_PROMPTS:
        raise ValueError(f"Unknown approach: {approach}")
    return SYSTEM_PROMPTS[approach]

def resolve_system_prompt_for_model(
    model_stem: str,
    approach: str,
    sample: Optional[VQASample] = None,
    dataset_name: str = "",
    specialist_dir: str = "./data/specialist_prompts",
) -> str:
    if (
        model_stem in MEDVLTHINKER_MODELS
        or model_stem in MEDMO_MODELS
        or model_stem in MEDIX_MODELS
        or model_stem in QWEN_VL_MODELS
        or model_stem in ADAPT_QWEN_VL_MODELS
        or model_stem in ADAPT_INTERNVL_MODELS
        or model_stem in INTERNVL_MODELS
        or model_stem in ADAPT_LLAMA_MODELS
        or model_stem in META_LLAMA_VISION_MODELS
    ):
        return ""
    if model_stem in LLAVA_MODELS and approach == "image_question":
        if model_stem in LLAVA_MED_MODELS:
            return LLAVA_SYSTEM_PROMPT_MED
        return LLAVA_SYSTEM_PROMPT_V0
    if model_stem in FLAMINGO_MODELS and approach == "image_question":
        if model_stem in MED_FLAMINGO_MODELS:
            return FLAMINGO_SYSTEM_PROMPT_MED
        return FLAMINGO_SYSTEM_PROMPT_OPEN
    return resolve_system_prompt(
        approach=approach,
        sample=sample,
        dataset_name=dataset_name,
        specialist_dir=specialist_dir,
    )

def select_few_shot_samples(
    train_data: VQADataset,
    n_shots: int,
    seed: int = 42,
) -> list[VQASample]:
    if n_shots <= 0:
        return []
    rng = random.Random(seed)
    if n_shots >= len(train_data.samples):
        selected = list(train_data.samples)
        rng.shuffle(selected)
        return selected
    return rng.sample(train_data.samples, n_shots)

def build_few_shot_example(
    sample: VQASample,
    options_mode: str,
    model_stem: Optional[str] = None,
    options_order: str = "default",
) -> dict:
    if model_stem in LLAVA_MODELS:
        return _build_llava_few_shot_example(
            sample, options_mode, options_order=options_order
        )
    if model_stem in FLAMINGO_MODELS:
        raise ValueError(
            "Flamingo few-shot examples require dataset context; use _build_flamingo_few_shot_example()."
        )
    if model_stem in MEDVLTHINKER_MODELS:
        return _build_medvlthinker_few_shot_example(
            sample, options_mode, options_order=options_order
        )
    if model_stem in MEDMO_MODELS:
        return _build_medmo_few_shot_example(
            sample, options_mode, options_order=options_order
        )
    if model_stem in MEDIX_MODELS:
        return _build_medix_few_shot_example(
            sample, options_mode, options_order=options_order
        )
    if model_stem in QWEN2_VL_MODELS:
        return _build_qwen2_vl_few_shot_example(
            sample, options_mode, options_order=options_order
        )
    if model_stem in ADAPT_QWEN_VL_MODELS:
        return _build_adapt_qwen_few_shot_example(
            sample, options_mode, options_order=options_order
        )
    if model_stem in ADAPT_INTERNVL_MODELS:
        return _build_adapt_internvl_few_shot_example(
            sample, options_mode, options_order=options_order
        )
    if model_stem in INTERNVL_MODELS:
        return _build_internvl_few_shot_example(
            sample, options_mode, options_order=options_order
        )
    if model_stem in QWEN_VL_MODELS:
        return _build_qwen25_vl_few_shot_example(
            sample, options_mode, options_order=options_order
        )
    if model_stem in ADAPT_LLAMA_MODELS:
        return _build_adapt_llama_few_shot_example(
            sample, options_mode, options_order=options_order
        )
    if model_stem in META_LLAMA_VISION_MODELS:
        return _build_meta_llama_few_shot_example(
            sample, options_mode, options_order=options_order
        )
    if _uses_simple_gemma_prompt(model_stem):
        effective_options, effective_answer_label = get_reordered_options(
            sample,
            options_order=options_order,
        )
        if options_mode == "no_options":
            user_text = _build_simple_gemma_few_shot_user_text(
                sample,
                options_mode,
                options_order=options_order,
            )
            assistant_text = sample.answer.strip()
        else:
            user_text = _build_simple_gemma_few_shot_user_text(
                sample,
                options_mode,
                options_order=options_order,
            )
            assistant_text = (
                f"{effective_answer_label}: {effective_options[effective_answer_label]}"
            )
        return {"user_text": user_text, "assistant_text": assistant_text}
    effective_options, effective_answer_label = get_reordered_options(
        sample, options_order=options_order
    )
    question_text = f"Question: {sample.question}"
    if options_mode == "no_options":
        user_text = question_text
        assistant_text = f"The answer is: {sample.answer}"
    else:
        lines = ["Options:"]
        for label, text in sorted(effective_options.items()):
            lines.append(f"  {label}) {text}")
        options_str = "\n".join(lines)
        user_text = f"{question_text}\n\n{options_str}"
        assistant_text = f"The answer is [{effective_answer_label}]"
    return {"user_text": user_text, "assistant_text": assistant_text}

def build_prompt(
    sample: VQASample,
    dataset: VQADataset,
    model_stem: str,
    approach: str,
    reasoning: str,
    options_mode: str,
    options_order: str = "default",
    specialist_dir: str = "./data/specialist_prompts",
) -> BasePrompt:
    if _uses_simple_gemma_prompt(model_stem):
        system_prompt = ""
        user_text = _build_simple_gemma_prompt_text(
            sample,
            options_mode,
            options_order=options_order,
        )
        prompt_options = None
    else:
        system_prompt = resolve_system_prompt_for_model(
            model_stem=model_stem,
            approach=approach,
            sample=sample,
            dataset_name=dataset.dataset_name,
            specialist_dir=specialist_dir,
        )
        if model_stem in MEDVLTHINKER_MODELS:
            system_prompt = MEDVLTHINKER_INSTRUCTION_PROMPT
        if model_stem in LLAVA_MODELS:
            user_text = _build_llava_user_text(
                sample, reasoning, options_mode, options_order=options_order
            )
        elif model_stem in FLAMINGO_MODELS:
            user_text = _build_flamingo_user_text(
                sample, reasoning, options_mode, options_order=options_order
            )
        elif model_stem in MEDVLTHINKER_MODELS:
            user_text = _build_medvlthinker_prompt_text(
                sample, options_mode, options_order=options_order
            )
            prompt_options = None
        elif model_stem in MEDMO_MODELS:
            user_text = _build_medmo_prompt_text(
                sample, options_mode, options_order=options_order
            )
            prompt_options = None
        elif model_stem in MEDIX_MODELS:
            user_text, prompt_options = _build_medix_prompt_parts(
                sample, options_mode, options_order=options_order
            )
        elif model_stem in QWEN2_VL_MODELS:
            user_text = _build_qwen2_vl_prompt_text(
                sample, options_mode, options_order=options_order
            )
            prompt_options = None
        elif model_stem in ADAPT_QWEN_VL_MODELS:
            user_text = _build_adapt_qwen_prompt_text(
                sample, options_mode, options_order=options_order
            )
            prompt_options = None
        elif model_stem in ADAPT_INTERNVL_MODELS:
            user_text = _build_adapt_internvl_prompt_text(
                sample, options_mode, options_order=options_order
            )
            prompt_options = None
        elif model_stem in INTERNVL_MODELS:
            user_text = _build_internvl_prompt_text(
                sample, options_mode, options_order=options_order
            )
            prompt_options = None
        elif model_stem in ADAPT_LLAMA_MODELS:
            user_text = _build_adapt_llama_prompt_text(
                sample, options_mode, options_order=options_order
            )
            prompt_options = None
        elif model_stem in META_LLAMA_VISION_MODELS:
            user_text = _build_meta_llama_prompt_text(
                sample, options_mode, options_order=options_order
            )
            prompt_options = None
        elif model_stem in QWEN_VL_MODELS:
            user_text = _build_qwen25_vl_prompt_text(
                sample, options_mode, options_order=options_order
            )
            prompt_options = None
        else:
            question_text = f"Question: {sample.question}"
            options_text = _build_options_text(
                sample, options_mode, options_order=options_order
            )
            reasoning_text = REASONING_TEMPLATES[reasoning]
            output_format = OUTPUT_FORMATS[options_mode]
            parts = [question_text]
            if options_text:
                parts.append(options_text)
            parts.append(reasoning_text)
            parts.append(output_format)
            user_text = "\n\n".join(parts)
            prompt_options = None
    image_path = None
    if model_stem in VISION_MODELS:
        image_path = os.path.join(dataset.image_base_dir, sample.image_path)
        if not os.path.exists(image_path):
            image_path = None
    PromptClass = get_prompt_class(model_stem)
    prompt_kwargs = {
        "user_text": user_text,
        "system_prompt": system_prompt,
        "image_path": image_path,
    }
    if model_stem in MEDIX_MODELS:
        prompt_kwargs["options"] = prompt_options
    prompt = PromptClass(**prompt_kwargs)
    return prompt

def build_few_shot_conversation(
    sample: VQASample,
    dataset: VQADataset,
    train_data: VQADataset,
    model_stem: str,
    approach: str,
    reasoning: str,
    options_mode: str,
    n_shots: int,
    options_order: str = "default",
    specialist_dir: str = "./data/specialist_prompts",
    seed: int = 42,
    shots_root: str = _DEFAULT_GEMMA_SHOTS_ROOT,
) -> tuple[str, list[dict], BasePrompt]:
    use_simple_gemma_few_shot = (
        _uses_simple_gemma_prompt(model_stem) and options_order == "default"
    )
    if _uses_simple_gemma_prompt(model_stem):
        system_prompt = ""
    else:
        system_prompt = resolve_system_prompt_for_model(
            model_stem=model_stem,
            approach=approach,
            sample=sample,
            dataset_name=dataset.dataset_name,
            specialist_dir=specialist_dir,
        )
        if model_stem in MEDVLTHINKER_MODELS:
            system_prompt = MEDVLTHINKER_INSTRUCTION_PROMPT
    if _uses_simple_gemma_prompt(model_stem) and not use_simple_gemma_few_shot:
        examples = []
    else:
        selected_samples = select_few_shot_samples(train_data, n_shots, seed=seed)
        if model_stem in MED_FLAMINGO_MODELS:
            flamingo_samples, flamingo_image_base = _select_gemma_few_shot_samples(
                dataset,
                train_data,
                sample,
                n_shots,
                seed,
                shots_root,
            )
            flamingo_data_ctx = VQADataset(
                samples=flamingo_samples,
                split=train_data.split,
                dataset_name=train_data.dataset_name,
                image_base_dir=flamingo_image_base or train_data.image_base_dir,
            )
            examples = [
                _build_flamingo_few_shot_example(
                    s,
                    flamingo_data_ctx,
                    options_mode,
                    options_order=options_order,
                )
                for s in flamingo_samples
            ]
        elif model_stem in OPEN_FLAMINGO_MODELS:
            flamingo_samples, flamingo_image_base = _select_gemma_few_shot_samples(
                dataset,
                train_data,
                sample,
                n_shots,
                seed,
                shots_root,
            )
            flamingo_data_ctx = VQADataset(
                samples=flamingo_samples,
                split=train_data.split,
                dataset_name=train_data.dataset_name,
                image_base_dir=flamingo_image_base or train_data.image_base_dir,
            )
            examples = [
                _build_flamingo_few_shot_example(
                    s,
                    flamingo_data_ctx,
                    options_mode,
                    options_order=options_order,
                )
                for s in flamingo_samples
            ]
        elif _uses_simple_gemma_prompt(model_stem):
            examples = []
            gemma_samples, gemma_image_base = _select_gemma_few_shot_samples(
                dataset,
                train_data,
                sample,
                n_shots,
                seed,
                shots_root,
            )
            for selected_sample in gemma_samples:
                example = build_few_shot_example(
                    selected_sample,
                    options_mode,
                    model_stem=model_stem,
                    options_order=options_order,
                )
                base = gemma_image_base or train_data.image_base_dir
                image_path = os.path.join(base, selected_sample.image_path)
                if os.path.exists(image_path):
                    example["image_path"] = image_path
                examples.append(example)
        elif model_stem in LLAVA_MED_MODELS:
            llava_samples, llava_image_base = _select_gemma_few_shot_samples(
                dataset,
                train_data,
                sample,
                n_shots,
                seed,
                shots_root,
            )
            examples = _build_llava_train_few_shot_examples_with_images(
                llava_samples,
                train_data,
                options_mode,
                model_stem,
                options_order=options_order,
                image_base_dir_override=llava_image_base,
            )
        elif model_stem in LLAVA_MODELS:
            llava_samples, llava_image_base = _select_gemma_few_shot_samples(
                dataset,
                train_data,
                sample,
                n_shots,
                seed,
                shots_root,
            )
            examples = _build_llava_train_few_shot_examples_with_images(
                llava_samples,
                train_data,
                options_mode,
                model_stem,
                options_order=options_order,
                image_base_dir_override=llava_image_base,
            )
        elif model_stem in ADAPT_QWEN_VL_MODELS:
            adapt_samples, adapt_image_base = _select_gemma_few_shot_samples(
                dataset,
                train_data,
                sample,
                n_shots,
                seed,
                shots_root,
            )
            adapt_data_ctx = VQADataset(
                samples=adapt_samples,
                split=train_data.split,
                dataset_name=train_data.dataset_name,
                image_base_dir=adapt_image_base or train_data.image_base_dir,
            )
            examples = _build_adapt_train_few_shot_examples_with_images(
                adapt_samples,
                adapt_data_ctx,
                options_mode,
                model_stem,
                options_order=options_order,
            )
        elif model_stem in ADAPT_INTERNVL_MODELS:
            adapt_samples, adapt_image_base = _select_gemma_few_shot_samples(
                dataset,
                train_data,
                sample,
                n_shots,
                seed,
                shots_root,
            )
            adapt_data_ctx = VQADataset(
                samples=adapt_samples,
                split=train_data.split,
                dataset_name=train_data.dataset_name,
                image_base_dir=adapt_image_base or train_data.image_base_dir,
            )
            examples = _build_adapt_train_few_shot_examples_with_images(
                adapt_samples,
                adapt_data_ctx,
                options_mode,
                model_stem,
                options_order=options_order,
            )
        elif model_stem in ADAPT_LLAMA_MODELS:
            adapt_samples, adapt_image_base = _select_gemma_few_shot_samples(
                dataset,
                train_data,
                sample,
                n_shots,
                seed,
                shots_root,
            )
            adapt_data_ctx = VQADataset(
                samples=adapt_samples,
                split=train_data.split,
                dataset_name=train_data.dataset_name,
                image_base_dir=adapt_image_base or train_data.image_base_dir,
            )
            examples = _build_adapt_train_few_shot_examples_with_images(
                adapt_samples,
                adapt_data_ctx,
                options_mode,
                model_stem,
                options_order=options_order,
            )
        elif model_stem in META_LLAMA_VISION_MODELS:
            examples = _build_meta_llama_train_few_shot_examples_with_images(
                selected_samples,
                train_data,
                options_mode,
                model_stem,
                options_order=options_order,
            )
        elif model_stem in INTERNVL_MODELS:
            intern_samples, intern_image_base = _select_gemma_few_shot_samples(
                dataset,
                train_data,
                sample,
                n_shots,
                seed,
                shots_root,
            )
            intern_data_ctx = VQADataset(
                samples=intern_samples,
                split=train_data.split,
                dataset_name=train_data.dataset_name,
                image_base_dir=intern_image_base or train_data.image_base_dir,
            )
            examples = _build_internvl_train_few_shot_examples_with_images(
                intern_samples,
                intern_data_ctx,
                options_mode,
                model_stem,
                options_order=options_order,
            )
        elif model_stem in QWEN_VL_MODELS:
            qwen_samples, qwen_image_base = _select_gemma_few_shot_samples(
                dataset,
                train_data,
                sample,
                n_shots,
                seed,
                shots_root,
            )
            qwen_data_ctx = VQADataset(
                samples=qwen_samples,
                split=train_data.split,
                dataset_name=train_data.dataset_name,
                image_base_dir=qwen_image_base or train_data.image_base_dir,
            )
            examples = _build_qwen_train_few_shot_examples_with_images(
                qwen_samples,
                qwen_data_ctx,
                options_mode,
                model_stem,
                options_order=options_order,
            )
        elif model_stem in QWEN2_VL_MODELS:
            qwen_samples, qwen_image_base = _select_gemma_few_shot_samples(
                dataset,
                train_data,
                sample,
                n_shots,
                seed,
                shots_root,
            )
            qwen_data_ctx = VQADataset(
                samples=qwen_samples,
                split=train_data.split,
                dataset_name=train_data.dataset_name,
                image_base_dir=qwen_image_base or train_data.image_base_dir,
            )
            examples = _build_qwen_train_few_shot_examples_with_images(
                qwen_samples,
                qwen_data_ctx,
                options_mode,
                model_stem,
                options_order=options_order,
            )
        elif model_stem in MEDIX_MODELS:
            med_samples, med_image_base = _select_gemma_few_shot_samples(
                dataset,
                train_data,
                sample,
                n_shots,
                seed,
                shots_root,
            )
            med_data_ctx = VQADataset(
                samples=med_samples,
                split=train_data.split,
                dataset_name=train_data.dataset_name,
                image_base_dir=med_image_base or train_data.image_base_dir,
            )
            examples = _build_medix_train_few_shot_examples_with_images(
                med_samples,
                med_data_ctx,
                options_mode,
                model_stem,
                options_order=options_order,
            )
        elif model_stem in MEDVLTHINKER_MODELS:
            med_samples, med_image_base = _select_gemma_few_shot_samples(
                dataset,
                train_data,
                sample,
                n_shots,
                seed,
                shots_root,
            )
            med_data_ctx = VQADataset(
                samples=med_samples,
                split=train_data.split,
                dataset_name=train_data.dataset_name,
                image_base_dir=med_image_base or train_data.image_base_dir,
            )
            examples = _build_medvlthinker_train_few_shot_examples_with_images(
                med_samples,
                med_data_ctx,
                options_mode,
                model_stem,
                options_order=options_order,
            )
        elif model_stem in MEDMO_MODELS:
            med_samples, med_image_base = _select_gemma_few_shot_samples(
                dataset,
                train_data,
                sample,
                n_shots,
                seed,
                shots_root,
            )
            med_data_ctx = VQADataset(
                samples=med_samples,
                split=train_data.split,
                dataset_name=train_data.dataset_name,
                image_base_dir=med_image_base or train_data.image_base_dir,
            )
            examples = _build_qwen_train_few_shot_examples_with_images(
                med_samples,
                med_data_ctx,
                options_mode,
                model_stem,
                options_order=options_order,
            )
        else:
            examples = [
                build_few_shot_example(
                    s,
                    options_mode,
                    model_stem=model_stem,
                    options_order=options_order,
                )
                for s in selected_samples
            ]
    if use_simple_gemma_few_shot:
        user_text = _build_simple_gemma_few_shot_user_text(
            sample,
            options_mode,
            options_order=options_order,
        )
        prompt_options = None
    elif _uses_simple_gemma_prompt(model_stem):
        user_text = _build_simple_gemma_prompt_text(
            sample,
            options_mode,
            options_order=options_order,
        )
        prompt_options = None
    elif model_stem in LLAVA_MODELS:
        user_text = _build_llava_user_text(
            sample, reasoning, options_mode, options_order=options_order
        )
    elif model_stem in FLAMINGO_MODELS:
        user_text = _build_flamingo_user_text(
            sample, reasoning, options_mode, options_order=options_order
        )
    elif model_stem in MEDVLTHINKER_MODELS:
        user_text = _build_medvlthinker_prompt_text(
            sample, options_mode, options_order=options_order
        )
        prompt_options = None
    elif model_stem in MEDMO_MODELS:
        user_text = _build_medmo_prompt_text(
            sample, options_mode, options_order=options_order
        )
        prompt_options = None
    elif model_stem in MEDIX_MODELS:
        user_text, prompt_options = _build_medix_prompt_parts(
            sample, options_mode, options_order=options_order
        )
    elif model_stem in QWEN2_VL_MODELS:
        user_text = _build_qwen2_vl_prompt_text(
            sample, options_mode, options_order=options_order
        )
        prompt_options = None
    elif model_stem in ADAPT_QWEN_VL_MODELS:
        user_text = _build_adapt_qwen_prompt_text(
            sample, options_mode, options_order=options_order
        )
        prompt_options = None
    elif model_stem in ADAPT_INTERNVL_MODELS:
        user_text = _build_adapt_internvl_prompt_text(
            sample, options_mode, options_order=options_order
        )
        prompt_options = None
    elif model_stem in INTERNVL_MODELS:
        user_text = _build_internvl_prompt_text(
            sample, options_mode, options_order=options_order
        )
        prompt_options = None
    elif model_stem in ADAPT_LLAMA_MODELS:
        user_text = _build_adapt_llama_prompt_text(
            sample, options_mode, options_order=options_order
        )
        prompt_options = None
    elif model_stem in META_LLAMA_VISION_MODELS:
        user_text = _build_meta_llama_prompt_text(
            sample, options_mode, options_order=options_order
        )
        prompt_options = None
    elif model_stem in QWEN_VL_MODELS:
        user_text = _build_qwen25_vl_prompt_text(
            sample, options_mode, options_order=options_order
        )
        prompt_options = None
    else:
        question_text = f"Question: {sample.question}"
        options_text = _build_options_text(
            sample, options_mode, options_order=options_order
        )
        reasoning_text = REASONING_TEMPLATES[reasoning]
        output_format = OUTPUT_FORMATS[options_mode]
        parts = [question_text]
        if options_text:
            parts.append(options_text)
        parts.append(reasoning_text)
        parts.append(output_format)
        user_text = "\n\n".join(parts)
        prompt_options = None
    image_path = None
    if model_stem in VISION_MODELS:
        image_path = os.path.join(dataset.image_base_dir, sample.image_path)
        if not os.path.exists(image_path):
            image_path = None
    PromptClass = get_prompt_class(model_stem)
    query_kwargs = {
        "user_text": user_text,
        "system_prompt": None,
        "image_path": image_path,
    }
    if model_stem in MEDIX_MODELS:
        query_kwargs["options"] = prompt_options
    query_prompt = PromptClass(**query_kwargs)
    return system_prompt, examples, query_prompt

def get_output_path(
    model_stem: str,
    dataset_name: str,
    approach: str,
    reasoning: str,
    options_mode: str,
    n_shots: int = 0,
    options_order: str = "default",
    task: str = "prompting",
    base_dir: str = "./data/sample_generations",
) -> str:
    filename = f"{approach}_{reasoning}_{options_mode}"
    if options_order != "default":
        filename += f"_{options_order}"
    if n_shots > 0:
        filename += f"_{n_shots}shots"
    filename += ".jsonl"
    path = os.path.join(base_dir, model_stem, dataset_name, task, filename)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path

def get_embedding_output_dir(
    model_stem: str,
    dataset_name: str,
    approach: str,
    reasoning: str,
    options_mode: str,
    n_shots: int = 0,
    options_order: str = "default",
    base_dir: str = "./data/sample_features",
) -> str:
    dirname = f"{approach}_{reasoning}_{options_mode}"
    if options_order != "default":
        dirname += f"_{options_order}"
    if n_shots > 0:
        dirname += f"_{n_shots}shots"
    path = os.path.join(base_dir, model_stem, dataset_name, dirname)
    os.makedirs(path, exist_ok=True)
    return path
