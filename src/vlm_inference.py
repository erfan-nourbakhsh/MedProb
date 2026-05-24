import argparse
import logging
import os
import sys

from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.constants import (
    ALL_MODELS,
    APPROACHES,
    REASONING_STRATEGIES,
    OPTION_MODES,
    OPTION_ORDER_MODES,
)
from utils.loaders import load_dataset
from utils.prompters import get_prompter
from utils.prompt_builders import (
    build_prompt,
    build_few_shot_conversation,
    get_output_path,
)
from utils.evaluation_helpers import evaluate_single
from utils.file_io import append_jsonl

BATCHED_QWEN_MODELS = {
    "medix-r1-2b",
    "medmo-4b-next",
    "medmo-8b-next",
    "medix-r1-30b",
    "qwen25-vl-3b-instruct",
    "qwen25-vl-7b-instruct",
    "qwen25-vl-7b-instruct-full-path-vqa",
    "qwen25-vl-7b-instruct-full-all-med-vqa",
    "qwen25-vl-7b-instruct-full-slake",
    "qwen25-vl-7b-instruct-full-vqa-rad",
    "qwen25-vl-32b-instruct",
    "qwen2-vl-2b-instruct",
    "qwen3-vl-2b-instruct",
    "qwen3-vl-4b-instruct",
    "qwen3-vl-8b-instruct",
    "qwen3-vl-30b-a3b-instruct",
    "adapt-qwen2-2b",
}
TOP_P_MODELS = BATCHED_QWEN_MODELS

def parse_optional_float(value: str) -> float | None:
    if value.lower() in {"none", "null"}:
        return None
    return float(value)

def parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "t", "yes", "y"}:
        return True
    if normalized in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")

def parse_args():
    p = argparse.ArgumentParser(description="Medical LVLM Prompting Pipeline")
    p.add_argument("--dataset", type=str, default="PATH-VQA", help="Dataset name")
    p.add_argument(
        "--dataset_dir",
        type=str,
        default="./samples/PATH-VQA",
        help="Path to dataset directory with train.json, test.json, images/",
    )
    p.add_argument(
        "--model", type=str, required=True, choices=ALL_MODELS, help="Model stem to use"
    )
    p.add_argument(
        "--approach",
        type=str,
        required=True,
        choices=APPROACHES,
        help="Prompting approach",
    )
    p.add_argument(
        "--reasoning",
        type=str,
        required=True,
        choices=REASONING_STRATEGIES,
        help="Reasoning strategy",
    )
    p.add_argument(
        "--options_mode",
        type=str,
        required=True,
        choices=OPTION_MODES,
        help="Option presentation mode",
    )
    p.add_argument(
        "--options_order",
        type=str,
        default="default",
        choices=OPTION_ORDER_MODES,
        help="Option ordering experiment for prompting/evaluation",
    )
    p.add_argument(
        "--n_shots",
        type=int,
        default=0,
        help="Number of few-shot examples (0 = zero-shot)",
    )
    p.add_argument("--batch_size", type=int, default=1, help="Batch size for inference")
    p.add_argument(
        "--max_new_tokens", type=int, default=200, help="Maximum new tokens to generate"
    )
    p.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Generation temperature (0.0 = greedy decoding)",
    )
    p.add_argument(
        "--top_p",
        type=parse_optional_float,
        default=None,
        help="Top-p sampling threshold; use 'null' to disable",
    )
    p.add_argument("--num_beams", type=int, default=1, help="Beam count for generation")
    p.add_argument(
        "--do_sample", type=parse_bool, default=False, help="Whether to enable sampling"
    )
    p.add_argument(
        "--specialist_dir",
        type=str,
        default="./data/specialist_prompts",
        help="Directory with pre-generated specialist prompts",
    )
    p.add_argument(
        "--use_gpt_judge",
        type=int,
        default=1,
        choices=[0, 1],
        help="Use GPT-4O as fallback judge for evaluation",
    )
    p.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "test"],
        help="Which split to evaluate on",
    )
    return p.parse_args()

def main():
    args = parse_args()
    log_dir = "./logs"
    os.makedirs(log_dir, exist_ok=True)
    log_suffix = f"{args.model}_{args.approach}_{args.reasoning}_{args.options_mode}_{args.options_order}"
    if args.n_shots > 0:
        log_suffix += f"_{args.n_shots}shots"
    log_file = os.path.join(log_dir, f"prompting_{log_suffix}.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(),
        ],
    )
    logger = logging.getLogger(__name__)
    logger.info("=" * 70)
    logger.info("MEDICAL LVLM PROMPTING PIPELINE")
    logger.info("=" * 70)
    logger.info(f"Dataset:      {args.dataset}")
    logger.info(f"Model:        {args.model}")
    logger.info(f"Approach:     {args.approach}")
    logger.info(f"Reasoning:    {args.reasoning}")
    logger.info(f"Options Mode: {args.options_mode}")
    logger.info(f"Options Order:{args.options_order}")
    logger.info(f"N Shots:      {args.n_shots}")
    logger.info(f"Split:        {args.split}")
    logger.info(f"Batch Size:   {args.batch_size}")
    logger.info(f"Max Tokens:   {args.max_new_tokens}")
    logger.info(f"Temperature:  {args.temperature}")
    if args.model in TOP_P_MODELS:
        logger.info(f"Top P:        {args.top_p}")
    if args.model in {
        "med-flamingo-9b",
        "random_med-flamingo-9b_1",
        "random_med-flamingo-9b_2",
        "random_med-flamingo-9b_3",
        "open-flamingo-9b",
        "random_open-flamingo-9b_1",
        "random_open-flamingo-9b_2",
        "random_open-flamingo-9b_3",
    }:
        logger.info(f"Num Beams:    {args.num_beams}")
        logger.info(f"Do Sample:    {args.do_sample}")
    logger.info("=" * 70)
    logger.info("Loading dataset...")
    train_data, test_data = load_dataset(args.dataset_dir, args.dataset)
    eval_data = test_data if args.split == "test" else train_data
    logger.info(f"Loaded {len(eval_data)} samples from {args.split} split")
    if args.n_shots > 0:
        logger.info(f"Few-shot source: {len(train_data)} training samples")
    logger.info(f"Loading model: {args.model}...")
    prompter_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
    }
    if args.model in TOP_P_MODELS:
        prompter_kwargs.update(top_p=args.top_p)
    if args.model in {
        "med-flamingo-9b",
        "random_med-flamingo-9b_1",
        "random_med-flamingo-9b_2",
        "random_med-flamingo-9b_3",
        "open-flamingo-9b",
        "random_open-flamingo-9b_1",
        "random_open-flamingo-9b_2",
        "random_open-flamingo-9b_3",
    }:
        prompter_kwargs.update(
            num_beams=args.num_beams,
            do_sample=args.do_sample,
        )
    prompter = get_prompter(args.model, **prompter_kwargs)
    logger.info("Model loaded successfully")
    out_file = get_output_path(
        model_stem=args.model,
        dataset_name=args.dataset,
        approach=args.approach,
        reasoning=args.reasoning,
        options_mode=args.options_mode,
        options_order=args.options_order,
        n_shots=args.n_shots,
        task="prompting",
    )
    logger.info(f"Output file: {out_file}")
    if os.path.exists(out_file):
        os.remove(out_file)
    use_few_shot = args.n_shots > 0
    correct_count = 0
    total_count = 0
    desc = f"Prompting | {args.model} | {args.approach} | {args.reasoning} | {args.options_mode} | {args.options_order}"
    if use_few_shot:
        desc += f" | {args.n_shots}-shot"
    for start_idx in tqdm(
        range(0, len(eval_data), args.batch_size),
        desc=desc,
    ):
        batch_samples = eval_data.samples[start_idx : start_idx + args.batch_size]
        batch_outputs = None
        if (
            not use_few_shot
            and args.model in BATCHED_QWEN_MODELS
            and len(batch_samples) > 1
        ):
            try:
                batch_prompts = [
                    build_prompt(
                        sample=sample,
                        dataset=eval_data,
                        model_stem=args.model,
                        approach=args.approach,
                        reasoning=args.reasoning,
                        options_mode=args.options_mode,
                        options_order=args.options_order,
                        specialist_dir=args.specialist_dir,
                    )
                    for sample in batch_samples
                ]
                batch_outputs = prompter.get_completion_batch(batch_prompts)
            except Exception as e:
                logger.warning(
                    f"Batch generation failed for samples {start_idx}-{start_idx + len(batch_samples) - 1}: {e}. "
                    "Falling back to per-sample generation."
                )
                batch_outputs = None
        for sample_idx, sample in enumerate(batch_samples):
            try:
                if batch_outputs is not None:
                    model_output = batch_outputs[sample_idx]
                elif use_few_shot:
                    system_prompt, examples, query_prompt = build_few_shot_conversation(
                        sample=sample,
                        dataset=eval_data,
                        train_data=train_data,
                        model_stem=args.model,
                        approach=args.approach,
                        reasoning=args.reasoning,
                        options_mode=args.options_mode,
                        n_shots=args.n_shots,
                        options_order=args.options_order,
                        specialist_dir=args.specialist_dir,
                    )
                    model_output = prompter.get_completion_conversation(
                        system_prompt=system_prompt,
                        few_shot_examples=examples,
                        query_prompt=query_prompt,
                    )
                else:
                    prompt = build_prompt(
                        sample=sample,
                        dataset=eval_data,
                        model_stem=args.model,
                        approach=args.approach,
                        reasoning=args.reasoning,
                        options_mode=args.options_mode,
                        options_order=args.options_order,
                        specialist_dir=args.specialist_dir,
                    )
                    model_output = prompter.get_completion(prompt)
            except Exception as e:
                logger.exception(f"Error on sample {sample.idx}")
                model_output = f"ERROR: {type(e).__name__}: {e!r}"
            eval_result = evaluate_single(
                model_output=model_output,
                sample=sample,
                options_mode=args.options_mode,
                options_order=args.options_order,
                use_gpt_judge=bool(args.use_gpt_judge),
                model_stem=args.model,
            )
            total_count += 1
            correct_count += eval_result.get("correct", 0)
            running_acc = correct_count / max(total_count, 1)
            append_jsonl(out_file, eval_result)
        if total_count % 10 == 0:
            logger.info(
                f"Progress: {total_count}/{len(eval_data)} | "
                f"Running Acc: {running_acc:.4f} ({running_acc:.2%})"
            )
    final_acc = correct_count / max(total_count, 1)
    logger.info("=" * 70)
    logger.info(f"FINISHED: {total_count} samples processed")
    logger.info(f"Accuracy: {final_acc:.4f} ({final_acc:.2%})")
    logger.info(f"Results saved to: {out_file}")
    logger.info("=" * 70)

if __name__ == "__main__":
    main()
