import argparse
import logging
import os
import sys
from typing import Optional

import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.constants import (
    ALL_MODELS,
    APPROACHES,
    REASONING_STRATEGIES,
    OPTION_MODES,
    OPTION_ORDER_MODES,
    EXTRACTION_POSITIONS,
)
from utils.loaders import load_dataset
from utils.prompters import (
    AdaptInternVL3Prompter,
    AdaptLlamaPrompter,
    AdaptQwen2VLPrompter,
    FlamingoPrompter,
    LlavaMedPrompter,
    MetaLlama32VisionPrompter,
    get_prompter,
)
from utils.prompt_builders import (
    build_prompt,
    build_few_shot_conversation,
    get_embedding_output_dir,
)
_FEW_SHOT_EMBEDDING_PROMPTERS = (
    LlavaMedPrompter,
    FlamingoPrompter,
    AdaptLlamaPrompter,
    AdaptQwen2VLPrompter,
    AdaptInternVL3Prompter,
    MetaLlama32VisionPrompter,
)
from utils.file_io import save_embeddings

def _extract_layer_embeddings(
    prompter, prompt, extraction_position: str, convo_few_shot: Optional[list]
):
    use_fs = convo_few_shot is not None and isinstance(
        prompter, _FEW_SHOT_EMBEDDING_PROMPTERS
    )
    fs_kw = {"few_shot_examples": convo_few_shot} if use_fs else {}
    if extraction_position == "first_generated_token":
        return prompter.get_first_generated_token_embeddings(prompt, **fs_kw)
    if extraction_position == "mean_all_tokens":
        return prompter.get_mean_all_tokens_embeddings(prompt, **fs_kw)
    if extraction_position == "mean_image_tokens":
        return prompter.get_mean_image_tokens_embeddings(prompt, **fs_kw)
    if extraction_position == "mean_text_tokens":
        return prompter.get_mean_text_tokens_embeddings(prompt, **fs_kw)
    if extraction_position == "concat_img_text_last":
        return prompter.get_concat_img_text_last_embeddings(prompt, **fs_kw)
    return prompter.get_all_layer_embeddings(prompt, **fs_kw)

def parse_args():
    p = argparse.ArgumentParser(
        description="Medical LVLM Embedding Extraction Pipeline"
    )
    p.add_argument("--dataset", type=str, default="PATH-VQA", help="Dataset name")
    p.add_argument(
        "--dataset_dir",
        type=str,
        default="./samples/PATH-VQA",
        help="Path to dataset directory",
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
        help="Option ordering mode",
    )
    p.add_argument(
        "--n_shots",
        type=int,
        default=0,
        help="Number of few-shot examples (0 = zero-shot)",
    )
    p.add_argument(
        "--batch_size", type=int, default=1, help="Batch size for embedding extraction"
    )
    p.add_argument(
        "--extraction_position",
        type=str,
        default="last_input_token",
        choices=EXTRACTION_POSITIONS,
        help="Token position for embedding extraction",
    )
    p.add_argument(
        "--specialist_dir",
        type=str,
        default="./data/specialist_prompts",
        help="Directory with specialist prompts",
    )
    return p.parse_args()

def main():
    args = parse_args()
    log_dir = "./logs"
    os.makedirs(log_dir, exist_ok=True)
    log_suffix = f"{args.model}_{args.approach}_{args.reasoning}_{args.options_mode}_{args.options_order}"
    if args.n_shots > 0:
        log_suffix += f"_{args.n_shots}shots"
    log_file = os.path.join(log_dir, f"embedding_{log_suffix}.log")
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
    logger.info("MEDICAL LVLM EMBEDDING EXTRACTION")
    logger.info("=" * 70)
    logger.info(f"Dataset:      {args.dataset}")
    logger.info(f"Model:        {args.model}")
    logger.info(f"Approach:     {args.approach}")
    logger.info(f"Reasoning:    {args.reasoning}")
    logger.info(f"Options Mode: {args.options_mode}")
    logger.info(f"Options Order:{args.options_order}")
    logger.info(f"N Shots:      {args.n_shots}")
    logger.info(f"Position:     {args.extraction_position}")
    logger.info("=" * 70)
    logger.info("Loading dataset...")
    train_data, test_data = load_dataset(args.dataset_dir, args.dataset)
    logger.info(f"Train: {len(train_data)} samples, Test: {len(test_data)} samples")
    logger.info(f"Loading model: {args.model}...")
    prompter = get_prompter(args.model)
    logger.info("Model loaded successfully")
    position_suffix = (
        ""
        if args.extraction_position == "last_input_token"
        else f"_{args.extraction_position}"
    )
    out_dir = get_embedding_output_dir(
        model_stem=args.model,
        dataset_name=args.dataset,
        approach=args.approach,
        reasoning=args.reasoning,
        options_mode=args.options_mode,
        n_shots=args.n_shots,
        options_order=args.options_order,
    )
    if position_suffix:
        out_dir = out_dir + position_suffix
    logger.info(f"Output dir: {out_dir}")
    use_few_shot = args.n_shots > 0
    embeddings = {"train": {}, "test": {}}
    trim_incomplete_meta_layers = args.model.startswith(
        "meta-llama3.2-11b-vision-instruct"
    )
    for split_name, split_data in [("train", train_data), ("test", test_data)]:
        batch_embeds: dict[str, list[np.ndarray]] = {}
        successful_samples = 0
        num_batches = (len(split_data) + args.batch_size - 1) // args.batch_size
        for start_idx in tqdm(
            range(0, len(split_data), args.batch_size),
            desc=f"Embedding {split_name} | {num_batches} batches | {len(split_data)} samples",
        ):
            batch_samples = split_data.samples[start_idx : start_idx + args.batch_size]
            for sample in batch_samples:
                examples = None
                if use_few_shot:
                    system_prompt, examples, query_prompt = build_few_shot_conversation(
                        sample=sample,
                        dataset=split_data,
                        train_data=train_data,
                        model_stem=args.model,
                        approach=args.approach,
                        reasoning=args.reasoning,
                        options_mode=args.options_mode,
                        options_order=args.options_order,
                        n_shots=args.n_shots,
                        specialist_dir=args.specialist_dir,
                    )
                    if system_prompt:
                        query_prompt.system_prompt = system_prompt
                    prompt = query_prompt
                else:
                    prompt = build_prompt(
                        sample=sample,
                        dataset=split_data,
                        model_stem=args.model,
                        approach=args.approach,
                        reasoning=args.reasoning,
                        options_mode=args.options_mode,
                        options_order=args.options_order,
                        specialist_dir=args.specialist_dir,
                    )
                convo_few_shot = (
                    examples
                    if isinstance(prompter, _FEW_SHOT_EMBEDDING_PROMPTERS)
                    else None
                )
                try:
                    all_layer_embs = _extract_layer_embeddings(
                        prompter, prompt, args.extraction_position, convo_few_shot
                    )
                    successful_samples += 1
                    for layer_key, emb_tensor in all_layer_embs.items():
                        if layer_key not in batch_embeds:
                            batch_embeds[layer_key] = []
                        batch_embeds[layer_key].append(
                            emb_tensor.detach().cpu().numpy()
                        )
                except Exception as e:
                    logger.exception(f"Error on {split_name} sample {sample.idx}")
                    continue
        if trim_incomplete_meta_layers and batch_embeds:
            complete_keys = [
                layer_key
                for layer_key, emb_list in batch_embeds.items()
                if len(emb_list) == successful_samples
            ]
            dropped_keys = sorted(
                set(batch_embeds) - set(complete_keys),
                key=lambda k: (0, int(k)) if k.lstrip("-").isdigit() else (1, k),
            )
            if dropped_keys:
                logger.warning(
                    "Dropping incomplete Meta-Llama layers for %s: %s",
                    split_name,
                    dropped_keys,
                )
            batch_embeds = {
                layer_key: batch_embeds[layer_key] for layer_key in complete_keys
            }
        for layer_key, emb_list in batch_embeds.items():
            if emb_list:
                embeddings[split_name][layer_key] = np.stack(emb_list, axis=0)
            else:
                embeddings[split_name][layer_key] = np.zeros((0, 0), dtype=float)
        num_layers = len(embeddings[split_name])
        if num_layers > 0:
            sample_shape = next(iter(embeddings[split_name].values())).shape
        else:
            sample_shape = (0, 0)
        logger.info(f"Finished {split_name}: {num_layers} layers, shape={sample_shape}")
    save_embeddings(
        train_embed=embeddings["train"],
        test_embed=embeddings["test"],
        save_path=out_dir,
    )
    logger.info(
        f"Embeddings saved to {out_dir}/ " f"({len(embeddings['train'])} layers)"
    )

if __name__ == "__main__":
    main()
