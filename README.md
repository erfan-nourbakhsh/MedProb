# MedProb: Probing Internal Representations of Vision-Language Models for Medical Question Answering

---

## Supported Datasets

| Dataset | Type | Download |
|---------|------|----------|
| PATH-VQA | Pathology, binary + open | [HuggingFace](https://huggingface.co/datasets/flaviagiammarino/path-vqa) |
| VQA-RAD | Radiology, binary + open | [OSF](https://osf.io/89kps/overview) |
| SLAKE | Radiology, binary + open | [med-vqa.com](https://www.med-vqa.com/slake/) |

---

## Supported Models

### General VLMs

| Model | Project Key | HuggingFace ID |
|-------|-------------|----------------|
| Gemma 4B | `gemma` | `google/gemma-3-4b-it` |
| Gemma 27B | `gemma-27b` | `google/gemma-3-27b-it` |
| LLaVA-V0-7B | `llava-v0-7b` | local checkpoint |
| OpenFlamingo-9B | `open-flamingo-9b` | `openflamingo/OpenFlamingo-9B-deprecated` |
| InternVL3-1B | `internvl3-1b` | `OpenGVLab/InternVL3-1B` |
| Llama-3.2-11B-Vision | `meta-llama3.2-11b-vision-instruct` | `meta-llama/Llama-3.2-11B-Vision-Instruct` |
| Qwen2-VL-2B | `qwen2-vl-2b-instruct` | `Qwen/Qwen2-VL-2B-Instruct` |
| Qwen2.5-VL-3B | `qwen25-vl-3b-instruct` | `Qwen/Qwen2.5-VL-3B-Instruct` |
| Qwen2.5-VL-7B | `qwen25-vl-7b-instruct` | `Qwen/Qwen2.5-VL-7B-Instruct` |
| Qwen2.5-VL-32B | `qwen25-vl-32b-instruct` | `Qwen/Qwen2.5-VL-32B-Instruct` |
| Qwen3-VL-2B | `qwen3-vl-2b-instruct` | `Qwen/Qwen3-VL-2B-Instruct` |
| Qwen3-VL-4B | `qwen3-vl-4b-instruct` | `Qwen/Qwen3-VL-4B-Instruct` |
| Qwen3-VL-8B | `qwen3-vl-8b-instruct` | `Qwen/Qwen3-VL-8B-Instruct` |
| Qwen3-VL-30B | `qwen3-vl-30b-a3b-instruct` | `Qwen/Qwen3-VL-30B-A3B-Instruct` |

### Biomedical VLMs

| Model | Project Key | HuggingFace ID |
|-------|-------------|----------------|
| MedGemma-4B | `medgemma` | `google/medgemma-4b-it` |
| MedGemma-27B | `medgemma-27b` | `google/medgemma-27b-it` |
| LLaVA-Med-7B | `llava-med-7b` | local checkpoint |
| Med-Flamingo-9B | `med-flamingo-9b` | `med-flamingo/med-flamingo` |
| BioMed-Llama-3.2-11B | `adapt-llama3.2-11b` | `AdaptLLM/biomed-Llama-3.2-11B-Vision-Instruct` |
| BioMed-Qwen2-VL-2B | `adapt-qwen2-2b` | `AdaptLLM/biomed-Qwen2-VL-2B-Instruct` |
| BioMed-InternVL3-1B | `adapt-internVL3-1b` | `AdaptLLM/biomed-InternVL3-1B` |
| MedVLThinker-RL-3B | `medvlthinker-3b` | `UCSC-VLAA/MedVLThinker-3B-RL_m23k` |
| MedVLThinker-RL-7B | `medvlthinker-7b` | `UCSC-VLAA/MedVLThinker-7B-RL_m23k` |
| MedVLThinker-RL-32B | `medvlthinker-32b` | `UCSC-VLAA/MedVLThinker-32B-RL_m23k` |
| MedMO-4B-Next | `medmo-4b-next` | `MBZUAI/MedMO-4B-Next` |
| MedMO-8B-Next | `medmo-8b-next` | `MBZUAI/MedMO-8B-Next` |
| MediX-R1-2B | `medix-r1-2b` | `MBZUAI/MediX-R1-2B` |
| MediX-R1-30B | `medix-r1-30b` | `MBZUAI/MediX-R1-30B` |

### Fine-tuned Checkpoints

Qwen2.5-VL-7B and LLaMA3.2-11B-Vision fine-tuned on PATH-VQA / VQA-RAD / SLAKE / ALL\_MED\_VQA via `fine-tune/`.

| Fine-tuned Model | Project Key |
|-----------------|-------------|
| Qwen2.5-VL-7B × PATH-VQA | `qwen25-vl-7b-instruct-full-path-vqa` |
| Qwen2.5-VL-7B × VQA-RAD | `qwen25-vl-7b-instruct-full-vqa-rad` |
| Qwen2.5-VL-7B × SLAKE | `qwen25-vl-7b-instruct-full-slake` |
| Qwen2.5-VL-7B × ALL | `qwen25-vl-7b-instruct-full-all-med-vqa` |
| Llama-3.2-11B-Vision × PATH-VQA | `meta-llama3.2-11b-vision-instruct-full-path-vqa` |
| Llama-3.2-11B-Vision × VQA-RAD | `meta-llama3.2-11b-vision-instruct-full-vqa-rad` |
| Llama-3.2-11B-Vision × SLAKE | `meta-llama3.2-11b-vision-instruct-full-slake` |
| Llama-3.2-11B-Vision × ALL | `meta-llama3.2-11b-vision-instruct-full-all-med-vqa` |

---

## Repository Structure

```
MedProb/
├── src/                               # Core Python source modules
│   ├── vlm_inference.py               # VLM inference for MCQ experiments
│   ├── extract_embeddings.py          # Per-layer hidden-state extraction
│   ├── evaluate.py                    # Accuracy, F1, optional GPT-judge evaluation
│   ├── linear_probe.py                # Linear probe training + evaluation (main)
│   ├── linear_probe_cross_dataset.py  # Cross-dataset probe generalization
│   ├── linear_probe_default_train.py  # Default-split probe training
│   ├── classify_modality.py           # Modality classification via GPT
│   ├── format_audit.py                # Output format compliance audit
│   ├── error_analysis.py              # Error distribution analysis
│   └── utils/
│       ├── prompt_builders.py         # Prompt construction (all models)
│       ├── prompters.py               # Model-specific prompter classes
│       ├── prompt_objects.py          # Prompt data structures
│       ├── loaders.py                 # Dataset + model loaders
│       ├── evaluation_helpers.py      # Scoring helpers
│       ├── file_io.py                 # JSONL / NPZ I/O
│       ├── constants.py               # Model + config constants
│       ├── answer_frequency.py        # Answer frequency utilities
│       └── preprocessing/             # Dataset-specific preprocessing
│           ├── path_vqa.py
│           ├── slake.py
│           └── vqa_rad.py
│
├── bin/                               # Shell runner scripts
│   ├── data_downloader/               # Download + preprocess each dataset
│   ├── generation/                    # Run inference and probe training
│   │   ├── run_inference.sh           # Main VLM inference runner (all combos)
│   │   ├── run_inference_v2.sh        # Inference v2 runner
│   │   ├── run_linear_probe.sh        # Train linear probe
│   │   ├── run_linear_probe_cross_dataset.sh
│   │   └── run_linear_probe_default_train.sh
│   ├── features/                      # Feature extraction runners
│   │   ├── run_extract_embeddings.sh  # Hidden-state extraction
│   │   └── classify_modality.sh       # Modality classification
│   ├── eval/                          # Evaluation runners
│   │   ├── eval_inference.sh          # Evaluate VLM inference outputs
│   │   └── eval_linear_probe.sh       # Evaluate linear probe outputs
│   ├── new_evals/                     # Cross-dataset + default-split evaluations
│   │   ├── eval_linear_probe_cross_dataset.sh
│   │   └── eval_linear_probe_default_train.sh
│   └── sample_generator/              # Export preprocessed samples
│
├── fine-tune/                         # Fine-tuning pipeline
│   ├── qwen25_vl_finetune.py          # Qwen2.5-VL-7B trainer
│   ├── llama32_vision_finetune.py     # LLaMA3.2-11B-Vision trainer
│   ├── run_qwen25_vl.sh               # Single-dataset fine-tune (Qwen)
│   ├── run_llama32_vision.sh          # Single-dataset fine-tune (LLaMA)
│   ├── run_all_qwen25_vl.sh           # All datasets (Qwen)
│   ├── run_qwen25_vl_bg.sh            # Background launcher with logs
│   └── README.md
│
└── requirements.txt
```

---

## Installation

```bash
git clone https://github.com/MedProb/MedProb.git
cd MedProb

python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Set environment variables:

```bash
export HF_TOKEN=hf_...          # HuggingFace token (for gated models)
export OPENAI_API_KEY=sk-...    # Optional: GPT-judge fallback in eval
```

---

## Dataset Setup

Download the raw data from the official sources, then run the project download/preprocess scripts:

| Dataset | Source |
|---------|--------|
| PATH-VQA | [HuggingFace — flaviagiammarino/path-vqa](https://huggingface.co/datasets/flaviagiammarino/path-vqa) |
| VQA-RAD | [OSF — VQA-RAD](https://osf.io/89kps/overview) |
| SLAKE | [med-vqa.com — SLAKE](https://www.med-vqa.com/slake/) |

```bash
bash bin/data_downloader/download_VQA-RAD.sh
bash bin/data_downloader/download_SLAKE.sh
# PATH-VQA: download from HuggingFace (link above), then run src/utils/preprocessing/path_vqa.py
```

Each script downloads the raw dataset, normalizes it to the project schema, and writes:

```
samples/<DATASET>/
  train.json
  test.json
  images/
```

Export split-specific sample sets:

```bash
bash bin/sample_generator/sampler.sh
```

---

## Pipeline: Step-by-Step

### Step 1 — VLM Inference

Run VLM inference for all experiment combinations (model × dataset × approach × reasoning × option mode × shot count):

```bash
# Edit DATASETS, MODELS, etc. at the top, then:
./bin/generation/run_inference.sh

# Or background with logs:
nohup ./bin/generation/run_inference.sh > ./logs/inference_master.log 2>&1 &
```

Key configuration options inside the script:
- `DATASETS`: `PATH-VQA`, `VQA-RAD`, `SLAKE`
- `APPROACHES`: `image_question`
- `REASONING`: `direct`
- `OPTIONS_MODES`: `with_options`
- `NSHOTS`: `0`, `3`

Output: `data/sample_generations/<model>/<dataset>/prompting/<config>.jsonl`

Direct call:

```bash
python src/vlm_inference.py \
  --dataset PATH-VQA \
  --dataset_dir ./samples/PATH-VQA \
  --model medgemma \
  --approach image_question \
  --reasoning direct \
  --options_mode with_options \
  --n_shots 0
```

### Step 2 — Evaluate Inference

Evaluate inference outputs (accuracy, F1, optional GPT-judge):

```bash
./bin/eval/eval_inference.sh
```

Direct call:

```bash
python src/evaluate.py \
  --pred_path data/sample_generations/medgemma/PATH-VQA/prompting/image_question_direct_with_options.jsonl \
  --dataset_dir ./samples/PATH-VQA
```

### Step 3 — Embedding Extraction

Extract per-layer hidden states for linear probing:

```bash
./bin/features/run_extract_embeddings.sh
```

Direct call:

```bash
python src/extract_embeddings.py \
  --dataset PATH-VQA \
  --dataset_dir ./samples/PATH-VQA \
  --model medgemma \
  --approach image_question \
  --reasoning direct \
  --options_mode with_options \
  --extraction_position last_input_token
```

Output: `data/sample_features/<model>/<dataset>/<config>/test_embeddings.npz`

### Step 4 — Linear Probing

Train and evaluate linear probes on extracted embeddings:

```bash
./bin/generation/run_linear_probe.sh
./bin/eval/eval_linear_probe.sh
```

Cross-dataset generalization:

```bash
./bin/generation/run_linear_probe_cross_dataset.sh
./bin/new_evals/eval_linear_probe_cross_dataset.sh
```

Default-split probing:

```bash
./bin/generation/run_linear_probe_default_train.sh
./bin/new_evals/eval_linear_probe_default_train.sh
```

---

## Fine-Tuning

Full fine-tuning (default) or LoRA / QLoRA for Qwen2.5-VL-7B-Instruct and LLaMA3.2-11B-Vision-Instruct on medical VQA datasets.

```bash
# Qwen2.5-VL-7B
bash fine-tune/run_qwen25_vl.sh PATH-VQA
bash fine-tune/run_qwen25_vl.sh VQA-RAD
bash fine-tune/run_qwen25_vl.sh SLAKE
bash fine-tune/run_all_qwen25_vl.sh        # all three datasets

# LLaMA3.2-11B-Vision
bash fine-tune/run_llama32_vision.sh PATH-VQA

# Background with logs
bash fine-tune/run_qwen25_vl_bg.sh PATH-VQA
```

Override training parameters:

```bash
GPU_ID=1 METHOD=qlora EPOCHS=2.0 bash fine-tune/run_qwen25_vl.sh VQA-RAD
```

Key parameters: `GPU_ID`, `METHOD` (`full` / `lora` / `qlora`), `EPOCHS`, `MAX_LENGTH`, `LEARNING_RATE`, `GRAD_ACCUM_STEPS`.

The training prompt format imports directly from `src/utils/prompt_builders.py` to ensure exact consistency with inference experiments.

Output: `fine-tune/results/<DATASET>/<model>-<method>/`

---

## Recommended Run Order

```
1. bin/data_downloader/              →  download + preprocess datasets
2. bin/sample_generator/             →  export samples/
3. bin/generation/run_inference.sh   →  run VLM inference
4. bin/eval/eval_inference.sh        →  evaluate inference accuracy
5. bin/features/run_extract_embeddings.sh  →  extract hidden states
6. bin/generation/run_linear_probe.sh      →  train linear probes
7. bin/eval/eval_linear_probe.sh           →  evaluate probes
```

---

## Output Artifact Layout

```
data/
├── sample_generations/
│   └── <model>/<dataset>/prompting/<config>.jsonl
└── sample_features/
    └── <model>/<dataset>/<config>/
        ├── test_embeddings.npz
        └── train_embeddings.npz
```