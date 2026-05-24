#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

for dataset in PATH-VQA VQA-RAD SLAKE; do
  bash "${PROJECT_ROOT}/fine-tune/run_qwen25_vl.sh" "${dataset}"
done
