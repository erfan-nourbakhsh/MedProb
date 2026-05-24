#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

DATASET="${1:-${DATASET:-}}"
if [[ -z "${DATASET}" ]]; then
  echo "Usage: $0 <PATH-VQA|VQA-RAD|SLAKE|ALL_MED_VQA>"
  exit 1
fi

case "${DATASET}" in
  PATH-VQA|VQA-RAD|SLAKE|ALL_MED_VQA) ;;
  *)
    echo "Unsupported dataset: ${DATASET}"
    echo "Valid datasets: PATH-VQA, VQA-RAD, SLAKE, ALL_MED_VQA"
    exit 1
    ;;
esac

GPU_ID="${GPU_ID:-1}"
METHOD="${METHOD:-full}"
LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/logs}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/qwen25_vl_${DATASET}_${METHOD}_gpu${GPU_ID}_${TIMESTAMP}.log"

mkdir -p "${LOG_DIR}"

nohup env GPU_ID="${GPU_ID}" METHOD="${METHOD}" \
  bash "${PROJECT_ROOT}/fine-tune/run_qwen25_vl.sh" "${DATASET}" \
  > "${LOG_FILE}" 2>&1 < /dev/null &

PID=$!
echo "Started ${DATASET} fine-tuning in background."
echo "PID: ${PID}"
echo "Log: ${LOG_FILE}"
