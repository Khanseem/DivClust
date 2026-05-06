#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash hpc_results/vanda/run_fashionmnist_20e.sh
# Optional overrides:
#   DATASET_PATH=/scratch/$USER/FashionMNIST GPUS=0

DATASET_PATH="${DATASET_PATH:-./data_folder/FashionMNIST}"
GPUS="${GPUS:-0}"

python3 main.py \
  --preset cc_fashionmnist \
  --dataset_path "${DATASET_PATH}" \
  --epochs 20 \
  --gpus "${GPUS}" \
  --gpu "${GPUS}" \
  --wandb_mode off
