#!/bin/bash
set -euo pipefail

cd /scratch/e0538389/DivClust
ART_DIR="artifacts/fmnist_divclust_vanda_cifar10cfg_k20_e100_2026-03-30"

mkdir -p "${ART_DIR}/results/logs"
mkdir -p "${ART_DIR}/results/outcomes"
mkdir -p "${ART_DIR}/results/per_run_config"

for DT in 0.5 0.6 0.7 0.8 0.9 0.95 1.0; do
  DT_SAFE=$(echo "${DT}" | tr '.' '_')
  RUN_NAME="CC_FMNIST_C10CFG_K20_E100_DT_${DT_SAFE}"
  PBS_LOG="logs/fmnist_c10cfg_k20_e100_dt_${DT_SAFE}.out"
  EXP_DIR="experiments/deep_clustering/${RUN_NAME}"

  if [ -f "${PBS_LOG}" ]; then
    cp "${PBS_LOG}" "${ART_DIR}/results/logs/"
  fi
  if [ -f "${EXP_DIR}/outcomes" ]; then
    cp "${EXP_DIR}/outcomes" "${ART_DIR}/results/outcomes/${RUN_NAME}.outcomes.pt"
  fi
  if [ -f "${EXP_DIR}/config.txt" ]; then
    cp "${EXP_DIR}/config.txt" "${ART_DIR}/results/per_run_config/${RUN_NAME}.config.txt"
  fi
done
