#!/bin/bash
set -euo pipefail
cd /scratch/e0538389/DivClust

OUT_CSV="artifacts/fmnist_divclust_vanda_cifar10cfg_k20_e100_2026-03-30/run_assets/submitted_jobs.csv"
mkdir -p "$(dirname "${OUT_CSV}")"
echo "dt,job_id,submitted_at" > "${OUT_CSV}"

for DT in 0.5 0.6 0.7 0.8 0.9 0.95 1.0; do
  JOB_ID=$(qsub -v DT="${DT}" artifacts/fmnist_divclust_vanda_cifar10cfg_k20_e100_2026-03-30/run_assets/run_divclust_fmnist_c10cfg_k20_e100.pbs)
  echo "${DT},${JOB_ID},$(date '+%Y-%m-%d %H:%M:%S')" | tee -a "${OUT_CSV}"
done
