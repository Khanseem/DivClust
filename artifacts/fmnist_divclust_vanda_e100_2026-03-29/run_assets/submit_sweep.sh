#!/bin/bash
set -euo pipefail
cd /scratch/e0538389/DivClust
for DT in 0.5 0.6 0.7 0.8 0.9 0.95 1.0; do
  echo "Submitting DT=${DT}"
  qsub -v DT=$DT run_divclust_fmnist_e100.pbs
done
