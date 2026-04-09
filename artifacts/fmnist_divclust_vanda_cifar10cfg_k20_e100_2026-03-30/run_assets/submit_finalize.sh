#!/bin/bash
set -euo pipefail
cd /scratch/e0538389/DivClust

# Usage:
#   bash submit_finalize.sh 822118.stdct-mgmt-02 822119.stdct-mgmt-02 ...
if [ "$#" -lt 1 ]; then
  echo "Provide at least one upstream job id."
  exit 1
fi

DEPS=""
for JOB in "$@"; do
  if [ -z "${DEPS}" ]; then
    DEPS="${JOB}"
  else
    DEPS="${DEPS}:${JOB}"
  fi
done

qsub -W depend=afterany:${DEPS} artifacts/fmnist_divclust_vanda_cifar10cfg_k20_e100_2026-03-30/run_assets/finalize_report.pbs
