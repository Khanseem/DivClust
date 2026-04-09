#!/bin/bash
set -euo pipefail
cd /scratch/e0538389/DivClust
qsub run_fmnist_cifar10_matched_e100.pbs
