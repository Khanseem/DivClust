#!/bin/bash
#PBS -q auto
#PBS -l select=1:ngpus=1
#PBS -l walltime=03:20:00
#PBS -N divclust_run
#PBS -o hpc_results/output_dt_1.log
#PBS -e hpc_results/error_dt_1.log

cd /scratch/e1536052/DivClust
source divclust_env/bin/activate

python main.py \
  --preset cc_cifar10 \
  --clusterings 20 \
  --epochs 100 \
  --NMI_target 1.0 \
  --run_name CC_dt_1.0	
