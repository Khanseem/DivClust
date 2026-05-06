# Fashion-MNIST Sweep (CIFAR10-Matched Config, K=20, E=100) on Vanda

This artifact folder is for the rerun requested on **March 30, 2026**:

- Dataset: `fashion_mnist`
- CIFAR-style core setup: `resnet34`, `crop_size=224`, `eval_interval=10`, `batch_size=256`
- Clustering setup: `clusters=10`, `clusterings=20`
- Epochs: `100`
- Swept target (`D^T`): `0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0`
- Scheduler: PBS (`auto` -> GPU queue)

## Structure

- `configs/`: base config + per-target frozen configs used for traceability
- `run_assets/`: PBS submit/run/collect/report scripts
- `results/`: logs, outcomes, resolved per-run configs, tables, and plots

## Run Flow (on HPC)

1. Submit jobs:

```bash
bash artifacts/fmnist_divclust_vanda_cifar10cfg_k20_e100_2026-03-30/run_assets/submit_sweep.sh
```

2. Wait until all jobs finish.

3. Collect outputs into this artifact:

```bash
bash artifacts/fmnist_divclust_vanda_cifar10cfg_k20_e100_2026-03-30/run_assets/collect_results.sh
```

4. Build summary tables + plots:

```bash
python3 artifacts/fmnist_divclust_vanda_cifar10cfg_k20_e100_2026-03-30/run_assets/fmnist_c10cfg_k20_sweep_report.py \
  --artifact-dir artifacts/fmnist_divclust_vanda_cifar10cfg_k20_e100_2026-03-30
```

## Expected Runtime

With `clusterings=20`, each run is usually significantly slower than the earlier `clusterings=5` runs.  
Typical walltime expectation per run: roughly **2.5 to 4.0 hours**, depending on GPU node/queue conditions.
