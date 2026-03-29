# Fashion-MNIST DivClust Sweep (Vanda, 100 epochs)

This folder is a self-contained bundle of the Fashion-MNIST DivClust sweep executed on NUS HPC Vanda.

## Sweep specification

- Dataset: `fashion_mnist`
- Preset: `cc_fashionmnist`
- Fixed epochs: `100`
- Swept parameter: `NMI_target` (`D^T`)
- Targets: `0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0`
- Scheduler: PBS (`auto` -> `batch_gpu`, 1 GPU per job)

## Folder layout

- `run_assets/`: run + submission + reporting scripts and submitted job map
- `configs/`: base config and per-target configs used for the sweep
- `results/`: final summaries, plots, raw logs, per-run resolved configs, and outcomes tensors

## Key outputs

- Summary CSV: `results/fmnist_sweep_summary.csv`
- Summary Markdown: `results/fmnist_sweep_summary.md`
- Combined metrics graph: `results/plots/all_metrics_same_graph.png`
- Accuracy-vs-epoch: `results/plots/accuracy_vs_epoch.png`
- Simulation heatmap: `results/plots/simulation_heatmap.png`

## Runtime

Observed walltime per run in this sweep: approximately `1h 09m` to `1h 20m`.
