# Teammate Feedback Response (2026-03-30)

This note responds directly to the comments in the two WhatsApp screenshots.

## 1) Config-setting questions

### Why `backbone: resnet34_small`?

- We used the existing Fashion-MNIST preset (`configs/cc_fashionmnist.yaml`) and only swept `NMI_target`.
- That preset is intentionally lighter than CIFAR10 (`resnet34`) because Fashion-MNIST is lower-resolution and simpler.
- It also keeps GPU runtime reasonable for parallel PBS sweeps.

### What is `crop_size: 32`?

- It is the image size after augmentation resize/crop in `CCTransforms`.
- Fashion-MNIST raw images are `28x28`; using `32x32` is a small upsize that works well with the model while preserving the low-resolution setting.
- In this codebase, `crop_size` also affects augmentation behavior (blur switch; see section 3 below).

### Why `clusterings: 5` and not `20`?

- We kept the Fashion-MNIST preset fixed (except `NMI_target`) to satisfy the "only sweep one parameter" requirement.
- `clusterings=5` means 5 clustering heads. Increasing to 20 increases pairwise interactions sharply.
- The diversity parts of the loss/metrics scale with pair counts: with 5 heads there are 10 pairs; with 20 heads there are 190 pairs.
- So moving 5 -> 20 is not a tiny tweak; it is a major compute and optimization change.

## 2) "Can we follow paper Table metrics and show D^R clearly?"

Yes. New artifacts now include explicit `D^R` (inter-clustering NMI) and paper-style comparisons:

- `results/feedback_review/fmnist_sweep_with_dr_summary.csv`
- `results/feedback_review/fmnist_sweep_with_dr_summary.md`
- `results/feedback_review/paper_comparison_dt_0_7.csv`
- `results/feedback_review/paper_comparison_dt_0_7.md`
- `results/feedback_review/paper_vs_simulation_overlap_targets.csv`
- `results/feedback_review/plots/paper_vs_simulation_overlap_metrics.png`
- `results/feedback_review/plots/dt_vs_dr_target_alignment.png`

### D^R target-achievement summary (from outcomes, avg off-diagonal NMI)

- `D^T=0.50 -> D^R=0.5106` (hit)
- `D^T=0.60 -> D^R=0.6014` (hit)
- `D^T=0.70 -> D^R=0.6965` (hit)
- `D^T=0.80 -> D^R=0.7992` (hit)
- `D^T=0.90 -> D^R=0.8968` (hit)
- `D^T=0.95 -> D^R=0.8752` (not hit)
- `D^T=1.00 -> D^R=0.9800` (hit)

Here, "hit" uses `|D^R - D^T| <= 0.03`.

## 3) "Can we do the same/similar heatmap as the paper?"

Yes. We added paper-style inter-clustering NMI matrix heatmaps (head-vs-head, values in `[0,1]`) for every target:

- `results/feedback_review/plots/inter_cluster_nmi_heatmap_dt_0_5.png`
- `results/feedback_review/plots/inter_cluster_nmi_heatmap_dt_0_6.png`
- `results/feedback_review/plots/inter_cluster_nmi_heatmap_dt_0_7.png`
- `results/feedback_review/plots/inter_cluster_nmi_heatmap_dt_0_8.png`
- `results/feedback_review/plots/inter_cluster_nmi_heatmap_dt_0_9.png`
- `results/feedback_review/plots/inter_cluster_nmi_heatmap_dt_0_95.png`
- `results/feedback_review/plots/inter_cluster_nmi_heatmap_dt_1_0.png`

Important distinction:

- Old `results/plots/simulation_heatmap.png` is a *sweep summary heatmap* across different metrics and targets.
- New `inter_cluster_nmi_heatmap_dt_*.png` are *paper-style inter-clustering similarity matrices* for one run at a time.

## 4) Why does FMNIST plateau early while CIFAR10 can plateau much later?

This is expected given setup differences:

- FMNIST run here is 100 epochs; CIFAR baseline discussed was 1000 epochs.
- FMNIST preset uses `resnet34_small`, `crop_size=32`, `clusterings=5`; CIFAR preset uses `resnet34`, `crop_size=224`, `clusterings=1`.
- FMNIST is grayscale-origin and generally easier than CIFAR10.
- Different augmentation regime and resolution imply different optimization dynamics.

So the curves are not directly apples-to-apples unless we also match backbone, crop pipeline, clusterings, eval interval, and total epochs.

## 5) Why was blur applied on FMNIST but not in the CIFAR10 run?

This came from code logic, not manual one-off tweaking:

- In `data/dataset_implementations/cc.py`, `_get_cc_datasets(...)` forces `blur=True` when `crop_size != 224`.
- FMNIST uses `crop_size=32` -> blur becomes active.
- CIFAR10 preset uses `crop_size=224` -> this auto-switch is not triggered.

## 6) Repro command for new feedback artifacts

From repo root:

```bash
python3 artifacts/fmnist_divclust_vanda_e100_2026-03-29/run_assets/fmnist_feedback_report.py \
  --artifact-dir artifacts/fmnist_divclust_vanda_e100_2026-03-29 \
  --output-subdir results/feedback_review
```

## 7) CIFAR10-matched config now added

Per request, a strict CIFAR10-matched Fashion-MNIST config is now included:

- `configs/cc_fashionmnist_cifar10_matched_e100.yaml` (repo root)
- `artifacts/fmnist_divclust_vanda_e100_2026-03-29/configs/cc_fashionmnist_cifar10_matched_e100.yaml`

Run command:

```bash
python main.py --preset cc_fashionmnist_cifar10_matched_e100 --wandb_mode off
```

PBS submit:

```bash
bash artifacts/fmnist_divclust_vanda_e100_2026-03-29/run_assets/submit_fmnist_cifar10_matched_e100.sh
```
