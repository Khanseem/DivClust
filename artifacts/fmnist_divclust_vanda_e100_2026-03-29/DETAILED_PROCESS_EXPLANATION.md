# Detailed End-to-End Explanation (Kid-Friendly)

This document explains, in great detail, what we did for the Fashion-MNIST DivClust experiments on Vanda, how data was transformed, how this compares to CIFAR-10, why some settings were different, and what the results mean.

Think of this as the "full story" of the experiment.

---

## 1) What we were trying to do

We wanted to run DivClust on **Fashion-MNIST** and test different values of one knob called `NMI_target` (also written as `D^T` in the paper discussion).

We kept everything else fixed and only changed this one knob:

- `0.5`
- `0.6`
- `0.7`
- `0.8`
- `0.9`
- `0.95`
- `1.0`

Each value was launched as a separate PBS job on Vanda GPU nodes so they could run in parallel.

---

## 2) Where the experiment assets are stored

Everything is bundled in this folder:

- `artifacts/fmnist_divclust_vanda_e100_2026-03-29/`

Main subfolders:

- `run_assets/`
- `configs/`
- `results/`

Important files:

- Run PBS script: `run_assets/run_divclust_fmnist_e100.pbs`
- Submit helper: `run_assets/submit_sweep.sh`
- Report generator: `run_assets/fmnist_sweep_report.py`
- Per-target config files: `configs/cc_fashionmnist_e100_dt_*.yaml`
- Final metrics table: `results/fmnist_sweep_summary.csv` and `results/fmnist_sweep_summary.md`
- Plots:
  - `results/plots/accuracy_vs_epoch.png`
  - `results/plots/simulation_heatmap.png`
  - `results/plots/all_metrics_same_graph.png`

---

## 3) Big picture workflow (what happened, step by step)

1. Connected to NUS HPC (`atlas9`), then to `vanda`.
2. Verified GPU scheduler behavior:
   - Direct `gpu` queue submit is route-only in this setup.
   - Correct submit path was via `#PBS -q auto`, which routed to `batch_gpu`.
3. Confirmed project allocation and GPU credits.
4. Prepared a run script that:
   - uses Fashion-MNIST preset,
   - forces `epochs=100`,
   - passes each `NMI_target`,
   - writes per-target logs.
5. Submitted 7 jobs (one per target).
6. Waited for all jobs to finish.
7. Parsed logs + outcomes to build summary tables and plots.
8. Packaged scripts/configs/logs/outcomes/results in this artifact folder.

---

## 4) What transformations were done on CIFAR-10

For CIFAR-10 in this repository's `cc_cifar10.yaml` preset:

- `crop_size: 224`
- `clusterings: 1`
- `backbone: resnet34`

The CC transform pipeline (training) is:

1. `RandomResizedCrop(size=224)`
2. `RandomHorizontalFlip()`
3. `RandomApply(ColorJitter(...), p=0.8)`
4. `RandomGrayscale(p=0.2)`
5. `ToTensor()`

Validation pipeline:

1. `Resize((224,224))`
2. `ToTensor()`

Important detail:

- In this CC pipeline, there is **no explicit mean/std normalization** step in `CCTransforms`.
- With `crop_size=224` and CIFAR preset, blur is not added in practice.

---

## 5) What transformations were done on Fashion-MNIST

For Fashion-MNIST in `cc_fashionmnist.yaml`:

- `crop_size: 32`
- `clusterings: 5`
- `backbone: resnet34_small`

First, raw Fashion-MNIST images are grayscale (`1 channel`), but before transforms they are converted to `RGB` (`3 channels`) by the dataset reader class. So the model always sees RGB-form images.

Training transform pipeline for Fashion-MNIST:

1. `RandomResizedCrop(size=32)`
2. `RandomHorizontalFlip()`
3. `RandomApply(ColorJitter(...), p=0.8)`
4. `RandomGrayscale(p=0.2)`
5. `GaussianBlur` (probabilistic, because `crop_size != 224` path enables blur)
6. `ToTensor()`

Validation transform pipeline:

1. `Resize((32,32))`
2. `ToTensor()`

And for CC training, two independent augmented views are created and stacked:

- view 1 = transformed image
- view 2 = transformed image again (different random choices)

So each training sample becomes a pair `(v1, v2)` fed to the CC loss.

### Why blur was ON for Fashion-MNIST but OFF for CIFAR-10 in our run

This was controlled by code logic, not by a random manual choice.

In `data/dataset_implementations/cc.py`, `_get_cc_datasets(...)` has:

- if `crop_size != 224`, then `blur = True`.

So:

- Fashion-MNIST run used `crop_size = 32` -> blur forced ON.
- CIFAR-10 preset used `crop_size = 224` -> this auto-switch did not activate.

Also, CIFAR-10 builder calls `_get_cc_datasets(..., blur=False)`, so with `crop_size=224` it stays OFF.

In short:

- The code uses `crop_size` as a heuristic switch for blur.
- It is not that “Fashion always gets blur and CIFAR never gets blur” by name.
- It happened in this experiment because Fashion was configured at 32 and CIFAR preset at 224.

---

## 6) Why Fashion-MNIST and CIFAR-10 were not identical

Great question. The short answer:

- We used the **Fashion-MNIST preset** and only swept `NMI_target`.
- We did **not** force a CIFAR-10 preset onto Fashion-MNIST.

Why this choice is sensible:

1. Fashion-MNIST is lower-resolution and grayscale by nature.
2. The project already includes a dedicated Fashion-MNIST config.
3. Your request said to sweep only one parameter (`NMI_target`) and keep others fixed; that naturally means "fixed within the Fashion setup."

So differences came from preset design, not from ad-hoc random changes.

---

## 7) Config comparison: CIFAR-10 vs Fashion-MNIST

### CIFAR-10 preset (`configs/cc_cifar10.yaml`)

- `dataset: cifar10`
- `backbone: resnet34`
- `epochs: 1000` (base preset; our Fashion run did 100)
- `eval_interval: 10`
- `crop_size: 224`
- `clusterings: 1`

### Fashion-MNIST preset (`configs/cc_fashionmnist.yaml`)

- `dataset: fashion_mnist`
- `backbone: resnet34_small`
- `epochs: 20` in base preset (then overridden to `100` for this sweep)
- `eval_interval: 1`
- `crop_size: 32`
- `clusterings: 5`
- includes DivClust knobs (`NMI_target`, `NMI_interval`, etc.)

### Why these differences matter

- `crop_size 224` vs `32`: changes geometry, compute, and blur behavior.
- `resnet34` vs `resnet34_small`: model capacity/compute differs.
- `clusterings 1` vs `5`: diversity mechanism is much more active with multiple clusterings.
- `eval_interval 10` vs `1`: Fashion setup logs eval every epoch, giving richer curves.

---

## 8) Exact sweep settings we used for Fashion-MNIST

Fixed:

- `dataset = fashion_mnist`
- `crop_size = 32`
- `clusterings = 5`
- `backbone = resnet34_small`
- `batch_size = 256`
- `epochs = 100`

Swept only:

- `NMI_target in {0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0}`

---

## 9) Runtime results

Per-target observed walltimes (from log START/END):

- `0.5 -> 01:11:30`
- `0.6 -> 01:20:05`
- `0.7 -> 01:10:08`
- `0.8 -> 01:11:51`
- `0.9 -> 01:18:52`
- `0.95 -> 01:19:06`
- `1.0 -> 01:09:05`

Summary:

- fastest: `01:09:05`
- slowest: `01:20:05`
- average: `01:14:22`

Why not exactly equal? GPU jobs can vary due to node load, scheduling, and tiny I/O differences.

---

## 10) Final metrics (what we got)

From `results/fmnist_sweep_summary.csv`:

- `DT=0.5`: CNF `0.895`, mean acc `49.30`, max acc `56.83`, mean NMI `0.523`, DivClust ACC `55.03`, DivClust ARI `0.4097`
- `DT=0.6`: CNF `0.899`, mean acc `56.89`, max acc `66.56`, mean NMI `0.574`, DivClust ACC `58.75`, DivClust ARI `0.4386`
- `DT=0.7`: CNF `0.911`, mean acc `63.15`, max acc `68.78`, mean NMI `0.624`, DivClust ACC `68.16`, DivClust ARI `0.5692`
- `DT=0.8`: CNF `0.903`, mean acc `66.63`, max acc `69.04`, mean NMI `0.652`, DivClust ACC `65.77`, DivClust ARI `0.5447`
- `DT=0.9`: CNF `0.899`, mean acc `67.73`, max acc `69.48`, mean NMI `0.648`, DivClust ACC `67.92`, DivClust ARI `0.5538`
- `DT=0.95`: CNF `0.910`, mean acc `68.08`, max acc `69.93`, mean NMI `0.652`, DivClust ACC `68.51`, DivClust ARI `0.5599`
- `DT=1.0`: CNF `0.903`, mean acc `68.07`, max acc `68.10`, mean NMI `0.648`, DivClust ACC `68.08`, DivClust ARI `0.5540`

---

## 11) What the results mean (explained simply)

Imagine 5 students trying to sort clothes into 10 boxes:

- If they are forced to be **too different** (`DT=0.5`), they disagree too much and overall sorting quality drops.
- If they are allowed to be **a bit more similar** (`DT=0.7` to `0.95`), they disagree in useful ways and do better.
- At very high target (`DT=1.0`), you remove diversity pressure and it behaves closer to a normal setting; still good, but not clearly best on every metric.

So for this Fashion-MNIST setup:

- very low `DT` hurts,
- middle/high `DT` works much better,
- best area is roughly around `0.9` to `0.95` for top accuracy,
- `0.7` gave strongest `DivClust_NMI` in this run.

In plain words:  
**Some diversity helps, too much diversity hurts.**

---

## 12) Why this is useful for future experiments

Now you have:

1. a reproducible sweep template,
2. per-target configs and logs,
3. saved outcomes for deeper analysis,
4. plots + summary tables,
5. a reasonable "good range" for `NMI_target` on Fashion-MNIST.

If you want to make the comparison even stricter to CIFAR style, next step would be:

- run Fashion-MNIST with a "CIFAR-like config clone" (same backbone/crop/clusterings behavior where possible), and then compare side-by-side.

---

## 13) Quick reproducibility commands

From repo root:

```bash
cd /scratch/e0538389/DivClust
bash artifacts/fmnist_divclust_vanda_e100_2026-03-29/run_assets/submit_sweep.sh
python artifacts/fmnist_divclust_vanda_e100_2026-03-29/run_assets/fmnist_sweep_report.py --base-dir . --output-dir artifacts/fmnist_divclust_vanda_e100_2026-03-29/results
```

---

## 14) One last important note

This explanation is tied to the exact code and logs in this bundle. If configs or transform code change later, results can change too.

---

## 15) Teammate feedback clarifications (added 2026-03-30)

### "Why not `clusterings: 20`?"

Because this experiment was designed as a **single-parameter sweep** on top of the Fashion preset.  
Changing 5 -> 20 would be a second major variable.

Also, compute complexity grows strongly with more heads:

- with 5 heads, pair count is `5 * 4 / 2 = 10`
- with 20 heads, pair count is `20 * 19 / 2 = 190`

So 20 heads is not "just 4x"; some pairwise computations jump by about **19x**.

### "Why can Fashion plateau much earlier than CIFAR10?"

Because these are not matched setups:

- Fashion run here: 100 epochs, `resnet34_small`, `crop_size=32`, `clusterings=5`
- CIFAR preset often discussed: 1000 epochs, `resnet34`, `crop_size=224`, `clusterings=1`

Different dataset difficulty + different model/augment/training horizon means different curve shapes are expected.

### "Can we show paper-style heatmaps and clear D^R?"

Yes. New files in `results/feedback_review/` now include:

- paper-style inter-clustering heatmaps for each target
- a `D^T` vs achieved `D^R` plot
- paper-style table comparison for `D^T=0.7`
- overlap comparison CSV/plot for `D^T in {0.7, 0.8, 0.9, 0.95, 1.0}`
