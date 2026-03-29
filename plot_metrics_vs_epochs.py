#import matplotlib
#matplotlib.use("Agg")

import os
import re
import math
import csv
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------
BASE = "/scratch/e1536052/DivClust/experiments/deep_clustering/CC_dt_0.8_1000epochs"
LOG_FILE = f"{BASE}/log.txt"

OUTPUT_DIR = "/scratch/e1536052/DivClust/Output_Results/epoch_plots_CC_dt_0.8_1000epochs"
PER_METRIC_DIR = os.path.join(OUTPUT_DIR, "per_metric")
os.makedirs(PER_METRIC_DIR, exist_ok=True)

# ------------------------------------------------------------
# Paper reference values for CC, CIFAR10, DT = 0.8
# Main paper table values + supplementary mean/max NMI/ARI
# ------------------------------------------------------------
paper_values = {
    "CNF": 0.930,
    "Mean ACC": 0.762,
    "Max ACC": 0.847,
    "Inter-clustering NMI": 0.814,
    "Mean NMI": 0.675,   # supplementary table
    "Max NMI": 0.762,    # supplementary table
    "Mean ARI": 0.632,   # supplementary table
    "Max ARI": 0.727,    # supplementary table
}

# ------------------------------------------------------------
# Regex helpers
# ------------------------------------------------------------
epoch_pat = re.compile(r"Ep\.\s*(\d+)/1000")
float_field = {
    "CNF": re.compile(r"eval_confidence=([0-9]*\.?[0-9]+(?:e[-+]?\d+)?)"),
    "Mean ACC": re.compile(r"mean_cluster_acc_eval=([0-9]*\.?[0-9]+(?:e[-+]?\d+)?)"),
    "Max ACC": re.compile(r"max_cluster_acc_eval=([0-9]*\.?[0-9]+(?:e[-+]?\d+)?)"),
    "Mean NMI": re.compile(r"mean_cluster_nmi_eval=([0-9]*\.?[0-9]+(?:e[-+]?\d+)?)"),
    "Max NMI": re.compile(r"max_cluster_nmi_eval=([0-9]*\.?[0-9]+(?:e[-+]?\d+)?)"),
    "Mean ARI": re.compile(r"mean_cluster_ari_eval=([0-9]*\.?[0-9]+(?:e[-+]?\d+)?)"),
    "Max ARI": re.compile(r"max_cluster_ari_eval=([0-9]*\.?[0-9]+(?:e[-+]?\d+)?)"),
    "Inter-clustering NMI": re.compile(r"interclustering_nmi_eval=([0-9]*\.?[0-9]+(?:e[-+]?\d+)?)"),
}

# Optional training-loss metrics (no paper horizontal lines for these)
loss_field = {
    "loss_cc": re.compile(r"loss_cc=([0-9]*\.?[0-9]+(?:e[-+]?\d+)?)"),
    "loss_ce": re.compile(r"loss_ce=([0-9]*\.?[0-9]+(?:e[-+]?\d+)?)"),
    "loss_ne": re.compile(r"loss_ne=([0-9]*\.?[0-9]+(?:e[-+]?\d+)?)"),
    "loss_div": re.compile(r"loss_div=([0-9]*\.?[0-9]+(?:e[-+]?\d+)?)"),
    "threshold": re.compile(r"threshold=([0-9]*\.?[0-9]+(?:e[-+]?\d+)?)"),
    "loss": re.compile(r"(?<!_)loss=([0-9]*\.?[0-9]+(?:e[-+]?\d+)?)"),
}

# ------------------------------------------------------------
# Parse log
# ------------------------------------------------------------
eval_history = {k: [] for k in float_field.keys()}
eval_epochs = []

loss_history = {k: [] for k in loss_field.keys()}
loss_epochs = []

if not os.path.exists(LOG_FILE):
    raise FileNotFoundError(f"Missing log file: {LOG_FILE}")

with open(LOG_FILE, "r", encoding="utf-8", errors="ignore") as f:
    for line in f:
        # Evaluation lines: contain eval_confidence
        if "eval_confidence=" in line:
            m_epoch = epoch_pat.search(line)
            if not m_epoch:
                continue
            epoch = int(m_epoch.group(1))
            eval_epochs.append(epoch)

            for metric, pat in float_field.items():
                m = pat.search(line)
                if m is None:
                    raise ValueError(f"Could not parse {metric} at epoch {epoch}")
                value = float(m.group(1))

                # ACC fields in log are percentages; convert to decimals
                if metric in {"Mean ACC", "Max ACC"}:
                    value /= 100.0

                eval_history[metric].append(value)

        # End-of-epoch training summaries: contain "Time:" but not eval_confidence
        elif " | Time: " in line and "eval_confidence=" not in line:
            m_epoch = epoch_pat.search(line)
            if not m_epoch:
                continue
            epoch = int(m_epoch.group(1))
            loss_epochs.append(epoch)

            for metric, pat in loss_field.items():
                m = pat.search(line)
                if m is None:
                    loss_history[metric].append(np.nan)
                else:
                    loss_history[metric].append(float(m.group(1)))

# ------------------------------------------------------------
# Save parsed CSVs
# ------------------------------------------------------------
eval_csv = os.path.join(OUTPUT_DIR, "eval_metrics_vs_epoch.csv")
with open(eval_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["Epoch"] + list(eval_history.keys()))
    for i, ep in enumerate(eval_epochs):
        writer.writerow([ep] + [eval_history[m][i] for m in eval_history.keys()])

loss_csv = os.path.join(OUTPUT_DIR, "training_losses_vs_epoch.csv")
with open(loss_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["Epoch"] + list(loss_history.keys()))
    for i, ep in enumerate(loss_epochs):
        writer.writerow([ep] + [loss_history[m][i] for m in loss_history.keys()])

print(f"Saved evaluation CSV to: {eval_csv}")
print(f"Saved loss CSV to: {loss_csv}")

# ------------------------------------------------------------
# Plot evaluation metrics with paper horizontal lines
# ------------------------------------------------------------
metrics_to_plot = list(eval_history.keys())
n_metrics = len(metrics_to_plot)
ncols = 2
nrows = math.ceil(n_metrics / ncols)

fig, axes = plt.subplots(nrows, ncols, figsize=(12, 3.8 * nrows))
axes = np.array(axes).reshape(-1)

for ax, metric in zip(axes, metrics_to_plot):
    ax.plot(eval_epochs, eval_history[metric], marker="o", linewidth=2, label="Simulation")

    if metric in paper_values:
        ax.axhline(
            paper_values[metric],
            linestyle="--",
            linewidth=2,
            label=f"Paper = {paper_values[metric]:.3f}"
        )

    ax.set_title(metric, fontsize=12)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Value")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()

for i in range(len(metrics_to_plot), len(axes)):
    axes[i].axis("off")

fig.suptitle("CC_dt_0.8_1000epochs: Evaluation Metrics vs Epoch", fontsize=15)
plt.tight_layout(rect=[0, 0, 1, 0.97])

combined_eval_fig = os.path.join(OUTPUT_DIR, "eval_metrics_vs_epoch_with_paper.png")
plt.savefig(combined_eval_fig, dpi=300, bbox_inches="tight")
plt.show()

print(f"Saved combined evaluation plot to: {combined_eval_fig}")

# ------------------------------------------------------------
# Plot one figure per evaluation metric
# ------------------------------------------------------------
for metric in metrics_to_plot:
    plt.figure(figsize=(6.5, 4.5))
    plt.plot(eval_epochs, eval_history[metric], marker="o", linewidth=2, label="Simulation")

    if metric in paper_values:
        plt.axhline(
            paper_values[metric],
            linestyle="--",
            linewidth=2,
            label=f"Paper = {paper_values[metric]:.3f}"
        )

    plt.title(metric)
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()

    safe_name = metric.replace(" ", "_").replace("/", "_")
    out_path = os.path.join(PER_METRIC_DIR, f"{safe_name}.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

print(f"Saved per-metric evaluation plots to: {PER_METRIC_DIR}")

# ------------------------------------------------------------
# Plot training losses vs epoch (no paper reference lines)
# ------------------------------------------------------------
loss_metrics = list(loss_history.keys())
n_loss = len(loss_metrics)
ncols = 2
nrows = math.ceil(n_loss / ncols)

fig, axes = plt.subplots(nrows, ncols, figsize=(12, 3.6 * nrows))
axes = np.array(axes).reshape(-1)

for ax, metric in zip(axes, loss_metrics):
    ax.plot(loss_epochs, loss_history[metric], linewidth=2)
    ax.set_title(metric, fontsize=12)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Value")
    ax.grid(True, linestyle="--", alpha=0.5)

for i in range(len(loss_metrics), len(axes)):
    axes[i].axis("off")

fig.suptitle("CC_dt_0.8_1000epochs: Training Loss Metrics vs Epoch", fontsize=15)
plt.tight_layout(rect=[0, 0, 1, 0.97])

combined_loss_fig = os.path.join(OUTPUT_DIR, "training_losses_vs_epoch.png")
plt.savefig(combined_loss_fig, dpi=300, bbox_inches="tight")
plt.show()

print(f"Saved combined loss plot to: {combined_loss_fig}")