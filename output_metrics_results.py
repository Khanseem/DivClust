#import matplotlib
#matplotlib.use("Agg")

import os
import re
import csv
import math
import numpy as np
import matplotlib.pyplot as plt

BASE_DIR = "/scratch/e1536052/DivClust/experiments/deep_clustering"
OUTPUT_DIR = "/scratch/e1536052/DivClust/Output_Results"
TARGETS = [0.7, 0.8, 0.9, 0.95, 1.0]

# Create output folders
os.makedirs(OUTPUT_DIR, exist_ok=True)
PER_METRIC_DIR = os.path.join(OUTPUT_DIR, "metric_plots")
os.makedirs(PER_METRIC_DIR, exist_ok=True)

# ---------------------------------------------------------------------
# 1) Hard-coded paper results
# ---------------------------------------------------------------------
paper_results = {
    "CNF (mean NMI)":       {0.7: 0.927, 0.8: 0.930, 0.9: 0.931, 0.95: 0.934, 1.0: 0.934},
    "Mean ACC":             {0.7: 0.703, 0.8: 0.762, 0.9: 0.794, 0.95: 0.762, 1.0: 0.763},
    "Max ACC":              {0.7: 0.818, 0.8: 0.847, 0.9: 0.818, 0.95: 0.773, 1.0: 0.763},
    "Inter-clustering NMI": {0.7: 0.699, 0.8: 0.814, 0.9: 0.900, 0.95: 0.946, 1.0: 0.976},
    "Consensus NMI":        {0.7: 0.710, 0.8: 0.724, 0.9: 0.678, 0.95: 0.677, 1.0: 0.678},
    "Consensus ACC":        {0.7: 0.815, 0.8: 0.819, 0.9: 0.789, 0.95: 0.760, 1.0: 0.763},
    "Consensus ARI":        {0.7: 0.675, 0.8: 0.681, 0.9: 0.641, 0.95: 0.602, 1.0: 0.604},
}

# ---------------------------------------------------------------------
# 2) Map paper metric names to exact labels in metrics_log.txt
# ---------------------------------------------------------------------
log_metric_map = {
    "CNF (mean NMI)": "CNF (mean NMI)",
    "Mean ACC": "Mean ACC",
    "Max ACC": "Max ACC",
    "Inter-clustering NMI": "Inter-clustering NMI",
    "Consensus NMI": "Consensus NMI",
    "Consensus ACC": "Consensus ACC",
    "Consensus ARI": "Consensus ARI",
}

# ---------------------------------------------------------------------
# 3) Parse metrics_log.txt
# ---------------------------------------------------------------------
def parse_metrics_log(filepath):
    metrics = {}

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Missing file: {filepath}")

    with open(filepath, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if ":" not in line:
                continue

            left, right = line.split(":", 1)
            key = left.strip()
            value_str = right.strip()

            # Skip non-scalar lines
            if key in {"Selected heads", "Saved labels to"}:
                continue

            m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", value_str)
            if m:
                metrics[key] = float(m.group())

    return metrics

# ---------------------------------------------------------------------
# 4) Load simulation results
# ---------------------------------------------------------------------
simulation_results = {metric: {} for metric in paper_results.keys()}

for target in TARGETS:
    folder = os.path.join(BASE_DIR, f"CC_dt_{target}")
    metrics_file = os.path.join(folder, "metrics_log.txt")

    parsed = parse_metrics_log(metrics_file)

    for plot_metric, log_metric in log_metric_map.items():
        if log_metric not in parsed:
            raise KeyError(
                f"Metric '{log_metric}' not found in {metrics_file}. "
                f"Found keys: {sorted(parsed.keys())}"
            )
        simulation_results[plot_metric][target] = parsed[log_metric]

# ---------------------------------------------------------------------
# 5) Save combined CSV summary
# ---------------------------------------------------------------------
csv_path = os.path.join(OUTPUT_DIR, "paper_vs_sim_metrics.csv")
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["Metric", "NMI_target", "Paper", "Simulation", "Difference"])
    for metric in paper_results:
        for target in TARGETS:
            p = paper_results[metric][target]
            y = simulation_results[metric][target]
            writer.writerow([metric, target, p, y, y - p])

print(f"Saved CSV summary to: {csv_path}")

# ---------------------------------------------------------------------
# 6) Plot all metrics in one figure
# ---------------------------------------------------------------------
metrics_to_plot = list(paper_results.keys())
n_metrics = len(metrics_to_plot)

ncols = 2
nrows = math.ceil(n_metrics / ncols)

fig, axes = plt.subplots(nrows, ncols, figsize=(12, 3.8 * nrows))
axes = np.array(axes).reshape(-1)

for ax, metric in zip(axes, metrics_to_plot):
    x = TARGETS
    y_paper = [paper_results[metric][t] for t in x]
    y_simulation = [simulation_results[metric][t] for t in x]

    ax.plot(x, y_paper, marker="o", linewidth=2, label="Paper")
    ax.plot(x, y_simulation, marker="s", linewidth=2, label="Simulation")

    ax.set_title(metric, fontsize=12)
    ax.set_xlabel("NMI_target")
    ax.set_ylabel("Value")
    ax.set_xticks(TARGETS)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()

for i in range(len(metrics_to_plot), len(axes)):
    axes[i].axis("off")

fig.suptitle("Paper Results vs Simulation Results across NMI_target", fontsize=15)
plt.tight_layout(rect=[0, 0, 1, 0.97])

combined_fig_path = os.path.join(OUTPUT_DIR, "paper_vs_simulation_metrics_all.png")
plt.savefig(combined_fig_path, dpi=300, bbox_inches="tight")
plt.show()

print(f"Saved combined figure to: {combined_fig_path}")

# ---------------------------------------------------------------------
# 7) Save one figure per metric
# ---------------------------------------------------------------------
for metric in metrics_to_plot:
    x = TARGETS
    y_paper = [paper_results[metric][t] for t in x]
    y_simulation = [simulation_results[metric][t] for t in x]

    plt.figure(figsize=(6.5, 4.5))
    plt.plot(x, y_paper, marker="o", linewidth=2, label="Paper")
    plt.plot(x, y_simulation, marker="s", linewidth=2, label="Simulation")
    plt.title(metric)
    plt.xlabel("NMI_target")
    plt.ylabel("Value")
    plt.xticks(TARGETS)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()

    safe_name = (
        metric.replace(" ", "_")
              .replace("(", "")
              .replace(")", "")
              .replace("/", "_")
    )
    out_path = os.path.join(PER_METRIC_DIR, f"{safe_name}.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

print(f"Saved per-metric plots to: {PER_METRIC_DIR}")