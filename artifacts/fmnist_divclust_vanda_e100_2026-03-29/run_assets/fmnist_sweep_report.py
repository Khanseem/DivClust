#!/usr/bin/env python3
import argparse
import csv
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


METRIC_KEYS = [
    "eval_confidence",
    "mean_cluster_acc_eval",
    "max_cluster_acc_eval",
    "mean_cluster_nmi_eval",
]

HEATMAP_COLUMNS = [
    "CNF (%)",
    "Mean_ACC (%)",
    "Max_ACC (%)",
    "Mean_NMI (%)",
    "DivClust_NMI (%)",
    "DivClust_ACC (%)",
    "DivClust_ARI (%)",
]

# Convert all heatmap metrics to percentage for consistent coloring.
# ACC metrics are already in percentage and keep scale factor 1.
HEATMAP_PERCENT_SCALES = np.array([100.0, 1.0, 1.0, 100.0, 100.0, 1.0, 100.0], dtype=np.float64)


@dataclass
class RunSummary:
    dt: str
    run_name: str
    status: str
    cnf: float = math.nan
    mean_acc: float = math.nan
    max_acc: float = math.nan
    mean_nmi: float = math.nan
    divclust_nmi: float = math.nan
    divclust_acc: float = math.nan
    divclust_ari: float = math.nan


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate Fashion-MNIST DivClust sweep results")
    parser.add_argument("--base-dir", type=str, default=".", help="DivClust repo root")
    parser.add_argument(
        "--targets",
        type=str,
        default="0.5,0.6,0.7,0.8,0.9,0.95,1.0",
        help="Comma-separated NMI_target values",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="hpc_results/fmnist_sweep_e100",
        help="Output directory for report artifacts",
    )
    parser.add_argument("--epochs", type=int, default=100, help="Expected total epochs")
    return parser.parse_args()


def dt_to_safe(dt: str) -> str:
    return dt.replace(".", "_")


def run_name_from_dt(dt: str) -> str:
    return f"CC_FMNIST_E100_DT_{dt_to_safe(dt)}"


def parse_epoch_metrics(log_path: Path, expected_epochs: int) -> Dict[int, Dict[str, float]]:
    epoch_metrics: Dict[int, Dict[str, float]] = {}
    if not log_path.exists():
        return epoch_metrics

    metric_pattern = re.compile(r"([A-Za-z0-9_]+)=(-?[0-9]+(?:\.[0-9]+)?(?:e-?[0-9]+)?)")
    epoch_pattern = re.compile(r"\| Ep\.\s*(\d+)/(\d+)\s*\|")

    with log_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if "| Ep." not in line or "|" not in line:
                continue
            ep_match = epoch_pattern.search(line)
            if not ep_match:
                continue
            epoch = int(ep_match.group(1))
            total = int(ep_match.group(2))
            if total != expected_epochs:
                continue
            metrics: Dict[str, float] = {}
            for m in metric_pattern.finditer(line):
                key = m.group(1)
                value = float(m.group(2))
                metrics[key] = value
            if metrics:
                epoch_metrics[epoch] = metrics

    return epoch_metrics


def clustering_acc(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    y_pred = y_pred.astype(np.int64)
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    d = int(max(y_pred.max(), y_true.max()) + 1)
    w = np.zeros((d, d), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    row_ind, col_ind = linear_sum_assignment(w.max() - w)
    return float(w[row_ind, col_ind].sum() * 100.0 / y_pred.size)


def _remap_to_base(labels: np.ndarray, base: np.ndarray) -> np.ndarray:
    labels = labels.astype(np.int64)
    base = base.astype(np.int64)
    d = int(max(labels.max(), base.max()) + 1)
    w = np.zeros((d, d), dtype=np.int64)
    for i in range(labels.size):
        w[labels[i], base[i]] += 1
    row_ind, col_ind = linear_sum_assignment(w.max() - w)
    mapping = {int(r): int(c) for r, c in zip(row_ind, col_ind)}
    remapped = np.array([mapping.get(int(x), int(x)) for x in labels], dtype=np.int64)
    return remapped


def consensus_labels_from_ensemble(clusters: np.ndarray) -> np.ndarray:
    # clusters shape: (K, N)
    base = clusters[0]
    aligned = [base]
    for k in range(1, clusters.shape[0]):
        aligned.append(_remap_to_base(clusters[k], base))
    aligned_arr = np.stack(aligned, axis=0)

    # majority vote among aligned cluster IDs
    n = aligned_arr.shape[1]
    consensus = np.zeros(n, dtype=np.int64)
    for i in range(n):
        vals, counts = np.unique(aligned_arr[:, i], return_counts=True)
        consensus[i] = vals[np.argmax(counts)]
    return consensus


def compute_divclust_metrics(outcomes_path: Path) -> Tuple[float, float, float]:
    if not outcomes_path.exists():
        return math.nan, math.nan, math.nan

    obj = torch.load(outcomes_path, map_location="cpu", weights_only=False)
    clusters = np.asarray(obj["clusters"])  # (K, N)
    ground_truth = np.asarray(obj["ground_truth"]).reshape(-1)

    if clusters.ndim == 1:
        consensus = clusters
    else:
        consensus = consensus_labels_from_ensemble(clusters)

    div_acc = clustering_acc(consensus, ground_truth)
    div_nmi = float(np.round(normalized_mutual_info_score(consensus, ground_truth), 5))
    div_ari = float(np.round(adjusted_rand_score(ground_truth, consensus), 5))
    return div_nmi, div_acc, div_ari


def build_summary(base_dir: Path, targets: List[str], epochs: int) -> Tuple[List[RunSummary], Dict[str, List[Tuple[int, float]]]]:
    results: List[RunSummary] = []
    curves: Dict[str, List[Tuple[int, float]]] = {}

    for dt in targets:
        run_name = run_name_from_dt(dt)
        exp_dir = base_dir / "experiments" / "deep_clustering" / run_name
        log_path = exp_dir / "log.txt"
        outcomes_path = exp_dir / "outcomes"

        epoch_metrics = parse_epoch_metrics(log_path, expected_epochs=epochs)
        final_epoch = epochs - 1

        if not log_path.exists():
            results.append(RunSummary(dt=dt, run_name=run_name, status="missing_log"))
            curves[dt] = []
            continue

        # Curve from per-epoch mean accuracy (only epochs where eval metrics exist)
        curve = []
        for ep in sorted(epoch_metrics.keys()):
            m = epoch_metrics[ep]
            if "mean_cluster_acc_eval" in m:
                curve.append((ep, m["mean_cluster_acc_eval"]))
        curves[dt] = curve

        if final_epoch not in epoch_metrics:
            status = "incomplete"
            last_ep = max(epoch_metrics.keys()) if epoch_metrics else None
            if last_ep is not None:
                status = f"incomplete_ep_{last_ep}"
            results.append(RunSummary(dt=dt, run_name=run_name, status=status))
            continue

        fm = epoch_metrics[final_epoch]
        div_nmi, div_acc, div_ari = compute_divclust_metrics(outcomes_path)

        results.append(
            RunSummary(
                dt=dt,
                run_name=run_name,
                status="complete",
                cnf=fm.get("eval_confidence", math.nan),
                mean_acc=fm.get("mean_cluster_acc_eval", math.nan),
                max_acc=fm.get("max_cluster_acc_eval", math.nan),
                mean_nmi=fm.get("mean_cluster_nmi_eval", math.nan),
                divclust_nmi=div_nmi,
                divclust_acc=div_acc,
                divclust_ari=div_ari,
            )
        )

    return results, curves


def write_summary_csv(output_dir: Path, summaries: List[RunSummary]) -> Path:
    out_csv = output_dir / "fmnist_sweep_summary.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "NMI_target",
            "run_name",
            "status",
            "CNF",
            "mean_accuracy",
            "max_accuracy",
            "mean_NMI",
            "DivClust_NMI",
            "DivClust_ACC",
            "DivClust_ARI",
        ])
        for r in summaries:
            writer.writerow([
                r.dt,
                r.run_name,
                r.status,
                r.cnf,
                r.mean_acc,
                r.max_acc,
                r.mean_nmi,
                r.divclust_nmi,
                r.divclust_acc,
                r.divclust_ari,
            ])
    return out_csv


def write_markdown_table(output_dir: Path, summaries: List[RunSummary]) -> Path:
    out_md = output_dir / "fmnist_sweep_summary.md"
    lines = []
    lines.append("| NMI_target | status | CNF | mean accuracy | max accuracy | mean NMI | DivClust NMI | DivClust ACC | DivClust ARI |")
    lines.append("|---:|---|---:|---:|---:|---:|---:|---:|---:|")
    for r in summaries:
        lines.append(
            f"| {r.dt} | {r.status} | {r.cnf:.5f} | {r.mean_acc:.5f} | {r.max_acc:.5f} | {r.mean_nmi:.5f} | {r.divclust_nmi:.5f} | {r.divclust_acc:.5f} | {r.divclust_ari:.5f} |"
        )
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_md


def plot_accuracy_curves(output_dir: Path, curves: Dict[str, List[Tuple[int, float]]]) -> Path:
    out_png = output_dir / "accuracy_vs_epoch.png"
    plt.figure(figsize=(10, 6))
    for dt in sorted(curves.keys(), key=lambda x: float(x)):
        curve = curves[dt]
        if not curve:
            continue
        xs = [ep for ep, _ in curve]
        ys = [acc for _, acc in curve]
        plt.plot(xs, ys, label=f"DT={dt}", linewidth=1.8)
    plt.title("Fashion-MNIST DivClust: Mean Accuracy vs Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Mean Cluster Accuracy (%)")
    plt.grid(alpha=0.25)
    if any(curves.values()):
        plt.legend(ncol=2, fontsize=9)
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()
    return out_png


def plot_heatmap(output_dir: Path, summaries: List[RunSummary]) -> Path:
    out_png = output_dir / "simulation_heatmap.png"
    dts = [r.dt for r in summaries]
    raw_matrix = np.array(
        [
            [r.cnf, r.mean_acc, r.max_acc, r.mean_nmi, r.divclust_nmi, r.divclust_acc, r.divclust_ari]
            for r in summaries
        ],
        dtype=np.float64,
    )
    matrix = raw_matrix * HEATMAP_PERCENT_SCALES.reshape(1, -1)

    plt.figure(figsize=(11, max(3.8, 0.55 * len(dts))))
    im = plt.imshow(matrix, aspect="auto", cmap="viridis")
    cbar = plt.colorbar(im, fraction=0.04, pad=0.02)
    cbar.set_label("Metric value (%)")
    plt.xticks(np.arange(len(HEATMAP_COLUMNS)), HEATMAP_COLUMNS, rotation=25, ha="right")
    plt.yticks(np.arange(len(dts)), [f"DT={dt}" for dt in dts])
    plt.title("Fashion-MNIST DivClust Sweep Simulation Heatmap (All Metrics in %)")

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix[i, j]
            text = "nan" if np.isnan(val) else f"{val:.1f}"
            plt.text(j, i, text, ha="center", va="center", fontsize=8, color="white")

    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()
    return out_png


def main() -> None:
    args = parse_args()
    base_dir = Path(args.base_dir).resolve()
    output_dir = (base_dir / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    targets = [x.strip() for x in args.targets.split(",") if x.strip()]
    targets = sorted(targets, key=lambda x: float(x))

    summaries, curves = build_summary(base_dir=base_dir, targets=targets, epochs=args.epochs)

    csv_path = write_summary_csv(output_dir, summaries)
    md_path = write_markdown_table(output_dir, summaries)
    acc_plot = plot_accuracy_curves(output_dir, curves)
    heatmap = plot_heatmap(output_dir, summaries)

    print(f"Wrote summary CSV: {csv_path}")
    print(f"Wrote summary Markdown: {md_path}")
    print(f"Wrote accuracy curve plot: {acc_plot}")
    print(f"Wrote simulation heatmap: {heatmap}")


if __name__ == "__main__":
    main()
