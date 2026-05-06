#!/usr/bin/env python3
import argparse
import csv
import math
import pickle
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import normalized_mutual_info_score


PAPER_RESULTS: Dict[float, Dict[str, float]] = {
    0.7: {
        "CNF": 0.927,
        "Mean_ACC": 0.703,
        "Max_ACC": 0.818,
        "Inter_clustering_NMI_D_R": 0.699,
        "DivClust_NMI": 0.710,
        "DivClust_ACC": 0.815,
        "DivClust_ARI": 0.675,
    },
    0.8: {
        "CNF": 0.930,
        "Mean_ACC": 0.762,
        "Max_ACC": 0.847,
        "Inter_clustering_NMI_D_R": 0.814,
        "DivClust_NMI": 0.724,
        "DivClust_ACC": 0.819,
        "DivClust_ARI": 0.681,
    },
    0.9: {
        "CNF": 0.931,
        "Mean_ACC": 0.794,
        "Max_ACC": 0.818,
        "Inter_clustering_NMI_D_R": 0.900,
        "DivClust_NMI": 0.678,
        "DivClust_ACC": 0.789,
        "DivClust_ARI": 0.641,
    },
    0.95: {
        "CNF": 0.934,
        "Mean_ACC": 0.762,
        "Max_ACC": 0.773,
        "Inter_clustering_NMI_D_R": 0.946,
        "DivClust_NMI": 0.677,
        "DivClust_ACC": 0.760,
        "DivClust_ARI": 0.602,
    },
    1.0: {
        "CNF": 0.934,
        "Mean_ACC": 0.763,
        "Max_ACC": 0.763,
        "Inter_clustering_NMI_D_R": 0.976,
        "DivClust_NMI": 0.678,
        "DivClust_ACC": 0.763,
        "DivClust_ARI": 0.604,
    },
}

PAPER_METRIC_ORDER = [
    "CNF",
    "Mean_ACC",
    "Max_ACC",
    "Inter_clustering_NMI_D_R",
    "DivClust_NMI",
    "DivClust_ACC",
    "DivClust_ARI",
]


@dataclass
class SweepRow:
    dt: float
    run_name: str
    status: str
    cnf: float
    mean_acc_pct: float
    max_acc_pct: float
    mean_nmi: float
    divclust_nmi: float
    divclust_acc_pct: float
    divclust_ari: float
    inter_nmi_dr: float = math.nan
    inter_nmi_dr_from_log_final: float = math.nan
    inter_nmi_delta_target: float = math.nan
    target_achieved_absdiff_le_0_03: bool = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate teammate-feedback artifacts for FMNIST DivClust sweep"
    )
    parser.add_argument(
        "--artifact-dir",
        type=str,
        default=str(Path(__file__).resolve().parents[1]),
        help="Path to artifact folder containing results/",
    )
    parser.add_argument(
        "--targets",
        type=str,
        default="0.5,0.6,0.7,0.8,0.9,0.95,1.0",
        help="Comma-separated D^T values to include",
    )
    parser.add_argument(
        "--target-achieve-tol",
        type=float,
        default=0.03,
        help="Tolerance for marking D^R as target-achieved (abs(D^R - D^T) <= tol)",
    )
    parser.add_argument(
        "--output-subdir",
        type=str,
        default="results/feedback_review",
        help="Output folder (relative to artifact-dir)",
    )
    return parser.parse_args()


def _safe_float(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _dt_to_safe(dt: float) -> str:
    text = f"{dt}"
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    if "." not in text:
        text = text + ".0"
    return text.replace(".", "_")


def load_sweep_rows(summary_csv: Path, targets: List[float]) -> List[SweepRow]:
    if not summary_csv.exists():
        raise FileNotFoundError(f"Missing summary CSV: {summary_csv}")

    rows: List[SweepRow] = []
    target_set = {round(t, 6) for t in targets}

    with summary_csv.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for raw in reader:
            dt = _safe_float(raw.get("NMI_target"))
            if round(dt, 6) not in target_set:
                continue
            rows.append(
                SweepRow(
                    dt=dt,
                    run_name=raw.get("run_name", ""),
                    status=raw.get("status", ""),
                    cnf=_safe_float(raw.get("CNF")),
                    mean_acc_pct=_safe_float(raw.get("mean_accuracy")),
                    max_acc_pct=_safe_float(raw.get("max_accuracy")),
                    mean_nmi=_safe_float(raw.get("mean_NMI")),
                    divclust_nmi=_safe_float(raw.get("DivClust_NMI")),
                    divclust_acc_pct=_safe_float(raw.get("DivClust_ACC")),
                    divclust_ari=_safe_float(raw.get("DivClust_ARI")),
                )
            )
    rows.sort(key=lambda r: r.dt)
    return rows


def load_outcome_clusters(outcomes_path: Path) -> np.ndarray:
    if not outcomes_path.exists():
        raise FileNotFoundError(f"Missing outcomes file: {outcomes_path}")

    obj = None
    try:
        with zipfile.ZipFile(outcomes_path, "r") as archive:
            candidate_names = ["data.pkl", "outcomes/data.pkl"]
            data_name = next((name for name in candidate_names if name in archive.namelist()), None)
            if data_name is not None:
                with archive.open(data_name) as handle:
                    obj = pickle.load(handle)
    except zipfile.BadZipFile:
        obj = None

    if obj is None:
        with outcomes_path.open("rb") as handle:
            obj = pickle.load(handle)

    clusters = np.asarray(obj["clusters"], dtype=np.int64)
    if clusters.ndim == 1:
        clusters = np.expand_dims(clusters, axis=0)
    return clusters


def compute_interclustering_matrix(clusters: np.ndarray) -> np.ndarray:
    heads = clusters.shape[0]
    matrix = np.zeros((heads, heads), dtype=np.float64)
    for i in range(heads):
        for j in range(heads):
            matrix[i, j] = normalized_mutual_info_score(clusters[i], clusters[j])
    return matrix


def average_offdiagonal(matrix: np.ndarray) -> float:
    if matrix.shape[0] <= 1:
        return float(matrix[0, 0])
    denom = matrix.shape[0] * (matrix.shape[0] - 1)
    return float((matrix.sum() - np.trace(matrix)) / denom)


def parse_final_interclustering_nmi_from_log(log_path: Path) -> float:
    if not log_path.exists():
        return math.nan

    epoch_pattern = re.compile(r"\| Ep\.\s*(\d+)/(\d+)\s*\|")
    metric_pattern = re.compile(r"interclustering_nmi_eval=([-+]?\d*\.?\d+(?:e[-+]?\d+)?)")

    best_epoch = -1
    best_value = math.nan
    with log_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            epoch_match = epoch_pattern.search(line)
            if not epoch_match:
                continue
            value_match = metric_pattern.search(line)
            if not value_match:
                continue
            epoch = int(epoch_match.group(1))
            if epoch >= best_epoch:
                best_epoch = epoch
                best_value = float(value_match.group(1))
    return best_value


def plot_interclustering_heatmap(matrix: np.ndarray, dt: float, dr: float, out_path: Path) -> None:
    heads = matrix.shape[0]
    fig, ax = plt.subplots(figsize=(6.4, 6.2))

    coords = np.arange(heads + 1)
    mesh = ax.pcolormesh(
        coords,
        coords,
        matrix,
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
        edgecolors="black",
        linewidth=0.8,
        shading="flat",
    )
    ax.set_xlim(0, heads)
    ax.set_ylim(heads, 0)
    ax.set_aspect("equal")

    ticks = np.arange(heads) + 0.5
    labels = [str(i) for i in range(heads)]
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Head")
    ax.set_ylabel("Head")
    ax.set_title(
        f"Inter-clustering Similarity (NMI)\nD^T={dt:.2f}, D^R(avg off-diagonal)={dr:.3f}"
    )

    for i in range(heads):
        for j in range(heads):
            value = matrix[i, j]
            text_color = "black" if value >= 0.8 else "white"
            ax.text(j + 0.5, i + 0.5, f"{value:.3f}", ha="center", va="center", fontsize=7, color=text_color)

    cbar = plt.colorbar(mesh, ax=ax, shrink=0.78, pad=0.04)
    cbar.set_label("NMI")
    cbar.set_ticks(np.linspace(0, 1, 6))

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_extended_summary(rows: List[SweepRow], out_csv: Path, out_md: Path) -> None:
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "NMI_target_D_T",
                "run_name",
                "status",
                "CNF",
                "Mean_ACC_pct",
                "Max_ACC_pct",
                "Mean_NMI",
                "Inter_clustering_NMI_D_R",
                "Inter_clustering_NMI_from_log_final",
                "D_R_minus_D_T",
                "target_achieved_absdiff_le_0_03",
                "DivClust_NMI",
                "DivClust_ACC_pct",
                "DivClust_ARI",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row.dt,
                    row.run_name,
                    row.status,
                    row.cnf,
                    row.mean_acc_pct,
                    row.max_acc_pct,
                    row.mean_nmi,
                    row.inter_nmi_dr,
                    row.inter_nmi_dr_from_log_final,
                    row.inter_nmi_delta_target,
                    row.target_achieved_absdiff_le_0_03,
                    row.divclust_nmi,
                    row.divclust_acc_pct,
                    row.divclust_ari,
                ]
            )

    lines = []
    lines.append(
        "| D^T | D^R (offdiag NMI) | D^R-D^T | target hit (|diff|<=0.03) | CNF | Mean ACC (%) | Max ACC (%) | Mean NMI | DivClust NMI | DivClust ACC (%) | DivClust ARI |"
    )
    lines.append("|---:|---:|---:|:---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        lines.append(
            f"| {row.dt:.2f} | {row.inter_nmi_dr:.4f} | {row.inter_nmi_delta_target:+.4f} | {str(row.target_achieved_absdiff_le_0_03)} | {row.cnf:.4f} | {row.mean_acc_pct:.2f} | {row.max_acc_pct:.2f} | {row.mean_nmi:.4f} | {row.divclust_nmi:.4f} | {row.divclust_acc_pct:.2f} | {row.divclust_ari:.4f} |"
        )
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_target_vs_achieved_dr(rows: List[SweepRow], out_path: Path) -> None:
    x = [r.dt for r in rows]
    y_target = x
    y_achieved = [r.inter_nmi_dr for r in rows]

    plt.figure(figsize=(8.4, 5.2))
    plt.plot(x, y_target, linestyle="--", linewidth=2.0, label="Target D^T")
    plt.plot(x, y_achieved, marker="o", linewidth=2.0, label="Achieved D^R (inter-clustering NMI)")
    plt.axhline(0.0, color="black", linewidth=0.5, alpha=0.2)
    for dt, dr in zip(x, y_achieved):
        plt.text(dt, dr + 0.01, f"{dr:.3f}", fontsize=8, ha="center")
    plt.title("Target (D^T) vs Achieved Inter-clustering NMI (D^R)")
    plt.xlabel("NMI target (D^T)")
    plt.ylabel("NMI")
    plt.xticks(x)
    plt.ylim(0.45, 1.02)
    plt.grid(alpha=0.28, linestyle="--")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=260, bbox_inches="tight")
    plt.close()


def simulation_metrics_decimal(row: SweepRow) -> Dict[str, float]:
    return {
        "CNF": row.cnf,
        "Mean_ACC": row.mean_acc_pct / 100.0,
        "Max_ACC": row.max_acc_pct / 100.0,
        "Inter_clustering_NMI_D_R": row.inter_nmi_dr,
        "DivClust_NMI": row.divclust_nmi,
        "DivClust_ACC": row.divclust_acc_pct / 100.0,
        "DivClust_ARI": row.divclust_ari,
    }


def write_paper_comparison(
    rows: List[SweepRow],
    out_csv: Path,
    out_plot: Path,
    dt07_csv: Path,
    dt07_md: Path,
) -> None:
    by_target = {round(r.dt, 6): r for r in rows}
    overlap_targets = sorted([t for t in PAPER_RESULTS.keys() if round(t, 6) in by_target])

    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["NMI_target", "Metric", "Paper_Result", "Simulation_Result", "Difference(Sim-Paper)"])
        for target in overlap_targets:
            row = by_target[round(target, 6)]
            sim = simulation_metrics_decimal(row)
            for metric in PAPER_METRIC_ORDER:
                paper_val = PAPER_RESULTS[target][metric]
                sim_val = sim[metric]
                writer.writerow([target, metric, paper_val, sim_val, sim_val - paper_val])

    if overlap_targets:
        fig, axes = plt.subplots(4, 2, figsize=(12.5, 13.5))
        axes = axes.reshape(-1)
        for ax, metric in zip(axes, PAPER_METRIC_ORDER):
            paper_series = [PAPER_RESULTS[t][metric] for t in overlap_targets]
            sim_series = [simulation_metrics_decimal(by_target[round(t, 6)])[metric] for t in overlap_targets]
            ax.plot(overlap_targets, paper_series, marker="o", linewidth=2.0, label="Paper")
            ax.plot(overlap_targets, sim_series, marker="s", linewidth=2.0, label="Simulation (FMNIST E100)")
            ax.set_title(metric.replace("_", " "))
            ax.set_xlabel("NMI target")
            ax.set_ylabel("Value")
            ax.set_xticks(overlap_targets)
            ax.grid(alpha=0.3, linestyle="--")
            ax.legend(fontsize=8)
        for idx in range(len(PAPER_METRIC_ORDER), len(axes)):
            axes[idx].axis("off")
        fig.suptitle("Paper Metrics vs FMNIST Simulation (Overlapping Targets)")
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        plt.savefig(out_plot, dpi=250, bbox_inches="tight")
        plt.close(fig)

    # Dedicated table for D^T = 0.7 (same format as teammate screenshot)
    dt = 0.7
    key = round(dt, 6)
    if key not in by_target:
        return

    row = by_target[key]
    sim = simulation_metrics_decimal(row)

    with dt07_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Metric", "Paper_Result", "Simulation_Result", "Difference(Sim-Paper)"])
        for metric in PAPER_METRIC_ORDER:
            paper_val = PAPER_RESULTS[dt][metric]
            sim_val = sim[metric]
            writer.writerow([metric, paper_val, sim_val, sim_val - paper_val])

    lines = []
    lines.append(f"## NMI_target = {dt}")
    lines.append("")
    lines.append("| Metric | Paper Result | Simulation Result | Difference (Sim-Paper) |")
    lines.append("|---|---:|---:|---:|")
    for metric in PAPER_METRIC_ORDER:
        paper_val = PAPER_RESULTS[dt][metric]
        sim_val = sim[metric]
        diff = sim_val - paper_val
        lines.append(f"| {metric} | {paper_val:.4f} | {sim_val:.4f} | {diff:+.4f} |")
    dt07_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    artifact_dir = Path(args.artifact_dir).resolve()
    targets = [float(x.strip()) for x in args.targets.split(",") if x.strip()]
    output_dir = (artifact_dir / args.output_subdir).resolve()
    plots_dir = output_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    summary_csv = artifact_dir / "results" / "fmnist_sweep_summary.csv"
    logs_dir = artifact_dir / "results" / "logs"
    outcomes_dir = artifact_dir / "results" / "outcomes"

    rows = load_sweep_rows(summary_csv, targets)
    if not rows:
        raise RuntimeError("No sweep rows were loaded from summary CSV for requested targets.")

    for row in rows:
        safe = _dt_to_safe(row.dt)
        outcomes_path = outcomes_dir / f"{row.run_name}.outcomes.pt"
        if not outcomes_path.exists():
            outcomes_path = outcomes_dir / f"CC_FMNIST_E100_DT_{safe}.outcomes.pt"
        clusters = load_outcome_clusters(outcomes_path)
        matrix = compute_interclustering_matrix(clusters)
        row.inter_nmi_dr = average_offdiagonal(matrix)

        log_path = logs_dir / f"fmnist_e100_dt_{safe}.out"
        row.inter_nmi_dr_from_log_final = parse_final_interclustering_nmi_from_log(log_path)
        row.inter_nmi_delta_target = row.inter_nmi_dr - row.dt
        row.target_achieved_absdiff_le_0_03 = abs(row.inter_nmi_delta_target) <= args.target_achieve_tol

        heatmap_path = plots_dir / f"inter_cluster_nmi_heatmap_dt_{safe}.png"
        plot_interclustering_heatmap(matrix, row.dt, row.inter_nmi_dr, heatmap_path)

    write_extended_summary(
        rows=rows,
        out_csv=output_dir / "fmnist_sweep_with_dr_summary.csv",
        out_md=output_dir / "fmnist_sweep_with_dr_summary.md",
    )

    plot_target_vs_achieved_dr(
        rows=rows,
        out_path=plots_dir / "dt_vs_dr_target_alignment.png",
    )

    write_paper_comparison(
        rows=rows,
        out_csv=output_dir / "paper_vs_simulation_overlap_targets.csv",
        out_plot=plots_dir / "paper_vs_simulation_overlap_metrics.png",
        dt07_csv=output_dir / "paper_comparison_dt_0_7.csv",
        dt07_md=output_dir / "paper_comparison_dt_0_7.md",
    )

    print(f"Wrote feedback review artifacts to: {output_dir}")


if __name__ == "__main__":
    main()
