#!/usr/bin/env python3
import argparse
import csv
import math
import pickle
import re
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


PLOT_HEATMAP_COLUMNS = [
    "CNF (%)",
    "Mean_ACC (%)",
    "Max_ACC (%)",
    "Mean_NMI (%)",
    "D^R (%)",
    "DivClust_NMI (%)",
    "DivClust_ACC (%)",
    "DivClust_ARI (%)",
]
PLOT_HEATMAP_PERCENT_SCALES = np.array([100.0, 1.0, 1.0, 100.0, 100.0, 100.0, 1.0, 100.0], dtype=np.float64)


@dataclass
class SweepResult:
    dt: float
    run_name: str
    status: str
    cnf: float = math.nan
    mean_acc: float = math.nan
    max_acc: float = math.nan
    mean_nmi: float = math.nan
    inter_nmi_dr: float = math.nan
    divclust_nmi: float = math.nan
    divclust_acc: float = math.nan
    divclust_ari: float = math.nan
    runtime_minutes: float = math.nan
    runtime_hhmmss: str = "NA"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize FMNIST C10CFG K20 E100 sweep outputs")
    parser.add_argument(
        "--artifact-dir",
        type=str,
        default=str(Path(__file__).resolve().parents[1]),
        help="Artifact directory",
    )
    parser.add_argument(
        "--targets",
        type=str,
        default="0.5,0.6,0.7,0.8,0.9,0.95,1.0",
        help="Comma-separated D^T values",
    )
    parser.add_argument("--epochs", type=int, default=100, help="Expected training epochs")
    return parser.parse_args()


def dt_to_safe(dt: float) -> str:
    txt = f"{dt}"
    if "." in txt:
        txt = txt.rstrip("0").rstrip(".")
    if "." not in txt:
        txt = txt + ".0"
    return txt.replace(".", "_")


def run_name_from_dt(dt: float) -> str:
    return f"CC_FMNIST_C10CFG_K20_E100_DT_{dt_to_safe(dt)}"


def log_name_from_dt(dt: float) -> str:
    return f"fmnist_c10cfg_k20_e100_dt_{dt_to_safe(dt)}.out"


def parse_epoch_metrics(log_path: Path, expected_epochs: int) -> Dict[int, Dict[str, float]]:
    rows: Dict[int, Dict[str, float]] = {}
    if not log_path.exists():
        return rows

    metric_pattern = re.compile(r"([A-Za-z0-9_]+)=(-?[0-9]+(?:\.[0-9]+)?(?:e-?[0-9]+)?)")
    epoch_pattern = re.compile(r"\| Ep\.\s*(\d+)/(\d+)\s*\|")

    with log_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if "| Ep." not in line:
                continue
            epoch_match = epoch_pattern.search(line)
            if not epoch_match:
                continue
            epoch = int(epoch_match.group(1))
            total = int(epoch_match.group(2))
            if total != expected_epochs:
                continue
            metrics: Dict[str, float] = {}
            for match in metric_pattern.finditer(line):
                metrics[match.group(1)] = float(match.group(2))
            if metrics:
                rows[epoch] = metrics
    return rows


def parse_runtime(log_path: Path) -> Tuple[float, str]:
    if not log_path.exists():
        return math.nan, "NA"

    start_ts: Optional[datetime] = None
    end_ts: Optional[datetime] = None
    # Example: START Sun Mar 29 03:08:19 PM +08 2026 DT=0.7 RUN_NAME=...
    pattern = re.compile(r"^(START|END)\s+([A-Za-z]{3}\s+[A-Za-z]{3}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2}\s+[AP]M\s+[+-]\d{2}\s+\d{4})")

    with log_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            match = pattern.match(line.strip())
            if not match:
                continue
            stamp = match.group(2)
            # Some PBS logs encode timezone as "+08" instead of "+0800".
            stamp_norm = re.sub(r" ([+-]\d{2}) ", r" \g<1>00 ", stamp)
            dt = datetime.strptime(stamp_norm, "%a %b %d %I:%M:%S %p %z %Y")
            if match.group(1) == "START":
                start_ts = dt
            else:
                end_ts = dt

    if start_ts is None or end_ts is None:
        return math.nan, "NA"

    duration = end_ts - start_ts
    minutes = duration.total_seconds() / 60.0
    total_sec = int(duration.total_seconds())
    hh = total_sec // 3600
    mm = (total_sec % 3600) // 60
    ss = total_sec % 60
    return minutes, f"{hh:02d}:{mm:02d}:{ss:02d}"


def load_outcomes(outcomes_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    if not outcomes_path.exists():
        raise FileNotFoundError(f"Missing outcomes: {outcomes_path}")

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
    labels = np.asarray(obj["ground_truth"], dtype=np.int64)
    if clusters.ndim == 1:
        clusters = np.expand_dims(clusters, axis=0)
    return clusters, labels


def clustering_acc(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    y_pred = y_pred.astype(np.int64)
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    dim = int(max(y_pred.max(), y_true.max()) + 1)
    w = np.zeros((dim, dim), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    row_ind, col_ind = linear_sum_assignment(w.max() - w)
    return float(w[row_ind, col_ind].sum() * 100.0 / y_pred.size)


def remap_to_base(labels: np.ndarray, base: np.ndarray) -> np.ndarray:
    labels = labels.astype(np.int64)
    base = base.astype(np.int64)
    dim = int(max(labels.max(), base.max()) + 1)
    w = np.zeros((dim, dim), dtype=np.int64)
    for i in range(labels.size):
        w[labels[i], base[i]] += 1
    row_ind, col_ind = linear_sum_assignment(w.max() - w)
    mapping = {int(r): int(c) for r, c in zip(row_ind, col_ind)}
    return np.array([mapping.get(int(v), int(v)) for v in labels], dtype=np.int64)


def consensus_labels(clusters: np.ndarray) -> np.ndarray:
    if clusters.shape[0] == 1:
        return clusters[0]
    base = clusters[0]
    aligned = [base]
    for idx in range(1, clusters.shape[0]):
        aligned.append(remap_to_base(clusters[idx], base))
    arr = np.stack(aligned, axis=0)
    n = arr.shape[1]
    out = np.zeros(n, dtype=np.int64)
    for i in range(n):
        vals, counts = np.unique(arr[:, i], return_counts=True)
        out[i] = vals[np.argmax(counts)]
    return out


def pairwise_nmi_matrix(clusters: np.ndarray) -> np.ndarray:
    heads = clusters.shape[0]
    matrix = np.zeros((heads, heads), dtype=np.float64)
    for i in range(heads):
        for j in range(heads):
            matrix[i, j] = normalized_mutual_info_score(clusters[i], clusters[j])
    return matrix


def inter_nmi_avg_offdiag(matrix: np.ndarray) -> float:
    if matrix.shape[0] <= 1:
        return float(matrix[0, 0])
    denom = matrix.shape[0] * (matrix.shape[0] - 1)
    return float((matrix.sum() - np.trace(matrix)) / denom)


def plot_interclustering_heatmap(matrix: np.ndarray, dt: float, dr: float, out_path: Path) -> None:
    heads = matrix.shape[0]
    fig, ax = plt.subplots(figsize=(6.4, 6.1))
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
    ax.set_xticks(np.arange(heads) + 0.5)
    ax.set_yticks(np.arange(heads) + 0.5)
    ax.set_xticklabels([str(i) for i in range(heads)])
    ax.set_yticklabels([str(i) for i in range(heads)])
    ax.set_xlabel("Head")
    ax.set_ylabel("Head")
    ax.set_title(f"Inter-clustering Similarity (NMI)\nD^T={dt:.2f}, D^R={dr:.3f}")
    cbar = plt.colorbar(mesh, ax=ax, shrink=0.78, pad=0.04)
    cbar.set_label("NMI")
    cbar.set_ticks(np.linspace(0, 1, 6))
    plt.tight_layout()
    plt.savefig(out_path, dpi=260, bbox_inches="tight")
    plt.close(fig)


def build_results(artifact_dir: Path, targets: List[float], epochs: int) -> Tuple[List[SweepResult], Dict[float, List[Tuple[int, float]]]]:
    results: List[SweepResult] = []
    curves: Dict[float, List[Tuple[int, float]]] = {}

    logs_dir = artifact_dir / "results" / "logs"
    outcomes_dir = artifact_dir / "results" / "outcomes"
    plots_dir = artifact_dir / "results" / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    for dt in targets:
        run_name = run_name_from_dt(dt)
        log_path = logs_dir / log_name_from_dt(dt)
        outcomes_path = outcomes_dir / f"{run_name}.outcomes.pt"

        runtime_m, runtime_hhmmss = parse_runtime(log_path)
        epoch_metrics = parse_epoch_metrics(log_path, epochs)
        curve = []
        for ep in sorted(epoch_metrics.keys()):
            row = epoch_metrics[ep]
            if "mean_cluster_acc_eval" in row:
                curve.append((ep, row["mean_cluster_acc_eval"]))
        curves[dt] = curve

        final_epoch = epochs - 1
        if not log_path.exists():
            results.append(SweepResult(dt=dt, run_name=run_name, status="missing_log"))
            continue
        if final_epoch not in epoch_metrics:
            status = "incomplete"
            if epoch_metrics:
                status = f"incomplete_ep_{max(epoch_metrics.keys())}"
            results.append(
                SweepResult(
                    dt=dt,
                    run_name=run_name,
                    status=status,
                    runtime_minutes=runtime_m,
                    runtime_hhmmss=runtime_hhmmss,
                )
            )
            continue
        if not outcomes_path.exists():
            results.append(
                SweepResult(
                    dt=dt,
                    run_name=run_name,
                    status="missing_outcomes",
                    runtime_minutes=runtime_m,
                    runtime_hhmmss=runtime_hhmmss,
                )
            )
            continue

        final = epoch_metrics[final_epoch]
        clusters, labels = load_outcomes(outcomes_path)
        matrix = pairwise_nmi_matrix(clusters)
        dr = inter_nmi_avg_offdiag(matrix)
        plot_interclustering_heatmap(
            matrix,
            dt=dt,
            dr=dr,
            out_path=plots_dir / f"inter_cluster_nmi_heatmap_dt_{dt_to_safe(dt)}.png",
        )

        consensus = consensus_labels(clusters)
        div_acc = clustering_acc(consensus, labels)
        div_nmi = float(normalized_mutual_info_score(consensus, labels))
        div_ari = float(adjusted_rand_score(labels, consensus))

        results.append(
            SweepResult(
                dt=dt,
                run_name=run_name,
                status="complete",
                cnf=final.get("eval_confidence", math.nan),
                mean_acc=final.get("mean_cluster_acc_eval", math.nan),
                max_acc=final.get("max_cluster_acc_eval", math.nan),
                mean_nmi=final.get("mean_cluster_nmi_eval", math.nan),
                inter_nmi_dr=dr,
                divclust_nmi=div_nmi,
                divclust_acc=div_acc,
                divclust_ari=div_ari,
                runtime_minutes=runtime_m,
                runtime_hhmmss=runtime_hhmmss,
            )
        )

    return results, curves


def save_tables(artifact_dir: Path, rows: List[SweepResult]) -> None:
    out_csv = artifact_dir / "results" / "fmnist_sweep_summary.csv"
    out_md = artifact_dir / "results" / "fmnist_sweep_summary.md"
    runtime_csv = artifact_dir / "results" / "runtime_summary.csv"

    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "NMI_target",
                "run_name",
                "status",
                "CNF",
                "mean_accuracy",
                "max_accuracy",
                "mean_NMI",
                "interclustering_NMI_D_R",
                "DivClust_NMI",
                "DivClust_ACC",
                "DivClust_ARI",
                "runtime_minutes",
                "runtime_hhmmss",
            ]
        )
        for r in rows:
            writer.writerow(
                [
                    r.dt,
                    r.run_name,
                    r.status,
                    r.cnf,
                    r.mean_acc,
                    r.max_acc,
                    r.mean_nmi,
                    r.inter_nmi_dr,
                    r.divclust_nmi,
                    r.divclust_acc,
                    r.divclust_ari,
                    r.runtime_minutes,
                    r.runtime_hhmmss,
                ]
            )

    md_lines = []
    md_lines.append("| D^T | status | CNF | Mean ACC | Max ACC | Mean NMI | D^R | DivClust NMI | DivClust ACC | DivClust ARI | Runtime |")
    md_lines.append("|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        md_lines.append(
            f"| {r.dt:.2f} | {r.status} | {r.cnf:.5f} | {r.mean_acc:.3f} | {r.max_acc:.3f} | {r.mean_nmi:.5f} | {r.inter_nmi_dr:.5f} | {r.divclust_nmi:.5f} | {r.divclust_acc:.3f} | {r.divclust_ari:.5f} | {r.runtime_hhmmss} |"
        )
    out_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    with runtime_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["NMI_target", "run_name", "runtime_hhmmss", "runtime_minutes"])
        for r in rows:
            writer.writerow([r.dt, r.run_name, r.runtime_hhmmss, r.runtime_minutes])


def plot_accuracy_vs_epoch(artifact_dir: Path, curves: Dict[float, List[Tuple[int, float]]]) -> None:
    out_path = artifact_dir / "results" / "plots" / "accuracy_vs_epoch.png"
    plt.figure(figsize=(10.0, 6.0))
    for dt in sorted(curves.keys()):
        points = curves[dt]
        if not points:
            continue
        xs = [ep for ep, _ in points]
        ys = [val for _, val in points]
        plt.plot(xs, ys, label=f"D^T={dt}", linewidth=1.8)
    plt.title("Fashion-MNIST (CIFAR-config, K=20): Mean Accuracy vs Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Mean Cluster Accuracy (%)")
    plt.grid(alpha=0.25)
    if any(curves.values()):
        plt.legend(ncol=2, fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def plot_all_metrics_graph(artifact_dir: Path, rows: List[SweepResult]) -> None:
    out_path = artifact_dir / "results" / "plots" / "all_metrics_same_graph.png"
    dts = [r.dt for r in rows if r.status == "complete"]
    if not dts:
        return
    metrics = [
        ("CNF", [r.cnf for r in rows if r.status == "complete"]),
        ("Mean ACC (%)", [r.mean_acc for r in rows if r.status == "complete"]),
        ("Max ACC (%)", [r.max_acc for r in rows if r.status == "complete"]),
        ("Mean NMI", [r.mean_nmi for r in rows if r.status == "complete"]),
        ("D^R (Inter-clustering NMI)", [r.inter_nmi_dr for r in rows if r.status == "complete"]),
        ("DivClust NMI", [r.divclust_nmi for r in rows if r.status == "complete"]),
        ("DivClust ACC (%)", [r.divclust_acc for r in rows if r.status == "complete"]),
        ("DivClust ARI", [r.divclust_ari for r in rows if r.status == "complete"]),
    ]
    fig, axes = plt.subplots(4, 2, figsize=(12.5, 14.0))
    axes = axes.reshape(-1)
    for ax, (label, values) in zip(axes, metrics):
        ax.plot(dts, values, marker="o", linewidth=2)
        ax.set_title(label)
        ax.set_xlabel("NMI target (D^T)")
        ax.set_ylabel("Value")
        ax.grid(alpha=0.25, linestyle="--")
    plt.tight_layout()
    plt.savefig(out_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_dt_vs_dr(artifact_dir: Path, rows: List[SweepResult]) -> None:
    out_path = artifact_dir / "results" / "plots" / "dt_vs_dr_target_alignment.png"
    completed = [r for r in rows if r.status == "complete"]
    if not completed:
        return
    x = [r.dt for r in completed]
    y = [r.inter_nmi_dr for r in completed]
    plt.figure(figsize=(8.3, 5.0))
    plt.plot(x, x, linestyle="--", linewidth=2, label="Target D^T")
    plt.plot(x, y, marker="o", linewidth=2, label="Achieved D^R")
    for xi, yi in zip(x, y):
        plt.text(xi, yi + 0.01, f"{yi:.3f}", ha="center", fontsize=8)
    plt.xlabel("NMI target (D^T)")
    plt.ylabel("Inter-clustering NMI (D^R)")
    plt.title("Target vs Achieved Inter-clustering NMI")
    plt.xticks(x)
    plt.grid(alpha=0.25, linestyle="--")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=240, bbox_inches="tight")
    plt.close()


def plot_normalized_heatmap(artifact_dir: Path, rows: List[SweepResult]) -> None:
    out_path = artifact_dir / "results" / "plots" / "simulation_heatmap.png"
    completed = [r for r in rows if r.status == "complete"]
    if not completed:
        return

    raw = np.array(
        [
            [
                r.cnf,
                r.mean_acc,
                r.max_acc,
                r.mean_nmi,
                r.inter_nmi_dr,
                r.divclust_nmi,
                r.divclust_acc,
                r.divclust_ari,
            ]
            for r in completed
        ],
        dtype=np.float64,
    )
    matrix = raw * PLOT_HEATMAP_PERCENT_SCALES.reshape(1, -1)
    ylabels = [f"D^T={r.dt}" for r in completed]

    plt.figure(figsize=(12, max(4.0, 0.55 * len(completed))))
    im = plt.imshow(matrix, aspect="auto", cmap="viridis")
    cbar = plt.colorbar(im, fraction=0.04, pad=0.02)
    cbar.set_label("Value (%)")
    plt.xticks(np.arange(len(PLOT_HEATMAP_COLUMNS)), PLOT_HEATMAP_COLUMNS, rotation=22, ha="right")
    plt.yticks(np.arange(len(ylabels)), ylabels)
    plt.title("FMNIST C10CFG K20 E100: Simulation Heatmap (all metrics as %)")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            v = matrix[i, j]
            txt = "nan" if np.isnan(v) else f"{v:.1f}"
            plt.text(j, i, txt, ha="center", va="center", fontsize=8, color="white")
    plt.tight_layout()
    plt.savefig(out_path, dpi=230, bbox_inches="tight")
    plt.close()


def main() -> None:
    args = parse_args()
    artifact_dir = Path(args.artifact_dir).resolve()
    targets = [float(x.strip()) for x in args.targets.split(",") if x.strip()]
    targets = sorted(targets)

    rows, curves = build_results(artifact_dir=artifact_dir, targets=targets, epochs=args.epochs)
    save_tables(artifact_dir=artifact_dir, rows=rows)
    plot_accuracy_vs_epoch(artifact_dir=artifact_dir, curves=curves)
    plot_all_metrics_graph(artifact_dir=artifact_dir, rows=rows)
    plot_dt_vs_dr(artifact_dir=artifact_dir, rows=rows)
    plot_normalized_heatmap(artifact_dir=artifact_dir, rows=rows)

    print(f"Generated summary and plots in: {artifact_dir / 'results'}")


if __name__ == "__main__":
    main()
