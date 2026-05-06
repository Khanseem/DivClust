import argparse
import csv
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def dt_to_tag(value):
    return str(value).replace(".", "_")


def load_summary(base_dir, target):
    run_name = f"CC_FMNIST_E20_DT_{dt_to_tag(target)}"
    summary_path = os.path.join(base_dir, run_name, "analysis_summary.json")
    with open(summary_path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    data["NMI_target"] = float(target)
    data["run_name"] = run_name
    return data


def write_csv(rows, out_path):
    fieldnames = [
        "NMI_target",
        "cnf",
        "mean_acc",
        "max_acc",
        "mean_nmi",
        "max_nmi",
        "mean_ari",
        "max_ari",
        "interclustering_nmi",
        "divclust_nmi",
        "divclust_acc",
        "divclust_ari",
        "divclust_best_gamma_by_nmi",
        "divclust_connected_components",
        "run_name",
    ]
    with open(out_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})


def write_markdown(rows, out_path):
    headers = [
        "D^T",
        "CNF",
        "Mean ACC",
        "Max ACC",
        "Mean NMI",
        "DivClust NMI",
        "DivClust ACC",
        "DivClust ARI",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "---|" * len(headers),
    ]
    for row in rows:
        lines.append(
            "| {dt:.2f} | {cnf:.4f} | {mean_acc:.4f} | {max_acc:.4f} | {mean_nmi:.4f} | {div_nmi:.4f} | {div_acc:.4f} | {div_ari:.4f} |".format(
                dt=row["NMI_target"],
                cnf=row["cnf"],
                mean_acc=row["mean_acc"],
                max_acc=row["max_acc"],
                mean_nmi=row["mean_nmi"],
                div_nmi=row["divclust_nmi"],
                div_acc=row["divclust_acc"],
                div_ari=row["divclust_ari"],
            )
        )
    with open(out_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def build_combined_plot(rows, out_path):
    targets = [row["NMI_target"] for row in rows]
    metrics = [
        ("cnf", "CNF"),
        ("mean_acc", "Mean ACC"),
        ("max_acc", "Max ACC"),
        ("mean_nmi", "Mean NMI"),
        ("interclustering_nmi", "Inter-clustering NMI"),
        ("divclust_nmi", "DivClust NMI"),
        ("divclust_acc", "DivClust ACC"),
        ("divclust_ari", "DivClust ARI"),
    ]

    fig, axes = plt.subplots(4, 2, figsize=(12, 14))
    axes = axes.reshape(-1)
    for ax, (key, label) in zip(axes, metrics):
        values = [row[key] for row in rows]
        ax.plot(targets, values, marker="o", linewidth=2)
        ax.set_title(label)
        ax.set_xlabel("NMI target")
        ax.set_ylabel("Value")
        ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", required=True, type=str)
    parser.add_argument("--targets", nargs="+", required=True, type=float)
    parser.add_argument("--output-dir", required=True, type=str)
    args = parser.parse_args()

    rows = [load_summary(args.base_dir, target) for target in args.targets]
    rows.sort(key=lambda row: row["NMI_target"])

    os.makedirs(args.output_dir, exist_ok=True)
    write_csv(rows, os.path.join(args.output_dir, "fashionmnist_nmi_sweep_summary.csv"))
    write_markdown(rows, os.path.join(args.output_dir, "fashionmnist_nmi_sweep_summary.md"))
    build_combined_plot(rows, os.path.join(args.output_dir, "fashionmnist_nmi_sweep_metrics.png"))


if __name__ == "__main__":
    main()
