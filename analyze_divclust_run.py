import argparse
import csv
import json
import os
import re
import zipfile
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from scipy.sparse import csr_matrix, hstack, vstack
from scipy.sparse.csgraph import connected_components
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float32

GAMMA_LIST = [2e-05, 3e-05, 4e-05, 5e-05, 6e-05, 7e-05, 8e-05, 1e-04]
MAXITER = 30
LR_W = 0.005
LR_S = 0.005
TOP_EDGES_PER_ROW = 2
KMEANS_ITERS = 50
KMEANS_RESTARTS = 10
SEED = 0


def clustering_accuracy(y_true, y_pred):
    y_true = y_true.astype(np.int64)
    y_pred = y_pred.astype(np.int64)

    dim = max(y_pred.max(), y_true.max()) + 1
    cost = np.zeros((dim, dim), dtype=np.int64)
    for idx in range(y_pred.size):
        cost[y_pred[idx], y_true[idx]] += 1

    row_ind, col_ind = linear_sum_assignment(cost.max() - cost)
    return cost[row_ind, col_ind].sum() / y_pred.size


def load_outcomes(path):
    try:
        data = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        with zipfile.ZipFile(path, "r") as archive:
            candidate_names = ["data.pkl", "outcomes/data.pkl"]
            data_name = next((name for name in candidate_names if name in archive.namelist()), None)
            if data_name is None:
                raise FileNotFoundError(f"Could not find serialized outcomes in {path}")
            with archive.open(data_name) as handle:
                data = pickle.load(handle)

    clusters = np.asarray(data["clusters"], dtype=np.int64)
    labels = np.asarray(data["ground_truth"], dtype=np.int64)
    return clusters, labels


def extract_final_eval_confidence(log_path):
    pattern = re.compile(r"eval_confidence=([0-9]*\.?[0-9]+)")
    last_conf = None
    with open(log_path, "r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            match = pattern.search(line)
            if match:
                last_conf = float(match.group(1))
    if last_conf is None:
        raise ValueError(f"No eval_confidence found in {log_path}")
    return last_conf


def extract_final_head_losses(log_path, num_heads):
    pattern = re.compile(r"loss_main_head_(\d+)=([0-9eE+\-.]+)")
    last_vals = {}
    with open(log_path, "r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            matches = pattern.findall(line)
            if matches:
                current = {int(head): float(value) for head, value in matches}
                if current:
                    last_vals = current

    if not last_vals:
        raise ValueError(f"No loss_main_head_k values found in {log_path}")

    missing = [head for head in range(num_heads) if head not in last_vals]
    if missing:
        raise ValueError(f"Missing head losses for heads {missing}")

    return np.array([last_vals[idx] for idx in range(num_heads)], dtype=float)


def extract_epoch_metrics(log_path):
    patterns = {
        "eval_confidence": re.compile(r"eval_confidence=([0-9]*\.?[0-9]+)"),
        "mean_cluster_acc_eval": re.compile(r"mean_cluster_acc_eval=([0-9]*\.?[0-9]+)"),
        "max_cluster_acc_eval": re.compile(r"max_cluster_acc_eval=([0-9]*\.?[0-9]+)"),
        "mean_cluster_nmi_eval": re.compile(r"mean_cluster_nmi_eval=([0-9]*\.?[0-9]+)"),
        "max_cluster_nmi_eval": re.compile(r"max_cluster_nmi_eval=([0-9]*\.?[0-9]+)"),
        "mean_cluster_ari_eval": re.compile(r"mean_cluster_ari_eval=([0-9]*\.?[0-9]+)"),
        "max_cluster_ari_eval": re.compile(r"max_cluster_ari_eval=([0-9]*\.?[0-9]+)"),
        "interclustering_nmi_eval": re.compile(r"interclustering_nmi_eval=([0-9]*\.?[0-9]+)"),
    }
    epoch_pattern = re.compile(r"\|\s+Ep\.\s+(\d+)/(\d+)\s+\|")

    rows = []
    with open(log_path, "r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            epoch_match = epoch_pattern.search(line)
            if not epoch_match or "mean_cluster_acc_eval" not in line:
                continue

            row = {
                "epoch": int(epoch_match.group(1)),
                "epochs_total": int(epoch_match.group(2)),
            }
            for key, pattern in patterns.items():
                match = pattern.search(line)
                row[key] = float(match.group(1)) if match else None
            rows.append(row)
    return rows


def compute_head_metrics(clusters, labels):
    acc_list = []
    nmi_list = []
    ari_list = []
    pairwise_nmi = []

    num_heads = clusters.shape[0]
    for i in range(num_heads):
        pred = clusters[i]
        acc_list.append(clustering_accuracy(labels, pred))
        nmi_list.append(normalized_mutual_info_score(labels, pred))
        ari_list.append(adjusted_rand_score(labels, pred))
        for j in range(i + 1, num_heads):
            pairwise_nmi.append(normalized_mutual_info_score(clusters[i], clusters[j]))

    return {
        "mean_acc": float(np.mean(acc_list)),
        "max_acc": float(np.max(acc_list)),
        "mean_nmi": float(np.mean(nmi_list)),
        "max_nmi": float(np.max(nmi_list)),
        "mean_ari": float(np.mean(ari_list)),
        "max_ari": float(np.max(ari_list)),
        "inter_nmi": float(np.mean(pairwise_nmi)),
    }


def hard_labels_to_onehot_torch(pred, n_clusters):
    pred_t = torch.as_tensor(pred, device=DEVICE, dtype=torch.long)
    return torch.nn.functional.one_hot(pred_t, num_classes=n_clusters).to(DTYPE)


def row_topk_sparsify_and_normalize(similarity, topk=2):
    vals, idx = torch.topk(similarity, k=min(topk, similarity.shape[1]), dim=1)
    updated = torch.zeros_like(similarity)
    updated.scatter_(1, idx, vals)
    updated = torch.clamp(updated, min=0.0)
    row_sum = updated.sum(dim=1, keepdim=True)
    return updated / torch.clamp(row_sum, min=1e-12)


def pairwise_sqdist_rows(x_mat, y_mat):
    x2 = (x_mat * x_mat).sum(dim=1, keepdim=True)
    y2 = (y_mat * y_mat).sum(dim=1, keepdim=True).T
    xy = x_mat @ y_mat.T
    return torch.clamp(x2 + y2 - 2 * xy, min=0.0)


def gpu_kmeans(x_mat, n_clusters, num_iters=25, num_restarts=3):
    num_rows = x_mat.shape[0]
    best_labels = None
    best_inertia = None

    for restart in range(num_restarts):
        torch.manual_seed(SEED + restart)
        perm = torch.randperm(num_rows, device=x_mat.device)
        centers = x_mat[perm[:n_clusters]].clone()

        for _ in range(num_iters):
            dist = pairwise_sqdist_rows(x_mat, centers)
            labels = torch.argmin(dist, dim=1)

            new_centers = []
            for cluster_idx in range(n_clusters):
                mask = labels == cluster_idx
                if mask.any():
                    new_centers.append(x_mat[mask].mean(dim=0))
                else:
                    new_centers.append(x_mat[torch.randint(0, num_rows, (1,), device=x_mat.device)].squeeze(0))
            centers = torch.stack(new_centers, dim=0)

        dist = pairwise_sqdist_rows(x_mat, centers)
        labels = torch.argmin(dist, dim=1)
        inertia = dist[torch.arange(num_rows, device=x_mat.device), labels].sum().item()

        if best_inertia is None or inertia < best_inertia:
            best_inertia = inertia
            best_labels = labels.clone()

    return best_labels.detach().cpu().numpy()


def connected_components_bipartite_from_s(similarity, thresh=1e-6):
    similarity_np = (similarity.detach().cpu().numpy() > thresh).astype(np.int8)
    n_rows, n_cols = similarity_np.shape

    sparse = csr_matrix(similarity_np)
    upper = hstack([csr_matrix((n_rows, n_rows)), sparse], format="csr")
    lower = hstack([sparse.T, csr_matrix((n_cols, n_cols))], format="csr")
    graph = vstack([upper, lower], format="csr")

    n_components, labels_all = connected_components(graph, directed=False, return_labels=True)
    return labels_all[:n_rows], n_components


def stabilized_sccbg_gpu(y_list, n_clusters, gamma):
    s0 = torch.cat(y_list, dim=1).to(device=DEVICE, dtype=DTYPE)
    similarity = s0.clone()
    spectral = s0.clone()

    n_rows, n_cols = similarity.shape
    affinity = s0.T @ s0
    weights = torch.zeros((n_rows, n_cols), device=DEVICE, dtype=DTYPE)

    lam = 1.0
    rho = 0.5

    for _ in range(MAXITER):
        sim_affinity = similarity @ affinity
        grad_w = 2.0 * gamma * weights * torch.clamp(sim_affinity, min=0.0)
        grad_w += 2.0 * weights * ((similarity - spectral) ** 2)
        grad_w -= rho
        weights = torch.clamp(weights - LR_W * grad_w, 0.0, 1.0)

        d1 = 1.0 / torch.sqrt(torch.clamp(similarity.sum(dim=1), min=1e-12))
        d2 = 1.0 / torch.sqrt(torch.clamp(similarity.sum(dim=0), min=1e-12))

        spec_1 = d1.unsqueeze(1) * similarity * d2.unsqueeze(0)
        spec_2 = spec_1.T @ spec_1

        evals, evecs = torch.linalg.eigh(spec_2)
        order = torch.argsort(evals, descending=True)
        evals = evals[order]
        evecs = evecs[:, order]

        v_emb = evecs[:, :n_clusters]
        ev0 = evals[:n_clusters]
        denom = torch.sqrt(torch.clamp(ev0, min=1e-12))
        u_mat = (spec_1 @ v_emb) / denom.unsqueeze(0)

        fn1 = evals[:n_clusters].sum().item()
        fn2 = evals[:min(n_clusters + 1, len(evals))].sum().item()
        if fn1 < n_clusters - 1e-7:
            lam *= 2.0
        elif fn2 > n_clusters + 1 - 1e-7:
            lam /= 2.0

        u1 = d1.unsqueeze(1) * u_mat
        v1 = d2.unsqueeze(1) * v_emb
        dist = pairwise_sqdist_rows(u1, v1)

        ww = weights.unsqueeze(2) * weights.unsqueeze(1) * affinity.unsqueeze(0)
        rowsum_ww = ww.sum(dim=2)
        ww_similarity = torch.bmm(ww, similarity.unsqueeze(2)).squeeze(2)

        grad_s = 2.0 * (weights * weights) * (similarity - spectral)
        grad_s += 2.0 * gamma * (rowsum_ww * similarity - ww_similarity)
        grad_s += lam * dist

        similarity = torch.clamp(similarity - LR_S * grad_s, 0.0, 1.0)
        similarity = row_topk_sparsify_and_normalize(similarity, topk=TOP_EDGES_PER_ROW)
        rho *= 2.0

    _, n_components = connected_components_bipartite_from_s(similarity, thresh=1e-5)
    final_labels = gpu_kmeans(u1, n_clusters=n_clusters, num_iters=KMEANS_ITERS, num_restarts=KMEANS_RESTARTS)
    return final_labels, n_components


def compute_metrics(labels, pred):
    return {
        "acc": clustering_accuracy(labels, pred),
        "nmi": normalized_mutual_info_score(labels, pred),
        "ari": adjusted_rand_score(labels, pred),
    }


def divclust_c_gpu_consensus(clusters, head_losses, labels):
    top_k_heads = min(10, clusters.shape[0])
    n_clusters = len(np.unique(labels))
    selected_idx = np.argsort(head_losses)[:top_k_heads]
    selected = clusters[selected_idx]
    y_list = [hard_labels_to_onehot_torch(selected[idx], n_clusters) for idx in range(top_k_heads)]

    all_results = []
    for gamma in GAMMA_LIST:
        pred, n_components = stabilized_sccbg_gpu(y_list, n_clusters, gamma)
        metrics = compute_metrics(labels, pred)
        all_results.append(
            {
                "gamma": gamma,
                "labels": pred,
                "components": n_components,
                "acc": metrics["acc"],
                "nmi": metrics["nmi"],
                "ari": metrics["ari"],
            }
        )
    return selected_idx, all_results


def build_nmi_heatmap(clusters, out_path):
    num_heads = clusters.shape[0]
    matrix = np.zeros((num_heads, num_heads), dtype=float)
    for i in range(num_heads):
        for j in range(num_heads):
            matrix[i, j] = normalized_mutual_info_score(clusters[i], clusters[j])

    avg_offdiag = (matrix.sum() - np.trace(matrix)) / (num_heads * (num_heads - 1))

    fig, ax = plt.subplots(figsize=(6.4, 6.4))
    coords = np.arange(num_heads + 1)
    mesh = ax.pcolormesh(
        coords,
        coords,
        matrix,
        cmap="viridis",
        vmin=0,
        vmax=1,
        edgecolors="black",
        linewidth=0.8,
        shading="flat",
    )
    ax.set_xlim(0, num_heads)
    ax.set_ylim(num_heads, 0)
    ax.set_aspect("equal")
    colorbar = plt.colorbar(mesh, ax=ax, shrink=0.75, pad=0.04)
    colorbar.set_ticks(np.linspace(0, 1, 6))
    ax.set_xlabel("Head")
    ax.set_ylabel("Head")
    ax.set_title(f"Inter-clustering similarity (NMI)\nAvg off-diagonal = {avg_offdiag:.3f}")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return avg_offdiag


def build_accuracy_curve(epoch_rows, out_path):
    epochs = [row["epoch"] + 1 for row in epoch_rows]
    mean_acc = [row["mean_cluster_acc_eval"] for row in epoch_rows]
    max_acc = [row["max_cluster_acc_eval"] for row in epoch_rows]

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    ax.plot(epochs, mean_acc, marker="o", linewidth=2, label="Mean ACC")
    ax.plot(epochs, max_acc, marker="s", linewidth=2, label="Max ACC")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy vs Epoch")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_epoch_metrics_csv(epoch_rows, out_path):
    fieldnames = [
        "epoch",
        "epochs_total",
        "eval_confidence",
        "mean_cluster_acc_eval",
        "max_cluster_acc_eval",
        "mean_cluster_nmi_eval",
        "max_cluster_nmi_eval",
        "mean_cluster_ari_eval",
        "max_cluster_ari_eval",
        "interclustering_nmi_eval",
    ]
    with open(out_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(epoch_rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-dir", required=True, type=str)
    args = parser.parse_args()

    exp_dir = os.path.abspath(args.experiment_dir)
    outcomes_path = os.path.join(exp_dir, "outcomes")
    log_path = os.path.join(exp_dir, "log.txt")

    clusters, labels = load_outcomes(outcomes_path)
    epoch_rows = extract_epoch_metrics(log_path)
    cnf = extract_final_eval_confidence(log_path)
    head_losses = extract_final_head_losses(log_path, clusters.shape[0])
    head_metrics = compute_head_metrics(clusters, labels)
    _, consensus_results = divclust_c_gpu_consensus(clusters, head_losses, labels)
    best_by_nmi = max(consensus_results, key=lambda item: item["nmi"])

    heatmap_path = os.path.join(exp_dir, "simulation_heatmap.png")
    accuracy_curve_path = os.path.join(exp_dir, "accuracy_curve.png")
    epoch_csv_path = os.path.join(exp_dir, "epoch_metrics.csv")
    summary_json_path = os.path.join(exp_dir, "analysis_summary.json")
    summary_txt_path = os.path.join(exp_dir, "analysis_summary.txt")

    inter_nmi_from_heatmap = build_nmi_heatmap(clusters, heatmap_path)
    build_accuracy_curve(epoch_rows, accuracy_curve_path)
    write_epoch_metrics_csv(epoch_rows, epoch_csv_path)

    summary = {
        "experiment_dir": exp_dir,
        "cnf": cnf,
        "mean_acc": head_metrics["mean_acc"],
        "max_acc": head_metrics["max_acc"],
        "mean_nmi": head_metrics["mean_nmi"],
        "max_nmi": head_metrics["max_nmi"],
        "mean_ari": head_metrics["mean_ari"],
        "max_ari": head_metrics["max_ari"],
        "interclustering_nmi": head_metrics["inter_nmi"],
        "simulation_heatmap_interclustering_nmi": inter_nmi_from_heatmap,
        "divclust_nmi": best_by_nmi["nmi"],
        "divclust_acc": best_by_nmi["acc"],
        "divclust_ari": best_by_nmi["ari"],
        "divclust_best_gamma_by_nmi": best_by_nmi["gamma"],
        "divclust_connected_components": best_by_nmi["components"],
        "num_heads": int(clusters.shape[0]),
        "epochs_logged": len(epoch_rows),
    }

    with open(summary_json_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)

    with open(summary_txt_path, "w", encoding="utf-8") as handle:
        handle.write("===== Experiment Summary =====\n")
        for key in sorted(summary.keys()):
            handle.write(f"{key}: {summary[key]}\n")

    np.save(os.path.join(exp_dir, "divclust_labels_best_nmi.npy"), best_by_nmi["labels"])


if __name__ == "__main__":
    main()
