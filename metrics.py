import os
import torch
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from sklearn.cluster import KMeans

BASE = "/scratch/e1536052/DivClust/experiments/deep_clustering/CC_dt_1.0"
OUTCOME_FILE = f"{BASE}/outcomes"
LOG_FILE = f"{BASE}/metrics_log.txt"

N_CLUSTERS = 10
TOP_K_HEADS = 10


def log(msg):
    print(msg)
    with open(LOG_FILE, "a") as f:
        f.write(msg + "\n")


def clustering_accuracy(y_true, y_pred):

    y_true = y_true.astype(np.int64)
    y_pred = y_pred.astype(np.int64)

    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)

    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1

    row_ind, col_ind = linear_sum_assignment(w.max() - w)

    return w[row_ind, col_ind].sum() / y_pred.size


def load_outcomes():

    data = torch.load(OUTCOME_FILE, map_location="cpu", weights_only=False)

    clusters = np.asarray(data["clusters"])
    labels = np.asarray(data["ground_truth"])

    return clusters, labels


def compute_head_metrics(clusters, labels):

    acc_list = []
    nmi_list = []
    ari_list = []
    pairwise_nmi = []

    num_heads = clusters.shape[0]

    for i in range(num_heads):

        pred = clusters[i]

        acc = clustering_accuracy(labels, pred)
        nmi = normalized_mutual_info_score(labels, pred)
        ari = adjusted_rand_score(labels, pred)

        acc_list.append(acc)
        nmi_list.append(nmi)
        ari_list.append(ari)

        for j in range(i + 1, num_heads):
            pairwise_nmi.append(
                normalized_mutual_info_score(clusters[i], clusters[j])
            )

    return {
        "mean_acc": np.mean(acc_list),
        "max_acc": np.max(acc_list),
        "mean_nmi": np.mean(nmi_list),
        "max_nmi": np.max(nmi_list),
        "mean_ari": np.mean(ari_list),
        "max_ari": np.max(ari_list),
        "inter_nmi": np.mean(pairwise_nmi),
        "per_head_acc": acc_list
    }


def fast_consensus(clusters, head_acc):

    # select best heads
    idx = np.argsort(head_acc)[::-1][:TOP_K_HEADS]
    selected = clusters[idx]

    # shape -> (N samples, K heads)
    X = selected.T

    # run KMeans consensus
    model = KMeans(
        n_clusters=N_CLUSTERS,
        n_init=20,
        random_state=0
    )

    consensus_labels = model.fit_predict(X)

    return consensus_labels, idx


def compute_consensus_metrics(labels, consensus):

    return {
        "acc": clustering_accuracy(labels, consensus),
        "nmi": normalized_mutual_info_score(labels, consensus),
        "ari": adjusted_rand_score(labels, consensus)
    }


def main():

    if os.path.exists(LOG_FILE):
        os.remove(LOG_FILE)

    clusters, labels = load_outcomes()

    head_metrics = compute_head_metrics(clusters, labels)

    log("===== Base-head Metrics =====")
    log(f"CNF (mean NMI)         : {head_metrics['mean_nmi']:.4f}")
    log(f"Max NMI                : {head_metrics['max_nmi']:.4f}")
    log("")
    log(f"Mean ACC               : {head_metrics['mean_acc']:.4f}")
    log(f"Max ACC                : {head_metrics['max_acc']:.4f}")
    log("")
    log(f"Mean ARI               : {head_metrics['mean_ari']:.4f}")
    log(f"Max ARI                : {head_metrics['max_ari']:.4f}")
    log("")
    log(f"Inter-clustering NMI   : {head_metrics['inter_nmi']:.4f}")

    log("")
    log("===== FAST DivClust Consensus =====")

    consensus, idx = fast_consensus(clusters, head_metrics["per_head_acc"])

    metrics = compute_consensus_metrics(labels, consensus)

    log(f"Selected heads         : {list(idx)}")
    log(f"Consensus NMI          : {metrics['nmi']:.4f}")
    log(f"Consensus ACC          : {metrics['acc']:.4f}")
    log(f"Consensus ARI          : {metrics['ari']:.4f}")

    np.save(f"{BASE}/fast_consensus_labels.npy", consensus)

    log("")
    log(f"Saved labels to        : {BASE}/fast_consensus_labels.npy")


if __name__ == "__main__":
    main()