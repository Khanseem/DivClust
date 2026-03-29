import os

# ------------------------------------------------------------
# Limit CPU thread oversubscription
# ------------------------------------------------------------
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["BLIS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import re
import random
import torch
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.sparse import csr_matrix, hstack, vstack
from scipy.sparse.csgraph import connected_components
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score

# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------
BASE = "/scratch/e1536052/DivClust/experiments/deep_clustering/CC_dt_0.8_rerun"
OUTCOME_FILE = f"{BASE}/outcomes"
TRAIN_LOG_FILE = f"{BASE}/log.txt"
METRICS_LOG_FILE = f"{BASE}/metrics_log_divclust_c_gpu.txt"

# ------------------------------------------------------------
# Settings
# ------------------------------------------------------------
N_CLUSTERS = 10
TOP_K_HEADS = 10

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float32

#GAMMA_LIST = [3e-05, 5e-05, 7e-05, 1e-04, 2e-04]
GAMMA_LIST = [2e-05, 3e-05, 4e-05, 5e-05, 6e-05, 7e-05, 8e-05, 1e-04]
MAXITER = 30
LR_W = 0.005
LR_S = 0.005
TOP_EDGES_PER_ROW = 2
KMEANS_ITERS = 50
KMEANS_RESTARTS = 10
SEED = 0

# ------------------------------------------------------------
# Reproducibility
# ------------------------------------------------------------
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
try:
    torch.use_deterministic_algorithms(True)
except Exception:
    pass


def log(msg):
    print(msg)
    with open(METRICS_LOG_FILE, "a", encoding="utf-8") as f:
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
    clusters = np.asarray(data["clusters"], dtype=np.int64)
    labels = np.asarray(data["ground_truth"], dtype=np.int64)
    return clusters, labels


def extract_final_eval_confidence(log_file):
    pattern = re.compile(r"eval_confidence=([0-9]*\.?[0-9]+)")
    last_conf = None

    with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = pattern.search(line)
            if m:
                last_conf = float(m.group(1))

    if last_conf is None:
        raise ValueError(f"No eval_confidence found in {log_file}")
    return last_conf


def extract_final_head_losses(log_file, num_heads):
    pattern = re.compile(r"loss_main_head_(\d+)=([0-9eE+\-.]+)")
    last_vals = {}

    with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            matches = pattern.findall(line)
            if matches:
                current = {int(k): float(v) for k, v in matches}
                if current:
                    last_vals = current

    if not last_vals:
        raise ValueError(f"No loss_main_head_k values found in {log_file}")

    missing = [k for k in range(num_heads) if k not in last_vals]
    if missing:
        raise ValueError(f"Missing loss_main_head values for heads: {missing}")

    return np.array([last_vals[k] for k in range(num_heads)], dtype=float)


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


def row_topk_sparsify_and_normalize(S, topk=2):
    vals, idx = torch.topk(S, k=min(topk, S.shape[1]), dim=1)
    S_new = torch.zeros_like(S)
    S_new.scatter_(1, idx, vals)
    S_new = torch.clamp(S_new, min=0.0)
    row_sum = S_new.sum(dim=1, keepdim=True)
    S_new = S_new / torch.clamp(row_sum, min=1e-12)
    return S_new


def pairwise_sqdist_rows(X, Y):
    x2 = (X * X).sum(dim=1, keepdim=True)
    y2 = (Y * Y).sum(dim=1, keepdim=True).T
    xy = X @ Y.T
    D = x2 + y2 - 2 * xy
    return torch.clamp(D, min=0.0)


def gpu_kmeans(X, n_clusters, num_iters=25, num_restarts=3):
    N, d = X.shape
    best_labels = None
    best_inertia = None

    for restart in range(num_restarts):
        torch.manual_seed(SEED + restart)
        perm = torch.randperm(N, device=X.device)
        centers = X[perm[:n_clusters]].clone()

        for _ in range(num_iters):
            dist = pairwise_sqdist_rows(X, centers)
            labels = torch.argmin(dist, dim=1)

            new_centers = []
            for k in range(n_clusters):
                mask = labels == k
                if mask.any():
                    new_centers.append(X[mask].mean(dim=0))
                else:
                    new_centers.append(X[torch.randint(0, N, (1,), device=X.device)].squeeze(0))
            centers = torch.stack(new_centers, dim=0)

        dist = pairwise_sqdist_rows(X, centers)
        labels = torch.argmin(dist, dim=1)
        inertia = dist[torch.arange(N, device=X.device), labels].sum().item()

        if best_inertia is None or inertia < best_inertia:
            best_inertia = inertia
            best_labels = labels.clone()

    return best_labels.detach().cpu().numpy()


def connected_components_bipartite_from_S(S, thresh=1e-6):
    S_np = (S.detach().cpu().numpy() > thresh).astype(np.int8)
    n, v = S_np.shape

    S_sparse = csr_matrix(S_np)
    upper = hstack([csr_matrix((n, n)), S_sparse], format="csr")
    lower = hstack([S_sparse.T, csr_matrix((v, v))], format="csr")
    SS0 = vstack([upper, lower], format="csr")

    n_components, labels_all = connected_components(SS0, directed=False, return_labels=True)
    return labels_all[:n], n_components


def stabilized_sccbg_gpu(Y_list, c, gamma):
    S0 = torch.cat(Y_list, dim=1).to(device=DEVICE, dtype=DTYPE)
    S = S0.clone()
    SS = S0.clone()

    N, V = S.shape
    A = S0.T @ S0
    W = torch.zeros((N, V), device=DEVICE, dtype=DTYPE)

    lam = 1.0
    rho = 0.5

    for _ in range(MAXITER):
        # update W
        SA = S @ A
        grad_W = 2.0 * gamma * W * torch.clamp(SA, min=0.0) \
                 + 2.0 * W * ((S - SS) ** 2) \
                 - rho
        W = torch.clamp(W - LR_W * grad_W, 0.0, 1.0)

        # spectral step
        d1 = 1.0 / torch.sqrt(torch.clamp(S.sum(dim=1), min=1e-12))
        d2 = 1.0 / torch.sqrt(torch.clamp(S.sum(dim=0), min=1e-12))

        SS1 = d1.unsqueeze(1) * S * d2.unsqueeze(0)
        SS2 = SS1.T @ SS1

        evals, evecs = torch.linalg.eigh(SS2)
        order = torch.argsort(evals, descending=True)
        evals = evals[order]
        evecs = evecs[:, order]

        V_emb = evecs[:, :c]
        ev0 = evals[:c]

        denom = torch.sqrt(torch.clamp(ev0, min=1e-12))
        U = (SS1 @ V_emb) / denom.unsqueeze(0)

        fn1 = evals[:c].sum().item()
        fn2 = evals[:min(c + 1, len(evals))].sum().item()

        if fn1 < c - 1e-7:
            lam *= 2.0
        elif fn2 > c + 1 - 1e-7:
            lam /= 2.0

        U1 = d1.unsqueeze(1) * U
        V1 = d2.unsqueeze(1) * V_emb
        dist = pairwise_sqdist_rows(U1, V1)

        # update S
        WW = W.unsqueeze(2) * W.unsqueeze(1) * A.unsqueeze(0)
        rowsum_WW = WW.sum(dim=2)
        WWS = torch.bmm(WW, S.unsqueeze(2)).squeeze(2)

        grad_S = (
            2.0 * (W * W) * (S - SS)
            + 2.0 * gamma * (rowsum_WW * S - WWS)
            + lam * dist
        )

        S = torch.clamp(S - LR_S * grad_S, 0.0, 1.0)
        S = row_topk_sparsify_and_normalize(S, topk=TOP_EDGES_PER_ROW)

        rho *= 2.0

    cc_labels, n_components = connected_components_bipartite_from_S(S, thresh=1e-5)
    final_labels = gpu_kmeans(U1, n_clusters=c, num_iters=KMEANS_ITERS, num_restarts=KMEANS_RESTARTS)

    return final_labels, n_components, S, U1.detach()


def compute_metrics(labels, pred):
    return {
        "acc": clustering_accuracy(labels, pred),
        "nmi": normalized_mutual_info_score(labels, pred),
        "ari": adjusted_rand_score(labels, pred),
    }


def divclust_c_gpu_consensus(clusters, head_losses, labels):
    selected_idx = np.argsort(head_losses)[:TOP_K_HEADS]
    selected = clusters[selected_idx]

    Y_list = [hard_labels_to_onehot_torch(selected[j], N_CLUSTERS) for j in range(TOP_K_HEADS)]

    all_results = []

    for gamma in GAMMA_LIST:
        pred, n_components, _, _ = stabilized_sccbg_gpu(Y_list, N_CLUSTERS, gamma)
        metrics = compute_metrics(labels, pred)

        result = {
            "gamma": gamma,
            "labels": pred,
            "components": n_components,
            "acc": metrics["acc"],
            "nmi": metrics["nmi"],
            "ari": metrics["ari"],
        }
        all_results.append(result)

        log(
            f"gamma={gamma:.0e} | "
            f"components={n_components} | "
            f"NMI={metrics['nmi']:.4f} | "
            f"ACC={metrics['acc']:.4f} | "
            f"ARI={metrics['ari']:.4f}"
        )

    return selected_idx, all_results


def get_best_result(results, key):
    return max(results, key=lambda x: x[key])


def main():
    if os.path.exists(METRICS_LOG_FILE):
        os.remove(METRICS_LOG_FILE)

    log(f"Device                  : {DEVICE}")

    clusters, labels = load_outcomes()
    num_heads = clusters.shape[0]

    cnf = extract_final_eval_confidence(TRAIN_LOG_FILE)
    head_losses = extract_final_head_losses(TRAIN_LOG_FILE, num_heads)
    head_metrics = compute_head_metrics(clusters, labels)

    log("===== Base-head Metrics =====")
    log(f"CNF                     : {cnf:.4f}")
    log(f"Mean NMI                : {head_metrics['mean_nmi']:.4f}")
    log(f"Max NMI                 : {head_metrics['max_nmi']:.4f}")
    log("")
    log(f"Mean ACC                : {head_metrics['mean_acc']:.4f}")
    log(f"Max ACC                 : {head_metrics['max_acc']:.4f}")
    log("")
    log(f"Mean ARI                : {head_metrics['mean_ari']:.4f}")
    log(f"Max ARI                 : {head_metrics['max_ari']:.4f}")
    log("")
    log(f"Inter-clustering NMI    : {head_metrics['inter_nmi']:.4f}")

    log("")
    log("===== DivClust C Consensus (stabilized GPU SCCBG-style) =====")
    log(f"Final per-head losses   : {head_losses.tolist()}")

    selected_idx, all_results = divclust_c_gpu_consensus(clusters, head_losses, labels)

    best_by_acc = get_best_result(all_results, "acc")
    best_by_nmi = get_best_result(all_results, "nmi")
    best_by_ari = get_best_result(all_results, "ari")

    np.save(f"{BASE}/divclust_c_labels_best_acc.npy", best_by_acc["labels"])
    np.save(f"{BASE}/divclust_c_labels_best_nmi.npy", best_by_nmi["labels"])
    np.save(f"{BASE}/divclust_c_labels_best_ari.npy", best_by_ari["labels"])

    log("")
    log(f"Selected heads          : {list(selected_idx)}")
    log(f"Selected losses         : {[float(head_losses[i]) for i in selected_idx]}")

    log("")
    log("===== Best by ACC =====")
    log(f"Best gamma              : {best_by_acc['gamma']:.0e}")
    log(f"Consensus NMI           : {best_by_acc['nmi']:.4f}")
    log(f"Consensus ACC           : {best_by_acc['acc']:.4f}")
    log(f"Consensus ARI           : {best_by_acc['ari']:.4f}")
    log(f"Connected components    : {best_by_acc['components']}")
    log(f"Saved labels to         : {BASE}/divclust_c_labels_best_acc.npy")

    log("")
    log("===== Best by NMI =====")
    log(f"Best gamma              : {best_by_nmi['gamma']:.0e}")
    log(f"Consensus NMI           : {best_by_nmi['nmi']:.4f}")
    log(f"Consensus ACC           : {best_by_nmi['acc']:.4f}")
    log(f"Consensus ARI           : {best_by_nmi['ari']:.4f}")
    log(f"Connected components    : {best_by_nmi['components']}")
    log(f"Saved labels to         : {BASE}/divclust_c_labels_best_nmi.npy")

    log("")
    log("===== Best by ARI =====")
    log(f"Best gamma              : {best_by_ari['gamma']:.0e}")
    log(f"Consensus NMI           : {best_by_ari['nmi']:.4f}")
    log(f"Consensus ACC           : {best_by_ari['acc']:.4f}")
    log(f"Consensus ARI           : {best_by_ari['ari']:.4f}")
    log(f"Connected components    : {best_by_ari['components']}")
    log(f"Saved labels to         : {BASE}/divclust_c_labels_best_ari.npy")


if __name__ == "__main__":
    main()