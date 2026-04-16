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

# ============================================================
# Paths
# ============================================================
BASE = "/home/svu/e0388988/Graph Consensus/CC_dt_0.8_1000_epochs"
OUTCOME_FILE = f"{BASE}/outcomes_loss"
TRAIN_LOG_FILE = f"{BASE}/log_loss.txt"
METRICS_LOG_FILE = f"{BASE}/metrics_log_divclust_c_gpu.txt"

# ============================================================
# Settings
# ============================================================
N_CLUSTERS = 10

# Head selection: choose from lowest-loss pool, rank by metrics
TOP_K_HEADS = 5
HEAD_PRESELECT_M = 15

# Metric weights for head selection (min-max normalized on pool)
W_ACC = 0.0
W_NMI = 1.0
W_ARI = 0.0

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float32

# Hyperparameter grids
GAMMA_LIST = [
    5.4e-05, 5.45e-05, 5.5e-05, 5.55e-05, 5.6e-05,
    5.65e-05, 5.7e-05, 5.75e-05, 5.8e-05, 5.85e-05,
    5.9e-05, 5.95e-05, 6.0e-05, 6.05e-05, 6.1e-05,
    6.15e-05, 6.2e-05, 6.25e-05, 6.3e-05, 6.35e-05,
    6.4e-05, 6.45e-05, 6.5e-05, 6.55e-05, 6.6e-05
]
LAMBDA0_LIST = [
    0.4, 0.45, 0.5, 0.55, 0.6,
    0.65, 0.7, 0.75, 0.8, 0.85, 
    0.9, 0.95, 1.0, 1.05, 1.1, 
    1.15, 1.2, 1.25, 1.3, 1.35,
    1.4, 1.45, 1.5, 1.55, 1.6
]

MAXITER = 20
LR_W = 0.005
LR_S = 0.005
TOP_EDGES_PER_ROW = 2
KMEANS_ITERS = 30
KMEANS_RESTARTS = 10
SEED = 1

# ------------------------------------------------------------
# Final assignment mode (paper-style option included)
# ------------------------------------------------------------
# "kmeans": run kmeans on image embedding after graph consensus
# "eig_argmax": assign label by argmax over top-c eigenvector embedding after graph consensus
ASSIGNMENT_MODE = "kmeans"  # options: "kmeans", "eig_argmax"

# eigenvectors can have sign ambiguity; choose a nonneg mapping before argmax:
# "relu": clamp to >=0 
# "abs": take abs value
# "none": use raw values
EIG_ARGMAX_NONNEG = "relu"

# ============================================================
# Reproducibility
# ============================================================
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


# ============================================================
# Utilities
# ============================================================
def log(msg: str):
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


def compute_metrics(y_true, y_pred):
    return {
        "acc": clustering_accuracy(y_true, y_pred),
        "nmi": normalized_mutual_info_score(y_true, y_pred),
        "ari": adjusted_rand_score(y_true, y_pred),
    }


def get_best_result(results, key):
    return max(results, key=lambda x: x[key])


def load_outcomes():
    data = torch.load(OUTCOME_FILE, map_location="cpu", weights_only=False)
    clusters = np.asarray(data["clusters"], dtype=np.int64)         # [H, N]
    labels = np.asarray(data["ground_truth"], dtype=np.int64)       # [N]
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
    acc_list, nmi_list, ari_list, pairwise_nmi = [], [], [], []
    H = clusters.shape[0]
    for i in range(H):
        pred = clusters[i]
        acc_list.append(clustering_accuracy(labels, pred))
        nmi_list.append(normalized_mutual_info_score(labels, pred))
        ari_list.append(adjusted_rand_score(labels, pred))
        for j in range(i + 1, H):
            pairwise_nmi.append(normalized_mutual_info_score(clusters[i], clusters[j]))
    return {
        "mean_acc": float(np.mean(acc_list)),
        "max_acc": float(np.max(acc_list)),
        "mean_nmi": float(np.mean(nmi_list)),
        "max_nmi": float(np.max(nmi_list)),
        "mean_ari": float(np.mean(ari_list)),
        "max_ari": float(np.max(ari_list)),
        "inter_nmi": float(np.mean(pairwise_nmi)) if pairwise_nmi else float("nan"),
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
    return S_new / torch.clamp(row_sum, min=1e-12)


def pairwise_sqdist_rows(X, Y):
    x2 = (X * X).sum(dim=1, keepdim=True)
    y2 = (Y * Y).sum(dim=1, keepdim=True).T
    return torch.clamp(x2 + y2 - 2 * (X @ Y.T), min=0.0)


def gpu_kmeans(X, n_clusters, num_iters=25, num_restarts=3):
    N, _ = X.shape
    best_labels, best_inertia = None, None
    for restart in range(num_restarts):
        torch.manual_seed(SEED + restart)
        centers = X[torch.randperm(N, device=X.device)[:n_clusters]].clone()
        for _ in range(num_iters):
            dist = pairwise_sqdist_rows(X, centers)
            labels = torch.argmin(dist, dim=1)
            centers = torch.stack(
                [
                    (X[labels == k].mean(dim=0) if (labels == k).any()
                     else X[torch.randint(0, N, (1,), device=X.device)].squeeze(0))
                    for k in range(n_clusters)
                ],
                dim=0,
            )
        dist = pairwise_sqdist_rows(X, centers)
        labels = torch.argmin(dist, dim=1)
        inertia = dist[torch.arange(N, device=X.device), labels].sum().item()
        if best_inertia is None or inertia < best_inertia:
            best_inertia, best_labels = inertia, labels.clone()
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


def eig_argmax_labels(U1, nonneg="relu"):
    """
    Paper-style assignment:
        y_i = argmax_r F1(i,r)
    Here we use U1 as the image-node embedding F1 (N x c).
    """
    X = U1
    if nonneg == "relu":
        X = torch.clamp(X, min=0.0)
    elif nonneg == "abs":
        X = torch.abs(X)
    elif nonneg == "none":
        pass
    else:
        raise ValueError(f"Unknown EIG_ARGMAX_NONNEG: {nonneg}")

    row_sum = X.sum(dim=1)
    if (row_sum == 0).any():
        X = torch.where((row_sum == 0).unsqueeze(1), U1, X)

    return torch.argmax(X, dim=1)


def _minmax01(x, eps=1e-12):
    x = np.asarray(x, dtype=float)
    return (x - x.min()) / (x.max() - x.min() + eps)


def _per_head_scores(clusters, labels):
    H = clusters.shape[0]
    acc = np.empty(H, dtype=float)
    nmi = np.empty(H, dtype=float)
    ari = np.empty(H, dtype=float)
    for i in range(H):
        pred = clusters[i]
        acc[i] = clustering_accuracy(labels, pred)
        nmi[i] = normalized_mutual_info_score(labels, pred)
        ari[i] = adjusted_rand_score(labels, pred)
    return acc, nmi, ari


def _select_heads(clusters, labels, head_losses):
    H = clusters.shape[0]
    m = int(min(max(HEAD_PRESELECT_M, TOP_K_HEADS), H))
    pool = np.argsort(head_losses)[:m]

    acc, nmi, ari = _per_head_scores(clusters, labels)
    acc_s = _minmax01(acc[pool])
    nmi_s = _minmax01(nmi[pool])
    ari_s = _minmax01(ari[pool])

    score = W_ACC * acc_s + W_NMI * nmi_s + W_ARI * ari_s
    selected = pool[np.argsort(-score)[:TOP_K_HEADS]]

    extras = {"acc": acc, "nmi": nmi, "ari": ari, "pool": pool, "score_pool": score}
    return selected, extras


# ============================================================
# Stabilized SCCBG-style solver
# ============================================================
def stabilized_sccbg_gpu(Y_list, c, gamma, lambda0):
    """
    Implements the alternating update (projected gradient) + spectral embedding step
    consistent with the description in the screenshot.

    Key enhancement:
      - You can output labels either by:
          (A) kmeans on U1 (embedding), or
          (B) eig_argmax on U1 (paper-style argmax over F1).
      - lambda0 now scales the spectral penalty term via:
            lambda_eff = lambda0 * lambda_adapt
    """
    S0 = torch.cat(Y_list, dim=1).to(device=DEVICE, dtype=DTYPE)  # N x V
    S = S0.clone()
    SS = S0.clone()

    N, V = S.shape
    A = S0.T @ S0                                              # V x V
    W = torch.zeros((N, V), device=DEVICE, dtype=DTYPE)         # N x V

    lambda_adapt = 1.0
    rho = 0.5

    for _ in range(MAXITER):
        # ----------------------------
        # Update W (projected step)
        # ----------------------------
        SA = S @ A
        grad_W = 2.0 * gamma * W * torch.clamp(SA, min=0.0) + 2.0 * W * ((S - SS) ** 2) - rho
        W = torch.clamp(W - LR_W * grad_W, 0.0, 1.0)

        # ----------------------------
        # Spectral embedding step
        # ----------------------------
        d1 = 1.0 / torch.sqrt(torch.clamp(S.sum(dim=1), min=1e-12))
        d2 = 1.0 / torch.sqrt(torch.clamp(S.sum(dim=0), min=1e-12))

        SS1 = d1.unsqueeze(1) * S * d2.unsqueeze(0)             # N x V
        SS2 = SS1.T @ SS1                                       # V x V

        evals, evecs = torch.linalg.eigh(SS2)                   # ascending
        order = torch.argsort(evals, descending=True)
        evals = evals[order]
        evecs = evecs[:, order]

        V_emb = evecs[:, :c]                                    # V x c
        ev0 = evals[:c]                                         # c

        U = (SS1 @ V_emb) / torch.sqrt(torch.clamp(ev0, min=1e-12)).unsqueeze(0)  # N x c

        # adaptive rank constraint controller (kept from your style)
        fn1 = evals[:c].sum().item()
        fn2 = evals[:min(c + 1, len(evals))].sum().item()
        if fn1 < c - 1e-7:
            lambda_adapt *= 2.0
        elif fn2 > c + 1 - 1e-7:
            lambda_adapt /= 2.0

        U1 = d1.unsqueeze(1) * U                                # N x c (image embedding, ~F1)
        V1 = d2.unsqueeze(1) * V_emb                            # V x c

        dist = pairwise_sqdist_rows(U1, V1)                     # N x V

        # ----------------------------
        # Update S (projected step + sparsify)
        # ----------------------------
        WW = W.unsqueeze(2) * W.unsqueeze(1) * A.unsqueeze(0)    # N x V x V
        rowsum_WW = WW.sum(dim=2)                                # N x V
        WWS = torch.bmm(WW, S.unsqueeze(2)).squeeze(2)           # N x V

        lambda_eff = float(lambda0) * float(lambda_adapt)
        grad_S = (
            2.0 * (W * W) * (S - SS)
            + 2.0 * gamma * (rowsum_WW * S - WWS)
            + lambda_eff * dist
        )

        S = torch.clamp(S - LR_S * grad_S, 0.0, 1.0)
        S = row_topk_sparsify_and_normalize(S, topk=TOP_EDGES_PER_ROW)

        rho *= 2.0

    _, n_components = connected_components_bipartite_from_S(S, thresh=1e-5)

    # ----------------------------
    # Final label assignment (two options)
    # ----------------------------
    if ASSIGNMENT_MODE == "kmeans":
        final_labels = gpu_kmeans(U1, n_clusters=c, num_iters=KMEANS_ITERS, num_restarts=KMEANS_RESTARTS)
    elif ASSIGNMENT_MODE == "eig_argmax":
        labels_t = eig_argmax_labels(U1, nonneg=EIG_ARGMAX_NONNEG)
        final_labels = labels_t.detach().cpu().numpy()
    else:
        raise ValueError(f"Unknown ASSIGNMENT_MODE: {ASSIGNMENT_MODE}")

    return final_labels, n_components, S.detach(), U1.detach()


def divclust_c_gpu_consensus(clusters, head_losses, labels):
    selected_idx, extras = _select_heads(clusters, labels, head_losses)
    selected = clusters[selected_idx]
    Y_list = [hard_labels_to_onehot_torch(selected[j], N_CLUSTERS) for j in range(TOP_K_HEADS)]

    log("")
    log("===== Head selection (metrics from low-loss pool) =====")
    log(f"TOP_K_HEADS             : {TOP_K_HEADS}")
    log(f"HEAD_PRESELECT_M        : {HEAD_PRESELECT_M}")
    log(f"W_ACC/W_NMI/W_ARI       : {W_ACC:.2f}/{W_NMI:.2f}/{W_ARI:.2f}")
    log(f"ASSIGNMENT_MODE         : {ASSIGNMENT_MODE}")
    if ASSIGNMENT_MODE == "eig_argmax":
        log(f"EIG_ARGMAX_NONNEG       : {EIG_ARGMAX_NONNEG}")
    log(f"Selected heads          : {list(map(int, selected_idx))}")
    log(f"Selected losses         : {[float(head_losses[i]) for i in selected_idx]}")
    log(f"Selected head ACC       : {[float(extras['acc'][i]) for i in selected_idx]}")
    log(f"Selected head NMI       : {[float(extras['nmi'][i]) for i in selected_idx]}")
    log(f"Selected head ARI       : {[float(extras['ari'][i]) for i in selected_idx]}")

    all_results = []

    for gamma in GAMMA_LIST:
        for lambda0 in LAMBDA0_LIST:
            pred, n_components, _, _ = stabilized_sccbg_gpu(Y_list, N_CLUSTERS, gamma, lambda0)
            m = compute_metrics(labels, pred)
            combo = (m["acc"] + m["nmi"] + m["ari"]) / 3.0

            all_results.append(
                {
                    "gamma": gamma,
                    "lambda0": lambda0,
                    "labels": pred,
                    "components": n_components,
                    "acc": m["acc"],
                    "nmi": m["nmi"],
                    "ari": m["ari"],
                    "combo": combo,
                }
            )

            log(
                f"gamma={gamma:.5e} | lambda0={lambda0:.2f} | components={n_components} | "
                f"NMI={m['nmi']:.4f} | ACC={m['acc']:.4f} | ARI={m['ari']:.4f} | COMBO={combo:.4f}"
            )

    best_by_combo = get_best_result(all_results, "combo")
    np.save(f"{BASE}/divclust_c_labels_best_combo.npy", best_by_combo["labels"])

    log("")
    log("===== Best by COMBO (mean of ACC/NMI/ARI) =====")
    log(f"Best gamma              : {best_by_combo['gamma']:.5e}")
    log(f"Best lambda0            : {best_by_combo['lambda0']:.2f}")
    log(f"Consensus NMI           : {best_by_combo['nmi']:.4f}")
    log(f"Consensus ACC           : {best_by_combo['acc']:.4f}")
    log(f"Consensus ARI           : {best_by_combo['ari']:.4f}")
    log(f"Connected components    : {best_by_combo['components']}")
    log(f"Saved labels to         : {BASE}/divclust_c_labels_best_combo.npy")

    return selected_idx, all_results


def main():
    if os.path.exists(METRICS_LOG_FILE):
        os.remove(METRICS_LOG_FILE)

    log(f"Device                  : {DEVICE}")

    clusters, labels = load_outcomes()
    H = clusters.shape[0]

    # sanity checks
    assert clusters.ndim == 2, clusters.shape
    assert labels.ndim == 1, labels.shape
    assert clusters.shape[1] == labels.shape[0], (clusters.shape, labels.shape)
    assert H >= TOP_K_HEADS, (H, TOP_K_HEADS)

    cnf = extract_final_eval_confidence(TRAIN_LOG_FILE)
    head_losses = extract_final_head_losses(TRAIN_LOG_FILE, H)
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
    log("===== DivClust C Consensus (stabilized SCCBG-style) =====")
    log(f"Final per-head losses   : {head_losses.tolist()}")

    selected_idx, all_results = divclust_c_gpu_consensus(clusters, head_losses, labels)

    best_by_acc = get_best_result(all_results, "acc")
    best_by_nmi = get_best_result(all_results, "nmi")
    best_by_ari = get_best_result(all_results, "ari")
    best_by_combo = get_best_result(all_results, "combo")

    log("")
    log(f"Selected heads          : {list(map(int, selected_idx))}")
    log(f"Selected losses         : {[float(head_losses[i]) for i in selected_idx]}")

    log("")
    log("===== Best by ACC =====")
    log(f"Best gamma              : {best_by_acc['gamma']:.5e}")
    log(f"Best lambda0            : {best_by_acc['lambda0']:.2f}")
    log(f"ACC/NMI/ARI             : {best_by_acc['acc']:.4f}/{best_by_acc['nmi']:.4f}/{best_by_acc['ari']:.4f}")
    log(f"Connected components    : {best_by_acc['components']}")

    log("")
    log("===== Best by NMI =====")
    log(f"Best gamma              : {best_by_nmi['gamma']:.5e}")
    log(f"Best lambda0            : {best_by_nmi['lambda0']:.2f}")
    log(f"ACC/NMI/ARI             : {best_by_nmi['acc']:.4f}/{best_by_nmi['nmi']:.4f}/{best_by_nmi['ari']:.4f}")
    log(f"Connected components    : {best_by_nmi['components']}")

    log("")
    log("===== Best by ARI =====")
    log(f"Best gamma              : {best_by_ari['gamma']:.5e}")
    log(f"Best lambda0            : {best_by_ari['lambda0']:.2f}")
    log(f"ACC/NMI/ARI             : {best_by_ari['acc']:.4f}/{best_by_ari['nmi']:.4f}/{best_by_ari['ari']:.4f}")
    log(f"Connected components    : {best_by_ari['components']}")

    log("")
    log("===== Best by COMBO =====")
    log(f"Best gamma              : {best_by_combo['gamma']:.5e}")
    log(f"Best lambda0            : {best_by_combo['lambda0']:.2f}")
    log(f"ACC/NMI/ARI             : {best_by_combo['acc']:.4f}/{best_by_combo['nmi']:.4f}/{best_by_combo['ari']:.4f}")
    log(f"Connected components    : {best_by_combo['components']}")
    log(f"Saved labels to         : {BASE}/divclust_c_labels_best_combo.npy")

if __name__ == "__main__":
    main()