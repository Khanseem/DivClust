#import matplotlib
#matplotlib.use("Agg")

import zipfile
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import normalized_mutual_info_score


BASE = "/scratch/e1536052/DivClust/experiments/deep_clustering/CC_dt_1.0"
OUTCOME_FILE = f"{BASE}/outcomes"

with zipfile.ZipFile(OUTCOME_FILE, "r") as z:
    print("Archive contents:")
    for name in z.namelist():
        print(name)

    candidate_names = ["data.pkl", "outcomes/data.pkl"]
    data_name = None
    for name in candidate_names:
        if name in z.namelist():
            data_name = name
            break

    if data_name is None:
        raise FileNotFoundError("Could not find data.pkl inside outcomes archive.")

    with z.open(data_name) as f:
        data = pickle.load(f)

clusters = data["clusters"]   # shape: (num_heads, N)
num_heads = clusters.shape[0]

# Compute pairwise NMI matrix
nmi_matrix = np.zeros((num_heads, num_heads), dtype=float)
for i in range(num_heads):
    for j in range(num_heads):
        nmi_matrix[i, j] = normalized_mutual_info_score(clusters[i], clusters[j])

# Average off-diagonal NMI
avg_offdiag = (nmi_matrix.sum() - np.trace(nmi_matrix)) / (num_heads * (num_heads - 1))

# Plot using pcolormesh for clean grid lines
fig, ax = plt.subplots(figsize=(6.4, 6.4))

x = np.arange(num_heads + 1)
y = np.arange(num_heads + 1)

mesh = ax.pcolormesh(
    x,
    y,
    nmi_matrix,
    cmap="viridis",
    vmin=0,
    vmax=1,
    edgecolors="black",
    linewidth=0.8,
    shading="flat"
)

# Put row 0 at the top, like imshow
ax.set_xlim(0, num_heads)
ax.set_ylim(num_heads, 0)
ax.set_aspect("equal")

# Shorter colorbar
cbar = plt.colorbar(mesh, ax=ax, shrink=0.75, pad=0.04)
cbar.set_ticks(np.linspace(0, 1, 6))
cbar.ax.tick_params(labelsize=10)

# Major ticks at cell centers
tick_pos = np.arange(0, num_heads, 2.5) + 0.5
tick_labels = [f"{t:.1f}" for t in np.arange(0, num_heads, 2.5)]

ax.set_xticks(tick_pos)
ax.set_yticks(tick_pos)
ax.set_xticklabels(tick_labels)
ax.set_yticklabels(tick_labels)

# Labels and title
ax.set_xlabel("Head", fontsize=11)
ax.set_ylabel("Head", fontsize=11)
ax.set_title(
    f"Inter-clustering similarity (NMI)\nAvg off-diagonal = {avg_offdiag:.3f}",
    fontsize=13
)

ax.tick_params(axis="both", which="major", labelsize=10)

plt.tight_layout()
plt.savefig(f"{BASE}/nmi_heatmap.png", dpi=300, bbox_inches="tight")
plt.show()

print(f"Saved figure to: {BASE}/nmi_heatmap.png")
print(f"Average off-diagonal NMI = {avg_offdiag:.3f}")