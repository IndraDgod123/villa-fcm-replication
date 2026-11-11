import os
import matplotlib; matplotlib.use("Agg")
os.makedirs("fig", exist_ok=True)
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
plt.rcParams["font.family"] = "DejaVu Serif"
boxes = [("Problem\nidentification", "research questions"), ("Data collection", "77 files, four villas,\n2019-2024"),
         ("Data integration", "extraction, schema\nharmonization\n(Apache Spark)"), ("Cleaning and\ndeduplication", "key-sensitivity\naudit, 'Unknown'\nhandling"),
         ("Feature encoding", "one-hot + Min-Max\n(17,762 x 20)"),
         ("Profiling and\ndashboard", "deviation profiles,\nStreamlit\nprototype"), ("External checks\nand baselines", "external variables;\nK-Means, GMM, LCA,\nk-prototypes, rules"),
         ("Stability\nassessment", "bootstrap, 2024\nhold-out, leave-one-\nvilla-out, subsampling"), ("Fuzzy C-Means", "k = 2-7, m = 1.3-3.0,\n20 restarts,\nfive validity indices"),
         ("Dimensionality\nreduction", "PCA, p = 8\n(sensitivity p = 4-20)")]
fig, ax = plt.subplots(figsize=(10, 3.3)); ax.set_xlim(0, 10); ax.set_ylim(0, 3.3); ax.axis("off")
w, h = 1.72, 1.35
pos = [(0.1 + i * 2.0, 1.85) for i in range(5)] + [(0.1 + i * 2.0, 0.2) for i in range(5)]
for (t, sub), (x, y) in zip(boxes, pos):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc="white", ec="black", lw=1.4))
    ax.text(x + w / 2, y + h - 0.12, t, ha="center", va="top", fontsize=10)
    ax.text(x + w / 2, y + 0.1, sub, ha="center", va="bottom", fontsize=7.6, color="#333333")
arr = dict(arrowstyle="-|>", lw=1.4, color="black", mutation_scale=12)
for i in range(4): ax.annotate("", (pos[i + 1][0], 1.85 + h / 2), (pos[i][0] + w, 1.85 + h / 2), arrowprops=arr)
ax.annotate("", (pos[9][0] + w / 2, 0.2 + h), (pos[4][0] + w / 2, 1.85), arrowprops=arr)
for i in range(9, 5, -1): ax.annotate("", (pos[i - 1][0] + w, 0.2 + h / 2), (pos[i][0], 0.2 + h / 2), arrowprops=arr)
fig.savefig("fig/Figure_1_flowchart.png", dpi=300, bbox_inches="tight"); print("ok")
