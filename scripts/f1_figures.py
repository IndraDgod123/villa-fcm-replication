# -*- coding: utf-8 -*-
import json, sys
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
sys.path.insert(0, "scripts")
from pipeline import build_features
plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 8, "axes.edgecolor": "#52514e",
    "axes.labelcolor": "#0b0b0b", "xtick.color": "#52514e", "ytick.color": "#52514e",
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": "#e6e5e0",
    "grid.linewidth": 0.6, "axes.axisbelow": True, "savefig.dpi": 300, "legend.frameon": False})
C = ["#2a78d6", "#eb6834", "#1baf7a"]; INK2 = "#52514e"
NAMES = ["C0 Mixed independent", "C1 Peak Asian DMC", "C2 Off-peak Asian DMC"]
S = np.load("out/main_solution.npz"); u, lab, Xp, evr = S["u"], S["lab"], S["Xp"], S["evr"]
R = json.load(open("out/a3_results.json"))
master = pd.read_csv("data/master_data.csv"); f = build_features(master)
W = 6.3  # inches, full text width approx 16 cm

# ---------- Figure 2: PCA variance + PCA sensitivity
from sklearn.decomposition import PCA
from pipeline import encode
X = encode(f).values.astype(float); full = PCA().fit(X).explained_variance_ratio_[:17]*100
fig, ax = plt.subplots(1, 2, figsize=(W, 2.3))
k = np.arange(1, 18)
ax[0].bar(k, full, color=[C[0] if i < 8 else "#b7d3f6" for i in range(17)], width=0.7)
ax[0].plot(k, np.cumsum(full), color="#0b0b0b", lw=1.5, marker="o", ms=3)
ax[0].axhline(90, color=INK2, lw=0.8, ls="--"); ax[0].text(17, 88, "90% threshold", ha="right", va="top", fontsize=7, color=INK2)
ax[0].annotate("PC8: 92.8%", (8, 92.8), (10, 60), fontsize=7, arrowprops=dict(arrowstyle="-", color=INK2, lw=0.6))
ax[0].set_xlabel("Principal component"); ax[0].set_ylabel("Explained variance (%)"); ax[0].set_xticks([1,4,8,12,17])
ax[0].set_ylim(0, 105); ax[0].set_title("(a) Per component (bars) and cumulative (line)", fontsize=8, loc="left")
ps = pd.read_csv("out/pca_sensitivity.csv")
xs = ["4", "5", "6", "7", "8", "10", "12", "17", "20-D"]
ax[1].plot(range(len(xs)), ps["ARI_vs_main"], color=C[0], lw=2, marker="o", ms=4)
for i, v in enumerate(ps["ARI_vs_main"]):
    if i in (0, 3, 8): ax[1].text(i, v - 0.07, f"{v:.2f}", ha="center", fontsize=7)
ax[1].set_xticks(range(len(xs))); ax[1].set_xticklabels(xs); ax[1].set_ylim(0, 1.08)
ax[1].set_xlabel("Retained components (20-D = no reduction)"); ax[1].set_ylabel("Adjusted Rand Index")
ax[1].set_title("(b) Agreement with the reported partition", fontsize=8, loc="left")
fig.tight_layout(); fig.savefig("fig/Figure_2_pca.png"); plt.close(fig)

# ---------- Figure 3: 2-D projection + deviation heatmap
fig, ax = plt.subplots(1, 2, figsize=(W, 3.0), gridspec_kw=dict(width_ratios=[1, 1.25]))
rng = np.random.default_rng(1); idx = rng.choice(len(lab), 6000, replace=False)
jit = rng.normal(0, 0.03, (len(idx), 2))
for c in range(3):
    s = idx[lab[idx] == c]
    ax[0].scatter(Xp[s, 0] + jit[lab[idx] == c, 0], Xp[s, 1] + jit[lab[idx] == c, 1], s=3, color=C[c], alpha=0.35, lw=0, label=NAMES[c])
ax[0].set_xlabel(f"PC1 ({evr[0]*100:.1f}%)"); ax[0].set_ylabel(f"PC2 ({evr[1]*100:.1f}%)")
fig.legend(*ax[0].get_legend_handles_labels(), loc="lower center", ncol=3, fontsize=7, markerscale=3, handletextpad=0.2)
ax[0].set_title("(a) Random 6,000 bookings on PC1-PC2", fontsize=8, loc="left")
rows = [("Peak", f.booking_season == "Peak"), ("Shoulder", f.booking_season == "Shoulder"), ("Low", f.booking_season == "Low"),
        ("Asia", f.nationality_region == "Asia"), ("Oceania", f.nationality_region == "Oceania"),
        ("Domestic", f.nationality_region == "Domestic"), ("Unknown origin", f.nationality_region == "Unknown"),
        ("Int. OTA", f.booking_platform_type == "International_OTA"), ("DMC", f.booking_platform_type == "DMC"),
        ("Local agent", f.booking_platform_type == "Local_Agent"), ("Other channel", f.booking_platform_type == "Other")]
M = np.array([[100 * (m[lab == c]).mean() - 100 * m.mean() for c in range(3)] for _, m in rows])
cmap = LinearSegmentedColormap.from_list("div", ["#2a78d6", "#f0efec", "#e34948"])
im = ax[1].imshow(M, cmap=cmap, vmin=-60, vmax=60, aspect="auto")
for i in range(M.shape[0]):
    for j in range(3):
        ax[1].text(j, i, f"{M[i, j]:+.1f}", ha="center", va="center", fontsize=6.5, color="#0b0b0b")
ax[1].set_xticks(range(3)); ax[1].set_xticklabels(["C0", "C1", "C2"]); ax[1].set_yticks(range(len(rows))); ax[1].set_yticklabels([r[0] for r in rows])
ax[1].grid(False); cb = fig.colorbar(im, ax=ax[1], fraction=0.05, pad=0.02); cb.set_label("Percentage points", fontsize=7)
ax[1].set_title("(b) Share in cluster minus overall share (pp)", fontsize=8, loc="left")
fig.tight_layout(rect=(0, 0.06, 1, 1)); fig.savefig("fig/Figure_3_profile.png"); plt.close(fig)

# ---------- Figure 4: membership distribution + m sensitivity
mx = u.max(axis=0)
fig, ax = plt.subplots(1, 2, figsize=(W, 2.3))
bins = np.linspace(0.33, 0.9, 40)
ax[0].hist([mx[lab == c] for c in range(3)], bins=bins, stacked=True, color=C, label=["C0", "C1", "C2"], edgecolor="white", lw=0.3)
for t in (0.5, 0.7): ax[0].axvline(t, color=INK2, ls="--", lw=0.8)
ax[0].text(0.415, ax[0].get_ylim()[1]*0.92, "Ambiguous\n60.8%", ha="center", fontsize=7, va="top")
ax[0].text(0.6, ax[0].get_ylim()[1]*0.92, "Moderate\n8.5%", ha="center", fontsize=7, va="top")
ax[0].text(0.8, ax[0].get_ylim()[1]*0.92, "Strong\n30.7%", ha="center", fontsize=7, va="top")
ax[0].set_xlabel("Highest membership degree"); ax[0].set_ylabel("Bookings"); ax[0].legend(fontsize=6.5, loc="center right")
ax[0].set_title("(a) Highest membership at m = 2.0", fontsize=8, loc="left")
ms = pd.read_csv("out/m_sensitivity.csv")
ax[1].plot(ms.m, ms.ambiguous_pct, color=C[1], lw=2, marker="o", ms=4, label="Ambiguous (< 0.5)")
ax[1].plot(ms.m, ms.strong_pct, color=C[0], lw=2, marker="s", ms=4, label="Strong (>= 0.7)")
ax[1].axvspan(2.45, 3.05, color="#f0efec"); ax[1].text(2.75, 50, "trivial\nsolution\n(PC = 1/c)", ha="center", fontsize=7, color=INK2)
ax[1].set_xlabel("Fuzzifier m"); ax[1].set_ylabel("Share of bookings (%)"); ax[1].set_ylim(0, 105)
ax[1].legend(fontsize=6.5, loc="upper center", bbox_to_anchor=(0.38, 1.0), ncol=2); ax[1].set_ylim(0, 125); ax[1].set_yticks(range(0, 101, 25)); ax[1].set_title("(b) Membership classes as a function of m", fontsize=8, loc="left")
fig.tight_layout(); fig.savefig("fig/Figure_4_membership.png"); plt.close(fig)

# ---------- Figure 5: yearly composition
yr = pd.DataFrame(R["cluster_by_year_rowpct"]).loc[["2019", "2020", "2021", "2022", "2023", "2024"]]
tot = R["year_totals"]
ds = f.villa.eq("Dream Suite"); yy = f.booking_year
dsp = pd.crosstab(yy[ds], lab[ds.values], normalize="index") * 100
fig, ax = plt.subplots(1, 2, figsize=(W, 2.3), sharey=True)
for a, tab, ttl, ns in [(ax[0], yr.astype(float), "(a) All villas", [tot[y] for y in yr.index]),
                        (ax[1], dsp.rename(columns=str), "(b) Villa Dream Suite only", list(yy[ds].value_counts().sort_index().values))]:
    yrs = [str(y) for y in tab.index]; bottom = np.zeros(len(yrs))
    for c in range(3):
        v = tab[str(c)].values
        a.bar(yrs, v, bottom=bottom, color=C[c], width=0.65, edgecolor="white", lw=1, label=f"C{c}")
        bottom += v
    for i, n in enumerate(ns): a.text(i, 102, f"n={n:,}", ha="center", fontsize=6.3, color=INK2)
    a.set_title(ttl, fontsize=8, loc="left", pad=12); a.set_xlabel("Check-in year")
ax[0].set_ylabel("Share of bookings (%)"); ax[0].set_ylim(0, 110); fig.legend(*ax[0].get_legend_handles_labels(), fontsize=7, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.0))
fig.tight_layout(rect=(0, 0, 1, 0.93)); fig.savefig("fig/Figure_5_yearly.png"); plt.close(fig)
print("ok")
