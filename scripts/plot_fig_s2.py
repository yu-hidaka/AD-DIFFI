import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pathlib import Path
REPO = Path(__file__).resolve().parent.parent
OUT = str(REPO / "results" / "supp_fig_s2_depth_purity") + "/"
d = pd.read_csv(OUT + "depth_sensitivity_s1.csv")
s = pd.read_csv(OUT + "split_purity_s1.csv")

LIMITS = [0, 1, 2, 3, 4, 5, 6, 12]
XPOS = np.arange(len(LIMITS))
XLAB = ["0\n(RSO)", "1", "2", "3", "4", "5", "6", "All"]

FEAT = {
    "X1_Strong_Cont": ("$X_1$ (cont. signal)", "#0072B2", "-", "o"),
    "X6_Cont_Noise": ("$X_6$ (cont. noise)", "#0072B2", "--", "o"),
    "X4_Strong_Bin": ("$X_4$ (bin. signal, strong)", "#D55E00", "-", "s"),
    "X5_Weak_Bin": ("$X_5$ (bin. signal, weak)", "#E69F00", "-", "^"),
    "X7_Bin_Noise_A": ("$X_7$ (bin. noise)", "#555555", "--", "s"),
    "X8_Bin_Noise_B": ("$X_8$ (bin. noise)", "#999999", "--", "^"),
}
BIN = ["X4_Strong_Bin", "X5_Weak_Bin", "X7_Bin_Noise_A", "X8_Bin_Noise_B"]


def mean_ci(g):
    m = g.mean()
    se = g.std(ddof=1) / np.sqrt(g.count())
    return m, 1.96 * se


plt.rcParams.update({"font.size": 8.5, "axes.titlesize": 9,
                     "axes.labelsize": 8.5, "legend.fontsize": 7.5,
                     "font.family": "DejaVu Sans"})
fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.0))

# (a) binary-only AUC vs d_max
ax = axes[0, 0]
for meth, col, mk in (("Original DIFFI", "#CC79A7", "o"),
                      ("Raw rule", "#009E73", "s")):
    sub = d[d.method == meth]
    m, h = zip(*[mean_ci(sub[sub.d_max == L].auc_binary) for L in LIMITS])
    lab = "Original DIFFI" if meth == "Original DIFFI" else "Raw enrichment rule"
    ax.errorbar(XPOS, m, yerr=h, color=col, marker=mk, ms=4, lw=1.4,
                capsize=2, label=lab)
ax.axhline(0.5, color="black", lw=0.8, ls=":")
ax.set_ylim(0, 1.02)
ax.set_ylabel("Binary-only ranking AUC")
ax.set_title("(a) Binary-only AUC by binary depth limit", loc="left")
ax.legend(loc="center right", bbox_to_anchor=(1.0, 0.64), frameon=False)

# (b), (c) mean binary-feature scores vs d_max
for ax, meth, title in ((axes[0, 1], "Original DIFFI",
                         "(b) Original DIFFI: binary-feature scores"),
                        (axes[1, 0], "Raw rule",
                         "(c) Raw enrichment rule: binary-feature scores")):
    sub = d[d.method == meth]
    for f in BIN:
        lab, col, ls, mk = FEAT[f]
        m, h = zip(*[mean_ci(sub[sub.d_max == L][f]) for L in LIMITS])
        ax.errorbar(XPOS, m, yerr=h, color=col, ls=ls, marker=mk, ms=4,
                    lw=1.3, capsize=2, label=lab)
    ax.set_ylabel("Mean feature score")
    ax.set_title(title, loc="left")

for ax in (axes[0, 0], axes[0, 1], axes[1, 0]):
    ax.set_xticks(XPOS)
    ax.set_xticklabels(XLAB)
    ax.set_xlabel("Maximum depth for binary-feature contributions ($d_{\\max}$)")
    ax.grid(True, ls=":", lw=0.5, alpha=0.7)

# (d) split-level outlier enrichment by depth
ax = axes[1, 1]
rep_mean = (s.groupby(["rep", "feature", "depth"]).enrichment.mean()
            .reset_index())
for f in ["X1_Strong_Cont", "X6_Cont_Noise"] + BIN:
    lab, col, ls, mk = FEAT[f]
    sub = rep_mean[rep_mean.feature == f]
    depths = sorted(sub.depth.unique())
    m, h = zip(*[mean_ci(sub[sub.depth == k].enrichment) for k in depths])
    ax.errorbar(depths, m, yerr=h, color=col, ls=ls, marker=mk, ms=4,
                lw=1.3, capsize=2, label=lab)
ax.set_xticks(range(0, 8))
ax.set_xlabel("Split depth")
ax.set_ylabel("Mean outlier enrichment $E(v)$")
ax.set_title("(d) Split-level outlier enrichment by depth", loc="left")
ax.grid(True, ls=":", lw=0.5, alpha=0.7)

handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False,
           bbox_to_anchor=(0.5, -0.005))
fig.tight_layout(rect=(0, 0.075, 1, 1), h_pad=1.6, w_pad=1.6)
fig.savefig(OUT + "supp_fig_s2_depth_purity.pdf", bbox_inches="tight", pad_inches=0.08)
fig.savefig(OUT + "supp_fig_s2_depth_purity.png", dpi=300,
            bbox_inches="tight", pad_inches=0.08)
print("saved")
