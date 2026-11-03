"""Figure for the Reviewer #3 (R2) matched-ablation evidence: cold-start PDR
(episodes 1-5) for the matched control versus the headline configuration and
the two single-component ablations already in the paper's Table 5."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

plt.rcParams.update({
    "font.size": 11, "axes.labelsize": 12, "axes.titlesize": 12,
    "legend.fontsize": 10, "figure.dpi": 110, "savefig.dpi": 300,
    "savefig.bbox": "tight", "axes.grid": True, "grid.alpha": 0.3,
})

# Published Table 5 numbers (episodes 1-5), from coldstart_summary.csv
PUBLISHED = {
    "SAC+GWO":  (7.541348369821595, 1.1147306627865337),
    "SAC+EHS":  (7.447244068088377, 1.1621856716626409),
}

cold = json.load(open("results/coldstart_matched.json"))


def ci95(x):
    x = np.asarray(x, dtype=float)
    return stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))


noehs = np.array(cold["PROPOSED+NOEHS"])[:, :5].mean(axis=1)
propp = np.array(cold["PROPOSED+"])[:, :5].mean(axis=1)

order = ["SAC+GWO", "SAC+EHS", "PROPOSED+NOEHS", "PROPOSED+"]
means = [PUBLISHED["SAC+GWO"][0], PUBLISHED["SAC+EHS"][0],
         noehs.mean(), propp.mean()]
cis = [PUBLISHED["SAC+GWO"][1], PUBLISHED["SAC+EHS"][1],
       ci95(noehs), ci95(propp)]
colors = ["#7f7f7f", "#bcbd22", "#d62728", "#12355b"]

fig, ax = plt.subplots(figsize=(6.6, 4.2))
bars = ax.bar(order, means, yerr=cis, capsize=4, color=colors, alpha=0.88)
ax.set_ylabel("Packet drop rate, % (episodes 1–5)")
ax.set_title("Matched EHS ablation (Reviewer #3, R2):\nPROPOSED+NOEHS isolates the EHS contribution")
for b, m in zip(bars, means):
    ax.text(b.get_x() + b.get_width() / 2, m + 0.35, f"{m:.2f}", ha="center", fontsize=10)
plt.xticks(rotation=12, ha="right")
plt.tight_layout()
plt.savefig("results/fig_r3_matched_ablation.png")
print("saved results/fig_r3_matched_ablation.png")

t, p = stats.ttest_ind(propp, noehs, equal_var=False)
print(f"PROPOSED+ ({propp.mean():.2f}) vs PROPOSED+NOEHS ({noehs.mean():.2f}): "
      f"Welch t={t:.3f}, p={p:.4f}")
