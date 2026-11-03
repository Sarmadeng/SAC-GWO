"""Publication-quality figures (300 dpi) from the campaign results."""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES = "results/results.csv"
CURVES = "results/curves.json"
OUT = "results/figs"
import os
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "font.size": 11, "axes.labelsize": 12, "axes.titlesize": 12,
    "legend.fontsize": 10, "figure.dpi": 110, "savefig.dpi": 300,
    "savefig.bbox": "tight", "axes.grid": True, "grid.alpha": 0.3,
})

STYLE = {
    "PROPOSED+":   dict(color="#12355b", lw=2.6, ls="-",  marker="o"),
    "PROPOSED":    dict(color="#1f77b4", lw=2.2, ls="-",  marker="s"),
    "PROPOSED-NG": dict(color="#5aa9e6", lw=1.8, ls="--", marker="^"),
    "SAC+GWO":     dict(color="#7f7f7f", lw=1.6, ls="-.", marker="v"),
    "SAC+EHS":     dict(color="#bcbd22", lw=1.6, ls="-.", marker="D"),
    "SAC":         dict(color="#d62728", lw=1.6, ls="--", marker="x"),
    "TD3":         dict(color="#ff7f0e", lw=1.6, ls="--", marker="+"),
    "PPO":         dict(color="#2ca02c", lw=1.6, ls="--", marker="*"),
    "RR":          dict(color="#8c564b", lw=1.4, ls=":",  marker="."),
    "WF":          dict(color="#9467bd", lw=1.4, ls=":",  marker="."),
    "GRD":         dict(color="#e377c2", lw=1.4, ls=":",  marker="."),
    "PF":          dict(color="#17becf", lw=1.4, ls=":",  marker="."),
}


def smooth(x, w=9):
    return pd.Series(x).rolling(w, min_periods=1).mean().values


def fig_convergence(curves, idx, ylabel, fname, title):
    plt.figure(figsize=(7, 4.4))
    for m, c in curves.items():
        if m not in STYLE:
            continue
        y = smooth([row[idx] for row in c])
        st = dict(STYLE[m]); st.pop("marker", None)
        plt.plot(y, label=m, **st)
    plt.xlabel("Training episode"); plt.ylabel(ylabel)
    plt.title(title); plt.legend(ncol=2)
    plt.savefig(f"{OUT}/{fname}"); plt.close()


def fig_bar_ci(df, metric, ylabel, fname, title, order):
    plt.figure(figsize=(7.4, 4.2))
    means, cis, labs = [], [], []
    from scipy import stats
    for m in order:
        g = df[df.method == m][metric].values
        if len(g) == 0:
            continue
        means.append(g.mean()); labs.append(m)
        cis.append(stats.t.ppf(0.975, len(g) - 1) * g.std(ddof=1) / np.sqrt(len(g))
                   if len(g) > 1 else 0.0)
    cols = [STYLE.get(m, {}).get("color", "#333") for m in labs]
    plt.bar(labs, means, yerr=cis, capsize=4, color=cols, alpha=0.85)
    plt.ylabel(ylabel); plt.title(title)
    plt.xticks(rotation=25, ha="right")
    plt.savefig(f"{OUT}/{fname}"); plt.close()


def fig_scale(df, metric, ylabel, fname, title):
    plt.figure(figsize=(7, 4.4))
    for m in df.method.unique():
        g = df[df.method == m].groupby("n_users")[metric]
        x = sorted(g.groups.keys())
        mu = [g.get_group(k).mean() for k in x]
        sd = [g.get_group(k).std(ddof=1) if len(g.get_group(k)) > 1 else 0 for k in x]
        st = dict(STYLE.get(m, dict(color="#333", lw=1.5, ls="-", marker="o")))
        plt.errorbar(x, mu, yerr=sd, capsize=3, label=m, **st)
    plt.xlabel("Number of users N"); plt.ylabel(ylabel)
    plt.title(title); plt.legend()
    plt.savefig(f"{OUT}/{fname}"); plt.close()


if __name__ == "__main__":
    df = pd.read_csv(RES)
    main = df[df.exp == "main"]
    order = [m for m in ["RR", "WF", "GRD", "PF", "PPO", "TD3", "SAC",
                         "SAC+EHS", "SAC+GWO", "PROPOSED-NG", "PROPOSED",
                         "PROPOSED+"] if m in set(main.method)]

    try:
        curves = json.load(open(CURVES))
        fig_convergence(curves, 1, "Packet drop rate (%)", "fig_conv_pdr.png",
                        "Convergence: deadline-violation rate")
        fig_convergence(curves, 2, "Goodput (Mbps)", "fig_conv_goodput.png",
                        "Convergence: delivered goodput")
        fig_convergence(curves, 0, "Average reward per slot", "fig_conv_reward.png",
                        "Convergence: objective F (Eq. 8)")
    except FileNotFoundError:
        pass

    fig_bar_ci(main, "pdr", "Packet drop rate (%)", "fig_pdr.png",
               "Reliability (mean, 95% CI)", order)
    fig_bar_ci(main, "goodput", "Goodput (Mbps)", "fig_goodput.png",
               "Delivered goodput (mean, 95% CI)", order)
    fig_bar_ci(main, "ee", "Energy efficiency (Mbit/J)", "fig_ee.png",
               "Energy efficiency (mean, 95% CI)", order)
    fig_bar_ci(main, "jain", "Jain index (satisfaction ratio)", "fig_jain.png",
               "Demand-aware fairness (mean, 95% CI)", order)

    sc = df[df.exp == "scale"]
    if len(sc):
        fig_scale(sc, "pdr", "Packet drop rate (%)", "fig_scale_pdr.png",
                  "Scalability: reliability vs user density")
        fig_scale(sc, "goodput", "Goodput (Mbps)", "fig_scale_goodput.png",
                  "Scalability: goodput vs user density")

    se = df[df.exp == "sens"]
    if len(se):
        plt.figure(figsize=(7, 4.2))
        for m in se.method.unique():
            g = se[se.method == m].groupby("w_pdr")
            x = sorted(g.groups.keys())
            plt.plot(x, [g.get_group(k)["pdr"].mean() for k in x],
                     label=f"{m} PDR", **{k: v for k, v in STYLE.get(m, {}).items()})
        plt.xlabel("Reliability weight $w_4$"); plt.ylabel("Packet drop rate (%)")
        plt.title("Sensitivity to the reliability weight"); plt.legend()
        plt.savefig(f"{OUT}/fig_sensitivity.png"); plt.close()

    cold_path = "results/coldstart.json"
    if os.path.exists(cold_path):
        cold = json.load(open(cold_path))
        plt.figure(figsize=(7, 4.4))
        for m, seeds in cold.items():
            if m not in STYLE:
                continue
            arr = np.array(seeds)
            mu = arr.mean(axis=0)
            sd = arr.std(axis=0, ddof=1) if arr.shape[0] > 1 else np.zeros_like(mu)
            x = np.arange(1, len(mu) + 1)
            st = dict(STYLE[m]); st.pop("marker", None)
            plt.plot(x, mu, label=m, **st)
            plt.fill_between(x, mu - sd, mu + sd, alpha=0.12,
                             color=STYLE[m]["color"])
        plt.xlabel("Training episode"); plt.ylabel("Packet drop rate (%)")
        plt.title("Cold-start reliability (mean $\\pm$ std over 10 seeds)")
        plt.legend(ncol=2)
        plt.savefig(f"{OUT}/fig_coldstart.png"); plt.close()

    print("figures written to", OUT)
