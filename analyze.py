"""Statistical analysis of the experiment campaign.

Reports mean +/- std, 95% confidence intervals (t-distribution), Welch's
unequal-variance t-test against the proposed method, Cliff's delta as a
non-parametric effect size, and Holm-Bonferroni correction across the family
of comparisons for each metric.
"""
import json
import os
import numpy as np
import pandas as pd
from scipy import stats

RES = "results/results.csv"
METRICS = ["pdr", "goodput", "ee", "jain", "t_tot_ms"]
LOWER_IS_BETTER = {"pdr", "t_tot_ms"}


def ci95(x):
    x = np.asarray(x, dtype=float)
    if len(x) < 2:
        return 0.0
    return stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))


def cliffs_delta(a, b):
    a, b = np.asarray(a), np.asarray(b)
    gt = sum((x > y) for x in a for y in b)
    lt = sum((x < y) for x in a for y in b)
    return (gt - lt) / (len(a) * len(b))


def delta_label(d):
    d = abs(d)
    return ("negligible" if d < 0.147 else "small" if d < 0.33
            else "medium" if d < 0.474 else "large")


def holm(pvals):
    idx = np.argsort(pvals)
    adj = np.empty(len(pvals))
    running = 0.0
    for rank, i in enumerate(idx):
        val = (len(pvals) - rank) * pvals[i]
        running = max(running, val)
        adj[i] = min(running, 1.0)
    return adj


def summarise(df, group="method"):
    rows = []
    for m, g in df.groupby(group, sort=False):
        r = {"method": m, "n_seeds": len(g)}
        for k in METRICS:
            r[f"{k}_mean"] = g[k].mean()
            r[f"{k}_std"] = g[k].std(ddof=1) if len(g) > 1 else 0.0
            r[f"{k}_ci"] = ci95(g[k])
        rows.append(r)
    return pd.DataFrame(rows)


def compare(df, ref="PROPOSED+"):
    out = []
    for k in METRICS:
        a = df[df.method == ref][k].values
        raw = []
        others = [m for m in df.method.unique() if m != ref]
        for m in others:
            b = df[df.method == m][k].values
            if len(a) < 2 or len(b) < 2:
                raw.append((m, np.nan, np.nan)); continue
            t, p = stats.ttest_ind(a, b, equal_var=False)
            raw.append((m, p, cliffs_delta(a, b)))
        ps = np.array([r[1] for r in raw], dtype=float)
        mask = ~np.isnan(ps)
        adj = np.full(len(ps), np.nan)
        if mask.any():
            adj[mask] = holm(ps[mask])
        for (m, p, d), pa in zip(raw, adj):
            out.append(dict(metric=k, ref=ref, other=m, p_raw=p, p_holm=pa,
                            cliffs_delta=d, effect=delta_label(d) if d == d else "",
                            ref_mean=a.mean() if len(a) else np.nan,
                            other_mean=df[df.method == m][k].mean()))
    return pd.DataFrame(out)


if __name__ == "__main__":
    df = pd.read_csv(RES)
    main = df[df.exp == "main"]
    print("=" * 100)
    print("MAIN RESULTS (N=30), mean +/- std over seeds")
    print("=" * 100)
    s = summarise(main)
    disp = s[["method", "n_seeds"] + [f"{k}_{q}" for k in METRICS for q in ("mean", "std")]]
    print(disp.round(3).to_string(index=False))
    s.to_csv("results/summary_main.csv", index=False)

    if "PROPOSED+" in set(main.method):
        cmp = compare(main, "PROPOSED+")
        print("\nSIGNIFICANCE vs PROPOSED+ (Welch t-test, Holm-corrected)")
        print(cmp.round(4).to_string(index=False))
        cmp.to_csv("results/significance.csv", index=False)

    sc = df[df.exp == "scale"]
    if len(sc):
        print("\n" + "=" * 100)
        print("SCALABILITY (mean PDR %)")
        print(sc.pivot_table(index="n_users", columns="method", values="pdr").round(2).to_string())
        print("\nSCALABILITY (mean goodput Mbps)")
        print(sc.pivot_table(index="n_users", columns="method", values="goodput").round(1).to_string())

    se = df[df.exp == "sens"]
    if len(se):
        print("\n" + "=" * 100)
        print("SENSITIVITY to w4 (reliability weight)")
        print(se.pivot_table(index="w_pdr", columns="method",
                             values=["pdr", "ee"]).round(3).to_string())

    cold_path = "results/coldstart.json"
    if os.path.exists(cold_path):
        print("\n" + "=" * 100)
        print("COLD-START (mean PDR % over seeds, by training-episode window)")
        cold = json.load(open(cold_path))
        rows = []
        for m, seeds in cold.items():
            arr = np.array(seeds)                       # (n_seeds, n_episodes)
            rows.append(dict(
                method=m, n_seeds=arr.shape[0],
                ep1=arr[:, 0].mean(), ep1_ci=ci95(arr[:, 0]),
                ep1_5=arr[:, :5].mean(), ep1_5_ci=ci95(arr[:, :5].mean(axis=1)),
                ep1_25=arr.mean(), ep1_25_ci=ci95(arr.mean(axis=1)),
            ))
        cold_df = pd.DataFrame(rows)
        print(cold_df.round(3).to_string(index=False))
        cold_df.to_csv("results/coldstart_summary.csv", index=False)

        # significance of cold-start PDR: PROPOSED+ vs each baseline, ep1-5 window
        if "PROPOSED+" in cold:
            ref = np.array(cold["PROPOSED+"])[:, :5].mean(axis=1)
            raw = []
            for m, seeds in cold.items():
                if m == "PROPOSED+":
                    continue
                other = np.array(seeds)[:, :5].mean(axis=1)
                if len(ref) > 1 and len(other) > 1:
                    _, p = stats.ttest_ind(ref, other, equal_var=False)
                    d = cliffs_delta(ref, other)
                    raw.append((m, p, d))
            if raw:
                ps = np.array([r[1] for r in raw])
                adj = holm(ps)
                print("\nCOLD-START significance (episodes 1-5, PROPOSED+ vs baseline):")
                for (m, p, d), pa in zip(raw, adj):
                    print(f"  {m:12s} p_raw={p:.4f}  p_holm={pa:.4f}  "
                          f"cliffs_delta={d:+.3f} ({delta_label(d)})")

