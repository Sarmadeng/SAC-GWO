"""
Reviewer #3 (R2 revision) matched cold-start ablation.

Runs PROPOSED+ and PROPOSED+NOEHS under IDENTICAL seeding (env seed, rng
seed, torch seed, per-episode reset seed) for the same 10 seeds and the same
COLD_EP=25 training episodes used for every other method in the paper's
Table 5 cold-start experiment. The two methods differ ONLY in whether the
candidate seed pool handed to GWO is the EHS heuristic-anchor ensemble
(PROPOSED+) or the same number of random seeds (PROPOSED+NOEHS); gating,
temporal warm-start, SAC agent, environment and evaluation protocol are
all identical (see sixg/runner.py for the exact code path).

This is real, executed code -- not a closed-form approximation -- using the
unmodified env.py / heuristics.py / agents.py extracted verbatim from the
author's own Colab notebook (heliyon_6g_colab.ipynb), with only runner.py
patched to add the PROPOSED+NOEHS control.
"""
import json, time
import numpy as np
from scipy import stats

from sixg.runner import run_one

COLD_EP = 25
SEEDS = list(range(10))
METHODS = ["PROPOSED+", "PROPOSED+NOEHS"]


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


def main():
    t_start = time.time()
    cold = {}
    for m in METHODS:
        rows = []
        for s in SEEDS:
            t0 = time.time()
            _, curve = run_one(m, n_users=30, seed=s, n_train_ep=COLD_EP, n_eval_ep=1)
            pdr_curve = [row[1] for row in curve]   # per-episode PDR, length COLD_EP
            rows.append(pdr_curve)
            print(f"[{m:16s}] seed={s} ep1={pdr_curve[0]:6.2f} "
                  f"ep1-5={np.mean(pdr_curve[:5]):6.2f}  ({time.time()-t0:.1f}s, "
                  f"total {time.time()-t_start:.0f}s)", flush=True)
        cold[m] = rows

    json.dump(cold, open("results/coldstart_matched.json", "w"))

    summary = []
    for m, rows in cold.items():
        arr = np.array(rows)
        summary.append(dict(
            method=m, n_seeds=arr.shape[0],
            ep1=arr[:, 0].mean(), ep1_ci=ci95(arr[:, 0]),
            ep1_5=arr[:, :5].mean(), ep1_5_ci=ci95(arr[:, :5].mean(axis=1)),
            ep1_25=arr.mean(), ep1_25_ci=ci95(arr.mean(axis=1)),
        ))

    print("\n" + "=" * 70)
    print("COLD-START SUMMARY (mean PDR %, 95% CI)")
    print("=" * 70)
    for r in summary:
        print(f"  {r['method']:16s} ep1={r['ep1']:.2f}±{r['ep1_ci']:.2f}  "
              f"ep1-5={r['ep1_5']:.2f}±{r['ep1_5_ci']:.2f}  "
              f"ep1-25={r['ep1_25']:.2f}±{r['ep1_25_ci']:.2f}")

    with open("results/coldstart_matched_summary.csv", "w") as f:
        f.write("method,n_seeds,ep1,ep1_ci,ep1_5,ep1_5_ci,ep1_25,ep1_25_ci\n")
        for r in summary:
            f.write(",".join(str(r[k]) for k in
                     ["method", "n_seeds", "ep1", "ep1_ci", "ep1_5", "ep1_5_ci",
                      "ep1_25", "ep1_25_ci"]) + "\n")

    # ---- the test Reviewer #3 actually asked for: PROPOSED+ vs PROPOSED+NOEHS
    ref = np.array(cold["PROPOSED+"])[:, :5].mean(axis=1)
    other = np.array(cold["PROPOSED+NOEHS"])[:, :5].mean(axis=1)
    t, p = stats.ttest_ind(ref, other, equal_var=False)
    d = cliffs_delta(ref, other)
    print("\n" + "=" * 70)
    print("SIGNIFICANCE: PROPOSED+ vs PROPOSED+NOEHS, episodes 1-5 mean PDR")
    print("=" * 70)
    print(f"  PROPOSED+      mean={ref.mean():.3f}  std={ref.std(ddof=1):.3f}")
    print(f"  PROPOSED+NOEHS mean={other.mean():.3f}  std={other.std(ddof=1):.3f}")
    print(f"  Welch t={t:.3f}  p={p:.4f}   Cliff's delta={d:+.3f} ({delta_label(d)})")

    result = dict(ref_mean=float(ref.mean()), ref_std=float(ref.std(ddof=1)),
                  other_mean=float(other.mean()), other_std=float(other.std(ddof=1)),
                  t=float(t), p=float(p), cliffs_delta=float(d),
                  effect=delta_label(d))
    json.dump(result, open("results/coldstart_matched_significance.json", "w"), indent=2)
    print(f"\nTotal time: {time.time()-t_start:.0f}s")


if __name__ == "__main__":
    main()
