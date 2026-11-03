"""
Unified, resumable experiment driver.

Run:
    python run_experiment.py

If interrupted (Ctrl+C, crash, power loss), simply run it again: completed
(exp, method, seed, n_users, w_pdr) combinations found in results.csv are
skipped automatically, so nothing is recomputed and nothing is duplicated.

Produces
--------
results/results.csv    -- one row per (experiment, method, seed) evaluation
results/curves.json    -- per-episode training curves (seed 0) for convergence plots
results/coldstart.json -- per-episode PDR during the first COLD_EP episodes, all seeds

Experiments
-----------
main   : N=30, 10 seeds (3 for the non-learning heuristics), all methods,
         used for the main comparison table + significance tests.
scale  : N in {50, 70, 90}, 5 seeds, a subset of methods, for scalability.
sens   : reliability weight w4 in {0.3, 0.6, 1.2}, 3 seeds, PROPOSED+ vs TD3.
cold   : first COLD_EP training episodes, 10 seeds, tests the paper's central
         cold-start claim directly (this is NOT the same as final performance).
"""
import csv, json, os, time
import numpy as np

from sixg.runner import run_one

OUT = "results"
RES = os.path.join(OUT, "results.csv")
CURVES = os.path.join(OUT, "curves.json")
COLD = os.path.join(OUT, "coldstart.json")

FIELDS = ["exp", "method", "seed", "n_users", "goodput", "offered", "pdr", "ee",
          "jain", "thr", "rho", "t_inf_ms", "t_ref_ms", "t_tot_ms", "gwo_rate",
          "w_pdr"]

TRAIN_EP, EVAL_EP = 120, 10
COLD_EP = 25

MAIN_METHODS = ["RR", "WF", "GRD", "PF", "PPO", "TD3", "SAC",
                "SAC+EHS", "SAC+GWO", "PROPOSED-NG", "PROPOSED", "PROPOSED+"]
MAIN_SEEDS = list(range(10))
# RR/WF/GRD/PF are deterministic (no learnable parameters, no stochastic
# policy) and are evaluated on the same fixed held-out episode set as every
# other method, so repeating them under different "seeds" reproduces bit-
# identical numbers -- there is no extra variance to sample.  One seed is
# both sufficient and correct; report their std as exactly 0 in the paper
# with this justification, not as a measurement artefact.
HEURISTIC_SEEDS = [0]

SCALE_N = [50, 70, 90]
SCALE_METHODS = ["RR", "PPO", "TD3", "SAC", "PROPOSED+"]
SCALE_SEEDS = list(range(5))

SENS_W = [0.3, 0.6, 1.2]
SENS_METHODS = ["PROPOSED+", "TD3"]
SENS_SEEDS = list(range(3))

COLD_METHODS = ["PPO", "TD3", "SAC", "SAC+GWO", "SAC+EHS", "PROPOSED+"]
COLD_SEEDS = list(range(10))


# ----------------------------------------------------------------------- I/O
def load_done():
    """Set of (exp, method, seed, n_users, w_pdr) already present in results.csv."""
    if not os.path.exists(RES):
        return set()
    import pandas as pd
    df = pd.read_csv(RES)
    keys = set()
    for _, r in df.iterrows():
        keys.add((r["exp"], r["method"], int(r["seed"]), int(r["n_users"]),
                  round(float(r["w_pdr"]), 3)))
    return keys


def writer():
    os.makedirs(OUT, exist_ok=True)
    new = not os.path.exists(RES)
    f = open(RES, "a", newline="")
    w = csv.DictWriter(f, fieldnames=FIELDS)
    if new:
        w.writeheader()
    return f, w


def record(w, f, exp, agg, w_pdr=0.60):
    row = {k: agg.get(k, "") for k in FIELDS}
    row["exp"] = exp
    row["w_pdr"] = w_pdr
    w.writerow(row)
    f.flush()


def load_json(path):
    if os.path.exists(path):
        return json.load(open(path))
    return {}


def save_json(obj, path):
    json.dump(obj, open(path, "w"))


# --------------------------------------------------------------------- main
def main():
    t_start = time.time()
    f, w = writer()
    done = load_done()
    curves = load_json(CURVES)
    cold = load_json(COLD)
    print(f"resuming with {len(done)} completed runs already in {RES}", flush=True)

    def log(tag, extra, agg, t0):
        print(f"[{tag}] {extra:16s} pdr={agg['pdr']:6.2f} good={agg['goodput']:6.1f} "
              f"ee={agg['ee']:6.2f} jain={agg['jain']:.3f} tref={agg['t_ref_ms']:6.2f}ms "
              f"gwo={agg['gwo_rate']:.2f}  ({time.time()-t0:.0f}s, total "
              f"{time.time()-t_start:.0f}s)", flush=True)

    # ------------------------------------------------------------- 1. main
    for m in MAIN_METHODS:
        seeds = HEURISTIC_SEEDS if m in ("RR", "WF", "GRD", "PF") else MAIN_SEEDS
        for s in seeds:
            key = ("main", m, s, 30, 0.6)
            if key in done:
                continue
            t0 = time.time()
            agg, curve = run_one(m, n_users=30, seed=s,
                                 n_train_ep=TRAIN_EP, n_eval_ep=EVAL_EP)
            record(w, f, "main", agg)
            done.add(key)
            if s == 0 and curve:
                curves[m] = curve
                save_json(curves, CURVES)
            log("main", f"{m} s{s}", agg, t0)

    # ---------------------------------------------------------- 2. scaling
    for n in SCALE_N:
        for m in SCALE_METHODS:
            seeds = SCALE_SEEDS[:3] if m == "RR" else SCALE_SEEDS
            for s in seeds:
                key = ("scale", m, s, n, 0.6)
                if key in done:
                    continue
                t0 = time.time()
                agg, _ = run_one(m, n_users=n, seed=s, n_train_ep=TRAIN_EP,
                                 n_eval_ep=EVAL_EP, log_curve=False)
                record(w, f, "scale", agg)
                done.add(key)
                log(f"scale N={n}", f"{m} s{s}", agg, t0)

    # ------------------------------------------------------- 3. sensitivity
    import sixg.env as E
    base_w = E.CFG["W_PDR"]
    for wp in SENS_W:
        E.CFG["W_PDR"] = wp
        for m in SENS_METHODS:
            for s in SENS_SEEDS:
                key = ("sens", m, s, 30, round(wp, 3))
                if key in done:
                    continue
                t0 = time.time()
                agg, _ = run_one(m, n_users=30, seed=s, n_train_ep=TRAIN_EP,
                                 n_eval_ep=EVAL_EP, log_curve=False)
                record(w, f, "sens", agg, w_pdr=wp)
                done.add(key)
                log(f"sens w4={wp}", f"{m} s{s}", agg, t0)
    E.CFG["W_PDR"] = base_w

    # ------------------------------------------------------- 4. cold-start
    for m in COLD_METHODS:
        if m in cold and len(cold[m]) >= len(COLD_SEEDS):
            continue
        rows = cold.get(m, [])
        for s in range(len(rows), len(COLD_SEEDS)):
            t0 = time.time()
            _, curve = run_one(m, n_users=30, seed=s, n_train_ep=COLD_EP, n_eval_ep=1)
            rows.append([row[1] for row in curve])       # per-episode PDR
            cold[m] = rows
            save_json(cold, COLD)
            arr = np.array(rows)
            print(f"[cold] {m:12s} s{s}  ep1={arr[-1][0]:6.2f}  "
                  f"ep1-5_mean_so_far={arr[:, :5].mean():6.2f}  "
                  f"({time.time()-t0:.0f}s, total {time.time()-t_start:.0f}s)", flush=True)

    f.close()
    print(f"ALL EXPERIMENTS DONE in {time.time()-t_start:.0f}s", flush=True)


if __name__ == "__main__":
    main()
