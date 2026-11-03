"""
Training / evaluation driver.

Methods
-------
  RR, WF, GRD      : classical non-learning heuristics
  SAC, TD3, PPO    : DRL baselines -- identical network capacity, optimiser,
                     batch size and number of environment interactions
  SAC+EHS          : ensemble seeding with best-of-seeds selection, no GWO
  SAC+GWO          : GWO refinement from a random initial population
  PROPOSED         : SAC + EHS + elitist GWO + behaviour cloning + adaptive gate
  PROPOSED-NG      : PROPOSED with the gate disabled (GWO at every slot)
  PROPOSED+NOEHS   : matched control for PROPOSED+, added for the R2 revision
                     (Reviewer #3). Identical adaptive gate and identical
                     temporal warm-start to PROPOSED+; the ONLY difference is
                     that the EHS heuristic-anchor ensemble is replaced by the
                     same number of random seeds. This isolates whether EHS
                     itself -- as opposed to gating/warm-start/architecture --
                     is responsible for PROPOSED+'s cold-start advantage.

Every method is evaluated on the SAME set of evaluation episode seeds, so the
channel realisations and packet arrivals are identical across methods.
"""

import time
import numpy as np

from .env import SixGEdgeEnv, project_action, inverse_project
from .heuristics import ANCHORS, ENSEMBLE, gwo_refine
from .agents import SAC, TD3, PPO, HP

HEURISTIC_METHODS = ("RR", "WF", "GRD", "URG", "PF")
ALG = dict(pop=12, iters=10, gate_margin=0.0)


def make_agent(method, sdim, adim, seed):
    if method == "TD3":
        return TD3(sdim, adim, seed=seed)
    if method == "PPO":
        return PPO(sdim, adim, seed=seed)
    if method.startswith("SAC") or method.startswith("PROPOSED"):
        return SAC(sdim, adim, seed=seed)
    return None


def decide(env, method, agent, state, rng, cfg, deterministic):
    """Return (b, p, rho), bc_target, timing."""
    if method in HEURISTIC_METHODS:
        t0 = time.perf_counter()
        act = ANCHORS[method](env)
        return act, None, dict(t_inf=time.perf_counter() - t0, t_ref=0.0, gwo=0)

    t0 = time.perf_counter()
    a_raw = agent.act(state, deterministic=deterministic)
    b, p, rho = project_action(a_raw, env.N)
    t_inf = time.perf_counter() - t0

    if method in ("SAC", "TD3", "PPO"):
        return (b, p, rho), None, dict(t_inf=t_inf, t_ref=0.0, gwo=0)

    # "NOEHS" marks the Reviewer-#3 matched control: same gate, same warm
    # start as PROPOSED+, but EHS anchors are replaced by random seeds of
    # the same count (len(ENSEMBLE)), so EHS is the ONLY thing that differs
    # from PROPOSED+.
    is_noehs = "NOEHS" in method
    use_ehs = (("EHS" in method) or method.startswith("PROPOSED")) and not is_noehs
    use_warm = method.startswith("PROPOSED+")
    f_sac = env.evaluate(b, p, rho)[0]
    t_ref = 0.0

    if method == "SAC+EHS":
        t0 = time.perf_counter()
        best, best_f = (b, p, rho), f_sac
        for key in ENSEMBLE:
            cand = ANCHORS[key](env)
            fc = env.evaluate(*cand)[0]
            if fc > best_f:
                best, best_f = cand, fc
        t_ref = time.perf_counter() - t0
        return best, inverse_project(*best), dict(t_inf=t_inf, t_ref=t_ref, gwo=0)

    # ---- GWO-refined variants ----
    t0 = time.perf_counter()
    if use_ehs:
        seeds = [(b, p, rho)] + [ANCHORS[k](env) for k in ENSEMBLE]
    elif is_noehs:
        # Matched-SIZE random control: len(ENSEMBLE) random seeds instead of
        # the len(ENSEMBLE) heuristic anchors used by PROPOSED+, so the
        # candidate pool size feeding GWO is identical and only the SOURCE
        # of the seeds (heuristic vs random) differs.
        seeds = [(b, p, rho)]
        for _ in range(len(ENSEMBLE)):
            seeds.append((rng.dirichlet(np.ones(env.N)),
                          rng.dirichlet(np.ones(env.N)),
                          float(rng.uniform(0.2, 1.0))))
    else:
        seeds = [(b, p, rho)]
        for _ in range(4):
            seeds.append((rng.dirichlet(np.ones(env.N)),
                          rng.dirichlet(np.ones(env.N)),
                          float(rng.uniform(0.2, 1.0))))
    # Temporal warm start is applied identically regardless of EHS/no-EHS,
    # so that PROPOSED+ and PROPOSED+NOEHS differ ONLY in the seed source.
    if use_warm and cfg.get("_prev") is not None:
        seeds.append(cfg["_prev"])
    t_ref += time.perf_counter() - t0

    if cfg.get("gate", False):
        t0 = time.perf_counter()
        f_anchor = max(env.evaluate(*s)[0] for s in seeds[1:])
        t_ref += time.perf_counter() - t0
        if f_sac >= f_anchor - cfg.get("gate_margin", 0.0):
            if use_warm:
                cfg["_prev"] = (b, p, rho)
            return (b, p, rho), None, dict(t_inf=t_inf, t_ref=t_ref, gwo=0)

    t0 = time.perf_counter()
    best, fbest, _ = gwo_refine(env, seeds, pop_size=cfg.get("pop", 10),
                                iters=cfg.get("iters", 10), rng=rng)
    t_ref += time.perf_counter() - t0
    if use_warm:
        cfg["_prev"] = best
    return best, inverse_project(*best), dict(t_inf=t_inf, t_ref=t_ref, gwo=1)


def run_one(method, n_users=30, seed=0, n_train_ep=120, n_eval_ep=15,
            cfg=None, log_curve=True):
    cfg = dict(ALG, **(cfg or {}))
    if method == "PROPOSED" or method.startswith("PROPOSED+"):
        cfg.setdefault("gate", True)

    env = SixGEdgeEnv(n_users=n_users, seed=1000 + seed)
    rng = np.random.default_rng(2000 + seed)
    np.random.seed(3000 + seed)
    agent = make_agent(method, env.state_dim, env.action_dim, seed)
    curve = []

    # -------------------------------------------------------------- training
    if agent is not None:
        for ep in range(n_train_ep):
            s = env.reset(seed=10_000 + seed * 977 + ep)
            cfg["_prev"] = None
            ep_r = 0.0
            for _ in range(env.T):
                (b, p, rho), bc, _ = decide(env, method, agent, s, rng, cfg,
                                            deterministic=False)
                s2, r, done, _ = env.step(b, p, rho)
                a_store = inverse_project(b, p, rho)
                agent.store(s.astype(np.float32), a_store.astype(np.float32),
                            np.float32(r), s2.astype(np.float32), np.float32(done))
                agent.update(bc_target=bc)
                s = s2
                ep_r += r
            if log_curve:
                st = env.episode_stats()
                curve.append((ep_r / env.T, st["pdr"], st["goodput"]))

    # ------------------------------------------------------------ evaluation
    rows, t_infs, t_refs, gwos = [], [], [], []
    for k in range(n_eval_ep):
        s = env.reset(seed=777_000 + k)
        cfg["_prev"] = None
        thr_l, jain_l, ee_l, rho_l = [], [], [], []
        for _ in range(env.T):
            (b, p, rho), _, tm = decide(env, method, agent, s, rng, cfg,
                                        deterministic=True)
            t_infs.append(tm["t_inf"]); t_refs.append(tm["t_ref"]); gwos.append(tm["gwo"])
            s, r, done, info = env.step(b, p, rho)
            thr_l.append(float(info["thr"])); jain_l.append(float(info["jain"]))
            ee_l.append(float(info["ee"])); rho_l.append(rho)
        st = env.episode_stats()
        rows.append(dict(thr=np.mean(thr_l), jain=np.mean(jain_l), ee=np.mean(ee_l),
                         rho=np.mean(rho_l), pdr=st["pdr"], goodput=st["goodput"],
                         offered=st["offered"]))

    agg = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
    agg.update(method=method, seed=seed, n_users=n_users,
               t_inf_ms=float(np.mean(t_infs) * 1e3),
               t_ref_ms=float(np.mean(t_refs) * 1e3),
               gwo_rate=float(np.mean(gwos)))
    agg["t_tot_ms"] = agg["t_inf_ms"] + agg["t_ref_ms"]
    return agg, curve
