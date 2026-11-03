"""
Deterministic heuristic anchors and an elitist, vectorised Grey Wolf Optimizer.

GWO follows Mirjalili, Mirjalili & Lewis, "Grey Wolf Optimizer", Advances in
Engineering Software 69:46-61, 2014:
    a decreases linearly from 2 to 0
    A = 2*a*r1 - a,  C = 2*r2
    D_k = |C_k X_k - X|,  X_k' = X_k - A_k D_k   for k in {alpha, beta, delta}
    X(t+1) = mean(X_alpha', X_beta', X_delta')
Positions are projected onto the feasible set after every update, and the
returned solution is the arg-max over every point ever evaluated (elitism).

Each candidate is a vector in R^{2N+1} -> (b, p, rho) via project_batch.
"""

import numpy as np
from .env import project_batch, inverse_project

# Classical heuristics operate at full transmit power, as in their standard
# formulations; the learned policies may scale total power via rho.
FULL_POWER = 1.0


def h_round_robin(env):
    n = env.N
    return np.ones(n) / n, np.ones(n) / n, FULL_POWER


def h_greedy(env):
    g = env._gain()
    w = g / (g.sum() + 1e-30)
    return w.copy(), w.copy(), FULL_POWER


def h_water_filling(env, iters=40):
    """Equal bandwidth split, power by water-filling (bisection on the level)."""
    n = env.N
    b = np.ones(n) / n
    g = env._gain()
    noise = env.N0 * (b * env.W) + env.I_inter
    inv_g = noise / (g + 1e-30)
    lo, hi = float(inv_g.min()), float(inv_g.max() + env.P_max)
    for _ in range(iters):
        mu = 0.5 * (lo + hi)
        if np.maximum(mu - inv_g, 0.0).sum() > env.P_max:
            hi = mu
        else:
            lo = mu
    p = np.maximum(0.5 * (lo + hi) - inv_g, 0.0)
    if p.sum() <= 1e-12:
        p = np.ones(n)
    return b, p / p.sum(), FULL_POWER


def h_urgency(env):
    """Backlog- and deadline-aware anchor (part of the proposed EHS ensemble).

    Not a classical heuristic: it is introduced here, and the ablation reports
    its contribution separately so the gain remains attributable.
    """
    c, n = env.cfg, env.N
    q = env._queue_pkts()
    occ = env.Q > 0
    hol = np.where(occ.any(axis=1), np.argmax(occ, axis=1), env.D).astype(float)
    urgency = (q + 0.5) * (env.D + 1.0 - hol)
    g = env._gain()
    score = urgency * np.sqrt(g / (g.mean() + 1e-30))
    if score.sum() <= 0:
        score = np.ones(n)
    w = score / score.sum()
    return w.copy(), w.copy(), FULL_POWER


def h_prop_fair(env):
    """Proportional-fair anchor: weight by instantaneous rate over the EMA of
    the delivered rate (Kelly / PF scheduling)."""
    n = env.N
    g = env._gain()
    inst = np.log2(1.0 + (env.P_max / n) * g / (env.N0 * env.W / n + env.I_inter))
    score = inst / (env.avg_rate + 1e-3)
    if score.sum() <= 0:
        score = np.ones(n)
    w = score / score.sum()
    return w.copy(), w.copy(), FULL_POWER


ANCHORS = {"WF": h_water_filling, "RR": h_round_robin, "GRD": h_greedy,
           "URG": h_urgency, "PF": h_prop_fair}
CLASSICAL = ("WF", "RR", "GRD", "PF")
ENSEMBLE = ("WF", "RR", "GRD", "URG", "PF")


def seeds_to_matrix(seeds):
    return np.array([inverse_project(b, p, r) for (b, p, r) in seeds])


def gwo_refine(env, seeds, pop_size=10, iters=10, rng=None):
    """Elitist GWO.

    Returns ((b, p, rho), fitness, n_fitness_evals).

    Elitism guarantee: the returned point maximises F over all evaluated
    points, the initial seeds included.  Therefore  F* >= max_k F(seed_k)
    holds by construction for the same F used throughout -- this is what
    Proposition 1 in the manuscript should state, and nothing stronger.
    """
    rng = np.random.default_rng() if rng is None else rng
    n = env.N
    X = seeds_to_matrix(seeds)

    if len(X) < pop_size:
        base = X[0]
        extra = np.abs(base[None, :] + rng.normal(0.0, 0.10, (pop_size - len(X), 2 * n + 1)))
        extra[:, -1] = base[-1] + rng.normal(0.0, 0.3, pop_size - len(X))
        X = np.vstack([X, extra])
    X = X[:pop_size]

    n_evals = 0

    def fitness(M):
        nonlocal n_evals
        B, P, RHO = project_batch(M, n)
        n_evals += M.shape[0]
        return env.evaluate_batch(B, P, RHO)[0]

    scores = fitness(X)
    order = np.argsort(-scores)
    best_vec, best_f = X[order[0]].copy(), float(scores[order[0]])
    Xa, Xb, Xd = X[order[0]].copy(), X[order[1]].copy(), X[order[2]].copy()

    for it in range(iters):
        a = 2.0 - 2.0 * it / max(iters, 1)
        newX = np.zeros_like(X)
        for lead in (Xa, Xb, Xd):
            r1 = rng.random(X.shape)
            r2 = rng.random(X.shape)
            A = 2.0 * a * r1 - a
            C = 2.0 * r2
            Dm = np.abs(C * lead[None, :] - X)
            newX += lead[None, :] - A * Dm
        X = newX / 3.0

        scores = fitness(X)
        order = np.argsort(-scores)
        if scores[order[0]] > best_f:
            best_f = float(scores[order[0]])
            best_vec = X[order[0]].copy()
        Xa, Xb, Xd = X[order[0]].copy(), X[order[1]].copy(), X[order[2]].copy()

    B, P, RHO = project_batch(best_vec[None, :], n)
    return (B[0], P[0], float(RHO[0])), best_f, n_evals
