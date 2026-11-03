"""
6G edge downlink environment with deadline-constrained queueing.

Design notes that matter for the review response
------------------------------------------------
1.  TRUE MDP, not a bandit.  Each user holds a FIFO packet queue with per-packet
    deadlines; the allocation at slot t drains queues and therefore determines
    the state at t+1.

2.  PACKET-LEVEL RELIABILITY.  PDR is the fraction of ARRIVED packets that
    expire before being fully served (plus buffer overflow).  It is a
    deadline-violation rate, which is what the paper should claim -- not
    "3GPP URLLC 1e-5", which this system does not and cannot demonstrate.

3.  GOODPUT, NOT SHANNON RATE.  A user with an empty queue cannot use the
    bandwidth assigned to it.  The delivered rate is min(R_i, backlog_i/TTI),
    so over-provisioning a strong-but-idle user is correctly penalised.  This
    is why a pure Greedy/SINR-maximising allocation does badly here.

4.  ACTUAL POWER CONTROL.  The action carries a total-power scaling factor
    rho in [RHO_MIN, 1], so the agent can spend less than P_max.  Without it,
    "energy efficiency" is just a rescaled throughput and any Green claim is
    vacuous.

Queues are stored as an N x (D+1) count matrix Q[i, k] = number of packets of
user i with k slots of life remaining, which makes every operation vectorised.
"""

import numpy as np

CFG = dict(
    # --- radio ---
    W_TOTAL_HZ      = 200e6,
    P_MAX_DBM       = 46.0,
    NOISE_PSD_DBM   = -174.0,
    NF_DB           = 7.0,
    P_CIRC_W        = 10.0,
    XI_PA           = 1.2,
    D_MIN_M         = 50.0,
    D_MAX_M         = 500.0,
    SHADOW_STD_DB   = 8.0,
    RHO_FADING      = 0.90,
    INTERF_DBM      = -95.0,
    RHO_MIN         = 0.05,

    # --- traffic / queueing ---
    TTI_S           = 1e-3,
    PKT_BITS        = 8000.0,
    DEADLINE_SLOTS  = 5,
    LAMBDA_MIN_MBPS = 5.0,
    LAMBDA_MAX_MBPS = 25.0,
    QUEUE_CAP_PKTS  = 200,

    # --- episode ---
    T_SLOTS         = 40,

    # --- objective weights (w1..w4 of Eq. 8) ---
    W_THR           = 0.35,
    W_FAIR          = 0.15,
    W_EE            = 0.20,
    W_PDR           = 0.60,

    MIN_FRAC        = 1e-4,
)


def dbm_to_w(x):
    return 10.0 ** ((x - 30.0) / 10.0)


class SixGEdgeEnv:
    def __init__(self, n_users=30, cfg=None, seed=0):
        self.cfg = dict(CFG) if cfg is None else dict(cfg)
        self.N = int(n_users)
        self.rng = np.random.default_rng(seed)
        c = self.cfg
        self.W = c["W_TOTAL_HZ"]
        self.P_max = dbm_to_w(c["P_MAX_DBM"])
        self.N0 = dbm_to_w(c["NOISE_PSD_DBM"] + c["NF_DB"])
        self.I_inter = dbm_to_w(c["INTERF_DBM"])
        self.D = int(c["DEADLINE_SLOTS"])
        self.T = c["T_SLOTS"]
        self.state_dim = 4 * self.N
        self.action_dim = 2 * self.N + 1

    # ---------------------------------------------------------------- reset
    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        c, N = self.cfg, self.N

        self.dist_m = self.rng.uniform(c["D_MIN_M"], c["D_MAX_M"], N)
        pl_db = 128.1 + 37.6 * np.log10(self.dist_m / 1000.0)
        shadow_db = self.rng.normal(0.0, c["SHADOW_STD_DB"], N)
        self.large_scale = 10.0 ** ((-pl_db + shadow_db) / 10.0)
        self.h = (self.rng.normal(0, np.sqrt(0.5), N)
                  + 1j * self.rng.normal(0, np.sqrt(0.5), N))

        self.lam_mbps = self.rng.uniform(c["LAMBDA_MIN_MBPS"], c["LAMBDA_MAX_MBPS"], N)
        self.lam_pkts = self.lam_mbps * 1e6 * c["TTI_S"] / c["PKT_BITS"]

        self.R_ref = float(self.lam_mbps.sum())
        self.EE_ref = self.R_ref / (c["P_CIRC_W"] + c["XI_PA"] * self.P_max)

        self.Q = np.zeros((N, self.D + 1))
        self.residual = np.zeros(N)
        self.avg_rate = np.full(N, 1.0)         # EMA of delivered rate (PF anchor)

        self.t = 0
        self.tot_arrived = 0
        self.tot_dropped = 0
        self.tot_delivered_bits = 0.0
        return self._observe()

    # ------------------------------------------------------------ observation
    def _gain(self):
        return self.large_scale * (np.abs(self.h) ** 2)

    def _queue_pkts(self):
        return self.Q.sum(axis=1)

    def _observe(self):
        c, N = self.cfg, self.N
        g = self._gain()
        snr_ref = (self.P_max / N) * g / (self.N0 * self.W / N + self.I_inter)
        s_snr = np.log10(snr_ref + 1e-12) / 3.0
        s_q = np.minimum(self._queue_pkts() / 50.0, 4.0)
        occupied = self.Q > 0
        hol = np.where(occupied.any(axis=1),
                       np.argmax(occupied, axis=1), self.D).astype(float)
        s_hol = hol / self.D
        s_lam = self.lam_mbps / c["LAMBDA_MAX_MBPS"]
        return np.concatenate([s_snr, s_q, s_hol, s_lam]).astype(np.float64)

    # --------------------------------------------------------------- physics
    def _rates_core(self, B, P, RHO):
        c = self.cfg
        B = np.maximum(B, c["MIN_FRAC"])
        P = np.maximum(P, c["MIN_FRAC"])
        g = self._gain()[None, :]
        bw = B * self.W
        noise = self.N0 * bw + self.I_inter
        p_w = P * (RHO[:, None] * self.P_max)
        sinr = p_w * g / noise
        return bw * np.log2(1.0 + sinr) / 1e6

    def evaluate_batch(self, B, P, RHO):
        c = self.cfg
        R = self._rates_core(B, P, RHO)
        backlog_bits = np.maximum(self._queue_pkts() - self.residual, 0.0) * c["PKT_BITS"]
        cap_mbps = backlog_bits / c["TTI_S"] / 1e6
        U = np.minimum(R, cap_mbps[None, :])
        thr = U.sum(axis=1)
        # Demand-aware fairness: Jain's index over the per-user satisfaction
        # ratio (delivered rate / offered load).  Plain Jain over raw rates
        # would penalise a scheduler for correctly giving less to users who
        # asked for less.
        sat = U / (self.lam_mbps[None, :] + 1e-9)
        jain = (sat.sum(axis=1) ** 2) / (self.N * np.sum(sat ** 2, axis=1) + 1e-12)
        p_tot = c["P_CIRC_W"] + c["XI_PA"] * RHO * self.P_max
        ee = thr / p_tot

        urgent_bits = self.Q[:, 1] * c["PKT_BITS"]
        deliver = R * 1e6 * c["TTI_S"]
        miss = np.maximum(urgent_bits[None, :] - deliver, 0.0).sum(axis=1)
        pdr_proxy = miss / (urgent_bits.sum() + 1e-9)

        f = (c["W_THR"] * thr / self.R_ref
             + c["W_FAIR"] * jain
             + c["W_EE"] * ee / self.EE_ref
             - c["W_PDR"] * pdr_proxy)
        return f, dict(thr=thr, jain=jain, ee=ee, pdr_proxy=pdr_proxy, R=R, U=U)

    def evaluate(self, b, p, rho=1.0):
        f, d = self.evaluate_batch(np.atleast_2d(b), np.atleast_2d(p),
                                   np.atleast_1d(rho))
        return float(f[0]), {k: v[0] for k, v in d.items()}

    # -------------------------------------------------------------- dynamics
    def step(self, b, p, rho=1.0):
        c, N = self.cfg, self.N
        f, info = self.evaluate(b, p, rho)
        R = np.atleast_1d(info["R"])

        avail_pkts = R * 1e6 * c["TTI_S"] / c["PKT_BITS"] + self.residual
        q_tot = self._queue_pkts()
        n_full = np.floor(avail_pkts)
        served = np.minimum(n_full, q_tot)
        self.residual = np.where(n_full <= q_tot, avail_pkts - n_full, 0.0)

        cum = np.cumsum(self.Q, axis=1)
        remove = np.clip(served[:, None] - (cum - self.Q), 0.0, self.Q)
        self.Q = self.Q - remove
        self.tot_delivered_bits += float(served.sum() * c["PKT_BITS"])
        served_mbps = served * c["PKT_BITS"] / c["TTI_S"] / 1e6
        self.avg_rate = 0.9 * self.avg_rate + 0.1 * served_mbps

        expired = float(self.Q[:, 1].sum())
        self.Q[:, 1] = 0.0
        self.Q[:, :-1] = self.Q[:, 1:]
        self.Q[:, -1] = 0.0
        self.tot_dropped += int(round(expired))

        arrivals = self.rng.poisson(self.lam_pkts).astype(float)
        free = np.maximum(c["QUEUE_CAP_PKTS"] - self._queue_pkts(), 0.0)
        admit = np.minimum(arrivals, free)
        self.tot_dropped += int(round(float((arrivals - admit).sum())))
        self.Q[:, self.D] += admit
        self.tot_arrived += int(arrivals.sum())

        rho_f = c["RHO_FADING"]
        innov = (self.rng.normal(0, np.sqrt(0.5), N)
                 + 1j * self.rng.normal(0, np.sqrt(0.5), N))
        self.h = rho_f * self.h + np.sqrt(1.0 - rho_f ** 2) * innov

        self.t += 1
        return self._observe(), float(f), self.t >= self.T, info

    def episode_stats(self):
        c = self.cfg
        return dict(
            pdr=100.0 * self.tot_dropped / max(self.tot_arrived, 1),
            goodput=self.tot_delivered_bits / (self.t * c["TTI_S"]) / 1e6,
            offered=float(self.lam_mbps.sum()),
        )


# ----------------------------------------------------------------------------
def project_action(x, n):
    x = np.asarray(x, dtype=float)
    xb = np.abs(x[:n]) + 1e-9
    xp = np.abs(x[n:2 * n]) + 1e-9
    b = xb / xb.sum()
    p = xp / xp.sum()
    r = 0.5 * (np.tanh(x[2 * n]) + 1.0) if x.shape[0] > 2 * n else 1.0
    rho = CFG["RHO_MIN"] + (1.0 - CFG["RHO_MIN"]) * r
    return b, p, float(rho)


def project_batch(X, n):
    X = np.atleast_2d(X)
    B = np.abs(X[:, :n]) + 1e-9
    P = np.abs(X[:, n:2 * n]) + 1e-9
    B = B / B.sum(axis=1, keepdims=True)
    P = P / P.sum(axis=1, keepdims=True)
    r = 0.5 * (np.tanh(X[:, 2 * n]) + 1.0)
    RHO = CFG["RHO_MIN"] + (1.0 - CFG["RHO_MIN"]) * r
    return B, P, RHO


def inverse_project(b, p, rho):
    r = (rho - CFG["RHO_MIN"]) / (1.0 - CFG["RHO_MIN"])
    r = np.clip(2.0 * r - 1.0, -0.999, 0.999)
    return np.concatenate([b / (b.max() + 1e-12),
                           p / (p.max() + 1e-12),
                           [np.arctanh(r)]])
