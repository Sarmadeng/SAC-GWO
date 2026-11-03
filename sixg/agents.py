"""
Reference implementations of SAC, TD3 and PPO for continuous control.

All three share the same MLP capacity, the same optimiser family, the same
batch size and the same number of environment interactions, so that the
comparison is not confounded by network size or training budget.

SAC  : Haarnoja et al., ICML 2018 (twin critics, tanh-Gaussian policy,
       automatic temperature tuning with target entropy -dim(A)).
TD3  : Fujimoto et al., ICML 2018 (twin critics, delayed policy updates,
       target policy smoothing).
PPO  : Schulman et al., 2017 (clipped surrogate objective, GAE-lambda).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEV = torch.device("cpu")
LOG_STD_MIN, LOG_STD_MAX = -5.0, 2.0

HP = dict(
    hidden        = 128,
    lr_actor      = 3e-4,
    lr_critic     = 3e-4,
    lr_alpha      = 3e-4,
    gamma         = 0.95,
    tau           = 0.005,
    batch_size    = 128,
    buffer_size   = 100_000,
    start_steps   = 500,      # uniform-random warm-up before learning
    updates_per_step = 1,
    # TD3
    policy_delay  = 2,
    expl_noise    = 0.1,
    target_noise  = 0.2,
    noise_clip    = 0.5,
    # PPO
    ppo_epochs    = 10,
    ppo_clip      = 0.2,
    ppo_lam       = 0.95,
    ppo_rollout   = 400,
    ppo_minibatch = 128,
    # imitation (proposed method only)
    bc_coef_init  = 1.0,
    bc_decay      = 0.995,
)


def mlp(inp, out, hidden, out_act=None):
    layers = [nn.Linear(inp, hidden), nn.ReLU(),
              nn.Linear(hidden, hidden), nn.ReLU(),
              nn.Linear(hidden, out)]
    if out_act is not None:
        layers.append(out_act)
    return nn.Sequential(*layers)


class ReplayBuffer:
    def __init__(self, sdim, adim, size):
        self.s = np.zeros((size, sdim), dtype=np.float32)
        self.a = np.zeros((size, adim), dtype=np.float32)
        self.r = np.zeros((size, 1), dtype=np.float32)
        self.s2 = np.zeros((size, sdim), dtype=np.float32)
        self.d = np.zeros((size, 1), dtype=np.float32)
        self.ptr, self.n, self.size = 0, 0, size

    def add(self, s, a, r, s2, d):
        i = self.ptr
        self.s[i], self.a[i], self.r[i], self.s2[i], self.d[i] = s, a, r, s2, d
        self.ptr = (self.ptr + 1) % self.size
        self.n = min(self.n + 1, self.size)

    def sample(self, bs):
        idx = np.random.randint(0, self.n, bs)
        t = lambda x: torch.as_tensor(x[idx], device=DEV)
        return t(self.s), t(self.a), t(self.r), t(self.s2), t(self.d)


# ------------------------------------------------------------------------ SAC
class SACActor(nn.Module):
    def __init__(self, sdim, adim, hidden):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(sdim, hidden), nn.ReLU(),
                                  nn.Linear(hidden, hidden), nn.ReLU())
        self.mu = nn.Linear(hidden, adim)
        self.log_std = nn.Linear(hidden, adim)

    def forward(self, s, deterministic=False, with_logp=True):
        h = self.body(s)
        mu = self.mu(h)
        log_std = torch.clamp(self.log_std(h), LOG_STD_MIN, LOG_STD_MAX)
        std = log_std.exp()
        if deterministic:
            u = mu
        else:
            u = mu + std * torch.randn_like(mu)
        a = torch.tanh(u)
        logp = None
        if with_logp and not deterministic:
            logp = (-0.5 * ((u - mu) / std) ** 2 - log_std
                    - 0.5 * np.log(2 * np.pi)).sum(-1, keepdim=True)
            logp = logp - (2 * (np.log(2) - u - F.softplus(-2 * u))).sum(-1, keepdim=True)
        return a, logp, torch.tanh(mu)


class SAC:
    name = "SAC"

    def __init__(self, sdim, adim, hp=None, seed=0):
        self.hp = dict(HP) if hp is None else dict(hp)
        torch.manual_seed(seed)
        h = self.hp["hidden"]
        self.actor = SACActor(sdim, adim, h).to(DEV)
        self.q1 = mlp(sdim + adim, 1, h).to(DEV)
        self.q2 = mlp(sdim + adim, 1, h).to(DEV)
        self.q1t = mlp(sdim + adim, 1, h).to(DEV)
        self.q2t = mlp(sdim + adim, 1, h).to(DEV)
        self.q1t.load_state_dict(self.q1.state_dict())
        self.q2t.load_state_dict(self.q2.state_dict())

        self.opt_a = torch.optim.Adam(self.actor.parameters(), lr=self.hp["lr_actor"])
        self.opt_q = torch.optim.Adam(list(self.q1.parameters()) + list(self.q2.parameters()),
                                      lr=self.hp["lr_critic"])
        self.log_alpha = torch.zeros(1, requires_grad=True, device=DEV)
        self.opt_alpha = torch.optim.Adam([self.log_alpha], lr=self.hp["lr_alpha"])
        self.target_entropy = -float(adim)

        self.buf = ReplayBuffer(sdim, adim, self.hp["buffer_size"])
        self.adim = adim
        self.steps = 0
        self.bc_coef = self.hp["bc_coef_init"]

    @torch.no_grad()
    def act(self, s, deterministic=False):
        st = torch.as_tensor(s, dtype=torch.float32, device=DEV).unsqueeze(0)
        a, _, mu = self.actor(st, deterministic=deterministic, with_logp=False)
        return (mu if deterministic else a).cpu().numpy()[0]

    def store(self, *tr):
        self.buf.add(*tr)

    def update(self, bc_target=None):
        hp = self.hp
        if self.buf.n < max(hp["batch_size"], hp["start_steps"]):
            return
        s, a, r, s2, d = self.buf.sample(hp["batch_size"])
        alpha = self.log_alpha.exp().detach()

        with torch.no_grad():
            a2, logp2, _ = self.actor(s2)
            qt = torch.min(self.q1t(torch.cat([s2, a2], -1)),
                           self.q2t(torch.cat([s2, a2], -1)))
            target = r + hp["gamma"] * (1 - d) * (qt - alpha * logp2)

        sa = torch.cat([s, a], -1)
        loss_q = F.mse_loss(self.q1(sa), target) + F.mse_loss(self.q2(sa), target)
        self.opt_q.zero_grad(); loss_q.backward(); self.opt_q.step()

        anew, logp, mu = self.actor(s)
        q = torch.min(self.q1(torch.cat([s, anew], -1)),
                      self.q2(torch.cat([s, anew], -1)))
        loss_a = (alpha * logp - q).mean()

        # optional behaviour-cloning term toward the GWO-refined expert action
        if bc_target is not None and self.bc_coef > 1e-3:
            bt = torch.as_tensor(bc_target, dtype=torch.float32, device=DEV).unsqueeze(0)
            st = s[:1]
            _, _, mu_s = self.actor(st, with_logp=False)
            loss_a = loss_a + self.bc_coef * F.mse_loss(mu_s, bt)

        self.opt_a.zero_grad(); loss_a.backward(); self.opt_a.step()

        loss_alpha = -(self.log_alpha.exp() * (logp.detach() + self.target_entropy)).mean()
        self.opt_alpha.zero_grad(); loss_alpha.backward(); self.opt_alpha.step()

        with torch.no_grad():
            for net, tgt in ((self.q1, self.q1t), (self.q2, self.q2t)):
                for p, pt in zip(net.parameters(), tgt.parameters()):
                    pt.mul_(1 - hp["tau"]).add_(hp["tau"] * p)
        self.bc_coef *= hp["bc_decay"]


# ------------------------------------------------------------------------ TD3
class TD3:
    name = "TD3"

    def __init__(self, sdim, adim, hp=None, seed=0):
        self.hp = dict(HP) if hp is None else dict(hp)
        torch.manual_seed(seed)
        h = self.hp["hidden"]
        self.actor = mlp(sdim, adim, h, nn.Tanh()).to(DEV)
        self.actor_t = mlp(sdim, adim, h, nn.Tanh()).to(DEV)
        self.actor_t.load_state_dict(self.actor.state_dict())
        self.q1 = mlp(sdim + adim, 1, h).to(DEV)
        self.q2 = mlp(sdim + adim, 1, h).to(DEV)
        self.q1t = mlp(sdim + adim, 1, h).to(DEV)
        self.q2t = mlp(sdim + adim, 1, h).to(DEV)
        self.q1t.load_state_dict(self.q1.state_dict())
        self.q2t.load_state_dict(self.q2.state_dict())
        self.opt_a = torch.optim.Adam(self.actor.parameters(), lr=self.hp["lr_actor"])
        self.opt_q = torch.optim.Adam(list(self.q1.parameters()) + list(self.q2.parameters()),
                                      lr=self.hp["lr_critic"])
        self.buf = ReplayBuffer(sdim, adim, self.hp["buffer_size"])
        self.adim = adim
        self.it = 0

    @torch.no_grad()
    def act(self, s, deterministic=False):
        st = torch.as_tensor(s, dtype=torch.float32, device=DEV).unsqueeze(0)
        a = self.actor(st).cpu().numpy()[0]
        if not deterministic:
            a = np.clip(a + np.random.normal(0, self.hp["expl_noise"], self.adim), -1, 1)
        return a

    def store(self, *tr):
        self.buf.add(*tr)

    def update(self, bc_target=None):
        hp = self.hp
        if self.buf.n < max(hp["batch_size"], hp["start_steps"]):
            return
        self.it += 1
        s, a, r, s2, d = self.buf.sample(hp["batch_size"])
        with torch.no_grad():
            noise = (torch.randn_like(a) * hp["target_noise"]
                     ).clamp(-hp["noise_clip"], hp["noise_clip"])
            a2 = (self.actor_t(s2) + noise).clamp(-1, 1)
            qt = torch.min(self.q1t(torch.cat([s2, a2], -1)),
                           self.q2t(torch.cat([s2, a2], -1)))
            target = r + hp["gamma"] * (1 - d) * qt
        sa = torch.cat([s, a], -1)
        loss_q = F.mse_loss(self.q1(sa), target) + F.mse_loss(self.q2(sa), target)
        self.opt_q.zero_grad(); loss_q.backward(); self.opt_q.step()

        if self.it % hp["policy_delay"] == 0:
            loss_a = -self.q1(torch.cat([s, self.actor(s)], -1)).mean()
            self.opt_a.zero_grad(); loss_a.backward(); self.opt_a.step()
            with torch.no_grad():
                for net, tgt in ((self.q1, self.q1t), (self.q2, self.q2t),
                                 (self.actor, self.actor_t)):
                    for p, pt in zip(net.parameters(), tgt.parameters()):
                        pt.mul_(1 - hp["tau"]).add_(hp["tau"] * p)


# ------------------------------------------------------------------------ PPO
class PPO:
    name = "PPO"

    def __init__(self, sdim, adim, hp=None, seed=0):
        self.hp = dict(HP) if hp is None else dict(hp)
        torch.manual_seed(seed)
        h = self.hp["hidden"]
        self.pi = mlp(sdim, adim, h, nn.Tanh()).to(DEV)
        self.log_std = nn.Parameter(torch.full((adim,), -0.5, device=DEV))
        self.v = mlp(sdim, 1, h).to(DEV)
        self.opt = torch.optim.Adam(
            list(self.pi.parameters()) + [self.log_std] + list(self.v.parameters()),
            lr=self.hp["lr_actor"])
        self.reset_rollout()
        self.adim = adim

    def reset_rollout(self):
        self.S, self.A, self.R, self.D, self.LP, self.V = [], [], [], [], [], []

    @torch.no_grad()
    def act(self, s, deterministic=False):
        st = torch.as_tensor(s, dtype=torch.float32, device=DEV).unsqueeze(0)
        mu = self.pi(st)
        if deterministic:
            self._last = (st, mu, torch.zeros(1, 1))
            return mu.cpu().numpy()[0]
        std = self.log_std.exp()
        a = mu + std * torch.randn_like(mu)
        logp = (-0.5 * ((a - mu) / std) ** 2 - self.log_std
                - 0.5 * np.log(2 * np.pi)).sum(-1, keepdim=True)
        self._last = (st, a, logp)
        return a.clamp(-1, 1).cpu().numpy()[0]

    def store(self, s, a, r, s2, d):
        st, at, lp = self._last
        self.S.append(st); self.A.append(at); self.LP.append(lp)
        self.R.append(r); self.D.append(d)
        with torch.no_grad():
            self.V.append(self.v(st))

    def update(self, bc_target=None):
        hp = self.hp
        if len(self.R) < hp["ppo_rollout"]:
            return
        S = torch.cat(self.S); A = torch.cat(self.A)
        LP = torch.cat(self.LP).detach(); V = torch.cat(self.V).squeeze(-1)
        R = torch.as_tensor(self.R, dtype=torch.float32, device=DEV)
        D = torch.as_tensor(self.D, dtype=torch.float32, device=DEV)

        adv = torch.zeros_like(R); last = 0.0
        for t in reversed(range(len(R))):
            nextv = V[t + 1] if t + 1 < len(R) else 0.0
            delta = R[t] + hp["gamma"] * nextv * (1 - D[t]) - V[t]
            last = delta + hp["gamma"] * hp["ppo_lam"] * (1 - D[t]) * last
            adv[t] = last
        ret = adv + V
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        idx = np.arange(len(R))
        for _ in range(hp["ppo_epochs"]):
            np.random.shuffle(idx)
            for k in range(0, len(idx), hp["ppo_minibatch"]):
                j = idx[k:k + hp["ppo_minibatch"]]
                mu = self.pi(S[j]); std = self.log_std.exp()
                logp = (-0.5 * ((A[j] - mu) / std) ** 2 - self.log_std
                        - 0.5 * np.log(2 * np.pi)).sum(-1, keepdim=True)
                ratio = (logp - LP[j]).exp().squeeze(-1)
                s1 = ratio * adv[j]
                s2 = ratio.clamp(1 - hp["ppo_clip"], 1 + hp["ppo_clip"]) * adv[j]
                loss = -torch.min(s1, s2).mean() \
                       + 0.5 * F.mse_loss(self.v(S[j]).squeeze(-1), ret[j])
                self.opt.zero_grad(); loss.backward(); self.opt.step()
        self.reset_rollout()
