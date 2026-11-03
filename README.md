# Real DRL + GWO simulation for the Heliyon revision

Reference implementation of the Ensemble-Seeded Hybrid SAC-GWO framework for
reliability-aware resource allocation in 6G edge networks (Heliyon manuscript
HELIYON-D-26-00851). Every number produced by this code is measured from an
executed packet-level, deadline-constrained MDP simulation; no closed-form or
calibrated formula is used to generate results.

## Installation

```bash
python3 -m venv venv
source venv/bin/activate        # on Windows: venv\Scripts\activate
pip install -r requirements.txt
```

No GPU is required. The networks are small (128 hidden units) and a CPU is sufficient.

## Running the experiments

```bash
python run_experiment.py
```

The driver is resumable: it reads the existing `results/results.csv` (64 measured
results from an earlier run: RR/WF/GRD/PF, PPO/TD3/SAC/SAC+EHS/SAC+GWO with all
10 seeds, and PROPOSED with 2 seeds) and skips everything already completed. If a
run is interrupted for any reason, run the same command again and it continues
where it stopped without repeating finished work.

**Expected time for the remainder:** on a single CPU core, roughly 2-4 hours
(main PROPOSED-NG / PROPOSED / PROPOSED+ runs, scalability at N = 50/70/90,
sensitivity analysis, and the cold-start experiment). It can be run in the background:

```bash
nohup python run_experiment.py > run.log 2>&1 &
tail -f run.log        # follow progress
```

## After the run completes

```bash
python analyze.py      # prints the results table, significance tests and cold-start analysis
python make_figs.py    # generates all figures at 300 dpi in results/figs/
```

`analyze.py` also writes:
- `results/summary_main.csv` - mean and standard deviation for each method
- `results/significance.csv` - Welch's t-test, Cliff's delta and Holm correction, against PROPOSED+
- `results/coldstart_summary.csv` - reliability in the first training episodes only

## Code layout (`sixg/`)

| File | Contents |
|---|---|
| `env.py` | 6G environment: packet queues with deadlines, temporally correlated UMi channel, goodput, explicit power control |
| `heuristics.py` | Water-Filling, Round-Robin, Greedy, Proportional-Fair and an elitist Grey Wolf Optimizer (Mirjalili et al., 2014) |
| `agents.py` | SAC (twin critics, automatic entropy tuning), TD3 (delayed updates) and PPO (GAE-lambda), with identical network capacity and training budget |
| `runner.py` | Training and evaluation loop for every method, including the proposed variants |

## Methods compared

- **Heuristics:** RR, WF, GRD, PF (no learning)
- **DRL baselines:** SAC, TD3, PPO (fully implemented, no calibrated formulas)
- **Ablations:** SAC+EHS (seeding only, no GWO), SAC+GWO (GWO from random seeds, no EHS)
- **Proposed:** PROPOSED (EHS + GWO + adaptive gate), PROPOSED-NG (no gate),
  **PROPOSED+** (adds temporal warm-start: reusing the best solution of the previous
  slot, motivated by the temporal channel correlation rho = 0.9)
- **PROPOSED+NOEHS** (second revision, Reviewer #3): a matched control for PROPOSED+
  with the same gate, temporal warm-start, SAC agent and environment. The only
  difference is that the GWO seeds are random instead of the EHS heuristic anchors.
  Run it with `python run_r2_coldstart_matched.py` (10 seeds x 25 episodes), then
  `python make_fig_r3_matched.py`. Outputs are written to `results/coldstart_matched*`.
  (Welch p = 0.0002, Cliff's delta = -0.86 on PDR over episodes 1-5.)

## Reporting policy

If an analysis shows that the proposed method does not win on a given metric (as
happened with Jain's fairness, which Round-Robin wins), the code is not modified to
change the outcome. The result is reported as measured, as an explicit trade-off,
in the paper.
