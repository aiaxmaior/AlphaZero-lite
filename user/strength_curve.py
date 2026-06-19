"""Strength curve — relative Elo of checkpoints across training.

WHY: loss plateaus and hides strength gains. To see *strength* over training you
have to make the nets play games. This produces an Elo-vs-iteration curve you can
run the same band/knee analysis on.

DESIGN — the cheap "ladder" (a.k.a. relative Elo, how AlphaZero reported it):
  Instead of a full round-robin (every pair plays — O(N^2) matches, expensive),
  play each sampled checkpoint against the PREVIOUS sampled one (O(N) matches).
  - score of (later vs earlier) -> a *local* Elo gain via the logistic formula
  - cumulative-sum the gains -> an Elo curve anchored at iter 1 = 0
  Adjacent nets are similar strength, so scores sit near 50% — maximally
  informative (a fixed weak/strong anchor would saturate to 100%/0% and tell you
  nothing). Trade-off: gains accumulate noise, so the absolute Elo drifts; the
  *shape* (where it rises vs flattens) is what you trust.
"""
import glob
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config import Config
from src.network import ChessNet
from src.evaluate import net_move, play_match

cfg = Config()
cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
cfg.mcts.simulations = 40          # eval search budget
GAMES = 8                          # games per adjacent matchup
STEP  = 50                         # sample a checkpoint every STEP iterations
CK    = "runs/run_0/checkpoints"


def load(it):
    n = ChessNet(cfg.net).to(cfg.device)
    n.load_state_dict(torch.load(f"{CK}/net_iter{it:03d}.pt", map_location=cfg.device))
    n.eval()
    return n


def elo_gain(score):
    """Elo difference implied by an expected score (draws count 0.5)."""
    score = min(max(score, 0.02), 0.98)         # clamp so a sweep doesn't -> inf
    return -400.0 * np.log10(1.0 / score - 1.0)


# which iterations to sample: 1, STEP, 2*STEP, ...
avail = sorted(int(re.search(r"(\d+)", os.path.basename(p)).group(1))
               for p in glob.glob(f"{CK}/net_iter*.pt"))
iters = [1] + [i for i in avail if i % STEP == 0]
print("sampling iters:", iters, flush=True)

elo = [0.0]                                      # anchor: iter 1 = 0 Elo
rng = np.random.default_rng(0)
prev = load(iters[0])
for k in range(1, len(iters)):
    cur = load(iters[k])
    w, d, l = play_match(cur, cfg, lambda b: net_move(prev, b, cfg), GAMES, rng)
    score = (w + 0.5 * d) / (w + d + l)
    elo.append(elo[-1] + elo_gain(score))
    print(f"iter {iters[k]:>3} vs {iters[k-1]:>3}:  score {score:.2f}  "
          f"local +{elo_gain(score):6.1f}  ->  cumulative {elo[-1]:7.1f} Elo", flush=True)
    prev = cur

plt.figure(figsize=(9, 5))
plt.plot(iters, elo, marker="o")
plt.xlabel("training iteration"); plt.ylabel("relative Elo  (iter 1 = 0)")
plt.title("run_0 — strength over training (ladder Elo)"); plt.grid(alpha=.3)
plt.tight_layout(); plt.savefig("runs/run_0/strength_curve.png", dpi=120)
print("saved runs/run_0/strength_curve.png", flush=True)
