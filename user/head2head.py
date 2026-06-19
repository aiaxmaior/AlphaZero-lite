"""Head-to-head between two checkpoints under IDENTICAL eval search.

Handles different architectures (reads net shape from each run's config.yaml), so
you can pit a 7-block net against a 6-block net fairly. Runs on CPU so it doesn't
fight GPU training. With --wait, it blocks until both checkpoints exist.

Example:
  python user/head2head.py --wait \
    --a runs/run_1/checkpoints/net_iter050.pt --a-cfg runs/run_1/config.yaml \
    --b runs/run_0/checkpoints/net_iter050.pt --b-cfg runs/run_0/config.yaml \
    --games 50
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import chess
import numpy as np
import torch

from src.config import Config, load_config
from src.network import ChessNet
from src.evaluate import net_move


def random_opening(rng, plies):
    """A random but legal, non-terminal opening position — to diversify games."""
    while True:
        board = chess.Board()
        ok = True
        for _ in range(plies):
            moves = list(board.legal_moves)
            if not moves or board.is_game_over():
                ok = False
                break
            board.push(moves[int(rng.integers(len(moves)))])
        if ok and not board.is_game_over():
            return board


def play_match_openings(net_a, net_b, cfg, n_games, opening_plies, rng):
    """Net A vs Net B, each game from a different random opening, colors alternating.
    Because eval is deterministic, the random opening is what makes the N games
    genuinely distinct (otherwise you replay 2 games N/2 times)."""
    w = d = l = 0
    for g in range(n_games):
        board = random_opening(rng, opening_plies)
        a_is_white = (g % 2 == 0)
        while not board.is_game_over(claim_draw=True):
            a_to_move = (board.turn == chess.WHITE) == a_is_white
            mv = net_move(net_a if a_to_move else net_b, board, cfg)
            board.push(mv)
        res = board.result(claim_draw=True)
        if res == "1/2-1/2":
            d += 1
        elif (res == "1-0") == a_is_white:
            w += 1
        else:
            l += 1
    return w, d, l


def build_net(ckpt, run_cfg_path, device):
    cfg = load_config(run_cfg_path)                    # read THIS net's architecture
    net = ChessNet(cfg.net).to(device)
    net.load_state_dict(torch.load(ckpt, map_location=device))
    net.eval()
    return net, f"{cfg.net.channels}x{cfg.net.blocks}"


def wait_for(path, timeout=7200):
    t0 = time.time()
    while not os.path.exists(path):
        if time.time() - t0 > timeout:
            return False
        time.sleep(20)
    return True


ap = argparse.ArgumentParser()
ap.add_argument("--a", required=True);     ap.add_argument("--a-cfg", required=True)
ap.add_argument("--b", required=True);     ap.add_argument("--b-cfg", required=True)
ap.add_argument("--games", type=int, default=50)
ap.add_argument("--sims",  type=int, default=50)
ap.add_argument("--cpuct", type=float, default=1.5)    # SHARED eval search for both
ap.add_argument("--openings", type=int, default=6)     # random plies per game (0 = none)
ap.add_argument("--wait", action="store_true")
args = ap.parse_args()

if args.wait:
    print(f"waiting for {args.a} and {args.b} ...", flush=True)
    if not (wait_for(args.a) and wait_for(args.b)):
        print("timed out waiting for checkpoints"); sys.exit(1)

ev = Config()
ev.device = "cpu"
torch.set_num_threads(6)
ev.mcts.simulations = args.sims
ev.mcts.c_puct = args.cpuct

net_a, shape_a = build_net(args.a, args.a_cfg, "cpu")
net_b, shape_b = build_net(args.b, args.b_cfg, "cpu")
print(f"A = {args.a} ({shape_a})\nB = {args.b} ({shape_b})", flush=True)
print(f"eval search: sims={args.sims}, c_puct={args.cpuct}, {args.games} games\n", flush=True)

rng = np.random.default_rng(0)
w, d, l = play_match_openings(net_a, net_b, ev, args.games, args.openings, rng)
score = (w + 0.5 * d) / (w + d + l)
# 95% confidence interval on the score (binomial-ish), so we don't over-read it
se = (score * (1 - score) / args.games) ** 0.5
print(f"\n>>> A vs B:  +{w} ={d} -{l}   score {score:.0%}  (±{2*se:.0%} 95% CI)", flush=True)
if 0 < score < 1:
    elo = -400 * np.log10(1 / score - 1)
    print(f">>> Elo(A − B) ≈ {elo:+.0f}", flush=True)
print(f"(games seeded from {args.openings} random opening plies — genuinely distinct)", flush=True)
