"""Measure strength by playing the current net against Stockfish.

Stockfish is used ONLY as a benchmark opponent here — never as a teacher. We set
it to a low skill level and count wins/draws/losses. This is how you'll know the
self-play loop is actually working: the win-rate against a fixed level should
climb over training iterations.

    python -m src.evaluate --vs stockfish --level 1 --games 20 --sims 200
    python -m src.evaluate --vs stockfish --engine /usr/games/stockfish

If you don't have Stockfish, `--vs random` plays against a random-move opponent,
which a half-trained net should already beat ~100% of the time.
"""
from __future__ import annotations

import argparse
import shutil

import numpy as np
import torch
import chess

from .config import Config
from .network import ChessNet
from .mcts import MCTS, select_move
from .encoding import index_to_move


def net_move(net, board, cfg):
    """Pick the net's move: run MCTS, play greedily (temperature 0)."""
    pi = MCTS(net, cfg.mcts, cfg.device).run(board, add_noise=False)
    return index_to_move(select_move(pi, temperature=0.0), board)


def random_move(board, rng):
    return rng.choice(list(board.legal_moves))


def play_match(net, cfg, opponent, n_games, rng):
    """Play n_games, alternating colours. Returns (wins, draws, losses) for net."""
    wins = draws = losses = 0
    for g in range(n_games):
        board = chess.Board()
        net_is_white = (g % 2 == 0)
        while not board.is_game_over(claim_draw=True):
            net_to_move = (board.turn == chess.WHITE) == net_is_white
            move = net_move(net, board, cfg) if net_to_move else opponent(board)
            board.push(move)

        result = board.result(claim_draw=True)     # "1-0", "0-1", "1/2-1/2"
        if result == "1/2-1/2":
            draws += 1
        elif (result == "1-0") == net_is_white:
            wins += 1
        else:
            losses += 1
        print(f"  game {g+1}/{n_games}: {result} "
              f"(net played {'White' if net_is_white else 'Black'})")
    return wins, draws, losses


def main():
    ap = argparse.ArgumentParser(description="evaluate the net vs an opponent")
    ap.add_argument("--checkpoint", default="checkpoints/latest.pt")
    ap.add_argument("--vs", choices=["stockfish", "random"], default="stockfish")
    ap.add_argument("--level", type=int, default=1, help="Stockfish skill level 0-20")
    ap.add_argument("--engine", default=None, help="path to stockfish binary")
    ap.add_argument("--games", type=int, default=10)
    ap.add_argument("--sims", type=int, default=None)
    args = ap.parse_args()

    cfg = Config()
    if args.sims is not None:
        cfg.mcts.simulations = args.sims
    if cfg.device == "cuda" and not torch.cuda.is_available():
        cfg.device = "cpu"

    net = ChessNet(cfg.net).to(cfg.device)
    net.load_state_dict(torch.load(args.checkpoint, map_location=cfg.device))
    net.eval()

    rng = np.random.default_rng(0)

    if args.vs == "random":
        opponent = lambda b: random_move(b, rng)
        w, d, l = play_match(net, cfg, opponent, args.games, rng)
    else:
        import chess.engine
        path = args.engine or shutil.which("stockfish")
        if path is None:
            raise SystemExit("Stockfish not found. Install it or pass --engine PATH, "
                             "or use --vs random.")
        engine = chess.engine.SimpleEngine.popen_uci(path)
        engine.configure({"Skill Level": args.level})
        limit = chess.engine.Limit(time=0.05)   # weak + fast; raise to strengthen

        def opponent(board):
            return engine.play(board, limit).move

        try:
            w, d, l = play_match(net, cfg, opponent, args.games, rng)
        finally:
            engine.quit()

    total = w + d + l
    score = (w + 0.5 * d) / total if total else 0.0
    print(f"\nResult vs {args.vs}: +{w} ={d} -{l}   score={score:.1%}")


if __name__ == "__main__":
    main()
