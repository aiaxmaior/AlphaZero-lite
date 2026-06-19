"""Play against the trained net from the terminal (you are White by default).

    python -m src.play
    python -m src.play --checkpoint checkpoints/net_iter010.pt --as black

Enter moves in UCI (e.g. e2e4) or SAN (e.g. Nf3). Type 'quit' to leave.
"""
from __future__ import annotations

import argparse

import torch
import chess

from .config import Config
from .network import ChessNet
from .evaluate import net_move


def read_human_move(board: chess.Board) -> chess.Move:
    while True:
        text = input("your move > ").strip()
        if text in ("quit", "exit"):
            raise SystemExit
        for parser in (board.parse_san, board.parse_uci):
            try:
                mv = parser(text)
                if mv in board.legal_moves:
                    return mv
            except ValueError:
                continue
        print("  illegal/unparseable move; try UCI like e2e4 or SAN like Nf3")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/latest.pt")
    ap.add_argument("--as", dest="human", choices=["white", "black"], default="white")
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

    board = chess.Board()
    human_white = args.human == "white"

    while not board.is_game_over(claim_draw=True):
        print("\n" + str(board))
        human_turn = (board.turn == chess.WHITE) == human_white
        if human_turn:
            board.push(read_human_move(board))
        else:
            mv = net_move(net, board, cfg)
            print(f"net plays: {board.san(mv)}")
            board.push(mv)

    print("\n" + str(board))
    print("result:", board.result(claim_draw=True))


if __name__ == "__main__":
    main()
