"""Run YOUR MCTS.   ->  python user/run_search.py

Two searches, both on a dummy evaluator (no network):
  1) the opening position, just to see a policy come out
  2) a forced mate-in-1 — if your search finds it, the logic is correct.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

import chess
from mcts_user import MCTS


def dummy_evaluate(board):
    """Stand-in for the network: uniform priors, neutral value."""
    moves = list(board.legal_moves)
    return {m: 1.0 / len(moves) for m in moves}, 0.0


mcts = MCTS(dummy_evaluate, simulations=50)

# ---------------------------------------------------------------- 1) opening
board = chess.Board()
policy = mcts.run(board)
print("== Opening search: top 5 moves by visit share ==")
for mv, p in sorted(policy.items(), key=lambda kv: kv[1], reverse=True)[:5]:
    print(f"  {mv.uci()}   {p:.3f}")

# ---------------------------------------------------------------- 2) mate-in-1
mate_fen = "6k1/5ppp/8/8/8/8/8/4R1K1 w - - 0 1"   # White: Re1 + Kg1 ; Re8 is mate
board = chess.Board(mate_fen)

# sanity: confirm the position really HAS a mate in 1, and which move it is
truth = None
for mv in board.legal_moves:
    b = board.copy(); b.push(mv)
    if b.is_checkmate():
        truth = mv
print("\n== Mate-in-1 test ==")
print(board)
print("a real mate-in-1 exists:", truth.uci() if truth else "NONE (bad FEN)")

policy = mcts.run(board)
best = max(policy, key=policy.get)
print(f"MCTS's most-visited move: {best.uci()}  (visit share {policy[best]:.3f})")

check = board.copy(); check.push(best)
print("does it deliver checkmate? ->", check.is_checkmate())
print("\n*** YOUR MCTS WORKS ***" if check.is_checkmate() else "\n(missed the mate — let's debug)")
