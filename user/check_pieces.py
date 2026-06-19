"""Run the pieces you've already written.  ->  python user/check_pieces.py

Exercises Node, MCTS._expand, and MCTS._backup with a fake evaluator, so the
code you wrote produces visible output. No network, no torch — just your logic.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))   # so `import mcts_user` works

import chess
from mcts_user import Node, MCTS


# A stand-in for the network: uniform priors over legal moves, neutral value.
def dummy_evaluate(board):
    moves = list(board.legal_moves)
    priors = {m: 1.0 / len(moves) for m in moves}
    return priors, 0.0


print("== 1) Node on its own ==")
n = Node(prior=0.25, to_play=chess.WHITE)
print("fresh node     -> value():", n.value(), " is_expanded():", n.is_expanded())
n.visit_count = 4
n.value_sum = 3.0
print("after N=4, W=3 -> value():", n.value(), "  (expect 0.75)")
print()

print("== 2) _expand: grow children at a leaf ==")
mcts = MCTS(dummy_evaluate)
board = chess.Board()
root = Node(prior=1.0, to_play=board.turn)
returned = mcts._expand(root, board)
print("root now expanded? ", root.is_expanded(), " #children:", len(root.children),
      "(expect 20 at the start)")
move, child = next(iter(root.children.items()))
print(f"one child: move {move.uci()} -> prior {child.prior:.4f}, to_play {child.to_play}")
print("evaluate's value was returned:", returned)
print()

print("== 3) _backup: push a value up a path, watch the sign flip ==")
# a fake 3-deep path: root -> mid -> leaf
root_n = Node(prior=1.0, to_play=chess.WHITE)
mid_n  = Node(prior=0.5, to_play=chess.BLACK)
leaf_n = Node(prior=0.5, to_play=chess.WHITE)
mcts._backup([root_n, mid_n, leaf_n], value=1.0)   # +1 from the LEAF mover's view
for label, nd in [("leaf", leaf_n), ("mid ", mid_n), ("root", root_n)]:
    print(f"{label}: N={nd.visit_count} W={nd.value_sum:+.1f} value()={nd.value():+.1f}")
print("note: +1 at the leaf becomes -1 one ply up, +1 again at root — the flip.")
