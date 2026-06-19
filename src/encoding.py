"""Encoding: chess position <-> tensor, and chess move <-> action index.

This is the bridge between the rules of chess (python-chess) and the neural net
(which only speaks tensors). Two jobs:

  1. encode_board(board)  ->  float32 array of shape (17, 8, 8)
  2. a *fixed* action space of 4672 moves, so the policy head has a constant size.

Two conventions worth internalising, because they trip everyone up:

  * EVERYTHING IS FROM THE SIDE-TO-MOVE'S PERSPECTIVE. We mirror the board
    vertically when Black is to move, so "forward" is always toward rank 8.
    The network therefore only ever has to learn "how do *I* win from here",
    never "how does White win" and separately "how does Black win".

  * The action space is AlphaZero's 8x8x73 encoding:
        from_square (64) x move_type (73) = 4672 logits.
    The 73 move types are: 56 "queen" moves (8 directions x 7 distances),
    8 knight moves, and 9 underpromotions (knight/bishop/rook x 3 pawn directions).
    Queen-promotions are just queen moves to the back rank.
"""
from __future__ import annotations

import argparse
import numpy as np
import chess

# ----------------------------------------------------------------------------
# Input encoding: board -> (17, 8, 8)
# ----------------------------------------------------------------------------
# Plane layout (all from the mover's perspective):
#   0-5   : our pieces   (pawn, knight, bishop, rook, queen, king)
#   6-11  : their pieces  (same order)
#   12-15 : castling rights (our K, our Q, their K, their Q)
#   16    : 1.0 everywhere if White is to move, else 0.0  (absolute-colour hint)
N_PLANES = 17
ACTION_SIZE = 64 * 73


def _orient(square: int, turn: bool) -> int:
    """Map a real square into the mover's frame (mirror vertically for Black)."""
    return square if turn == chess.WHITE else chess.square_mirror(square)


def encode_board(board: chess.Board) -> np.ndarray:
    """Return the (17, 8, 8) float32 planes for `board`, mover's perspective."""
    planes = np.zeros((N_PLANES, 8, 8), dtype=np.float32)
    us = board.turn

    for square, piece in board.piece_map().items():
        osq = _orient(square, us)
        row, col = divmod(osq, 8)          # rank, file in oriented frame
        plane = (piece.piece_type - 1) + (0 if piece.color == us else 6)
        planes[plane, row, col] = 1.0

    # Castling rights, expressed as "ours" / "theirs".
    them = not us
    rights = [
        board.has_kingside_castling_rights(us),
        board.has_queenside_castling_rights(us),
        board.has_kingside_castling_rights(them),
        board.has_queenside_castling_rights(them),
    ]
    for i, has in enumerate(rights):
        if has:
            planes[12 + i, :, :] = 1.0

    if us == chess.WHITE:
        planes[16, :, :] = 1.0

    return planes


# ----------------------------------------------------------------------------
# Action encoding: chess.Move <-> index in [0, 4672)
# ----------------------------------------------------------------------------
# We build, once, a table mapping each of the 73 "move type" planes to a
# (delta_file, delta_rank, promotion) triple in the oriented frame.

# 8 sliding directions, ordered N, NE, E, SE, S, SW, W, NW.
_QUEEN_DIRS = [(0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1)]
# 8 knight deltas, fixed order.
_KNIGHT_DELTAS = [(1, 2), (2, 1), (2, -1), (1, -2), (-1, -2), (-2, -1), (-2, 1), (-1, 2)]
# Underpromotion piece types, in the order used by the 9 underpromo planes.
_UNDERPROMO = [chess.KNIGHT, chess.BISHOP, chess.ROOK]


def _build_plane_table():
    """plane index -> (df, dr, promotion or None), in the oriented frame."""
    table = [None] * 73
    # 0..55: queen moves = direction * 7 + (distance-1)
    for d, (dx, dy) in enumerate(_QUEEN_DIRS):
        for dist in range(1, 8):
            table[d * 7 + (dist - 1)] = (dx * dist, dy * dist, None)
    # 56..63: knight moves
    for k, (dx, dy) in enumerate(_KNIGHT_DELTAS):
        table[56 + k] = (dx, dy, None)
    # 64..72: underpromotions, piece * 3 + (df+1); pawn always moves dr=+1.
    for p, piece in enumerate(_UNDERPROMO):
        for df in (-1, 0, 1):
            table[64 + p * 3 + (df + 1)] = (df, 1, piece)
    return table


_PLANE_TABLE = _build_plane_table()
# Reverse lookup for encoding: (df, dr, promo) -> plane.
_DELTA_TO_PLANE = {v: i for i, v in enumerate(_PLANE_TABLE)}


def move_to_index(move: chess.Move, turn: bool) -> int:
    """Encode a legal `move` (in real coords) to an action index in [0, 4672)."""
    frm = _orient(move.from_square, turn)
    to = _orient(move.to_square, turn)
    ff, fr = divmod(frm, 8)[::-1]          # file, rank
    tf, tr = divmod(to, 8)[::-1]
    df, dr = tf - ff, tr - fr

    promo = move.promotion
    if promo is not None and promo != chess.QUEEN:
        # Underpromotion: forward step is always dr=+1 in oriented frame.
        plane = _DELTA_TO_PLANE[(df, 1, promo)]
    else:
        # Queen-move (covers normal moves, queen promotions, castling, knights).
        if (abs(df), abs(dr)) in ((1, 2), (2, 1)):
            plane = _DELTA_TO_PLANE[(df, dr, None)]          # knight
        else:
            plane = _DELTA_TO_PLANE[(df, dr, None)]          # sliding/king/pawn
    return frm * 73 + plane


def index_to_move(index: int, board: chess.Board) -> chess.Move:
    """Decode an action index back to a chess.Move in real coordinates.

    Needs the board to (a) de-orient and (b) know when a queen-plane move that
    lands on the back rank is actually a (queen) promotion.
    """
    turn = board.turn
    frm_o, plane = divmod(index, 73)
    df, dr, promo = _PLANE_TABLE[plane]

    ff, fr = divmod(frm_o, 8)[::-1]
    tf, tr = ff + df, fr + dr
    if not (0 <= tf < 8 and 0 <= tr < 8):
        raise ValueError(f"action {index} leaves the board")
    to_o = tr * 8 + tf

    # De-orient both squares back to real coordinates (mirror is its own inverse).
    frm = _orient(frm_o, turn)
    to = _orient(to_o, turn)

    if promo is None:
        # Promote to queen automatically if a pawn reaches the last rank.
        piece = board.piece_at(frm)
        last_rank = 7 if turn == chess.WHITE else 0
        if piece is not None and piece.piece_type == chess.PAWN and chess.square_rank(to) == last_rank:
            promo = chess.QUEEN
    return chess.Move(frm, to, promotion=promo)


def legal_action_mask(board: chess.Board) -> np.ndarray:
    """Boolean (4672,) mask: True for indices that are legal moves right now."""
    mask = np.zeros(ACTION_SIZE, dtype=bool)
    for mv in board.legal_moves:
        mask[move_to_index(mv, board.turn)] = True
    return mask


# ----------------------------------------------------------------------------
# Self-test: encode -> decode must be the identity on every legal move.
# ----------------------------------------------------------------------------
def _selftest(n_positions: int = 2000) -> None:
    import random
    rng = random.Random(0)
    checked = 0
    for _ in range(n_positions):
        board = chess.Board()
        for _ in range(rng.randint(0, 40)):          # random reachable position
            moves = list(board.legal_moves)
            if not moves:
                break
            board.push(rng.choice(moves))
        for mv in board.legal_moves:
            idx = move_to_index(mv, board.turn)
            back = index_to_move(idx, board)
            assert back == mv, f"round-trip failed: {mv} -> {idx} -> {back}\n{board.fen()}"
            assert 0 <= idx < ACTION_SIZE
            checked += 1
    print(f"OK: {checked} legal moves round-tripped through the {ACTION_SIZE}-action space.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="encoding self-test")
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()
    if args.selftest:
        _selftest()
    else:
        b = chess.Board()
        print("planes:", encode_board(b).shape, "action size:", ACTION_SIZE)
