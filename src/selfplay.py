"""Self-play: the network plays itself to generate training data.

For each game we record, at every move, a training example:
    (board_planes, search_policy_pi, player_to_move)
and once the game ends we fill in z = the final result from that player's view.
That triple (s, pi, z) is exactly what train.py fits the two heads to.

There is no opponent and no labels from outside — the only signal is who won the
games the current network played against itself. That is the RL part.
"""
from __future__ import annotations

from dataclasses import dataclass
import sys
import numpy as np
import chess

from .config import Config
from .encoding import encode_board, index_to_move
from .mcts import MCTS, select_move


@dataclass
class Example:
    planes: np.ndarray      # (17, 8, 8)
    pi: np.ndarray          # (4672,) MCTS visit-count policy
    to_play: bool           # chess.WHITE / chess.BLACK at this position
    z: float = 0.0          # filled in once the game ends


def play_game(net, cfg: Config, rng=np.random) -> list[Example]:
    """Play one self-play game; return its list of (filled-in) examples."""
    board = chess.Board()
    mcts = MCTS(net, cfg.mcts, cfg.device)
    examples: list[Example] = []

    while not board.is_game_over(claim_draw=True) and board.fullmove_number * 2 <= cfg.selfplay.max_moves:
        pi = mcts.run(board, add_noise=True)

        examples.append(Example(encode_board(board), pi, board.turn))

        # Explore early (sample), then exploit (greedy) — controlled by temperature.
        ply = len(examples)
        temp = 1.0 if ply <= cfg.mcts.temperature_moves else 0.0
        action = select_move(pi, temp, rng)

        from .encoding import index_to_move
        board.push(index_to_move(action, board))

    z = _final_result(board)        # +1 white win, -1 black win, 0 draw
    for ex in examples:
        # Convert the absolute result into "did the player to move win?".
        ex.z = z if ex.to_play == chess.WHITE else -z
    return examples


def _final_result(board: chess.Board) -> float:
    """+1 if White won, -1 if Black won, 0 for any draw / adjudicated game."""
    if board.is_checkmate():
        # The side to move was just mated, so the *other* side won.
        return -1.0 if board.turn == chess.WHITE else 1.0
    return 0.0


def generate_games(net, cfg: Config, n_games: int, rng=np.random):
    """Generate `n_games` self-play games; yield each game's examples."""
    try:
        from tqdm import trange
        iterator = trange(n_games, desc="self-play")
    except ImportError:
        iterator = range(n_games)
    for _ in iterator:
        yield play_game(net, cfg, rng)


def generate_games_parallel(net, cfg: Config, n_games: int, rng=np.random):
    """GPU-efficient self-play: keep `parallel_games` games in flight at once and
    batch their MCTS leaf evaluations. Yields each game's examples as it finishes.

    All in-flight games are searched together (one batched network call per
    simulation), then each advances one move. When a game ends it's finalised and
    a fresh game takes its slot until `n_games` have been produced.
    """
    from types import SimpleNamespace
    from .mcts import BatchedMCTS

    n_parallel = max(1, min(cfg.selfplay.parallel_games, n_games))
    mcts = BatchedMCTS(net, cfg.mcts, cfg.device, cfg.precision)

    def fresh():
        return SimpleNamespace(board=chess.Board(), examples=[], ply=0)

    active = [fresh() for _ in range(n_parallel)]
    started = n_parallel
    finished = 0

    try:
        from tqdm import tqdm
        bar = tqdm(total=n_games, desc="self-play", disable=not sys.stderr.isatty())
    except ImportError:
        bar = None

    while active:
        policies = mcts.run([g.board for g in active], add_noise=True)
        still = []
        for g, pi in zip(active, policies):
            g.examples.append(Example(encode_board(g.board), pi, g.board.turn))
            g.ply += 1
            temp = 1.0 if g.ply <= cfg.mcts.temperature_moves else 0.0
            g.board.push(index_to_move(select_move(pi, temp, rng), g.board))

            over = (g.board.is_game_over(claim_draw=True)
                    or g.board.fullmove_number * 2 > cfg.selfplay.max_moves)
            if over:
                z = _final_result(g.board)
                for ex in g.examples:
                    ex.z = z if ex.to_play == chess.WHITE else -z
                finished += 1
                if bar:
                    bar.update(1)
                yield g.examples
                if started < n_games:                  # refill the slot
                    still.append(fresh())
                    started += 1
            else:
                still.append(g)
        active = still

    if bar:
        bar.close()
