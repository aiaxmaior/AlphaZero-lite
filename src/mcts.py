"""Monte-Carlo Tree Search, network-guided (the AlphaZero variant).

Plain MCTS rolls out random games to estimate who's winning. AlphaZero throws
that away and uses the network instead:

  * the *policy* head tells the search which moves are worth exploring (priors),
  * the *value* head replaces the random rollout with a single position eval.

Each simulation does four steps:
  1. SELECT   walk down the tree picking the child with the highest PUCT score
  2. EXPAND   at a leaf, ask the network for (priors, value)
  3. (no separate simulation step — the value head *is* the estimate)
  4. BACKUP   propagate the value up the path, flipping sign each ply
              (a position good for me is bad for my opponent)

The output we actually use for learning is the *visit-count distribution* at the
root: moves the search spent more time on are the moves we train the policy toward.
"""
from __future__ import annotations

import math
import numpy as np
import chess

from .config import MCTSConfig
from .encoding import encode_board, legal_action_mask, index_to_move, move_to_index


class Node:
    """One board position in the search tree."""

    __slots__ = ("prior", "visit_count", "value_sum", "children", "to_play")

    def __init__(self, prior: float, to_play: bool):
        self.prior = prior              # P(this move) from the policy head
        self.visit_count = 0
        self.value_sum = 0.0            # sum of backed-up values (mover's view)
        self.children: dict[int, "Node"] = {}   # action index -> child Node
        self.to_play = to_play          # whose turn it is AT this node

    @property
    def value(self) -> float:
        return self.value_sum / self.visit_count if self.visit_count else 0.0

    def is_expanded(self) -> bool:
        return bool(self.children)


def _ucb_score(parent: Node, child: Node, c_puct: float) -> float:
    """PUCT: exploitation (Q) + exploration (U) terms."""
    u = c_puct * child.prior * math.sqrt(parent.visit_count) / (1 + child.visit_count)
    # child.value is from the child-mover's perspective; negate for the parent.
    q = -child.value if child.visit_count else 0.0
    return q + u


class MCTS:
    def __init__(self, net, cfg: MCTSConfig, device: str):
        self.net = net
        self.cfg = cfg
        self.device = device

    def run(self, board: chess.Board, add_noise: bool = True) -> np.ndarray:
        """Run `simulations` rollouts from `board`; return a (4672,) policy
        proportional to root visit counts (zeros on illegal moves)."""
        root = Node(prior=1.0, to_play=board.turn)
        self._expand(root, board)
        if add_noise:
            self._add_dirichlet_noise(root)

        for _ in range(self.cfg.simulations):
            node = root
            scratch = board.copy()
            path = [node]

            # 1. SELECT to a leaf.
            while node.is_expanded():
                action, node = self._select_child(node)
                scratch.push(index_to_move(action, scratch))
                path.append(node)

            # 2. EXPAND + evaluate (unless the game is already over here).
            value = self._evaluate_terminal(scratch)
            if value is None:
                value = self._expand(node, scratch)

            # 4. BACKUP, flipping sign each step up the path.
            for n in reversed(path):
                n.visit_count += 1
                n.value_sum += value
                value = -value

        return self._visit_policy(root)

    # -- internals ---------------------------------------------------------
    def _select_child(self, node: Node):
        return max(
            node.children.items(),
            key=lambda kv: _ucb_score(node, kv[1], self.cfg.c_puct),
        )

    def _expand(self, node: Node, board: chess.Board) -> float:
        """Ask the network to score this position; create children. Returns the
        value estimate (mover's perspective) for backup."""
        mask = legal_action_mask(board)
        priors, value = self.net.predict(encode_board(board), mask, self.device)
        next_to_play = not board.turn
        for action in np.nonzero(mask)[0]:
            node.children[int(action)] = Node(float(priors[action]), next_to_play)
        return value

    @staticmethod
    def _evaluate_terminal(board: chess.Board):
        """Return a value in {-1, 0} if the game is over, else None.

        From the perspective of the player to move at `board`: a checkmate means
        *they* just got mated -> -1; stalemate/draw -> 0.
        """
        if board.is_checkmate():
            return -1.0
        if board.is_game_over(claim_draw=True):
            return 0.0
        return None

    def _add_dirichlet_noise(self, root: Node):
        eps, alpha = self.cfg.dirichlet_eps, self.cfg.dirichlet_alpha
        if eps <= 0:
            return
        actions = list(root.children)
        noise = np.random.dirichlet([alpha] * len(actions))
        for a, n in zip(actions, noise):
            root.children[a].prior = (1 - eps) * root.children[a].prior + eps * n

    @staticmethod
    def _visit_policy(root: Node) -> np.ndarray:
        from .encoding import ACTION_SIZE
        pi = np.zeros(ACTION_SIZE, dtype=np.float32)
        total = sum(c.visit_count for c in root.children.values())
        if total == 0:
            return pi
        for a, c in root.children.items():
            pi[a] = c.visit_count / total
        return pi


def select_move(pi: np.ndarray, temperature: float, rng=np.random) -> int:
    """Pick an action index from a visit-count policy.

    temperature -> 0 : play the most-visited move (deterministic, strongest).
    temperature = 1   : sample in proportion to visit counts (exploration).
    """
    if temperature <= 1e-3:
        return int(pi.argmax())
    logits = np.log(np.maximum(pi, 1e-12)) / temperature
    probs = np.exp(logits - logits.max())
    probs /= probs.sum()
    return int(rng.choice(len(probs), p=probs))


# ----------------------------------------------------------------------------
# Batched MCTS — run many games in lockstep so leaf evals batch into one call.
# ----------------------------------------------------------------------------
def _backup(path, value):
    """Propagate `value` up `path`, flipping sign each ply (zero-sum)."""
    for node in reversed(path):
        node.visit_count += 1
        node.value_sum += value
        value = -value


def _expand_from(node: Node, mask: np.ndarray, priors: np.ndarray, to_play: bool):
    """Create children of `node` for every legal action in `mask`."""
    for action in np.nonzero(mask)[0]:
        node.children[int(action)] = Node(float(priors[action]), to_play)


class BatchedMCTS:
    """MCTS over a *list* of games advanced together.

    The single-game MCTS calls the network once per simulation (batch 1), which
    starves the GPU. Here we run G games in lockstep: each simulation descends
    one leaf per game, then ALL G leaves are evaluated in a single
    `predict_batch` call. Batch size ≈ G, so a 4070 Ti finally has work to do.
    The batching is *across games*; each game still gets `simulations` sims.
    """

    def __init__(self, net, cfg: MCTSConfig, device: str, precision: str = "fp32"):
        self.net = net
        self.cfg = cfg
        self.device = device
        import torch
        self._autocast = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(precision)

    def run(self, boards, add_noise: bool = True):
        """boards: list[chess.Board] (all non-terminal). Returns a list of
        (4672,) visit-count policies, one per board."""
        roots = [Node(prior=1.0, to_play=b.turn) for b in boards]
        self._expand_roots(roots, boards)
        if add_noise:
            for root in roots:
                self._add_dirichlet_noise(root)

        for _ in range(self.cfg.simulations):
            self._simulate_step(roots, boards)

        return [MCTS._visit_policy(root) for root in roots]

    # -- one batched simulation across all games ---------------------------
    def _simulate_step(self, roots, boards):
        pending = []  # (path, scratch_board) for leaves that need a network eval
        for root, board in zip(roots, boards):
            node = root
            scratch = board.copy()
            path = [node]
            while node.is_expanded():                    # SELECT
                action, node = max(
                    node.children.items(),
                    key=lambda kv: _ucb_score(path[-1], kv[1], self.cfg.c_puct),
                )
                scratch.push(index_to_move(action, scratch))
                path.append(node)

            value = MCTS._evaluate_terminal(scratch)
            if value is not None:                        # terminal: no eval needed
                _backup(path, value)
            else:
                pending.append((path, scratch))

        if not pending:
            return

        # EXPAND + EVALUATE all leaves in one forward pass.
        planes = np.stack([encode_board(b) for _, b in pending])
        masks = np.stack([legal_action_mask(b) for _, b in pending])
        priors, values = self.net.predict_batch(planes, masks, self.device, self._autocast)
        for k, (path, board) in enumerate(pending):
            _expand_from(path[-1], masks[k], priors[k], not board.turn)
            _backup(path, float(values[k]))

    def _expand_roots(self, roots, boards):
        planes = np.stack([encode_board(b) for b in boards])
        masks = np.stack([legal_action_mask(b) for b in boards])
        priors, _ = self.net.predict_batch(planes, masks, self.device, self._autocast)
        for k, (root, board) in enumerate(zip(roots, boards)):
            _expand_from(root, masks[k], priors[k], not board.turn)

    def _add_dirichlet_noise(self, root: Node):
        eps, alpha = self.cfg.dirichlet_eps, self.cfg.dirichlet_alpha
        if eps <= 0:
            return
        actions = list(root.children)
        noise = np.random.dirichlet([alpha] * len(actions))
        for a, n in zip(actions, noise):
            root.children[a].prior = (1 - eps) * root.children[a].prior + eps * n
