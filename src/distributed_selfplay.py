"""Multiprocessing self-play — put all those CPU cores to work.

Single-process self-play pins ONE core at 100% (the Python board/tree search) and
leaves the GPU ~idle, because the network forward is a tiny sliver of each step.
This module runs N worker processes, each generating a share of the games with the
current net, so every core works at once and the GPU stays fed by many processes.

Actor/learner split:
  - the trainer (main process) writes the current net to a file each iteration,
  - workers reload it and play their chunk of games,
  - the trainer collects the examples and trains.

A persistent 'spawn' pool means each worker's CUDA context is created ONCE, not
per iteration. 'spawn' (not 'fork') is required for CUDA in subprocesses.
"""
from __future__ import annotations

import multiprocessing as mp
import os
import random

import numpy as np
import torch

from .config import Config
from .network import ChessNet
from .selfplay import generate_games_parallel

# Per-worker process-global state (each worker keeps its own net loaded).
_STATE: dict = {}


def _init_worker(cfg: Config, net_path: str) -> None:
    torch.set_num_threads(1)              # one BLAS thread per worker — no oversubscription
    net = ChessNet(cfg.net).to(cfg.device)
    net.eval()
    _STATE["cfg"] = cfg
    _STATE["net"] = net
    _STATE["net_path"] = net_path
    _STATE["mtime"] = None


def _reload_if_changed() -> None:
    path = _STATE["net_path"]
    if not os.path.exists(path):
        return
    m = os.path.getmtime(path)
    if m != _STATE["mtime"]:
        _STATE["net"].load_state_dict(torch.load(path, map_location=_STATE["cfg"].device))
        _STATE["net"].eval()
        _STATE["mtime"] = m


def _play_chunk(args):
    n_games, seed = args
    _reload_if_changed()
    torch.manual_seed(seed)
    np.random.seed(seed % (2**31 - 1))
    random.seed(seed)
    rng = np.random.default_rng(seed)
    return list(generate_games_parallel(_STATE["net"], _STATE["cfg"], n_games, rng))


class DistributedSelfPlay:
    """Persistent pool of self-play workers. Call generate() once per iteration."""

    def __init__(self, cfg: Config, n_workers: int, net_path: str):
        self.n_workers = n_workers
        self.net_path = net_path
        ctx = mp.get_context("spawn")
        self.pool = ctx.Pool(n_workers, initializer=_init_worker, initargs=(cfg, net_path))

    def generate(self, net, cfg, n_games: int, iteration: int = 0):
        """Play `n_games` across the workers; return a flat list of games
        (each game is a list of Example)."""
        torch.save(net.state_dict(), self.net_path)   # publish current net for workers

        per = [n_games // self.n_workers] * self.n_workers
        for i in range(n_games % self.n_workers):
            per[i] += 1
        base = 1_000_000 * (iteration + 1)
        args = [(per[i], base + i) for i in range(self.n_workers) if per[i] > 0]

        chunks = self.pool.map(_play_chunk, args)
        return [game for chunk in chunks for game in chunk]

    def close(self) -> None:
        self.pool.close()
        self.pool.join()
