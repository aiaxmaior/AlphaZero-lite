"""The outer RL loop: self-play -> train -> checkpoint, repeated, with logging.

One "iteration" is:
    1. play `games_per_iter` self-play games with the current net
    2. add their (s, pi, z) examples to a replay buffer (last N games)
    3. take a few SGD steps fitting policy+value to the buffer
    4. checkpoint, and log metrics (loss, timings, ETA) to the run directory

Everything lands in runs/<run_name>/ :
    config.yaml      the fully-resolved config this run used
    progress.log     one human line per iteration  ->  tail -f this
    metrics.csv      same data as columns           ->  plot / inspect
    metrics.jsonl    same data as json lines
    tb/              TensorBoard scalars (if installed)
    checkpoints/     net_iterNNN.pt + latest.pt

Run it:
    python -m src.train --config configs/default.yaml
    python -m src.train --config configs/default.yaml --name run_0 --sims 200
    python -m src.train --iterations 10 --games 50 --sims 100   # no yaml, pure CLI
"""
from __future__ import annotations

import argparse
import os
import random
import time
from collections import deque

import numpy as np
import torch

from .config import Config, load_config, save_config
from .metrics import MetricsLogger
from .network import ChessNet, policy_value_loss
from .selfplay import generate_games, generate_games_parallel


def _resolve_device(requested: str) -> str:
    if requested.startswith("cuda") and not torch.cuda.is_available():
        print("[train] CUDA not available, falling back to CPU.")
        return "cpu"
    return requested


def train_on_buffer(net, buffer, cfg: Config, optim) -> dict:
    """Run `epochs_per_iter` passes of SGD over the replay buffer.

    Returns the mean {loss, policy_loss, value_loss} over the iteration's steps.
    """
    net.train()
    device = cfg.device
    examples = list(buffer)
    if not examples:
        return {"loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0}

    tot = tot_p = tot_v = 0.0
    steps = 0
    for _ in range(cfg.train.epochs_per_iter):
        random.shuffle(examples)
        for i in range(0, len(examples), cfg.train.batch_size):
            batch = examples[i : i + cfg.train.batch_size]
            planes = torch.from_numpy(np.stack([e.planes for e in batch])).to(device)
            target_pi = torch.from_numpy(np.stack([e.pi for e in batch])).to(device)
            target_z = torch.tensor([e.z for e in batch], dtype=torch.float32, device=device)

            logits, value = net(planes)
            loss, p_loss, v_loss = policy_value_loss(
                logits, value, target_pi, target_z, cfg.train.value_weight
            )

            optim.zero_grad()
            loss.backward()
            optim.step()

            tot += loss.item()
            tot_p += p_loss.item()
            tot_v += v_loss.item()
            steps += 1

    steps = max(1, steps)
    return {"loss": tot / steps, "policy_loss": tot_p / steps, "value_loss": tot_v / steps}


def _build_config(args) -> Config:
    cfg = load_config(args.config) if args.config else Config()
    if args.iterations is not None:
        cfg.iterations = args.iterations
    if args.games is not None:
        cfg.selfplay.games_per_iter = args.games
    if args.sims is not None:
        cfg.mcts.simulations = args.sims
    if args.parallel is not None:
        cfg.selfplay.parallel_games = args.parallel
    if args.workers is not None:
        cfg.selfplay.num_workers = args.workers
    if args.precision is not None:
        cfg.precision = args.precision
    if args.name is not None:
        cfg.run_name = args.name
    cfg.device = _resolve_device(cfg.device)
    if not cfg.run_name:
        cfg.run_name = "run_" + time.strftime("%Y%m%d_%H%M%S")
    return cfg


def main():
    ap = argparse.ArgumentParser(description="self-play training loop")
    ap.add_argument("--config", type=str, default=None, help="YAML config file")
    ap.add_argument("--name", type=str, default=None, help="run name (-> runs/<name>/)")
    ap.add_argument("--iterations", type=int, default=None)
    ap.add_argument("--games", type=int, default=None, help="self-play games per iteration")
    ap.add_argument("--sims", type=int, default=None, help="MCTS simulations per move")
    ap.add_argument("--parallel", type=int, default=None, help="games batched in lockstep")
    ap.add_argument("--workers", type=int, default=None, help="self-play worker processes")
    ap.add_argument("--precision", choices=["bf16", "fp16", "fp32"], default=None)
    ap.add_argument("--resume", type=str, default=None, help="checkpoint to load")
    args = ap.parse_args()

    cfg = _build_config(args)

    run_dir = os.path.join(cfg.runs_dir, cfg.run_name)
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    save_config(cfg, os.path.join(run_dir, "config.yaml"))

    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    net = ChessNet(cfg.net).to(cfg.device)
    if args.resume:
        net.load_state_dict(torch.load(args.resume, map_location=cfg.device))
        print(f"[train] resumed from {args.resume}")

    optim = torch.optim.Adam(net.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    buffer = deque(maxlen=cfg.train.replay_buffer_games * 40)

    logger = MetricsLogger(run_dir, cfg.iterations, use_tb=cfg.tensorboard)
    logger.banner(f"[train] run '{cfg.run_name}' -> {run_dir}")
    logger.banner(
        f"[train] device={cfg.device} precision={cfg.precision} "
        f"net={cfg.net.channels}x{cfg.net.blocks} sims={cfg.mcts.simulations} "
        f"games/iter={cfg.selfplay.games_per_iter} parallel={cfg.selfplay.parallel_games} "
        f"iters={cfg.iterations} tensorboard={'on' if logger.tb else 'off'}"
    )

    dsp = None
    if cfg.selfplay.num_workers > 1:
        from .distributed_selfplay import DistributedSelfPlay
        worker_net_path = os.path.join(run_dir, "_worker_net.pt")
        dsp = DistributedSelfPlay(cfg, cfg.selfplay.num_workers, worker_net_path)
        logger.banner(f"[train] {cfg.selfplay.num_workers} self-play worker processes")
    gen = generate_games_parallel if cfg.selfplay.parallel_games > 1 else generate_games

    for it in range(1, cfg.iterations + 1):
        logger.banner(f"[iter {it}/{cfg.iterations}] self-play "
                      f"{cfg.selfplay.games_per_iter} games @ {cfg.mcts.simulations} sims ...")

        t0 = time.time()
        positions = games = 0
        games_iter = (dsp.generate(net, cfg, cfg.selfplay.games_per_iter, it)
                      if dsp else gen(net, cfg, cfg.selfplay.games_per_iter))
        for game in games_iter:
            buffer.extend(game)
            positions += len(game)
            games += 1
        selfplay_s = time.time() - t0

        t1 = time.time()
        losses = train_on_buffer(net, buffer, cfg, optim)
        train_s = time.time() - t1

        torch.save(net.state_dict(), os.path.join(ckpt_dir, f"net_iter{it:03d}.pt"))
        torch.save(net.state_dict(), os.path.join(ckpt_dir, "latest.pt"))

        logger.log(it, {
            **losses,
            "games": games,
            "positions": positions,
            "buffer": len(buffer),
            "selfplay_s": round(selfplay_s, 1),
            "train_s": round(train_s, 1),
            "games_per_s": round(games / max(selfplay_s, 1e-9), 3),
        })

    if dsp:
        dsp.close()
    logger.close()
    logger.banner(f"[train] done — {cfg.iterations} iterations. checkpoints in {ckpt_dir}")


if __name__ == "__main__":
    main()
