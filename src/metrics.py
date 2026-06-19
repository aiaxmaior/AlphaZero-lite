"""Training metrics: a tail-friendly console line + CSV/JSONL + optional TensorBoard.

One `MetricsLogger` per run. Call `.log(iteration, {...})` once per iteration; it:
  - appends a row to runs/<name>/metrics.csv and metrics.jsonl,
  - mirrors scalars to TensorBoard (runs/<name>/tb) if available,
  - writes a one-line human summary (with elapsed + ETA) to stdout AND to
    runs/<name>/progress.log, so `tail -f progress.log` always works.
"""
from __future__ import annotations

import csv
import json
import os
import time


def _hms(seconds: float) -> str:
    seconds = int(max(0, seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}"


class MetricsLogger:
    def __init__(self, run_dir: str, total_iterations: int, use_tb: bool = True):
        self.run_dir = run_dir
        self.total = total_iterations
        os.makedirs(run_dir, exist_ok=True)
        self.csv_path = os.path.join(run_dir, "metrics.csv")
        self.jsonl_path = os.path.join(run_dir, "metrics.jsonl")
        self.log_path = os.path.join(run_dir, "progress.log")
        self.start = time.time()
        self.last = self.start
        self.iter_times: list[float] = []

        self.tb = None
        if use_tb:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.tb = SummaryWriter(os.path.join(run_dir, "tb"))
            except Exception:
                self.tb = None

    def banner(self, text: str) -> None:
        """A free-form line to both stdout and progress.log (e.g. run header)."""
        print(text, flush=True)
        with open(self.log_path, "a") as f:
            f.write(text + "\n")

    def log(self, iteration: int, metrics: dict) -> None:
        now = time.time()
        iter_time = now - self.last
        self.last = now
        self.iter_times.append(iter_time)
        avg = sum(self.iter_times) / len(self.iter_times)
        eta = avg * (self.total - iteration)

        row = {
            "iteration": iteration,
            "elapsed_s": round(now - self.start, 1),
            "iter_s": round(iter_time, 1),
            **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in metrics.items()},
        }

        new = (not os.path.exists(self.csv_path)) or os.path.getsize(self.csv_path) == 0
        with open(self.csv_path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row.keys()))
            if new:
                w.writeheader()
            w.writerow(row)
        with open(self.jsonl_path, "a") as f:
            f.write(json.dumps(row) + "\n")

        if self.tb:
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    self.tb.add_scalar(k, v, iteration)

        m = metrics
        line = (
            f"[iter {iteration:>3}/{self.total}] "
            f"loss {m.get('loss', 0):.3f} (p {m.get('policy_loss', 0):.3f} "
            f"v {m.get('value_loss', 0):.3f}) | "
            f"games {m.get('games', 0)} pos {m.get('positions', 0)} buf {m.get('buffer', 0)} | "
            f"selfplay {m.get('selfplay_s', 0):.0f}s train {m.get('train_s', 0):.0f}s "
            f"({m.get('games_per_s', 0):.2f} g/s) | "
            f"elapsed {_hms(now - self.start)} ETA {_hms(eta)}"
        )
        print(line, flush=True)
        with open(self.log_path, "a") as f:
            f.write(line + "\n")

    def close(self) -> None:
        if self.tb:
            self.tb.close()
