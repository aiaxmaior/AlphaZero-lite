"""Live training monitor — a nicer alternative to `tail` / `watch`.

    python -m src.monitor                 # show the most recent run
    python -m src.monitor runs/run_0      # a specific run
    python -m src.monitor --watch 5       # refresh every 5 seconds

Shows the current iteration, latest losses, timing, ETA, a short loss trend,
the latest checkpoint, and whether a training process is actually alive.
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import subprocess
import time


def latest_run(runs_dir: str = "runs") -> str | None:
    dirs = [d for d in glob.glob(os.path.join(runs_dir, "*")) if os.path.isdir(d)]
    return max(dirs, key=os.path.getmtime) if dirs else None


def read_rows(run_dir: str) -> list[dict]:
    path = os.path.join(run_dir, "metrics.csv")
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def training_alive() -> bool:
    """True if an actual `python -m src.train` process is running (not this
    monitor, and not a shell that merely mentions the string)."""
    try:
        out = subprocess.run(["pgrep", "-af", r"src\.train"], capture_output=True, text=True)
        for line in out.stdout.splitlines():
            if "src.train" in line and "python" in line and "monitor" not in line:
                return True
        return False
    except Exception:
        return False


def show(run_dir: str) -> None:
    rows = read_rows(run_dir)
    print("=" * 70)
    print(f"run: {run_dir}")
    print(f"training process: {'● ALIVE' if training_alive() else '○ not running'}")

    if not rows:
        print("no metrics yet — self-play of iteration 1 is probably still running.")
        log = os.path.join(run_dir, "progress.log")
        if os.path.exists(log):
            print("\nprogress.log tail:")
            with open(log) as f:
                for line in f.readlines()[-5:]:
                    print("  " + line.rstrip())
        return

    last = rows[-1]
    print(f"iteration : {last.get('iteration')} / (see config)   "
          f"elapsed {float(last.get('elapsed_s', 0)) / 3600:.2f} h")
    print(f"loss      : {last.get('loss')}   (policy {last.get('policy_loss')}  "
          f"value {last.get('value_loss')})")
    print(f"last iter : games {last.get('games')}  pos {last.get('positions')}  "
          f"buffer {last.get('buffer')}  iter_s {last.get('iter_s')}  "
          f"({last.get('games_per_s')} g/s)")

    trend = rows[-8:]
    print("loss trend: " + "  ".join(
        f"{r.get('iteration')}:{float(r.get('loss', 0)):.2f}" for r in trend))

    ckpt = os.path.join(run_dir, "checkpoints", "latest.pt")
    print(f"checkpoint: {ckpt if os.path.exists(ckpt) else 'none yet'}")


def main():
    ap = argparse.ArgumentParser(description="live training monitor")
    ap.add_argument("run_dir", nargs="?", default=None, help="defaults to most recent run")
    ap.add_argument("--watch", type=float, default=None, help="refresh every N seconds")
    ap.add_argument("--runs-dir", default="runs")
    args = ap.parse_args()

    run_dir = args.run_dir or latest_run(args.runs_dir)
    if not run_dir:
        print(f"no runs found under {args.runs_dir}/")
        return

    if args.watch:
        try:
            while True:
                os.system("clear")
                show(run_dir)
                time.sleep(args.watch)
        except KeyboardInterrupt:
            pass
    else:
        show(run_dir)


if __name__ == "__main__":
    main()
