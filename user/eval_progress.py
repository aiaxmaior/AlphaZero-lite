"""Strength eval: latest checkpoint vs Stockfish AND vs its younger selves.

Two independent signals:
  - vs Stockfish (external bar): where does it actually stand?
  - vs earlier checkpoints (internal): did training produce real strength,
    independent of the (flat) loss?
"""
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # project root

import chess.engine                                                                             #         ⠀⠀⠀⠀⢀⣀⣤⣤⣤⣀⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀   ⢀⣀⣤⣤⣤⣀⡀⠀⠀⠀⠀⠀⠀⠀⠀        ⢀⣀⣤⣤ ⣤⣀⡀
import numpy as np                                                                              #      ⠀⠀⠀⠀⠀⣰⠟⠉  ⠀⠀⠉⠙⠻⣶⣄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀ ⣰⠟⠉⠀⠀  ⠉⠙⠻⣶⣄⠀⠀⠀⠀⠀⠀⠀  ⠀⣰⠟⠉⠀⠀     ⠙⣶⣄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
import torch                                                                                    #⠀⠀⠀⠀⠀⠀     ⢏   ⠀⠈⠉⠲⣄⠀⠀⠈⢻⣦⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢏⠀ ⠈⠉⠲⣄ ⠀ ⠀⠈⢻⣦⠀⠀⠀⠀  ⠀ ⢏ ⠀  ⠉⠲⣄    ⠈⢻⣦⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀  
import subprocess                                                                               #          ⠀⠀⠀⠈⠂  ⠀⠀⠀⢹⡇⠀⠀⠀⠹⣷⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠂⠀⠀⠀ ⢹⡇ ⠀ ⠀⠀⠹⣷⡀⠀  ⠀⠀⠀⠈      ⢹⡇⠀⠀⠀ ⠹⣷⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀
import re                                                                                       #⠀⠀⠀      ⠀   ⣀⣀⣀⣀⣀⣠⣿⠃⠀⠀⠀⠀⠈⠻⣦⣀⠀      ⣠⣤⣀⣀⣀⣀⣠⣿⠃⠀⠀   ⠀⠀⠈⠻⣦⣀⣀⣠⣤⣀⣀⣀⣀⣀⣠⣿⠃⠀⠀ ⠀ ⠈⠻⣦⣀⠀⠀⠀⠀⠀⠀⠀
from src.config import Config, load_config                                                      #⠀⠀⠀        ⠀⠀⠀⠉⠛⠛⠛⠉⠀⠀⠻⣦⣀⠀⠀⠀⠀⠉⠉⠒⠒⠻⣦⠛⠛⠛⠉⠀⠀        ⠀⠀⠀⠀⠀⠀⠀⠉⠉⠒⠒⠈⠉⠛⠛⠛⠉⠀⠀⠀⠀⠀⠀⠀⠀⠀     ⠉⠉⠒⠒⠀⠀⠀⠀
from src.network import ChessNet                                                                #                       ⠻⣦⣀                ⠻⣦⣀                          ⠻⣦⣀
from src.evaluate import net_move, play_match                                                   #               ⠻⣦⣀⣦⣀                                  ⠻⣦⣀
                                                                                                #        ⠀⠀⠀⠀   ⠀⠀  ⢀⣀⣤⣤⣤⣀⡀⠀⠀⠀⢀⣀⣤⣤⣤⣀⡀⠀⠀⠀⢀⣀⣤⣤⣤⣀⡀⠀  ⢀⣀⣤⣤⣤⣀⡀ ⠀⠀⢀⣀⣤⣤⣤⣀⡀⠀⠀⠀                              
print("*"*20,"Alpha-Zero (Ajax-style) Evaluation","*"*20)                                       #                   ⣰⠟⠉⠀⠀⠉⠙⠻⣶⣄⣰⠟⠉⠀⠀⠉⠙⠻⣶⣄⣰⠟⠉⠀⠀⠉⠙⠻⣶⣄⣰⠟⠉⠀⠀⠉⠙⠻⣶⣰⠟⠉  ⠀⠀⠉⠙⠻⣶
print("Test run checkpoint performance against self or against other runs")
print("*"*76,"\n")
cfg = Config()
device_choice = input("Select the device to run the simulations on: \n 1. GPU (specify which GPU if multiple, i.e. 0,1 etc) \n 2. CPU \n Your answer: ").strip()
if not device_choice or device_choice == "1":                                                   #     ⠀                                        ⠀  ⠀         ⣀⣠⠤⢾⣞⣿⣿⣿⣶       WHALE HELLO THERE!
    if not device_choice:                                                                       #                   ⠀ ⠀⠀⠀⠀⠀                       ⠀⠀⠀⣀⡤⠔⠚ ⠉⢁⣤⣶⣾⣿⣟⠻⢇⣿     
        print("defaulting to GPU")                                                              #                   ⠀⠀  ⠀⠀  ⠀⠀⠀ ⠀⠀⠀⠀⠀⠀ ⠀⠀  ⠀⠀⣀⡤ ⠒⠋⠉⠀⠀⠀⣠⣶⣿⣿⡿⢋⡴⠟⣾⡾⠉
    # 1. Show available GPUs                                                                    #                    ⠀⠀  ⠀ ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀ ⠀   ⣀⡴⠚⠁ ⠀⠀⠀⠀⣠⣴⣾⣿⢟⡿⢋⡴⠛⣠⢾⠏⠁⠀
    try:                                                                                        #         ⢀⡶⡄   ⠀  ⠀⠀         ⠀ ⠀⠀⠀⠀⠀⠀⠀⢀⡴⠋⠀(O )⣤⢶⣶⣾⡟⢠⠋⢠⠏⡰⢻⠋⡠⢋⡞⠁⠀⠀⠀⠀
        result = subprocess.run(                                                                #      ⠀⣠⡿  ⡇⠀             ⠀⠀⠀  ⠀⠀⠀⠀⠀⢀⡴⠋⠀⢀⣰⣶⣿ ⣷⣇⣾⣿⢿⣤⠃⢠⢏⡞⣡⢃⣞⣡⠋⠀⠀⠀⠀⠀⠀
            ["nvidia-smi", "--query-gpu=index,name", "--format=csv"],                           # ⣤⣠⣴⠗⠛⠀  ⢠⡇  ⠀⠀⠀⠀⠀⠀               ⢀⡴⠋⠀⠀⣴⠋⠁⣠⣿⣿⣿⢟⣽⡿ ⢃⡴⢃⠞⣰⣿⣿⠟⠁⠀⠀⠀⠀⠀⠀⠀
            capture_output=True,                                                                # ⣾⡍⠉⠁⠀⠀⣠⡾⠁⠀  ⠀⠀⠀ ⠀⠀⠀⠀⠀       ⠀⠀⡴⠋⠀⠀⠀⣸⠁⠀⢠⣾⣿⣯⣕⣿⣯⠖⢉⡴⢋⣼⣽⣵⣯⠀⠀⠀⠀⠀⠀⠀⠀⠀
            text=True,                                                                          #⠈⡇     ⡾⠛         ⠀⠀⠀ ⠀⠀⠀ ⠀⠀⠀⠀⣠⠎⠀⠀⠀⢠⣼⡇⠀⠀⠀⠻⣯⢛⡵⠚⢁⡴⠊⣠⣾⣿⣿⣿⣇⠀⠀⠀⠀⠀⠀⠀⠀
            check=True                                                                          # ⣿⠉⠀        ⠀ ⠀⠀⠀⠀      ⠀⠀⠀⣀⣤⣾⠁⠀⠀⠀⢠⣿⣿⡇⠀⠀⠀⢡⣿⠏⣀⠔⢋⡠⣪⣿⣿⣿⠻⣿⣇⠀⠀⠀⠀⠀ ⠀⠀⠀⠀⠀
        )                                                                                       #  ⠓⠲                       ⣰⣿⣿⣿⠀⠀⠀    ⠀ ⣆⣿⡏⢁⣴⣯⡾⠋⣿⣿⣿⣿   ⡼
        print("\nAvailable GPUs:\n" + result.stdout)                                            #    ⠓⠲               ⠀⢀⣠⣶⣿⣯⡿⣿⡽⣷⢞⣽⠿     ⢸⡇⠀⠀⠀⠀⠀⠀⠀⠳⡆    ⢸⡇ 
    except FileNotFoundError:                                                                   #      ⠈⢳⡀⣾⣿⠓⠲⢤⣤⣤⠤⠔⣲⠟⠛⠋⠁⢀⣴⣴⣿⣿⡁⠀⠀⠀⢰⣻⡿⠀⠀⠳⡆⣧⠀⠀     ⣼⠇   ⢸⡇⠀⠀⠀⠀⠀⠀⠀
        print("\n[Warning] nvidia-smi not found. Defaulting to cpu")                            # ⠀ ⠀    ⠙⠿⠿⢾⣻⣴⣿⣿⡿⠿⠀⠀⠀⠀⣾⡟⠀⠀⠀⠀⠀⠀⠀⣇⠀  ⣼⡿⠀   ⠘⣿⠿⠂   ⢸⡇ c  ⡼
        cfg.device='cpu'                                                                        # ⠀⠀ ⠀    ⠈⠙⢿⣿⣶⣾⣿⣶⣶⣦⣤⣤⣴⣖⣶⣾⡿⠿⠛⣶⣿⡇⠀⠀⠀ ⢰⡟⠀⠀⠀ ⠀⣿     ⣿⠀⠀⠀⠀⡼⠀⠀⠀⠀⠀
                                                                                                #⠀ ⠀⠀ ⠀⠀⠀    ⠈⠙⠻⠿⢿⣻⣿⣿⣿⠛⣭⣴⣒⣒⣚⣛⣯⠋⣿⡇⠀ ⠀⢠⡞⠁⠀⠀⠀ ⡿    ⢠⡇⠀⠀⠀⡼⠀⠀⠀⠀⠀                                                                                                
    # 2. Capture the specific GPU index                                                         #                 ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠉⠉⠉⠉⠉⠉⠉⠀⠀⠀⠀⢻⣷⠀  ⢀⣿⠀⠀  ⣞   ⣀⠏⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
    gpu_idx = input("Specify GPU index (e.g., 0, 1) or press Enter for default (0): ").strip()  # ⠀⠀⠀                              ⠀⠀⠀⠀  ⠀⠸⣿⣀   ⣮⡷⠀  ⠀⠸⠿⠋⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
    if not gpu_idx:                                                                             #                                         ⠀⠀⠀⢿ ⣿ ⣻⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
        gpu_idx = "0"                                                                           # ⠀  ⠀⠀⠀                             ⠀⠀⠀⠀ ⠀⠀⠘⣿ ⣿⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
                                                                                                #⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀     ⠀⠀⠀⠀⠀⠀⠀⠀⠀     ⠀⠀ ⠀ ⠀⠀⠀⠘⣿⡇⠀⠀⠀⠀        
    # 3. Assign the correct PyTorch device string
    cfg.device = f"cuda:{gpu_idx}"

else:
    # 4. Handle CPU assignment and thread limiting
    cfg.device = "cpu"                  # run on CPU — don't fight the GPU-bound training
    torch.set_num_threads(6)

run_num = input("Enter the number of the run you want to test: ").strip()
if not run_num:
    run_num = len(os.listdir("runs")) - 1
    print(f"Defaulting to latest run: {run_num}")
comp_run_num = input("Enter the run to compare against (blank = compare against itself): ").strip()
if not comp_run_num:
    comp_run_num = run_num
    print("Self-comparison commencing...")
else:
    print(f"Comparing run_{run_num} against run_{comp_run_num}...")
GAMES = 8


def load_run(run, ckpt_name):
    """Load a checkpoint, building the net with THAT run's own architecture.
    Both the config and the checkpoint path come from the same run number, so
    they can never mismatch (e.g. a 7-block run vs a 6-block run)."""
    rc = load_config(f"runs/run_{run}/config.yaml")               # this run's net shape
    n = ChessNet(rc.net).to(cfg.device)
    n.load_state_dict(torch.load(f"runs/run_{run}/checkpoints/{ckpt_name}.pt",
                                 map_location=cfg.device))
    n.eval()
    return n

import csv
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(PROJECT_ROOT, "eval", "eval_comparisons")
os.makedirs(OUT_DIR, exist_ok=True)
results = []                               # one row per matchup, exported at the end


def report(tag, w, d, l):
    total = w + d + l
    score = (w + 0.5 * d) / total if total else 0.0
    print(f"\n>>> latest vs {tag}:  +{w} ={d} -{l}   score {score:.0%}\n", flush=True)
    results.append({
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "test_run": run_num, "comp_run": comp_run_num,
        "opponent": tag, "wins": w, "draws": d, "losses": l,
        "score": round(score, 3), "games": GAMES, "device": cfg.device,
    })

if os.path.isfile(f"runs/run_{run_num}/checkpoints/latest.pt"):
    print("latest checkpoint found")
    latest = load_run(run_num, "latest")        # pass the NAME, not a full path
else:
    print("no checkpoint found")
    latest = None



rng = np.random.default_rng(0)

# 1) external bar — Stockfish at its weakest (skill 1, 50 ms/move)
sf = shutil.which("stockfish")
if sf:
    engine = chess.engine.SimpleEngine.popen_uci(sf)
    engine.configure({"Skill Level": 0})
    limit = chess.engine.Limit(time=0.05)
    try:
        w, d, l = play_match(latest, cfg, lambda b: engine.play(b, limit).move, GAMES, rng)
        report("Stockfish(skill1, 50ms)", w, d, l)
    finally:
        engine.quit()
else:
    print(">>> Stockfish not on PATH — skipping external benchmark.\n", flush=True)

# 2) internal — vs its younger selves (self) or another run's checkpoints (compare)
ck_dir = f"runs/run_{comp_run_num}/checkpoints"

# the highest iteration that got checkpointed
max_iter = max(int(re.search(r"net_iter(\d+)", f).group(1))
               for f in os.listdir(ck_dir) if f.startswith("net_iter"))

# every 50th, 50..max, zero-padded to match the filenames
ckpts = [f"net_iter{i:03d}" for i in range(50, max_iter + 1, 50)]
extra_iter = input("Add a custom iteration for opponent (or self) comparison, if wanted. Eval will default compare latest to every 50th checkpoint")
if extra_iter:
    ckpts.append(f"net_iter{int(extra_iter):03d}")
for opp_name in ckpts:
    opp = load_run(comp_run_num, opp_name)      # comp_run_num == run_num for self-comparison
    w, d, l = play_match(latest, cfg, lambda b: net_move(opp, b, cfg), GAMES, rng)
    report(opp_name, w, d, l)

# --- export this eval's results to eval/eval_comparisons/ (one file per run) ---
if results:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(OUT_DIR, f"run{run_num}_vs_run{comp_run_num}_{stamp}.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    print(f"\nsaved {len(results)} comparisons -> {out_path}")

