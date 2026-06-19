import time, numpy as np, torch
from src.config import Config
from src.network import ChessNet
from src.selfplay import generate_games_parallel

cfg = Config()
# device and precision

cfg.device='cuda0'
cfg.precision='bf16'

# default-ish net so the per-eval cost is realistic
cfg.net.blocks=6
cfg.net.channels=128
cfg.mcts.simulations=800
cfg.selfplay.parallel_games=8
cfg.selfplay.max_moves=40
torch.manual_seed(0); np.random.seed(0)
net = ChessNet(cfg.net)

t=time.time()
games=list(generate_games_parallel(net, cfg, n_games=8))
dt=time.time()-t
moves=sum(len(g) for g in games)
print(f'net: 128ch/6blk  sims/move: {cfg.mcts.simulations}  parallel: {cfg.selfplay.parallel_games}  (CPU, fp32)')
print(f'{len(games)} games, {moves} total moves, {dt:.1f}s')
print(f'-> {dt/len(games):.2f} s/game,  {dt/moves*1000:.0f} ms/move,  {dt/moves/cfg.mcts.simulations*1000:.2f} ms/simulation')