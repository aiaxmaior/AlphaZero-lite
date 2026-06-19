"""External benchmark: latest checkpoint vs Stockfish 16 across a skill ladder.

CPU-only (don't fight the GPU training). Stockfish is throttled to its weakest
useful settings (low skill, 50 ms/move). A small overnight net will likely lose
to even weak Stockfish — that's expected; this just calibrates *where* it stands.
"""
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # project root

import chess.engine# #               ⢀⣀⣤⣤⣤⣀⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀ ⠀⢀⣀⣤⣤⣤⣀⡀⠀⠀⠀⠀⠀⠀ ⠀⠀⠀    ⢀⣀⣤⣤ ⣤⣀⡀
import numpy as np   #              ⣰⠟⠉⠀⠀⠉⠙⠻⣶⣄⠀⠀⠀⠀⠀⠀⣰⠟⠉⠀ ⠀  ⠉⠙⠻⣶⣄⠀⠀⠀⠀⠀⠀⠀⠀⠀⣰⠟⠉⠀⠀    ⠙⣶⣄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
import torch         #              ⢏⠀⠈⠉⠲⣄⠀⠀⠈⢻⣦⠀⠀⠀⠀⠀⢏⠀   ⠈⠉⠲⣄⠀ ⠀⠈⢻⣦⠀⠀⠀⠀⠀ ⠀⢏⠀  ⠉⠲⣄    ⠈⢻⣦⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
                     #               ⠈⠂⠀⠀⠀⢹⡇⠀⠀⠀⠹⣷⡀⠀ ⠀⠈⠂⠀  ⠀⠀⢹⡇⠀ ⠀⠀⠹⣷⡀⠀⠀⠀⠀⠀⠀⠈   ⠀ ⢹⡇⠀⠀⠀ ⠹⣷⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀
                     #          ⣀⣠⣤⣀⣀⣀⣀⣀⣠⣿⠃⠀⠀⠀⠀⠈⠻⣦⣀⣀⣠⣤⣀⣀⣀⣀⣠⣿⠃⠀⠀ ⠀⠀⠈⠻⣦⣀⣀⣠⣤⣀⣀⣀⣀⣀⣠⣿⠃⠀⠀⠀⠀ ⠈⠻⣦⣀⠀⠀⠀⠀⠀⠀⠀
                     #     ⠀⠀   ⠀⠀⠀⠈⠉⠛⠛⠛⠉⠀⠀⠻⣦⣀⠀⠀⠀⠀      ⠛⠛⠛⠉⠀  ⠀⠀⠀⠀⠀⠀⠀⠀⠉⠉⠒⠒⠈⠉⠛⠛⠛⠉⠀⠀⠀⠀⠀⠀⠀⠀⠀   ⠉⠉⠒⠒⠀⠀⠀⠀
                     #               ⠻⣦⣀            ⠻⣦⣀                ⠻⣦⣀          ⠻⣦⣀
                     #                                       ⠻⣦⣀                             ⠻⣦⣀
from src.config import Config        #⠀⠀⠀⠀⠀⠀⢀⣀⣤⣤⣤⣀⡀⠀⠀⠀⢀⣀⣤⣤⣤⣀⡀⠀⠀⠀⢀⣀⣤⣤⣤⣀⡀⠀  ⢀⣀⣤⣤⣤⣀⡀⠀⠀⠀⢀⣀⣤⣤⣤⣀⡀⠀⠀⠀                              
from src.network import ChessNet     #      ⣰⠟⠉⠀⠀⠉⠙⠻⣶⣄⣰⠟⠉⠀⠀⠉⠙⠻⣶⣄⣰⠟⠉⠀⠀⠉⠙⠻⣶⣄⣰⠟⠉⠀⠀⠉⠙⠻⣶⣰⠟⠉⠀⠀⠉⠙⠻⣶
from src.evaluate import play_match  #              
                              #                                  ____________________________
                              #    ><((-o>                       |... Did someone say fish???|
                              #                                  |___________________________|
                              #    ><((-o>                      /
cfg = Config()                #                                /      /`·.¸                       
cfg.device = "cpu"            #                               /      /¸...¸`:·
torch.set_num_threads(6)      #                              |   ¸.·´  ¸   `·.¸.·´)                  ><((((-0 >                                                                         
cfg.mcts.simulations = 40     #                               -- : © ):´;      ¸  {  
GAMES = 8                     #                                  `·.¸ `·  ¸.·´\`·¸)
                              #                                      `\\´´\¸.·´ 
net = ChessNet(cfg.net).to("cpu")  #                                                                     ><((((-0 >                                                                         
net.load_state_dict(torch.load("runs/run_0/checkpoints/net_iter120.pt", map_location="cuda:0"))
net.eval()

sf = shutil.which("stockfish") or "/usr/games/stockfish"                                                            #    ><((((-o> (suckas)

for level in (0, 1, 3):
    engine = chess.engine.SimpleEngine.popen_uci(sf)
    engine.configure({"Skill Level": level})
    limit = chess.engine.Limit(time=0.05)
    rng = np.random.default_rng(100 + level)
    try:
        w, d, l = play_match(net, cfg, lambda b: engine.play(b, limit).move, GAMES, rng)
    finally:
        engine.quit()
    score = (w + 0.5 * d) / (w + d + l)
    print(f">>> latest vs Stockfish(skill {level}, 50ms):  +{w} ={d} -{l}   score {score:.0%}", flush=True)
