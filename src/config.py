"""Central configuration.

Everything tunable lives here so the rest of the code reads cleanly. The defaults
are chosen to *run* on a laptop and to *train well* on a single GPU. Where a value
trades quality for speed, the comment says which direction to push it.
"""
from dataclasses import dataclass, field


@dataclass
class NetConfig:
    channels: int = 128          # width of the residual tower (AlphaZero used 256)
    blocks: int = 6              # number of residual blocks (AlphaZero used 19/39)
    # The input has this many feature planes (see encoding.py: 12 piece planes
    # + 5 state planes). Kept here so the net and encoder never disagree.
    in_planes: int = 17


@dataclass
class MCTSConfig:
    simulations: int = 100       # rollouts per move. More = stronger + slower.
    c_puct: float = 1.5          # exploration constant in the PUCT formula
    dirichlet_alpha: float = 0.3 # root noise spreads exploration over moves
    dirichlet_eps: float = 0.25  # how much root noise to mix in (0 = none)
    # During the first N plies of a self-play game we sample moves in proportion
    # to visit counts (exploration); after that we play greedily (best move).
    temperature_moves: int = 20


@dataclass
class TrainConfig:
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 256
    epochs_per_iter: int = 1     # passes over the replay buffer each iteration
    replay_buffer_games: int = 2000   # keep positions from the last N games
    # Loss = policy_cross_entropy + value_weight * value_mse
    value_weight: float = 1.0


@dataclass
class SelfPlayConfig:
    games_per_iter: int = 50
    max_moves: int = 200         # adjudicate a draw if a game runs this long
    resign_threshold: float = -0.95   # resign if value drops below this (None to disable)
    # Number of self-play games run *in lockstep* so their leaf evaluations batch
    # into one GPU call per simulation. This is THE knob that feeds the GPU — set
    # it to ~64–128 on a 12 GB card. 1 disables batching (slow, batch-1 path).
    parallel_games: int = 64
    # Number of self-play worker PROCESSES. The MCTS tree search is pure-Python
    # and CPU-bound, so 1 process pins a single core. Set this near your core
    # count to parallelize across cores (see src/distributed_selfplay.py).
    # 1 = single-process (no multiprocessing).
    num_workers: int = 1


@dataclass
class Config:
    net: NetConfig = field(default_factory=NetConfig)
    mcts: MCTSConfig = field(default_factory=MCTSConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    selfplay: SelfPlayConfig = field(default_factory=SelfPlayConfig)

    device: str = "cuda"         # falls back to cpu automatically in train.py
    # Inference/training precision for self-play. "bf16" uses the Ada/Ampere
    # tensor cores (≈2x throughput, half VRAM); "fp32" disables autocast.
    precision: str = "bf16"
    checkpoint_dir: str = "checkpoints"
    seed: int = 0

    # --- run / logging ---
    iterations: int = 80         # self-play -> train cycles to run
    run_name: str = ""           # blank -> auto timestamped name (set in train.py)
    runs_dir: str = "runs"       # each run lives in runs/<run_name>/
    tensorboard: bool = True     # also log scalars to TensorBoard if installed


# A single shared default; scripts can construct their own and override fields.
DEFAULT = Config()


def load_config(path: str) -> Config:
    """Build a Config from a YAML file, overriding the dataclass defaults.

    YAML layout mirrors the dataclasses: top-level keys (device, precision,
    iterations, ...) plus nested `net:`, `mcts:`, `train:`, `selfplay:` blocks.
    Unknown keys are ignored so old configs still load.
    """
    import yaml
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    cfg = Config()
    for section in ("net", "mcts", "train", "selfplay"):
        sub = getattr(cfg, section)
        for k, v in (raw.get(section) or {}).items():
            if hasattr(sub, k):
                setattr(sub, k, v)
    for k in ("device", "precision", "checkpoint_dir", "seed",
              "iterations", "run_name", "runs_dir", "tensorboard"):
        if k in raw:
            setattr(cfg, k, raw[k])
    return cfg


def save_config(cfg: Config, path: str) -> None:
    """Dump the fully-resolved config to YAML (a record of what the run used)."""
    import dataclasses
    import yaml
    with open(path, "w") as f:
        yaml.safe_dump(dataclasses.asdict(cfg), f, sort_keys=False)
