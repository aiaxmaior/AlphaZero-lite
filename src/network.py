"""The neural network: one body, two heads.

Input : (batch, 17, 8, 8) board planes from encoding.encode_board
Output:
    policy_logits : (batch, 4672)  raw scores over the fixed action space
    value         : (batch,)       in [-1, 1], the expected game result for the
                                    side to move (+1 = we win, -1 = we lose)

This is deliberately small: a 6-block / 128-channel tower trains in hours on one
GPU and is plenty to beat low-level Stockfish. Scale `blocks`/`channels` in
config.py for more strength at the cost of speed.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import NetConfig
from .encoding import N_PLANES, ACTION_SIZE


class ResidualBlock(nn.Module):
    """Standard conv-bn-relu x2 with a skip connection."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        return F.relu(x + y)


class ChessNet(nn.Module):
    def __init__(self, cfg: NetConfig | None = None):
        super().__init__()
        cfg = cfg or NetConfig()
        c = cfg.channels

        # Body: a "stem" conv that lifts 17 planes to `c` channels, then a tower.
        self.stem = nn.Sequential(
            nn.Conv2d(N_PLANES, c, 3, padding=1, bias=False),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
        )
        self.tower = nn.Sequential(*[ResidualBlock(c) for _ in range(cfg.blocks)])

        # Policy head: 1x1 conv -> flatten -> linear to the action space.
        self.policy_conv = nn.Sequential(
            nn.Conv2d(c, 32, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU(inplace=True)
        )
        self.policy_fc = nn.Linear(32 * 8 * 8, ACTION_SIZE)

        # Value head: 1x1 conv -> linear -> scalar in [-1, 1].
        self.value_conv = nn.Sequential(
            nn.Conv2d(c, 8, 1, bias=False), nn.BatchNorm2d(8), nn.ReLU(inplace=True)
        )
        self.value_fc = nn.Sequential(
            nn.Linear(8 * 8 * 8, 128), nn.ReLU(inplace=True), nn.Linear(128, 1), nn.Tanh()
        )

    def forward(self, x):
        x = self.tower(self.stem(x))
        p = self.policy_fc(self.policy_conv(x).flatten(1))
        v = self.value_fc(self.value_conv(x).flatten(1)).squeeze(-1)
        return p, v

    # -- convenience for MCTS: evaluate a single position ------------------
    @torch.no_grad()
    def predict(self, planes: np.ndarray, mask: np.ndarray, device: str):
        """Return (move_priors over the *legal* mask, value) for one position.

        `planes` is (17,8,8), `mask` is the boolean legal-action mask. We zero out
        illegal moves *before* softmax so the priors are a proper distribution
        over legal moves only.
        """
        self.eval()
        x = torch.from_numpy(planes).unsqueeze(0).to(device)
        logits, value = self.forward(x)
        logits = logits.squeeze(0).cpu().numpy()

        logits[~mask] = -np.inf
        logits -= logits.max()                 # stabilise before exp
        priors = np.exp(logits)
        priors /= priors.sum()
        return priors, float(value.item())

    @torch.no_grad()
    def predict_batch(self, planes, masks, device, autocast_dtype=None):
        """Batched version of `predict` — the key to GPU utilisation.

        planes : (B, 17, 8, 8) float32   board encodings for B positions
        masks  : (B, 4672) bool          legal-move mask per position
        Returns (priors (B,4672) float32 over each row's legal mask, values (B,)).

        One forward pass for the whole batch; with B≈64 this finally keeps the
        GPU busy instead of running thousands of batch-1 calls. autocast_dtype
        (e.g. torch.bfloat16) runs the pass in low precision on CUDA.
        """
        self.eval()
        x = torch.from_numpy(planes).to(device)
        if autocast_dtype is not None and "cuda" in str(device):
            with torch.autocast(device_type="cuda", dtype=autocast_dtype):
                logits, value = self.forward(x)
        else:
            logits, value = self.forward(x)

        logits = logits.float().cpu().numpy()
        values = value.float().cpu().numpy()

        # Mask illegal moves to a huge negative, then row-wise softmax.
        logits = np.where(masks, logits, np.float32(-1e30))
        logits -= logits.max(axis=1, keepdims=True)
        np.exp(logits, out=logits)
        logits /= logits.sum(axis=1, keepdims=True)
        return logits, values


def policy_value_loss(logits, value, target_pi, target_z, value_weight: float):
    """AlphaZero loss: cross-entropy(policy) + mse(value).

    target_pi is a full (batch, 4672) distribution from MCTS visit counts;
    target_z is the game outcome from the perspective of the player to move.
    """
    log_probs = F.log_softmax(logits, dim=1)
    policy_loss = -(target_pi * log_probs).sum(dim=1).mean()
    value_loss = F.mse_loss(value, target_z)
    return policy_loss + value_weight * value_loss, policy_loss, value_loss
