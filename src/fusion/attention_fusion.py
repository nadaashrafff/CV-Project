"""Attention/gating fusion network (E10).

Learns per-sample weights to blend classical and deep features.
"""
import torch
import torch.nn as nn
from src.utils.seed import SEED


class AttentionFusion(nn.Module):
    def __init__(self, classical_dim: int = 252, deep_dim: int = 2048,
                 hidden_dim: int = 128, num_classes: int = 6, dropout: float = 0.3):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(classical_dim + deep_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
            nn.Softmax(dim=1),
        )
        self.classical_proj = nn.Linear(classical_dim, hidden_dim)
        self.deep_proj = nn.Linear(deep_dim, hidden_dim)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, classical: torch.Tensor, deep: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([classical, deep], dim=1)
        gates = self.gate(combined)               # (B, 2) — weights for each stream
        c_proj = self.classical_proj(classical)   # (B, H)
        d_proj = self.deep_proj(deep)             # (B, H)
        fused = gates[:, 0:1] * c_proj + gates[:, 1:2] * d_proj
        return self.classifier(fused)
