"""Dimensionality reduction wrappers for E5 (PCA), E6 (Autoencoder), E7 (feature selection)."""
import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from src.utils.seed import SEED


# ── E5: PCA ──────────────────────────────────────────────────────────────────

def fit_pca(X_train: np.ndarray, n_components: float = 0.95) -> PCA:
    pca = PCA(n_components=n_components, random_state=SEED)
    pca.fit(X_train)
    return pca


# ── E6: Autoencoder ──────────────────────────────────────────────────────────

class Autoencoder(nn.Module):
    def __init__(self, input_dim: int = 2300, latent_dim: int = 256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 1024), nn.ReLU(),
            nn.Linear(1024, 512),      nn.ReLU(),
            nn.Linear(512, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 512), nn.ReLU(),
            nn.Linear(512, 1024),       nn.ReLU(),
            nn.Linear(1024, input_dim),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        return self.decoder(z), z


# ── E7: Feature selection ─────────────────────────────────────────────────────

def select_top_features(
    X_train: np.ndarray, y_train: np.ndarray, top_k: int = 200
) -> tuple[np.ndarray, np.ndarray]:
    """Return importance scores and top-k feature indices."""
    rf = RandomForestClassifier(n_estimators=100, class_weight="balanced",
                                 n_jobs=-1, random_state=SEED)
    rf.fit(X_train, y_train)
    importances = rf.feature_importances_
    top_indices = np.argsort(importances)[::-1][:top_k]
    return importances, top_indices
