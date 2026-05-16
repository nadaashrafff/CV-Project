"""E6: Train autoencoder for dimensionality reduction of fused 2300-dim features."""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

from src.fusion.dimensionality_reduction import Autoencoder
from src.utils.seed import set_seed, SEED
from src.utils.logger import get_logger

logger = get_logger(__name__)


def train(
    X_train: np.ndarray,
    latent_dim: int = 256,
    epochs: int = 50,
    batch_size: int = 256,
    lr: float = 0.001,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Autoencoder:
    set_seed(SEED)
    model = Autoencoder(input_dim=X_train.shape[1], latent_dim=latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.MSELoss()

    X_tensor = torch.FloatTensor(X_train)
    loader = DataLoader(TensorDataset(X_tensor), batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for (batch,) in loader:
            batch = batch.to(device)
            recon, _ = model(batch)
            loss = criterion(recon, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch + 1) % 10 == 0:
            logger.info(f"Epoch {epoch+1}/{epochs} — loss: {total_loss/len(loader):.4f}")

    return model


def encode(model: Autoencoder, X: np.ndarray, device: str = "cpu") -> np.ndarray:
    model.train(False)  # set to inference mode
    with torch.no_grad():
        z = model.encoder(torch.FloatTensor(X).to(device))
    return z.cpu().numpy()
