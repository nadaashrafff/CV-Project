"""E10: Train attention/gating fusion network."""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

from src.fusion.attention_fusion import AttentionFusion
from src.utils.seed import set_seed, SEED
from src.utils.logger import get_logger

logger = get_logger(__name__)


def train(
    X_classical_train: np.ndarray,
    X_deep_train: np.ndarray,
    y_train: np.ndarray,
    epochs: int = 50,
    batch_size: int = 128,
    lr: float = 0.001,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> AttentionFusion:
    set_seed(SEED)
    model = AttentionFusion(
        classical_dim=X_classical_train.shape[1],
        deep_dim=X_deep_train.shape[1],
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    dataset = TensorDataset(
        torch.FloatTensor(X_classical_train),
        torch.FloatTensor(X_deep_train),
        torch.LongTensor(y_train),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for xc, xd, yt in loader:
            xc, xd, yt = xc.to(device), xd.to(device), yt.to(device)
            logits = model(xc, xd)
            loss = criterion(logits, yt)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch + 1) % 10 == 0:
            logger.info(f"Epoch {epoch+1}/{epochs} — loss: {total_loss/len(loader):.4f}")

    return model
