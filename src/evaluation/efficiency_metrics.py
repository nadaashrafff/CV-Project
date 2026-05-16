"""Inference time and memory benchmarking utilities."""
import time
from pathlib import Path

import numpy as np


def measure_inference_time(predict_fn, X: np.ndarray, n_runs: int = 100) -> float:
    """Return mean inference time in milliseconds per sample."""
    _ = predict_fn(X[:1])  # warm-up
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        predict_fn(X[:1])
        times.append((time.perf_counter() - start) * 1000)
    return float(np.mean(times))


def get_model_size_mb(path: str) -> float:
    return Path(path).stat().st_size / 1e6
