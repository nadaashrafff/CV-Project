import random
import numpy as np

SEED = 42

try:
    import torch as _torch
    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if _HAS_TORCH:
        _torch.manual_seed(seed)
        if _torch.cuda.is_available():
            _torch.cuda.manual_seed_all(seed)
