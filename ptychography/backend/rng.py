# ptychography/backend/rng.py

import numpy as np

_rng = None

def set_seed(seed: int):
    global _rng
    _rng = np.random.default_rng(seed)

def get_rng():
    global _rng
    if _rng is None:
        _rng = np.random.default_rng()
    return _rng
