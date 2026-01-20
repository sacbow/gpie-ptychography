from typing import Tuple, Optional
import numpy as np


def _apply_jitter(
    y: float, x: float, jitter_um: float, rng: Optional[np.random.Generator] = None
) -> Tuple[float, float]:
    """Add Gaussian jitter (position noise) to (y, x)."""
    if jitter_um == 0.0:
        return y, x
    rng = rng or np.random.default_rng()
    dy, dx = rng.normal(0.0, jitter_um, size=2)
    return y + dy, x + dx