from .utils import _apply_jitter
import numpy as np
from typing import Generator, Optional, Tuple

def raster_scan(
    stride: float = 1.0,
    jitter: float = 0.0,
    rng: Optional[np.random.Generator] = None,
) -> Generator[Tuple[float, float], None, None]:
    """
    Generator that yields scan positions in real space, starting from (0, 0),
    following a spiral raster pattern (Manhattan spiral).

    Parameters
    ----------
    stride_um : float
        Base spacing between scan positions.
    jitter : float, optional
        Standard deviation of Gaussian positional jitter. Default = 0 (no jitter).
    rng : np.random.Generator, optional
        Optional random number generator for reproducibility.

    Yields
    ------
    (y, x) : tuple of floats
        Real-space scan coordinates with optional jitter.
    """
    yield _apply_jitter(0.0, 0.0, jitter, rng)

    step = 1
    x, y = 0, 0

    while True:
        # Move right
        for _ in range(step):
            x += 1
            yield _apply_jitter(x * stride, y * stride, jitter, rng)

        # Move up
        for _ in range(step):
            y += 1
            yield _apply_jitter(x * stride, y * stride, jitter, rng)

        step += 1

        # Move left
        for _ in range(step):
            x -= 1
            yield _apply_jitter(x * stride, y * stride, jitter, rng)

        # Move down
        for _ in range(step):
            y -= 1
            yield _apply_jitter(x * stride, y * stride, jitter, rng)

        step += 1