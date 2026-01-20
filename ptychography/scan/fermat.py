from .utils import _apply_jitter
import numpy as np
import math
from typing import Generator, Optional, Tuple

def fermat_spiral(
    step: float = 1.0,
    golden_angle_rad: float = 2.399967,  # ≈ 137.5 deg
    jitter: float = 0.0,
    rng: Optional[np.random.Generator] = None,
) -> Generator[Tuple[float, float], None, None]:
    """
    Generator that yields scan positions along a Fermat spiral in real space.

    Parameters
    ----------
    step : float
        Radial scaling factor.
    golden_angle_rad : float
        Angle between successive points in radians (default ≈ 137.5°).
    jitter : float, optional
        Standard deviation of Gaussian positional jitter. Default = 0 (no jitter).
    rng : np.random.Generator, optional
        Random number generator for reproducibility.

    Yields
    ------
    (y_um, x_um) : tuple of floats
        Real-space coordinates along Fermat spiral, with optional jitter.
    
    References
    ----------
    X. Huang et al., "Effects of overlap uniformness for ptychography,"
    Opt. Express 22(11), 12634–12644 (2014).
    """
    i = 0
    rng = rng or np.random.default_rng()

    while True:
        r = step * math.sqrt(i)
        theta = i * golden_angle_rad
        x = r * math.cos(theta)
        y = r * math.sin(theta)
        yield _apply_jitter(y, x, jitter, rng)
        i += 1