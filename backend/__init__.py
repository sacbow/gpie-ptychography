# backend/__init__.py

"""
Numerical backend utilities for gpie-ptychography.

This module provides thin abstractions over:
- array backends (NumPy / CuPy)
- random number generators
- FFT implementations (NumPy / CuPy / FFTW)

Design principles
-----------------
- Backend utilities are implementation details.
- Public APIs (algorithms, datasets) must use NumPy arrays.
- FFT backend abstraction is primarily intended for solver implementations.
"""

# Array backend
from .array import xp, set_backend, to_numpy

# RNG utilities
from .rng import get_rng, normal, poisson

# FFT utilities
from .fft import fft2c, ifft2c

__all__ = [
    # array
    "xp",
    "set_backend",
    "to_numpy",
    # rng
    "get_rng",
    "normal",
    "poisson",
    # fft
    "fft2c",
    "ifft2c",
]
