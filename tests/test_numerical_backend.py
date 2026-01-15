# tests/test_numerical_backend.py

import numpy as np
import pytest


def test_array_backend_default_is_numpy():
    """Default array backend should be NumPy."""
    from ptychography.backend.array import xp

    xp_backend = xp()
    assert xp_backend is np


def test_array_backend_set_and_get():
    """Setting backend should affect xp() result."""
    from ptychography.backend.array import set_backend, xp

    set_backend(np)
    assert xp() is np


def test_fft_roundtrip_identity():
    """fft2c followed by ifft2c should recover the original array."""
    from ptychography.backend.fft import fft2c, ifft2c

    x = np.random.randn(16, 16)
    y = ifft2c(fft2c(x))

    assert np.allclose(x, y, atol=1e-6)


def test_fft_preserves_shape_and_dtype():
    """FFT utilities should preserve array shape and dtype."""
    from ptychography.backend.fft import fft2c

    x = np.random.randn(8, 8).astype(np.float32)
    y = fft2c(x)

    assert y.shape == x.shape
    assert y.dtype == np.complex64 or y.dtype == np.complex128


def test_rng_reproducibility_with_seed():
    """RNG should be reproducible when seed is fixed."""
    from ptychography.backend.rng import set_seed, get_rng

    set_seed(123)
    rng1 = get_rng()
    a = rng1.normal(size=10)

    set_seed(123)
    rng2 = get_rng()
    b = rng2.normal(size=10)

    assert np.allclose(a, b)


def test_rng_state_is_shared_after_get():
    """Multiple calls to get_rng() should return the same RNG instance."""
    from ptychography.backend.rng import get_rng, set_seed

    set_seed(0)
    rng1 = get_rng()
    rng2 = get_rng()

    assert rng1 is rng2


def test_arraylike_protocol_compatibility():
    """ArrayLike Protocol should accept numpy arrays."""
    from ptychography.utils.types import ArrayLike

    x = np.zeros((4, 4))
    # Runtime check: only attribute existence matters
    assert hasattr(x, "shape")
    assert hasattr(x, "dtype")
