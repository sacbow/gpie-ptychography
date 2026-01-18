import numpy as np
import pytest
import importlib.util

from ptychography.backend.array import set_backend, xp
from ptychography.backend.rng import get_rng, normal


# ------------------------------------------------------------
# Backend availability
# ------------------------------------------------------------

cupy_spec = importlib.util.find_spec("cupy")
has_cupy = cupy_spec is not None

if has_cupy:
    import cupy as cp


backends = [np]
if has_cupy:
    backends.append(cp)


# ------------------------------------------------------------
# Array backend basics
# ------------------------------------------------------------

def test_array_backend_default_is_numpy():
    """Default array backend should be NumPy."""
    xp_backend = xp()
    assert xp_backend is np


@pytest.mark.parametrize("backend", backends)
def test_array_backend_set_and_get(backend):
    """Setting backend should affect xp() result."""
    set_backend(backend)
    assert xp() is backend


# ------------------------------------------------------------
# FFT utilities
# ------------------------------------------------------------

@pytest.mark.parametrize("backend", backends)
def test_fft_roundtrip_identity(backend):
    """fft2c followed by ifft2c should recover the original array."""
    set_backend(backend)

    from ptychography.backend.fft import fft2c, ifft2c

    xp_ = xp()
    x = xp_.random.randn(16, 16)

    y = ifft2c(fft2c(x))

    # Bring back to NumPy for comparison if needed
    if backend is not np:
        x = x.get()
        y = y.get()

    assert np.allclose(x, y, atol=1e-6)


@pytest.mark.parametrize("backend", backends)
def test_fft_preserves_shape_and_dtype(backend):
    """FFT utilities should preserve array shape and return complex dtype."""
    set_backend(backend)

    from ptychography.backend.fft import fft2c

    xp_ = xp()
    x = xp_.random.randn(8, 8).astype(xp_.float32)
    y = fft2c(x)

    assert y.shape == x.shape
    assert y.dtype.kind == "c"  # complex


# ------------------------------------------------------------
# RNG utilities
# ------------------------------------------------------------

@pytest.mark.parametrize("backend", backends)
def test_rng_reproducibility_with_seed(backend):
    """RNG should be reproducible when seed is fixed."""
    set_backend(backend)

    rng1 = get_rng(123)
    a = normal(rng = rng1, size=10)

    rng2 = get_rng(123)
    b = normal(rng = rng2, size=10)

    if backend is not np:
        a = a.get()
        b = b.get()

    assert np.allclose(a, b)


# ------------------------------------------------------------
# ArrayLike protocol
# ------------------------------------------------------------

@pytest.mark.parametrize("backend", backends)
def test_arraylike_protocol_compatibility(backend):
    """ArrayLike Protocol should accept numpy and cupy arrays."""
    set_backend(backend)

    from ptychography.utils.types import ArrayLike

    xp_ = xp()
    x = xp_.zeros((4, 4))

    # Runtime check: Protocol is structural
    assert hasattr(x, "shape")
    assert hasattr(x, "dtype")
