import numpy as np
import pytest

from ptychography.backend.array import set_backend
from ptychography.core.wave import Wave
from ptychography.core.ops import Add, Multiply, Abs, Power, FFT2, Slice


# ---------------------------------------------------------------------
# Test utilities
# ---------------------------------------------------------------------

@pytest.fixture(autouse=True)
def numpy_backend():
    set_backend(np)


def make_wave(data, generation=0):
    return Wave(data=data, generation=generation)


def run_forward(out_wave):
    """Run forward on parent propagator and return output data."""
    prop = out_wave.parent
    prop.forward()
    return out_wave.data


# ---------------------------------------------------------------------
# Add
# ---------------------------------------------------------------------

def test_add_forward():
    a = make_wave(np.ones((4, 4)))
    b = make_wave(2 * np.ones((4, 4)))

    out = Add() @ (a, b)
    y = run_forward(out)

    assert np.allclose(y, 3.0)


def test_add_shape_mismatch_raises():
    a = make_wave(np.ones((4, 4)))
    b = make_wave(np.ones((3, 4)))

    out = Add() @ (a, b)
    with pytest.raises(ValueError):
        out.parent.forward()


def test_add_wrong_input_count_raises():
    a = make_wave(np.ones((4, 4)))

    out = Add() @ a
    with pytest.raises(RuntimeError):
        out.parent.forward()


# ---------------------------------------------------------------------
# Multiply
# ---------------------------------------------------------------------

def test_multiply_forward():
    a = make_wave(np.ones((4, 4)))
    b = make_wave(3 * np.ones((4, 4)))

    out = Multiply() @ (a, b)
    y = run_forward(out)

    assert np.allclose(y, 3.0)


def test_multiply_broadcastable():
    a = make_wave(np.ones((4, 4)))
    b = make_wave(np.ones((1, 4)))

    out = Multiply() @ (a, b)
    y = run_forward(out)

    assert y.shape == (4, 4)


def test_multiply_shape_mismatch_raises():
    a = make_wave(np.ones((4, 4)))
    b = make_wave(np.ones((2, 3)))

    out = Multiply() @ (a, b)
    with pytest.raises(ValueError):
        out.parent.forward()


# ---------------------------------------------------------------------
# Abs
# ---------------------------------------------------------------------

def test_abs_forward():
    x = make_wave(np.array([[-1.0, 2.0], [-3.0, 4.0]]))

    out = Abs() @ x
    y = run_forward(out)

    assert np.allclose(y, [[1.0, 2.0], [3.0, 4.0]])


def test_abs_wrong_input_count_raises():
    a = make_wave(np.ones((2, 2)))
    b = make_wave(np.ones((2, 2)))

    out = Abs() @ (a, b)
    with pytest.raises(RuntimeError):
        out.parent.forward()


# ---------------------------------------------------------------------
# Power
# ---------------------------------------------------------------------

def test_power_forward():
    x = make_wave(np.array([[2.0, 3.0]]))

    out = Power(2) @ x
    y = run_forward(out)

    assert np.allclose(y, [[4.0, 9.0]])


def test_power_wrong_input_count_raises():
    a = make_wave(np.ones((2, 2)))
    b = make_wave(np.ones((2, 2)))

    out = Power(2) @ (a, b)
    with pytest.raises(RuntimeError):
        out.parent.forward()


# ---------------------------------------------------------------------
# FFT2
# ---------------------------------------------------------------------

def test_fft2_forward_matches_numpy():
    x = np.random.randn(8, 8)
    w = make_wave(x)

    out = FFT2() @ w
    y = run_forward(out)

    ref = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(x), norm = "ortho"))
    assert np.allclose(y, ref)


def test_fft2_ndim_check():
    w = make_wave(np.ones((8,)))

    out = FFT2() @ w
    with pytest.raises(ValueError):
        out.parent.forward()


# ---------------------------------------------------------------------
# Slice
# ---------------------------------------------------------------------

def test_slice_forward():
    x = np.arange(16).reshape(4, 4)
    w = make_wave(x)

    indices = [
        (slice(0, 2), slice(0, 2)),
        (slice(2, 4), slice(2, 4)),
    ]

    out = Slice(indices) @ w
    y = run_forward(out)

    assert y.shape == (2, 2, 2)
    assert np.allclose(y[0], x[0:2, 0:2])
    assert np.allclose(y[1], x[2:4, 2:4])


def test_slice_wrong_input_count_raises():
    a = make_wave(np.ones((4, 4)))
    b = make_wave(np.ones((4, 4)))

    out = Slice([]) @ (a, b)
    with pytest.raises(RuntimeError):
        out.parent.forward()
