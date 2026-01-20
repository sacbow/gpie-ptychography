import numpy as np
import pytest

from ptychography.core.wave import Wave
from ptychography.core.propagator import Propagator
from ptychography.core.ops import Add, Multiply, Abs, Power, FFT2
from ptychography.core.shortcuts import fft2
from ptychography.backend.array import set_backend



# ------------------------------------------------------------
# Basic binary ops
# ------------------------------------------------------------

def test_add_operator_builds_graph():
    w1 = Wave(label="w1", generation=0)
    w2 = Wave(label="w2", generation=2)

    out = w1 + w2

    assert isinstance(out.parent, Add)
    assert out.parent.inputs == [w1, w2]
    assert out.parent.output is out

    assert out.generation == 3
    assert out.parent in w1.children
    assert out.parent in w2.children


def test_multiply_operator_builds_graph():
    w1 = Wave(label="w1", generation=1)
    w2 = Wave(label="w2", generation=4)

    out = w1 * w2

    assert isinstance(out.parent, Multiply)
    assert out.parent.inputs == [w1, w2]
    assert out.generation == 5


# ------------------------------------------------------------
# Unary ops
# ------------------------------------------------------------

def test_abs_operator():
    w = Wave(label="w", generation=3)

    out = abs(w)

    assert isinstance(out.parent, Abs)
    assert out.parent.inputs == [w]
    assert out.generation == 4


def test_power_operator():
    w = Wave(label="w", generation=5)

    out = w ** 2

    assert isinstance(out.parent, Power)
    assert out.parent.power == 2
    assert out.parent.inputs == [w]
    assert out.generation == 6


# ------------------------------------------------------------
# FFT shortcut
# ------------------------------------------------------------

def test_fft2_shortcut():
    w = Wave(label="input", generation=7)

    out = fft2(w)

    assert isinstance(out.parent, FFT2)
    assert out.parent.inputs == [w]
    assert out.generation == 8



# ------------------------------------------------------------
# Chained expressions
# ------------------------------------------------------------

def test_chained_expression_generations():
    w1 = Wave(label="w1", generation=0)
    w2 = Wave(label="w2", generation=1)

    out = abs(fft2(w1 + w2) ** 2)

    # Check final wave
    assert isinstance(out, Wave)
    assert out.generation == 5

    # Walk parents
    p_abs = out.parent
    p_pow = p_abs.inputs[0].parent
    p_fft = p_pow.inputs[0].parent
    p_add = p_fft.inputs[0].parent

    assert isinstance(p_add, Add)
    assert isinstance(p_fft, FFT2)
    assert isinstance(p_pow, Power)
    assert isinstance(p_abs, Abs)

# ------------------------------------------------------------
# Wave <-> scalar / ndarray binary ops
# ------------------------------------------------------------

def test_add_wave_and_scalar():
    w = Wave(label="w", generation=3)

    out = w + 2.0

    assert isinstance(out.parent, Add)
    lhs, rhs = out.parent.inputs

    assert lhs is w
    assert isinstance(rhs, Wave)
    assert rhs.generation == 0
    np.testing.assert_allclose(rhs.numpy(), 2.0)

    assert out.generation == 4


def test_add_scalar_and_wave():
    w = Wave(label="w", generation=3)

    out = 2.0 + w

    assert isinstance(out.parent, Add)
    lhs, rhs = out.parent.inputs

    assert rhs is w
    assert isinstance(lhs, Wave)
    assert lhs.generation == 0
    np.testing.assert_allclose(lhs.numpy(), 2.0)

    assert out.generation == 4


def test_multiply_wave_and_scalar():
    w = Wave(label="w", generation=5)

    out = w * 10.0

    assert isinstance(out.parent, Multiply)
    lhs, rhs = out.parent.inputs

    assert lhs is w
    assert isinstance(rhs, Wave)
    assert rhs.generation == 0
    np.testing.assert_allclose(rhs.numpy(), 10.0)

    assert out.generation == 6


def test_multiply_scalar_and_wave():
    w = Wave(label="w", generation=5)

    out = 10.0 * w

    assert isinstance(out.parent, Multiply)
    lhs, rhs = out.parent.inputs

    assert rhs is w
    assert isinstance(lhs, Wave)
    assert lhs.generation == 0
    np.testing.assert_allclose(lhs.numpy(), 10.0)

    assert out.generation == 6


def test_add_wave_and_ndarray():
    w = Wave(label="w", generation=2)
    arr = np.ones((4, 4), dtype=np.float32)

    out = w + arr

    assert isinstance(out.parent, Add)
    lhs, rhs = out.parent.inputs

    assert lhs is w
    assert isinstance(rhs, Wave)
    assert rhs.generation == 0
    np.testing.assert_allclose(rhs.numpy(), arr)

    assert out.generation == 3


# ------------------------------------------------------------
# Subtract / Divide (once implemented)
# ------------------------------------------------------------

def test_subtract_wave_and_scalar():
    from ptychography.core.ops import Subtract

    w = Wave(label="w", generation=4)

    out = w - 1.5

    assert isinstance(out.parent, Subtract)
    lhs, rhs = out.parent.inputs

    assert lhs is w
    assert isinstance(rhs, Wave)
    np.testing.assert_allclose(rhs.numpy(), 1.5)

    assert out.generation == 5


def test_divide_wave_and_scalar():
    from ptychography.core.ops import Divide

    w = Wave(label="w", generation=6)

    out = w / 2.0

    assert isinstance(out.parent, Divide)
    lhs, rhs = out.parent.inputs

    assert lhs is w
    assert isinstance(rhs, Wave)
    np.testing.assert_allclose(rhs.numpy(), 2.0)

    assert out.generation == 7


def test_divide_scalar_and_wave():
    from ptychography.core.ops import Divide

    w = Wave(label="w", generation=6)

    out = 2.0 / w

    assert isinstance(out.parent, Divide)
    lhs, rhs = out.parent.inputs

    assert rhs is w
    assert isinstance(lhs, Wave)
    np.testing.assert_allclose(lhs.numpy(), 2.0)

    assert out.generation == 7
