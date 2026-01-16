import numpy as np
import pytest

from ptychography.core.wave import Wave
from ptychography.core.propagator import Propagator
from ptychography.core.ops import Add, Multiply, Abs, Power, FFT2
from ptychography.core.shortcuts import fft2


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
# ndarray mixing should fail cleanly
# ------------------------------------------------------------

def test_wave_add_numpy_array_raises():
    w = Wave(label="w")

    arr = np.ones((4, 4))

    with pytest.raises(TypeError):
        _ = w + arr


def test_numpy_array_add_wave_raises():
    w = Wave(label="w")
    arr = np.ones((4, 4))

    with pytest.raises(TypeError):
        _ = arr + w


def test_wave_mul_scalar_not_supported():
    w = Wave(label="w")

    with pytest.raises(TypeError):
        _ = w * 2


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
