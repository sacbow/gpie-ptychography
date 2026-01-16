# tests/test_wave_and_propagator.py

import numpy as np
import pytest

from ptychography.core.wave import Wave
from ptychography.core.propagator import Propagator


class DummyPropagator(Propagator):
    """
    Minimal propagator for testing forward graph behavior.
    Simply adds all inputs together.
    """

    def compute(self, *inputs):
        out = inputs[0]
        for x in inputs[1:]:
            out = out + x
        return out


# ---------------------------------------------------------------------
# Graph wiring tests
# ---------------------------------------------------------------------

def test_single_input_propagator_connection():
    """Single-input propagator should connect waves correctly."""
    w0 = Wave(label="input", generation=0)

    prop = DummyPropagator()
    w1 = prop @ w0

    # Parent / child relationship
    assert w1.parent is prop
    assert prop in w0.children
    assert prop.inputs == [w0]
    assert prop.output is w1

    # Generation rule
    assert prop.generation == 0
    assert w1.generation == 1


def test_multi_input_propagator_generation():
    """Generation should be max(input generations)."""
    w0 = Wave(label="w0", generation=0)
    w1 = Wave(label="w1", generation=3)

    prop = DummyPropagator()
    out = prop @ (w0, w1)

    assert prop.inputs == [w0, w1]
    assert prop.generation == 3
    assert out.generation == 4

    assert prop in w0.children
    assert prop in w1.children
    assert out.parent is prop


def test_wave_parent_set_once():
    """Wave should not accept multiple parents."""
    w0 = Wave(label="root")

    prop1 = DummyPropagator()
    w1 = prop1 @ w0

    prop2 = DummyPropagator()

    with pytest.raises(RuntimeError):
        w1.set_parent(prop2)


# ---------------------------------------------------------------------
# Data handling tests
# ---------------------------------------------------------------------

def test_wave_data_set_and_read():
    """Wave should store and expose ndarray-like data."""
    arr = np.ones((4, 4), dtype=np.float32)
    w = Wave(label="data", data=arr)

    assert w.is_set()
    assert w.shape == (4, 4)
    assert w.dtype == np.float32
    assert w.ndim == 2
    assert np.allclose(w.data, arr)


def test_wave_data_unset_raises():
    """Accessing data when unset should raise."""
    w = Wave(label="empty")

    assert not w.is_set()

    with pytest.raises(RuntimeError):
        _ = w.data


# ---------------------------------------------------------------------
# Forward execution tests
# ---------------------------------------------------------------------

def test_propagator_forward_sets_output_data():
    """forward() should read inputs and write to output wave."""
    a = np.ones((2, 2))
    b = np.full((2, 2), 2.0)

    w0 = Wave(label="a", data=a)
    w1 = Wave(label="b", data=b)

    prop = DummyPropagator()
    out = prop @ (w0, w1)

    prop.forward()
    result = out.data

    assert out.is_set()
    assert np.allclose(result, a + b)


# ---------------------------------------------------------------------
# Representation tests
# ---------------------------------------------------------------------

def test_wave_repr():
    """__repr__ should include label, generation, and shape."""
    w = Wave(label="test", generation=2)
    s = repr(w)

    assert "test" in s
    assert "gen=2" in s
    assert "shape" in s


def test_propagator_repr():
    """Propagator __repr__ should include name and generation."""
    prop = DummyPropagator(name="dummy")
    s = repr(prop)

    assert "dummy" in s
    assert "gen" in s
