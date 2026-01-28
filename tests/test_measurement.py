# tests/test_measurement.py

import numpy as np
import pytest

from ptychography.core.wave import Wave
from ptychography.core.measurement import Measurement, PoissonMeasurement
from backend.array import set_backend


# ---------------------------------------------------------------------
# Base Measurement behavior
# ---------------------------------------------------------------------

class DummyMeasurement(Measurement):
    """Minimal concrete Measurement for testing base class behavior."""

    def compute(self, x):
        return x


def test_measurement_connects_wave():
    """Measurement << Wave should register input correctly."""
    w = Wave(label="input")
    meas = DummyMeasurement(name="dummy")

    meas << w

    assert meas.input is w
    assert "connected" in repr(meas)


def test_measurement_double_connect_raises():
    """Measurement should not accept multiple input waves."""
    w1 = Wave(label="w1")
    w2 = Wave(label="w2")
    meas = DummyMeasurement()

    meas << w1
    with pytest.raises(RuntimeError):
        meas << w2


def test_measurement_invalid_input_type_raises():
    """Only Wave objects can be connected."""
    meas = DummyMeasurement()

    with pytest.raises(TypeError):
        meas << np.ones((4, 4))


def test_measurement_forward_requires_input():
    """Calling forward without input should raise."""
    meas = DummyMeasurement()

    with pytest.raises(RuntimeError):
        meas.forward()


def test_measurement_forward_stores_data():
    """Measurement.forward() should call compute and store result."""
    set_backend(np)

    arr = np.ones((4, 4), dtype=np.float32)
    w = Wave(label="w", data=arr)

    meas = DummyMeasurement()
    meas << w

    out = meas.forward()

    assert meas.data is out
    assert np.allclose(out, arr)


# ---------------------------------------------------------------------
# PoissonMeasurement behavior
# ---------------------------------------------------------------------

def test_poisson_measurement_basic():
    """PoissonMeasurement should return non-negative integer array."""
    set_backend(np)

    x = np.full((8, 8), 5.0, dtype=np.float32)
    w = Wave(label="intensity", data=x)

    meas = PoissonMeasurement()
    meas << w

    out = meas.forward()

    assert out.shape == x.shape
    assert np.issubdtype(out.dtype, np.integer)
    assert np.all(out >= 0)


def test_poisson_measurement_rejects_non_float():
    """PoissonMeasurement should reject non-floating input."""
    set_backend(np)

    x = np.ones((4, 4), dtype=np.int32)
    w = Wave(label="bad", data=x)

    meas = PoissonMeasurement()
    meas << w

    with pytest.raises(RuntimeError):
        meas.forward()


def test_poisson_measurement_rejects_negative_values():
    """PoissonMeasurement should reject negative input."""
    set_backend(np)

    x = np.array([[1.0, -1.0]], dtype=np.float32)
    w = Wave(label="bad", data=x)

    meas = PoissonMeasurement()
    meas << w

    with pytest.raises(RuntimeError):
        meas.forward()


def test_poisson_measurement_repr():
    """__repr__ should reflect connection state."""
    meas = PoissonMeasurement()

    assert "unconnected" in repr(meas)

    w = Wave(label="w", data=np.ones((2, 2), dtype=np.float32))
    meas << w

    assert "connected" in repr(meas)
