# tests/test_diffraction_data.py

import pytest
import numpy as np

from backend.array import set_backend, xp
from ptychography.data.diffraction import DiffractionData


def test_diffraction_data_attributes():
    """Basic attribute storage and access."""
    set_backend(np)

    intensity = xp().ones((16, 16), dtype=xp().float32)
    indices = (slice(0, 16), slice(0, 16))

    data = DiffractionData(
        position=(10.0, 20.0),
        intensity=intensity,
        indices=indices,
        meta={"id": 1},
    )

    assert data.position == (10.0, 20.0)
    assert data.intensity.shape == (16, 16)
    assert data.indices == indices
    assert data.meta["id"] == 1


def test_validate_accepts_float_intensity():
    """validate() should accept floating-point 2D intensity arrays."""
    set_backend(np)

    intensity = xp().random.rand(8, 8).astype(xp().float64)
    data = DiffractionData(position=(0.0, 0.0), intensity=intensity)

    # Should not raise
    data.validate()


def test_validate_rejects_non_2d_intensity():
    """validate() should reject non-2D intensity arrays."""
    set_backend(np)

    intensity = xp().ones((4, 4, 4), dtype=xp().float32)
    data = DiffractionData(position=(0.0, 0.0), intensity=intensity)

    with pytest.raises(ValueError):
        data.validate()


def test_validate_rejects_non_float_intensity():
    """validate() should reject non-floating intensity arrays."""
    set_backend(np)

    intensity = xp().ones((4, 4), dtype=xp().int32)
    data = DiffractionData(position=(0.0, 0.0), intensity=intensity)

    with pytest.raises(TypeError):
        data.validate()


def test_shape_property():
    """shape property should reflect intensity shape."""
    set_backend(np)

    intensity = xp().zeros((32, 24), dtype=xp().float32)
    data = DiffractionData(position=(0.0, 0.0), intensity=intensity)

    assert data.shape == (32, 24)


def test_amplitude_computation():
    """Amplitude should be sqrt(intensity)."""
    set_backend(np)

    intensity = xp().ones((8, 8), dtype=xp().float32) * 4.0
    data = DiffractionData(position=(0.0, 0.0), intensity=intensity)

    amp = data.amplitude()

    assert amp.shape == (8, 8)
    assert np.allclose(amp, 2.0)


def test_amplitude_with_eps():
    """Amplitude computation should support epsilon offset."""
    set_backend(np)

    intensity = xp().zeros((4, 4), dtype=xp().float32)
    data = DiffractionData(position=(0.0, 0.0), intensity=intensity)

    amp = data.amplitude(eps=1e-6)

    assert np.all(amp > 0.0)


def test_with_indices_creates_copy():
    """with_indices() should return a new DiffractionData with updated indices."""
    set_backend(np)

    intensity = xp().ones((16, 16), dtype=xp().float32)
    data = DiffractionData(position=(0.0, 0.0), intensity=intensity)

    new_indices = (slice(4, 12), slice(4, 12))
    data2 = data.with_indices(new_indices)

    assert data.indices is None
    assert data2.indices == new_indices
    assert data2.intensity is data.intensity  # shallow copy
    assert data2.position == data.position


def test_summary_string():
    """Summary string should include position, shape, and index status."""
    set_backend(np)

    intensity = xp().zeros((4, 5), dtype=xp().float32)
    data = DiffractionData(
        position=(1.5, -2.0),
        intensity=intensity,
    )

    summary = data.summary()

    assert "position=(1.5, -2.0)" in summary
    assert "shape=(4, 5)" in summary
    assert "has_indices=False" in summary


def test_to_numpy_returns_numpy_array():
    """to_numpy() should always return a NumPy ndarray."""
    set_backend(np)

    intensity = xp().ones((8, 8), dtype=xp().float32)
    data = DiffractionData(position=(0.0, 0.0), intensity=intensity)

    arr = data.to_numpy()

    assert isinstance(arr, np.ndarray)
    assert arr.shape == (8, 8)


@pytest.mark.skipif(
    pytest.importorskip("matplotlib", reason="matplotlib not installed") is None,
    reason="matplotlib not available",
)
def test_show_runs_without_error():
    """Ensure DiffractionData.show() runs without GUI errors."""
    import matplotlib
    matplotlib.use("Agg")  # headless backend for CI
    import matplotlib.pyplot as plt

    set_backend(np)

    intensity = xp().random.rand(8, 8).astype(xp().float32)
    data = DiffractionData(position=(0.0, 0.0), intensity=intensity)

    ax = data.show()
    assert ax is not None

    plt.close(ax.figure)
