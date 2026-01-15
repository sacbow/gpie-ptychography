# tests/test_ptychography.py

import pytest
import numpy as np

from ptychography.backend.array import set_backend, xp
from ptychography.data.diffraction import DiffractionData
from ptychography.data.ptychography import Ptychography


def _make_dummy_diffraction(position=(0.0, 0.0), shape=(8, 8)):
    """Utility to create a dummy DiffractionData."""
    intensity = xp().ones(shape, dtype=xp().float32)
    indices = (slice(0, shape[0]), slice(0, shape[1]))
    return DiffractionData(
        position=position,
        intensity=intensity,
        indices=indices,
    )


def test_empty_ptychography():
    """Empty Ptychography dataset should be valid."""
    ptycho = Ptychography()

    assert len(ptycho) == 0
    ptycho.validate()  # should not raise


def test_append_and_len():
    """Appending diffraction data should increase length."""
    set_backend(np)

    ptycho = Ptychography()
    d = _make_dummy_diffraction(position=(1.0, 2.0))

    ptycho.append(d)

    assert len(ptycho) == 1
    assert ptycho[0] is d


def test_extend():
    """Extending with multiple diffraction items."""
    set_backend(np)

    ptycho = Ptychography()
    items = [
        _make_dummy_diffraction(position=(0.0, 0.0)),
        _make_dummy_diffraction(position=(1.0, 1.0)),
        _make_dummy_diffraction(position=(2.0, 2.0)),
    ]

    ptycho.extend(items)

    assert len(ptycho) == 3
    assert ptycho.positions == [(0.0, 0.0), (1.0, 1.0), (2.0, 2.0)]


def test_positions_indices_intensities_properties():
    """positions / indices / intensities accessors should work."""
    set_backend(np)

    ptycho = Ptychography()
    d1 = _make_dummy_diffraction(position=(0.0, 0.0))
    d2 = _make_dummy_diffraction(position=(1.0, 2.0))

    ptycho.extend([d1, d2])

    positions = ptycho.positions
    indices = ptycho.indices
    intensities = ptycho.intensities

    assert positions == [(0.0, 0.0), (1.0, 2.0)]
    assert indices[0] == d1.indices
    assert intensities[1].shape == (8, 8)


def test_validate_consistent_shapes():
    """validate() should pass when all diffraction shapes match."""
    set_backend(np)

    ptycho = Ptychography()
    ptycho.extend([
        _make_dummy_diffraction(shape=(8, 8)),
        _make_dummy_diffraction(shape=(8, 8)),
    ])

    ptycho.validate()  # should not raise


def test_validate_inconsistent_shapes_raises():
    """validate() should raise if diffraction shapes differ."""
    set_backend(np)

    ptycho = Ptychography()
    ptycho.extend([
        _make_dummy_diffraction(shape=(8, 8)),
        _make_dummy_diffraction(shape=(16, 16)),
    ])

    with pytest.raises(ValueError):
        ptycho.validate()


def test_summary_string():
    """summary() should return a readable summary string."""
    set_backend(np)

    ptycho = Ptychography(
        pixel_pitch=(1.0, 1.0),
        wavelength=1e-10,
        detector_distance=1.0,
    )

    ptycho.append(_make_dummy_diffraction())

    summary = ptycho.summary()

    assert "Ptychography Dataset" in summary
    assert "Number of diffraction patterns" in summary
    assert "Diffraction shape" in summary
    assert "Pixel pitch" in summary
    assert "Wavelength" in summary


@pytest.mark.skipif(
    pytest.importorskip("matplotlib", reason="matplotlib not installed") is None,
    reason="matplotlib not available",
)
def test_show_diffraction_runs_without_error():
    """show_diffraction() should run without GUI errors."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    set_backend(np)

    ptycho = Ptychography()
    ptycho.append(_make_dummy_diffraction())

    ax = ptycho.show_diffraction(index=0)
    assert ax is not None

    plt.close(ax.figure)


@pytest.mark.skipif(
    pytest.importorskip("matplotlib", reason="matplotlib not installed") is None,
    reason="matplotlib not available",
)
def test_show_positions_runs_without_error():
    """show_positions() should run without GUI errors."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    set_backend(np)

    ptycho = Ptychography()
    ptycho.extend([
        _make_dummy_diffraction(position=(0.0, 0.0)),
        _make_dummy_diffraction(position=(1.0, 2.0)),
        _make_dummy_diffraction(position=(2.0, 4.0)),
    ])

    ax = ptycho.show_positions()
    assert ax is not None

    plt.close(ax.figure)
