import pytest
import numpy as np

from backend.array import xp
from ptychography.optics.aperture import make_aperture
from ptychography.optics.probe import (
    make_smooth_random_phase,
    generate_probe,
)


# ------------------------------------------------------------
# Aperture tests
# ------------------------------------------------------------

@pytest.mark.parametrize("kind", ["circular", "square"])
def test_make_aperture_basic_properties(kind):
    shape = (64, 64)
    pixel_size = 1.0
    aperture_radius = 10.0

    mask = make_aperture(
        shape=shape,
        pixel_size=pixel_size,
        aperture_radius=aperture_radius,
        kind=kind,
        smooth_edge_sigma=0.05,
    )

    assert mask.shape == shape
    assert mask.dtype.kind in ("f",)  # float
    assert xp().all(mask >= 0.0)
    assert xp().all(mask <= 1.0)


def test_make_aperture_invalid_kind():
    with pytest.raises(ValueError):
        make_aperture(
            shape=(32, 32),
            pixel_size=1.0,
            aperture_radius=5.0,
            kind="triangle",  # invalid
        )


# ------------------------------------------------------------
# Random phase tests
# ------------------------------------------------------------

def test_make_smooth_random_phase_basic():
    shape = (64, 64)
    rng = np.random.default_rng(0)

    phase = make_smooth_random_phase(
        shape=shape,
        cutoff_radius=0.1,
        rng=rng,
    )

    assert phase.shape == shape
    assert phase.dtype.kind == "f"
    assert xp().all(phase >= 0.0)
    assert xp().all(phase < 2 * xp().pi)


def test_make_smooth_random_phase_reproducibility():
    shape = (32, 32)

    rng1 = np.random.default_rng(123)
    rng2 = np.random.default_rng(123)

    phase1 = make_smooth_random_phase(shape, rng=rng1)
    phase2 = make_smooth_random_phase(shape, rng=rng2)

    xp().testing.assert_allclose(phase1, phase2)


# ------------------------------------------------------------
# Probe generation tests
# ------------------------------------------------------------

@pytest.mark.parametrize("space", ["real", "fourier"])
@pytest.mark.parametrize("kind", ["circular", "square"])
def test_generate_probe_basic(space, kind):
    shape = (64, 64)

    probe = generate_probe(
        shape=shape,
        pixel_size=1.0,
        kind=kind,
        space=space,
    )

    assert probe.shape == shape
    assert probe.dtype.kind == "c"  # complex
    assert xp().isfinite(probe).all()

    # Normalization: max amplitude == 1
    max_amp = xp().max(xp().abs(probe))
    xp().testing.assert_allclose(max_amp, 1.0, rtol=1e-6)


def test_generate_probe_with_random_phase_reproducibility():
    shape = (64, 64)

    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(42)

    probe1 = generate_probe(
        shape=shape,
        random_phase=True,
        rng=rng1,
    )

    probe2 = generate_probe(
        shape=shape,
        random_phase=True,
        rng=rng2,
    )

    xp().testing.assert_allclose(probe1, probe2)


def test_generate_probe_invalid_space():
    with pytest.raises(ValueError):
        generate_probe(
            shape=(32, 32),
            space="kspace",  # invalid
        )


# ------------------------------------------------------------
# Integration-style sanity check
# ------------------------------------------------------------

def test_probe_nontrivial_structure():
    """
    Sanity check that probe is not identically zero
    and has spatial structure.
    """
    shape = (64, 64)
    probe = generate_probe(
        shape=shape,
        random_phase=True,
        rng=np.random.default_rng(0),
    )

    amp = xp().abs(probe)

    assert xp().max(amp) > 0.0
    assert xp().std(amp) > 0.0
