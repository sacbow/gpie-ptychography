import numpy as np
import pytest

from ptychography.data.context import PtychoContext
from ptychography.core.wave import Wave
from ptychography.core.decorator import ptycho_model
from ptychography.core.measurement import Measurement, PoissonMeasurement
from ptychography.core.shortcuts import get_slice, replicate, fft2
from ptychography.backend.array import set_backend


# ---------------------------------------------------------------------
# Optional CuPy availability
# ---------------------------------------------------------------------

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False


# ---------------------------------------------------------------------
# Dummy identity measurement (deterministic)
# ---------------------------------------------------------------------

class IdentityMeasurement(Measurement):
    """
    Measurement that returns the input array as-is.
    """
    def compute(self, x):
        return x


# ---------------------------------------------------------------------
# Backend fixture (numpy / cupy)
# ---------------------------------------------------------------------

@pytest.fixture(params=["numpy", "cupy"])
def backend(request):
    import numpy as np   # ← ここで必ず定義

    name = request.param

    if name == "numpy":
        set_backend(np)

    elif name == "cupy":
        try:
            import cupy as cp
        except ImportError:
            pytest.skip("CuPy is not available.")
        set_backend(cp)

    yield name

    # restore default backend
    set_backend(np)



# ---------------------------------------------------------------------
# Context & data fixtures
# ---------------------------------------------------------------------

@pytest.fixture
def simple_context():
    scan_positions = [
        (0.0, 0.0),
        (2.0, 0.0),
        (0.0, 2.0),
        (2.0, 2.0),
    ]
    return PtychoContext(
        scan_positions=scan_positions,
        object_shape=(8, 8),
        probe_shape=(4, 4),
        pixel_pitch=1.0,
    )


@pytest.fixture
def object_and_probe(simple_context):
    # IMPORTANT:
    # object/probe are injected as NumPy arrays;
    # Graph may move them to GPU internally if backend == "cupy"
    obj = np.ones(simple_context.object_shape, dtype=np.float32)
    probe = np.ones(simple_context.probe_shape, dtype=np.float32)
    return obj, probe


# ---------------------------------------------------------------------
# Test ①: Identity measurement (NumPy reference)
# ---------------------------------------------------------------------

def test_ptycho_identity_measurement_matches_numpy(
    backend, simple_context, object_and_probe
):
    ctx = simple_context
    obj, probe = object_and_probe

    @ptycho_model
    def model(ctx):
        object_wave = Wave(label="object")
        probe_wave = Wave(label="probe")

        obj_sliced = get_slice(object_wave, ctx)
        probe_rep = replicate(probe_wave, ctx)

        exit_wave = obj_sliced * probe_rep
        intensity = abs(fft2(exit_wave)) ** 2

        IdentityMeasurement() << intensity

    pmodel = model(ctx)
    pmodel.set_wave("object", obj)
    pmodel.set_wave("probe", probe)

    pmodel.forward()
    diff_items = pmodel.get_diffraction_data()

    assert len(diff_items) == ctx.n_scan

    # NumPy reference
    expected = []
    for (sy, sx) in ctx.slice_indices:
        patch = obj[sy, sx] * probe
        ref = np.abs(np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(patch), norm = "ortho"))) ** 2
        expected.append(ref.astype(np.float32, copy=False))
    expected = np.stack(expected, axis=0)

    for i, item in enumerate(diff_items):
        # DiffractionData.intensity must ALWAYS be NumPy ndarray
        assert isinstance(item.intensity, np.ndarray)

        np.testing.assert_allclose(
            item.intensity,
            expected[i],
            rtol=1e-6,
            atol=1e-6,
        )

        assert item.indices == ctx.slice_indices[i]
        assert item.subpixel_shift == ctx.subpixel_shifts[i]


# ---------------------------------------------------------------------
# Test ②: PoissonMeasurement (statistical, backend-agnostic)
# ---------------------------------------------------------------------

def test_ptycho_poisson_measurement_runs(
    backend, simple_context, object_and_probe
):
    ctx = simple_context
    obj, probe = object_and_probe

    @ptycho_model
    def model(ctx):
        object_wave = Wave(label="object")
        probe_wave = Wave(label="probe")

        obj_sliced = get_slice(object_wave, ctx)
        probe_rep = replicate(probe_wave, ctx)

        exit_wave = obj_sliced * probe_rep
        intensity = abs(fft2(exit_wave)) ** 2

        PoissonMeasurement() << intensity * 1000.0

    pmodel = model(ctx)
    pmodel.set_wave("object", obj)
    pmodel.set_wave("probe", probe)

    pmodel.forward()
    diff_items = pmodel.get_diffraction_data()

    assert len(diff_items) == ctx.n_scan

    for item in diff_items:
        intensity = item.intensity

        # DiffractionData must hold NumPy arrays
        assert isinstance(intensity, np.ndarray)

        assert intensity.shape == ctx.probe_shape
        assert np.all(intensity >= 0)
        assert np.issubdtype(intensity.dtype, np.integer)
