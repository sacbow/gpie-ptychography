import numpy as np
import pytest

from ptychography.data.context import PtychoContext
from ptychography.core.wave import Wave
from ptychography.core.decorator import ptycho_model
from ptychography.core.measurement import Measurement
from ptychography.core.shortcuts import get_slice, replicate
from ptychography.backend.array import xp


# ------------------------------------------------------------
# Dummy measurement for testing
# ------------------------------------------------------------

class IdentityMeasurement(Measurement):
    """
    Measurement that simply passes through the input array.
    Useful for testing forward graph execution.
    """

    def compute(self, x):
        return x


# ------------------------------------------------------------
# End-to-end forward simulation test
# ------------------------------------------------------------

def test_simple_forward_simulation():
    """
    Test that:
      - context-dependent propagators work correctly
      - graph builds and executes end-to-end
      - measurement receives expected array shape
    """

    # ----------------------------
    # Define context
    # ----------------------------

    scan_positions = [
        (0.0, 0.0),
        (1.0, 0.0),
        (0.0, 1.0),
    ]

    ctx = PtychoContext(
        scan_positions=scan_positions,
        object_shape=(8, 8),
        probe_shape=(4, 4),
        pixel_pitch=1.0,
    )

    n_scan = ctx.n_scan

    # ----------------------------
    # Define forward model
    # ----------------------------

    @ptycho_model
    def simple_model(ctx):
        obj = Wave(label="object")
        probe = Wave(label="probe")

        obj_sliced = get_slice(obj, ctx)
        probe_rep = replicate(probe, ctx)

        exit_wave = obj_sliced * probe_rep

        IdentityMeasurement() << exit_wave

    model = simple_model(ctx)

    # ----------------------------
    # Inject concrete data
    # ----------------------------

    backend = xp()

    object_array = backend.ones(ctx.object_shape, dtype=np.float32)
    probe_array = backend.ones(ctx.probe_shape, dtype=np.float32)

    model.set_wave("object", object_array)
    model.set_wave("probe", probe_array)

    # ----------------------------
    # Execute forward simulation
    # ----------------------------

    model.forward()

    measurements = model.get_measurements()

    assert len(measurements) == 1

    data = measurements[0]

    # ----------------------------
    # Validate output shape
    # ----------------------------

    assert data.shape == (
        n_scan,
        ctx.probe_shape[0],
        ctx.probe_shape[1],
    )

    # ----------------------------
    # Validate numerical values
    # ----------------------------

    # object and probe are all ones, so exit_wave should be all ones
    np.testing.assert_allclose(
        data,
        np.ones(data.shape),
    )
