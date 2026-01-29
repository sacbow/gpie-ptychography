from __future__ import annotations

from typing import Sequence, Tuple, Optional, Any
import numpy as np

from gpie import model, GaussianPrior, fft2, AmplitudeMeasurement


@model
def ptychography_graph_known_probe(
    obj_shape: Tuple[int, int],
    prb: np.ndarray,
    indices: Sequence[Tuple[slice, slice]],
    noise: float,
    *,
    dtype: Any = np.complex64,
    meas_label: str = "meas",
    obj_label: str = "object",
    damping = "auto"
) -> None:
    """
    Ptychography factor graph with known probe.

    Notes
    -----
    - `obj_label` is attached to the variable node created by the Prior (Wave label).
    - Measurement data must be injected later via graph.get_factor(meas_label).set_observed(...).
    """
    # Prior over object
    obj = ~GaussianPrior(event_shape=obj_shape, label=obj_label, dtype=dtype)

    # Extract object patches at scan positions
    patches = obj.extract_patches(indices)

    # Exit waves and measurement
    exit_waves = prb * patches
    AmplitudeMeasurement(var=noise, label=meas_label, damping = damping) << fft2(exit_waves)
    return


@model
def ptychography_graph_unknown_probe(
    obj_shape: Tuple[int, int],
    prb_shape: Tuple[int, int],
    indices: Sequence[Tuple[slice, slice]],
    noise: float,
    *,
    dtype: Any = np.complex64,
    meas_label: str = "meas",
    obj_label: str = "object",
    prb_label: str = "probe",
    damping = "auto"
) -> None:
    """
    Ptychography factor graph with unknown probe.

    Notes
    -----
    - Both object and probe are created by Priors, hence both can be manually initialized.
    - Measurement data must be injected later via graph.get_factor(meas_label).set_observed(...).
    """
    # Prior over object
    obj = ~GaussianPrior(event_shape=obj_shape, label=obj_label, dtype=dtype)

    # Prior over probe
    prb = ~GaussianPrior(event_shape=prb_shape, label=prb_label, dtype=dtype)

    # Extract patches and form exit waves
    patches = obj.extract_patches(indices)
    exit_waves = prb * patches

    # Measurement factor
    AmplitudeMeasurement(var=noise, label=meas_label, damping = damping) << fft2(exit_waves)
    return
