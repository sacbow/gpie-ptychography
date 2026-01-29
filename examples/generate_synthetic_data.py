"""
Synthetic ptychography data generation script.

This script generates:
- scan positions
- ptychography context
- synthetic probe
- synthetic object
- diffraction data via forward simulation

The output is intended to be consumed by preprocessing
and reconstruction algorithms (PIE / gPIE).
"""

from __future__ import annotations

import itertools
import numpy as np
from typing import List, Tuple

# ------------------------------------------------------------
# Core ptychography components
# ------------------------------------------------------------

from ptychography.data import PtychoContext
from ptychography.scan import raster_scan
from ptychography.optics import generate_probe
from ptychography.utils.io import load_sample_image
from ptychography.core import (
    ptycho_model,
    Wave,
    PoissonMeasurement,
    fft2,
    get_slice,
)
from ptychography.data.diffraction import DiffractionData


# ------------------------------------------------------------
# Main data generation routine
# ------------------------------------------------------------

def generate_synthetic_dataset(
    *,
    n_scan: int = 64,
    object_shape: Tuple[int, int] = (256, 256),
    probe_shape: Tuple[int, int] = (128, 128),
    pixel_pitch: float = 0.1,
    scan_stride: float = 2.0,
    scan_jitter: float = 0.01,
    photon_scale: float = 1000.0,
    rng_seed: int = 0,
) -> Tuple[PtychoContext, np.ndarray, np.ndarray, List[DiffractionData]]:
    """
    Generate a synthetic ptychography dataset.

    Returns
    -------
    ctx : PtychoContext
        Ptychography experimental context.
    probe : np.ndarray
        Complex probe array.
    obj : np.ndarray
        Real-valued object array.
    diffraction_data : list of DiffractionData
        Simulated diffraction measurements.
    """

    rng = np.random.default_rng(rng_seed)

    # ------------------------------------------------------------
    # Scan positions
    # ------------------------------------------------------------

    scan_positions = list(
        itertools.islice(
            raster_scan(stride=scan_stride, jitter=scan_jitter),
            n_scan,
        )
    )

    # ------------------------------------------------------------
    # Ptychography context
    # ------------------------------------------------------------

    ctx = PtychoContext(
        scan_positions=scan_positions,
        object_shape=object_shape,
        probe_shape=probe_shape,
        pixel_pitch=pixel_pitch,
    )

    # ------------------------------------------------------------
    # Probe generation
    # ------------------------------------------------------------

    probe = generate_probe(
        shape=ctx.probe_shape,
        pixel_size=ctx.pixel_pitch,
        aperture_radius=2.0,
        kind="circular",
        space="real",
        smooth_edge_sigma=0.0,
        random_phase=False,
        rng=rng,
    )

    # ------------------------------------------------------------
    # Object generation
    # ------------------------------------------------------------

    obj_amp = load_sample_image(
        name="coins",
        shape=ctx.object_shape,
    )

    # Real-valued object (no phase modulation for simplicity)
    obj = obj_amp.astype(np.float32)

    # ------------------------------------------------------------
    # Forward model definition
    # ------------------------------------------------------------

    @ptycho_model
    def forward_model(ctx):
        obj_wave = Wave(label="object")
        prb_wave = Wave(label="probe")

        obj_slices = get_slice(obj_wave, ctx)
        exit_wave = obj_slices * prb_wave

        intensity = abs(fft2(exit_wave)) ** 2
        PoissonMeasurement() << intensity * photon_scale

    # ------------------------------------------------------------
    # Compile and run forward model
    # ------------------------------------------------------------

    model = forward_model(ctx)
    model.set_wave("object", obj)
    model.set_wave("probe", probe)

    model.forward()

    diffraction_data = model.get_diffraction_data()

    return ctx, probe, obj, diffraction_data


# ------------------------------------------------------------
# Minimal CLI-style execution
# ------------------------------------------------------------

if __name__ == "__main__":
    ctx, probe, obj, diffraction_data = generate_synthetic_dataset()

    print("Synthetic dataset generated:")
    print(f"  n_scan          : {len(diffraction_data)}")
    print(f"  object_shape    : {ctx.object_shape}")
    print(f"  probe_shape     : {ctx.probe_shape}")
