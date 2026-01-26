# generate_data.py
#
# Synthetic ptychography data generation script.
#
# This script mirrors the forward-model construction used in
# examples/user_guide/user_guide.ipynb, but strips out all
# visualization and explanatory code.
#
# It provides a minimal interface to obtain:
#   - ground-truth object
#   - ground-truth probe
#   - a fully constructed and executed PtychoModel
#
# The resulting objects can be directly reused in reconstruction
# demo notebooks (e.g., known_probe).

import numpy as np
import itertools

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

# ------------------------------------------------------------
# Forward model definition (identical to user_guide)
# ------------------------------------------------------------

@ptycho_model
def forward_model(ctx):
    """
    Ptychography forward model.

    Latent variables:
        - object : complex object transmission function
        - probe  : complex probe illumination

    Measurement:
        - Poisson-distributed diffraction intensities
    """
    obj = Wave(label="object")
    prb = Wave(label="probe")

    # Slice object according to scan positions
    obj_slices = get_slice(obj, ctx)

    # Exit wave
    exit_wave = obj_slices * prb

    # Far-field diffraction intensity
    intensity = abs(fft2(exit_wave)) ** 2

    # Poisson measurement with fixed scaling
    PoissonMeasurement() << intensity * 1000.0


# ------------------------------------------------------------
# Public API
# ------------------------------------------------------------

def generate_dataset(
    *,
    object_shape=(256, 256),
    probe_shape=(128, 128),
    n_scan=25,
    scan_stride=2.0,
    scan_jitter=0.05,
    pixel_pitch=0.1,
    seed=0,
):
    """
    Generate synthetic ptychography data with a known probe.

    This function reproduces the data-generation procedure used
    in examples/user_guide, but returns only the essential objects
    needed for reconstruction demos.

    Args:
        object_shape (tuple): Shape of the object array (Ny, Nx)
        probe_shape (tuple): Shape of the probe array (Ny, Nx)
        n_scan (int): Number of scan positions
        scan_stride (float): Spacing between scan positions
        scan_jitter (float): Standard deviation of random scan jitter
        pixel_pitch (float): Real-space pixel size
        seed (int): RNG seed for reproducibility

    Returns:
        dict with the following entries:
            - ctx            : PtychoContext
            - object_true    : ground-truth object array (real-valued)
            - probe_true     : ground-truth probe array (complex-valued)
            - model          : PtychoModel after forward simulation
            - diffraction    : simulated diffraction data
    """

    rng = np.random.default_rng(seed)

    # --------------------------------------------------------
    # Generate scan positions
    # --------------------------------------------------------

    scan_positions = list(
        itertools.islice(
            raster_scan(
                stride=scan_stride,
                jitter=scan_jitter,
            ),
            n_scan,
        )
    )

    # --------------------------------------------------------
    # Construct ptychography context
    # --------------------------------------------------------

    ctx = PtychoContext(
        scan_positions=scan_positions,
        object_shape=object_shape,
        probe_shape=probe_shape,
        pixel_pitch=pixel_pitch,
    )

    # --------------------------------------------------------
    # Generate probe (ground truth)
    # --------------------------------------------------------

    probe = generate_probe(
        shape=ctx.probe_shape,
        pixel_size=ctx.pixel_pitch,
        aperture_radius=0.75,
        kind="circular",
        space="fourier",
        smooth_edge_sigma=0.1,
        random_phase=False,
        rng=rng,
    )

    # --------------------------------------------------------
    # Generate object (ground truth)
    # --------------------------------------------------------

    obj_amp = load_sample_image(
        name="coins",
        shape=ctx.object_shape,
    )

    # Purely real-valued object (no phase modulation)
    obj = obj_amp.astype(np.float32)

    # --------------------------------------------------------
    # Build model and run forward simulation
    # --------------------------------------------------------

    model = forward_model(ctx)

    model.set_wave("object", obj)
    model.set_wave("probe", probe)

    model.forward()

    diffraction_data = model.get_diffraction_data()

    # --------------------------------------------------------
    # Return minimal dataset
    # --------------------------------------------------------

    return {
        "ctx": ctx,
        "object_true": obj,
        "probe_true": probe,
        "model": model,
        "diffraction": diffraction_data,
    }
