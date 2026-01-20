import math
import numpy as np
import pytest
from itertools import islice

from ptychography.scan.raster import raster_scan
from ptychography.scan.fermat import fermat_spiral


# ------------------------------------------------------------
# Raster scan
# ------------------------------------------------------------

def test_raster_scan_first_points_no_jitter():
    gen = raster_scan(stride=1.0, jitter=0.0)
    points = list(islice(gen, 9))

    # Expected Manhattan spiral (y, x)
    expected = [
        (0.0, 0.0),
        (0.0, 1.0),
        (1.0, 1.0),
        (1.0, 0.0),
        (1.0, -1.0),
        (0.0, -1.0),
        (-1.0, -1.0),
        (-1.0, 0.0),
        (-1.0, 1.0),
    ]

    assert points == expected


def test_raster_scan_stride_scaling():
    gen = raster_scan(stride=2.5, jitter=0.0)
    points = list(islice(gen, 4))

    # Only scaling should change
    expected = [
        (0.0, 0.0),
        (0.0, 2.5),
        (2.5, 2.5),
        (2.5, 0.0),
    ]

    assert points == expected


def test_raster_scan_with_jitter_is_reproducible():
    rng = np.random.default_rng(1234)

    gen1 = raster_scan(stride=1.0, jitter=0.1, rng=rng)
    pts1 = list(islice(gen1, 5))

    rng = np.random.default_rng(1234)
    gen2 = raster_scan(stride=1.0, jitter=0.1, rng=rng)
    pts2 = list(islice(gen2, 5))

    np.testing.assert_allclose(pts1, pts2)


# ------------------------------------------------------------
# Fermat spiral
# ------------------------------------------------------------

def test_fermat_spiral_starts_at_origin():
    gen = fermat_spiral(step=1.0, jitter=0.0)
    y, x = next(gen)

    assert y == pytest.approx(0.0)
    assert x == pytest.approx(0.0)


def test_fermat_spiral_radial_growth():
    gen = fermat_spiral(step=2.0, jitter=0.0)
    points = list(islice(gen, 10))

    radii = [math.hypot(y, x) for (y, x) in points]

    # r_i = step * sqrt(i)
    expected = [2.0 * math.sqrt(i) for i in range(10)]

    np.testing.assert_allclose(radii, expected, rtol=1e-6)


def test_fermat_spiral_with_jitter_is_reproducible():
    rng = np.random.default_rng(42)

    gen1 = fermat_spiral(step=1.0, jitter=0.05, rng=rng)
    pts1 = list(islice(gen1, 8))

    rng = np.random.default_rng(42)
    gen2 = fermat_spiral(step=1.0, jitter=0.05, rng=rng)
    pts2 = list(islice(gen2, 8))

    np.testing.assert_allclose(pts1, pts2)


# ------------------------------------------------------------
# General properties
# ------------------------------------------------------------

def test_scan_generators_are_infinite():
    gen = raster_scan()
    for _ in range(1000):
        next(gen)  # should not StopIteration

    gen = fermat_spiral()
    for _ in range(1000):
        next(gen)
