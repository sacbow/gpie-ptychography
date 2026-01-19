import pytest
from ptychography.data.context import PtychoContext

def test_context_initialization_basic():
    ctx = PtychoContext(
        scan_positions=[(0.0, 0.0), (1.0, -1.0)],
        object_shape=(128, 128),
        probe_shape=(16, 16),
        pixel_pitch=1.0,
    )

    assert ctx.n_scan == 2
    assert len(ctx.slice_indices) == 2
    assert len(ctx.subpixel_shifts) == 2


def test_real_to_index_origin_mapping():
    ctx = PtychoContext(
        scan_positions=[(0.0, 0.0)],
        object_shape=(11, 11),
        probe_shape=(3, 3),
        pixel_pitch=1.0,
    )

    (iy, ix), _ = ctx._int_indices[0], ctx.subpixel_shifts[0]

    assert iy == 5
    assert ix == 5


def test_real_space_axis_convention():
    ctx = PtychoContext(
        scan_positions=[(1.0, 1.0)],
        object_shape=(11, 11),
        probe_shape=(3, 3),
        pixel_pitch=1.0,
    )

    (iy, ix) = ctx._int_indices[0]

    # +x -> right (column increases)
    assert ix == 6
    # +y -> up (row decreases)
    assert iy == 4

def test_subpixel_shift_computation():
    ctx = PtychoContext(
        scan_positions=[(0.25, -0.25)],
        object_shape=(11, 11),
        probe_shape=(3, 3),
        pixel_pitch=1.0,
    )

    (dy, dx) = ctx.subpixel_shifts[0]

    # y_real = -0.25 -> shift downward (positive dy)
    assert dy > 0
    # x_real = +0.25 -> shift rightward (positive dx)
    assert dx > 0

def test_slice_indices_shape():
    ctx = PtychoContext(
        scan_positions=[(0.0, 0.0)],
        object_shape=(32, 32),
        probe_shape=(8, 8),
        pixel_pitch=1.0,
    )

    sy, sx = ctx.slice_indices[0]
    assert sy.stop - sy.start == 8
    assert sx.stop - sx.start == 8

def test_slice_out_of_bounds_raises():
    with pytest.raises(ValueError):
        PtychoContext(
            scan_positions=[(100.0, 100.0)],
            object_shape=(32, 32),
            probe_shape=(8, 8),
            pixel_pitch=1.0,
        )


def test_invalid_pixel_pitch():
    with pytest.raises(ValueError):
        PtychoContext(
            scan_positions=[(0.0, 0.0)],
            object_shape=(16, 16),
            probe_shape=(4, 4),
            pixel_pitch=0.0,
        )


def test_invalid_scan_positions_format():
    with pytest.raises(ValueError):
        PtychoContext(
            scan_positions=[(0.0,)],
            object_shape=(16, 16),
            probe_shape=(4, 4),
            pixel_pitch=1.0,
        )
