from typing import Tuple, Literal
from ptychography.backend.array import xp


def make_aperture(
    shape: Tuple[int, int],
    pixel_size: float,
    aperture_radius: float,
    kind: Literal["circular", "square"] = "circular",
    smooth_edge_sigma: float = 0.01,
) -> xp().ndarray:
    """
    Generate a soft aperture mask (circular or square) defined on a
    frequency grid.

    This function constructs an aperture mask using coordinates generated
    by `fftfreq`, and therefore assumes a frequency-domain interpretation
    by default. Interpretation as a real-space aperture is possible, but
    the physical meaning of `pixel_size` and `aperture_radius` must then
    be handled consistently by the caller.

    Parameters
    ----------
    shape : (ny, nx)
        Size of the aperture mask in pixels.
    pixel_size : float
        Physical pixel size used to define frequency coordinates.
        Passed to `fftfreq` as the sampling interval.
    aperture_radius : float
        Aperture radius (for circular) or half-width (for square), expressed
        in the same units as the frequency coordinates.
    kind : {"circular", "square"}
        Aperture shape.
    smooth_edge_sigma : float
        Width of the Gaussian edge smoothing, expressed as a fraction of
        `aperture_radius`.

    Returns
    -------
    mask : xp.ndarray
        Aperture mask with values in [0, 1], defined on a frequency grid.
    """

    ny, nx = shape
    fy = xp().fft.fftfreq(ny, d=pixel_size)
    fx = xp().fft.fftfreq(nx, d=pixel_size)
    FX, FY = xp().meshgrid(fx, fy, indexing="ij")

    if kind == "circular":
        R = xp().sqrt(FX**2 + FY**2)
        sigma = aperture_radius * smooth_edge_sigma
        mask = xp().exp(-0.5 * ((R - aperture_radius) / sigma) ** 2)
        mask[R <= aperture_radius] = 1.0

    elif kind == "square":
        # half-width along each axis
        wx = wy = aperture_radius
        smooth_x = xp().exp(-0.5 * ((xp().abs(FX) - wx) / (wx * smooth_edge_sigma)) ** 2)
        smooth_y = xp().exp(-0.5 * ((xp().abs(FY) - wy) / (wy * smooth_edge_sigma)) ** 2)
        mask = xp().minimum(smooth_x, smooth_y)
        mask[(xp().abs(FX) <= wx) & (xp().abs(FY) <= wy)] = 1.0
    else:
        raise ValueError(f"Unknown aperture kind: {kind}")

    return mask
