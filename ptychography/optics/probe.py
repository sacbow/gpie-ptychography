from typing import Tuple, Literal, Optional
from ptychography.backend.array import xp
from ptychography.backend.rng import get_rng, normal
from .aperture import make_aperture


def make_smooth_random_phase(
    shape: Tuple[int, int],
    cutoff_radius: float = 0.05,
    rng=None,
) -> xp().ndarray:
    """
    Generate a smooth random phase map by applying a Gaussian low-pass filter
    to white noise in frequency space.

    Parameters
    ----------
    shape : (ny, nx)
        Output phase map size.
    cutoff_radius : float
        Normalized cutoff frequency for Gaussian low-pass filter
        (relative to Nyquist frequency = 0.5).
    rng : numpy.random.Generator or cupy.random.Generator, optional
        Random number generator. If None, a default RNG is used.

    Returns
    -------
    phase : xp.ndarray
        Smooth random phase map (radians in [0, 2π)).
    """
    ny, nx = shape
    fft2 = xp().fft.fft2
    ifft2 = xp().fft.ifft2
    fftfreq = xp().fft.fftfreq
    fftshift = xp().fft.fftshift

    rng = get_rng(rng)

    # Complex Gaussian noise
    noise = normal(rng=rng, size=shape, mean=0.0, std=1.0)
    noise = noise.astype(xp().complex64)
    K = fft2(noise)

    # Gaussian low-pass filter in frequency space
    fy = fftfreq(ny)
    fx = fftfreq(nx)
    FX, FY = xp().meshgrid(fx, fy, indexing="ij")
    R = xp().sqrt(FX**2 + FY**2)
    gaussian_lpf = xp().exp(-(R / cutoff_radius) ** 2)

    # Filter and transform back
    smoothed = fftshift(ifft2(K * gaussian_lpf))
    phase = xp().angle(smoothed)

    return (phase + 2 * xp().pi) % (2 * xp().pi)


def generate_probe(
    shape: Tuple[int, int],
    pixel_size: float = 1.0,
    aperture_radius: float = 0.1,
    *,
    kind: Literal["circular", "square"] = "circular",
    space: Literal["fourier", "real"] = "real",
    smooth_edge_sigma: float = 0.05,
    random_phase: bool = False,
    cutoff_radius: float = 0.05,
    rng=None,
) -> xp().ndarray:
    """
    Generate a soft-aperture probe in real or Fourier space.

    Parameters
    ----------
    shape : (ny, nx)
        Probe shape in pixels.
    pixel_size : float
        Physical pixel size.
    aperture_radius : float, optional
        Aperture radius or half-width (depending on `kind`).
    kind : {"circular", "square"}, optional
        Aperture shape. Default: "circular".
    space : {"fourier", "real"}, optional
        Domain in which the aperture mask is interpreted:
            - "fourier": aperture defined in Fourier space, then IFFT to real space
            - "real": aperture interpreted directly in real space
    smooth_edge_sigma : float, optional
        Gaussian falloff width at aperture edge, as a fraction of radius.
    random_phase : bool, optional
        If True, apply smooth random phase modulation to the aperture.
    cutoff_radius : float, optional
        Low-pass cutoff for random phase generator (normalized frequency).
    rng : numpy.random.Generator or cupy.random.Generator, optional
        Random number generator. If None, a default RNG is used.

    Returns
    -------
    probe : xp.ndarray
        Complex-valued real-space probe field, normalized so that
        max(abs(probe)) = 1.
    """
    fft2 = xp().fft.fft2
    ifft2 = xp().fft.ifft2
    ifftshift = xp().fft.ifftshift

    rng = get_rng(rng)

    if aperture_radius is None:
        aperture_radius = shape[0] * 0.5

    # --- Generate aperture mask ---
    mask = make_aperture(
        shape=shape,
        pixel_size=pixel_size,
        aperture_radius=aperture_radius,
        kind=kind,
        smooth_edge_sigma=smooth_edge_sigma,
    )

    # --- Apply random phase aberration if requested ---
    if random_phase:
        phase = make_smooth_random_phase(
            shape,
            cutoff_radius=cutoff_radius,
            rng=rng,
        )
        mask = mask * xp().exp(1j * phase)

    # --- Domain handling ---
    if space == "fourier":
        probe = ifftshift(ifft2(mask))
    elif space == "real":
        probe = ifftshift(mask.astype(xp().complex64))
    else:
        raise ValueError(
            f"Invalid 'space' argument: {space}. Must be 'fourier' or 'real'."
        )

    # --- Normalize amplitude ---
    probe /= xp().max(xp().abs(probe))

    return probe
