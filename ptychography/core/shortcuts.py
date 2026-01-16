# ptychography/core/shortcuts.py

from ptychography.core.ops import FFT2
from ptychography.core.wave import Wave


def fft2(
    wave: Wave,
    s=None,
    axes=(-2, -1),
    norm=None,
):
    """
    2D FFT shortcut (NumPy-compatible API).
    """
    return FFT2(s=s, axes=axes, norm=norm) @ wave
