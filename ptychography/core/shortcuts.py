from __future__ import annotations

from ptychography.core.ops import FFT2
from ptychography.core.wave import Wave
from ptychography.core.slice import Slice
from ptychography.core.replicate import Replicate


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

def get_slice(wave: Wave, context) -> Wave:
    """
    Slice an object wave according to scan positions.
    """
    return Slice(context) @ wave


def replicate(wave: Wave, context) -> Wave:
    """
    Replicate a wave for each scan position.
    """
    return Replicate(context) @ wave
