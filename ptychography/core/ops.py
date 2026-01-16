# ptychography/core/ops.py

from __future__ import annotations
from typing import Tuple, Optional

from ptychography.core.propagator import Propagator
from ptychography.utils.types import ArrayLike
from ptychography.backend.array import xp


# ---------------------------------------------------------------------
# Arithmetic propagators
# ---------------------------------------------------------------------

class Add(Propagator):
    """
    Elementwise addition: y = x1 + x2 + ...
    """

    def compute(self, *inputs: ArrayLike) -> ArrayLike:
        backend = xp()
        out = backend.zeros_like(inputs[0])
        for x in inputs:
            out = out + x
        return out


class Multiply(Propagator):
    """
    Elementwise multiplication: y = x1 * x2 * ...
    """

    def compute(self, *inputs: ArrayLike) -> ArrayLike:
        backend = xp()
        out = backend.ones_like(inputs[0])
        for x in inputs:
            out = out * x
        return out


class Abs(Propagator):
    """
    Elementwise absolute value: y = |x|
    """

    def compute(self, x: ArrayLike) -> ArrayLike:
        backend = xp()
        return backend.abs(x)


class Power(Propagator):
    """
    Elementwise power: y = x ** p
    """

    def __init__(self, p: float, name: Optional[str] = None):
        super().__init__(name=name or f"Power({p})")
        self.power = p

    def compute(self, x: ArrayLike) -> ArrayLike:
        backend = xp()
        return backend.power(x, self.p)


# ---------------------------------------------------------------------
# FFT propagators
# ---------------------------------------------------------------------

class FFT2(Propagator):
    """
    2D FFT propagator.

    API intentionally mirrors numpy.fft.fft2.
    """

    def __init__(
        self,
        s: Optional[Tuple[int, int]] = None,
        axes: Tuple[int, int] = (-2, -1),
        norm: Optional[str] = None,
        name: Optional[str] = None,
    ):
        super().__init__(name=name or "FFT2")
        self.s = s
        self.axes = axes
        self.norm = norm

    def compute(self, x: ArrayLike) -> ArrayLike:
        backend = xp()
        return backend.fft.fft2(
            x,
            s=self.s,
            axes=self.axes,
            norm=self.norm,
        )
