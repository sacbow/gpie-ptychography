# ptychography/core/ops.py

from __future__ import annotations
from typing import Tuple, Optional, Sequence

from ptychography.core.propagator import Propagator
from ptychography.utils.types import ArrayLike
from backend.array import xp


# ---------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------

def _broadcastable(shape_a, shape_b) -> bool:
    """
    Check NumPy-style broadcast compatibility.
    """
    for a, b in zip(shape_a[::-1], shape_b[::-1]):
        if a != b and a != 1 and b != 1:
            return False
    return True


# ---------------------------------------------------------------------
# Arithmetic propagators
# ---------------------------------------------------------------------

class Binary(Propagator):
    """
    Elementwise addition: y = x1 + x2
    """

    def check_inputs(self):
        if len(self.inputs) != 2:
            raise RuntimeError("Add expects exactly two input waves.")

        s0 = self.inputs[0].shape
        s1 = self.inputs[1].shape
        if s0 is None or s1 is None:
            return

        if not _broadcastable(s0, s1):
            raise ValueError(f"Add: shapes {s0} and {s1} are not broadcastable.")

    def compute(self, x1: ArrayLike, x2: ArrayLike) -> ArrayLike:
        raise NotImplementedError



class Add(Binary):
    """
    Elementwise addition: y = x1 + x2
    """
    def compute(self, x1: ArrayLike, x2: ArrayLike) -> ArrayLike:
        return x1 + x2


class Subtract(Binary):
    """
    Elementwise subtraction: y = x1 - x2
    """
    def compute(self, x1: ArrayLike, x2: ArrayLike) -> ArrayLike:
        return x1 - x2
    


class Multiply(Binary):
    """
    Elementwise multiplication: y = x1 * x2
    """
    def compute(self, x1: ArrayLike, x2: ArrayLike) -> ArrayLike:
        return x1 * x2


class Divide(Binary):
    """
    Elementwise division: y = x1 / x2
    """
    def compute(self, x, y):
        backend = xp()
        return backend.divide(x, y)




class Abs(Propagator):
    """
    Elementwise absolute value: y = |x|
    """

    def check_inputs(self):
        if len(self.inputs) != 1:
            raise RuntimeError("Abs expects exactly one input wave.")

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

    def check_inputs(self):
        if len(self.inputs) != 1:
            raise RuntimeError("Power expects exactly one input wave.")

    def compute(self, x: ArrayLike) -> ArrayLike:
        backend = xp()
        return backend.power(x, self.power)


# ---------------------------------------------------------------------
# FFT propagators
# ---------------------------------------------------------------------

class FFT2(Propagator):
    """
    2D FFT propagator.

    By default, a centered FFT is applied:
        fftshift(fft2(ifftshift(x)))

    This matches the standard convention in ptychography and
    coherent diffraction imaging.
    """

    def __init__(
        self,
        s: Optional[Tuple[int, int]] = None,
        axes: Tuple[int, int] = (-2, -1),
        norm: Optional[str] = None,
        *,
        centered: bool = True,
        name: Optional[str] = None,
    ):
        super().__init__(name=name or "FFT2")
        self.s = s
        self.axes = axes
        self.norm = "ortho"
        self.centered = centered

    def check_inputs(self):
        if len(self.inputs) != 1:
            raise RuntimeError("FFT2 expects exactly one input wave.")

        ndim = self.inputs[0].ndim
        if ndim is not None and ndim < 2:
            raise ValueError("FFT2 requires input with ndim >= 2.")

    def compute(self, x: ArrayLike) -> ArrayLike:
        backend = xp()
        fft2 = backend.fft.fft2
        fftshift = backend.fft.fftshift
        ifftshift = backend.fft.ifftshift

        if self.centered:
            x = ifftshift(x, axes=self.axes)
            y = fft2(
                x,
                s=self.s,
                axes=self.axes,
                norm=self.norm,
            )
            return fftshift(y, axes=self.axes)
        else:
            return fft2(
                x,
                s=self.s,
                axes=self.axes,
                norm=self.norm,
            )
