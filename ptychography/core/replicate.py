from __future__ import annotations
from typing import Any

from ptychography.core.propagator import Propagator
from backend.array import xp


class Replicate(Propagator):
    """
    Replicate a wave for each scan position.

    Input shape
    -----------
    (Ny, Nx)

    Output shape
    ------------
    (n_scan, Ny, Nx)
    """

    def __init__(self, context, name: str | None = None):
        super().__init__(name=name or "Replicate")
        self.context = context

    def check_inputs(self) -> None:
        if len(self.inputs) != 1:
            raise RuntimeError("Replicate expects exactly one input wave.")

    def compute(self, x):
        backend = xp()
        n = self.context.n_scan

        # Broadcast by explicit expansion
        return backend.broadcast_to(
            x,
            (n,) + x.shape,
        )
