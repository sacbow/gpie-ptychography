from __future__ import annotations

from ptychography.core.propagator import Propagator
from backend.array import xp


class Slice(Propagator):
    """
    Slice an object wave according to scan positions.

    Input shape
    -----------
    (Ny_obj, Nx_obj)

    Output shape
    ------------
    (n_scan, Ny_probe, Nx_probe)
    """

    def __init__(self, context, name: str | None = None):
        super().__init__(name=name or "Slice")
        self.context = context

    def check_inputs(self) -> None:
        if len(self.inputs) != 1:
            raise RuntimeError("Slice expects exactly one input wave.")

    def compute(self, x):
        backend = xp()
        slices = self.context.slice_indices

        patches = [
            x[sy, sx]
            for (sy, sx) in slices
        ]

        return backend.stack(patches, axis=0)
