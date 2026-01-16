# ptychography/core/propagator.py

from __future__ import annotations
from typing import Iterable, List

from ptychography.core.wave import Wave


class Propagator:
    """
    Base class for forward-only propagators in a ptychography computation graph.

    A Propagator:
    - consumes one or more input Wave objects
    - produces a single output Wave
    - defines a numerical operation via compute()
    """

    def __init__(self, name: str | None = None):
        self.name: str = name or self.__class__.__name__

        # Graph structure
        self.inputs: List[Wave] = []
        self.output: Wave | None = None

        # Topological ordering
        self.generation: int | None = None

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def __matmul__(self, waves: Wave | Iterable[Wave]) -> Wave:
        """
        Connect propagator to input wave(s).

        Usage
        -----
        out = prop @ wave
        out = prop @ (wave1, wave2)
        """
        if isinstance(waves, Wave):
            waves = [waves]
        else:
            waves = list(waves)

        # Register inputs
        self.inputs = waves
        for w in waves:
            w.add_child(self)

        # Determine generation
        self.generation = max(w.generation for w in waves)

        # Create output wave
        out = Wave(generation=self.generation + 1)
        out.set_parent(self)

        self.output = out
        return out

    # --------------------------------------------------
    # Forward execution
    # --------------------------------------------------

    def forward(self) -> None:
        """
        Execute forward computation.

        This method should NOT be overridden.
        """
        self.check_inputs()

        data_inputs = [w.data for w in self.inputs]
        out = self.compute(*data_inputs)

        self.output.set_data(out)

    # --------------------------------------------------
    # Hooks for subclasses
    # --------------------------------------------------

    def check_inputs(self) -> None:
        """
        Validate input waves.

        Subclasses may override this.
        """
        pass

    def compute(self, *inputs):
        """
        Compute output array from input arrays.
        """
        raise NotImplementedError

    # --------------------------------------------------

    def __repr__(self):
        gen = self.generation if self.generation is not None else "?"
        return f"<{self.name} gen={gen}>"