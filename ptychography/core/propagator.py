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

    # ------------------------------------------------------------------
    # Forward execution
    # ------------------------------------------------------------------

    def forward(self):
        """
        Execute forward computation for this propagator.

        Reads data from input waves and writes result to output wave.
        """
        if self.output is None:
            raise RuntimeError("Propagator has no output wave.")

        inputs = [w.data for w in self.inputs]
        result = self.compute(*inputs)
        self.output.set_data(result)
        return result

    def compute(self, *inputs):
        """
        Numerical computation implemented by subclasses.

        Parameters
        ----------
        *inputs : ArrayLike
            Concrete array values from input waves.

        Returns
        -------
        ArrayLike
            Output array.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        gen = self.generation if self.generation is not None else "?"
        return f"<{self.name} gen={gen}>"
