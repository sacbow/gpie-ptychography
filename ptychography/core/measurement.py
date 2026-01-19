# ptychography/core/measurement.py

from __future__ import annotations
from typing import Optional, Any

from ptychography.core.wave import Wave
from ptychography.backend.array import xp
from ptychography.backend.rng import get_rng


class Measurement:
    """
    Terminal measurement node in a forward ptychography computation graph.

    A Measurement:
    - consumes exactly one input Wave
    - does not produce an output Wave
    - records measured data internally

    Usage
    -----
    meas = Measurement(...)
    meas << wave
    meas.forward()
    """

    def __init__(self, name: Optional[str] = None):
        self.name = name or self.__class__.__name__

        # Graph connectivity
        self.input: Optional[Wave] = None

        # Stored measurement result (DiffractionData or similar)
        self.data: Optional[Any] = None

    # ------------------------------------------------------------------
    # Graph wiring
    # ------------------------------------------------------------------

    def __lshift__(self, wave: Wave):
        if not isinstance(wave, Wave):
            raise TypeError("Measurement can only be connected to a Wave.")
        
        if self.input is not None:
            raise RuntimeError("Measurement already has an input wave.")

        self.input = wave

        # Register to active graph if exists
        from ptychography.core.graph import Graph
        graph = Graph.get_active_graph()
        if graph is not None:
            graph.add_measurement(self)

        return self

    # ------------------------------------------------------------------
    # Forward execution
    # ------------------------------------------------------------------

    def forward(self):
        """
        Execute measurement.

        This method:
        - reads concrete data from input Wave
        - converts it into a DiffractionData-like object
        - stores it internally

        Note
        ----
        Exact semantics (Poisson, Gaussian, etc.) are implemented
        in subclasses via `compute()`.
        """
        if self.input is None:
            raise RuntimeError("Measurement has no input wave.")

        x = self.input.data
        self.data = self.compute(x)
        return self.data

    def compute(self, x):
        """
        Measurement-specific computation.

        Subclasses should override this method.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        state = "connected" if self.input is not None else "unconnected"
        return f"<{self.name} ({state})>"


class PoissonMeasurement(Measurement):
    """
    Poisson measurement node.

    Applies Poisson noise to non-negative real-valued input.
    """

    def __init__(self, rng=None, name=None):
        super().__init__(name=name or "PoissonMeasurement")
        self.rng = rng

    def compute(self, x):
        xp_ = xp()

        # ----------------------------
        # Input validation
        # ----------------------------
        if not xp_.issubdtype(x.dtype, xp_.floating):
            raise RuntimeError(
                "PoissonMeasurement expects a real-valued input."
            )

        if xp_.any(x < 0):
            raise RuntimeError(
                "PoissonMeasurement expects non-negative input values."
            )

        # ----------------------------
        # Poisson sampling
        # ----------------------------
        rng = self.rng or get_rng()
        return rng.poisson(x)



