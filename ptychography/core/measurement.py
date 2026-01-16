# ptychography/core/measurement.py

from __future__ import annotations
from typing import Optional, Any

from ptychography.core.wave import Wave
from ptychography.data.diffraction import DiffractionData


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
        """
        Connect measurement to an input wave.

        Usage
        -----
        measurement << wave
        """
        if not isinstance(wave, Wave):
            raise TypeError("Measurement can only be connected to a Wave.")

        if self.input is not None:
            raise RuntimeError("Measurement already has an input wave.")

        self.input = wave
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


