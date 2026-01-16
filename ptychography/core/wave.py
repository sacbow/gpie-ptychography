# ptychography/core/wave.py

from __future__ import annotations
from typing import Optional, Tuple

from ptychography.utils.types import ArrayLike


class Wave:
    """
    Forward-only wave node in a ptychography computation graph.

    A Wave represents both:
    - a node in a forward computation graph
    - an ndarray-like container holding concrete data (NumPy or CuPy)

    Parameters
    ----------
    label : str, optional
        Human-readable identifier (e.g. 'object', 'probe').
    data : ArrayLike, optional
        Concrete array value (NumPy or CuPy).
    generation : int, default=0
        Topological generation index.
    """
    __array_priority__ = 1000.0  # Ensure Wave ops override ndarray ops

    def __init__(
        self,
        label: Optional[str] = None,
        data: Optional[ArrayLike] = None,
        generation: int = 0,
    ):
        self.label: Optional[str] = label
        self._data: Optional[ArrayLike] = data

        # Graph structure
        self.parent = None            # Upstream Propagator
        self.children = []            # Downstream Propagators

        # Topological ordering
        self.generation: int = generation

    # ------------------------------------------------------------------
    # Graph connectivity
    # ------------------------------------------------------------------

    def set_parent(self, propagator) -> None:
        """
        Register upstream propagator.
        """
        if self.parent is not None:
            raise RuntimeError("Wave already has a parent.")
        self.parent = propagator

    def add_child(self, propagator) -> None:
        """
        Register downstream propagator.
        """
        self.children.append(propagator)

    # ------------------------------------------------------------------
    # ndarray-like interface (read-only)
    # ------------------------------------------------------------------

    @property
    def data(self) -> ArrayLike:
        """
        Read-only access to underlying array data.

        Raises
        ------
        RuntimeError
            If data is not set.
        """
        if self._data is None:
            raise RuntimeError("Wave data is not set.")
        return self._data

    def set_data(self, data: ArrayLike) -> None:
        """
        Attach concrete array data to this wave.
        """
        self._data = data

    def is_set(self) -> bool:
        """
        Whether this wave holds concrete data.
        """
        return self._data is not None

    # ------------------------------------------------------------------
    # ndarray metadata
    # ------------------------------------------------------------------

    @property
    def shape(self) -> Optional[Tuple[int, ...]]:
        """
        Shape of underlying array, if present.
        """
        return None if self._data is None else self._data.shape

    @property
    def dtype(self):
        """
        dtype of underlying array, if present.
        """
        return None if self._data is None else self._data.dtype

    @property
    def ndim(self) -> Optional[int]:
        """
        Number of dimensions of underlying array, if present.
        """
        return None if self._data is None else self._data.ndim

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        name = self.label if self.label is not None else "Wave"
        shape = "∅" if self._data is None else str(self._data.shape)
        return f"<Wave {name} gen={self.generation} shape={shape}>"
    
    # ----------------------------
    # Operator overloading
    # ----------------------------

    def __add__(self, other):
        if isinstance(other, Wave):
            from .ops import Add
            return Add() @ (self, other)
        return NotImplemented

    def __radd__(self, other):
        return self.__add__(other)

    def __mul__(self, other):
        if isinstance(other, Wave):
            from .ops import Multiply
            return Multiply() @ (self, other)
        return NotImplemented

    def __rmul__(self, other):
        return self.__mul__(other)

    def __abs__(self):
        from .ops import Abs
        return Abs() @ self

    def __pow__(self, power):
        if isinstance(power, (int, float)):
            from .ops import Power
            return Power(power) @ self
        return NotImplemented
