# ptychography/data/diffraction.py

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple, Optional, Dict, Any

from ptychography.backend.array import xp


@dataclass
class DiffractionData:
    """
    Container class for a single diffraction intensity pattern and its metadata.
    Design principles
    -----------------
    - This class represents *measured data* (detector output), not algorithm-specific
      preprocessed quantities (e.g., amplitude, Anscombe-transformed data).
    - The minimal information required for ptychography simulations/inference is:
        (1) scan position (continuous coordinates)
        (2) measured diffraction intensity (float array)
        (3) slicing indices on the object grid (discrete indices)

    Attributes
    ----------
    position : Tuple[float, float]
        (y, x) scan position in real space. Units are arbitrary but should be
        consistent across a dataset (e.g., µm or nm).
    intensity : object
        Measured diffraction intensity image. Must be a float ndarray-like object
        (NumPy or CuPy). Shape is typically (H, W).
    indices : Optional[Tuple[slice, slice]]
        Object-grid slicing indices corresponding to this measurement. This is an
        algorithm-side discretization artifact derived from position + geometry.
        If None, it can be assigned later by preprocessing/geometry logic.
    meta : Dict[str, Any]
        Arbitrary metadata dictionary. Intended for future extensions such as
        sub-pixel shift parameters, detector geometry info, exposure time, etc.
    """

    position: Tuple[float, float]
    intensity: object  # ndarray-like (NumPy or CuPy), float dtype recommended
    indices: Optional[Tuple[slice, slice]] = None
    subpixel_shift: Optional[Tuple[float, float]] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    # ---------------------------------------------------------------------
    # Validation / basic properties
    # ---------------------------------------------------------------------

    def validate(self) -> None:
        xp_ = xp()
        arr = xp_.asarray(self.intensity)

        if arr.ndim != 2:
            raise ValueError(
                f"intensity must be 2D, got ndim={arr.ndim} (shape={arr.shape})."
            )

        if not xp_.issubdtype(arr.dtype, xp_.floating):
            raise TypeError(
                f"intensity must be floating dtype, got dtype={arr.dtype}."
            )

        if self.subpixel_shift is not None:
            dy, dx = self.subpixel_shift
            if not all(isinstance(v, (int, float)) for v in (dy, dx)):
                raise TypeError(
                    f"subpixel_shift must be numeric (dy, dx), got {self.subpixel_shift}."
                )

    @property
    def shape(self) -> Tuple[int, int]:
        """Return intensity image shape."""
        return getattr(self.intensity, "shape", None)

    # ---------------------------------------------------------------------
    # Derived quantities (algorithm-side conveniences)
    # ---------------------------------------------------------------------

    def amplitude(self, *, eps: float = 0.0):
        """
        Return sqrt(intensity) as an amplitude-like quantity.

        Notes
        -----
        - This is a convenience method only.
        - For Poisson-like noise, consider using an Anscombe transform or other
          variance-stabilizing preprocessing outside this class.

        Parameters
        ----------
        eps : float
            Optional small constant added before sqrt to avoid sqrt(0) issues
            in downstream code.

        Returns
        -------
        ndarray-like
            sqrt(intensity + eps)
        """
        xp_ = xp()
        return xp_.sqrt(xp_.asarray(self.intensity) + xp_.asarray(eps, dtype=xp_.asarray(self.intensity).dtype))

    # ---------------------------------------------------------------------
    # Utilities
    # ---------------------------------------------------------------------

    def summary(self) -> str:
        """Return a compact string summarizing this diffraction item."""
        shape = getattr(self.intensity, "shape", None)
        has_idx = self.indices is not None
        return f"DiffractionData(position={self.position}, shape={shape}, has_indices={has_idx})"

    def with_indices(self, indices: Tuple[slice, slice]) -> "DiffractionData":
        """
        Return a shallow-copied DiffractionData with updated indices.
        """
        return DiffractionData(
            position=self.position,
            intensity=self.intensity,
            indices=indices,
            meta=dict(self.meta),
        )

    def to_numpy(self):
        """
        Return intensity as a NumPy ndarray (for visualization / serialization).

        This does not modify the stored intensity.
        """
        import numpy as np

        data = self.intensity
        if isinstance(data, np.ndarray):
            return data

        # CuPy arrays typically provide .get()
        try:
            return data.get()
        except Exception:
            return np.asarray(data)

    def show(self, ax=None, log_scale: bool = True, cmap: str = "viridis"):
        """
        Visualize diffraction intensity.

        Parameters
        ----------
        ax : matplotlib axis, optional
            If None, a new figure/axis is created.
        log_scale : bool
            If True, plot log10(intensity + 1e-12).
        cmap : str
            Matplotlib colormap name.

        Returns
        -------
        matplotlib axis
        """
        import numpy as np
        import matplotlib.pyplot as plt

        img = self.to_numpy().astype(np.float64, copy=False)

        if log_scale:
            img = np.log10(img + 1e-12)

        if ax is None:
            _, ax = plt.subplots()

        ax.imshow(img, cmap=cmap)
        ax.set_title(f"Diffraction intensity @ {self.position}")
        ax.axis("off")
        return ax
