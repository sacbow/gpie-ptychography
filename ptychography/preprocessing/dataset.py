from dataclasses import dataclass
from typing import List, Optional, Tuple, Any


@dataclass
class PreprocessedDiffractionDataset:
    """
    Dataset object directly consumable by ptychography solvers.

    All preprocessing steps (sorting, transforms, normalization)
    are assumed to be completed before constructing this object.
    """

    amplitude: Any
    """
    Amplitude array of shape (N, H, W).
    ndarray-like (NumPy or CuPy).
    """

    object_shape: Tuple[int, int]
    """
    shape of object.
    """

    probe_shape: Tuple[int, int]
    """
    shape of probe (= (H, W)).
    """

    variance: float
    """
    Scalar noise variance shared by all diffraction patterns.
    """

    indices: List[Tuple[slice, slice]]
    """
    Object-grid slicing indices corresponding to each diffraction pattern.
    Ordering must match the first axis of `amplitude`.
    """

    subpixel_shift: Optional[
        List[Tuple[float, float]]
    ] = None
    """
    Optional list of subpixel shifts (dy, dx) per diffraction pattern.
    """

    meta: Optional[dict] = None
    """
    Optional metadata for bookkeeping or debugging.
    """
