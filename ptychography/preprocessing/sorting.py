from typing import Callable, List, Optional, Any
import numpy as np

from ptychography.data.diffraction import DiffractionData
from ptychography.backend.rng import get_rng


def reorder_diffraction_data(
    data: List[DiffractionData],
    *,
    sort_key: Optional[Callable[[DiffractionData], object]] = None,
    shuffle: bool = False,
    rng: Any = None,
) -> List[DiffractionData]:
    """
    Reorder diffraction data by sorting or random shuffling.

    Exactly one of `sort_key` or `shuffle` may be specified.

    Args:
        data: List of DiffractionData.
        sort_key: Key function for sorting.
        shuffle: Whether to randomly shuffle the data.
        seed: Optional RNG seed for shuffling.

    Returns:
        Reordered list of DiffractionData.
    """

    if sort_key is not None and shuffle:
        raise ValueError("Specify either sort_key or shuffle, not both.")

    data = list(data)  # shallow copy

    if sort_key is not None:
        data.sort(key=sort_key)
        return data

    if shuffle:
        rng = rng if rng is not None else get_rng()
        rng.shuffle(data)
        return data

    return data
