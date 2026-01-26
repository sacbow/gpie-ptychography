from typing import List
from ptychography.backend.array import xp
from ptychography.data.diffraction import DiffractionData


def anscombe_transform(
    data: List[DiffractionData],
    *,
    offset: float = 3.0 / 8.0,
):
    """
    Apply Anscombe transform to diffraction intensities.

    This function does NOT normalize or stack data.
    It returns a list of amplitude arrays corresponding
    to the input DiffractionData list.

    Args:
        data: List of DiffractionData.
        offset: Offset added before sqrt (default: 3/8).

    Returns:
        List of ndarray-like amplitude arrays.
    """
    xp_ = xp()
    amplitudes = []

    for d in data:
        I = xp_.asarray(d.intensity)
        A = xp_.sqrt(I + offset)
        amplitudes.append(A)

    return amplitudes
