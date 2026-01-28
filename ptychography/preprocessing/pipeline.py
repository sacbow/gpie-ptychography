from typing import Callable, List, Optional, Any

from ptychography.data.diffraction import DiffractionData
from backend.array import to_numpy

from .dataset import PreprocessedDiffractionDataset
from .sorting import reorder_diffraction_data
from .anscombe import anscombe_transform
from .normalize import global_max_normalize


def preprocess_diffraction_data(
    data: List[DiffractionData],
    *,
    sort_key: Optional[Callable[[DiffractionData], object]] = None,
    shuffle: bool = False,
    rng: Any = None,
    anscombe: bool = True,
    normalize: bool = True,
) -> PreprocessedDiffractionDataset:
    """
    Preprocess diffraction data for ptychography solvers.

    Processing steps:
        (1) sorting / shuffling
        (2) Anscombe transform
        (3) global normalization

    Args:
        data: List of DiffractionData.
        sort_key: Optional key function for sorting.
        shuffle: Whether to randomly shuffle data.
        seed: RNG seed for shuffling.
        anscombe: Whether to apply Anscombe transform.
        normalize: Whether to apply global normalization.

    Returns:
        PreprocessedDiffractionDataset
    """

    # --------------------------------------------------
    # Step 1: reorder
    # --------------------------------------------------
    data = reorder_diffraction_data(
        data,
        sort_key=sort_key,
        shuffle=shuffle,
        rng=rng,
    )

    indices = [d.indices for d in data]
    shifts = [d.subpixel_shift for d in data]

    # --------------------------------------------------
    # Step 2: Anscombe transform
    # --------------------------------------------------
    if anscombe:
        amplitudes = anscombe_transform(data)
        base_variance = 1.0
    else:
        amplitudes = [d.amplitude() for d in data]
        base_variance = 0.0  # undefined; user responsibility

    # --------------------------------------------------
    # Step 3: normalization
    # --------------------------------------------------
    if normalize:
        amplitude_array, variance, scale = global_max_normalize(
            amplitudes,
            base_variance=base_variance,
        )
    else:
        from backend.array import xp
        xp_ = xp()
        amplitude_array = xp_.stack(amplitudes, axis=0)
        variance = base_variance
        scale = 1.0

    return PreprocessedDiffractionDataset(
        amplitude=to_numpy(amplitude_array),
        variance=float(variance),
        indices=indices,
        subpixel_shift=shifts,
        meta={
            "anscombe": anscombe,
            "normalize": normalize,
            "scale": scale,
        },
    )
