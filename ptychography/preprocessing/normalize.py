from typing import List, Tuple
from backend.array import xp


def global_max_normalize(
    amplitudes: List[object],
    *,
    base_variance: float = 1.0,
) -> Tuple[object, float, float]:
    """
    Apply global max normalization to amplitude data.

    Args:
        amplitudes: List of amplitude arrays.
        base_variance: Variance before normalization
                       (Anscombe => typically 1.0).

    Returns:
        normalized_amplitudes: stacked array (N, H, W)
        variance: scalar noise variance after normalization
        scale: normalization constant
    """
    xp_ = xp()

    # Determine global maximum
    max_vals = [xp_.max(A) for A in amplitudes]
    scale = xp_.max(xp_.stack(max_vals))

    if scale <= 0:
        raise ValueError("Normalization scale must be positive.")

    # Normalize and stack
    normalized = xp_.stack([A / scale for A in amplitudes], axis=0)

    variance = base_variance / (scale ** 2)

    return normalized, float(variance), float(scale)
