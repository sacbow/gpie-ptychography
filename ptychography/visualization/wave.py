from ptychography.visualization._mpl import require_matplotlib
from ptychography.core.wave import Wave
from backend.array import to_numpy
import numpy as np

def plot_wave(
    wave: Wave,
    *,
    mode: str = "abs",
    ax=None,
    **imshow_kwargs,
):
    """
    Visualize a Wave object.

    mode: "abs", "phase", "real", "imag"
    """
    plt = require_matplotlib()

    if ax is None:
        fig, ax = plt.subplots()

    data = to_numpy(wave.data)

    if mode == "abs":
        img = np.abs(data)
    elif mode == "phase":
        img = np.angle(data)
    elif mode == "real":
        img = np.real(data)
    elif mode == "imag":
        img = np.imag(data)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    ax.imshow(img, **imshow_kwargs)
    return ax
