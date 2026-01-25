# visualize/diffraction.py
import numpy as np
from ptychography.visualization._mpl import require_matplotlib
from ptychography.backend.array import to_numpy

def plot_diffraction_grid(
    diffraction_list,
    *,
    ncols=4,
    log_scale=True,
    ax=None,
    **imshow_kwargs,
):
    plt = require_matplotlib()

    n = len(diffraction_list)
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols)
    axes = np.atleast_1d(axes).ravel()

    for i, (d, ax_) in enumerate(zip(diffraction_list, axes)):
        img = to_numpy(d.intensity)
        if log_scale:
            img = np.log1p(img)
        ax_.imshow(img, **imshow_kwargs)
        ax_.axis("off")

    for ax_ in axes[n:]:
        ax_.axis("off")

    return fig, axes
