from ptychography.visualization._mpl import require_matplotlib

def plot_scan_positions(context, *, ax=None, **scatter_kwargs):
    """
    Plot scan positions in real space.

    Parameters
    ----------
    context : PtychoContext
        Ptychography context containing scan_positions.
    ax : matplotlib.axes.Axes, optional
        Target axes.
    **scatter_kwargs
        Passed directly to ax.scatter().

    Returns
    -------
    ax : matplotlib.axes.Axes
    """
    plt = require_matplotlib()

    if ax is None:
        fig, ax = plt.subplots()
    
    xs, ys = zip(*context.scan_positions)
    ax.scatter(xs, ys, **scatter_kwargs)
    ax.set_aspect("equal")

    return ax
