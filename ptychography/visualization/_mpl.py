try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

def require_matplotlib():
    if plt is None:
        raise ImportError(
            "matplotlib is required for visualization. "
            "Install it via `pip install matplotlib`."
        )
    return plt
