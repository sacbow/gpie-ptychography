import numpy as _np

_backend = _np

def set_backend(lib):
    global _backend
    _backend = lib

def xp():
    """Return current array backend (numpy or cupy)."""
    return _backend

def to_numpy(x):
    try:
        import cupy as cp
        if isinstance(x, cp.ndarray):
            return x.get()
    except ImportError:
        pass
    return _np.asarray(x)
