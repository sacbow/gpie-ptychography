# ptychography/backend/array.py

import numpy as _np

_backend = _np

def set_backend(lib):
    global _backend
    _backend = lib

def xp():
    """Return current array backend (numpy or cupy)."""
    return _backend
