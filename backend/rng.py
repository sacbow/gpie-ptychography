from backend.array import xp
import warnings

def get_rng(seed=None):
    backend = xp()

    import numpy as np
    if backend is np:
        return np.random.default_rng(seed)

    try:
        import cupy as cp
        if backend is cp:
            return cp.random.default_rng(seed)
    except ImportError:
        warnings.warn("CuPy backend selected but CuPy not installed.")

    return np.random.default_rng(seed)

def _ensure_rng_backend(rng):
    backend = xp()

    import numpy as np
    try:
        import cupy as cp
    except ImportError:
        cp = None

    if backend is np and not isinstance(rng, np.random.Generator):
        warnings.warn("[rng] RNG backend mismatch, replacing with numpy RNG.")
        return np.random.default_rng()

    if cp and backend is cp and not isinstance(rng, cp.random.Generator):
        warnings.warn("[rng] RNG backend mismatch, replacing with cupy RNG.")
        return cp.random.default_rng()

    return rng



def normal(rng, size, mean=0.0, std=1.0):
    rng = _ensure_rng_backend(rng)
    import numpy as np
    try:
        import cupy as cp
    except ImportError:
        cp = None

    if isinstance(rng, np.random.Generator):
        return rng.normal(loc=mean, scale=std, size=size)
    elif cp and isinstance(rng, cp.random.Generator):
        size = (size,) if isinstance(size, int) else size
        return std * rng.standard_normal(size) + mean


def poisson(rng, lam):
    rng = _ensure_rng_backend(rng)
    import numpy as np
    try:
        import cupy as cp
    except ImportError:
        cp = None

    return rng.poisson(lam)

