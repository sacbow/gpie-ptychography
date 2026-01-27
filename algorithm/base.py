# algorithm/base.py

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any

import numpy as np

from backend import to_numpy
from ptychography.preprocessing.dataset import PreprocessedDiffractionDataset


class PtychographyAlgorithm(ABC):
    """
    Base class for ptychographic phase retrieval algorithms.

    Design principles
    -----------------
    - Public API uses NumPy ndarray only.
    - GPU acceleration is an internal implementation detail.
    - FFTs must be performed via backend.fft utilities.
    """

    def __init__(
        self,
        dataset: PreprocessedDiffractionDataset,
        *,
        init_object: np.ndarray,
        init_probe: Optional[np.ndarray] = None,
        estimate_probe: bool = False,
        use_gpu: bool = False,
    ):
        """
        Parameters
        ----------
        dataset : PreprocessedDiffractionDataset
            Preprocessed diffraction dataset.
        init_object : np.ndarray
            Initial object estimate.
        init_probe : np.ndarray, optional
            Initial probe estimate.
        estimate_probe : bool
            Whether to update the probe during reconstruction.
        use_gpu : bool
            Whether to use GPU internally.
        """

        self.dataset = dataset

        self._object = np.asarray(init_object)
        self._probe = None if init_probe is None else np.asarray(init_probe)

        if estimate_probe and self._probe is None:
            raise ValueError(
                "init_probe must be provided when estimate_probe=True."
            )

        self.estimate_probe = estimate_probe
        self.use_gpu = use_gpu

        self.iteration: int = 0
        self.history: Dict[str, list] = {}

        self._setup_backend()
        self._initialize_state()

    # ------------------------------------------------------------------
    # Backend handling
    # ------------------------------------------------------------------

    def _setup_backend(self) -> None:
        """
        Setup internal numerical backend.

        Notes
        -----
        - Subclasses should not override this method.
        - Backend choice does not affect public API.
        """
        if self.use_gpu:
            try:
                import cupy as cp
                self.xp = cp
            except ImportError as e:
                raise ImportError(
                    "use_gpu=True but CuPy is not installed."
                ) from e
        else:
            self.xp = np

        # Move state to backend
        self._object = self.xp.asarray(self._object)
        if self._probe is not None:
            self._probe = self.xp.asarray(self._probe)

    # ------------------------------------------------------------------
    # Initialization hook
    # ------------------------------------------------------------------

    def _initialize_state(self) -> None:
        """
        Initialize algorithm-specific internal state.

        Subclasses may override this.
        """
        pass

    # ------------------------------------------------------------------
    # Core iteration API
    # ------------------------------------------------------------------

    @abstractmethod
    def step(self) -> None:
        """
        Perform one iteration of the algorithm.

        Notes
        -----
        - Sequential algorithms: one full sweep over all diffraction patterns.
        - Parallel algorithms: one global update.
        """
        raise NotImplementedError

    def run(self, n_iter: int) -> None:
        """
        Run the algorithm for a given number of iterations.
        """
        for _ in range(n_iter):
            self.step()
            self.iteration += 1

    # ------------------------------------------------------------------
    # Public state accessors (NumPy only)
    # ------------------------------------------------------------------

    @property
    def object(self) -> np.ndarray:
        """Current object estimate (NumPy array)."""
        return to_numpy(self._object)

    @property
    def probe(self) -> Optional[np.ndarray]:
        """Current probe estimate (NumPy array), if present."""
        if self._probe is None:
            return None
        return to_numpy(self._probe)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def get_state(self) -> Dict[str, Any]:
        """
        Return a snapshot of the current solver state.
        """
        return {
            "iteration": self.iteration,
            "object": self.object,
            "probe": self.probe,
        }
