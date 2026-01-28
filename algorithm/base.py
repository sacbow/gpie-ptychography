from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Any

import numpy as np

from backend import to_numpy
from ptychography.preprocessing.dataset import PreprocessedDiffractionDataset


class PtychographyAlgorithm(ABC):
    """
    Base class for ptychographic phase retrieval algorithms.

    Design principles
    -----------------
    - Public API uses NumPy ndarray only.
    - Object and probe are always defined.
    - Whether the probe is updated is controlled by `estimate_probe`.
    - GPU acceleration is an internal implementation detail.
    - FFTs must be performed via backend.fft utilities.
    """

    def __init__(
        self,
        dataset: PreprocessedDiffractionDataset,
        *,
        init_object: np.ndarray,
        init_probe: np.ndarray,
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
        init_probe : np.ndarray
            Initial probe estimate (ground truth or initial guess).
        estimate_probe : bool, optional
            Whether to update the probe during reconstruction.
        use_gpu : bool, optional
            Whether to use GPU internally.
        """

        # -------------------------------
        # Store problem definition
        # -------------------------------
        self.dataset = dataset
        self.estimate_probe = bool(estimate_probe)
        self.use_gpu = bool(use_gpu)

        # -------------------------------
        # Store initial state (NumPy)
        # -------------------------------
        self._object = np.asarray(init_object)
        self._probe = np.asarray(init_probe)

        # -------------------------------
        # Bookkeeping
        # -------------------------------
        self.iteration: int = 0
        self.history: Dict[str, list] = {}

        # -------------------------------
        # Backend setup and state transfer
        # -------------------------------
        self._setup_backend()
        self._initialize_state()

    # ------------------------------------------------------------------
    # Backend handling
    # ------------------------------------------------------------------

    def _setup_backend(self) -> None:
        """
        Setup numerical backend and move state to backend memory.

        Notes
        -----
        - Subclasses must not override this method.
        - After this method, internal arrays live on the backend.
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

        self._to_backend()

    def _to_backend(self) -> None:
        """
        Move internal state to the current numerical backend.
        """
        self._object = self.xp.asarray(self._object)
        self._probe = self.xp.asarray(self._probe)

    # ------------------------------------------------------------------
    # Initialization hook
    # ------------------------------------------------------------------

    def _initialize_state(self) -> None:
        """
        Initialize algorithm-specific internal state.

        Subclasses may override this method to allocate buffers,
        precompute constants, or initialize auxiliary variables.
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
        - Sequential algorithms: one full sweep over diffraction patterns.
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
        """
        Current object estimate (NumPy array).
        """
        return to_numpy(self._object)

    @property
    def probe(self) -> np.ndarray:
        """
        Current probe estimate (NumPy array).
        """
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
