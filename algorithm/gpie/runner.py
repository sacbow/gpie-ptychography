from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Literal, Optional

import numpy as np

from ptychography.preprocessing.dataset import PreprocessedDiffractionDataset
from backend.rng import get_rng

from .models import (
    ptychography_graph_known_probe,
    ptychography_graph_unknown_probe,
)


InitMode = Literal["uninformative", "sample", "manual"]
Schedule = Literal["parallel", "sequential"]
Backend = Literal["numpy", "cupy"]


@dataclass(frozen=True)
class GPieRunConfig:
    """
    Configuration for running gPIE belief propagation.
    """
    n_iter: int = 50
    schedule: Schedule = "parallel"
    block_size: int = 1
    verbose: bool = False


class GPiePtychographyRunner:
    """
    Adapter for running gPIE-based ptychography reconstruction.

    Design principles
    -----------------
    - gPIE backend (NumPy / CuPy) is controlled explicitly by this runner.
    - PreprocessedDiffractionDataset is the sole input describing the experiment.
    - All outputs exposed to users are NumPy ndarrays.
    - gPIE internals (beliefs, messages, UA) are not leaked.
    """

    def __init__(
        self,
        dataset: PreprocessedDiffractionDataset,
        *,
        known_probe: bool,
        probe: Optional[np.ndarray] = None,
        damping: Any = "auto",
        backend: Backend = "numpy",
        obj_dtype: Any = np.complex64,
        meas_label: str = "meas",
        obj_label: str = "object",
        prb_label: str = "probe",
    ) -> None:
        self.dataset = dataset
        self.known_probe = bool(known_probe)

        self.meas_label = meas_label
        self.obj_label = obj_label
        self.prb_label = prb_label
        self.obj_dtype = obj_dtype
        self.damping = damping

        # --------------------------------------------------
        # Configure gPIE numerical backend
        # --------------------------------------------------
        self._set_gpie_backend(backend)

        # --------------------------------------------------
        # Validate dataset
        # --------------------------------------------------
        self._validate_dataset()

        # --------------------------------------------------
        # Build factor graph
        # --------------------------------------------------
        if self.known_probe:
            if probe is None:
                raise ValueError("probe must be provided when known_probe=True.")
            self._probe_value = np.asarray(probe)

            self.graph = ptychography_graph_known_probe(
                obj_shape=self.dataset.object_shape,
                prb=self._probe_value,
                indices=self.dataset.indices,
                noise=float(self.dataset.variance),
                dtype=self.obj_dtype,
                meas_label=self.meas_label,
                obj_label=self.obj_label,
                damping=self.damping,
            )
        else:
            self._probe_value = None
            self.graph = ptychography_graph_unknown_probe(
                obj_shape=self.dataset.object_shape,
                prb_shape=self.dataset.probe_shape,
                indices=self.dataset.indices,
                noise=float(self.dataset.variance),
                dtype=self.obj_dtype,
                meas_label=self.meas_label,
                obj_label=self.obj_label,
                prb_label=self.prb_label,
                damping=self.damping,
            )

        # --------------------------------------------------
        # Inject measurements
        # --------------------------------------------------
        self.meas_node = self.graph.get_factor(self.meas_label)
        self.meas_node.set_observed(np.asarray(self.dataset.amplitude))

        self._rng: Any = None

    # ------------------------------------------------------------------
    # Backend handling
    # ------------------------------------------------------------------

    def _set_gpie_backend(self, backend: Backend) -> None:
        from gpie.core.backend import set_backend

        if backend == "numpy":
            set_backend(np)
        elif backend == "cupy":
            try:
                import cupy as cp
            except ImportError as e:
                raise ImportError("backend='cupy' requested but CuPy is not installed.") from e
            set_backend(cp)
        else:
            raise ValueError(f"Unknown backend '{backend}'")

    # ------------------------------------------------------------------
    # Dataset validation
    # ------------------------------------------------------------------

    def _validate_dataset(self) -> None:
        amp = np.asarray(self.dataset.amplitude)

        if amp.ndim != 3:
            raise ValueError(
                "dataset.amplitude must have shape (n_scan, H, W). "
                f"Got shape={amp.shape}."
            )

        if len(self.dataset.indices) != amp.shape[0]:
            raise ValueError(
                "dataset.indices length must match amplitude batch size."
            )

        if tuple(amp.shape[1:]) != tuple(self.dataset.probe_shape):
            raise ValueError(
                "dataset.probe_shape must match amplitude frame shape."
            )

        if not np.isscalar(self.dataset.variance):
            raise ValueError("dataset.variance must be a scalar.")

    # ------------------------------------------------------------------
    # Initialization control (via Graph API)
    # ------------------------------------------------------------------

    def set_init_rng(self, seed: Optional[int] = None) -> None:
        """
        Set RNG used by gPIE initialization.
        """
        self._rng = get_rng(seed)

    def set_object_init(self, mode: InitMode, data: Optional[np.ndarray] = None) -> None:
        if mode == "manual":
            if data is None:
                raise ValueError("manual init selected but data is None.")
            data = np.asarray(data)

        self.graph.set_init_strategy(
            label=self.obj_label,
            mode=mode,
            data=data,
        )

    def set_probe_init(self, mode: InitMode, data: Optional[np.ndarray] = None) -> None:
        if self.known_probe:
            raise RuntimeError("Probe is known; probe init is not applicable.")

        if mode == "manual":
            if data is None:
                raise ValueError("manual init selected but data is None.")
            data = np.asarray(data)

        self.graph.set_init_strategy(
            label=self.prb_label,
            mode=mode,
            data=data,
        )

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def run(
        self,
        config: GPieRunConfig,
        *,
        callback: Optional[Callable[[int, Dict[str, Any]], None]] = None,
        compute_fitness: bool = True,
    ) -> None:
        if self._rng is not None:
            self.graph.set_init_rng(self._rng)

        def _callback(graph: Any, t: int) -> None:
            metrics: Dict[str, Any] = {}
            if compute_fitness:
                metrics["fitness"] = self.compute_fitness()
            if callback is not None:
                callback(t, metrics)

        self.graph.run(
            n_iter=int(config.n_iter),
            schedule=config.schedule,
            block_size=int(config.block_size),
            callback=_callback if (callback or compute_fitness) else None,
            verbose=bool(config.verbose),
        )

    # ------------------------------------------------------------------
    # Public outputs (NumPy only)
    # ------------------------------------------------------------------

    def compute_fitness(self) -> float:
        return float(self.meas_node.compute_fitness())

    def get_object(self) -> np.ndarray:
        wave = self.graph.get_wave(self.obj_label)
        belief = wave.compute_belief()
        return self._to_numpy(belief.data[0])

    def get_probe(self) -> np.ndarray:
        if self.known_probe:
            return np.asarray(self._probe_value)

        wave = self.graph.get_wave(self.prb_label)
        belief = wave.compute_belief()
        return self._to_numpy(belief.data[0])

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _to_numpy(x: Any) -> np.ndarray:
        try:
            import cupy as cp
            if isinstance(x, cp.ndarray):
                return x.get()
        except ImportError:
            pass
        return np.asarray(x)
