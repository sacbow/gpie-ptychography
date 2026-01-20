from __future__ import annotations

import contextlib
import threading
from typing import Dict, Iterable, List, Optional, Set, Tuple, Any

from ptychography.core.propagator import Propagator
from ptychography.core.measurement import Measurement
from ptychography.core.wave import Wave


_current_graph = threading.local()


class Graph:
    """
    Forward-only computation graph for ptychography models.

    This Graph supports:
    - Thread-local "active graph" context (with graph.observe(): ...)
    - Measurement registration during DSL construction
    - Upstream traversal from measurements to collect:
        * propagators
        * waves (and label-based wave registry)
    - Topological execution of propagators by generation
    - Measurement execution and result collection
    - Label-based injection of concrete data into Waves after build()
    """

    def __init__(self) -> None:
        # User-registered terminal nodes (typically via Measurement.__lshift__)
        self._measurements: List[Measurement] = []

        # Collected by build()
        self._propagators: List[Propagator] = []
        self._propagators_sorted: List[Propagator] = []
        self._waves: Set[Wave] = set()

        # Label registry for Waves (only labeled waves are stored here)
        self._waves_by_label: Dict[str, Wave] = {}

        # Build state
        self._built: bool = False

        # associated context
        self.context = None

    # ------------------------------------------------------------------
    # Active graph context (thread-local)
    # ------------------------------------------------------------------

    @staticmethod
    def get_active_graph() -> Optional["Graph"]:
        """
        Return the currently active graph inside a `with graph.observe():` block.
        """
        return getattr(_current_graph, "value", None)

    @contextlib.contextmanager
    def observe(self):
        """
        Context manager that makes this Graph the active graph.

        Typical usage:
            graph = Graph()
            with graph.observe():
                # build DSL here, including Measurement << Wave
                ...
        """
        _current_graph.value = self
        try:
            yield
        finally:
            _current_graph.value = None

    # ------------------------------------------------------------------
    # Registration API (called during DSL construction)
    # ------------------------------------------------------------------

    def add_measurement(self, measurement: Measurement) -> None:
        """
        Register a measurement node.

        Measurements are the entry points for upstream traversal in build().
        """
        if measurement not in self._measurements:
            self._measurements.append(measurement)
        # Invalidate build cache
        self._built = False

    def register_wave(self, wave: Wave) -> None:
        """
        Register a wave into the graph's wave set and label registry (if labeled).

        This method is safe to call multiple times for the same wave object.
        """
        if wave not in self._waves:
            self._waves.add(wave)

        label = getattr(wave, "label", None)
        if label is None:
            return

        if not isinstance(label, str) or label.strip() == "":
            raise ValueError(f"Wave label must be a non-empty string, got: {label!r}")

        if label in self._waves_by_label and self._waves_by_label[label] is not wave:
            raise ValueError(
                f"Duplicate Wave label '{label}' detected in graph."
            )

        self._waves_by_label[label] = wave

    def add_propagator(self, propagator: Propagator) -> None:
        """
        Register a propagator node.

        In the "measurement-rooted" build pattern, users typically do not call this
        directly. It is kept for completeness.
        """
        if propagator not in self._propagators:
            self._propagators.append(propagator)
        self._built = False

    # ------------------------------------------------------------------
    # Graph building (measurement-rooted upstream traversal)
    # ------------------------------------------------------------------

    def build(self) -> None:
        """
        Traverse upstream from measurements and prepare execution order.

        After build():
        - self._propagators_sorted is populated
        - self._waves and self._waves_by_label are populated (as much as possible)
        """
        self._collect_nodes()
        self._sort_propagators()
        self._built = True

    def _collect_nodes(self) -> None:
        """
        Collect all propagators and waves reachable upstream from registered measurements.
        """
        collected_props: Set[Propagator] = set()
        collected_waves: Set[Wave] = set()

        # Stack contains propagators to process
        stack: List[Propagator] = []

        # Seed traversal from measurement inputs
        for meas in self._measurements:
            wave = getattr(meas, "input", None)
            if wave is None:
                continue

            collected_waves.add(wave)
            self.register_wave(wave)

            parent = getattr(wave, "parent", None)
            if parent is not None:
                stack.append(parent)

        # DFS upstream traversal
        while stack:
            prop = stack.pop()
            if prop in collected_props:
                continue

            collected_props.add(prop)

            # Register all input waves and continue traversing via each wave.parent
            for w in getattr(prop, "inputs", []):
                if w not in collected_waves:
                    collected_waves.add(w)
                self.register_wave(w)

                parent = getattr(w, "parent", None)
                if parent is not None and parent not in collected_props:
                    stack.append(parent)

            # Also register output wave if present (helps label lookup for intermediate waves)
            out = getattr(prop, "output", None)
            if out is not None:
                if out not in collected_waves:
                    collected_waves.add(out)
                self.register_wave(out)

        # Commit collected sets
        self._propagators = list(collected_props)
        self._waves |= collected_waves  # ensure set union with any pre-registered waves

    def _sort_propagators(self) -> None:
        """
        Sort propagators by generation (topological order surrogate).
        """
        # Fail fast if any propagator has generation unset
        unset = [p for p in self._propagators if getattr(p, "generation", None) is None]
        if unset:
            raise RuntimeError(
                "Graph.build(): some propagators have generation=None. "
                "Ensure propagators are connected via `prop @ wave` before build()."
            )

        self._propagators_sorted = sorted(
            self._propagators,
            key=lambda p: p.generation,  # type: ignore[arg-type]
        )

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def forward(self) -> None:
        """
        Execute forward computation for the entire graph.
        """
        if not self._built:
            raise RuntimeError("Graph.forward(): graph is not built. Call build() first.")

        for prop in self._propagators_sorted:
            prop.forward()

        for meas in self._measurements:
            meas.forward()

    # ------------------------------------------------------------------
    # Data injection / retrieval (label-based)
    # ------------------------------------------------------------------

    def get_wave(self, label: str) -> Wave:
        """
        Get a Wave by its label.

        Notes
        -----
        - Only labeled waves can be retrieved by this method.
        - Labeled waves are typically created by `Wave(label="...")` within
          a `with graph.observe():` block, or discovered during build().
        """
        if label not in self._waves_by_label:
            known = ", ".join(sorted(self._waves_by_label.keys()))
            raise KeyError(f"Unknown wave label '{label}'. Known labels: [{known}]")
        return self._waves_by_label[label]

    def set_wave_data(self, label: str, data: Any) -> None:
        """
        Attach concrete array data to a labeled Wave.
        """
        w = self.get_wave(label)
        w.set_data(data)

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------

    def collect_measurements(self) -> List[object]:
        """
        Collect data objects from all measurement nodes (order preserved).
        """
        return [m.data for m in self._measurements]

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def propagators(self) -> List[Propagator]:
        return list(self._propagators)

    @property
    def measurements(self) -> List[Measurement]:
        return list(self._measurements)

    @property
    def waves(self) -> Set[Wave]:
        return set(self._waves)

    @property
    def wave_labels(self) -> List[str]:
        return sorted(self._waves_by_label.keys())

    def summary(self) -> str:
        """
        Human-readable summary of the graph contents.
        """
        return (
            "Graph Summary\n"
            f"  built            : {self._built}\n"
            f"  propagators      : {len(self._propagators)}\n"
            f"  measurements     : {len(self._measurements)}\n"
            f"  waves            : {len(self._waves)}\n"
            f"  labeled_waves    : {len(self._waves_by_label)}\n"
        )

    def __repr__(self) -> str:
        return (
            "Graph(\n"
            f"  built = {self._built},\n"
            f"  propagators = {len(self._propagators)},\n"
            f"  measurements = {len(self._measurements)},\n"
            f"  waves = {len(self._waves)},\n"
            f"  labeled_waves = {len(self._waves_by_label)}\n"
            ")"
        )
