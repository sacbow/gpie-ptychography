# ptychography/data/ptychography.py

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any

import numpy as np

from ptychography.data.diffraction import DiffractionData


@dataclass
class Ptychography:
    """
    Container class for a ptychography dataset.

    This class manages diffraction data, scan geometry, and basic
    inspection/visualization utilities. It deliberately does NOT
    implement any forward or inverse simulation logic.

    Design notes
    ------------
    - Forward simulation and inference are handled by separate layers.
    - This class corresponds conceptually to a dataset / experiment,
      not to a specific physical model (single-state, mixed-state, etc.).
    """

    # ------------------------------------------------------------------
    # Core data
    # ------------------------------------------------------------------

    data: List[DiffractionData] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Scan geometry (shared across all diffraction patterns)
    # ------------------------------------------------------------------

    pixel_pitch: Optional[Tuple[float, float]] = None
    wavelength: Optional[float] = None
    detector_distance: Optional[float] = None

    # Units (purely informational)
    position_unit: str = "um"
    wavelength_unit: str = "m"

    # Arbitrary metadata (beam energy, exposure time, notes, etc.)
    meta: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Basic container interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> DiffractionData:
        return self.data[idx]

    def append(self, item: DiffractionData) -> None:
        self.data.append(item)

    def extend(self, items: List[DiffractionData]) -> None:
        self.data.extend(items)

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    @property
    def positions(self) -> List[Tuple[float, float]]:
        return [d.position for d in self.data]

    @property
    def indices(self):
        return [d.indices for d in self.data]

    @property
    def intensities(self):
        return [d.intensity for d in self.data]

    # ------------------------------------------------------------------
    # Validation / consistency checks
    # ------------------------------------------------------------------

    def validate(self) -> None:
        """
        Validate internal consistency of the dataset.

        Raises
        ------
        ValueError
            If diffraction data have inconsistent shapes.
        """
        if not self.data:
            return

        shapes = [d.shape for d in self.data]
        if len(set(shapes)) != 1:
            raise ValueError(f"Inconsistent diffraction shapes: {set(shapes)}")

    # ------------------------------------------------------------------
    # Summary / inspection
    # ------------------------------------------------------------------

    def summary(self) -> str:
        """
        Return a human-readable summary of the dataset.
        """
        n = len(self.data)
        shape = self.data[0].shape if n > 0 else None

        return (
            "Ptychography Dataset\n"
            f"  Number of diffraction patterns : {n}\n"
            f"  Diffraction shape              : {shape}\n"
            f"  Pixel pitch                    : {self.pixel_pitch}\n"
            f"  Wavelength                     : {self.wavelength}\n"
            f"  Detector distance              : {self.detector_distance}\n"
        )

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------

    def show_diffraction(
        self,
        index: int = 0,
        *,
        log_scale: bool = True,
        cmap: str = "viridis",
        ax=None,
    ):
        """
        Visualize a single diffraction pattern.

        Parameters
        ----------
        index : int
            Index of diffraction pattern to visualize.
        """
        return self.data[index].show(
            ax=ax,
            log_scale=log_scale,
            cmap=cmap,
        )

    def show_positions(self, ax=None):
        """
        Plot scan positions in real space.
        """
        import matplotlib.pyplot as plt

        positions = np.array(self.positions)

        if ax is None:
            _, ax = plt.subplots()

        ax.plot(positions[:, 1], positions[:, 0], "o")
        ax.set_xlabel(f"x [{self.position_unit}]")
        ax.set_ylabel(f"y [{self.position_unit}]")
        ax.set_title("Scan positions")
        ax.axis("equal")
        ax.grid(True)

        return ax
