from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Tuple, List


@dataclass
class PtychoContext:
    """
    Semantic context for a ptychography forward model.

    This object defines the mapping between:
      - real-space coordinates (continuous, physical units)
      - array indices (discrete, pixel units)

    The context is immutable and fully validated at initialization time.
    All derived quantities (slice indices, sub-pixel shifts) are computed
    eagerly and cached.

    Design policy
    -------------
    - Scan positions that lead to out-of-bounds probe footprints are
      skipped by default.
    - A summary warning is emitted once if any scan positions are skipped.
    - Strict behavior can be enforced by setting skip_out_of_bounds=False.
    """

    # ------------------------------------------------------------------
    # User-provided parameters
    # ------------------------------------------------------------------

    scan_positions: List[Tuple[float, float]]
    """
    Scan positions in real space.

    Convention
    ----------
    scan_positions[i] = (x, y)
      - +x : right
      - +y : up

    Units must be consistent with pixel_pitch.
    """

    object_shape: Tuple[int, int]
    """
    Shape of the object array: (Ny_obj, Nx_obj).
    """

    probe_shape: Tuple[int, int]
    """
    Shape of the probe array: (Ny_probe, Nx_probe).
    """

    pixel_pitch: float
    """
    Real-space length corresponding to one pixel.
    """

    skip_out_of_bounds: bool = True
    """
    Whether to skip scan positions whose probe footprint exceeds
    object boundaries.
    """

    warn_on_skip: bool = True
    """
    Whether to emit a warning when out-of-bounds scan positions are skipped.
    """

    # ------------------------------------------------------------------
    # Cached derived quantities (filled in __post_init__)
    # ------------------------------------------------------------------

    _float_indices: List[Tuple[float, float]] = field(init=False, repr=False)
    _int_indices: List[Tuple[int, int]] = field(init=False, repr=False)
    _subpixel_shifts: List[Tuple[float, float]] = field(init=False, repr=False)
    _slice_indices: List[Tuple[slice, slice]] = field(init=False, repr=False)

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def __post_init__(self) -> None:
        """
        Validate inputs and compute all derived quantities eagerly.
        """
        self._validate()
        self._build_geometry()

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def n_scan(self) -> int:
        """
        Number of valid scan positions after filtering.
        """
        return len(self._slice_indices)

    @property
    def slice_indices(self) -> List[Tuple[slice, slice]]:
        """
        Precomputed object slice indices for each valid scan position.
        """
        return self._slice_indices

    @property
    def subpixel_shifts(self) -> List[Tuple[float, float]]:
        """
        Sub-pixel shifts (dy, dx) in pixel units for each scan position.

        Convention
        ----------
        dy, dx are defined in array-index coordinates:
          - +dy : downward
          - +dx : rightward
        """
        return self._subpixel_shifts

    # ------------------------------------------------------------------
    # Core coordinate mapping
    # ------------------------------------------------------------------

    def _real_to_float_index(self, pos: Tuple[float, float]) -> Tuple[float, float]:
        """
        Convert real-space coordinates (x, y) to floating-point array indices.

        Mapping
        -------
        Real space:
          +x : right
          +y : up

        Array indices:
          +x : right (column)
          +y : down  (row)

        Origin
        ------
        Real-space (0, 0) corresponds to the center of the object array:
          ((Ny - 1) / 2, (Nx - 1) / 2)
        """
        x_real, y_real = pos
        ny, nx = self.object_shape

        center_y = (ny - 1) / 2.0
        center_x = (nx - 1) / 2.0

        # NOTE:
        #   y_real (upward) maps to decreasing row index
        #   x_real (rightward) maps to increasing column index
        y_idx = center_y - y_real / self.pixel_pitch
        x_idx = center_x + x_real / self.pixel_pitch

        return y_idx, x_idx

    # ------------------------------------------------------------------
    # Geometry builder
    # ------------------------------------------------------------------

    def _build_geometry(self) -> None:
        """
        Compute and cache all geometry-related derived quantities.

        This method performs the following steps atomically per scan position:
          - real -> float index conversion
          - integer index rounding
          - slice construction
          - out-of-bounds validation

        Invalid scan positions are skipped or rejected according to policy.
        """
        float_indices: List[Tuple[float, float]] = []
        int_indices: List[Tuple[int, int]] = []
        shifts: List[Tuple[float, float]] = []
        slices: List[Tuple[slice, slice]] = []

        skipped = 0

        ny_probe, nx_probe = self.probe_shape
        ny_obj, nx_obj = self.object_shape

        half_y = ny_probe // 2
        half_x = nx_probe // 2

        for pos in self.scan_positions:
            y_f, x_f = self._real_to_float_index(pos)

            iy = int(round(y_f))
            ix = int(round(x_f))

            dy = y_f - iy
            dx = x_f - ix

            sy = slice(iy - half_y, iy - half_y + ny_probe)
            sx = slice(ix - half_x, ix - half_x + nx_probe)

            out_of_bounds = (
                sy.start < 0
                or sx.start < 0
                or sy.stop > ny_obj
                or sx.stop > nx_obj
            )

            if out_of_bounds:
                skipped += 1
                if not self.skip_out_of_bounds:
                    raise ValueError(
                        f"Scan position {pos} leads to out-of-bounds slice: "
                        f"y={sy}, x={sx}, object_shape={self.object_shape}"
                    )
                continue

            float_indices.append((y_f, x_f))
            int_indices.append((iy, ix))
            shifts.append((dy, dx))
            slices.append((sy, sx))

        if skipped > 0 and self.skip_out_of_bounds and self.warn_on_skip:
            warnings.warn(
                f"{skipped} / {len(self.scan_positions)} scan positions were skipped "
                f"because the probe footprint exceeded object boundaries.",
                UserWarning,
            )

        self._float_indices = float_indices
        self._int_indices = int_indices
        self._subpixel_shifts = shifts
        self._slice_indices = slices

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate(self) -> None:
        if self.pixel_pitch <= 0:
            raise ValueError(
                f"pixel_pitch must be positive, got {self.pixel_pitch}"
            )

        for name, shape in [
            ("object_shape", self.object_shape),
            ("probe_shape", self.probe_shape),
        ]:
            if not isinstance(shape, (tuple, list)):
                raise ValueError(f"{name} must be a tuple or list, got {shape}")

            if len(shape) < 2:
                raise ValueError(
                    f"{name} must have at least 2 dimensions, got {shape}"
                )

            ny, nx = shape[-2:]
            if ny <= 0 or nx <= 0:
                raise ValueError(
                    f"{name} has invalid spatial dimensions: {shape}"
                )

        for i, pos in enumerate(self.scan_positions):
            if not isinstance(pos, (tuple, list)) or len(pos) != 2:
                raise ValueError(
                    f"scan_positions[{i}] must be an (x, y) pair, got {pos}"
                )

            if not all(isinstance(v, (int, float)) for v in pos):
                raise ValueError(
                    f"scan_positions[{i}] must contain numeric values, got {pos}"
                )

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            "PtychoContext(\n"
            f"  n_scan       = {self.n_scan}\n"
            f"  requested    = {len(self.scan_positions)}\n"
            f"  object_shape = {self.object_shape}\n"
            f"  probe_shape  = {self.probe_shape}\n"
            f"  pixel_pitch  = {self.pixel_pitch}\n"
            ")"
        )
