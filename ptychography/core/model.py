from typing import List
from ptychography.backend.array import xp, to_numpy
from ptychography.data.diffraction import DiffractionData


class PtychoModel:
    """
    Compiled ptychography forward model.

    This object binds:
      - a forward computation graph
      - a semantic context
    """

    def __init__(self, graph, context):
        self._graph = graph
        self._context = context
        self._graph.context = context

    def set_wave(self, label: str, data):
        """
        Set concrete data for a labeled wave.
        Data is converted to current backend array.
        """
        backend = xp()
        array = backend.asarray(data)
        self._graph.set_wave_data(label, array)

    def forward(self):
        """
        Execute forward simulation.
        """
        self._graph.forward()

    # ------------------------------------------------------------------
    # DiffractionData construction
    # ------------------------------------------------------------------

    def get_diffraction_data(self) -> List[DiffractionData]:
        """
        Convert measurement outputs into DiffractionData objects.

        Returns
        -------
        List[DiffractionData]
            One DiffractionData per scan position.

        Notes
        -----
        - This method assumes that measurement.data has shape
          (n_scan, H, W).
        - Each scan index i is paired with context-provided metadata:
            * scan_positions[i]
            * slice_indices[i]
            * subpixel_shifts[i]
        - More complex cases (e.g., extra reference measurements)
          should be handled by custom Measurement subclasses.
        """
        ctx = self._context

        diffraction_items: List[DiffractionData] = []

        measurements = self._graph.collect_measurements()

        for meas in measurements:
            data = meas.data
            if data is None:
                continue

            # Expect batched diffraction patterns
            if getattr(data, "ndim", None) != 3:
                raise ValueError(
                    "Measurement data must have shape (n_scan, H, W). "
                    f"Got shape={getattr(data, 'shape', None)}."
                )

            if data.shape[0] != ctx.n_scan:
                raise ValueError(
                    "Leading dimension of measurement data does not match "
                    f"number of scan positions: data.shape[0]={data.shape[0]}, "
                    f"n_scan={ctx.n_scan}."
                )

            for i in range(ctx.n_scan):
                frame = xp().take(data, i, axis=0)
                item = DiffractionData(
                    position=ctx.scan_positions[i],
                    intensity=to_numpy(frame),
                    indices=ctx.slice_indices[i],
                    subpixel_shift=ctx.subpixel_shifts[i],
                    meta={
                        "measurement": meas.name,
                        "scan_index": i,
                    },
                )
                diffraction_items.append(item)

        return diffraction_items
    
    def __repr__(self) -> str:
        return (
            "PtychoModel(\n"
            "  context = "
            + repr(self._context).replace("\n", "\n  ")
            + ",\n"
            "  graph = "
            + repr(self._graph).replace("\n", "\n  ")
            + "\n"
            ")"
        )