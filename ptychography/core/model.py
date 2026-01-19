# ptychography/core/model.py

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
        """
        self._graph.set_wave_data(label, data)

    def forward(self):
        self._graph.forward()

    def get_measurements(self):
        return self._graph.collect_measurements()


