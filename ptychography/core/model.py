# ptychography/core/model.py

class PtychoModel:
    """
    Compiled ptychography forward model.

    This object binds:
      - a forward computation graph
      - a semantic context
    """

    def __init__(self, graph, context):
        self.graph = graph
        self.context = context

    def forward(self):
        self.graph.forward()