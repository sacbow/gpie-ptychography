# ptychography/core/decorators.py

from functools import wraps
from ptychography.core.graph import Graph
from ptychography.core.model import PtychoModel


def model(func):
    """
    Decorator for defining a ptychography forward model.

    The decorated function must have signature:
        f(context) -> None
    """

    @wraps(func)
    def wrapper(context):
        graph = Graph()

        with graph.observe():
            result = func(context)

        # Explicit return from model function is not allowed
        if result is not None:
            raise RuntimeError(
                "@model-decorated functions must not return a value."
            )

        graph.build()
        return PtychoModel(graph=graph, context=context)

    return wrapper
