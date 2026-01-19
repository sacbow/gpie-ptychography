# tests/test_model_graph_interop.py

import numpy as np
import pytest

from ptychography.core.wave import Wave
from ptychography.core.measurement import Measurement
from ptychography.core.decorator import model


# ------------------------------------------------------------
# Dummy propagator for testing
# ------------------------------------------------------------

from ptychography.core.propagator import Propagator


class Identity(Propagator):
    """
    Simple propagator: y = x
    """

    def compute(self, x):
        return x


# ------------------------------------------------------------
# Dummy context
# ------------------------------------------------------------

class DummyContext:
    """
    Minimal stand-in for PtychoContext.
    """
    pass

# ------------------------------------------------------------
# Dummy measurement
# ------------------------------------------------------------

class DummyMeasurement(Measurement):
    """
    Minimal stand-in for PtychoContext.
    """
    def compute(self, x):
        return x



# ------------------------------------------------------------
# Tests
# ------------------------------------------------------------

def test_model_decorator_builds_graph_and_model():
    """
    @model should:
    - create a Graph
    - run the DSL under graph.observe()
    - build the graph
    - return a PtychoModel with graph and context
    """

    ctx = DummyContext()

    @model
    def simple_model(ctx):
        x = Wave(label="x")
        y = Identity() @ x
        DummyMeasurement() << y

    m = simple_model(ctx)

    # Basic sanity checks
    assert m.context is ctx
    assert m.graph is not None

    # Graph should have collected nodes
    graph = m.graph
    assert len(graph.measurements) == 1
    assert len(graph.propagators) == 1

    # Wave registry should contain labeled wave
    assert "x" in graph.wave_labels


def test_forward_execution_propagates_data():
    """
    Data injected into a labeled Wave should propagate
    through the graph to the Measurement.
    """

    ctx = DummyContext()

    @model
    def simple_model(ctx):
        x = Wave(label="x")
        y = Identity() @ x
        DummyMeasurement() << y

    m = simple_model(ctx)
    graph = m.graph

    # Inject concrete data
    x_data = np.array([[1.0, 2.0], [3.0, 4.0]])
    graph.set_wave_data("x", x_data)

    # Execute forward
    m.forward()

    # Collect measurement
    results = graph.collect_measurements()
    assert len(results) == 1

    measured = results[0]
    np.testing.assert_array_equal(measured, x_data)


def test_multiple_measurements_and_order():
    """
    Multiple Measurement nodes should be supported,
    and results should be collected in registration order.
    """

    ctx = DummyContext()

    @model
    def multi_measurement_model(ctx):
        x = Wave(label="x")
        y = Identity() @ x
        DummyMeasurement(name="m1") << y
        DummyMeasurement(name="m2") << y

    m = multi_measurement_model(ctx)
    graph = m.graph

    graph.set_wave_data("x", np.array([1, 2, 3]))
    m.forward()

    results = graph.collect_measurements()
    assert len(results) == 2

    np.testing.assert_array_equal(results[0], np.array([1, 2, 3]))
    np.testing.assert_array_equal(results[1], np.array([1, 2, 3]))


def test_model_function_must_not_return_value():
    """
    @model-decorated functions must not return a value.
    """

    ctx = DummyContext()

    @model
    def bad_model(ctx):
        x = Wave(label="x")
        return x  # illegal

    with pytest.raises(RuntimeError):
        bad_model(ctx)


def test_forward_without_data_raises():
    """
    Forward execution without setting required Wave data
    should raise an error.
    """

    ctx = DummyContext()

    @model
    def simple_model(ctx):
        x = Wave(label="x")
        y = Identity() @ x
        DummyMeasurement() << y

    m = simple_model(ctx)

    with pytest.raises(RuntimeError):
        m.forward()
