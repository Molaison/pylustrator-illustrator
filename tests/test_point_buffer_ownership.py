from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.lines import Line2D

import pylustrator.transform_engine as transform_engine
from pylustrator.artist_adapters import PointHandleModel
from pylustrator.transform_engine import PointEditPlan, PointEditSource


def _manual_source(control_points: np.ndarray) -> PointEditSource:
    target = Line2D([0.0, 1.0], [0.0, 1.0])
    model = PointHandleModel(
        target=target,
        keys=(0, 1),
        display_positions=np.array([[10.0, 20.0], [30.0, 40.0]]),
        topology_token=("manual", 2),
    )
    return PointEditSource(
        target=target,
        handle_model=model,
        source_fingerprint=("manual",),
        topology_token=model.topology_token,
        control_points=control_points,
        selection_points=np.array([[10.0, 20.0], [30.0, 40.0]]),
    )


def test_public_point_source_owns_its_frozen_control_buffer():
    external = np.array([[10.0, 20.0], [30.0, 40.0]])
    expected = external.copy()

    source = _manual_source(external)

    assert not np.shares_memory(source.control_array(), external)
    assert not source.control_array().flags.writeable
    external[:] = -1.0
    np.testing.assert_array_equal(source.control_array(), expected)
    with pytest.raises(ValueError, match="read-only"):
        source.control_array()[0, 0] = 99.0


def test_public_point_plan_owns_its_frozen_control_buffer():
    source = _manual_source(np.array([[10.0, 20.0], [30.0, 40.0]]))
    external = np.array([[11.0, 21.0], [30.0, 40.0]])
    expected = external.copy()

    plan = PointEditPlan(
        source=source,
        point_keys=(0,),
        destination_positions=np.array([[11.0, 21.0]]),
        is_noop=False,
        control_points=external,
        selection_points=np.array([[11.0, 21.0], [30.0, 40.0]]),
    )

    assert not np.shares_memory(plan.control_array(), external)
    assert not plan.control_array().flags.writeable
    external[:] = -1.0
    np.testing.assert_array_equal(plan.control_array(), expected)
    with pytest.raises(ValueError, match="read-only"):
        plan.control_array()[0, 0] = 99.0


def test_internal_capture_and_preview_freeze_owned_100k_buffers_without_copy(
    monkeypatch,
):
    count = 100_000
    x = np.linspace(0.05, 0.95, count)
    y = 0.5 + 0.2 * np.sin(np.linspace(0.0, 16.0, count))
    fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
    line = ax.plot(x, y, linewidth=0.8)[0]
    fig.canvas.draw()

    transferred = []
    freeze = transform_engine._readonly_point_preview

    def record_transfer(points):
        if isinstance(points, transform_engine._OwnedPointPreview):
            transferred.append(points.values)
        return freeze(points)

    monkeypatch.setattr(transform_engine, "_readonly_point_preview", record_transfer)
    try:
        source = PointEditSource.capture(line)
        assert len(transferred) == 1
        capture_buffer = transferred.pop()
        assert np.shares_memory(source.control_array(), capture_buffer)

        key = source.handle_model.keys[0]
        destination = source.handle_model.positions_array()[0] + (3.0, -2.0)
        plan = PointEditPlan.preview(source, key, destination)
        assert len(transferred) == 1
        preview_buffer = transferred.pop()
        assert np.shares_memory(plan.control_array(), preview_buffer)
        assert not source.control_array().flags.writeable
        assert not plan.control_array().flags.writeable
    finally:
        plt.close(fig)
