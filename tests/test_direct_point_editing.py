from __future__ import annotations

import gc
from copy import deepcopy
from dataclasses import replace
from time import perf_counter
import weakref

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.backend_bases import KeyEvent, MouseEvent
from matplotlib.lines import Line2D
from matplotlib.patches import PathPatch, Polygon
from matplotlib.path import Path
from qtpy import QtWidgets

from pylustrator.artist_adapters import (
    PolygonAdapter,
    UnsupportedArtistError,
    artist_adapter_registry,
    get_artist_adapter,
)
from pylustrator.change_tracker import init_figure
from pylustrator.operations import TransformOperation
from pylustrator.interaction import SelectionMode
from pylustrator.commands import (
    LineEndpointReplayConflictError,
    ensure_line_endpoint_replay_api,
    install_line_endpoint_replay_api,
)
from pylustrator.transform_engine import (
    PointEditPlan,
    PointEditSource,
    StaleTransformPlanError,
)
from test_selection_indicator import attach_drag_manager


def _figure_with_manager():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
    manager = attach_drag_manager(fig)
    return app, fig, ax, manager


def test_closed_polygon_exposes_one_handle_for_duplicated_closure_and_commits_both():
    app, fig, ax, manager = _figure_with_manager()
    polygon = ax.add_patch(
        Polygon(
            [(0.2, 0.2), (0.8, 0.2), (0.7, 0.75), (0.2, 0.2)],
            closed=True,
        )
    )
    fig.canvas.draw()
    adapter = get_artist_adapter(polygon)
    model = adapter.point_handle_model()

    assert adapter.operation_support(TransformOperation.EDIT_POINTS).supported
    assert model.keys == (0, 1, 2)
    assert model.aliases_for(0) == (0, 3)

    source = PointEditSource.capture(polygon, handle_model=model)
    destination = model.positions_array()[0] + np.array((17.0, -9.0))
    before = polygon.get_xy().copy()
    plan = PointEditPlan.preview(source, 0, destination)

    np.testing.assert_array_equal(polygon.get_xy(), before)
    assert manager.figure.change_tracker.changes == []
    assert plan.commit() is True
    np.testing.assert_allclose(polygon.get_xy()[0], polygon.get_xy()[-1])
    np.testing.assert_allclose(
        adapter.control_points()[0], destination, atol=0.25, rtol=0
    )
    assert manager.figure.change_tracker.change[1].startswith(".set_xy(")

    manager.selection.clear_targets()
    plt.close(fig)
    assert app is not None


def test_linear_pathpatch_hides_close_dummy_and_repeated_start_but_curves_are_denied():
    app, fig, ax, manager = _figure_with_manager()
    vertices = np.array(
        [
            (0.2, 0.2),
            (0.8, 0.2),
            (0.8, 0.7),
            (0.2, 0.7),
            (0.2, 0.2),
            (0.0, 0.0),
        ]
    )
    codes = [
        Path.MOVETO,
        Path.LINETO,
        Path.LINETO,
        Path.LINETO,
        Path.LINETO,
        Path.CLOSEPOLY,
    ]
    patch = ax.add_patch(PathPatch(Path(vertices, codes)))
    curve = ax.add_patch(
        PathPatch(
            Path(
                [(0.1, 0.1), (0.4, 0.8), (0.9, 0.2)],
                [Path.MOVETO, Path.CURVE3, Path.CURVE3],
            )
        )
    )
    fig.canvas.draw()

    adapter = get_artist_adapter(patch)
    model = adapter.point_handle_model()
    assert model.keys == (0, 1, 2, 3)
    assert model.aliases_for(0) == (0, 4)
    assert len(model.path_array()) == 6

    curve_adapter = get_artist_adapter(curve)
    support = curve_adapter.operation_support(TransformOperation.EDIT_POINTS)
    assert not support.supported
    with pytest.raises(UnsupportedArtistError, match="edit_points"):
        curve_adapter.point_handle_model()

    manager.selection.clear_targets()
    plt.close(fig)
    assert app is not None


def test_open_path_with_coincident_endpoints_keeps_distinct_anchor_identity():
    app, fig, ax, manager = _figure_with_manager()
    patch = ax.add_patch(
        PathPatch(Path([(0.2, 0.2), (0.8, 0.6), (0.2, 0.2)]))
    )
    fig.canvas.draw()
    adapter = get_artist_adapter(patch)
    model = adapter.point_handle_model()

    assert model.keys == (0, 1, 2)
    assert model.aliases_for(0) == (0,)
    assert model.aliases_for(2) == (2,)

    manager.selection.clear_targets()
    plt.close(fig)
    assert app is not None


def test_point_plan_rejects_stale_vertices_before_any_setter_or_recording():
    app, fig, ax, manager = _figure_with_manager()
    polygon = ax.add_patch(Polygon([(0.2, 0.2), (0.8, 0.2), (0.5, 0.8)]))
    fig.canvas.draw()
    source = PointEditSource.capture(polygon)
    destination = source.handle_model.positions_array()[1] + (8.0, 4.0)
    plan = PointEditPlan.preview(source, 1, destination)

    changed = polygon.get_xy().copy()
    changed[2] += (0.01, 0.0)
    polygon.set_xy(changed)
    recording_before = deepcopy(manager.figure.change_tracker.changes)
    live_before = polygon.get_xy().copy()

    with pytest.raises(StaleTransformPlanError, match="stale"):
        plan.commit()

    np.testing.assert_array_equal(polygon.get_xy(), live_before)
    assert manager.figure.change_tracker.changes == recording_before
    manager.selection.clear_targets()
    plt.close(fig)
    assert app is not None


def test_line2d_exposes_only_finite_outer_endpoints_and_preserves_masked_raw_payload():
    app, fig, ax, manager = _figure_with_manager()
    x = np.ma.array(
        [0.1, 0.3, 0.5, 0.8],
        mask=[False, True, False, False],
        fill_value=-123.5,
    )
    y = np.ma.array(
        [0.2, 0.4, np.nan, 0.7],
        mask=[False, True, False, False],
        fill_value=456.5,
    )
    line = ax.add_line(Line2D(x, y, marker="o"))
    fig.canvas.draw()
    adapter = get_artist_adapter(line)
    model = adapter.point_handle_model()

    assert model.keys == (0, 3)
    source = PointEditSource.capture(line, handle_model=model)
    destination = model.positions_array()[1] + np.array((-13.0, 11.0))
    raw_x_before = deepcopy(line.get_xdata(orig=True))
    raw_y_before = deepcopy(line.get_ydata(orig=True))
    plan = PointEditPlan.preview(source, 3, destination)

    np.testing.assert_equal(line.get_xdata(orig=True), raw_x_before)
    np.testing.assert_equal(line.get_ydata(orig=True), raw_y_before)
    assert plan.commit() is True
    raw_x_after = line.get_xdata(orig=True)
    raw_y_after = line.get_ydata(orig=True)
    np.testing.assert_array_equal(raw_x_after.mask, raw_x_before.mask)
    np.testing.assert_array_equal(raw_y_after.mask, raw_y_before.mask)
    assert raw_x_after.fill_value == raw_x_before.fill_value
    assert raw_y_after.fill_value == raw_y_before.fill_value
    assert raw_x_after.data[1] == raw_x_before.data[1]
    assert raw_y_after.data[1] == raw_y_before.data[1]
    np.testing.assert_allclose(
        adapter.control_points()[3], destination, atol=0.25, rtol=0
    )
    assert manager.figure.change_tracker.change[1].startswith(
        "._pylustrator_set_line_endpoints("
    )

    manager.selection.clear_targets()
    plt.close(fig)
    assert app is not None


def test_markerless_line_endpoint_preview_keeps_the_unedited_endpoint_segment():
    app, fig, ax, manager = _figure_with_manager()
    line = ax.add_line(Line2D([0.1, 0.5, 0.9], [0.2, 0.8, 0.3]))
    fig.canvas.draw()
    adapter = get_artist_adapter(line)
    source = PointEditSource.capture(line)
    key = source.handle_model.keys[-1]
    destination = source.handle_model.positions_array()[-1] + np.array((9.0, -6.0))
    plan = PointEditPlan.preview(source, key, destination)

    assert plan.commit() is True
    fig.canvas.draw()
    np.testing.assert_allclose(
        adapter.control_points()[key], destination, atol=0.25, rtol=0
    )
    np.testing.assert_allclose(
        adapter.selection_points(), plan.selection_array(), atol=0.25, rtol=0
    )

    manager.selection.clear_targets()
    plt.close(fig)
    assert app is not None


def test_zero_length_markerless_line_denies_endpoint_editing_at_preflight():
    app, fig, ax, manager = _figure_with_manager()
    line = ax.add_line(Line2D([0.26, 0.26], [4.17, 4.17], marker=""))
    fig.canvas.draw()
    adapter = get_artist_adapter(line)
    support = adapter.operation_support(TransformOperation.EDIT_POINTS)

    assert not support.supported
    assert not adapter.capabilities.can_edit_points
    assert "visible" in support.reason.lower()
    with pytest.raises(UnsupportedArtistError, match="visible"):
        adapter.point_handle_model()

    manager.selection.clear_targets()
    plt.close(fig)
    assert app is not None


def test_point_commit_uses_edit_points_capability_not_translate_capability():
    class PointOnlyPolygon(Polygon):
        pass

    class PointOnlyPolygonAdapter(PolygonAdapter):
        @classmethod
        def capabilities_for(cls, target):
            return replace(
                super().capabilities_for(target),
                can_translate=False,
                can_edit_points=True,
            )

    app, fig, ax, manager = _figure_with_manager()
    polygon = PointOnlyPolygon([(0.2, 0.2), (0.8, 0.2), (0.5, 0.8)])
    ax.add_patch(polygon)
    artist_adapter_registry.register(PointOnlyPolygon, PointOnlyPolygonAdapter)
    try:
        fig.canvas.draw()
        adapter = get_artist_adapter(polygon)
        assert not adapter.operation_support(TransformOperation.TRANSLATE).supported
        assert adapter.operation_support(TransformOperation.EDIT_POINTS).supported
        source = PointEditSource.capture(polygon)
        key = source.handle_model.keys[1]
        destination = source.handle_model.positions_array()[1] + (8.0, -5.0)
        plan = PointEditPlan.preview(source, key, destination)

        assert plan.commit() is True
        np.testing.assert_allclose(
            adapter.control_points()[key], destination, atol=0.25, rtol=0
        )
    finally:
        artist_adapter_registry.unregister(PointOnlyPolygon, PointOnlyPolygonAdapter)
        manager.selection.clear_targets()
        plt.close(fig)
    assert app is not None


def test_point_commit_drops_destination_canonicalized_to_native_noop():
    app, fig, ax, manager = _figure_with_manager()
    line = ax.add_line(
        Line2D(
            np.asarray([0.1, 0.5, 0.9], dtype=np.float32),
            np.asarray([0.2, 0.8, 0.3], dtype=np.float32),
            marker="o",
        )
    )
    fig.canvas.draw()
    source = PointEditSource.capture(line)
    key = source.handle_model.keys[0]
    destination = source.handle_model.positions_array()[0] + (1e-6, 0.0)
    before_x = line.get_xdata(orig=True).copy()
    before_y = line.get_ydata(orig=True).copy()
    recording_before = fig.change_tracker.capture_recording_state()
    plan = PointEditPlan.preview(source, key, destination)

    assert not plan.is_noop
    assert plan.commit() is False
    np.testing.assert_array_equal(line.get_xdata(orig=True), before_x)
    np.testing.assert_array_equal(line.get_ydata(orig=True), before_y)
    assert fig.change_tracker.capture_recording_state() == recording_before

    manager.selection.clear_targets()
    plt.close(fig)
    assert app is not None


def test_100k_line_point_model_is_two_handles_and_warm_preview_stays_bounded():
    app, fig, ax, manager = _figure_with_manager()
    count = 100_000
    x = np.linspace(0.05, 0.95, count)
    y = 0.5 + 0.2 * np.sin(np.linspace(0.0, 16.0, count))
    line = ax.plot(x, y, linewidth=0.8)[0]
    fig.canvas.draw()
    adapter = get_artist_adapter(line)
    model = adapter.point_handle_model()
    assert model.keys == (0, count - 1)
    assert model.path_array().shape == (0, 2)

    source = PointEditSource.capture(line, handle_model=model)
    destination = model.positions_array()[0] + np.array((3.0, -2.0))
    PointEditPlan.preview(source, 0, destination)
    samples = []
    for offset in np.linspace(0.0, 1.0, 25):
        start = perf_counter()
        PointEditPlan.preview(source, 0, destination + (offset, 0.0))
        samples.append((perf_counter() - start) * 1000.0)

    assert np.percentile(samples, 95) < 4.0
    assert source.control_array().nbytes <= count * 2 * 8
    manager.selection.clear_targets()
    plt.close(fig)
    assert app is not None


def test_100k_line_endpoint_commit_uses_compact_replay_and_bounded_release_time():
    app, fig, ax, manager = _figure_with_manager()
    count = 100_000
    x = np.linspace(0.05, 0.95, count)
    y = 0.5 + 0.2 * np.sin(np.linspace(0.0, 16.0, count))
    line = ax.plot(x, y, linewidth=0.8)[0]
    fig.canvas.draw()
    source = PointEditSource.capture(line)
    key = source.handle_model.keys[-1]
    destination = source.handle_model.positions_array()[-1] + np.array((3.0, -2.0))
    plan = PointEditPlan.preview(source, key, destination)

    started = perf_counter()
    assert plan.commit() is True
    release_ms = (perf_counter() - started) * 1000.0
    command = fig.change_tracker.change[1]

    assert release_ms < 80.0
    assert command.startswith("._pylustrator_set_line_endpoints(")
    assert len(command) < 512

    replay_fig, replay_ax = plt.subplots(figsize=(4, 3), dpi=100)
    replay_line = replay_ax.plot(x, y, linewidth=0.8)[0]
    ensure_line_endpoint_replay_api(replay_line)
    exec(f"target{command}", {"target": replay_line})
    np.testing.assert_allclose(
        replay_line.get_xydata(), line.get_xydata(), atol=1e-15, rtol=0
    )

    manager.selection.clear_targets()
    plt.close(replay_fig)
    plt.close(fig)
    assert app is not None


def test_line_endpoint_replay_binding_is_figure_local_and_skips_managed_lines():
    fig1, ax1 = plt.subplots(figsize=(4, 3), dpi=100)
    source1 = ax1.plot([0.1, 0.5, 0.9], [0.2, 0.8, 0.3], label="source")[0]
    legend = ax1.legend()
    fig1.canvas.draw()
    legend_proxy = legend.legend_handles[0]
    tick_line = ax1.xaxis.get_major_ticks()[0].tick1line
    fig2, ax2 = plt.subplots(figsize=(4, 3), dpi=100)
    source2 = ax2.plot([0.1, 0.5, 0.9], [0.3, 0.4, 0.6])[0]

    install_line_endpoint_replay_api(fig1)

    replay_attribute = "_pylustrator_set_line_endpoints"
    assert replay_attribute in vars(source1)
    assert replay_attribute not in vars(source2)
    assert replay_attribute not in vars(legend_proxy)
    assert replay_attribute not in vars(tick_line)
    assert replay_attribute not in vars(Line2D)

    install_line_endpoint_replay_api(fig2)
    plt.close(fig1)
    source2._pylustrator_set_line_endpoints(
        (0, 2),
        ((0.2, 0.25), (0.8, 0.75)),
    )
    np.testing.assert_allclose(
        source2.get_xydata()[[0, 2]],
        ((0.2, 0.25), (0.8, 0.75)),
        atol=1e-15,
        rtol=0,
    )
    assert replay_attribute not in vars(Line2D)

    plt.close(fig2)


def test_line_endpoint_replay_method_cycle_does_not_retain_figure_or_line():
    fig = matplotlib.figure.Figure(figsize=(4, 3), dpi=100)
    ax = fig.subplots()
    line = ax.plot([0.1, 0.9], [0.2, 0.8])[0]
    ensure_line_endpoint_replay_api(line)
    figure_ref = weakref.ref(fig)
    line_ref = weakref.ref(line)

    del line
    del ax
    del fig
    gc.collect()

    assert line_ref() is None
    assert figure_ref() is None
    assert "_pylustrator_set_line_endpoints" not in vars(Line2D)


def test_line_endpoint_replay_binding_has_no_render_or_recording_side_effect(
    monkeypatch,
):
    fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
    line = ax.plot([0.1, 0.9], [0.2, 0.8])[0]
    tracker = type("Tracker", (), {})()
    tracker.changes = {(line, ".set_visible"): (line, ".set_visible(True)")}
    tracker.saved = False
    fig.change_tracker = tracker
    draw_calls = []
    monkeypatch.setattr(fig.canvas, "draw", lambda: draw_calls.append("draw"))
    monkeypatch.setattr(
        fig.canvas,
        "draw_idle",
        lambda: draw_calls.append("draw_idle"),
    )
    line.stale = False
    fig.stale = False
    changes_before = dict(tracker.changes)

    ensure_line_endpoint_replay_api(line)

    assert draw_calls == []
    assert line.stale is False
    assert fig.stale is False
    assert tracker.changes == changes_before
    assert tracker.saved is False

    plt.close(fig)


def test_line_endpoint_replay_class_collision_only_denies_point_editing(
    monkeypatch,
):
    replay_attribute = "_pylustrator_set_line_endpoints"
    third_party_attribute = object()
    monkeypatch.setattr(
        Line2D,
        replay_attribute,
        third_party_attribute,
        raising=False,
    )
    fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
    line = ax.plot([0.1, 0.9], [0.2, 0.8])[0]

    init_figure(fig)
    adapter = get_artist_adapter(line)
    assert adapter.operation_support(TransformOperation.TRANSLATE).supported
    edit_support = adapter.operation_support(TransformOperation.EDIT_POINTS)
    assert not edit_support.supported
    assert "class attribute" in edit_support.reason
    with pytest.raises(LineEndpointReplayConflictError) as error:
        ensure_line_endpoint_replay_api(line)
    assert error.value.scope == "class"
    assert vars(Line2D)[replay_attribute] is third_party_attribute
    assert replay_attribute not in vars(line)

    plt.close(fig)


def test_line_endpoint_replay_instance_collision_is_atomic_and_not_overwritten():
    replay_attribute = "_pylustrator_set_line_endpoints"
    fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
    clean, occupied = ax.plot(
        [0.1, 0.9],
        [[0.2, 0.8], [0.3, 0.7]],
    )
    third_party_attribute = object()
    vars(occupied)[replay_attribute] = third_party_attribute

    with pytest.raises(LineEndpointReplayConflictError) as error:
        install_line_endpoint_replay_api(fig)

    assert error.value.scope == "instance"
    assert replay_attribute not in vars(clean)
    assert vars(occupied)[replay_attribute] is third_party_attribute
    adapter = get_artist_adapter(occupied)
    assert adapter.operation_support(TransformOperation.TRANSLATE).supported
    assert not adapter.operation_support(TransformOperation.EDIT_POINTS).supported
    assert replay_attribute not in vars(Line2D)

    plt.close(fig)


def test_line_sparse_point_history_restores_integer_storage_promotion():
    app, fig, ax, manager = _figure_with_manager()
    line = ax.add_line(
        Line2D(
            np.asarray([0, 1, 2], dtype=np.int32),
            np.asarray([0, 2, 1], dtype=np.int32),
            marker="o",
        )
    )
    fig.canvas.draw()
    manager.set_selection_mode(SelectionMode.DIRECT)
    manager.select_element(line)
    selection = manager.selection
    model = selection.direct_point_model
    key = model.keys[-1]
    start = model.positions_array()[-1]
    before_x = line.get_xdata(orig=True).copy()
    before_y = line.get_ydata(orig=True).copy()
    compact = get_artist_adapter(line).point_edit_history_snapshot((key,))
    assert compact.xaxis.fallback is None
    assert compact.yaxis.fallback is None
    assert len(compact.xaxis.values) <= 2
    assert len(compact.yaxis.values) <= 2

    destination = start + np.array((7.0, -5.0))
    manager.button_press_event0(
        MouseEvent("button_press_event", fig.canvas, *start, button=1)
    )
    selection.direct_point_editor.on_motion(
        MouseEvent("motion_notify_event", fig.canvas, *destination, button=1)
    )
    manager.button_release_event0(
        MouseEvent("button_release_event", fig.canvas, *destination, button=1)
    )
    after_x = line.get_xdata(orig=True).copy()
    after_y = line.get_ydata(orig=True).copy()
    assert after_x.dtype.kind == "f" or after_y.dtype.kind == "f"

    undo, redo = fig.change_tracker.edit[:2]
    undo()
    np.testing.assert_array_equal(line.get_xdata(orig=True), before_x)
    np.testing.assert_array_equal(line.get_ydata(orig=True), before_y)
    assert line.get_xdata(orig=True).dtype == np.dtype(np.int32)
    assert line.get_ydata(orig=True).dtype == np.dtype(np.int32)
    redo()
    np.testing.assert_array_equal(line.get_xdata(orig=True), after_x)
    np.testing.assert_array_equal(line.get_ydata(orig=True), after_y)

    manager.selection.clear_targets()
    plt.close(fig)
    assert app is not None


def test_line_endpoint_replay_composes_after_and_is_superseded_by_full_data():
    class DictTracker:
        def __init__(self):
            self.changes = {}
            self.saved = True

        def addChange(self, target, command):
            key = command.split("(", 1)[0]
            self.changes[target, key] = (target, command)
            self.saved = False

        def capture_recording_state(self):
            return dict(self.changes), self.saved

        def restore_recording_state(self, state):
            self.changes, self.saved = dict(state[0]), bool(state[1])

    app, fig, ax, manager = _figure_with_manager()
    original_x = np.asarray([0.1, 0.5, 0.9], dtype=float)
    original_y = np.asarray([0.2, 0.8, 0.3], dtype=float)
    line = ax.add_line(Line2D(original_x, original_y, marker="o"))
    fig.canvas.draw()
    tracker = DictTracker()
    fig.change_tracker = tracker
    adapter = get_artist_adapter(line)

    adapter.translate((4.0, -3.0))
    assert (line, ".set_data") in tracker.changes
    source = PointEditSource.capture(line)
    key = source.handle_model.keys[-1]
    destination = source.handle_model.positions_array()[-1] + (6.0, 2.0)
    assert PointEditPlan.preview(source, key, destination).commit()
    assert (line, "._pylustrator_set_line_endpoints") in tracker.changes

    replay_fig, replay_ax = plt.subplots(figsize=(4, 3), dpi=100)
    replay_line = replay_ax.add_line(Line2D(original_x, original_y, marker="o"))
    ensure_line_endpoint_replay_api(replay_line)
    for command_key in (".set_data", "._pylustrator_set_line_endpoints"):
        command = tracker.changes[line, command_key][1]
        exec(f"target{command}", {"target": replay_line, "np": np})
    np.testing.assert_allclose(
        replay_line.get_xydata(), line.get_xydata(), atol=1e-15, rtol=0
    )

    adapter.translate((-2.0, 1.0))
    assert (line, ".set_data") in tracker.changes
    assert (line, "._pylustrator_set_line_endpoints") not in tracker.changes

    manager.selection.clear_targets()
    plt.close(replay_fig)
    plt.close(fig)
    assert app is not None


def test_saved_sparse_line_replay_bootstraps_only_its_exact_target(
    monkeypatch,
):
    from pylustrator.change_tracker import ChangeTracker as RealChangeTracker
    import pylustrator.change_tracker as change_tracker_module

    app, fig, ax, manager = _figure_with_manager()
    line = ax.plot([0.1, 0.5, 0.9], [0.2, 0.8, 0.3], marker="o")[0]
    untouched = ax.plot([0.1, 0.9], [0.7, 0.4])[0]
    fig.canvas.draw()
    tracker = RealChangeTracker.__new__(RealChangeTracker)
    tracker.figure = fig
    tracker.changes = {}
    tracker.saved = True
    tracker.edits = []
    tracker.last_edit = -1
    tracker.update_changes_signal = None
    tracker.no_save = False
    fig.change_tracker = tracker
    source = PointEditSource.capture(line)
    key = source.handle_model.keys[-1]
    destination = source.handle_model.positions_array()[-1] + (4.0, -3.0)
    assert PointEditPlan.preview(source, key, destination).commit()
    saved = {}
    monkeypatch.setattr(change_tracker_module, "getTextFromFile", lambda *_args: [])
    monkeypatch.setattr(
        change_tracker_module,
        "stack_position",
        object(),
        raising=False,
    )
    monkeypatch.setattr(
        change_tracker_module,
        "insertTextToFile",
        lambda output, *_args: saved.setdefault("lines", list(output)),
    )

    tracker.save()

    generated = saved["lines"]
    replay_index = next(
        index
        for index, line_source in enumerate(generated)
        if "._pylustrator_set_line_endpoints(" in line_source
    )
    ensure_source = generated[replay_index - 1]
    replay_source = generated[replay_index]
    assert ensure_source.startswith(
        "_pylustrator_ensure_line_endpoint_replay_api("
    )
    assert ensure_source.removeprefix(
        "_pylustrator_ensure_line_endpoint_replay_api("
    ).removesuffix(")") == replay_source.split(
        "._pylustrator_set_line_endpoints(", 1
    )[0]
    assert sum(
        "_pylustrator_ensure_line_endpoint_replay_api(" in line_source
        for line_source in generated
    ) == 1
    assert any(
        line_source
        == "from pylustrator.commands import ensure_line_endpoint_replay_api as "
        "_pylustrator_ensure_line_endpoint_replay_api"
        for line_source in generated
    )
    assert "_pylustrator_set_line_endpoints" not in vars(Line2D)
    assert "_pylustrator_set_line_endpoints" not in vars(untouched)

    manager.selection.clear_targets()
    expected = line.get_xydata().copy()
    figure_number = fig.number
    plt.close(fig)
    replay_fig, replay_ax = plt.subplots(
        num=figure_number,
        clear=True,
        figsize=(4, 3),
        dpi=100,
    )
    replay_line = replay_ax.plot(
        [0.1, 0.5, 0.9],
        [0.2, 0.8, 0.3],
        marker="o",
    )[0]
    replay_untouched = replay_ax.plot([0.1, 0.9], [0.7, 0.4])[0]
    exec("\n".join(generated), {"plt": plt})
    np.testing.assert_allclose(
        replay_line.get_xydata(),
        expected,
        atol=1e-15,
        rtol=0,
    )
    assert "_pylustrator_set_line_endpoints" in vars(replay_line)
    assert "_pylustrator_set_line_endpoints" not in vars(replay_untouched)
    assert "_pylustrator_set_line_endpoints" not in vars(Line2D)
    plt.close(replay_fig)
    assert app is not None


def test_direct_point_overlay_drag_previews_without_setter_then_commits_one_history_item(
    monkeypatch,
):
    app, fig, ax, manager = _figure_with_manager()
    polygon = ax.add_patch(Polygon([(0.2, 0.2), (0.8, 0.2), (0.5, 0.75)]))
    fig.canvas.draw()
    manager.set_selection_mode(SelectionMode.DIRECT)
    manager.select_element(polygon)
    selection = manager.selection
    model = selection.direct_point_model
    assert model is not None
    assert selection._direct_point_overlay_visible
    assert len(selection._direct_point_overlay_items) == 3
    assert all(
        np.allclose(
            (
                grabber.ellipse.rect().center().x(),
                grabber.ellipse.rect().center().y(),
            ),
            (-100, -100),
        )
        for grabber in selection.grabbers
    )
    assert not selection.rotation_grabber.handle.isVisible()

    key = model.keys[1]
    start = model.positions_array()[1]
    before = polygon.get_xy().copy()
    press = MouseEvent("button_press_event", fig.canvas, *start, button=1)

    def forbidden_hit(*_args, **_kwargs):
        raise AssertionError("anchor hits must not re-enter the Artist hit stack")

    monkeypatch.setattr(manager, "_resolve_top_hit", forbidden_hit)
    manager.button_press_event0(press)
    assert manager.grab_element is selection.direct_point_editor
    assert selection.direct_point_editor.got_artist

    destination = start + np.array((19.0, -11.0))
    motion = MouseEvent("motion_notify_event", fig.canvas, *destination, button=1)
    selection.direct_point_editor.on_motion(motion)
    np.testing.assert_array_equal(polygon.get_xy(), before)
    assert fig.change_tracker.changes == []
    preview = polygon._pylustrator_preview_positions
    np.testing.assert_allclose(preview[key], destination, atol=0.25, rtol=0)

    release = MouseEvent("button_release_event", fig.canvas, *destination, button=1)
    manager.button_release_event0(release)
    assert not hasattr(polygon, "_pylustrator_preview_positions")
    np.testing.assert_allclose(
        get_artist_adapter(polygon).control_points()[key],
        destination,
        atol=0.25,
        rtol=0,
    )
    assert len(fig.change_tracker.edits) == 1
    assert fig.change_tracker.edit[2] == "Edit points"
    assert fig.change_tracker.change_count == 1
    assert selection.direct_point_editor.active_key == key

    fig.change_tracker.edit[0]()
    np.testing.assert_allclose(polygon.get_xy(), before, atol=1e-15, rtol=0)
    assert selection.direct_point_editor.active_key == key
    fig.change_tracker.edit[1]()
    np.testing.assert_allclose(
        get_artist_adapter(polygon).control_points()[key],
        destination,
        atol=0.25,
        rtol=0,
    )
    assert selection.direct_point_editor.active_key == key

    manager.selection.clear_targets()
    plt.close(fig)
    assert app is not None


def _start_two_anchor_point_drag(fig, manager, polygon):
    manager.set_selection_mode(SelectionMode.DIRECT)
    manager.select_element(polygon)
    selection = manager.selection
    editor = selection.direct_point_editor
    model = selection.direct_point_model
    selected_keys = tuple(model.keys[:2])
    active_key = selected_keys[-1]
    selection.restore_direct_point_selection(
        selected_keys,
        primary=active_key,
    )
    start = model.positions_array()[model.keys.index(active_key)]
    manager.button_press_event0(
        MouseEvent("button_press_event", fig.canvas, *start, button=1)
    )
    destination = start + np.array((13.0, -7.0))
    editor.on_motion(
        MouseEvent("motion_notify_event", fig.canvas, *destination, button=1)
    )
    assert editor.plan is not None
    assert editor.selected_keys == selected_keys
    assert editor.active_key == active_key
    return selection, editor, selected_keys, active_key, destination


def _assert_failed_point_release_cleanup(
    polygon,
    geometry_before,
    manager,
    editor,
    selected_keys,
    active_key,
):
    np.testing.assert_allclose(
        polygon.get_xy(), geometry_before, atol=1e-15, rtol=0
    )
    assert not hasattr(polygon, "_pylustrator_preview_positions")
    assert not hasattr(polygon, "_pylustrator_preview_selection_points")
    assert editor.source is None
    assert editor.plan is None
    assert editor.last_error is None
    assert not editor.moved
    assert not editor.got_artist
    assert manager.grab_element is None
    assert editor.active_key == active_key
    assert editor.selected_keys == selected_keys
    assert [target.target for target in manager.selection.targets] == [polygon]
    assert manager.selected_element is polygon


def test_direct_point_first_save_point_failure_cleans_precommit_gesture(
    monkeypatch,
):
    app, fig, ax, manager = _figure_with_manager()
    polygon = ax.add_patch(
        Polygon([(0.2, 0.2), (0.8, 0.2), (0.5, 0.75)])
    )
    fig.canvas.draw()
    geometry_before = polygon.get_xy().copy()
    tracker = fig.change_tracker
    tracker.changes = [(polygon, ".set_visible(True)")]
    tracker.saved = False
    existing_edit = [lambda: None, lambda: None, "Existing edit"]
    tracker.edits = [existing_edit]
    tracker.last_edit = 0
    recording_before = tracker.capture_recording_state()
    history_before = (list(tracker.edits), tracker.last_edit)
    selection, editor, selected_keys, active_key, destination = (
        _start_two_anchor_point_drag(fig, manager, polygon)
    )
    calls = 0

    def fail_first_save_point(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        raise RuntimeError("injected first save-point failure")

    monkeypatch.setattr(selection, "get_save_point", fail_first_save_point)
    with pytest.raises(RuntimeError, match="injected first save-point failure"):
        manager.button_release_event0(
            MouseEvent("button_release_event", fig.canvas, *destination, button=1)
        )

    assert calls == 1
    assert tracker.capture_recording_state() == recording_before
    assert (tracker.edits, tracker.last_edit) == history_before
    _assert_failed_point_release_cleanup(
        polygon,
        geometry_before,
        manager,
        editor,
        selected_keys,
        active_key,
    )

    manager.selection.clear_targets()
    plt.close(fig)
    assert app is not None


def test_direct_point_recording_capture_failure_cleans_precommit_gesture(
    monkeypatch,
):
    app, fig, ax, manager = _figure_with_manager()
    polygon = ax.add_patch(
        Polygon([(0.2, 0.2), (0.8, 0.2), (0.5, 0.75)])
    )
    fig.canvas.draw()
    geometry_before = polygon.get_xy().copy()
    tracker = fig.change_tracker
    tracker.changes = [(polygon, ".set_visible(True)")]
    tracker.saved = False
    existing_edit = [lambda: None, lambda: None, "Existing edit"]
    tracker.edits = [existing_edit]
    tracker.last_edit = 0
    original_capture = tracker.capture_recording_state
    original_get_save_point = manager.selection.get_save_point
    recording_before = original_capture()
    history_before = (list(tracker.edits), tracker.last_edit)
    selection, editor, selected_keys, active_key, destination = (
        _start_two_anchor_point_drag(fig, manager, polygon)
    )
    capture_calls = 0
    save_point_calls = 0
    restore_calls = 0

    def counted_get_save_point(*args, **kwargs):
        nonlocal save_point_calls, restore_calls
        save_point_calls += 1
        restore = original_get_save_point(*args, **kwargs)

        def counted_restore():
            nonlocal restore_calls
            restore_calls += 1
            restore()

        return counted_restore

    def fail_second_capture():
        nonlocal capture_calls
        capture_calls += 1
        if capture_calls == 2:
            raise RuntimeError("injected recording capture failure")
        return original_capture()

    monkeypatch.setattr(selection, "get_save_point", counted_get_save_point)
    monkeypatch.setattr(tracker, "capture_recording_state", fail_second_capture)
    with pytest.raises(RuntimeError, match="injected recording capture failure"):
        manager.button_release_event0(
            MouseEvent("button_release_event", fig.canvas, *destination, button=1)
        )

    assert capture_calls == 2
    assert save_point_calls == 1
    assert restore_calls == 0
    assert original_capture() == recording_before
    assert (tracker.edits, tracker.last_edit) == history_before
    _assert_failed_point_release_cleanup(
        polygon,
        geometry_before,
        manager,
        editor,
        selected_keys,
        active_key,
    )

    manager.selection.clear_targets()
    plt.close(fig)
    assert app is not None


def test_direct_point_late_history_failure_rolls_back_document_and_gesture(
    monkeypatch,
):
    app, fig, ax, manager = _figure_with_manager()
    polygon = ax.add_patch(Polygon([(0.2, 0.2), (0.8, 0.2), (0.5, 0.75)]))
    fig.canvas.draw()
    manager.set_selection_mode(SelectionMode.DIRECT)
    manager.select_element(polygon)
    selection = manager.selection
    editor = selection.direct_point_editor
    model = selection.direct_point_model
    key = model.keys[1]
    start = model.positions_array()[1]
    geometry_before = polygon.get_xy().copy()
    tracker = fig.change_tracker
    recording_before = tracker.capture_recording_state()
    history_before = (list(tracker.edits), tracker.last_edit)

    manager.button_press_event0(
        MouseEvent("button_press_event", fig.canvas, *start, button=1)
    )
    destination = start + np.array((13.0, -7.0))
    editor.on_motion(
        MouseEvent("motion_notify_event", fig.canvas, *destination, button=1)
    )

    def fail_add_edit(_edit):
        raise RuntimeError("injected addEdit failure")

    monkeypatch.setattr(tracker, "addEdit", fail_add_edit)
    with pytest.raises(RuntimeError, match="injected addEdit failure"):
        manager.button_release_event0(
            MouseEvent("button_release_event", fig.canvas, *destination, button=1)
        )

    np.testing.assert_allclose(polygon.get_xy(), geometry_before, atol=1e-15, rtol=0)
    assert tracker.capture_recording_state() == recording_before
    assert (tracker.edits, tracker.last_edit) == history_before
    assert not hasattr(polygon, "_pylustrator_preview_positions")
    assert editor.source is None
    assert editor.plan is None
    assert not editor.got_artist
    assert manager.grab_element is None
    assert editor.active_key == key
    assert editor.selected_keys == (key,)

    manager.selection.clear_targets()
    plt.close(fig)
    assert app is not None


def test_direct_point_undo_redo_restores_anchor_after_production_draw_refresh():
    app, fig, ax, manager = _figure_with_manager()
    polygon = ax.add_patch(Polygon([(0.2, 0.2), (0.8, 0.2), (0.5, 0.75)]))
    fig.canvas.draw()
    manager.activate()
    manager.set_selection_mode(SelectionMode.DIRECT)
    manager.select_element(polygon)
    fig.canvas.draw()
    selection = manager.selection
    model = selection.direct_point_model
    key = model.keys[1]
    start = model.positions_array()[1]
    destination = start + np.array((11.0, -6.0))

    manager.button_press_event0(
        MouseEvent("button_press_event", fig.canvas, *start, button=1)
    )
    selection.direct_point_editor.on_motion(
        MouseEvent("motion_notify_event", fig.canvas, *destination, button=1)
    )
    manager.button_release_event0(
        MouseEvent("button_release_event", fig.canvas, *destination, button=1)
    )

    undo, redo = fig.change_tracker.edit[:2]
    for restore in (undo, redo):
        restore()
        fig.canvas.draw()
        assert [target.target for target in selection.targets] == [polygon]
        assert selection._direct_point_overlay_visible
        assert selection.direct_point_editor.active_key == key
        assert selection.direct_point_editor.selected_keys == (key,)

    manager.deactivate(redraw=False)
    plt.close(fig)
    assert app is not None


def test_escape_cancels_point_preview_but_keeps_parent_selection_and_anchor():
    app, fig, ax, manager = _figure_with_manager()
    line = ax.plot([0.2, 0.8], [0.3, 0.7])[0]
    fig.canvas.draw()
    manager.set_selection_mode(SelectionMode.DIRECT)
    manager.select_element(line)
    selection = manager.selection
    model = selection.direct_point_model
    key = model.keys[0]
    start = model.positions_array()[0]
    before_x = line.get_xdata(orig=True).copy()
    before_y = line.get_ydata(orig=True).copy()

    manager.button_press_event0(
        MouseEvent("button_press_event", fig.canvas, *start, button=1)
    )
    selection.direct_point_editor.on_motion(
        MouseEvent(
            "motion_notify_event",
            fig.canvas,
            *(start + (14.0, 8.0)),
            button=1,
        )
    )
    assert hasattr(line, "_pylustrator_preview_positions")

    manager.key_press_event(KeyEvent("key_press_event", fig.canvas, "escape"))

    np.testing.assert_array_equal(line.get_xdata(orig=True), before_x)
    np.testing.assert_array_equal(line.get_ydata(orig=True), before_y)
    assert not hasattr(line, "_pylustrator_preview_positions")
    assert [target.target for target in selection.targets] == [line]
    assert manager.selected_element is line
    assert selection.direct_point_editor.active_key == key
    assert fig.change_tracker.edits == []

    manager.selection.clear_targets()
    plt.close(fig)
    assert app is not None


def test_shift_click_adds_anchor_and_one_drag_moves_the_selected_point_set():
    app, fig, ax, manager = _figure_with_manager()
    polygon = ax.add_patch(Polygon([(0.2, 0.2), (0.8, 0.2), (0.5, 0.8)]))
    fig.canvas.draw()
    manager.set_selection_mode(SelectionMode.DIRECT)
    manager.select_element(polygon)
    selection = manager.selection
    model = selection.direct_point_model
    first_key, second_key = model.keys[:2]
    first, second = model.positions_array()[:2]

    manager.button_press_event0(
        MouseEvent("button_press_event", fig.canvas, *first, button=1)
    )
    manager.button_release_event0(
        MouseEvent("button_release_event", fig.canvas, *first, button=1)
    )
    assert selection.direct_point_editor.selected_keys == (first_key,)
    assert fig.change_tracker.edits == []

    manager.button_press_event0(
        MouseEvent(
            "button_press_event",
            fig.canvas,
            *second,
            button=1,
            key="shift",
        )
    )
    assert selection.direct_point_editor.selected_keys == (
        first_key,
        second_key,
    )
    delta = np.array((15.0, 6.0))
    selection.direct_point_editor.on_motion(
        MouseEvent(
            "motion_notify_event",
            fig.canvas,
            *(second + delta),
            button=1,
        )
    )
    manager.button_release_event0(
        MouseEvent(
            "button_release_event",
            fig.canvas,
            *(second + delta),
            button=1,
        )
    )

    controls = np.asarray(get_artist_adapter(polygon).control_points())
    np.testing.assert_allclose(controls[first_key], first + delta, atol=0.25, rtol=0)
    np.testing.assert_allclose(controls[second_key], second + delta, atol=0.25, rtol=0)
    assert selection.direct_point_editor.selected_keys == (
        first_key,
        second_key,
    )
    assert len(fig.change_tracker.edits) == 1
    assert fig.change_tracker.edit[2] == "Edit points"

    manager.selection.clear_targets()
    plt.close(fig)
    assert app is not None


def test_v_a_switch_hides_and_restores_batched_point_overlay():
    app, fig, ax, manager = _figure_with_manager()
    patch = ax.add_patch(
        PathPatch(
            Path(
                [(0.2, 0.2), (0.8, 0.2), (0.5, 0.8), (0.0, 0.0)],
                [Path.MOVETO, Path.LINETO, Path.LINETO, Path.CLOSEPOLY],
            )
        )
    )
    fig.canvas.draw()
    manager.set_selection_mode(SelectionMode.DIRECT)
    manager.select_element(patch)
    selection = manager.selection
    item_ids = tuple(id(item) for item in selection._direct_point_overlay_items)
    assert selection._direct_point_overlay_visible

    manager.set_selection_mode(SelectionMode.OBJECT)
    assert not selection._direct_point_overlay_visible
    assert all(not item.isVisible() for item in selection._direct_point_overlay_items)

    manager.set_selection_mode(SelectionMode.DIRECT)
    assert selection._direct_point_overlay_visible
    assert tuple(id(item) for item in selection._direct_point_overlay_items) == item_ids

    manager.selection.clear_targets()
    plt.close(fig)
    assert app is not None


def test_unchanged_large_direct_overlay_skips_qt_path_rebuild_on_draw(monkeypatch):
    app, fig, ax, manager = _figure_with_manager()
    angles = np.linspace(0.0, 2.0 * np.pi, 1024, endpoint=False)
    polygon = ax.add_patch(
        Polygon(
            np.column_stack(
                (0.5 + 0.35 * np.cos(angles), 0.5 + 0.35 * np.sin(angles))
            ),
            closed=True,
        )
    )
    fig.canvas.draw()
    manager.activate()
    manager.set_selection_mode(SelectionMode.DIRECT)
    manager.select_element(polygon)
    fig.canvas.draw()
    selection = manager.selection
    assert len(selection.direct_point_model.keys) == 1024
    real_render = selection._render_direct_point_model
    render_calls = []

    def counted_render():
        render_calls.append(True)
        return real_render()

    monkeypatch.setattr(selection, "_render_direct_point_model", counted_render)
    samples = []
    for _ in range(12):
        started = perf_counter()
        assert selection.refresh_direct_point_overlay(force=True)
        samples.append((perf_counter() - started) * 1000.0)
    fig.canvas.draw()
    fig.canvas.draw()

    assert render_calls == []
    assert np.percentile(samples, 95) < 4.0

    moved = polygon.get_xy().copy()
    moved[17, 0] += 0.01
    polygon.set_xy(moved)
    fig.canvas.draw()
    assert len(render_calls) == 1

    manager.deactivate(redraw=False)
    plt.close(fig)
    assert app is not None


def test_permanent_manager_dispose_removes_direct_overlay_scene_tree():
    app, fig, ax, manager = _figure_with_manager()
    polygon = ax.add_patch(Polygon([(0.2, 0.2), (0.8, 0.2), (0.5, 0.75)]))
    fig.canvas.draw()
    manager.activate()
    manager.set_selection_mode(SelectionMode.DIRECT)
    manager.select_element(polygon)
    selection = manager.selection
    scene = selection.graphics_scene.scene()
    overlay_items = tuple(selection._direct_point_overlay_items)
    count_before = len(scene.items())
    assert len(overlay_items) == 3
    assert all(item.scene() is scene for item in overlay_items)

    assert manager.dispose(redraw=False) is True

    assert manager.dispose(redraw=False) is False
    assert selection._direct_point_overlay_items == []
    for item in overlay_items:
        try:
            assert item.scene() is None
        except RuntimeError:
            # Deleting the detached parent may eagerly delete its Qt children.
            pass
    assert len(scene.items()) <= count_before - 3
    assert fig.figure_dragger is None
    assert fig.selection is None
    with pytest.raises(RuntimeError, match="disposed"):
        manager.activate()

    plt.close(fig)
    assert app is not None


def test_annotation_exposes_two_mixed_coordinate_anchors_with_exact_arrow_preview():
    app, fig, ax, manager = _figure_with_manager()
    annotation = ax.annotate(
        "reference",
        xy=(0.25, 0.3),
        xycoords="data",
        xytext=(0.72, 0.78),
        textcoords="axes fraction",
        arrowprops={"arrowstyle": "->", "connectionstyle": "arc3"},
        annotation_clip=False,
    )
    fig.canvas.draw()
    adapter = get_artist_adapter(annotation)
    model = adapter.point_handle_model()
    assert model.keys == (0, 1)
    source = PointEditSource.capture(annotation, handle_model=model)
    before_position = annotation.get_position()
    before_xy = annotation.xy
    destination = model.positions_array()[1] + np.array((12.0, -7.0))

    samples = []
    plan = None
    for offset in np.linspace(0.0, 1.0, 30):
        started = perf_counter()
        plan = PointEditPlan.preview(source, 1, destination + np.array((offset, 0.0)))
        samples.append((perf_counter() - started) * 1000.0)
    assert samples[0] < 16.7
    assert np.percentile(samples[10:], 95) < 4.0
    assert annotation.get_position() == before_position
    assert annotation.xy == before_xy
    assert plan.commit()
    fig.canvas.draw()

    np.testing.assert_allclose(
        adapter.control_points()[1],
        plan.destination_array()[0],
        atol=0.25,
        rtol=0,
    )
    np.testing.assert_allclose(
        adapter.selection_points(),
        plan.selection_array(),
        atol=0.25,
        rtol=0,
    )
    assert annotation.get_position() == before_position
    assert annotation.xy != before_xy
    assert fig.change_tracker.text_change_count == 1
    assert fig.change_tracker.change[1].startswith(".xy = ")

    manager.selection.clear_targets()
    plt.close(fig)
    assert app is not None


def test_annotation_point_capture_and_preview_isolate_live_arrow_state():
    app, fig, ax, manager = _figure_with_manager()
    annotation = ax.annotate(
        "isolated",
        xy=(0.25, 0.3),
        xycoords="data",
        xytext=(0.72, 0.78),
        textcoords="axes fraction",
        arrowprops={"arrowstyle": "->", "connectionstyle": "arc3"},
        annotation_clip=False,
    )
    fig.canvas.draw()
    live_arrow = annotation.arrow_patch
    live_positions = [
        np.asarray(position, dtype=float).copy()
        for position in live_arrow._posA_posB
    ]
    live_path = live_arrow.get_path()
    live_vertices = live_path.vertices.copy()
    live_codes = None if live_path.codes is None else live_path.codes.copy()
    stale_before = tuple(
        getattr(owner, "_stale", None)
        for owner in (annotation, live_arrow, ax, fig)
    )
    native_before = (annotation.get_position(), annotation.xy)

    source = PointEditSource.capture(annotation)
    preview_arrow = source.preview_context.target.arrow_patch

    assert preview_arrow is not live_arrow
    assert preview_arrow._posA_posB is not live_arrow._posA_posB
    assert all(
        preview is not live
        for preview, live in zip(preview_arrow._posA_posB, live_arrow._posA_posB)
    )
    assert preview_arrow.stale_callback is None
    assert preview_arrow._remove_method is None
    assert source.preview_context.target.stale_callback is None
    assert source.preview_context.target._remove_method is None
    np.testing.assert_array_equal(live_arrow.get_path().vertices, live_vertices)
    if live_codes is not None:
        np.testing.assert_array_equal(live_arrow.get_path().codes, live_codes)
    assert stale_before == tuple(
        getattr(owner, "_stale", None)
        for owner in (annotation, live_arrow, ax, fig)
    )

    destination = source.handle_model.positions_array()[1] + np.array((20.0, -13.0))
    PointEditPlan.preview(source, 1, destination)

    for actual, expected in zip(live_arrow._posA_posB, live_positions):
        np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(live_arrow.get_path().vertices, live_vertices)
    if live_codes is not None:
        np.testing.assert_array_equal(live_arrow.get_path().codes, live_codes)
    assert annotation.get_position() == native_before[0]
    assert annotation.xy == native_before[1]
    assert stale_before == tuple(
        getattr(owner, "_stale", None)
        for owner in (annotation, live_arrow, ax, fig)
    )

    manager.selection.clear_targets()
    plt.close(fig)
    assert app is not None


def test_path_topology_change_rejects_frozen_plan_before_mutation():
    app, fig, ax, manager = _figure_with_manager()
    path = Path(
        [(0.2, 0.2), (0.8, 0.2), (0.5, 0.8), (0.0, 0.0)],
        [Path.MOVETO, Path.LINETO, Path.LINETO, Path.CLOSEPOLY],
    )
    patch = ax.add_patch(PathPatch(path))
    fig.canvas.draw()
    source = PointEditSource.capture(patch)
    plan = PointEditPlan.preview(
        source,
        1,
        source.handle_model.positions_array()[1] + (6.0, -4.0),
    )
    changed_codes = path.codes.copy()
    changed_codes[2] = Path.MOVETO
    patch.set_path(Path(path.vertices.copy(), changed_codes))
    before = patch.get_path().vertices.copy()

    with pytest.raises(StaleTransformPlanError, match="topology|stale"):
        plan.commit()

    np.testing.assert_array_equal(patch.get_path().vertices, before)
    assert fig.change_tracker.changes == []
    manager.selection.clear_targets()
    plt.close(fig)
    assert app is not None


def test_path_handle_budget_and_legend_owned_line_are_typed_denials():
    app, fig, ax, manager = _figure_with_manager()
    dense = ax.add_patch(
        PathPatch(
            Path(
                np.column_stack(
                    (
                        np.linspace(0.1, 0.9, 1025),
                        np.linspace(0.2, 0.8, 1025),
                    )
                )
            )
        )
    )
    ax.plot([0, 1], [0, 1], label="managed")
    legend = ax.legend()
    fig.canvas.draw()
    dense_adapter = get_artist_adapter(dense)
    dense_support = dense_adapter.operation_support(TransformOperation.EDIT_POINTS)
    assert not dense_support.supported
    with pytest.raises(UnsupportedArtistError):
        dense_adapter.point_handle_model()

    legend_line = legend.get_lines()[0]
    line_support = get_artist_adapter(legend_line).operation_support(
        TransformOperation.EDIT_POINTS
    )
    assert not line_support.supported
    assert "Legend" in line_support.reason

    manager.selection.clear_targets()
    plt.close(fig)
    assert app is not None


def test_line_broadcast_axis_rejects_independent_endpoint_preview_without_mutation():
    app, fig, ax, manager = _figure_with_manager()
    line = ax.add_line(Line2D([0.5], [0.2, 0.5, 0.8]))
    fig.canvas.draw()
    source = PointEditSource.capture(line)
    destination = source.handle_model.positions_array()[0] + (9.0, 0.0)
    before_x = line.get_xdata(orig=True).copy()
    before_y = line.get_ydata(orig=True).copy()

    with pytest.raises(UnsupportedArtistError, match="broadcast storage"):
        PointEditPlan.preview(source, source.handle_model.keys[0], destination)

    np.testing.assert_array_equal(line.get_xdata(orig=True), before_x)
    np.testing.assert_array_equal(line.get_ydata(orig=True), before_y)
    assert fig.change_tracker.changes == []
    manager.selection.clear_targets()
    plt.close(fig)
    assert app is not None


def test_point_commit_setter_failure_restores_geometry_and_recording_atomically():
    class FailingPolygon(Polygon):
        fail_writes = 0

        def set_xy(self, xy):
            super().set_xy(xy)
            if self.fail_writes:
                self.fail_writes -= 1
                raise RuntimeError("injected point setter failure")

    app, fig, ax, manager = _figure_with_manager()
    polygon = FailingPolygon([(0.2, 0.2), (0.8, 0.2), (0.5, 0.8)])
    ax.add_patch(polygon)
    artist_adapter_registry.register(FailingPolygon, PolygonAdapter)
    try:
        fig.canvas.draw()
        source = PointEditSource.capture(polygon)
        plan = PointEditPlan.preview(
            source,
            1,
            source.handle_model.positions_array()[1] + (10.0, 5.0),
        )
        before = polygon.get_xy().copy()
        recording_before = manager.figure.change_tracker.capture_recording_state()
        polygon.fail_writes = 1

        with pytest.raises(RuntimeError, match="injected point setter failure"):
            plan.commit()

        np.testing.assert_allclose(polygon.get_xy(), before, atol=1e-15, rtol=0)
        assert (
            manager.figure.change_tracker.capture_recording_state() == recording_before
        )
        assert manager.figure.change_tracker.edits == []
    finally:
        artist_adapter_registry.unregister(FailingPolygon, PolygonAdapter)
        manager.selection.clear_targets()
        plt.close(fig)
    assert app is not None
