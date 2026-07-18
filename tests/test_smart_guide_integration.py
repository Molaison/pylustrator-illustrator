"""End-to-end smart-guide gesture, scope, overlay, and history contracts."""

from __future__ import annotations

from time import monotonic, perf_counter
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.backend_bases import KeyEvent, MouseEvent
from matplotlib.patches import Rectangle
from matplotlib.text import Text
from qtpy import QtCore, QtWidgets

import pylustrator.drag_helper as drag_helper
from pylustrator.components.plot_layout import selection_scene_transform
from pylustrator.drag_helper import DragManager, GrabbableRectangleSelection
from pylustrator.interaction import SelectionMode
from pylustrator.smart_guide_ui import (
    SmartGuideOverlay,
    _cached_scene_guides,
    _capture_scene_guides,
    schedule_smart_guide_warmup,
)
from pylustrator.smart_guides import (
    Axis,
    GuideKind,
    GuideLine,
    SnapPlan,
    StaleGuideSnapshotError,
)
from pylustrator.snap import TargetWrapper


class _Signal:
    def __init__(self) -> None:
        self.calls = []

    def emit(self, *args) -> None:
        self.calls.append(args)


class _Signals:
    def __init__(self) -> None:
        self.figure_selection_moved = _Signal()
        self.figure_element_selected = _Signal()
        self.figure_selection_update = _Signal()
        self.figure_selection_property_changed = _Signal()


class _ChangeTracker:
    def __init__(self) -> None:
        self.edits = []
        self.last_edit = -1
        self.changes = []

    def addEdit(self, edit) -> None:
        self.edits.append(edit)
        self.last_edit = len(self.edits) - 1

    def addChange(self, target, command) -> None:
        self.changes.append((target, command))

    def addNewLegendChange(self, target) -> None:
        self.changes.append((target, "legend"))

    def addNewTextChange(self, target) -> None:
        self.changes.append((target, "text"))

    def addNewAxesChange(self, target) -> None:
        self.changes.append((target, "axes"))


@pytest.fixture(scope="module", autouse=True)
def _qt_application():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield app


def _attach_manager(fig, *, device_pixel_ratio: float = 1.0) -> DragManager:
    scene = QtWidgets.QGraphicsScene()
    origin = QtWidgets.QGraphicsRectItem()
    height = float(fig.canvas.get_width_height()[1])
    origin.setTransform(selection_scene_transform(device_pixel_ratio, height))
    origin.view = SimpleNamespace(
        h=height,
        device_pixel_ratio=device_pixel_ratio,
        grabber_found=False,
    )
    scene.addItem(origin)
    fig._pyl_scene_scene = scene
    fig._pyl_scene = origin
    fig.signals = _Signals()
    fig.change_tracker = _ChangeTracker()

    manager = DragManager.__new__(DragManager)
    manager.figure = fig
    manager.selected_element = None
    manager.grab_element = None
    manager.marquee_start = None
    manager.marquee_rect = None
    manager.marquee_active = False
    manager.marquee_additive = False
    manager.marquee_click_element = None
    manager._smart_guide_idle_warmup_enabled = False
    manager.make_figure_draggable(fig)
    manager.make_axes_draggable(fig.axes)
    manager.selection = GrabbableRectangleSelection(fig, origin)
    manager.selection.smart_guides_allow_blocking_capture = True
    fig.selection = manager.selection
    fig.figure_dragger = manager
    return manager


def _figure_rectangle(fig, bounds, **kwargs) -> Rectangle:
    rectangle = Rectangle(
        bounds[:2],
        bounds[2],
        bounds[3],
        transform=fig.transFigure,
        linewidth=0,
        **kwargs,
    )
    fig.add_artist(rectangle)
    return rectangle


def _display_bounds(artist) -> np.ndarray:
    points = np.asarray(TargetWrapper(artist).get_selection_points(), dtype=float)
    return np.array(
        [
            np.min(points[:, 0]),
            np.min(points[:, 1]),
            np.max(points[:, 0]),
            np.max(points[:, 1]),
        ],
        dtype=float,
    )


def _event(fig, name: str, x: float, y: float, *, key=None) -> MouseEvent:
    return MouseEvent(name, fig.canvas, x, y, button=1, key=key)


def _start_snap_drag(manager, moving, source, *, miss_px: float = 2.0):
    manager.select_element(moving)
    selection = manager.selection
    moving_bounds = _display_bounds(moving)
    source_bounds = _display_bounds(source)
    raw_dx = float(source_bounds[0] - moving_bounds[2] - miss_px)
    press_x = float((moving_bounds[0] + moving_bounds[2]) / 2)
    press_y = float((moving_bounds[1] + moving_bounds[3]) / 2)
    press = _event(manager.figure, "button_press_event", press_x, press_y)
    selection.button_press_event(press)
    assert selection.smart_guide_session is not None
    return selection, moving_bounds, raw_dx, press_x, press_y


def test_drag_preview_commit_undo_and_redo_share_one_snap_result() -> None:
    fig = plt.figure(figsize=(4, 3), dpi=100)
    moving = _figure_rectangle(fig, (0.15, 0.1, 0.1, 0.1))
    source = _figure_rectangle(fig, (0.55, 0.65, 0.1, 0.1))
    fig.canvas.draw()
    manager = _attach_manager(fig)
    native_before = tuple(moving.get_xy())
    selection, before, raw_dx, press_x, press_y = _start_snap_drag(
        manager, moving, source
    )
    session = selection.smart_guide_session
    overlay_item = session.overlay.item

    motion = _event(
        fig,
        "motion_notify_event",
        press_x + raw_dx,
        press_y,
    )
    selection.on_motion(motion)

    plan = session.last_plan
    assert plan is not None
    assert plan.delta_px[0] == pytest.approx(2.0)
    preview = selection.move_current_selection_points[id(moving)]
    preview_bounds = np.array(
        [
            np.min(preview[:, 0]),
            np.min(preview[:, 1]),
            np.max(preview[:, 0]),
            np.max(preview[:, 1]),
        ]
    )
    assert preview_bounds - before == pytest.approx([raw_dx + 2, 0, raw_dx + 2, 0])
    assert tuple(moving.get_xy()) == pytest.approx(native_before)
    assert overlay_item is not None and overlay_item.isVisible()

    selection.button_release_event(
        _event(fig, "button_release_event", press_x + raw_dx, press_y)
    )

    committed = _display_bounds(moving)
    assert committed == pytest.approx(preview_bounds)
    assert len(fig.change_tracker.edits) == 1
    assert overlay_item.scene() is None
    assert selection.smart_guide_session is None

    undo, redo = fig.change_tracker.edits[0][:2]
    undo()
    assert _display_bounds(moving) == pytest.approx(before)
    redo()
    assert _display_bounds(moving) == pytest.approx(committed)
    assert len(fig.change_tracker.edits) == 1
    selection.clear_targets()
    plt.close(fig)


def test_alt_disables_guides_temporarily_and_shift_restricts_snap_axis() -> None:
    fig = plt.figure(figsize=(4, 3), dpi=100)
    moving = _figure_rectangle(fig, (0.15, 0.15, 0.1, 0.1))
    source = _figure_rectangle(fig, (0.55, 0.65, 0.1, 0.1))
    fig.canvas.draw()
    manager = _attach_manager(fig)
    selection, before, raw_dx, press_x, press_y = _start_snap_drag(
        manager, moving, source
    )
    session = selection.smart_guide_session

    selection.on_motion(
        _event(
            fig,
            "motion_notify_event",
            press_x + raw_dx,
            press_y,
            key="alt",
        )
    )
    alt_preview = selection.move_current_selection_points[id(moving)]
    assert np.min(alt_preview[:, 0]) - before[0] == pytest.approx(raw_dx)
    assert session.last_plan is None

    selection.on_motion(
        _event(
            fig,
            "motion_notify_event",
            press_x + raw_dx,
            press_y + 17,
            key="shift",
        )
    )
    shift_preview = selection.move_current_selection_points[id(moving)]
    assert np.min(shift_preview[:, 0]) - before[0] == pytest.approx(raw_dx + 2)
    assert np.min(shift_preview[:, 1]) - before[1] == pytest.approx(0)
    assert session.last_plan is not None
    assert {hit.axis for hit in session.last_plan.hits} <= {Axis.X}

    manager.key_press_event(KeyEvent("key_press_event", fig.canvas, key="escape"))
    assert _display_bounds(moving) == pytest.approx(before)
    assert [target.target for target in selection.targets] == [moving]
    assert not selection.got_artist
    assert selection.smart_guide_session is None
    assert not fig.change_tracker.edits
    selection.clear_targets()
    plt.close(fig)


def test_equal_gap_plan_is_available_through_real_gesture_session() -> None:
    fig = plt.figure(figsize=(4, 3), dpi=100)
    _figure_rectangle(fig, (0.10, 0.3, 0.05, 0.1))
    _figure_rectangle(fig, (0.20, 0.3, 0.05, 0.1))
    moving = _figure_rectangle(fig, (0.375, 0.3, 0.05, 0.1))
    fig.canvas.draw()
    manager = _attach_manager(fig)
    manager.select_element(moving)
    bounds = _display_bounds(moving)
    press_x, press_y = float(bounds[0]), float(bounds[1])
    manager.selection.button_press_event(
        _event(fig, "button_press_event", press_x, press_y)
    )
    session = manager.selection.smart_guide_session
    assert session is not None

    plan = session.query((-28.0, 0.0))

    equal_gap = [hit for hit in plan.hits if hit.kind is GuideKind.EQUAL_GAP]
    assert len(equal_gap) == 1
    assert equal_gap[0].axis is Axis.X
    assert equal_gap[0].delta_px == pytest.approx(-2.0)
    manager.key_press_event(KeyEvent("key_press_event", fig.canvas, key="escape"))
    manager.selection.clear_targets()
    plt.close(fig)


def test_overlay_uses_parent_display_transform_and_close_removes_item() -> None:
    fig = plt.figure(figsize=(4, 3), dpi=100)
    fig.canvas.draw()
    manager = _attach_manager(fig, device_pixel_ratio=2.0)
    overlay = SmartGuideOverlay(fig)
    plan = SnapPlan(
        "snapshot",
        (0.0, 0.0),
        (),
        (
            GuideLine(Axis.X, 100.0, (20.0, 80.0), GuideKind.EDGE, ("x",)),
            GuideLine(Axis.Y, 60.0, (10.0, 90.0), GuideKind.EDGE, ("y",)),
        ),
        0,
    )

    overlay.render(plan)

    item = overlay.item
    assert item is not None
    path = item.path()
    coordinates = [
        (path.elementAt(index).x, path.elementAt(index).y)
        for index in range(path.elementCount())
    ]
    assert coordinates == pytest.approx(
        [(100, 20), (100, 80), (10, 60), (90, 60)]
    )
    mapped = item.mapToScene(QtCore.QPointF(100, 20))
    assert (mapped.x(), mapped.y()) == pytest.approx((50, 290))

    overlay.close()
    assert item.scene() is None
    plt.close(fig)
    assert manager is not None


def test_scene_revision_stales_session_and_hides_overlay() -> None:
    fig = plt.figure(figsize=(4, 3), dpi=100)
    moving = _figure_rectangle(fig, (0.1, 0.1, 0.1, 0.1))
    source = _figure_rectangle(fig, (0.5, 0.7, 0.1, 0.1))
    fig.canvas.draw()
    manager = _attach_manager(fig)
    selection, _before, raw_dx, press_x, press_y = _start_snap_drag(
        manager, moving, source
    )
    session = selection.smart_guide_session
    session.query((raw_dx, 0))
    item = session.overlay.item
    assert item is not None and item.isVisible()

    manager._invalidate_interaction_index()

    with pytest.raises(StaleGuideSnapshotError):
        session.query((raw_dx, 0))
    assert session.last_plan is None
    assert not item.isVisible()
    manager.key_press_event(KeyEvent("key_press_event", fig.canvas, key="escape"))
    selection.clear_targets()
    plt.close(fig)


def test_release_commits_last_accepted_preview_without_a_second_solve() -> None:
    fig = plt.figure(figsize=(4, 3), dpi=100)
    moving = _figure_rectangle(fig, (0.1, 0.1, 0.1, 0.1))
    source = _figure_rectangle(fig, (0.5, 0.7, 0.1, 0.1))
    fig.canvas.draw()
    manager = _attach_manager(fig)
    selection, _before, raw_dx, press_x, press_y = _start_snap_drag(
        manager, moving, source
    )
    selection.on_motion(
        _event(fig, "motion_notify_event", press_x + raw_dx, press_y)
    )
    accepted_preview = np.array(
        selection.move_current_selection_points[id(moving)], copy=True
    )

    # An external draw can invalidate future queries after the last motion.
    # Release still commits the exact accepted frame the user saw; re-solving
    # here would create a preview/commit jump.
    manager._invalidate_interaction_index()
    assert not selection.smart_guide_session.active
    selection.button_release_event(
        _event(fig, "button_release_event", press_x + raw_dx, press_y)
    )

    expected = np.array(
        [
            np.min(accepted_preview[:, 0]),
            np.min(accepted_preview[:, 1]),
            np.max(accepted_preview[:, 0]),
            np.max(accepted_preview[:, 1]),
        ]
    )
    assert _display_bounds(moving) == pytest.approx(expected)
    assert len(fig.change_tracker.edits) == 1
    selection.clear_targets()
    plt.close(fig)


def test_runtime_planner_failure_disables_once_and_keeps_raw_dragging() -> None:
    fig = plt.figure(figsize=(4, 3), dpi=100)
    moving = _figure_rectangle(fig, (0.1, 0.1, 0.1, 0.1))
    source = _figure_rectangle(fig, (0.5, 0.7, 0.1, 0.1))
    fig.canvas.draw()
    manager = _attach_manager(fig)
    selection, before, raw_dx, press_x, press_y = _start_snap_drag(
        manager, moving, source
    )
    session = selection.smart_guide_session
    calls = 0

    def broken_query(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        raise RuntimeError("synthetic planner failure")

    session.query = broken_query
    selection.on_motion(
        _event(fig, "motion_notify_event", press_x + raw_dx, press_y)
    )
    preview = selection.move_current_selection_points[id(moving)]
    assert np.min(preview[:, 0]) - before[0] == pytest.approx(raw_dx)
    assert not session.active
    assert session.last_plan is None

    selection.on_motion(
        _event(fig, "motion_notify_event", press_x + raw_dx - 5, press_y)
    )
    assert calls == 1
    manager.key_press_event(KeyEvent("key_press_event", fig.canvas, key="escape"))
    assert _display_bounds(moving) == pytest.approx(before)
    selection.clear_targets()
    plt.close(fig)


def test_preview_failure_never_publishes_unapplied_guide_plan() -> None:
    fig = plt.figure(figsize=(4, 3), dpi=100)
    moving = _figure_rectangle(fig, (0.1, 0.1, 0.1, 0.1))
    source = _figure_rectangle(fig, (0.5, 0.7, 0.1, 0.1))
    fig.canvas.draw()
    manager = _attach_manager(fig)
    selection, before, raw_dx, press_x, press_y = _start_snap_drag(
        manager, moving, source
    )
    session = selection.smart_guide_session
    original_add_offset = selection.addOffset

    def fail_preview(*_args, **_kwargs):
        raise RuntimeError("synthetic adapter preview failure")

    selection.addOffset = fail_preview
    with pytest.raises(RuntimeError, match="adapter preview"):
        selection.on_motion(
            _event(fig, "motion_notify_event", press_x + raw_dx, press_y)
        )
    assert session.last_plan is None
    assert session.overlay.item is None or not session.overlay.item.isVisible()
    selection.addOffset = original_add_offset
    manager.key_press_event(KeyEvent("key_press_event", fig.canvas, key="escape"))
    assert _display_bounds(moving) == pytest.approx(before)
    selection.clear_targets()
    plt.close(fig)


def test_undo_and_tool_switch_cancel_active_gesture_before_policy_change() -> None:
    fig = plt.figure(figsize=(4, 3), dpi=100)
    moving = _figure_rectangle(fig, (0.1, 0.1, 0.1, 0.1))
    source = _figure_rectangle(fig, (0.5, 0.7, 0.1, 0.1))
    fig.canvas.draw()
    manager = _attach_manager(fig)
    selection, before, raw_dx, press_x, press_y = _start_snap_drag(
        manager, moving, source
    )
    selection.on_motion(
        _event(fig, "motion_notify_event", press_x + raw_dx, press_y)
    )

    # The probe tracker intentionally has no backEdit method.  Returning after
    # cancellation is therefore also proof that Undo did not touch old history.
    manager.undo()
    assert _display_bounds(moving) == pytest.approx(before)
    assert [target.target for target in selection.targets] == [moving]
    assert not fig.change_tracker.edits

    selection, _before, raw_dx, press_x, press_y = _start_snap_drag(
        manager, moving, source
    )
    selection.on_motion(
        _event(fig, "motion_notify_event", press_x + raw_dx, press_y)
    )
    manager.set_selection_mode(SelectionMode.DIRECT)
    assert manager.selection_mode is SelectionMode.DIRECT
    assert _display_bounds(moving) == pytest.approx(before)
    assert not selection.got_artist
    assert selection.smart_guide_session is None
    assert not fig.change_tracker.edits
    selection.clear_targets()
    plt.close(fig)


def test_idle_warmup_publishes_cache_before_nonblocking_gesture() -> None:
    app = QtWidgets.QApplication.instance()
    fig = plt.figure(figsize=(4, 3), dpi=100)
    moving = _figure_rectangle(fig, (0.1, 0.1, 0.1, 0.1))
    source = _figure_rectangle(fig, (0.5, 0.7, 0.1, 0.1))
    fig.canvas.draw()
    manager = _attach_manager(fig)
    manager._smart_guide_idle_warmup_enabled = True
    manager.selection.smart_guides_allow_blocking_capture = False
    manager._smart_guide_scene_cache = None

    assert schedule_smart_guide_warmup(manager, batch_budget_ms=0.25)
    roster, _editable, _order = manager._interaction_roster_snapshot()
    deadline = monotonic() + 2.0
    while (
        (
            _cached_scene_guides(manager) is None
            or not manager._interaction_index.is_current(
                revision=manager._interaction_revision,
                source_ids=roster.source_ids,
            )
        )
        and monotonic() < deadline
    ):
        app.processEvents()
    assert _cached_scene_guides(manager) is not None
    assert manager._interaction_index.is_current(
        revision=manager._interaction_revision,
        source_ids=roster.source_ids,
    )

    selection, _before, raw_dx, press_x, press_y = _start_snap_drag(
        manager, moving, source
    )
    session = selection.smart_guide_session
    assert session._index is None
    selection.on_motion(
        _event(fig, "motion_notify_event", press_x + raw_dx, press_y)
    )
    assert session._index is not None
    manager.key_press_event(KeyEvent("key_press_event", fig.canvas, key="escape"))
    manager._smart_guide_idle_warmup_enabled = False
    selection.clear_targets()
    plt.close(fig)


def test_idle_warmup_makes_first_fig2_scale_hit_query_hot(monkeypatch) -> None:
    app = QtWidgets.QApplication.instance()
    fig, ax = plt.subplots(figsize=(7, 5), dpi=100)
    rectangles = []
    for ix in range(24):
        for iy in range(18):
            rectangles.append(
                ax.add_patch(
                    Rectangle(
                        (ix / 24, iy / 18),
                        0.7 / 24,
                        0.7 / 18,
                        linewidth=0.2,
                    )
                )
            )
    fig.canvas.draw()
    manager = _attach_manager(fig)
    manager._smart_guide_idle_warmup_enabled = True
    manager._smart_guide_scene_cache = None
    roster, _editable, _order = manager._interaction_roster_snapshot()
    measured = 0
    original_bounds = manager._interaction_index_bounds

    def counted_bounds(*args, **kwargs):
        nonlocal measured
        measured += 1
        return original_bounds(*args, **kwargs)

    monkeypatch.setattr(manager, "_interaction_index_bounds", counted_bounds)
    assert schedule_smart_guide_warmup(manager, batch_budget_ms=0.25)
    # Scheduling performs no synchronous per-Artist hit-envelope work on draw.
    assert measured == 0

    deadline = monotonic() + 5.0
    while (
        not manager._interaction_index.is_current(
            revision=manager._interaction_revision,
            source_ids=roster.source_ids,
        )
        and monotonic() < deadline
    ):
        app.processEvents()
    assert manager._interaction_index.is_current(
        revision=manager._interaction_revision,
        source_ids=roster.source_ids,
    )
    assert measured == len(roster.artists)

    def unexpected_pointer_measurement(*_args, **_kwargs):
        raise AssertionError("first pointer query rebuilt the full scene")

    monkeypatch.setattr(
        manager, "_interaction_index_bounds", unexpected_pointer_measurement
    )
    x, y = ax.transData.transform((0.51, 0.51))
    event = _event(fig, "button_press_event", float(x), float(y))
    samples = []
    for _ in range(80):
        started = perf_counter()
        stack = manager.get_hit_stack(event)
        samples.append(perf_counter() - started)
    assert any(rectangle in stack.artists for rectangle in rectangles)
    assert np.percentile(samples, 95) < 0.004

    manager._smart_guide_idle_warmup_enabled = False
    manager.selection.clear_targets()
    plt.close(fig)


def test_object_direct_and_isolation_scopes_filter_guide_sources() -> None:
    fig = plt.figure(figsize=(4, 3), dpi=100)
    first = _figure_rectangle(fig, (0.1, 0.1, 0.1, 0.1))
    second = _figure_rectangle(fig, (0.3, 0.1, 0.1, 0.1))
    outsider = _figure_rectangle(fig, (0.7, 0.7, 0.1, 0.1))
    fig.canvas.draw()
    manager = _attach_manager(fig)
    manager.select_elements([first, second], primary=second)
    group = manager.group_selection("pair")
    fig.change_tracker.edits.clear()
    fig.change_tracker.last_edit = -1

    manager.select_element(outsider)
    bounds = _display_bounds(outsider)
    manager.selection.button_press_event(
        _event(fig, "button_press_event", bounds[0], bounds[1])
    )
    object_sources = manager.selection.smart_guide_session.index.snapshot.objects
    assert len(object_sources) == 1
    assert "EditorGroup" in object_sources[0].stable_id
    manager.key_press_event(KeyEvent("key_press_event", fig.canvas, key="escape"))

    manager.set_selection_mode(SelectionMode.DIRECT)
    manager.select_element(outsider)
    manager.selection.button_press_event(
        _event(fig, "button_press_event", bounds[0], bounds[1])
    )
    direct_sources = manager.selection.smart_guide_session.index.snapshot.objects
    assert len(direct_sources) == 2
    assert all("Rectangle" in source.stable_id for source in direct_sources)
    manager.key_press_event(KeyEvent("key_press_event", fig.canvas, key="escape"))

    assert manager.enter_isolation(group)
    manager.set_selection_mode(SelectionMode.DIRECT)
    manager.select_element(first)
    first_bounds = _display_bounds(first)
    manager.selection.button_press_event(
        _event(fig, "button_press_event", first_bounds[0], first_bounds[1])
    )
    isolated_sources = manager.selection.smart_guide_session.index.snapshot.objects
    assert len(isolated_sources) == 1
    source_bounds = isolated_sources[0].bounds
    assert (
        source_bounds.x0,
        source_bounds.y0,
        source_bounds.x1,
        source_bounds.y1,
    ) == pytest.approx(tuple(_display_bounds(second)))
    manager.key_press_event(KeyEvent("key_press_event", fig.canvas, key="escape"))
    manager.selection.clear_targets()
    plt.close(fig)


def test_text_guides_capture_insertion_anchor_and_unrotated_baseline() -> None:
    fig = plt.figure(figsize=(4, 3), dpi=100)
    normal = fig.text(0.2, 0.3, "baseline")
    rotated = fig.text(0.7, 0.6, "rotated", rotation=30)
    fig.canvas.draw()
    manager = _attach_manager(fig)

    entries = _capture_scene_guides(manager).entries
    by_artist = {entry.artist: entry.guide for entry in entries}

    assert by_artist[normal].baseline_y is not None
    assert len(by_artist[normal].anchors) == 1
    assert by_artist[rotated].baseline_y is None
    assert len(by_artist[rotated].anchors) == 1
    plt.close(fig)


def test_hit_and_guides_share_one_renderer_revision_geometry_measurement() -> None:
    fig = plt.figure(figsize=(4, 3), dpi=100)
    rectangle = _figure_rectangle(fig, (0.2, 0.3, 0.2, 0.2))
    fig.canvas.draw()
    calls = 0
    original_get_window_extent = rectangle.get_window_extent

    def counted_get_window_extent(renderer=None):
        nonlocal calls
        calls += 1
        return original_get_window_extent(renderer)

    rectangle.get_window_extent = counted_get_window_extent
    manager = _attach_manager(fig)
    x, y = fig.transFigure.transform((0.3, 0.4))
    event = _event(fig, "button_press_event", float(x), float(y))

    assert rectangle in manager.get_hit_stack(event).artists
    _capture_scene_guides(manager)

    geometry = manager._ensure_display_geometry_cache()
    assert calls == 1
    assert geometry.revision == manager._interaction_revision
    assert geometry.renderer is fig.canvas.get_renderer()
    assert geometry.roster is manager._selectable_roster_snapshot()

    geometry.bind(
        revision=manager._interaction_revision,
        roster=manager._selectable_roster_snapshot(),
        renderer=object(),
    )
    assert geometry.selection_bounds(rectangle) is not None
    assert calls == 2

    manager.invalidate_geometry_cache()
    _capture_scene_guides(manager)
    assert calls == 3
    plt.close(fig)


def test_formatter_owned_tick_labels_are_not_unstable_guide_sources() -> None:
    fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
    ax.plot([0, 1], [0, 1])
    fig.canvas.draw()
    manager = _attach_manager(fig)
    tagged = [
        artist
        for artist in manager._selectable_artists
        if getattr(artist, "_pylustrator_formatter_owned_tick_label", False)
    ]
    assert tagged

    cache = _capture_scene_guides(manager)

    assert not any(entry.artist in tagged for entry in cache.entries)
    assert not any(
        isinstance(entry.artist, Text) and entry.artist.get_text() == ""
        for entry in cache.entries
    )
    assert any(entry.artist is ax for entry in cache.entries)
    manager.selection.clear_targets()
    plt.close(fig)


def test_session_build_failure_lazily_falls_back_to_legacy_snaps(
    monkeypatch,
) -> None:
    fig = plt.figure(figsize=(4, 3), dpi=100)
    moving = _figure_rectangle(fig, (0.1, 0.1, 0.1, 0.1))
    _figure_rectangle(fig, (0.6, 0.7, 0.1, 0.1))
    fig.canvas.draw()
    manager = _attach_manager(fig)
    manager.select_element(moving)

    class _LegacySnap:
        removed = False

        def remove(self) -> None:
            self.removed = True

    legacy = _LegacySnap()

    def fail_session(*_args, **_kwargs):
        raise RuntimeError("synthetic guide capture failure")

    monkeypatch.setattr(drag_helper, "create_smart_guide_drag_session", fail_session)
    monkeypatch.setattr(drag_helper, "getSnaps", lambda *_args, **_kwargs: [legacy])
    bounds = _display_bounds(moving)

    manager.selection.button_press_event(
        _event(fig, "button_press_event", bounds[0], bounds[1])
    )

    assert manager.selection.smart_guide_session is None
    assert manager.selection.snaps == [legacy]
    manager.key_press_event(KeyEvent("key_press_event", fig.canvas, key="escape"))
    assert legacy.removed
    assert [target.target for target in manager.selection.targets] == [moving]
    assert not fig.change_tracker.edits
    manager.selection.clear_targets()
    plt.close(fig)


def test_pointer_press_never_performs_blocking_cold_scene_capture(
    monkeypatch,
) -> None:
    fig = plt.figure(figsize=(4, 3), dpi=100)
    moving = _figure_rectangle(fig, (0.1, 0.1, 0.1, 0.1))
    _figure_rectangle(fig, (0.6, 0.7, 0.1, 0.1))
    fig.canvas.draw()
    manager = _attach_manager(fig)
    manager.selection.smart_guides_allow_blocking_capture = False
    manager._smart_guide_scene_cache = None

    class _LegacySnap:
        removed = False

        def remove(self) -> None:
            self.removed = True

    legacy = _LegacySnap()
    monkeypatch.setattr(drag_helper, "getSnaps", lambda *_args, **_kwargs: [legacy])
    manager.select_element(moving)
    bounds = _display_bounds(moving)

    manager.selection.button_press_event(
        _event(fig, "button_press_event", bounds[0], bounds[1])
    )

    assert manager.selection.smart_guide_session is None
    assert manager.selection.snaps == [legacy]
    manager.key_press_event(KeyEvent("key_press_event", fig.canvas, key="escape"))
    assert legacy.removed
    manager.selection.clear_targets()
    plt.close(fig)
