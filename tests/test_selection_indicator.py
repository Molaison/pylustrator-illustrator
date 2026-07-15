from __future__ import annotations

from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.backend_bases import KeyEvent, MouseEvent
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.patches import (
    Circle,
    ConnectionPatch,
    Ellipse,
    FancyArrowPatch,
    FancyBboxPatch,
    Patch,
    PathPatch,
    Rectangle,
    RegularPolygon,
    Wedge,
)
from matplotlib.path import Path
from qtpy import QtCore, QtGui, QtWidgets

from pylustrator.artist_adapters import selection_geometry_snapshot
from pylustrator.components.plot_layout import (
    scene_point_to_canvas_pixels,
    selection_scene_transform,
)
from pylustrator.components.qpos_and_size import QPosAndSize
from pylustrator.components.tree_view import MyTreeView
from pylustrator.drag_helper import (
    DIR_X0,
    DIR_X1,
    DIR_Y0,
    DIR_Y1,
    DragManager,
    GrabbableRectangleSelection,
)
from pylustrator.snap import SnapSamePos, TargetWrapper
from pylustrator.interaction import SelectionMode
from pylustrator.editor_model import EditorGroup
from pylustrator.commands import semantic_equal


def test_degenerate_stroked_path_patch_is_reachable_by_click() -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
    path = Path(
        [(0.4, 0.2), (0.4, 0.8), (0.4, 0.8), (0.4, 0.2), (0.4, 0.2)],
        [Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO, Path.CLOSEPOLY],
    )
    patch = ax.add_patch(
        PathPatch(path, facecolor="white", edgecolor="black", linewidth=0.65)
    )
    fig.canvas.draw()
    manager = attach_drag_manager(fig)
    display_path = patch.get_transform().transform_path(path)
    center = np.mean(display_path.vertices[:2], axis=0)
    event = MouseEvent(
        "button_press_event", fig.canvas, float(center[0]), float(center[1]), button=1
    )

    assert patch in manager.get_hit_candidates(event)
    picked, _finished = manager.get_picked_element(event)
    assert picked is patch

    miss = MouseEvent(
        "button_press_event",
        fig.canvas,
        float(center[0] + 6),
        float(center[1]),
        button=1,
    )
    assert patch not in manager.get_hit_candidates(miss)
    manager.selection.clear_targets()
    plt.close(fig)
    assert app is not None


class SelectionView:
    h = 200
    device_pixel_ratio = 1.0
    grabber_found = False


class ChangeTracker:
    def __init__(self):
        self.axes_change_count = 0
        self.legend_change_count = 0
        self.text_change_count = 0
        self.change_count = 0

    def addEdit(self, edit):
        self.edit = edit

    def addNewLegendChange(self, target):
        self.legend_change_count += 1
        self.legend = target

    def addNewTextChange(self, target):
        self.text_change_count += 1
        self.text = target

    def addNewAxesChange(self, target):
        self.axes_change_count += 1
        self.axes = target

    def addChange(self, target, command):
        self.change_count += 1
        self.change = (target, command)

    def removeElement(self, target):
        self.removed = target
        target.set_visible(False)


class Signals:
    def __init__(self):
        self.selected = []
        self.moved = False

        class SelectionMoved:
            def __init__(self, parent):
                self.parent = parent

            def emit(self):
                self.parent.moved = True

        class ElementSelected:
            def __init__(self, parent):
                self.parent = parent

            def emit(self, element):
                self.parent.selected.append(element)

        self.figure_selection_moved = SelectionMoved(self)
        self.figure_element_selected = ElementSelected(self)


class Signal:
    def __init__(self):
        self.callbacks = []

    def connect(self, callback):
        self.callbacks.append(callback)

    def emit(self, *args):
        for callback in list(self.callbacks):
            callback(*args)


class WidgetSignals:
    def __init__(self):
        self.figure_changed = Signal()
        self.figure_element_selected = Signal()
        self.figure_selection_moved = Signal()
        self.figure_selection_property_changed = Signal()
        self.figure_size_changed = Signal()


class TreeSignals:
    def __init__(self):
        self.figure_changed = Signal()
        self.figure_element_selected = Signal()
        self.figure_element_child_created = Signal()


def make_selection_scene():
    scene = QtWidgets.QGraphicsScene()
    origin = QtWidgets.QGraphicsRectItem()
    origin.view = SelectionView()
    scene.addItem(origin)
    return scene, origin


def attach_drag_manager(fig):
    scene, origin = make_selection_scene()
    fig._pyl_scene_scene = scene
    fig._pyl_scene = origin
    fig.signals = Signals()
    fig.change_tracker = ChangeTracker()
    manager = DragManager.__new__(DragManager)
    manager.figure = fig
    manager.selected_element = None
    manager.grab_element = None
    manager.marquee_start = None
    manager.marquee_rect = None
    manager.marquee_active = False
    manager.marquee_additive = False
    manager.marquee_click_element = None
    manager.make_figure_draggable(fig)
    manager.make_axes_draggable(fig.axes)
    manager.selection = GrabbableRectangleSelection(fig, origin)
    fig.selection = manager.selection
    fig.figure_dragger = manager
    return manager


class EmptySelection:
    lock_aspect_ratio = False
    targets = []

    def update_selection_rectangles(self):
        pass


def selection_target_extents(selection):
    extents = []
    for target in selection.targets:
        points = TargetWrapper(target.target).get_selection_points()
        x_values = [point[0] for point in points]
        y_values = [point[1] for point in points]
        extents.append((min(x_values), min(y_values), max(x_values), max(y_values)))
    return extents


def selection_rect_extents(selection):
    extents = []
    for index in range(0, len(selection.targets_rects), 2):
        rect = selection.targets_rects[index].rect()
        extents.append(
            (
                rect.x(),
                rect.y(),
                rect.x() + rect.width(),
                rect.y() + rect.height(),
            )
        )
    return extents


def _center_event(fig, artist, *, key=None, dblclick=False):
    bbox = artist.get_window_extent(fig.canvas.get_renderer())
    return MouseEvent(
        "button_press_event",
        fig.canvas,
        (bbox.x0 + bbox.x1) / 2,
        (bbox.y0 + bbox.y1) / 2,
        button=1,
        key=key,
        dblclick=dblclick,
    )


def test_hit_stack_is_front_to_back_and_exposes_all_overlapping_candidates() -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
    lower = ax.add_patch(Rectangle((0.2, 0.2), 0.6, 0.6, zorder=2))
    upper = ax.add_patch(Rectangle((0.2, 0.2), 0.6, 0.6, zorder=5))
    fig.canvas.draw()
    manager = attach_drag_manager(fig)
    x, y = ax.transData.transform((0.5, 0.5))
    event = MouseEvent("button_press_event", fig.canvas, x, y, button=1)

    stack = manager.get_hit_stack(event)

    assert stack.artists.index(upper) < stack.artists.index(lower)
    assert manager.get_hit_candidates(event)[:2] == (upper, lower)
    manager.selection.clear_targets()
    plt.close(fig)
    assert app is not None


def test_hover_preselection_and_candidate_entries_use_same_resolver() -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
    lower = ax.add_patch(Rectangle((0.2, 0.2), 0.6, 0.6, zorder=2))
    upper = ax.add_patch(Rectangle((0.2, 0.2), 0.6, 0.6, zorder=5))
    fig.canvas.draw()
    manager = attach_drag_manager(fig)
    x, y = ax.transData.transform((0.5, 0.5))
    event = MouseEvent("motion_notify_event", fig.canvas, x, y)

    hovered = manager.update_preselection(event)
    entries = manager.get_hit_candidate_entries(event)

    assert hovered is upper
    assert manager.preselection_artist is upper
    assert manager.preselection_rect.isVisible()
    assert entries[0][0] is upper
    assert entries[1][0] is lower

    manager._hide_preselection()
    manager.selection.clear_targets()
    plt.close(fig)
    assert app is not None


def test_object_and_direct_tools_resolve_legend_children_differently() -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
    ax.plot([0, 1], [0, 1], label="line")
    legend = ax.legend()
    fig.canvas.draw()
    manager = attach_drag_manager(fig)
    text = legend.get_texts()[0]
    event = _center_event(fig, text)

    manager.set_selection_mode(SelectionMode.OBJECT)
    assert manager.get_hit_candidates(event)[0] is legend

    manager.set_selection_mode(SelectionMode.DIRECT)
    assert manager.get_hit_candidates(event)[0] is text

    manager.selection.clear_targets()
    plt.close(fig)
    assert app is not None


def test_alt_click_cycles_resolved_candidates_in_visual_order() -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
    lower = ax.add_patch(Rectangle((0.2, 0.2), 0.6, 0.6, zorder=2))
    upper = ax.add_patch(Rectangle((0.2, 0.2), 0.6, 0.6, zorder=5))
    fig.canvas.draw()
    manager = attach_drag_manager(fig)
    x, y = ax.transData.transform((0.5, 0.5))

    first = MouseEvent("button_press_event", fig.canvas, x, y, button=1)
    manager.button_press_event0(first)
    manager.button_release_event0(first)
    assert manager.selected_element is upper

    cycle = MouseEvent(
        "button_press_event", fig.canvas, x, y, button=1, key="alt"
    )
    manager.button_press_event0(cycle)
    manager.button_release_event0(cycle)
    assert manager.selected_element is lower

    manager.selection.clear_targets()
    plt.close(fig)
    assert app is not None


def test_double_click_enters_legend_isolation_and_escape_exits_one_scope() -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
    ax.plot([0, 1], [0, 1], label="line")
    legend = ax.legend()
    fig.canvas.draw()
    manager = attach_drag_manager(fig)
    text = legend.get_texts()[0]
    event = _center_event(fig, text, dblclick=True)

    manager.button_press_event0(event)

    assert manager.isolation_breadcrumbs == ("Legend",)
    assert manager.selected_element is text
    assert [target.target for target in manager.selection.targets] == [text]

    manager.key_press_event(KeyEvent("key_press_event", fig.canvas, "escape"))
    assert manager.isolation_breadcrumbs == ()
    assert manager.selected_element is legend
    assert [target.target for target in manager.selection.targets] == [legend]

    manager.selection.clear_targets()
    plt.close(fig)
    assert app is not None


def test_logical_group_selects_as_one_object_but_direct_tool_reaches_member() -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
    first = ax.add_patch(Rectangle((0.15, 0.2), 0.25, 0.3, zorder=3))
    second = ax.add_patch(Rectangle((0.55, 0.2), 0.25, 0.3, zorder=3))
    fig.canvas.draw()
    manager = attach_drag_manager(fig)
    manager.select_elements([first, second], primary=second)

    group = manager.group_selection("Pair")

    assert isinstance(group, EditorGroup)
    assert manager.selected_element is group
    assert [target.target for target in manager.selection.targets] == [group]
    x, y = ax.transData.transform((0.25, 0.3))
    event = MouseEvent("button_press_event", fig.canvas, x, y, button=1)
    assert manager.get_hit_candidates(event)[0] is group

    manager.set_selection_mode(SelectionMode.DIRECT)
    assert manager.get_hit_candidates(event)[0] is first

    manager.selection.clear_targets()
    plt.close(fig)
    assert app is not None


def test_direct_marquee_skips_empty_logical_group_bounds() -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
    first = ax.add_patch(Rectangle((0.1, 0.3), 0.15, 0.2))
    second = ax.add_patch(Rectangle((0.75, 0.3), 0.15, 0.2))
    fig.canvas.draw()
    manager = attach_drag_manager(fig)
    manager.select_elements([first, second], primary=second)
    group = manager.group_selection("Separated pair")
    manager.selection.clear_targets()
    x, y = ax.transData.transform((0.5, 0.4))

    manager.set_selection_mode(SelectionMode.OBJECT)
    assert manager.select_elements_in_bbox(x - 1, y - 1, x + 1, y + 1) == [group]
    manager.selection.clear_targets()

    manager.set_selection_mode(SelectionMode.DIRECT)
    assert manager.select_elements_in_bbox(x - 1, y - 1, x + 1, y + 1) == []
    assert manager.selection.targets == []
    plt.close(fig)
    assert app is not None


def test_logical_group_drag_transforms_every_member_with_one_selection_box() -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
    first = ax.add_patch(Rectangle((0.15, 0.2), 0.25, 0.3))
    second = ax.add_patch(Rectangle((0.55, 0.2), 0.25, 0.3))
    fig.canvas.draw()
    manager = attach_drag_manager(fig)
    before = [
        TargetWrapper(artist).get_selection_points().copy()
        for artist in (first, second)
    ]
    manager.select_elements([first, second], primary=second)
    group = manager.group_selection("Pair")

    manager.selection.start_move()
    manager.selection.move(
        (13, -8),
        DIR_X0 | DIR_X1 | DIR_Y0 | DIR_Y1,
        [],
        ignore_snaps=True,
    )
    manager.selection.end_move()
    fig.canvas.draw()

    assert [target.target for target in manager.selection.targets] == [group]
    assert len(manager.selection.targets_rects) == 2
    for artist, original in zip((first, second), before):
        moved = TargetWrapper(artist).get_selection_points()
        assert np.allclose(moved - original, [13, -8])

    manager.selection.clear_targets()
    plt.close(fig)
    assert app is not None


def test_group_undo_redo_rebuilds_same_stable_group_identity() -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
    first = ax.add_patch(Rectangle((0.15, 0.2), 0.25, 0.3))
    second = ax.add_patch(Rectangle((0.55, 0.2), 0.25, 0.3))
    fig.canvas.draw()
    manager = attach_drag_manager(fig)
    manager.select_elements([first, second], primary=second)
    group = manager.group_selection("Pair")
    group_id = group.group_id
    edit = fig.change_tracker.edit

    edit[0]()
    assert manager.editor_scene.groups == {}
    assert manager.editor_scene.selection_parent(first) is ax

    edit[1]()
    restored = manager.editor_scene.groups[group_id]
    assert restored.name == "Pair"
    assert restored.members == [first, second]
    assert manager.editor_scene.selection_parent(first) is restored

    manager.selection.clear_targets()
    plt.close(fig)
    assert app is not None


def test_object_tree_shows_logical_group_under_common_matplotlib_owner() -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
    first = ax.add_patch(Rectangle((0.15, 0.2), 0.25, 0.3))
    second = ax.add_patch(Rectangle((0.55, 0.2), 0.25, 0.3))
    fig.canvas.draw()
    manager = attach_drag_manager(fig)
    manager.select_elements([first, second], primary=second)
    group = manager.group_selection("Pair")

    signals = TreeSignals()
    container = QtWidgets.QWidget()
    layout = QtWidgets.QVBoxLayout(container)
    tree = MyTreeView(signals, layout)
    signals.figure_changed.emit(fig)
    tree.expand(fig)
    tree.expand(ax)

    item = tree.getItemFromEntry(group)
    assert item is not None
    assert item.text() == "Pair"
    assert tree.getParentEntry(group) is ax
    assert tree.queryToExpandEntry(group) == [first, second]

    manager.selection.clear_targets()
    tree.deleteLater()
    container.deleteLater()
    plt.close(fig)
    assert app is not None


def test_locked_and_hidden_layer_state_is_serializable_and_excluded_from_hits() -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
    rectangle = ax.add_patch(Rectangle((0.2, 0.2), 0.6, 0.6, zorder=5))
    fig.canvas.draw()
    manager = attach_drag_manager(fig)
    x, y = ax.transData.transform((0.5, 0.5))
    event = MouseEvent("button_press_event", fig.canvas, x, y, button=1)
    assert rectangle in manager.get_hit_candidates(event)

    manager.select_element(rectangle)
    assert manager.set_selection_locked(True)
    assert rectangle not in manager.get_hit_candidates(event)
    assert manager.editor_scene.export_state()["locked"]

    assert manager.unlock_all()
    manager.select_element(rectangle)
    assert manager.set_selection_visible(False)
    assert not rectangle.get_visible()
    assert rectangle not in manager.get_hit_stack(event).artists
    assert manager.editor_scene.export_state()["hidden"]

    assert manager.show_all()
    assert rectangle.get_visible()
    assert rectangle in manager.get_hit_candidates(event)

    manager.selection.clear_targets()
    plt.close(fig)
    assert app is not None


def _install_real_history_tracker(fig):
    from pylustrator.change_tracker import ChangeTracker as RealChangeTracker

    tracker = RealChangeTracker.__new__(RealChangeTracker)
    tracker.figure = fig
    tracker.changes = {}
    tracker.saved = True
    tracker.edits = []
    tracker.last_edit = -1
    tracker.update_changes_signal = None
    tracker.no_save = False
    fig.change_tracker = tracker
    return tracker


def test_semantic_noop_drag_restores_exact_state_without_dirtying_history() -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
    rectangle = ax.add_patch(Rectangle((0.2, 0.3), 0.25, 0.2))
    fig.canvas.draw()
    manager = attach_drag_manager(fig)
    tracker = _install_real_history_tracker(fig)
    manager.select_element(rectangle)
    before = TargetWrapper(rectangle).get_restore_state()

    manager.selection.start_move()
    manager.selection.addOffset((12, -7), DIR_X0 | DIR_X1 | DIR_Y0 | DIR_Y1)
    manager.selection.addOffset((0, 0), DIR_X0 | DIR_X1 | DIR_Y0 | DIR_Y1)
    manager.selection.has_moved = True
    manager.selection.end_move()

    assert tracker.edits == []
    assert tracker.changes == {}
    assert tracker.saved
    assert semantic_equal(TargetWrapper(rectangle).get_restore_state(), before, atol=0)

    manager.selection.clear_targets()
    plt.close(fig)
    assert app is not None


def test_undo_redo_preserves_selection_and_isolation_scope() -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
    ax.plot([0, 1], [0, 1], label="line")
    legend = ax.legend()
    fig.canvas.draw()
    manager = attach_drag_manager(fig)
    tracker = _install_real_history_tracker(fig)
    text = legend.get_texts()[0]
    assert manager.enter_isolation(legend)
    manager.select_element(text)

    manager.selection.start_move()
    manager.selection.addOffset((8, -3), DIR_X0 | DIR_X1 | DIR_Y0 | DIR_Y1)
    manager.selection.has_moved = True
    manager.selection.end_move()
    assert len(tracker.edits) == 1

    manager.undo()
    assert manager.isolation_breadcrumbs == ("Legend",)
    assert manager.selected_element is text
    assert [target.target for target in manager.selection.targets] == [text]

    manager.redo()
    assert manager.isolation_breadcrumbs == ("Legend",)
    assert manager.selected_element is text
    assert [target.target for target in manager.selection.targets] == [text]

    manager.selection.clear_targets()
    plt.close(fig)
    assert app is not None


def test_repeated_arrow_nudges_coalesce_into_one_undoable_command() -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
    rectangle = ax.add_patch(Rectangle((0.2, 0.3), 0.25, 0.2))
    fig.canvas.draw()
    manager = attach_drag_manager(fig)
    tracker = _install_real_history_tracker(fig)
    manager.select_element(rectangle)
    before = TargetWrapper(rectangle).get_selection_points().copy()

    event = KeyEvent("key_press_event", fig.canvas, "right")
    manager.selection.keyPressEvent(event)
    manager.selection.keyPressEvent(event)

    assert len(tracker.edits) == 1
    moved = TargetWrapper(rectangle).get_selection_points().copy()
    assert np.allclose(moved - before, [2, 0])
    manager.undo()
    assert np.allclose(TargetWrapper(rectangle).get_selection_points(), before)
    manager.redo()
    assert np.allclose(TargetWrapper(rectangle).get_selection_points(), moved)

    manager.selection.clear_targets()
    plt.close(fig)
    assert app is not None


def test_multi_selection_has_visible_per_target_indicators() -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    _scene, origin = make_selection_scene()

    fig, axes = plt.subplots(1, 2, figsize=(4, 2), dpi=100)
    fig._pyl_scene = origin
    selection = GrabbableRectangleSelection(fig, origin)

    selection.add_target(axes[0])
    selection.add_target(axes[1])

    assert len(selection.targets) == 2
    assert len(selection.targets_rects) == 4

    primary_rects = selection.targets_rects[0::2]
    contrast_rects = selection.targets_rects[1::2]
    for rect in primary_rects:
        assert rect.isVisible()
        assert rect.zValue() >= 900
        assert rect.pen().width() >= 3
        assert rect.pen().color().name().lower() == "#1e88e5"
        assert rect.brush().color().alpha() > 0
    for rect in contrast_rects:
        assert rect.isVisible()
        assert rect.zValue() > primary_rects[0].zValue()
        assert rect.pen().style() == QtCore.Qt.DashLine

    selection.clear_targets()
    plt.close(fig)
    assert app is not None


def test_axes_selection_indicator_updates_after_axes_move() -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    _scene, origin = make_selection_scene()

    fig, ax = plt.subplots(figsize=(4, 2), dpi=100)
    fig._pyl_scene = origin
    fig.canvas.draw()
    selection = GrabbableRectangleSelection(fig, origin)
    selection.add_target(ax)

    wrapper = TargetWrapper(ax)
    original = ax.get_position().frozen()
    moved = [original.x0 + 0.08, original.y0 + 0.04, original.width, original.height]
    ax.set_position(moved)
    selection.update_selection_rectangles()

    expected_points = wrapper.get_positions()
    expected_x0 = min(point[0] for point in expected_points)
    expected_y0 = min(point[1] for point in expected_points)
    rect = selection.targets_rects[0].rect()

    assert rect.x() == expected_x0
    assert rect.y() == expected_y0
    selection.clear_targets()
    plt.close(fig)
    assert app is not None


def test_single_selection_aligns_to_canvas_bounds() -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    fig, ax = plt.subplots(figsize=(4, 2), dpi=100)
    fig.canvas.draw()
    manager = attach_drag_manager(fig)
    ax.set_position([0.2, 0.2, 0.3, 0.3])
    manager.selection.add_target(ax)

    manager.selection.align_points("left_x")
    fig.canvas.draw()

    assert abs(ax.get_position().x0) < 1e-9
    manager.selection.clear_targets()
    plt.close(fig)
    assert app is not None


def test_multi_selection_aligns_to_selection_bounds() -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    fig, axes = plt.subplots(1, 2, figsize=(4, 2), dpi=100)
    fig.canvas.draw()
    manager = attach_drag_manager(fig)
    axes[0].set_position([0.2, 0.2, 0.3, 0.3])
    axes[1].set_position([0.6, 0.1, 0.2, 0.2])
    manager.selection.add_target(axes[0])
    manager.selection.add_target(axes[1])

    manager.selection.align_points("left_x")
    fig.canvas.draw()

    assert abs(axes[0].get_position().x0 - 0.2) < 1e-9
    assert abs(axes[1].get_position().x0 - 0.2) < 1e-9
    manager.selection.clear_targets()
    plt.close(fig)
    assert app is not None


def test_restore_point_uses_artist_local_coordinates_after_parent_moves() -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
    text = ax.text(0.5, 0.5, "label")
    fig.canvas.draw()
    manager = attach_drag_manager(fig)
    manager.selection.add_target(text)

    restore = manager.selection.get_save_point()
    ax.set_position([0.1, 0.1, 0.8, 0.8])
    text.set_position((0.2, 0.2))
    restore()

    assert abs(text.get_position()[0] - 0.5) < 1e-9
    assert abs(text.get_position()[1] - 0.5) < 1e-9
    manager.selection.clear_targets()
    plt.close(fig)
    assert app is not None


def test_legend_restore_point_uses_anchor_coordinates_after_figure_resize() -> None:
    from pylustrator.snap import TargetWrapper

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
    fig.change_tracker = ChangeTracker()
    legend = fig.legend(
        handles=[Patch(label="A"), Patch(label="B")],
        loc="upper center",
        bbox_to_anchor=(0.6, 0.9),
    )
    fig.canvas.draw()
    wrapper = TargetWrapper(legend)
    state = wrapper.get_restore_state()

    fig.set_size_inches(6, 3, forward=True)
    legend.set_bbox_to_anchor((0.1, 0.1), transform=fig.transFigure)
    wrapper.restore_state(state)
    fig.canvas.draw()

    expected = fig.transFigure.transform((0.6, 0.9))
    actual = legend.get_bbox_to_anchor().p0
    assert abs(actual[0] - expected[0]) < 1e-9
    assert abs(actual[1] - expected[1]) < 1e-9
    plt.close(fig)
    assert app is not None


def test_drag_motion_uses_original_display_geometry_for_legend_text() -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
    fig.canvas.draw()
    manager = attach_drag_manager(fig)
    legend = fig.legend(handles=[Patch(label="A"), Patch(label="B")])
    fig.canvas.draw()
    text = legend.get_texts()[0]
    before = text.get_window_extent(fig.canvas.get_renderer()).frozen()
    manager.selection.add_target(text)

    manager.selection.start_move()
    for offset in ((3, -2), (7, -4), (12, -7)):
        manager.selection.addOffset(offset, DIR_X0 | DIR_X1 | DIR_Y0 | DIR_Y1)
    manager.selection.has_moved = True
    manager.selection.end_move()
    fig.canvas.draw()

    after = text.get_window_extent(fig.canvas.get_renderer())
    assert abs((after.x0 - before.x0) - 12) < 1e-9
    assert abs((after.y0 - before.y0) + 7) < 1e-9
    manager.selection.clear_targets()
    plt.close(fig)
    assert app is not None


def test_legend_child_drag_moves_only_selected_child() -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
    fig.canvas.draw()
    manager = attach_drag_manager(fig)
    legend = fig.legend(handles=[Patch(label="A"), Patch(label="B")])
    fig.canvas.draw()
    text = legend.get_texts()[0]
    renderer = fig.canvas.get_renderer()
    text_before = text.get_window_extent(renderer).frozen()
    legend_before = legend.get_window_extent(renderer).frozen()
    text_position = text.get_position()
    anchor_before = legend.get_bbox_to_anchor().bounds
    manager.selection.defer_artist_updates = True
    manager.selection.add_target(text)

    manager.selection.start_move()
    manager.selection.move(
        (12, -7),
        DIR_X0 | DIR_X1 | DIR_Y0 | DIR_Y1,
        [],
        ignore_snaps=True,
    )
    manager.selection.end_move()
    fig.canvas.draw()

    text_after = text.get_window_extent(renderer)
    legend_after = legend.get_window_extent(renderer)
    assert text.get_position() != text_position
    assert abs((text_after.x0 - text_before.x0) - 12) < 1e-9
    assert abs((text_after.y0 - text_before.y0) + 7) < 1e-9
    assert legend.get_bbox_to_anchor().bounds == anchor_before
    assert legend_after.bounds == legend_before.bounds
    assert fig.change_tracker.legend_change_count == 0
    assert fig.change_tracker.text is text
    assert fig.change_tracker.text_change_count == 1
    manager.selection.clear_targets()
    plt.close(fig)
    assert app is not None


def test_extra_axes_legend_is_registered_and_directly_selectable() -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
    first = ax.legend(handles=[Patch(label="first")], loc="upper left")
    ax.add_artist(first)
    current = ax.legend(handles=[Patch(label="current")], loc="upper right")
    fig.canvas.draw()

    manager = attach_drag_manager(fig)
    registered = manager._selectable_artists

    assert first in registered
    assert current in registered
    assert first.pickable()
    assert current.pickable()
    manager.select_element(first)
    assert manager.selected_element is first
    assert [target.target for target in manager.selection.targets] == [first]
    manager.selection.clear_targets()
    plt.close(fig)
    assert app is not None


def test_overlapping_legends_follow_matplotlib_draw_order_for_child_hits() -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
    extra = ax.legend(handles=[Patch(label="same")], loc="center")
    ax.add_artist(extra)
    current = ax.legend(handles=[Patch(label="same")], loc="center")
    fig.canvas.draw()
    manager = attach_drag_manager(fig)
    current_text = current.get_texts()[0]
    extra_text = extra.get_texts()[0]
    bbox = current_text.get_window_extent(fig.canvas.get_renderer())
    x = (bbox.x0 + bbox.x1) / 2
    y = (bbox.y0 + bbox.y1) / 2
    event = MouseEvent("button_press_event", fig.canvas, x, y, button=1)

    assert extra_text.contains(event)[0]
    assert current_text.contains(event)[0]
    assert ax.get_children().index(current) > ax.get_children().index(extra)
    assert manager.get_picked_element(event)[0] is current_text

    manager.selection.clear_targets()
    plt.close(fig)
    assert app is not None


def test_yaxis_label_drag_keeps_selection_box_and_text_together() -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
    ax.set_ylabel("Y label")
    fig.canvas.draw()
    manager = attach_drag_manager(fig)
    manager.selection.defer_artist_updates = True
    label = ax.yaxis.label
    renderer = fig.canvas.get_renderer()
    manager.selection.add_target(label)
    text_before = label.get_window_extent(renderer).frozen()
    rect_before = selection_rect_extents(manager.selection)[0]

    manager.selection.start_move()
    manager.selection.addOffset((-20, 0), DIR_X0 | DIR_X1 | DIR_Y0 | DIR_Y1)
    preview_rect = selection_rect_extents(manager.selection)[0]
    assert abs((preview_rect[0] - rect_before[0]) + 20) < 1e-9
    assert abs(preview_rect[1] - rect_before[1]) < 1e-9
    assert label.get_window_extent(renderer).bounds == text_before.bounds

    manager.selection.has_moved = True
    manager.selection.end_move()
    fig.canvas.draw()
    manager.selection.update_selection_rectangles()

    text_after = label.get_window_extent(renderer)
    rect_after = selection_rect_extents(manager.selection)[0]
    assert abs((text_after.x0 - text_before.x0) + 20) < 1e-9
    assert abs(text_after.y0 - text_before.y0) < 1e-9
    assert abs((rect_after[0] - rect_before[0]) + 20) < 1e-9
    assert abs(rect_after[1] - rect_before[1]) < 1e-9
    manager.selection.clear_targets()
    plt.close(fig)
    assert app is not None


def test_annotation_with_mixed_coordinate_systems_moves_as_one_visual_object() -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
    annotation = ax.annotate(
        "mixed coords",
        xy=(0.2, 0.3),
        xycoords="data",
        xytext=(0.7, 0.8),
        textcoords="axes fraction",
        arrowprops={"arrowstyle": "->"},
    )
    fig.canvas.draw()
    manager = attach_drag_manager(fig)
    renderer = fig.canvas.get_renderer()
    before = annotation.get_window_extent(renderer).frozen()
    manager.select_element(annotation)

    manager.selection.start_move()
    manager.selection.move(
        (19, -11),
        DIR_X0 | DIR_X1 | DIR_Y0 | DIR_Y1,
        [],
        ignore_snaps=True,
    )
    manager.selection.end_move()
    fig.canvas.draw()
    manager.selection.update_selection_rectangles()

    after = annotation.get_window_extent(renderer)
    rect = selection_rect_extents(manager.selection)[0]
    visible_after = selection_target_extents(manager.selection)[0]
    # Arrow clipping/shrink is recomputed from the translated endpoints and may
    # move an extreme by a few thousandths of a pixel.
    assert abs((after.x0 - before.x0) - 19) < 0.02
    assert abs((after.y0 - before.y0) + 11) < 0.02
    assert np.allclose(rect, visible_after)
    manager.selection.clear_targets()
    plt.close(fig)
    assert app is not None


def test_cross_axes_clipped_annotation_alignment_is_rejected_before_mutation() -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    fig, axes = plt.subplots(1, 2, figsize=(6, 3), dpi=100)
    annotations = [
        ax.annotate(
            "QA",
            xy=(0.28, 0.32),
            xycoords="data",
            xytext=(0.7, 0.78),
            textcoords="axes fraction",
            arrowprops={"arrowstyle": "->"},
        )
        for ax in axes
    ]
    fig.canvas.draw()
    manager = attach_drag_manager(fig)
    manager.select_elements(annotations, primary=annotations[-1])
    before = [TargetWrapper(item).get_restore_state() for item in annotations]

    support = manager.selection.operation_support("translate")
    assert "annotated_point_within_owning_axes" in support.constraints
    with pytest.raises(TypeError, match="outside the owning Axes"):
        manager.selection.align_points("left_x")

    after = [TargetWrapper(item).get_restore_state() for item in annotations]
    assert all(semantic_equal(old, new) for old, new in zip(before, after))
    assert not hasattr(fig.change_tracker, "edit")
    manager.selection.clear_targets()
    plt.close(fig)
    assert app is not None


def test_cross_axes_fully_clipped_patch_alignment_is_rejected_atomically() -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    fig, axes = plt.subplots(1, 2, figsize=(6, 3), dpi=100)
    rectangles = [
        ax.add_patch(Rectangle((0.6, 0.4), 0.2, 0.2)) for ax in axes
    ]
    fig.canvas.draw()
    manager = attach_drag_manager(fig)
    manager.select_elements(rectangles, primary=rectangles[-1])
    before = [TargetWrapper(item).get_restore_state() for item in rectangles]

    with pytest.raises(TypeError, match="entirely outside the active clip region"):
        manager.selection.align_points("left_x")

    after = [TargetWrapper(item).get_restore_state() for item in rectangles]
    assert all(semantic_equal(old, new) for old, new in zip(before, after))
    assert not hasattr(fig.change_tracker, "edit")
    manager.selection.clear_targets()
    plt.close(fig)
    assert app is not None


def test_deferred_translation_preflight_uses_pre_preview_geometry() -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
    rectangle = ax.add_patch(Rectangle((0.88, 0.4), 0.04, 0.1))
    fig.canvas.draw()
    manager = attach_drag_manager(fig)
    manager.select_element(rectangle)
    wrapper = TargetWrapper(rectangle)
    before = wrapper.get_positions().copy()
    delta = ax.transData.transform((0.07, 0.0)) - ax.transData.transform((0.0, 0.0))

    manager.selection.start_move()
    manager.selection.move(
        delta,
        DIR_X0 | DIR_X1 | DIR_Y0 | DIR_Y1,
        [],
        ignore_snaps=True,
    )
    manager.selection.end_move()
    fig.canvas.draw()

    assert np.allclose(wrapper.get_positions(), before + delta)
    fig.change_tracker.edit[0]()
    assert np.allclose(wrapper.get_positions(), before)
    manager.selection.clear_targets()
    plt.close(fig)
    assert app is not None


def test_partially_clipped_drag_keeps_preview_selection_on_visible_paint() -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
    rectangle = ax.add_patch(
        Rectangle((0.72, 0.35), 0.2, 0.25, linewidth=2, color="red")
    )
    fig.canvas.draw()
    manager = attach_drag_manager(fig)
    manager.select_element(rectangle)
    delta = np.array([40.0, 0.0])

    manager.selection.start_move()
    manager.selection.move(
        delta,
        DIR_X0 | DIR_X1 | DIR_Y0 | DIR_Y1,
        [],
        ignore_snaps=True,
    )
    preview = manager.selection.move_current_selection_points[id(rectangle)].copy()
    assert np.max(preview[:, 0]) == pytest.approx(ax.bbox.x1)
    manager.selection.end_move()
    fig.canvas.draw()
    committed = TargetWrapper(rectangle).get_selection_points()

    painted = np.asarray(fig.canvas.buffer_rgba()).copy()
    rectangle.set_visible(False)
    fig.canvas.draw()
    without_rectangle = np.asarray(fig.canvas.buffer_rgba()).copy()
    changed_rows, changed_columns = np.where(
        np.any(painted != without_rectangle, axis=2)
    )
    painted_right = float(np.max(changed_columns))

    assert np.allclose(committed, preview, atol=0.25)
    assert np.max(committed[:, 0]) - painted_right <= 1.5
    rectangle.set_visible(True)
    fig.change_tracker.edit[0]()
    manager.selection.clear_targets()
    plt.close(fig)
    assert app is not None


def test_alignment_rejects_partial_clip_that_would_miss_requested_edge() -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    fig = plt.figure(figsize=(4, 3), dpi=100)
    left_axes = fig.add_axes((0.1, 0.1, 0.5, 0.8))
    right_axes = fig.add_axes((0.5, 0.1, 0.4, 0.8))
    left = left_axes.add_patch(Rectangle((0.75, 0.4), 0.1, 0.2))
    right = right_axes.add_patch(Rectangle((0.5, 0.4), 0.2, 0.2))
    fig.canvas.draw()
    manager = attach_drag_manager(fig)
    manager.select_elements([left, right], primary=right)
    before = [TargetWrapper(artist).get_restore_state() for artist in (left, right)]

    with pytest.raises(TypeError, match="visible bounds at the active clip region"):
        manager.selection.align_points("left_x")

    after = [TargetWrapper(artist).get_restore_state() for artist in (left, right)]
    assert all(semantic_equal(old, new) for old, new in zip(before, after))
    assert not hasattr(fig.change_tracker, "edit")
    manager.selection.clear_targets()
    plt.close(fig)
    assert app is not None


def test_non_rectangular_clip_rejects_bbox_overlap_without_paint_overlap() -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
    ax.set_aspect("equal")
    rectangle = ax.add_patch(Rectangle((0.45, 0.45), 0.05, 0.05, color="red"))
    rectangle.set_clip_path(Circle((0.5, 0.5), 0.3, transform=ax.transData))
    fig.canvas.draw()
    manager = attach_drag_manager(fig)
    wrapper = TargetWrapper(rectangle)
    before = wrapper.get_positions().copy()
    delta = ax.transData.transform((0.28, 0.28)) - ax.transData.transform((0.0, 0.0))

    with pytest.raises(TypeError, match="entirely outside the active clip region"):
        wrapper.translate(delta)

    assert np.allclose(wrapper.get_positions(), before)
    manager.selection.clear_targets()
    plt.close(fig)
    assert app is not None


def test_non_rectangular_partial_clip_selection_matches_raster_paint_bounds() -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
    ax.set_aspect("equal")
    rectangle = ax.add_patch(
        Rectangle((0.45, 0.45), 0.05, 0.05, color="red", linewidth=0)
    )
    rectangle.set_clip_path(Circle((0.5, 0.5), 0.3, transform=ax.transData))
    fig.canvas.draw()
    manager = attach_drag_manager(fig)
    wrapper = TargetWrapper(rectangle)
    delta = ax.transData.transform((0.23, 0.23)) - ax.transData.transform((0.0, 0.0))

    wrapper.translate(delta)
    fig.canvas.draw()
    selection_points = wrapper.get_selection_points()
    selection = np.array(
        [
            np.min(selection_points[:, 0]),
            np.min(selection_points[:, 1]),
            np.max(selection_points[:, 0]),
            np.max(selection_points[:, 1]),
        ],
        dtype=float,
    )
    painted = np.asarray(fig.canvas.buffer_rgba()).copy()
    rectangle.set_visible(False)
    fig.canvas.draw()
    without_rectangle = np.asarray(fig.canvas.buffer_rgba()).copy()
    rows, columns = np.where(np.any(painted != without_rectangle, axis=2))
    height = painted.shape[0]
    paint_bounds = np.array(
        [
            np.min(columns),
            height - 1 - np.max(rows),
            np.max(columns),
            height - 1 - np.min(rows),
        ],
        dtype=float,
    )

    assert np.allclose(selection, paint_bounds, atol=2.0)
    manager.selection.clear_targets()
    plt.close(fig)
    assert app is not None


def test_cross_axes_editor_group_preflights_every_clipped_member() -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    fig, axes = plt.subplots(1, 2, figsize=(6, 3), dpi=100)
    left = axes[0].add_patch(Rectangle((0.2, 0.4), 0.15, 0.2))
    right_members = [
        axes[1].add_patch(Rectangle((x, 0.4), 0.12, 0.2))
        for x in (0.55, 0.75)
    ]
    fig.canvas.draw()
    manager = attach_drag_manager(fig)
    manager.select_elements(right_members, primary=right_members[-1])
    group = manager.group_selection("Clipped pair")
    manager.select_elements([left, group], primary=group)
    before = TargetWrapper(group).get_restore_state()

    with pytest.raises(TypeError, match="entirely outside the active clip region"):
        manager.selection.align_points("left_x")

    assert semantic_equal(before, TargetWrapper(group).get_restore_state())
    manager.selection.clear_targets()
    plt.close(fig)
    assert app is not None


def test_unclipped_annotation_large_translation_invalidates_arrow_geometry() -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
    annotation = ax.annotate(
        "unclipped",
        xy=(0.28, 0.32),
        xycoords="data",
        xytext=(0.7, 0.78),
        textcoords="axes fraction",
        arrowprops={"arrowstyle": "->"},
        annotation_clip=False,
    )
    fig.canvas.draw()
    manager = attach_drag_manager(fig)
    wrapper = TargetWrapper(annotation)
    before = wrapper.get_selection_points().copy()
    delta = np.array([-80.0, 0.0])

    wrapper.translate(delta)
    fig.canvas.draw()
    after = wrapper.get_selection_points()

    assert np.allclose(after, before + delta, atol=0.05)
    manager.selection.clear_targets()
    plt.close(fig)
    assert app is not None


def test_text_visible_bbox_is_preserved_and_used_for_selection() -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
    text = ax.text(
        0.4,
        0.5,
        "boxed",
        bbox={"facecolor": "yellow", "edgecolor": "red", "pad": 8},
    )
    fig.canvas.draw()
    patch = text.get_bbox_patch()
    facecolor = patch.get_facecolor()
    manager = attach_drag_manager(fig)
    fig.canvas.draw()
    manager.select_element(text)
    renderer = fig.canvas.get_renderer()
    patch_bounds = patch.get_window_extent(renderer).extents
    stroke_radius = patch.get_linewidth() * fig.dpi / 72.0 / 2.0
    visible_patch_bounds = patch_bounds + np.array(
        [-stroke_radius, -stroke_radius, stroke_radius, stroke_radius]
    )
    selection_bounds = np.array(selection_rect_extents(manager.selection)[0])

    assert text.get_bbox_patch() is patch
    assert text.get_bbox_patch().get_facecolor() == facecolor
    assert np.allclose(selection_bounds, visible_patch_bounds)
    manager.selection.clear_targets()
    plt.close(fig)
    assert app is not None


def test_regular_polygon_drag_does_not_promote_selection_to_parent_axes() -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
    polygon = RegularPolygon((0.4, 0.5), 6, radius=0.15, transform=ax.transAxes)
    ax.add_patch(polygon)
    fig.canvas.draw()
    manager = attach_drag_manager(fig)
    renderer = fig.canvas.get_renderer()
    before = polygon.get_window_extent(renderer).frozen()
    axes_before = ax.get_position().frozen()
    center_before = polygon.xy

    manager.select_element(polygon)
    manager.selection.start_move()
    manager.selection.move(
        (16, -8),
        DIR_X0 | DIR_X1 | DIR_Y0 | DIR_Y1,
        [],
        ignore_snaps=True,
    )
    manager.selection.end_move()
    fig.canvas.draw()

    after = polygon.get_window_extent(renderer)
    assert manager.selected_element is polygon
    assert ax.get_position().bounds == axes_before.bounds
    assert abs((after.x0 - before.x0) - 16) < 1e-9
    assert abs((after.y0 - before.y0) + 8) < 1e-9

    fig.change_tracker.edit[0]()
    assert np.allclose(polygon.xy, center_before)
    manager.selection.clear_targets()
    plt.close(fig)
    assert app is not None


def test_unsupported_artist_is_not_silently_promoted_to_parent_axes() -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
    connection = ConnectionPatch(
        (0.2, 0.3),
        (0.8, 0.7),
        coordsA="data",
        coordsB="axes fraction",
        axesA=ax,
        axesB=ax,
    )
    ax.add_artist(connection)
    fig.canvas.draw()
    manager = attach_drag_manager(fig)

    manager.select_element(connection)

    assert manager.selected_element is None
    assert manager.selection.targets == []
    manager.selection.clear_targets()
    plt.close(fig)
    assert app is not None


def test_click_on_unsupported_artist_does_not_fall_through_to_parent_axes() -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
    connection = ConnectionPatch(
        (0.2, 0.3),
        (0.8, 0.7),
        coordsA="data",
        coordsB="data",
        axesA=ax,
        axesB=ax,
        linewidth=8,
        zorder=10,
    )
    ax.add_artist(connection)
    fig.canvas.draw()
    manager = attach_drag_manager(fig)
    x, y = ax.transData.transform((0.5, 0.5))
    press = MouseEvent("button_press_event", fig.canvas, x, y, button=1)
    release = MouseEvent("button_release_event", fig.canvas, x, y, button=1)

    assert connection.contains(press)[0]
    assert connection in manager._uneditable_artists
    assert manager.get_picked_element(press)[0] is None
    manager.select_element(ax)
    assert manager.selected_element is ax

    manager.button_press_event0(press)
    manager.button_release_event0(release)

    assert manager.selected_element is None
    assert manager.selection.targets == []
    manager.selection.clear_targets()
    plt.close(fig)
    assert app is not None


def test_marquee_on_unsupported_artist_does_not_select_parent_axes() -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
    connection = ConnectionPatch(
        (0.2, 0.3),
        (0.8, 0.7),
        coordsA="data",
        coordsB="data",
        axesA=ax,
        axesB=ax,
        linewidth=8,
        zorder=10,
    )
    ax.add_artist(connection)
    fig.canvas.draw()
    manager = attach_drag_manager(fig)
    x, y = ax.transData.transform((0.5, 0.5))

    selected = manager.select_elements_in_bbox(x - 8, y - 8, x + 8, y + 8)

    assert selected == []
    assert manager.selected_element is None
    assert manager.selection.targets == []
    manager.selection.clear_targets()
    plt.close(fig)
    assert app is not None


def test_supported_foreground_artist_wins_over_unsupported_blocker() -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
    connection = ConnectionPatch(
        (0.2, 0.3),
        (0.8, 0.7),
        coordsA="data",
        coordsB="data",
        axesA=ax,
        axesB=ax,
        linewidth=8,
        zorder=10,
    )
    ax.add_artist(connection)
    text = ax.text(0.5, 0.5, "top", ha="center", va="center", zorder=20)
    fig.canvas.draw()
    manager = attach_drag_manager(fig)
    x, y = ax.transData.transform((0.5, 0.5))
    event = MouseEvent("button_press_event", fig.canvas, x, y, button=1)

    assert connection.contains(event)[0]
    assert text.contains(event)[0]
    assert manager.get_picked_element(event)[0] is text
    assert not manager._last_pick_blocked

    manager.selection.clear_targets()
    plt.close(fig)
    assert app is not None


def test_figure_level_supported_artist_is_registered_and_referenceable() -> None:
    from pylustrator.change_tracker import getReference

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    fig = plt.figure(figsize=(4, 3), dpi=100)
    rectangle = Rectangle(
        (0.2, 0.3),
        0.25,
        0.2,
        transform=fig.transFigure,
    )
    fig.add_artist(rectangle)
    fig.canvas.draw()
    manager = attach_drag_manager(fig)

    assert rectangle in manager._selectable_artists
    assert getReference(rectangle).endswith(".artists[0]")
    assert eval(getReference(rectangle)) is rectangle
    manager.select_element(rectangle)
    assert manager.selected_element is rectangle

    manager.selection.clear_targets()
    plt.close(fig)
    assert app is not None


def test_scatter_collection_is_selectable_and_moves_in_its_own_coordinates() -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
    scatter = ax.scatter([0.2, 0.7], [0.3, 0.8], s=[25, 100])
    fig.canvas.draw()
    manager = attach_drag_manager(fig)
    before_offsets = np.asarray(scatter.get_offsets(), dtype=float).copy()
    wrapper = TargetWrapper(scatter)
    before_bounds = wrapper.get_selection_points().copy()
    expected_visible_bounds = wrapper.preview_translation_selection_points((14, -6))

    manager.select_element(scatter)
    manager.selection.start_move()
    manager.selection.move(
        (14, -6),
        DIR_X0 | DIR_X1 | DIR_Y0 | DIR_Y1,
        [],
        ignore_snaps=True,
    )
    manager.selection.end_move()
    fig.canvas.draw()
    after_bounds = wrapper.get_selection_points()

    assert manager.selected_element is scatter
    assert np.allclose(after_bounds, expected_visible_bounds)
    assert np.allclose(after_bounds[0] - before_bounds[0], [14, -6])
    assert after_bounds[1, 0] == pytest.approx(ax.bbox.x1)
    assert not np.allclose(scatter.get_offsets(), before_offsets)
    fig.change_tracker.edit[0]()
    assert np.allclose(scatter.get_offsets(), before_offsets)
    manager.selection.clear_targets()
    plt.close(fig)
    assert app is not None


def test_canvas_click_selects_scatter_instead_of_parent_axes() -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
    scatter = ax.scatter([0.25, 0.75], [0.3, 0.8], s=180)
    fig.canvas.draw()
    manager = attach_drag_manager(fig)
    x, y = ax.transData.transform((0.25, 0.3))
    press = MouseEvent("button_press_event", fig.canvas, x, y, button=1)
    release = MouseEvent("button_release_event", fig.canvas, x, y, button=1)

    manager.button_press_event0(press)
    manager.button_release_event0(release)

    assert manager.selected_element is scatter
    assert [target.target for target in manager.selection.targets] == [scatter]
    manager.selection.clear_targets()
    plt.close(fig)
    assert app is not None


def test_line_and_poly_collections_move_undo_and_replay_in_native_groups() -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    factories = [
        lambda: LineCollection(
            [
                [[0.2, 0.3], [0.8, 0.9]],
                [[1.0, 0.4], [1.3, 1.2]],
            ]
        ),
        lambda: PolyCollection(
            [
                [[0.2, 0.3], [0.8, 0.4], [0.6, 1.1]],
                [[1.0, 0.2], [1.4, 0.3], [1.2, 0.8]],
            ]
        ),
    ]

    for factory in factories:
        fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
        collection = factory()
        ax.add_collection(collection)
        ax.set_xlim(0, 2)
        ax.set_ylim(0, 2)
        fig.canvas.draw()
        manager = attach_drag_manager(fig)
        before = TargetWrapper(collection).get_selection_points().copy()

        manager.select_element(collection)
        manager.selection.start_move()
        manager.selection.move(
            (12, -5),
            DIR_X0 | DIR_X1 | DIR_Y0 | DIR_Y1,
            [],
            ignore_snaps=True,
        )
        manager.selection.end_move()
        fig.canvas.draw()

        moved = TargetWrapper(collection).get_selection_points().copy()
        assert np.allclose(moved - before, [12, -5])
        command_target, command = fig.change_tracker.change
        assert command_target is collection
        fig.change_tracker.edit[0]()
        fig.canvas.draw()
        assert np.allclose(TargetWrapper(collection).get_selection_points(), before)

        eval("collection" + command)
        fig.canvas.draw()
        assert np.allclose(TargetWrapper(collection).get_selection_points(), moved)
        manager.selection.clear_targets()
        plt.close(fig)
    assert app is not None


def test_resize_handles_require_lossless_scaling_for_every_selected_artist() -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
    scalable = ax.add_patch(Rectangle((0.1, 0.1), 0.2, 0.3))
    non_scalable = [
        ax.add_patch(Rectangle((0.4, 0.1), 0.2, 0.3, angle=25)),
        ax.add_patch(Ellipse((0.5, 0.6), 0.2, 0.3, angle=25)),
        ax.add_patch(FancyArrowPatch((0.1, 0.8), (0.3, 0.9))),
        ax.add_patch(FancyBboxPatch((0.4, 0.4), 0.2, 0.2, boxstyle="round,pad=0.1")),
        ax.add_patch(RegularPolygon((0.7, 0.7), 6, radius=0.1)),
        ax.add_patch(Wedge((0.8, 0.3), 0.1, 10, 250)),
        ax.scatter([0.2, 0.3], [0.5, 0.6]),
    ]
    fig.canvas.draw()
    manager = attach_drag_manager(fig)

    manager.select_element(scalable)
    assert manager.selection.do_target_scale()
    for artist in non_scalable:
        assert not TargetWrapper(artist).do_scale
        manager.select_elements([scalable, artist], primary=artist)
        assert not manager.selection.do_target_scale()

    manager.selection.clear_targets()
    plt.close(fig)
    assert app is not None


def test_rotation_routes_through_artist_capabilities_and_undo() -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
    text = ax.text(0.2, 0.7, "rotate")
    rectangle = ax.add_patch(Rectangle((0.5, 0.3), 0.2, 0.25))
    fig.canvas.draw()
    manager = attach_drag_manager(fig)
    fig.signals.figure_selection_property_changed = Signal()
    manager.select_elements([text, rectangle], primary=rectangle)

    assert manager.selection.rotate_selection(17)
    assert text.get_rotation() == 17
    assert rectangle.get_angle() == 17

    fig.change_tracker.edit[0]()
    assert text.get_rotation() == 0
    assert rectangle.get_angle() == 0
    manager.selection.clear_targets()
    plt.close(fig)
    assert app is not None


def test_rotation_handle_commits_arbitrary_native_angle_and_single_undo() -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
    text = ax.text(0.35, 0.55, "rotate by handle")
    rectangle = ax.add_patch(Rectangle((0.65, 0.2), 0.15, 0.2))
    fig.canvas.draw()
    manager = attach_drag_manager(fig)
    fig.signals.figure_selection_property_changed = Signal()
    manager.select_element(text)
    selection = manager.selection

    assert selection.rotation_handle_supported()
    assert selection.rotation_grabber.handle.isVisible()
    pivot = selection.rotation_pivot()
    start = np.asarray(selection.rotation_grabber.get_xy(), dtype=float)
    vector = start - pivot
    angle = np.deg2rad(37.0)
    rotated = pivot + np.array(
        [
            vector[0] * np.cos(angle) - vector[1] * np.sin(angle),
            vector[0] * np.sin(angle) + vector[1] * np.cos(angle),
        ]
    )

    selection.start_rotation(SimpleNamespace(x=start[0], y=start[1], key=None))
    preview = selection.preview_rotation(
        SimpleNamespace(x=rotated[0], y=rotated[1], key=None)
    )
    assert preview == pytest.approx(37.0)
    assert text.get_rotation() == pytest.approx(37.0)
    assert selection.end_rotation()
    assert text.get_rotation() == pytest.approx(37.0)

    fig.change_tracker.edit[0]()
    assert text.get_rotation() == pytest.approx(0.0)

    manager.select_elements([text, rectangle], primary=rectangle)
    assert not selection.rotation_handle_supported()
    assert not selection.rotation_grabber.handle.isVisible()
    manager.selection.clear_targets()
    plt.close(fig)
    assert app is not None


def test_legend_managed_text_hides_rotation_handle_when_layout_moves_pivot() -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
    ax.plot([0, 1], [0, 1], label="line")
    legend = ax.legend()
    fig.canvas.draw()
    manager = attach_drag_manager(fig)
    legend_text = legend.get_texts()[0]

    manager.select_elements([legend_text], primary=legend_text)

    support = TargetWrapper(legend_text).operation_support("rotate")
    assert not support.supported
    assert "stable native pivot" in support.reason
    assert not manager.selection.rotation_handle_supported()
    assert not manager.selection.rotation_grabber.handle.isVisible()
    manager.selection.clear_targets()
    plt.close(fig)
    assert app is not None


def test_non_affine_fancy_bbox_is_blocking_instead_of_inexactly_editable() -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
    ax.set_xscale("log")
    fancy = FancyBboxPatch(
        (0.2, 0.3),
        0.4,
        0.3,
        boxstyle="round,pad=0.1",
        linewidth=8,
        zorder=10,
    )
    ax.add_patch(fancy)
    ax.set_xlim(0.1, 1)
    ax.set_ylim(0, 1)
    fig.canvas.draw()
    manager = attach_drag_manager(fig)
    bbox = fancy.get_window_extent(fig.canvas.get_renderer())
    x = (bbox.x0 + bbox.x1) / 2
    y = (bbox.y0 + bbox.y1) / 2
    event = MouseEvent("button_press_event", fig.canvas, x, y, button=1)

    assert not TargetWrapper.supports_target(fancy)
    assert fancy in manager._uneditable_artists
    assert fancy.contains(event)[0]
    assert manager.get_picked_element(event)[0] is None

    manager.selection.clear_targets()
    plt.close(fig)
    assert app is not None


def test_axes_image_drag_preserves_camera_and_matches_selection_preview() -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
    image = ax.imshow([[0, 1], [1, 0]], extent=(0.2, 0.6, 0.3, 0.7))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.canvas.draw()
    manager = attach_drag_manager(fig)
    renderer = fig.canvas.get_renderer()
    before = image.get_window_extent(renderer).frozen()
    extent_before = image.get_extent()
    limits_before = (ax.get_xlim(), ax.get_ylim())

    manager.select_element(image)
    manager.selection.start_move()
    manager.selection.move(
        (13, -7),
        DIR_X0 | DIR_X1 | DIR_Y0 | DIR_Y1,
        [],
        ignore_snaps=True,
    )
    manager.selection.end_move()
    fig.canvas.draw()
    manager.selection.update_selection_rectangles()

    after = image.get_window_extent(renderer)
    rect = selection_rect_extents(manager.selection)[0]
    assert ax.get_xlim() == limits_before[0]
    assert ax.get_ylim() == limits_before[1]
    assert abs((after.x0 - before.x0) - 13) < 1e-9
    assert abs((after.y0 - before.y0) + 7) < 1e-9
    assert all(
        abs(actual - selected) < 1e-9 for actual, selected in zip(after.extents, rect)
    )
    moved_extent = image.get_extent()
    command_target, command = fig.change_tracker.change
    assert command_target is image
    image.set_extent(extent_before)
    ax.set_xlim(limits_before[0])
    ax.set_ylim(limits_before[1])
    eval("image" + command)
    assert np.allclose(image.get_extent(), moved_extent)
    assert ax.get_xlim() == limits_before[0]
    assert ax.get_ylim() == limits_before[1]
    manager.selection.clear_targets()
    plt.close(fig)
    assert app is not None


def test_shift_drag_constrains_selection_to_cardinal_direction() -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    fig, ax = plt.subplots(figsize=(4, 2), dpi=100)
    fig.canvas.draw()
    manager = attach_drag_manager(fig)
    manager.selection.defer_artist_updates = True
    ax.set_position([0.2, 0.2, 0.3, 0.4])
    manager.selection.add_target(ax)
    original = TargetWrapper(ax).get_positions()

    manager.selection.mouse_xy = (0, 0)
    manager.selection.start_move()
    event = MouseEvent(
        "motion_notify_event",
        fig.canvas,
        30,
        10,
        button=1,
        key="shift",
    )
    manager.selection.movedEvent(event)

    preview = manager.selection.move_current_positions[id(ax)]
    assert abs((preview[0][0] - original[0][0]) - 30) < 1e-9
    assert abs(preview[0][1] - original[0][1]) < 1e-9

    manager.selection.end_move()
    manager.selection.clear_targets()
    plt.close(fig)
    assert app is not None


def test_number_widget_ignores_non_scalar_linked_values() -> None:
    from pylustrator.QLinkableWidgets import NumberWidget

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    container = QtWidgets.QWidget()
    layout = QtWidgets.QHBoxLayout(container)
    signal = Signal()
    widget = NumberWidget(layout, "Linewidth:")
    widget.link("linewidth", signal)
    fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
    ax.scatter([0, 1], [0, 1], linewidths=[0.5, 1.0], label="points")
    legend = ax.legend()
    fig.canvas.draw()

    signal.emit(legend.legend_handles[0])

    widget.deleteLater()
    container.deleteLater()
    plt.close(fig)
    assert app is not None


def test_deferred_drag_updates_overlay_before_artist_and_commits_once() -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    fig, ax = plt.subplots(figsize=(4, 2), dpi=100)
    fig.canvas.draw()
    manager = attach_drag_manager(fig)
    manager.selection.defer_artist_updates = True
    ax.set_position([0.2, 0.2, 0.3, 0.4])
    manager.selection.add_target(ax)
    original = ax.get_position().frozen()

    manager.selection.start_move()
    manager.selection.addOffset((20, -10), DIR_X0 | DIR_X1 | DIR_Y0 | DIR_Y1)

    assert ax.get_position().bounds == original.bounds
    assert fig.change_tracker.axes_change_count == 0
    preview_rect = manager.selection.targets_rects[0].rect()
    preview_x = manager.selection.move_current_positions[id(ax)][0, 0]
    assert abs(preview_rect.x() - preview_x) < 1e-9

    manager.selection.has_moved = True
    manager.selection.end_move()

    assert ax.get_position().bounds != original.bounds
    assert fig.change_tracker.axes_change_count == 1
    manager.selection.clear_targets()
    plt.close(fig)
    assert app is not None


def test_deferred_thick_patch_resize_preview_matches_committed_visible_bounds() -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
    rectangle = ax.add_patch(
        Rectangle(
            (0.2, 0.25),
            0.35,
            0.3,
            facecolor="none",
            edgecolor="black",
            linewidth=18,
            label="qa-thick-resize",
        )
    )
    fig.canvas.draw()
    manager = attach_drag_manager(fig)
    manager.selection.defer_artist_updates = True
    manager.selection.add_target(rectangle)
    geometry_before = (
        tuple(rectangle.get_xy()),
        rectangle.get_width(),
        rectangle.get_height(),
    )
    visible_before = np.asarray(
        TargetWrapper(rectangle).get_selection_points(), dtype=float
    )

    manager.selection.start_move()
    manager.selection.move(
        (40, 0),
        DIR_X1,
        [],
        keep_aspect_ratio=False,
        ignore_snaps=True,
    )
    preview = manager.selection.move_current_selection_points[id(rectangle)]
    preview_bounds = np.array(
        [
            np.min(preview[:, 0]),
            np.min(preview[:, 1]),
            np.max(preview[:, 0]),
            np.max(preview[:, 1]),
        ]
    )
    before_bounds = np.array(
        [
            np.min(visible_before[:, 0]),
            np.min(visible_before[:, 1]),
            np.max(visible_before[:, 0]),
            np.max(visible_before[:, 1]),
        ]
    )

    assert (
        tuple(rectangle.get_xy()),
        rectangle.get_width(),
        rectangle.get_height(),
    ) == geometry_before
    assert preview_bounds[0] == pytest.approx(before_bounds[0])
    assert preview_bounds[2] == pytest.approx(before_bounds[2] + 40)
    assert np.allclose(selection_rect_extents(manager.selection)[0], preview_bounds)

    manager.selection.end_move()
    fig.canvas.draw()
    committed = selection_target_extents(manager.selection)[0]
    assert np.allclose(committed, preview_bounds, atol=0.25, rtol=0)
    assert rectangle.get_linewidth() == pytest.approx(18)
    manager.selection.clear_targets()
    plt.close(fig)
    assert app is not None


def test_deferred_drag_invalidates_preview_extent_cache() -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    fig, ax = plt.subplots(figsize=(4, 2), dpi=100)
    fig.canvas.draw()
    manager = attach_drag_manager(fig)
    manager.selection.defer_artist_updates = True
    ax.set_position([0.2, 0.2, 0.3, 0.4])
    manager.selection.add_target(ax)
    original_extent = TargetWrapper(ax).get_extent()

    manager.selection.start_move()
    manager.selection.addOffset((20, -10), DIR_X0 | DIR_X1 | DIR_Y0 | DIR_Y1)

    preview_extent = TargetWrapper(ax).get_extent()
    assert abs((preview_extent[0] - original_extent[0]) - 20) < 1e-9

    manager.selection.end_move()
    manager.selection.clear_targets()
    plt.close(fig)
    assert app is not None


def test_deferred_text_snap_position_uses_preview_geometry() -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    fig, ax = plt.subplots(figsize=(4, 2), dpi=100)
    text = ax.text(0.2, 0.5, "source")
    other = ax.text(0.8, 0.5, "other")
    fig.canvas.draw()
    manager = attach_drag_manager(fig)
    manager.selection.defer_artist_updates = True
    manager.selection.add_target(text)
    original_anchor = TargetWrapper(text).get_positions()[0]

    manager.selection.start_move()
    manager.selection.addOffset((25, 5), DIR_X0 | DIR_X1 | DIR_Y0 | DIR_Y1)

    preview_anchor = TargetWrapper(text).get_positions()[0]
    actual_anchor = text.get_transform().transform(text.get_position())
    snap = SnapSamePos(text, other, 0)
    try:
        assert abs(actual_anchor[0] - original_anchor[0]) < 1e-9
        assert abs((preview_anchor[0] - original_anchor[0]) - 25) < 1e-9
        assert abs(snap.getPosition(snap.ax_source)[0] - preview_anchor[0]) < 1e-9
    finally:
        snap.remove()
        manager.selection.end_move()
        manager.selection.clear_targets()
        plt.close(fig)
    assert app is not None


def test_immediate_drag_mode_still_updates_artist_during_motion() -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    fig, ax = plt.subplots(figsize=(4, 2), dpi=100)
    fig.canvas.draw()
    manager = attach_drag_manager(fig)
    manager.selection.defer_artist_updates = False
    ax.set_position([0.2, 0.2, 0.3, 0.4])
    manager.selection.add_target(ax)
    original = ax.get_position().frozen()

    manager.selection.start_move()
    manager.selection.addOffset((20, -10), DIR_X0 | DIR_X1 | DIR_Y0 | DIR_Y1)

    assert ax.get_position().bounds != original.bounds
    assert fig.change_tracker.axes_change_count == 1
    manager.selection.clear_targets()
    plt.close(fig)
    assert app is not None


def test_match_width_can_preserve_aspect_ratio() -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    fig, axes = plt.subplots(1, 2, figsize=(4, 2), dpi=100)
    fig.canvas.draw()
    manager = attach_drag_manager(fig)
    axes[0].set_position([0.1, 0.2, 0.4, 0.2])
    axes[1].set_position([0.6, 0.2, 0.2, 0.3])
    manager.selection.add_target(axes[0])
    manager.selection.add_target(axes[1])
    manager.selection.lock_aspect_ratio = True

    manager.selection.match_size("width")

    pos = axes[1].get_position()
    assert abs(pos.width - 0.4) < 1e-9
    assert abs(pos.height - 0.6) < 1e-9
    manager.selection.clear_targets()
    plt.close(fig)
    assert app is not None


def test_match_width_uses_visible_bounds_without_scaling_strokes() -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    fig, ax = plt.subplots(figsize=(4, 2), dpi=100)
    reference = ax.add_patch(
        Rectangle(
            (0.1, 0.2),
            0.35,
            0.3,
            facecolor="none",
            edgecolor="black",
            linewidth=4,
            label="qa-reference",
        )
    )
    target = ax.add_patch(
        Rectangle(
            (0.65, 0.25),
            0.12,
            0.2,
            facecolor="none",
            edgecolor="black",
            linewidth=20,
            label="qa-target",
        )
    )
    fig.canvas.draw()
    manager = attach_drag_manager(fig)
    manager.selection.add_target(reference)
    manager.selection.add_target(target)

    manager.selection.match_size("width", keep_aspect_ratio=False)
    fig.canvas.draw()
    reference_bounds, target_bounds = selection_target_extents(manager.selection)

    assert target_bounds[2] - target_bounds[0] == pytest.approx(
        reference_bounds[2] - reference_bounds[0], abs=0.25
    )
    assert reference.get_linewidth() == pytest.approx(4)
    assert target.get_linewidth() == pytest.approx(20)
    manager.selection.clear_targets()
    plt.close(fig)
    assert app is not None


def test_match_width_preflights_all_targets_before_clip_limited_resize() -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    fig, ax = plt.subplots(figsize=(4, 2), dpi=100)
    reference = Rectangle(
        (0.05, 0.05), 0.9, 0.1, transform=fig.transFigure, facecolor="none"
    )
    fig.add_artist(reference)
    safe_target = ax.add_patch(Rectangle((0.2, 0.2), 0.1, 0.2))
    clipped_target = ax.imshow(np.arange(16).reshape((4, 4)))
    fig.canvas.draw()
    manager = attach_drag_manager(fig)
    manager.select_elements(
        [reference, safe_target, clipped_target], primary=clipped_target
    )
    before = [
        TargetWrapper(target).get_restore_state()
        for target in (safe_target, clipped_target)
    ]

    with pytest.raises(TypeError, match="active clip region"):
        manager.selection.match_size("width", keep_aspect_ratio=False)

    after = [
        TargetWrapper(target).get_restore_state()
        for target in (safe_target, clipped_target)
    ]
    assert all(semantic_equal(old, new) for old, new in zip(before, after))
    assert not hasattr(fig.change_tracker, "edit")
    manager.selection.clear_targets()
    plt.close(fig)
    assert app is not None


def test_size_widget_undo_restores_axes_position() -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    container = QtWidgets.QWidget()
    layout = QtWidgets.QVBoxLayout(container)
    signals = WidgetSignals()
    fig, ax = plt.subplots(figsize=(4, 2), dpi=100)
    fig.signals = signals
    fig.selection = EmptySelection()
    fig.change_tracker = ChangeTracker()
    widget = QPosAndSize(layout, signals)
    widget.setFigure(fig)
    widget.setElement(ax)
    original = ax.get_position().frozen()

    widget.changeSize((0.4, 0.5))
    fig.change_tracker.edit[0]()

    restored = ax.get_position()
    assert abs(restored.x0 - original.x0) < 1e-9
    assert abs(restored.y0 - original.y0) < 1e-9
    assert abs(restored.width - original.width) < 1e-9
    assert abs(restored.height - original.height) < 1e-9
    container.deleteLater()
    plt.close(fig)
    assert app is not None


def test_size_widget_lock_aspect_resizes_from_changed_axis() -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    container = QtWidgets.QWidget()
    layout = QtWidgets.QVBoxLayout(container)
    signals = WidgetSignals()
    fig, ax = plt.subplots(figsize=(4, 2), dpi=100)
    fig.signals = signals
    fig.selection = EmptySelection()
    fig.change_tracker = ChangeTracker()
    widget = QPosAndSize(layout, signals)
    widget.setFigure(fig)
    widget.setElement(ax)
    ax.set_position([0.1, 0.2, 0.2, 0.4])
    widget.input_lock_aspect.setChecked(True)

    widget.changeSize((0.4, 0.1), changed_axis=0)

    pos = ax.get_position()
    assert abs(pos.width - 0.4) < 1e-9
    assert abs(pos.height - 0.8) < 1e-9
    container.deleteLater()
    plt.close(fig)
    assert app is not None


def test_selection_reference_point_is_non_mutating_editor_state() -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
    rectangle = ax.add_patch(Rectangle((0.2, 0.3), 0.25, 0.2, linewidth=3))
    fig.canvas.draw()
    manager = attach_drag_manager(fig)
    manager.select_element(rectangle)
    before = TargetWrapper(rectangle).get_selection_points().copy()
    bounds = manager.selection.selection_bounds()

    manager.selection.set_reference_point((1.0, 1.0))

    assert np.allclose(TargetWrapper(rectangle).get_selection_points(), before)
    assert np.allclose(manager.selection.reference_position(), bounds[2:])
    assert not hasattr(fig.change_tracker, "edit")
    manager.selection.clear_targets()
    plt.close(fig)
    assert app is not None


def test_numeric_selection_resize_keeps_active_reference_fixed_and_undoes() -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
    rectangle = ax.add_patch(Rectangle((0.2, 0.3), 0.25, 0.2, linewidth=3))
    fig.canvas.draw()
    manager = attach_drag_manager(fig)
    manager.select_element(rectangle)
    manager.selection.set_reference_point((0.0, 0.0))
    before = manager.selection.selection_bounds().copy()
    desired = before[2:] - before[:2] + np.array([30.0, 18.0])

    assert manager.selection.resize_selection_to(desired)
    fig.canvas.draw()
    after = manager.selection.selection_bounds()

    assert np.allclose(after[:2], before[:2], atol=0.25)
    assert np.allclose(after[2:] - after[:2], desired, atol=0.25)
    fig.change_tracker.edit[0]()
    fig.canvas.draw()
    assert np.allclose(manager.selection.selection_bounds(), before, atol=0.25)
    manager.selection.clear_targets()
    plt.close(fig)
    assert app is not None


def test_numeric_image_resize_rejects_clip_limited_visible_size_atomically() -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
    image = ax.imshow(np.arange(16).reshape((4, 4)))
    fig.canvas.draw()
    manager = attach_drag_manager(fig)
    manager.select_element(image)
    before = TargetWrapper(image).get_restore_state()
    bounds = manager.selection.selection_bounds()
    desired = bounds[2:] - bounds[:2] + np.array([9.0, 6.0])

    with pytest.raises(TypeError, match="active clip region"):
        manager.selection.resize_selection_to(desired)

    assert semantic_equal(before, TargetWrapper(image).get_restore_state())
    assert not hasattr(fig.change_tracker, "edit")
    manager.selection.clear_targets()
    plt.close(fig)
    assert app is not None


def test_transform_panel_moves_mixed_selection_by_visible_reference_point() -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    container = QtWidgets.QWidget()
    layout = QtWidgets.QVBoxLayout(container)
    signals = WidgetSignals()
    fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
    axes_text = ax.text(0.2, 0.4, "axes", transform=ax.transAxes)
    figure_text = fig.text(0.7, 0.7, "figure", transform=fig.transFigure)
    fig.canvas.draw()
    manager = attach_drag_manager(fig)
    fig.signals.figure_selection_property_changed = Signal()
    manager.select_elements([axes_text, figure_text], primary=axes_text)
    widget = QPosAndSize(layout, signals)
    widget.setFigure(fig)
    widget.setElement(axes_text)
    renderer = fig.canvas.get_renderer()
    axes_before = axes_text.get_window_extent(renderer).frozen()
    figure_before = figure_text.get_window_extent(renderer).frozen()
    reference_before = manager.selection.reference_position().copy()
    desired_display = reference_before + np.array([31.0, 0.0])
    desired_native = TargetWrapper(axes_text).transform_inverted_points(
        [desired_display]
    )[0]

    widget.changePos(float(desired_native[0]), None)
    fig.canvas.draw()

    axes_after = axes_text.get_window_extent(renderer)
    figure_after = figure_text.get_window_extent(renderer)
    assert axes_after.x0 - axes_before.x0 == pytest.approx(31.0, abs=1e-9)
    assert figure_after.x0 - figure_before.x0 == pytest.approx(31.0, abs=1e-9)
    assert np.allclose(manager.selection.reference_position(), desired_display)
    widget.input_reference.setValue((0.0, 0.0), emit=True)
    assert manager.selection.reference_point == (0.0, 0.0)
    manager.selection.clear_targets()
    container.deleteLater()
    plt.close(fig)
    assert app is not None


def test_position_widget_moves_mixed_transforms_by_one_display_delta() -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    container = QtWidgets.QWidget()
    layout = QtWidgets.QVBoxLayout(container)
    signals = WidgetSignals()
    fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
    axes_text = ax.text(0.2, 0.4, "axes", transform=ax.transAxes)
    figure_text = fig.text(0.7, 0.7, "figure", transform=fig.transFigure)
    fig.canvas.draw()
    manager = attach_drag_manager(fig)
    fig.signals = signals
    manager.select_elements([axes_text, figure_text], primary=axes_text)
    manager.selection.set_reference_point((0.0, 0.0))
    widget = QPosAndSize(layout, signals)
    widget.setFigure(fig)
    widget.setElement(axes_text)
    renderer = fig.canvas.get_renderer()
    axes_before = axes_text.get_window_extent(renderer).frozen()
    figure_before = figure_text.get_window_extent(renderer).frozen()
    positions_before = (axes_text.get_position(), figure_text.get_position())

    widget.changePos(0.3, None)
    fig.canvas.draw()

    axes_after = axes_text.get_window_extent(renderer)
    figure_after = figure_text.get_window_extent(renderer)
    axes_delta = axes_after.x0 - axes_before.x0
    figure_delta = figure_after.x0 - figure_before.x0
    assert abs(axes_delta - 31) < 1e-9
    assert abs(figure_delta - axes_delta) < 1e-9
    assert abs(axes_after.y0 - axes_before.y0) < 1e-9
    assert abs(figure_after.y0 - figure_before.y0) < 1e-9

    fig.change_tracker.edit[0]()
    assert np.allclose(axes_text.get_position(), positions_before[0])
    assert np.allclose(figure_text.get_position(), positions_before[1])
    manager.selection.clear_targets()
    container.deleteLater()
    plt.close(fig)
    assert app is not None


def test_position_widget_uses_subfigure_transform_for_axes() -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    container = QtWidgets.QWidget()
    layout = QtWidgets.QVBoxLayout(container)
    signals = WidgetSignals()
    fig = plt.figure(figsize=(6, 3), dpi=100)
    _left, right = fig.subfigures(1, 2)
    ax = right.subplots()
    fig.canvas.draw()
    manager = attach_drag_manager(fig)
    fig.signals = signals
    manager.select_element(ax)
    widget = QPosAndSize(layout, signals)
    widget.setFigure(fig)
    widget.transform_index = 2

    native = ax.get_position().p0
    widget_display = widget.getTransform(ax).transform(native)
    interaction_display = TargetWrapper(ax).get_positions()[0]

    assert np.allclose(widget_display, interaction_display)
    assert manager._selection_parent_by_id[id(ax)] is right
    manager.selection.clear_targets()
    container.deleteLater()
    plt.close(fig)
    assert app is not None


def test_transform_panel_converts_display_coordinates_to_physical_units_last() -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    container = QtWidgets.QWidget()
    layout = QtWidgets.QVBoxLayout(container)
    signals = WidgetSignals()
    fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
    text = ax.text(0.2, 0.4, "physical units", transform=ax.transAxes)
    fig.canvas.draw()
    fig.signals = signals
    fig.selection = EmptySelection()
    widget = QPosAndSize(layout, signals)
    widget.setFigure(fig)
    display = text.get_transform().transform(text.get_position())

    widget.transform_index = 0
    centimeters = widget.getTransform(text).transform(text.get_position())
    assert np.allclose(centimeters, display / fig.dpi * 2.54)

    widget.transform_index = 1
    inches = widget.getTransform(text).transform(text.get_position())
    assert np.allclose(inches, display / fig.dpi)

    widget.transform_index = 2
    figure_pixels = widget.getTransform(fig).transform(fig.get_size_inches())
    assert np.allclose(figure_pixels, np.asarray(fig.get_size_inches()) * fig.dpi)
    container.deleteLater()
    plt.close(fig)
    assert app is not None


def test_backspace_deletes_selected_object() -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    fig, ax = plt.subplots(figsize=(4, 2), dpi=100)
    fig.canvas.draw()
    manager = attach_drag_manager(fig)
    manager.selection.add_target(ax)

    manager.selection.keyPressEvent(
        KeyEvent("key_press_event", fig.canvas, "backspace")
    )

    assert fig.change_tracker.removed is ax
    assert not ax.get_visible()
    plt.close(fig)
    assert app is not None


def test_selection_scene_transform_maps_physical_canvas_pixels_to_logical_scene() -> (
    None
):
    transform = selection_scene_transform(2.0, 200)

    assert transform.map(100, 300) == (50, 50)


def test_scene_point_to_canvas_pixels_restores_physical_canvas_coordinates() -> None:
    view = SelectionView()
    view.h = 200
    view.device_pixel_ratio = 2.0

    assert scene_point_to_canvas_pixels(view, QtCore.QPointF(50, 50)) == (100, 300)


def test_drag_manager_select_elements_uses_single_multi_selection_model() -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    fig, axes = plt.subplots(1, 2, figsize=(4, 2), dpi=100)
    fig.canvas.draw()
    manager = attach_drag_manager(fig)

    manager.select_elements([axes[0], axes[1]], primary=axes[1])

    assert [target.target for target in manager.selection.targets] == [axes[0], axes[1]]
    assert manager.selected_element is axes[1]
    manager.selection.clear_targets()
    plt.close(fig)
    assert app is not None


def test_bulk_selection_updates_combined_extent_once() -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
    texts = [ax.text(0.1 + index * 0.08, 0.5, str(index)) for index in range(10)]
    fig.canvas.draw()
    manager = attach_drag_manager(fig)
    original_update = manager.selection.update_extent
    calls = 0

    def counted_update():
        nonlocal calls
        calls += 1
        original_update()

    manager.selection.update_extent = counted_update
    manager.select_elements(texts, primary=texts[-1])

    assert calls == 1
    assert [target.target for target in manager.selection.targets] == texts
    manager.selection.clear_targets()
    plt.close(fig)
    assert app is not None


def test_drag_rectangle_omits_containing_axes_by_default() -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
    text = ax.text(0.5, 0.5, "inside")
    fig.canvas.draw()
    manager = attach_drag_manager(fig)
    bbox = text.get_window_extent(fig.canvas.get_renderer()).expanded(1.2, 1.4)

    selected = manager.select_elements_in_bbox(bbox.x0, bbox.y0, bbox.x1, bbox.y1)

    assert selected == [text]
    assert text in [target.target for target in manager.selection.targets]
    assert ax not in [target.target for target in manager.selection.targets]
    manager.selection.clear_targets()
    plt.close(fig)
    assert app is not None


def test_marquee_reuses_one_geometry_snapshot_per_artist_and_action() -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
    rectangle = ax.add_patch(
        Rectangle((0.2, 0.25), 0.3, 0.35, label="qa-marquee-cache")
    )
    fig.canvas.draw()
    original_get_window_extent = rectangle.get_window_extent
    calls = 0

    def counted_get_window_extent(renderer=None):
        nonlocal calls
        calls += 1
        return original_get_window_extent(renderer)

    rectangle.get_window_extent = counted_get_window_extent
    manager = attach_drag_manager(fig)
    bounds = fig.bbox.extents

    manager.select_elements_in_bbox(*bounds)
    assert calls == 1

    manager.select_elements_in_bbox(*bounds)
    assert calls == 2
    manager.selection.clear_targets()
    plt.close(fig)
    assert app is not None


def test_marquee_reuses_nested_legend_child_geometry_per_action() -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
    ax.plot([0.1, 0.9], [0.2, 0.8], label="first")
    ax.plot([0.1, 0.9], [0.8, 0.3], label="second")
    legend = ax.legend(title="Legend title")
    fig.canvas.draw()
    children = [*legend.get_texts(), legend.get_title()]
    calls = {id(child): 0 for child in children}
    for child in children:
        original = child.get_window_extent

        def counted_get_window_extent(renderer=None, *, _child=child, _original=original):
            calls[id(_child)] += 1
            return _original(renderer)

        child.get_window_extent = counted_get_window_extent
    manager = attach_drag_manager(fig)

    manager.select_elements_in_bbox(*fig.bbox.extents)

    assert set(calls.values()) == {1}
    manager.selection.clear_targets()
    plt.close(fig)
    assert app is not None


def test_selection_snapshot_reuses_nested_editor_group_member_geometry() -> None:
    fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
    first = ax.add_patch(Rectangle((0.1, 0.2), 0.2, 0.25))
    second = ax.add_patch(Rectangle((0.6, 0.55), 0.25, 0.2))
    group = EditorGroup(
        fig, "qa-cache-group", [first, second], name="QA cache group"
    )
    fig.canvas.draw()
    calls = {id(first): 0, id(second): 0}
    for member in (first, second):
        original = member.get_window_extent

        def counted_get_window_extent(renderer=None, *, _member=member, _original=original):
            calls[id(_member)] += 1
            return _original(renderer)

        member.get_window_extent = counted_get_window_extent

    with selection_geometry_snapshot():
        TargetWrapper(group).get_selection_points()
        TargetWrapper(first).get_selection_points()
        TargetWrapper(second).get_selection_points()

    assert set(calls.values()) == {1}
    plt.close(fig)


def test_drag_rectangle_container_only_mode_keeps_parent_axes() -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
    text = ax.text(0.5, 0.5, "inside")
    fig.canvas.draw()
    manager = attach_drag_manager(fig)
    manager.marquee_select_containers_only = True
    bbox = text.get_window_extent(fig.canvas.get_renderer()).expanded(1.2, 1.4)

    selected = manager.select_elements_in_bbox(bbox.x0, bbox.y0, bbox.x1, bbox.y1)

    assert selected == [ax]
    assert [target.target for target in manager.selection.targets] == [ax]
    manager.selection.clear_targets()
    plt.close(fig)
    assert app is not None


def test_drag_rectangle_container_only_mode_replaces_selected_children() -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
    text = ax.text(0.5, 0.5, "inside")
    fig.canvas.draw()
    manager = attach_drag_manager(fig)
    manager.marquee_select_containers_only = True
    manager.select_element(text)
    bbox = text.get_window_extent(fig.canvas.get_renderer()).expanded(1.2, 1.4)

    manager.select_elements_in_bbox(bbox.x0, bbox.y0, bbox.x1, bbox.y1, additive=True)

    assert [target.target for target in manager.selection.targets] == [ax]
    manager.selection.clear_targets()
    plt.close(fig)
    assert app is not None


def test_drag_rectangle_empty_plot_area_requires_container_toggle() -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
    fig.canvas.draw()
    manager = attach_drag_manager(fig)
    bbox = ax.get_window_extent(fig.canvas.get_renderer())
    cx = (bbox.x0 + bbox.x1) / 2
    cy = (bbox.y0 + bbox.y1) / 2

    selected = manager.select_elements_in_bbox(cx - 10, cy - 10, cx + 10, cy + 10)

    assert selected == []
    assert manager.selection.targets == []

    manager.marquee_select_containers_only = True
    selected = manager.select_elements_in_bbox(cx - 10, cy - 10, cx + 10, cy + 10)

    assert selected == [ax]
    assert [target.target for target in manager.selection.targets] == [ax]
    manager.selection.clear_targets()
    plt.close(fig)
    assert app is not None


def test_multi_selection_prefers_axes_children_over_parent_axes() -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
    text = ax.text(0.5, 0.5, "inside")
    fig.canvas.draw()
    manager = attach_drag_manager(fig)

    manager.select_elements([ax, text], primary=ax)

    assert [target.target for target in manager.selection.targets] == [text]
    assert manager.selected_element is text
    manager.selection.clear_targets()
    plt.close(fig)
    assert app is not None


def test_drag_rectangle_prefers_specific_children_over_containing_axes() -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
    text = ax.text(0.5, 0.5, "inside")
    fig.canvas.draw()
    manager = attach_drag_manager(fig)
    bbox = ax.get_window_extent(fig.canvas.get_renderer()).expanded(1.02, 1.02)

    selected = manager.select_elements_in_bbox(bbox.x0, bbox.y0, bbox.x1, bbox.y1)

    assert selected == [text]
    assert [target.target for target in manager.selection.targets] == [text]
    assert manager.selected_element is text
    manager.selection.clear_targets()
    plt.close(fig)
    assert app is not None


def test_canvas_drag_rectangle_starts_on_axes_and_selects_after_release() -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
    text = ax.text(0.5, 0.5, "inside")
    fig.canvas.draw()
    manager = attach_drag_manager(fig)
    bbox = text.get_window_extent(fig.canvas.get_renderer()).expanded(1.2, 1.4)

    press = MouseEvent("button_press_event", fig.canvas, bbox.x0, bbox.y0, button=1)
    move = MouseEvent("motion_notify_event", fig.canvas, bbox.x1, bbox.y1, button=1)
    release = MouseEvent("button_release_event", fig.canvas, bbox.x1, bbox.y1, button=1)

    manager.button_press_event0(press)
    manager.motion_notify_event0(move)
    manager.button_release_event0(release)

    assert text in [target.target for target in manager.selection.targets]
    assert ax not in [target.target for target in manager.selection.targets]
    manager.selection.clear_targets()
    plt.close(fig)
    assert app is not None


def test_transparent_inset_children_remain_directly_selectable() -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    fig = plt.figure(figsize=(7.35, 8.8), dpi=150)
    fig.add_axes([0.1067, 0.1104, 0.525, 0.2646], label="panel_g")
    inset = fig.add_axes(
        [0.140742, 0.225633, 0.080784, 0.067473],
        label="panel_g_inset_1",
    )
    inset.imshow([[0, 1], [1, 0]], extent=(0.10, 0.90, 0.10, 0.90))
    fill_hex = RegularPolygon(
        (0.5, 0.5),
        numVertices=6,
        radius=0.492,
        orientation=0,
        transform=inset.transAxes,
        facecolor="white",
        edgecolor="none",
        zorder=1,
    )
    outline = RegularPolygon(
        (0.5, 0.5),
        numVertices=6,
        radius=0.492,
        orientation=0,
        transform=inset.transAxes,
        facecolor="none",
        edgecolor="#6F6F6F",
        linewidth=0.6,
        zorder=3,
    )
    inset.add_patch(fill_hex)
    inset.add_patch(outline)
    inset.set_xlim(0, 1)
    inset.set_ylim(0, 1)
    inset.set_xticks([])
    inset.set_yticks([])
    inset.set_facecolor("none")
    inset.patch.set_alpha(0.0)
    for spine in inset.spines.values():
        spine.set_visible(False)

    fig.canvas.draw()
    manager = attach_drag_manager(fig)
    manager.select_element(fill_hex)
    assert manager.selected_element is fill_hex
    assert [target.target for target in manager.selection.targets] == [fill_hex]
    manager.selection.clear_targets()
    manager.selected_element = None

    bbox = inset.get_window_extent(fig.canvas.get_renderer())
    x = (bbox.x0 + bbox.x1) / 2
    y = (bbox.y0 + bbox.y1) / 2
    press = MouseEvent("button_press_event", fig.canvas, x, y, button=1)
    release = MouseEvent("button_release_event", fig.canvas, x, y, button=1)

    picked, _finished = manager.get_picked_element(press)
    manager.button_press_event0(press)
    manager.button_release_event0(release)

    assert picked in (fill_hex, outline)
    assert manager.selected_element in (fill_hex, outline)
    assert [target.target for target in manager.selection.targets] == [
        manager.selected_element
    ]
    manager.selection.clear_targets()
    plt.close(fig)
    assert app is not None


def test_top_inset_axes_wins_over_artists_in_lower_axes_layer() -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    fig = plt.figure(figsize=(4, 3), dpi=100)
    lower = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    lower.scatter([0.5], [0.5], s=300)
    inset = fig.add_axes([0.35, 0.35, 0.3, 0.3])
    inset.patch.set_alpha(0)
    inset.set_xticks([])
    inset.set_yticks([])
    fig.canvas.draw()
    manager = attach_drag_manager(fig)
    x, y = inset.transAxes.transform((0.5, 0.5))
    event = MouseEvent("button_press_event", fig.canvas, x, y, button=1)

    picked, _finished = manager.get_picked_element(event)

    assert picked is inset
    manager.selection.clear_targets()
    plt.close(fig)
    assert app is not None


def test_tree_view_extended_selection_updates_drag_manager_selection() -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    signals = TreeSignals()
    container = QtWidgets.QWidget()
    layout = QtWidgets.QVBoxLayout(container)
    tree = MyTreeView(signals, layout)
    fig, axes = plt.subplots(1, 2, figsize=(4, 2), dpi=100)
    manager = attach_drag_manager(fig)

    signals.figure_changed.emit(fig)
    tree.expand(fig)
    first = tree.getItemFromEntry(axes[0]).index()
    second = tree.getItemFromEntry(axes[1]).index()

    tree.selectionModel().select(
        first,
        QtCore.QItemSelectionModel.ClearAndSelect | QtCore.QItemSelectionModel.Rows,
    )
    tree.selectionModel().select(
        second,
        QtCore.QItemSelectionModel.Select | QtCore.QItemSelectionModel.Rows,
    )
    tree.setCurrentIndex(second)

    assert tree.selectionMode() == QtWidgets.QAbstractItemView.ExtendedSelection
    assert [target.target for target in manager.selection.targets] == [axes[0], axes[1]]
    assert manager.selected_element is axes[1]
    manager.selection.clear_targets()
    plt.close(fig)
    container.deleteLater()
    assert app is not None


def test_tree_view_ctrl_click_uses_additive_row_selection_command() -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    signals = TreeSignals()
    container = QtWidgets.QWidget()
    layout = QtWidgets.QVBoxLayout(container)
    tree = MyTreeView(signals, layout)
    fig, axes = plt.subplots(1, 2, figsize=(4, 2), dpi=100)
    manager = attach_drag_manager(fig)

    signals.figure_changed.emit(fig)
    tree.expand(fig)
    first = tree.getItemFromEntry(axes[0]).index()
    second = tree.getItemFromEntry(axes[1]).index()
    event = QtGui.QMouseEvent(
        QtCore.QEvent.MouseButtonPress,
        QtCore.QPointF(0, 0),
        QtCore.Qt.LeftButton,
        QtCore.Qt.LeftButton,
        QtCore.Qt.ControlModifier,
    )
    command = tree.selectionCommand(second, event)

    tree.selectionModel().select(
        first,
        QtCore.QItemSelectionModel.ClearAndSelect | QtCore.QItemSelectionModel.Rows,
    )
    tree.selectionModel().select(second, command)
    tree.setCurrentIndex(second)

    assert command & QtCore.QItemSelectionModel.Toggle
    assert command & QtCore.QItemSelectionModel.Rows
    assert [target.target for target in manager.selection.targets] == [axes[0], axes[1]]
    assert manager.selected_element is axes[1]
    manager.selection.clear_targets()
    plt.close(fig)
    container.deleteLater()
    assert app is not None


def test_plot_window_marquee_container_toggle_updates_drag_manager() -> None:
    from pylustrator.QtGuiDrag import PlotWindow

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = PlotWindow(1)
    fig, _ax = plt.subplots(figsize=(4, 3), dpi=100)
    manager = attach_drag_manager(fig)

    window.setFigure(fig)
    assert manager.marquee_select_containers_only is False

    window.setMarqueeSelectContainersOnly(True)
    assert manager.marquee_select_containers_only is True

    window.setMarqueeSelectContainersOnly(False)
    assert manager.marquee_select_containers_only is False
    manager.selection.clear_targets()
    plt.close(fig)
    window.deleteLater()
    assert app is not None


def test_multi_selection_drag_keeps_legend_child_rectangles_aligned() -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    fig, ax = plt.subplots(figsize=(3.56, 3.35), dpi=100)
    legend = fig.legend(
        handles=[
            Patch(label="ipTM-oriented"),
            Patch(label="+ pocket-oriented"),
            Patch(label="+ trajectory rescue"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.628, 0.99),
        ncol=3,
        fontsize=5.25,
        handlelength=0.72,
        columnspacing=0.3,
        handletextpad=0.2,
        borderaxespad=0.0,
        labelspacing=0.3,
        borderpad=0.28,
        frameon=False,
    )
    fig.canvas.draw()
    manager = attach_drag_manager(fig)
    bbox = legend.get_window_extent(fig.canvas.get_renderer()).expanded(1.05, 1.2)

    manager.select_elements_in_bbox(bbox.x0, bbox.y0, bbox.x1, bbox.y1)
    manager.selection.start_move()
    manager.selection.addOffset((12, -7), DIR_X0 | DIR_X1 | DIR_Y0 | DIR_Y1)
    fig.canvas.draw()

    for target_extent, rect_extent in zip(
        selection_target_extents(manager.selection),
        selection_rect_extents(manager.selection),
    ):
        assert all(
            abs(target_value - rect_value) < 1e-9
            for target_value, rect_value in zip(target_extent, rect_extent)
        )

    manager.selection.clear_targets()
    plt.close(fig)
    assert app is not None


def test_failed_legend_commit_rolls_back_without_recording_again() -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
    legend = ax.legend(handles=[Patch(label="A")], frameon=False)
    fig.canvas.draw()
    manager = attach_drag_manager(fig)
    manager.select_element(legend)
    anchor_before = legend.get_bbox_to_anchor().bounds

    class FailingLegendTracker(ChangeTracker):
        def addNewLegendChange(self, target):
            raise RuntimeError("simulated legend serialization failure")

    fig.change_tracker = FailingLegendTracker()
    manager.selection.start_move()
    manager.selection.addOffset((12, -7), DIR_X0 | DIR_X1 | DIR_Y0 | DIR_Y1)
    manager.selection.has_moved = True

    try:
        manager.selection.end_move()
    except RuntimeError as error:
        assert str(error) == "simulated legend serialization failure"
    else:
        raise AssertionError("Legend commit unexpectedly succeeded")

    fig.canvas.draw()
    manager.selection.update_extent()
    manager.selection.update_selection_rectangles()
    assert legend.get_bbox_to_anchor().bounds == anchor_before
    assert np.allclose(
        selection_target_extents(manager.selection)[0],
        selection_rect_extents(manager.selection)[0],
    )
    manager.selection.clear_targets()
    plt.close(fig)
    assert app is not None


def test_frameon_property_keeps_the_live_drag_selection() -> None:
    from pylustrator.components.qitem_properties import LegendPropertiesWidget

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
    legend = ax.legend(handles=[Patch(label="A")], frameon=False)
    fig.canvas.draw()
    manager = attach_drag_manager(fig)
    manager.select_element(legend)
    renderer = fig.canvas.get_renderer()
    bounds_before = legend.get_window_extent(renderer).bounds

    widget = LegendPropertiesWidget.__new__(LegendPropertiesWidget)
    widget.properties = {"frameon": False}
    widget.target = legend
    widget.changePropertiy("frameon", True)

    assert ax.get_legend() is legend
    assert manager.selected_element is legend
    assert [target.target for target in manager.selection.targets] == [legend]
    assert legend.get_frame_on()

    manager.selection.start_move()
    manager.selection.addOffset((12, -7), DIR_X0 | DIR_X1 | DIR_Y0 | DIR_Y1)
    manager.selection.has_moved = True
    manager.selection.end_move()
    fig.canvas.draw()

    assert ax.get_legend() is legend
    bounds_after = legend.get_window_extent(renderer).bounds
    assert np.allclose(
        (bounds_after[0] - bounds_before[0], bounds_after[1] - bounds_before[1]),
        (12, -7),
    )
    manager.selection.clear_targets()
    plt.close(fig)
    assert app is not None


def test_pyl_show_initializes_axes_defaults_before_drag_changes() -> None:
    from matplotlib import _pylab_helpers
    from pylustrator import QtGuiDrag

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    created_windows = []

    class PlotWindowDouble:
        def __init__(self):
            self.fig = None
            self.updated = False
            created_windows.append(self)

        def setFigure(self, fig):
            self.fig = fig

        def addFigure(self, fig):
            assert self.fig is fig

        def update(self):
            self.updated = True

    original_plot_window = QtGuiDrag.PlotWindow
    original_app = QtGuiDrag.app
    try:
        QtGuiDrag.app = app
        fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
        _scene, origin = make_selection_scene()
        fig._pyl_scene = origin
        assert getattr(ax, "_pylustrator_old_args", None) is None

        QtGuiDrag.PlotWindow = PlotWindowDouble
        QtGuiDrag.pyl_show(hide_window=True)

        assert created_windows[0].updated is True
        assert getattr(ax, "_pylustrator_old_args", None) is not None
        selection = fig.selection
        fig.figure_dragger.select_element(ax)
        selection.start_move()
        selection.addOffset((2, 0), selection.dir)
        selection.end_move()
    finally:
        plt.close("all")
        _pylab_helpers.Gcf.destroy_all()
        QtGuiDrag.PlotWindow = original_plot_window
        QtGuiDrag.app = original_app
    assert app is not None
