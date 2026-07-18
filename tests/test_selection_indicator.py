from __future__ import annotations

import gc
from time import perf_counter
from types import SimpleNamespace
import weakref

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
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
    Polygon,
    Rectangle,
    RegularPolygon,
    Wedge,
)
from matplotlib.path import Path
from matplotlib.transforms import IdentityTransform
from qtpy import QtCore, QtGui, QtWidgets

from pylustrator.artist_adapters import (
    PolygonAdapter,
    UnsupportedArtistError,
    artist_adapter_registry,
    selection_geometry_snapshot,
)
from pylustrator.components.plot_layout import (
    scene_point_to_canvas_pixels,
    selection_scene_transform,
)
from pylustrator.components.align import Align
from pylustrator.components.qpos_and_size import QPosAndSize, ReferencePointWidget
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
from pylustrator.operations import TransformOperation


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
        self.edits = []
        self.changes = []
        self.last_edit = -1
        self.saved = True

    def addEdit(self, edit):
        self.edit = edit
        self.edits.append(edit)
        self.last_edit = len(self.edits) - 1

    def capture_recording_state(self):
        return list(self.changes), bool(self.saved)

    def restore_recording_state(self, state):
        changes, saved = state
        self.changes = list(changes)
        self.saved = bool(saved)

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
        self.changes.append((target, command))

    def removeElement(self, target):
        self.removed = target
        target.set_visible(False)


class Signals:
    def __init__(self):
        self.selected = []
        self.moved = False
        self.moved_count = 0

        class SelectionMoved:
            def __init__(self, parent):
                self.parent = parent

            def emit(self):
                self.parent.moved = True
                self.parent.moved_count += 1

        class ElementSelected:
            def __init__(self, parent):
                self.parent = parent

            def emit(self, element):
                self.parent.selected.append(element)

        self.figure_selection_moved = SelectionMoved(self)
        self.figure_element_selected = ElementSelected(self)
        self.figure_selection_update = Signal()
        self.figure_selection_property_changed = Signal()


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
        self.figure_selection_update = Signal()
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


def artist_visible_extent(artist):
    points = np.asarray(TargetWrapper(artist).get_selection_points(), dtype=float)
    return (
        float(np.min(points[:, 0])),
        float(np.min(points[:, 1])),
        float(np.max(points[:, 0])),
        float(np.max(points[:, 1])),
    )


def alignment_coordinate(extent, mode):
    coordinates = {
        "left_x": extent[0],
        "center_x": (extent[0] + extent[2]) / 2,
        "right_x": extent[2],
        "bottom_y": extent[1],
        "center_y": (extent[1] + extent[3]) / 2,
        "top_y": extent[3],
    }
    return coordinates[mode]


def add_figure_rectangle(fig, bounds, **kwargs):
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


def test_pointer_press_uses_one_resolution_and_foreground_wins_old_selection() -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
    lower = ax.add_patch(Rectangle((0.2, 0.2), 0.6, 0.6, zorder=2))
    upper = ax.add_patch(Rectangle((0.2, 0.2), 0.6, 0.6, zorder=5))
    fig.canvas.draw()
    manager = attach_drag_manager(fig)
    manager.select_element(lower)
    x, y = ax.transData.transform((0.5, 0.5))
    press = MouseEvent("button_press_event", fig.canvas, x, y, button=1)
    release = MouseEvent("button_release_event", fig.canvas, x, y, button=1)

    calls = {"top_hit": 0, "hit_stack": 0, "lower_contains": 0}
    original_get_hit_stack = manager.get_hit_stack
    original_resolve_top_hit = manager._resolve_top_hit
    original_lower_contains = lower.contains

    def counted_hit_stack(event):
        calls["hit_stack"] += 1
        return original_get_hit_stack(event)

    def counted_top_hit(event):
        calls["top_hit"] += 1
        return original_resolve_top_hit(event)

    def counted_lower_contains(event):
        calls["lower_contains"] += 1
        return original_lower_contains(event)

    def forbidden_legacy_pick(*_args, **_kwargs):
        raise AssertionError("pointer press must not perform a second raw-leaf lookup")

    manager.get_hit_stack = counted_hit_stack
    manager._resolve_top_hit = counted_top_hit
    manager.get_picked_element = forbidden_legacy_pick
    lower.contains = counted_lower_contains

    manager.button_press_event0(press)

    assert calls == {"top_hit": 1, "hit_stack": 0, "lower_contains": 0}
    assert manager.selected_element is upper
    assert [target.target for target in manager.selection.targets] == [upper]
    manager.button_release_event0(release)
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
    fig.signals.selected.clear()

    manager.button_press_event0(event)

    assert manager.isolation_breadcrumbs == ("Legend",)
    assert manager.selected_element is text
    assert [target.target for target in manager.selection.targets] == [text]
    assert fig.signals.selected == [text]

    fig.signals.selected.clear()
    manager.key_press_event(KeyEvent("key_press_event", fig.canvas, "escape"))
    assert manager.isolation_breadcrumbs == ()
    assert manager.selected_element is legend
    assert [target.target for target in manager.selection.targets] == [legend]
    assert fig.signals.selected == [legend]

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


def test_v_a_switch_reconciles_group_and_leaf_selection_semantics() -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
    first = ax.add_patch(Rectangle((0.15, 0.2), 0.25, 0.3, zorder=3))
    second = ax.add_patch(Rectangle((0.55, 0.2), 0.25, 0.3, zorder=3))
    fig.canvas.draw()
    manager = attach_drag_manager(fig)
    manager.select_elements([first, second], primary=second)
    group = manager.group_selection("Pair")

    manager.key_press_event(KeyEvent("key_press_event", fig.canvas, "a"))

    assert manager.selection_mode is SelectionMode.DIRECT
    assert manager.selection.targets == []
    assert manager.selected_element is None

    press = _center_event(fig, first)
    manager.button_press_event0(press)
    manager.button_release_event0(press)
    assert [target.target for target in manager.selection.targets] == [first]

    manager.key_press_event(KeyEvent("key_press_event", fig.canvas, "v"))

    assert manager.selection_mode is SelectionMode.OBJECT
    assert [target.target for target in manager.selection.targets] == [group]
    assert manager.selected_element is group
    manager.selection.clear_targets()
    plt.close(fig)
    assert app is not None


def test_direct_isolation_never_falls_back_to_leaf_outside_scope() -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
    first = ax.add_patch(Rectangle((0.1, 0.2), 0.2, 0.25, zorder=3))
    second = ax.add_patch(Rectangle((0.4, 0.2), 0.2, 0.25, zorder=3))
    outside = ax.add_patch(Rectangle((0.72, 0.65), 0.18, 0.2, zorder=8))
    fig.canvas.draw()
    manager = attach_drag_manager(fig)
    manager.select_elements([first, second], primary=second)
    group = manager.group_selection("Isolated pair")
    assert manager.enter_isolation(group)
    manager.set_selection_mode(SelectionMode.DIRECT)
    press = _center_event(fig, outside)

    manager.button_press_event0(press)

    assert manager.marquee_start is not None
    assert manager.selected_element is None
    assert manager.selection.targets == []
    manager.button_release_event0(press)
    assert outside not in [target.target for target in manager.selection.targets]
    manager.selection.clear_targets()
    plt.close(fig)
    assert app is not None


def test_shift_click_selected_member_toggles_without_starting_drag() -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
    first = ax.add_patch(Rectangle((0.12, 0.2), 0.18, 0.2))
    second = ax.add_patch(Rectangle((0.64, 0.58), 0.16, 0.17))
    fig.canvas.draw()
    manager = attach_drag_manager(fig)
    manager.select_elements([first, second], primary=second)
    manager.selection.set_alignment_reference("key_object", key=second)
    press = _center_event(fig, first, key="shift")
    fig.signals.selected.clear()

    manager.button_press_event0(press)

    assert [target.target for target in manager.selection.targets] == [second]
    assert manager.selected_element is second
    assert manager.selection.alignment_reference_mode == "selection"
    assert manager.selection.alignment_key is None
    assert not manager.selection.got_artist
    assert fig.signals.selected == [second]
    manager.button_release_event0(press)
    assert [target.target for target in manager.selection.targets] == [second]
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


def test_marquee_revalidates_leaf_promoted_to_logical_group() -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
    first = ax.add_patch(Rectangle((0.15, 0.2), 0.25, 0.3, zorder=3))
    second = ax.add_patch(Rectangle((0.55, 0.2), 0.25, 0.3, zorder=3))
    fig.canvas.draw()
    manager = attach_drag_manager(fig)
    manager.select_elements([first, second], primary=second)
    group = manager.group_selection("Pair")
    manager.selection.clear_targets()
    manager.selected_element = None
    original_has_geometry = manager._artist_has_selection_geometry

    def group_has_no_valid_geometry(artist):
        return False if artist is group else original_has_geometry(artist)

    manager._artist_has_selection_geometry = group_has_no_valid_geometry
    bounds = artist_visible_extent(first)
    selected = manager.select_elements_in_bbox(*bounds)

    assert selected == []
    assert manager.selection.targets == []

    manager._artist_has_selection_geometry = original_has_geometry
    manager.editor_scene.set_locked([group], True)
    assert manager.select_elements_in_bbox(*bounds) == []
    manager.selection.clear_targets()
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

    fig.signals.selected.clear()
    manager.undo()
    assert manager.isolation_breadcrumbs == ("Legend",)
    assert manager.selected_element is text
    assert [target.target for target in manager.selection.targets] == [text]
    assert fig.signals.selected == [text]

    fig.signals.selected.clear()
    manager.redo()
    assert manager.isolation_breadcrumbs == ("Legend",)
    assert manager.selected_element is text
    assert [target.target for target in manager.selection.targets] == [text]
    assert fig.signals.selected == [text]

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


def test_single_selection_alignment_is_noop_until_artboard_is_explicit() -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    fig, ax = plt.subplots(figsize=(4, 2), dpi=100)
    fig.canvas.draw()
    manager = attach_drag_manager(fig)
    ax.set_position([0.2, 0.2, 0.3, 0.3])
    manager.selection.add_target(ax)

    before = ax.get_position().frozen()
    assert manager.selection.align_points("left_x") is False
    fig.canvas.draw()

    assert ax.get_position().bounds == before.bounds
    assert not fig.change_tracker.edits
    assert fig.signals.moved_count == 0

    manager.selection.set_alignment_reference("artboard")
    assert manager.selection.align_points("left_x") is True
    fig.canvas.draw()

    assert abs(ax.get_position().x0) < 1e-9
    assert len(fig.change_tracker.edits) == 1
    manager.selection.clear_targets()
    plt.close(fig)
    assert app is not None


def test_key_reference_is_transactional_and_visually_distinct() -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    fig, axes = plt.subplots(1, 3, figsize=(6, 2), dpi=100)
    fig.canvas.draw()
    manager = attach_drag_manager(fig)
    manager.select_elements(axes[:2], primary=axes[1])

    manager.selection.set_alignment_reference("key_object")
    assert manager.selection.alignment_key is axes[1]
    manager.selection.set_alignment_key(axes[0])

    assert manager.selected_element is axes[1]
    assert manager.selection.targets_rects[0].pen().width() == 5
    assert manager.selection.targets_rects[2].pen().width() == 3
    assert not fig.change_tracker.edits

    with pytest.raises(ValueError, match="part of the current selection"):
        manager.selection.set_alignment_reference("key_object", key=axes[2])
    assert manager.selection.alignment_reference_mode == "key_object"
    assert manager.selection.alignment_key is axes[0]

    manager.selection.set_alignment_reference("selection")
    manager.selection.clear_targets()
    with pytest.raises(ValueError, match="at least two"):
        manager.selection.set_alignment_reference("key_object")
    assert manager.selection.alignment_reference_mode == "selection"
    assert manager.selection.alignment_key is None
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


@pytest.mark.parametrize(
    "mode",
    ["left_x", "center_x", "right_x", "bottom_y", "center_y", "top_y"],
)
def test_key_object_alignment_keeps_explicit_key_and_undo_state(mode) -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    fig = plt.figure(figsize=(4, 3), dpi=100)
    key = add_figure_rectangle(fig, (0.12, 0.18, 0.24, 0.19))
    target = add_figure_rectangle(fig, (0.61, 0.58, 0.13, 0.11))
    fig.canvas.draw()
    manager = attach_drag_manager(fig)
    manager.select_elements([key, target], primary=target)
    manager.selection.set_alignment_reference("key_object", key=key)
    key_before = artist_visible_extent(key)
    target_before = artist_visible_extent(target)

    assert manager.selection.align_points(mode) is True
    fig.canvas.draw()
    target_after = artist_visible_extent(target)

    assert artist_visible_extent(key) == pytest.approx(key_before, abs=1e-8)
    assert alignment_coordinate(target_after, mode) == pytest.approx(
        alignment_coordinate(key_before, mode), abs=1e-8
    )
    assert len(fig.change_tracker.edits) == 1
    assert {artist for artist, _command in fig.change_tracker.changes} == {target}
    assert fig.signals.moved_count == 1
    assert manager.selected_element is target
    assert manager.selection.alignment_key is key

    undo, redo = fig.change_tracker.edit[:2]
    undo()
    fig.canvas.draw()
    assert artist_visible_extent(key) == pytest.approx(key_before, abs=1e-8)
    assert artist_visible_extent(target) == pytest.approx(target_before, abs=1e-8)
    assert manager.selected_element is target
    assert manager.selection.alignment_reference_mode == "key_object"
    assert manager.selection.alignment_key is key

    redo()
    fig.canvas.draw()
    assert alignment_coordinate(artist_visible_extent(target), mode) == pytest.approx(
        alignment_coordinate(key_before, mode), abs=1e-8
    )
    assert manager.selected_element is target
    assert manager.selection.alignment_key is key
    manager.selection.clear_targets()
    plt.close(fig)
    assert app is not None


@pytest.mark.parametrize(
    "mode",
    ["left_x", "center_x", "right_x", "bottom_y", "center_y", "top_y"],
)
def test_artboard_alignment_uses_figure_bbox(mode) -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    fig = plt.figure(figsize=(4, 3), dpi=100)
    target = add_figure_rectangle(fig, (0.31, 0.42, 0.17, 0.13))
    fig.canvas.draw()
    manager = attach_drag_manager(fig)
    manager.select_element(target)
    manager.selection.set_alignment_reference("artboard")
    artboard = tuple(float(value) for value in fig.bbox.extents)

    assert manager.selection.align_points(mode) is True
    fig.canvas.draw()

    assert alignment_coordinate(artist_visible_extent(target), mode) == pytest.approx(
        alignment_coordinate(artboard, mode), abs=1e-8
    )
    assert len(fig.change_tracker.edits) == 1
    manager.selection.clear_targets()
    plt.close(fig)
    assert app is not None


def test_clicking_selected_object_changes_only_alignment_key() -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
    first = ax.add_patch(Rectangle((0.12, 0.2), 0.18, 0.2))
    second = ax.add_patch(Rectangle((0.64, 0.58), 0.16, 0.17))
    fig.canvas.draw()
    manager = attach_drag_manager(fig)
    manager.select_elements([first, second], primary=second)
    manager.selection.set_alignment_reference("key_object", key=second)
    selected_before = [target.target for target in manager.selection.targets]
    x, y = ax.transData.transform((0.21, 0.3))
    press = MouseEvent("button_press_event", fig.canvas, x, y, button=1)
    release = MouseEvent("button_release_event", fig.canvas, x, y, button=1)

    manager.button_press_event0(press)
    manager.button_release_event0(release)

    assert manager.selection.alignment_key is first
    assert manager.selected_element is second
    assert [target.target for target in manager.selection.targets] == selected_before
    assert not fig.change_tracker.edits
    manager.selection.clear_targets()
    plt.close(fig)
    assert app is not None


def test_key_reference_falls_back_to_selection_when_only_one_object_remains() -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
    first = ax.add_patch(Rectangle((0.12, 0.2), 0.18, 0.2))
    second = ax.add_patch(Rectangle((0.64, 0.58), 0.16, 0.17))
    fig.canvas.draw()
    manager = attach_drag_manager(fig)
    manager.select_elements([first, second], primary=second)
    manager.selection.set_alignment_reference("key_object", key=second)

    manager.select_element(first)

    assert [target.target for target in manager.selection.targets] == [first]
    assert manager.selection.alignment_reference_mode == "selection"
    assert manager.selection.alignment_key is None
    manager.selection.clear_targets()
    plt.close(fig)
    assert app is not None


def test_single_selection_drag_ignores_stale_key_object_mode() -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
    target = ax.add_patch(Rectangle((0.24, 0.3), 0.2, 0.18))
    fig.canvas.draw()
    manager = attach_drag_manager(fig)
    manager.select_element(target)
    # Simulate an invalid state restored by an older session.  Pointer input
    # must still remain safe even though the public setter rejects this state.
    manager.selection.alignment_reference_mode = "key_object"
    manager.selection.alignment_key = target
    before = artist_visible_extent(target)
    x = (before[0] + before[2]) / 2
    y = (before[1] + before[3]) / 2
    press = MouseEvent("button_press_event", fig.canvas, x, y, button=1)
    move = MouseEvent(
        "motion_notify_event", fig.canvas, x + 10, y - 6, button=1, key="alt"
    )
    release = MouseEvent(
        "button_release_event", fig.canvas, x + 10, y - 6, button=1, key="alt"
    )

    manager.button_press_event0(press)
    manager.selection.on_motion(move)
    manager.button_release_event0(release)
    fig.canvas.draw()

    after = artist_visible_extent(target)
    assert (after[0] - before[0], after[1] - before[1]) == pytest.approx(
        (10, -6), abs=1e-8
    )
    manager.selection.clear_targets()
    plt.close(fig)
    assert app is not None


def test_selection_distribution_keeps_end_objects_and_equalizes_gaps() -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    fig = plt.figure(figsize=(4, 3), dpi=100)
    rectangles = [
        add_figure_rectangle(fig, bounds)
        for bounds in (
            (0.08, 0.2, 0.12, 0.1),
            (0.31, 0.2, 0.08, 0.1),
            (0.52, 0.2, 0.15, 0.1),
            (0.81, 0.2, 0.07, 0.1),
        )
    ]
    fig.canvas.draw()
    manager = attach_drag_manager(fig)
    manager.select_elements(rectangles, primary=rectangles[-1])
    before = [artist_visible_extent(artist) for artist in rectangles]

    assert manager.selection.align_points("distribute_x") is True
    fig.canvas.draw()
    after = [artist_visible_extent(artist) for artist in rectangles]
    gaps = [right[0] - left[2] for left, right in zip(after, after[1:])]

    assert after[0] == pytest.approx(before[0], abs=1e-8)
    assert after[-1] == pytest.approx(before[-1], abs=1e-8)
    assert np.ptp(gaps) < 1e-8
    assert len(fig.change_tracker.edits) == 1
    manager.selection.clear_targets()
    plt.close(fig)
    assert app is not None


def test_two_object_selection_distribution_is_strict_noop() -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    fig = plt.figure(figsize=(4, 3), dpi=100)
    rectangles = [
        add_figure_rectangle(fig, bounds)
        for bounds in ((0.1, 0.2, 0.16, 0.1), (0.72, 0.2, 0.09, 0.1))
    ]
    fig.canvas.draw()
    manager = attach_drag_manager(fig)
    manager.select_elements(rectangles, primary=rectangles[-1])
    before = [artist_visible_extent(artist) for artist in rectangles]

    assert manager.selection.align_points("distribute_x") is False

    assert [artist_visible_extent(artist) for artist in rectangles] == pytest.approx(
        before, abs=1e-8
    )
    assert not fig.change_tracker.edits
    assert fig.signals.moved_count == 0
    manager.selection.clear_targets()
    plt.close(fig)
    assert app is not None


@pytest.mark.parametrize("axis", [0, 1])
def test_artboard_distribution_uses_canvas_edges_and_equal_gaps(axis) -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    fig = plt.figure(figsize=(4, 3), dpi=100)
    rectangles = [
        add_figure_rectangle(fig, bounds)
        for bounds in (
            (0.18, 0.12, 0.1, 0.11),
            (0.46, 0.43, 0.16, 0.08),
            (0.73, 0.76, 0.08, 0.14),
        )
    ]
    fig.canvas.draw()
    manager = attach_drag_manager(fig)
    manager.select_elements(rectangles, primary=rectangles[-1])
    manager.selection.set_alignment_reference("artboard")
    mode = ("distribute_x", "distribute_y")[axis]

    assert manager.selection.align_points(mode) is True
    fig.canvas.draw()
    extents = [artist_visible_extent(artist) for artist in rectangles]
    low, high = ((0, 2), (1, 3))[axis]
    gaps = [
        right[low] - left[high] for left, right in zip(extents, extents[1:])
    ]
    artboard = fig.bbox.extents

    assert extents[0][low] == pytest.approx(artboard[low], abs=1e-8)
    assert extents[-1][high] == pytest.approx(artboard[high], abs=1e-8)
    assert np.ptp(gaps) < 1e-8
    manager.selection.clear_targets()
    plt.close(fig)
    assert app is not None


@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("key_index", [0, 1, 3])
@pytest.mark.parametrize("spacing", [17.0, 0.0, -9.0])
def test_key_distribution_uses_exact_signed_display_gap(
    axis, key_index, spacing
) -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    fig = plt.figure(figsize=(4, 3), dpi=100)
    rectangles = [
        add_figure_rectangle(fig, bounds)
        for bounds in (
            (0.08, 0.10, 0.10, 0.12),
            (0.29, 0.31, 0.15, 0.08),
            (0.57, 0.54, 0.08, 0.16),
            (0.82, 0.79, 0.12, 0.10),
        )
    ]
    fig.canvas.draw()
    manager = attach_drag_manager(fig)
    manager.select_elements(rectangles, primary=rectangles[-1])
    key = rectangles[key_index]
    manager.selection.set_alignment_reference("key_object", key=key)
    key_before = artist_visible_extent(key)
    mode = ("distribute_x", "distribute_y")[axis]

    assert manager.selection.align_points(mode, spacing=spacing) is True
    fig.canvas.draw()
    extents = [artist_visible_extent(artist) for artist in rectangles]
    low, high = ((0, 2), (1, 3))[axis]
    gaps = [
        right[low] - left[high] for left, right in zip(extents, extents[1:])
    ]

    assert gaps == pytest.approx([spacing] * 3, abs=1e-8)
    assert artist_visible_extent(key) == pytest.approx(key_before, abs=1e-8)
    assert manager.selected_element is rectangles[-1]
    assert manager.selection.alignment_key is key
    assert len(fig.change_tracker.edits) == 1
    manager.selection.clear_targets()
    plt.close(fig)
    assert app is not None


def test_key_auto_distribution_reuses_mean_current_gap() -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    fig = plt.figure(figsize=(4, 3), dpi=100)
    rectangles = [
        add_figure_rectangle(fig, bounds)
        for bounds in (
            (0.08, 0.2, 0.10, 0.1),
            (0.27, 0.2, 0.13, 0.1),
            (0.56, 0.2, 0.08, 0.1),
            (0.83, 0.2, 0.11, 0.1),
        )
    ]
    fig.canvas.draw()
    manager = attach_drag_manager(fig)
    manager.select_elements(rectangles, primary=rectangles[-1])
    key = rectangles[1]
    manager.selection.set_alignment_reference("key_object", key=key)
    before = [artist_visible_extent(artist) for artist in rectangles]
    expected_gap = np.mean(
        [right[0] - left[2] for left, right in zip(before, before[1:])]
    )
    key_before = before[1]

    assert manager.selection.align_points("distribute_x") is True
    fig.canvas.draw()
    after = [artist_visible_extent(artist) for artist in rectangles]
    gaps = [right[0] - left[2] for left, right in zip(after, after[1:])]

    assert gaps == pytest.approx([expected_gap] * 3, abs=1e-8)
    assert after[1] == pytest.approx(key_before, abs=1e-8)
    manager.selection.clear_targets()
    plt.close(fig)
    assert app is not None


def test_key_distribution_tie_order_is_stable_and_repeat_is_noop() -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    fig = plt.figure(figsize=(4, 3), dpi=100)
    rectangles = [
        add_figure_rectangle(fig, bounds)
        for bounds in (
            (0.2, 0.2, 0.08, 0.1),
            (0.2, 0.4, 0.12, 0.1),
            (0.48, 0.6, 0.09, 0.1),
            (0.78, 0.75, 0.10, 0.1),
        )
    ]
    fig.canvas.draw()
    manager = attach_drag_manager(fig)
    manager.select_elements(rectangles, primary=rectangles[-1])
    manager.selection.set_alignment_reference("key_object", key=rectangles[-1])

    assert manager.selection.align_points("distribute_x", spacing=5) is True
    fig.canvas.draw()
    first = [artist_visible_extent(artist) for artist in rectangles]
    assert [right[0] - left[2] for left, right in zip(first, first[1:])] == (
        pytest.approx([5.0, 5.0, 5.0], abs=1e-8)
    )
    edit_count = len(fig.change_tracker.edits)

    assert manager.selection.align_points("distribute_x", spacing=5) is False
    assert [artist_visible_extent(artist) for artist in rectangles] == pytest.approx(
        first, abs=1e-8
    )
    assert len(fig.change_tracker.edits) == edit_count
    manager.selection.clear_targets()
    plt.close(fig)
    assert app is not None


def test_numeric_distribution_validation_never_pollutes_reference_state() -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    fig = plt.figure(figsize=(4, 3), dpi=100)
    rectangles = [
        add_figure_rectangle(fig, bounds)
        for bounds in ((0.1, 0.2, 0.1, 0.1), (0.7, 0.2, 0.1, 0.1))
    ]
    fig.canvas.draw()
    manager = attach_drag_manager(fig)
    manager.select_elements(rectangles, primary=rectangles[-1])
    manager.selection.set_alignment_reference("key_object", key=rectangles[0])
    before = [artist_visible_extent(artist) for artist in rectangles]

    for spacing in (np.nan, np.inf, -np.inf):
        with pytest.raises(ValueError, match="finite"):
            manager.selection.align_points("distribute_x", spacing=spacing)

    assert [artist_visible_extent(artist) for artist in rectangles] == pytest.approx(
        before, abs=1e-8
    )
    assert manager.selection.alignment_reference_mode == "key_object"
    assert manager.selection.alignment_key is rectangles[0]
    assert manager.selected_element is rectangles[-1]
    assert not fig.change_tracker.edits
    manager.selection.clear_targets()
    plt.close(fig)
    assert app is not None


def test_alignment_key_survives_membership_add_and_clears_when_removed() -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    fig, axes = plt.subplots(1, 3, figsize=(6, 2), dpi=100)
    fig.canvas.draw()
    manager = attach_drag_manager(fig)
    manager.select_elements(axes[:2], primary=axes[1])
    manager.selection.set_alignment_reference("key_object", key=axes[0])

    manager.select_elements(axes, primary=axes[2])
    assert manager.selection.alignment_key is axes[0]
    assert manager.selected_element is axes[2]

    manager.select_elements(axes[1:], primary=axes[2])
    assert manager.selection.alignment_reference_mode == "key_object"
    assert manager.selection.alignment_key is None
    assert manager.selected_element is axes[2]
    with pytest.raises(ValueError, match="Choose a selected key object"):
        manager.selection.align_points("left_x")
    assert not fig.change_tracker.edits
    manager.selection.clear_targets()
    plt.close(fig)
    assert app is not None


def test_key_object_match_size_resizes_every_non_key_target() -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    fig = plt.figure(figsize=(4, 3), dpi=100)
    rectangles = [
        add_figure_rectangle(fig, bounds)
        for bounds in (
            (0.08, 0.15, 0.10, 0.12),
            (0.31, 0.38, 0.27, 0.14),
            (0.76, 0.68, 0.08, 0.09),
        )
    ]
    fig.canvas.draw()
    manager = attach_drag_manager(fig)
    manager.select_elements(rectangles, primary=rectangles[-1])
    key = rectangles[1]
    manager.selection.set_alignment_reference("key_object", key=key)
    before = [artist_visible_extent(artist) for artist in rectangles]
    key_width = before[1][2] - before[1][0]

    assert manager.selection.match_size("width", keep_aspect_ratio=False) is True
    fig.canvas.draw()
    after = [artist_visible_extent(artist) for artist in rectangles]

    assert [bounds[2] - bounds[0] for bounds in after] == pytest.approx(
        [key_width] * 3, abs=1e-8
    )
    assert after[1] == pytest.approx(before[1], abs=1e-8)
    assert manager.selection.alignment_key is key
    assert manager.selected_element is rectangles[-1]
    assert len(fig.change_tracker.edits) == 1
    assert {
        artist for artist, _command in fig.change_tracker.changes
    } == {rectangles[0], rectangles[2]}

    fig.change_tracker.edit[0]()
    fig.canvas.draw()
    assert np.allclose(
        [artist_visible_extent(artist) for artist in rectangles],
        before,
        atol=1e-8,
    )
    assert manager.selection.alignment_key is key
    assert manager.selected_element is rectangles[-1]
    manager.selection.clear_targets()
    plt.close(fig)
    assert app is not None


def test_key_alignment_clip_failure_is_atomic_and_preserves_ui_state() -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    fig, axes = plt.subplots(1, 2, figsize=(6, 3), dpi=100)
    key = axes[0].add_patch(Rectangle((0.25, 0.35), 0.12, 0.18))
    safe = add_figure_rectangle(fig, (0.72, 0.15, 0.08, 0.1))
    clipped = axes[1].add_patch(Rectangle((0.58, 0.38), 0.14, 0.17))
    fig.canvas.draw()
    manager = attach_drag_manager(fig)
    artists = [key, safe, clipped]
    manager.select_elements(artists, primary=clipped)
    manager.selection.set_alignment_reference("key_object", key=key)
    before = [TargetWrapper(artist).get_restore_state() for artist in artists]

    with pytest.raises(TypeError, match="clip region"):
        manager.selection.align_points("left_x")

    after = [TargetWrapper(artist).get_restore_state() for artist in artists]
    assert all(semantic_equal(old, new) for old, new in zip(before, after))
    assert not fig.change_tracker.edits
    assert fig.change_tracker.change_count == 0
    assert manager.selection.alignment_reference_mode == "key_object"
    assert manager.selection.alignment_key is key
    assert manager.selected_element is clipped
    manager.selection.clear_targets()
    plt.close(fig)
    assert app is not None


def test_interaction_state_roundtrip_includes_alignment_reference_and_key() -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    fig, axes = plt.subplots(1, 3, figsize=(6, 2), dpi=100)
    fig.canvas.draw()
    manager = attach_drag_manager(fig)
    manager.select_elements(axes[:2], primary=axes[1])
    manager.selection.set_alignment_reference("key_object", key=axes[0])
    state = manager.capture_interaction_state()

    manager.select_element(axes[2])
    manager.selection.set_alignment_reference("artboard")
    manager.restore_interaction_state(state)

    assert [target.target for target in manager.selection.targets] == list(axes[:2])
    assert manager.selected_element is axes[1]
    assert manager.selection.alignment_reference_mode == "key_object"
    assert manager.selection.alignment_key is axes[0]
    manager.selection.clear_targets()
    plt.close(fig)
    assert app is not None


def test_align_widget_tracks_reference_and_spacing_controls() -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    signals = WidgetSignals()
    container = QtWidgets.QWidget()
    layout = QtWidgets.QVBoxLayout(container)
    widget = Align(layout, signals)
    fig, axes = plt.subplots(1, 2, figsize=(4, 2), dpi=100)
    polygons = [
        axes[0].add_patch(
            Polygon(
                [[x, 0.3], [x + 0.16, 0.32], [x + 0.08, 0.5]],
                closed=True,
            )
        )
        for x in (0.25, 0.58)
    ]
    fig.canvas.draw()
    manager = attach_drag_manager(fig)
    fig.signals = signals
    signals.figure_changed.emit(fig)
    manager.select_elements(axes, primary=axes[1])
    signals.figure_element_selected.emit(axes[1])
    assert not widget.buttons_by_action["rotate_left"].isEnabled()
    assert not widget.buttons_by_action["rotate_right"].isEnabled()

    key_index = widget.reference_combo.findData("key_object")
    widget.reference_combo.setCurrentIndex(key_index)
    assert manager.selection.alignment_reference_mode == "key_object"
    assert manager.selection.alignment_key is axes[1]
    assert widget.spacing_enabled.isEnabled()

    widget.spacing_enabled.setChecked(True)
    assert widget.spacing_input.isEnabled()
    widget.spacing_input.setValue(-12.5)

    artboard_index = widget.reference_combo.findData("artboard")
    widget.reference_combo.setCurrentIndex(artboard_index)
    assert manager.selection.alignment_reference_mode == "artboard"
    assert not widget.spacing_enabled.isEnabled()
    assert not widget.spacing_input.isEnabled()

    manager.select_elements(polygons, primary=polygons[-1])
    signals.figure_element_selected.emit(polygons[-1])
    assert widget.buttons_by_action["rotate_left"].isEnabled()
    assert widget.buttons_by_action["rotate_right"].isEnabled()

    calls = []

    def capture_alignment(mode, *, spacing=None):
        calls.append((mode, spacing))
        return False

    manager.selection.align_points = capture_alignment
    widget.execute_action("distribute_x")
    assert calls == [("distribute_x", None)]

    manager.selection.clear_targets()
    plt.close(fig)
    container.deleteLater()
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


def test_move_reuses_legend_ownership_inventory_per_interaction_phase(
    monkeypatch,
) -> None:
    import pylustrator.artist_adapters as artist_adapters

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
    lines = [
        ax.plot([0, 1], [index, index + 0.5], label=f"line {index}")[0]
        for index in np.linspace(0, 1, 24)
    ]
    ax.legend(handles=[lines[0]], labels=["representative"])
    fig.canvas.draw()
    manager = attach_drag_manager(fig)
    manager.select_elements(lines)

    calls = 0
    original = artist_adapters.iter_figure_legends

    def counted(figure):
        nonlocal calls
        calls += 1
        return original(figure)

    monkeypatch.setattr(artist_adapters, "iter_figure_legends", counted)
    try:
        manager.selection.start_move()
        manager.selection.addOffset(
            (4, -2), DIR_X0 | DIR_X1 | DIR_Y0 | DIR_Y1
        )
        manager.selection.has_moved = True
        manager.selection.end_move()

        # Ownership may be refreshed once for each public gesture phase, but
        # it must not be rediscovered once per selected artist/capability read.
        assert calls <= 6
    finally:
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


def test_align_appearance_buttons_are_explicit_atomic_and_refresh_properties() -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    signals = WidgetSignals()
    container = QtWidgets.QWidget()
    layout = QtWidgets.QVBoxLayout(container)
    widget = Align(layout, signals)
    fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
    text = ax.text(0.28, 0.72, "appearance", fontsize=10)
    rectangle = ax.add_patch(Rectangle((0.6, 0.25), 0.18, 0.2))
    fig.canvas.draw()
    manager = attach_drag_manager(fig)
    fig.signals = signals
    signals.figure_changed.emit(fig)

    manager.select_element(rectangle)
    signals.figure_element_selected.emit(rectangle)
    assert not widget.buttons_by_action["appearance_up"].isEnabled()
    assert "geometry" in widget.buttons_by_action["scale_up"].toolTip()

    manager.select_elements([text, rectangle], primary=text)
    signals.figure_element_selected.emit(text)
    assert not widget.buttons_by_action["appearance_up"].isEnabled()
    assert "Rectangle" in widget.buttons_by_action["appearance_up"].toolTip()

    manager.select_element(text)
    signals.figure_element_selected.emit(text)
    assert widget.buttons_by_action["appearance_up"].isEnabled()
    assert widget.buttons_by_action["appearance_down"].isEnabled()

    refreshed = []
    signals.figure_element_selected.connect(refreshed.append)
    draw_count = 0
    original_draw = fig.canvas.draw

    def counted_draw(*args, **kwargs):
        nonlocal draw_count
        draw_count += 1
        return original_draw(*args, **kwargs)

    fig.canvas.draw = counted_draw
    position_before = text.get_position()
    widget.execute_action("appearance_up")

    assert text.get_fontsize() == pytest.approx(11.0)
    assert text.get_position() == position_before
    assert draw_count == 1
    assert len(fig.change_tracker.edits) == 1
    assert fig.change_tracker.edit[2] == "Scale appearance"
    assert refreshed[-1] is text

    fig.change_tracker.backEdit = lambda: fig.change_tracker.edit[0]()
    fig.change_tracker.forwardEdit = lambda: fig.change_tracker.edit[1]()
    manager.undo()
    assert text.get_fontsize() == pytest.approx(10.0)
    assert refreshed[-1] is text
    manager.redo()
    assert text.get_fontsize() == pytest.approx(11.0)
    assert refreshed[-1] is text

    manager.selection.clear_targets()
    plt.close(fig)
    container.deleteLater()
    assert app is not None


def test_selection_appearance_factor_one_is_noop_but_near_one_is_an_edit() -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
    text = ax.text(0.28, 0.72, "appearance", fontsize=10)
    fig.canvas.draw()
    manager = attach_drag_manager(fig)
    manager.select_element(text)

    assert manager.selection.scale_appearance_selection(1.0) is False
    assert text.get_fontsize() == pytest.approx(10.0)
    assert not fig.change_tracker.edits
    assert fig.change_tracker.text_change_count == 0

    assert manager.selection.scale_appearance_selection(1.000001) is True
    assert text.get_fontsize() == pytest.approx(10.00001)
    assert len(fig.change_tracker.edits) == 1
    assert fig.change_tracker.text_change_count == 0
    assert fig.change_tracker.change_count == 1
    assert fig.change_tracker.change[1].startswith(".set_fontsize(")

    manager.selection.clear_targets()
    plt.close(fig)
    assert app is not None


@pytest.mark.parametrize(
    "selection_mode", [SelectionMode.OBJECT, SelectionMode.DIRECT]
)
def test_visible_tick_label_is_canvas_reachable_without_starting_parent_drag(
    selection_mode,
) -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
    ax.set_yticks([0, 1], labels=["first method", "second method"])
    fig.canvas.draw()
    manager = attach_drag_manager(fig)
    manager.set_selection_mode(selection_mode)
    label = ax.yaxis.get_major_ticks()[0].label1
    bbox = label.get_window_extent(fig.canvas.get_renderer())
    event = MouseEvent(
        "button_press_event",
        fig.canvas,
        (bbox.x0 + bbox.x1) / 2,
        (bbox.y0 + bbox.y1) / 2,
        button=1,
    )

    assert id(label) in manager._interaction_artist_ids
    assert id(label) in manager._selectable_artist_ids
    assert label.contains(event)[0]
    assert label in manager.get_hit_candidates(event)
    assert manager.get_picked_element(event)[0] is label

    manager.button_press_event0(event)
    assert manager.selected_element is label
    assert [target.target for target in manager.selection.targets] == [label]
    assert not manager.selection.got_artist
    manager.button_release_event0(event)
    assert [target.target for target in manager.selection.targets] == [label]

    manager.selection.clear_targets()
    marquee = manager.select_elements_in_bbox(
        bbox.x0 - 1,
        bbox.y0 - 1,
        bbox.x1 + 1,
        bbox.y1 + 1,
    )
    assert label not in marquee

    manager.selection.clear_targets()
    plt.close(fig)
    assert app is not None


def test_tick_label_translation_is_typed_rejected_without_mutation() -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
    ax.set_yticks([0], labels=["generated label"])
    fig.canvas.draw()
    manager = attach_drag_manager(fig)
    label = ax.yaxis.get_major_ticks()[0].label1
    wrapper = TargetWrapper(label)
    before = wrapper.get_selection_points().copy()
    support = wrapper.operation_support(TransformOperation.TRANSLATE)

    assert not support.supported
    assert "managed by its Axis" in support.reason
    with pytest.raises(UnsupportedArtistError, match="managed by its Axis"):
        wrapper.translate((12, -7))
    fig.canvas.draw()
    np.testing.assert_allclose(wrapper.get_selection_points(), before, atol=0, rtol=0)
    assert manager.figure.change_tracker.changes == []

    manager.selection.clear_targets()
    plt.close(fig)
    assert app is not None


def test_tick_labels_materialized_after_draw_join_hit_inventory() -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
    ax.set_yticks([0], labels=["first"])
    fig.canvas.draw()
    manager = attach_drag_manager(fig)

    ax.set_yticks([0, 1], labels=["first", "later"])
    fig.canvas.draw()
    later = ax.yaxis.get_major_ticks()[1].label1
    assert id(later) not in manager._selectable_artist_ids

    manager.invalidate_geometry_cache()
    assert id(later) in manager._interaction_artist_ids
    assert id(later) in manager._selectable_artist_ids

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


def test_deferred_drag_rejects_empty_clipped_preview_before_mutation() -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
    rectangle = ax.add_patch(Rectangle((0.4, 0.4), 0.1, 0.1))
    fig.canvas.draw()
    manager = attach_drag_manager(fig)
    manager.select_element(rectangle)
    before = TargetWrapper(rectangle).get_restore_state()

    manager.selection.start_move()
    manager.selection.move(
        (400.0, 0.0),
        DIR_X0 | DIR_X1 | DIR_Y0 | DIR_Y1,
        [],
        ignore_snaps=True,
    )
    with pytest.raises(TypeError, match="leave no visible geometry"):
        manager.selection.end_move()

    assert semantic_equal(before, TargetWrapper(rectangle).get_restore_state())
    assert not hasattr(fig.change_tracker, "edit")
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


def test_partially_clipped_drag_preview_includes_newly_revealed_geometry() -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
    rectangle = ax.add_patch(
        Rectangle((0.9, 0.35), 0.2, 0.25, color="red", linewidth=0)
    )
    fig.canvas.draw()
    manager = attach_drag_manager(fig)
    manager.select_element(rectangle)
    before = manager.selection.selection_bounds()

    manager.selection.start_move()
    manager.selection.move(
        (-40.0, 0.0),
        DIR_X0 | DIR_X1 | DIR_Y0 | DIR_Y1,
        [],
        ignore_snaps=True,
    )
    preview = manager.selection.move_current_selection_points[id(rectangle)].copy()
    manager.selection.end_move()
    fig.canvas.draw()
    committed = TargetWrapper(rectangle).get_selection_points()

    assert np.allclose(committed, preview, atol=0.25)
    assert np.ptp(committed[:, 0]) > np.ptp(before[[0, 2]])
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


def test_multi_rotation_rejects_local_angle_only_selection_atomically() -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
    text = ax.text(0.2, 0.7, "rotate")
    rectangle = ax.add_patch(Rectangle((0.5, 0.3), 0.2, 0.25))
    fig.canvas.draw()
    manager = attach_drag_manager(fig)
    fig.signals.figure_selection_property_changed = Signal()
    manager.select_elements([text, rectangle], primary=rectangle)

    with pytest.raises(UnsupportedArtistError, match="common-pivot"):
        manager.selection.rotate_selection(17)
    assert text.get_rotation() == 0
    assert rectangle.get_angle() == 0
    assert not fig.change_tracker.edits
    manager.selection.clear_targets()
    plt.close(fig)
    assert app is not None


def test_toolbar_multi_rotation_uses_shared_reference_pivot_and_one_undo() -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
    first = ax.add_patch(
        Polygon([[0.25, 0.3], [0.35, 0.32], [0.3, 0.42]], closed=True)
    )
    second = ax.add_patch(
        Polygon([[0.58, 0.55], [0.7, 0.57], [0.65, 0.68]], closed=True)
    )
    fig.canvas.draw()
    manager = attach_drag_manager(fig)
    fig.signals.figure_selection_property_changed = Signal()
    manager.select_elements([first, second], primary=second)
    manager.selection.set_alignment_reference("key_object", key=first)
    manager.selection.set_reference_point((0.0, 0.0))
    pivot = manager.selection.reference_position().copy()
    before = [TargetWrapper(artist).get_positions().copy() for artist in (first, second)]
    matrix = TargetWrapper(first).adapter.display_rotation_matrix(17, pivot)

    assert manager.selection.rotate_selection(360) is False
    assert not fig.change_tracker.edits
    assert manager.selection.rotate_selection(17)
    fig.canvas.draw()
    after = [TargetWrapper(artist).get_positions() for artist in (first, second)]

    for original, rotated in zip(before, after):
        expected = TargetWrapper(first).adapter._transform_points(matrix, original)
        assert np.allclose(rotated, expected, atol=1e-8)
    assert len(fig.change_tracker.edits) == 1
    assert manager.selection.alignment_key is first
    assert manager.selected_element is second

    undo, redo = fig.change_tracker.edit[:2]
    undo()
    fig.canvas.draw()
    assert all(
        np.allclose(TargetWrapper(artist).get_positions(), original, atol=1e-8)
        for artist, original in zip((first, second), before)
    )
    assert manager.selection.alignment_key is first
    assert manager.selected_element is second
    redo()
    fig.canvas.draw()
    assert all(
        np.allclose(TargetWrapper(artist).get_positions(), rotated, atol=1e-8)
        for artist, rotated in zip((first, second), after)
    )
    manager.selection.clear_targets()
    plt.close(fig)
    assert app is not None


def test_custom_rotation_pivot_is_physical_non_document_state(monkeypatch) -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
    artists = [
        ax.add_patch(
            Polygon([[0.2, 0.25], [0.35, 0.27], [0.28, 0.42]], closed=True)
        ),
        ax.add_patch(
            Polygon([[0.58, 0.55], [0.74, 0.57], [0.66, 0.72]], closed=True)
        ),
    ]
    for artist in artists:
        artist.set_clip_on(False)
    fig.canvas.draw()
    manager = attach_drag_manager(fig)
    manager.select_elements(artists, primary=artists[-1])
    selection = manager.selection
    before = [TargetWrapper(artist).get_positions().copy() for artist in artists]
    draw_calls = []
    original_draw = fig.canvas.draw
    monkeypatch.setattr(fig.canvas, "draw", lambda *args, **kwargs: draw_calls.append(1))

    assert selection.set_rotation_pivot((85.0, 70.0)) == (85.0, 70.0)

    assert np.allclose(selection.custom_rotation_pivot_state(), (0.85, 0.7))
    assert np.allclose(selection.rotation_pivot(), (85.0, 70.0))
    assert not draw_calls
    assert not fig.change_tracker.edits
    assert not fig.change_tracker.changes
    assert all(
        np.allclose(TargetWrapper(artist).get_positions(), state)
        for artist, state in zip(artists, before)
    )

    monkeypatch.setattr(fig.canvas, "draw", original_draw)
    fig.set_dpi(200)
    assert np.allclose(selection.custom_rotation_pivot_position(), (170.0, 140.0))
    fig.set_size_inches(7, 5, forward=False)
    assert np.allclose(selection.custom_rotation_pivot_position(), (170.0, 140.0))
    selection.clear_targets()
    plt.close(fig)
    assert app is not None


def test_toolbar_rotation_uses_custom_pivot_and_undo_restores_pivot_state() -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
    first = ax.add_patch(
        Polygon([[0.22, 0.24], [0.36, 0.27], [0.29, 0.42]], closed=True)
    )
    second = ax.add_patch(
        Polygon([[0.57, 0.53], [0.72, 0.57], [0.64, 0.71]], closed=True)
    )
    for artist in (first, second):
        artist.set_clip_on(False)
    fig.canvas.draw()
    manager = attach_drag_manager(fig)
    manager.select_elements([first, second], primary=second)
    selection = manager.selection
    selection.set_alignment_reference("key_object", key=first)
    selection.set_reference_point((0.0, 0.0))
    custom = np.array([175.0, 128.0])
    selection.set_rotation_pivot(custom)
    custom_state = selection.custom_rotation_pivot_state()
    before = [TargetWrapper(artist).get_positions().copy() for artist in (first, second)]
    matrix = TargetWrapper(first).adapter.display_rotation_matrix(17.0, custom)

    assert selection.rotate_selection(17.0)
    fig.canvas.draw()
    after = [TargetWrapper(artist).get_positions().copy() for artist in (first, second)]

    for original, current in zip(before, after):
        expected = TargetWrapper(first).adapter._transform_points(matrix, original)
        assert np.allclose(current, expected, atol=1e-8)
    assert len(fig.change_tracker.edits) == 1
    assert selection.custom_rotation_pivot_state() == custom_state
    assert manager.selected_element is second
    assert selection.alignment_key is first

    undo, redo = fig.change_tracker.edit[:2]
    selection.set_reference_point((1.0, 1.0))
    assert selection.custom_rotation_pivot_state() is None
    undo()
    fig.canvas.draw()
    assert selection.reference_point == (0.0, 0.0)
    assert selection.custom_rotation_pivot_state() == custom_state
    assert np.allclose(selection.rotation_pivot(), custom)
    assert manager.selected_element is second
    assert selection.alignment_key is first
    redo()
    fig.canvas.draw()
    assert selection.reference_point == (0.0, 0.0)
    assert selection.custom_rotation_pivot_state() == custom_state
    assert all(
        np.allclose(TargetWrapper(artist).get_positions(), state, atol=1e-8)
        for artist, state in zip((first, second), after)
    )
    selection.clear_targets()
    plt.close(fig)
    assert app is not None


def test_rotation_handle_uses_custom_pivot_and_keeps_a_nonzero_lever_arm() -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
    artists = [
        ax.add_patch(
            Polygon([[0.22, 0.27], [0.36, 0.29], [0.29, 0.44]], closed=True)
        ),
        ax.add_patch(
            Polygon([[0.57, 0.52], [0.73, 0.56], [0.64, 0.7]], closed=True)
        ),
    ]
    for artist in artists:
        artist.set_clip_on(False)
    fig.canvas.draw()
    manager = attach_drag_manager(fig)
    manager.select_elements(artists, primary=artists[-1])
    selection = manager.selection
    original_handle = np.asarray(selection.rotation_grabber.get_xy(), dtype=float)
    selection.set_rotation_pivot(original_handle)
    pivot = selection.rotation_pivot().copy()
    handle = np.asarray(selection.rotation_grabber.get_xy(), dtype=float)

    assert np.linalg.norm(handle - pivot) > 12.0
    before = [TargetWrapper(artist).get_positions().copy() for artist in artists]
    vector = handle - pivot
    requested = np.deg2rad(37.0)
    pointer = pivot + np.array(
        [
            vector[0] * np.cos(requested) - vector[1] * np.sin(requested),
            vector[0] * np.sin(requested) + vector[1] * np.cos(requested),
        ]
    )
    selection.start_rotation(SimpleNamespace(x=handle[0], y=handle[1], key=None))
    assert selection.preview_rotation(
        SimpleNamespace(x=pointer[0], y=pointer[1], key="shift")
    ) == pytest.approx(30.0)
    matrix = TargetWrapper(artists[0]).adapter.display_rotation_matrix(30.0, pivot)
    for artist, original in zip(artists, before):
        expected = TargetWrapper(artist).adapter._transform_points(matrix, original)
        assert np.allclose(TargetWrapper(artist).get_positions(), expected, atol=1e-8)
    assert selection.end_rotation()
    selection.clear_targets()
    plt.close(fig)
    assert app is not None


def test_escape_cancels_pivot_drag_without_mutating_document_or_selection() -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
    artists = [
        ax.add_patch(Polygon([[0.2, 0.25], [0.35, 0.28], [0.28, 0.43]])),
        ax.add_patch(Polygon([[0.58, 0.53], [0.73, 0.57], [0.65, 0.71]])),
    ]
    for artist in artists:
        artist.set_clip_on(False)
    fig.canvas.draw()
    manager = attach_drag_manager(fig)
    manager.select_elements(artists, primary=artists[-1])
    selection = manager.selection
    original_pivot = np.array([132.0, 104.0])
    selection.set_rotation_pivot(original_pivot)
    original_state = selection.custom_rotation_pivot_state()
    artist_states = [TargetWrapper(artist).get_positions().copy() for artist in artists]
    grabber = selection.rotation_grabber

    grabber.pivot_button_press_event(
        SimpleNamespace(x=original_pivot[0], y=original_pivot[1])
    )
    grabber.on_pivot_motion(SimpleNamespace(x=78.0, y=62.0))
    assert np.allclose(selection.rotation_pivot(), (78.0, 62.0))
    manager.key_press_event(SimpleNamespace(key="escape"))

    assert selection.custom_rotation_pivot_state() == original_state
    assert np.allclose(selection.rotation_pivot(), original_pivot)
    assert not grabber.pivot_got_artist
    assert [target.target for target in selection.targets] == artists
    assert manager.selected_element is artists[-1]
    assert not fig.change_tracker.edits
    assert not fig.change_tracker.changes
    assert all(
        np.allclose(TargetWrapper(artist).get_positions(), state)
        for artist, state in zip(artists, artist_states)
    )
    selection.clear_targets()
    plt.close(fig)
    assert app is not None


def test_same_reference_locator_click_resets_custom_rotation_pivot() -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
    artists = [
        ax.add_patch(Polygon([[0.2, 0.2], [0.36, 0.24], [0.29, 0.4]])),
        ax.add_patch(Polygon([[0.58, 0.52], [0.74, 0.56], [0.65, 0.7]])),
    ]
    fig.canvas.draw()
    manager = attach_drag_manager(fig)
    manager.select_elements(artists, primary=artists[-1])
    selection = manager.selection
    selection.set_rotation_pivot((120.0, 90.0))
    widget = ReferencePointWidget()
    widget.referenceChanged.connect(selection.set_reference_point)

    widget.setValue(widget.value(), emit=True)

    assert selection.reference_point == (0.5, 0.5)
    assert selection.custom_rotation_pivot_state() is None
    assert np.allclose(selection.rotation_pivot(), selection.reference_position())
    widget.deleteLater()
    selection.clear_targets()
    plt.close(fig)
    assert app is not None


def test_custom_pivot_survives_primary_change_and_interaction_state_restore() -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
    artists = [
        ax.add_patch(Polygon([[0.15, 0.2], [0.28, 0.23], [0.22, 0.37]])),
        ax.add_patch(Polygon([[0.43, 0.45], [0.57, 0.48], [0.5, 0.63]])),
        ax.add_patch(Polygon([[0.7, 0.22], [0.84, 0.25], [0.77, 0.4]])),
    ]
    fig.canvas.draw()
    manager = attach_drag_manager(fig)
    manager.select_elements(artists[:2], primary=artists[1])
    selection = manager.selection
    selection.set_reference_point((0.0, 1.0))
    selection.set_rotation_pivot((145.0, 112.0))
    pivot_state = selection.custom_rotation_pivot_state()

    manager.select_elements(artists[:2], primary=artists[0])
    assert selection.custom_rotation_pivot_state() == pivot_state
    assert manager.selected_element is artists[0]
    interaction = manager.capture_interaction_state()

    manager.select_element(artists[2])
    assert selection.custom_rotation_pivot_state() is None
    manager.restore_interaction_state(interaction)

    assert [target.target for target in selection.targets] == [artists[1], artists[0]]
    assert manager.selected_element is artists[0]
    assert selection.reference_point == (0.0, 1.0)
    assert selection.custom_rotation_pivot_state() == pivot_state
    assert np.allclose(selection.rotation_pivot(), (145.0, 112.0))
    selection.clear_targets()
    plt.close(fig)
    assert app is not None


def test_group_undo_restores_the_pre_group_custom_pivot() -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
    first = ax.add_patch(Polygon([[0.2, 0.22], [0.35, 0.25], [0.28, 0.4]]))
    second = ax.add_patch(Polygon([[0.58, 0.52], [0.74, 0.56], [0.65, 0.71]]))
    fig.canvas.draw()
    manager = attach_drag_manager(fig)
    manager.select_elements([first, second], primary=second)
    selection = manager.selection
    selection.set_reference_point((1.0, 0.0))
    selection.set_rotation_pivot((142.0, 108.0))
    pivot_state = selection.custom_rotation_pivot_state()

    group = manager.group_selection("Pivot group")

    assert [target.target for target in selection.targets] == [group]
    assert selection.custom_rotation_pivot_state() is None
    undo, redo = fig.change_tracker.edit[:2]
    undo()
    assert [target.target for target in selection.targets] == [first, second]
    assert manager.selected_element is second
    assert selection.reference_point == (1.0, 0.0)
    assert selection.custom_rotation_pivot_state() == pivot_state
    assert np.allclose(selection.rotation_pivot(), (142.0, 108.0))
    redo()
    assert len(selection.targets) == 1
    assert selection.targets[0].target.group_id == group.group_id
    assert selection.custom_rotation_pivot_state() is None
    selection.clear_targets()
    plt.close(fig)
    assert app is not None


def test_custom_pivot_survives_failed_rotation_preflight_without_mutation() -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
    artists = [
        ax.add_patch(Polygon([[0.08, 0.08], [0.2, 0.1], [0.13, 0.23]])),
        ax.add_patch(Polygon([[0.72, 0.72], [0.86, 0.75], [0.79, 0.89]])),
    ]
    fig.canvas.draw()
    manager = attach_drag_manager(fig)
    manager.select_elements(artists, primary=artists[-1])
    selection = manager.selection
    selection.set_rotation_pivot((55.0, 45.0))
    pivot_state = selection.custom_rotation_pivot_state()
    before = [TargetWrapper(artist).get_positions().copy() for artist in artists]

    with pytest.raises(UnsupportedArtistError, match="clip"):
        selection.rotate_selection(90.0)

    assert selection.custom_rotation_pivot_state() == pivot_state
    assert np.allclose(selection.rotation_pivot(), (55.0, 45.0))
    assert all(
        np.allclose(TargetWrapper(artist).get_positions(), state)
        for artist, state in zip(artists, before)
    )
    assert not fig.change_tracker.edits
    assert not fig.change_tracker.changes
    selection.clear_targets()
    plt.close(fig)
    assert app is not None


def test_pivot_drag_release_commits_only_editor_state() -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
    artists = [
        ax.add_patch(Polygon([[0.2, 0.23], [0.35, 0.26], [0.28, 0.41]])),
        ax.add_patch(Polygon([[0.58, 0.53], [0.73, 0.56], [0.65, 0.7]])),
    ]
    fig.canvas.draw()
    manager = attach_drag_manager(fig)
    manager.select_elements(artists, primary=artists[-1])
    selection = manager.selection
    before = [TargetWrapper(artist).get_positions().copy() for artist in artists]
    start = selection.rotation_pivot().copy()
    destination = np.array([96.0, 76.0])
    grabber = selection.rotation_grabber
    update_emissions = []
    fig.signals.figure_selection_update.connect(
        lambda: update_emissions.append(True)
    )

    grabber.pivot_button_press_event(SimpleNamespace(x=start[0], y=start[1]))
    grabber.on_pivot_motion(
        SimpleNamespace(x=destination[0], y=destination[1])
    )
    assert not update_emissions
    grabber.pivot_button_release_event(
        SimpleNamespace(x=destination[0], y=destination[1])
    )

    assert np.allclose(selection.rotation_pivot(), destination)
    assert len(update_emissions) == 1
    assert not grabber.pivot_got_artist
    assert not fig.change_tracker.edits
    assert not fig.change_tracker.changes
    assert all(
        np.allclose(TargetWrapper(artist).get_positions(), state)
        for artist, state in zip(artists, before)
    )
    selection.clear_targets()
    plt.close(fig)
    assert app is not None


def test_rotation_history_insertion_failure_rolls_back_the_entire_gesture() -> None:
    from pylustrator.change_tracker import ChangeTracker as RealChangeTracker

    class RaisingSignal:
        def emit(self, *_args):
            raise RuntimeError("simulated history insertion failure")

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
    artists = [
        ax.add_patch(Polygon([[0.2, 0.23], [0.35, 0.26], [0.28, 0.41]])),
        ax.add_patch(Polygon([[0.58, 0.53], [0.73, 0.56], [0.65, 0.7]])),
    ]
    for artist in artists:
        artist.set_clip_on(False)
    fig.canvas.draw()
    manager = attach_drag_manager(fig)
    manager.select_elements(artists, primary=artists[-1])
    selection = manager.selection
    selection.set_reference_point((0.0, 1.0))
    selection.set_rotation_pivot((132.0, 102.0))
    pivot_state = selection.custom_rotation_pivot_state()
    before = [TargetWrapper(artist).get_positions().copy() for artist in artists]

    tracker = RealChangeTracker.__new__(RealChangeTracker)
    tracker.figure = fig
    tracker.changes = {}
    tracker.saved = True
    tracker.edits = []
    tracker.last_edit = -1
    tracker.update_changes_signal = None
    tracker.no_save = False
    fig.change_tracker = tracker
    recording_before = tracker.capture_recording_state()
    original_add_edit = tracker.addEdit

    def fail_after_history_insert(edit):
        tracker.update_changes_signal = RaisingSignal()
        try:
            original_add_edit(edit)
        finally:
            tracker.update_changes_signal = None

    tracker.addEdit = fail_after_history_insert

    with pytest.raises(RuntimeError, match="history insertion failure"):
        selection.rotate_selection(17.0)

    assert all(
        np.allclose(TargetWrapper(artist).get_positions(), state)
        for artist, state in zip(artists, before)
    )
    assert tracker.capture_recording_state() == recording_before
    assert tracker.edits == []
    assert tracker.last_edit == -1
    assert selection.reference_point == (0.0, 1.0)
    assert selection.custom_rotation_pivot_state() == pivot_state
    assert np.allclose(selection.rotation_pivot(), (132.0, 102.0))
    assert [target.target for target in selection.targets] == artists
    assert manager.selected_element is artists[-1]
    selection.clear_targets()
    plt.close(fig)
    assert app is not None


def test_native_only_rotation_never_accepts_a_custom_pivot() -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
    text = ax.text(0.3, 0.5, "native angle only")
    fig.canvas.draw()
    manager = attach_drag_manager(fig)
    manager.select_element(text)
    selection = manager.selection

    assert selection.rotation_operation() is TransformOperation.ROTATE
    assert not selection.custom_rotation_pivot_supported()
    with pytest.raises(UnsupportedArtistError, match="shared rigid-rotation"):
        selection.set_rotation_pivot((120.0, 90.0))
    assert selection.custom_rotation_pivot_state() is None
    assert not selection.rotation_grabber.pivot_marker.isVisible()
    assert not fig.change_tracker.edits
    assert not fig.change_tracker.changes
    selection.clear_targets()
    plt.close(fig)
    assert app is not None


def test_toolbar_mixed_geometry_rotation_shares_one_pivot_and_transaction() -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    fig, ax = plt.subplots(figsize=(5, 5), dpi=100)
    text = ax.text(0.47, 0.52, "anchor", rotation_mode="anchor")
    line = ax.plot(
        [0.25, 0.38],
        [0.7, 0.78],
        marker="o",
        markevery=[0],
    )[0]
    polygon = ax.add_patch(
        Polygon([[0.25, 0.25], [0.4, 0.27], [0.34, 0.4]], closed=True)
    )
    path_patch = ax.add_patch(
        PathPatch(
            Path(
                [[0.58, 0.24], [0.74, 0.28], [0.65, 0.42], [0.58, 0.24]],
                [Path.MOVETO, Path.LINETO, Path.LINETO, Path.CLOSEPOLY],
            )
        )
    )
    line_collection = LineCollection(
        [[[0.6, 0.66], [0.74, 0.78]], [[0.54, 0.72], [0.68, 0.84]]]
    )
    poly_collection = PolyCollection(
        [[[0.4, 0.62], [0.5, 0.65], [0.46, 0.76]]]
    )
    ax.add_collection(line_collection)
    ax.add_collection(poly_collection)
    artists = [text, line, polygon, path_patch, line_collection, poly_collection]
    for artist in artists:
        artist.set_clip_on(False)
    fig.canvas.draw()
    manager = attach_drag_manager(fig)
    fig.signals.figure_selection_property_changed = Signal()
    manager.select_elements(artists, primary=path_patch)
    selection = manager.selection
    pivot = selection.reference_position().copy()
    before = [TargetWrapper(artist).get_positions().copy() for artist in artists]
    matrix = TargetWrapper(polygon).adapter.display_rotation_matrix(-11.0, pivot)

    assert all(
        TargetWrapper(artist).supports_operation(TransformOperation.RIGID_ROTATE)
        for artist in artists
    )
    assert selection.rotation_operation() is TransformOperation.RIGID_ROTATE
    assert selection.rotate_selection(-11.0)
    fig.canvas.draw()
    after = [TargetWrapper(artist).get_positions().copy() for artist in artists]

    for original, current in zip(before, after):
        expected = TargetWrapper(polygon).adapter._transform_points(matrix, original)
        assert np.allclose(current, expected, atol=1e-8, equal_nan=True)
    assert len(fig.change_tracker.edits) == 1
    assert {target for target, _command in fig.change_tracker.changes} == set(
        artists[1:]
    )
    assert fig.change_tracker.text_change_count == 1

    undo, redo = fig.change_tracker.edit[:2]
    undo()
    fig.canvas.draw()
    assert all(
        np.allclose(
            TargetWrapper(artist).get_positions(),
            original,
            atol=1e-8,
            equal_nan=True,
        )
        for artist, original in zip(artists, before)
    )
    redo()
    fig.canvas.draw()
    assert all(
        np.allclose(
            TargetWrapper(artist).get_positions(),
            rotated,
            atol=1e-8,
            equal_nan=True,
        )
        for artist, rotated in zip(artists, after)
    )
    selection.clear_targets()
    plt.close(fig)
    assert app is not None


def test_rotation_handle_commits_arbitrary_native_angle_and_single_undo() -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    fig = plt.figure(figsize=(4, 3), dpi=100)
    target = Rectangle(
        (110.0, 105.0),
        52.0,
        34.0,
        rotation_point=(125.0, 118.0),
        transform=IdentityTransform(),
        clip_on=False,
    )
    second = Rectangle(
        (220.0, 85.0),
        38.0,
        45.0,
        transform=IdentityTransform(),
        clip_on=False,
    )
    fig.add_artist(target)
    fig.add_artist(second)
    fig.canvas.draw()
    manager = attach_drag_manager(fig)
    fig.signals.figure_selection_property_changed = Signal()
    manager.select_element(target)
    selection = manager.selection

    assert selection.rotation_operation() is TransformOperation.ROTATE
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
    assert target.get_angle() == pytest.approx(37.0)
    assert selection.end_rotation()
    assert target.get_angle() == pytest.approx(37.0)

    fig.change_tracker.edit[0]()
    assert target.get_angle() == pytest.approx(0.0)

    manager.select_elements([target, second], primary=second)
    assert not selection.rotation_handle_supported()
    assert not selection.rotation_grabber.handle.isVisible()
    manager.selection.clear_targets()
    plt.close(fig)
    assert app is not None


def test_multi_rotation_handle_previews_shared_pivot_and_shift_snap() -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
    artists = [
        ax.add_patch(
            Polygon([[0.24, 0.3], [0.36, 0.31], [0.3, 0.43]], closed=True)
        ),
        ax.add_patch(
            Polygon([[0.58, 0.55], [0.71, 0.58], [0.64, 0.69]], closed=True)
        ),
    ]
    fig.canvas.draw()
    manager = attach_drag_manager(fig)
    fig.signals.figure_selection_property_changed = Signal()
    manager.select_elements(artists, primary=artists[-1])
    selection = manager.selection
    pivot = selection.reference_position().copy()
    before = [TargetWrapper(artist).get_positions().copy() for artist in artists]

    assert selection.rotation_handle_supported()
    assert selection.rotation_grabber.handle.isVisible()
    assert np.allclose(selection.rotation_pivot(), pivot)
    start = np.asarray(selection.rotation_grabber.get_xy(), dtype=float)
    vector = start - pivot
    requested = np.deg2rad(37.0)
    pointer = pivot + np.array(
        [
            vector[0] * np.cos(requested) - vector[1] * np.sin(requested),
            vector[0] * np.sin(requested) + vector[1] * np.cos(requested),
        ]
    )

    selection.start_rotation(SimpleNamespace(x=start[0], y=start[1], key=None))
    preview = selection.preview_rotation(
        SimpleNamespace(x=pointer[0], y=pointer[1], key="shift")
    )
    preview_points = [TargetWrapper(artist).get_positions().copy() for artist in artists]

    assert preview == pytest.approx(30.0)
    assert not fig.change_tracker.changes
    matrix = TargetWrapper(artists[0]).adapter.display_rotation_matrix(30, pivot)
    for original, current in zip(before, preview_points):
        expected = TargetWrapper(artists[0]).adapter._transform_points(
            matrix, original
        )
        assert np.allclose(current, expected, atol=1e-8)

    assert selection.end_rotation()
    fig.canvas.draw()
    assert all(
        np.allclose(TargetWrapper(artist).get_positions(), previewed, atol=1e-8)
        for artist, previewed in zip(artists, preview_points)
    )
    assert len(fig.change_tracker.edits) == 1
    assert {target for target, _command in fig.change_tracker.changes} == set(artists)

    fig.change_tracker.edit[0]()
    fig.canvas.draw()
    assert all(
        np.allclose(TargetWrapper(artist).get_positions(), original, atol=1e-8)
        for artist, original in zip(artists, before)
    )
    manager.select_element(artists[0])
    assert selection.rotation_operation() is TransformOperation.RIGID_ROTATE
    assert selection.rotation_handle_supported()
    assert selection.rotation_grabber.handle.isVisible()
    selection.clear_targets()
    plt.close(fig)
    assert app is not None


@pytest.mark.parametrize("commit_path", ["numeric", "handle"])
def test_multi_rigid_rotation_prepares_every_target_before_first_apply(
    monkeypatch, commit_path
) -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
    artists = [
        ax.add_patch(
            Polygon([[0.22, 0.27], [0.36, 0.29], [0.29, 0.44]], closed=True)
        ),
        ax.add_patch(
            Polygon([[0.57, 0.52], [0.73, 0.56], [0.64, 0.7]], closed=True)
        ),
    ]
    for artist in artists:
        artist.set_clip_on(False)
    fig.canvas.draw()
    manager = attach_drag_manager(fig)
    fig.signals.figure_selection_property_changed = Signal()
    manager.select_elements(artists, primary=artists[-1])
    selection = manager.selection
    before = [TargetWrapper(artist).get_positions().copy() for artist in artists]

    if commit_path == "handle":
        pivot = selection.rotation_pivot().copy()
        start = np.asarray(selection.rotation_grabber.get_xy(), dtype=float)
        vector = start - pivot
        angle = np.deg2rad(19.0)
        pointer = pivot + np.array(
            [
                vector[0] * np.cos(angle) - vector[1] * np.sin(angle),
                vector[0] * np.sin(angle) + vector[1] * np.cos(angle),
            ]
        )
        selection.start_rotation(
            SimpleNamespace(x=start[0], y=start[1], key=None)
        )
        assert selection.preview_rotation(
            SimpleNamespace(x=pointer[0], y=pointer[1], key=None)
        ) == pytest.approx(19.0)

    first_adapter = selection.targets[0].adapter
    second_adapter = selection.targets[1].adapter
    original_apply = first_adapter._apply_prevalidated_rigid_rotation_plan
    original_setter = first_adapter._apply_native_control_points
    apply_calls = []
    setter_calls_during_apply = []
    applying = False

    def capture_apply(plan, *, record_changes=True):
        nonlocal applying
        apply_calls.append(plan)
        applying = True
        try:
            return original_apply(
                plan,
                record_changes=record_changes,
            )
        finally:
            applying = False

    def capture_setter(points):
        if applying:
            setter_calls_during_apply.append(np.asarray(points, dtype=float).copy())
        return original_setter(points)

    def fail_second_prepare(_plan):
        raise UnsupportedArtistError("QA second rigid prepare failure")

    monkeypatch.setattr(
        first_adapter,
        "_apply_prevalidated_rigid_rotation_plan",
        capture_apply,
    )
    monkeypatch.setattr(first_adapter, "_apply_native_control_points", capture_setter)
    monkeypatch.setattr(
        second_adapter,
        "revalidate_rigid_rotation_plan",
        fail_second_prepare,
    )

    try:
        with pytest.raises(
            UnsupportedArtistError, match="second rigid prepare failure"
        ):
            if commit_path == "handle":
                selection.end_rotation()
            else:
                selection.rotate_selection(19.0)

        assert apply_calls == []
        assert setter_calls_during_apply == []
        assert all(
            np.allclose(TargetWrapper(artist).get_positions(), state, atol=1e-8)
            for artist, state in zip(artists, before)
        )
        assert not fig.change_tracker.edits
        assert not fig.change_tracker.changes
    finally:
        selection.clear_targets()
        plt.close(fig)
    assert app is not None


def test_rotation_handle_reuses_cached_gesture_source_geometry(monkeypatch) -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
    target = ax.plot(
        np.linspace(0.2, 0.8, 100),
        np.linspace(0.3, 0.7, 100),
        linestyle="none",
        marker="o",
        markevery=5,
        clip_on=False,
    )[0]
    fig.canvas.draw()
    manager = attach_drag_manager(fig)
    fig.signals.figure_selection_property_changed = Signal()
    manager.select_element(target)
    selection = manager.selection
    pivot = selection.rotation_pivot().copy()
    start = np.asarray(selection.rotation_grabber.get_xy(), dtype=float)
    vector = start - pivot
    requested = np.deg2rad(23.0)
    pointer = pivot + np.array(
        [
            vector[0] * np.cos(requested) - vector[1] * np.sin(requested),
            vector[0] * np.sin(requested) + vector[1] * np.cos(requested),
        ]
    )

    selection.start_rotation(SimpleNamespace(x=start[0], y=start[1], key=None))
    wrapper = selection.targets[0]
    original = wrapper.adapter.plan_rigid_rotation
    calls = []

    def capture_plan(angle, plan_pivot, **kwargs):
        calls.append(kwargs)
        return original(angle, plan_pivot, **kwargs)

    monkeypatch.setattr(wrapper.adapter, "plan_rigid_rotation", capture_plan)

    try:
        selection.preview_rotation(
            SimpleNamespace(x=pointer[0], y=pointer[1], key=None)
        )
        assert len(calls) == 1
        assert calls[0]["control_points"] is selection.move_start_positions[
            id(target)
        ]
        assert calls[0][
            "selection_points"
        ] is selection.move_start_raw_selection_points[id(target)]
        selection.cancel_rotation()
    finally:
        selection.clear_targets()
        plt.close(fig)
    assert app is not None


def test_large_line_rigid_preview_skips_commit_revalidation(monkeypatch) -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
    count = 100_000
    target = ax.plot(
        np.linspace(0.2, 0.8, count),
        np.linspace(0.3, 0.7, count),
        linewidth=1.0,
        clip_on=False,
    )[0]
    fig.canvas.draw()
    manager = attach_drag_manager(fig)
    fig.signals.figure_selection_property_changed = Signal()
    manager.select_element(target)
    selection = manager.selection
    pivot = selection.rotation_pivot().copy()
    start = np.asarray(selection.rotation_grabber.get_xy(), dtype=float)
    vector = start - pivot
    angle = np.deg2rad(17.0)
    pointer = pivot + np.array(
        [
            vector[0] * np.cos(angle) - vector[1] * np.sin(angle),
            vector[0] * np.sin(angle) + vector[1] * np.cos(angle),
        ]
    )

    selection.start_rotation(SimpleNamespace(x=start[0], y=start[1], key=None))
    adapter = selection.targets[0].adapter
    revalidation_calls = []

    def unexpected_revalidation(plan):
        revalidation_calls.append(plan)
        raise AssertionError("pointer preview entered commit revalidation")

    monkeypatch.setattr(
        adapter,
        "revalidate_rigid_rotation_plan",
        unexpected_revalidation,
    )

    try:
        assert selection.preview_rotation(
            SimpleNamespace(x=pointer[0], y=pointer[1], key=None)
        ) == pytest.approx(17.0)
        assert revalidation_calls == []
        selection.cancel_rotation()
    finally:
        selection.clear_targets()
        plt.close(fig)
    assert app is not None


def test_escape_cancels_rigid_rotation_preview_before_clearing_selection() -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
    artists = [
        ax.add_patch(
            Polygon([[0.24, 0.3], [0.37, 0.32], [0.3, 0.45]], closed=True)
        ),
        ax.add_patch(
            Polygon([[0.58, 0.54], [0.72, 0.57], [0.65, 0.7]], closed=True)
        ),
    ]
    fig.canvas.draw()
    manager = attach_drag_manager(fig)
    fig.signals.figure_selection_property_changed = Signal()
    manager.select_elements(artists, primary=artists[-1])
    selection = manager.selection
    before = [TargetWrapper(artist).get_positions().copy() for artist in artists]
    pivot = selection.rotation_pivot()
    start = np.asarray(selection.rotation_grabber.get_xy(), dtype=float)
    vector = start - pivot
    angle = np.deg2rad(30.0)
    pointer = pivot + np.array(
        [
            vector[0] * np.cos(angle) - vector[1] * np.sin(angle),
            vector[0] * np.sin(angle) + vector[1] * np.cos(angle),
        ]
    )

    selection.start_rotation(SimpleNamespace(x=start[0], y=start[1], key=None))
    selection.preview_rotation(
        SimpleNamespace(x=pointer[0], y=pointer[1], key=None)
    )
    assert any(
        not np.allclose(TargetWrapper(artist).get_positions(), original)
        for artist, original in zip(artists, before)
    )

    manager.key_press_event(KeyEvent("key_press_event", fig.canvas, "escape"))

    assert all(
        np.allclose(TargetWrapper(artist).get_positions(), original, atol=1e-8)
        for artist, original in zip(artists, before)
    )
    assert [target.target for target in selection.targets] == artists
    assert manager.selected_element is artists[-1]
    assert not hasattr(selection, "rotation_drag_mode")
    assert not fig.change_tracker.edits
    assert not fig.change_tracker.changes
    assert selection.rotation_grabber.handle.isVisible()

    manager.key_press_event(KeyEvent("key_press_event", fig.canvas, "escape"))
    assert not selection.targets
    assert manager.selected_element is None
    plt.close(fig)
    assert app is not None


def test_escape_cancels_native_rotation_preview_without_recording() -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    fig = plt.figure(figsize=(4, 3), dpi=100)
    target = Rectangle(
        (110.0, 105.0),
        52.0,
        34.0,
        rotation_point=(125.0, 118.0),
        transform=IdentityTransform(),
        clip_on=False,
    )
    fig.add_artist(target)
    fig.canvas.draw()
    manager = attach_drag_manager(fig)
    fig.signals.figure_selection_property_changed = Signal()
    manager.select_element(target)
    selection = manager.selection
    pivot = selection.rotation_pivot()
    start = np.asarray(selection.rotation_grabber.get_xy(), dtype=float)
    vector = start - pivot
    angle = np.deg2rad(37.0)
    pointer = pivot + np.array(
        [
            vector[0] * np.cos(angle) - vector[1] * np.sin(angle),
            vector[0] * np.sin(angle) + vector[1] * np.cos(angle),
        ]
    )

    selection.start_rotation(SimpleNamespace(x=start[0], y=start[1], key=None))
    selection.preview_rotation(
        SimpleNamespace(x=pointer[0], y=pointer[1], key=None)
    )
    assert target.get_angle() == pytest.approx(37.0)

    manager.key_press_event(KeyEvent("key_press_event", fig.canvas, "escape"))

    assert target.get_angle() == pytest.approx(0.0)
    assert [wrapper.target for wrapper in selection.targets] == [target]
    assert manager.selected_element is target
    assert not fig.change_tracker.edits
    assert fig.change_tracker.change_count == 0
    assert not hasattr(selection, "rotation_drag_mode")
    selection.clear_targets()
    plt.close(fig)
    assert app is not None


def test_rigid_preview_rollback_continues_after_one_target_restore_fails(
    request,
) -> None:
    class PersistentFailurePolygon(Polygon):
        def __init__(self, *args, **kwargs):
            self.set_xy_calls = 0
            self.fail_after = None
            super().__init__(*args, **kwargs)

        def set_xy(self, xy):
            self.set_xy_calls += 1
            if self.fail_after is not None and self.set_xy_calls > self.fail_after:
                raise RuntimeError("QA persistent rigid failure")
            return super().set_xy(xy)

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
    first = ax.add_patch(
        Polygon([[0.24, 0.3], [0.37, 0.32], [0.3, 0.45]], closed=True)
    )
    second = ax.add_patch(
        PersistentFailurePolygon(
            [[0.58, 0.54], [0.72, 0.57], [0.65, 0.7]], closed=True
        )
    )
    artists = [first, second]
    artist_adapter_registry.register(PersistentFailurePolygon, PolygonAdapter)
    request.addfinalizer(
        lambda: artist_adapter_registry.unregister(
            PersistentFailurePolygon, PolygonAdapter
        )
    )
    fig.canvas.draw()
    manager = attach_drag_manager(fig)
    fig.signals.figure_selection_property_changed = Signal()
    manager.select_elements(artists, primary=second)
    selection = manager.selection
    before = [TargetWrapper(artist).get_positions().copy() for artist in artists]
    pivot = selection.rotation_pivot()
    start = np.asarray(selection.rotation_grabber.get_xy(), dtype=float)
    vector = start - pivot

    def pointer(angle_degrees):
        angle = np.deg2rad(angle_degrees)
        return pivot + np.array(
            [
                vector[0] * np.cos(angle) - vector[1] * np.sin(angle),
                vector[0] * np.sin(angle) + vector[1] * np.cos(angle),
            ]
        )

    selection.start_rotation(SimpleNamespace(x=start[0], y=start[1], key=None))
    first_pointer = pointer(30.0)
    selection.preview_rotation(
        SimpleNamespace(x=first_pointer[0], y=first_pointer[1], key=None)
    )
    second.fail_after = second.set_xy_calls + 1
    failing_pointer = pointer(45.0)

    with pytest.raises(RuntimeError, match="persistent rigid failure") as error:
        selection.preview_rotation(
            SimpleNamespace(x=failing_pointer[0], y=failing_pointer[1], key=None)
        )

    assert all(
        np.allclose(TargetWrapper(artist).get_positions(), original, atol=1e-8)
        for artist, original in zip(artists, before)
    )
    assert not fig.change_tracker.edits
    assert not fig.change_tracker.changes
    assert not hasattr(selection, "rotation_drag_mode")
    assert selection.move_rollback_failures
    assert any(
        "rollback failures" in note
        for note in getattr(error.value, "__notes__", ())
    )
    selection.clear_targets()
    plt.close(fig)
    assert app is not None


def test_single_polygon_rotation_handle_uses_rigid_plan_and_one_undo() -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
    target = ax.add_patch(
        Polygon([[0.25, 0.3], [0.48, 0.34], [0.36, 0.58]], closed=True)
    )
    fig.canvas.draw()
    manager = attach_drag_manager(fig)
    fig.signals.figure_selection_property_changed = Signal()
    manager.select_element(target)
    selection = manager.selection
    before = TargetWrapper(target).get_positions().copy()

    assert selection.rotation_operation() is TransformOperation.RIGID_ROTATE
    assert selection.rotation_handle_supported()
    pivot = selection.rotation_pivot()
    start = np.asarray(selection.rotation_grabber.get_xy(), dtype=float)
    vector = start - pivot
    angle = np.deg2rad(23.0)
    pointer = pivot + np.array(
        [
            vector[0] * np.cos(angle) - vector[1] * np.sin(angle),
            vector[0] * np.sin(angle) + vector[1] * np.cos(angle),
        ]
    )
    matrix = TargetWrapper(target).adapter.display_rotation_matrix(23.0, pivot)
    expected = TargetWrapper(target).adapter._transform_points(matrix, before)

    selection.start_rotation(SimpleNamespace(x=start[0], y=start[1], key=None))
    preview = selection.preview_rotation(
        SimpleNamespace(x=pointer[0], y=pointer[1], key=None)
    )

    assert preview == pytest.approx(23.0)
    assert np.allclose(TargetWrapper(target).get_positions(), expected, atol=1e-8)
    assert not fig.change_tracker.changes
    assert selection.end_rotation()
    fig.canvas.draw()
    assert np.allclose(TargetWrapper(target).get_positions(), expected, atol=1e-8)
    assert len(fig.change_tracker.edits) == 1
    assert {artist for artist, _command in fig.change_tracker.changes} == {target}

    fig.change_tracker.edit[0]()
    fig.canvas.draw()
    assert np.allclose(TargetWrapper(target).get_positions(), before, atol=1e-8)
    selection.clear_targets()
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


def test_owner_managed_rotation_leaves_never_expose_native_fallback_handle() -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    fig, ax = plt.subplots(figsize=(4, 4), dpi=100)
    title = ax.set_title("layout title", rotation_mode="anchor")
    legend = ax.legend(handles=[Patch(label="patch handle")], title="legend")
    legend.get_title().set_rotation_mode("anchor")
    fig.canvas.draw()
    manager = attach_drag_manager(fig)
    targets = [title, legend.get_title(), legend.legend_handles[0], ax.patch, fig.patch]

    for target in targets:
        manager.select_elements([target], primary=target)
        assert manager.selection.rotation_operation() is TransformOperation.ROTATE
        assert not manager.selection.rotation_handle_supported()
        assert not manager.selection.rotation_grabber.handle.isVisible()
        assert not TargetWrapper(target).operation_support("rigid_rotate").supported

    manager.selection.clear_targets()
    plt.close(fig)
    assert app is not None


def test_native_angle_property_does_not_imply_a_visual_rotation_handle() -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
    anisotropic = ax.add_patch(Rectangle((0.15, 0.15), 0.2, 0.25))
    hatched = Rectangle(
        (80.0, 80.0),
        42.0,
        31.0,
        hatch="//",
        transform=IdentityTransform(),
        clip_on=False,
    )
    effected = Rectangle(
        (150.0, 85.0),
        42.0,
        31.0,
        transform=IdentityTransform(),
        clip_on=False,
    )
    effected.set_path_effects([path_effects.SimplePatchShadow(offset=(20, 0))])
    fig.add_artist(hatched)
    fig.add_artist(effected)
    default_text = ax.text(0.45, 0.55, "default text")
    effected_text = ax.text(0.45, 0.7, "effected text")
    effected_text.set_path_effects([path_effects.withStroke(linewidth=4)])
    annotation = ax.annotate(
        "annotation",
        xy=(0.8, 0.25),
        xytext=(0.62, 0.48),
        arrowprops={"arrowstyle": "->"},
    )
    anchor_text = ax.text(0.72, 0.75, "anchor", rotation_mode="anchor")
    fig.canvas.draw()
    manager = attach_drag_manager(fig)
    fig.signals.figure_selection_property_changed = Signal()
    unsafe = [
        anisotropic,
        hatched,
        effected,
        default_text,
        effected_text,
        annotation,
    ]

    for target in unsafe:
        wrapper = TargetWrapper(target)
        old_angle = wrapper.get_rotation()
        manager.select_element(target)
        assert wrapper.operation_support(TransformOperation.ROTATE).supported
        assert not wrapper.native_rotation_handle_support().supported
        assert not manager.selection.rotation_handle_supported()
        assert not manager.selection.rotation_grabber.handle.isVisible()
        with pytest.raises(UnsupportedArtistError):
            manager.selection.rotate_selection(13.0)
        assert wrapper.get_rotation() == pytest.approx(old_angle)
        assert not fig.change_tracker.edits
        assert not fig.change_tracker.changes

    manager.select_element(anchor_text)
    assert manager.selection.rotation_operation() is TransformOperation.RIGID_ROTATE
    assert manager.selection.rotation_handle_supported()
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

    assert not ax.get_visible()
    assert len(fig.change_tracker.edits) == 1
    assert fig.change_tracker.edits[0][2] == "Delete object"

    fig.change_tracker.edits[0][0]()
    assert ax.get_visible()
    assert [target.target for target in manager.selection.targets] == [ax]
    plt.close(fig)
    assert app is not None


def test_delete_undo_redo_emit_one_final_selection_refresh() -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    fig, ax = plt.subplots(figsize=(4, 2), dpi=100)
    rectangle = ax.add_patch(Rectangle((0.2, 0.3), 0.25, 0.2))
    fig.canvas.draw()
    manager = attach_drag_manager(fig)
    _install_real_history_tracker(fig)
    manager.select_element(rectangle)

    fig.signals.selected.clear()
    manager.selection.delete_targets()
    assert fig.signals.selected == [None]

    fig.signals.selected.clear()
    manager.undo()
    assert rectangle.get_visible()
    assert fig.signals.selected == [rectangle]

    fig.signals.selected.clear()
    manager.redo()
    assert not rectangle.get_visible()
    assert fig.signals.selected == [None]

    manager.selection.clear_targets()
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


def test_marquee_reuses_revision_geometry_until_explicit_invalidation() -> None:
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
    assert calls == 1

    manager.invalidate_geometry_cache()
    manager.select_elements_in_bbox(*bounds)
    assert calls == 2
    manager.selection.clear_targets()
    plt.close(fig)
    assert app is not None


def test_marquee_spatial_index_preserves_targets_without_full_roster_scan() -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    fig = plt.figure(figsize=(8, 6), dpi=100)
    rectangles = []
    for ix in range(25):
        for iy in range(16):
            rectangle = Rectangle(
                (ix / 25 + 0.004, iy / 16 + 0.004),
                0.018,
                0.025,
                transform=fig.transFigure,
            )
            fig.add_artist(rectangle)
            rectangles.append(rectangle)
    fig.canvas.draw()
    manager = attach_drag_manager(fig)
    target = rectangles[8 * 16 + 7]
    bounds = artist_visible_extent(target)
    calls = 0
    original_pick_candidate = manager._is_pick_candidate

    def counted_pick_candidate(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original_pick_candidate(*args, **kwargs)

    manager._is_pick_candidate = counted_pick_candidate
    selected = manager.select_elements_in_bbox(*bounds)

    assert selected == [target]
    assert calls < len(rectangles) // 10
    assert manager._marquee_index.built_revision == manager._interaction_revision
    manager.selection.clear_targets()
    plt.close(fig)
    assert app is not None


def test_large_selection_uses_constant_overlay_items_and_fast_warm_reselect() -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    fig = plt.figure(figsize=(8, 6), dpi=100)
    texts = [
        fig.text((index % 25) / 25, (index // 25) / 15, str(index), fontsize=4)
        for index in range(365)
    ]
    fig.canvas.draw()
    manager = attach_drag_manager(fig)

    start = perf_counter()
    selected = manager.select_elements_in_bbox(*fig.bbox.extents)
    cold_elapsed = perf_counter() - start

    assert selected == texts
    assert manager.selection.targets_rects == []
    overlay_items = manager.selection._batched_selection_overlay_items
    assert len(overlay_items) == 3
    assert overlay_items[0].path().elementCount() == 5 * len(texts)
    assert overlay_items[2].path().isEmpty()
    assert cold_elapsed < 0.100

    manager.selection.set_alignment_reference("key_object", key=texts[0])
    assert overlay_items[0].path().elementCount() == 5 * (len(texts) - 1)
    assert overlay_items[2].path().elementCount() == 5
    assert overlay_items[2].pen().width() == 5

    # Measure the warm selection algorithm, not a generation-2 collection of
    # cyclic Matplotlib/Qt objects accumulated by unrelated earlier tests.
    # This mirrors the smart-guide performance harness: collect outside the
    # timed region and restore the caller's GC policy without relaxing the
    # latency budget.
    gc.collect()
    gc_was_enabled = gc.isenabled()
    if gc_was_enabled:
        gc.disable()
    try:
        start = perf_counter()
        manager.select_elements_in_bbox(*fig.bbox.extents)
        warm_elapsed = perf_counter() - start
    finally:
        if gc_was_enabled:
            gc.enable()
    assert warm_elapsed < 0.050

    manager.selection.update_selection_rectangles(target_indices=(0, len(texts) - 1))
    manager.selection.refresh_targets_after_draw((0, len(texts) - 1))
    manager.selection.remove_target(texts[0])
    assert len(manager.selection.targets) == len(texts) - 1
    assert manager.selection.targets_rects == []
    assert manager.selection._batched_selection_overlay_items == overlay_items
    assert overlay_items[2].path().isEmpty()

    manager.selection.clear_targets()
    assert all(item.scene() is None for item in overlay_items)
    manager.select_elements(texts[:2], primary=texts[1])
    assert len(manager.selection.targets_rects) == 4
    assert manager.selection._batched_selection_overlay_items == []
    manager.selection.clear_targets()
    plt.close(fig)
    assert app is not None


def test_draw_invalidation_releases_externally_removed_artist_rosters() -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
    rectangle = ax.add_patch(Rectangle((0.2, 0.25), 0.3, 0.35))
    fig.canvas.draw()
    manager = attach_drag_manager(fig)
    rectangle_id = id(rectangle)
    reference = weakref.ref(rectangle)
    bounds = artist_visible_extent(rectangle)
    event = MouseEvent(
        "button_press_event",
        fig.canvas,
        (bounds[0] + bounds[2]) / 2,
        (bounds[1] + bounds[3]) / 2,
        button=1,
    )
    assert rectangle in manager.get_hit_stack(event).artists
    assert rectangle in manager._selectable_roster_snapshot().artists
    assert rectangle_id in manager.editor_scene._known_artists

    rectangle.remove()
    manager.invalidate_geometry_cache()

    assert rectangle not in manager._interaction_artists
    assert rectangle not in manager._selectable_artists
    assert rectangle not in manager._interaction_roster_snapshot()[0].artists
    assert rectangle not in manager._selectable_roster_snapshot().artists
    assert rectangle_id not in manager.editor_scene._known_artists
    assert rectangle_id not in manager._selection_parent_by_id
    assert manager._display_geometry_cache.roster is None

    del rectangle
    gc.collect()
    assert reference() is None
    plt.close(fig)
    assert app is not None


def test_delaxes_prunes_axes_descendants_and_releases_strong_references() -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    fig, axes = plt.subplots(figsize=(4, 3), dpi=100)
    (line,) = axes.plot([0.2, 0.8], [0.3, 0.7])
    text = axes.text(0.5, 0.5, "detached")
    fig.canvas.draw()
    manager = attach_drag_manager(fig)
    references = tuple(weakref.ref(artist) for artist in (axes, line, text))
    detached_ids = {id(axes), id(line), id(text)}

    assert all(manager._is_artist_attached(artist) for artist in (axes, line, text))
    fig.delaxes(axes)
    assert axes.figure is fig
    assert line.figure is fig
    assert not manager._is_artist_attached(axes)
    assert not manager._is_artist_attached(line)
    assert not manager._is_artist_attached(text)

    manager.invalidate_geometry_cache()

    assert detached_ids.isdisjoint(manager._interaction_artist_ids)
    assert detached_ids.isdisjoint(manager._selectable_artist_ids)
    assert detached_ids.isdisjoint(manager.editor_scene._known_artists)
    assert detached_ids.isdisjoint(manager._selection_parent_by_id)
    event = MouseEvent("button_press_event", fig.canvas, 200, 150, button=1)
    assert not detached_ids.intersection(
        id(artist) for artist in manager.get_hit_stack(event).artists
    )

    del axes, line, text
    gc.collect()
    assert all(reference() is None for reference in references)
    plt.close(fig)
    assert app is not None


def test_child_and_subfigure_axes_follow_their_live_owner_inventory() -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    fig = plt.figure(figsize=(6, 3), dpi=100)
    left, _right = fig.subfigures(1, 2)
    axes = left.subplots()
    child = axes.inset_axes([0.2, 0.2, 0.4, 0.4])
    child_text = child.text(0.5, 0.5, "inset")
    fig.canvas.draw()
    manager = attach_drag_manager(fig)

    assert axes.figure is left
    assert axes in left.axes
    assert child not in fig.axes
    assert child in axes.child_axes
    assert manager._is_artist_attached(axes)
    assert manager._is_artist_attached(child)
    assert manager._is_artist_attached(child_text)
    manager.invalidate_geometry_cache()
    assert axes in manager._selectable_artists
    assert child in manager._selectable_artists

    left.delaxes(axes)
    assert axes.figure is left
    assert child.figure is left
    assert child in axes.child_axes
    assert not manager._is_artist_attached(axes)
    assert not manager._is_artist_attached(child)
    assert not manager._is_artist_attached(child_text)
    manager.invalidate_geometry_cache()
    assert axes not in manager._interaction_artists
    assert child not in manager._interaction_artists
    assert child_text not in manager._interaction_artists
    plt.close(fig)
    assert app is not None


def test_undoable_hidden_axes_remain_attached_and_in_rosters() -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    fig, axes = plt.subplots(figsize=(4, 3), dpi=100)
    fig.canvas.draw()
    manager = attach_drag_manager(fig)
    manager.select_element(axes)

    assert manager.set_selection_visible(False)
    undo = fig.change_tracker.edit[0]
    manager.invalidate_geometry_cache()

    assert axes in fig.axes
    assert manager._is_artist_attached(axes)
    assert axes in manager._interaction_artists
    assert axes in manager._selectable_artists
    assert not axes.get_visible()

    undo()
    manager.invalidate_geometry_cache()
    assert axes.get_visible()
    assert manager._is_artist_attached(axes)
    assert axes in manager._selectable_roster_snapshot().artists
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
    assert np.allclose(manager.selection.positions, artist_visible_extent(legend))

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


def test_legend_property_rebuild_keeps_overall_selection_extent_live() -> None:
    from pylustrator.components.qitem_properties import LegendPropertiesWidget

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
    legend = ax.legend(handles=[Patch(label="A")], frameon=False, borderpad=0.4)
    fig.canvas.draw()
    manager = attach_drag_manager(fig)
    manager.select_element(legend)

    widget = LegendPropertiesWidget.__new__(LegendPropertiesWidget)
    widget.properties = {"frameon": False, "borderpad": 0.4}
    widget.target = legend

    def assert_live_overlay():
        current = ax.get_legend()
        actual = artist_visible_extent(current)
        assert manager.selected_element is current
        assert [target.target for target in manager.selection.targets] == [current]
        assert np.allclose(manager.selection.positions, actual, atol=1e-9)
        assert np.allclose(
            selection_rect_extents(manager.selection)[0], actual, atol=1e-9
        )

    widget.changePropertiy("frameon", True)
    frame_edit = fig.change_tracker.edits[-1]
    assert_live_overlay()

    widget.changePropertiy("borderpad", 0.6)
    property_edit = fig.change_tracker.edits[-1]
    assert_live_overlay()

    for action in (
        property_edit[0],
        frame_edit[0],
        frame_edit[1],
        property_edit[1],
    ):
        action()
        assert_live_overlay()

    manager.selection.clear_targets()
    plt.close(fig)
    assert app is not None


def test_draw_event_refreshes_legend_selection_after_move_undo() -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
    legend = ax.legend(handles=[Patch(label="A")], frameon=False)
    fig.canvas.draw()
    manager = attach_drag_manager(fig)
    manager._selection_refresh_on_draw = True
    draw_connection = fig.canvas.mpl_connect(
        "draw_event", manager.invalidate_geometry_cache
    )
    manager.select_element(legend)

    manager.selection.start_move()
    manager.selection.addOffset((1, -1), DIR_X0 | DIR_X1 | DIR_Y0 | DIR_Y1)
    manager.selection.has_moved = True
    manager.selection.end_move()
    fig.canvas.draw()

    fig.change_tracker.edit[0]()
    fig.canvas.draw()
    actual = artist_visible_extent(legend)
    assert np.allclose(manager.selection.positions, actual, atol=1e-9)
    assert np.allclose(
        selection_rect_extents(manager.selection)[0], actual, atol=1e-9
    )

    fig.canvas.mpl_disconnect(draw_connection)
    manager.selection.clear_targets()
    plt.close(fig)
    assert app is not None


def test_draw_event_remeasures_only_layout_late_selection_targets(
    monkeypatch,
) -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
    lines = [
        ax.plot([0, 1], [offset, offset + 0.2])[0]
        for offset in np.linspace(0, 1, 24)
    ]
    legend = ax.legend(handles=[lines[0]], labels=["representative"])
    fig.canvas.draw()
    manager = attach_drag_manager(fig)
    manager.select_elements([*lines, legend])

    calls = {}
    original = TargetWrapper.get_selection_points

    def counted(wrapper):
        key = id(wrapper.target)
        calls[key] = calls.get(key, 0) + 1
        return original(wrapper)

    monkeypatch.setattr(TargetWrapper, "get_selection_points", counted)
    manager._selection_refresh_on_draw = True
    draw_connection = fig.canvas.mpl_connect(
        "draw_event", manager.invalidate_geometry_cache
    )
    fig.canvas.draw()

    assert calls.get(id(legend), 0) == 1
    assert all(calls.get(id(line), 0) == 0 for line in lines)

    fig.canvas.mpl_disconnect(draw_connection)
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
