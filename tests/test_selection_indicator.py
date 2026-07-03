from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.backend_bases import KeyEvent, MouseEvent
from matplotlib.patches import Patch
from qtpy import QtCore, QtGui, QtWidgets

from pylustrator.components.plot_layout import scene_point_to_canvas_pixels, selection_scene_transform
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
        points = TargetWrapper(target.target).get_positions()
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


def test_legend_child_drag_moves_parent_legend_without_internal_offset() -> None:
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
    assert text.get_position() == text_position
    assert abs((text_after.x0 - text_before.x0) - 12) < 1e-9
    assert abs((text_after.y0 - text_before.y0) + 7) < 1e-9
    assert abs((legend_after.x0 - legend_before.x0) - 12) < 1e-9
    assert abs((legend_after.y0 - legend_before.y0) + 7) < 1e-9
    assert fig.change_tracker.legend is legend
    assert fig.change_tracker.legend_change_count == 1
    assert fig.change_tracker.text_change_count == 0
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


def test_backspace_deletes_selected_object() -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    fig, ax = plt.subplots(figsize=(4, 2), dpi=100)
    fig.canvas.draw()
    manager = attach_drag_manager(fig)
    manager.selection.add_target(ax)

    manager.selection.keyPressEvent(KeyEvent("key_press_event", fig.canvas, "backspace"))

    assert fig.change_tracker.removed is ax
    assert not ax.get_visible()
    plt.close(fig)
    assert app is not None


def test_selection_scene_transform_maps_physical_canvas_pixels_to_logical_scene() -> None:
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


def test_drag_rectangle_selects_intersecting_artists_and_axes_consistently() -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
    text = ax.text(0.5, 0.5, "inside")
    fig.canvas.draw()
    manager = attach_drag_manager(fig)
    bbox = text.get_window_extent(fig.canvas.get_renderer()).expanded(1.2, 1.4)

    selected = manager.select_elements_in_bbox(bbox.x0, bbox.y0, bbox.x1, bbox.y1)

    assert text in selected
    assert ax in selected
    assert text in [target.target for target in manager.selection.targets]
    assert ax in [target.target for target in manager.selection.targets]
    manager.selection.clear_targets()
    plt.close(fig)
    assert app is not None


def test_drag_rectangle_selects_empty_plot_area_axes() -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
    fig.canvas.draw()
    manager = attach_drag_manager(fig)
    bbox = ax.get_window_extent(fig.canvas.get_renderer())
    cx = (bbox.x0 + bbox.x1) / 2
    cy = (bbox.y0 + bbox.y1) / 2

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

    assert ax in selected
    assert text in selected
    assert [target.target for target in manager.selection.targets] == [ax, text]
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
    assert ax in [target.target for target in manager.selection.targets]
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
