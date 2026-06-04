from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.backend_bases import MouseEvent
from qtpy import QtCore, QtWidgets

from pylustrator.components.plot_layout import scene_point_to_canvas_pixels, selection_scene_transform
from pylustrator.components.tree_view import MyTreeView
from pylustrator.drag_helper import DragManager, GrabbableRectangleSelection
from pylustrator.snap import TargetWrapper


class SelectionView:
    h = 200
    device_pixel_ratio = 1.0
    grabber_found = False


class ChangeTracker:
    def addEdit(self, edit):
        self.edit = edit


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


def test_drag_rectangle_selects_intersecting_artists_without_background_axes() -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
    text = ax.text(0.5, 0.5, "inside")
    fig.canvas.draw()
    manager = attach_drag_manager(fig)
    bbox = text.get_window_extent(fig.canvas.get_renderer()).expanded(1.2, 1.4)

    selected = manager.select_elements_in_bbox(bbox.x0, bbox.y0, bbox.x1, bbox.y1)

    assert text in selected
    assert ax not in selected
    assert text in [target.target for target in manager.selection.targets]
    assert ax not in [target.target for target in manager.selection.targets]
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
