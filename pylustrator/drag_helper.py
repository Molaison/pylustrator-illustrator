#!/usr/bin/env python
# -*- coding: utf-8 -*-
# drag_helper.py

# Copyright (c) 2016-2020, Richard Gerum
#
# This file is part of Pylustrator.
#
# Pylustrator is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Pylustrator is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Pylustrator. If not, see <http://www.gnu.org/licenses/>

import numpy as np
from matplotlib.artist import Artist
from matplotlib.figure import Figure, SubFigure
from matplotlib.axes import Axes
from matplotlib.legend import Legend
from matplotlib.text import Text
from matplotlib.patches import Rectangle
from matplotlib.backend_bases import MouseEvent, KeyEvent
from typing import Iterable, Sequence
from qtpy import QtCore, QtGui, QtWidgets

from .artist_adapters import (
    UnsupportedArtistError,
    iter_figure_legends,
    iter_legend_children,
    selection_geometry_snapshot,
    suspend_change_recording,
)
from .snap import (
    TargetWrapper,
    getSnaps,
    checkSnaps,
    checkSnapsActive,
    SnapBase,
)
from .change_tracker import ChangeTracker, add_text_default
from .components.plot_layout import scene_point_to_canvas_pixels
from .editor_model import EditorGroup, EditorScene
from .interaction import HitCandidate, HitStack, SelectionKernel, SelectionMode
from .operations import OperationSupport, TransformOperation
from .commands import InteractionState, ObjectLocator, semantic_equal
from pylustrator.change_tracker import UndoRedo
import time

DIR_X0 = 1
DIR_Y0 = 2
DIR_X1 = 4
DIR_Y1 = 8

blit = False


def _legend_selectable_children(legend: Legend) -> list[Artist]:
    """Return legend parts that Matplotlib does not expose reliably as children."""
    return list(iter_legend_children(legend))


def iter_artist_children(element: Artist) -> list[tuple[Artist, bool]]:
    """Return normal children plus explicit editable children.

    The boolean marks explicit children that Pylustrator exposes even when
    Matplotlib gives them private labels, such as legend handles.
    """
    children: list[tuple[Artist, bool]] = [
        (child, False) for child in element.get_children()
    ]
    if isinstance(element, Legend):
        children.extend((child, True) for child in _legend_selectable_children(element))

    by_id: dict[int, int] = {}
    unique: list[tuple[Artist, bool]] = []
    for child, explicit in children:
        key = id(child)
        if key in by_id:
            index = by_id[key]
            unique[index] = (unique[index][0], unique[index][1] or explicit)
            continue
        by_id[key] = len(unique)
        unique.append((child, explicit))
    return unique


def get_artist_children(element: Artist) -> list[Artist]:
    return [child for child, _explicit in iter_artist_children(element)]


def _is_internal_label(artist: Artist, explicit: bool = False) -> bool:
    if explicit or getattr(artist, "_pylustrator_explicitly_editable", False):
        return False
    label = artist.get_label()
    return isinstance(label, str) and label.startswith("_")


def _container_yields_to_children(artist: Artist) -> bool:
    return isinstance(artist, (Figure, SubFigure, Axes))


def _container_keeps_children(artist: Artist) -> bool:
    return isinstance(artist, Legend)


def _event_has_modifier(event, modifier: str) -> bool:
    key = getattr(event, "key", None)
    return (
        modifier in key.split("+")
        if event is not None and key is not None
        else False
    )


def _constrain_to_cardinal_direction(pos: Sequence[float], dir: int) -> list[float]:
    """Constrain full-object drags to horizontal or vertical movement."""
    pos = [float(pos[0]), float(pos[1])]
    if dir != (DIR_X0 | DIR_X1 | DIR_Y0 | DIR_Y1):
        return pos
    if abs(pos[0]) >= abs(pos[1]):
        pos[1] = 0.0
    else:
        pos[0] = 0.0
    return pos


class GrabFunctions(object):
    """basic functionality used by all grabbers"""

    figure = None
    target = None
    dir = None
    snaps = None

    got_artist = False

    def __init__(self, parent, dir: int, no_height=False):
        self.figure = parent.figure
        self.parent = parent
        self.dir = dir
        self.snaps = []
        self.no_height = no_height

    def on_motion(self, evt: MouseEvent):
        """callback when the object is moved"""
        if self.got_artist:
            self.movedEvent(evt)
            self.moved = True

    def button_press_event(self, evt: MouseEvent):
        """when the mouse is pressed"""
        self.got_artist = True
        self.moved = False

        self._c1 = self.figure.canvas.mpl_connect("motion_notify_event", self.on_motion)
        self.clickedEvent(evt)

    def button_release_event(self, event: MouseEvent):
        """when the mouse is released"""
        if self.got_artist:
            self.got_artist = False
            self.figure.canvas.mpl_disconnect(self._c1)
            try:
                self.releasedEvent(event)
            except UnsupportedArtistError as error:
                QtWidgets.QMessageBox.warning(None, "Pylustrator", str(error))
                self.figure.canvas.draw_idle()

    def clickedEvent(self, event: MouseEvent):
        """when the mouse is clicked"""
        self.parent.start_move()
        self.mouse_xy = (event.x, event.y)

        for s in self.snaps:
            s.remove()
        self.snaps = []

        self.snaps = getSnaps(self.targets, self.dir, no_height=self.no_height)

        if blit is True:
            for target in self.targets:
                target.target.set_animated(True)

            self.figure.canvas.draw()
            self.bg = self.figure.canvas.copy_from_bbox(self.figure.bbox)
        else:
            pass
        self.time = time.time()

    def releasedEvent(self, event: MouseEvent):
        """when the mouse is released"""
        for snap in self.snaps:
            snap.remove()
        self.snaps = []

        self.parent.end_move()

        if blit is True:
            for target in self.targets:
                target.target.set_animated(False)
        else:
            pass

    def movedEvent(self, event: MouseEvent):
        """when the mouse is moved"""
        if len(self.targets) == 0:
            return

        dx = event.x - self.mouse_xy[0]
        dy = event.y - self.mouse_xy[1]

        control_toggles_aspect = _event_has_modifier(event, "control")
        keep_aspect = (
            bool(getattr(self.parent, "lock_aspect_ratio", False))
            ^ control_toggles_aspect
        )
        constrain_direction = _event_has_modifier(event, "shift")
        ignore_snaps = _event_has_modifier(event, "alt") or _event_has_modifier(
            event, "option"
        )

        self.parent.move(
            [dx, dy],
            self.dir,
            self.snaps,
            keep_aspect_ratio=keep_aspect,
            ignore_snaps=ignore_snaps,
            constrain_direction=constrain_direction,
        )

        if blit is True:
            fig = self.figure
            fig.canvas.restore_region(self.bg)
            for target in self.targets:
                fig.draw_artist(target.target)
            # copy the image to the GUI state, but screen might not be changed yet
            fig.canvas.blit(fig.bbox)
            # flush any pending GUI events, re-painting the screen if needed
            fig.canvas.flush_events()
        else:
            if not getattr(self.parent, "defer_current_move", False):
                self.figure.canvas.schedule_draw()


class GrabbableRectangleSelection(GrabFunctions):
    grabbers = None

    def addGrabber(self, x: float, y: float, dir: int, GrabberClass: object):
        # add a grabber object at the given coordinates
        self.grabbers.append(GrabberClass(self, x, y, dir, self.graphics_scene))

    def __init__(self, figure: Figure, graphics_scene=None):
        self.grabbers = []
        pos = [0, 0, 0, 0]
        self.positions = np.array(pos, dtype=float)
        self.p1 = self.positions[:2]
        self.p2 = self.positions[2:]
        self.figure = figure
        self.graphics_scene = graphics_scene
        self.graphics_scene_myparent = QtWidgets.QGraphicsRectItem(
            0, 0, 0, 0, self.graphics_scene
        )
        self.graphics_scene_snapparent = QtWidgets.QGraphicsRectItem(
            0, 0, 0, 0, self.graphics_scene
        )
        figure._pyl_graphics_scene_snapparent = self.graphics_scene_snapparent

        GrabFunctions.__init__(
            self, self, DIR_X0 | DIR_X1 | DIR_Y0 | DIR_Y1, no_height=True
        )

        self.addGrabber(0, 0, DIR_X0 | DIR_Y0, GrabberGenericRound)
        self.addGrabber(0.5, 0, DIR_Y0, GrabberGenericRectangle)
        self.addGrabber(1, 1, DIR_X1 | DIR_Y1, GrabberGenericRound)
        self.addGrabber(1, 0.5, DIR_X1, GrabberGenericRectangle)
        self.addGrabber(0, 1, DIR_X0 | DIR_Y1, GrabberGenericRound)
        self.addGrabber(0.5, 1, DIR_Y1, GrabberGenericRectangle)
        self.addGrabber(1, 0, DIR_X1 | DIR_Y0, GrabberGenericRound)
        self.addGrabber(0, 0.5, DIR_X0, GrabberGenericRectangle)
        self.rotation_grabber = GrabberRotation(self, self.graphics_scene)

        self.c4 = self.figure.canvas.mpl_connect("key_press_event", self.keyPressEvent)

        self.targets = []
        self.targets_rects = []
        self.lock_aspect_ratio = False
        self.reference_point = (0.5, 0.5)
        self.defer_artist_updates = True

        self.hide_grabber()

    def add_target(self, target: Artist, update: bool = True):
        """add an artist to the selection"""
        if target in [wrapped.target for wrapped in self.targets]:
            return
        target = TargetWrapper(target)
        if not target.supported:
            return

        new_points = np.array(target.get_selection_points())
        if len(new_points) == 0:
            return

        self.targets.append(target)

        x0, y0, x1, y1 = (
            np.min(new_points[:, 0]),
            np.min(new_points[:, 1]),
            np.max(new_points[:, 0]),
            np.max(new_points[:, 1]),
        )
        if 0:
            rect1 = Rectangle(
                (x0, y0),
                x1 - x0,
                y1 - y0,
                picker=False,
                figure=self.figure,
                linestyle="-",
                edgecolor="w",
                facecolor="#FFFFFF00",
                zorder=900,
                label="_rect for %s" % str(target),
            )
            rect2 = Rectangle(
                (x0, y0),
                x1 - x0,
                y1 - y0,
                picker=False,
                figure=self.figure,
                linestyle="--",
                edgecolor="k",
                facecolor="#FFFFFF00",
                zorder=900,
                label="_rect2 for %s" % str(target),
            )
            self.figure.patches.append(rect1)
            self.figure.patches.append(rect2)
            self.targets_rects.append(rect1)
            self.targets_rects.append(rect2)
        else:
            pen1 = QtGui.QPen(QtGui.QColor("#1E88E5"), 3)
            pen2 = QtGui.QPen(QtGui.QColor("white"), 1)
            pen2.setStyle(QtCore.Qt.DashLine)
            brush1 = QtGui.QBrush(QtGui.QColor(30, 136, 229, 32))

            w0, h0 = x1 - x0, y1 - y0
            rect1 = QtWidgets.QGraphicsRectItem(
                x0, y0, w0, h0, self.graphics_scene_myparent
            )
            rect1.setPen(pen1)
            rect1.setBrush(brush1)
            rect1.setZValue(900)
            rect2 = QtWidgets.QGraphicsRectItem(
                x0, y0, w0, h0, self.graphics_scene_myparent
            )
            rect2.setPen(pen2)
            rect2.setZValue(901)

            self.targets_rects.append(rect1)
            self.targets_rects.append(rect2)

        if update:
            self.update_extent()

    def update_extent(self):
        """updates the extend of the selection to all the selected elements"""
        points = None
        for target in self.targets:
            new_points = np.array(target.get_selection_points())
            if new_points.ndim != 2 or not len(new_points):
                continue

            if points is None:
                points = new_points
            else:
                points = np.concatenate((points, new_points))

        if points is None or not len(points):
            self.hide_grabber()
            return

        for grabber in self.grabbers:
            grabber.targets = self.targets
        self.rotation_grabber.targets = self.targets

        self.positions[0] = np.min(points[:, 0])
        self.positions[1] = np.min(points[:, 1])
        self.positions[2] = np.max(points[:, 0])
        self.positions[3] = np.max(points[:, 1])

        if self.positions[2] - self.positions[0] < 0.01:
            self.positions[0], self.positions[2] = (
                self.positions[0] - 0.01,
                self.positions[0] + 0.01,
            )
        if self.positions[3] - self.positions[1] < 0.01:
            self.positions[1], self.positions[3] = (
                self.positions[1] - 0.01,
                self.positions[1] + 0.01,
            )

        self.update_grabber()

    @staticmethod
    def _unique_wrappers(wrappers: Iterable[TargetWrapper]) -> list[TargetWrapper]:
        unique = []
        seen = set()
        for wrapper in wrappers:
            key = id(wrapper.target)
            if key in seen:
                continue
            seen.add(key)
            unique.append(wrapper)
        return unique

    def operation_support(
        self, operation: TransformOperation | str
    ) -> OperationSupport:
        operation = TransformOperation.coerce(operation)
        if not self.targets:
            return OperationSupport.denied(operation, "No objects are selected")
        supports = [target.operation_support(operation) for target in self.targets]
        failures = [
            (target, support)
            for target, support in zip(self.targets, supports)
            if not support.supported
        ]
        if failures:
            reason = "; ".join(
                f"{type(target.target).__name__}: {support.reason}"
                for target, support in failures
            )
            return OperationSupport.denied(operation, reason)
        constraints = tuple(
            dict.fromkeys(
                constraint
                for support in supports
                for constraint in support.constraints
            )
        )
        strategies = {support.preview_strategy for support in supports}
        return OperationSupport.allowed(
            operation,
            constraints=constraints,
            preview_strategy=(strategies.pop() if len(strategies) == 1 else "mixed"),
        )

    def _resolve_alignment_items(self) -> list[tuple[TargetWrapper, TargetWrapper]]:
        """Measure and move the exact artists selected by the user.

        Every wrapper accepts display-space deltas, so alignment no longer needs
        to promote children to an ancestor merely to find a shared coordinate
        system.
        """
        return [(target, target) for target in self.targets]

    def _delta_plan(
        self,
        items: list[tuple[TargetWrapper, TargetWrapper]],
        deltas: list[float],
    ) -> list[tuple[TargetWrapper, float]]:
        plan = {}
        for (_measure, move_target), delta in zip(items, deltas):
            key = id(move_target.target)
            if key in plan:
                _target, existing_delta = plan[key]
                if not np.isclose(existing_delta, delta, atol=1e-9):
                    raise ValueError(
                        "Selected subobjects resolve to the same parent with conflicting alignment deltas."
                    )
                continue
            plan[key] = (move_target, delta)
        return list(plan.values())

    @staticmethod
    def _bounds_from_points(points_list: list[np.ndarray]) -> np.ndarray:
        points = np.concatenate(points_list)
        return np.array(
            [
                np.min(points[:, 0]),
                np.min(points[:, 1]),
                np.max(points[:, 0]),
                np.max(points[:, 1]),
            ],
            dtype=float,
        )

    def _measure_points(
        self, measure_target: TargetWrapper, move_target: TargetWrapper
    ) -> np.ndarray:
        """Return display-space bounds used to compute alignment deltas."""
        return np.array(measure_target.get_selection_points(), dtype=float)

    @staticmethod
    def _measure_size(points: np.ndarray, y: int, direct_target: bool) -> float:
        return np.max(points[:, y]) - np.min(points[:, y])

    def align_points(self, mode: str):
        """a function to apply the alignment options, e.g. align all selected elements at the top or with equal spacing."""
        if len(self.targets) == 0:
            return

        if mode == "group":
            from pylustrator.helper_functions import axes_to_grid

            # return axes_to_grid([target.target for target in self.targets], track_changes=True)
            with UndoRedo(
                [
                    target.target
                    for target in self.targets
                    if isinstance(target.target, Axes)
                ],
                "Grid Align",
            ):
                axes_to_grid(
                    [
                        target.target
                        for target in self.targets
                        if isinstance(target.target, Axes)
                    ],
                    track_changes=False,
                )

        def reference_bounds(points_list: list[np.ndarray]):
            if len(points_list) == 1:
                bbox = self.figure.bbox
                return np.array([bbox.x0, bbox.y0, bbox.x1, bbox.y1])
            return self._bounds_from_points(points_list)

        def execute_translation_plan(plan, axis: int, edit_name: str) -> None:
            vectors = []
            for target, delta in plan:
                display_delta = np.zeros(2, dtype=float)
                display_delta[axis] = delta
                target.preflight_rigid_visible_translation(display_delta)
                vectors.append((target, display_delta))
            self.start_move(save_targets=[target for target, _delta in vectors])
            try:
                for target, display_delta in vectors:
                    target.translate(display_delta)
                self.update_extent()
                self.has_moved = True
                self.end_move(edit_name)
            except Exception:
                self._restore_move_start()
                self.has_moved = False
                self.end_move(edit_name)
                raise

        def align(y: int, func: callable):
            items = self._resolve_alignment_items()
            measure_points = [
                self._measure_points(measure_target, move_target)
                for measure_target, move_target in items
            ]
            centers = []
            for points in measure_points:
                centers.append(func(points[:, y]))
            new_center = func(reference_bounds(measure_points)[y::2])
            deltas = [new_center - center for center in centers]
            plan = self._delta_plan(items, deltas)
            execute_translation_plan(plan, y, "Align")

            self.figure.canvas.draw()
            self.update_selection_rectangles()

        def distribute(y: int):
            items = self._resolve_alignment_items()
            sizes = []
            positions = []
            measure_points = [
                self._measure_points(measure_target, move_target)
                for measure_target, move_target in items
            ]
            for points, (measure_target, move_target) in zip(measure_points, items):
                sizes.append(
                    self._measure_size(
                        points, y, measure_target.target is move_target.target
                    )
                )
                positions.append(np.min(points[:, y]))
            order = np.argsort(positions)
            bounds = self._bounds_from_points(measure_points)
            spaces = np.diff(bounds[y::2])[0] - np.sum(sizes)
            spaces /= max([(len(items) - 1), 1])
            pos = np.min(bounds[y::2])
            deltas = [0.0] * len(items)
            for index in order:
                points = measure_points[index]
                deltas[index] = pos - np.min(points[:, y])
                pos += sizes[index] + spaces
            plan = self._delta_plan(items, deltas)
            execute_translation_plan(plan, y, "Distribute")

            self.figure.canvas.draw()
            self.update_selection_rectangles()

        if mode == "center_x":
            align(0, np.mean)

        if mode == "left_x":
            align(0, np.min)

        if mode == "right_x":
            align(0, np.max)

        if mode == "center_y":
            align(1, np.mean)

        if mode == "bottom_y":
            align(1, np.min)

        if mode == "top_y":
            align(1, np.max)

        if mode == "distribute_x":
            distribute(0)

        if mode == "distribute_y":
            distribute(1)

        self.figure.signals.figure_selection_moved.emit()

    @staticmethod
    def _points_bounds(points: np.ndarray) -> np.ndarray:
        """Return x0, y0, x1, y1 bounds for a TargetWrapper point set."""
        return np.array(
            [
                np.min(points[:, 0]),
                np.min(points[:, 1]),
                np.max(points[:, 0]),
                np.max(points[:, 1]),
            ],
            dtype=float,
        )

    def match_size(self, mode: str, keep_aspect_ratio: bool = None) -> bool:
        """Resize selected targets to the first selected target's width/height."""
        modes = {
            "width": (True, False),
            "height": (False, True),
            "size": (True, True),
        }
        if mode not in modes:
            raise ValueError(f"Unknown size matching mode: {mode}")
        if len(self.targets) < 2:
            raise ValueError("Select at least two objects to match size.")
        if keep_aspect_ratio is None:
            keep_aspect_ratio = self.lock_aspect_ratio

        match_width, match_height = modes[mode]
        reference_points = np.array(self.targets[0].get_selection_points(), dtype=float)
        reference_bounds = self._points_bounds(reference_points)
        reference_width = reference_bounds[2] - reference_bounds[0]
        reference_height = reference_bounds[3] - reference_bounds[1]
        if match_width and reference_width <= 0:
            raise ValueError("Reference object has no width.")
        if match_height and reference_height <= 0:
            raise ValueError("Reference object has no height.")

        unsupported = [
            (target.target, target.operation_support(TransformOperation.RESIZE_GEOMETRY))
            for target in self.targets[1:]
            if not target.supports_operation(TransformOperation.RESIZE_GEOMETRY)
        ]
        if unsupported:
            reasons = "; ".join(
                f"{type(target).__name__}: {support.reason}"
                for target, support in unsupported
            )
            raise ValueError(reasons)

        planned: list[tuple[TargetWrapper, np.ndarray]] = []
        for target in self.targets[1:]:
            bounds = self._points_bounds(
                np.array(target.get_selection_points(), dtype=float)
            )
            current_width = bounds[2] - bounds[0]
            current_height = bounds[3] - bounds[1]
            if match_width and current_width <= 0:
                raise ValueError("Selected object has no width.")
            if match_height and current_height <= 0:
                raise ValueError("Selected object has no height.")
            scale_x = reference_width / current_width if match_width else 1.0
            scale_y = reference_height / current_height if match_height else 1.0
            if keep_aspect_ratio:
                if match_width and match_height:
                    scale = min(scale_x, scale_y)
                    scale_x = scale_y = scale
                elif match_width:
                    scale_y = scale_x
                elif match_height:
                    scale_x = scale_y
            if np.isclose(scale_x, 1.0) and np.isclose(scale_y, 1.0):
                continue
            center_x = (bounds[0] + bounds[2]) / 2
            center_y = (bounds[1] + bounds[3]) / 2
            transform = np.array(
                [
                    [scale_x, 0.0, center_x * (1.0 - scale_x)],
                    [0.0, scale_y, center_y * (1.0 - scale_y)],
                    [0.0, 0.0, 1.0],
                ],
                dtype=float,
            )
            target.preflight_rigid_visible_resize(transform)
            planned.append((target, transform))

        if not planned:
            return False
        self.start_move()
        try:
            for target, transform in planned:
                target.resize(transform)
            self.update_extent()
            self.has_moved = True
            self.end_move("Resize")
        except Exception:
            self._restore_move_start()
            self.has_moved = False
            self.end_move("Resize")
            raise
        self.figure.canvas.draw()
        self.update_selection_rectangles()
        return True

    def scale_selection(self, factor: float) -> bool:
        """Scale the current selection around its combined center."""
        if factor <= 0:
            raise ValueError("Scale factor must be positive.")
        if len(self.targets) == 0:
            return False
        support = self.operation_support(TransformOperation.RESIZE_GEOMETRY)
        if not support.supported:
            raise ValueError(support.reason)
        if np.isclose(factor, 1.0):
            return False

        center_x, center_y = self.reference_position()
        transform = np.array(
            [
                [factor, 0.0, center_x * (1.0 - factor)],
                [0.0, factor, center_y * (1.0 - factor)],
                [0.0, 0.0, 1.0],
            ],
            dtype=float,
        )

        for target in self.targets:
            target.preflight_rigid_visible_resize(transform)
        self.start_move()
        try:
            for target in self.targets:
                target.resize(transform)
            self.update_extent()
            self.has_moved = True
            self.end_move("Scale")
        except Exception:
            self._restore_move_start()
            self.has_moved = False
            self.end_move("Scale")
            raise
        self.figure.canvas.draw()
        self.update_selection_rectangles()
        return True

    @staticmethod
    def _rotatable_value(target: Artist) -> float | None:
        wrapped = TargetWrapper(target)
        return (
            wrapped.get_rotation()
            if wrapped.supports_operation(TransformOperation.ROTATE)
            else None
        )

    def _set_rotation_value(self, target: Artist, value: float) -> None:
        TargetWrapper(target).set_rotation(value)

    def rotation_handle_supported(self) -> bool:
        """Expose a handle only when one exact native rotation pivot exists."""

        return bool(
            len(self.targets) == 1
            and self.operation_support(TransformOperation.ROTATE).supported
        )

    def rotation_pivot(self) -> np.ndarray:
        if not self.rotation_handle_supported():
            raise ValueError(
                "Rotation handles require one selected object with native rotation support"
            )
        return np.asarray(self.targets[0].get_rotation_pivot(), dtype=float)

    def start_rotation(self, event: MouseEvent) -> None:
        """Begin one deferred, atomic native-rotation gesture."""

        if not self.rotation_handle_supported():
            support = self.operation_support(TransformOperation.ROTATE)
            raise ValueError(support.reason or "Rotation handle is unavailable")
        pivot = self.rotation_pivot()
        pointer = np.asarray((event.x, event.y), dtype=float)
        vector = pointer - pivot
        if not np.all(np.isfinite(vector)) or np.linalg.norm(vector) <= 1e-9:
            raise ValueError("Rotation handle is too close to its native pivot")

        self.start_move(save_targets=self.targets)
        self.rotation_drag_target = self.targets[0]
        self.rotation_drag_pivot = pivot
        self.rotation_drag_start_pointer_angle = float(
            np.degrees(np.arctan2(vector[1], vector[0]))
        )
        self.rotation_drag_start_value = self.rotation_drag_target.get_rotation()
        self.rotation_drag_preview_value = self.rotation_drag_start_value

    def preview_rotation(self, event: MouseEvent) -> float:
        """Preview the exact native angle that will be committed on release."""

        target = getattr(self, "rotation_drag_target", None)
        if target is None:
            raise RuntimeError("No rotation gesture is active")
        pointer = np.asarray((event.x, event.y), dtype=float)
        vector = pointer - self.rotation_drag_pivot
        if not np.all(np.isfinite(vector)) or np.linalg.norm(vector) <= 1e-9:
            return self.rotation_drag_preview_value
        pointer_angle = float(np.degrees(np.arctan2(vector[1], vector[0])))
        delta = (
            pointer_angle - self.rotation_drag_start_pointer_angle + 180.0
        ) % 360.0 - 180.0
        if _event_has_modifier(event, "shift"):
            delta = round(delta / 15.0) * 15.0
        value = self.rotation_drag_start_value + delta
        try:
            with suspend_change_recording():
                target.set_rotation(value)
        except Exception:
            self._restore_move_start()
            self.has_moved = False
            self.end_move("Rotate")
            self._clear_rotation_gesture()
            raise
        self.rotation_drag_preview_value = value
        self.has_moved = not np.isclose(value, self.rotation_drag_start_value)
        self.update_extent()
        self.update_selection_rectangles()
        self.hide_grabber()
        canvas = self.figure.canvas
        if hasattr(canvas, "schedule_draw"):
            canvas.schedule_draw()
        else:
            canvas.draw_idle()
        return value

    def _clear_rotation_gesture(self) -> None:
        for name in (
            "rotation_drag_target",
            "rotation_drag_pivot",
            "rotation_drag_start_pointer_angle",
            "rotation_drag_start_value",
            "rotation_drag_preview_value",
        ):
            try:
                delattr(self, name)
            except AttributeError:
                pass

    def end_rotation(self) -> bool:
        """Commit one generated change and one undo item for a handle gesture."""

        target = getattr(self, "rotation_drag_target", None)
        if target is None:
            return False
        changed = not np.isclose(
            self.rotation_drag_preview_value, self.rotation_drag_start_value
        )
        try:
            if changed:
                # The preview was deliberately unrecorded. Reapplying the final
                # absolute value emits exactly one stable generated change.
                target.set_rotation(self.rotation_drag_preview_value)
                self.update_extent()
                self.has_moved = True
            else:
                self.has_moved = False
            self.end_move("Rotate")
        except Exception:
            self._restore_move_start()
            self.has_moved = False
            self.end_move("Rotate")
            self._clear_rotation_gesture()
            raise
        self._clear_rotation_gesture()
        signal = getattr(self.figure.signals, "figure_selection_property_changed", None)
        if signal is not None:
            signal.emit()
        self.figure.canvas.draw()
        self.update_selection_rectangles()
        return changed

    def rotate_selection(self, angle_degrees: float) -> bool:
        """Rotate selected objects that have a native saveable rotation property."""
        if len(self.targets) == 0:
            return False
        if np.isclose(angle_degrees, 0.0):
            return False

        support = self.operation_support(TransformOperation.ROTATE)
        if not support.supported:
            raise ValueError(support.reason)

        old_values: list[tuple[Artist, float]] = []
        for target in self.targets:
            value = self._rotatable_value(target.target)
            if value is not None:
                old_values.append((target.target, value))

        new_values = [(target, value + angle_degrees) for target, value in old_values]

        def apply(values: list[tuple[Artist, float]]) -> None:
            for target, value in values:
                self._set_rotation_value(target, value)
            self.figure.canvas.draw()
            self.update_extent()
            self.update_selection_rectangles()

        def undo() -> None:
            apply(old_values)

        def redo() -> None:
            apply(new_values)

        redo()
        self.figure.change_tracker.addEdit([undo, redo, "Rotate"])
        signal = getattr(self.figure.signals, "figure_selection_property_changed", None)
        if signal is not None:
            signal.emit()
        self.figure.signals.figure_selection_moved.emit()
        return True

    def delete_targets(self):
        """Delete all selected targets."""
        if len(self.targets) == 0:
            return
        for target in self.targets[::-1]:
            self.figure.change_tracker.removeElement(target.target)
        self.figure.canvas.draw()

    def update_selection_rectangles(self, use_previous_offset=False):
        """update the selection visualisation"""
        if len(self.targets) == 0:
            return
        if 0:
            for index, target in enumerate(self.targets):
                new_points = np.array(target.get_positions())
                for i in range(2):
                    rect = self.targets_rects[index * 2 + i]
                    rect.set_xy(new_points[0])
                    rect.set_width(new_points[1][0] - new_points[0][0])
                    rect.set_height(new_points[1][1] - new_points[0][1])
        else:
            for index, target in enumerate(self.targets):
                new_points = None
                if use_previous_offset:
                    new_points = getattr(self, "move_current_selection_points", {}).get(
                        id(target.target)
                    )
                if new_points is None:
                    new_points = np.array(target.get_selection_points())
                if new_points.ndim != 2 or not len(new_points):
                    for i in range(2):
                        self.targets_rects[index * 2 + i].setRect(-100, -100, 0, 0)
                    continue
                x0, y0, x1, y1 = (
                    np.min(new_points[:, 0]),
                    np.min(new_points[:, 1]),
                    np.max(new_points[:, 0]),
                    np.max(new_points[:, 1]),
                )
                w0, h0 = x1 - x0, y1 - y0
                for i in range(2):
                    rect = self.targets_rects[index * 2 + i]
                    rect.setRect(x0, y0, w0, h0)

    def remove_target(self, target: Artist):
        """remove an artist from the current selection"""
        targets_non_wrapped = [t.target for t in self.targets]
        if target not in targets_non_wrapped:
            return
        index = targets_non_wrapped.index(target)
        self._clear_preview(self.targets[index])
        self.targets.pop(index)
        rect1 = self.targets_rects.pop(index * 2)
        rect2 = self.targets_rects.pop(index * 2)
        rect1.scene().removeItem(rect1)
        rect2.scene().removeItem(rect2)
        # self.figure.patches.remove(rect1)
        # self.figure.patches.remove(rect2)
        if len(self.targets) == 0:
            self.clear_targets()
        else:
            self.update_extent()

    def update_grabber(self):
        """update the position of the grabber elements"""
        if self.do_target_scale():
            for grabber in self.grabbers:
                grabber.updatePos()
        else:
            for grabber in self.grabbers:
                grabber.set_xy((-100, -100))
        if self.rotation_handle_supported():
            self.rotation_grabber.updatePos()
        else:
            self.rotation_grabber.hide()

    def hide_grabber(self):
        """hide the grabber elements"""
        for grabber in self.grabbers:
            grabber.set_xy((-100, -100))
        self.rotation_grabber.hide()

    def clear_targets(self):
        """remove all elements from the selection"""
        self.clear_move_previews()
        for rect in self.targets_rects:
            self.graphics_scene.scene().removeItem(rect)
            # self.figure.patches.remove(rect)
        self.targets_rects = []
        self.targets = []

        self.hide_grabber()

    def do_target_scale(self) -> bool:
        """Only expose resize handles when every selected artist can scale."""
        return self.operation_support(TransformOperation.RESIZE_GEOMETRY).supported

    def do_change_aspect_ratio(self) -> bool:
        """if any of the element sin the selection wants to perserve its aspect ratio"""
        return np.any([target.fixed_aspect for target in self.targets])

    def width(self) -> float:
        """the width of the current selection"""
        return (self.p2 - self.p1)[0]

    def height(self) -> float:
        """the height of the current selection"""
        return (self.p2 - self.p1)[1]

    def size(self) -> (float, float):
        """the size of the current selection (width and height)"""
        return self.p2 - self.p1

    def selection_bounds(self) -> np.ndarray:
        """Return the exact visible bounds shared by selection UI operations."""

        point_sets = []
        for target in self.targets:
            points = np.asarray(target.get_selection_points(), dtype=float)
            if points.ndim != 2 or points.shape[1] < 2:
                continue
            points = points[np.all(np.isfinite(points[:, :2]), axis=1), :2]
            if len(points):
                point_sets.append(points)
        if not point_sets:
            raise ValueError("No selected object has finite visible geometry")
        return self._bounds_from_points(point_sets)

    def set_reference_point(self, point: Sequence[float]) -> tuple[float, float]:
        """Set the normalized transform-panel anchor without mutating the figure."""

        point = tuple(float(value) for value in point)
        if len(point) != 2 or not np.all(np.isfinite(point)):
            raise ValueError("Reference point must contain two finite values")
        if any(value not in (0.0, 0.5, 1.0) for value in point):
            raise ValueError("Reference point values must use the 3x3 transform grid")
        self.reference_point = point
        return point

    def reference_position(self) -> np.ndarray:
        """Resolve the normalized reference point in display coordinates."""

        bounds = self.selection_bounds()
        low = bounds[:2]
        size = bounds[2:] - low
        return low + np.asarray(self.reference_point, dtype=float) * size

    def translate_reference_to(self, display_position: Sequence[float]) -> bool:
        """Move the whole selection so its active reference reaches a display point."""

        if not self.targets:
            return False
        support = self.operation_support(TransformOperation.TRANSLATE)
        if not support.supported:
            raise ValueError(support.reason)
        desired = np.asarray(display_position, dtype=float)
        if desired.shape != (2,) or not np.all(np.isfinite(desired)):
            raise ValueError("Reference position must contain two finite values")
        delta = desired - self.reference_position()
        if np.allclose(delta, 0.0):
            return False

        for target in self.targets:
            target.preflight_rigid_visible_translation(delta)
        self.start_move()
        try:
            for target in self.targets:
                target.translate(delta)
            self.update_extent()
            self.has_moved = True
            self.end_move("Change position")
        except Exception:
            self._restore_move_start()
            self.has_moved = False
            self.end_move("Change position")
            raise
        signal = getattr(self.figure.signals, "figure_selection_property_changed", None)
        if signal is not None:
            signal.emit()
        self.figure.canvas.draw()
        self.update_selection_rectangles()
        return True

    def resize_selection_to(
        self,
        display_size: Sequence[float],
        *,
        keep_aspect_ratio: bool = False,
        changed_axis: int | None = None,
    ) -> bool:
        """Resize visible selection bounds about the active reference point."""

        if not self.targets:
            return False
        support = self.operation_support(TransformOperation.RESIZE_GEOMETRY)
        if not support.supported:
            raise ValueError(support.reason)
        desired = np.asarray(display_size, dtype=float)
        if desired.shape != (2,) or not np.all(np.isfinite(desired)):
            raise ValueError("Selection size must contain two finite values")

        bounds = self.selection_bounds()
        current = bounds[2:] - bounds[:2]
        if np.any(current <= np.finfo(float).eps):
            raise ValueError("Selection has a zero visible width or height")
        if np.any(desired <= 0.0):
            raise ValueError("Selection width and height must be positive")

        if keep_aspect_ratio:
            if changed_axis in (0, 1):
                scale = desired[changed_axis] / current[changed_axis]
            else:
                scale = min(desired[0] / current[0], desired[1] / current[1])
            desired = current * scale
        scale_x, scale_y = desired / current
        if np.allclose((scale_x, scale_y), (1.0, 1.0)):
            return False

        pivot_x, pivot_y = self.reference_position()
        matrix = np.array(
            [
                [scale_x, 0.0, pivot_x * (1.0 - scale_x)],
                [0.0, scale_y, pivot_y * (1.0 - scale_y)],
                [0.0, 0.0, 1.0],
            ],
            dtype=float,
        )

        planned_selection = [
            target.preflight_rigid_visible_resize(matrix) for target in self.targets
        ]
        expected_bounds = self._bounds_from_points(
            [
                self.apply_transform(
                    matrix,
                    np.array(
                        [[bounds[0], bounds[1]], [bounds[2], bounds[3]]],
                        dtype=float,
                    ),
                )
            ]
        )
        planned_bounds = self._bounds_from_points(planned_selection)
        if not np.allclose(
            planned_bounds, expected_bounds, atol=0.25, rtol=0.0
        ):
            raise UnsupportedArtistError(
                "Numeric resize cannot reach the requested visible bounds because "
                "an active clip would change the selection envelope"
            )

        self.start_move()
        try:
            for target in self.targets:
                target.resize(matrix)
            self.update_extent()
            self.has_moved = True
            self.end_move("Resize")
        except Exception:
            self._restore_move_start()
            self.has_moved = False
            self.end_move("Resize")
            raise
        signal = getattr(self.figure.signals, "figure_selection_property_changed", None)
        if signal is not None:
            signal.emit()
        self.figure.canvas.draw()
        self.update_selection_rectangles()
        return True

    def get_trans_matrix(self):
        """the transformation matrix for the current displacement and scaling of the selection"""
        x, y = self.p1
        w, h = self.size()
        return np.array([[w, 0, x], [0, h, y], [0, 0, 1]], dtype=float)

    def get_inv_trans_matrix(self):
        """the inverse transformation for the current displacement and scaling of the selection"""
        x, y = self.p1
        w, h = self.size()
        if not np.isfinite(w) or abs(w) < np.finfo(float).eps:
            w = np.copysign(np.finfo(float).eps, w if w else 1.0)
        if not np.isfinite(h) or abs(h) < np.finfo(float).eps:
            h = np.copysign(np.finfo(float).eps, h if h else 1.0)
        return np.array(
            [[1.0 / w, 0, -x / w], [0, 1.0 / h, -y / h], [0, 0, 1]], dtype=float
        )

    def transform(self, pos: Sequence) -> np.ndarray:
        """apply the current transformation to a point"""
        return np.dot(self.get_trans_matrix(), [pos[0], pos[1], 1.0])

    def inv_transform(self, pos: Sequence) -> np.ndarray:
        """apply the inverse current transformation to a point"""
        return np.dot(self.get_inv_trans_matrix(), [pos[0], pos[1], 1.0])

    def get_pos(self, pos: Sequence) -> np.ndarray:
        """transform a point"""
        return self.transform(pos)

    def get_save_point(self, targets: Iterable[TargetWrapper] = None) -> callable:
        """gather the current positions in a restore point for the undo function"""
        selected_targets = [target.target for target in self.targets]
        wrapped_targets = self._unique_wrappers(targets or self.targets)
        restore_targets = [target.target for target in wrapped_targets]
        states = [target.get_restore_state() for target in wrapped_targets]
        tracker = getattr(self.figure, "change_tracker", None)
        capture = getattr(tracker, "capture_recording_state", None)
        recording_state = capture() if capture is not None else None

        def undo():
            self.clear_targets()
            for target, state in zip(restore_targets, states):
                target = TargetWrapper(target)
                target.restore_state(
                    state, record_changes=recording_state is None
                )
            restore_recording = getattr(tracker, "restore_recording_state", None)
            if recording_state is not None and restore_recording is not None:
                restore_recording(recording_state)
            for target in selected_targets:
                self.add_target(target, update=False)
            if self.targets:
                self.update_extent()

        return undo

    def start_move(self, save_targets: Iterable[TargetWrapper] = None):
        """start to move a grabber"""
        self.start_p1 = self.p1.copy()
        self.start_p2 = self.p2.copy()
        self.start_inv_transform = self.get_inv_trans_matrix()
        self.hide_grabber()
        self.has_moved = False
        self.defer_current_move = bool(self.defer_artist_updates)
        self.save_targets = self._unique_wrappers(save_targets or self.targets)
        for target in self._unique_wrappers(list(self.targets) + self.save_targets):
            target.refresh_offset()
        self.move_start_positions = {
            id(target.target): np.array(target.get_positions(), dtype=float)
            for target in self.targets
        }
        self.move_start_selection_points = {
            id(target.target): np.array(target.get_selection_points(), dtype=float)
            for target in self.targets
        }
        self.move_current_positions = {}
        self.move_current_selection_points = {}
        self.move_start_states = {
            id(target.target): target.get_restore_state()
            for target in self.save_targets
        }
        tracker = getattr(self.figure, "change_tracker", None)
        capture = getattr(tracker, "capture_recording_state", None)
        self.move_start_tracker_state = capture() if capture is not None else None

        self.store_start = self.get_save_point(self.save_targets)

    @staticmethod
    def _clear_preview(target: TargetWrapper):
        for attribute in (
            "_pylustrator_preview_positions",
            "_pylustrator_preview_selection_points",
        ):
            try:
                delattr(target.target, attribute)
            except AttributeError:
                pass
        setattr(target.target, "_pylustrator_cached_get_extend", None)

    def clear_move_previews(self):
        for target in self._unique_wrappers(getattr(self, "targets", [])):
            self._clear_preview(target)

    def _set_preview_positions(
        self,
        target: TargetWrapper,
        points: np.ndarray,
        selection_points: np.ndarray = None,
    ):
        target.target._pylustrator_preview_positions = [
            np.array(point, dtype=float).copy() for point in points
        ]
        if selection_points is not None:
            target.target._pylustrator_preview_selection_points = [
                np.array(point, dtype=float).copy() for point in selection_points
            ]
        setattr(target.target, "_pylustrator_cached_get_extend", None)

    def _commit_deferred_positions(self):
        if not getattr(self, "defer_current_move", False):
            return
        pending = []
        for target in self.targets:
            points = self.move_current_positions.get(id(target.target))
            if points is None:
                continue
            translation_delta = None
            start = self.move_start_positions.get(id(target.target))
            if start is not None and np.shape(start) == np.shape(points) and len(points):
                deltas = np.asarray(points, dtype=float) - np.asarray(start, dtype=float)
                if np.allclose(deltas, deltas[0]):
                    translation_delta = deltas[0]
            pending.append((target, points, translation_delta))

        # Preview geometry is already at the proposed destination. Clear every
        # preview before validating the display delta, otherwise adapters would
        # add the delta to the preview a second time. Preflight the complete
        # transaction before mutating any target.
        for target, _points, _translation_delta in pending:
            self._clear_preview(target)
        for target, _points, translation_delta in pending:
            if translation_delta is not None:
                target.preflight_translation(translation_delta)
        for target, points, _translation_delta in pending:
            target.set_positions(points)

    def _move_changed_semantically(self) -> bool:
        start_states = getattr(self, "move_start_states", {})
        for target in self.save_targets:
            before = start_states.get(id(target.target))
            if before is None or not semantic_equal(before, target.get_restore_state()):
                return True
        return False

    def _restore_move_start(self) -> None:
        start_states = getattr(self, "move_start_states", {})
        for target in reversed(self.save_targets):
            state = start_states.get(id(target.target))
            if state is not None:
                target.restore_state(state, record_changes=False)
        tracker_state = getattr(self, "move_start_tracker_state", None)
        tracker = getattr(self.figure, "change_tracker", None)
        restore = getattr(tracker, "restore_recording_state", None)
        if tracker_state is not None and restore is not None:
            restore(tracker_state)
        self.clear_move_previews()
        if self.targets:
            self.update_extent()
            self.update_selection_rectangles()

    def end_move(self, edit_name: str = "Move", coalesce_key: str = None):
        """a grabber move stopped"""
        if self.has_moved is True:
            try:
                self._commit_deferred_positions()
            except Exception:
                self._restore_move_start()
                self.has_moved = False
                raise
            if not self._move_changed_semantically():
                self._restore_move_start()
                self.has_moved = False
        else:
            self.clear_move_previews()
        self.update_grabber()

        self.store_end = self.get_save_point(self.save_targets)
        if self.has_moved is True:
            self.figure.signals.figure_selection_moved.emit()
            edit = [self.store_start, self.store_end, edit_name]
            if coalesce_key is not None:
                edit.append(
                    {
                        "coalesce_key": coalesce_key,
                        "targets": tuple(id(target.target) for target in self.save_targets),
                        "timestamp": time.monotonic(),
                    }
                )
            self.figure.change_tracker.addEdit(edit)
            if getattr(self, "defer_current_move", False):
                canvas = getattr(self.figure, "canvas", None)
                if hasattr(canvas, "schedule_draw"):
                    canvas.schedule_draw()
                elif hasattr(canvas, "draw_idle"):
                    canvas.draw_idle()
        self.move_start_positions = {}
        self.move_start_selection_points = {}
        self.move_current_positions = {}
        self.move_current_selection_points = {}
        self.move_start_states = {}
        self.move_start_tracker_state = None
        self.defer_current_move = False

    def addOffset(self, pos: Sequence, dir: int, keep_aspect_ratio: bool = True):
        """move the whole selection (e.g. for the use of the arrow keys)"""
        pos = list(pos)
        whole_object = bool(
            dir & DIR_X0 and dir & DIR_X1 and dir & DIR_Y0 and dir & DIR_Y1
        )
        operation = (
            TransformOperation.TRANSLATE
            if whole_object
            else TransformOperation.RESIZE_GEOMETRY
        )
        support = self.operation_support(operation)
        if not support.supported:
            raise ValueError(support.reason)
        if (keep_aspect_ratio or self.do_change_aspect_ratio()) and not whole_object:
            if (dir & DIR_X0 and dir & DIR_Y0) or (dir & DIR_X1 and dir & DIR_Y1):
                dx = pos[1] * self.width() / self.height()
                dy = pos[0] * self.height() / self.width()
                if abs(dx) < abs(dy):
                    pos[0] = dx
                else:
                    pos[1] = dy
            elif (dir & DIR_X0 and dir & DIR_Y1) or (dir & DIR_X1 and dir & DIR_Y0):
                dx = -pos[1] * self.width() / self.height()
                dy = -pos[0] * self.height() / self.width()
                if abs(dx) < abs(dy):
                    pos[0] = dx
                else:
                    pos[1] = dy
            elif dir & DIR_X0 or dir & DIR_X1:
                dy = pos[0] * self.height() / self.width()
                if dir & DIR_X0:
                    self.p1[1] = self.start_p1[1] + dy / 2
                    self.p2[1] = self.start_p2[1] - dy / 2
                else:
                    self.p1[1] = self.start_p1[1] - dy / 2
                    self.p2[1] = self.start_p2[1] + dy / 2
            elif dir & DIR_Y0 or dir & DIR_Y1:
                dx = pos[1] * self.width() / self.height()
                if dir & DIR_Y0:
                    self.p1[0] = self.start_p1[0] + dx / 2
                    self.p2[0] = self.start_p2[0] - dx / 2
                else:
                    self.p1[0] = self.start_p1[0] - dx / 2
                    self.p2[0] = self.start_p2[0] + dx / 2

        if dir & DIR_X0:
            self.p1[0] = self.start_p1[0] + pos[0]
        if dir & DIR_X1:
            self.p2[0] = self.start_p2[0] + pos[0]
        if dir & DIR_Y0:
            self.p1[1] = self.start_p1[1] + pos[1]
        if dir & DIR_Y1:
            self.p2[1] = self.start_p2[1] + pos[1]

        start_inv_transform = getattr(
            self, "start_inv_transform", self.get_inv_trans_matrix()
        )
        transform = np.dot(self.get_trans_matrix(), start_inv_transform)
        start_positions = getattr(self, "move_start_positions", {})
        start_selection_points = getattr(self, "move_start_selection_points", {})
        self.move_current_positions = {}
        self.move_current_selection_points = {}
        for target in self.targets:
            points = start_positions.get(id(target.target))
            if points is None:
                points = np.array(target.get_positions(), dtype=float)
            selection_points = start_selection_points.get(id(target.target))
            if selection_points is None:
                selection_points = np.array(target.get_selection_points(), dtype=float)
            if whole_object:
                points = self.apply_transform(transform, points)
                selection_points = self.apply_transform(transform, selection_points)
            else:
                resize_selection_points = target.preview_resize_selection_points(
                    transform,
                    control_points=points,
                    selection_points=selection_points,
                )
                points = target.preview_resize_control_points(
                    transform,
                    control_points=points,
                    selection_points=selection_points,
                )
                selection_points = resize_selection_points
            self.move_current_positions[id(target.target)] = points
            if getattr(self, "defer_current_move", False):
                self._set_preview_positions(target, points, selection_points)
                # TargetWrapper applies the active paint clip to the preview.
                # Store the same envelope that the user actually sees so the
                # preview indicator and committed selection cannot diverge.
                selection_points = np.asarray(
                    target.get_selection_points(), dtype=float
                )
            else:
                target.set_positions(points)
                selection_points = np.asarray(
                    target.get_selection_points(), dtype=float
                )
            self.move_current_selection_points[id(target.target)] = selection_points

        self.update_selection_rectangles(use_previous_offset=True)
        # for rect in self.targets_rects:
        #    self.transform_target(transform, TargetWrapper(rect))

    def move(
        self,
        pos: Sequence[float],
        dir: int,
        snaps: Sequence[SnapBase],
        keep_aspect_ratio: bool = False,
        ignore_snaps: bool = False,
        constrain_direction: bool = False,
    ):
        """called from a grabber to move the selection."""
        if constrain_direction:
            pos = _constrain_to_cardinal_direction(pos, dir)
        self.addOffset(pos, dir, keep_aspect_ratio)
        self.has_moved = True

        if not ignore_snaps:
            offx, offy = checkSnaps(snaps)
            adjusted_pos = (pos[0] - offx, pos[1] - offy)
            if constrain_direction:
                adjusted_pos = _constrain_to_cardinal_direction(adjusted_pos, dir)
            self.addOffset(adjusted_pos, dir, keep_aspect_ratio)

            offx, offy = checkSnaps(self.snaps)

        checkSnapsActive(snaps)

    def apply_transform(self, transform: np.ndarray, point: Sequence[float]):
        """apply the given transformation to a point"""
        point = np.array(point)
        point = np.hstack((point, np.ones((point.shape[0], 1)))).T
        return np.dot(transform, point)[:2].T

    def transform_target(self, transform: np.ndarray, target: TargetWrapper):
        """transform the position of an artist."""
        target.apply_display_transform(transform)

    def keyPressEvent(self, event: KeyEvent):
        """when a key is pressed. Arrow keys move the selection, Pageup/down movein z"""
        def schedule_draw():
            canvas = self.figure.canvas
            if hasattr(canvas, "schedule_draw"):
                canvas.schedule_draw()
            else:
                canvas.draw_idle()

        # if not self.selected:
        #    return
        # move last axis in z order
        if event.key == "pagedown":
            self.figure.figure_dragger.change_selection_zorder("backward")
        if event.key == "pageup":
            self.figure.figure_dragger.change_selection_zorder("forward")
        if event.key == "left":
            self.start_move()
            self.addOffset((-1, 0), self.dir)
            self.has_moved = True
            self.end_move("Nudge", coalesce_key="nudge")
            schedule_draw()
        if event.key == "right":
            self.start_move()
            self.addOffset((+1, 0), self.dir)
            self.has_moved = True
            self.end_move("Nudge", coalesce_key="nudge")
            schedule_draw()
        if event.key == "down":
            self.start_move()
            self.addOffset((0, -1), self.dir)
            self.has_moved = True
            self.end_move("Nudge", coalesce_key="nudge")
            schedule_draw()
        if event.key == "up":
            self.start_move()
            self.addOffset((0, +1), self.dir)
            self.has_moved = True
            self.end_move("Nudge", coalesce_key="nudge")
            schedule_draw()
        if event.key in ["delete", "backspace"]:
            self.delete_targets()


class DragManager:
    """a class to manage the selection and the moving of artists in a figure"""

    selected_element = None
    grab_element = None

    def __init__(self, figure: Figure, no_save):
        self.figure = figure
        self.figure.figure_dragger = self
        self._selectable_artists = []
        self._selectable_artist_ids = set()
        self._uneditable_artists = []
        self._uneditable_artist_ids = set()
        self._interaction_artists = []
        self._interaction_artist_ids = set()
        self._selection_parent_by_id = {}
        self._draw_child_orders = {}
        self.editor_scene = EditorScene(
            figure, ownership_parent=self._draw_parent
        )
        self.marquee_select_containers_only = False
        self.marquee_start = None
        self.marquee_rect = None
        self.marquee_active = False
        self.marquee_additive = False
        self.marquee_click_element = None
        self._last_pick_blocked = False
        self.preselection_rect = None
        self.preselection_artist = None
        self._candidate_menu = None
        self.selection_kernel = SelectionKernel(
            parent_of=self._interaction_parent,
            is_group=self._interaction_is_group,
            label_of=self._interaction_label,
        )

        self.figure.canvas.mpl_disconnect(
            self.figure.canvas.manager.key_press_handler_id
        )

        self.activate()

        self.make_figure_draggable(self.figure)
        self.make_axes_draggable(self.figure.axes)
        self.editor_scene.restore_persisted_state()
        self._sync_editor_groups()
        self.selection = GrabbableRectangleSelection(figure, figure._pyl_scene)
        self.figure.selection = self.selection
        self.change_tracker = ChangeTracker(figure, no_save)
        self.figure.change_tracker = self.change_tracker

    def activate(self):
        """activate the interaction callbacks from the figure"""
        self.c3 = self.figure.canvas.mpl_connect(
            "button_release_event", self.button_release_event0
        )
        self.c2 = self.figure.canvas.mpl_connect(
            "button_press_event", self.button_press_event0
        )
        self.c4 = self.figure.canvas.mpl_connect(
            "key_press_event", self.key_press_event
        )
        self.c5 = self.figure.canvas.mpl_connect(
            "motion_notify_event", self.motion_notify_event0
        )
        self.c6 = self.figure.canvas.mpl_connect(
            "draw_event", self.invalidate_geometry_cache
        )

    def _draw_parent(self, artist: Artist) -> Artist | None:
        if isinstance(artist, EditorGroup):
            return artist.owner
        parent = getattr(self, "_selection_parent_by_id", {}).get(id(artist))
        if parent is None and isinstance(artist, SubFigure):
            parent = getattr(artist, "_parent", None)
        return parent

    def _ensure_editor_scene(self) -> EditorScene:
        scene = getattr(self, "editor_scene", None)
        if scene is None:
            scene = EditorScene(self.figure, ownership_parent=self._draw_parent)
            self.editor_scene = scene
        return scene

    def _interaction_parent(self, artist: Artist) -> Artist | None:
        """Return editor grouping independently from Matplotlib ownership."""

        return self._ensure_editor_scene().selection_parent(artist)

    def _interaction_is_group(self, artist: Artist) -> bool:
        return self._ensure_editor_scene().is_group(artist)

    @staticmethod
    def _interaction_label(artist: Artist) -> str:
        if isinstance(artist, EditorGroup):
            return artist.name
        if isinstance(artist, Legend):
            title = artist.get_title().get_text()
            return title or "Legend"
        label = getattr(artist, "get_label", lambda: "")()
        if isinstance(label, str) and label and not label.startswith("_"):
            return label
        return type(artist).__name__

    @property
    def selection_mode(self) -> SelectionMode:
        return self._ensure_selection_kernel().mode

    @property
    def isolation_breadcrumbs(self) -> tuple[str, ...]:
        return self._ensure_selection_kernel().breadcrumbs

    def _ensure_selection_kernel(self) -> SelectionKernel:
        kernel = getattr(self, "selection_kernel", None)
        if kernel is None:
            kernel = SelectionKernel(
                parent_of=self._interaction_parent,
                is_group=self._interaction_is_group,
                label_of=self._interaction_label,
            )
            self.selection_kernel = kernel
        return kernel

    def _update_interaction_controls(self) -> None:
        window = getattr(self.figure, "window", None)
        if window is not None and hasattr(window, "updateSelectionControls"):
            window.updateSelectionControls()

    def set_selection_mode(self, mode: SelectionMode | str) -> SelectionMode:
        result = self._ensure_selection_kernel().set_mode(mode)
        self._update_interaction_controls()
        return result

    def enter_isolation(self, element: Artist) -> bool:
        entered = self._ensure_selection_kernel().enter_isolation(element)
        if entered:
            self.selection.clear_targets()
            self.selected_element = None
            self.on_select(None, None)
            self._update_interaction_controls()
            self.figure.canvas.draw_idle()
        return entered

    def exit_isolation(self) -> Artist | None:
        exited = self._ensure_selection_kernel().exit_isolation()
        if exited is not None:
            self.select_element(exited)
            self._update_interaction_controls()
            self.figure.canvas.draw_idle()
        return exited

    def _selected_artists(self) -> list[Artist]:
        return [target.target for target in self.selection.targets]

    @staticmethod
    def _object_locator(artist: Artist) -> ObjectLocator | None:
        try:
            return ObjectLocator.from_artist(artist)
        except (TypeError, ValueError):
            return None

    def capture_interaction_state(self) -> InteractionState:
        selected = tuple(
            locator
            for artist in self._selected_artists()
            if (locator := self._object_locator(artist)) is not None
        )
        primary = (
            self._object_locator(self.selected_element)
            if self.selected_element is not None
            else None
        )
        scopes = tuple(
            locator
            for scope in self._ensure_selection_kernel().scopes
            if (locator := self._object_locator(scope.root)) is not None
        )
        return InteractionState(
            self.selection_mode.value, selected, primary, scopes
        )

    def restore_interaction_state(self, state: InteractionState) -> None:
        scene = self._ensure_editor_scene()
        kernel = self._ensure_selection_kernel()
        kernel.clear_isolation()
        kernel.set_mode(state.mode)
        for locator in state.scopes:
            root = locator.resolve(scene)
            if root is not None:
                kernel.enter_isolation(root)
        selected = [
            artist
            for locator in state.selected
            if (artist := locator.resolve(scene)) is not None
        ]
        primary = state.primary.resolve(scene) if state.primary is not None else None
        self.select_elements(selected, primary=primary)
        self._update_interaction_controls()

    def _notify_editor_scene_changed(self, owner: Artist = None) -> None:
        self._sync_editor_groups()
        signals = getattr(self.figure, "signals", None)
        signal = getattr(signals, "figure_element_child_created", None)
        if signal is not None:
            signal.emit(owner or self.figure)
        self.invalidate_geometry_cache()
        self.figure.canvas.draw_idle()

    def _sync_editor_groups(self) -> None:
        """Keep transient group nodes aligned with the persisted editor scene."""

        scene = self._ensure_editor_scene()
        active_groups = tuple(scene.groups.values())
        active_ids = {id(group) for group in active_groups}

        def remove_stale(items):
            return [
                item
                for item in items
                if not isinstance(item, EditorGroup) or id(item) in active_ids
            ]

        if hasattr(self, "_selectable_artists"):
            self._selectable_artists = remove_stale(self._selectable_artists)
            self._selectable_artist_ids = {
                id(item) for item in self._selectable_artists
            }
        if hasattr(self, "_interaction_artists"):
            self._interaction_artists = remove_stale(self._interaction_artists)
            self._interaction_artist_ids = {
                id(item) for item in self._interaction_artists
            }
        if hasattr(self, "_uneditable_artists"):
            self._uneditable_artists = remove_stale(self._uneditable_artists)
            self._uneditable_artist_ids = {
                id(item) for item in self._uneditable_artists
            }
        for key, parent in list(getattr(self, "_selection_parent_by_id", {}).items()):
            if isinstance(parent, EditorGroup) and id(parent) not in active_ids:
                self._selection_parent_by_id.pop(key, None)
        for group in active_groups:
            self.make_draggable(group, group.owner)

    def _hide_preselection(self) -> None:
        rect = getattr(self, "preselection_rect", None)
        if rect is not None:
            rect.setVisible(False)
        self.preselection_artist = None

    def _ensure_preselection_rect(self):
        current = getattr(self, "preselection_rect", None)
        if current is not None:
            return current
        rect = QtWidgets.QGraphicsRectItem(0, 0, 0, 0, self.figure._pyl_scene)
        pen = QtGui.QPen(QtGui.QColor("#00A8E8"), 2)
        pen.setStyle(QtCore.Qt.DotLine)
        rect.setPen(pen)
        rect.setBrush(QtGui.QBrush(QtCore.Qt.NoBrush))
        rect.setZValue(898)
        self.preselection_rect = rect
        return rect

    def update_preselection(self, event: MouseEvent) -> Artist | None:
        if event.x is None or event.y is None or getattr(self, "grab_element", None):
            self._hide_preselection()
            return None
        target = self._ensure_selection_kernel().pick(self.get_hit_stack(event))
        if target is None:
            self._hide_preselection()
            return None
        try:
            points = np.asarray(TargetWrapper(target).get_selection_points(), dtype=float)
        except (AttributeError, TypeError, ValueError, RuntimeError):
            self._hide_preselection()
            return None
        if points.size == 0:
            self._hide_preselection()
            return None
        x0, y0 = np.min(points[:, :2], axis=0)
        x1, y1 = np.max(points[:, :2], axis=0)
        rect = self._ensure_preselection_rect()
        rect.setRect(float(x0), float(y0), float(x1 - x0), float(y1 - y0))
        rect.setVisible(True)
        rect.setToolTip(self._interaction_label(target))
        self.preselection_artist = target
        return target

    def _apply_editor_state(self, state: dict) -> None:
        self.selection.clear_targets()
        self.selected_element = None
        self._ensure_selection_kernel().clear_isolation()
        scene = self._ensure_editor_scene()
        scene.apply_state(state)
        scene.record_state()
        self._notify_editor_scene_changed()

    def group_selection(self, name: str = None) -> EditorGroup:
        members = self._selected_artists()
        scene = self._ensure_editor_scene()
        before = scene.export_state()
        interaction_before = self.capture_interaction_state()
        group = scene.create_group(members, name=name)
        after = scene.export_state()
        scene.record_state()
        self.select_element(group)
        interaction_after = self.capture_interaction_state()

        def undo():
            self._apply_editor_state(before)
            self.restore_interaction_state(interaction_before)

        def redo():
            self._apply_editor_state(after)
            self.restore_interaction_state(interaction_after)

        self.figure.change_tracker.addEdit([undo, redo, "Group"])
        self._notify_editor_scene_changed(group.owner)
        return group

    def ungroup_selection(self) -> list[Artist]:
        groups = [
            artist for artist in self._selected_artists() if isinstance(artist, EditorGroup)
        ]
        if not groups:
            raise ValueError("Select an editor group to ungroup.")
        scene = self._ensure_editor_scene()
        before = scene.export_state()
        interaction_before = self.capture_interaction_state()
        members: list[Artist] = []
        for group in groups:
            members.extend(scene.remove_group(group))
        after = scene.export_state()
        scene.record_state()
        self.select_elements(members, primary=members[-1] if members else None)
        interaction_after = self.capture_interaction_state()

        def undo():
            self._apply_editor_state(before)
            self.restore_interaction_state(interaction_before)

        def redo():
            self._apply_editor_state(after)
            self.restore_interaction_state(interaction_after)

        self.figure.change_tracker.addEdit([undo, redo, "Ungroup"])
        self._notify_editor_scene_changed()
        return members

    def set_selection_locked(self, locked: bool = True) -> bool:
        artists = self._selected_artists()
        if not artists:
            return False
        scene = self._ensure_editor_scene()
        before = scene.export_state()
        interaction_before = self.capture_interaction_state()
        if not scene.set_locked(artists, locked):
            return False
        after = scene.export_state()
        scene.record_state()
        if locked:
            self.select_element(None)
        interaction_after = self.capture_interaction_state()

        def undo():
            self._apply_editor_state(before)
            self.restore_interaction_state(interaction_before)

        def redo():
            self._apply_editor_state(after)
            self.restore_interaction_state(interaction_after)

        self.figure.change_tracker.addEdit(
            [undo, redo, "Lock" if locked else "Unlock"]
        )
        self._notify_editor_scene_changed()
        return True

    def unlock_all(self) -> bool:
        scene = self._ensure_editor_scene()
        before = scene.export_state()
        interaction_before = self.capture_interaction_state()
        artists = [
            artist
            for key in list(scene._locked_ids)
            if (artist := scene._known_artists.get(key)) is not None
        ]
        if not scene.set_locked(artists, False):
            return False
        after = scene.export_state()
        scene.record_state()
        interaction_after = self.capture_interaction_state()

        def undo():
            self._apply_editor_state(before)
            self.restore_interaction_state(interaction_before)

        def redo():
            self._apply_editor_state(after)
            self.restore_interaction_state(interaction_after)

        self.figure.change_tracker.addEdit([undo, redo, "Unlock All"])
        self._notify_editor_scene_changed()
        return True

    def set_selection_visible(self, visible: bool) -> bool:
        artists = self._selected_artists()
        if not artists:
            return False
        scene = self._ensure_editor_scene()
        before = scene.export_state()
        interaction_before = self.capture_interaction_state()
        if not scene.set_visible(artists, visible):
            return False
        after = scene.export_state()
        scene.record_state()
        if not visible:
            self.select_element(None)
        interaction_after = self.capture_interaction_state()

        def undo():
            self._apply_editor_state(before)
            self.restore_interaction_state(interaction_before)

        def redo():
            self._apply_editor_state(after)
            self.restore_interaction_state(interaction_after)

        self.figure.change_tracker.addEdit(
            [undo, redo, "Show" if visible else "Hide"]
        )
        self._notify_editor_scene_changed()
        return True

    def show_all(self) -> bool:
        scene = self._ensure_editor_scene()
        before = scene.export_state()
        interaction_before = self.capture_interaction_state()
        artists = [
            artist
            for key in list(scene._explicitly_hidden_ids)
            if (artist := scene._known_artists.get(key)) is not None
        ]
        if not scene.set_visible(artists, True):
            return False
        after = scene.export_state()
        scene.record_state()
        interaction_after = self.capture_interaction_state()

        def undo():
            self._apply_editor_state(before)
            self.restore_interaction_state(interaction_before)

        def redo():
            self._apply_editor_state(after)
            self.restore_interaction_state(interaction_after)

        self.figure.change_tracker.addEdit([undo, redo, "Show All"])
        self._notify_editor_scene_changed()
        return True

    @staticmethod
    def _zorder_leaves(artists: Iterable[Artist]) -> list[Artist]:
        leaves: list[Artist] = []
        seen: set[int] = set()

        def add(artist: Artist):
            if isinstance(artist, EditorGroup):
                for member in artist.members:
                    add(member)
            elif id(artist) not in seen:
                seen.add(id(artist))
                leaves.append(artist)

        for artist in artists:
            add(artist)
        return leaves

    def change_selection_zorder(self, mode: str) -> bool:
        leaves = self._zorder_leaves(self._selected_artists())
        if not leaves:
            return False
        modes = {"forward", "backward", "front", "back"}
        if mode not in modes:
            raise ValueError(f"Unknown z-order action: {mode}")
        old_values = [(artist, float(artist.get_zorder())) for artist in leaves]
        selected_ids = {id(artist) for artist in leaves}
        others = [
            artist
            for artist in getattr(self, "_selectable_artists", [])
            if id(artist) not in selected_ids and artist.get_visible()
        ]
        if mode == "forward":
            delta = 1.0
        elif mode == "backward":
            delta = -1.0
        elif mode == "front":
            target = max((float(artist.get_zorder()) for artist in others), default=0.0) + 1
            delta = target - max(value for _artist, value in old_values)
        else:
            target = min((float(artist.get_zorder()) for artist in others), default=0.0) - 1
            delta = target - min(value for _artist, value in old_values)
        new_values = [(artist, value + delta) for artist, value in old_values]

        def apply(values):
            for artist, value in values:
                artist.set_zorder(value)
                self.figure.change_tracker.addChange(
                    artist, f".set_zorder({value!r})"
                )
            self.invalidate_geometry_cache()
            self.figure.canvas.draw_idle()

        def undo():
            apply(old_values)

        def redo():
            apply(new_values)

        redo()
        self.figure.change_tracker.addEdit([undo, redo, "Change stacking order"])
        return True

    def invalidate_geometry_cache(self, _event=None):
        """Drop visible-bound caches after any render/transform change."""
        for artist in getattr(self, "_selectable_artists", []):
            setattr(artist, "_pylustrator_cached_get_extend", None)

    def deactivate(self):
        """deactivate the interaction callbacks from the figure"""
        self.figure.canvas.mpl_disconnect(self.c3)
        self.figure.canvas.mpl_disconnect(self.c2)
        self.figure.canvas.mpl_disconnect(self.c4)
        self.figure.canvas.mpl_disconnect(self.c5)
        self.figure.canvas.mpl_disconnect(self.c6)

        self.selection.clear_targets()
        self.selected_element = None
        self.on_select(None, None)
        self.figure.canvas.draw()

    def make_draggable(self, target: Artist, parent: Artist = None):
        """make an artist draggable"""
        if getattr(self, "figure", None) is None:
            self.figure = getattr(target, "figure", None)
        if not hasattr(self, "_selectable_artists"):
            self._selectable_artists = []
            self._selectable_artist_ids = set()
            self._uneditable_artists = []
            self._uneditable_artist_ids = set()
            self._interaction_artists = []
            self._interaction_artist_ids = set()
            self._selection_parent_by_id = {}
            self._draw_child_orders = {}
        if not hasattr(self, "_uneditable_artists"):
            self._uneditable_artists = []
            self._uneditable_artist_ids = set()
        if not hasattr(self, "_interaction_artists"):
            self._interaction_artists = []
            self._interaction_artist_ids = set()
        if parent is None:
            if isinstance(target, Legend):
                parent = getattr(target, "parent", None)
            if parent is None:
                parent = getattr(target, "axes", None)
            if parent is target or parent is None:
                parent = getattr(target, "figure", None)
        if parent is not None and parent is not target:
            self._selection_parent_by_id[id(target)] = parent
        self._ensure_editor_scene().register_artist(target)
        if id(target) not in self._interaction_artist_ids:
            self._interaction_artists.append(target)
            self._interaction_artist_ids.add(id(target))
            self._draw_child_orders = {}
        if not TargetWrapper.supports_target(target):
            if (
                id(target) not in self._selectable_artist_ids
                and id(target) not in self._uneditable_artist_ids
            ):
                self._uneditable_artists.append(target)
                self._uneditable_artist_ids.add(id(target))
            return False
        if id(target) not in self._selectable_artist_ids:
            self._selectable_artists.append(target)
            self._selectable_artist_ids.add(id(target))
        target._pylustrator_explicitly_editable = True
        if not target.pickable():
            target.set_picker(True)
        if isinstance(target, Text):
            add_text_default(target)
        if isinstance(target, Legend):
            for handle in target.legend_handles:
                self.make_draggable(handle, target)
            for text in target.get_texts():
                self.make_draggable(text, target)
            title = target.get_title()
            if title is not None:
                self.make_draggable(title, target)
        return True

    def make_axes_draggable(self, axes: list[Axes], parent: Artist = None) -> None:
        for index, ax in enumerate(axes):
            ax.set_picker(True)
            leg = ax.get_legend()
            if leg:
                self.make_draggable(leg, ax)
            for artist in ax.artists:
                self.make_draggable(artist, ax)
            for text in ax.texts:
                self.make_draggable(text, ax)
            for attribute_name in ["title", "_left_title", "_right_title"]:
                text = getattr(ax, attribute_name, None)
                if text is not None:
                    self.make_draggable(text, ax)
            for patch in ax.patches:
                self.make_draggable(patch, ax)
            for line in ax.lines:
                self.make_draggable(line, ax)
            for collection in ax.collections:
                self.make_draggable(collection, ax)
            for image in ax.images:
                self.make_draggable(image, ax)
            self.make_draggable(ax.xaxis.get_label(), ax)
            self.make_draggable(ax.yaxis.get_label(), ax)
            self.make_draggable(ax, parent or ax.figure or self.figure)
            self.make_axes_draggable(ax.child_axes, parent=ax)

    def make_figure_draggable(self, fig: Figure | SubFigure) -> None:
        for artist in fig.artists:
            self.make_draggable(artist, fig)
        for text in fig.texts:
            self.make_draggable(text, fig)
        for patch in fig.patches:
            self.make_draggable(patch, fig)
        for leg in iter_figure_legends(fig):
            self.make_draggable(leg)
        for subfig in fig.subfigs:
            self.make_figure_draggable(subfig)

    def iter_selectable_artists(
        self, element: Artist = None, seen: set[int] = None
    ) -> Iterable[Artist]:
        if element is None and hasattr(self, "_selectable_artists"):
            for artist in self._selectable_artists:
                if self._is_pick_candidate(artist, explicit=True):
                    yield artist
            return
        if element is None:
            element = self.figure
        if seen is None:
            seen = set()

        for child, explicit in iter_artist_children(element):
            key = id(child)
            if key in seen:
                continue
            seen.add(key)
            if self._is_pick_candidate(child, explicit=explicit):
                yield child
            yield from self.iter_selectable_artists(child, seen)

    def _is_pick_candidate(
        self, child: Artist, event: MouseEvent = None, explicit: bool = False
    ) -> bool:
        if not self._is_artist_attached(child):
            return False
        if not child.get_visible():
            return False
        scene = self._ensure_editor_scene()
        if scene.is_locked(child) or scene.is_explicitly_hidden(child):
            return False
        if isinstance(child, Text) and child.get_text() == "":
            return False
        if _is_internal_label(child, explicit):
            return False
        if not (child.pickable() or isinstance(child, GrabberGeneric)):
            return False
        if event is not None and not self._is_interaction_hit(child, event):
            return False
        if isinstance(child, GrabberGeneric):
            return True
        if event is None:
            return self._artist_has_selection_geometry(child)
        return self._resolve_selectable_artist(child) is not None

    def _is_interaction_hit(self, artist: Artist, event: MouseEvent) -> bool:
        """Return whether an explicitly discovered foreground artist was hit.

        Unsupported foreground objects must participate in hit ordering even
        though they cannot become selection targets.  Otherwise a click passes
        through them and selects a containing Axes, recreating the dangerous
        "clicked child, moved parent" behaviour through a different code path.
        """
        if not self._is_artist_attached(artist) or not artist.get_visible():
            return False
        scene = self._ensure_editor_scene()
        if scene.is_locked(artist) or scene.is_explicitly_hidden(artist):
            return False
        try:
            target = TargetWrapper(artist)
            if target.supported:
                return target.adapter.hit_test(event)
            return bool(artist.contains(event)[0])
        except (AttributeError, TypeError, ValueError, RuntimeError):
            return False

    def _artist_has_selection_geometry(self, artist: Artist) -> bool:
        if not self._is_artist_attached(artist):
            return False
        scene = self._ensure_editor_scene()
        if scene.is_locked(artist) or scene.is_explicitly_hidden(artist):
            return False
        try:
            target = TargetWrapper(artist)
            if not target.supported:
                return False
            points = np.array(target.get_selection_points(), dtype=float)
        except (AttributeError, TypeError, ValueError, RuntimeError):
            return False
        return (
            points.ndim == 2
            and points.shape[0] > 0
            and points.shape[1] >= 2
            and np.all(np.isfinite(points[:, :2]))
        )

    def _is_artist_attached(self, artist: Artist) -> bool:
        """Reject replaced legends and all children registered beneath them."""

        if getattr(artist, "figure", None) is None:
            return False
        if isinstance(artist, EditorGroup) and artist not in self._ensure_editor_scene().groups.values():
            return False
        current = artist
        seen = set()
        parent_map = getattr(self, "_selection_parent_by_id", {})
        while current is not None and id(current) not in seen:
            seen.add(id(current))
            if isinstance(current, Legend):
                axes = current.axes
                if axes is not None:
                    if axes.get_legend() is not current and current not in axes.artists:
                        return False
                else:
                    figure = current.figure
                    if figure is None or (
                        current not in figure.legends and current not in figure.artists
                    ):
                        return False
            current = parent_map.get(id(current))
        return True

    def _resolve_selectable_artist(self, artist: Artist) -> Artist | None:
        if artist is None:
            return None
        if self._artist_has_selection_geometry(artist):
            return artist
        return None

    def _resolve_selectable_elements(
        self, elements: Iterable[Artist], primary: Artist = None
    ) -> tuple[list[Artist], Artist]:
        resolved = []
        for element in elements:
            selectable = self._resolve_selectable_artist(element)
            if selectable is not None:
                resolved.append(selectable)
        primary = self._resolve_selectable_artist(primary)
        return resolved, primary

    def _artist_contains_descendant(self, parent: Artist, descendant: Artist) -> bool:
        if self._ensure_editor_scene().contains(parent, descendant):
            return True
        parent_map = getattr(self, "_selection_parent_by_id", {})
        if id(descendant) in parent_map:
            current = parent_map.get(id(descendant))
            seen = set()
            while current is not None and id(current) not in seen:
                if current is parent:
                    return True
                seen.add(id(current))
                current = parent_map.get(id(current))
            return False
        for child, _explicit in iter_artist_children(parent):
            if child is descendant:
                return True
            if self._artist_contains_descendant(child, descendant):
                return True
        return False

    def _normalize_selection(
        self,
        elements: Iterable[Artist],
        primary: Artist = None,
        preserve_axes: bool = False,
        prefer_containers: bool = False,
    ) -> tuple[list[Artist], Artist]:
        unique = []
        for element in elements:
            if element is not None and element not in unique:
                unique.append(element)

        unique_by_id = {id(element): element for element in unique}
        direct_selection = self.selection_mode is SelectionMode.DIRECT
        remove_ids = set()
        for descendant in unique:
            current = self._interaction_parent(descendant)
            seen = set()
            while current is not None and id(current) not in seen:
                current_id = id(current)
                seen.add(current_id)
                if current_id in unique_by_id:
                    if (
                        _container_keeps_children(current)
                        and not direct_selection
                    ) or (
                        prefer_containers and _container_yields_to_children(current)
                    ):
                        remove_ids.add(id(descendant))
                    elif (
                        _container_keeps_children(current)
                        and direct_selection
                        and not prefer_containers
                    ):
                        remove_ids.add(current_id)
                    elif (
                        _container_yields_to_children(current)
                        and not (preserve_axes and isinstance(current, Axes))
                        and not prefer_containers
                    ):
                        remove_ids.add(current_id)
                current = self._interaction_parent(current)

        # Programmatic callers may submit supported artists before the manager
        # has registered their semantic parent.  Keep the recursive fallback
        # for only those rare objects instead of paying O(n^2) for every marquee.
        for descendant in unique:
            if self._interaction_parent(descendant) is not None:
                continue
            for possible_parent in unique:
                if possible_parent is descendant or not self._artist_contains_descendant(
                    possible_parent, descendant
                ):
                    continue
                if (
                    _container_keeps_children(possible_parent)
                    and not direct_selection
                ) or (
                    prefer_containers
                    and _container_yields_to_children(possible_parent)
                ):
                    remove_ids.add(id(descendant))
                elif (
                    _container_keeps_children(possible_parent)
                    and direct_selection
                    and not prefer_containers
                ):
                    remove_ids.add(id(possible_parent))
                elif (
                    _container_yields_to_children(possible_parent)
                    and not (
                        preserve_axes and isinstance(possible_parent, Axes)
                    )
                    and not prefer_containers
                ):
                    remove_ids.add(id(possible_parent))

        normalized = [
            element for element in unique if id(element) not in remove_ids
        ]

        if primary not in normalized:
            for element in normalized:
                if primary is not None and self._artist_contains_descendant(
                    primary, element
                ):
                    primary = element
                    break
            else:
                primary = normalized[-1] if normalized else None
        return normalized, primary

    def get_hit_stack(self, event: MouseEvent) -> HitStack:
        """Return every visual hit from front to back using one draw-order model."""

        selectable_ids = getattr(self, "_selectable_artist_ids", set())
        interaction_artists = [
            (artist, id(artist) in selectable_ids)
            for artist in getattr(
                self, "_interaction_artists", getattr(self, "_selectable_artists", [])
            )
        ]
        registration_order = {
            id(artist): index
            for index, (artist, _editable) in enumerate(interaction_artists)
        }
        child_orders: dict[int, dict[int, int]] = getattr(
            self, "_draw_child_orders", {}
        )
        self._draw_child_orders = child_orders

        def child_order(parent, child, fallback):
            parent_key = id(parent)
            if parent_key not in child_orders:
                try:
                    children = get_artist_children(parent)
                except (AttributeError, TypeError, ValueError, RuntimeError):
                    children = []
                child_orders[parent_key] = {
                    id(item): index for index, item in enumerate(children)
                }
            return child_orders[parent_key].get(id(child), fallback)

        def pick_order(entry):
            index, (artist, _editable) = entry
            path = []
            current = artist
            seen = set()
            while current is not None and not isinstance(current, Figure):
                current_key = id(current)
                if current_key in seen:
                    break
                seen.add(current_key)
                parent = self._draw_parent(current)
                fallback = registration_order.get(current_key, index)
                order = (
                    child_order(parent, current, fallback)
                    if parent is not None
                    else fallback
                )
                path.append((float(current.get_zorder()), order))
                current = parent
            return tuple(reversed(path)), index

        hits: list[HitCandidate] = []
        for index, (candidate, registered_editable) in sorted(
            enumerate(interaction_artists), key=pick_order, reverse=True
        ):
            if not self._is_interaction_hit(candidate, event):
                continue
            editable = bool(
                registered_editable
                and self._resolve_selectable_artist(candidate) is not None
            )
            hits.append(
                HitCandidate(
                    candidate,
                    editable,
                    pick_order((index, (candidate, registered_editable))),
                    index,
                )
            )
        return HitStack(tuple(hits))

    def get_hit_candidates(self, event: MouseEvent) -> tuple[Artist, ...]:
        """Public candidate-list API resolved through the active selection tool."""

        return self._ensure_selection_kernel().candidates(self.get_hit_stack(event))

    def get_hit_candidate_entries(
        self, event: MouseEvent
    ) -> tuple[tuple[Artist, str], ...]:
        entries = []
        for artist in self.get_hit_candidates(event):
            name = self._interaction_label(artist)
            type_name = type(artist).__name__
            entries.append((artist, f"{name} [{type_name}]"))
        return tuple(entries)

    def show_hit_candidate_menu(self, event: MouseEvent):
        entries = self.get_hit_candidate_entries(event)
        window = getattr(self.figure, "window", None)
        if not entries or window is None:
            return entries
        menu = QtWidgets.QMenu(window)
        for artist, label in entries:
            action = menu.addAction(label)
            action.triggered.connect(
                lambda _checked=False, artist=artist: self.select_element(artist)
            )
        self._candidate_menu = menu
        menu.popup(QtGui.QCursor.pos())
        return entries

    def get_picked_element(
        self,
        event: MouseEvent,
        element: Artist = None,
        picked_element: Artist = None,
        last_selected: Artist = None,
    ):
        """Get the exact leaf Artist hit by an event.

        The legacy return shape is retained for callers that need direct Artist
        hits.  Canvas selection resolves this stack separately through the active
        object/direct tool.
        """
        if element is None and hasattr(self, "_selectable_artists"):
            self._last_pick_blocked = False
            available: list[Artist] = []
            for candidate in self.get_hit_stack(event):
                if not candidate.editable:
                    if not available:
                        self._last_pick_blocked = True
                    break
                available.append(candidate.artist)
            if last_selected is not None and last_selected in available:
                index = available.index(last_selected) + 1
                return (available[index] if index < len(available) else None), True
            return (available[0] if available else picked_element), False
        # start with the figure
        if element is None:
            element = self.figure
        finished = False
        # iterate over all children
        children = sorted(
            iter_artist_children(element),
            key=lambda entry: (entry[0].get_zorder(), int(entry[1])),
        )
        for child, explicit in children:
            # check if the element is contained in the event and has an active dragger
            # if child.contains(event)[0] and ((getattr(child, "_draggable", None) and getattr(child, "_draggable",
            #                                                                               None).connected) or isinstance(child, GrabberGeneric) or isinstance(child, GrabbableRectangleSelection)):
            if self._is_pick_candidate(child, event, explicit):
                selectable_child = (
                    child
                    if isinstance(child, GrabberGeneric)
                    else self._resolve_selectable_artist(child)
                )
                if selectable_child is None:
                    continue
                # if the element is the last selected, finish the search
                if selectable_child == last_selected:
                    return picked_element, True
                # use this element as the current best matching element
                picked_element = selectable_child
            # iterate over the children's children
            picked_element, finished = self.get_picked_element(
                event, child, picked_element, last_selected=last_selected
            )
            # if the subcall wants to finish, just break the loop
            if finished:
                break
        return picked_element, finished

    def _start_marquee_selection(self, event: MouseEvent, click_element: Artist = None):
        self.marquee_start = np.array([event.x, event.y], dtype=float)
        self.marquee_active = False
        self.marquee_additive = _event_has_modifier(event, "shift")
        self.marquee_click_element = click_element
        self._remove_marquee_rect()

    def _remove_marquee_rect(self):
        if self.marquee_rect is not None:
            scene = self.marquee_rect.scene()
            if scene is not None:
                scene.removeItem(self.marquee_rect)
            self.marquee_rect = None

    def _ensure_marquee_rect(self):
        if self.marquee_rect is not None:
            return
        pen = QtGui.QPen(QtGui.QColor("#1E88E5"), 1)
        pen.setStyle(QtCore.Qt.DashLine)
        brush = QtGui.QBrush(QtGui.QColor(30, 136, 229, 24))
        self.marquee_rect = QtWidgets.QGraphicsRectItem(
            0, 0, 0, 0, self.figure._pyl_scene
        )
        self.marquee_rect.setPen(pen)
        self.marquee_rect.setBrush(brush)
        self.marquee_rect.setZValue(899)

    def _update_marquee_rect(self, event: MouseEvent):
        if self.marquee_start is None:
            return
        current = np.array([event.x, event.y], dtype=float)
        delta = current - self.marquee_start
        if not self.marquee_active and np.max(np.abs(delta)) < 3:
            return
        self.marquee_active = True
        self._ensure_marquee_rect()
        x0, y0 = np.minimum(self.marquee_start, current)
        x1, y1 = np.maximum(self.marquee_start, current)
        self.marquee_rect.setRect(float(x0), float(y0), float(x1 - x0), float(y1 - y0))

    def _finish_marquee_selection(self, event: MouseEvent):
        start = self.marquee_start
        active = self.marquee_active
        additive = self.marquee_additive
        click_element = self.marquee_click_element
        self.marquee_start = None
        self.marquee_active = False
        self.marquee_additive = False
        self.marquee_click_element = None
        self._remove_marquee_rect()
        if start is None:
            return
        if active:
            self.select_elements_in_bbox(
                start[0], start[1], event.x, event.y, additive=additive
            )
        else:
            self.select_element(click_element, event)

    def motion_notify_event0(self, event: MouseEvent):
        if self.marquee_start is not None:
            self._hide_preselection()
            self._update_marquee_rect(event)
        elif not any(getattr(grabber, "got_artist", False) for grabber in self.selection.grabbers):
            self.update_preselection(event)

    def _artist_intersects_bbox(
        self, artist: Artist, x0: float, y0: float, x1: float, y1: float
    ) -> bool:
        bounds = self._artist_display_bounds(artist)
        if bounds is None:
            return False
        return self._bounds_intersect(*bounds, x0, y0, x1, y1)

    @staticmethod
    def _bounds_intersect(
        ax0: float,
        ay0: float,
        ax1: float,
        ay1: float,
        x0: float,
        y0: float,
        x1: float,
        y1: float,
    ) -> bool:
        return ax1 >= x0 and ax0 <= x1 and ay1 >= y0 and ay0 <= y1

    @staticmethod
    def _artist_display_bounds(
        artist: Artist,
    ) -> tuple[float, float, float, float] | None:
        points = np.array(TargetWrapper(artist).get_selection_points())
        if len(points) == 0:
            return None
        ax0, ay0 = np.min(points[:, 0]), np.min(points[:, 1])
        ax1, ay1 = np.max(points[:, 0]), np.max(points[:, 1])
        return float(ax0), float(ay0), float(ax1), float(ay1)

    def _uneditable_artist_display_bounds(
        self, artist: Artist
    ) -> tuple[float, float, float, float] | None:
        if getattr(artist, "figure", None) is None:
            return None
        try:
            bbox = artist.get_window_extent(self.figure.canvas.get_renderer())
            bounds = np.asarray(bbox.extents, dtype=float)
        except (AttributeError, TypeError, ValueError, RuntimeError):
            return None
        if bounds.shape != (4,) or not np.all(np.isfinite(bounds)):
            return None
        return tuple(float(value) for value in bounds)

    def select_elements_in_bbox(
        self, x0: float, y0: float, x1: float, y1: float, additive: bool = False
    ) -> list[Artist]:
        with selection_geometry_snapshot():
            return self._select_elements_in_bbox(x0, y0, x1, y1, additive)

    def _select_elements_in_bbox(
        self, x0: float, y0: float, x1: float, y1: float, additive: bool = False
    ) -> list[Artist]:
        x0, x1 = sorted((float(x0), float(x1)))
        y0, y1 = sorted((float(y0), float(y1)))
        artists = list(self.iter_selectable_artists())
        elements = [
            artist
            for artist in artists
            if self._artist_intersects_bbox(artist, x0, y0, x1, y1)
        ]
        prefer_containers = bool(getattr(self, "marquee_select_containers_only", False))
        if prefer_containers:
            elements = [
                element for element in elements if _container_yields_to_children(element)
            ]
        else:
            # Panel containers are never an implicit fallback for an otherwise
            # empty marquee. This keeps the opt-in toggle semantically real and
            # prevents an empty plot-area drag from moving the entire Axes.
            elements = [
                element
                for element in elements
                if not _container_yields_to_children(element)
            ]
            blockers = [
                artist
                for artist in getattr(self, "_uneditable_artists", [])
                if (
                    (bounds := self._uneditable_artist_display_bounds(artist))
                    is not None
                    and self._bounds_intersect(*bounds, x0, y0, x1, y1)
                )
            ]
            if blockers:
                elements = [
                    element
                    for element in elements
                    if not (
                        _container_yields_to_children(element)
                        and any(
                            self._artist_contains_descendant(element, blocker)
                            for blocker in blockers
                        )
                    )
                ]
            elements = self._ensure_selection_kernel().map_artists(elements)
        if elements or not additive:
            selected_elements = self.select_elements(
                elements,
                additive=additive,
                preserve_axes=prefer_containers,
                prefer_containers=prefer_containers,
            )
        else:
            selected_elements = []
        return selected_elements

    def button_release_event0(self, event: MouseEvent):
        """when the mouse button is released"""
        if self.marquee_start is not None:
            self._finish_marquee_selection(event)
            return
        # release the grabber
        if self.grab_element:
            self.grab_element.button_release_event(event)
            self.grab_element = None
        # or notify the selected element
        elif len(self.selection.targets):
            self.selection.button_release_event(event)

    def button_press_event0(self, event: MouseEvent):
        """when the mouse button is pressed"""
        if event.button == 3:
            self.show_hit_candidate_menu(event)
            return
        if event.button == 1:
            self._hide_preselection()
            last = (
                self.selection.targets[-1].target
                if len(self.selection.targets)
                else None
            )
            contained = np.any(
                [t.target.contains(event)[0] for t in self.selection.targets]
            )

            hit_stack = self.get_hit_stack(event)
            # Keep the exact-leaf API active so unsupported foreground objects
            # retain their blocking semantics.
            raw_picked, _ = self.get_picked_element(event)
            click_through = _event_has_modifier(
                event, "alt"
            ) or _event_has_modifier(event, "option")
            kernel = self._ensure_selection_kernel()
            picked_element = kernel.pick(
                hit_stack,
                cycle_from=last if click_through else None,
                wrap=click_through,
            )
            if kernel.mode is SelectionMode.DIRECT and picked_element is None:
                picked_element = raw_picked

            if event.dblclick and picked_element is not None:
                if self.enter_isolation(picked_element):
                    inner = kernel.pick(hit_stack)
                    if inner is not None:
                        self.select_element(inner, event)
                    return

            # if the element is a grabber, store it
            if getattr(self, "_last_pick_blocked", False):
                self._start_marquee_selection(event)
                return
            if isinstance(picked_element, GrabberGeneric):
                self.grab_element = picked_element
            elif (
                isinstance(picked_element, Axes)
                and not contained
                and not event.dblclick
            ):
                self._start_marquee_selection(event, click_element=picked_element)
                return
            elif picked_element is None and not contained:
                self._start_marquee_selection(event)
                return
            # if not, we want to keep our selected element, if the click was in the area of the selected element
            elif (
                len(self.selection.targets) == 0
                or not contained
                or event.dblclick
                or click_through
            ):
                self.select_element(picked_element, event)
                contained = True

            # if we have a grabber, notify it
            if self.grab_element:
                self.grab_element.button_press_event(event)
            # if not, notify the selected element
            elif contained:
                self.selection.button_press_event(event)

    def select_element(self, element: Artist, event: MouseEvent = None):
        """select an artist in a figure"""
        self.select_elements(
            [] if element is None else [element],
            event=event,
            additive=_event_has_modifier(event, "shift"),
            primary=element,
        )

    def select_elements(
        self,
        elements: Iterable[Artist],
        event: MouseEvent = None,
        additive: bool = False,
        primary: Artist = None,
        preserve_axes: bool = False,
        prefer_containers: bool = False,
    ):
        """Select one or more artists through the same model used by the canvas."""
        if additive:
            elements = [target.target for target in self.selection.targets] + list(
                elements
            )
        elements, primary = self._resolve_selectable_elements(elements, primary)
        elements, primary = self._normalize_selection(
            elements,
            primary,
            preserve_axes=preserve_axes,
            prefer_containers=prefer_containers,
        )

        current = [target.target for target in self.selection.targets]
        if primary == self.selected_element and current == elements:
            return elements

        self.selection.clear_targets()

        for element in elements:
            if element != primary:
                self.selection.add_target(element, update=False)

        if primary is not None:
            self.selection._batch_add_targets = True
            try:
                self.on_select(primary, event)
            finally:
                self.selection._batch_add_targets = False
        else:
            self.on_select(None, event)
        self.selected_element = primary
        if self.selection.targets:
            self.selection.update_extent()
        return elements

    def on_deselect(self, event: MouseEvent):
        """deselect currently selected artists"""
        modifier = _event_has_modifier(event, "shift")
        # only if the modifier key is not used
        if not modifier:
            self.selection.clear_targets()

    def on_select(self, element: Artist, event: MouseEvent):
        """when an artist is selected"""
        if element is not None:
            self.selection.add_target(
                element,
                update=not getattr(self.selection, "_batch_add_targets", False),
            )

    def undo(self):
        print("back edit")
        self.figure.change_tracker.backEdit()
        current = [target.target for target in self.selection.targets]
        self.selected_element = current[-1] if current else None
        self._update_interaction_controls()
        self.figure.canvas.draw()

    def redo(self):
        print("forward edit")
        self.figure.change_tracker.forwardEdit()
        current = [target.target for target in self.selection.targets]
        self.selected_element = current[-1] if current else None
        self._update_interaction_controls()
        self.figure.canvas.draw()

    def key_press_event(self, event: KeyEvent):
        """when a key is pressed"""
        if event.key == "v":
            self.set_selection_mode(SelectionMode.OBJECT)
            return
        if event.key == "a":
            self.set_selection_mode(SelectionMode.DIRECT)
            return
        # space: print code to restore current configuration
        if event.key == "ctrl+s":
            self.figure.change_tracker.save()
        if event.key == "ctrl+z":
            self.undo()
        if event.key == "ctrl+y":
            self.redo()
        if event.key == "escape":
            if self._ensure_selection_kernel().scope_root is not None:
                self.exit_isolation()
                return
            self.selection.clear_targets()
            self.selected_element = None
            self.on_select(None, None)
            self.figure.canvas.draw()


class GrabberGeneric(GrabFunctions):
    """a generic grabber object to move a selection"""

    _no_save = True

    def __init__(
        self, parent: GrabbableRectangleSelection, x: float, y: float, dir: int
    ):
        self._animated = True
        GrabFunctions.__init__(self, parent, dir)
        self.pos = (x, y)
        self.updatePos()

    def get_xy(self):
        return self.center

    def set_xy(self, xy: (float, float)):
        self.center = xy

    def getPos(self):
        x, y = self.get_xy()
        return self.transform.transform((x, y))

    def updatePos(self):
        self.set_xy(self.parent.get_pos(self.pos))

    def applyOffset(self, pos: (float, float), event: MouseEvent):
        self.set_xy((self.ox + pos[0], self.oy + pos[1]))


class GrabberGenericRound(GrabberGeneric):
    """a rectangle with a round appearance"""

    d = 10
    shape = "round"

    def __init__(
        self, parent: GrabbableRectangleSelection, x: float, y: float, dir: int, scene
    ):
        pen3 = QtGui.QPen(QtGui.QColor("black"), 2)
        brush1 = QtGui.QBrush(QtGui.QColor("red"))

        self.ellipse = MyEllipse(x, y, 10, 10, scene)
        self.ellipse.view = scene.view
        self.ellipse.grabber = self
        self.ellipse.setPen(pen3)
        self.ellipse.setBrush(brush1)
        self.center = (x, y)

        GrabberGeneric.__init__(self, parent, x, y, dir)

    def set_xy(self, xy: (float, float)):
        self.xy = xy
        self.ellipse.setRect(xy[0] - 5, xy[1] - 5, 10, 10)


class GrabberGenericRectangle(GrabberGeneric):
    """a rectangle with a square appearance"""

    d = 10
    shape = "rect"

    def __init__(
        self, parent: GrabbableRectangleSelection, x: float, y: float, dir: int, scene
    ):
        # somehow the original "self" rectangle does not show up in the current matplotlib version, therefore this doubling
        # self.rect = Rectangle((0, 0), self.d, self.d, figure=parent.figure, edgecolor="k", facecolor="r", zorder=1000, label="grabber")
        # self.rect._no_save = True
        # parent.figure.patches.append(self.rect)

        # Rectangle.__init__(self, (0, 0), self.d, self.d, picker=True, figure=parent.figure, edgecolor="k", facecolor="r", zorder=1000, label="grabber")

        # self.figure.patches.append(self)

        pen3 = QtGui.QPen(QtGui.QColor("black"), 2)
        brush1 = QtGui.QBrush(QtGui.QColor("red"))

        self.ellipse = MyRect(x - 5, y - 5, 10, 10, scene)
        self.ellipse.view = scene.view
        self.ellipse.grabber = self
        self.ellipse.setPen(pen3)
        self.ellipse.setBrush(brush1)

        self.xy = (x, y)
        # self.updatePos()

        GrabberGeneric.__init__(self, parent, x, y, dir)

    def get_xy(self):
        return self.xy
        xy = Rectangle.get_xy(self)
        return xy[0] + self.d / 2, xy[1] + self.d / 2

    def set_xy(self, xy: (float, float)):
        self.xy = xy

        self.ellipse.setRect(xy[0] - 5, xy[1] - 5, 10, 10)
        return
        Rectangle.set_xy(self, (xy[0] - self.d / 2, xy[1] - self.d / 2))
        self.rect.set_xy((xy[0] - self.d / 2, xy[1] - self.d / 2))


class GrabberRotation(GrabFunctions):
    """One native-angle handle anchored to the selected artist's true pivot."""

    def __init__(self, parent: GrabbableRectangleSelection, scene):
        GrabFunctions.__init__(self, parent, 0, no_height=True)
        self.targets = []
        line_pen = QtGui.QPen(QtGui.QColor("#1E88E5"), 1)
        pivot_pen = QtGui.QPen(QtGui.QColor("#1E88E5"), 2)
        handle_pen = QtGui.QPen(QtGui.QColor("black"), 2)
        handle_brush = QtGui.QBrush(QtGui.QColor("#1E88E5"))

        self.line = QtWidgets.QGraphicsLineItem(scene)
        self.line.setPen(line_pen)
        self.line.setZValue(902)
        self.pivot_marker = QtWidgets.QGraphicsEllipseItem(-3, -3, 6, 6, scene)
        self.pivot_marker.setPen(pivot_pen)
        self.pivot_marker.setBrush(QtGui.QBrush(QtCore.Qt.NoBrush))
        self.pivot_marker.setZValue(903)
        self.handle = MyEllipse(-5, -5, 10, 10, scene)
        self.handle.view = scene.view
        self.handle.grabber = self
        self.handle.setPen(handle_pen)
        self.handle.setBrush(handle_brush)
        self.handle.setZValue(904)
        self.handle.setToolTip("Drag to rotate; hold Shift to snap to 15°")
        self.handle.setCursor(QtCore.Qt.CrossCursor)
        self.xy = (-100.0, -100.0)
        self.hide()

    def get_xy(self):
        return self.xy

    def set_xy(self, xy) -> None:
        self.xy = (float(xy[0]), float(xy[1]))
        self.handle.setRect(self.xy[0] - 5, self.xy[1] - 5, 10, 10)

    def updatePos(self) -> None:
        bounds = self.parent.selection_bounds()
        pivot = self.parent.rotation_pivot()
        handle = np.array([(bounds[0] + bounds[2]) / 2, bounds[3] + 24.0])
        self.set_xy(handle)
        self.line.setLine(
            float(pivot[0]), float(pivot[1]), float(handle[0]), float(handle[1])
        )
        self.pivot_marker.setRect(
            float(pivot[0]) - 3, float(pivot[1]) - 3, 6, 6
        )
        self.line.setVisible(True)
        self.pivot_marker.setVisible(True)
        self.handle.setVisible(True)

    def hide(self) -> None:
        self.line.setVisible(False)
        self.pivot_marker.setVisible(False)
        self.handle.setVisible(False)

    def clickedEvent(self, event: MouseEvent):
        self.parent.start_rotation(event)

    def movedEvent(self, event: MouseEvent):
        self.parent.preview_rotation(event)
        self.moved = True

    def releasedEvent(self, event: MouseEvent):
        self.parent.end_rotation()


class MyItem:
    w = 10

    def mousePressEvent(self, e):
        super().mousePressEvent(e)
        self.view.grabber_found = True
        self.scene().grabber_pressed = self
        x, y = scene_point_to_canvas_pixels(self.view, e.scenePos())
        self.grabber.button_press_event(MyEvent(x, y))

    def mouseReleaseEvent(self, e):
        super().mouseReleaseEvent(e)
        self.scene().grabber_pressed = None
        self.view.grabber_found = True
        x, y = scene_point_to_canvas_pixels(self.view, e.scenePos())
        self.grabber.button_release_event(MyEvent(x, y))


class MyRect(MyItem, QtWidgets.QGraphicsRectItem):
    pass


class MyEllipse(MyItem, QtWidgets.QGraphicsEllipseItem):
    pass


class MyEvent:
    def __init__(self, x, y):
        self.x = x
        self.y = y
