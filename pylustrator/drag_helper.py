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
from matplotlib.lines import Line2D
from matplotlib.text import Text
from matplotlib.patches import Patch, Rectangle
from matplotlib.backend_bases import MouseEvent, KeyEvent
from typing import Iterable, Sequence
from qtpy import QtCore, QtGui, QtWidgets

from .snap import (
    TargetWrapper,
    checkXLabel,
    checkYLabel,
    getSnaps,
    checkSnaps,
    checkSnapsActive,
    SnapBase,
)
from .change_tracker import ChangeTracker, add_text_default
from .components.plot_layout import scene_point_to_canvas_pixels
from pylustrator.change_tracker import UndoRedo
import time

DIR_X0 = 1
DIR_Y0 = 2
DIR_X1 = 4
DIR_Y1 = 8

blit = False


def _legend_selectable_children(legend: Legend) -> list[Artist]:
    """Return legend parts that Matplotlib does not expose reliably as children."""
    children = []
    children.extend(getattr(legend, "legend_handles", []))
    children.extend(legend.get_texts())
    title = legend.get_title()
    if title is not None:
        children.append(title)
    return children


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
    if explicit:
        return False
    label = artist.get_label()
    return isinstance(label, str) and label.startswith("_")


def _container_yields_to_children(artist: Artist) -> bool:
    return isinstance(artist, (Figure, SubFigure, Axes))


def _container_keeps_children(artist: Artist) -> bool:
    return isinstance(artist, Legend)


def _event_has_modifier(event, modifier: str) -> bool:
    return (
        modifier in event.key.split("+")
        if event is not None and event.key is not None
        else False
    )


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
            self.releasedEvent(event)

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

        keep_aspect = (
            "control" in event.key.split("+") if event.key is not None else False
        )
        ignore_snaps = (
            "shift" in event.key.split("+") if event.key is not None else False
        )

        self.parent.move(
            [dx, dy],
            self.dir,
            self.snaps,
            keep_aspect_ratio=keep_aspect,
            ignore_snaps=ignore_snaps,
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

        self.c4 = self.figure.canvas.mpl_connect("key_press_event", self.keyPressEvent)

        self.targets = []
        self.targets_rects = []

        self.hide_grabber()

    def add_target(self, target: Artist):
        """add an artist to the selection"""
        if target in [wrapped.target for wrapped in self.targets]:
            return
        target = TargetWrapper(target)

        new_points = np.array(target.get_positions())
        if len(new_points) == 0:
            return

        self.targets.append(target)

        if new_points.shape[0] == 3:
            x0, y0, x1, y1 = (
                np.min(new_points[1:, 0]),
                np.min(new_points[1:, 1]),
                np.max(new_points[1:, 0]),
                np.max(new_points[1:, 1]),
            )
        else:
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

        self.update_extent()

    def update_extent(self):
        """updates the extend of the selection to all the selected elements"""
        points = None
        for target in self.targets:
            new_points = np.array(target.get_positions())

            if points is None:
                points = new_points
            else:
                points = np.concatenate((points, new_points))

        if points is None:
            return

        for grabber in self.grabbers:
            grabber.targets = self.targets

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

        if self.do_target_scale():
            self.update_grabber()
        else:
            self.hide_grabber()

    def _known_legends(self) -> list[Legend]:
        legends = list(getattr(self.figure, "legends", []))
        for axes in self.figure.axes:
            legend = axes.get_legend()
            if legend is not None:
                legends.append(legend)
            legends.extend(
                artist for artist in axes.artists if isinstance(artist, Legend)
            )

        unique = []
        seen = set()
        for legend in legends:
            if id(legend) not in seen:
                unique.append(legend)
                seen.add(id(legend))
        return unique

    def _find_direct_parent(self, parent: Artist, target: Artist, seen=None):
        if seen is None:
            seen = set()
        if id(parent) in seen:
            return None
        seen.add(id(parent))

        children = get_artist_children(parent)
        if target in children:
            return parent
        for child in children:
            found = self._find_direct_parent(child, target, seen)
            if found is not None:
                return found
        return None

    def _artist_parent(self, artist: Artist):
        if isinstance(artist, Figure):
            return None
        if isinstance(artist, Legend):
            return getattr(artist, "parent", None) or artist.figure
        if isinstance(artist, Text):
            label_axes = checkXLabel(artist) or checkYLabel(artist)
            if label_axes is not None:
                return label_axes
        for legend in self._known_legends():
            if artist in get_artist_children(legend):
                return legend
        found = self._find_direct_parent(self.figure, artist)
        if found is not None:
            return found
        axes = getattr(artist, "axes", None)
        if axes is not None and axes is not artist:
            return axes
        figure = getattr(artist, "figure", None)
        if figure is not None and figure is not artist:
            return figure
        return None

    def _ancestor_chain(self, artist: Artist) -> list[Artist]:
        chain = []
        current = artist
        seen = set()
        while current is not None and id(current) not in seen:
            seen.add(id(current))
            if isinstance(current, Figure):
                break
            chain.append(current)
            current = self._artist_parent(current)
        return chain

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

    def _resolve_alignment_items(self) -> list[tuple[TargetWrapper, TargetWrapper]]:
        """Return pairs of selected measurement wrapper and same-layer move wrapper."""
        if len(self.targets) <= 1:
            return [(target, target) for target in self.targets]

        chains = [self._ancestor_chain(target.target) for target in self.targets]
        if any(len(chain) == 0 for chain in chains):
            raise ValueError("Selected object has no parent layer.")

        best_indices = None
        best_score = None

        def visit(indices: list[int], index: int) -> None:
            nonlocal best_indices, best_score
            if index == len(chains):
                candidates = [chains[i][indices[i]] for i in range(len(chains))]
                parents = [self._artist_parent(candidate) for candidate in candidates]
                if any(parent is None for parent in parents):
                    return
                if len({id(parent) for parent in parents}) != 1:
                    return
                score = (max(indices), sum(indices), tuple(indices))
                if best_score is None or score < best_score:
                    best_score = score
                    best_indices = tuple(indices)
                return
            for candidate_index in range(len(chains[index])):
                indices.append(candidate_index)
                visit(indices, index + 1)
                indices.pop()

        visit([], 0)
        if best_indices is None:
            raise ValueError("Selected objects do not resolve to one parent layer.")

        wrappers_by_artist = {
            id(target.target): target for target in self.targets
        }
        items = []
        for target, chain, chain_index in zip(self.targets, chains, best_indices):
            move_artist = chain[chain_index]
            move_wrapper = wrappers_by_artist.get(id(move_artist))
            if move_wrapper is None:
                move_wrapper = TargetWrapper(move_artist)
                wrappers_by_artist[id(move_artist)] = move_wrapper
            items.append((target, move_wrapper))
        return items

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
        if measure_target.target is move_target.target:
            return np.array(measure_target.get_positions(), dtype=float)
        if isinstance(measure_target.target, (Text, Legend, Line2D)):
            bbox = measure_target.target.get_window_extent(
                self.figure.canvas.get_renderer()
            )
            return np.array([[bbox.x0, bbox.y0], [bbox.x1, bbox.y1]], dtype=float)
        points = np.array(measure_target.get_positions(), dtype=float)
        if points.shape[0] == 3:
            return points[1:]
        return points

    @staticmethod
    def _measure_size(points: np.ndarray, y: int, direct_target: bool) -> float:
        if direct_target:
            return np.diff(points[:, y])[0]
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
            self.start_move(save_targets=[target for target, _delta in plan])
            for target, delta in plan:
                new_points = np.array(target.get_positions())
                new_points[:, y] += delta
                target.set_positions(new_points)
            self.update_extent()
            self.has_moved = True
            self.end_move()

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
            self.start_move(save_targets=[target for target, _delta in plan])
            for target, delta in plan:
                new_points = np.array(target.get_positions())
                new_points[:, y] += delta
                target.set_positions(new_points)
            self.has_moved = True
            self.end_move()

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
        if points.shape[0] == 3:
            points = points[1:]
        return np.array(
            [
                np.min(points[:, 0]),
                np.min(points[:, 1]),
                np.max(points[:, 0]),
                np.max(points[:, 1]),
            ],
            dtype=float,
        )

    def match_size(self, mode: str) -> bool:
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

        match_width, match_height = modes[mode]
        reference_points = np.array(self.targets[0].get_positions(), dtype=float)
        reference_bounds = self._points_bounds(reference_points)
        reference_width = reference_bounds[2] - reference_bounds[0]
        reference_height = reference_bounds[3] - reference_bounds[1]
        if match_width and reference_width <= 0:
            raise ValueError("Reference object has no width.")
        if match_height and reference_height <= 0:
            raise ValueError("Reference object has no height.")

        non_scalable = [target.target for target in self.targets[1:] if not target.do_scale]
        if non_scalable:
            names = ", ".join(type(target).__name__ for target in non_scalable)
            raise ValueError(f"Selected object cannot be resized: {names}")

        planned: list[tuple[TargetWrapper, np.ndarray, np.ndarray]] = []
        for target in self.targets[1:]:
            points = np.array(target.get_positions(), dtype=float)
            bounds = self._points_bounds(points)
            current_width = bounds[2] - bounds[0]
            current_height = bounds[3] - bounds[1]
            if match_width and current_width <= 0:
                raise ValueError("Selected object has no width.")
            if match_height and current_height <= 0:
                raise ValueError("Selected object has no height.")
            planned.append((target, points, bounds))

        self.start_move()
        changed = False
        for target, points, bounds in planned:
            current_width = bounds[2] - bounds[0]
            current_height = bounds[3] - bounds[1]
            scale_x = reference_width / current_width if match_width else 1.0
            scale_y = reference_height / current_height if match_height else 1.0
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
            target.set_positions(self.apply_transform(transform, points))
            changed = True

        if changed:
            self.update_extent()
            self.has_moved = True
        self.end_move("Resize")
        self.figure.canvas.draw()
        self.update_selection_rectangles()
        return changed

    def scale_selection(self, factor: float) -> bool:
        """Scale the current selection around its combined center."""
        if factor <= 0:
            raise ValueError("Scale factor must be positive.")
        if len(self.targets) == 0:
            return False
        non_scalable = [target.target for target in self.targets if not target.do_scale]
        if non_scalable:
            names = ", ".join(type(target).__name__ for target in non_scalable)
            raise ValueError(f"Selected object cannot be scaled: {names}")
        if np.isclose(factor, 1.0):
            return False

        center_x = (self.positions[0] + self.positions[2]) / 2
        center_y = (self.positions[1] + self.positions[3]) / 2
        transform = np.array(
            [
                [factor, 0.0, center_x * (1.0 - factor)],
                [0.0, factor, center_y * (1.0 - factor)],
                [0.0, 0.0, 1.0],
            ],
            dtype=float,
        )

        self.start_move()
        for target in self.targets:
            points = np.array(target.get_positions(), dtype=float)
            target.set_positions(self.apply_transform(transform, points))
        self.update_extent()
        self.has_moved = True
        self.end_move("Scale")
        self.figure.canvas.draw()
        self.update_selection_rectangles()
        return True

    @staticmethod
    def _rotatable_value(target: Artist) -> float | None:
        if isinstance(target, Text):
            return float(target.get_rotation())
        if (
            isinstance(target, Patch)
            and hasattr(target, "get_angle")
            and hasattr(target, "set_angle")
        ):
            return float(target.get_angle())
        return None

    def _set_rotation_value(self, target: Artist, value: float) -> None:
        if isinstance(target, Text):
            add_text_default(target)
            target.set_rotation(value)
            self.figure.change_tracker.addNewTextChange(target)
            return
        if isinstance(target, Patch) and hasattr(target, "set_angle"):
            target.set_angle(value)
            self.figure.change_tracker.addChange(
                target, ".set_angle(%f)" % target.get_angle()
            )
            return
        raise ValueError(f"Selected object cannot be rotated: {type(target).__name__}")

    def rotate_selection(self, angle_degrees: float) -> bool:
        """Rotate selected objects that have a native saveable rotation property."""
        if len(self.targets) == 0:
            return False
        if np.isclose(angle_degrees, 0.0):
            return False

        old_values: list[tuple[Artist, float]] = []
        unsupported: list[Artist] = []
        for target in self.targets:
            value = self._rotatable_value(target.target)
            if value is None:
                unsupported.append(target.target)
            else:
                old_values.append((target.target, value))
        if unsupported:
            names = ", ".join(type(target).__name__ for target in unsupported)
            raise ValueError(f"Selected object cannot be rotated: {names}")

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
        self.figure.signals.figure_selection_property_changed.emit()
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
                new_points = np.array(
                    target.get_positions(use_previous_offset)
                )
                if new_points.shape[0] == 3:
                    x0, y0, x1, y1 = (
                        np.min(new_points[1:, 0]),
                        np.min(new_points[1:, 1]),
                        np.max(new_points[1:, 0]),
                        np.max(new_points[1:, 1]),
                    )
                else:
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
            self.hide_grabber()

    def hide_grabber(self):
        """hide the grabber elements"""
        for grabber in self.grabbers:
            grabber.set_xy((-100, -100))

    def clear_targets(self):
        """remove all elements from the selection"""
        for rect in self.targets_rects:
            self.graphics_scene.scene().removeItem(rect)
            # self.figure.patches.remove(rect)
        self.targets_rects = []
        self.targets = []

        self.hide_grabber()

    def do_target_scale(self) -> bool:
        """if any of the elements in the selection allows scaling"""
        return np.any([target.do_scale for target in self.targets])

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

    def get_trans_matrix(self):
        """the transformation matrix for the current displacement and scaling of the selection"""
        x, y = self.p1
        w, h = self.size()
        return np.array([[w, 0, x], [0, h, y], [0, 0, 1]], dtype=float)

    def get_inv_trans_matrix(self):
        """the inverse transformation for the current displacement and scaling of the selection"""
        x, y = self.p1
        w, h = self.size()
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
        positions = [target.get_positions() for target in wrapped_targets]

        def undo():
            self.clear_targets()
            for target, pos in zip(restore_targets, positions):
                target = TargetWrapper(target)
                target.set_positions(pos)
            for target in selected_targets:
                self.add_target(target)

        return undo

    def start_move(self, save_targets: Iterable[TargetWrapper] = None):
        """start to move a grabber"""
        self.start_p1 = self.p1.copy()
        self.start_p2 = self.p2.copy()
        self.hide_grabber()
        self.has_moved = False
        self.save_targets = self._unique_wrappers(save_targets or self.targets)
        for target in self._unique_wrappers(list(self.targets) + self.save_targets):
            target.refresh_offset()

        self.store_start = self.get_save_point(self.save_targets)

    def end_move(self, edit_name: str = "Move"):
        """a grabber move stopped"""
        self.update_grabber()

        self.store_end = self.get_save_point(self.save_targets)
        if self.has_moved is True:
            self.figure.signals.figure_selection_moved.emit()
            self.figure.change_tracker.addEdit(
                [self.store_start, self.store_end, edit_name]
            )

    def addOffset(self, pos: Sequence, dir: int, keep_aspect_ratio: bool = True):
        """move the whole selection (e.g. for the use of the arrow keys)"""
        pos = list(pos)
        self.old_inv_transform = self.get_inv_trans_matrix()

        if (keep_aspect_ratio or self.do_change_aspect_ratio()) and not (
            dir & DIR_X0 and dir & DIR_X1 and dir & DIR_Y0 and dir & DIR_Y1
        ):
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

        transform = np.dot(self.get_trans_matrix(), self.old_inv_transform)
        for target in self.targets:
            self.transform_target(transform, target)

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
    ):
        """called from a grabber to move the selection."""
        self.addOffset(pos, dir, keep_aspect_ratio)
        self.has_moved = True

        if not ignore_snaps:
            offx, offy = checkSnaps(snaps)
            self.addOffset((pos[0] - offx, pos[1] - offy), dir, keep_aspect_ratio)

            offx, offy = checkSnaps(self.snaps)

        checkSnapsActive(snaps)

    def apply_transform(self, transform: np.ndarray, point: Sequence[float]):
        """apply the given transformation to a point"""
        point = np.array(point)
        point = np.hstack((point, np.ones((point.shape[0], 1)))).T
        return np.dot(transform, point)[:2].T

    def transform_target(self, transform: np.ndarray, target: TargetWrapper):
        """transform the position of an artist."""
        points = target.get_positions()
        points = self.apply_transform(transform, points)
        target.set_positions(points)

    def keyPressEvent(self, event: KeyEvent):
        """when a key is pressed. Arrow keys move the selection, Pageup/down movein z"""
        # if not self.selected:
        #    return
        # move last axis in z order
        if event.key == "pagedown":
            for target in self.targets:
                target.target.set_zorder(target.target.get_zorder() - 1)
                self.figure.change_tracker.addChange(
                    target.target, ".set_zorder(%d)" % target.target.get_zorder()
                )
            self.figure.canvas.draw()
        if event.key == "pageup":
            for target in self.targets:
                target.target.set_zorder(target.target.get_zorder() + 1)
                self.figure.change_tracker.addChange(
                    target.target, ".set_zorder(%d)" % target.target.get_zorder()
                )
            self.figure.canvas.draw()
        if event.key == "left":
            self.start_move()
            self.addOffset((-1, 0), self.dir)
            self.has_moved = True
            self.end_move()
            self.figure.canvas.schedule_draw()
        if event.key == "right":
            self.start_move()
            self.addOffset((+1, 0), self.dir)
            self.has_moved = True
            self.end_move()
            self.figure.canvas.schedule_draw()
        if event.key == "down":
            self.start_move()
            self.addOffset((0, -1), self.dir)
            self.has_moved = True
            self.end_move()
            self.figure.canvas.schedule_draw()
        if event.key == "up":
            self.start_move()
            self.addOffset((0, +1), self.dir)
            self.has_moved = True
            self.end_move()
            self.figure.canvas.schedule_draw()
        if event.key in ["delete", "backspace"]:
            self.delete_targets()


class DragManager:
    """a class to manage the selection and the moving of artists in a figure"""

    selected_element = None
    grab_element = None

    def __init__(self, figure: Figure, no_save):
        self.figure = figure
        self.figure.figure_dragger = self
        self.marquee_start = None
        self.marquee_rect = None
        self.marquee_active = False
        self.marquee_additive = False
        self.marquee_click_element = None

        self.figure.canvas.mpl_disconnect(
            self.figure.canvas.manager.key_press_handler_id
        )

        self.activate()

        self.make_figure_draggable(self.figure)
        self.make_axes_draggable(self.figure.axes)
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

    def deactivate(self):
        """deactivate the interaction callbacks from the figure"""
        self.figure.canvas.mpl_disconnect(self.c3)
        self.figure.canvas.mpl_disconnect(self.c2)
        self.figure.canvas.mpl_disconnect(self.c4)
        self.figure.canvas.mpl_disconnect(self.c5)

        self.selection.clear_targets()
        self.selected_element = None
        self.on_select(None, None)
        self.figure.canvas.draw()

    def make_draggable(self, target: Artist):
        """make an artist draggable"""
        target.set_picker(True)
        if isinstance(target, Text):
            add_text_default(target)
            target.set_bbox(dict(facecolor="none", edgecolor="none"))
        if isinstance(target, Legend):
            for handle in target.legend_handles:
                self.make_draggable(handle)
            for text in target.get_texts():
                self.make_draggable(text)
            title = target.get_title()
            if title is not None:
                self.make_draggable(title)

    def make_axes_draggable(self, axes: list[Axes]) -> None:
        for index, ax in enumerate(axes):
            ax.set_picker(True)
            leg = ax.get_legend()
            if leg:
                self.make_draggable(leg)
            for text in ax.texts:
                self.make_draggable(text)
            for attribute_name in ["title", "_left_title", "_right_title"]:
                text = getattr(ax, attribute_name, None)
                if text is not None:
                    self.make_draggable(text)
            for patch in ax.patches:
                self.make_draggable(patch)
            self.make_draggable(ax.xaxis.get_label())
            self.make_draggable(ax.yaxis.get_label())
            self.make_draggable(ax)
            self.make_axes_draggable(ax.child_axes)

    def make_figure_draggable(self, fig: Figure | SubFigure) -> None:
        for text in fig.texts:
            self.make_draggable(text)
        for patch in fig.patches:
            self.make_draggable(patch)
        for leg in fig.legends:
            self.make_draggable(leg)
        for subfig in fig.subfigs:
            self.make_figure_draggable(subfig)

    def iter_selectable_artists(
        self, element: Artist = None, seen: set[int] = None
    ) -> Iterable[Artist]:
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
        if not child.get_visible():
            return False
        if isinstance(child, Text) and child.get_text() == "":
            return False
        if _is_internal_label(child, explicit):
            return False
        if not (child.pickable() or isinstance(child, GrabberGeneric)):
            return False
        if event is not None and not child.contains(event)[0]:
            return False
        return True

    def _artist_contains_descendant(self, parent: Artist, descendant: Artist) -> bool:
        for child, _explicit in iter_artist_children(parent):
            if child is descendant:
                return True
            if self._artist_contains_descendant(child, descendant):
                return True
        return False

    def _normalize_selection(
        self, elements: Iterable[Artist], primary: Artist = None
    ) -> tuple[list[Artist], Artist]:
        unique = []
        for element in elements:
            if element is not None and element not in unique:
                unique.append(element)

        normalized = [
            element
            for element in unique
            if not any(
                other is not element
                and (
                    (
                        _container_yields_to_children(element)
                        and self._artist_contains_descendant(element, other)
                    )
                    or (
                        _container_keeps_children(other)
                        and self._artist_contains_descendant(other, element)
                    )
                )
                for other in unique
            )
        ]

        if primary not in normalized:
            for element in normalized:
                if primary is not None and self._artist_contains_descendant(primary, element):
                    primary = element
                    break
            else:
                primary = normalized[-1] if normalized else None
        return normalized, primary

    def get_picked_element(
        self,
        event: MouseEvent,
        element: Artist = None,
        picked_element: Artist = None,
        last_selected: Artist = None,
    ):
        """get the picked element that an event refers to.
        To implement selection of elements at the back with multiple clicks.
        """
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
                # if the element is the last selected, finish the search
                if child == last_selected:
                    return picked_element, True
                # use this element as the current best matching element
                picked_element = child
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
        self.marquee_rect = QtWidgets.QGraphicsRectItem(0, 0, 0, 0, self.figure._pyl_scene)
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
            self.select_elements_in_bbox(start[0], start[1], event.x, event.y, additive=additive)
        else:
            self.select_element(click_element, event)

    def motion_notify_event0(self, event: MouseEvent):
        self._update_marquee_rect(event)

    def _artist_intersects_bbox(
        self, artist: Artist, x0: float, y0: float, x1: float, y1: float
    ) -> bool:
        points = np.array(TargetWrapper(artist).get_positions())
        if len(points) == 0:
            return False
        ax0, ay0 = np.min(points[:, 0]), np.min(points[:, 1])
        ax1, ay1 = np.max(points[:, 0]), np.max(points[:, 1])
        if isinstance(artist, (Axes, Legend)):
            return ax0 >= x0 and ax1 <= x1 and ay0 >= y0 and ay1 <= y1
        return ax1 >= x0 and ax0 <= x1 and ay1 >= y0 and ay0 <= y1

    def select_elements_in_bbox(
        self, x0: float, y0: float, x1: float, y1: float, additive: bool = False
    ) -> list[Artist]:
        x0, x1 = sorted((float(x0), float(x1)))
        y0, y1 = sorted((float(y0), float(y1)))
        elements = [
            artist
            for artist in self.iter_selectable_artists()
            if self._artist_intersects_bbox(artist, x0, y0, x1, y1)
        ]
        if elements or not additive:
            self.select_elements(elements, additive=additive)
        return elements

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
        if event.button == 1:
            last = (
                self.selection.targets[-1].target
                if len(self.selection.targets)
                else None
            )
            contained = np.any(
                [t.target.contains(event)[0] for t in self.selection.targets]
            )

            # recursively iterate over all elements
            picked_element, _ = self.get_picked_element(
                event, last_selected=last if event.dblclick else None
            )

            # if the element is a grabber, store it
            if isinstance(picked_element, GrabberGeneric):
                self.grab_element = picked_element
            elif isinstance(picked_element, Axes) and not contained and not event.dblclick:
                self._start_marquee_selection(event, click_element=picked_element)
                return
            elif picked_element is None and not contained:
                self._start_marquee_selection(event)
                return
            # if not, we want to keep our selected element, if the click was in the area of the selected element
            elif len(self.selection.targets) == 0 or not contained or event.dblclick:
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
    ):
        """Select one or more artists through the same model used by the canvas."""
        if additive:
            elements = [target.target for target in self.selection.targets] + list(elements)
        elements, primary = self._normalize_selection(elements, primary)

        current = [target.target for target in self.selection.targets]
        if primary == self.selected_element and current == elements:
            return

        self.selection.clear_targets()

        for element in elements:
            if element != primary:
                self.selection.add_target(element)

        if primary is not None:
            self.on_select(primary, event)
        else:
            self.on_select(None, event)
        self.selected_element = primary

    def on_deselect(self, event: MouseEvent):
        """deselect currently selected artists"""
        modifier = _event_has_modifier(event, "shift")
        # only if the modifier key is not used
        if not modifier:
            self.selection.clear_targets()

    def on_select(self, element: Artist, event: MouseEvent):
        """when an artist is selected"""
        if element is not None:
            self.selection.add_target(element)

    def undo(self):
        print("back edit")
        self.figure.change_tracker.backEdit()
        self.selection.clear_targets()
        self.selected_element = None
        self.on_select(None, None)
        self.figure.canvas.draw()

    def redo(self):
        print("forward edit")
        self.figure.change_tracker.forwardEdit()
        self.selection.clear_targets()
        self.selected_element = None
        self.on_select(None, None)
        self.figure.canvas.draw()

    def key_press_event(self, event: KeyEvent):
        """when a key is pressed"""
        # space: print code to restore current configuration
        if event.key == "ctrl+s":
            self.figure.change_tracker.save()
        if event.key == "ctrl+z":
            self.undo()
        if event.key == "ctrl+y":
            self.redo()
        if event.key == "escape":
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
