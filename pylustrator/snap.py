#!/usr/bin/env python
# -*- coding: utf-8 -*-
# snap.py

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

from typing import List, Optional

import numpy as np
from matplotlib.artist import Artist
from matplotlib.legend import Legend
from matplotlib.text import Text
from qtpy import QtCore, QtGui, QtWidgets

from .artist_adapters import (
    ArtistCapabilities,
    artist_adapter_registry,
    checkXLabel,
    checkYLabel,
    get_artist_adapter,
    legend_anchor_is_point as legend_anchor_is_point,
    legend_anchor_transform as legend_anchor_transform,
    legend_display_loc as legend_display_loc,
    legend_loc_transform as legend_loc_transform,
    set_legend_point_anchor_display as set_legend_point_anchor_display,
    suspend_change_recording,
)
from .helper_functions import main_figure
from .operations import OperationSupport, TransformOperation


DIR_X0 = 1
DIR_Y0 = 2
DIR_X1 = 4
DIR_Y1 = 8


class TargetWrapper:
    """Backward-compatible facade over the artist adapter registry.

    Existing selection and snapping callers keep their historical method names,
    while every operation is delegated to one type-specific adapter.  New code
    should prefer :mod:`pylustrator.artist_adapters` directly.
    """

    target = None

    def __init__(self, target: Artist):
        self.target = target
        self.adapter = get_artist_adapter(target)
        self.figure = self.adapter.figure

    @classmethod
    def supports_target(cls, target: Artist) -> bool:
        return artist_adapter_registry.supports(target)

    @property
    def capabilities(self) -> ArtistCapabilities:
        return self.adapter.capabilities

    @property
    def supported(self) -> bool:
        return self.adapter.supported

    @property
    def do_scale(self) -> bool:
        return self.supports_operation(TransformOperation.RESIZE_GEOMETRY)

    @property
    def fixed_aspect(self) -> bool:
        return self.capabilities.fixed_aspect

    def operation_support(
        self, operation: TransformOperation | str
    ) -> OperationSupport:
        return self.adapter.operation_support(operation)

    def supports_operation(self, operation: TransformOperation | str) -> bool:
        return self.adapter.supports_operation(operation)

    def get_transform(self):
        return self.adapter.get_transform()

    def get_selection_points(self) -> np.ndarray:
        return self.adapter.selection_points()

    def get_positions(self, use_previous_offset=False, update_offset=False):
        return self.adapter.control_points()

    def get_local_positions(
        self, use_previous_offset=False, update_offset=False
    ) -> list[np.ndarray]:
        return self.adapter.local_control_points()

    def set_positions(self, points) -> None:
        self.adapter.apply_control_points(points)

    def set_local_positions(self, points) -> None:
        self.adapter.apply_native_control_points(points)

    def translate(self, delta) -> None:
        self.adapter.translate(delta)

    def apply_display_transform(self, matrix) -> None:
        self.adapter.apply_display_transform(matrix)

    def resize(self, matrix) -> None:
        self.adapter.resize(matrix)

    def get_rotation(self) -> float:
        return self.adapter.rotation()

    def set_rotation(self, value: float) -> None:
        self.adapter.set_rotation(value)

    def get_restore_state(self):
        return self.adapter.snapshot()

    def restore_state(self, state, *, record_changes: bool = True) -> None:
        if record_changes:
            self.adapter.restore(state)
            return
        with suspend_change_recording():
            self.adapter.restore(state)

    def get_extent(self):
        return self.adapter.get_extent()

    def do_get_extent(self):
        return self.adapter.do_get_extent()

    def transform_points(self, points):
        return self.adapter.native_to_display(points)

    def transform_inverted_points(self, points):
        return self.adapter.display_to_native(points)

    def refresh_offset(self) -> None:
        """Compatibility hook retained for older drag code."""


def _text_display_position(text: TargetWrapper) -> np.ndarray:
    return np.array(text.get_positions()[0], dtype=float)


class SnapBase:
    """The base class to implement snaps."""

    data = None

    def __init__(self, ax_source: Artist, ax_target: Artist, edge: int):
        # wrap both object with a TargetWrapper
        self.ax_source = TargetWrapper(ax_source)
        self.ax_target = TargetWrapper(ax_target)
        self.edge = edge
        # initialize a line object for the visualisation of the snap
        self.draw_path = QtWidgets.QGraphicsPathItem()
        parent = main_figure(ax_source)._pyl_graphics_scene_snapparent
        parent.scene().addItem(self.draw_path)
        pen1 = QtGui.QPen(QtGui.QColor("red"), 2)
        pen1.setStyle(QtCore.Qt.DashLine)
        self.draw_path.setPen(pen1)

    def getPosition(self, target: TargetWrapper):
        """get the position of a target"""
        try:
            return target.get_extent()
        except AttributeError:
            return np.array(
                target.figure.transFigure.transform(target.get_position())
            ).flatten()

    def getDistance(self, index: int) -> (int, int):
        """Calculate the distance of the snap to its target"""
        return 0, 0

    def checkSnap(self, index: int) -> Optional[float]:
        """Return the distance to the targets or None"""
        distance = self.getDistance(index)
        if abs(distance) < 10:
            return distance
        return None

    def checkSnapActive(self):
        """Test if the snap condition is fullfilled"""
        distance = min([self.getDistance(index) for index in [0, 1]])
        # show the snap if the distance to a target is smaller than 1
        if abs(distance) < 1:
            self.show()
        else:
            self.hide()

    def show(self):
        """Implements a visualisation of the snap, e.g. lines to indicate what objects are snapped to what"""
        pass

    def set_data(self, xdata, ydata):
        painter_path = QtGui.QPainterPath()
        move = True
        current_pos = (0, 0)
        for x, y in zip(xdata, ydata):
            if np.isnan(x):
                move = True
                continue
            y = self.ax_target.figure.canvas.height() - y
            if move is True:
                painter_path.moveTo(x, y)
                current_pos = (x, y)
                move = False
            else:
                if current_pos[0] > x:
                    painter_path.moveTo(x, y)
                    painter_path.lineTo(*current_pos)
                    current_pos = (x, y)
                else:
                    painter_path.lineTo(x, y)
                    current_pos = (x, y)
        self.draw_path.setPath(painter_path)
        self.data = (xdata, ydata)

    def hide(self):
        """Hides the visualisation"""
        self.set_data((), ())

    def remove(self):
        """Remove the snap and its visualisation"""
        self.hide()
        try:
            self.draw_path.scene().removeItem(self.draw_path)
        except ValueError:
            pass


class SnapSameEdge(SnapBase):
    """a snap that checks if two objects share an edge"""

    def getDistance(self, index: int) -> (int, int):
        """Calculate the distance of the snap to its target"""
        # only if the right edge index (x or y) is queried, if not the distance is infinite
        if self.edge % 2 != index:
            return np.inf
        # get the position of both objects
        p1 = self.getPosition(self.ax_source)
        p2 = self.getPosition(self.ax_target)
        # and return the difference in the target dimension
        return p1[self.edge] - p2[self.edge]

    def show(self):
        """A visualisation of the snap, e.g. lines to indicate what objects are snapped to what"""
        # get the position of both objects
        p1 = self.getPosition(self.ax_source)
        p2 = self.getPosition(self.ax_target)
        # if the focus edge is x, draw a line along the edge
        if self.edge % 2 == 0:
            self.set_data(
                (p1[self.edge], p1[self.edge], p2[self.edge], p2[self.edge]),
                (
                    p1[self.edge - 1],
                    p1[self.edge + 1],
                    p2[self.edge - 1],
                    p2[self.edge + 1],
                ),
            )
        # if the focus edge is y
        else:
            self.set_data(
                (
                    p1[self.edge - 1],
                    p1[self.edge - 3],
                    p2[self.edge - 1],
                    p2[self.edge - 3],
                ),
                (p1[self.edge], p1[self.edge], p2[self.edge], p2[self.edge]),
            )


class SnapSameDimension(SnapBase):
    """a snap that checks if two objects have the same width or height"""

    def getDistance(self, index: int) -> (int, int):
        """Calculate the distance of the snap to its target"""
        # only if the right edge index (x or y) is queried, if not the distance is infinite
        if self.edge % 2 != index:
            return np.inf
        # get the position of both objects
        p1 = self.getPosition(self.ax_source)
        p2 = self.getPosition(self.ax_target)
        # and the difference of the widths (or heights) of the objects
        return (p2[self.edge - 2] - p2[self.edge]) - (p1[self.edge - 2] - p1[self.edge])

    def show(self):
        """A visualisation of the snap, e.g. lines to indicate what objects are snapped to what"""
        # get the position of both objects
        p1 = self.getPosition(self.ax_source)
        p2 = self.getPosition(self.ax_target)
        # if the focus edge is x, draw a line though the center of each object
        if self.edge % 2 == 0:
            self.set_data(
                (p1[0], p1[2], np.nan, p2[0], p2[2]),
                (
                    p1[1] * 0.5 + p1[3] * 0.5,
                    p1[1] * 0.5 + p1[3] * 0.5,
                    np.nan,
                    p2[1] * 0.5 + p2[3] * 0.5,
                    p2[1] * 0.5 + p2[3] * 0.5,
                ),
            )
        # if the focus edge is y
        else:
            self.set_data(
                (
                    p1[0] * 0.5 + p1[2] * 0.5,
                    p1[0] * 0.5 + p1[2] * 0.5,
                    np.nan,
                    p2[0] * 0.5 + p2[2] * 0.5,
                    p2[0] * 0.5 + p2[2] * 0.5,
                ),
                (p1[1], p1[3], np.nan, p2[1], p2[3]),
            )


class SnapSamePos(SnapBase):
    """a snap that checks if two objects have the same position"""

    def getPosition(self, text: TargetWrapper) -> (int, int):
        # get the position of an object
        return _text_display_position(text)

    def getDistance(self, index: int) -> int:
        """Calculate the distance of the snap to its target"""
        # only if the right edge index (x or y) is queried, if not the distance is infinite
        if self.edge % 2 != index:
            return np.inf
        # get the position of both objects
        p1 = self.getPosition(self.ax_source)
        p2 = self.getPosition(self.ax_target)
        # get the distance of the two objects in the target dimension
        return p1[self.edge] - p2[self.edge]

    def show(self):
        """A visualisation of the snap, e.g. lines to indicate what objects are snapped to what"""
        # get the position of both objects
        p1 = self.getPosition(self.ax_source)
        p2 = self.getPosition(self.ax_target)
        # draw a line connecting the centers of the objects
        self.set_data((p1[0], p2[0]), (p1[1], p2[1]))


class SnapSameBorder(SnapBase):
    """A snap that checks if tree axes share the space between them"""

    def __init__(
        self, ax_source: Artist, ax_target: Artist, ax_target2: Artist, edge: int
    ):
        super().__init__(ax_source, ax_target, edge)
        self.ax_target2 = ax_target2

    def overlap(self, p1: list, p2: list, dir: int):
        """Test if two objects have an overlapping x or y region"""
        if p1[dir + 2] < p2[dir] or p1[dir] > p2[dir + 2]:
            return False
        return True

    def getBorders(self, p1: list, p2: list):
        borders = []
        for edge in [0, 1]:
            if self.overlap(p1, p2, 1 - edge):
                if p1[edge + 2] < p2[edge]:
                    dist = p2[edge] - p1[edge + 2]
                    borders.append([edge * 2 + 0, dist])
                if p1[edge] > p2[edge + 2]:
                    dist = p1[edge] - p2[edge + 2]
                    borders.append([edge * 2 + 1, dist])
        return np.array(borders)

    def getDistance(self, index: int):
        """Calculate the distance of the snap to its target"""
        # get the positions of all three targets
        p1 = self.getPosition(self.ax_source)
        p2 = self.getPosition(self.ax_target)
        p3 = self.getPosition(self.ax_target2)

        for edge in [index]:
            if not (self.edge & DIR_X1) and not (self.edge & DIR_Y1):
                if p1[edge + 2] < p2[edge]:
                    continue
            if not (self.edge & DIR_X0) and not (self.edge & DIR_Y0):
                if p1[edge] > p2[edge + 2]:
                    continue
            if (p1[edge + 2] < p2[edge] or p1[edge] > p2[edge + 2]) and self.overlap(
                p1, p2, 1 - edge
            ):
                distances = np.array([p2[edge] - p1[edge + 2], p1[edge] - p2[edge + 2]])
                index1 = np.argmax(distances)
                distance = distances[index1]
                borders = self.getBorders(p2, p3)
                if len(borders):
                    deltas = distance - borders[:, 1]
                    index2 = np.argmin(np.abs(deltas))
                    self.dir2 = borders[index2, 0]
                    self.dir1 = edge * 2 + index1
                    return deltas[index2] * (-1 + 2 * index1)
        return np.inf

    def getConnection(self, p1: list, p2: list, dir: int):
        """return the coordinates of a line that spans the space between to axes"""
        # check which edge (e.g. x, y) and which direction (e.g. if to change the order of p1 and p2)
        edge, order = dir // 2, dir % 2
        # optionally change p1 with p2
        if order == 1:
            p1, p2 = p2, p1
        # if edge is x
        if edge == 0:
            y = np.mean([max(p1[1], p2[1]), min(p1[3], p2[3])])
            return [[p1[2], p2[0], np.nan], [y, y, np.nan]]
        # if edge is y
        x = np.mean([max(p1[0], p2[0]), min(p1[2], p2[2])])
        return [[x, x, np.nan], [p1[3], p2[1], np.nan]]

    def show(self):
        """A visualisation of the snap, e.g. lines to indicate what objects are snapped to what"""
        # get the positions of all three axes
        p1 = self.getPosition(self.ax_source)
        p2 = self.getPosition(self.ax_target)
        p3 = self.getPosition(self.ax_target2)
        # get the
        x1, y1 = self.getConnection(p1, p2, self.dir1)
        x2, y2 = self.getConnection(p2, p3, self.dir2)
        x1.extend(x2)
        y1.extend(y2)
        self.set_data(x1, y1)


class SnapCenterWith(SnapBase):
    """A snap that checks if a text is centered with an axes"""

    def getPosition(self, text: TargetWrapper) -> (int, int):
        """get the position of the first object"""
        return _text_display_position(text)

    def getPosition2(self, axes: TargetWrapper) -> int:
        """get the position of the second object"""
        pos = np.array(axes.get_positions())
        p = pos[0, :]
        p[self.edge] = np.mean(pos, axis=0)[self.edge]
        return p

    def getDistance(self, index: int) -> int:
        """Calculate the distance of the snap to its target"""
        # only if the right edge index (x or y) is queried, if not the distance is infinite
        if self.edge % 2 != index:
            return np.inf
        # get the position of both objects
        p1 = self.getPosition(self.ax_source)
        p2 = self.getPosition2(self.ax_target)
        # get the distance of the two objects in the target dimension
        return p1[self.edge] - p2[self.edge]

    def show(self):
        """A visualisation of the snap, e.g. lines to indicate what objects are snapped to what"""
        # get the position of both objects
        p1 = self.getPosition(self.ax_source)
        p2 = self.getPosition2(self.ax_target)
        # draw a line connecting the centers of the objects
        self.set_data((p1[0], p2[0]), (p1[1], p2[1]))


def checkSnaps(snaps: List[SnapBase]) -> (int, int):
    """get the x and y offsets the snaps suggest"""
    result = [0, 0]
    # iterate over x and y
    for index in range(2):
        # find the best snap
        best = np.inf
        for snap in snaps:
            delta = snap.checkSnap(index)
            if delta is not None and abs(delta) < abs(best):
                best = delta
        # if there is a snap suggestion, store it
        if best < np.inf:
            result[index] = best
    # return the best suggestion
    return result


def checkSnapsActive(snaps: List[SnapBase]):
    """check if snaps are active and show them if yes"""
    for snap in snaps:
        snap.checkSnapActive()


def getSnaps(targets: List[TargetWrapper], dir: int, no_height=False) -> List[SnapBase]:
    """get all snap objects for the target and the direction"""
    snaps = []
    targets = [t.target for t in targets]
    for target in targets:
        if isinstance(target, Legend):
            continue
        if isinstance(target, Text):
            if checkXLabel(target):
                snaps.append(SnapCenterWith(target, checkXLabel(target), 0))
            elif checkYLabel(target):
                snaps.append(SnapCenterWith(target, checkYLabel(target), 1))
            for ax in target.figure.axes + [target.figure]:
                for txt in ax.texts:
                    # for other texts
                    if txt in targets or not txt.get_visible():
                        continue
                    # snap to the x and the y coordinate
                    snaps.append(SnapSamePos(target, txt, 0))
                    snaps.append(SnapSamePos(target, txt, 1))
            continue
        for index, axes in enumerate(target.figure.axes):
            if axes not in targets and axes.get_visible():
                # axes edged
                if dir & DIR_X0:
                    snaps.append(SnapSameEdge(target, axes, 0))
                if dir & DIR_Y0:
                    snaps.append(SnapSameEdge(target, axes, 1))
                if dir & DIR_X1:
                    snaps.append(SnapSameEdge(target, axes, 2))
                if dir & DIR_Y1:
                    snaps.append(SnapSameEdge(target, axes, 3))

                # snap same dimensions
                if not no_height:
                    if dir & DIR_X0:
                        snaps.append(SnapSameDimension(target, axes, 0))
                    if dir & DIR_X1:
                        snaps.append(SnapSameDimension(target, axes, 2))
                    if dir & DIR_Y0:
                        snaps.append(SnapSameDimension(target, axes, 1))
                    if dir & DIR_Y1:
                        snaps.append(SnapSameDimension(target, axes, 3))

                for axes2 in target.figure.axes:
                    if axes2 != axes and axes2 not in targets and axes2.get_visible():
                        snaps.append(SnapSameBorder(target, axes, axes2, dir))
    return snaps
