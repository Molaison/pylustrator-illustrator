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
from packaging import version
from qtpy import QtCore, QtGui, QtWidgets

import matplotlib as mpl
import numpy as np
from matplotlib.artist import Artist

try:  # starting from mpl version 3.6.0
    from matplotlib.axes import Axes
except ImportError:
    from matplotlib.axes._subplots import Axes
from matplotlib.collections import (
    Collection,
    LineCollection,
    PathCollection,
    PolyCollection,
)
from matplotlib.image import AxesImage
from matplotlib.legend import Legend
from matplotlib.lines import Line2D
from matplotlib.path import Path
from matplotlib.patches import (
    Ellipse,
    ConnectionPatch,
    FancyArrowPatch,
    FancyBboxPatch,
    PathPatch,
    Polygon,
    Rectangle,
    RegularPolygon,
    Wedge,
)
from matplotlib.text import Annotation, Text
from matplotlib.transforms import (
    Bbox,
    BboxTransformFrom,
    BboxTransformTo,
    IdentityTransform,
)

from .helper_functions import main_figure


DIR_X0 = 1
DIR_Y0 = 2
DIR_X1 = 4
DIR_Y1 = 8


def checkXLabel(target: Artist):
    """checks if the target is the xlabel of an axis"""
    for axes in target.figure.axes:
        if axes.xaxis.get_label() == target:
            return axes


def checkYLabel(target: Artist):
    """checks if the target is the ylabel of an axis"""
    for axes in target.figure.axes:
        if axes.yaxis.get_label() == target:
            return axes


def cache_property(object, name):
    if getattr(object, f"_pylustrator_cached_{name}", False) is True:
        return
    setattr(object, f"_pylustrator_cached_{name}", True)
    getter = getattr(object, f"get_{name}")
    setter = getattr(object, f"set_{name}")

    def new_getter(*args, **kwargs):
        if getattr(object, f"_pylustrator_cache_{name}", None) is None:
            setattr(object, f"_pylustrator_cache_{name}", getter(*args, **kwargs))
        return getattr(object, f"_pylustrator_cache_{name}", None)

    def new_setter(*args, **kwargs):
        result = setter(*args, **kwargs)
        setattr(object, f"_pylustrator_cache_{name}", None)
        return result

    setattr(object, f"get_{name}", new_getter)
    setattr(object, f"set_{name}", new_setter)


def legend_loc_transform(legend: Legend):
    return BboxTransformFrom(legend.get_bbox_to_anchor())


def legend_anchor_transform(legend: Legend):
    return getattr(
        legend.get_bbox_to_anchor(),
        "_transform",
        BboxTransformTo(legend.parent.bbox),
    )


def legend_anchor_is_point(legend: Legend):
    bbox = legend.get_bbox_to_anchor()
    return bbox.width == 0 and bbox.height == 0


def set_legend_point_anchor_display(legend: Legend, point, transform=None):
    if transform is None:
        transform = getattr(
            legend.get_bbox_to_anchor(),
            "_transform",
            BboxTransformTo(legend.parent.bbox),
        )
    legend.set_bbox_to_anchor(
        tuple(float(x) for x in transform.inverted().transform(point)),
        transform=transform,
    )


def legend_display_loc(legend: Legend):
    if legend_anchor_is_point(legend):
        return np.array(legend.get_bbox_to_anchor().p0)
    bbox = legend.get_frame().get_bbox()
    if isinstance(legend._get_loc(), int):
        return np.array([bbox.x0, bbox.y0])
    return BboxTransformTo(legend.get_bbox_to_anchor()).transform(legend._get_loc())


class TargetWrapper(object):
    """Expose one display-space interaction contract for supported artists.

    ``get_positions`` returns writable control points.  Selection and alignment
    use ``get_selection_points`` instead, because an artist's visible bounds are
    not necessarily writable state (text and annotations are the most obvious
    examples).  Keeping those concepts separate is what makes preview, commit,
    and undo agree across artist types.
    """

    target = None

    _patch_types = (
        Rectangle,
        Ellipse,
        FancyArrowPatch,
        FancyBboxPatch,
        PathPatch,
        Polygon,
        RegularPolygon,
        Wedge,
    )
    _collection_types = (PathCollection, LineCollection, PolyCollection)

    @classmethod
    def supports_target(cls, target: Artist) -> bool:
        """Return whether the artist has a lossless move implementation."""
        if isinstance(target, ConnectionPatch):
            return False
        if isinstance(target, FancyBboxPatch) and not target.get_data_transform().is_affine:
            # BoxStyle padding and corner geometry are expressed around the
            # native bounds.  Under a non-affine transform, changing those
            # bounds does not translate the visible box by one display delta.
            return False
        return isinstance(
            target,
            (Axes, Text, Legend, Line2D, AxesImage)
            + cls._patch_types
            + cls._collection_types,
        )

    def __init__(self, target: Artist):
        self.target = target
        self.figure = target.figure
        self.supported = self.supports_target(target)
        self.do_scale = True
        self.fixed_aspect = False
        # a patch uses the data_transform
        if isinstance(self.target, self._patch_types):
            self.get_transform = self.target.get_data_transform
            if isinstance(
                self.target,
                (FancyArrowPatch, FancyBboxPatch, RegularPolygon, Wedge),
            ):
                # These artists expose a movable center but not a general affine
                # shape setter, or have BoxStyle padding/corners that do not
                # follow a bounds-only scale.  Hiding resize handles is safer
                # than committing a result that cannot match the preview.
                self.do_scale = False
            elif isinstance(self.target, (Rectangle, Ellipse)) and not np.isclose(
                float(self.target.get_angle()) % 180.0, 0.0
            ):
                # Width/height setters operate in the unrotated local frame, so
                # an axis-aligned resize preview would not describe the result.
                self.do_scale = False
        # axes use the figure_transform
        elif isinstance(self.target, Axes):
            # and optionally have a fixed aspect ratio
            if (
                self.target.get_aspect() != "auto"
                and self.target.get_adjustable() != "datalim"
            ):
                self.fixed_aspect = True
            # old matplotlib version
            if version.parse(mpl.__version__) < version.parse("3.4.0"):
                self.get_transform = lambda: self.target.figure.transFigure
            else:
                self.get_transform = (
                    lambda: self.target.figure.transSubfigure
                    if self.target.figure.transSubfigure
                    else self.target.figure.transFigure
                )

            # cache the get_position
            cache_property(self.target, "position")
        # texts use get_transform
        elif isinstance(self.target, Text):
            self.do_scale = False
            if checkXLabel(self.target):
                self.label_factor = self.figure.dpi / 72.0
                if getattr(self.target, "pad_offset", None) is None:
                    self.target.pad_offset = (
                        self.target.get_position()[1]
                        + checkXLabel(self.target).xaxis.labelpad * self.label_factor
                    )
                self.label_y = self.target.get_position()[1]
            elif checkYLabel(self.target):
                self.label_factor = self.figure.dpi / 72.0
                if getattr(self.target, "pad_offset", None) is None:
                    self.target.pad_offset = (
                        self.target.get_position()[0]
                        + checkYLabel(self.target).yaxis.labelpad * self.label_factor
                    )
                self.label_x = self.target.get_position()[0]
            self.get_transform = self.target.get_transform
        elif isinstance(self.target, Legend):
            self.get_transform = IdentityTransform
            self.do_scale = False
        elif isinstance(self.target, Line2D):
            self.get_transform = IdentityTransform
            self.do_scale = False
        elif isinstance(self.target, AxesImage):
            self.get_transform = self.target.get_transform
        elif isinstance(self.target, self._collection_types):
            self.get_transform = self.target.get_transform
            # Scaling a collection's anchors without also scaling marker/stroke
            # geometry is misleading.  Collections remain directly movable.
            self.do_scale = False
        # the default is to use get_transform
        else:
            self.get_transform = getattr(
                self.target, "get_transform", IdentityTransform
            )
            self.do_scale = False

    def _renderer(self):
        return self.figure.canvas.get_renderer()

    def _annotation_xy_transform(self):
        return self.target._get_xy_transform(self._renderer(), self.target.xycoords)

    @staticmethod
    def _point_array(points) -> np.ndarray:
        points = np.ma.asarray(points, dtype=float)
        if np.ma.isMaskedArray(points):
            points = points.filled(np.nan)
        points = np.asarray(points, dtype=float)
        if points.ndim != 2 or points.shape[1] < 2:
            return np.empty((0, 2), dtype=float)
        return points[:, :2]

    @staticmethod
    def _finite_points(points) -> np.ndarray:
        points = TargetWrapper._point_array(points)
        return points[np.all(np.isfinite(points), axis=1)]

    def _collection_local_groups(self) -> list[np.ndarray]:
        if isinstance(self.target, PathCollection):
            return [self._point_array(self.target.get_offsets())]
        if isinstance(self.target, LineCollection):
            return [
                self._point_array(segment) for segment in self.target.get_segments()
            ]
        if isinstance(self.target, PolyCollection):
            return [
                self._point_array(path.vertices) for path in self.target.get_paths()
            ]
        return []

    def _collection_display_groups(self) -> list[np.ndarray]:
        groups = self._collection_local_groups()
        if isinstance(self.target, PathCollection):
            transform = self.target.get_offset_transform()
        else:
            transform = self.target.get_transform()
        return [
            self._finite_points(transform.transform(group))
            for group in groups
            if len(group)
        ]

    def _collection_padding(self) -> float:
        """Approximate display padding contributed by markers and strokes."""
        dpi_per_point = float(self.figure.dpi) / 72.0
        linewidths = np.asarray(self.target.get_linewidths(), dtype=float)
        stroke = (
            float(np.max(linewidths)) * dpi_per_point / 2 if linewidths.size else 0.0
        )
        if isinstance(self.target, PathCollection):
            sizes = np.asarray(self.target.get_sizes(), dtype=float)
            marker = (
                float(np.sqrt(np.max(sizes))) * dpi_per_point / 2 if sizes.size else 0.0
            )
            return marker + stroke
        return stroke

    @staticmethod
    def _bounds_points(points: np.ndarray, padding: float = 0.0) -> np.ndarray:
        points = TargetWrapper._finite_points(points)
        if len(points) == 0:
            return np.empty((0, 2), dtype=float)
        return np.array(
            [
                [np.min(points[:, 0]) - padding, np.min(points[:, 1]) - padding],
                [np.max(points[:, 0]) + padding, np.max(points[:, 1]) + padding],
            ],
            dtype=float,
        )

    def get_selection_points(self) -> np.ndarray:
        """Return the visible display-space bounds used by selection/alignment."""
        preview = getattr(self.target, "_pylustrator_preview_selection_points", None)
        if preview is not None:
            return np.asarray(preview, dtype=float).copy()

        if isinstance(self.target, Collection):
            groups = self._collection_display_groups()
            if groups:
                return self._bounds_points(
                    np.concatenate(groups), padding=self._collection_padding()
                )

        try:
            bbox = self.target.get_window_extent(self._renderer())
        except (AttributeError, TypeError, ValueError, RuntimeError):
            bbox = None
        if isinstance(self.target, Text):
            bbox_patch = self.target.get_bbox_patch()
            if bbox_patch is not None:
                face_alpha = bbox_patch.get_facecolor()[-1]
                edge_alpha = bbox_patch.get_edgecolor()[-1]
                if face_alpha > 0 or edge_alpha > 0:
                    self.target.update_bbox_position_size(self._renderer())
                    patch_bbox = bbox_patch.get_window_extent(self._renderer())
                    if bbox is None:
                        bbox = patch_bbox
                    else:
                        bbox = Bbox.from_extents(
                            min(bbox.x0, patch_bbox.x0),
                            min(bbox.y0, patch_bbox.y0),
                            max(bbox.x1, patch_bbox.x1),
                            max(bbox.y1, patch_bbox.y1),
                        )
        if bbox is not None:
            bounds = np.asarray(bbox.extents, dtype=float)
            if bounds.shape == (4,) and np.all(np.isfinite(bounds)):
                return np.array([bounds[:2], bounds[2:]], dtype=float)

        return self._bounds_points(np.asarray(self.get_positions(), dtype=float))

    def get_positions(
        self, use_previous_offset=False, update_offset=False
    ) -> (int, int, int, int):
        """Return writable control points in display coordinates."""
        preview = getattr(self.target, "_pylustrator_preview_positions", None)
        if preview is not None:
            return [np.array(point, dtype=float).copy() for point in preview]

        points = []
        if isinstance(self.target, Rectangle):
            points.append(self.target.get_xy())
            p2 = (
                self.target.get_x() + self.target.get_width(),
                self.target.get_y() + self.target.get_height(),
            )
            points.append(p2)
        elif isinstance(self.target, Ellipse):
            c = self.target.center
            w = self.target.width
            h = self.target.height
            points.append((c[0] - w / 2, c[1] - h / 2))
            points.append((c[0] + w / 2, c[1] + h / 2))
        elif isinstance(self.target, FancyArrowPatch):
            points.append(self.target._posA_posB[0])
            points.append(self.target._posA_posB[1])
        elif isinstance(self.target, FancyBboxPatch):
            points.append((self.target.get_x(), self.target.get_y()))
            points.append(
                (
                    self.target.get_x() + self.target.get_width(),
                    self.target.get_y() + self.target.get_height(),
                )
            )
        elif isinstance(self.target, RegularPolygon):
            points.append(self.target.xy)
        elif isinstance(self.target, Wedge):
            points.append(self.target.center)
        elif isinstance(self.target, Polygon):
            points.extend(self.target.get_xy())
        elif isinstance(self.target, PathPatch):
            points.extend(self.target.get_path().vertices)
        elif isinstance(self.target, Annotation):
            points.append(self.target.get_position())
            points.append(self.target.xy)
        elif isinstance(self.target, Text):
            points.append(self.target.get_position())
            if checkXLabel(self.target):
                points[0] = (points[0][0], self.label_y)
            elif checkYLabel(self.target):
                points[0] = (self.label_x, points[0][1])
        elif isinstance(self.target, Axes):
            p1, p2 = np.array(self.target.get_position())
            points.append(p1)
            points.append(p2)
        elif isinstance(self.target, Legend):
            points.append(legend_display_loc(self.target))
        elif isinstance(self.target, Line2D):
            points.extend(self.target.get_xydata())
        elif isinstance(self.target, AxesImage):
            left, right, bottom, top = self.target.get_extent()
            points.extend(((left, bottom), (right, top)))
        elif isinstance(self.target, self._collection_types):
            groups = self._collection_local_groups()
            if groups:
                points.extend(np.concatenate(groups))
        return self.transform_points(points)

    def refresh_offset(self):
        """Compatibility hook retained for older drag code."""

    def get_local_positions(
        self, use_previous_offset=False, update_offset=False
    ) -> list[np.ndarray]:
        """Return positions in the artist's own coordinate system.

        Dragging happens in display coordinates, but undo/redo restore points must
        survive later changes to a parent axes or figure transform.
        """
        points = self.get_positions(
            use_previous_offset=use_previous_offset, update_offset=update_offset
        )
        return [
            np.array(point, dtype=float).copy()
            for point in self.transform_inverted_points(points)
        ]

    def set_local_positions(self, points: list[np.ndarray]):
        """Restore positions captured with get_local_positions."""
        self.set_positions(self.transform_points(points))

    def get_restore_state(self):
        """Return a transform-independent state for undo/redo restore points."""
        label_axes = (
            checkXLabel(self.target) or checkYLabel(self.target)
            if isinstance(self.target, Text)
            else None
        )
        if label_axes is not None:
            axis_name = "x" if checkXLabel(self.target) is not None else "y"
            axis = label_axes.xaxis if axis_name == "x" else label_axes.yaxis
            return {
                "type": "axis_label",
                "axis": axis_name,
                "position": tuple(float(value) for value in self.target.get_position()),
                "labelpad": float(axis.labelpad),
            }
        if isinstance(self.target, Legend):
            bbox = self.target.get_bbox_to_anchor()
            transform = legend_anchor_transform(self.target)
            inverted = transform.inverted()
            p0 = inverted.transform(bbox.p0)
            p1 = inverted.transform(bbox.p1)
            return {
                "type": "legend",
                "is_point": legend_anchor_is_point(self.target),
                "anchor": (
                    float(p0[0]),
                    float(p0[1]),
                    float(p1[0] - p0[0]),
                    float(p1[1] - p0[1]),
                ),
                "transform": transform,
                "loc": self.target._loc,
            }
        return {"type": "positions", "positions": self.get_local_positions()}

    def restore_state(self, state):
        """Restore a state captured with get_restore_state."""
        if state["type"] == "axis_label":
            axes = checkXLabel(self.target) or checkYLabel(self.target)
            axis = axes.xaxis if state["axis"] == "x" else axes.yaxis
            self.target.set_position(state["position"])
            axis.labelpad = state["labelpad"]
            wrapper = TargetWrapper(self.target)
            if state["axis"] == "x":
                wrapper.label_y = self.target.get_position()[1]
                self.target.pad_offset = (
                    wrapper.label_y + axis.labelpad * wrapper.label_factor
                )
            else:
                wrapper.label_x = self.target.get_position()[0]
                self.target.pad_offset = (
                    wrapper.label_x + axis.labelpad * wrapper.label_factor
                )
            change_tracker = (
                self.figure.figure.change_tracker
                if self.figure.figure is not None
                else self.figure.change_tracker
            )
            change_tracker.addChange(
                axes, f".{state['axis']}axis.labelpad = {axis.labelpad:f}"
            )
            change_tracker.addNewTextChange(self.target)
            return
        if state["type"] == "legend":
            anchor = state["anchor"]
            if state["is_point"]:
                self.target.set_bbox_to_anchor(anchor[:2], transform=state["transform"])
            else:
                self.target.set_bbox_to_anchor(anchor, transform=state["transform"])
            self.target._loc = state["loc"]
            change_tracker = (
                self.figure.figure.change_tracker
                if self.figure.figure is not None
                else self.figure.change_tracker
            )
            change_tracker.addNewLegendChange(self.target)
            return
        self.set_local_positions(state["positions"])

    def set_positions(self, points: (int, int)):
        """set the position of the target Artist"""
        points = self.transform_inverted_points(points)

        if self.figure.figure is not None:
            change_tracker = self.figure.figure.change_tracker
        else:
            change_tracker = self.figure.change_tracker

        if isinstance(self.target, Rectangle):
            self.target.set_xy(points[0])
            self.target.set_width(points[1][0] - points[0][0])
            self.target.set_height(points[1][1] - points[0][1])
            if (
                self.target.get_label() is None
                or not self.target.get_label().startswith("_rect")
            ):
                change_tracker.addChange(
                    self.target, ".set_xy([%f, %f])" % tuple(self.target.get_xy())
                )
                change_tracker.addChange(
                    self.target, ".set_width(%f)" % self.target.get_width()
                )
                change_tracker.addChange(
                    self.target, ".set_height(%f)" % self.target.get_height()
                )
        elif isinstance(self.target, Ellipse):
            self.target.center = np.mean(points, axis=0)
            self.target.width = points[1][0] - points[0][0]
            self.target.height = points[1][1] - points[0][1]
            change_tracker.addChange(
                self.target, ".center = (%f, %f)" % tuple(self.target.center)
            )
            change_tracker.addChange(self.target, ".width = %f" % self.target.width)
            change_tracker.addChange(self.target, ".height = %f" % self.target.height)
        elif isinstance(self.target, FancyArrowPatch):
            self.target.set_positions(points[0], points[1])
            change_tracker.addChange(
                self.target,
                ".set_positions(%s, %s)" % (tuple(points[0]), tuple(points[1])),
            )
        elif isinstance(self.target, FancyBboxPatch):
            bounds = (
                float(points[0][0]),
                float(points[0][1]),
                float(points[1][0] - points[0][0]),
                float(points[1][1] - points[0][1]),
            )
            self.target.set_bounds(*bounds)
            change_tracker.addChange(self.target, f".set_bounds{bounds!r}")
        elif isinstance(self.target, RegularPolygon):
            self.target.xy = tuple(float(value) for value in points[0])
            self.target.stale = True
            change_tracker.addChange(self.target, f".xy = {self.target.xy!r}")
        elif isinstance(self.target, Wedge):
            center = tuple(float(value) for value in points[0])
            self.target.set_center(center)
            change_tracker.addChange(self.target, f".set_center({center!r})")
        elif isinstance(self.target, Polygon):
            vertices = [[float(x), float(y)] for x, y in points]
            self.target.set_xy(vertices)
            change_tracker.addChange(self.target, f".set_xy({vertices!r})")
        elif isinstance(self.target, PathPatch):
            old_path = self.target.get_path()
            vertices = np.asarray(points, dtype=float)
            codes = None if old_path.codes is None else old_path.codes.copy()
            self.target.set_path(Path(vertices, codes))
            vertices_literal = [[float(x), float(y)] for x, y in vertices]
            codes_literal = None if codes is None else [int(code) for code in codes]
            change_tracker.addChange(
                self.target,
                f".set_path(mpl.path.Path({vertices_literal!r}, {codes_literal!r}))",
            )
        elif isinstance(self.target, Line2D):
            new_xy = np.asarray(points, dtype=float)
            self.target.set_data(new_xy[:, 0], new_xy[:, 1])
            change_tracker.addChange(
                self.target,
                ".set_data(%s, %s)"
                % (
                    [float(value) for value in new_xy[:, 0]],
                    [float(value) for value in new_xy[:, 1]],
                ),
            )
        elif isinstance(self.target, Annotation):
            self.target.set_position(points[0])
            self.target.xy = tuple(float(value) for value in points[1])
            change_tracker.addNewTextChange(self.target)
            change_tracker.addChange(self.target, f".xy = {self.target.xy!r}")
        elif isinstance(self.target, Text):
            if checkXLabel(self.target):
                axes = checkXLabel(self.target)
                axes.xaxis.labelpad = (
                    self.target.pad_offset - points[0][1]
                ) / self.label_factor
                change_tracker.addChange(
                    axes, ".xaxis.labelpad = %f" % axes.xaxis.labelpad
                )

                self.target.set_position(points[0])
                self.label_y = points[0][1]
                change_tracker.addNewTextChange(self.target)
            elif checkYLabel(self.target):
                axes = checkYLabel(self.target)
                axes.yaxis.labelpad = (
                    self.target.pad_offset - points[0][0]
                ) / self.label_factor
                change_tracker.addChange(
                    axes, ".yaxis.labelpad = %f" % axes.yaxis.labelpad
                )

                self.target.set_position(points[0])
                self.label_x = points[0][0]
                change_tracker.addNewTextChange(self.target)
            else:
                self.target.set_position(points[0])
                change_tracker.addNewTextChange(self.target)
        elif isinstance(self.target, Legend):
            bbox = self.target.get_bbox_to_anchor()
            if bbox.width == 0 and bbox.height == 0:
                set_legend_point_anchor_display(
                    self.target, self.transform_inverted_points(points)[0]
                )
            else:
                point = legend_loc_transform(self.target).transform(
                    self.transform_inverted_points(points)[0]
                )
                self.target._loc = tuple(point)
            change_tracker.addNewLegendChange(self.target)
            # change_tracker.addChange(self.target, "._set_loc((%f, %f))" % tuple(point))
        elif isinstance(self.target, Axes):
            position = np.array([points[0], points[1] - points[0]]).flatten()
            if self.fixed_aspect:
                position[3] = (
                    position[2]
                    * self.target.get_position().height
                    / self.target.get_position().width
                )
            self.target.set_position(position)
            change_tracker.addNewAxesChange(self.target)
            # change_tracker.addChange(self.target, ".set_position([%f, %f, %f, %f])" % tuple(
            #    np.array([points[0], points[1] - points[0]]).flatten()))
        elif isinstance(self.target, AxesImage):
            extent = tuple(
                float(value)
                for value in (points[0][0], points[1][0], points[0][1], points[1][1])
            )
            axes = self.target.axes
            xlim = tuple(float(value) for value in axes.get_xlim())
            ylim = tuple(float(value) for value in axes.get_ylim())
            self.target.set_extent(extent)
            # AxesImage.set_extent may autoscale the parent axes.  A move should
            # transform the image inside its current canvas, not move the camera
            # with it and leave the pixels apparently stationary.
            axes.set_xlim(xlim)
            axes.set_ylim(ylim)
            from .change_tracker import getReference

            axes_reference = getReference(axes)
            change_tracker.addChange(
                self.target,
                f".set_extent({extent!r}), "
                f"{axes_reference}.set_xlim({xlim!r}), "
                f"{axes_reference}.set_ylim({ylim!r})",
            )
        elif isinstance(self.target, self._collection_types):
            lengths = [len(group) for group in self._collection_local_groups()]
            groups = []
            start = 0
            for length in lengths:
                groups.append(np.asarray(points[start : start + length], dtype=float))
                start += length
            literal = [[[float(x), float(y)] for x, y in group] for group in groups]
            if isinstance(self.target, PathCollection):
                offsets = groups[0] if groups else np.empty((0, 2))
                self.target.set_offsets(offsets)
                change_tracker.addChange(self.target, f".set_offsets({literal[0]!r})")
            elif isinstance(self.target, LineCollection):
                self.target.set_segments(groups)
                change_tracker.addChange(self.target, f".set_segments({literal!r})")
            else:
                codes = [path.codes for path in self.target.get_paths()]
                codes_literal = [
                    None if path_codes is None else [int(code) for code in path_codes]
                    for path_codes in codes
                ]
                self.target.set_verts_and_codes(groups, codes)
                change_tracker.addChange(
                    self.target,
                    f".set_verts_and_codes({literal!r}, {codes_literal!r})",
                )
        setattr(self.target, "_pylustrator_cached_get_extend", None)

    def get_extent(self):
        # get get_extent as it can be called very frequently when checking snap conditions
        if not getattr(self.target, "_pylustrator_cached_get_extend_added", False):
            setattr(self.target, "_pylustrator_cached_get_extend_added", True)
        if getattr(self.target, "_pylustrator_cached_get_extend", None) is None:
            setattr(self.target, "_pylustrator_cached_get_extend", self.do_get_extent())
        return getattr(self.target, "_pylustrator_cached_get_extend")

    def do_get_extent(self) -> (int, int, int, int):
        """get the extent of the target"""
        points = np.array(self.get_selection_points())
        return [
            np.min(points[:, 0]),
            np.min(points[:, 1]),
            np.max(points[:, 0]),
            np.max(points[:, 1]),
        ]

    def transform_points(self, points: (int, int)) -> (int, int):
        """Transform native control points to display coordinates."""
        if isinstance(self.target, Annotation):
            if len(points) == 0:
                return []
            transformed = [self.target.get_transform().transform(points[0])]
            if len(points) > 1:
                transformed.append(self._annotation_xy_transform().transform(points[1]))
            return transformed
        if isinstance(self.target, Line2D):
            return [self.target.get_transform().transform(point) for point in points]
        if isinstance(self.target, PathCollection):
            transform = self.target.get_offset_transform()
            return [transform.transform(point) for point in points]
        if isinstance(self.target, (LineCollection, PolyCollection)):
            transform = self.target.get_transform()
            return [transform.transform(point) for point in points]
        transform = self.get_transform()
        return [transform.transform(p) for p in points]

    def transform_inverted_points(self, points: (int, int)) -> (int, int):
        """Transform display control points back to native coordinates."""
        if isinstance(self.target, Annotation):
            if len(points) == 0:
                return []
            transformed = [self.target.get_transform().inverted().transform(points[0])]
            if len(points) > 1:
                transformed.append(
                    self._annotation_xy_transform().inverted().transform(points[1])
                )
            return transformed
        if isinstance(self.target, Line2D):
            transform = self.target.get_transform()
            return [transform.inverted().transform(point) for point in points]
        if isinstance(self.target, PathCollection):
            transform = self.target.get_offset_transform()
            return [transform.inverted().transform(point) for point in points]
        if isinstance(self.target, (LineCollection, PolyCollection)):
            transform = self.target.get_transform()
            return [transform.inverted().transform(point) for point in points]
        transform = self.get_transform()
        return [transform.inverted().transform(p) for p in points]


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
