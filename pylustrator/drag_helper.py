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

import sys
import time

import numpy as np
from matplotlib.artist import Artist
from matplotlib.figure import Figure, SubFigure
from matplotlib.axes import Axes
from matplotlib.collections import Collection
from matplotlib.image import AxesImage
from matplotlib.legend import Legend
from matplotlib.lines import Line2D
from matplotlib.text import Annotation, Text
from matplotlib.patches import Patch, Rectangle
from matplotlib.backend_bases import MouseEvent, KeyEvent
from typing import Iterable, Sequence
from qtpy import QtCore, QtGui, QtWidgets

from .artist_adapters import (
    ArtistAdapter,
    PatchAdapter,
    UnsupportedArtistError,
    invalidate_legend_owner_inventory,
    iter_figure_legends,
    iter_legend_children,
    legend_owner_snapshot,
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
from .change_tracker import ChangeTracker, add_text_default, getReference
from .components.plot_layout import scene_point_to_canvas_pixels
from .editor_model import EditorGroup, EditorScene
from .display_geometry import ArtistRoster, DisplayGeometryCache
from .interaction import (
    HitCandidate,
    HitStack,
    SelectionKernel,
    SelectionMode,
    TopHitStatus,
)
from .interaction_index import DisplaySpaceHitIndex
from .operations import OperationSupport, TransformIntent, TransformOperation
from .smart_guides import Axis, StaleGuideSnapshotError
from .smart_guide_ui import (
    create_smart_guide_drag_session,
    invalidate_smart_guide_cache,
    schedule_smart_guide_warmup,
)
from .transform_engine import TransformPlan
from .property_adapters import axis_tick_label_reference
from .property_transactions import PropertyOperation, PropertyPlan
from .commands import InteractionState, ObjectLocator, semantic_equal
from .lifecycle_commands import delete_selection
from .content_preview_cache import (
    DEFAULT_MAX_ARTISTS,
    DEFAULT_MEMORY_BUDGET_BYTES,
    DEFAULT_SOURCE_FINGERPRINT_BUDGET_BYTES,
    activate_content_preview,
    close_content_preview_cache,
    deactivate_content_preview,
    invalidate_content_preview_cache,
    schedule_content_preview_warmup,
    update_content_preview,
)
from pylustrator.change_tracker import UndoRedo

DIR_X0 = 1
DIR_Y0 = 2
DIR_X1 = 4
DIR_Y1 = 8

blit = False


# Only native containment implementations with a display-bounded contract are
# eligible for coarse indexing.  Custom ``contains`` or adapter ``hit_test``
# implementations remain always-tested, so extension authors cannot create an
# invisible false-negative by returning hits outside a conventional bbox.
_BOUNDED_NATIVE_CONTAINS = frozenset(
    {
        Figure.contains,
        SubFigure.contains,
        Axes.contains,
        Collection.contains,
        AxesImage.contains,
        Legend.contains,
        Line2D.contains,
        Annotation.contains,
        Text.contains,
        Patch.contains,
        EditorGroup.contains,
    }
)
_BOUNDED_ADAPTER_HIT_TESTS = frozenset(
    {ArtistAdapter.hit_test, PatchAdapter.hit_test}
)


class InteractionRollbackError(RuntimeError):
    """Raised when a preview transaction cannot fully restore its start state."""

    def __init__(self, failures):
        self.failures = tuple(failures)
        details = "; ".join(
            f"{type(target).__name__}: {error}"
            for target, error in self.failures
        )
        super().__init__(f"Interaction rollback was incomplete: {details}")


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
        self.smart_guide_session = None
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

    def cancel_event(self) -> None:
        """Stop pointer delivery without committing the active gesture."""

        if self.got_artist:
            self.got_artist = False
            connection = getattr(self, "_c1", None)
            if connection is not None:
                self.figure.canvas.mpl_disconnect(connection)
        for snap in self.snaps:
            snap.remove()
        self.snaps = []
        self._close_smart_guide_session()
        self.moved = False

    def _close_smart_guide_session(self) -> None:
        session = getattr(self, "smart_guide_session", None)
        self.smart_guide_session = None
        if session is not None:
            session.close()

    def clickedEvent(self, event: MouseEvent):
        """when the mouse is clicked"""
        self.parent.start_move()
        self.mouse_xy = (event.x, event.y)

        for s in self.snaps:
            s.remove()
        self.snaps = []
        self._close_smart_guide_session()

        whole_object = bool(
            self.dir & DIR_X0
            and self.dir & DIR_X1
            and self.dir & DIR_Y0
            and self.dir & DIR_Y1
        )
        if whole_object and bool(
            getattr(self.parent, "smart_guides_enabled", True)
        ):
            manager = getattr(self.figure, "figure_dragger", None)
            if manager is not None:
                try:
                    self.smart_guide_session = create_smart_guide_drag_session(
                        manager,
                        self.parent,
                        [target.target for target in self.targets],
                        tolerance_px=float(
                            getattr(
                                self.parent,
                                "smart_guide_tolerance_px",
                                5.0,
                            )
                        ),
                        include_equal_gaps=bool(
                            getattr(
                                self.parent,
                                "smart_guides_equal_gaps",
                                True,
                            )
                        ),
                        allow_cold_capture=bool(
                            getattr(
                                self.parent,
                                "smart_guides_allow_blocking_capture",
                                False,
                            )
                        ),
                    )
                except (
                    AttributeError,
                    IndexError,
                    LookupError,
                    TypeError,
                    ValueError,
                    RuntimeError,
                    np.linalg.LinAlgError,
                ):
                    self.smart_guide_session = None

        # Resize retains the legacy dimension snaps.  Translation uses the
        # generic indexed guide session when it could be built; falling back is
        # behavior-preserving and keeps third-party/custom Artists fail-open.
        if self.smart_guide_session is None:
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
        session = getattr(self, "smart_guide_session", None)
        self.smart_guide_session = None
        draw_pending = False
        try:
            # Commit the already accepted preview verbatim.  Re-solving at
            # release would make the final position jump away from the last
            # frame the user actually saw.
            self.parent.end_move()
            draw_pending = bool(getattr(self.parent, "has_moved", False))
        finally:
            if session is not None:
                session.close()
            manager = getattr(self.figure, "figure_dragger", None)
            if manager is not None and not draw_pending:
                # A committed move schedules a draw, whose post-layout event
                # starts the authoritative warmup.  Only a click/no-op needs a
                # warmup here because no draw is pending.
                try:
                    schedule_smart_guide_warmup(manager)
                except (
                    AttributeError,
                    IndexError,
                    LookupError,
                    TypeError,
                    ValueError,
                    RuntimeError,
                ):
                    # Guide warmup is an independent fail-open accelerator;
                    # the unchanged content token/pixmap remains reusable.
                    pass

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
            smart_guide_session=self.smart_guide_session,
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
        self.target_bounds = []
        self.selection_overlay_batch_threshold = 128
        self._batched_selection_overlay = False
        self._batched_selection_overlay_items = []
        self.lock_aspect_ratio = False
        self.reference_point = (0.5, 0.5)
        self._custom_rotation_pivot_inches = None
        self.alignment_reference_mode = "selection"
        self.alignment_key = None
        self.defer_artist_updates = True
        self.content_preview_enabled = True
        self.content_preview_max_artists = DEFAULT_MAX_ARTISTS
        self.content_preview_memory_budget_bytes = DEFAULT_MEMORY_BUDGET_BYTES
        self.content_preview_source_budget_bytes = (
            DEFAULT_SOURCE_FINGERPRINT_BUDGET_BYTES
        )
        self.smart_guides_enabled = True
        self.smart_guide_tolerance_px = 5.0
        self.smart_guides_equal_gaps = True
        self.smart_guides_allow_blocking_capture = False

        self.hide_grabber()

    def configure_target_overlay(self, target_count: int) -> bool:
        """Use a fixed number of scene items for a large target inventory."""

        threshold = max(int(self.selection_overlay_batch_threshold), 1)
        self._batched_selection_overlay = int(target_count) >= threshold
        if self._batched_selection_overlay:
            self._ensure_batched_selection_overlay()
        return self._batched_selection_overlay

    def _ensure_batched_selection_overlay(self):
        items = getattr(self, "_batched_selection_overlay_items", [])
        if items:
            return items

        outline = QtWidgets.QGraphicsPathItem(self.graphics_scene_myparent)
        outline.setPen(QtGui.QPen(QtGui.QColor("#1E88E5"), 3))
        outline.setBrush(QtGui.QBrush(QtGui.QColor(30, 136, 229, 32)))
        outline.setZValue(900)

        contrast = QtWidgets.QGraphicsPathItem(self.graphics_scene_myparent)
        contrast_pen = QtGui.QPen(QtGui.QColor("white"), 1)
        contrast_pen.setStyle(QtCore.Qt.DashLine)
        contrast.setPen(contrast_pen)
        contrast.setBrush(QtGui.QBrush(QtCore.Qt.NoBrush))
        contrast.setZValue(901)

        key = QtWidgets.QGraphicsPathItem(self.graphics_scene_myparent)
        key.setPen(QtGui.QPen(QtGui.QColor("#0D47A1"), 5))
        key.setBrush(QtGui.QBrush(QtGui.QColor(30, 136, 229, 32)))
        key.setZValue(905)
        items = [outline, contrast, key]
        self._batched_selection_overlay_items = items
        return items

    def _remove_batched_selection_overlay(self) -> None:
        for item in getattr(self, "_batched_selection_overlay_items", []):
            scene = item.scene()
            if scene is not None:
                scene.removeItem(item)
        self._batched_selection_overlay_items = []

    def _update_batched_selection_overlay(self) -> None:
        if not getattr(self, "_batched_selection_overlay", False):
            return
        outline, contrast, key_item = self._ensure_batched_selection_overlay()
        ordinary_path = QtGui.QPainterPath()
        key_path = QtGui.QPainterPath()
        key = (
            self.alignment_key
            if self.alignment_reference_mode == "key_object"
            else None
        )
        for target, bounds in zip(self.targets, self.target_bounds):
            if bounds is None:
                continue
            x0, y0, x1, y1 = (float(value) for value in bounds)
            path = key_path if target.target is key else ordinary_path
            path.addRect(QtCore.QRectF(x0, y0, x1 - x0, y1 - y0))
        outline.setPath(ordinary_path)
        contrast.setPath(ordinary_path)
        key_item.setPath(key_path)

    def add_target(self, target: Artist, update: bool = True):
        """add an artist to the selection"""
        if target in [wrapped.target for wrapped in self.targets]:
            return
        target = TargetWrapper(target)
        if (
            not getattr(self, "_batch_targets_prevalidated", False)
            and not target.supported
        ):
            return
        if not getattr(self, "_batch_add_targets", False):
            self._clear_custom_rotation_pivot()

        new_points = np.asarray(target.get_selection_points(), dtype=float)
        if (
            new_points.ndim != 2
            or new_points.shape[0] == 0
            or new_points.shape[1] < 2
            or not np.all(np.isfinite(new_points[:, :2]))
        ):
            return
        new_points = new_points[:, :2]

        self.targets.append(target)

        x0, y0, x1, y1 = (
            np.min(new_points[:, 0]),
            np.min(new_points[:, 1]),
            np.max(new_points[:, 0]),
            np.max(new_points[:, 1]),
        )
        self.target_bounds.append(np.array([x0, y0, x1, y1], dtype=float))
        if getattr(self, "_batched_selection_overlay", False):
            if update:
                self.update_extent()
            return
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
        manager = getattr(self.figure, "figure_dragger", None)
        if manager is not None:
            schedule_content_preview_warmup(manager)

    @selection_geometry_snapshot()
    def update_extent(self):
        """updates the extend of the selection to all the selected elements"""
        bounds = []
        self.target_bounds = [None] * len(self.targets)
        for index, target in enumerate(self.targets):
            new_points = np.array(target.get_selection_points())
            if new_points.ndim != 2 or not len(new_points):
                continue
            target_bounds = self._bounds_from_points([new_points])
            self.target_bounds[index] = target_bounds
            bounds.append(target_bounds)

        if not bounds:
            self.hide_grabber()
            self._update_batched_selection_overlay()
            return

        self._apply_target_bounds(bounds, refresh_capabilities=True)

    def _apply_target_bounds(
        self, bounds=None, *, refresh_capabilities: bool
    ) -> None:
        """Update the overall grabber from cached per-target display bounds."""

        if bounds is None:
            bounds = [value for value in self.target_bounds if value is not None]
        if not bounds:
            self.hide_grabber()
            return
        bounds = np.asarray(bounds, dtype=float)

        for grabber in self.grabbers:
            grabber.targets = self.targets
        self.rotation_grabber.targets = self.targets

        self.positions[0] = np.min(bounds[:, 0])
        self.positions[1] = np.min(bounds[:, 1])
        self.positions[2] = np.max(bounds[:, 2])
        self.positions[3] = np.max(bounds[:, 3])

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

        if refresh_capabilities or not hasattr(self, "_grabber_scale_supported"):
            self.update_grabber()
        else:
            self._position_grabbers(
                self._grabber_scale_supported,
                self._grabber_rotation_supported,
            )
        self._update_batched_selection_overlay()

    def refresh_targets_after_draw(self, target_indices: Iterable[int]) -> None:
        """Remeasure only targets whose layout is finalized during draw."""

        target_indices = tuple(dict.fromkeys(int(index) for index in target_indices))
        if not target_indices:
            return
        with selection_geometry_snapshot():
            self.update_selection_rectangles(target_indices=target_indices)
            self._apply_target_bounds(refresh_capabilities=False)

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

    @legend_owner_snapshot()
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

    def align_points(self, mode: str, *, spacing: float = None) -> bool:
        """Align visible bounds in display space using the active reference.

        The complete delta plan is measured and preflighted before any artist
        mutates. Selection alignment uses the selected objects' envelope,
        key-object alignment keeps the explicit key fixed, and artboard
        alignment uses the figure canvas bounds.
        """

        alignment_modes = {
            "left_x": (0, np.min),
            "center_x": (0, np.mean),
            "right_x": (0, np.max),
            "bottom_y": (1, np.min),
            "center_y": (1, np.mean),
            "top_y": (1, np.max),
        }
        distribution_modes = {"distribute_x": 0, "distribute_y": 1}
        if mode not in {*alignment_modes, *distribution_modes, "group"}:
            raise ValueError(f"Unknown alignment mode: {mode}")
        if len(self.targets) == 0:
            return False
        if spacing is not None:
            spacing = float(spacing)
            if not np.isfinite(spacing):
                raise ValueError("Distribution spacing must be finite")
            if mode not in distribution_modes:
                raise ValueError("Numeric spacing only applies to distribution")
            if self.alignment_reference_mode != "key_object":
                raise ValueError("Numeric spacing requires key-object alignment")

        if mode == "group":
            from pylustrator.helper_functions import axes_to_grid

            # return axes_to_grid([target.target for target in self.targets], track_changes=True)
            axes = [
                target.target
                for target in self.targets
                if isinstance(target.target, Axes)
            ]
            if not axes:
                return False
            with UndoRedo(axes, "Grid Align"):
                axes_to_grid(
                    axes,
                    track_changes=False,
                )
            return True

        key_wrapper = None
        if self.alignment_reference_mode == "key_object":
            if len(self.targets) < 2:
                raise ValueError(
                    "Select at least two objects for key-object alignment"
                )
            key_wrapper = self.alignment_key_wrapper()

        with selection_geometry_snapshot():
            items = self._resolve_alignment_items()
            measure_points = [
                self._measure_points(measure_target, move_target)
                for measure_target, move_target in items
            ]

            if mode in alignment_modes:
                axis, function = alignment_modes[mode]
                if self.alignment_reference_mode == "artboard":
                    bbox = self.figure.bbox
                    reference = np.array(
                        [bbox.x0, bbox.y0, bbox.x1, bbox.y1], dtype=float
                    )
                elif key_wrapper is not None:
                    key_index = next(
                        index
                        for index, (_measure, move) in enumerate(items)
                        if move.target is key_wrapper.target
                    )
                    reference = self._bounds_from_points(
                        [measure_points[key_index]]
                    )
                else:
                    reference = self._bounds_from_points(measure_points)
                current = [function(points[:, axis]) for points in measure_points]
                destination = function(reference[axis::2])
                deltas = [destination - value for value in current]
                edit_name = "Align"
            else:
                axis = distribution_modes[mode]
                if len(items) < 2:
                    raise ValueError("Select at least two objects to distribute")
                sizes = np.asarray(
                    [
                        self._measure_size(
                            points,
                            axis,
                            measure_target.target is move_target.target,
                        )
                        for points, (measure_target, move_target) in zip(
                            measure_points, items
                        )
                    ],
                    dtype=float,
                )
                positions = np.asarray(
                    [np.min(points[:, axis]) for points in measure_points],
                    dtype=float,
                )
                order = np.argsort(positions, kind="stable").tolist()
                deltas = [0.0] * len(items)

                if key_wrapper is not None:
                    key_index = next(
                        index
                        for index, (_measure, move) in enumerate(items)
                        if move.target is key_wrapper.target
                    )
                    key_rank = order.index(key_index)
                    if spacing is None:
                        current_gaps = [
                            positions[right]
                            - (positions[left] + sizes[left])
                            for left, right in zip(order, order[1:])
                        ]
                        gap = float(np.mean(current_gaps))
                    else:
                        gap = spacing

                    cursor = positions[key_index]
                    for index in reversed(order[:key_rank]):
                        cursor -= gap + sizes[index]
                        deltas[index] = cursor - positions[index]
                    cursor = positions[key_index] + sizes[key_index]
                    for index in order[key_rank + 1 :]:
                        cursor += gap
                        deltas[index] = cursor - positions[index]
                        cursor += sizes[index]
                else:
                    if self.alignment_reference_mode == "artboard":
                        bbox = self.figure.bbox
                        layout_low = (bbox.x0, bbox.y0)[axis]
                        layout_high = (bbox.x1, bbox.y1)[axis]
                    else:
                        if len(items) == 2:
                            # Two objects have no interior position to solve;
                            # preserve both exactly instead of inventing motion.
                            layout_low = positions[order[0]]
                            layout_high = (
                                positions[order[1]] + sizes[order[1]]
                            )
                        else:
                            layout_bounds = self._bounds_from_points(
                                measure_points
                            )
                            layout_low = layout_bounds[axis]
                            layout_high = layout_bounds[axis + 2]
                    gap = (
                        layout_high - layout_low - float(np.sum(sizes))
                    ) / (len(items) - 1)
                    cursor = layout_low
                    for index in order:
                        deltas[index] = cursor - positions[index]
                        cursor += sizes[index] + gap
                edit_name = "Distribute"

            plan = self._delta_plan(items, deltas)
            vectors = []
            for target, delta in plan:
                if (
                    key_wrapper is not None
                    and target.target is key_wrapper.target
                ) or np.isclose(delta, 0.0, atol=1e-12):
                    continue
                display_delta = np.zeros(2, dtype=float)
                display_delta[axis] = delta
                target.preflight_rigid_visible_translation(display_delta)
                vectors.append((target, display_delta))

        if not vectors:
            return False
        self.start_move(save_targets=[target for target, _delta in vectors])
        try:
            for target, display_delta in vectors:
                target.translate(display_delta)
            self.update_extent()
            self.has_moved = True
            self.end_move(edit_name)
        except Exception:
            self._restore_move_start(strict=False)
            self.has_moved = False
            self.end_move(edit_name)
            raise
        self.figure.canvas.draw()
        self.update_selection_rectangles()
        return True

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
        """Match visible size to the key object, or the first selected object."""
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
        reference = (
            self.alignment_key_wrapper()
            if self.alignment_reference_mode == "key_object"
            else self.targets[0]
        )
        resize_targets = [
            target for target in self.targets if target.target is not reference.target
        ]

        planned: list[tuple[TargetWrapper, np.ndarray]] = []
        with selection_geometry_snapshot():
            reference_points = np.array(
                reference.get_selection_points(), dtype=float
            )
            reference_bounds = self._points_bounds(reference_points)
            reference_width = reference_bounds[2] - reference_bounds[0]
            reference_height = reference_bounds[3] - reference_bounds[1]
            if match_width and reference_width <= 0:
                raise ValueError("Reference object has no width.")
            if match_height and reference_height <= 0:
                raise ValueError("Reference object has no height.")

            for target in resize_targets:
                bounds = self._points_bounds(
                    np.array(target.get_selection_points(), dtype=float)
                )
                current_width = bounds[2] - bounds[0]
                current_height = bounds[3] - bounds[1]
                if match_width and current_width <= 0:
                    raise ValueError("Selected object has no width.")
                if match_height and current_height <= 0:
                    raise ValueError("Selected object has no height.")
                scale_x = (
                    reference_width / current_width if match_width else 1.0
                )
                scale_y = (
                    reference_height / current_height if match_height else 1.0
                )
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
        self.start_move(save_targets=[target for target, _transform in planned])
        try:
            for target, transform in planned:
                target.resize(transform)
            self.update_extent()
            self.has_moved = True
            self.end_move("Resize")
        except Exception:
            self._restore_move_start(strict=False)
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
            self._restore_move_start(strict=False)
            self.has_moved = False
            self.end_move("Scale")
            raise
        self.figure.canvas.draw()
        self.update_selection_rectangles()
        return True

    def scale_appearance_selection(self, factor: float) -> bool:
        """Scale font/stroke/marker appearance without changing geometry."""

        factor = float(factor)
        if not np.isfinite(factor) or factor <= 0.0:
            raise ValueError("Appearance scale factor must be finite and positive")
        if not self.targets or factor == 1.0:
            return False
        plan = TransformPlan.preflight(
            [target.target for target in self.targets],
            TransformIntent.scale_appearance(factor),
        )
        save_targets = self._unique_wrappers(self.targets)
        store_start = self.get_save_point(
            save_targets, appearance_only=True
        )
        try:
            plan.commit()
            self.figure.canvas.draw()
            self.update_extent()
        except Exception as error:
            rollback_failures = []
            try:
                store_start()
            except Exception as rollback_error:
                rollback_failures.append((self, rollback_error))
            ArtistAdapter.annotate_rollback_failures(error, rollback_failures)
            raise
        store_end = self.get_save_point(
            save_targets, appearance_only=True
        )
        self.figure.signals.figure_selection_moved.emit()
        self.figure.change_tracker.addEdit(
            [store_start, store_end, "Scale appearance"]
        )
        self.update_selection_rectangles()
        dragger = getattr(self.figure, "figure_dragger", None)
        notify = getattr(dragger, "_notify_selected_element_changed", None)
        if callable(notify):
            notify()
        return True

    @staticmethod
    def _rotatable_value(target: Artist) -> float | None:
        wrapped = TargetWrapper(target)
        return (
            wrapped.get_rotation()
            if wrapped.supports_operation(TransformOperation.ROTATE)
            else None
        )

    def rotation_interaction_support(
        self, operation: TransformOperation | None = None
    ) -> OperationSupport:
        if operation is None:
            operation = self.rotation_operation()
        if operation is None:
            return OperationSupport.denied(
                TransformOperation.ROTATE, "No objects are selected"
            )
        if operation is TransformOperation.ROTATE:
            if len(self.targets) != 1:
                return OperationSupport.denied(
                    operation,
                    "Native visual rotation requires exactly one selected object",
                )
            return self.targets[0].native_rotation_handle_support()
        return self.operation_support(operation)

    def rotation_operation(self) -> TransformOperation | None:
        if not self.targets:
            return None
        if self.operation_support(TransformOperation.RIGID_ROTATE).supported:
            return TransformOperation.RIGID_ROTATE
        if (
            len(self.targets) == 1
            and self.targets[0].native_rotation_handle_support().supported
        ):
            return TransformOperation.ROTATE
        return (
            TransformOperation.ROTATE
            if len(self.targets) == 1
            else TransformOperation.RIGID_ROTATE
        )

    def rotation_handle_supported(self) -> bool:
        """Expose native single-object or common-pivot multi-object rotation."""

        if not self.targets:
            return False
        if len(self.targets) == 1:
            target = self.targets[0]
            return bool(
                target.operation_support(TransformOperation.RIGID_ROTATE).supported
                or target.native_rotation_handle_support().supported
            )
        return all(
            target.operation_support(TransformOperation.RIGID_ROTATE).supported
            for target in self.targets
        )

    def rotation_pivot(self) -> np.ndarray:
        if not self.rotation_handle_supported():
            raise ValueError(
                "Rotation handle requires native single-object rotation or a "
                "complete common-pivot geometry plan"
            )
        if self.rotation_operation() is TransformOperation.ROTATE:
            return np.asarray(self.targets[0].get_rotation_pivot(), dtype=float)
        custom = self.custom_rotation_pivot_position()
        if custom is not None:
            return custom
        return np.asarray(self.reference_position(), dtype=float)

    def start_rotation(self, event: MouseEvent) -> None:
        """Begin one deferred native or common-pivot rotation gesture."""

        if not self.rotation_handle_supported():
            operation = self.rotation_operation() or TransformOperation.ROTATE
            support = self.rotation_interaction_support(operation)
            raise ValueError(support.reason or "Rotation handle is unavailable")
        pivot = self.rotation_pivot()
        pointer = np.asarray((event.x, event.y), dtype=float)
        vector = pointer - pivot
        if not np.all(np.isfinite(vector)) or np.linalg.norm(vector) <= 1e-9:
            raise ValueError("Rotation handle is too close to its native pivot")

        self.start_move(save_targets=self.targets)
        self.rotation_drag_mode = (
            "native"
            if self.rotation_operation() is TransformOperation.ROTATE
            else "rigid"
        )
        self.rotation_drag_pivot = pivot
        self.rotation_drag_start_pointer_angle = float(
            np.degrees(np.arctan2(vector[1], vector[0]))
        )
        if self.rotation_drag_mode == "native":
            self.rotation_drag_target = self.targets[0]
            self.rotation_drag_start_value = self.rotation_drag_target.get_rotation()
            self.rotation_drag_preview_value = self.rotation_drag_start_value
        else:
            self.rotation_drag_preview_delta = 0.0
            self.rotation_drag_plans = ()

    def preview_rotation(self, event: MouseEvent) -> float:
        """Preview the exact native or shared rigid rotation destination."""

        mode = getattr(self, "rotation_drag_mode", None)
        if mode is None:
            raise RuntimeError("No rotation gesture is active")
        pointer = np.asarray((event.x, event.y), dtype=float)
        vector = pointer - self.rotation_drag_pivot
        if not np.all(np.isfinite(vector)) or np.linalg.norm(vector) <= 1e-9:
            return (
                self.rotation_drag_preview_value
                if mode == "native"
                else self.rotation_drag_preview_delta
            )
        pointer_angle = float(np.degrees(np.arctan2(vector[1], vector[0])))
        delta = (
            pointer_angle - self.rotation_drag_start_pointer_angle + 180.0
        ) % 360.0 - 180.0
        if _event_has_modifier(event, "shift"):
            delta = round(delta / 15.0) * 15.0
        try:
            if mode == "native":
                value = self.rotation_drag_start_value + delta
                with suspend_change_recording():
                    self.rotation_drag_target.set_rotation(value)
                self.rotation_drag_preview_value = value
                self.has_moved = not np.isclose(
                    value, self.rotation_drag_start_value
                )
                result = value
            else:
                self._restore_move_start()
                with selection_geometry_snapshot():
                    plans = tuple(
                        target.plan_rigid_rotation(
                            delta,
                            self.rotation_drag_pivot,
                            control_points=self.move_start_positions.get(
                                id(target.target)
                            ),
                            selection_points=self.move_start_raw_selection_points.get(
                                id(target.target)
                            ),
                        )
                        for target in self.targets
                    )
                for target, plan in zip(self.targets, plans):
                    target._apply_prevalidated_rigid_rotation_plan(
                        plan,
                        record_changes=False,
                    )
                self.rotation_drag_plans = plans
                self.rotation_drag_preview_delta = delta
                self.has_moved = not np.isclose(delta, 0.0)
                result = delta
        except Exception:
            self._restore_move_start(strict=False)
            self.has_moved = False
            self.end_move("Rotate")
            self._clear_rotation_gesture()
            self.rotation_grabber.cancel_event()
            raise
        self.update_extent()
        self.update_selection_rectangles()
        self.hide_grabber()
        canvas = self.figure.canvas
        if hasattr(canvas, "schedule_draw"):
            canvas.schedule_draw()
        else:
            canvas.draw_idle()
        return result

    def _clear_rotation_gesture(self) -> None:
        for name in (
            "rotation_drag_target",
            "rotation_drag_mode",
            "rotation_drag_pivot",
            "rotation_drag_start_pointer_angle",
            "rotation_drag_start_value",
            "rotation_drag_preview_value",
            "rotation_drag_preview_delta",
            "rotation_drag_plans",
        ):
            try:
                delattr(self, name)
            except AttributeError:
                pass

    def end_rotation(self) -> bool:
        """Commit one generated change and one undo item for a handle gesture."""

        mode = getattr(self, "rotation_drag_mode", None)
        if mode is None:
            return False
        changed = (
            not np.isclose(
                self.rotation_drag_preview_value,
                self.rotation_drag_start_value,
            )
            if mode == "native"
            else not np.isclose(self.rotation_drag_preview_delta, 0.0)
        )
        try:
            if changed:
                if mode == "native":
                    # The preview was deliberately unrecorded. Reapplying the
                    # final absolute value emits one stable generated change.
                    self.rotation_drag_target.set_rotation(
                        self.rotation_drag_preview_value
                    )
                else:
                    self._restore_move_start()
                    with selection_geometry_snapshot():
                        prepared_plans = tuple(
                            target.revalidate_rigid_rotation_plan(plan)
                            for target, plan in zip(
                                self.targets, self.rotation_drag_plans
                            )
                        )
                    for target, plan in zip(self.targets, prepared_plans):
                        target._apply_prevalidated_rigid_rotation_plan(plan)
                self.update_extent()
                self.has_moved = True
            else:
                self.has_moved = False
            self.end_move("Rotate")
        except Exception:
            self._restore_move_start(strict=False)
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

    def cancel_rotation(self) -> bool:
        """Restore the pre-gesture transaction and discard a rotation preview."""

        if getattr(self, "rotation_drag_mode", None) is None:
            return False
        rollback_error = None
        try:
            try:
                self._restore_move_start()
            except InteractionRollbackError as error:
                rollback_error = error
            self.has_moved = False
            self.end_move("Rotate")
        finally:
            self._clear_rotation_gesture()
        self.figure.canvas.draw()
        self.update_selection_rectangles()
        if rollback_error is not None:
            raise rollback_error
        return True

    def rotate_selection(self, angle_degrees: float) -> bool:
        """Rotate one native object or a multi-selection around one pivot."""
        if len(self.targets) == 0:
            return False
        angle_degrees = float(angle_degrees)
        if not np.isfinite(angle_degrees):
            raise ValueError("Rotation angle must be finite")
        if np.isclose(angle_degrees % 360.0, 0.0, atol=1e-12):
            return False

        operation = self.rotation_operation()
        if operation is None:
            return False
        support = self.rotation_interaction_support(operation)
        if not support.supported:
            raise UnsupportedArtistError(support.reason)

        plans = ()
        if operation is TransformOperation.RIGID_ROTATE:
            pivot = self.rotation_pivot()
            with selection_geometry_snapshot():
                plans = tuple(
                    target.plan_rigid_rotation(angle_degrees, pivot)
                    for target in self.targets
                )

        self.start_move(save_targets=self.targets)
        try:
            if operation is TransformOperation.ROTATE:
                target = self.targets[0]
                target.set_rotation(target.get_rotation() + angle_degrees)
            else:
                with selection_geometry_snapshot():
                    prepared_plans = tuple(
                        target.revalidate_rigid_rotation_plan(plan)
                        for target, plan in zip(self.targets, plans)
                    )
                for target, plan in zip(self.targets, prepared_plans):
                    target._apply_prevalidated_rigid_rotation_plan(plan)
            self.update_extent()
            self.has_moved = True
            self.end_move("Rotate")
        except Exception:
            self._restore_move_start(strict=False)
            self.has_moved = False
            self.end_move("Rotate")
            raise
        signal = getattr(self.figure.signals, "figure_selection_property_changed", None)
        if signal is not None:
            signal.emit()
        self.figure.canvas.draw()
        self.update_selection_rectangles()
        return True

    def delete_targets(self):
        """Delete all selected targets as one atomic lifecycle command."""
        targets = [target.target for target in self.targets]
        if delete_selection(self.figure.figure_dragger, targets):
            self.figure.canvas.draw()

    def update_selection_rectangles(
        self, use_previous_offset=False, target_indices: Iterable[int] = None
    ):
        """update the selection visualisation"""
        if len(self.targets) == 0:
            return
        indices = (
            range(len(self.targets))
            if target_indices is None
            else (
                index
                for index in target_indices
                if 0 <= index < len(self.targets)
            )
        )
        if getattr(self, "_batched_selection_overlay", False):
            for index in indices:
                target = self.targets[index]
                new_points = None
                if use_previous_offset:
                    new_points = getattr(
                        self, "move_current_selection_points", {}
                    ).get(id(target.target))
                if new_points is None:
                    new_points = np.asarray(target.get_selection_points())
                if new_points.ndim != 2 or not len(new_points):
                    self.target_bounds[index] = None
                    continue
                self.target_bounds[index] = self._bounds_from_points([new_points])
            self._update_batched_selection_overlay()
            return
        if 0:
            for index in indices:
                target = self.targets[index]
                new_points = np.array(target.get_positions())
                for i in range(2):
                    rect = self.targets_rects[index * 2 + i]
                    rect.set_xy(new_points[0])
                    rect.set_width(new_points[1][0] - new_points[0][0])
                    rect.set_height(new_points[1][1] - new_points[0][1])
        else:
            for index in indices:
                target = self.targets[index]
                new_points = None
                if use_previous_offset:
                    new_points = getattr(self, "move_current_selection_points", {}).get(
                        id(target.target)
                    )
                if new_points is None:
                    new_points = np.array(target.get_selection_points())
                if new_points.ndim != 2 or not len(new_points):
                    self.target_bounds[index] = None
                    for i in range(2):
                        self.targets_rects[index * 2 + i].setRect(-100, -100, 0, 0)
                    continue
                x0, y0, x1, y1 = (
                    np.min(new_points[:, 0]),
                    np.min(new_points[:, 1]),
                    np.max(new_points[:, 0]),
                    np.max(new_points[:, 1]),
                )
                self.target_bounds[index] = np.array(
                    [x0, y0, x1, y1], dtype=float
                )
                w0, h0 = x1 - x0, y1 - y0
                for i in range(2):
                    rect = self.targets_rects[index * 2 + i]
                    rect.setRect(x0, y0, w0, h0)
        self._update_alignment_key_style()

    def _update_alignment_key_style(self) -> None:
        """Draw the active key object with Illustrator-style emphasis."""

        if getattr(self, "_batched_selection_overlay", False):
            self._update_batched_selection_overlay()
            return

        key = (
            self.alignment_key
            if self.alignment_reference_mode == "key_object"
            else None
        )
        for index, target in enumerate(self.targets):
            rect_index = index * 2
            if rect_index >= len(self.targets_rects):
                continue
            is_key = target.target is key
            pen = QtGui.QPen(QtGui.QColor("#0D47A1" if is_key else "#1E88E5"))
            pen.setWidth(5 if is_key else 3)
            rect = self.targets_rects[rect_index]
            rect.setPen(pen)
            rect.setZValue(905 if is_key else 900)

    def remove_target(self, target: Artist):
        """remove an artist from the current selection"""
        targets_non_wrapped = [t.target for t in self.targets]
        if target not in targets_non_wrapped:
            return
        self.rotation_grabber.cancel_pivot_event(restore=True)
        index = targets_non_wrapped.index(target)
        self._clear_preview(self.targets[index])
        self.targets.pop(index)
        self.target_bounds.pop(index)
        if not getattr(self, "_batched_selection_overlay", False):
            rect1 = self.targets_rects.pop(index * 2)
            rect2 = self.targets_rects.pop(index * 2)
            rect1.scene().removeItem(rect1)
            rect2.scene().removeItem(rect2)
        self._clear_custom_rotation_pivot()
        # self.figure.patches.remove(rect1)
        # self.figure.patches.remove(rect2)
        if len(self.targets) == 0:
            self.clear_targets()
        else:
            if self.alignment_key is target:
                self.alignment_key = None
            self.update_extent()
            self.update_selection_rectangles()
        self._notify_alignment_state_changed()
        manager = getattr(self.figure, "figure_dragger", None)
        if manager is not None and self.targets:
            schedule_content_preview_warmup(manager)

    def update_grabber(self):
        """update the position of the grabber elements"""
        self._grabber_scale_supported = self.do_target_scale()
        self._grabber_rotation_supported = self.rotation_handle_supported()
        if (
            self._custom_rotation_pivot_inches is not None
            and not self.custom_rotation_pivot_supported()
        ):
            self._clear_custom_rotation_pivot()
        self._position_grabbers(
            self._grabber_scale_supported,
            self._grabber_rotation_supported,
        )

    def _position_grabbers(self, can_scale: bool, can_rotate: bool) -> None:
        """Reposition handles without re-evaluating immutable capabilities."""

        if can_scale:
            for grabber in self.grabbers:
                grabber.updatePos()
        else:
            for grabber in self.grabbers:
                grabber.set_xy((-100, -100))
        if can_rotate:
            self.rotation_grabber.updatePos()
        else:
            self.rotation_grabber.hide()

    def hide_grabber(self):
        """hide the grabber elements"""
        for grabber in self.grabbers:
            grabber.set_xy((-100, -100))
        self.rotation_grabber.hide()

    def clear_targets(self, *, preserve_rotation_pivot: bool = False):
        """remove all elements from the selection"""
        self.rotation_grabber.cancel_pivot_event(restore=True)
        self.clear_move_previews()
        manager = getattr(self.figure, "figure_dragger", None)
        if manager is not None:
            invalidate_content_preview_cache(manager, "selection-cleared")
        for rect in self.targets_rects:
            self.graphics_scene.scene().removeItem(rect)
            # self.figure.patches.remove(rect)
        self.targets_rects = []
        self._remove_batched_selection_overlay()
        self._batched_selection_overlay = False
        self.targets = []
        self.target_bounds = []
        self.alignment_key = None
        if not preserve_rotation_pivot:
            self._clear_custom_rotation_pivot()

        self.hide_grabber()

    def do_target_scale(self) -> bool:
        """Only expose resize handles when every selected artist can scale."""
        return bool(self.targets) and all(
            target.operation_support(TransformOperation.RESIZE_GEOMETRY).supported
            for target in self.targets
        )

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

    def _notify_alignment_state_changed(self) -> None:
        signals = getattr(self.figure, "signals", None)
        signal = getattr(signals, "figure_selection_update", None)
        if signal is not None:
            signal.emit()

    def _restore_alignment_reference_state(
        self, mode: str, key: Artist | None
    ) -> None:
        """Restore reference UI state without creating a document edit."""

        if mode not in {"selection", "key_object", "artboard"}:
            mode = "selection"
        selected = [target.target for target in self.targets]
        if mode == "key_object" and len(selected) < 2:
            # A key object only has meaning inside a multi-selection.  Keep
            # the interaction state valid when selection changes or an older
            # session restores a now-impossible key-object mode.
            mode = "selection"
            key = None
        self.alignment_reference_mode = mode
        self.alignment_key = (
            key
            if mode == "key_object"
            and any(target is key for target in selected)
            else None
        )
        self._update_alignment_key_style()
        self._notify_alignment_state_changed()

    def set_alignment_reference(
        self, mode: str, *, key: Artist | None = None
    ) -> str:
        """Choose selection, key-object, or artboard alignment semantics."""

        mode = str(mode).lower()
        if mode not in {"selection", "key_object", "artboard"}:
            raise ValueError(f"Unknown alignment reference mode: {mode!r}")
        if mode == "key_object":
            if len(self.targets) < 2:
                raise ValueError(
                    "Select at least two objects before choosing key-object alignment"
                )
            if key is None:
                primary = getattr(
                    getattr(self.figure, "figure_dragger", None),
                    "selected_element",
                    None,
                )
                if any(target.target is primary for target in self.targets):
                    key = primary
                else:
                    key = self.targets[-1].target
            if isinstance(key, TargetWrapper):
                key = key.target
            if not any(target.target is key for target in self.targets):
                raise ValueError("The key object must be part of the current selection")
            new_key = key
        else:
            new_key = None
        self._restore_alignment_reference_state(mode, new_key)
        return mode

    def set_alignment_key(self, artist: Artist) -> Artist:
        """Designate one already-selected object without mutating the figure."""

        if len(self.targets) < 2:
            raise ValueError("Select at least two objects before choosing a key object")
        if isinstance(artist, TargetWrapper):
            artist = artist.target
        if not any(target.target is artist for target in self.targets):
            raise ValueError("The key object must be part of the current selection")
        self.alignment_key = artist
        self._update_alignment_key_style()
        self._notify_alignment_state_changed()
        return artist

    def alignment_key_wrapper(self) -> TargetWrapper:
        if self.alignment_reference_mode != "key_object":
            raise ValueError("Key-object alignment is not active")
        for target in self.targets:
            if target.target is self.alignment_key:
                return target
        raise ValueError("Choose a selected key object before aligning")

    def set_reference_point(self, point: Sequence[float]) -> tuple[float, float]:
        """Set the normalized transform-panel anchor without mutating the figure."""

        point = tuple(float(value) for value in point)
        if len(point) != 2 or not np.all(np.isfinite(point)):
            raise ValueError("Reference point must contain two finite values")
        if any(value not in (0.0, 0.5, 1.0) for value in point):
            raise ValueError("Reference point values must use the 3x3 transform grid")
        self.reference_point = point
        self._clear_custom_rotation_pivot()
        if self.rotation_handle_supported():
            self.rotation_grabber.updatePos()
        self._notify_alignment_state_changed()
        return point

    def custom_rotation_pivot_supported(self) -> bool:
        """Return whether the selection can honor an arbitrary shared pivot."""

        return bool(
            self.rotation_operation() is TransformOperation.RIGID_ROTATE
            and self.rotation_interaction_support(
                TransformOperation.RIGID_ROTATE
            ).supported
        )

    def custom_rotation_pivot_state(self) -> tuple[float, float] | None:
        """Return the non-document pivot in root-Figure physical inches."""

        if self._custom_rotation_pivot_inches is None:
            return None
        return tuple(self._custom_rotation_pivot_inches)

    def custom_rotation_pivot_position(self) -> np.ndarray | None:
        """Resolve the custom pivot to current display pixels, if one is active."""

        state = self.custom_rotation_pivot_state()
        if state is None:
            return None
        position = np.asarray(self.figure.dpi_scale_trans.transform(state), dtype=float)
        if position.shape != (2,) or not np.all(np.isfinite(position)):
            return None
        return position

    def _set_custom_rotation_pivot_state(
        self, state: Sequence[float] | None, *, notify: bool = True
    ) -> None:
        if state is None:
            self._custom_rotation_pivot_inches = None
        else:
            state = tuple(float(value) for value in state)
            if len(state) != 2 or not np.all(np.isfinite(state)):
                raise ValueError("Custom rotation pivot state must be finite")
            if not self.custom_rotation_pivot_supported():
                raise UnsupportedArtistError(
                    "A custom rotation pivot requires a complete shared "
                    "rigid-rotation plan"
                )
            self._custom_rotation_pivot_inches = state
        if self.rotation_handle_supported():
            self.rotation_grabber.updatePos()
        if notify:
            self._notify_alignment_state_changed()

    def _clear_custom_rotation_pivot(self) -> None:
        self._custom_rotation_pivot_inches = None

    def set_rotation_pivot(
        self,
        display_position: Sequence[float],
        *,
        notify: bool = True,
    ) -> tuple[float, float]:
        """Set a shared rotation pivot without creating a document edit."""

        if getattr(self, "rotation_drag_mode", None) is not None:
            raise RuntimeError("Cannot move the pivot during a rotation gesture")
        if not self.custom_rotation_pivot_supported():
            raise UnsupportedArtistError(
                "A custom rotation pivot requires a complete shared "
                "rigid-rotation plan"
            )
        display_position = np.asarray(display_position, dtype=float)
        if display_position.shape != (2,) or not np.all(np.isfinite(display_position)):
            raise ValueError("Rotation pivot must contain two finite display values")
        physical_position = np.asarray(
            self.figure.dpi_scale_trans.inverted().transform(display_position),
            dtype=float,
        )
        self._set_custom_rotation_pivot_state(
            physical_position, notify=notify
        )
        return tuple(float(value) for value in display_position)

    def reset_rotation_pivot(self) -> bool:
        """Return rigid rotation to the active 3x3 reference point."""

        changed = self._custom_rotation_pivot_inches is not None
        self._set_custom_rotation_pivot_state(None)
        return changed

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
            self._restore_move_start(strict=False)
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
            self._restore_move_start(strict=False)
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

    @selection_geometry_snapshot()
    def get_save_point(
        self,
        targets: Iterable[TargetWrapper] = None,
        *,
        appearance_only: bool = False,
    ) -> callable:
        """gather the current positions in a restore point for the undo function"""
        selected_targets = [target.target for target in self.targets]
        alignment_reference_mode = self.alignment_reference_mode
        alignment_key = self.alignment_key
        reference_point = tuple(self.reference_point)
        custom_rotation_pivot = self.custom_rotation_pivot_state()
        dragger = getattr(self.figure, "figure_dragger", None)
        selected_primary = getattr(dragger, "selected_element", None)
        wrapped_targets = self._unique_wrappers(targets or self.targets)
        restore_targets = [target.target for target in wrapped_targets]
        states = [
            (
                target.get_appearance_state()
                if appearance_only
                else target.get_restore_state()
            )
            for target in wrapped_targets
        ]
        tracker = getattr(self.figure, "change_tracker", None)
        capture = getattr(tracker, "capture_recording_state", None)
        recording_state = capture() if capture is not None else None

        @legend_owner_snapshot()
        def undo():
            self.clear_targets()
            for target, state in zip(restore_targets, states):
                target = TargetWrapper(target)
                if appearance_only:
                    target.restore_appearance_state(
                        state, record_changes=recording_state is None
                    )
                else:
                    target.restore_state(
                        state, record_changes=recording_state is None
                    )
            restore_recording = getattr(tracker, "restore_recording_state", None)
            if recording_state is not None and restore_recording is not None:
                restore_recording(recording_state)
            for target in selected_targets:
                self.add_target(target, update=False)
            if self.targets and not bool(
                getattr(dragger, "_selection_refresh_on_draw", False)
            ):
                self.update_extent()
            if dragger is not None:
                dragger.selected_element = (
                    selected_primary
                    if any(target is selected_primary for target in selected_targets)
                    else None
                )
            self._restore_alignment_reference_state(
                alignment_reference_mode, alignment_key
            )
            self.set_reference_point(reference_point)
            if custom_rotation_pivot is not None:
                self._set_custom_rotation_pivot_state(
                    custom_rotation_pivot
                )

        return undo

    @selection_geometry_snapshot()
    def start_move(self, save_targets: Iterable[TargetWrapper] = None):
        """start to move a grabber"""
        self.start_p1 = self.p1.copy()
        self.start_p2 = self.p2.copy()
        self.start_inv_transform = self.get_inv_trans_matrix()
        self.hide_grabber()
        self.has_moved = False
        self.move_rollback_failures = ()
        self.defer_current_move = bool(self.defer_artist_updates)
        self.save_targets = self._unique_wrappers(save_targets or self.targets)
        for target in self._unique_wrappers(list(self.targets) + self.save_targets):
            target.refresh_offset()
        self.move_start_positions = {
            id(target.target): np.array(target.get_positions(), dtype=float)
            for target in self.targets
        }
        self.move_start_raw_selection_points = {}
        self.move_start_selection_points = {}
        for target in self.targets:
            raw = np.asarray(target.adapter.selection_points(), dtype=float)
            key = id(target.target)
            self.move_start_raw_selection_points[key] = raw
            self.move_start_selection_points[key] = np.asarray(
                target.adapter.clip_selection_points(raw), dtype=float
            )
        self.move_current_positions = {}
        self.move_current_selection_points = {}
        self.move_start_states = {
            id(target.target): target.get_restore_state()
            for target in self.save_targets
        }
        tracker = getattr(self.figure, "change_tracker", None)
        capture = getattr(tracker, "capture_recording_state", None)
        self.move_start_tracker_state = capture() if capture is not None else None
        self.move_start_edit_history_state = None
        if hasattr(tracker, "edits") and hasattr(tracker, "last_edit"):
            self.move_start_edit_history_state = (
                self._copy_edit_history(tracker.edits),
                int(tracker.last_edit),
            )
        self.move_start_reference_point = tuple(self.reference_point)
        self.move_start_custom_rotation_pivot = self.custom_rotation_pivot_state()
        self.move_start_ui_state_captured = True

        self.store_start = self.get_save_point(self.save_targets)
        if self.defer_current_move:
            activate_content_preview(self)

    @staticmethod
    def _copy_edit_history(edits) -> list:
        copied = []
        for edit in edits:
            if not isinstance(edit, list):
                copied.append(edit)
                continue
            item = list(edit)
            if len(item) > 3 and isinstance(item[3], dict):
                item[3] = dict(item[3])
            copied.append(item)
        return copied

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
        deactivate_content_preview(self)
        for target in self._unique_wrappers(getattr(self, "targets", [])):
            self._clear_preview(target)

    def _set_preview_positions(
        self,
        target: TargetWrapper,
        points: np.ndarray,
        selection_points: np.ndarray = None,
    ):
        # Keep one owned, contiguous buffer.  A Line2D can expose hundreds of
        # thousands of control points; allocating one tiny ndarray per vertex
        # dominated deferred-drag frame time and memory without adding any
        # isolation beyond a single array copy.
        target.target._pylustrator_preview_positions = np.array(
            points, dtype=float, copy=True, order="C"
        )
        if selection_points is not None:
            target.target._pylustrator_preview_selection_points = np.array(
                selection_points, dtype=float, copy=True, order="C"
            )
        setattr(target.target, "_pylustrator_cached_get_extend", None)

    @legend_owner_snapshot()
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
            pending.append(
                (
                    target,
                    points,
                    translation_delta,
                    start,
                    self.move_start_raw_selection_points.get(id(target.target)),
                    self.move_current_selection_points.get(id(target.target)),
                )
            )

        # Preview geometry is already at the proposed destination. Clear every
        # preview before validating the display delta, otherwise adapters would
        # add the delta to the preview a second time. Preflight the complete
        # transaction before mutating any target.
        for (
            target,
            _points,
            _translation_delta,
            _start,
            _selection,
            _destination,
        ) in pending:
            self._clear_preview(target)
        for (
            target,
            _points,
            translation_delta,
            start,
            selection_points,
            destination_selection_points,
        ) in pending:
            if translation_delta is not None:
                target.preflight_translation(
                    translation_delta,
                    control_points=start,
                    selection_points=selection_points,
                    destination_selection_points=destination_selection_points,
                )
        for (
            target,
            points,
            _translation_delta,
            _start,
            _selection,
            _destination,
        ) in pending:
            target.set_positions(points)

    def _move_changed_semantically(self) -> bool:
        start_states = getattr(self, "move_start_states", {})
        for target in self.save_targets:
            before = start_states.get(id(target.target))
            if before is None or not semantic_equal(before, target.get_restore_state()):
                return True
        return False

    def _restore_move_start(self, *, strict: bool = True) -> tuple:
        """Best-effort rollback that never strands earlier targets or previews."""

        failures = []
        start_states = getattr(self, "move_start_states", {})
        for target in reversed(self.save_targets):
            state = start_states.get(id(target.target))
            if state is not None:
                try:
                    target.restore_state(state, record_changes=False)
                except Exception as error:
                    failures.append((target.target, error))
        tracker_state = getattr(self, "move_start_tracker_state", None)
        tracker = getattr(self.figure, "change_tracker", None)
        edit_history_state = getattr(
            self, "move_start_edit_history_state", None
        )
        if edit_history_state is not None:
            try:
                edits, last_edit = edit_history_state
                tracker.edits = self._copy_edit_history(edits)
                tracker.last_edit = int(last_edit)
            except Exception as error:
                failures.append((tracker, error))
        restore = getattr(tracker, "restore_recording_state", None)
        if tracker_state is not None and restore is not None:
            try:
                restore(tracker_state)
            except Exception as error:
                failures.append((tracker, error))
        if bool(getattr(self, "move_start_ui_state_captured", False)):
            self.reference_point = tuple(self.move_start_reference_point)
            self._custom_rotation_pivot_inches = (
                self.move_start_custom_rotation_pivot
            )
        try:
            self.clear_move_previews()
        except Exception as error:
            failures.append((self, error))
        if self.targets:
            try:
                self.update_extent()
                self.update_selection_rectangles()
            except Exception as error:
                failures.append((self, error))

        self.move_rollback_failures = tuple(failures)
        active_error = sys.exc_info()[1]
        if failures and active_error is not None:
            try:
                active_error.pylustrator_rollback_failures = tuple(failures)
            except (AttributeError, TypeError):
                pass
            add_note = getattr(active_error, "add_note", None)
            if callable(add_note):
                details = "; ".join(
                    f"{type(target).__name__}: {error}"
                    for target, error in failures
                )
                add_note(f"Pylustrator rollback failures: {details}")
        if failures and strict:
            raise InteractionRollbackError(failures)
        return tuple(failures)

    def _clear_move_transaction(self) -> None:
        deactivate_content_preview(self)
        self.move_start_positions = {}
        self.move_start_raw_selection_points = {}
        self.move_start_selection_points = {}
        self.move_current_positions = {}
        self.move_current_selection_points = {}
        self.move_start_states = {}
        self.move_start_tracker_state = None
        self.move_start_edit_history_state = None
        self.move_start_reference_point = None
        self.move_start_custom_rotation_pivot = None
        self.move_start_ui_state_captured = False
        self.defer_current_move = False

    @legend_owner_snapshot()
    def end_move(self, edit_name: str = "Move", coalesce_key: str = None):
        """a grabber move stopped"""
        try:
            if self.has_moved is True:
                self._commit_deferred_positions()
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
                            "targets": tuple(
                                id(target.target) for target in self.save_targets
                            ),
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
        except Exception:
            self._restore_move_start(strict=False)
            self.has_moved = False
            self._clear_move_transaction()
            dragger = getattr(self.figure, "figure_dragger", None)
            invalidate = getattr(dragger, "_invalidate_interaction_index", None)
            if callable(invalidate):
                invalidate()
            raise
        if self.has_moved:
            dragger = getattr(self.figure, "figure_dragger", None)
            invalidate = getattr(dragger, "_invalidate_interaction_index", None)
            if callable(invalidate):
                invalidate()
        self._clear_move_transaction()

    @legend_owner_snapshot()
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
        start_raw_selection_points = getattr(
            self, "move_start_raw_selection_points", {}
        )
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
                selection_points = start_raw_selection_points.get(
                    id(target.target), selection_points
                )
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
        update_content_preview(self, transform)
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
        smart_guide_session=None,
    ):
        """called from a grabber to move the selection."""
        original_pos = np.asarray(pos, dtype=float)
        if constrain_direction:
            pos = _constrain_to_cardinal_direction(pos, dir)
        pos = np.asarray(pos, dtype=float)

        if smart_guide_session is not None:
            adjusted_pos = pos.copy()
            plan = None
            if ignore_snaps:
                smart_guide_session.hide()
            elif bool(getattr(smart_guide_session, "active", True)):
                axes = None
                if constrain_direction:
                    if np.isclose(pos[0], 0.0) and not np.isclose(pos[1], 0.0):
                        axes = frozenset((Axis.Y,))
                    elif np.isclose(pos[1], 0.0) and not np.isclose(pos[0], 0.0):
                        axes = frozenset((Axis.X,))
                    elif np.allclose(original_pos, 0.0):
                        axes = frozenset()
                try:
                    plan = smart_guide_session.query(
                        pos,
                        axes=axes,
                        render=False,
                    )
                except StaleGuideSnapshotError:
                    smart_guide_session.invalidate()
                except (
                    AttributeError,
                    IndexError,
                    LookupError,
                    TypeError,
                    ValueError,
                    RuntimeError,
                    np.linalg.LinAlgError,
                ):
                    # Smart guides are an interaction aid, never a reason to
                    # strand a drag gesture.  An invalid planner is disabled
                    # once; subsequent frames remain raw and exception-free.
                    smart_guide_session.invalidate()
                else:
                    adjusted_pos += np.asarray(plan.delta_px, dtype=float)
            try:
                self.addOffset(adjusted_pos, dir, keep_aspect_ratio)
            except Exception:
                smart_guide_session.hide()
                raise
            if plan is not None:
                try:
                    smart_guide_session.accept(plan)
                except (
                    AttributeError,
                    TypeError,
                    ValueError,
                    RuntimeError,
                ):
                    smart_guide_session.invalidate()
            self.has_moved = True
            return

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
            try:
                self.figure.figure_dragger.change_selection_zorder("backward")
            except ValueError:
                return
        if event.key == "pageup":
            try:
                self.figure.figure_dragger.change_selection_zorder("forward")
            except ValueError:
                return
        if event.key in {"left", "right", "down", "up"} and not self.operation_support(
            TransformOperation.TRANSLATE
        ).supported:
            return
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

    @staticmethod
    def _capture_figure_structure(figure: Figure) -> tuple[int, ...]:
        """Capture the explicit Artist inventory defining an editor session.

        Renderer-managed ticks are deliberately excluded because an ordinary
        draw may materialize them. Replacing Axes or direct Figure/Axes
        children through ``clf`` or external mutation requires a fresh manager
        and source tracker even when the Figure object itself is reused.
        """

        result: list[int] = []
        seen: set[int] = set()

        def add(artist) -> bool:
            if artist is None or id(artist) in seen:
                return False
            seen.add(id(artist))
            result.append(id(artist))
            return True

        def add_axes(axes) -> None:
            if not add(axes):
                return
            add(getattr(axes, "legend_", None))
            for name in ("title", "_left_title", "_right_title"):
                add(getattr(axes, name, None))
            add(axes.xaxis.get_label())
            add(axes.yaxis.get_label())
            for name in (
                "artists",
                "texts",
                "patches",
                "lines",
                "collections",
                "images",
            ):
                for artist in getattr(axes, name, ()):
                    add(artist)
            for child in getattr(axes, "child_axes", ()):
                add_axes(child)

        def add_owner(owner) -> None:
            add(owner)
            for name in ("artists", "texts", "patches", "legends"):
                for artist in getattr(owner, name, ()):
                    add(artist)
            for subfigure in getattr(owner, "subfigs", ()):
                add_owner(subfigure)
            for axes in getattr(owner, "axes", ()):
                add_axes(axes)

        add_owner(figure)
        return tuple(result)

    def _refresh_figure_structure_signature(self) -> tuple[int, ...]:
        signature = self._capture_figure_structure(self.figure)
        self._figure_structure_signature = signature
        return signature

    def figure_structure_matches(self) -> bool:
        return getattr(self, "_figure_structure_signature", None) == (
            self._capture_figure_structure(self.figure)
        )

    def __init__(self, figure: Figure, no_save, source_stack_position=None):
        self.figure = figure
        previous = getattr(figure, "figure_dragger", None)
        if previous is not None and previous is not self:
            previous.deactivate(redraw=False)
        self.figure.figure_dragger = self
        self._selectable_artists = []
        self._selectable_artist_ids = set()
        self._uneditable_artists = []
        self._uneditable_artist_ids = set()
        self._interaction_artists = []
        self._interaction_artist_ids = set()
        self._selection_parent_by_id = {}
        self._draw_child_orders = {}
        self._interaction_revision = 0
        self._interaction_index = DisplaySpaceHitIndex()
        self._marquee_index = DisplaySpaceHitIndex()
        self._interaction_roster_cache = None
        self._selectable_roster_cache = None
        self._marquee_roster_cache = None
        self._display_geometry_cache = DisplayGeometryCache()
        self._smart_guide_idle_warmup_enabled = True
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
        self._selection_semantic_mode = SelectionMode.OBJECT

        self.figure.canvas.mpl_disconnect(
            self.figure.canvas.manager.key_press_handler_id
        )

        self.activate()

        self.make_figure_draggable(self.figure)
        self.make_axes_draggable(self.figure.axes)
        self.editor_scene.restore_persisted_state()
        self._sync_editor_groups()
        self.selection = GrabbableRectangleSelection(figure, figure._pyl_scene)
        self._selection_callback_canvas = figure.canvas
        self.figure.selection = self.selection
        self.change_tracker = ChangeTracker(
            figure,
            no_save,
            source_stack_position=source_stack_position,
        )
        self.figure.change_tracker = self.change_tracker
        self._refresh_figure_structure_signature()
        schedule_smart_guide_warmup(self)

    def _ensure_interaction_index(self) -> DisplaySpaceHitIndex:
        index = getattr(self, "_interaction_index", None)
        if index is None:
            index = DisplaySpaceHitIndex()
            self._interaction_index = index
        if not hasattr(self, "_interaction_revision"):
            self._interaction_revision = 0
        return index

    def _ensure_marquee_index(self) -> DisplaySpaceHitIndex:
        index = getattr(self, "_marquee_index", None)
        if index is None:
            index = DisplaySpaceHitIndex()
            self._marquee_index = index
        return index

    def _invalidate_artist_rosters(self) -> None:
        """Drop immutable inventories only after a structural roster change."""

        self._interaction_roster_cache = None
        self._selectable_roster_cache = None
        self._marquee_roster_cache = None

    def _interaction_roster_snapshot(self):
        cached = getattr(self, "_interaction_roster_cache", None)
        if cached is not None:
            return cached
        roster = ArtistRoster.capture(
            getattr(
                self,
                "_interaction_artists",
                getattr(self, "_selectable_artists", ()),
            )
        )
        selectable_ids = getattr(self, "_selectable_artist_ids", set())
        editable = tuple(id(artist) in selectable_ids for artist in roster.artists)
        registration_order = {
            source_id: index for index, source_id in enumerate(roster.source_ids)
        }
        cached = roster, editable, registration_order
        self._interaction_roster_cache = cached
        return cached

    def _selectable_roster_snapshot(self) -> ArtistRoster:
        cached = getattr(self, "_selectable_roster_cache", None)
        if cached is None:
            artists = getattr(self, "_selectable_artists", None)
            if artists is None:
                artists = tuple(self.iter_selectable_artists())
            cached = ArtistRoster.capture(artists)
            self._selectable_roster_cache = cached
        return cached

    def _marquee_roster_snapshot(self) -> ArtistRoster:
        """Exclude formatter output that can never join a marquee selection."""

        cached = getattr(self, "_marquee_roster_cache", None)
        if cached is None:
            cached = ArtistRoster.capture(
                tuple(
                    artist
                    for artist in getattr(self, "_selectable_artists", ())
                    if not (
                        isinstance(artist, Text)
                        and getattr(
                            artist,
                            "_pylustrator_formatter_owned_tick_label",
                            False,
                        )
                    )
                )
            )
            self._marquee_roster_cache = cached
        return cached

    def _ensure_display_geometry_cache(self) -> DisplayGeometryCache:
        cache = getattr(self, "_display_geometry_cache", None)
        if cache is None:
            cache = DisplayGeometryCache()
            self._display_geometry_cache = cache
        try:
            renderer = self.figure.canvas.get_renderer()
        except (AttributeError, TypeError, ValueError, RuntimeError):
            renderer = None
        return cache.bind(
            revision=getattr(self, "_interaction_revision", 0),
            roster=self._selectable_roster_snapshot(),
            renderer=renderer,
        )

    def _invalidate_interaction_index(self) -> None:
        """Advance the geometry/inventory version used by pointer queries."""

        index = self._ensure_interaction_index()
        self._interaction_revision = int(self._interaction_revision) + 1
        index.invalidate()
        self._ensure_marquee_index().invalidate()
        geometry = getattr(self, "_display_geometry_cache", None)
        if geometry is not None:
            geometry.invalidate()
        active_session = getattr(self, "_active_smart_guide_session", None)
        if active_session is not None:
            active_session.invalidate()
        invalidate_smart_guide_cache(self)
        invalidate_content_preview_cache(self, "interaction-revision")

    @staticmethod
    def _interaction_hit_components(artist: Artist) -> tuple[Artist, ...]:
        """Return child Artists whose native hits contribute to *artist*."""

        components: list[Artist] = []
        if isinstance(artist, Legend):
            components.append(artist.get_frame())
        if isinstance(artist, Axes):
            components.append(artist.patch)
        if isinstance(artist, Annotation) and artist.arrow_patch is not None:
            components.append(artist.arrow_patch)
        if isinstance(artist, Text):
            bbox_patch = artist.get_bbox_patch()
            if bbox_patch is not None:
                components.append(bbox_patch)
        unique: list[Artist] = []
        seen: set[int] = set()
        for component in components:
            if id(component) not in seen:
                seen.add(id(component))
                unique.append(component)
        return tuple(unique)

    @staticmethod
    def _has_bounded_native_contains(artist: Artist) -> bool:
        """Reject subclass and instance-level custom containment contracts."""

        contains = getattr(artist, "contains", None)
        return getattr(contains, "__func__", None) in _BOUNDED_NATIVE_CONTAINS

    def _interaction_hit_padding(
        self,
        artist: Artist,
        components: tuple[Artist, ...] | None = None,
    ) -> float:
        """Return conservative display-pixel picker and antialias padding."""

        # PatchAdapter's thin-stroke fallback uses three display pixels.  Two
        # further pixels cover antialias fringes and integer event rounding.
        tolerance = 3.0
        renderer = None
        try:
            renderer = self.figure.canvas.get_renderer()
        except (AttributeError, TypeError, ValueError, RuntimeError):
            pass

        def include(value) -> None:
            nonlocal tolerance
            if isinstance(value, (bool, np.bool_)) or not isinstance(
                value, (int, float, np.integer, np.floating)
            ):
                return
            value = abs(float(value))
            if not np.isfinite(value):
                return
            # Collection pick radii are consumed as display pixels while line
            # radii and numeric pickers are conventionally typographic points.
            # Taking both interpretations is conservative across artist types.
            tolerance = max(tolerance, value)
            if renderer is not None:
                try:
                    tolerance = max(
                        tolerance, abs(float(renderer.points_to_pixels(value)))
                    )
                except (AttributeError, TypeError, ValueError, RuntimeError):
                    pass
            else:
                try:
                    tolerance = max(
                        tolerance,
                        value * abs(float(self.figure.dpi)) / 72.0,
                    )
                except (AttributeError, TypeError, ValueError, RuntimeError):
                    pass

        if components is None:
            components = self._interaction_hit_components(artist)
        hit_sources = (artist, *components)
        for source in hit_sources:
            for getter_name in ("get_picker", "get_pickradius", "get_linewidth"):
                getter = getattr(source, getter_name, None)
                if callable(getter):
                    try:
                        include(getter())
                    except (AttributeError, TypeError, ValueError, RuntimeError):
                        pass
        return tolerance + 2.0

    def _interaction_index_bounds(
        self,
        artist: Artist,
        geometry: DisplayGeometryCache | None = None,
    ) -> tuple[float, float, float, float] | None:
        """Return a conservative bounded hit envelope, otherwise ``None``.

        ``None`` means the artist is always tested; it never means the artist
        can be omitted.  This is how unsupported foreground objects, logical
        groups, and third-party custom hit contracts remain fail-open.
        """

        if id(artist) not in getattr(self, "_selectable_artist_ids", set()):
            return None
        if isinstance(artist, EditorGroup):
            return None
        try:
            adapter = TargetWrapper(artist).adapter
        except (AttributeError, LookupError, TypeError, ValueError, RuntimeError):
            return None
        if type(adapter).hit_test not in _BOUNDED_ADAPTER_HIT_TESTS:
            return None
        if not self._has_bounded_native_contains(artist):
            return None
        components = self._interaction_hit_components(artist)
        if any(not self._has_bounded_native_contains(item) for item in components):
            return None

        point_sets: list[np.ndarray] = []
        needs_adapter_envelope = isinstance(artist, Collection)
        if needs_adapter_envelope:
            try:
                effective_clip = bool(artist.get_clip_on()) and (
                    artist.get_clip_box() is not None
                    or artist.get_clip_path() is not None
                )
                shared = (
                    None
                    if geometry is None or effective_clip
                    else geometry.selection_bounds(artist)
                )
                if shared is not None:
                    point_sets.append(np.asarray(shared, dtype=float).reshape(2, 2))
                else:
                    points = np.asarray(adapter.selection_points(), dtype=float)
                    if points.ndim == 2 and points.shape[1] >= 2:
                        points = points[:, :2]
                        points = points[np.all(np.isfinite(points), axis=1)]
                        if len(points):
                            point_sets.append(points)
            except (
                AttributeError,
                IndexError,
                LookupError,
                NotImplementedError,
                OverflowError,
                TypeError,
                ValueError,
                RuntimeError,
                np.linalg.LinAlgError,
            ):
                return None
            if not point_sets:
                return None

        # Native Matplotlib ``contains`` commonly ignores clipping.  Use raw
        # window extents and explicitly union every composite child whose
        # native containment contributes to its parent contract.
        # Collection.get_window_extent historically reports data-limit space,
        # not display space.  Its adapter envelope is the bounded native
        # collection hit geometry, so do not union that incompatible extent.
        if not isinstance(artist, Collection):
            try:
                renderer = (
                    geometry.renderer
                    if geometry is not None and geometry.renderer is not None
                    else self.figure.canvas.get_renderer()
                )
                if isinstance(artist, Text) and artist.get_bbox_patch() is not None:
                    artist.update_bbox_position_size(renderer)
                effective_clip = bool(artist.get_clip_on()) and (
                    artist.get_clip_box() is not None
                    or artist.get_clip_path() is not None
                )
                shared = (
                    None
                    if geometry is None
                    or effective_clip
                    or isinstance(artist, Annotation)
                    else geometry.selection_bounds(artist)
                )
                extent_artists = (artist, *components)
                if shared is not None:
                    point_sets.append(np.asarray(shared, dtype=float).reshape(2, 2))
                    extent_artists = components
                for extent_artist in extent_artists:
                    extent = np.asarray(
                        extent_artist.get_window_extent(renderer).extents,
                        dtype=float,
                    )
                    if extent.shape != (4,) or not np.all(np.isfinite(extent)):
                        return None
                    point_sets.append(extent.reshape(2, 2))
                if isinstance(artist, Annotation):
                    # Annotation.get_window_extent may collapse to Bbox.unit()
                    # under annotation clipping even though Annotation.contains
                    # still delegates directly to Text.contains.
                    text_extent = np.asarray(
                        Text.get_window_extent(artist, renderer).extents,
                        dtype=float,
                    )
                    if text_extent.shape != (4,) or not np.all(
                        np.isfinite(text_extent)
                    ):
                        return None
                    point_sets.append(text_extent.reshape(2, 2))
            except (
                AttributeError,
                IndexError,
                NotImplementedError,
                OverflowError,
                TypeError,
                ValueError,
                RuntimeError,
            ):
                return None

        if not point_sets:
            return None
        points = np.concatenate(point_sets)
        low = np.min(points, axis=0)
        high = np.max(points, axis=0)
        if not np.all(np.isfinite((low, high))):
            return None
        padding = self._interaction_hit_padding(artist, components)
        return (
            float(low[0] - padding),
            float(low[1] - padding),
            float(high[0] + padding),
            float(high[1] + padding),
        )

    def _interaction_candidate_indices(
        self,
        event: MouseEvent,
        artists: Sequence[Artist],
        *,
        source_ids: tuple[int, ...] | None = None,
    ) -> Sequence[int]:
        """Return indexed candidates, falling back to the original full scan."""

        index = self._ensure_interaction_index()
        geometry = self._ensure_display_geometry_cache()
        try:
            candidates = index.candidate_indices(
                event.x,
                event.y,
                artists,
                revision=self._interaction_revision,
                bounds_provider=lambda artist: self._interaction_index_bounds(
                    artist, geometry
                ),
                source_ids=source_ids,
            )
        except Exception:
            candidates = None
        return range(len(artists)) if candidates is None else candidates

    def activate(self):
        """activate the interaction callbacks from the figure"""
        if getattr(self, "_interaction_active", False):
            return False
        canvas = self.figure.canvas
        self._callback_canvas = canvas
        self.c3 = canvas.mpl_connect(
            "button_release_event", self.button_release_event0
        )
        self.c2 = canvas.mpl_connect(
            "button_press_event", self.button_press_event0
        )
        self.c4 = canvas.mpl_connect(
            "key_press_event", self.key_press_event
        )
        self.c5 = canvas.mpl_connect(
            "motion_notify_event", self.motion_notify_event0
        )
        self.c6 = canvas.mpl_connect(
            "draw_event", self.invalidate_geometry_cache
        )
        selection = getattr(self, "selection", None)
        if selection is not None and getattr(selection, "c4", None) is None:
            selection.c4 = canvas.mpl_connect(
                "key_press_event", selection.keyPressEvent
            )
            self._selection_callback_canvas = canvas
        self._selection_refresh_on_draw = True
        self._interaction_active = True
        return True

    def _post_draw_selection_indices(self) -> tuple[int, ...]:
        """Return selected targets whose geometry can settle during draw."""

        selection = getattr(self, "selection", None)
        if selection is None:
            return ()
        get_layout_engine = getattr(self.figure, "get_layout_engine", None)
        if callable(get_layout_engine) and get_layout_engine() is not None:
            return tuple(range(len(selection.targets)))

        parent_map = getattr(self, "_selection_parent_by_id", {})

        def needs_refresh(artist: Artist) -> bool:
            if isinstance(artist, Legend) or bool(
                getattr(artist, "_pylustrator_geometry_finalized_on_draw", False)
            ):
                return True
            if isinstance(artist, EditorGroup):
                return any(needs_refresh(member) for member in artist.members)
            current = parent_map.get(id(artist))
            seen = set()
            while current is not None and id(current) not in seen:
                if isinstance(current, Legend):
                    return True
                seen.add(id(current))
                current = parent_map.get(id(current))
            return False

        return tuple(
            index
            for index, target in enumerate(selection.targets)
            if needs_refresh(target.target)
        )

    def refresh_selection_geometry(self, *, post_draw: bool = False) -> None:
        """Synchronize selection overlays with current rendered geometry."""

        selection = getattr(self, "selection", None)
        if selection is None or not getattr(selection, "targets", None):
            return
        if post_draw:
            selection.refresh_targets_after_draw(
                self._post_draw_selection_indices()
            )
            return
        with selection_geometry_snapshot():
            selection.update_extent()
            selection.update_selection_rectangles()

    def _draw_parent(self, artist: Artist) -> Artist | None:
        if isinstance(artist, EditorGroup):
            return artist.owner
        parent = getattr(self, "_selection_parent_by_id", {}).get(id(artist))
        if parent is None and isinstance(artist, SubFigure):
            parent = getattr(artist, "_parent", None)
        return parent

    def _paint_order_key(
        self,
        artist: Artist,
        *,
        fallback: int = 0,
        registration_order: dict[int, int] | None = None,
    ) -> tuple:
        """Return the authoritative back-to-front Matplotlib paint key.

        Hit testing and cached content ghosts share this exact ownership/zorder
        model so a same-z multi-selection cannot reverse overlapping paint.
        """

        if registration_order is None:
            registration_order = {
                id(item): index
                for index, item in enumerate(
                    getattr(self, "_interaction_artists", ())
                )
            }
        child_orders: dict[int, dict[int, int]] = getattr(
            self, "_draw_child_orders", {}
        )
        self._draw_child_orders = child_orders

        def child_order(parent, child, default):
            parent_key = id(parent)
            if parent_key not in child_orders:
                try:
                    children = get_artist_children(parent)
                except (AttributeError, TypeError, ValueError, RuntimeError):
                    children = []
                child_orders[parent_key] = {
                    id(item): index for index, item in enumerate(children)
                }
            return child_orders[parent_key].get(id(child), default)

        path = []
        current = artist
        seen = set()
        while current is not None and not isinstance(current, Figure):
            current_key = id(current)
            if current_key in seen:
                break
            seen.add(current_key)
            parent = self._draw_parent(current)
            default = registration_order.get(current_key, int(fallback))
            order = (
                child_order(parent, current, default)
                if parent is not None
                else default
            )
            path.append((float(current.get_zorder()), order))
            current = parent
        return tuple(reversed(path)), int(fallback)

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

    def _notify_selected_element_changed(self) -> None:
        """Refresh property panels without changing the current selection."""

        current = [target.target for target in self.selection.targets]
        element = self.selected_element if self.selected_element in current else None
        if element is None and current:
            element = current[-1]
            self.selected_element = element
        signals = getattr(self.figure, "signals", None)
        selected = getattr(signals, "figure_element_selected", None)
        emit = getattr(selected, "emit", None)
        if callable(emit):
            self._selection_notification_revision = (
                int(getattr(self, "_selection_notification_revision", 0)) + 1
            )
            self.figure.no_figure_dragger_selection_update = True
            try:
                emit(element)
            finally:
                self.figure.no_figure_dragger_selection_update = False

    def _reconcile_selection_for_mode(self) -> bool:
        """Make the retained selection obey the active tool's object policy.

        Tool changes must not leave a logical group draggable by Direct
        Selection, or a direct group member draggable as though Object
        Selection were still active. Programmatic mode changes are uncommon,
        so this work only rebuilds overlays when semantic targets differ.
        """

        selection = getattr(self, "selection", None)
        if selection is None:
            return False
        current = [target.target for target in selection.targets]
        if not current:
            if self.selected_element is None:
                return False
            self.selected_element = None
            self._notify_selected_element_changed()
            return True

        kernel = self._ensure_selection_kernel()
        mapped = kernel.map_artists(current)
        mapped_primary = kernel.map_artists(
            [] if self.selected_element is None else [self.selected_element]
        )
        primary = mapped_primary[-1] if mapped_primary else None
        if primary is None and mapped:
            primary = mapped[-1]
        unchanged = len(current) == len(mapped) and all(
            before is after for before, after in zip(current, mapped)
        )
        if unchanged and self.selected_element is primary:
            return False

        self.select_elements(mapped, primary=primary)
        return True

    def set_selection_mode(self, mode: SelectionMode | str) -> SelectionMode:
        if hasattr(self, "selection"):
            self._cancel_active_pointer_transform()
        result = self._ensure_selection_kernel().set_mode(mode)
        self._reconcile_selection_for_mode()
        self._selection_semantic_mode = result
        invalidate_smart_guide_cache(self)
        schedule_smart_guide_warmup(self)
        self._update_interaction_controls()
        return result

    def enter_isolation(self, element: Artist, *, notify: bool = True) -> bool:
        if hasattr(self, "selection"):
            self._cancel_active_pointer_transform()
        entered = self._ensure_selection_kernel().enter_isolation(element)
        if entered:
            invalidate_smart_guide_cache(self)
            self.selection.clear_targets()
            self.selected_element = None
            self.on_select(None, None)
            if notify:
                self._notify_selected_element_changed()
            self._update_interaction_controls()
            self.figure.canvas.draw_idle()
            schedule_smart_guide_warmup(self)
        return entered

    def exit_isolation(self) -> Artist | None:
        if hasattr(self, "selection"):
            self._cancel_active_pointer_transform()
        exited = self._ensure_selection_kernel().exit_isolation()
        if exited is not None:
            invalidate_smart_guide_cache(self)
            self.select_element(exited)
            self._update_interaction_controls()
            self.figure.canvas.draw_idle()
            schedule_smart_guide_warmup(self)
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
        alignment_key = (
            self._object_locator(self.selection.alignment_key)
            if self.selection.alignment_key is not None
            else None
        )
        return InteractionState(
            mode=self.selection_mode.value,
            selected=selected,
            primary=primary,
            scopes=scopes,
            alignment_reference_mode=self.selection.alignment_reference_mode,
            alignment_key=alignment_key,
            reference_point=tuple(self.selection.reference_point),
            custom_rotation_pivot_inches=(
                self.selection.custom_rotation_pivot_state()
            ),
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
        alignment_key = (
            state.alignment_key.resolve(scene)
            if state.alignment_key is not None
            else None
        )
        self.selection._restore_alignment_reference_state(
            state.alignment_reference_mode, alignment_key
        )
        self.selection.set_reference_point(state.reference_point)
        if state.custom_rotation_pivot_inches is not None:
            self.selection._set_custom_rotation_pivot_state(
                state.custom_rotation_pivot_inches
            )
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
        inventory_before = tuple(
            id(item) for item in getattr(self, "_interaction_artists", ())
        )
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
        inventory_after = tuple(
            id(item) for item in getattr(self, "_interaction_artists", ())
        )
        if inventory_after != inventory_before:
            self._invalidate_artist_rosters()
            self._invalidate_interaction_index()

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
        target = self._resolve_top_hit(event).target
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

    def _zorder_paint_siblings(
        self, parent: Artist, selected: Sequence[Artist]
    ) -> tuple[list[Artist], dict[int, int]]:
        """Return visible editor siblings in authoritative child-list order.

        Matplotlib paints Figure/SubFigure/Axes children by a stable z-order
        sort.  Registration order is not a substitute for the stable portion
        of that sort, and managed descendants such as tick labels are not
        direct paint siblings even when the editor exposes them beneath an
        Axes.  Restricting this inventory to actual children prevents a local
        arrange command from pretending that unrelated containers share a
        global layer stack.
        """

        if not isinstance(parent, (Figure, SubFigure, Axes)):
            raise ValueError(
                f"{type(parent).__name__} does not expose a sortable paint stack"
            )
        children = get_artist_children(parent)
        child_order = {id(child): index for index, child in enumerate(children)}
        interaction_ids = set(getattr(self, "_interaction_artist_ids", ()))
        interaction_ids.update(id(artist) for artist in selected)
        siblings = []
        seen = set()
        for child in children:
            key = id(child)
            if key in seen or key not in interaction_ids:
                continue
            seen.add(key)
            if self._draw_parent(child) is not parent or not child.get_visible():
                continue
            if isinstance(child, Text) and child.get_text() == "":
                continue
            siblings.append(child)

        sibling_ids = {id(artist) for artist in siblings}
        missing = [
            artist for artist in selected if id(artist) not in sibling_ids
        ]
        if missing:
            names = ", ".join(type(artist).__name__ for artist in missing)
            raise ValueError(
                f"Selected {names} is not a visible direct paint sibling"
            )
        siblings.sort(
            key=lambda artist: (
                float(artist.get_zorder()),
                child_order[id(artist)],
            )
        )
        return siblings, child_order

    @staticmethod
    def _reorder_siblings(
        siblings: Sequence[Artist], selected_ids: set[int], mode: str
    ) -> list[Artist]:
        """Apply Illustrator-style stable arrange semantics to one stack."""

        result = list(siblings)
        if mode == "forward":
            for index in range(len(result) - 2, -1, -1):
                if (
                    id(result[index]) in selected_ids
                    and id(result[index + 1]) not in selected_ids
                ):
                    result[index], result[index + 1] = (
                        result[index + 1],
                        result[index],
                    )
        elif mode == "backward":
            for index in range(1, len(result)):
                if (
                    id(result[index]) in selected_ids
                    and id(result[index - 1]) not in selected_ids
                ):
                    result[index], result[index - 1] = (
                        result[index - 1],
                        result[index],
                    )
        elif mode == "front":
            result = [
                artist for artist in result if id(artist) not in selected_ids
            ] + [artist for artist in result if id(artist) in selected_ids]
        else:
            result = [
                artist for artist in result if id(artist) in selected_ids
            ] + [artist for artist in result if id(artist) not in selected_ids]
        return result

    @staticmethod
    def _zorder_run_levels(
        base: float,
        count: int,
        direction: float,
        bound: float | None,
    ) -> list[float]:
        levels = []
        current = float(base)
        destination = np.inf if direction > 0 else -np.inf
        for _index in range(count):
            current = float(np.nextafter(current, destination))
            if not np.isfinite(current) or (
                bound is not None
                and (
                    (direction > 0 and current >= bound)
                    or (direction < 0 and current <= bound)
                )
            ):
                raise ValueError(
                    "The local z-order values are too dense to preserve paint order"
                )
            levels.append(current)
        if direction < 0:
            levels.reverse()
        return levels

    @classmethod
    def _zorder_values_for_order(
        cls,
        current: Sequence[Artist],
        destination: Sequence[Artist],
        child_order: dict[int, int],
        selected_ids: set[int],
    ) -> dict[int, float]:
        """Relabel one stable sort so it exactly realizes *destination*.

        Existing z-order levels are permuted first.  Equal-level inversions
        are split into the minimum number of adjacent floating-point levels;
        the longest child-order run remains unchanged.  This preserves the
        normal semantic bands (patch/line/text) while still handling a stack
        whose siblings all share exactly the same zorder.
        """

        levels = sorted(float(artist.get_zorder()) for artist in current)
        if not all(np.isfinite(level) for level in levels):
            raise ValueError("Stacking order requires finite z-order values")
        assigned: dict[int, float] = {}
        start = 0
        while start < len(destination):
            base = levels[start]
            stop = start + 1
            while stop < len(destination) and levels[stop] == base:
                stop += 1
            group = list(destination[start:stop])

            runs: list[list[Artist]] = []
            for artist in group:
                if (
                    not runs
                    or child_order[id(artist)]
                    <= child_order[id(runs[-1][-1])]
                ):
                    runs.append([])
                runs[-1].append(artist)

            anchors = sorted(
                range(len(runs)),
                key=lambda index: (
                    len(runs[index]),
                    sum(
                        id(artist) not in selected_ids
                        for artist in runs[index]
                    ),
                    -index,
                ),
                reverse=True,
            )
            lower = levels[start - 1] if start else None
            upper = levels[stop] if stop < len(levels) else None
            for anchor in anchors:
                try:
                    before_levels = cls._zorder_run_levels(
                        base, anchor, -1.0, lower
                    )
                    after_levels = cls._zorder_run_levels(
                        base, len(runs) - anchor - 1, 1.0, upper
                    )
                except ValueError:
                    continue
                break
            else:
                raise ValueError(
                    "The local z-order values are too dense to preserve paint order"
                )
            run_levels = [*before_levels, base, *after_levels]
            for run, level in zip(runs, run_levels):
                for artist in run:
                    assigned[id(artist)] = level
            start = stop

        resolved = sorted(
            current,
            key=lambda artist: (
                assigned[id(artist)],
                child_order[id(artist)],
            ),
        )
        if any(
            actual is not expected
            for actual, expected in zip(resolved, destination)
        ):
            raise ValueError("Could not represent the requested paint order")
        return assigned

    def change_selection_zorder(self, mode: str) -> bool:
        leaves = self._zorder_leaves(self._selected_artists())
        if not leaves:
            return False
        modes = {"forward", "backward", "front", "back"}
        if mode not in modes:
            raise ValueError(f"Unknown z-order action: {mode}")
        selected_ids = {id(artist) for artist in leaves}
        parents = {id(parent): parent for parent in map(self._draw_parent, leaves)}
        if None in parents.values() or len(parents) != 1:
            raise ValueError(
                "Arrange requires selected objects in one paint container"
            )
        parent = next(iter(parents.values()))
        siblings, child_order = self._zorder_paint_siblings(parent, leaves)
        destination = self._reorder_siblings(siblings, selected_ids, mode)
        if all(before is after for before, after in zip(siblings, destination)):
            return False
        values = self._zorder_values_for_order(
            siblings, destination, child_order, selected_ids
        )
        operations = [
            PropertyOperation.for_setter(
                artist,
                artist,
                "zorder",
                values[id(artist)],
            )
            for artist in siblings
            if float(artist.get_zorder()) != values[id(artist)]
        ]
        if not operations:
            return False
        for operation in operations:
            try:
                getReference(operation.target)
            except (AttributeError, IndexError, TypeError, ValueError) as error:
                raise ValueError(
                    f"Cannot replay stacking order for "
                    f"{type(operation.target).__name__}"
                ) from error
        return PropertyPlan(self.figure, operations).commit(
            "Change stacking order"
        )

    def _prune_detached_interaction_artists(self) -> int:
        """Release externally removed Artists at the authoritative draw boundary.

        Editor delete keeps undoable objects attached and merely hidden, so it
        is unaffected.  This only removes Artists that Matplotlib (or a removed
        semantic parent) has actually detached from the live figure.
        """

        interaction = list(getattr(self, "_interaction_artists", ()))
        if not interaction:
            return 0
        detached_ids = {
            id(artist) for artist in interaction if not self._is_artist_attached(artist)
        }
        parent_map = getattr(self, "_selection_parent_by_id", {})
        for artist in interaction:
            current = parent_map.get(id(artist))
            seen = set()
            while current is not None and id(current) not in seen:
                if id(current) in detached_ids or getattr(current, "figure", None) is None:
                    detached_ids.add(id(artist))
                    break
                seen.add(id(current))
                current = parent_map.get(id(current))

        scene = getattr(self, "editor_scene", None)
        if scene is not None:
            for group in list(scene.groups.values()):
                group.members = [
                    member for member in group.members if id(member) not in detached_ids
                ]
                if not group.members:
                    scene.remove_group(group)
                    detached_ids.add(id(group))
        if not detached_ids:
            return 0

        def keep(items):
            return [item for item in items if id(item) not in detached_ids]

        self._interaction_artists = keep(interaction)
        self._interaction_artist_ids = {
            id(item) for item in self._interaction_artists
        }
        self._selectable_artists = keep(
            list(getattr(self, "_selectable_artists", ()))
        )
        self._selectable_artist_ids = {
            id(item) for item in self._selectable_artists
        }
        self._uneditable_artists = keep(
            list(getattr(self, "_uneditable_artists", ()))
        )
        self._uneditable_artist_ids = {
            id(item) for item in self._uneditable_artists
        }
        self._selection_parent_by_id = {
            key: parent
            for key, parent in parent_map.items()
            if key not in detached_ids and id(parent) not in detached_ids
        }
        if scene is not None:
            for key in detached_ids:
                scene._known_artists.pop(key, None)
                scene._logical_parent_by_id.pop(key, None)
                scene._locked_ids.discard(key)
                scene._explicitly_hidden_ids.discard(key)

        selection = getattr(self, "selection", None)
        if selection is not None:
            for target in list(selection.targets):
                if id(target.target) in detached_ids:
                    selection.remove_target(target.target)
        if id(getattr(self, "selected_element", None)) in detached_ids:
            remaining = [target.target for target in getattr(selection, "targets", ())]
            self.selected_element = remaining[-1] if remaining else None
        if id(getattr(self, "preselection_artist", None)) in detached_ids:
            self._hide_preselection()
        self._invalidate_artist_rosters()
        return len(detached_ids)

    def invalidate_geometry_cache(self, _event=None):
        """Drop visible-bound caches after any render/transform change."""
        self._prune_detached_interaction_artists()
        for artist in getattr(self, "_selectable_artists", []):
            setattr(artist, "_pylustrator_cached_get_extend", None)
        # Child list order is part of the authoritative paint/hit order and can
        # change independently from zorder (including through external
        # Matplotlib mutations). Never retain it across a completed draw.
        self._draw_child_orders = {}
        # Locators may materialize additional Tick/Text objects during draw.
        # Keep visible labels in the same explicit hit inventory as all other
        # direct-selection targets without rebuilding the complete scene.
        for axes in getattr(self.figure, "axes", ()):
            self.register_axis_tick_labels(axes)
        # Draw is the authoritative boundary for layout, visibility, z-order,
        # and external Matplotlib mutations.  Never query pre-draw envelopes.
        self._invalidate_interaction_index()
        # Legend/text layout is finalized by Matplotlib during draw.  Overlay
        # geometry measured before that point is necessarily provisional.
        self.refresh_selection_geometry(post_draw=True)
        schedule_smart_guide_warmup(self)
        schedule_content_preview_warmup(self)

    def deactivate(self, *, redraw: bool = True):
        """deactivate the interaction callbacks from the figure"""
        if not getattr(self, "_interaction_active", False):
            return False
        self._cancel_active_pointer_transform()
        invalidate_smart_guide_cache(self)
        close_content_preview_cache(self)
        session = getattr(self, "_active_smart_guide_session", None)
        if session is not None:
            session.close()
        canvas = getattr(self, "_callback_canvas", self.figure.canvas)
        for name in ("c3", "c2", "c4", "c5", "c6"):
            connection = getattr(self, name, None)
            if connection is not None:
                canvas.mpl_disconnect(connection)
            setattr(self, name, None)
        selection = getattr(self, "selection", None)
        if selection is not None:
            selection_connection = getattr(selection, "c4", None)
            if selection_connection is not None:
                selection_canvas = getattr(
                    self, "_selection_callback_canvas", canvas
                )
                selection_canvas.mpl_disconnect(selection_connection)
                selection.c4 = None
        self._selection_refresh_on_draw = False
        self._interaction_active = False
        self._callback_canvas = None
        self._selection_callback_canvas = None

        self.selection.clear_targets()
        self.selected_element = None
        self.on_select(None, None)
        self._notify_selected_element_changed()
        if redraw:
            self.figure.canvas.draw_idle()
        return True

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
        self._ensure_interaction_index()
        index_changed = False
        roster_changed = False
        if parent is None:
            if isinstance(target, Legend):
                parent = getattr(target, "parent", None)
            if parent is None:
                parent = getattr(target, "axes", None)
            if parent is target or parent is None:
                parent = getattr(target, "figure", None)
        if parent is not None and parent is not target:
            if self._selection_parent_by_id.get(id(target)) is not parent:
                self._selection_parent_by_id[id(target)] = parent
                index_changed = True
        finalized_on_draw = bool(
            getattr(target, "_pylustrator_geometry_finalized_on_draw", False)
            or isinstance(target, Legend)
            or isinstance(parent, Legend)
        )
        if isinstance(target, Text) and isinstance(parent, Axes):
            axes_layout_texts = (
                parent.title,
                parent._left_title,
                parent._right_title,
                parent.xaxis.label,
                parent.yaxis.label,
                parent.xaxis.offsetText,
                parent.yaxis.offsetText,
            )
            finalized_on_draw = finalized_on_draw or any(
                target is text for text in axes_layout_texts
            )
        if isinstance(target, Text) and isinstance(parent, (Figure, SubFigure)):
            finalized_on_draw = finalized_on_draw or any(
                target is getattr(parent, name, None)
                for name in ("_suptitle", "_supxlabel", "_supylabel")
            )
        if finalized_on_draw:
            target._pylustrator_geometry_finalized_on_draw = True
        if isinstance(target, Text):
            target._pylustrator_legend_owner = (
                parent if isinstance(parent, Legend) else None
            )
        self._ensure_editor_scene().register_artist(target)
        if id(target) not in self._interaction_artist_ids:
            self._interaction_artists.append(target)
            self._interaction_artist_ids.add(id(target))
            self._draw_child_orders = {}
            index_changed = True
            roster_changed = True
        if not TargetWrapper.supports_target(target):
            if (
                id(target) not in self._selectable_artist_ids
                and id(target) not in self._uneditable_artist_ids
            ):
                self._uneditable_artists.append(target)
                self._uneditable_artist_ids.add(id(target))
            if roster_changed:
                self._invalidate_artist_rosters()
            if index_changed:
                self._invalidate_interaction_index()
            return False
        if id(target) not in self._selectable_artist_ids:
            self._selectable_artists.append(target)
            self._selectable_artist_ids.add(id(target))
            index_changed = True
            roster_changed = True
        target._pylustrator_explicitly_editable = True
        if not target.pickable():
            target.set_picker(True)
            index_changed = True
        if roster_changed:
            self._invalidate_artist_rosters()
        if index_changed:
            self._invalidate_interaction_index()
        if isinstance(target, Text):
            add_text_default(target)
        if isinstance(target, Legend):
            invalidate_legend_owner_inventory(self.figure)
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
            self.register_axis_tick_labels(ax)
            self.make_draggable(ax, parent or ax.figure or self.figure)
            self.make_axes_draggable(ax.child_axes, parent=ax)

    def register_axis_tick_labels(self, axes: Axes) -> None:
        """Expose every currently visible non-empty tick label as a Text leaf."""

        seen_axes = set()
        for axis_name in ("xaxis", "yaxis", "zaxis"):
            axis = getattr(axes, axis_name, None)
            if axis is None or id(axis) in seen_axes:
                continue
            seen_axes.add(id(axis))
            ticks = (
                *getattr(axis, "majorTicks", ()),
                *getattr(axis, "minorTicks", ()),
            )
            for tick in ticks:
                for label in (tick.label1, tick.label2):
                    if label.get_visible() and label.get_text() != "":
                        label._pylustrator_geometry_finalized_on_draw = True
                        was_formatter_owned = bool(
                            getattr(
                                label,
                                "_pylustrator_formatter_owned_tick_label",
                                False,
                            )
                        )
                        label._pylustrator_formatter_owned_tick_label = True
                        if not was_formatter_owned:
                            self._marquee_roster_cache = None
                        # Draw events revisit the same Tick/Text identities very
                        # frequently. Registration and capability discovery are
                        # inventory work, not geometry invalidation work.
                        if id(label) not in self._interaction_artist_ids:
                            self.make_draggable(label, axes)

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

    def _is_interaction_hit(
        self,
        artist: Artist,
        event: MouseEvent,
        *,
        registered_editable: bool | None = None,
    ) -> bool:
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
            if registered_editable is True or target.supported:
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
            if isinstance(current, Axes):
                owner = parent_map.get(id(current))
                if isinstance(owner, Axes):
                    if not any(child is current for child in owner.child_axes):
                        return False
                elif isinstance(owner, (Figure, SubFigure)):
                    if not any(axes is current for axes in owner.axes):
                        return False
                else:
                    figure = getattr(current, "figure", None)
                    if not isinstance(figure, (Figure, SubFigure)):
                        return False
                    live = any(axes is current for axes in figure.axes)
                    if not live:
                        live = any(
                            child is current
                            for axes in figure.axes
                            for child in axes.child_axes
                        )
                    if not live:
                        return False
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
        geometry = self._ensure_display_geometry_cache()
        with geometry.snapshot():
            return self._get_hit_stack(event)

    def _get_hit_stack(self, event: MouseEvent) -> HitStack:
        """Return every visual hit from front to back using one draw-order model."""

        return HitStack(tuple(self._iter_hit_candidates(event)))

    def _iter_hit_candidates(self, event: MouseEvent):
        """Yield actual hits front-to-back, allowing top-hit short circuiting."""

        roster, editable_flags, registration_order = (
            self._interaction_roster_snapshot()
        )
        artists = roster.artists

        def pick_order(index):
            return self._paint_order_key(
                artists[index],
                fallback=index,
                registration_order=registration_order,
            )

        candidate_indices = self._interaction_candidate_indices(
            event,
            artists,
            source_ids=roster.source_ids,
        )
        ordered_entries = sorted(
            ((pick_order(index), index) for index in candidate_indices),
            key=lambda item: item[0],
            reverse=True,
        )
        for draw_key, index in ordered_entries:
            candidate = artists[index]
            registered_editable = editable_flags[index]
            if not self._is_interaction_hit(
                candidate,
                event,
                registered_editable=registered_editable,
            ):
                continue
            editable = bool(
                registered_editable
                and self._resolve_selectable_artist(candidate) is not None
            )
            yield HitCandidate(
                candidate,
                editable,
                draw_key,
                index,
            )

    def _resolve_top_hit(self, event: MouseEvent):
        """Resolve an ordinary click without materializing the full hit stack."""

        geometry = self._ensure_display_geometry_cache()
        with geometry.snapshot():
            iterator = iter(self._iter_hit_candidates(event))
            consumed: list[HitCandidate] = []

            def recording_stream():
                for candidate in iterator:
                    consumed.append(candidate)
                    yield candidate

            kernel = self._ensure_selection_kernel()
            decision = kernel.resolve_top(recording_stream())
            if decision.status is TopHitStatus.RESOLVED:
                return decision
            # Direct Selection encountered a group shell.  Continue the same
            # iterator and reuse already-tested hits in the full oracle path.
            return kernel.resolve(HitStack(tuple(consumed) + tuple(iterator)))

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
        self,
        artist: Artist,
        x0: float,
        y0: float,
        x1: float,
        y1: float,
        geometry: DisplayGeometryCache | None = None,
    ) -> bool:
        bounds = (
            geometry.selection_bounds(artist)
            if geometry is not None
            else self._artist_display_bounds(artist)
        )
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
        geometry = self._ensure_display_geometry_cache()
        with geometry.snapshot():
            return self._select_elements_in_bbox(
                x0, y0, x1, y1, additive, geometry=geometry
            )

    def _select_elements_in_bbox(
        self,
        x0: float,
        y0: float,
        x1: float,
        y1: float,
        additive: bool = False,
        *,
        geometry: DisplayGeometryCache | None = None,
    ) -> list[Artist]:
        x0, x1 = sorted((float(x0), float(x1)))
        y0, y1 = sorted((float(y0), float(y1)))
        if geometry is None:
            geometry = self._ensure_display_geometry_cache()
        if hasattr(self, "_selectable_artists"):
            roster = self._marquee_roster_snapshot()
            candidates = self._ensure_marquee_index().candidate_indices_for_bounds(
                x0,
                y0,
                x1,
                y1,
                roster.artists,
                revision=getattr(self, "_interaction_revision", 0),
                bounds_provider=geometry.selection_bounds,
                source_ids=roster.source_ids,
            )
            candidate_indices = (
                range(len(roster.artists)) if candidates is None else candidates
            )
            candidate_artists = (
                roster.artists[index]
                for index in candidate_indices
                if self._is_pick_candidate(roster.artists[index], explicit=True)
            )
        else:
            # Preserve the lightweight legacy/test-double surface. Production
            # managers always own the explicit revisioned roster above.
            candidate_artists = self.iter_selectable_artists()
        # Formatter-owned tick labels are click-selectable for property edits,
        # but cannot participate in a rigid marquee transform.  Requiring an
        # explicit click also prevents one generated label from disabling an
        # otherwise movable mixed marquee selection.
        artists = [
            artist
            for artist in candidate_artists
            if not (
                isinstance(artist, Text)
                and (
                    getattr(
                        artist,
                        "_pylustrator_formatter_owned_tick_label",
                        False,
                    )
                    or axis_tick_label_reference(artist) is not None
                )
            )
        ]
        elements = [
            artist
            for artist in artists
            if self._artist_intersects_bbox(
                artist, x0, y0, x1, y1, geometry=geometry
            )
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
            validated_ids = {id(element) for element in elements}
            mapped = self._ensure_selection_kernel().map_artists(elements)
            # Object Selection may promote a validated leaf to a logical
            # group. The group is a distinct transform target and must pass
            # attachment/lock/capability/geometry checks in its own right.
            elements = [
                element
                for element in mapped
                if id(element) in validated_ids
                or self._is_pick_candidate(element, explicit=True)
            ]
        if elements or not additive:
            selected_elements = self.select_elements(
                elements,
                additive=additive,
                preserve_axes=prefer_containers,
                prefer_containers=prefer_containers,
                _prevalidated=True,
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
            if (
                getattr(self, "_selection_semantic_mode", None)
                is not self.selection_mode
            ):
                self._reconcile_selection_for_mode()
                self._selection_semantic_mode = self.selection_mode
            selected = [target.target for target in self.selection.targets]
            last = (
                self.selected_element
                if any(target is self.selected_element for target in selected)
                else (selected[-1] if selected else None)
            )

            click_through = _event_has_modifier(
                event, "alt"
            ) or _event_has_modifier(event, "option")
            shift = _event_has_modifier(event, "shift")
            kernel = self._ensure_selection_kernel()
            hit_stack = None
            if click_through or event.dblclick:
                hit_stack = self.get_hit_stack(event)
                resolution = kernel.resolve(
                    hit_stack,
                    cycle_from=last if click_through else None,
                    wrap=click_through,
                )
            else:
                resolution = self._resolve_top_hit(event)
            picked_element = resolution.target
            self._last_pick_blocked = resolution.blocked
            picked_is_selected = any(
                target is picked_element for target in selected
            )

            if (
                self.selection.alignment_reference_mode == "key_object"
                and len(self.selection.targets) < 2
            ):
                self.selection._restore_alignment_reference_state(
                    "selection", None
                )
            if (
                self.selection.alignment_reference_mode == "key_object"
                and len(self.selection.targets) >= 2
                and picked_is_selected
                and not click_through
                and not shift
            ):
                self.selection.set_alignment_key(picked_element)

            if event.dblclick and picked_element is not None:
                if self.enter_isolation(picked_element, notify=False):
                    assert hit_stack is not None
                    inner = kernel.resolve(hit_stack).target
                    if inner is not None:
                        self.select_element(inner, event)
                    else:
                        self._notify_selected_element_changed()
                    return

            # if the element is a grabber, store it
            if getattr(self, "_last_pick_blocked", False):
                self._start_marquee_selection(event)
                return
            if isinstance(picked_element, GrabberGeneric):
                self.grab_element = picked_element
            elif shift and picked_is_selected:
                remaining = [
                    target for target in selected if target is not picked_element
                ]
                primary = (
                    self.selected_element
                    if any(
                        target is self.selected_element for target in remaining
                    )
                    else (remaining[-1] if remaining else None)
                )
                self.select_elements(remaining, primary=primary)
                return
            elif (
                isinstance(picked_element, Axes)
                and not picked_is_selected
                and not event.dblclick
            ):
                self._start_marquee_selection(event, click_element=picked_element)
                return
            elif picked_element is None:
                self._start_marquee_selection(event)
                return
            # Keep a multi-selection only when the resolved target is one of
            # its members. An overlapped old selection cannot suppress the
            # visually foreground target.
            elif (
                not picked_is_selected
                or event.dblclick
                or click_through
            ):
                self.select_element(picked_element, event)
                picked_is_selected = any(
                    target.target is picked_element
                    for target in self.selection.targets
                )

            # if we have a grabber, notify it
            if self.grab_element:
                self.grab_element.button_press_event(event)
            # if not, notify the selected element
            elif picked_is_selected and self.selection.operation_support(
                TransformOperation.TRANSLATE
            ).supported:
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
        _prevalidated: bool = False,
    ):
        """Select one or more artists through the same model used by the canvas."""
        if additive:
            elements = [target.target for target in self.selection.targets] + list(
                elements
            )
        if _prevalidated:
            elements = list(elements)
        else:
            elements, primary = self._resolve_selectable_elements(elements, primary)
        elements, primary = self._normalize_selection(
            elements,
            primary,
            preserve_axes=preserve_axes,
            prefer_containers=prefer_containers,
        )

        current = [target.target for target in self.selection.targets]
        self._selection_semantic_mode = self.selection_mode
        if primary == self.selected_element and current == elements:
            return elements

        previous_alignment_key = self.selection.alignment_key
        same_membership = len(current) == len(elements) and {
            id(element) for element in current
        } == {id(element) for element in elements}
        self.selection.clear_targets(
            preserve_rotation_pivot=same_membership
        )
        self.selection.configure_target_overlay(len(elements))

        self.selection._batch_add_targets = True
        self.selection._batch_targets_prevalidated = True
        try:
            for element in elements:
                if element != primary:
                    self.selection.add_target(element, update=False)

            if primary is not None:
                self.on_select(primary, event)
            else:
                self.on_select(None, event)
        finally:
            self.selection._batch_add_targets = False
            self.selection._batch_targets_prevalidated = False
        self.selected_element = primary
        if self.selection.targets:
            self.selection.update_extent()
        if self.selection.alignment_reference_mode == "key_object":
            retained_key = next(
                (
                    element
                    for element in elements
                    if element is previous_alignment_key
                ),
                None,
            )
            self.selection._restore_alignment_reference_state(
                "key_object", retained_key
            )
        else:
            self.selection._notify_alignment_state_changed()
        self._notify_selected_element_changed()
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
        if self._cancel_active_pointer_transform():
            return
        print("back edit")
        notification_revision = int(
            getattr(self, "_selection_notification_revision", 0)
        )
        self.figure.change_tracker.backEdit()
        current = [target.target for target in self.selection.targets]
        self.selected_element = current[-1] if current else None
        self._update_interaction_controls()
        if int(getattr(self, "_selection_notification_revision", 0)) == (
            notification_revision
        ):
            self._notify_selected_element_changed()

    def redo(self):
        if self._cancel_active_pointer_transform():
            return
        print("forward edit")
        notification_revision = int(
            getattr(self, "_selection_notification_revision", 0)
        )
        self.figure.change_tracker.forwardEdit()
        current = [target.target for target in self.selection.targets]
        self.selected_element = current[-1] if current else None
        self._update_interaction_controls()
        if int(getattr(self, "_selection_notification_revision", 0)) == (
            notification_revision
        ):
            self._notify_selected_element_changed()

    def _cancel_active_pointer_transform(self) -> bool:
        """Rollback an in-flight move/resize and keep the selection intact."""

        active = None
        grabber = self.grab_element
        if grabber is not None and bool(getattr(grabber, "got_artist", False)):
            active = grabber
        elif bool(getattr(self.selection, "got_artist", False)):
            active = self.selection
        if active is None or not hasattr(self.selection, "move_start_states"):
            return False

        try:
            self.selection._restore_move_start(strict=False)
            self.selection.has_moved = False
            self.selection._clear_move_transaction()
        finally:
            active.cancel_event()
            self.grab_element = None
            scene = getattr(self.figure, "_pyl_scene", None)
            if scene is not None:
                scene.grabber_pressed = None
        canvas = self.figure.canvas
        if hasattr(canvas, "schedule_draw"):
            canvas.schedule_draw()
        else:
            canvas.draw_idle()
        return True

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
            self._cancel_active_pointer_transform()
            self.figure.change_tracker.save()
        if event.key == "ctrl+z":
            self.undo()
        if event.key == "ctrl+y":
            self.redo()
        if event.key == "escape":
            pivot_active = bool(
                getattr(
                    self.selection.rotation_grabber,
                    "pivot_got_artist",
                    False,
                )
            )
            if pivot_active:
                self.selection.rotation_grabber.cancel_pivot_event(
                    restore=True
                )
                self.grab_element = None
                scene = getattr(self.figure, "_pyl_scene", None)
                if scene is not None:
                    scene.grabber_pressed = None
                return
            rotation_active = (
                getattr(self.selection, "rotation_drag_mode", None) is not None
            )
            if rotation_active:
                try:
                    self.selection.cancel_rotation()
                finally:
                    self.selection.rotation_grabber.cancel_event()
                    if self.grab_element is not None:
                        self.grab_element.cancel_event()
                    self.grab_element = None
                    scene = getattr(self.figure, "_pyl_scene", None)
                    if scene is not None:
                        scene.grabber_pressed = None
                return
            if self._cancel_active_pointer_transform():
                return
            if self._ensure_selection_kernel().scope_root is not None:
                self.exit_isolation()
                return
            self.selection.clear_targets()
            self.selected_element = None
            self.on_select(None, None)
            self._notify_selected_element_changed()
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
    """Rotation handle plus an optional movable shared-pivot marker."""

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
        self.pivot_marker = RotationPivotMarker(-3, -3, 6, 6, scene)
        self.pivot_marker.view = scene.view
        self.pivot_marker.grabber = self
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
        if np.linalg.norm(handle - pivot) <= 12.0:
            candidates = np.array(
                [
                    [bounds[2] + 24.0, (bounds[1] + bounds[3]) / 2],
                    [(bounds[0] + bounds[2]) / 2, bounds[1] - 24.0],
                    [bounds[0] - 24.0, (bounds[1] + bounds[3]) / 2],
                ],
                dtype=float,
            )
            handle = candidates[
                int(np.argmax(np.linalg.norm(candidates - pivot, axis=1)))
            ]
        self.set_xy(handle)
        self._position_pivot(pivot)
        movable = self.parent.custom_rotation_pivot_supported()
        self.pivot_marker.setAcceptedMouseButtons(
            QtCore.Qt.LeftButton if movable else QtCore.Qt.NoButton
        )
        self.pivot_marker.setCursor(
            QtCore.Qt.SizeAllCursor if movable else QtCore.Qt.ArrowCursor
        )
        self.pivot_marker.setToolTip(
            "Drag to move the shared rotation pivot; double-click to reset"
            if movable
            else "This object rotates around its fixed native pivot"
        )
        self.line.setVisible(True)
        self.pivot_marker.setVisible(True)
        self.handle.setVisible(True)

    def _position_pivot(self, pivot: Sequence[float]) -> None:
        self.line.setLine(
            float(pivot[0]), float(pivot[1]), float(self.xy[0]), float(self.xy[1])
        )
        self.pivot_marker.setRect(
            float(pivot[0]) - 3, float(pivot[1]) - 3, 6, 6
        )

    def on_pivot_motion(self, event: MouseEvent) -> None:
        if getattr(self, "pivot_got_artist", False):
            self.parent.set_rotation_pivot(
                (event.x, event.y), notify=False
            )

    def pivot_button_press_event(self, event: MouseEvent) -> None:
        if not self.parent.custom_rotation_pivot_supported():
            raise UnsupportedArtistError(
                "A custom rotation pivot requires a complete shared "
                "rigid-rotation plan"
            )
        self.pivot_drag_start_state = self.parent.custom_rotation_pivot_state()
        self.pivot_got_artist = True
        self._pivot_c1 = self.figure.canvas.mpl_connect(
            "motion_notify_event", self.on_pivot_motion
        )
        self.parent.set_rotation_pivot(
            (event.x, event.y), notify=False
        )

    def pivot_button_release_event(self, event: MouseEvent) -> None:
        if not getattr(self, "pivot_got_artist", False):
            return
        try:
            self.parent.set_rotation_pivot((event.x, event.y))
        finally:
            self.cancel_pivot_event()

    def cancel_pivot_event(self, *, restore: bool = False) -> None:
        was_active = bool(getattr(self, "pivot_got_artist", False))
        start_state = getattr(self, "pivot_drag_start_state", None)
        if was_active:
            self.pivot_got_artist = False
            connection = getattr(self, "_pivot_c1", None)
            if connection is not None:
                self.figure.canvas.mpl_disconnect(connection)
        if restore and was_active:
            self.parent._set_custom_rotation_pivot_state(start_state)
        self._pivot_c1 = None
        self.pivot_drag_start_state = None

    def reset_pivot_event(self) -> None:
        self.cancel_pivot_event()
        self.parent.reset_rotation_pivot()

    def cancel_event(self) -> None:
        super().cancel_event()
        self.cancel_pivot_event(restore=True)

    def hide(self) -> None:
        self.cancel_pivot_event(restore=True)
        self.pivot_marker.setAcceptedMouseButtons(QtCore.Qt.NoButton)
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


class RotationPivotMarker(QtWidgets.QGraphicsEllipseItem):
    """Qt overlay item that moves editor pivot state, never figure Artists."""

    def _canvas_event(self, event):
        x, y = scene_point_to_canvas_pixels(self.view, event.scenePos())
        return MyEvent(x, y)

    def _claim_event(self) -> None:
        self.view.grabber_found = True
        self.scene().grabber_pressed = self

    def mousePressEvent(self, event):
        QtWidgets.QGraphicsEllipseItem.mousePressEvent(self, event)
        if not self.grabber.parent.custom_rotation_pivot_supported():
            event.ignore()
            return
        self._claim_event()
        self.grabber.pivot_button_press_event(self._canvas_event(event))
        event.accept()

    def mouseReleaseEvent(self, event):
        QtWidgets.QGraphicsEllipseItem.mouseReleaseEvent(self, event)
        self.scene().grabber_pressed = None
        self.view.grabber_found = True
        self.grabber.pivot_button_release_event(self._canvas_event(event))
        event.accept()

    def mouseDoubleClickEvent(self, event):
        QtWidgets.QGraphicsEllipseItem.mouseDoubleClickEvent(self, event)
        if not self.grabber.parent.custom_rotation_pivot_supported():
            event.ignore()
            return
        self._claim_event()
        self.grabber.reset_pivot_event()
        self.scene().grabber_pressed = None
        event.accept()


class MyEvent:
    def __init__(self, x, y):
        self.x = x
        self.y = y
