"""Matplotlib/Qt integration for the renderer-independent smart-guide kernel.

The expensive part of smart guides is measuring live Artists, not querying the
sorted scalar index.  This module therefore keeps a revisioned scene snapshot,
filters it through the active Object/Direct Selection scope at gesture start,
and reuses one immutable :class:`~pylustrator.smart_guides.GuideCandidateIndex`
for every pointer frame.  The computational kernel remains Qt-free; only the
small overlay renderer below knows about QGraphicsScene.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Iterable, Sequence

import numpy as np
from matplotlib.artist import Artist
from matplotlib.text import Text
from qtpy import QtCore, QtGui, QtWidgets

from .smart_guides import (
    Axis,
    DisplayBounds,
    EqualGapOverlay,
    ExplicitAnchor,
    GuideCandidateIndex,
    GuideLine,
    GuideObject,
    GuideSnapshot,
    MovingGeometry,
    SnapPlan,
    StaleGuideSnapshotError,
)


def _finite_bounds(points) -> DisplayBounds | None:
    try:
        values = np.asarray(points, dtype=float)
    except (TypeError, ValueError):
        return None
    if values.ndim != 2 or values.shape[1] < 2:
        return None
    values = values[:, :2]
    values = values[np.all(np.isfinite(values), axis=1)]
    if not len(values):
        return None
    low = np.min(values, axis=0)
    high = np.max(values, axis=0)
    return DisplayBounds(float(low[0]), float(low[1]), float(high[0]), float(high[1]))


def _text_features(
    target: Artist, renderer
) -> tuple[float | None, tuple[ExplicitAnchor, ...]]:
    """Return an unrotated Text baseline and its insertion anchor in pixels."""

    if not isinstance(target, Text) or target.get_text() == "":
        return None, ()
    try:
        position = target.get_transform().transform(target.get_unitless_position())
        position = np.asarray(position, dtype=float)
        if position.shape != (2,) or not np.all(np.isfinite(position)):
            return None, ()
        anchor = ExplicitAnchor(float(position[0]), float(position[1]), "insertion")
    except (AttributeError, TypeError, ValueError, RuntimeError):
        return None, ()

    baseline = None
    try:
        rotation = float(target.get_rotation()) % 360.0
        if np.isclose(rotation, 0.0, atol=1e-12):
            _bbox, layout, _descent = target._get_layout(renderer)
            if layout:
                # Matplotlib stores each line's baseline-relative y offset as
                # the fourth item in the private layout tuple.  This is the
                # same layout consumed by Text.draw and get_window_extent.
                offset_y = float(layout[0][3])
                value = float(position[1] + offset_y)
                if np.isfinite(value):
                    baseline = value
    except (AttributeError, IndexError, TypeError, ValueError, RuntimeError):
        pass
    return baseline, (anchor,)


@dataclass(frozen=True)
class _SceneGuideEntry:
    artist: Artist
    guide: GuideObject


@dataclass(frozen=True)
class _SceneGuideCache:
    revision: int
    inventory_ids: tuple[int, ...]
    semantic_key: tuple[str, tuple[int, ...]]
    entries: tuple[_SceneGuideEntry, ...]


@dataclass
class _PendingSceneGuideCapture:
    revision: int
    inventory_ids: tuple[int, ...]
    semantic_key: tuple[str, tuple[int, ...]]
    artists: tuple[tuple[int, Artist], ...]
    next_index: int = 0
    entries: list[_SceneGuideEntry] = field(default_factory=list)


def _scene_revision(manager) -> int:
    return int(getattr(manager, "_interaction_revision", 0))


def _scene_semantic_key(manager) -> tuple[str, tuple[int, ...]]:
    kernel = manager._ensure_selection_kernel()
    return kernel.mode.value, tuple(id(scope.root) for scope in kernel.scopes)


def _scene_state(manager):
    roster = manager._selectable_roster_snapshot()
    return (
        roster.artists,
        _scene_revision(manager),
        roster.source_ids,
        _scene_semantic_key(manager),
    )


def _cache_matches(
    cache,
    revision: int,
    inventory_ids: tuple[int, ...],
    semantic_key: tuple[str, tuple[int, ...]],
) -> bool:
    return (
        isinstance(cache, _SceneGuideCache)
        and cache.revision == revision
        and cache.inventory_ids is inventory_ids
        and cache.semantic_key == semantic_key
    )


def _measure_scene_guide(
    manager,
    scene,
    renderer,
    geometry,
    scope_id: str,
    order: int,
    artist: Artist,
) -> _SceneGuideEntry | None:
    try:
        if (
            not manager._is_artist_attached(artist)
            or not artist.get_visible()
            or scene.is_locked(artist)
            or scene.is_explicitly_hidden(artist)
            # Tick labels owned by an Axis formatter are generated output, not
            # independently movable editor objects.  They remain click-
            # selectable for text/style edits, but would otherwise create a
            # dense, unstable guide lattice in scientific figures.
            or getattr(
                artist,
                "_pylustrator_formatter_owned_tick_label",
                False,
            )
            or (isinstance(artist, Text) and artist.get_text() == "")
        ):
            return None
        measured_bounds = geometry.selection_bounds(artist)
        if measured_bounds is None:
            return None
        bounds = DisplayBounds(*measured_bounds)
        baseline, anchors = _text_features(artist, renderer)
        z_order = float(artist.get_zorder())
        if not np.isfinite(z_order):
            return None
        stable_id = f"{order:08d}:{type(artist).__name__}:{id(artist):x}"
        guide = GuideObject(
            stable_id,
            bounds,
            z_order=z_order,
            order=order,
            baseline_y=baseline,
            anchors=anchors,
            scope_id=scope_id,
        )
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
    return _SceneGuideEntry(artist, guide)


def _capture_scene_guides(manager) -> _SceneGuideCache:
    artists, revision, inventory_ids, semantic_key = _scene_state(manager)
    cached = getattr(manager, "_smart_guide_scene_cache", None)
    if _cache_matches(cached, revision, inventory_ids, semantic_key):
        return cached

    # A synchronous caller supersedes any incomplete idle capture.  This path
    # is retained for tests and non-Qt embedders; normal pointer handling only
    # consumes a completed idle cache and otherwise keeps the legacy snaps.
    manager._smart_guide_pending_capture = None
    artists = tuple(_logical_source_artists(manager, artists))
    renderer = manager.figure.canvas.get_renderer()
    geometry = manager._ensure_display_geometry_cache()
    scene = manager._ensure_editor_scene()
    entries: list[_SceneGuideEntry] = []
    scope_id = f"figure:{id(manager.figure)}"
    with geometry.snapshot():
        for order, artist in enumerate(artists):
            entry = _measure_scene_guide(
                manager, scene, renderer, geometry, scope_id, order, artist
            )
            if entry is not None:
                entries.append(entry)

    result = _SceneGuideCache(
        revision,
        inventory_ids,
        semantic_key,
        tuple(entries),
    )
    manager._smart_guide_scene_cache = result
    return result


def _cached_scene_guides(manager) -> _SceneGuideCache | None:
    _artists, revision, inventory_ids, semantic_key = _scene_state(manager)
    cached = getattr(manager, "_smart_guide_scene_cache", None)
    return (
        cached
        if _cache_matches(cached, revision, inventory_ids, semantic_key)
        else None
    )


def schedule_smart_guide_warmup(manager, *, batch_budget_ms: float = 4.0) -> bool:
    """Measure one scene in small Qt-idle slices without delaying pointer input."""

    if not bool(getattr(manager, "_smart_guide_idle_warmup_enabled", True)):
        return False
    if _cached_scene_guides(manager) is not None:
        return False
    if QtWidgets.QApplication.instance() is None:
        return False
    artists, revision, inventory_ids, semantic_key = _scene_state(manager)
    pending = getattr(manager, "_smart_guide_pending_capture", None)
    if (
        isinstance(pending, _PendingSceneGuideCapture)
        and pending.revision == revision
        and pending.inventory_ids is inventory_ids
        and pending.semantic_key == semantic_key
    ):
        return False
    logical = tuple(enumerate(_logical_source_artists(manager, artists)))
    pending = _PendingSceneGuideCapture(
        revision,
        inventory_ids,
        semantic_key,
        logical,
    )
    manager._smart_guide_pending_capture = pending
    budget = max(float(batch_budget_ms), 0.25) / 1000.0

    def step() -> None:
        if getattr(manager, "_smart_guide_pending_capture", None) is not pending:
            return
        selection = getattr(manager, "selection", None)
        grabber = getattr(manager, "grab_element", None)
        if bool(getattr(selection, "got_artist", False)) or bool(
            getattr(grabber, "got_artist", False)
        ):
            # Never measure deferred preview geometry.  Release/draw schedules
            # a fresh idle capture from the committed or restored scene.
            manager._smart_guide_pending_capture = None
            return
        _, current_revision, current_ids, current_key = _scene_state(
            manager
        )
        if (
            current_revision != pending.revision
            or current_ids is not pending.inventory_ids
            or current_key != pending.semantic_key
        ):
            manager._smart_guide_pending_capture = None
            return
        renderer = manager.figure.canvas.get_renderer()
        geometry = manager._ensure_display_geometry_cache()
        scene = manager._ensure_editor_scene()
        scope_id = f"figure:{id(manager.figure)}"
        deadline = perf_counter() + budget
        with geometry.snapshot():
            while pending.next_index < len(pending.artists):
                order, artist = pending.artists[pending.next_index]
                pending.next_index += 1
                entry = _measure_scene_guide(
                    manager, scene, renderer, geometry, scope_id, order, artist
                )
                if entry is not None:
                    pending.entries.append(entry)
                if perf_counter() >= deadline:
                    break
        if pending.next_index < len(pending.artists):
            QtCore.QTimer.singleShot(0, step)
            return
        result = _SceneGuideCache(
            pending.revision,
            pending.inventory_ids,
            pending.semantic_key,
            tuple(pending.entries),
        )
        manager._smart_guide_scene_cache = result
        manager._smart_guide_pending_capture = None

    QtCore.QTimer.singleShot(0, step)
    return True


def invalidate_smart_guide_cache(manager) -> None:
    """Drop measured scene guides after a renderer/inventory revision."""

    setattr(manager, "_smart_guide_scene_cache", None)
    setattr(manager, "_smart_guide_pending_capture", None)


def _logical_source_artists(manager, artists: Iterable[Artist]) -> list[Artist]:
    kernel = manager._ensure_selection_kernel()
    mapped = kernel.map_artists(artists)
    try:
        normalized, _primary = manager._normalize_selection(
            mapped,
            preserve_axes=True,
            prefer_containers=False,
        )
    except (AttributeError, TypeError, ValueError, RuntimeError):
        normalized = mapped
    root = kernel.scope_root
    return [artist for artist in normalized if artist is not root]


def _is_selected_descendant(manager, selected: Sequence[Artist], artist: Artist) -> bool:
    scene = manager._ensure_editor_scene()
    for target in selected:
        if artist is target or scene.contains(target, artist):
            return True
    return False


def _translated_moving_geometry(
    source: MovingGeometry, delta_px: tuple[float, float]
) -> MovingGeometry:
    dx, dy = (float(delta_px[0]), float(delta_px[1]))
    return MovingGeometry(
        source.bounds.translated((dx, dy)),
        baseline_y=(None if source.baseline_y is None else source.baseline_y + dy),
        anchors=tuple(
            ExplicitAnchor(anchor.x + dx, anchor.y + dy, anchor.name)
            for anchor in source.anchors
        ),
        scope_id=source.scope_id,
    )


class SmartGuideOverlay:
    """Render one small SnapPlan in the existing canvas QGraphicsScene."""

    def __init__(self, figure) -> None:
        self.figure = figure
        parent = getattr(figure, "_pyl_graphics_scene_snapparent", None)
        self.item = None
        if parent is None:
            return
        try:
            item = QtWidgets.QGraphicsPathItem(parent)
            pen = QtGui.QPen(QtGui.QColor("#D414D4"), 1.5)
            pen.setCosmetic(True)
            pen.setStyle(QtCore.Qt.DashLine)
            item.setPen(pen)
            item.setBrush(QtGui.QBrush(QtCore.Qt.NoBrush))
            item.setZValue(950)
            item.setToolTip("Smart Guides")
            self.item = item
        except (AttributeError, RuntimeError, TypeError):
            self.item = None

    def render(self, plan: SnapPlan) -> None:
        if self.item is None:
            return
        path = QtGui.QPainterPath()

        def segment(x0: float, y0: float, x1: float, y1: float) -> None:
            # This item is a child of the same selection-scene origin used by
            # grabbers.  That parent already converts Matplotlib's bottom-left
            # display pixels to Qt's top-left scene coordinates (and applies
            # HiDPI scaling).  A second y flip here mirrors the guide away from
            # the geometry that actually snapped.
            path.moveTo(float(x0), float(y0))
            path.lineTo(float(x1), float(y1))

        for overlay in plan.overlays:
            if isinstance(overlay, GuideLine):
                if overlay.axis is Axis.X:
                    segment(
                        overlay.position_px,
                        overlay.span_px[0],
                        overlay.position_px,
                        overlay.span_px[1],
                    )
                else:
                    segment(
                        overlay.span_px[0],
                        overlay.position_px,
                        overlay.span_px[1],
                        overlay.position_px,
                    )
                continue
            if not isinstance(overlay, EqualGapOverlay):
                continue
            tick = 3.0
            for start, end in overlay.intervals_px:
                if overlay.axis is Axis.X:
                    segment(start, overlay.cross_position_px, end, overlay.cross_position_px)
                    segment(
                        start,
                        overlay.cross_position_px - tick,
                        start,
                        overlay.cross_position_px + tick,
                    )
                    segment(
                        end,
                        overlay.cross_position_px - tick,
                        end,
                        overlay.cross_position_px + tick,
                    )
                else:
                    segment(overlay.cross_position_px, start, overlay.cross_position_px, end)
                    segment(
                        overlay.cross_position_px - tick,
                        start,
                        overlay.cross_position_px + tick,
                        start,
                    )
                    segment(
                        overlay.cross_position_px - tick,
                        end,
                        overlay.cross_position_px + tick,
                        end,
                    )
        try:
            self.item.setPath(path)
            self.item.setVisible(bool(plan.overlays))
        except RuntimeError:
            self.item = None

    def hide(self) -> None:
        if self.item is None:
            return
        try:
            self.item.setPath(QtGui.QPainterPath())
            self.item.setVisible(False)
        except RuntimeError:
            self.item = None

    def close(self) -> None:
        item = self.item
        self.item = None
        if item is None:
            return
        try:
            scene = item.scene()
            item.setParentItem(None)
            if scene is not None:
                scene.removeItem(item)
        except RuntimeError:
            pass


class SmartGuideDragSession:
    """One immutable-source smart-guide transaction for a move gesture."""

    def __init__(
        self,
        manager,
        source_guides: Sequence[GuideObject],
        moving: MovingGeometry,
        *,
        revision: int,
        tolerance_px: float,
        include_equal_gaps: bool,
    ) -> None:
        self.manager = manager
        self.source_guides = tuple(source_guides)
        self._index: GuideCandidateIndex | None = None
        self.moving = moving
        self.revision = int(revision)
        self.tolerance_px = float(tolerance_px)
        self.include_equal_gaps = bool(include_equal_gaps)
        self.overlay = SmartGuideOverlay(manager.figure)
        self.last_plan: SnapPlan | None = None
        self.active = True
        previous = getattr(manager, "_active_smart_guide_session", None)
        if previous is not None and previous is not self:
            previous.close()
        manager._active_smart_guide_session = self

    @property
    def index(self) -> GuideCandidateIndex:
        index = self._index
        if index is None:
            snapshot = GuideSnapshot.capture(
                self.source_guides,
                revision=self.revision,
            )
            index = GuideCandidateIndex(
                snapshot,
                include_equal_gaps=self.include_equal_gaps,
            )
            self._index = index
        return index

    def query(
        self,
        delta_px: Sequence[float],
        *,
        axes: frozenset[Axis] | None = None,
        render: bool = True,
    ) -> SnapPlan:
        if not self.active or _scene_revision(self.manager) != self.revision:
            self.invalidate()
            raise StaleGuideSnapshotError(
                "smart-guide scene changed during the active drag gesture"
            )
        delta = (float(delta_px[0]), float(delta_px[1]))
        moving = _translated_moving_geometry(self.moving, delta)
        plan = self.index.query(
            moving,
            tolerance_px=self.tolerance_px,
            expected_fingerprint=self.index.snapshot.fingerprint,
            include_equal_gaps=self.include_equal_gaps,
            axes=axes,
        )
        if render:
            self.accept(plan)
        return plan

    def accept(self, plan: SnapPlan) -> None:
        if not self.active:
            raise StaleGuideSnapshotError("smart-guide session is no longer active")
        plan.require_fingerprint(self.index.snapshot.fingerprint)
        self.last_plan = plan
        self.overlay.render(plan)

    def hide(self) -> None:
        self.last_plan = None
        self.overlay.hide()

    def invalidate(self) -> None:
        self.active = False
        self.hide()

    def close(self) -> None:
        self.active = False
        self.last_plan = None
        self.overlay.close()
        if getattr(self.manager, "_active_smart_guide_session", None) is self:
            self.manager._active_smart_guide_session = None


def create_smart_guide_drag_session(
    manager,
    selection,
    selected: Iterable[Artist],
    *,
    tolerance_px: float = 5.0,
    include_equal_gaps: bool = True,
    allow_cold_capture: bool = True,
) -> SmartGuideDragSession | None:
    """Capture exact source geometry and return one drag-session planner."""

    selected = tuple(dict.fromkeys(selected))
    if not selected:
        return None
    start_p1 = np.asarray(getattr(selection, "start_p1", selection.p1), dtype=float)
    start_p2 = np.asarray(getattr(selection, "start_p2", selection.p2), dtype=float)
    if start_p1.shape != (2,) or start_p2.shape != (2,):
        return None
    moving_bounds = DisplayBounds(
        float(min(start_p1[0], start_p2[0])),
        float(min(start_p1[1], start_p2[1])),
        float(max(start_p1[0], start_p2[0])),
        float(max(start_p1[1], start_p2[1])),
    )
    if not moving_bounds.is_finite_and_ordered:
        return None

    cache = _cached_scene_guides(manager)
    if cache is None:
        if not allow_cold_capture:
            schedule_smart_guide_warmup(manager)
            return None
        cache = _capture_scene_guides(manager)
    entry_by_id = {id(entry.artist): entry for entry in cache.entries}
    source_guides = [
        entry.guide
        for entry in cache.entries
        if not _is_selected_descendant(manager, selected, entry.artist)
    ]
    if not source_guides:
        return None

    baseline = None
    anchors: tuple[ExplicitAnchor, ...] = ()
    if len(selected) == 1 and isinstance(selected[0], Text):
        selected_entry = entry_by_id.get(id(selected[0]))
        if selected_entry is not None:
            baseline = selected_entry.guide.baseline_y
            anchors = selected_entry.guide.anchors

    scope_id = source_guides[0].scope_id
    moving = MovingGeometry(
        moving_bounds,
        baseline_y=baseline,
        anchors=anchors,
        scope_id=scope_id,
    )
    return SmartGuideDragSession(
        manager,
        source_guides,
        moving,
        revision=cache.revision,
        tolerance_px=tolerance_px,
        include_equal_gaps=include_equal_gaps,
    )


__all__ = [
    "SmartGuideDragSession",
    "SmartGuideOverlay",
    "create_smart_guide_drag_session",
    "invalidate_smart_guide_cache",
    "schedule_smart_guide_warmup",
]
