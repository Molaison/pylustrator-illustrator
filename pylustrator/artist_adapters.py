"""Artist-specific interaction contracts.

Pylustrator edits objects in display coordinates, while Matplotlib stores each
artist in its own native coordinate system.  This module is the single boundary
between those two worlds.  Every editable artist is resolved to one adapter
which owns its geometry, capabilities, mutations, undo snapshots, and change
records.

The registry deliberately resolves by MRO specificity.  That makes subclass
semantics explicit: for example, ``Annotation`` is handled before ``Text`` and
``ConnectionPatch`` before ``FancyArrowPatch`` without depending on a fragile
order of ``isinstance`` branches.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from threading import RLock
from typing import Iterable, Optional, Sequence

import matplotlib as mpl
import numpy as np
from matplotlib.artist import Artist

try:  # starting from mpl version 3.6.0
    from matplotlib.axes import Axes
except ImportError:
    from matplotlib.axes._subplots import Axes
from matplotlib.collections import LineCollection, PathCollection, PolyCollection
from matplotlib.image import AxesImage
from matplotlib.legend import Legend
from matplotlib.lines import Line2D
from matplotlib.path import Path
from matplotlib.patches import (
    ConnectionPatch,
    Ellipse,
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
    Transform,
)
from packaging import version

from .editor_model import EditorGroup
from .operations import OperationSupport, TransformOperation


_CHANGE_RECORDING_ENABLED = ContextVar(
    "pylustrator_change_recording_enabled", default=True
)


@contextmanager
def suspend_change_recording():
    """Restore interaction state without emitting a second set of changes."""

    token = _CHANGE_RECORDING_ENABLED.set(False)
    try:
        yield
    finally:
        _CHANGE_RECORDING_ENABLED.reset(token)


def checkXLabel(target: Artist):
    """Return the owning axes when *target* is its x-axis label."""
    figure = getattr(target, "figure", None)
    for axes in getattr(figure, "axes", []):
        if axes.xaxis.get_label() is target:
            return axes
    return None


def checkYLabel(target: Artist):
    """Return the owning axes when *target* is its y-axis label."""
    figure = getattr(target, "figure", None)
    for axes in getattr(figure, "axes", []):
        if axes.yaxis.get_label() is target:
            return axes
    return None


def cache_property(obj, name: str) -> None:
    """Cache an expensive Matplotlib getter until its paired setter runs."""
    if getattr(obj, f"_pylustrator_cached_{name}", False) is True:
        return
    setattr(obj, f"_pylustrator_cached_{name}", True)
    getter = getattr(obj, f"get_{name}")
    setter = getattr(obj, f"set_{name}")

    def new_getter(*args, **kwargs):
        if getattr(obj, f"_pylustrator_cache_{name}", None) is None:
            setattr(obj, f"_pylustrator_cache_{name}", getter(*args, **kwargs))
        return getattr(obj, f"_pylustrator_cache_{name}", None)

    def new_setter(*args, **kwargs):
        result = setter(*args, **kwargs)
        setattr(obj, f"_pylustrator_cache_{name}", None)
        return result

    setattr(obj, f"get_{name}", new_getter)
    setattr(obj, f"set_{name}", new_setter)


def legend_loc_transform(legend: Legend):
    return BboxTransformFrom(legend.get_bbox_to_anchor())


def legend_anchor_transform(legend: Legend):
    return getattr(
        legend.get_bbox_to_anchor(),
        "_transform",
        BboxTransformTo(legend.parent.bbox),
    )


def legend_anchor_is_point(legend: Legend) -> bool:
    bbox = legend.get_bbox_to_anchor()
    return bbox.width == 0 and bbox.height == 0


def set_legend_point_anchor_display(
    legend: Legend, point: Sequence[float], transform: Optional[Transform] = None
) -> None:
    if transform is None:
        transform = legend_anchor_transform(legend)
    legend.set_bbox_to_anchor(
        tuple(float(x) for x in transform.inverted().transform(point)),
        transform=transform,
    )


def iter_legend_children(legend: Legend) -> tuple[Artist, ...]:
    """Return every persistent, directly editable child of a Legend."""

    children = [*getattr(legend, "legend_handles", []), *legend.get_texts()]
    title = legend.get_title()
    if title is not None:
        children.append(title)
    return tuple(dict.fromkeys(child for child in children if child is not None))


def iter_figure_legends(figure) -> tuple[Legend, ...]:
    """Return one authoritative inventory of live Legends below *figure*.

    Matplotlib stores legends in three places: ``Figure.legends``, the current
    ``Axes.legend_``, and ``artists`` lists for additional retained legends.
    Selection and persistent-reference resolution must see the same union.
    """

    legends = []
    seen = set()

    def add(legend) -> None:
        if isinstance(legend, Legend) and id(legend) not in seen:
            seen.add(id(legend))
            legends.append(legend)

    def visit(owner) -> None:
        for legend in getattr(owner, "legends", []):
            add(legend)
        for artist in getattr(owner, "artists", []):
            add(artist)
        for axes in getattr(owner, "axes", []):
            add(axes.get_legend())
            for artist in axes.artists:
                add(artist)
        for subfigure in getattr(owner, "subfigs", []):
            visit(subfigure)

    visit(figure)
    return tuple(legends)


def legend_display_loc(legend: Legend) -> np.ndarray:
    if legend_anchor_is_point(legend):
        return np.array(legend.get_bbox_to_anchor().p0)
    bbox = legend.get_frame().get_bbox()
    if isinstance(legend._get_loc(), int):
        return np.array([bbox.x0, bbox.y0])
    return BboxTransformTo(legend.get_bbox_to_anchor()).transform(legend._get_loc())


@dataclass(frozen=True)
class ArtistCapabilities:
    """Operations an adapter can perform without lossy approximations."""

    can_select: bool = False
    can_translate: bool = False
    can_resize: bool = False
    can_snapshot: bool = False
    can_serialize: bool = False
    fixed_aspect: bool = False
    can_rotate: bool = False

    @property
    def editable(self) -> bool:
        return self.can_select and self.can_translate


@dataclass(frozen=True)
class ChangeRecord:
    """One adapter-owned instruction for updating the ChangeTracker."""

    kind: str
    target: Artist
    command: Optional[str] = None

    @classmethod
    def command_change(cls, target: Artist, command: str) -> ChangeRecord:
        return cls("command", target, command)

    @classmethod
    def text_change(cls, target: Text) -> ChangeRecord:
        return cls("text", target)

    @classmethod
    def legend_change(cls, target: Legend) -> ChangeRecord:
        return cls("legend", target)

    @classmethod
    def axes_change(cls, target: Axes) -> ChangeRecord:
        return cls("axes", target)

    def apply(self, tracker) -> None:
        if self.kind == "command":
            tracker.addChange(self.target, self.command)
        elif self.kind == "text":
            tracker.addNewTextChange(self.target)
        elif self.kind == "legend":
            tracker.addNewLegendChange(self.target)
        elif self.kind == "axes":
            tracker.addNewAxesChange(self.target)
        else:  # pragma: no cover - protects third-party adapter mistakes
            raise ValueError(f"Unknown change-record kind: {self.kind!r}")


class UnsupportedArtistError(TypeError):
    """Raised when an adapter is asked to perform an unsupported operation."""


class ArtistAdapter:
    """Base display/native interaction protocol for one Matplotlib artist."""

    default_capabilities = ArtistCapabilities()
    unsupported_operation_reasons: dict[TransformOperation, str] = {}

    def __init__(self, target: Artist):
        self.target = target
        self.figure = getattr(target, "figure", None)

    @classmethod
    def capabilities_for(cls, target: Artist) -> ArtistCapabilities:
        return cls.default_capabilities

    @property
    def capabilities(self) -> ArtistCapabilities:
        return self.capabilities_for(self.target)

    @property
    def supported(self) -> bool:
        return self.capabilities.editable

    def operation_support(
        self, operation: TransformOperation | str
    ) -> OperationSupport:
        operation = TransformOperation.coerce(operation)
        capabilities = self.capabilities
        legacy_support = {
            TransformOperation.SELECT: capabilities.can_select,
            TransformOperation.TRANSLATE: capabilities.can_translate,
            TransformOperation.RESIZE_GEOMETRY: capabilities.can_resize,
            TransformOperation.ROTATE: capabilities.can_rotate,
            TransformOperation.SNAPSHOT: capabilities.can_snapshot,
            TransformOperation.SERIALIZE: capabilities.can_serialize,
            TransformOperation.SCALE_APPEARANCE: False,
            TransformOperation.REFLOW_LAYOUT: False,
            TransformOperation.EDIT_POINTS: False,
        }
        if legacy_support[operation]:
            constraints = ("fixed_aspect",) if (
                operation is TransformOperation.RESIZE_GEOMETRY
                and capabilities.fixed_aspect
            ) else ()
            preview_strategy = (
                "native_rotation"
                if operation is TransformOperation.ROTATE
                else "control_points"
            )
            return OperationSupport.allowed(
                operation,
                constraints=constraints,
                preview_strategy=preview_strategy,
            )
        reason = self.unsupported_operation_reasons.get(
            operation,
            f"{type(self.target).__name__} has no lossless {operation.value} adapter",
        )
        return OperationSupport.denied(operation, reason)

    def supports_operation(self, operation: TransformOperation | str) -> bool:
        return self.operation_support(operation).supported

    def get_transform(self) -> Transform:
        getter = getattr(self.target, "get_transform", None)
        return getter() if getter is not None else IdentityTransform()

    def renderer(self):
        return self.figure.canvas.get_renderer()

    def change_tracker(self):
        root_figure = getattr(self.figure, "figure", None)
        if root_figure is not None:
            return root_figure.change_tracker
        return self.figure.change_tracker

    @staticmethod
    def point_array(points: Iterable[Sequence[float]]) -> np.ndarray:
        points = np.ma.asarray(points, dtype=float)
        if np.ma.isMaskedArray(points):
            points = points.filled(np.nan)
        points = np.asarray(points, dtype=float)
        if points.size == 0:
            return np.empty((0, 2), dtype=float)
        if points.ndim != 2 or points.shape[1] < 2:
            return np.empty((0, 2), dtype=float)
        return points[:, :2]

    @classmethod
    def finite_points(cls, points) -> np.ndarray:
        points = cls.point_array(points)
        return points[np.all(np.isfinite(points), axis=1)]

    @classmethod
    def bounds_points(cls, points, padding: float = 0.0) -> np.ndarray:
        points = cls.finite_points(points)
        if len(points) == 0:
            return np.empty((0, 2), dtype=float)
        return np.array(
            [
                [np.min(points[:, 0]) - padding, np.min(points[:, 1]) - padding],
                [np.max(points[:, 0]) + padding, np.max(points[:, 1]) + padding],
            ],
            dtype=float,
        )

    def _preview_points(self, attribute: str):
        preview = getattr(self.target, attribute, None)
        if preview is None:
            return None
        return np.asarray(preview, dtype=float).copy()

    def selection_points(self) -> np.ndarray:
        """Visible display-space bounds used for picking and alignment."""
        preview = self._preview_points("_pylustrator_preview_selection_points")
        if preview is not None:
            return preview
        try:
            bbox = self.target.get_window_extent(self.renderer())
        except (AttributeError, TypeError, ValueError, RuntimeError):
            bbox = None
        if bbox is not None:
            bounds = np.asarray(bbox.extents, dtype=float)
            if bounds.shape == (4,) and np.all(np.isfinite(bounds)):
                return np.array([bounds[:2], bounds[2:]], dtype=float)
        return self.bounds_points(self.control_points())

    def native_control_points(self) -> list[np.ndarray]:
        """Writable points in the artist's own coordinate systems."""
        return []

    def control_points(self) -> list[np.ndarray]:
        """Writable points transformed to display coordinates."""
        preview = self._preview_points("_pylustrator_preview_positions")
        if preview is not None:
            return [np.array(point, dtype=float).copy() for point in preview]
        return [
            np.array(point, dtype=float).copy()
            for point in self.native_to_display(self.native_control_points())
        ]

    def native_to_display(self, points) -> list[np.ndarray]:
        transform = self.get_transform()
        return [np.asarray(transform.transform(point), dtype=float) for point in points]

    def display_to_native(self, points) -> list[np.ndarray]:
        transform = self.get_transform().inverted()
        return [np.asarray(transform.transform(point), dtype=float) for point in points]

    def local_control_points(self) -> list[np.ndarray]:
        """Current controls in native coordinates, including active previews."""
        return [
            np.array(point, dtype=float).copy()
            for point in self.display_to_native(self.control_points())
        ]

    def _apply_native_control_points(self, points) -> None:
        raise UnsupportedArtistError(
            f"{type(self.target).__name__} has no lossless move adapter"
        )

    def serialize_changes(self) -> tuple[ChangeRecord, ...]:
        return ()

    def _record_change_records(self, records) -> None:
        if not _CHANGE_RECORDING_ENABLED.get():
            return
        records = tuple(records)
        if records and not self.capabilities.can_serialize:
            raise UnsupportedArtistError(
                f"{type(self).__name__} emitted changes without serialization capability"
            )
        if not records:
            return
        tracker = self.change_tracker()
        for record in records:
            record.apply(tracker)

    def record_changes(self) -> None:
        if not _CHANGE_RECORDING_ENABLED.get():
            return
        self._record_change_records(self.serialize_changes())

    def invalidate_geometry_cache(self) -> None:
        setattr(self.target, "_pylustrator_cached_get_extend", None)

    def apply_native_control_points(self, points) -> None:
        if not self.capabilities.can_translate:
            raise UnsupportedArtistError(
                f"{type(self.target).__name__} cannot be translated losslessly"
            )
        self._apply_native_control_points(points)
        self.record_changes()
        self.invalidate_geometry_cache()

    def apply_control_points(self, points) -> None:
        self.apply_native_control_points(self.display_to_native(points))

    def translate(self, delta: Sequence[float]) -> None:
        delta = np.asarray(delta, dtype=float)
        if delta.shape != (2,):
            raise ValueError("Display-space translation must contain exactly x and y")
        points = np.asarray(self.control_points(), dtype=float)
        self.apply_control_points(points + delta)

    @staticmethod
    def _transform_points(matrix, points) -> np.ndarray:
        points = np.asarray(points, dtype=float)
        homogeneous = np.concatenate(
            [points, np.ones((len(points), 1), dtype=float)], axis=1
        )
        return np.asarray(homogeneous @ np.asarray(matrix, dtype=float).T)[:, :2]

    def apply_display_transform(self, matrix) -> None:
        self.apply_control_points(self._transform_points(matrix, self.control_points()))

    def resize(self, matrix) -> None:
        if not self.capabilities.can_resize:
            raise UnsupportedArtistError(
                f"{type(self.target).__name__} cannot be resized losslessly"
            )
        self.apply_display_transform(matrix)

    def rotation(self) -> float:
        raise UnsupportedArtistError(
            f"{type(self.target).__name__} has no native rotation property"
        )

    def _apply_rotation(self, value: float) -> None:
        raise UnsupportedArtistError(
            f"{type(self.target).__name__} has no native rotation property"
        )

    def serialize_rotation_changes(self) -> tuple[ChangeRecord, ...]:
        return ()

    def set_rotation(self, value: float) -> None:
        if not self.capabilities.can_rotate:
            raise UnsupportedArtistError(
                f"{type(self.target).__name__} cannot be rotated losslessly"
            )
        self._apply_rotation(float(value))
        self._record_change_records(self.serialize_rotation_changes())
        self.invalidate_geometry_cache()

    def _record_restored_state(self, *, include_rotation: bool = False) -> None:
        records = list(self.serialize_changes())
        if include_rotation:
            for record in self.serialize_rotation_changes():
                if record not in records:
                    records.append(record)
        self._record_change_records(records)

    def snapshot(self):
        if not self.capabilities.can_snapshot:
            raise UnsupportedArtistError(
                f"{type(self.target).__name__} does not support interaction snapshots"
            )
        state = {"type": "positions", "positions": self.local_control_points()}
        if self.capabilities.can_rotate:
            state["rotation"] = self.rotation()
        return state

    def restore(self, state) -> None:
        if state.get("type") != "positions":
            raise ValueError(f"Unsupported snapshot for {type(self).__name__}: {state!r}")
        self._apply_native_control_points(state["positions"])
        include_rotation = "rotation" in state and not np.isclose(
            self.rotation(), float(state["rotation"])
        )
        if include_rotation:
            self._apply_rotation(float(state["rotation"]))
        self._record_restored_state(include_rotation=include_rotation)
        self.invalidate_geometry_cache()

    def get_extent(self) -> list[float]:
        if not getattr(self.target, "_pylustrator_cached_get_extend_added", False):
            setattr(self.target, "_pylustrator_cached_get_extend_added", True)
        cached = getattr(self.target, "_pylustrator_cached_get_extend", None)
        if cached is None:
            cached = self.do_get_extent()
            setattr(self.target, "_pylustrator_cached_get_extend", cached)
        return cached

    def do_get_extent(self) -> list[float]:
        points = np.asarray(self.selection_points(), dtype=float)
        if len(points) == 0:
            raise UnsupportedArtistError(
                f"{type(self.target).__name__} has no finite selection geometry"
            )
        return [
            float(np.min(points[:, 0])),
            float(np.min(points[:, 1])),
            float(np.max(points[:, 0])),
            float(np.max(points[:, 1])),
        ]


@dataclass(frozen=True)
class AdapterRegistration:
    artist_type: type
    adapter_type: type[ArtistAdapter]
    priority: int
    order: int


class ArtistAdapterRegistry:
    """Resolve an artist to the most specific registered adapter class."""

    def __init__(self):
        self._registrations: list[AdapterRegistration] = []
        self._cache: dict[type, type[ArtistAdapter]] = {}
        self._lock = RLock()
        self._next_order = 0

    def register(
        self,
        artist_type: type,
        adapter_type: type[ArtistAdapter],
        *,
        priority: int = 0,
        replace: bool = False,
    ) -> type[ArtistAdapter]:
        if not isinstance(artist_type, type) or not issubclass(artist_type, Artist):
            raise TypeError("artist_type must be an Artist subclass")
        if not isinstance(adapter_type, type) or not issubclass(
            adapter_type, ArtistAdapter
        ):
            raise TypeError("adapter_type must be an ArtistAdapter subclass")
        with self._lock:
            if replace:
                self._registrations = [
                    item
                    for item in self._registrations
                    if item.artist_type is not artist_type
                ]
            self._registrations.append(
                AdapterRegistration(
                    artist_type, adapter_type, int(priority), self._next_order
                )
            )
            self._next_order += 1
            self._cache.clear()
        return adapter_type

    def unregister(
        self,
        artist_type: type,
        adapter_type: Optional[type[ArtistAdapter]] = None,
    ) -> None:
        with self._lock:
            self._registrations = [
                item
                for item in self._registrations
                if not (
                    item.artist_type is artist_type
                    and (adapter_type is None or item.adapter_type is adapter_type)
                )
            ]
            self._cache.clear()

    def registrations(self) -> tuple[AdapterRegistration, ...]:
        with self._lock:
            return tuple(self._registrations)

    @staticmethod
    def _mro_distance(concrete: type, registered: type) -> int:
        try:
            return concrete.mro().index(registered)
        except ValueError:
            # Supports virtual/ABC subclass registrations while keeping true
            # MRO matches more specific.
            return len(concrete.mro()) + 1

    def resolve_type(self, target_or_type) -> type[ArtistAdapter]:
        concrete = target_or_type if isinstance(target_or_type, type) else type(target_or_type)
        with self._lock:
            cached = self._cache.get(concrete)
            if cached is not None:
                return cached
            matches = [
                item
                for item in self._registrations
                if issubclass(concrete, item.artist_type)
            ]
            if not matches:
                raise LookupError(f"No artist adapter registered for {concrete!r}")
            selected = min(
                matches,
                key=lambda item: (
                    self._mro_distance(concrete, item.artist_type),
                    -item.priority,
                    -item.order,
                ),
            )
            self._cache[concrete] = selected.adapter_type
            return selected.adapter_type

    def create(self, target: Artist) -> ArtistAdapter:
        return self.resolve_type(target)(target)

    def capabilities_for(self, target: Artist) -> ArtistCapabilities:
        return self.resolve_type(target).capabilities_for(target)

    def supports(self, target: Artist) -> bool:
        return self.capabilities_for(target).editable


class PatchAdapter(ArtistAdapter):
    default_capabilities = ArtistCapabilities(
        can_select=True,
        can_translate=True,
        can_resize=True,
        can_snapshot=True,
        can_serialize=True,
    )

    @classmethod
    def capabilities_for(cls, target) -> ArtistCapabilities:
        capabilities = cls.default_capabilities
        return ArtistCapabilities(
            can_select=capabilities.can_select,
            can_translate=capabilities.can_translate,
            can_resize=capabilities.can_resize,
            can_snapshot=capabilities.can_snapshot,
            can_serialize=capabilities.can_serialize,
            fixed_aspect=capabilities.fixed_aspect,
            can_rotate=hasattr(target, "get_angle") and hasattr(target, "set_angle"),
        )

    def get_transform(self) -> Transform:
        return self.target.get_data_transform()

    def rotation(self) -> float:
        return float(self.target.get_angle())

    def _apply_rotation(self, value: float) -> None:
        self.target.set_angle(value)

    def serialize_rotation_changes(self):
        return (
            ChangeRecord.command_change(
                self.target, ".set_angle(%f)" % self.target.get_angle()
            ),
        )


class RectangleAdapter(PatchAdapter):
    @classmethod
    def capabilities_for(cls, target: Rectangle) -> ArtistCapabilities:
        can_resize = np.isclose(float(target.get_angle()) % 180.0, 0.0)
        return ArtistCapabilities(
            can_select=True,
            can_translate=True,
            can_resize=bool(can_resize),
            can_snapshot=True,
            can_serialize=True,
            can_rotate=True,
        )

    def native_control_points(self):
        return [
            np.asarray(self.target.get_xy(), dtype=float),
            np.asarray(
                (
                    self.target.get_x() + self.target.get_width(),
                    self.target.get_y() + self.target.get_height(),
                ),
                dtype=float,
            ),
        ]

    def _apply_native_control_points(self, points) -> None:
        self.target.set_xy(points[0])
        self.target.set_width(points[1][0] - points[0][0])
        self.target.set_height(points[1][1] - points[0][1])

    def serialize_changes(self):
        label = self.target.get_label()
        if label is not None and label.startswith("_rect"):
            return ()
        return (
            ChangeRecord.command_change(
                self.target, ".set_xy([%f, %f])" % tuple(self.target.get_xy())
            ),
            ChangeRecord.command_change(
                self.target, ".set_width(%f)" % self.target.get_width()
            ),
            ChangeRecord.command_change(
                self.target, ".set_height(%f)" % self.target.get_height()
            ),
        )


class EllipseAdapter(PatchAdapter):
    @classmethod
    def capabilities_for(cls, target: Ellipse) -> ArtistCapabilities:
        can_resize = np.isclose(float(target.get_angle()) % 180.0, 0.0)
        return ArtistCapabilities(
            can_select=True,
            can_translate=True,
            can_resize=bool(can_resize),
            can_snapshot=True,
            can_serialize=True,
            can_rotate=True,
        )

    def native_control_points(self):
        center = np.asarray(self.target.center, dtype=float)
        size = np.asarray((self.target.width, self.target.height), dtype=float)
        return [center - size / 2, center + size / 2]

    def _apply_native_control_points(self, points) -> None:
        self.target.center = np.mean(points, axis=0)
        self.target.width = points[1][0] - points[0][0]
        self.target.height = points[1][1] - points[0][1]

    def serialize_changes(self):
        return (
            ChangeRecord.command_change(
                self.target, ".center = (%f, %f)" % tuple(self.target.center)
            ),
            ChangeRecord.command_change(
                self.target, ".width = %f" % self.target.width
            ),
            ChangeRecord.command_change(
                self.target, ".height = %f" % self.target.height
            ),
        )


class FancyArrowPatchAdapter(PatchAdapter):
    default_capabilities = ArtistCapabilities(
        can_select=True,
        can_translate=True,
        can_snapshot=True,
        can_serialize=True,
    )

    def native_control_points(self):
        return [
            np.asarray(self.target._posA_posB[0], dtype=float),
            np.asarray(self.target._posA_posB[1], dtype=float),
        ]

    def _apply_native_control_points(self, points) -> None:
        self.target.set_positions(points[0], points[1])

    def serialize_changes(self):
        point_a, point_b = self.target._posA_posB
        return (
            ChangeRecord.command_change(
                self.target,
                ".set_positions(%s, %s)" % (tuple(point_a), tuple(point_b)),
            ),
        )


class ConnectionPatchAdapter(FancyArrowPatchAdapter):
    """ConnectionPatch endpoints may live in two unrelated coordinate systems."""

    default_capabilities = ArtistCapabilities()


class FancyBboxPatchAdapter(PatchAdapter):
    @classmethod
    def capabilities_for(cls, target: FancyBboxPatch) -> ArtistCapabilities:
        # BoxStyle padding and corners do not follow a display delta through a
        # non-affine data transform, so such boxes are blockers, not editables.
        movable = bool(target.get_data_transform().is_affine)
        return ArtistCapabilities(
            can_select=movable,
            can_translate=movable,
            can_snapshot=movable,
            can_serialize=movable,
            can_rotate=hasattr(target, "get_angle") and hasattr(target, "set_angle"),
        )

    def native_control_points(self):
        return [
            np.asarray((self.target.get_x(), self.target.get_y()), dtype=float),
            np.asarray(
                (
                    self.target.get_x() + self.target.get_width(),
                    self.target.get_y() + self.target.get_height(),
                ),
                dtype=float,
            ),
        ]

    def _apply_native_control_points(self, points) -> None:
        self.target.set_bounds(
            float(points[0][0]),
            float(points[0][1]),
            float(points[1][0] - points[0][0]),
            float(points[1][1] - points[0][1]),
        )

    def serialize_changes(self):
        bounds = tuple(float(value) for value in self.target.get_bbox().bounds)
        return (
            ChangeRecord.command_change(self.target, f".set_bounds{bounds!r}"),
        )


class RegularPolygonAdapter(PatchAdapter):
    unsupported_operation_reasons = {
        TransformOperation.RESIZE_GEOMETRY: (
            "RegularPolygon resize must change its semantic radius, not stretch its center point"
        )
    }
    default_capabilities = ArtistCapabilities(
        can_select=True,
        can_translate=True,
        can_snapshot=True,
        can_serialize=True,
    )

    def native_control_points(self):
        return [np.asarray(self.target.xy, dtype=float)]

    def _apply_native_control_points(self, points) -> None:
        self.target.xy = tuple(float(value) for value in points[0])
        self.target.stale = True

    def serialize_changes(self):
        return (
            ChangeRecord.command_change(self.target, f".xy = {self.target.xy!r}"),
        )


class WedgeAdapter(PatchAdapter):
    default_capabilities = ArtistCapabilities(
        can_select=True,
        can_translate=True,
        can_snapshot=True,
        can_serialize=True,
    )

    def native_control_points(self):
        return [np.asarray(self.target.center, dtype=float)]

    def _apply_native_control_points(self, points) -> None:
        self.target.set_center(tuple(float(value) for value in points[0]))

    def serialize_changes(self):
        center = tuple(float(value) for value in self.target.center)
        return (
            ChangeRecord.command_change(self.target, f".set_center({center!r})"),
        )


class PolygonAdapter(PatchAdapter):
    def native_control_points(self):
        return [np.asarray(point, dtype=float) for point in self.target.get_xy()]

    def _apply_native_control_points(self, points) -> None:
        self.target.set_xy([[float(x), float(y)] for x, y in points])

    def serialize_changes(self):
        vertices = [[float(x), float(y)] for x, y in self.target.get_xy()]
        return (
            ChangeRecord.command_change(self.target, f".set_xy({vertices!r})"),
        )


class PathPatchAdapter(PatchAdapter):
    def native_control_points(self):
        return [
            np.asarray(point, dtype=float)
            for point in self.target.get_path().vertices
        ]

    def _apply_native_control_points(self, points) -> None:
        old_path = self.target.get_path()
        codes = None if old_path.codes is None else old_path.codes.copy()
        self.target.set_path(Path(np.asarray(points, dtype=float), codes))

    def serialize_changes(self):
        path = self.target.get_path()
        vertices = [[float(x), float(y)] for x, y in path.vertices]
        codes = None if path.codes is None else [int(code) for code in path.codes]
        return (
            ChangeRecord.command_change(
                self.target,
                f".set_path(mpl.path.Path({vertices!r}, {codes!r}))",
            ),
        )


class TextAdapter(ArtistAdapter):
    unsupported_operation_reasons = {
        TransformOperation.RESIZE_GEOMETRY: (
            "Text bounds come from font metrics; use appearance scaling instead of geometry resize"
        ),
        TransformOperation.SCALE_APPEARANCE: (
            "Text appearance scaling is not implemented yet"
        ),
    }
    default_capabilities = ArtistCapabilities(
        can_select=True,
        can_translate=True,
        can_snapshot=True,
        can_serialize=True,
        can_rotate=True,
    )

    def __init__(self, target: Text):
        super().__init__(target)
        self._label_axis = None
        self.label_factor = None
        self.label_x = None
        self.label_y = None
        self._ensure_label_state()

    def _axis_label_owner(self):
        axes = checkXLabel(self.target)
        if axes is not None:
            return axes, "x"
        axes = checkYLabel(self.target)
        if axes is not None:
            return axes, "y"
        return None

    def _ensure_label_state(self) -> None:
        owner = self._axis_label_owner()
        self._label_axis = owner
        if owner is None:
            return
        axes, axis_name = owner
        self.label_factor = self.figure.dpi / 72.0
        position = self.target.get_position()
        axis = axes.xaxis if axis_name == "x" else axes.yaxis
        coordinate = position[1] if axis_name == "x" else position[0]
        if getattr(self.target, "pad_offset", None) is None:
            self.target.pad_offset = coordinate + axis.labelpad * self.label_factor
        if axis_name == "x":
            self.label_y = position[1]
        else:
            self.label_x = position[0]

    def get_transform(self) -> Transform:
        return self.target.get_transform()

    def selection_points(self) -> np.ndarray:
        preview = self._preview_points("_pylustrator_preview_selection_points")
        if preview is not None:
            return preview
        try:
            bbox = self.target.get_window_extent(self.renderer())
        except (AttributeError, TypeError, ValueError, RuntimeError):
            bbox = None
        bbox_patch = self.target.get_bbox_patch()
        if bbox_patch is not None:
            face_alpha = bbox_patch.get_facecolor()[-1]
            edge_alpha = bbox_patch.get_edgecolor()[-1]
            if face_alpha > 0 or edge_alpha > 0:
                self.target.update_bbox_position_size(self.renderer())
                patch_bbox = bbox_patch.get_window_extent(self.renderer())
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
        return self.bounds_points(self.control_points())

    def native_control_points(self):
        position = list(self.target.get_position())
        if self._label_axis is not None:
            _axes, axis_name = self._label_axis
            if axis_name == "x":
                position[1] = self.label_y
            else:
                position[0] = self.label_x
        return [np.asarray(position, dtype=float)]

    def _apply_native_control_points(self, points) -> None:
        point = np.asarray(points[0], dtype=float)
        if self._label_axis is not None:
            axes, axis_name = self._label_axis
            axis = axes.xaxis if axis_name == "x" else axes.yaxis
            coordinate = point[1] if axis_name == "x" else point[0]
            axis.labelpad = (self.target.pad_offset - coordinate) / self.label_factor
        self.target.set_position(point)
        if self._label_axis is not None:
            _axes, axis_name = self._label_axis
            if axis_name == "x":
                self.label_y = point[1]
            else:
                self.label_x = point[0]

    def serialize_changes(self):
        records = []
        if self._label_axis is not None:
            axes, axis_name = self._label_axis
            axis = axes.xaxis if axis_name == "x" else axes.yaxis
            records.append(
                ChangeRecord.command_change(
                    axes, f".{axis_name}axis.labelpad = {axis.labelpad:f}"
                )
            )
        records.append(ChangeRecord.text_change(self.target))
        return tuple(records)

    def rotation(self) -> float:
        return float(self.target.get_rotation())

    def _apply_rotation(self, value: float) -> None:
        from .change_tracker import add_text_default

        add_text_default(self.target)
        self.target.set_rotation(value)

    def serialize_rotation_changes(self):
        return (ChangeRecord.text_change(self.target),)

    def snapshot(self):
        if self._label_axis is None:
            return super().snapshot()
        axes, axis_name = self._label_axis
        axis = axes.xaxis if axis_name == "x" else axes.yaxis
        return {
            "type": "axis_label",
            "axis": axis_name,
            "position": tuple(float(value) for value in self.target.get_position()),
            "labelpad": float(axis.labelpad),
            "rotation": self.rotation(),
        }

    def restore(self, state) -> None:
        if state.get("type") != "axis_label":
            super().restore(state)
            return
        owner = self._axis_label_owner()
        if owner is None:
            raise ValueError("Axis-label snapshot cannot be restored to ordinary text")
        axes, axis_name = owner
        if state["axis"] != axis_name:
            raise ValueError("Axis-label snapshot belongs to a different axis")
        axis = axes.xaxis if axis_name == "x" else axes.yaxis
        self.target.set_position(state["position"])
        axis.labelpad = state["labelpad"]
        self._ensure_label_state()
        if axis_name == "x":
            self.label_y = self.target.get_position()[1]
            self.target.pad_offset = self.label_y + axis.labelpad * self.label_factor
        else:
            self.label_x = self.target.get_position()[0]
            self.target.pad_offset = self.label_x + axis.labelpad * self.label_factor
        include_rotation = "rotation" in state and not np.isclose(
            self.rotation(), float(state["rotation"])
        )
        if include_rotation:
            self._apply_rotation(float(state["rotation"]))
        self._record_restored_state(include_rotation=include_rotation)
        self.invalidate_geometry_cache()


class AnnotationAdapter(TextAdapter):
    def __init__(self, target: Annotation):
        ArtistAdapter.__init__(self, target)

    def _xy_transform(self):
        return self.target._get_xy_transform(self.renderer(), self.target.xycoords)

    def native_control_points(self):
        return [
            np.asarray(self.target.get_position(), dtype=float),
            np.asarray(self.target.xy, dtype=float),
        ]

    def native_to_display(self, points):
        if len(points) == 0:
            return []
        result = [
            np.asarray(self.target.get_transform().transform(points[0]), dtype=float)
        ]
        if len(points) > 1:
            result.append(np.asarray(self._xy_transform().transform(points[1]), dtype=float))
        return result

    def display_to_native(self, points):
        if len(points) == 0:
            return []
        result = [
            np.asarray(
                self.target.get_transform().inverted().transform(points[0]), dtype=float
            )
        ]
        if len(points) > 1:
            result.append(
                np.asarray(self._xy_transform().inverted().transform(points[1]), dtype=float)
            )
        return result

    def _apply_native_control_points(self, points) -> None:
        self.target.set_position(points[0])
        self.target.xy = tuple(float(value) for value in points[1])

    def serialize_changes(self):
        return (
            ChangeRecord.text_change(self.target),
            ChangeRecord.command_change(self.target, f".xy = {self.target.xy!r}"),
        )

    def snapshot(self):
        # Annotation has two ordinary native control points; axis-label state
        # from TextAdapter must not leak into this more-specific subclass.
        return ArtistAdapter.snapshot(self)

    def restore(self, state) -> None:
        ArtistAdapter.restore(self, state)


class AxesAdapter(ArtistAdapter):
    default_capabilities = ArtistCapabilities(
        can_select=True,
        can_translate=True,
        can_resize=True,
        can_snapshot=True,
        can_serialize=True,
    )

    def __init__(self, target: Axes):
        super().__init__(target)
        cache_property(self.target, "position")

    @classmethod
    def capabilities_for(cls, target: Axes) -> ArtistCapabilities:
        fixed_aspect = (
            target.get_aspect() != "auto" and target.get_adjustable() != "datalim"
        )
        return ArtistCapabilities(
            can_select=True,
            can_translate=True,
            can_resize=True,
            can_snapshot=True,
            can_serialize=True,
            fixed_aspect=fixed_aspect,
        )

    def get_transform(self) -> Transform:
        if version.parse(mpl.__version__) < version.parse("3.4.0"):
            return self.target.figure.transFigure
        return self.target.figure.transSubfigure or self.target.figure.transFigure

    def native_control_points(self):
        p1, p2 = np.asarray(self.target.get_position(), dtype=float)
        return [p1, p2]

    def _apply_native_control_points(self, points) -> None:
        position = np.array([points[0], points[1] - points[0]]).flatten()
        if self.capabilities.fixed_aspect:
            current = self.target.get_position()
            position[3] = position[2] * current.height / current.width
        self.target.set_position(position)

    def serialize_changes(self):
        return (ChangeRecord.axes_change(self.target),)


class LegendAdapter(ArtistAdapter):
    unsupported_operation_reasons = {
        TransformOperation.RESIZE_GEOMETRY: (
            "Legend size is controlled by layout; use legend reflow instead of stretching its bounds"
        ),
        TransformOperation.REFLOW_LAYOUT: (
            "Legend layout reflow is not implemented yet"
        ),
    }
    default_capabilities = ArtistCapabilities(
        can_select=True,
        can_translate=True,
        can_snapshot=True,
        can_serialize=True,
    )

    def __init__(self, target: Legend):
        super().__init__(target)
        if not hasattr(target, "_pylustrator_original_frameon"):
            target._pylustrator_original_frameon = target.get_frame_on()

    def get_transform(self) -> Transform:
        return IdentityTransform()

    def selection_points(self) -> np.ndarray:
        """Measure the visible legend artwork, not only its layout frame.

        Matplotlib permits legend texts and handles to be positioned outside
        the packer's nominal bbox.  Figure-editing selection and alignment must
        follow those visible children.  A visible frame remains part of the
        artwork; an invisible frame contributes no selection padding.
        """

        preview = self._preview_points("_pylustrator_preview_selection_points")
        if preview is not None:
            return preview

        point_sets = []
        if self.target.get_frame_on():
            point_sets.append(super().selection_points())

        children = list(iter_legend_children(self.target))
        title = self.target.get_title()
        for child in children:
            if (
                child is None
                or not child.get_visible()
                or (child is title and not title.get_text())
            ):
                continue
            try:
                points = np.asarray(
                    get_artist_adapter(child).selection_points(), dtype=float
                )
            except (AttributeError, TypeError, ValueError, RuntimeError):
                continue
            if points.ndim == 2 and len(points) and np.all(np.isfinite(points)):
                point_sets.append(points)

        if point_sets:
            return self.bounds_points(np.concatenate(point_sets))
        return super().selection_points()

    def native_control_points(self):
        return [legend_display_loc(self.target)]

    def _apply_native_control_points(self, points) -> None:
        if legend_anchor_is_point(self.target):
            set_legend_point_anchor_display(self.target, points[0])
        else:
            self.target._loc = tuple(
                legend_loc_transform(self.target).transform(points[0])
            )

    def serialize_changes(self):
        return (ChangeRecord.legend_change(self.target),)

    def set_frame_on(self, visible: bool) -> bool:
        """Toggle the legend frame without replacing the Legend object."""

        visible = bool(visible)
        if self.target.get_frame_on() == visible:
            return False
        self.target.set_frame_on(visible)
        self.record_changes()
        self.invalidate_geometry_cache()
        return True

    def snapshot(self):
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

    def restore(self, state) -> None:
        if state.get("type") != "legend":
            raise ValueError(f"Unsupported legend snapshot: {state!r}")
        anchor = state["anchor"]
        if state["is_point"]:
            self.target.set_bbox_to_anchor(anchor[:2], transform=state["transform"])
        else:
            self.target.set_bbox_to_anchor(anchor, transform=state["transform"])
        self.target._loc = state["loc"]
        self.record_changes()
        self.invalidate_geometry_cache()


class EditorGroupAdapter(ArtistAdapter):
    """Apply one display-space operation atomically to logical group members."""

    @classmethod
    def capabilities_for(cls, target: EditorGroup) -> ArtistCapabilities:
        if not target.members:
            return ArtistCapabilities()
        capabilities = [get_artist_adapter(member).capabilities for member in target.members]
        return ArtistCapabilities(
            can_select=all(value.can_select for value in capabilities),
            can_translate=all(value.can_translate for value in capabilities),
            can_resize=all(value.can_resize for value in capabilities),
            can_snapshot=all(value.can_snapshot for value in capabilities),
            can_serialize=all(value.can_serialize for value in capabilities),
            fixed_aspect=any(value.fixed_aspect for value in capabilities),
            # Rotation of a group also changes member positions around one pivot;
            # native per-member rotation alone is not an equivalent operation.
            can_rotate=False,
        )

    def get_transform(self) -> Transform:
        return IdentityTransform()

    def _member_adapters(self) -> list[ArtistAdapter]:
        return [get_artist_adapter(member) for member in self.target.members]

    def native_control_points(self):
        points = []
        for adapter in self._member_adapters():
            points.extend(adapter.control_points())
        return points

    def selection_points(self) -> np.ndarray:
        preview = self._preview_points("_pylustrator_preview_selection_points")
        if preview is not None:
            return preview
        points = [
            adapter.selection_points()
            for adapter in self._member_adapters()
            if adapter.capabilities.can_select
        ]
        points = [value for value in points if len(value)]
        return np.concatenate(points) if points else np.empty((0, 2), dtype=float)

    def _apply_native_control_points(self, points) -> None:
        start = 0
        for adapter in self._member_adapters():
            length = len(adapter.control_points())
            adapter.apply_control_points(points[start : start + length])
            start += length
        if start != len(points):
            raise ValueError("Editor-group control-point count changed during transform")

    def serialize_changes(self):
        records = []
        for adapter in self._member_adapters():
            records.extend(adapter.serialize_changes())
        return tuple(records)

    def snapshot(self):
        return {
            "type": "editor_group",
            "id": self.target.group_id,
            "members": [
                (member, get_artist_adapter(member).snapshot())
                for member in self.target.members
            ],
        }

    def restore(self, state) -> None:
        if state.get("type") != "editor_group" or state.get("id") != self.target.group_id:
            raise ValueError(f"Snapshot does not belong to group {self.target.group_id!r}")
        for member, member_state in state["members"]:
            get_artist_adapter(member).restore(member_state)
        self.invalidate_geometry_cache()

    def invalidate_geometry_cache(self) -> None:
        super().invalidate_geometry_cache()
        for adapter in self._member_adapters():
            adapter.invalidate_geometry_cache()


class Line2DAdapter(ArtistAdapter):
    unsupported_operation_reasons = {
        TransformOperation.RESIZE_GEOMETRY: (
            "Line geometry resize requires an affine coordinate preflight"
        )
    }
    default_capabilities = ArtistCapabilities(
        can_select=True,
        can_translate=True,
        can_snapshot=True,
        can_serialize=True,
    )

    def get_transform(self) -> Transform:
        return IdentityTransform()

    def native_control_points(self):
        return [np.asarray(point, dtype=float) for point in self.target.get_xydata()]

    def native_to_display(self, points):
        transform = self.target.get_transform()
        return [np.asarray(transform.transform(point), dtype=float) for point in points]

    def display_to_native(self, points):
        transform = self.target.get_transform().inverted()
        return [np.asarray(transform.transform(point), dtype=float) for point in points]

    def _apply_native_control_points(self, points) -> None:
        xy = np.asarray(points, dtype=float)
        self.target.set_data(xy[:, 0], xy[:, 1])

    def serialize_changes(self):
        xy = np.asarray(self.target.get_xydata(), dtype=float)
        return (
            ChangeRecord.command_change(
                self.target,
                ".set_data(%s, %s)"
                % (
                    [float(value) for value in xy[:, 0]],
                    [float(value) for value in xy[:, 1]],
                ),
            ),
        )


class AxesImageAdapter(ArtistAdapter):
    default_capabilities = ArtistCapabilities(
        can_select=True,
        can_translate=True,
        can_resize=True,
        can_snapshot=True,
        can_serialize=True,
    )

    def native_control_points(self):
        left, right, bottom, top = self.target.get_extent()
        return [
            np.asarray((left, bottom), dtype=float),
            np.asarray((right, top), dtype=float),
        ]

    def _apply_native_control_points(self, points) -> None:
        extent = (
            float(points[0][0]),
            float(points[1][0]),
            float(points[0][1]),
            float(points[1][1]),
        )
        axes = self.target.axes
        xlim = tuple(float(value) for value in axes.get_xlim())
        ylim = tuple(float(value) for value in axes.get_ylim())
        self.target.set_extent(extent)
        # set_extent may autoscale the camera, making a translated image appear
        # stationary.  Moving the image must leave its parent viewport intact.
        axes.set_xlim(xlim)
        axes.set_ylim(ylim)

    def serialize_changes(self):
        from .change_tracker import getReference

        left, right, bottom, top = self.target.get_extent()
        extent = tuple(float(v) for v in (left, right, bottom, top))
        axes = self.target.axes
        xlim = tuple(float(value) for value in axes.get_xlim())
        ylim = tuple(float(value) for value in axes.get_ylim())
        axes_reference = getReference(axes)
        return (
            ChangeRecord.command_change(
                self.target,
                f".set_extent({extent!r}), "
                f"{axes_reference}.set_xlim({xlim!r}), "
                f"{axes_reference}.set_ylim({ylim!r})",
            ),
        )


class CollectionAdapter(ArtistAdapter):
    unsupported_operation_reasons = {
        TransformOperation.RESIZE_GEOMETRY: (
            "Collection resize must distinguish item positions from marker appearance"
        ),
        TransformOperation.SCALE_APPEARANCE: (
            "Collection appearance scaling is not implemented yet"
        ),
    }
    default_capabilities = ArtistCapabilities(
        can_select=True,
        can_translate=True,
        can_snapshot=True,
        can_serialize=True,
    )

    def local_groups(self) -> list[np.ndarray]:
        return []

    def group_transform(self) -> Transform:
        return self.target.get_transform()

    def native_control_points(self):
        groups = self.local_groups()
        if not groups:
            return []
        return [np.asarray(point, dtype=float) for point in np.concatenate(groups)]

    def native_to_display(self, points):
        transform = self.group_transform()
        return [np.asarray(transform.transform(point), dtype=float) for point in points]

    def display_to_native(self, points):
        transform = self.group_transform().inverted()
        return [np.asarray(transform.transform(point), dtype=float) for point in points]

    def display_groups(self) -> list[np.ndarray]:
        transform = self.group_transform()
        return [
            self.finite_points(transform.transform(group))
            for group in self.local_groups()
            if len(group)
        ]

    def selection_padding(self) -> float:
        dpi_per_point = float(self.figure.dpi) / 72.0
        linewidths = np.asarray(self.target.get_linewidths(), dtype=float)
        return (
            float(np.max(linewidths)) * dpi_per_point / 2
            if linewidths.size
            else 0.0
        )

    def selection_points(self) -> np.ndarray:
        preview = self._preview_points("_pylustrator_preview_selection_points")
        if preview is not None:
            return preview
        groups = self.display_groups()
        if groups:
            return self.bounds_points(
                np.concatenate(groups), padding=self.selection_padding()
            )
        return super().selection_points()

    def split_points(self, points) -> list[np.ndarray]:
        lengths = [len(group) for group in self.local_groups()]
        groups = []
        start = 0
        for length in lengths:
            groups.append(np.asarray(points[start : start + length], dtype=float))
            start += length
        return groups


class PathCollectionAdapter(CollectionAdapter):
    def local_groups(self):
        return [self.point_array(self.target.get_offsets())]

    def group_transform(self) -> Transform:
        return self.target.get_offset_transform()

    def selection_padding(self) -> float:
        stroke = super().selection_padding()
        sizes = np.asarray(self.target.get_sizes(), dtype=float)
        marker = (
            float(np.sqrt(np.max(sizes))) * float(self.figure.dpi) / 72.0 / 2
            if sizes.size
            else 0.0
        )
        return marker + stroke

    def _apply_native_control_points(self, points) -> None:
        groups = self.split_points(points)
        offsets = groups[0] if groups else np.empty((0, 2))
        self.target.set_offsets(offsets)

    def serialize_changes(self):
        offsets = [
            [float(x), float(y)] for x, y in self.point_array(self.target.get_offsets())
        ]
        return (
            ChangeRecord.command_change(self.target, f".set_offsets({offsets!r})"),
        )


class LineCollectionAdapter(CollectionAdapter):
    def local_groups(self):
        return [self.point_array(segment) for segment in self.target.get_segments()]

    def _apply_native_control_points(self, points) -> None:
        self.target.set_segments(self.split_points(points))

    def serialize_changes(self):
        groups = [
            [[float(x), float(y)] for x, y in self.point_array(segment)]
            for segment in self.target.get_segments()
        ]
        return (
            ChangeRecord.command_change(self.target, f".set_segments({groups!r})"),
        )


class PolyCollectionAdapter(CollectionAdapter):
    def local_groups(self):
        return [self.point_array(path.vertices) for path in self.target.get_paths()]

    def _apply_native_control_points(self, points) -> None:
        codes = [path.codes for path in self.target.get_paths()]
        self.target.set_verts_and_codes(self.split_points(points), codes)

    def serialize_changes(self):
        paths = self.target.get_paths()
        groups = [
            [[float(x), float(y)] for x, y in self.point_array(path.vertices)]
            for path in paths
        ]
        codes = [
            None if path.codes is None else [int(code) for code in path.codes]
            for path in paths
        ]
        return (
            ChangeRecord.command_change(
                self.target, f".set_verts_and_codes({groups!r}, {codes!r})"
            ),
        )


artist_adapter_registry = ArtistAdapterRegistry()


def register_artist_adapter(
    artist_type: type,
    *,
    priority: int = 0,
    replace: bool = False,
    registry: ArtistAdapterRegistry = artist_adapter_registry,
):
    """Decorator for built-in or third-party adapter registration."""

    def decorator(adapter_type: type[ArtistAdapter]):
        registry.register(
            artist_type, adapter_type, priority=priority, replace=replace
        )
        return adapter_type

    return decorator


def get_artist_adapter(target: Artist) -> ArtistAdapter:
    return artist_adapter_registry.create(target)


# Registration order is intentionally not semantic.  Resolution uses MRO
# distance, then priority and only then registration order for true ties.
for _artist_type, _adapter_type in (
    (Artist, ArtistAdapter),
    (EditorGroup, EditorGroupAdapter),
    (Axes, AxesAdapter),
    (Text, TextAdapter),
    (Annotation, AnnotationAdapter),
    (Legend, LegendAdapter),
    (Line2D, Line2DAdapter),
    (AxesImage, AxesImageAdapter),
    (Rectangle, RectangleAdapter),
    (Ellipse, EllipseAdapter),
    (FancyArrowPatch, FancyArrowPatchAdapter),
    (ConnectionPatch, ConnectionPatchAdapter),
    (FancyBboxPatch, FancyBboxPatchAdapter),
    (RegularPolygon, RegularPolygonAdapter),
    (Wedge, WedgeAdapter),
    (Polygon, PolygonAdapter),
    (PathPatch, PathPatchAdapter),
    (PathCollection, PathCollectionAdapter),
    (LineCollection, LineCollectionAdapter),
    (PolyCollection, PolyCollectionAdapter),
):
    artist_adapter_registry.register(_artist_type, _adapter_type)


__all__ = [
    "AdapterRegistration",
    "AnnotationAdapter",
    "ArtistAdapter",
    "ArtistAdapterRegistry",
    "ArtistCapabilities",
    "AxesAdapter",
    "AxesImageAdapter",
    "ChangeRecord",
    "CollectionAdapter",
    "ConnectionPatchAdapter",
    "EllipseAdapter",
    "EditorGroupAdapter",
    "FancyArrowPatchAdapter",
    "FancyBboxPatchAdapter",
    "LegendAdapter",
    "Line2DAdapter",
    "LineCollectionAdapter",
    "PathCollectionAdapter",
    "PathPatchAdapter",
    "PolyCollectionAdapter",
    "PolygonAdapter",
    "RectangleAdapter",
    "RegularPolygonAdapter",
    "TextAdapter",
    "UnsupportedArtistError",
    "WedgeAdapter",
    "artist_adapter_registry",
    "checkXLabel",
    "checkYLabel",
    "get_artist_adapter",
    "iter_figure_legends",
    "iter_legend_children",
    "legend_anchor_is_point",
    "legend_anchor_transform",
    "legend_display_loc",
    "legend_loc_transform",
    "register_artist_adapter",
    "set_legend_point_anchor_display",
    "suspend_change_recording",
]
