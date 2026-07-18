"""Artist-specific interaction contracts.

Pylustrator edits objects in display coordinates, while Matplotlib stores each
artist in its own native coordinate system.  This module is the single boundary
between those two worlds.  Every editable artist is resolved to one adapter
which owns its geometry, capabilities, mutations, undo snapshots, and change
records.

The registry deliberately resolves by MRO specificity while making inheritance
an explicit contract.  Registrations match their exact Artist type by default;
an adapter must opt in before subclasses may inherit its mutation semantics.
That keeps semantic subclasses such as Matplotlib's 3D artists from silently
using incompatible 2D writers while still allowing explicitly validated
extension hierarchies.
"""

from __future__ import annotations

import hashlib
from copy import copy, deepcopy
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field, replace
from enum import Enum
from numbers import Integral, Real
from threading import RLock
from typing import Iterable, Optional, Sequence

import matplotlib as mpl
import matplotlib.collections as mpl_collections
import numpy as np
from matplotlib.artist import Artist

try:  # starting from mpl version 3.6.0
    from matplotlib.axes import Axes
except ImportError:
    from matplotlib.axes._subplots import Axes  # ty: ignore[unresolved-import]
from matplotlib.collections import LineCollection, PathCollection, PolyCollection
from matplotlib.image import AxesImage
from matplotlib.legend import Legend
from matplotlib.lines import Line2D, _mark_every_path  # ty: ignore[unresolved-import]
from matplotlib.path import Path, get_path_collection_extents
from matplotlib.patches import (
    Arc,
    Circle,
    CirclePolygon,
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
    Affine2D,
    Bbox,
    BboxTransformFrom,
    BboxTransformTo,
    IdentityTransform,
    Transform,
)
from packaging import version

from .editor_model import EditorGroup
from .legend_layout import (
    LegendLayoutError,
    LegendLayoutPlan,
    LegendLayoutSpec,
    LegendLayoutState,
    capture_legend_layout_state,
    ensure_legend_layout_baseline,
    plan_legend_layout,
    restore_legend_layout_state,
)
from .operations import OperationSupport, TransformOperation
from .property_adapters import axis_tick_label_reference
from .replay import replay_literal


_CHANGE_RECORDING_ENABLED = ContextVar(
    "pylustrator_change_recording_enabled", default=True
)
_SELECTION_GEOMETRY_CACHE = ContextVar(
    "pylustrator_selection_geometry_cache", default=None
)
_LEGEND_OWNER_CACHE = ContextVar("pylustrator_legend_owner_cache", default=None)
_LEGEND_OWNER_INVENTORY_ATTR = "_pylustrator_legend_owner_inventory"
_ACTIVE_LAYOUT_OWNER_SNAPSHOT_KEY = object()
_ACTIVE_LAYOUT_EXTRAS_SNAPSHOT_KEY = object()
_CONTAINER_OWNER_SNAPSHOT_KEY = object()


@contextmanager
def suspend_change_recording():
    """Restore interaction state without emitting a second set of changes."""

    token = _CHANGE_RECORDING_ENABLED.set(False)
    try:
        yield
    finally:
        _CHANGE_RECORDING_ENABLED.reset(token)


@contextmanager
def legend_owner_snapshot():
    """Resolve Legend ownership once during one non-structural action phase.

    Ownership is structural state, not geometry.  Keeping it in a dedicated
    snapshot lets a move reuse the inventory while artists change position,
    without allowing selection bounds measured before the mutation to leak
    into the committed state.
    """

    existing = _LEGEND_OWNER_CACHE.get()
    if existing is not None:
        yield existing
        return
    cache = {}
    token = _LEGEND_OWNER_CACHE.set(cache)
    try:
        yield cache
    finally:
        _LEGEND_OWNER_CACHE.reset(token)


@contextmanager
def selection_geometry_snapshot(cache: dict | None = None):
    """Reuse one immutable geometry measurement during a selection action.

    A caller-owned cache may span several read-only interaction phases that
    belong to the same renderer revision.  Nested callers always reuse the
    active cache, so mutation/preview code cannot accidentally switch snapshots
    midway through one operation.
    """

    with legend_owner_snapshot():
        existing = _SELECTION_GEOMETRY_CACHE.get()
        if existing is not None:
            yield existing
            return
        cache = {} if cache is None else cache
        token = _SELECTION_GEOMETRY_CACHE.set(cache)
        try:
            yield cache
        finally:
            _SELECTION_GEOMETRY_CACHE.reset(token)


def _display_clip_components(
    target: Artist,
) -> tuple[np.ndarray | None, Path | None]:
    """Resolve the display-space clips that actually constrain leaf paint."""

    # Legend is a layout container whose own Artist clip metadata is not used
    # when its frame, handles, and text children draw. Treating that metadata as
    # paint clipping makes visible outside-Axes legends impossible to move.
    if isinstance(target, (Legend, EditorGroup)) or not bool(
        getattr(target, "get_clip_on", lambda: False)()
    ):
        return None, None

    clip_box_bounds = None
    clip_box = getattr(target, "get_clip_box", lambda: None)()
    if clip_box is not None:
        bounds = np.asarray(clip_box.extents, dtype=float)
        if bounds.shape == (4,) and np.all(np.isfinite(bounds)):
            clip_box_bounds = bounds

    clip_path = None
    try:
        path, affine = target.get_transformed_clip_path_and_affine()
        if path is not None:
            transformed = affine.transform_path(path)
            bounds = np.asarray(transformed.get_extents().extents, dtype=float)
            if bounds.shape == (4,) and np.all(np.isfinite(bounds)):
                clip_path = transformed
    except (AttributeError, TypeError, ValueError, RuntimeError):
        pass
    return clip_box_bounds, clip_path


def _clip_polygon_to_bbox(vertices, bounds: np.ndarray) -> np.ndarray:
    """Clip one display-space polygon to an axis-aligned bbox."""

    polygon = np.asarray(vertices, dtype=float)
    polygon = polygon[np.all(np.isfinite(polygon[:, :2]), axis=1), :2]
    if len(polygon) < 3:
        return np.empty((0, 2), dtype=float)

    def clip_edge(points, *, axis: int, limit: float, keep_greater: bool):
        if not len(points):
            return points
        result = []

        def inside(point) -> bool:
            return bool(point[axis] >= limit) if keep_greater else bool(
                point[axis] <= limit
            )

        previous = points[-1]
        previous_inside = inside(previous)
        for current in points:
            current_inside = inside(current)
            if current_inside != previous_inside:
                denominator = current[axis] - previous[axis]
                if not np.isclose(denominator, 0.0):
                    fraction = (limit - previous[axis]) / denominator
                    result.append(previous + fraction * (current - previous))
            if current_inside:
                result.append(current)
            previous = current
            previous_inside = current_inside
        return np.asarray(result, dtype=float)

    for axis, limit, keep_greater in (
        (0, bounds[0], True),
        (0, bounds[2], False),
        (1, bounds[1], True),
        (1, bounds[3], False),
    ):
        polygon = clip_edge(
            polygon, axis=axis, limit=float(limit), keep_greater=keep_greater
        )
        if not len(polygon):
            break
    return polygon


def _clip_path_intersection_bounds(
    clip_path: Path, low: np.ndarray, high: np.ndarray
) -> np.ndarray | None:
    """Approximate the visible bbox of a rectangular envelope under a Path clip."""

    bounds = np.array([*low, *high], dtype=float)
    points = []
    try:
        polygons = clip_path.to_polygons(closed_only=True)
    except (AttributeError, TypeError, ValueError, RuntimeError):
        polygons = ()
    for polygon in polygons:
        clipped = _clip_polygon_to_bbox(polygon, bounds)
        if len(clipped):
            points.append(clipped)
    if not points:
        return None
    points = np.concatenate(points)
    return np.array(
        [
            np.min(points[:, 0]),
            np.min(points[:, 1]),
            np.max(points[:, 0]),
            np.max(points[:, 1]),
        ],
        dtype=float,
    )


def _clip_selection_points(target: Artist, points) -> np.ndarray:
    """Clip a selection envelope to the Artist's active paint region."""

    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] < 2 or not len(points):
        return points
    clip_box, clip_path = _display_clip_components(target)
    if clip_box is None and clip_path is None:
        return points
    finite = points[np.all(np.isfinite(points[:, :2]), axis=1), :2]
    if not len(finite):
        return np.empty((0, 2), dtype=float)

    low = np.min(finite, axis=0)
    high = np.max(finite, axis=0)
    if clip_box is not None:
        low = np.maximum(low, clip_box[:2])
        high = np.minimum(high, clip_box[2:])
    if np.any(high < low):
        return np.empty((0, 2), dtype=float)

    if clip_path is not None:
        candidate = Bbox.from_extents(*low, *high)
        try:
            if not clip_path.intersects_bbox(candidate, filled=True):
                return np.empty((0, 2), dtype=float)
        except (TypeError, ValueError, RuntimeError):
            return np.empty((0, 2), dtype=float)
        intersection = _clip_path_intersection_bounds(clip_path, low, high)
        if intersection is None:
            return np.empty((0, 2), dtype=float)
        low = intersection[:2]
        high = intersection[2:]
    return np.asarray((low, high), dtype=float)


def cached_selection_points(target: Artist, compute) -> np.ndarray:
    """Measure once inside :func:`selection_geometry_snapshot`."""

    cache = _SELECTION_GEOMETRY_CACHE.get()
    if cache is None:
        return _clip_selection_points(target, compute())
    key = id(target)
    if key not in cache:
        cache[key] = _clip_selection_points(target, compute()).copy()
    return cache[key].copy()


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


def iter_legend_managed_artists(legend: Legend) -> tuple[Artist, ...]:
    """Return every Artist whose geometry is owned by a Legend packer.

    ``legend_handles`` only exposes the top-level proxy for each entry.  Error
    bars, stem plots, tuple handlers, and similar composite entries contain
    additional Line2D/Collection/Patch leaves below private OffsetBox nodes.
    Those leaves are reachable by Direct Selection, but cannot be transformed
    or serialized independently of the Legend that lays them out.
    """

    managed = []
    seen = {id(legend)}
    stack = [*iter_legend_children(legend), *legend.get_children()]
    while stack:
        child = stack.pop()
        if not isinstance(child, Artist) or id(child) in seen:
            continue
        seen.add(id(child))
        managed.append(child)
        try:
            stack.extend(child.get_children())
        except (AttributeError, TypeError, RuntimeError):
            continue
    return tuple(managed)


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


def _legend_inventory_signature(legends: Sequence[Legend]) -> tuple:
    """Cheaply detect public Legend replacement and packer reconstruction."""

    return tuple(
        (
            id(legend),
            id(getattr(legend, "_legend_box", None)),
            id(legend.get_frame()),
            tuple(id(handle) for handle in getattr(legend, "legend_handles", ())),
            tuple(id(text) for text in legend.get_texts()),
            id(legend.get_title()),
        )
        for legend in legends
    )


def _legend_owner_inventory(figure) -> dict[int, tuple[Artist, Legend]]:
    """Map every managed descendant to its live Legend once per structure."""

    snapshot = _LEGEND_OWNER_CACHE.get()
    snapshot_key = id(figure)
    if snapshot is not None:
        snapshot_entry = snapshot.get(snapshot_key)
        if snapshot_entry is not None and snapshot_entry[0] is figure:
            return snapshot_entry[1]

    legends = iter_figure_legends(figure)
    signature = _legend_inventory_signature(legends)
    cached = getattr(figure, _LEGEND_OWNER_INVENTORY_ATTR, None)
    if cached is not None and cached[0] == signature:
        inventory = cached[1]
        if snapshot is not None:
            snapshot[snapshot_key] = (figure, inventory)
        return inventory

    inventory = {}
    for legend in legends:
        for child in iter_legend_managed_artists(legend):
            inventory[id(child)] = (child, legend)
    setattr(figure, _LEGEND_OWNER_INVENTORY_ATTR, (signature, inventory))
    if snapshot is not None:
        snapshot[snapshot_key] = (figure, inventory)
    return inventory


def invalidate_legend_owner_inventory(figure) -> None:
    """Drop ownership state after a Legend or its packer is reconstructed."""

    if figure is None:
        return
    try:
        delattr(figure, _LEGEND_OWNER_INVENTORY_ATTR)
    except AttributeError:
        pass
    snapshot = _LEGEND_OWNER_CACHE.get()
    if snapshot is not None:
        snapshot.pop(id(figure), None)


def legend_owner_for_artist(target: Artist) -> Legend | None:
    """Return whether an Artist is managed by a live Legend packer."""

    figure = getattr(target, "figure", None)
    if figure is not None:
        entry = _legend_owner_inventory(figure).get(id(target))
        if entry is not None and entry[0] is target:
            target._pylustrator_legend_owner = entry[1]
            return entry[1]
    target._pylustrator_legend_owner = None
    return None


def legend_owner_for_text(target: Text) -> Legend | None:
    """Backward-compatible Text-specific legend ownership query."""

    return legend_owner_for_artist(target)


def layout_owner_for_text(target: Text) -> Artist | None:
    """Return a Matplotlib layout owner that may rewrite Text position."""

    figure = getattr(target, "figure", None)
    for axes in getattr(figure, "axes", []):
        if any(
            target is title
            for title in (axes.title, axes._left_title, axes._right_title)
        ):
            return axes
        axis_map = getattr(axes, "_axis_map", None)
        axes_axes = (
            tuple(dict.fromkeys(axis_map.values()))
            if isinstance(axis_map, dict)
            else tuple(
                dict.fromkeys(
                    axis
                    for name in ("xaxis", "yaxis", "zaxis")
                    if (axis := getattr(axes, name, None)) is not None
                )
            )
        )
        for axis in axes_axes:
            if target is getattr(axis, "label", None) or target is getattr(
                axis, "offsetText", None
            ):
                return axis
            ticks = (
                *getattr(axis, "majorTicks", ()),
                *getattr(axis, "minorTicks", ()),
            )
            for tick in ticks:
                if target is tick.label1 or target is tick.label2:
                    return axis

    def visit(owner):
        for name in ("_suptitle", "_supxlabel", "_supylabel"):
            if target is getattr(owner, name, None):
                return owner
        for subfigure in getattr(owner, "subfigs", []):
            found = visit(subfigure)
            if found is not None:
                return found
        return None

    return visit(figure) if figure is not None else None


def active_layout_owner_for_artist(target: Artist) -> Artist | None:
    """Return an Axes whose active layout can feed back through *target*."""

    figure = getattr(target, "figure", None)
    if figure is None:
        return None
    get_layout_engine = getattr(figure, "get_layout_engine", None)
    if callable(get_layout_engine):
        layout_active = get_layout_engine() is not None
    else:
        layout_active = bool(
            getattr(figure, "get_constrained_layout", lambda: False)()
            or getattr(figure, "get_tight_layout", lambda: False)()
        )
    if not layout_active:
        return None

    if isinstance(target, Text):
        owner = layout_owner_for_text(target)
        if target is getattr(owner, "label", None):
            # Axis.draw always owns one label coordinate, while an active
            # Figure layout can also move the Axes itself. ``in_layout=False``
            # on the Text does not disable either channel; the labelpad-aware
            # adapter is exact only while the Figure layout is inactive.
            return getattr(owner, "axes", None) or owner

    if not target.get_visible() or not getattr(
        target, "get_in_layout", lambda: True
    )():
        return None

    # Constrained layout rewrites the automatic coordinate of Figure and
    # SubFigure super labels after a drag preview. Matplotlib marks exactly
    # those auto-positioned labels with ``_autopos``. A manually positioned
    # super label, ordinary ``figure.text``, or an object explicitly removed
    # from layout remains independently movable.
    if (
        isinstance(target, Text)
        and getattr(target, "axes", None) is None
        and bool(getattr(target, "_autopos", False))
    ):
        owner = layout_owner_for_text(target)
        if owner is not None:
            return owner

    axes = getattr(target, "axes", None)
    if axes is None:
        return None

    snapshot = _SELECTION_GEOMETRY_CACHE.get()
    if snapshot is None:
        # Transform capability preflight already has an ownership snapshot.
        # Reuse it so a mixed selection asks Matplotlib for each Axes' bbox
        # extras once rather than once per selected Artist.
        snapshot = _LEGEND_OWNER_CACHE.get()
    snapshot_key = (_ACTIVE_LAYOUT_OWNER_SNAPSHOT_KEY, id(target))
    if snapshot is not None:
        entry = snapshot.get(snapshot_key)
        if entry is not None and entry[0] is target:
            return entry[1]

    extras = None
    extras_inventory = None
    extras_key = (_ACTIVE_LAYOUT_EXTRAS_SNAPSHOT_KEY, id(axes))
    if snapshot is not None:
        entry = snapshot.get(extras_key)
        if entry is not None and entry[0] is axes:
            extras = entry[1]
            extras_inventory = entry[2]
    if extras is None:
        try:
            extras = tuple(axes.get_default_bbox_extra_artists())
        except (AttributeError, TypeError, ValueError, RuntimeError):
            extras = None
        extras_inventory = (
            None
            if extras is None
            else {id(artist): artist for artist in extras}
        )
        if snapshot is not None:
            snapshot[extras_key] = (axes, extras, extras_inventory)

    if extras is None:
        owner = axes
    else:
        owner = None
        if extras_inventory is not None:
            candidate = extras_inventory.get(id(target))
            is_extra = candidate is target
        else:
            is_extra = any(target is artist for artist in extras)
        if is_extra:
            try:
                bbox = target.get_tightbbox(figure.canvas.get_renderer())
                bounds = np.asarray(bbox.extents, dtype=float)
            except (AttributeError, TypeError, ValueError, RuntimeError):
                owner = axes
            else:
                if (
                    bounds.shape == (4,)
                    and np.all(np.isfinite(bounds))
                    and (bbox.width != 0 or bbox.height != 0)
                ):
                    owner = axes
    if snapshot is not None:
        snapshot[snapshot_key] = (target, owner)
    return owner


def container_owner_for_artist(target: Artist) -> Artist | None:
    """Return a Figure/SubFigure/Axes whose background patch is *target*."""

    figure = getattr(target, "figure", None)
    if figure is None:
        return None

    get_root = getattr(target, "get_figure", None)
    if callable(get_root):
        try:
            root = get_root(root=True)
        except (AttributeError, TypeError, ValueError):
            root = figure
    else:
        root = figure

    snapshot = _LEGEND_OWNER_CACHE.get()
    snapshot_key = (_CONTAINER_OWNER_SNAPSHOT_KEY, id(root))
    if snapshot is not None:
        entry = snapshot.get(snapshot_key)
        if entry is not None and entry[0] is root:
            owned = entry[1].get(id(target))
            return owned[1] if owned is not None and owned[0] is target else None

        inventory = {}

        def collect(owner) -> None:
            patch = getattr(owner, "patch", None)
            if patch is not None:
                inventory[id(patch)] = (patch, owner)
            for axes in getattr(owner, "axes", []):
                patch = getattr(axes, "patch", None)
                if patch is not None:
                    inventory[id(patch)] = (patch, axes)
            for subfigure in getattr(owner, "subfigs", []):
                collect(subfigure)

        collect(root)
        snapshot[snapshot_key] = (root, inventory)
        owned = inventory.get(id(target))
        return owned[1] if owned is not None and owned[0] is target else None

    def visit(owner):
        if target is getattr(owner, "patch", None):
            return owner
        for axes in getattr(owner, "axes", []):
            if target is getattr(axes, "patch", None):
                return axes
        for subfigure in getattr(owner, "subfigs", []):
            found = visit(subfigure)
            if found is not None:
                return found
        return None

    return visit(figure)


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
    can_rigid_rotate: bool = False

    @property
    def editable(self) -> bool:
        """Backward-compatible shorthand for directly movable selections.

        Use ``can_select`` for selection admission and the individual
        operation fields for actions; a selectable Artist need not move.
        """

        return self.can_select and self.can_translate


@dataclass(frozen=True)
class RigidRotationPlan:
    """Absolute destination for one display-space rigid rotation."""

    target: Artist
    angle_degrees: float
    pivot: tuple[float, float]
    control_points: np.ndarray = field(repr=False, compare=False)
    native_control_points: np.ndarray = field(repr=False, compare=False)
    selection_points: np.ndarray = field(repr=False, compare=False)
    rotation_value: Optional[float] = None
    member_plans: tuple["RigidRotationPlan", ...] = ()
    source_fingerprint: object | None = field(
        default=None, repr=False, compare=False
    )

    @staticmethod
    def _immutable_points(points) -> np.ndarray:
        values = np.asarray(points, dtype=float)
        if values.size == 0:
            values = np.empty((0, 2), dtype=float)
        if values.ndim != 2 or values.shape[1] != 2:
            raise ValueError("Rigid-rotation plan points must have shape (N, 2)")
        values = np.ascontiguousarray(values, dtype=float)
        return np.frombuffer(values.tobytes(), dtype=float).reshape(values.shape)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "control_points", self._immutable_points(self.control_points)
        )
        object.__setattr__(
            self,
            "native_control_points",
            self._immutable_points(self.native_control_points),
        )
        object.__setattr__(
            self,
            "selection_points",
            self._immutable_points(self.selection_points),
        )

    def control_array(self) -> np.ndarray:
        return self.control_points

    def native_array(self) -> np.ndarray:
        return self.native_control_points

    def selection_array(self) -> np.ndarray:
        return self.selection_points


@dataclass(frozen=True)
class TextAppearanceState:
    fontsize: float


@dataclass(frozen=True)
class Line2DAppearanceState:
    linewidth: float
    markersize: float
    markeredgewidth: float


@dataclass(frozen=True)
class _Line2DAxisSpec:
    """Read-only description of one raw Line2D coordinate container."""

    raw: object = field(repr=False, compare=False)
    data: np.ndarray = field(repr=False, compare=False)
    shape: tuple[int, ...]
    size: int
    dtype: np.dtype
    fixed_dtype: bool


@dataclass(frozen=True)
class _Line2DRawWrite:
    """Preflighted Line2D destination, optionally with materialized containers."""

    native_points: np.ndarray = field(repr=False, compare=False)
    xdata: object | None = field(default=None, repr=False, compare=False)
    ydata: object | None = field(default=None, repr=False, compare=False)


@dataclass(frozen=True)
class CollectionAppearanceState:
    linewidths: tuple[float, ...]


@dataclass(frozen=True)
class PathCollectionAppearanceState(CollectionAppearanceState):
    sizes: tuple[float, ...]


@dataclass(frozen=True)
class AppearanceScalePlan:
    """Absolute, preflighted appearance destination for one Artist."""

    target: Artist
    factor: float
    state: object
    selection_points: tuple[tuple[float, float], ...]

    def selection_array(self) -> np.ndarray:
        return np.asarray(self.selection_points, dtype=float)


@dataclass(frozen=True)
class ChangeRecord:
    """One adapter-owned instruction for updating the ChangeTracker."""

    kind: str
    target: Artist
    command: Optional[str] = None
    reference_target: Optional[Artist] = None
    reference_command: Optional[str] = None

    @classmethod
    def command_change(
        cls,
        target: Artist,
        command: str,
        *,
        reference_target: Optional[Artist] = None,
        reference_command: Optional[str] = None,
    ) -> ChangeRecord:
        return cls(
            "command",
            target,
            command,
            reference_target,
            reference_command,
        )

    @classmethod
    def text_change(cls, target: Text) -> ChangeRecord:
        return cls("text", target)

    @classmethod
    def legend_change(cls, target: Legend) -> ChangeRecord:
        return cls("legend", target)

    @classmethod
    def legend_layout_change(cls, target: Legend) -> ChangeRecord:
        return cls("legend_layout", target)

    @classmethod
    def axes_change(cls, target: Axes) -> ChangeRecord:
        return cls("axes", target)

    def apply(self, tracker) -> None:
        if self.kind == "command":
            add_owned = getattr(tracker, "addOwnedChange", None)
            if self.reference_target is not None and add_owned is not None:
                add_owned(
                    self.target,
                    self.command,
                    self.reference_target,
                    self.reference_command,
                )
            else:
                tracker.addChange(self.target, self.command)
        elif self.kind == "text":
            tracker.addNewTextChange(self.target)
        elif self.kind == "legend":
            tracker.addNewLegendChange(self.target)
        elif self.kind == "legend_layout":
            add_layout = getattr(tracker, "addNewLegendLayoutChange", None)
            if callable(add_layout):
                add_layout(self.target)
            else:
                tracker.addChange(
                    self.target,
                    LegendLayoutSpec.from_legend(self.target).replay_command(),
                )
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
        """Whether the Artist may enter selection.

        Selection is intentionally independent from translation.  Property-
        only and layout-managed Artists still need an honest selection box;
        each mutation checks its own semantic operation at gesture start.
        """

        return self.capabilities.can_select

    def operation_support(
        self, operation: TransformOperation | str
    ) -> OperationSupport:
        operation = TransformOperation.coerce(operation)
        container_geometry_operation = operation in {
            TransformOperation.TRANSLATE,
            TransformOperation.RESIZE_GEOMETRY,
            TransformOperation.ROTATE,
            TransformOperation.RIGID_ROTATE,
        }
        if operation in {
            TransformOperation.ROTATE,
            TransformOperation.RIGID_ROTATE,
        }:
            if self.target.get_agg_filter() is not None:
                return OperationSupport.denied(
                    operation,
                    f"{type(self.target).__name__} has an Agg filter whose pixel "
                    "offset is not guaranteed to rotate with its geometry",
                )
            legend_owner = legend_owner_for_artist(self.target)
            if legend_owner is not None:
                return OperationSupport.denied(
                    operation,
                    f"{type(self.target).__name__} is managed by Legend layout "
                    "and has no stable native pivot or independent replay identity",
                )
            if isinstance(self.target, Text):
                layout_owner = layout_owner_for_text(self.target)
                if layout_owner is not None:
                    return OperationSupport.denied(
                        operation,
                        "Text position is managed by "
                        f"{type(layout_owner).__name__} layout and has no stable "
                        "native pivot",
                    )
        if container_geometry_operation:
            container_owner = container_owner_for_artist(self.target)
            if container_owner is not None:
                return OperationSupport.denied(
                    operation,
                    f"{type(self.target).__name__} is the background of "
                    f"{type(container_owner).__name__} and cannot be transformed "
                    "independently; select the container instead",
                )
            active_layout_owner = active_layout_owner_for_artist(self.target)
            if active_layout_owner is not None:
                return OperationSupport.denied(
                    operation,
                    f"{type(self.target).__name__} participates in active "
                    f"{type(active_layout_owner).__name__} layout; a draw could "
                    "move its coordinate system after the preview. Set "
                    "in_layout=False before transforming it independently",
                )
        capabilities = self.capabilities
        legacy_support = {
            TransformOperation.SELECT: capabilities.can_select,
            TransformOperation.TRANSLATE: capabilities.can_translate,
            TransformOperation.RESIZE_GEOMETRY: capabilities.can_resize,
            TransformOperation.ROTATE: capabilities.can_rotate,
            TransformOperation.RIGID_ROTATE: capabilities.can_rigid_rotate,
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
                else (
                    "rigid_rotation"
                    if operation is TransformOperation.RIGID_ROTATE
                    else "control_points"
                )
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

    def native_rotation_handle_support(self) -> OperationSupport:
        """Whether native angle editing is an honest visual handle operation."""

        operation = TransformOperation.ROTATE
        support = self.operation_support(operation)
        if not support.supported:
            return support
        return OperationSupport.denied(
            operation,
            f"{type(self.target).__name__} exposes a native angle property but "
            "does not guarantee rigid visual rotation around a stable pivot",
        )

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

    @classmethod
    def path_has_drawable_segment(
        cls, path: Path, transform: Optional[Transform] = None
    ) -> bool:
        """Whether a Path contains a non-degenerate strokable segment."""

        try:
            if transform is not None:
                path = transform.transform_path(path)
            current = None
            subpath_start = None
            for vertices, code in path.iter_segments(
                remove_nans=True, simplify=False, curves=True
            ):
                points = np.asarray(vertices, dtype=float).reshape(-1, 2)
                if code == Path.MOVETO:
                    current = points[-1]
                    subpath_start = current
                    continue
                if code == Path.CLOSEPOLY:
                    destination = subpath_start
                    candidates = [] if destination is None else [destination]
                else:
                    destination = points[-1] if len(points) else None
                    candidates = points
                if current is not None and any(
                    np.all(np.isfinite(point))
                    and not np.array_equal(point, current)
                    for point in candidates
                ):
                    return True
                if destination is not None:
                    current = destination
            return False
        except (AttributeError, TypeError, ValueError, RuntimeError):
            return False

    @classmethod
    def path_has_fill_area(
        cls, path: Path, transform: Optional[Transform] = None
    ) -> bool:
        """Whether filling a Path can cover a non-zero display area."""

        try:
            if transform is not None:
                path = transform.transform_path(path)
            polygons = path.to_polygons(closed_only=False)
        except (AttributeError, TypeError, ValueError, RuntimeError):
            return False
        for polygon in polygons:
            polygon = cls.finite_points(polygon)
            if len(polygon) < 3:
                continue
            x = polygon[:, 0]
            y = polygon[:, 1]
            area = 0.5 * abs(
                float(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
            )
            if np.isfinite(area) and area > np.finfo(float).eps:
                return True
        return False

    def _preview_points(self, attribute: str):
        preview = getattr(self.target, attribute, None)
        if preview is None:
            return None
        return np.asarray(preview, dtype=float).copy()

    @staticmethod
    def colors_are_visible(colors) -> bool:
        """Return whether at least one resolved RGBA color has non-zero alpha."""

        try:
            values = np.asarray(colors, dtype=float)
        except (TypeError, ValueError):
            try:
                values = np.asarray(mpl.colors.to_rgba(colors), dtype=float)
            except (TypeError, ValueError):
                return False
        if values.size == 0:
            return False
        if values.ndim == 1:
            return values.shape[0] < 4 or bool(values[3] > 0)
        return values.shape[-1] < 4 or bool(np.any(values[..., 3] > 0))

    def points_to_pixels(self, value: float) -> float:
        """Convert typographic points to the current renderer's display pixels."""

        try:
            return float(self.renderer().points_to_pixels(float(value)))
        except (AttributeError, TypeError, ValueError, RuntimeError):
            return float(value) * float(self.figure.dpi) / 72.0

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

    def clip_selection_points(self, points) -> np.ndarray:
        """Apply the active paint clip to a display-space selection envelope."""

        return _clip_selection_points(self.target, points)

    def has_visible_selection_bounds(self) -> bool:
        """Whether the current rendered envelope intersects its active clip."""

        try:
            points = self.clip_selection_points(self.selection_points())
        except (AttributeError, TypeError, ValueError, RuntimeError):
            return False
        return bool(len(self.finite_points(points)))

    def hit_test(self, event) -> bool:
        """Return whether a canvas event hits the artist's visible paint.

        Matplotlib's native ``contains`` remains authoritative by default.
        Adapters may add a renderer-faithful fallback for artist classes whose
        native picker ignores visible degenerate geometry.
        """

        try:
            return bool(self.target.contains(event)[0])
        except (AttributeError, TypeError, ValueError, RuntimeError):
            return False

    def geometry_bounds(self, control_points=None) -> np.ndarray:
        """Transformable display-space geometry, excluding paint appearance."""

        if control_points is None:
            control_points = self.control_points()
        return self.bounds_points(control_points)

    def appearance_outsets(
        self, *, control_points=None, selection_points=None
    ) -> np.ndarray:
        """Fixed display-pixel outsets from geometry to visible selection bounds."""

        geometry = self.geometry_bounds(control_points)
        if selection_points is None:
            selection_points = self.selection_points()
        visible = self.bounds_points(selection_points)
        if not len(geometry) or not len(visible):
            return np.zeros(4, dtype=float)
        return np.maximum(
            np.array(
                [
                    geometry[0, 0] - visible[0, 0],
                    geometry[0, 1] - visible[0, 1],
                    visible[1, 0] - geometry[1, 0],
                    visible[1, 1] - geometry[1, 1],
                ],
                dtype=float,
            ),
            0.0,
        )

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

    def serialize_appearance_changes(self) -> tuple[ChangeRecord, ...]:
        """Generated records owned specifically by appearance state."""

        return ()

    def _record_change_records(self, records) -> None:
        if not _CHANGE_RECORDING_ENABLED.get():
            return
        records = tuple(records)
        layout_only = records and all(
            record.kind == "legend_layout" for record in records
        )
        if records and not self.capabilities.can_serialize and not layout_only:
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
        self.validate_native_control_points(points)
        self._apply_native_control_points(points)
        self.record_changes()
        self.invalidate_geometry_cache()

    def validate_native_control_points(self, points) -> None:
        """Reject destination-specific geometry before mutating an artist."""

    def canonicalize_native_control_points(self, points):
        """Return the exact writable native destination after preflight.

        Most artists store ordinary floating-point coordinates, so validation
        leaves their destination untouched.  Adapters with narrower storage
        (for example, a float32 or integer ``Line2D``) may return the closest
        representation that still round-trips through display space within the
        interaction tolerance.
        """

        self.validate_native_control_points(points)
        return points

    def display_clip_bounds(self) -> np.ndarray | None:
        """Return a conservative display-space bbox for the active paint clip."""

        clip_box, clip_path = _display_clip_components(self.target)
        clip_bounds = [] if clip_box is None else [clip_box]
        if clip_path is not None:
            clip_bounds.append(np.asarray(clip_path.get_extents().extents, dtype=float))
        if not clip_bounds:
            return None
        result = clip_bounds[0].copy()
        for bounds in clip_bounds[1:]:
            result[:2] = np.maximum(result[:2], bounds[:2])
            result[2:] = np.minimum(result[2:], bounds[2:])
        return result

    def validate_translation_visibility(self, visible_points) -> None:
        """Reject translations whose paint becomes wholly clipped away."""

        clip_box, clip_path = _display_clip_components(self.target)
        if clip_box is None and clip_path is None:
            return
        visible = self.bounds_points(visible_points)
        if not len(visible):
            return
        candidate = Bbox.from_extents(*visible[0], *visible[1])
        visible_in_clip_box = clip_box is None or Bbox.from_extents(
            *clip_box
        ).overlaps(candidate)
        visible_in_clip_path = True
        if clip_path is not None:
            try:
                visible_in_clip_path = clip_path.intersects_bbox(
                    candidate, filled=True
                )
            except (TypeError, ValueError, RuntimeError):
                visible_in_clip_path = False
        if not visible_in_clip_box or not visible_in_clip_path:
            raise UnsupportedArtistError(
                f"{type(self.target).__name__} translation would move its visible "
                "geometry entirely outside the active clip region"
            )

    def preflight_translation(
        self,
        delta: Sequence[float],
        *,
        control_points=None,
        selection_points=None,
        destination_selection_points=None,
    ) -> None:
        support = self.operation_support(TransformOperation.TRANSLATE)
        if not support.supported:
            raise UnsupportedArtistError(support.reason)
        delta = np.asarray(delta, dtype=float)
        if delta.shape != (2,) or not np.all(np.isfinite(delta)):
            raise ValueError("Display-space translation must contain two finite values")
        if control_points is None:
            control_points = self.control_points()
        if selection_points is None:
            selection_points = self.selection_points()
        points = self.point_array(control_points)
        self.validate_native_control_points(self.display_to_native(points + delta))
        if destination_selection_points is None:
            self.validate_translation_visibility(
                self.point_array(selection_points) + delta
            )
        elif not len(self.finite_points(destination_selection_points)):
            raise UnsupportedArtistError(
                f"{type(self.target).__name__} translation would leave no visible "
                "geometry inside the active clip region"
            )

    def preview_translation_selection_points(
        self, delta: Sequence[float]
    ) -> np.ndarray:
        """Return the clipped visible envelope after a display translation."""

        delta = np.asarray(delta, dtype=float)
        if delta.shape != (2,) or not np.all(np.isfinite(delta)):
            raise ValueError("Display-space translation must contain two finite values")
        proposed = self.point_array(self.selection_points()) + delta
        return _clip_selection_points(self.target, proposed)

    def preflight_rigid_visible_translation(self, delta: Sequence[float]) -> None:
        """Require an exact visible-envelope shift for numeric/layout commands."""

        self.preflight_translation(delta)
        delta = np.asarray(delta, dtype=float)
        current = self.bounds_points(
            _clip_selection_points(self.target, self.selection_points())
        )
        proposed = self.bounds_points(self.preview_translation_selection_points(delta))
        if not len(current) or not len(proposed):
            raise UnsupportedArtistError(
                f"{type(self.target).__name__} has no visible bounds for exact translation"
            )
        expected = current + delta
        _clip_box, clip_path = _display_clip_components(self.target)
        if clip_path is not None:
            raw = self.bounds_points(self.point_array(self.selection_points()) + delta)
            corners = np.array(
                [
                    raw[0],
                    (raw[0, 0], raw[1, 1]),
                    raw[1],
                    (raw[1, 0], raw[0, 1]),
                ],
                dtype=float,
            )
            try:
                contained = bool(
                    np.all(clip_path.contains_points(corners, radius=1e-7))
                )
            except (TypeError, ValueError, RuntimeError):
                contained = False
            if not contained:
                raise UnsupportedArtistError(
                    f"{type(self.target).__name__} exact translation cannot preserve "
                    "visible bounds inside its non-rectangular clip path"
                )
        if not np.allclose(proposed, expected, atol=0.25, rtol=0.0):
            raise UnsupportedArtistError(
                f"{type(self.target).__name__} exact translation would change its "
                "visible bounds at the active clip region"
            )

    def apply_control_points(self, points) -> None:
        self.apply_native_control_points(self.display_to_native(points))

    def translate(self, delta: Sequence[float]) -> None:
        support = self.operation_support(TransformOperation.TRANSLATE)
        if not support.supported:
            raise UnsupportedArtistError(support.reason)
        delta = np.asarray(delta, dtype=float)
        if delta.shape != (2,):
            raise ValueError("Display-space translation must contain exactly x and y")
        if np.all(delta == 0):
            return
        self.preflight_translation(delta)
        points = self.point_array(self.control_points())
        self.apply_control_points(points + delta)

    @staticmethod
    def _transform_points(matrix, points) -> np.ndarray:
        points = ArtistAdapter.point_array(points)
        if not len(points):
            return points
        homogeneous = np.concatenate(
            [points, np.ones((len(points), 1), dtype=float)], axis=1
        )
        return np.asarray(homogeneous @ np.asarray(matrix, dtype=float).T)[:, :2]

    def apply_display_transform(self, matrix) -> None:
        support = self.operation_support(TransformOperation.TRANSLATE)
        if not support.supported:
            raise UnsupportedArtistError(support.reason)
        matrix = np.asarray(matrix, dtype=float)
        translation_matrix = np.eye(3)
        if matrix.shape == (3, 3):
            translation_matrix[:2, 2] = matrix[:2, 2]
        if matrix.shape != (3, 3) or not np.allclose(matrix, translation_matrix):
            raise UnsupportedArtistError(
                "apply_display_transform only accepts translation matrices; "
                "use the semantic resize or rotation operation instead"
            )
        self.translate(matrix[:2, 2])

    def preview_resize_control_points(
        self, matrix, *, control_points=None, selection_points=None
    ) -> np.ndarray:
        """Plan geometry for a resize whose handles transform visible bounds."""

        if control_points is None:
            control_points = self.control_points()
        return self._transform_points(matrix, control_points)

    def preview_resize_selection_points(
        self, matrix, *, control_points=None, selection_points=None
    ):
        """Plan the visible selection envelope shown during a resize gesture."""

        if selection_points is None:
            selection_points = self.selection_points()
        return self._transform_points(matrix, selection_points)

    def preflight_resize(
        self, matrix, *, control_points=None, selection_points=None
    ) -> np.ndarray:
        """Plan a resize and return its clipped visible envelope without mutation."""

        support = self.operation_support(TransformOperation.RESIZE_GEOMETRY)
        if not support.supported:
            raise UnsupportedArtistError(support.reason)
        matrix = np.asarray(matrix, dtype=float)
        if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
            raise ValueError("Display-space resize requires a finite 3x3 matrix")
        if not np.allclose(matrix[2], (0.0, 0.0, 1.0)) or not np.allclose(
            matrix[[0, 1], [1, 0]], 0.0
        ):
            raise UnsupportedArtistError(
                "Resize only accepts axis-aligned scale/translation matrices; "
                "use rigid rotation for off-diagonal geometry transforms"
            )
        if control_points is None:
            control_points = self.control_points()
        if selection_points is None:
            selection_points = self.selection_points()
        planned_control = self.preview_resize_control_points(
            matrix,
            control_points=control_points,
            selection_points=selection_points,
        )
        self.validate_native_control_points(self.display_to_native(planned_control))
        planned_selection = self.preview_resize_selection_points(
            matrix,
            control_points=control_points,
            selection_points=selection_points,
        )
        visible = _clip_selection_points(self.target, planned_selection)
        if not len(visible):
            raise UnsupportedArtistError(
                f"{type(self.target).__name__} resize would leave no visible geometry "
                "inside the active clip region"
            )
        return visible

    def preflight_rigid_visible_resize(self, matrix) -> np.ndarray:
        """Require a resize to preserve the planned visible-envelope transform."""

        current = self.bounds_points(
            _clip_selection_points(self.target, self.selection_points())
        )
        planned = self.bounds_points(self.preflight_resize(matrix))
        expected = self.bounds_points(self._transform_points(matrix, current))
        if not len(current) or not len(planned) or not len(expected):
            raise UnsupportedArtistError(
                f"{type(self.target).__name__} has no visible bounds for exact resize"
            )
        if not np.allclose(planned, expected, atol=0.25, rtol=0.0):
            raise UnsupportedArtistError(
                f"{type(self.target).__name__} exact resize would change its visible "
                "bounds at the active clip region"
            )
        return planned

    def resize(self, matrix) -> None:
        if not self.capabilities.can_resize:
            raise UnsupportedArtistError(
                f"{type(self.target).__name__} cannot be resized losslessly"
            )
        self.preflight_resize(matrix)
        self.apply_control_points(self.preview_resize_control_points(matrix))

    @staticmethod
    def validate_scale_factor(factor: float) -> float:
        factor = float(factor)
        if not np.isfinite(factor) or factor <= 0.0:
            raise ValueError("Scale factor must be finite and positive")
        return factor

    @classmethod
    def scale_nonnegative_dimensions(
        cls, values, factor: float, *, power: int = 1, label: str
    ) -> tuple[float, ...]:
        """Scale point/area dimensions without overflow or lossy underflow."""

        factor = cls.validate_scale_factor(factor)
        values = np.asarray(values, dtype=float)
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            scaled = values.copy()
            for _index in range(power):
                scaled = scaled * factor
        if not np.all(np.isfinite(scaled)):
            raise ValueError(f"{label} would overflow to a non-finite value")
        if np.any((values > 0.0) & (scaled <= 0.0)):
            raise ValueError(f"{label} would underflow to zero")
        return tuple(float(value) for value in scaled)

    def appearance_state(self):
        raise UnsupportedArtistError(
            f"{type(self.target).__name__} has no lossless appearance state"
        )

    def scaled_appearance_state(self, factor: float):
        raise UnsupportedArtistError(
            f"{type(self.target).__name__} has no lossless appearance scale"
        )

    def _apply_appearance_state(self, state) -> None:
        raise UnsupportedArtistError(
            f"{type(self.target).__name__} has no lossless appearance scale"
        )

    def preview_appearance_state_selection_points(self, state) -> np.ndarray:
        """Measure an absolute appearance state without mutating the target."""

        clone = copy(self.target)
        for attribute in (
            "_pylustrator_preview_positions",
            "_pylustrator_preview_selection_points",
        ):
            try:
                delattr(clone, attribute)
            except AttributeError:
                pass
        adapter = type(self)(clone)
        adapter._apply_appearance_state(state)
        return np.asarray(adapter.selection_points(), dtype=float)

    def plan_appearance_scale(self, factor: float) -> AppearanceScalePlan:
        support = self.operation_support(TransformOperation.SCALE_APPEARANCE)
        if not support.supported:
            raise UnsupportedArtistError(support.reason)
        return self._plan_preflighted_appearance_scale(factor)

    def _plan_preflighted_appearance_scale(
        self, factor: float
    ) -> AppearanceScalePlan:
        """Build a plan after the outer selection capability preflight."""

        factor = self.validate_scale_factor(factor)
        state = self.scaled_appearance_state(factor)
        planned = self.clip_selection_points(
            self.preview_appearance_state_selection_points(state)
        )
        if not len(self.finite_points(planned)):
            raise UnsupportedArtistError(
                f"{type(self.target).__name__} appearance scaling would leave no "
                "visible geometry inside the active clip region"
            )
        return AppearanceScalePlan(
            self.target,
            factor,
            state,
            tuple(tuple(float(value) for value in point) for point in planned),
        )

    def apply_appearance_scale_plan(
        self, plan: AppearanceScalePlan, *, record_changes: bool = True
    ) -> bool:
        if plan.target is not self.target:
            raise ValueError("Appearance-scale plan belongs to another artist")
        support = self.operation_support(TransformOperation.SCALE_APPEARANCE)
        if not support.supported:
            raise UnsupportedArtistError(support.reason)
        if plan.factor == 1.0:
            return False
        before = self.appearance_state()
        tracker = None
        tracker_state = None
        if record_changes:
            tracker = self.change_tracker()
            capture = getattr(tracker, "capture_recording_state", None)
            if callable(capture):
                tracker_state = capture()
        try:
            self._apply_preflighted_appearance_scale_plan(plan)
            if record_changes:
                self._record_change_records(self.serialize_appearance_changes())
        except Exception as error:
            rollback_failures = []
            try:
                self._apply_appearance_state(before)
            except Exception as rollback_error:
                rollback_failures.append((self.target, rollback_error))
            restore_tracker = getattr(tracker, "restore_recording_state", None)
            if tracker_state is not None and callable(restore_tracker):
                try:
                    restore_tracker(tracker_state)
                except Exception as rollback_error:
                    rollback_failures.append((tracker, rollback_error))
            self.annotate_rollback_failures(error, rollback_failures)
            self.invalidate_geometry_cache()
            raise
        return True

    def _apply_preflighted_appearance_scale_plan(
        self, plan: AppearanceScalePlan
    ) -> bool:
        """Apply one immutable plan inside an existing outer transaction."""

        if plan.target is not self.target:
            raise ValueError("Appearance-scale plan belongs to another artist")
        if plan.factor == 1.0:
            return False
        self._apply_appearance_state(plan.state)
        self.invalidate_geometry_cache()
        return True

    @staticmethod
    def annotate_rollback_failures(error: Exception, failures) -> None:
        failures = tuple(failures)
        if not failures:
            return
        try:
            setattr(error, "pylustrator_rollback_failures", failures)
        except (AttributeError, TypeError):
            pass
        add_note = getattr(error, "add_note", None)
        if callable(add_note):
            details = "; ".join(
                f"{type(target).__name__}: {rollback_error}"
                for target, rollback_error in failures
            )
            add_note(f"Pylustrator rollback failures: {details}")

    def scale_appearance(self, factor: float) -> bool:
        plan = self.plan_appearance_scale(factor)
        return self.apply_appearance_scale_plan(plan)

    def restore_appearance_state(
        self, state, *, record_changes: bool = True
    ) -> None:
        """Restore only paint/font state, independent of geometry snapshots."""

        before = self.appearance_state()
        tracker = None
        tracker_state = None
        if record_changes:
            tracker = self.change_tracker()
            capture = getattr(tracker, "capture_recording_state", None)
            if callable(capture):
                tracker_state = capture()
        try:
            self._apply_appearance_state(state)
            if record_changes:
                self._record_change_records(self.serialize_appearance_changes())
        except Exception as error:
            rollback_failures = []
            try:
                self._apply_appearance_state(before)
            except Exception as rollback_error:
                rollback_failures.append((self.target, rollback_error))
            restore_tracker = getattr(tracker, "restore_recording_state", None)
            if tracker_state is not None and callable(restore_tracker):
                try:
                    restore_tracker(tracker_state)
                except Exception as rollback_error:
                    rollback_failures.append((tracker, rollback_error))
            self.annotate_rollback_failures(error, rollback_failures)
            self.invalidate_geometry_cache()
            raise
        self.invalidate_geometry_cache()

    @staticmethod
    def display_rotation_matrix(angle_degrees: float, pivot) -> np.ndarray:
        angle_degrees = float(angle_degrees)
        pivot = np.asarray(pivot, dtype=float)
        if not np.isfinite(angle_degrees):
            raise ValueError("Rotation angle must be finite")
        if pivot.shape != (2,) or not np.all(np.isfinite(pivot)):
            raise ValueError("Rotation pivot must contain two finite display values")
        angle = np.deg2rad(angle_degrees)
        cosine = float(np.cos(angle))
        sine = float(np.sin(angle))
        linear = np.array([[cosine, -sine], [sine, cosine]], dtype=float)
        translation = pivot - linear @ pivot
        return np.array(
            [
                [linear[0, 0], linear[0, 1], translation[0]],
                [linear[1, 0], linear[1, 1], translation[1]],
                [0.0, 0.0, 1.0],
            ],
            dtype=float,
        )

    @staticmethod
    def transform_is_invertible_affine(transform: Transform) -> bool:
        if not bool(getattr(transform, "is_affine", False)) or not bool(
            getattr(transform, "has_inverse", True)
        ):
            return False
        try:
            transform.inverted()
        except (
            AttributeError,
            TypeError,
            ValueError,
            NotImplementedError,
            RuntimeError,
            np.linalg.LinAlgError,
        ):
            return False
        return True

    @staticmethod
    def transform_is_similarity(transform: Transform) -> bool:
        """Whether native rotations remain rigid rotations in display space."""

        if not bool(getattr(transform, "is_affine", False)):
            return False
        try:
            linear = np.asarray(
                transform.get_affine().get_matrix(), dtype=float
            )[:2, :2]
        except (AttributeError, TypeError, ValueError, RuntimeError):
            return False
        gram = linear.T @ linear
        scale_squared = float(np.trace(gram) / 2.0)
        return bool(
            np.isfinite(scale_squared)
            and scale_squared > np.finfo(float).eps
            and np.allclose(
                gram,
                np.eye(2) * scale_squared,
                atol=max(scale_squared, 1.0) * 1e-10,
                rtol=1e-10,
            )
        )

    def rigid_rotation_uses_native_angle(self) -> bool:
        return False

    def rigid_rotation_angle_delta(self, angle_degrees: float) -> float:
        return float(angle_degrees)

    def preview_rigid_rotation_control_points(
        self, matrix, *, control_points
    ) -> np.ndarray:
        control_points = self.point_array(control_points)
        if not self.rigid_rotation_uses_native_angle():
            return self._transform_points(matrix, control_points)
        current_pivot = np.asarray(self.rotation_pivot(), dtype=float)
        destination_pivot = self._transform_points(matrix, [current_pivot])[0]
        return control_points + destination_pivot - current_pivot

    def preview_rigid_rotation_selection_points(
        self,
        matrix,
        *,
        control_points,
        selection_points,
        planned_control_points,
        planned_native_control_points,
        rotation_value: float | None,
    ) -> np.ndarray:
        """Predict the visible envelope for geometry-backed rigid rotation."""

        geometry = self.geometry_bounds(planned_control_points)
        if not len(geometry):
            return np.empty((0, 2), dtype=float)
        outsets = self.appearance_outsets(
            control_points=control_points,
            selection_points=selection_points,
        )
        left, bottom, right, top = outsets
        return np.array(
            [
                [geometry[0, 0] - left, geometry[0, 1] - bottom],
                [geometry[1, 0] + right, geometry[1, 1] + top],
            ],
            dtype=float,
        )

    def preflight_rigid_rotation_clip(
        self, source_selection, destination_selection
    ) -> None:
        """Require v1 rigid rotation to remain wholly inside a rectangular clip."""

        clip_box, clip_path = _display_clip_components(self.target)
        if clip_path is not None:
            raise UnsupportedArtistError(
                f"{type(self.target).__name__} rigid rotation inside a "
                "non-rectangular clip path is not exact yet"
            )
        if clip_box is None:
            return
        source = self.bounds_points(source_selection)
        destination = self.bounds_points(destination_selection)
        if not len(source) or not len(destination):
            raise UnsupportedArtistError(
                f"{type(self.target).__name__} has no finite rigid-rotation bounds"
            )
        clip = np.asarray(clip_box, dtype=float).reshape(2, 2)

        def contained(bounds) -> bool:
            return bool(
                np.all(bounds[0] >= clip[0] - 0.25)
                and np.all(bounds[1] <= clip[1] + 0.25)
            )

        if not contained(source) or not contained(destination):
            raise UnsupportedArtistError(
                f"{type(self.target).__name__} rigid rotation cannot preserve a "
                "partially clipped visible envelope"
            )

    def plan_rigid_rotation(
        self,
        angle_degrees: float,
        pivot,
        *,
        control_points=None,
        selection_points=None,
    ) -> RigidRotationPlan:
        """Preflight one absolute display-space rotation destination."""

        support = self.operation_support(TransformOperation.RIGID_ROTATE)
        if not support.supported:
            raise UnsupportedArtistError(support.reason)
        angle_degrees = (
            float(angle_degrees) + 180.0
        ) % 360.0 - 180.0
        if np.isclose(angle_degrees, 0.0, atol=1e-12):
            angle_degrees = 0.0
        matrix = self.display_rotation_matrix(angle_degrees, pivot)
        if control_points is None:
            control_points = np.asarray(self.control_points(), dtype=float)
        else:
            control_points = np.asarray(control_points, dtype=float)
        if selection_points is None:
            selection_points = np.asarray(self.selection_points(), dtype=float)
        else:
            selection_points = np.asarray(selection_points, dtype=float)
        planned_control = self.preview_rigid_rotation_control_points(
            matrix, control_points=control_points
        )
        try:
            planned_native = self.point_array(
                self.display_to_native(planned_control)
            )
            planned_native = self.point_array(
                self.canonicalize_native_control_points(planned_native)
            )
            representable_control = self.point_array(
                self.native_to_display(planned_native)
            )
        except UnsupportedArtistError:
            raise
        except (
            AttributeError,
            TypeError,
            ValueError,
            NotImplementedError,
            RuntimeError,
            np.linalg.LinAlgError,
        ) as error:
            raise UnsupportedArtistError(
                f"{type(self.target).__name__} rigid rotation cannot convert its "
                "display destination through native coordinates"
            ) from error
        expected = self.point_array(planned_control)
        expected_finite = np.isfinite(expected)
        actual_finite = np.isfinite(representable_control)
        nonfinite_matches = bool(
            expected.shape == representable_control.shape
            and np.array_equal(expected_finite, actual_finite)
            and np.array_equal(np.isnan(expected), np.isnan(representable_control))
            and np.array_equal(np.isposinf(expected), np.isposinf(representable_control))
            and np.array_equal(np.isneginf(expected), np.isneginf(representable_control))
        )
        if expected.shape == representable_control.shape and np.any(expected_finite):
            round_trip_error = float(
                np.max(
                    np.abs(
                        expected[expected_finite]
                        - representable_control[expected_finite]
                    )
                )
            )
        else:
            round_trip_error = float("inf")
        if not nonfinite_matches or round_trip_error > 0.25:
            raise UnsupportedArtistError(
                f"{type(self.target).__name__} rigid rotation cannot round-trip "
                "its display destination through native coordinates within "
                f"0.25 px (error {round_trip_error:.6g} px)"
            )
        planned_control = representable_control
        rotation_value = (
            self.rotation() + self.rigid_rotation_angle_delta(angle_degrees)
            if self.rigid_rotation_uses_native_angle()
            else None
        )
        planned_selection = self.preview_rigid_rotation_selection_points(
            matrix,
            control_points=control_points,
            selection_points=selection_points,
            planned_control_points=planned_control,
            planned_native_control_points=planned_native,
            rotation_value=rotation_value,
        )
        self.preflight_rigid_rotation_clip(
            selection_points, planned_selection
        )
        visible = _clip_selection_points(self.target, planned_selection)
        if not len(self.finite_points(visible)):
            raise UnsupportedArtistError(
                f"{type(self.target).__name__} rigid rotation would leave no visible "
                "geometry inside the active clip region"
            )
        return RigidRotationPlan(
            target=self.target,
            angle_degrees=float(angle_degrees),
            pivot=tuple(float(value) for value in np.asarray(pivot, dtype=float)),
            control_points=planned_control,
            native_control_points=planned_native,
            selection_points=visible,
            rotation_value=(
                None if rotation_value is None else float(rotation_value)
            ),
            source_fingerprint=self.rigid_rotation_source_fingerprint(),
        )

    def rigid_rotation_source_fingerprint(self):
        """Return an inexpensive token used to reject stale absolute plans."""

        return None

    def validate_rigid_rotation_plan_source(self, plan: RigidRotationPlan) -> None:
        """Reject a plan whose source storage changed after preflight."""

    def apply_rigid_rotation_plan(
        self, plan: RigidRotationPlan, *, record_changes: bool = True
    ) -> None:
        if plan.target is not self.target:
            raise ValueError("Rigid-rotation plan belongs to another artist")
        support = self.operation_support(TransformOperation.RIGID_ROTATE)
        if not support.supported:
            raise UnsupportedArtistError(support.reason)
        self.validate_rigid_rotation_plan_source(plan)
        if not self.rigid_rotation_plan_changes(plan):
            return
        native = plan.native_array()
        self.validate_native_control_points(native)
        with suspend_change_recording():
            self._apply_native_control_points(native)
            if plan.rotation_value is not None:
                self._apply_rotation(plan.rotation_value)
        if record_changes:
            self._record_restored_state(
                include_rotation=plan.rotation_value is not None
            )
        self.invalidate_geometry_cache()

    def rigid_rotation_plan_changes(self, plan: RigidRotationPlan) -> bool:
        current = np.asarray(self.control_points(), dtype=float)
        planned = plan.control_array()
        if current.shape != planned.shape or not np.allclose(
            current, planned, equal_nan=True
        ):
            return True
        return bool(
            plan.rotation_value is not None
            and not np.isclose(self.rotation(), plan.rotation_value)
        )

    def rigid_rotate(self, angle_degrees: float, pivot) -> None:
        plan = self.plan_rigid_rotation(angle_degrees, pivot)
        self.apply_rigid_rotation_plan(plan)

    def rotation(self) -> float:
        raise UnsupportedArtistError(
            f"{type(self.target).__name__} has no native rotation property"
        )

    def preview_native_rotation_selection_points(self, value: float) -> np.ndarray:
        """Measure one absolute native angle without mutating the live Artist."""

        clone = copy(self.target)
        for attribute in (
            "_pylustrator_preview_positions",
            "_pylustrator_preview_selection_points",
        ):
            try:
                delattr(clone, attribute)
            except AttributeError:
                pass
        adapter = type(self)(clone)
        with suspend_change_recording():
            adapter._apply_rotation(float(value))
        return adapter.point_array(
            adapter.clip_selection_points(adapter.selection_points())
        )

    def rotation_pivot(self) -> np.ndarray:
        """Return the native-rotation anchor in display coordinates."""

        if not self.capabilities.can_rotate:
            raise UnsupportedArtistError(
                f"{type(self.target).__name__} cannot be rotated losslessly"
            )
        points = self.finite_points(self.control_points())
        if not len(points):
            raise UnsupportedArtistError(
                f"{type(self.target).__name__} has no finite rotation anchor"
            )
        return np.asarray(points[0], dtype=float)

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
        capabilities = self.capabilities
        if not capabilities.can_snapshot:
            raise UnsupportedArtistError(
                f"{type(self.target).__name__} does not support interaction snapshots"
            )
        state = {"type": "positions", "positions": self.local_control_points()}
        if capabilities.can_rotate:
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


class AdapterInheritancePolicy(str, Enum):
    """Whether one adapter registration may serve Artist subclasses.

    ``EXACT`` is the safe default: only the registered concrete Artist type is
    accepted.  ``VALIDATED`` is an explicit assertion by the adapter author
    that every subclass covered by that registration preserves the adapter's
    geometry, mutation, snapshot, and serialization contracts.
    """

    EXACT = "exact"
    VALIDATED = "validated"

    @classmethod
    def coerce(
        cls, value: "AdapterInheritancePolicy | str"
    ) -> "AdapterInheritancePolicy":
        if isinstance(value, cls):
            return value
        try:
            return cls(str(value).lower())
        except ValueError as error:
            choices = ", ".join(policy.value for policy in cls)
            raise ValueError(
                f"inheritance_policy must be one of: {choices}"
            ) from error


class UnsupportedSubclassAdapter(ArtistAdapter):
    """Fail-closed adapter for a subclass without an inheritance contract."""

    def __init__(
        self,
        target: Artist,
        *,
        blocked_registration: "AdapterRegistration | None" = None,
    ):
        super().__init__(target)
        self.blocked_registration = blocked_registration

    def operation_support(
        self, operation: TransformOperation | str
    ) -> OperationSupport:
        operation = TransformOperation.coerce(operation)
        registration = self.blocked_registration
        if registration is None:  # pragma: no cover - registry always supplies it
            reason = (
                f"{type(self.target).__name__} has no validated adapter "
                "inheritance contract"
            )
        else:
            reason = (
                f"{type(self.target).__name__} is a subclass of registered "
                f"{registration.artist_type.__name__}, but "
                f"{registration.adapter_type.__name__} is exact-only; register "
                "this concrete Artist type or explicitly use validated "
                "inheritance"
            )
        return OperationSupport.denied(operation, reason)

@dataclass(frozen=True)
class AdapterRegistration:
    artist_type: type
    adapter_type: type[ArtistAdapter]
    priority: int
    order: int
    inheritance_policy: AdapterInheritancePolicy = AdapterInheritancePolicy.EXACT

    def accepts(self, concrete: type) -> bool:
        return concrete is self.artist_type or (
            self.inheritance_policy is AdapterInheritancePolicy.VALIDATED
            and issubclass(concrete, self.artist_type)
        )


class ArtistAdapterRegistry:
    """Resolve an artist to the most specific registered adapter class."""

    def __init__(self):
        self._registrations: list[AdapterRegistration] = []
        self._cache: dict[type, type[ArtistAdapter]] = {}
        self._blocked_cache: dict[type, AdapterRegistration] = {}
        self._lock = RLock()
        self._next_order = 0

    def register(
        self,
        artist_type: type,
        adapter_type: type[ArtistAdapter],
        *,
        priority: int = 0,
        replace: bool = False,
        inheritance_policy: AdapterInheritancePolicy | str = (
            AdapterInheritancePolicy.EXACT
        ),
    ) -> type[ArtistAdapter]:
        if not isinstance(artist_type, type) or not issubclass(artist_type, Artist):
            raise TypeError("artist_type must be an Artist subclass")
        if not isinstance(adapter_type, type) or not issubclass(
            adapter_type, ArtistAdapter
        ):
            raise TypeError("adapter_type must be an ArtistAdapter subclass")
        inheritance_policy = AdapterInheritancePolicy.coerce(inheritance_policy)
        with self._lock:
            if replace:
                self._registrations = [
                    item
                    for item in self._registrations
                    if item.artist_type is not artist_type
                ]
            self._registrations.append(
                AdapterRegistration(
                    artist_type,
                    adapter_type,
                    int(priority),
                    self._next_order,
                    inheritance_policy,
                )
            )
            self._next_order += 1
            self._cache.clear()
            self._blocked_cache.clear()
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
            self._blocked_cache.clear()

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

    def _registration_rank(
        self, concrete: type, registration: AdapterRegistration
    ) -> tuple[int, int, int]:
        return (
            self._mro_distance(concrete, registration.artist_type),
            -registration.priority,
            -registration.order,
        )

    def _resolve_uncached(self, concrete: type) -> type[ArtistAdapter]:
        candidates = [
            item
            for item in self._registrations
            if issubclass(concrete, item.artist_type)
        ]
        if not candidates:
            raise LookupError(f"No artist adapter registered for {concrete!r}")
        selected = min(
            candidates,
            key=lambda item: self._registration_rank(concrete, item),
        )
        if selected.accepts(concrete):
            adapter_type = selected.adapter_type
        else:
            adapter_type = UnsupportedSubclassAdapter
            self._blocked_cache[concrete] = selected
        self._cache[concrete] = adapter_type
        return adapter_type

    def resolve_type(self, target_or_type) -> type[ArtistAdapter]:
        concrete = target_or_type if isinstance(target_or_type, type) else type(target_or_type)
        with self._lock:
            cached = self._cache.get(concrete)
            if cached is not None:
                return cached
            return self._resolve_uncached(concrete)

    def create(self, target: Artist) -> ArtistAdapter:
        with self._lock:
            concrete = type(target)
            adapter_type = self._cache.get(concrete)
            if adapter_type is None:
                adapter_type = self._resolve_uncached(concrete)
            blocked_registration = self._blocked_cache.get(concrete)
        if blocked_registration is not None:
            return UnsupportedSubclassAdapter(
                target,
                blocked_registration=blocked_registration,
            )
        return adapter_type(target)

    def capabilities_for(self, target: Artist) -> ArtistCapabilities:
        return self.resolve_type(target).capabilities_for(target)

    def supports(self, target: Artist) -> bool:
        return self.capabilities_for(target).can_select


class PatchAdapter(ArtistAdapter):
    unsupported_operation_reasons = {
        TransformOperation.RIGID_ROTATE: (
            "Patch common-pivot rotation requires writable path geometry or a "
            "similarity transform"
        )
    }
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

    def native_rotation_handle_support(self) -> OperationSupport:
        operation = TransformOperation.ROTATE
        support = self.operation_support(operation)
        if not support.supported:
            return support
        transform = self.target.get_data_transform()
        if not self.transform_is_similarity(transform):
            return OperationSupport.denied(
                operation,
                f"{type(self.target).__name__} native angle is not a rigid "
                "display rotation under its non-similarity transform",
            )
        linear = np.asarray(
            transform.get_affine().get_matrix(), dtype=float
        )[:2, :2]
        if np.linalg.det(linear) <= 0:
            return OperationSupport.denied(
                operation,
                f"{type(self.target).__name__} native angle reverses direction "
                "under its reflected transform",
            )
        if self.target.get_hatch() or self.target.get_path_effects():
            return OperationSupport.denied(
                operation,
                f"{type(self.target).__name__} hatch/path effects do not "
                "guarantee rigid visual rotation with its native angle",
            )
        return OperationSupport.allowed(
            operation,
            constraints=("stable_native_pivot",),
            preview_strategy="native_rotation",
        )

    def selection_points(self) -> np.ndarray:
        preview = self._preview_points("_pylustrator_preview_selection_points")
        if preview is not None:
            return preview
        points = super().selection_points()
        if not len(points) or not self.colors_are_visible(self.target.get_edgecolor()):
            return points
        padding = self.points_to_pixels(max(float(self.target.get_linewidth()), 0.0)) / 2
        return self.bounds_points(points, padding=padding)

    @staticmethod
    def _point_segment_distance(point, start, end) -> float:
        point = np.asarray(point, dtype=float)
        start = np.asarray(start, dtype=float)
        end = np.asarray(end, dtype=float)
        segment = end - start
        length_squared = float(np.dot(segment, segment))
        if not np.isfinite(length_squared) or length_squared <= 0:
            return float(np.linalg.norm(point - start))
        offset = float(np.dot(point - start, segment) / length_squared)
        closest = start + np.clip(offset, 0.0, 1.0) * segment
        return float(np.linalg.norm(point - closest))

    def _stroke_hit_test(self, point, tolerance: float) -> bool:
        """Hit-test the transformed path centerline in display coordinates."""

        try:
            path = self.target.get_transform().transform_path(self.target.get_path())
        except (AttributeError, TypeError, ValueError, RuntimeError):
            return False

        current = None
        start = None
        try:
            segments = path.iter_segments(curves=False, simplify=False)
            for vertices, code in segments:
                vertex = np.asarray(vertices[-2:], dtype=float)
                if vertex.shape != (2,) or not np.all(np.isfinite(vertex)):
                    current = None
                    start = None
                    continue
                if code == Path.MOVETO or current is None:
                    current = vertex
                    start = vertex
                    continue
                endpoint = start if code == Path.CLOSEPOLY and start is not None else vertex
                if self._point_segment_distance(point, current, endpoint) <= tolerance:
                    return True
                current = endpoint
        except (AttributeError, TypeError, ValueError, RuntimeError):
            return False
        return False

    def hit_test(self, event) -> bool:
        if super().hit_test(event):
            return True
        if not self.colors_are_visible(self.target.get_edgecolor()):
            return False
        point = np.asarray((event.x, event.y), dtype=float)
        if not np.all(np.isfinite(point)):
            return False

        # Three pixels matches the editor's direct-manipulation tolerance while
        # the stroke half-width keeps thick outlines selectable across their
        # complete painted envelope. A numeric Matplotlib picker is expressed in
        # typographic points and may request a larger tolerance.
        tolerance = max(
            3.0,
            self.points_to_pixels(max(float(self.target.get_linewidth()), 0.0)) / 2,
        )
        picker = self.target.get_picker()
        if isinstance(picker, (int, float)) and not isinstance(picker, bool):
            tolerance = max(tolerance, self.points_to_pixels(float(picker)))
        return self._stroke_hit_test(point, tolerance)

    def preview_resize_control_points(
        self, matrix, *, control_points=None, selection_points=None
    ) -> np.ndarray:
        """Resize path geometry while keeping stroke width fixed in display pixels."""

        matrix = np.asarray(matrix, dtype=float)
        if control_points is None:
            control_points = np.asarray(self.control_points(), dtype=float)
        else:
            control_points = np.asarray(control_points, dtype=float)
        if selection_points is None:
            selection_points = np.asarray(self.selection_points(), dtype=float)
        else:
            selection_points = np.asarray(selection_points, dtype=float)

        geometry = self.geometry_bounds(control_points)
        visible = self.bounds_points(selection_points)
        if not len(geometry) or not len(visible):
            return super().preview_resize_control_points(
                matrix,
                control_points=control_points,
                selection_points=selection_points,
            )
        if matrix.shape != (3, 3) or not np.allclose(
            matrix[[0, 1, 2, 2], [1, 0, 0, 1]], 0.0
        ):
            return super().preview_resize_control_points(
                matrix,
                control_points=control_points,
                selection_points=selection_points,
            )

        outsets = self.appearance_outsets(
            control_points=control_points, selection_points=selection_points
        )
        transformed_visible = self._transform_points(matrix, visible)
        sx, sy = float(matrix[0, 0]), float(matrix[1, 1])
        left, bottom, right, top = outsets
        target_x0 = transformed_visible[0, 0] + (left if sx >= 0 else -right)
        target_x1 = transformed_visible[1, 0] - (right if sx >= 0 else -left)
        target_y0 = transformed_visible[0, 1] + (bottom if sy >= 0 else -top)
        target_y1 = transformed_visible[1, 1] - (top if sy >= 0 else -bottom)

        def clamp_collapsed_axis(
            original_low,
            original_high,
            desired_first,
            desired_second,
            low_outset,
            high_outset,
            target_first,
            target_second,
            scale,
        ):
            if not np.isclose(scale, 0.0) and scale * (
                target_second - target_first
            ) >= 0:
                return target_first, target_second
            desired_low = min(desired_first, desired_second)
            desired_high = max(desired_first, desired_second)
            if np.isclose(desired_low, original_low):
                collapsed = desired_low + low_outset
            elif np.isclose(desired_high, original_high):
                collapsed = desired_high - high_outset
            else:
                collapsed = (
                    desired_low + desired_high + low_outset - high_outset
                ) / 2
            return collapsed, collapsed

        target_x0, target_x1 = clamp_collapsed_axis(
            visible[0, 0],
            visible[1, 0],
            transformed_visible[0, 0],
            transformed_visible[1, 0],
            left,
            right,
            target_x0,
            target_x1,
            sx,
        )
        target_y0, target_y1 = clamp_collapsed_axis(
            visible[0, 1],
            visible[1, 1],
            transformed_visible[0, 1],
            transformed_visible[1, 1],
            bottom,
            top,
            target_y0,
            target_y1,
            sy,
        )

        width = float(geometry[1, 0] - geometry[0, 0])
        height = float(geometry[1, 1] - geometry[0, 1])
        if np.isclose(width, 0.0) or np.isclose(height, 0.0):
            return super().preview_resize_control_points(
                matrix,
                control_points=control_points,
                selection_points=selection_points,
            )
        geometry_matrix = np.array(
            [
                [
                    (target_x1 - target_x0) / width,
                    0.0,
                    target_x0
                    - (target_x1 - target_x0) / width * geometry[0, 0],
                ],
                [
                    0.0,
                    (target_y1 - target_y0) / height,
                    target_y0
                    - (target_y1 - target_y0) / height * geometry[0, 1],
                ],
                [0.0, 0.0, 1.0],
            ],
            dtype=float,
        )
        return self._transform_points(geometry_matrix, control_points)

    def preview_resize_selection_points(
        self, matrix, *, control_points=None, selection_points=None
    ) -> np.ndarray:
        """Derive the preview envelope from the same fixed-stroke geometry plan."""

        if control_points is None:
            control_points = np.asarray(self.control_points(), dtype=float)
        else:
            control_points = np.asarray(control_points, dtype=float)
        if selection_points is None:
            selection_points = np.asarray(self.selection_points(), dtype=float)
        else:
            selection_points = np.asarray(selection_points, dtype=float)
        outsets = self.appearance_outsets(
            control_points=control_points, selection_points=selection_points
        )
        planned = self.preview_resize_control_points(
            matrix,
            control_points=control_points,
            selection_points=selection_points,
        )
        geometry = self.geometry_bounds(planned)
        if not len(geometry):
            return super().preview_resize_selection_points(
                matrix,
                control_points=control_points,
                selection_points=selection_points,
            )
        left, bottom, right, top = outsets
        return np.array(
            [
                [geometry[0, 0] - left, geometry[0, 1] - bottom],
                [geometry[1, 0] + right, geometry[1, 1] + top],
            ],
            dtype=float,
        )

    def rigid_rotation_uses_native_angle(self) -> bool:
        return bool(
            self.capabilities.can_rigid_rotate and self.capabilities.can_rotate
        )

    def rigid_rotation_angle_delta(self, angle_degrees: float) -> float:
        if not self.rigid_rotation_uses_native_angle():
            return super().rigid_rotation_angle_delta(angle_degrees)
        linear = np.asarray(
            self.target.get_data_transform().get_affine().get_matrix(),
            dtype=float,
        )[:2, :2]
        orientation = 1.0 if np.linalg.det(linear) >= 0 else -1.0
        return orientation * float(angle_degrees)

    def preview_rigid_rotation_selection_points(
        self,
        matrix,
        *,
        control_points,
        selection_points,
        planned_control_points,
        planned_native_control_points,
        rotation_value: float | None,
    ) -> np.ndarray:
        if not self.rigid_rotation_uses_native_angle():
            return super().preview_rigid_rotation_selection_points(
                matrix,
                control_points=control_points,
                selection_points=selection_points,
                planned_control_points=planned_control_points,
                planned_native_control_points=planned_native_control_points,
                rotation_value=rotation_value,
            )
        clone = copy(self.target)
        for attribute in (
            "_pylustrator_preview_positions",
            "_pylustrator_preview_selection_points",
        ):
            try:
                delattr(clone, attribute)
            except AttributeError:
                pass
        adapter = type(self)(clone)
        adapter._apply_native_control_points(planned_native_control_points)
        adapter._apply_rotation(float(rotation_value))
        return np.asarray(adapter.selection_points(), dtype=float)

    def rotation(self) -> float:
        return float(self.target.get_angle())

    def _apply_rotation(self, value: float) -> None:
        self.target.set_angle(value)

    def serialize_rotation_changes(self):
        return (
            ChangeRecord.command_change(
                self.target,
                f".set_angle({replay_literal(self.target.get_angle())})",
            ),
        )


class RectangleAdapter(PatchAdapter):
    @classmethod
    def capabilities_for(cls, target: Rectangle) -> ArtistCapabilities:
        # A 180-degree Rectangle normally rotates around its ``xy`` anchor and
        # therefore occupies the opposite side of the unrotated writable
        # controls. Only a true 0-degree equivalent has matching controls and
        # rendered geometry. Ellipse is center-anchored and has different rules.
        can_resize = np.isclose(float(target.get_angle()) % 360.0, 0.0)
        rotation_point = getattr(
            target, "rotation_point", getattr(target, "_rotation_point", "xy")
        )
        owner_managed = bool(
            legend_owner_for_artist(target) is not None
            or container_owner_for_artist(target) is not None
            or active_layout_owner_for_artist(target) is not None
            or target.get_agg_filter() is not None
        )
        can_rigid_rotate = (
            not owner_managed
            and cls.transform_is_similarity(target.get_data_transform())
            and isinstance(rotation_point, str)
            and rotation_point in {"xy", "center"}
            and target.get_agg_filter() is None
            and not target.get_path_effects()
            and not target.get_hatch()
        )
        return ArtistCapabilities(
            can_select=True,
            can_translate=True,
            can_resize=bool(can_resize),
            can_snapshot=True,
            can_serialize=True,
            can_rotate=not owner_managed,
            can_rigid_rotate=bool(can_rigid_rotate),
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

    def rotation_pivot(self) -> np.ndarray:
        rotation_point = getattr(
            self.target,
            "rotation_point",
            getattr(self.target, "_rotation_point", "xy"),
        )
        if rotation_point == "center":
            return np.mean(np.asarray(self.control_points(), dtype=float), axis=0)
        if rotation_point != "xy":
            return np.asarray(self.native_to_display([rotation_point])[0], dtype=float)
        return np.asarray(self.control_points()[0], dtype=float)

    def _apply_native_control_points(self, points) -> None:
        self.target.set_xy(points[0])
        self.target.set_width(points[1][0] - points[0][0])
        self.target.set_height(points[1][1] - points[0][1])

    def serialize_changes(self):
        return (
            ChangeRecord.command_change(
                self.target, f".set_xy({replay_literal(self.target.get_xy())})"
            ),
            ChangeRecord.command_change(
                self.target,
                f".set_width({replay_literal(self.target.get_width())})",
            ),
            ChangeRecord.command_change(
                self.target,
                f".set_height({replay_literal(self.target.get_height())})",
            ),
        )


class EllipseAdapter(PatchAdapter):
    @classmethod
    def capabilities_for(cls, target: Ellipse) -> ArtistCapabilities:
        can_resize = np.isclose(float(target.get_angle()) % 180.0, 0.0)
        owner_managed = bool(
            legend_owner_for_artist(target) is not None
            or container_owner_for_artist(target) is not None
            or active_layout_owner_for_artist(target) is not None
            or target.get_agg_filter() is not None
        )
        can_rigid_rotate = (
            not owner_managed
            and cls.transform_is_similarity(target.get_data_transform())
            and target.get_agg_filter() is None
            and not target.get_path_effects()
            and not target.get_hatch()
        )
        return ArtistCapabilities(
            can_select=True,
            can_translate=True,
            can_resize=bool(can_resize),
            can_snapshot=True,
            can_serialize=True,
            can_rotate=not owner_managed,
            can_rigid_rotate=bool(can_rigid_rotate),
        )

    def native_control_points(self):
        center = np.asarray(self.target.center, dtype=float)
        size = np.asarray((self.target.width, self.target.height), dtype=float)
        return [center - size / 2, center + size / 2]

    def rotation_pivot(self) -> np.ndarray:
        return np.mean(np.asarray(self.control_points(), dtype=float), axis=0)

    def _apply_native_control_points(self, points) -> None:
        self.target.center = np.mean(points, axis=0)
        self.target.width = points[1][0] - points[0][0]
        self.target.height = points[1][1] - points[0][1]

    def serialize_changes(self):
        return (
            ChangeRecord.command_change(
                self.target, f".center = {replay_literal(tuple(self.target.center))}"
            ),
            ChangeRecord.command_change(
                self.target, f".width = {replay_literal(self.target.width)}"
            ),
            ChangeRecord.command_change(
                self.target, f".height = {replay_literal(self.target.height)}"
            ),
        )


class _CenterOnlyPatchAdapter(PatchAdapter):
    """Conservative contract for semantic patches positioned by a center."""

    default_capabilities = ArtistCapabilities(
        can_select=True,
        can_translate=True,
        can_snapshot=True,
        can_serialize=True,
    )
    unsupported_operation_reasons = {
        TransformOperation.TRANSLATE: (
            "Semantic patch translation requires an invertible affine "
            "transform and independent layout ownership"
        ),
        TransformOperation.RESIZE_GEOMETRY: (
            "Semantic patch resize requires a type-specific size contract"
        ),
        TransformOperation.ROTATE: (
            "Semantic patch rotation requires a type-specific angle contract"
        ),
        TransformOperation.RIGID_ROTATE: (
            "Semantic patch common-pivot rotation has not been validated"
        ),
    }

    @classmethod
    def capabilities_for(cls, target) -> ArtistCapabilities:
        transform = target.get_data_transform()
        movable = bool(
            transform.is_affine
            and getattr(transform, "has_inverse", True)
            and legend_owner_for_artist(target) is None
            and container_owner_for_artist(target) is None
            and active_layout_owner_for_artist(target) is None
        )
        if movable:
            try:
                transform.inverted()
            except (
                TypeError,
                ValueError,
                NotImplementedError,
                RuntimeError,
                np.linalg.LinAlgError,
            ):
                movable = False
        return ArtistCapabilities(
            can_select=True,
            can_translate=movable,
            can_snapshot=movable,
            can_serialize=True,
        )

    def native_control_points(self):
        return [np.asarray(self.target.center, dtype=float)]

    def _apply_native_control_points(self, points) -> None:
        self.target.set_center(tuple(float(value) for value in points[0]))

    def serialize_changes(self):
        return (
            ChangeRecord.command_change(
                self.target,
                f".set_center({replay_literal(tuple(self.target.center))})",
            ),
        )


class ArcAdapter(_CenterOnlyPatchAdapter):
    """Translate Arc centers without rewriting its ellipse/angle semantics."""

    unsupported_operation_reasons = {
        **_CenterOnlyPatchAdapter.unsupported_operation_reasons,
        TransformOperation.RESIZE_GEOMETRY: (
            "Arc resize must preserve width, height, and angular span semantics"
        ),
        TransformOperation.ROTATE: (
            "Arc rotation must update its semantic angle contract"
        ),
        TransformOperation.RIGID_ROTATE: (
            "Arc common-pivot rotation has not been validated"
        ),
    }


class CircleAdapter(_CenterOnlyPatchAdapter):
    """Translate Circle centers without stretching its radius semantics."""

    unsupported_operation_reasons = {
        **_CenterOnlyPatchAdapter.unsupported_operation_reasons,
        TransformOperation.RESIZE_GEOMETRY: (
            "Circle resize must update one semantic radius without stretching"
        ),
        TransformOperation.ROTATE: (
            "Circle native rotation has no independently saveable visual effect"
        ),
        TransformOperation.RIGID_ROTATE: (
            "Circle common-pivot rotation has not been validated"
        ),
    }


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
                f".set_positions({replay_literal(tuple(point_a))}, "
                f"{replay_literal(tuple(point_b))})",
            ),
        )


class ConnectionPatchAdapter(FancyArrowPatchAdapter):
    """ConnectionPatch endpoints may live in two unrelated coordinate systems."""

    default_capabilities = ArtistCapabilities()


class FancyBboxPatchAdapter(PatchAdapter):
    def operation_support(
        self, operation: TransformOperation | str
    ) -> OperationSupport:
        operation = TransformOperation.coerce(operation)
        if (
            operation is TransformOperation.TRANSLATE
            and legend_owner_for_artist(self.target) is not None
        ):
            return OperationSupport.denied(
                operation,
                "Legend frame geometry is owned by its layout and is recomputed "
                "on draw; select the Legend instead",
            )
        return super().operation_support(operation)

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
            ChangeRecord.command_change(
                self.target, f".set_bounds{replay_literal(bounds)}"
            ),
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
            ChangeRecord.command_change(
                self.target, f".xy = {replay_literal(self.target.xy)}"
            ),
        )


class CirclePolygonAdapter(RegularPolygonAdapter):
    """Exact, translation-only contract for CirclePolygon."""

    unsupported_operation_reasons = {
        TransformOperation.TRANSLATE: (
            "CirclePolygon translation requires an invertible affine "
            "transform and independent layout ownership"
        ),
        TransformOperation.RESIZE_GEOMETRY: (
            "CirclePolygon resize must update its semantic radius"
        ),
        TransformOperation.ROTATE: (
            "CirclePolygon rotation must update its semantic orientation"
        ),
        TransformOperation.RIGID_ROTATE: (
            "CirclePolygon common-pivot rotation has not been validated"
        ),
    }

    @classmethod
    def capabilities_for(cls, target) -> ArtistCapabilities:
        return _CenterOnlyPatchAdapter.capabilities_for(target)


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
            ChangeRecord.command_change(
                self.target, f".set_center({replay_literal(center)})"
            ),
        )


class PolygonAdapter(PatchAdapter):
    @classmethod
    def capabilities_for(cls, target: Polygon) -> ArtistCapabilities:
        if not len(cls.finite_points(target.get_xy())):
            return ArtistCapabilities()
        capabilities = super().capabilities_for(target)
        return replace(
            capabilities,
            can_rigid_rotate=bool(
                legend_owner_for_artist(target) is None
                and active_layout_owner_for_artist(target) is None
                and cls.transform_is_invertible_affine(target.get_data_transform())
                and target.get_agg_filter() is None
                and not target.get_path_effects()
                and not target.get_hatch()
            ),
        )

    def native_control_points(self):
        return [np.asarray(point, dtype=float) for point in self.target.get_xy()]

    def _apply_native_control_points(self, points) -> None:
        self.target.set_xy([[float(x), float(y)] for x, y in points])

    def serialize_changes(self):
        vertices = [[float(x), float(y)] for x, y in self.target.get_xy()]
        return (
            ChangeRecord.command_change(
                self.target, f".set_xy({replay_literal(vertices)})"
            ),
        )


class PathPatchAdapter(PatchAdapter):
    @classmethod
    def capabilities_for(cls, target: PathPatch) -> ArtistCapabilities:
        if not len(cls.finite_points(target.get_path().vertices)):
            return ArtistCapabilities()
        capabilities = super().capabilities_for(target)
        return replace(
            capabilities,
            can_rigid_rotate=bool(
                legend_owner_for_artist(target) is None
                and active_layout_owner_for_artist(target) is None
                and cls.transform_is_invertible_affine(target.get_data_transform())
                and target.get_agg_filter() is None
                and not target.get_path_effects()
                and not target.get_hatch()
            ),
        )

    def geometry_bounds(self, control_points=None) -> np.ndarray:
        """Measure the rendered Bezier path, not its off-curve control hull."""

        if control_points is None:
            control_points = self.control_points()
        old_path = self.target.get_path()
        codes = None if old_path.codes is None else old_path.codes.copy()
        path = Path(self.point_array(control_points), codes)
        bounds = np.asarray(path.get_extents().get_points(), dtype=float)
        return bounds if np.all(np.isfinite(bounds)) else np.empty((0, 2), dtype=float)

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
                f".set_path(mpl.path.Path({replay_literal(vertices)}, "
                f"{replay_literal(codes)}))",
            ),
        )


class TextAdapter(ArtistAdapter):
    unsupported_operation_reasons = {
        TransformOperation.RESIZE_GEOMETRY: (
            "Text bounds come from font metrics; use appearance scaling instead of geometry resize"
        ),
        TransformOperation.ROTATE: (
            "Legend-managed Text rotation reflows its layout and cannot preserve "
            "a stable native pivot"
        ),
        TransformOperation.RIGID_ROTATE: (
            "Text common-pivot rotation requires rotation_mode='anchor', no "
            "transform-relative text angle, and no layout-managed owner"
        ),
    }
    default_capabilities = ArtistCapabilities(
        can_select=True,
        can_translate=True,
        can_snapshot=True,
        can_serialize=True,
        can_rotate=True,
    )

    @classmethod
    def capabilities_for(cls, target: Text) -> ArtistCapabilities:
        if (
            legend_owner_for_text(target) is not None
            or layout_owner_for_text(target) is not None
            or active_layout_owner_for_artist(target) is not None
            or target.get_agg_filter() is not None
        ):
            capabilities = cls.default_capabilities
            return ArtistCapabilities(
                can_select=capabilities.can_select,
                can_translate=capabilities.can_translate,
                can_resize=capabilities.can_resize,
                can_snapshot=capabilities.can_snapshot,
                can_serialize=capabilities.can_serialize,
                fixed_aspect=capabilities.fixed_aspect,
                can_rotate=False,
            )
        capabilities = cls.default_capabilities
        transform_rotates = bool(
            getattr(target, "get_transform_rotates_text", lambda: False)()
        )
        can_rigid_rotate = bool(
            not isinstance(target, Annotation)
            and target.get_rotation_mode() == "anchor"
            and not transform_rotates
            and cls.transform_is_invertible_affine(target.get_transform())
            and target.get_agg_filter() is None
            and target.get_bbox_patch() is None
            and not target.get_wrap()
            and not target.get_path_effects()
        )
        return replace(
            capabilities, can_rigid_rotate=can_rigid_rotate
        )

    def native_rotation_handle_support(self) -> OperationSupport:
        operation = TransformOperation.ROTATE
        support = self.operation_support(operation)
        if not support.supported:
            return support
        return OperationSupport.denied(
            operation,
            f"{type(self.target).__name__} native angle does not rotate its "
            "complete visible bounds rigidly around the displayed pivot; use "
            "an anchor-mode rigid rotation plan",
        )

    def operation_support(
        self, operation: TransformOperation | str
    ) -> OperationSupport:
        operation = TransformOperation.coerce(operation)
        if operation is TransformOperation.SCALE_APPEARANCE:
            if isinstance(self.target, Annotation):
                return OperationSupport.denied(
                    operation,
                    "Annotation appearance scaling must include its text and arrow",
                )
            owner = legend_owner_for_text(self.target)
            if owner is not None:
                return OperationSupport.denied(
                    operation,
                    "Legend-managed Text must scale through its Legend layout",
                )
            layout_owner = layout_owner_for_text(self.target)
            if layout_owner is not None:
                return OperationSupport.denied(
                    operation,
                    f"Text is managed by {type(layout_owner).__name__} layout",
                )
            active_layout_owner = active_layout_owner_for_artist(self.target)
            if active_layout_owner is not None:
                return OperationSupport.denied(
                    operation,
                    "Text participates in active layout and cannot scale independently",
                )
            if self.target.get_agg_filter() is not None:
                return OperationSupport.denied(
                    operation,
                    "Text has an Agg filter whose pixel effect cannot be scaled losslessly",
                )
            if self.target.get_bbox_patch() is not None:
                return OperationSupport.denied(
                    operation,
                    "Text with a bbox must scale text and box appearance as one plan",
                )
            if self.target.get_wrap():
                return OperationSupport.denied(
                    operation,
                    "Wrapped Text can reflow nonlinearly when its font size changes",
                )
            if self.target.get_path_effects():
                return OperationSupport.denied(
                    operation,
                    "Text path effects have independent display-space dimensions",
                )
            if self.target.get_sketch_params() is not None:
                return OperationSupport.denied(
                    operation,
                    "Text sketch effects have independent display-space dimensions",
                )
            if self.target.get_usetex():
                return OperationSupport.denied(
                    operation,
                    "TeX Text metrics are external and cannot be previewed losslessly",
                )
            if bool(
                getattr(self.target, "get_transform_rotates_text", lambda: False)()
            ):
                return OperationSupport.denied(
                    operation,
                    "Transform-relative Text angle cannot be previewed independently",
                )
            if not self.transform_is_invertible_affine(self.target.get_transform()):
                return OperationSupport.denied(
                    operation,
                    "Text appearance scaling requires an invertible affine transform",
                )
            if not self.target.get_visible() or not self.target.get_text().strip():
                return OperationSupport.denied(
                    operation,
                    "Text has no visible glyphs to scale",
                )
            try:
                text_color = mpl.colors.to_rgba(
                    self.target.get_color(), self.target.get_alpha()
                )
            except (TypeError, ValueError):
                return OperationSupport.denied(
                    operation,
                    "Text color cannot be resolved to visible paint",
                )
            if not self.colors_are_visible(text_color):
                return OperationSupport.denied(
                    operation,
                    "Text has no visible paint to scale",
                )
            if not self.has_visible_selection_bounds():
                return OperationSupport.denied(
                    operation,
                    "Text has no visible geometry inside its active clip region",
                )
            fontsize = float(self.target.get_fontsize())
            if not np.isfinite(fontsize) or fontsize <= 0.0:
                return OperationSupport.denied(
                    operation,
                    "Text font size must be finite and positive",
                )
            return OperationSupport.allowed(
                operation,
                constraints=("positive_uniform_factor",),
                preview_strategy="redraw",
            )
        if (
            operation is TransformOperation.TRANSLATE
            and (
                getattr(
                    self.target,
                    "_pylustrator_formatter_owned_tick_label",
                    False,
                )
                or axis_tick_label_reference(self.target) is not None
            )
        ):
            return OperationSupport.denied(
                operation,
                "Tick-label position is managed by its Axis; edit tick content "
                "or Axis spacing instead of dragging the generated Text",
            )
        if operation is TransformOperation.TRANSLATE:
            legend_owner = legend_owner_for_text(self.target)
            if (
                legend_owner is not None
                and self.target is legend_owner.get_title()
                and (
                    not self.target.get_visible()
                    or not self.target.get_text().strip()
                )
            ):
                return OperationSupport.denied(
                    operation,
                    "An empty or hidden Legend title has no stable visible "
                    "geometry and its position is recomputed on draw",
                )
            layout_owner = layout_owner_for_text(self.target)
            if isinstance(layout_owner, Axes) and bool(
                getattr(layout_owner, "_autotitlepos", True)
            ):
                return OperationSupport.denied(
                    operation,
                    "Axes title position is managed by automatic title layout "
                    "and is recomputed on draw; pass an explicit title y position "
                    "before translating it independently",
                )
            if self.target is getattr(layout_owner, "offsetText", None):
                return OperationSupport.denied(
                    operation,
                    "Axis offset-text position is formatter-owned and is "
                    "recomputed on draw",
                )
        return super().operation_support(operation)

    def appearance_state(self):
        return TextAppearanceState(float(self.target.get_fontsize()))

    def scaled_appearance_state(self, factor: float):
        (fontsize,) = self.scale_nonnegative_dimensions(
            [self.target.get_fontsize()], factor, label="Text font size"
        )
        return TextAppearanceState(fontsize)

    @staticmethod
    def validate_appearance_state(state) -> TextAppearanceState:
        if not isinstance(state, TextAppearanceState):
            raise TypeError("Text appearance plan has an invalid state type")
        if not np.isfinite(state.fontsize) or state.fontsize <= 0.0:
            raise ValueError("Text font size must be finite and positive")
        return state

    def _apply_appearance_state(self, state) -> None:
        state = self.validate_appearance_state(state)
        from .change_tracker import add_text_default

        add_text_default(self.target)
        self.target.set_fontsize(state.fontsize)

    def preview_appearance_state_selection_points(self, state) -> np.ndarray:
        """Measure font changes without sharing mutable FontProperties state."""

        clone = copy(self.target)
        clone.set_fontproperties(copy(self.target.get_fontproperties()))
        for attribute in (
            "_pylustrator_preview_positions",
            "_pylustrator_preview_selection_points",
        ):
            try:
                delattr(clone, attribute)
            except AttributeError:
                pass
        adapter = type(self)(clone)
        adapter._apply_appearance_state(state)
        return np.asarray(adapter.selection_points(), dtype=float)

    def serialize_appearance_changes(self):
        return (
            ChangeRecord.command_change(
                self.target,
                f".set_fontsize({replay_literal(self.target.get_fontsize())})",
            ),
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
                if edge_alpha > 0 and float(bbox_patch.get_linewidth()) > 0:
                    padding = self.points_to_pixels(
                        float(bbox_patch.get_linewidth())
                    ) / 2
                    patch_bbox = patch_bbox.padded(padding)
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
                    axes,
                    f".{axis_name}axis.labelpad = {replay_literal(axis.labelpad)}",
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

    def rigid_rotation_uses_native_angle(self) -> bool:
        return self.capabilities.can_rigid_rotate

    def preview_rigid_rotation_selection_points(
        self,
        matrix,
        *,
        control_points,
        selection_points,
        planned_control_points,
        planned_native_control_points,
        rotation_value: float | None,
    ) -> np.ndarray:
        clone = copy(self.target)
        for attribute in (
            "_pylustrator_preview_positions",
            "_pylustrator_preview_selection_points",
        ):
            try:
                delattr(clone, attribute)
            except AttributeError:
                pass
        adapter = type(self)(clone)
        adapter._apply_native_control_points(planned_native_control_points)
        adapter._apply_rotation(float(rotation_value))
        return np.asarray(adapter.selection_points(), dtype=float)

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

    def operation_support(self, operation: TransformOperation | str) -> OperationSupport:
        operation = TransformOperation.coerce(operation)
        if operation is TransformOperation.SCALE_APPEARANCE:
            return OperationSupport.denied(
                operation,
                "Annotation appearance scaling must include its text and optional arrow",
            )
        support = super().operation_support(operation)
        if support.supported and operation is TransformOperation.TRANSLATE:
            clip = self.target.get_annotation_clip()
            if clip or (clip is None and self.target.xycoords == "data"):
                return OperationSupport.allowed(
                    operation,
                    constraints=(*support.constraints, "annotated_point_within_owning_axes"),
                    preview_strategy=support.preview_strategy,
                )
        return support

    def validate_native_control_points(self, points) -> None:
        clip = self.target.get_annotation_clip()
        clipped_to_annotated_point = clip or (
            clip is None and self.target.xycoords == "data"
        )
        if not clipped_to_annotated_point or len(points) < 2:
            return
        axes = self.target.axes
        if axes is None:
            raise UnsupportedArtistError(
                "Clipped Annotation translation requires an owning Axes"
            )
        xy_display = np.asarray(self._xy_transform().transform(points[1]), dtype=float)
        if not axes.contains_point(xy_display):
            raise UnsupportedArtistError(
                "Annotation translation would move its annotated point outside "
                "the owning Axes while annotation clipping is active"
            )

    def invalidate_geometry_cache(self) -> None:
        super().invalidate_geometry_cache()
        arrow = self.target.arrow_patch
        if arrow is not None:
            setattr(arrow, "_pylustrator_cached_get_extend", None)

    def selection_points(self) -> np.ndarray:
        preview = self._preview_points("_pylustrator_preview_selection_points")
        if preview is not None:
            return preview
        point_sets = [super().selection_points()]
        arrow = self.target.arrow_patch
        if arrow is not None and arrow.get_visible():
            try:
                arrow_adapter = get_artist_adapter(arrow)
                point_sets.append(
                    cached_selection_points(arrow, arrow_adapter.selection_points)
                )
            except (AttributeError, TypeError, ValueError, RuntimeError):
                pass
        point_sets = [points for points in point_sets if len(points)]
        if point_sets:
            return self.bounds_points(np.concatenate(point_sets))
        return np.empty((0, 2), dtype=float)

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
            ChangeRecord.command_change(
                self.target, f".xy = {replay_literal(self.target.xy)}"
            ),
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

    def _constrain_native_control_points(self, points) -> np.ndarray:
        """Apply the same fixed-aspect rule to previews and committed geometry."""

        points = self.point_array(points).copy()
        if not self.capabilities.fixed_aspect or len(points) < 2:
            return points
        current = self.target.get_position()
        if np.isclose(current.width, 0.0):
            return points
        ratio = float(current.height / current.width)
        points[1, 1] = points[0, 1] + (points[1, 0] - points[0, 0]) * ratio
        return points

    def preview_resize_control_points(
        self, matrix, *, control_points=None, selection_points=None
    ) -> np.ndarray:
        points = super().preview_resize_control_points(
            matrix,
            control_points=control_points,
            selection_points=selection_points,
        )
        native = self._constrain_native_control_points(self.display_to_native(points))
        return self.point_array(self.native_to_display(native))

    def preview_resize_selection_points(
        self, matrix, *, control_points=None, selection_points=None
    ) -> np.ndarray:
        if not self.capabilities.fixed_aspect:
            return super().preview_resize_selection_points(
                matrix,
                control_points=control_points,
                selection_points=selection_points,
            )
        return self.bounds_points(
            self.preview_resize_control_points(
                matrix,
                control_points=control_points,
                selection_points=selection_points,
            )
        )

    def _apply_native_control_points(self, points) -> None:
        points = self._constrain_native_control_points(points)
        position = np.array([points[0], points[1] - points[0]]).flatten()
        self.target.set_position(position)

    def snapshot(self):
        state = super().snapshot()
        state["in_layout"] = bool(self.target.get_in_layout())
        return state

    def restore(self, state) -> None:
        if state.get("type") != "positions":
            raise ValueError(
                f"Unsupported snapshot for {type(self).__name__}: {state!r}"
            )
        self._apply_native_control_points(state["positions"])
        if "in_layout" in state:
            self.target.set_in_layout(bool(state["in_layout"]))
        self._record_restored_state()
        self.invalidate_geometry_cache()

    def serialize_changes(self):
        return (ChangeRecord.axes_change(self.target),)


class LegendAdapter(ArtistAdapter):
    unsupported_operation_reasons = {
        TransformOperation.RESIZE_GEOMETRY: (
            "Legend size is controlled by layout; use legend reflow instead of stretching its bounds"
        ),
    }
    default_capabilities = ArtistCapabilities(
        can_select=True,
        can_translate=True,
        can_snapshot=True,
        can_serialize=True,
    )

    @classmethod
    def capabilities_for(cls, target) -> ArtistCapabilities:
        if target.axes is None or target.axes.get_legend() is not target:
            return cls.default_capabilities

        from .legend_replay import (
            UnsupportedLegendEntry,
            axes_handles_reproduce_legend,
            frozen_legend_handles_code,
            original_axes_legend_handles_labels,
        )

        axes_handles, axes_labels = original_axes_legend_handles_labels(target.axes)
        if axes_handles_reproduce_legend(target, axes_handles, axes_labels):
            return cls.default_capabilities
        try:
            frozen_legend_handles_code(target)
        except UnsupportedLegendEntry:
            # The entry remains selectable, but transforms and snapshots would
            # need to emit a lossy creation command for this composite handler.
            return ArtistCapabilities(can_select=True)
        return cls.default_capabilities

    def __init__(self, target: Legend):
        super().__init__(target)
        if not hasattr(target, "_pylustrator_original_frameon"):
            target._pylustrator_original_frameon = target.get_frame_on()
        ensure_legend_layout_baseline(target)

    def operation_support(
        self, operation: TransformOperation | str
    ) -> OperationSupport:
        operation = TransformOperation.coerce(operation)
        if operation is not TransformOperation.REFLOW_LAYOUT:
            return super().operation_support(operation)
        try:
            LegendLayoutPlan.preflight(
                self.target, LegendLayoutSpec.from_legend(self.target)
            )
        except LegendLayoutError as error:
            return OperationSupport.denied(operation, str(error))
        return OperationSupport.allowed(
            operation,
            constraints=("standard_offsetbox_tree", "identity_preserving"),
            preview_strategy="post_draw_layout",
        )

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
            try:
                frame = self.target.get_frame()
                frame_adapter = get_artist_adapter(frame)
                frame_points = cached_selection_points(
                    frame, frame_adapter.selection_points
                )
            except (AttributeError, TypeError, ValueError, RuntimeError):
                frame_points = super().selection_points()
            if len(frame_points):
                point_sets.append(frame_points)

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
                child_adapter = get_artist_adapter(child)
                points = np.asarray(
                    cached_selection_points(child, child_adapter.selection_points),
                    dtype=float,
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
        frame = self.target.get_frame()
        frame_properties = {
            "linewidth": float(frame.get_linewidth()),
            "edgecolor": tuple(float(value) for value in frame.get_edgecolor()),
            "facecolor": tuple(float(value) for value in frame.get_facecolor()),
            "alpha": frame.get_alpha(),
        }
        frame_command = (
            ".get_frame().set("
            + ", ".join(
                f"{name}={replay_literal(value)}"
                for name, value in frame_properties.items()
            )
            + ")"
        )
        frame_record = ChangeRecord.command_change(self.target, frame_command)
        if self.target.axes is not None and self.target.axes.get_legend() is self.target:
            frame_record = ChangeRecord.command_change(
                self.target.axes,
                ".get_legend()" + frame_command,
                reference_target=self.target,
                reference_command=".get_frame",
            )
        return (
            ChangeRecord.legend_change(self.target),
            frame_record,
        )

    def set_frame_on(self, visible: bool) -> bool:
        """Toggle the legend frame without replacing the Legend object."""

        visible = bool(visible)
        if self.target.get_frame_on() == visible:
            return False
        support = self.operation_support(TransformOperation.SERIALIZE)
        if not support.supported:
            raise UnsupportedArtistError(support.reason)
        self.target.set_frame_on(visible)
        self.record_changes()
        self.invalidate_geometry_cache()
        return True

    def layout_state(self) -> LegendLayoutState:
        support = self.operation_support(TransformOperation.REFLOW_LAYOUT)
        if not support.supported:
            raise UnsupportedArtistError(support.reason)
        return capture_legend_layout_state(self.target)

    def plan_layout_reflow(
        self,
        destination: LegendLayoutSpec | dict[str, object],
        *,
        selected_artists: Iterable[Artist] = (),
    ) -> LegendLayoutPlan:
        try:
            return plan_legend_layout(
                self.target,
                destination,
                selected_artists=selected_artists,
            )
        except LegendLayoutError as error:
            raise UnsupportedArtistError(str(error)) from error

    def apply_layout_reflow_plan(
        self,
        plan: LegendLayoutPlan,
        *,
        record_changes: bool = True,
    ) -> bool:
        if plan.target is not self.target:
            raise ValueError("Legend layout plan belongs to another object")
        if plan.destination == plan.source_spec:
            return False
        before = capture_legend_layout_state(self.target)
        tracker = None
        tracker_state = None
        if record_changes:
            tracker = self.change_tracker()
            capture = getattr(tracker, "capture_recording_state", None)
            tracker_state = capture() if callable(capture) else None
        try:
            changed = plan.apply()
            invalidate_legend_owner_inventory(self.figure)
            self.invalidate_geometry_cache()
            if record_changes:
                self._record_change_records(
                    (ChangeRecord.legend_layout_change(self.target),)
                )
        except Exception as error:
            rollback_failures = []
            try:
                restore_legend_layout_state(self.target, before)
                invalidate_legend_owner_inventory(self.figure)
                self.invalidate_geometry_cache()
            except Exception as rollback_error:
                rollback_failures.append((self.target, rollback_error))
            restore_tracker = getattr(tracker, "restore_recording_state", None)
            if tracker_state is not None and callable(restore_tracker):
                try:
                    restore_tracker(tracker_state)
                except Exception as rollback_error:
                    rollback_failures.append((tracker, rollback_error))
            self.annotate_rollback_failures(error, rollback_failures)
            raise
        return changed

    def restore_layout_state(
        self,
        state: LegendLayoutState,
        *,
        record_changes: bool = True,
    ) -> None:
        before = capture_legend_layout_state(self.target)
        tracker = None
        tracker_state = None
        if record_changes:
            tracker = self.change_tracker()
            capture = getattr(tracker, "capture_recording_state", None)
            tracker_state = capture() if callable(capture) else None
        try:
            restore_legend_layout_state(self.target, state)
            invalidate_legend_owner_inventory(self.figure)
            self.invalidate_geometry_cache()
            if record_changes:
                self._record_change_records(
                    (ChangeRecord.legend_layout_change(self.target),)
                )
        except Exception as error:
            rollback_failures = []
            try:
                restore_legend_layout_state(self.target, before)
                invalidate_legend_owner_inventory(self.figure)
                self.invalidate_geometry_cache()
            except Exception as rollback_error:
                rollback_failures.append((self.target, rollback_error))
            restore_tracker = getattr(tracker, "restore_recording_state", None)
            if tracker_state is not None and callable(restore_tracker):
                try:
                    restore_tracker(tracker_state)
                except Exception as rollback_error:
                    rollback_failures.append((tracker, rollback_error))
            self.annotate_rollback_failures(error, rollback_failures)
            raise

    def reflow_layout(
        self,
        destination: LegendLayoutSpec | dict[str, object],
        *,
        selected_artists: Iterable[Artist] = (),
        record_changes: bool = True,
    ) -> bool:
        plan = self.plan_layout_reflow(
            destination, selected_artists=selected_artists
        )
        return self.apply_layout_reflow_plan(
            plan, record_changes=record_changes
        )

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
            can_rigid_rotate=all(
                value.can_rigid_rotate for value in capabilities
            ),
        )

    def operation_support(
        self, operation: TransformOperation | str
    ) -> OperationSupport:
        operation = TransformOperation.coerce(operation)
        if operation is not TransformOperation.RIGID_ROTATE:
            return super().operation_support(operation)
        supports = [
            (adapter.target, adapter.operation_support(operation))
            for adapter in self._member_adapters()
        ]
        failures = [
            (target, support)
            for target, support in supports
            if not support.supported
        ]
        if failures:
            reason = "; ".join(
                f"{type(target).__name__}: {support.reason}"
                for target, support in failures
            )
            return OperationSupport.denied(operation, reason)
        return OperationSupport.allowed(
            operation, preview_strategy="rigid_rotation"
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
            cached_selection_points(adapter.target, adapter.selection_points)
            for adapter in self._member_adapters()
            if adapter.capabilities.can_select and adapter.target.get_visible()
        ]
        points = [value for value in points if len(value)]
        return np.concatenate(points) if points else np.empty((0, 2), dtype=float)

    def preflight_translation(
        self,
        delta: Sequence[float],
        *,
        control_points=None,
        selection_points=None,
        destination_selection_points=None,
    ) -> None:
        support = self.operation_support(TransformOperation.TRANSLATE)
        if not support.supported:
            raise UnsupportedArtistError(support.reason)
        # The non-rendering group has no clip of its own. Every leaf must accept
        # the same display delta before any member is allowed to mutate.
        for adapter in self._member_adapters():
            adapter.preflight_translation(delta)

    def preflight_rigid_visible_translation(self, delta: Sequence[float]) -> None:
        for adapter in self._member_adapters():
            adapter.preflight_rigid_visible_translation(delta)

    def preflight_resize(
        self, matrix, *, control_points=None, selection_points=None
    ) -> np.ndarray:
        support = self.operation_support(TransformOperation.RESIZE_GEOMETRY)
        if not support.supported:
            raise UnsupportedArtistError(support.reason)
        adapters = self._member_adapters()
        if control_points is None:
            control_points = np.asarray(self.control_points(), dtype=float)
        else:
            control_points = np.asarray(control_points, dtype=float)
        planned = []
        start = 0
        for adapter in adapters:
            length = len(adapter.control_points())
            member_points = control_points[start : start + length]
            if adapter.target.get_visible():
                planned.extend(
                    adapter.preflight_resize(
                        matrix,
                        control_points=member_points,
                        selection_points=adapter.selection_points(),
                    )
                )
            start += length
        if start != len(control_points):
            raise ValueError("Editor-group control-point count changed during preflight")
        return self.bounds_points(planned)

    def preflight_rigid_visible_resize(self, matrix) -> np.ndarray:
        planned = [
            adapter.preflight_rigid_visible_resize(matrix)
            for adapter in self._member_adapters()
            if adapter.target.get_visible()
        ]
        if not planned:
            raise UnsupportedArtistError(
                "EditorGroup has no visible bounds for exact resize"
            )
        return self.bounds_points(planned)

    def preview_resize_control_points(
        self, matrix, *, control_points=None, selection_points=None
    ) -> np.ndarray:
        """Plan each leaf geometry while the group shares one visible transform."""

        adapters = self._member_adapters()
        if control_points is None:
            control_points = np.asarray(self.control_points(), dtype=float)
        else:
            control_points = np.asarray(control_points, dtype=float)
        planned = []
        start = 0
        for adapter in adapters:
            length = len(adapter.control_points())
            member_points = control_points[start : start + length]
            planned.extend(
                adapter.preview_resize_control_points(
                    matrix,
                    control_points=member_points,
                    selection_points=cached_selection_points(
                        adapter.target, adapter.selection_points
                    ),
                )
            )
            start += length
        if start != len(control_points):
            raise ValueError("Editor-group control-point count changed during preview")
        return np.asarray(planned, dtype=float)

    def preview_resize_selection_points(
        self, matrix, *, control_points=None, selection_points=None
    ) -> np.ndarray:
        adapters = self._member_adapters()
        if control_points is None:
            control_points = np.asarray(self.control_points(), dtype=float)
        else:
            control_points = np.asarray(control_points, dtype=float)
        planned = []
        start = 0
        for adapter in adapters:
            length = len(adapter.control_points())
            member_points = control_points[start : start + length]
            if adapter.target.get_visible():
                planned.extend(
                    adapter.preview_resize_selection_points(
                        matrix,
                        control_points=member_points,
                        selection_points=cached_selection_points(
                            adapter.target, adapter.selection_points
                        ),
                    )
                )
            start += length
        if start != len(control_points):
            raise ValueError("Editor-group control-point count changed during preview")
        return self.bounds_points(planned)

    def plan_rigid_rotation(
        self,
        angle_degrees: float,
        pivot,
        *,
        control_points=None,
        selection_points=None,
    ) -> RigidRotationPlan:
        support = self.operation_support(TransformOperation.RIGID_ROTATE)
        if not support.supported:
            raise UnsupportedArtistError(support.reason)
        self.display_rotation_matrix(angle_degrees, pivot)
        adapters = self._member_adapters()
        member_plans = tuple(
            adapter.plan_rigid_rotation(angle_degrees, pivot)
            for adapter in adapters
        )
        control = np.concatenate(
            [plan.control_array() for plan in member_plans], axis=0
        )
        native = np.concatenate(
            [plan.native_array() for plan in member_plans], axis=0
        )
        visible = [
            plan.selection_array()
            for adapter, plan in zip(adapters, member_plans)
            if adapter.target.get_visible() and len(plan.selection_points)
        ]
        if not visible:
            raise UnsupportedArtistError(
                "EditorGroup has no visible bounds for rigid rotation"
            )
        selection = self.bounds_points(np.concatenate(visible))
        return RigidRotationPlan(
            target=self.target,
            angle_degrees=float(angle_degrees),
            pivot=tuple(float(value) for value in np.asarray(pivot, dtype=float)),
            control_points=control,
            native_control_points=native,
            selection_points=selection,
            member_plans=member_plans,
        )

    def apply_rigid_rotation_plan(
        self, plan: RigidRotationPlan, *, record_changes: bool = True
    ) -> None:
        if plan.target is not self.target:
            raise ValueError("Rigid-rotation plan belongs to another artist")
        adapters = self._member_adapters()
        if len(adapters) != len(plan.member_plans):
            raise ValueError("Editor-group membership changed after rotation preflight")
        if not any(
            adapter.rigid_rotation_plan_changes(member_plan)
            for adapter, member_plan in zip(adapters, plan.member_plans)
        ):
            return
        snapshots = [adapter.snapshot() for adapter in adapters]
        tracker_states = []
        seen_trackers = set()
        for adapter in adapters:
            try:
                tracker = adapter.change_tracker()
            except AttributeError:
                continue
            if id(tracker) in seen_trackers:
                continue
            capture = getattr(tracker, "capture_recording_state", None)
            restore = getattr(tracker, "restore_recording_state", None)
            if not callable(capture) or not callable(restore):
                continue
            seen_trackers.add(id(tracker))
            tracker_states.append((tracker, capture()))
        try:
            with suspend_change_recording():
                for adapter, member_plan in zip(adapters, plan.member_plans):
                    adapter.apply_rigid_rotation_plan(
                        member_plan, record_changes=False
                    )
            if record_changes:
                self._record_restored_state()
        except Exception:
            with suspend_change_recording():
                for adapter, state in zip(reversed(adapters), reversed(snapshots)):
                    try:
                        adapter.restore(state)
                    except Exception:
                        continue
            for tracker, state in tracker_states:
                tracker.restore_recording_state(state)
            self.invalidate_geometry_cache()
            raise
        self.invalidate_geometry_cache()

    def _apply_native_control_points(self, points) -> None:
        start = 0
        # A logical group is the transaction/serialization boundary.  Member
        # adapters still own their mutation semantics, but the outer group
        # records their resulting commands exactly once after all succeed.
        with suspend_change_recording():
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
        ),
        TransformOperation.RIGID_ROTATE: (
            "Line common-pivot rotation requires an affine transform, default "
            "drawstyle, and either no visible marker or a continuously "
            "rotation-symmetric marker"
        ),
        TransformOperation.SCALE_APPEARANCE: (
            "Line appearance scaling requires finite stroke and marker dimensions"
        ),
    }
    default_capabilities = ArtistCapabilities(
        can_select=True,
        can_translate=True,
        can_snapshot=True,
        can_serialize=True,
    )

    @staticmethod
    def _plain_numeric_scalar(value) -> bool:
        return bool(
            not isinstance(value, (bool, np.bool_, complex, np.complexfloating))
            and isinstance(value, (int, float, np.integer, np.floating))
        )

    @classmethod
    def _raw_axis_metadata_reason(cls, raw, axis_name: str) -> str | None:
        """Cheap domain probe used by frequently queried capability paths."""

        if np.ma.isMaskedArray(raw):
            data = np.asarray(raw.data)
        elif type(raw) is np.ndarray:
            data = raw
        elif type(raw) in (list, tuple):
            if not raw:
                return f"Line2D raw {axis_name} coordinates are empty"
            sample = raw[0]
            depth = 1
            while type(sample) in (list, tuple):
                if not sample:
                    return (
                        f"Line2D raw {axis_name} coordinates have an empty "
                        "nested dimension"
                    )
                sample = sample[0]
                depth += 1
                if depth > 2:
                    return (
                        f"Line2D raw {axis_name} coordinates have more than "
                        "two dimensions"
                    )
            if isinstance(sample, np.ndarray):
                data = sample
            elif not cls._plain_numeric_scalar(sample):
                return (
                    f"Line2D raw {axis_name} coordinates use categorical, "
                    "datetime, or custom-unit values"
                )
            else:
                return None
        else:
            return (
                f"Line2D raw {axis_name} coordinates use unsupported "
                f"{type(raw).__name__} storage"
            )
        if data.ndim not in (1, 2):
            return (
                f"Line2D raw {axis_name} coordinates must be one- or "
                "two-dimensional"
            )
        kind = data.dtype.kind
        if kind in "fiu":
            return None
        if kind == "O" and data.size and cls._plain_numeric_scalar(data.flat[0]):
            return None
        return (
            f"Line2D raw {axis_name} coordinates use categorical, datetime, "
            f"complex, boolean, or custom-unit dtype {data.dtype}"
        )

    @classmethod
    def _raw_metadata_reason(cls, target: Line2D) -> str | None:
        for axis_name, raw in (
            ("x", target.get_xdata(orig=True)),
            ("y", target.get_ydata(orig=True)),
        ):
            reason = cls._raw_axis_metadata_reason(raw, axis_name)
            if reason is not None:
                return reason
        return None

    @classmethod
    def capabilities_for(cls, target: Line2D) -> ArtistCapabilities:
        if not len(cls.finite_points(target.get_xydata())):
            return ArtistCapabilities()
        adapter = cls(target)
        raw_reason = cls._raw_metadata_reason(target)
        try:
            marker_rotation_supported = adapter._marker_rigid_rotation_supported(
                strict=True, resolve_positions=False
            )
            line_rotation_supported = adapter._line_rigid_rotation_supported(
                strict=True
            )
        except UnsupportedArtistError:
            marker_rotation_supported = False
            line_rotation_supported = False
        can_rigid_rotate = bool(
            raw_reason is None
            and legend_owner_for_artist(target) is None
            and active_layout_owner_for_artist(target) is None
            and cls.transform_is_invertible_affine(target.get_transform())
            and target.get_drawstyle() == "default"
            and marker_rotation_supported
            and line_rotation_supported
            and target.get_agg_filter() is None
            and not target.get_path_effects()
            and target.get_sketch_params() is None
        )
        return replace(
            cls.default_capabilities,
            can_rigid_rotate=can_rigid_rotate,
        )

    @classmethod
    def _raw_axis_spec(
        cls, raw, axis_name: str, processed_length: int
    ) -> _Line2DAxisSpec:
        reason = cls._raw_axis_metadata_reason(raw, axis_name)
        if reason is not None:
            raise UnsupportedArtistError(reason)
        if np.ma.isMaskedArray(raw):
            data = np.asarray(raw.data)
            fixed_dtype = True
        else:
            try:
                data = np.asarray(raw)
            except (TypeError, ValueError) as error:
                raise UnsupportedArtistError(
                    f"Line2D raw {axis_name} coordinates cannot be flattened "
                    "losslessly"
                ) from error
            fixed_dtype = type(raw) is np.ndarray
        if data.ndim not in (1, 2) or data.size not in (1, processed_length):
            raise UnsupportedArtistError(
                f"Line2D raw {axis_name} coordinate shape {data.shape} cannot "
                f"represent {processed_length} processed vertices; only exact "
                "ravel length or length-one broadcast storage is supported"
            )
        if data.dtype.kind == "O":
            if not all(cls._plain_numeric_scalar(value) for value in data.flat):
                raise UnsupportedArtistError(
                    f"Line2D raw {axis_name} object coordinates contain "
                    "categorical, datetime, or custom-unit values"
                )
        return _Line2DAxisSpec(
            raw=raw,
            data=data,
            shape=tuple(int(value) for value in data.shape),
            size=int(data.size),
            dtype=data.dtype,
            fixed_dtype=fixed_dtype,
        )

    @staticmethod
    def _cast_axis_destination(
        spec: _Line2DAxisSpec, values, axis_name: str
    ) -> np.ndarray:
        values = np.asarray(values, dtype=float)
        if not np.all(np.isfinite(values)):
            raise UnsupportedArtistError(
                f"Line2D {axis_name} destination contains non-finite values "
                "for a currently visible vertex"
            )
        if not spec.fixed_dtype:
            return values
        kind = spec.dtype.kind
        if kind == "f":
            with np.errstate(over="ignore", invalid="ignore"):
                encoded = values.astype(spec.dtype, copy=False)
        elif kind in "iu":
            rounded = np.rint(values)
            limits = np.iinfo(spec.dtype)
            if np.any(rounded < limits.min) or np.any(rounded > limits.max):
                raise UnsupportedArtistError(
                    f"Line2D {axis_name} destination exceeds {spec.dtype} range"
                )
            encoded = rounded.astype(spec.dtype)
        elif kind == "O":
            encoded = values.astype(object)
        else:  # protected by _raw_axis_spec; keep third-party subclasses typed
            raise UnsupportedArtistError(
                f"Line2D {axis_name} dtype {spec.dtype} is not losslessly writable"
            )
        numeric = np.asarray(encoded, dtype=float)
        if not np.all(np.isfinite(numeric)):
            raise UnsupportedArtistError(
                f"Line2D {axis_name} destination overflows {spec.dtype} storage"
            )
        return encoded

    @classmethod
    def _sequence_with_values(cls, template, values):
        if type(template) is list:
            return [
                cls._sequence_with_values(source, value)
                for source, value in zip(template, values)
            ]
        if type(template) is tuple:
            return tuple(
                cls._sequence_with_values(source, value)
                for source, value in zip(template, values)
            )
        if isinstance(template, np.ndarray):
            return np.asarray(values, dtype=template.dtype).reshape(template.shape)
        return values

    @classmethod
    def _materialize_raw_axis(
        cls,
        spec: _Line2DAxisSpec,
        representable_axis: np.ndarray,
        eligible: np.ndarray,
        *,
        promote: bool = False,
    ):
        raw = spec.raw
        if spec.size == 1:
            indices = np.array([0], dtype=int)
            values = np.asarray([representable_axis[np.flatnonzero(eligible)[0]]])
        else:
            indices = np.flatnonzero(eligible)
            values = representable_axis[eligible]
        if promote:
            if spec.dtype.kind == "f":
                promoted_dtype = np.dtype(float)
            else:
                integers = np.asarray(spec.data).reshape(-1)
                exactly_float64 = all(
                    np.isfinite(float(value))
                    and int(float(value)) == int(value)
                    for value in integers
                )
                if np.ma.isMaskedArray(raw):
                    fill = raw.fill_value
                    try:
                        exactly_float64 = bool(
                            exactly_float64
                            and np.isfinite(float(fill))
                            and int(float(fill)) == int(fill)
                        )
                    except (OverflowError, TypeError, ValueError):
                        exactly_float64 = False
                promoted_dtype = np.dtype(float if exactly_float64 else object)
            promoted_data = np.asarray(spec.data, dtype=promoted_dtype).copy()
            promoted_data.reshape(-1)[indices] = values
            if np.ma.isMaskedArray(raw):
                mask = (
                    np.ma.nomask
                    if raw.mask is np.ma.nomask
                    else np.array(raw.mask, dtype=bool, copy=True)
                )
                return np.ma.array(
                    promoted_data,
                    mask=mask,
                    fill_value=raw.fill_value,
                    hard_mask=bool(raw.hardmask),
                    dtype=promoted_dtype,
                )
            return promoted_data
        if (
            type(raw) is np.ndarray
            and spec.dtype == np.dtype(float)
            and spec.size == len(representable_axis)
            and np.all(eligible)
        ):
            return representable_axis.reshape(spec.shape)
        if np.ma.isMaskedArray(raw):
            result = deepcopy(raw)
            result.data.reshape(-1)[indices] = values
            return result
        if type(raw) is np.ndarray:
            result = raw.copy()
            result.reshape(-1)[indices] = values
            return result
        result = np.asarray(raw, dtype=object).copy()
        result.reshape(-1)[indices] = values
        nested = result.reshape(spec.shape).tolist()
        return cls._sequence_with_values(raw, nested)

    def _prepare_raw_write(
        self, points, *, materialize: bool
    ) -> _Line2DRawWrite:
        processed = np.asarray(self.target.get_xydata(), dtype=float)
        destination = np.asarray(points, dtype=float)
        if (
            processed.ndim != 2
            or processed.shape[1] != 2
            or destination.shape != processed.shape
        ):
            raise UnsupportedArtistError(
                "Line2D destination must match its processed Nx2 geometry"
            )
        length = len(processed)
        xspec = self._raw_axis_spec(
            self.target.get_xdata(orig=True), "x", length
        )
        yspec = self._raw_axis_spec(
            self.target.get_ydata(orig=True), "y", length
        )
        eligible = np.all(np.isfinite(processed), axis=1)
        if not np.any(eligible):
            raise UnsupportedArtistError(
                "Line2D has no jointly finite coordinate rows to transform"
            )
        all_eligible = bool(np.all(eligible))
        desired_visible = destination if all_eligible else destination[eligible]
        if not np.all(np.isfinite(desired_visible)):
            raise UnsupportedArtistError(
                "Line2D destination must stay finite for every visible vertex"
            )

        encoded_axes = []
        for axis, (axis_name, spec) in enumerate(
            (("x", xspec), ("y", yspec))
        ):
            encoded = self._cast_axis_destination(
                spec, desired_visible[:, axis], axis_name
            )
            numeric = np.asarray(encoded, dtype=float)
            if spec.size == 1:
                first = numeric[0]
                if not np.all(numeric == first):
                    raise UnsupportedArtistError(
                        f"Line2D length-one {axis_name} broadcast storage cannot "
                        "represent different transformed vertex coordinates"
                    )
                original = float(np.asarray(spec.data).reshape(-1)[0])
                if np.any(~eligible) and first != original:
                    raise UnsupportedArtistError(
                        f"Line2D length-one {axis_name} broadcast storage is shared "
                        "with non-finite rows and cannot change their raw payload"
                    )
                numeric = np.full(np.count_nonzero(eligible), first, dtype=float)
            encoded_axes.append(numeric)

        def stores_float64_exactly(spec: _Line2DAxisSpec) -> bool:
            return bool(
                not spec.fixed_dtype
                or spec.dtype.kind == "O"
                or (spec.dtype.kind == "f" and spec.dtype.itemsize >= 8)
            )

        exact_storage = bool(
            stores_float64_exactly(xspec)
            and stores_float64_exactly(yspec)
        )
        exact_destination = bool(
            exact_storage
            and xspec.size == length
            and yspec.size == length
            and all_eligible
        )
        promotion = [False, False]
        if exact_destination:
            representable = destination
        else:
            representable = processed.copy()
            representable[eligible, 0] = encoded_axes[0]
            representable[eligible, 1] = encoded_axes[1]
            if not exact_storage:
                try:
                    desired_display = np.asarray(
                        self.target.get_transform().transform(desired_visible),
                        dtype=float,
                    )
                    actual_display = np.asarray(
                        self.target.get_transform().transform(
                            representable[eligible]
                        ),
                        dtype=float,
                    )
                except (
                    AttributeError,
                    TypeError,
                    ValueError,
                    RuntimeError,
                    np.linalg.LinAlgError,
                ) as error:
                    raise UnsupportedArtistError(
                        "Line2D encoded coordinates cannot be checked in display "
                        "space"
                    ) from error
                if (
                    desired_display.shape != actual_display.shape
                    or not np.all(np.isfinite(desired_display))
                    or not np.all(np.isfinite(actual_display))
                ):
                    error_px = float("inf")
                else:
                    error_px = float(
                        np.max(np.abs(desired_display - actual_display), initial=0.0)
                    )
                if error_px > 0.25:
                    specs = (xspec, yspec)
                    for axis, spec in enumerate(specs):
                        promotion[axis] = bool(
                            not stores_float64_exactly(spec)
                            and not np.array_equal(
                                encoded_axes[axis], desired_visible[:, axis]
                            )
                        )
                        if promotion[axis]:
                            representable[eligible, axis] = desired_visible[:, axis]
                    if not any(promotion):
                        raise UnsupportedArtistError(
                            "Line2D numeric storage cannot represent its display "
                            "destination within 0.25 px and cannot be promoted "
                            "losslessly"
                        )

        if not materialize:
            return _Line2DRawWrite(representable)
        return _Line2DRawWrite(
            representable,
            self._materialize_raw_axis(
                xspec, representable[:, 0], eligible, promote=promotion[0]
            ),
            self._materialize_raw_axis(
                yspec, representable[:, 1], eligible, promote=promotion[1]
            ),
        )

    def _line_stroke_is_configured(self) -> bool:
        linestyle = self.target.get_linestyle()
        if linestyle in (None, "", " ", "none", "None"):
            return False
        colors = []
        try:
            colors.append(
                mpl.colors.to_rgba(
                    self.target.get_color(), self.target.get_alpha()
                )
            )
        except (TypeError, ValueError):
            pass
        gap_color = getattr(self.target, "get_gapcolor", lambda: None)()
        if self.target.is_dashed() and gap_color is not None:
            try:
                colors.append(
                    mpl.colors.to_rgba(gap_color, self.target.get_alpha())
                )
            except (TypeError, ValueError):
                pass
        if not self.colors_are_visible(colors):
            return False
        return self.path_has_drawable_segment(self.target.get_path())

    def _line_paint_is_visible(self) -> bool:
        if not self._line_stroke_is_configured():
            return False
        try:
            linewidth = float(self.target.get_linewidth())
        except (TypeError, ValueError):
            return False
        return bool(np.isfinite(linewidth) and linewidth > 0.0)

    def _line_rigid_rotation_supported(self, *, strict: bool = False) -> bool:
        if not self._line_stroke_is_configured():
            return True
        try:
            linewidth = float(self.target.get_linewidth())
            rendered_linewidth = self.points_to_pixels(linewidth)
        except (OverflowError, TypeError, ValueError):
            linewidth = float("nan")
            rendered_linewidth = float("nan")
        if (
            not np.isfinite(linewidth)
            or linewidth < 0.0
            or not np.isfinite(rendered_linewidth)
            or rendered_linewidth < 0.0
        ):
            if strict:
                raise UnsupportedArtistError(
                    "Line2D linewidth must be finite and non-negative"
                )
            return False
        return True

    def _finite_marker_dimensions(self) -> tuple[float, float] | None:
        try:
            values = np.asarray(
                (
                    float(self.target.get_markersize()),
                    float(self.target.get_markeredgewidth()),
                ),
                dtype=float,
            )
        except (OverflowError, TypeError, ValueError):
            return None
        if not np.all(np.isfinite(values)) or np.any(values < 0.0):
            return None
        return float(values[0]), float(values[1])

    def _marker_painted_paths(self) -> list[tuple[Path, Transform]]:
        marker = self.target._marker
        if not marker:
            return []
        dimensions = self._finite_marker_dimensions()
        if dimensions is None:
            return []
        _markersize, markeredgewidth = dimensions
        alpha = self.target.get_alpha()
        edge_visible = False
        if markeredgewidth > 0.0:
            try:
                edge_visible = self.colors_are_visible(
                    mpl.colors.to_rgba(self.target.get_markeredgecolor(), alpha)
                )
            except (TypeError, ValueError):
                pass

        components = []
        candidates = [
            (
                marker.get_path(),
                marker.get_transform(),
                self.target.get_markerfacecolor,
            )
        ]
        alt_path = marker.get_alt_path()
        if alt_path is not None:
            candidates.append(
                (
                    alt_path,
                    marker.get_alt_transform(),
                    self.target.get_markerfacecoloralt,
                )
            )
        for path, transform, face_getter in candidates:
            try:
                face_visible = self.colors_are_visible(
                    mpl.colors.to_rgba(face_getter(), alpha)
                )
            except (TypeError, ValueError):
                face_visible = False
            fills = bool(
                marker.is_filled()
                and face_visible
                and self.path_has_fill_area(path, transform)
            )
            strokes = bool(
                edge_visible and self.path_has_drawable_segment(path, transform)
            )
            if fills or strokes:
                components.append((path, transform))
        return components

    def _markevery_schema_is_valid(self) -> bool:
        markevery = self.target.get_markevery()
        if markevery is None:
            return True
        if isinstance(markevery, Integral):
            return int(markevery) != 0
        if isinstance(markevery, Real):
            try:
                value = float(markevery)
            except (OverflowError, TypeError, ValueError):
                return False
            return bool(np.isfinite(value) and value != 0.0)
        if isinstance(markevery, tuple):
            if len(markevery) != 2:
                return False
            start, step = markevery
            if isinstance(step, Integral):
                return isinstance(start, Integral) and int(step) != 0
            if isinstance(step, Real):
                try:
                    return bool(
                        isinstance(start, Real)
                        and np.isfinite(float(start))
                        and np.isfinite(float(step))
                        and float(step) != 0.0
                    )
                except (OverflowError, TypeError, ValueError):
                    return False
            return False
        if isinstance(markevery, slice):
            return bool(
                all(
                    value is None or isinstance(value, Integral)
                    for value in (
                        markevery.start,
                        markevery.stop,
                        markevery.step,
                    )
                )
                and (markevery.step is None or int(markevery.step) != 0)
            )
        try:
            indices = np.asarray(markevery)
        except (TypeError, ValueError):
            return False
        if indices.ndim != 1:
            return False
        if indices.size == 0:
            return True
        if np.issubdtype(indices.dtype, np.bool_):
            return len(indices) == len(self.target.get_xydata())
        if not np.issubdtype(indices.dtype, np.integer):
            return False
        length = len(self.target.get_xydata())
        return bool(np.all(indices >= -length) and np.all(indices < length))

    def _markevery_is_definitely_empty(self) -> bool:
        markevery = self.target.get_markevery()
        length = len(self.target.get_xydata())
        try:
            if isinstance(markevery, Integral):
                return len(range(length)[slice(0, None, int(markevery))]) == 0
            if (
                isinstance(markevery, tuple)
                and len(markevery) == 2
                and isinstance(markevery[0], Integral)
                and isinstance(markevery[1], Integral)
                and int(markevery[1]) != 0
            ):
                return (
                    len(
                        range(length)[
                            slice(int(markevery[0]), None, int(markevery[1]))
                        ]
                    )
                    == 0
                )
            if isinstance(markevery, slice) and self._markevery_schema_is_valid():
                return len(range(length)[markevery]) == 0
        except (OverflowError, TypeError, ValueError):
            return False
        if isinstance(markevery, (list, np.ndarray)):
            try:
                indices = np.asarray(markevery)
            except (TypeError, ValueError):
                return False
            return bool(
                indices.size == 0
                or (
                    indices.ndim == 1
                    and np.issubdtype(indices.dtype, np.bool_)
                    and not np.any(indices)
                )
            )
        return False

    def _marker_paint_is_visible(
        self, *, strict: bool = False, resolve_positions: bool = True
    ) -> bool:
        marker = self.target._marker
        dimensions = self._finite_marker_dimensions()
        if (
            not marker
            or dimensions is None
            or dimensions[0] <= 0.0
        ):
            return False
        if not self._markevery_schema_is_valid():
            if strict:
                raise UnsupportedArtistError(
                    "Line2D markevery is not valid for rigid rotation"
                )
            return False
        if self._markevery_is_definitely_empty():
            return False
        if resolve_positions and not len(
            self._marker_display_positions(strict=strict)
        ):
            return False
        return bool(self._marker_painted_paths())

    def _marker_rigid_rotation_supported(
        self, *, strict: bool = False, resolve_positions: bool = True
    ) -> bool:
        """Whether marker paint remains identical under arbitrary rotation."""

        marker = self.target._marker
        if marker:
            dimensions = self._finite_marker_dimensions()
            if dimensions is None:
                if strict:
                    raise UnsupportedArtistError(
                        "Line2D marker dimensions must be finite and non-negative"
                    )
                return False
            try:
                rendered_marker_edgewidth = self.points_to_pixels(dimensions[1])
            except (OverflowError, TypeError, ValueError):
                rendered_marker_edgewidth = float("nan")
            if (
                not np.isfinite(rendered_marker_edgewidth)
                or rendered_marker_edgewidth < 0.0
            ):
                if strict:
                    raise UnsupportedArtistError(
                        "Line2D marker dimensions must be finite and non-negative"
                    )
                return False
        if not self._markevery_schema_is_valid():
            if strict:
                raise UnsupportedArtistError(
                    "Line2D markevery is not valid for rigid rotation"
                )
            return False
        if not self._marker_paint_is_visible(
            strict=strict, resolve_positions=resolve_positions
        ):
            return True
        path = marker.get_path()
        circle = Path.unit_circle()
        transform = marker.get_transform()
        try:
            matrix = np.asarray(transform.get_affine().get_matrix(), dtype=float)
            marker_scale_pixels = self.points_to_pixels(
                float(self.target.get_markersize())
            )
            singular_values = np.linalg.svd(
                matrix[:2, :2], compute_uv=False
            )
            rendered_anisotropy = marker_scale_pixels * abs(
                float(singular_values[0] - singular_values[1])
            )
            rendered_offset_sweep = (
                2.0
                * marker_scale_pixels
                * float(np.linalg.norm(matrix[:2, 2]))
            )
        except (
            AttributeError,
            IndexError,
            OverflowError,
            TypeError,
            ValueError,
            RuntimeError,
            np.linalg.LinAlgError,
        ):
            return False
        centered_circle = bool(
            path.vertices.shape == circle.vertices.shape
            and np.array_equal(path.vertices, circle.vertices)
            and (
                (path.codes is None and circle.codes is None)
                or (
                    path.codes is not None
                    and circle.codes is not None
                    and np.array_equal(path.codes, circle.codes)
                )
            )
            and matrix.shape == (3, 3)
            and np.all(np.isfinite(matrix))
            and np.all(np.isfinite(singular_values))
            and np.isfinite(marker_scale_pixels)
            and marker_scale_pixels > 0.0
            and singular_values[-1] > np.finfo(float).eps
            and np.isfinite(rendered_anisotropy)
            and np.isfinite(rendered_offset_sweep)
            and rendered_anisotropy + rendered_offset_sweep <= 0.25
        )
        supported = bool(
            marker.get_fillstyle() in {"full", "none"}
            and marker.get_alt_path() is None
            and centered_circle
        )
        if supported and resolve_positions:
            marker_bounds = self._marker_selection_points(strict=strict)
            supported = bool(
                len(marker_bounds) and np.all(np.isfinite(marker_bounds))
            )
        return supported

    def _marker_is_pixel(self) -> bool:
        marker_value = self.target._marker.get_marker()
        return bool(isinstance(marker_value, str) and marker_value == ",")

    def operation_support(
        self, operation: TransformOperation | str
    ) -> OperationSupport:
        operation = TransformOperation.coerce(operation)
        if operation in {
            TransformOperation.TRANSLATE,
            TransformOperation.RIGID_ROTATE,
        }:
            raw_reason = self._raw_metadata_reason(self.target)
            if raw_reason is not None:
                return OperationSupport.denied(operation, raw_reason)
        if operation is not TransformOperation.SCALE_APPEARANCE:
            return super().operation_support(operation)
        if not self.capabilities.can_select or not self.capabilities.can_serialize:
            return OperationSupport.denied(
                operation,
                "Line2D has no finite geometry for an atomic appearance edit",
            )
        if legend_owner_for_artist(self.target) is not None:
            return OperationSupport.denied(
                operation,
                "Legend-managed Line2D appearance is owned by its Legend layout",
            )
        if active_layout_owner_for_artist(self.target) is not None:
            return OperationSupport.denied(
                operation,
                "Line2D participates in active layout and cannot scale independently",
            )
        if self.target.get_agg_filter() is not None or self.target.get_path_effects():
            return OperationSupport.denied(
                operation,
                "Line2D filters or path effects have independent pixel dimensions",
            )
        if self.target.get_sketch_params() is not None:
            return OperationSupport.denied(
                operation,
                "Line2D sketch effects have independent display-space dimensions",
            )
        marker_dimensions = self._finite_marker_dimensions()
        if (
            self._marker_is_pixel()
            and marker_dimensions is not None
            and marker_dimensions[0] > 0.0
        ):
            return OperationSupport.denied(
                operation,
                "The pixel marker has a renderer-fixed one-pixel appearance",
            )
        try:
            values = np.asarray(
                (
                    self.target.get_linewidth(),
                    self.target.get_markersize(),
                    self.target.get_markeredgewidth(),
                ),
                dtype=float,
            )
            rendered_values = np.asarray(
                [self.points_to_pixels(value) for value in values], dtype=float
            )
        except (OverflowError, TypeError, ValueError):
            values = np.asarray((np.nan,), dtype=float)
            rendered_values = np.asarray((np.nan,), dtype=float)
        if (
            not np.all(np.isfinite(values))
            or np.any(values < 0.0)
            or not np.all(np.isfinite(rendered_values))
            or np.any(rendered_values < 0.0)
        ):
            return OperationSupport.denied(
                operation,
                "Line2D stroke and marker dimensions must be finite and non-negative",
            )
        if not self.target.get_visible() or not (
            self._line_paint_is_visible() or self._marker_paint_is_visible()
        ):
            return OperationSupport.denied(
                operation,
                "Line2D has no visible stroke or marker paint to scale",
            )
        if not self.has_visible_selection_bounds():
            return OperationSupport.denied(
                operation,
                "Line2D has no visible geometry inside its active clip region",
            )
        return OperationSupport.allowed(
            operation,
            constraints=("positive_uniform_factor",),
            preview_strategy="redraw",
        )

    def appearance_state(self):
        return Line2DAppearanceState(
            float(self.target.get_linewidth()),
            float(self.target.get_markersize()),
            float(self.target.get_markeredgewidth()),
        )

    def scaled_appearance_state(self, factor: float):
        state = self.appearance_state()
        values = self.scale_nonnegative_dimensions(
            (state.linewidth, state.markersize, state.markeredgewidth),
            factor,
            label="Line2D appearance",
        )
        return Line2DAppearanceState(*values)

    @staticmethod
    def validate_appearance_state(state) -> Line2DAppearanceState:
        if not isinstance(state, Line2DAppearanceState):
            raise TypeError("Line2D appearance plan has an invalid state type")
        values = np.asarray(
            (state.linewidth, state.markersize, state.markeredgewidth), dtype=float
        )
        if not np.all(np.isfinite(values)) or np.any(values < 0.0):
            raise ValueError(
                "Line2D stroke and marker dimensions must be finite and non-negative"
            )
        return state

    def _apply_appearance_state(self, state) -> None:
        state = self.validate_appearance_state(state)
        self.target.set_linewidth(state.linewidth)
        self.target.set_markersize(state.markersize)
        self.target.set_markeredgewidth(state.markeredgewidth)

    def get_transform(self) -> Transform:
        return IdentityTransform()

    def _display_xydata(self) -> np.ndarray:
        points = self.point_array(self.target.get_xydata())
        if not len(points):
            return points
        return self.point_array(self.target.get_transform().transform(points))

    @staticmethod
    def _float_markevery_parameters(markevery) -> tuple[float, float] | None:
        """Return Matplotlib's axes-relative ``markevery`` parameters."""

        if isinstance(markevery, Real) and not isinstance(markevery, Integral):
            return 0.0, float(markevery)
        if (
            isinstance(markevery, tuple)
            and len(markevery) == 2
            and isinstance(markevery[0], Real)
            and isinstance(markevery[1], Real)
            and not isinstance(markevery[1], Integral)
        ):
            return float(markevery[0]), float(markevery[1])
        return None

    def _stable_float_marker_display_positions(
        self, points: np.ndarray, markevery
    ) -> np.ndarray | None:
        """Resolve float ``markevery`` and reject hardware-sensitive ties.

        Matplotlib selects the vertex nearest each axes-relative path distance.
        An exact or near tie can flip between vertices after a mathematically
        rigid transform because ``hypot`` and cumulative sums differ by a few
        ulps across CPUs.  Reproduce Matplotlib's resolver in one pass and fail
        closed when two materially different vertices are within the numerical
        error envelope.  This runs only during strict transform preflight; the
        ordinary selection and pointer paths remain unchanged.
        """

        parameters = self._float_markevery_parameters(markevery)
        if parameters is None:
            return None
        start, step = parameters
        axes = self.target.axes
        if axes is None:
            raise ValueError(
                "float markevery requires the Line2D to have an Axes parent"
            )

        finite = np.isfinite(points).all(axis=1)
        vertices = points[finite]
        delta_vectors = np.empty((len(vertices), 2), dtype=float)
        delta_vectors[0, :] = 0.0
        delta_vectors[1:, :] = vertices[1:, :] - vertices[:-1, :]
        cumulative = np.hypot(*delta_vectors.T).cumsum()
        (x0, y0), (x1, y1) = axes.transAxes.transform([[0, 0], [1, 1]])
        axes_diagonal = np.hypot(x1 - x0, y1 - y0)
        start_distance = start * axes_diagonal
        step_distance = step * axes_diagonal
        coordinate_scale = max(
            1.0,
            float(np.max(np.abs(vertices), initial=0.0)),
            float(np.max(np.abs(cumulative), initial=0.0)),
            abs(float(start_distance)),
            abs(float(step_distance)),
        )
        # Subtraction, hypot, and a cumulative sum contribute error.  The
        # small linear factor covers cross-runner ulp variation.  Pathological
        # huge-coordinate inputs may conservatively reject instead of risking
        # a preview/commit mismatch.
        ambiguity_tolerance = (
            16.0
            * np.finfo(float).eps
            * max(1, len(vertices))
            * coordinate_scale
        )
        if (
            not np.all(np.isfinite(cumulative))
            or not np.isfinite(axes_diagonal)
            or not np.isfinite(start_distance)
            or not np.isfinite(step_distance)
            or not np.isfinite(ambiguity_tolerance)
            or abs(step_distance) <= ambiguity_tolerance
        ):
            raise UnsupportedArtistError(
                "Line2D markevery may select different marker vertices after "
                "rigid rotation because its distance grid is numerically "
                "ambiguous"
            )
        marker_distances = np.arange(
            start_distance,
            cumulative[-1],
            step_distance,
        )

        # ``cumulative`` is monotonic, so the nearest vertex must be one of
        # the two insertion neighbours.  Matplotlib currently materializes an
        # M-by-N distance matrix here; searchsorted preserves its lower-index
        # tie rule while keeping preflight memory linear in M + N.
        def resolve_nearest(distances):
            right = np.searchsorted(cumulative, distances, side="left")
            left = np.clip(right - 1, 0, len(vertices) - 1)
            right = np.clip(right, 0, len(vertices) - 1)
            left_error = np.abs(cumulative[left] - distances)
            right_error = np.abs(cumulative[right] - distances)
            nearest = np.where(left_error <= right_error, left, right)
            # Zero-length segments have duplicate cumulative distances.
            # Argmin returns the first such vertex, so canonicalize likewise.
            nearest = np.searchsorted(
                cumulative, cumulative[nearest], side="left"
            )
            return nearest, left, right, left_error, right_error

        (
            indices,
            left_indices,
            right_indices,
            left_distances,
            right_distances,
        ) = resolve_nearest(marker_distances)

        if len(indices) and len(vertices) > 1:
            distance_gaps = np.abs(left_distances - right_distances)
            materially_distinct = np.any(
                np.abs(vertices[left_indices] - vertices[right_indices]) > 0.25,
                axis=1,
            )
            if np.any(
                (left_indices != right_indices)
                & (distance_gaps <= ambiguity_tolerance)
                & materially_distinct
            ):
                raise UnsupportedArtistError(
                    "Line2D markevery may select different marker vertices "
                    "after rigid rotation because a marker distance is "
                    "numerically ambiguous"
                )

        def marker_is_materially_new(index, existing) -> bool:
            if not len(existing):
                return True
            return not bool(
                np.any(
                    np.all(
                        np.abs(vertices[existing] - vertices[index]) <= 0.25,
                        axis=1,
                    )
                )
            )

        # ``arange`` excludes its stop.  If the path endpoint is within the
        # error envelope of the last included or first excluded grid point,
        # different CPUs can also disagree on the marker count, even when no
        # nearest-vertex midpoint is ambiguous.
        if len(marker_distances):
            last_distance = marker_distances[-1]
            if (
                abs(cumulative[-1] - last_distance) <= ambiguity_tolerance
                and marker_is_materially_new(indices[-1], indices[:-1])
            ):
                raise UnsupportedArtistError(
                    "Line2D markevery may select different marker vertices "
                    "after rigid rotation because its sequence endpoint is "
                    "numerically ambiguous"
                )
            next_distance = last_distance + step_distance
        else:
            next_distance = start_distance
        if not np.isfinite(next_distance):
            raise UnsupportedArtistError(
                "Line2D markevery may select different marker vertices after "
                "rigid rotation because its distance grid is numerically "
                "ambiguous"
            )
        if abs(cumulative[-1] - next_distance) <= ambiguity_tolerance:
            next_index = resolve_nearest(np.asarray([next_distance]))[0][0]
            if marker_is_materially_new(next_index, indices):
                raise UnsupportedArtistError(
                    "Line2D markevery may select different marker vertices "
                    "after rigid rotation because its sequence endpoint is "
                    "numerically ambiguous"
                )

        return vertices[np.unique(indices)]

    def _marker_display_positions(
        self, points=None, *, strict: bool = False
    ) -> np.ndarray:
        if points is None:
            points = self._display_xydata()
        else:
            points = self.point_array(points)
        if not len(points):
            return np.empty((0, 2), dtype=float)
        markevery = self.target.get_markevery()
        if markevery is None:
            return self.finite_points(points)
        try:
            resolved = (
                self._stable_float_marker_display_positions(points, markevery)
                if strict
                else None
            )
            if resolved is None:
                points = _mark_every_path(
                    markevery,
                    Path(points),
                    IdentityTransform(),
                    self.target.axes,
                ).vertices
            else:
                points = resolved
        except UnsupportedArtistError:
            raise
        except (
            IndexError,
            OverflowError,
            TypeError,
            ValueError,
            ZeroDivisionError,
        ) as error:
            if strict:
                raise UnsupportedArtistError(
                    "Line2D markevery cannot be resolved exactly for rigid rotation"
                ) from error
            # Selection stays conservative for invalid or renderer-dependent
            # inputs, but rigid-rotation capability uses the strict path above.
        return self.finite_points(points)

    def _marker_selection_points(
        self,
        positions=None,
        *,
        strict: bool = False,
        positions_are_resolved: bool = False,
    ) -> np.ndarray:
        marker = self.target._marker
        dimensions = self._finite_marker_dimensions()
        if not marker or dimensions is None or dimensions[0] <= 0.0:
            return np.empty((0, 2), dtype=float)
        markersize, markeredgewidth = dimensions
        alpha = self.target.get_alpha()
        painted_paths = self._marker_painted_paths()
        if not painted_paths:
            return np.empty((0, 2), dtype=float)

        if positions_are_resolved:
            positions = self.finite_points(positions)
        else:
            positions = self._marker_display_positions(positions, strict=strict)
        if not len(positions):
            return np.empty((0, 2), dtype=float)
        is_pixel = self._marker_is_pixel()
        try:
            rendered_markersize = self.points_to_pixels(markersize)
            rendered_markeredgewidth = self.points_to_pixels(markeredgewidth)
        except (OverflowError, TypeError, ValueError):
            return np.empty((0, 2), dtype=float)
        if not is_pixel and (
            not np.isfinite(rendered_markersize)
            or not np.isfinite(rendered_markeredgewidth)
        ):
            return np.empty((0, 2), dtype=float)
        relative = []
        for path, transform in painted_paths:
            transform = transform.frozen()
            if not is_pixel:
                transform.scale(rendered_markersize)
            relative.append(
                np.asarray(
                    path.get_extents(transform).get_points(), dtype=float
                )
            )
        relative = self.bounds_points(np.concatenate(relative))
        if not len(relative):
            return np.empty((0, 2), dtype=float)
        edge_padding = 0.0
        try:
            edge = mpl.colors.to_rgba(self.target.get_markeredgecolor(), alpha)
        except (TypeError, ValueError):
            edge = (0.0, 0.0, 0.0, 0.0)
        if not is_pixel and self.colors_are_visible(edge):
            edge_padding = rendered_markeredgewidth / 2
        relative = self.bounds_points(relative, padding=edge_padding)
        return np.array(
            [
                [
                    np.min(positions[:, 0]) + relative[0, 0],
                    np.min(positions[:, 1]) + relative[0, 1],
                ],
                [
                    np.max(positions[:, 0]) + relative[1, 0],
                    np.max(positions[:, 1]) + relative[1, 1],
                ],
            ],
            dtype=float,
        )

    def _line_selection_points(self, display_points) -> np.ndarray:
        display_points = self.point_array(display_points)
        try:
            linewidth = float(self.target.get_linewidth())
        except (TypeError, ValueError):
            linewidth = float("nan")
        if (
            self._line_stroke_is_configured()
            and np.isfinite(linewidth)
            and linewidth > 0.0
        ):
            finite = np.all(np.isfinite(display_points), axis=1)
            used = np.zeros(len(display_points), dtype=bool)
            if len(display_points) >= 2:
                drawable = (
                    finite[:-1]
                    & finite[1:]
                    & np.any(display_points[1:] != display_points[:-1], axis=1)
                )
                used[:-1] |= drawable
                used[1:] |= drawable
            line = self.bounds_points(
                display_points[used],
                padding=self.points_to_pixels(linewidth) / 2,
            )
            if len(line):
                return line
        return np.empty((0, 2), dtype=float)

    def selection_points(self) -> np.ndarray:
        preview = self._preview_points("_pylustrator_preview_selection_points")
        if preview is not None:
            return preview
        visible_groups = []
        display_points = self._display_xydata()
        if not len(display_points):
            return np.empty((0, 2), dtype=float)
        line = self._line_selection_points(display_points)
        if len(line):
            visible_groups.append(line)
        markers = self._marker_selection_points(display_points)
        if len(markers):
            visible_groups.append(markers)
        if visible_groups:
            return self.bounds_points(np.concatenate(visible_groups))
        return super().selection_points()

    def preview_rigid_rotation_control_points(
        self, matrix, *, control_points=None
    ) -> np.ndarray:
        """Rotate only jointly finite Line2D vertices.

        A row containing one masked/non-finite coordinate is a path break, not
        a partially writable point.  Keeping that display row untouched lets
        the raw codec preserve both hidden coordinate payloads exactly.
        """

        if control_points is None:
            control_points = self.control_points()
        result = self.point_array(control_points).copy()
        eligible = np.all(np.isfinite(result), axis=1)
        if np.any(eligible):
            result[eligible] = self._transform_points(matrix, result[eligible])
        return result

    def preview_rigid_rotation_selection_points(
        self,
        matrix,
        *,
        control_points,
        selection_points,
        planned_control_points,
        planned_native_control_points,
        rotation_value: float | None,
    ) -> np.ndarray:
        """Recompute Line2D paint from the exact destination marker subset."""

        destination_markers = np.empty((0, 2), dtype=float)
        if self._marker_paint_is_visible(
            strict=True, resolve_positions=False
        ):
            source_markers = self._marker_display_positions(
                control_points, strict=True
            )
            destination_markers = self._marker_display_positions(
                planned_control_points, strict=True
            )
            expected_markers = self._transform_points(matrix, source_markers)
            if (
                destination_markers.shape != expected_markers.shape
                or not np.allclose(
                    destination_markers,
                    expected_markers,
                    atol=0.25,
                    rtol=0.0,
                )
            ):
                raise UnsupportedArtistError(
                    "Line2D markevery would select different marker vertices "
                    "after rigid rotation"
                )
        visible_groups = []
        line = self._line_selection_points(planned_control_points)
        if len(line):
            visible_groups.append(line)
        markers = self._marker_selection_points(
            destination_markers,
            strict=True,
            positions_are_resolved=True,
        )
        if len(markers):
            visible_groups.append(markers)
        if visible_groups:
            return self.bounds_points(np.concatenate(visible_groups))
        return np.empty((0, 2), dtype=float)

    def native_control_points(self) -> np.ndarray:
        return self.point_array(self.target.get_xydata()).copy()

    def control_points(self) -> np.ndarray:
        preview = self._preview_points("_pylustrator_preview_positions")
        if preview is not None:
            return self.point_array(preview).copy()
        return self.point_array(self.native_to_display(self.native_control_points()))

    def native_to_display(self, points) -> np.ndarray:
        points = self.point_array(points)
        if not len(points):
            return points
        transform = self.target.get_transform()
        return self.point_array(transform.transform(points))

    def display_to_native(self, points) -> np.ndarray:
        points = self.point_array(points)
        if not len(points):
            return points
        transform = self.target.get_transform().inverted()
        return self.point_array(transform.transform(points))

    def local_control_points(self) -> np.ndarray:
        return self.point_array(self.display_to_native(self.control_points()))

    def validate_native_control_points(self, points) -> None:
        self._prepare_raw_write(points, materialize=False)

    def canonicalize_native_control_points(self, points) -> np.ndarray:
        return self._prepare_raw_write(points, materialize=False).native_points

    @staticmethod
    def _update_fingerprint_array(hasher, values) -> None:
        array = np.asarray(values)
        hasher.update(array.dtype.str.encode("ascii", errors="backslashreplace"))
        hasher.update(repr(tuple(int(value) for value in array.shape)).encode())
        if array.dtype.hasobject:
            for value in array.flat:
                encoded = repr(value).encode("utf-8", errors="backslashreplace")
                hasher.update(len(encoded).to_bytes(8, "little"))
                hasher.update(encoded)
            return
        if array.flags.c_contiguous:
            hasher.update(memoryview(array).cast("B"))
        else:
            hasher.update(array.tobytes(order="C"))

    @classmethod
    def _update_raw_fingerprint(cls, hasher, raw, axis_name: str) -> None:
        hasher.update(axis_name.encode("ascii"))
        container = f"{type(raw).__module__}.{type(raw).__qualname__}"
        hasher.update(container.encode("utf-8", errors="backslashreplace"))
        if np.ma.isMaskedArray(raw):
            cls._update_fingerprint_array(hasher, raw.data)
            if raw.mask is np.ma.nomask:
                hasher.update(b"nomask")
            else:
                hasher.update(b"mask")
                cls._update_fingerprint_array(hasher, raw.mask)
            fill = repr(raw.fill_value).encode(
                "utf-8", errors="backslashreplace"
            )
            hasher.update(len(fill).to_bytes(8, "little"))
            hasher.update(fill)
            hasher.update(b"hard" if raw.hardmask else b"soft")
            return
        cls._update_fingerprint_array(hasher, raw)

    @classmethod
    def _update_markevery_fingerprint(cls, hasher, markevery) -> None:
        """Hash valid markevery state by semantics rather than object identity."""

        if markevery is None:
            token = ("none",)
        elif isinstance(markevery, Integral):
            token = ("int", int(markevery))
        elif isinstance(markevery, Real):
            token = ("float", float(markevery))
        elif isinstance(markevery, tuple):
            start, step = markevery
            if isinstance(step, Integral):
                token = ("int_tuple", int(start), int(step))
            else:
                token = ("float_tuple", float(start), float(step))
        elif isinstance(markevery, slice):
            token = (
                "slice",
                None if markevery.start is None else int(markevery.start),
                None if markevery.stop is None else int(markevery.stop),
                None if markevery.step is None else int(markevery.step),
            )
        else:
            values = np.asarray(markevery)
            hasher.update(b"markevery-array")
            if values.size == 0:
                hasher.update(b"empty")
                return
            if np.issubdtype(values.dtype, np.bool_):
                hasher.update(b"bool")
                normalized = np.asarray(values, dtype=np.bool_)
            else:
                hasher.update(b"int")
                normalized = np.asarray(values, dtype=np.int64)
            cls._update_fingerprint_array(hasher, normalized)
            return
        encoded = repr(token).encode("ascii", errors="backslashreplace")
        hasher.update(len(encoded).to_bytes(8, "little"))
        hasher.update(encoded)

    def _update_line_context_fingerprint(self, hasher) -> None:
        """Hash display state used by Line2D float markevery and rotation."""

        transform = self.target.get_transform()
        transform_token = (
            type(transform).__module__,
            type(transform).__qualname__,
            bool(getattr(transform, "is_affine", False)),
            bool(getattr(transform, "has_inverse", True)),
        )
        hasher.update(repr(transform_token).encode("utf-8"))
        self._update_fingerprint_array(
            hasher, transform.get_affine().get_matrix()
        )

        axes = self.target.axes
        if axes is None:
            hasher.update(b"no-axes")
        else:
            hasher.update(id(axes).to_bytes(8, "little", signed=False))
            self._update_fingerprint_array(
                hasher, axes.transAxes.get_affine().get_matrix()
            )
        figure = self.figure
        self._update_fingerprint_array(
            hasher,
            np.asarray(
                (float(figure.dpi), self.points_to_pixels(1.0)), dtype=float
            ),
        )

    def rigid_rotation_source_fingerprint(self):
        xdata = self.target.get_xdata(orig=True)
        ydata = self.target.get_ydata(orig=True)
        # OpenSSL-backed SHA-256 is faster than hashlib.blake2b for the common
        # pair of contiguous 100k-coordinate arrays on supported runtimes.
        # Sixteen bytes retain a compact 128-bit stale-state token.
        hasher = hashlib.sha256()
        self._update_raw_fingerprint(hasher, xdata, "x")
        self._update_raw_fingerprint(hasher, ydata, "y")
        self._update_markevery_fingerprint(
            hasher, self.target.get_markevery()
        )
        self._update_line_context_fingerprint(hasher)
        return id(xdata), id(ydata), hasher.digest()[:16]

    def validate_rigid_rotation_plan_source(self, plan: RigidRotationPlan) -> None:
        if plan.source_fingerprint != self.rigid_rotation_source_fingerprint():
            raise UnsupportedArtistError(
                "Line2D coordinates, markevery, transform, or viewport changed "
                "after rigid-rotation preflight; discard the stale plan and "
                "start a new gesture"
            )

    def _atomic_set_raw_data(self, xdata, ydata, *, recache: bool = False) -> None:
        """Set both raw axes or restore every touched Line2D cache field."""

        missing = object()
        cache_names = (
            "_xorig",
            "_yorig",
            "_x",
            "_y",
            "_xy",
            "_path",
            "_transformed_path",
            "_invalidx",
            "_invalidy",
            "_subslice",
            "_x_filled",
        )
        cache = {
            name: getattr(self.target, name, missing) for name in cache_names
        }
        stale_owners = []
        for owner in (self.target, self.target.axes, self.figure):
            if owner is not None and all(owner is not item[0] for item in stale_owners):
                stale_owners.append((owner, getattr(owner, "_stale", missing)))
        try:
            self.target.set_data(xdata, ydata)
            if recache:
                self.target.recache(always=True)
        except Exception:
            for name, value in cache.items():
                if value is missing:
                    if hasattr(self.target, name):
                        delattr(self.target, name)
                else:
                    setattr(self.target, name, value)
            for owner, stale in stale_owners:
                if stale is not missing:
                    setattr(owner, "_stale", stale)
            raise

    @classmethod
    def _raw_snapshot_metadata(cls, raw) -> tuple:
        """Expose storage semantics that numeric ndarray equality would hide."""

        def container_layout(value, depth: int = 0):
            kind = f"{type(value).__module__}.{type(value).__qualname__}"
            if depth >= 2 or type(value) not in (list, tuple) or not value:
                return kind
            return kind, container_layout(value[0], depth + 1)

        try:
            data = np.asarray(raw.data if np.ma.isMaskedArray(raw) else raw)
            shape = tuple(int(value) for value in data.shape)
            dtype = data.dtype.str
            dtype_descriptor = repr(data.dtype.descr) if data.dtype.fields else ""
        except (AttributeError, TypeError, ValueError):
            shape = ()
            dtype = "unavailable"
            dtype_descriptor = ""
        return container_layout(raw), shape, dtype, dtype_descriptor

    def _apply_native_control_points(self, points) -> None:
        prepared = self._prepare_raw_write(points, materialize=True)
        self._atomic_set_raw_data(prepared.xdata, prepared.ydata)

    def apply_native_control_points(self, points) -> None:
        support = self.operation_support(TransformOperation.TRANSLATE)
        if not support.supported:
            raise UnsupportedArtistError(support.reason)
        prepared = self._prepare_raw_write(points, materialize=True)
        self._atomic_set_raw_data(prepared.xdata, prepared.ydata)
        self.record_changes()
        self.invalidate_geometry_cache()

    def snapshot(self):
        if not self.capabilities.can_snapshot:
            raise UnsupportedArtistError(
                "Line2D does not support interaction snapshots"
            )
        xdata = self.target.get_xdata(orig=True)
        ydata = self.target.get_ydata(orig=True)
        return {
            "type": "positions",
            "xdata": deepcopy(xdata),
            "ydata": deepcopy(ydata),
            "xdata_metadata": self._raw_snapshot_metadata(xdata),
            "ydata_metadata": self._raw_snapshot_metadata(ydata),
        }

    def restore(self, state) -> None:
        if state.get("type") != "positions":
            raise ValueError(f"Unsupported snapshot for Line2DAdapter: {state!r}")
        if "xdata" not in state or "ydata" not in state:
            super().restore(state)
            return
        self._atomic_set_raw_data(
            deepcopy(state["xdata"]),
            deepcopy(state["ydata"]),
            recache=True,
        )
        if _CHANGE_RECORDING_ENABLED.get():
            try:
                self._record_restored_state()
            except UnsupportedArtistError:
                # Snapshot restoration remains available for custom unit
                # containers even when generated Python cannot reproduce them.
                pass
        self.invalidate_geometry_cache()

    def serialize_changes(self):
        support = self.operation_support(TransformOperation.SERIALIZE)
        if not support.supported:
            raise UnsupportedArtistError(support.reason)
        xdata = self.target.get_xdata(orig=True)
        ydata = self.target.get_ydata(orig=True)
        try:
            xliteral = replay_literal(xdata, preserve_ndarray=True)
            yliteral = replay_literal(ydata, preserve_ndarray=True)
        except TypeError as error:
            raise UnsupportedArtistError(
                "Line2D raw coordinates contain values that generated Python "
                "cannot serialize losslessly"
            ) from error
        return (
            ChangeRecord.command_change(
                self.target,
                f".set_data({xliteral}, {yliteral})",
            ),
        )

    def serialize_appearance_changes(self):
        state = self.appearance_state()
        return (
            ChangeRecord.command_change(
                self.target,
                f".set_linewidth({replay_literal(state.linewidth)})",
            ),
            ChangeRecord.command_change(
                self.target,
                f".set_markersize({replay_literal(state.markersize)})",
            ),
            ChangeRecord.command_change(
                self.target,
                ".set_markeredgewidth("
                f"{replay_literal(state.markeredgewidth)})",
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
                f".set_extent({replay_literal(extent)}), "
                f"{axes_reference}.set_xlim({replay_literal(xlim)}), "
                f"{axes_reference}.set_ylim({replay_literal(ylim)})",
            ),
        )


class CollectionAdapter(ArtistAdapter):
    unsupported_operation_reasons = {
        TransformOperation.RESIZE_GEOMETRY: (
            "Collection resize must distinguish item positions from marker appearance"
        ),
        TransformOperation.SCALE_APPEARANCE: (
            "Collection appearance scaling requires finite stroke/item dimensions"
        ),
        TransformOperation.RIGID_ROTATE: (
            "Collection common-pivot rotation requires writable non-offset paths "
            "with an affine transform and no hatch or path effect"
        ),
    }
    default_capabilities = ArtistCapabilities(
        can_select=True,
        can_translate=True,
        can_snapshot=True,
        can_serialize=True,
    )

    @classmethod
    def capabilities_for(cls, target) -> ArtistCapabilities:
        paths = target.get_paths()
        if not any(len(cls.finite_points(path.vertices)) for path in paths):
            return ArtistCapabilities()

        has_rendered_offsets = cls.target_uses_rendered_offsets(target)
        if has_rendered_offsets:
            groups = [target.get_offsets()]
        else:
            groups = [path.vertices for path in paths]
        if not any(len(cls.finite_points(group)) for group in groups):
            return ArtistCapabilities()

        transform = (
            target.get_offset_transform()
            if has_rendered_offsets
            else target.get_transform()
        )
        if not getattr(transform, "has_inverse", True):
            return ArtistCapabilities(can_select=True, can_serialize=True)
        try:
            transform.inverted()
        except (
            TypeError,
            ValueError,
            NotImplementedError,
            RuntimeError,
            np.linalg.LinAlgError,
        ):
            return ArtistCapabilities(can_select=True, can_serialize=True)
        return cls.default_capabilities

    def operation_support(
        self, operation: TransformOperation | str
    ) -> OperationSupport:
        operation = TransformOperation.coerce(operation)
        if operation is not TransformOperation.SCALE_APPEARANCE:
            return super().operation_support(operation)
        if not self.capabilities.can_select or not self.capabilities.can_serialize:
            return OperationSupport.denied(
                operation,
                "Collection has no finite serializable appearance target",
            )
        if legend_owner_for_artist(self.target) is not None:
            return OperationSupport.denied(
                operation,
                "Legend-managed Collection appearance is owned by its Legend layout",
            )
        if active_layout_owner_for_artist(self.target) is not None:
            return OperationSupport.denied(
                operation,
                "Collection participates in active layout and cannot scale independently",
            )
        if self.target.get_agg_filter() is not None or self.target.get_path_effects():
            return OperationSupport.denied(
                operation,
                "Collection filters or path effects have independent pixel dimensions",
            )
        if self.target.get_sketch_params() is not None:
            return OperationSupport.denied(
                operation,
                "Collection sketch effects have independent display-space dimensions",
            )
        if getattr(self.target, "get_hatch", lambda: None)():
            return OperationSupport.denied(
                operation,
                "Hatch appearance has no complete public scaling contract",
            )
        linewidths = np.asarray(self.target.get_linewidths(), dtype=float)
        if not np.all(np.isfinite(linewidths)) or np.any(linewidths < 0.0):
            return OperationSupport.denied(
                operation,
                "Collection linewidths must be finite and non-negative",
            )
        if not self.has_scalable_appearance():
            return OperationSupport.denied(
                operation,
                "Collection has no visible stroke or marker area to scale",
            )
        if not self.has_visible_selection_bounds():
            return OperationSupport.denied(
                operation,
                "Collection has no visible geometry inside its active clip region",
            )
        return OperationSupport.allowed(
            operation,
            constraints=("positive_uniform_factor",),
            preview_strategy="redraw",
        )

    def has_scalable_appearance(self) -> bool:
        linewidths = np.asarray(self.target.get_linewidths(), dtype=float)
        edgecolors = np.asarray(self.target.get_edgecolors(), dtype=float)
        try:
            _transform, _offset_transform, _offsets, paths, _transforms, count = (
                self._prepared_offset_items()
            )
        except (AttributeError, TypeError, ValueError, RuntimeError):
            paths = self.target.get_paths()
            count = max(len(paths), len(linewidths), len(edgecolors))
        if count == 0 or not len(paths):
            return False
        widths = self._cycle_values(linewidths, count)
        edges = self._cycled_color_visibility(edgecolors, count)
        path_drawable = np.asarray(
            [self.path_has_drawable_segment(path) for path in paths], dtype=bool
        )
        drawable = path_drawable[np.arange(count) % len(path_drawable)]
        return bool(np.any((widths > 0.0) & edges & drawable))

    @staticmethod
    def _cycled_color_visibility(colors, count: int) -> np.ndarray:
        colors = np.asarray(colors, dtype=float)
        if count <= 0 or colors.size == 0:
            return np.zeros(max(count, 0), dtype=bool)
        if colors.ndim == 1:
            colors = colors.reshape(1, -1)
        alpha = colors[:, 3] if colors.shape[1] >= 4 else np.ones(len(colors))
        return alpha[np.arange(count) % len(alpha)] > 0.0

    def appearance_state(self):
        return CollectionAppearanceState(
            tuple(float(value) for value in self.target.get_linewidths())
        )

    def scaled_appearance_state(self, factor: float):
        state = self.appearance_state()
        return CollectionAppearanceState(
            self.scale_nonnegative_dimensions(
                state.linewidths, factor, label="Collection linewidths"
            )
        )

    @staticmethod
    def validate_appearance_state(state) -> CollectionAppearanceState:
        if not isinstance(state, CollectionAppearanceState):
            raise TypeError("Collection appearance plan has an invalid state type")
        linewidths = np.asarray(state.linewidths, dtype=float)
        if not np.all(np.isfinite(linewidths)) or np.any(linewidths < 0.0):
            raise ValueError(
                "Collection linewidths must be finite and non-negative"
            )
        return state

    def _apply_appearance_state(self, state) -> None:
        state = self.validate_appearance_state(state)
        self.target.set_linewidths(state.linewidths)

    def serialize_appearance_changes(self):
        state = self.appearance_state()
        return (
            ChangeRecord.command_change(
                self.target,
                f".set_linewidths({replay_literal(state.linewidths)})",
            ),
        )

    def local_groups(self) -> list[np.ndarray]:
        return []

    @classmethod
    def target_uses_rendered_offsets(cls, target) -> bool:
        if isinstance(target, PathCollection):
            return True
        offsets = getattr(target, "_offsets", None)
        return isinstance(target, (LineCollection, PolyCollection)) and (
            offsets is not None and len(cls.point_array(offsets)) > 0
        )

    def uses_rendered_offsets(self) -> bool:
        return self.target_uses_rendered_offsets(self.target)

    def group_transform(self) -> Transform:
        if self.uses_rendered_offsets():
            return self.target.get_offset_transform()
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
        if self.uses_rendered_offsets():
            return self._rendered_offset_groups()
        transform = self.group_transform()
        return [
            self.finite_points(transform.transform(group))
            for group in self.local_groups()
            if len(group)
        ]

    @staticmethod
    def _rendered_item_count(offsets, paths, transforms) -> int:
        """Match ``RendererBase._iter_collection`` item-count semantics."""

        return max(len(offsets), len(paths), len(transforms))

    def _prepared_offset_items(self):
        transform, offset_transform, offsets, paths = self.target._prepare_points()
        offsets = self.point_array(offsets)
        transforms = np.asarray(self.target.get_transforms(), dtype=float)
        count = self._rendered_item_count(offsets, paths, transforms)
        return transform, offset_transform, offsets, paths, transforms, count

    def _rendered_offset_groups(self) -> list[np.ndarray]:
        """Return every rendered path envelope at its cycled display offset."""

        (
            transform,
            offset_transform,
            offsets,
            paths,
            transforms,
            count,
        ) = self._prepared_offset_items()
        if count == 0 or not len(offsets) or not len(paths):
            return []

        groups = []
        for index in range(count):
            offset = offsets[index % len(offsets)]
            if not np.all(np.isfinite(offset)):
                continue
            path_transform = transform
            if len(transforms):
                item_transform = transforms[index % len(transforms)]
                if not np.all(np.isfinite(item_transform)):
                    continue
                path_transform = Affine2D(item_transform) + transform
            path = paths[index % len(paths)]
            path_bounds = np.asarray(
                path.get_extents(path_transform).get_points(), dtype=float
            )
            offset = np.asarray(offset_transform.transform(offset), dtype=float)
            if np.all(np.isfinite(path_bounds)) and np.all(np.isfinite(offset)):
                groups.append(path_bounds + offset)
        return groups

    def _rendered_offset_selection_points(self) -> np.ndarray:
        """Union path x offset items through Matplotlib's C renderer iterator."""

        (
            transform,
            offset_transform,
            offsets,
            paths,
            transforms,
            count,
        ) = self._prepared_offset_items()
        if count == 0 or not len(offsets) or not len(paths):
            return np.empty((0, 2), dtype=float)

        item_indices = np.arange(count)
        paddings = self.selection_paddings(count)
        item_offsets = offsets[item_indices % len(offsets)]
        valid_items = np.all(np.isfinite(item_offsets), axis=1) & np.isfinite(paddings)
        if len(transforms):
            item_transforms = transforms[item_indices % len(transforms)]
            valid_items &= np.all(np.isfinite(item_transforms), axis=(1, 2))
        item_indices = item_indices[valid_items]
        paddings = paddings[valid_items]
        if not len(item_indices):
            return np.empty((0, 2), dtype=float)

        envelopes = []
        for padding in np.unique(paddings):
            indices = item_indices[paddings == padding]
            item_paths = [paths[index % len(paths)] for index in indices]
            item_transforms = (
                transforms[indices % len(transforms)] if len(transforms) else []
            )
            item_offsets = offsets[indices % len(offsets)]
            bounds = get_path_collection_extents(
                transform,
                item_paths,
                item_transforms,
                item_offsets,
                offset_transform,
            )
            points = np.asarray(bounds.get_points(), dtype=float)
            if np.all(np.isfinite(points)):
                envelopes.append(self.bounds_points(points, padding=float(padding)))
        if envelopes:
            return self.bounds_points(np.concatenate(envelopes))
        return np.empty((0, 2), dtype=float)

    @staticmethod
    def _cycle_values(values, count: int, *, default: float = 0.0) -> np.ndarray:
        values = np.ma.asarray(values, dtype=float)
        if np.ma.isMaskedArray(values):
            values = values.filled(default)
        values = np.asarray(values, dtype=float).reshape(-1)
        if count <= 0:
            return np.empty(0, dtype=float)
        if values.size == 0:
            return np.full(count, default, dtype=float)
        cycled = values[np.arange(count) % len(values)]
        return np.where(np.isfinite(cycled), cycled, default)

    def selection_paddings(self, count: int) -> np.ndarray:
        linewidths = np.ma.asarray(self.target.get_linewidths(), dtype=float)
        paddings = self._cycle_values(linewidths, count)
        paddings = np.maximum(paddings, 0.0)
        paddings *= self.points_to_pixels(1.0) / 2

        edgecolors = np.asarray(self.target.get_edgecolors(), dtype=float)
        if edgecolors.size == 0:
            return np.zeros(count, dtype=float)
        if edgecolors.ndim == 1:
            edgecolors = edgecolors.reshape(1, -1)
        alphas = edgecolors[:, 3] if edgecolors.shape[1] >= 4 else np.ones(len(edgecolors))
        visible_edges = alphas[np.arange(count) % len(alphas)] > 0
        return np.where(visible_edges, paddings, 0.0)

    def selection_padding(self) -> float:
        """Largest item padding retained for third-party adapter compatibility."""

        groups = self.display_groups()
        paddings = self.selection_paddings(len(groups))
        return float(np.max(paddings)) if len(paddings) else 0.0

    def selection_points(self) -> np.ndarray:
        preview = self._preview_points("_pylustrator_preview_selection_points")
        if preview is not None:
            return preview
        if self.uses_rendered_offsets():
            return self._rendered_offset_selection_points()
        groups = self.display_groups()
        if groups:
            envelopes = [
                self.bounds_points(group, padding=float(padding))
                for group, padding in zip(groups, self.selection_paddings(len(groups)))
            ]
            envelopes = [points for points in envelopes if len(points)]
            if envelopes:
                return self.bounds_points(np.concatenate(envelopes))
        return super().selection_points()

    def preview_rigid_rotation_selection_points(
        self,
        matrix,
        *,
        control_points,
        selection_points,
        planned_control_points,
        planned_native_control_points,
        rotation_value: float | None,
    ) -> np.ndarray:
        if self.uses_rendered_offsets():
            return np.empty((0, 2), dtype=float)
        groups = self.split_points(planned_control_points)
        paddings = self.selection_paddings(len(groups))
        envelopes = [
            self.bounds_points(group, padding=float(padding))
            for group, padding in zip(groups, paddings)
            if len(self.finite_points(group))
        ]
        envelopes = [points for points in envelopes if len(points)]
        if envelopes:
            return self.bounds_points(np.concatenate(envelopes))
        return np.empty((0, 2), dtype=float)

    def split_points(self, points) -> list[np.ndarray]:
        lengths = [len(group) for group in self.local_groups()]
        groups = []
        start = 0
        for length in lengths:
            groups.append(np.asarray(points[start : start + length], dtype=float))
            start += length
        return groups

    def serialize_offset_change(self) -> ChangeRecord:
        offsets = [
            [float(x), float(y)] for x, y in self.point_array(self.target.get_offsets())
        ]
        return ChangeRecord.command_change(
            self.target, f".set_offsets({replay_literal(offsets)})"
        )


class PathCollectionAdapter(CollectionAdapter):
    def has_scalable_appearance(self) -> bool:
        sizes = np.asarray(self.target.get_sizes(), dtype=float)
        facecolors = np.asarray(self.target.get_facecolors(), dtype=float)
        edgecolors = np.asarray(self.target.get_edgecolors(), dtype=float)
        linewidths = np.asarray(self.target.get_linewidths(), dtype=float)
        try:
            _transform, _offset_transform, _offsets, paths, _transforms, count = (
                self._prepared_offset_items()
            )
            count = int(count)
        except (AttributeError, TypeError, ValueError, RuntimeError):
            paths = self.target.get_paths()
            count = 0
        count = max(count, len(sizes), len(facecolors), len(edgecolors), len(linewidths))
        if count == 0 or not len(sizes) or not len(paths):
            return False
        item_sizes = self._cycle_values(sizes, count)
        item_widths = self._cycle_values(linewidths, count)
        faces = self._cycled_color_visibility(facecolors, count)
        edges = self._cycled_color_visibility(edgecolors, count)
        path_fillable = np.asarray(
            [self.path_has_fill_area(path) for path in paths], dtype=bool
        )
        path_drawable = np.asarray(
            [self.path_has_drawable_segment(path) for path in paths], dtype=bool
        )
        path_indices = np.arange(count) % len(paths)
        fillable = path_fillable[path_indices]
        drawable = path_drawable[path_indices]
        painted = (faces & fillable) | (
            edges & (item_widths > 0.0) & drawable
        )
        return bool(np.any((item_sizes > 0.0) & painted))

    def operation_support(
        self, operation: TransformOperation | str
    ) -> OperationSupport:
        operation = TransformOperation.coerce(operation)
        if operation is TransformOperation.SCALE_APPEARANCE:
            sizes = np.asarray(self.target.get_sizes(), dtype=float)
            if not len(sizes):
                return OperationSupport.denied(
                    operation,
                    "PathCollection marker sizes are implicit and cannot be scaled exactly",
                )
            if not np.all(np.isfinite(sizes)) or np.any(sizes < 0.0):
                return OperationSupport.denied(
                    operation,
                    "PathCollection marker areas must be finite and non-negative",
                )
        return super().operation_support(operation)

    def appearance_state(self):
        state = super().appearance_state()
        return PathCollectionAppearanceState(
            state.linewidths,
            tuple(float(value) for value in self.target.get_sizes()),
        )

    def scaled_appearance_state(self, factor: float):
        state = self.appearance_state()
        return PathCollectionAppearanceState(
            self.scale_nonnegative_dimensions(
                state.linewidths, factor, label="PathCollection linewidths"
            ),
            # Matplotlib collection sizes are marker areas in pt^2.
            self.scale_nonnegative_dimensions(
                state.sizes,
                factor,
                power=2,
                label="PathCollection marker areas",
            ),
        )

    @staticmethod
    def validate_appearance_state(state) -> PathCollectionAppearanceState:
        if not isinstance(state, PathCollectionAppearanceState):
            raise TypeError("PathCollection appearance plan has an invalid state type")
        CollectionAdapter.validate_appearance_state(state)
        sizes = np.asarray(state.sizes, dtype=float)
        if not np.all(np.isfinite(sizes)) or np.any(sizes < 0.0):
            raise ValueError(
                "PathCollection marker areas must be finite and non-negative"
            )
        return state

    def _apply_appearance_state(self, state) -> None:
        state = self.validate_appearance_state(state)
        super()._apply_appearance_state(state)
        self.target.set_sizes(state.sizes, dpi=float(self.figure.dpi))

    def serialize_appearance_changes(self):
        return (
            *super().serialize_appearance_changes(),
            ChangeRecord.command_change(
                self.target,
                f".set_sizes({replay_literal(self.appearance_state().sizes)})",
            ),
        )

    def local_groups(self):
        return [self.point_array(self.target.get_offsets())]

    def _apply_native_control_points(self, points) -> None:
        groups = self.split_points(points)
        offsets = groups[0] if groups else np.empty((0, 2))
        self.target.set_offsets(offsets)

    def serialize_changes(self):
        return (self.serialize_offset_change(),)


class LineCollectionAdapter(CollectionAdapter):
    @classmethod
    def capabilities_for(cls, target: LineCollection) -> ArtistCapabilities:
        capabilities = super().capabilities_for(target)
        can_rigid_rotate = bool(
            capabilities.editable
            and legend_owner_for_artist(target) is None
            and active_layout_owner_for_artist(target) is None
            and not cls.target_uses_rendered_offsets(target)
            and cls.transform_is_invertible_affine(target.get_transform())
            and target.get_agg_filter() is None
            and not target.get_path_effects()
            and not target.get_hatch()
        )
        return replace(
            capabilities, can_rigid_rotate=can_rigid_rotate
        )

    def local_groups(self):
        if self.uses_rendered_offsets():
            return [self.point_array(self.target.get_offsets())]
        # ``get_segments()`` drops masked and non-finite rows, which are
        # meaningful path separators.  Transform the authoritative Path
        # vertices so a zero-delta edit cannot connect disjoint runs.
        return [self.point_array(path.vertices) for path in self.target.get_paths()]

    def _apply_native_control_points(self, points) -> None:
        if self.uses_rendered_offsets():
            groups = self.split_points(points)
            self.target.set_offsets(groups[0] if groups else np.empty((0, 2)))
            return
        self.target.set_segments(self.split_points(points))

    def serialize_changes(self):
        groups = [
            [[float(x), float(y)] for x, y in self.point_array(path.vertices)]
            for path in self.target.get_paths()
        ]
        records = [
            ChangeRecord.command_change(
                self.target, f".set_segments({replay_literal(groups)})"
            )
        ]
        if self.uses_rendered_offsets():
            records.append(self.serialize_offset_change())
        return tuple(records)


class PolyCollectionAdapter(CollectionAdapter):
    @classmethod
    def capabilities_for(cls, target: PolyCollection) -> ArtistCapabilities:
        capabilities = super().capabilities_for(target)
        can_rigid_rotate = bool(
            capabilities.editable
            and legend_owner_for_artist(target) is None
            and active_layout_owner_for_artist(target) is None
            and not cls.target_uses_rendered_offsets(target)
            and cls.transform_is_invertible_affine(target.get_transform())
            and target.get_agg_filter() is None
            and not target.get_path_effects()
            and not target.get_hatch()
        )
        return replace(
            capabilities, can_rigid_rotate=can_rigid_rotate
        )

    def local_groups(self):
        if self.uses_rendered_offsets():
            return [self.point_array(self.target.get_offsets())]
        return [self.point_array(path.vertices) for path in self.target.get_paths()]

    def _apply_native_control_points(self, points) -> None:
        if self.uses_rendered_offsets():
            groups = self.split_points(points)
            self.target.set_offsets(groups[0] if groups else np.empty((0, 2)))
            return
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
        records = [
            ChangeRecord.command_change(
                self.target,
                f".set_verts_and_codes({replay_literal(groups)}, "
                f"{replay_literal(codes)})",
            )
        ]
        if self.uses_rendered_offsets():
            records.append(self.serialize_offset_change())
        return tuple(records)


artist_adapter_registry = ArtistAdapterRegistry()


def register_artist_adapter(
    artist_type: type,
    *,
    priority: int = 0,
    replace: bool = False,
    inheritance_policy: AdapterInheritancePolicy | str = (
        AdapterInheritancePolicy.EXACT
    ),
    registry: ArtistAdapterRegistry = artist_adapter_registry,
):
    """Decorator for built-in or third-party adapter registration.

    Registrations are exact-only unless the adapter author explicitly sets
    ``inheritance_policy=AdapterInheritancePolicy.VALIDATED`` after validating
    the full subclass geometry, mutation, snapshot, and replay contract.
    """

    def decorator(adapter_type: type[ArtistAdapter]):
        registry.register(
            artist_type,
            adapter_type,
            priority=priority,
            replace=replace,
            inheritance_policy=inheritance_policy,
        )
        return adapter_type

    return decorator


def get_artist_adapter(target: Artist) -> ArtistAdapter:
    return artist_adapter_registry.create(target)


# Registration order is intentionally not semantic.  Resolution uses MRO
# distance, then priority and only then registration order for true ties.
artist_adapter_registry.register(
    Artist,
    ArtistAdapter,
    inheritance_policy=AdapterInheritancePolicy.VALIDATED,
)

_BUILTIN_ADAPTER_REGISTRATIONS = (
    (EditorGroup, EditorGroupAdapter),
    (Axes, AxesAdapter),
    (Text, TextAdapter),
    (Annotation, AnnotationAdapter),
    (Legend, LegendAdapter),
    (Line2D, Line2DAdapter),
    (AxesImage, AxesImageAdapter),
    (Rectangle, RectangleAdapter),
    (Ellipse, EllipseAdapter),
    (Arc, ArcAdapter),
    (Circle, CircleAdapter),
    (FancyArrowPatch, FancyArrowPatchAdapter),
    (ConnectionPatch, ConnectionPatchAdapter),
    (FancyBboxPatch, FancyBboxPatchAdapter),
    (RegularPolygon, RegularPolygonAdapter),
    (CirclePolygon, CirclePolygonAdapter),
    (Wedge, WedgeAdapter),
    (Polygon, PolygonAdapter),
    (PathPatch, PathPatchAdapter),
    (PathCollection, PathCollectionAdapter),
    (LineCollection, LineCollectionAdapter),
    (PolyCollection, PolyCollectionAdapter),
)
_fill_between_type = getattr(
    mpl_collections, "FillBetweenPolyCollection", None
)
if _fill_between_type is not None:
    _BUILTIN_ADAPTER_REGISTRATIONS += (
        (_fill_between_type, PolyCollectionAdapter),
    )

for _artist_type, _adapter_type in _BUILTIN_ADAPTER_REGISTRATIONS:
    artist_adapter_registry.register(_artist_type, _adapter_type)


__all__ = [
    "AdapterInheritancePolicy",
    "AdapterRegistration",
    "AnnotationAdapter",
    "ArcAdapter",
    "AppearanceScalePlan",
    "ArtistAdapter",
    "ArtistAdapterRegistry",
    "ArtistCapabilities",
    "AxesAdapter",
    "AxesImageAdapter",
    "ChangeRecord",
    "CircleAdapter",
    "CirclePolygonAdapter",
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
    "RigidRotationPlan",
    "TextAdapter",
    "UnsupportedArtistError",
    "UnsupportedSubclassAdapter",
    "WedgeAdapter",
    "artist_adapter_registry",
    "active_layout_owner_for_artist",
    "cached_selection_points",
    "checkXLabel",
    "checkYLabel",
    "container_owner_for_artist",
    "get_artist_adapter",
    "iter_figure_legends",
    "iter_legend_children",
    "iter_legend_managed_artists",
    "invalidate_legend_owner_inventory",
    "legend_anchor_is_point",
    "legend_anchor_transform",
    "legend_display_loc",
    "legend_loc_transform",
    "legend_owner_for_artist",
    "legend_owner_for_text",
    "legend_owner_snapshot",
    "layout_owner_for_text",
    "register_artist_adapter",
    "set_legend_point_anchor_display",
    "selection_geometry_snapshot",
    "suspend_change_recording",
]
