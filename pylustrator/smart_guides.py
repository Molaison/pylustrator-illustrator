"""Deterministic, renderer-independent smart-guide planning.

The interaction layer should capture artist geometry in *display pixels* once at
the beginning of a gesture, build :class:`GuideCandidateIndex`, and reuse that
index for every motion event.  A query returns a small immutable
:class:`SnapPlan`; the exact same plan can be drawn for preview and applied on
commit, so the preview cannot disagree with the final transform.

This module deliberately knows nothing about Matplotlib artists or Qt.  Artist
adapters own conversion to display geometry, while a UI renderer owns drawing
the returned overlay primitives.  The current index uses sorted scalar ranges
(``O(log n + k)`` per moving feature).  Its public query boundary can later be
backed by a shared spatial index without changing ``SnapPlan`` consumers.
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from enum import Enum
import hashlib
import math
from types import MappingProxyType
from typing import (
    TYPE_CHECKING,
    Any,
    Iterable,
    Iterator,
    Mapping,
    Protocol,
    TypeAlias,
    TypeVar,
    runtime_checkable,
)

import numpy as np

if TYPE_CHECKING:
    # ty.toml intentionally checks the legacy codebase as Python 3.9 even
    # though the package runtime is >=3.11.
    from typing_extensions import dataclass_transform
else:
    from typing import dataclass_transform


_T = TypeVar("_T")


@dataclass_transform(frozen_default=True)
def _frozen_slots(cls: type[_T], /) -> type[_T]:
    """Keep >=3.11 runtime slot savings under the legacy 3.9 type target."""

    runtime_dataclass: Any = dataclass
    return runtime_dataclass(frozen=True, slots=True)(cls)


class Axis(str, Enum):
    """A display-coordinate axis."""

    X = "x"
    Y = "y"


class GuideKind(str, Enum):
    """Semantic guide types, in user-facing tie-break priority order."""

    ANCHOR = "anchor"
    BASELINE = "baseline"
    EDGE = "edge"
    CENTER = "center"
    EQUAL_GAP = "equal_gap"


class FeatureKind(str, Enum):
    """The feature on a source or moving object that participates in a snap."""

    MIN_EDGE = "min_edge"
    MAX_EDGE = "max_edge"
    CENTER = "center"
    BASELINE = "baseline"
    ANCHOR = "anchor"


_GUIDE_PRIORITY: Mapping[GuideKind, int] = MappingProxyType(
    {
        GuideKind.ANCHOR: 0,
        GuideKind.BASELINE: 1,
        GuideKind.EDGE: 2,
        GuideKind.CENTER: 3,
        GuideKind.EQUAL_GAP: 4,
    }
)

class StaleGuideSnapshotError(RuntimeError):
    """Raised when a gesture tries to use an index or plan from old geometry."""


@_frozen_slots
class DisplayBounds:
    """An axis-aligned bounding box expressed exclusively in display pixels."""

    x0: float
    y0: float
    x1: float
    y1: float

    def __post_init__(self) -> None:
        for name in ("x0", "y0", "x1", "y1"):
            object.__setattr__(self, name, float(getattr(self, name)))

    @property
    def width(self) -> float:
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        return self.y1 - self.y0

    @property
    def center_x(self) -> float:
        return (self.x0 + self.x1) * 0.5

    @property
    def center_y(self) -> float:
        return (self.y0 + self.y1) * 0.5

    @property
    def is_finite_and_ordered(self) -> bool:
        return (
            all(math.isfinite(value) for value in (self.x0, self.y0, self.x1, self.y1))
            and self.x0 <= self.x1
            and self.y0 <= self.y1
        )

    def translated(self, delta_px: tuple[float, float]) -> DisplayBounds:
        dx, dy = delta_px
        return DisplayBounds(self.x0 + dx, self.y0 + dy, self.x1 + dx, self.y1 + dy)

    def minimum(self, axis: Axis) -> float:
        return self.x0 if axis is Axis.X else self.y0

    def maximum(self, axis: Axis) -> float:
        return self.x1 if axis is Axis.X else self.y1

    def center(self, axis: Axis) -> float:
        return self.center_x if axis is Axis.X else self.center_y

    def size(self, axis: Axis) -> float:
        return self.width if axis is Axis.X else self.height

    def cross_interval(self, axis: Axis) -> tuple[float, float]:
        return (self.y0, self.y1) if axis is Axis.X else (self.x0, self.x1)


@_frozen_slots
class ExplicitAnchor:
    """A semantic point, such as the insertion anchor of a Text artist."""

    x: float
    y: float
    name: str = "anchor"

    def __post_init__(self) -> None:
        object.__setattr__(self, "x", float(self.x))
        object.__setattr__(self, "y", float(self.y))
        if not isinstance(self.name, str):
            raise TypeError("anchor name must be a string")

    @property
    def is_finite(self) -> bool:
        return math.isfinite(self.x) and math.isfinite(self.y)


@_frozen_slots
class GuideObject:
    """Immutable display geometry captured from one source artist.

    ``order`` is the artist's paint/insertion order.  When distances and guide
    semantics tie, the highest z-order, then latest paint order, then lexical
    stable id wins.  ``scope_id`` prevents guides from unrelated coordinate
    spaces (for example, separate canvases) from interacting.
    """

    stable_id: str
    bounds: DisplayBounds
    z_order: float = 0.0
    order: int = 0
    visible: bool = True
    locked: bool = False
    baseline_y: float | None = None
    anchors: tuple[ExplicitAnchor, ...] = ()
    scope_id: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.stable_id, str) or not self.stable_id:
            raise ValueError("stable_id must be a non-empty string")
        if not isinstance(self.bounds, DisplayBounds):
            try:
                bounds = DisplayBounds(*self.bounds)  # type: ignore[arg-type]
            except (TypeError, ValueError) as exc:
                raise TypeError("bounds must be DisplayBounds or four coordinates") from exc
            object.__setattr__(self, "bounds", bounds)
        object.__setattr__(self, "z_order", float(self.z_order))
        object.__setattr__(self, "order", int(self.order))
        if self.baseline_y is not None:
            object.__setattr__(self, "baseline_y", float(self.baseline_y))
        anchors: list[ExplicitAnchor] = []
        for anchor in self.anchors:
            if isinstance(anchor, ExplicitAnchor):
                anchors.append(anchor)
            else:
                try:
                    anchors.append(ExplicitAnchor(*anchor))  # type: ignore[arg-type]
                except (TypeError, ValueError) as exc:
                    raise TypeError("anchors must contain ExplicitAnchor values") from exc
        object.__setattr__(self, "anchors", tuple(anchors))
        if self.scope_id is not None and not isinstance(self.scope_id, str):
            raise TypeError("scope_id must be a string or None")

    @property
    def is_finite(self) -> bool:
        return (
            self.bounds.is_finite_and_ordered
            and math.isfinite(self.z_order)
            and (self.baseline_y is None or math.isfinite(self.baseline_y))
            and all(anchor.is_finite for anchor in self.anchors)
        )


@_frozen_slots
class MovingGeometry:
    """Proposed (pre-snap) geometry of the current selection in display pixels."""

    bounds: DisplayBounds
    baseline_y: float | None = None
    anchors: tuple[ExplicitAnchor, ...] = ()
    scope_id: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.bounds, DisplayBounds):
            try:
                bounds = DisplayBounds(*self.bounds)  # type: ignore[arg-type]
            except (TypeError, ValueError) as exc:
                raise TypeError("bounds must be DisplayBounds or four coordinates") from exc
            object.__setattr__(self, "bounds", bounds)
        if not self.bounds.is_finite_and_ordered:
            raise ValueError("moving bounds must be finite and ordered")
        if self.baseline_y is not None:
            baseline = float(self.baseline_y)
            if not math.isfinite(baseline):
                raise ValueError("moving baseline must be finite")
            object.__setattr__(self, "baseline_y", baseline)
        anchors: list[ExplicitAnchor] = []
        for anchor in self.anchors:
            if isinstance(anchor, ExplicitAnchor):
                value = anchor
            else:
                try:
                    value = ExplicitAnchor(*anchor)  # type: ignore[arg-type]
                except (TypeError, ValueError) as exc:
                    raise TypeError("anchors must contain ExplicitAnchor values") from exc
            if not value.is_finite:
                raise ValueError("moving anchors must be finite")
            anchors.append(value)
        object.__setattr__(self, "anchors", tuple(anchors))
        if self.scope_id is not None and not isinstance(self.scope_id, str):
            raise TypeError("scope_id must be a string or None")

    @classmethod
    def from_object(cls, source: GuideObject) -> MovingGeometry:
        return cls(
            bounds=source.bounds,
            baseline_y=source.baseline_y,
            anchors=source.anchors,
            scope_id=source.scope_id,
        )


class _Digest(Protocol):
    def update(self, value: bytes) -> None: ...


def _fingerprint_feed(digest: _Digest, value: object) -> None:
    encoded = repr(value).encode("utf-8")
    digest.update(len(encoded).to_bytes(8, "big"))
    digest.update(encoded)


def _snapshot_fingerprint(
    objects: tuple[GuideObject, ...],
    selected_ids: frozenset[str],
    revision: str | int | None,
) -> str:
    digest = hashlib.blake2b(digest_size=16, person=b"pyl-smart-guide")
    _fingerprint_feed(digest, "smart-guides-v1")
    _fingerprint_feed(digest, revision)
    _fingerprint_feed(digest, tuple(sorted(selected_ids)))
    for source in objects:
        _fingerprint_feed(
            digest,
            (
                source.stable_id,
                source.bounds.x0,
                source.bounds.y0,
                source.bounds.x1,
                source.bounds.y1,
                source.z_order,
                source.order,
                source.baseline_y,
                tuple((anchor.x, anchor.y, anchor.name) for anchor in source.anchors),
                source.scope_id,
            ),
        )
    return digest.hexdigest()


@_frozen_slots
class GuideSnapshot:
    """A filtered, immutable gesture-start view of all guide sources."""

    objects: tuple[GuideObject, ...]
    selected_ids: frozenset[str]
    revision: str | int | None
    fingerprint: str

    @classmethod
    def capture(
        cls,
        objects: Iterable[GuideObject],
        *,
        selected_ids: Iterable[str] = (),
        revision: str | int | None = None,
    ) -> GuideSnapshot:
        if revision is not None and not isinstance(revision, (str, int)):
            raise TypeError("revision must be a string, integer, or None")
        selected = frozenset(selected_ids)
        if any(not isinstance(stable_id, str) for stable_id in selected):
            raise TypeError("selected ids must be strings")

        materialized = tuple(objects)
        if any(not isinstance(source, GuideObject) for source in materialized):
            raise TypeError("snapshot sources must be GuideObject values")
        ids = [source.stable_id for source in materialized]
        if len(set(ids)) != len(ids):
            raise ValueError("source stable ids must be unique")

        eligible = tuple(
            sorted(
                (
                    source
                    for source in materialized
                    if source.visible
                    and not source.locked
                    and source.stable_id not in selected
                    and source.is_finite
                ),
                key=lambda source: source.stable_id,
            )
        )
        fingerprint = _snapshot_fingerprint(eligible, selected, revision)
        return cls(eligible, selected, revision, fingerprint)

    def require_fingerprint(self, fingerprint: str) -> None:
        if fingerprint != self.fingerprint:
            raise StaleGuideSnapshotError(
                "smart-guide source geometry changed during the gesture"
            )


@_frozen_slots
class GuideLine:
    """A renderer-neutral infinite-guide segment for a direct alignment hit."""

    axis: Axis
    position_px: float
    span_px: tuple[float, float]
    kind: GuideKind
    source_ids: tuple[str, ...]


@_frozen_slots
class EqualGapOverlay:
    """Two dimension segments showing a repeated/equal display-pixel gap."""

    axis: Axis
    intervals_px: tuple[tuple[float, float], tuple[float, float]]
    cross_position_px: float
    gap_px: float
    source_ids: tuple[str, ...]


OverlayPrimitive: TypeAlias = GuideLine | EqualGapOverlay


@_frozen_slots
class SnapHit:
    """One deterministic winning guide for one display axis."""

    axis: Axis
    kind: GuideKind
    delta_px: float
    source_ids: tuple[str, ...]
    source_position_px: float
    target_position_px: float
    source_feature: str
    target_feature: str
    gap_px: float | None = None


@_frozen_slots
class SnapPlan:
    """Small immutable output shared verbatim by preview and commit."""

    snapshot_fingerprint: str
    delta_px: tuple[float, float]
    hits: tuple[SnapHit, ...]
    overlays: tuple[OverlayPrimitive, ...]
    examined_candidate_count: int

    @property
    def has_snap(self) -> bool:
        return bool(self.hits)

    def require_fingerprint(self, fingerprint: str) -> None:
        if fingerprint != self.snapshot_fingerprint:
            raise StaleGuideSnapshotError(
                "smart-guide plan belongs to an older source snapshot"
            )

    def apply_to_point(self, point_px: tuple[float, float]) -> tuple[float, float]:
        return (point_px[0] + self.delta_px[0], point_px[1] + self.delta_px[1])

    def apply_to_bounds(self, bounds: DisplayBounds) -> DisplayBounds:
        return bounds.translated(self.delta_px)


@runtime_checkable
class SmartGuideQueryIndex(Protocol):
    """Replaceable query boundary for a future shared 2-D spatial index."""

    @property
    def snapshot(self) -> GuideSnapshot: ...

    def query(
        self,
        moving: MovingGeometry,
        *,
        tolerance_px: float = 5.0,
        expected_fingerprint: str | None = None,
        include_equal_gaps: bool = True,
        axes: frozenset[Axis] | None = None,
    ) -> SnapPlan: ...


@_frozen_slots
class _AxisCandidate:
    axis: Axis
    kind: GuideKind
    position_px: float
    source_id: str
    source_feature: str
    source_span_px: tuple[float, float]
    z_order: float
    order: int
    feature_order: int


class _GapMode(str, Enum):
    BEFORE = "equal_gap_before"
    BETWEEN = "equal_gap_between"
    AFTER = "equal_gap_after"


@_frozen_slots
class _GapReference:
    axis: Axis
    position_px: float
    target_feature: FeatureKind
    mode: _GapMode
    source_ids: tuple[str, str]
    source_bounds: tuple[DisplayBounds, DisplayBounds]
    existing_gap_px: float
    z_order: float
    order: int
    stable_key: str


_IndexItem: TypeAlias = _AxisCandidate | _GapReference


@_frozen_slots
class _SortedBucket:
    positions: tuple[float, ...]
    items: tuple[_IndexItem, ...]

    def range(self, lower: float, upper: float) -> tuple[int, int]:
        return (
            bisect_left(self.positions, lower),
            bisect_right(self.positions, upper),
        )


@_frozen_slots
class _TargetFeature:
    axis: Axis
    kind: GuideKind
    feature: FeatureKind
    label: str
    position_px: float
    span_px: tuple[float, float]
    order: int


@_frozen_slots
class _Proposal:
    axis: Axis
    kind: GuideKind
    delta_px: float
    source_ids: tuple[str, ...]
    source_position_px: float
    target_position_px: float
    source_feature: str
    target_feature: str
    z_order: float
    order: int
    stable_key: str
    feature_order: int
    source_span_px: tuple[float, float] | None = None
    target_span_px: tuple[float, float] | None = None
    gap_reference: _GapReference | None = None
    gap_px: float | None = None


def _freeze_bucket(items: list[_IndexItem]) -> _SortedBucket:
    def sort_key(item: _IndexItem) -> tuple[object, ...]:
        if isinstance(item, _AxisCandidate):
            return (
                item.position_px,
                -item.z_order,
                -item.order,
                item.source_id,
                item.feature_order,
                item.source_feature,
            )
        return (
            item.position_px,
            -item.z_order,
            -item.order,
            item.stable_key,
            item.mode.value,
        )

    frozen = tuple(sorted(items, key=sort_key))
    return _SortedBucket(tuple(item.position_px for item in frozen), frozen)


def _intervals_overlap(
    first: tuple[float, float], second: tuple[float, float]
) -> bool:
    return max(first[0], second[0]) <= min(first[1], second[1])


_NeighborRank: TypeAlias = tuple[float, float, int, str]


@_frozen_slots
class _NeighborEntry:
    rank: _NeighborRank
    source: GuideObject


def _best_neighbor(
    first: _NeighborEntry | None, second: _NeighborEntry | None
) -> _NeighborEntry | None:
    if first is None:
        return second
    if second is None or first.rank <= second.rank:
        return first
    return second


class _IntervalBestIndex:
    """Insert-only interval index returning the best intersecting payload.

    An interval is decomposed into ``O(log n)`` canonical segment-tree nodes.
    Each node retains only its best local payload and best subtree payload, so
    both insertion and interval-intersection queries remain ``O(log n)`` and
    storage remains linear.  It is intentionally small enough to replace with
    a shared scene spatial index later.
    """

    __slots__ = ("_coordinate_count", "_local", "_subtree")

    def __init__(self, coordinate_count: int):
        if coordinate_count <= 0:
            raise ValueError("coordinate_count must be positive")
        self._coordinate_count = coordinate_count
        capacity = coordinate_count * 4
        self._local: list[_NeighborEntry | None] = [None] * capacity
        self._subtree: list[_NeighborEntry | None] = [None] * capacity

    def add(self, left: int, right: int, entry: _NeighborEntry) -> None:
        self._add(1, 0, self._coordinate_count - 1, left, right, entry)

    def _add(
        self,
        node: int,
        node_left: int,
        node_right: int,
        query_left: int,
        query_right: int,
        entry: _NeighborEntry,
    ) -> None:
        if query_left <= node_left and node_right <= query_right:
            self._local[node] = _best_neighbor(self._local[node], entry)
        else:
            midpoint = (node_left + node_right) // 2
            if query_left <= midpoint:
                self._add(
                    node * 2,
                    node_left,
                    midpoint,
                    query_left,
                    query_right,
                    entry,
                )
            if query_right > midpoint:
                self._add(
                    node * 2 + 1,
                    midpoint + 1,
                    node_right,
                    query_left,
                    query_right,
                    entry,
                )
        children_best = None
        if node_left != node_right:
            children_best = _best_neighbor(
                self._subtree[node * 2], self._subtree[node * 2 + 1]
            )
        self._subtree[node] = _best_neighbor(self._local[node], children_best)

    def query(self, left: int, right: int) -> _NeighborEntry | None:
        return self._query(1, 0, self._coordinate_count - 1, left, right)

    def _query(
        self,
        node: int,
        node_left: int,
        node_right: int,
        query_left: int,
        query_right: int,
    ) -> _NeighborEntry | None:
        if query_left <= node_left and node_right <= query_right:
            return self._subtree[node]

        result = self._local[node]
        midpoint = (node_left + node_right) // 2
        if query_left <= midpoint:
            result = _best_neighbor(
                result,
                self._query(
                    node * 2,
                    node_left,
                    midpoint,
                    query_left,
                    query_right,
                ),
            )
        if query_right > midpoint:
            result = _best_neighbor(
                result,
                self._query(
                    node * 2 + 1,
                    midpoint + 1,
                    node_right,
                    query_left,
                    query_right,
                ),
            )
        return result


def _nearest_gap_pairs_sweep(
    sources: list[GuideObject], axis: Axis
) -> tuple[tuple[GuideObject, GuideObject], ...]:
    """Find each source's nearest compatible neighbour before and after it.

    A simple adjacent-pair walk is incorrect because an intervening rectangle
    in another row/column can shadow the true neighbour.  These two offline
    sweeps index orthogonal intervals while moving the primary-axis frontier.
    """

    if len(sources) < 2:
        return ()
    cross_coordinates = sorted(
        {
            coordinate
            for source in sources
            for coordinate in source.bounds.cross_interval(axis)
        }
    )
    coordinate_index = {
        coordinate: index for index, coordinate in enumerate(cross_coordinates)
    }

    def cross_indices(source: GuideObject) -> tuple[int, int]:
        lower, upper = source.bounds.cross_interval(axis)
        return coordinate_index[lower], coordinate_index[upper]

    pairs: dict[tuple[str, str], tuple[GuideObject, GuideObject]] = {}

    # Forward sweep: candidates have min > query.max.  Of all orthogonally
    # intersecting candidates, the smallest min is the closest one after it.
    forward_index = _IntervalBestIndex(len(cross_coordinates))
    forward_candidates = sorted(
        sources,
        key=lambda source: (
            -source.bounds.minimum(axis),
            -source.z_order,
            -source.order,
            source.stable_id,
        ),
    )
    forward_queries = sorted(
        sources,
        key=lambda source: (
            -source.bounds.maximum(axis),
            source.stable_id,
        ),
    )
    candidate_index = 0
    for first in forward_queries:
        while (
            candidate_index < len(forward_candidates)
            and forward_candidates[candidate_index].bounds.minimum(axis)
            > first.bounds.maximum(axis)
        ):
            candidate = forward_candidates[candidate_index]
            lower, upper = cross_indices(candidate)
            forward_index.add(
                lower,
                upper,
                _NeighborEntry(
                    (
                        candidate.bounds.minimum(axis),
                        -candidate.z_order,
                        -candidate.order,
                        candidate.stable_id,
                    ),
                    candidate,
                ),
            )
            candidate_index += 1
        lower, upper = cross_indices(first)
        entry = forward_index.query(lower, upper)
        if entry is not None:
            second = entry.source
            pairs[(first.stable_id, second.stable_id)] = (first, second)

    # Backward sweep: candidates have max < query.min.  Negating max makes the
    # largest (closest) max the minimum rank returned by the same index.
    backward_index = _IntervalBestIndex(len(cross_coordinates))
    backward_candidates = sorted(
        sources,
        key=lambda source: (
            source.bounds.maximum(axis),
            -source.z_order,
            -source.order,
            source.stable_id,
        ),
    )
    backward_queries = sorted(
        sources,
        key=lambda source: (
            source.bounds.minimum(axis),
            source.stable_id,
        ),
    )
    candidate_index = 0
    for second in backward_queries:
        while (
            candidate_index < len(backward_candidates)
            and backward_candidates[candidate_index].bounds.maximum(axis)
            < second.bounds.minimum(axis)
        ):
            candidate = backward_candidates[candidate_index]
            lower, upper = cross_indices(candidate)
            backward_index.add(
                lower,
                upper,
                _NeighborEntry(
                    (
                        -candidate.bounds.maximum(axis),
                        -candidate.z_order,
                        -candidate.order,
                        candidate.stable_id,
                    ),
                    candidate,
                ),
            )
            candidate_index += 1
        lower, upper = cross_indices(second)
        entry = backward_index.query(lower, upper)
        if entry is not None:
            first = entry.source
            pairs[(first.stable_id, second.stable_id)] = (first, second)

    return tuple(
        sorted(
            pairs.values(),
            key=lambda pair: (
                pair[0].bounds.minimum(axis),
                pair[1].bounds.minimum(axis),
                pair[0].stable_id,
                pair[1].stable_id,
            ),
        )
    )


def _nearest_gap_pairs_vectorized(
    sources: list[GuideObject], axis: Axis
) -> tuple[tuple[GuideObject, GuideObject], ...]:
    """Exact small-scene neighbour search using bounded NumPy matrices.

    The sweep above remains asymptotically necessary for large documents.  At
    the few hundred objects typical of a scientific figure, however, its many
    Python segment-tree calls dominate gesture setup.  Two bounded boolean
    matrices reproduce the same forward/backward rank rules much faster.
    """

    count = len(sources)
    if count < 2:
        return ()
    if axis is Axis.X:
        minimum = np.fromiter((item.bounds.x0 for item in sources), float, count)
        maximum = np.fromiter((item.bounds.x1 for item in sources), float, count)
        cross_minimum = np.fromiter(
            (item.bounds.y0 for item in sources), float, count
        )
        cross_maximum = np.fromiter(
            (item.bounds.y1 for item in sources), float, count
        )
    else:
        minimum = np.fromiter((item.bounds.y0 for item in sources), float, count)
        maximum = np.fromiter((item.bounds.y1 for item in sources), float, count)
        cross_minimum = np.fromiter(
            (item.bounds.x0 for item in sources), float, count
        )
        cross_maximum = np.fromiter(
            (item.bounds.x1 for item in sources), float, count
        )
    z_order = np.fromiter((item.z_order for item in sources), float, count)
    paint_order = np.fromiter((item.order for item in sources), np.int64, count)
    stable_ids = np.asarray([item.stable_id for item in sources])
    cross_overlaps = (
        cross_minimum[None, :] <= cross_maximum[:, None]
    ) & (cross_maximum[None, :] >= cross_minimum[:, None])
    pairs: dict[tuple[str, str], tuple[GuideObject, GuideObject]] = {}

    def choose(
        eligible: np.ndarray,
        ranking: np.ndarray,
        *,
        before: bool,
    ) -> None:
        ranked = eligible[:, ranking]
        available = np.any(ranked, axis=1)
        choices = ranking[np.argmax(ranked, axis=1)]
        for query_index in np.flatnonzero(available):
            candidate_index = int(choices[query_index])
            query_index = int(query_index)
            if before:
                first, second = sources[candidate_index], sources[query_index]
            else:
                first, second = sources[query_index], sources[candidate_index]
            pairs[(first.stable_id, second.stable_id)] = (first, second)

    forward = (minimum[None, :] > maximum[:, None]) & cross_overlaps
    forward_rank = np.lexsort(
        (stable_ids, -paint_order, -z_order, minimum)
    )
    choose(forward, forward_rank, before=False)

    backward = (maximum[None, :] < minimum[:, None]) & cross_overlaps
    backward_rank = np.lexsort(
        (stable_ids, -paint_order, -z_order, -maximum)
    )
    choose(backward, backward_rank, before=True)

    return tuple(
        sorted(
            pairs.values(),
            key=lambda pair: (
                pair[0].bounds.minimum(axis),
                pair[1].bounds.minimum(axis),
                pair[0].stable_id,
                pair[1].stable_id,
            ),
        )
    )


def _nearest_gap_pairs(
    sources: list[GuideObject], axis: Axis
) -> tuple[tuple[GuideObject, GuideObject], ...]:
    # At 1024 objects the two temporary boolean matrices remain only a few MB.
    # Larger scenes keep the O(n log n), O(n) sweep and never approach a
    # quadratic allocation.
    if len(sources) <= 1024:
        return _nearest_gap_pairs_vectorized(sources, axis)
    return _nearest_gap_pairs_sweep(sources, axis)


def _target_features(moving: MovingGeometry) -> Iterator[_TargetFeature]:
    bounds = moving.bounds
    yield _TargetFeature(
        Axis.X,
        GuideKind.EDGE,
        FeatureKind.MIN_EDGE,
        "x_min_edge",
        bounds.x0,
        (bounds.y0, bounds.y1),
        0,
    )
    yield _TargetFeature(
        Axis.X,
        GuideKind.EDGE,
        FeatureKind.MAX_EDGE,
        "x_max_edge",
        bounds.x1,
        (bounds.y0, bounds.y1),
        1,
    )
    yield _TargetFeature(
        Axis.X,
        GuideKind.CENTER,
        FeatureKind.CENTER,
        "x_center",
        bounds.center_x,
        (bounds.y0, bounds.y1),
        2,
    )
    yield _TargetFeature(
        Axis.Y,
        GuideKind.EDGE,
        FeatureKind.MIN_EDGE,
        "y_min_edge",
        bounds.y0,
        (bounds.x0, bounds.x1),
        0,
    )
    yield _TargetFeature(
        Axis.Y,
        GuideKind.EDGE,
        FeatureKind.MAX_EDGE,
        "y_max_edge",
        bounds.y1,
        (bounds.x0, bounds.x1),
        1,
    )
    yield _TargetFeature(
        Axis.Y,
        GuideKind.CENTER,
        FeatureKind.CENTER,
        "y_center",
        bounds.center_y,
        (bounds.x0, bounds.x1),
        2,
    )
    if moving.baseline_y is not None:
        yield _TargetFeature(
            Axis.Y,
            GuideKind.BASELINE,
            FeatureKind.BASELINE,
            "baseline",
            moving.baseline_y,
            (bounds.x0, bounds.x1),
            3,
        )
    for index, anchor in enumerate(moving.anchors):
        suffix = anchor.name or str(index)
        yield _TargetFeature(
            Axis.X,
            GuideKind.ANCHOR,
            FeatureKind.ANCHOR,
            f"anchor:{suffix}:x",
            anchor.x,
            (anchor.y, anchor.y),
            4 + index,
        )
        yield _TargetFeature(
            Axis.Y,
            GuideKind.ANCHOR,
            FeatureKind.ANCHOR,
            f"anchor:{suffix}:y",
            anchor.y,
            (anchor.x, anchor.x),
            4 + index,
        )


class GuideCandidateIndex:
    """Pre-indexed smart-guide candidates for one immutable source snapshot."""

    __slots__ = (
        "_snapshot",
        "_guide_buckets",
        "_gap_buckets",
        "_alignment_candidate_count",
        "_gap_reference_count",
    )

    def __init__(
        self,
        snapshot: GuideSnapshot,
        *,
        include_equal_gaps: bool = True,
    ):
        if not isinstance(snapshot, GuideSnapshot):
            raise TypeError("snapshot must be a GuideSnapshot")
        self._snapshot = snapshot
        guide_lists: dict[tuple[str | None, Axis], list[_IndexItem]] = {}
        gap_lists: dict[
            tuple[str | None, Axis, FeatureKind], list[_IndexItem]
        ] = {}

        for source in snapshot.objects:
            self._add_source_candidates(guide_lists, source)
        if include_equal_gaps:
            self._add_gap_references(gap_lists, snapshot.objects)

        self._guide_buckets = MappingProxyType(
            {key: _freeze_bucket(items) for key, items in guide_lists.items()}
        )
        self._gap_buckets = MappingProxyType(
            {key: _freeze_bucket(items) for key, items in gap_lists.items()}
        )
        self._alignment_candidate_count = sum(
            len(bucket.items) for bucket in self._guide_buckets.values()
        )
        self._gap_reference_count = sum(
            len(bucket.items) for bucket in self._gap_buckets.values()
        )

    @property
    def snapshot(self) -> GuideSnapshot:
        return self._snapshot

    @property
    def source_count(self) -> int:
        return len(self._snapshot.objects)

    @property
    def alignment_candidate_count(self) -> int:
        return self._alignment_candidate_count

    @property
    def gap_reference_count(self) -> int:
        return self._gap_reference_count

    @staticmethod
    def _append(
        lists: dict[tuple[object, ...], list[_IndexItem]],
        key: tuple[object, ...],
        item: _IndexItem,
    ) -> None:
        lists.setdefault(key, []).append(item)

    @classmethod
    def _add_source_candidates(
        cls,
        lists: dict[tuple[str | None, Axis], list[_IndexItem]],
        source: GuideObject,
    ) -> None:
        bounds = source.bounds
        definitions = (
            (Axis.X, GuideKind.EDGE, bounds.x0, "x_min_edge", (bounds.y0, bounds.y1), 0),
            (Axis.X, GuideKind.EDGE, bounds.x1, "x_max_edge", (bounds.y0, bounds.y1), 1),
            (Axis.X, GuideKind.CENTER, bounds.center_x, "x_center", (bounds.y0, bounds.y1), 2),
            (Axis.Y, GuideKind.EDGE, bounds.y0, "y_min_edge", (bounds.x0, bounds.x1), 0),
            (Axis.Y, GuideKind.EDGE, bounds.y1, "y_max_edge", (bounds.x0, bounds.x1), 1),
            (Axis.Y, GuideKind.CENTER, bounds.center_y, "y_center", (bounds.x0, bounds.x1), 2),
        )
        for axis, kind, position, feature, span, feature_order in definitions:
            cls._append(
                lists,
                (source.scope_id, axis),
                _AxisCandidate(
                    axis,
                    kind,
                    position,
                    source.stable_id,
                    feature,
                    span,
                    source.z_order,
                    source.order,
                    feature_order,
                ),
            )
        if source.baseline_y is not None:
            cls._append(
                lists,
                (source.scope_id, Axis.Y),
                _AxisCandidate(
                    Axis.Y,
                    GuideKind.BASELINE,
                    source.baseline_y,
                    source.stable_id,
                    "baseline",
                    (bounds.x0, bounds.x1),
                    source.z_order,
                    source.order,
                    3,
                ),
            )
        for index, anchor in enumerate(source.anchors):
            suffix = anchor.name or str(index)
            cls._append(
                lists,
                (source.scope_id, Axis.X),
                _AxisCandidate(
                    Axis.X,
                    GuideKind.ANCHOR,
                    anchor.x,
                    source.stable_id,
                    f"anchor:{suffix}:x",
                    (anchor.y, anchor.y),
                    source.z_order,
                    source.order,
                    4 + index,
                ),
            )
            cls._append(
                lists,
                (source.scope_id, Axis.Y),
                _AxisCandidate(
                    Axis.Y,
                    GuideKind.ANCHOR,
                    anchor.y,
                    source.stable_id,
                    f"anchor:{suffix}:y",
                    (anchor.x, anchor.x),
                    source.z_order,
                    source.order,
                    4 + index,
                ),
            )

    @classmethod
    def _add_gap_references(
        cls,
        lists: dict[tuple[str | None, Axis, FeatureKind], list[_IndexItem]],
        sources: tuple[GuideObject, ...],
    ) -> None:
        by_scope: dict[str | None, list[GuideObject]] = {}
        for source in sources:
            by_scope.setdefault(source.scope_id, []).append(source)

        for scope_id, scoped_sources in by_scope.items():
            for axis in (Axis.X, Axis.Y):
                for first, second in _nearest_gap_pairs(scoped_sources, axis):
                    first_max = first.bounds.maximum(axis)
                    second_min = second.bounds.minimum(axis)
                    gap = second_min - first_max
                    assert gap > 0.0
                    assert _intervals_overlap(
                        first.bounds.cross_interval(axis),
                        second.bounds.cross_interval(axis),
                    )
                    source_ids = (first.stable_id, second.stable_id)
                    common = {
                        "axis": axis,
                        "source_ids": source_ids,
                        "source_bounds": (first.bounds, second.bounds),
                        "existing_gap_px": gap,
                        "z_order": max(first.z_order, second.z_order),
                        "order": max(first.order, second.order),
                        "stable_key": "\x1f".join(sorted(source_ids)),
                    }
                    references = (
                        _GapReference(
                            position_px=first.bounds.minimum(axis) - gap,
                            target_feature=FeatureKind.MAX_EDGE,
                            mode=_GapMode.BEFORE,
                            **common,
                        ),
                        _GapReference(
                            position_px=(first_max + second_min) * 0.5,
                            target_feature=FeatureKind.CENTER,
                            mode=_GapMode.BETWEEN,
                            **common,
                        ),
                        _GapReference(
                            position_px=second.bounds.maximum(axis) + gap,
                            target_feature=FeatureKind.MIN_EDGE,
                            mode=_GapMode.AFTER,
                            **common,
                        ),
                    )
                    for reference in references:
                        cls._append(
                            lists,
                            (scope_id, axis, reference.target_feature),
                            reference,
                        )

    def query(
        self,
        moving: MovingGeometry,
        *,
        tolerance_px: float = 5.0,
        expected_fingerprint: str | None = None,
        include_equal_gaps: bool = True,
        axes: frozenset[Axis] | None = None,
    ) -> SnapPlan:
        if not isinstance(moving, MovingGeometry):
            raise TypeError("moving must be MovingGeometry")
        tolerance = float(tolerance_px)
        if not math.isfinite(tolerance) or tolerance < 0.0:
            raise ValueError("tolerance_px must be a finite non-negative display-pixel value")
        if expected_fingerprint is not None:
            self._snapshot.require_fingerprint(expected_fingerprint)
        if axes is not None:
            if not isinstance(axes, frozenset) or any(
                not isinstance(axis, Axis) for axis in axes
            ):
                raise TypeError("axes must be a frozenset of Axis values or None")

        winners_by_axis: dict[Axis, _Proposal] = {}
        winner_keys: dict[
            Axis, tuple[float, int, float, int, str, int, str, str, float]
        ] = {}
        examined = 0

        def consider(proposal: _Proposal) -> None:
            proposal_key = _proposal_sort_key(proposal)
            current_key = winner_keys.get(proposal.axis)
            if current_key is None or proposal_key < current_key:
                winners_by_axis[proposal.axis] = proposal
                winner_keys[proposal.axis] = proposal_key

        for target in _target_features(moving):
            if axes is not None and target.axis not in axes:
                continue
            # Illustrator treats every moving semantic feature as a point on
            # the guide axis: an edge may meet a centre, a text anchor may meet
            # an edge, and a baseline may meet an ordinary feature.  Every
            # source semantic remains on its candidate, but one merged bucket
            # per axis needs only one binary range lookup per moving feature.
            bucket = self._guide_buckets.get((moving.scope_id, target.axis))
            if bucket is None:
                continue
            start, stop = bucket.range(
                target.position_px - tolerance,
                target.position_px + tolerance,
            )
            examined += stop - start
            for index in range(start, stop):
                raw_candidate = bucket.items[index]
                candidate = raw_candidate
                assert isinstance(candidate, _AxisCandidate)
                delta = candidate.position_px - target.position_px
                consider(
                    _Proposal(
                        target.axis,
                        candidate.kind,
                        _normalise_zero(delta),
                        (candidate.source_id,),
                        candidate.position_px,
                        target.position_px,
                        candidate.source_feature,
                        target.label,
                        candidate.z_order,
                        candidate.order,
                        candidate.source_id,
                        candidate.feature_order * 1000 + target.order,
                        source_span_px=candidate.source_span_px,
                        target_span_px=target.span_px,
                    )
                )

        if include_equal_gaps:
            for axis in (Axis.X, Axis.Y):
                if axes is not None and axis not in axes:
                    continue
                for target_feature, target_position in (
                    (FeatureKind.MIN_EDGE, moving.bounds.minimum(axis)),
                    (FeatureKind.CENTER, moving.bounds.center(axis)),
                    (FeatureKind.MAX_EDGE, moving.bounds.maximum(axis)),
                ):
                    bucket = self._gap_buckets.get(
                        (moving.scope_id, axis, target_feature)
                    )
                    if bucket is None:
                        continue
                    start, stop = bucket.range(
                        target_position - tolerance,
                        target_position + tolerance,
                    )
                    examined += stop - start
                    for index in range(start, stop):
                        raw_reference = bucket.items[index]
                        reference = raw_reference
                        assert isinstance(reference, _GapReference)
                        if not self._gap_is_applicable(reference, moving):
                            continue
                        gap = reference.existing_gap_px
                        if reference.mode is _GapMode.BETWEEN:
                            gap = (gap - moving.bounds.size(axis)) * 0.5
                        delta = reference.position_px - target_position
                        consider(
                            _Proposal(
                                axis,
                                GuideKind.EQUAL_GAP,
                                _normalise_zero(delta),
                                reference.source_ids,
                                reference.position_px,
                                target_position,
                                reference.mode.value,
                                target_feature.value,
                                reference.z_order,
                                reference.order,
                                reference.stable_key,
                                0,
                                gap_reference=reference,
                                gap_px=_normalise_zero(gap),
                            )
                        )

        winners = [
            winners_by_axis[axis]
            for axis in (Axis.X, Axis.Y)
            if axis in winners_by_axis
        ]

        delta_by_axis = {winner.axis: winner.delta_px for winner in winners}
        delta_px = (
            delta_by_axis.get(Axis.X, 0.0),
            delta_by_axis.get(Axis.Y, 0.0),
        )
        hits = tuple(
            SnapHit(
                winner.axis,
                winner.kind,
                winner.delta_px,
                winner.source_ids,
                winner.source_position_px,
                winner.target_position_px,
                winner.source_feature,
                winner.target_feature,
                winner.gap_px,
            )
            for winner in winners
        )
        final_bounds = moving.bounds.translated(delta_px)
        overlays = tuple(
            _proposal_overlay(winner, final_bounds, delta_px) for winner in winners
        )
        return SnapPlan(
            self._snapshot.fingerprint,
            delta_px,
            hits,
            overlays,
            examined,
        )

    @staticmethod
    def _gap_is_applicable(
        reference: _GapReference, moving: MovingGeometry
    ) -> bool:
        axis = reference.axis
        moving_cross = moving.bounds.cross_interval(axis)
        if not all(
            _intervals_overlap(moving_cross, bounds.cross_interval(axis))
            for bounds in reference.source_bounds
        ):
            return False
        if reference.mode is _GapMode.BETWEEN:
            return moving.bounds.size(axis) <= reference.existing_gap_px
        return True


def _normalise_zero(value: float) -> float:
    return 0.0 if value == 0.0 else value


def _proposal_sort_key(
    proposal: _Proposal,
) -> tuple[float, int, float, int, str, int, str, str, float]:
    return (
        abs(proposal.delta_px),
        _GUIDE_PRIORITY[proposal.kind],
        -proposal.z_order,
        -proposal.order,
        proposal.stable_key,
        proposal.feature_order,
        proposal.source_feature,
        proposal.target_feature,
        proposal.delta_px,
    )


def _proposal_overlay(
    proposal: _Proposal,
    final_bounds: DisplayBounds,
    delta_px: tuple[float, float],
) -> OverlayPrimitive:
    if proposal.gap_reference is not None:
        reference = proposal.gap_reference
        first, second = reference.source_bounds
        axis = proposal.axis
        if reference.mode is _GapMode.AFTER:
            intervals = (
                (first.maximum(axis), second.minimum(axis)),
                (second.maximum(axis), final_bounds.minimum(axis)),
            )
        elif reference.mode is _GapMode.BEFORE:
            intervals = (
                (final_bounds.maximum(axis), first.minimum(axis)),
                (first.maximum(axis), second.minimum(axis)),
            )
        else:
            intervals = (
                (first.maximum(axis), final_bounds.minimum(axis)),
                (final_bounds.maximum(axis), second.minimum(axis)),
            )
        cross_position = max(
            final_bounds.cross_interval(axis)[1],
            first.cross_interval(axis)[1],
            second.cross_interval(axis)[1],
        )
        return EqualGapOverlay(
            axis,
            (
                (min(intervals[0]), max(intervals[0])),
                (min(intervals[1]), max(intervals[1])),
            ),
            cross_position,
            proposal.gap_px if proposal.gap_px is not None else 0.0,
            proposal.source_ids,
        )

    assert proposal.source_span_px is not None
    assert proposal.target_span_px is not None
    orthogonal_delta = delta_px[1] if proposal.axis is Axis.X else delta_px[0]
    moved_target_span = (
        proposal.target_span_px[0] + orthogonal_delta,
        proposal.target_span_px[1] + orthogonal_delta,
    )
    span = (
        min(proposal.source_span_px[0], moved_target_span[0]),
        max(proposal.source_span_px[1], moved_target_span[1]),
    )
    return GuideLine(
        proposal.axis,
        proposal.source_position_px,
        span,
        proposal.kind,
        proposal.source_ids,
    )


__all__ = [
    "Axis",
    "DisplayBounds",
    "EqualGapOverlay",
    "ExplicitAnchor",
    "FeatureKind",
    "GuideCandidateIndex",
    "GuideKind",
    "GuideLine",
    "GuideObject",
    "GuideSnapshot",
    "MovingGeometry",
    "OverlayPrimitive",
    "SmartGuideQueryIndex",
    "SnapHit",
    "SnapPlan",
    "StaleGuideSnapshotError",
]
