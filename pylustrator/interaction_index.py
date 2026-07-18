"""Fail-open display-space acceleration for pointer hit testing.

The index deliberately knows nothing about Matplotlib hit semantics.  It only
returns a conservative subset of source indices whose expanded display bounds
cover a query point.  Callers must still run their authoritative adapter or
native hit test for every returned item.

Items without a provably bounded hit envelope are kept in ``always`` and are
therefore tested for every query.  A failed build never publishes a partial
index: :meth:`candidate_indices` returns ``None`` so the caller can perform its
original full scan.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from math import floor
from typing import Optional

import numpy as np
from matplotlib.artist import Artist


BoundsProvider = Callable[[Artist], Optional[Sequence[float]]]


class _IncrementalDisplaySpaceHitIndexBuild:
    """Private, conservative staging area for one idle index build.

    Bounds are accumulated without publishing a partial index.  Pointer input
    that arrives before :meth:`finish` sees every unmeasured Artist as an
    always-test candidate, so correctness never depends on the idle callback
    winning a race with user input.
    """

    def __init__(
        self,
        owner: "DisplaySpaceHitIndex",
        *,
        artist_count: int,
        revision: int,
        source_ids: tuple[int, ...],
    ) -> None:
        self._owner = owner
        self.artist_count = int(artist_count)
        self.revision = int(revision)
        self.source_ids = source_ids
        self.next_index = 0
        self.cells: dict[tuple[int, int], list[int]] = {}
        self.always: list[int] = []
        self.active = True

    @property
    def complete(self) -> bool:
        return self.next_index == self.artist_count

    def matches(
        self,
        artists: Sequence[Artist],
        *,
        revision: int,
        source_ids: tuple[int, ...],
    ) -> bool:
        return (
            self.active
            and self.artist_count == len(artists)
            and self.revision == int(revision)
            and (
                self.source_ids is source_ids
                or self.source_ids == source_ids
            )
        )

    def add_bounds(self, bounds: Optional[Sequence[float]]) -> bool:
        """Stage the next roster entry, failing the whole build atomically."""

        if not self.active or self.complete:
            return False
        index = self.next_index
        try:
            self._owner._insert_bounds(self.cells, self.always, index, bounds)
        except Exception:
            self._owner._fail_incremental_build(self)
            return False
        self.next_index += 1
        return True

    def candidate_indices(self, x: float, y: float) -> Sequence[int]:
        """Conservatively query staged cells plus every unmeasured Artist."""

        if self.next_index == 0:
            return range(self.artist_count)
        cell = (
            floor(float(x) / self._owner.cell_size),
            floor(float(y) / self._owner.cell_size),
        )
        found = set(self.always)
        found.update(self.cells.get(cell, ()))
        return (*sorted(found), *range(self.next_index, self.artist_count))

    def candidate_indices_for_bounds(
        self, x0: float, y0: float, x1: float, y1: float
    ) -> Sequence[int]:
        """Conservatively query staged cells over a display-space rectangle."""

        ix0, iy0 = (
            floor(float(x0) / self._owner.cell_size),
            floor(float(y0) / self._owner.cell_size),
        )
        ix1, iy1 = (
            floor(float(x1) / self._owner.cell_size),
            floor(float(y1) / self._owner.cell_size),
        )
        cell_count = (ix1 - ix0 + 1) * (iy1 - iy0 + 1)
        if cell_count > self._owner.max_query_cells or self.next_index == 0:
            return range(self.artist_count)
        found = set(self.always)
        for ix in range(ix0, ix1 + 1):
            for iy in range(iy0, iy1 + 1):
                found.update(self.cells.get((ix, iy), ()))
        return (*sorted(found), *range(self.next_index, self.artist_count))

    def finish(self) -> bool:
        """Atomically publish a complete still-current staging build."""

        return self._owner._finish_incremental_build(self)

    def cancel(self) -> None:
        self._owner._cancel_incremental_build(self)


class DisplaySpaceHitIndex:
    """A revisioned uniform-grid index with conservative fail-open behavior."""

    def __init__(
        self,
        *,
        cell_size: float = 64.0,
        max_cells_per_artist: int = 4096,
        max_query_cells: int = 65536,
    ) -> None:
        cell_size = float(cell_size)
        if not np.isfinite(cell_size) or cell_size <= 0.0:
            raise ValueError("cell_size must be finite and positive")
        if int(max_cells_per_artist) <= 0:
            raise ValueError("max_cells_per_artist must be positive")
        if int(max_query_cells) <= 0:
            raise ValueError("max_query_cells must be positive")
        self.cell_size = cell_size
        self.max_cells_per_artist = int(max_cells_per_artist)
        self.max_query_cells = int(max_query_cells)
        self._built_revision: Optional[int] = None
        self._source_ids: tuple[int, ...] = ()
        self._cells: dict[tuple[int, int], tuple[int, ...]] = {}
        self._always: tuple[int, ...] = ()
        self._failed_key: Optional[tuple[int, tuple[int, ...]]] = None
        self._pending_build: _IncrementalDisplaySpaceHitIndexBuild | None = None

    @property
    def built_revision(self) -> Optional[int]:
        return self._built_revision

    @property
    def always_count(self) -> int:
        return len(self._always)

    def invalidate(self) -> None:
        """Make the current snapshot unavailable until an atomic rebuild."""

        pending = self._pending_build
        if pending is not None:
            pending.active = False
        self._pending_build = None
        self._built_revision = None
        self._source_ids = ()
        self._cells = {}
        self._always = ()
        self._failed_key = None

    @staticmethod
    def _normalize_bounds(bounds: Sequence[float]) -> tuple[float, float, float, float]:
        values = np.asarray(bounds, dtype=float)
        if values.shape == (2, 2):
            values = values.reshape(4)
        elif values.shape != (4,):
            raise ValueError("interaction bounds must have shape (4,) or (2, 2)")
        if not np.all(np.isfinite(values)):
            raise ValueError("interaction bounds must be finite")
        x0, y0, x1, y1 = (float(value) for value in values)
        if x1 < x0 or y1 < y0:
            raise ValueError("interaction bounds must be ordered")
        return x0, y0, x1, y1

    def _insert_bounds(
        self,
        cells: dict[tuple[int, int], list[int]],
        always: list[int],
        index: int,
        bounds: Optional[Sequence[float]],
    ) -> None:
        """Insert one conservative envelope into local, unpublished storage."""

        if bounds is None:
            always.append(index)
            return
        x0, y0, x1, y1 = self._normalize_bounds(bounds)
        ix0, iy0 = floor(x0 / self.cell_size), floor(y0 / self.cell_size)
        ix1, iy1 = floor(x1 / self.cell_size), floor(y1 / self.cell_size)
        cell_count = (ix1 - ix0 + 1) * (iy1 - iy0 + 1)
        if cell_count > self.max_cells_per_artist:
            always.append(index)
            return
        for ix in range(ix0, ix1 + 1):
            for iy in range(iy0, iy1 + 1):
                cells.setdefault((ix, iy), []).append(index)

    @staticmethod
    def _validated_source_ids(
        artists: Sequence[Artist], source_ids: tuple[int, ...] | None
    ) -> tuple[int, ...] | None:
        if source_ids is None:
            return tuple(id(artist) for artist in artists)
        return source_ids if len(source_ids) == len(artists) else None

    def is_current(self, *, revision: int, source_ids: tuple[int, ...]) -> bool:
        """Return whether the published index matches these truth tokens."""

        return self._built_revision == int(revision) and (
            self._source_ids is source_ids or self._source_ids == source_ids
        )

    def begin_incremental_build(
        self,
        artists: Sequence[Artist],
        *,
        revision: int,
        source_ids: tuple[int, ...] | None = None,
    ) -> _IncrementalDisplaySpaceHitIndexBuild | None:
        """Start or reuse one non-published build for bounded idle work."""

        validated_ids = self._validated_source_ids(artists, source_ids)
        if validated_ids is None or self.is_current(
            revision=revision, source_ids=validated_ids
        ):
            return None
        key = int(revision), validated_ids
        if self._failed_key == key:
            return None
        pending = self._pending_build
        if pending is not None and pending.matches(
            artists, revision=revision, source_ids=validated_ids
        ):
            return pending
        if pending is not None:
            pending.active = False
        pending = _IncrementalDisplaySpaceHitIndexBuild(
            self,
            artist_count=len(artists),
            revision=revision,
            source_ids=validated_ids,
        )
        self._pending_build = pending
        return pending

    def _cancel_incremental_build(
        self, build: _IncrementalDisplaySpaceHitIndexBuild
    ) -> None:
        build.active = False
        if self._pending_build is build:
            self._pending_build = None

    def _fail_incremental_build(
        self, build: _IncrementalDisplaySpaceHitIndexBuild
    ) -> None:
        if self._pending_build is build:
            self._pending_build = None
            self._failed_key = build.revision, build.source_ids
        build.active = False

    def _finish_incremental_build(
        self, build: _IncrementalDisplaySpaceHitIndexBuild
    ) -> bool:
        if (
            self._pending_build is not build
            or not build.active
            or not build.complete
        ):
            return False
        self._cells = {
            key: tuple(indices) for key, indices in build.cells.items()
        }
        self._always = tuple(build.always)
        self._source_ids = build.source_ids
        self._built_revision = build.revision
        self._failed_key = None
        self._pending_build = None
        build.active = False
        build.cells = {}
        build.always = []
        return True

    def _build(
        self,
        artists: Sequence[Artist],
        *,
        revision: int,
        source_ids: tuple[int, ...],
        bounds_provider: BoundsProvider,
    ) -> None:
        """Build into locals and publish only after every item succeeds."""

        cells: dict[tuple[int, int], list[int]] = {}
        always: list[int] = []
        for index, artist in enumerate(artists):
            self._insert_bounds(
                cells, always, index, bounds_provider(artist)
            )

        # Atomic publication: an exception above leaves no partial snapshot.
        self._cells = {key: tuple(indices) for key, indices in cells.items()}
        self._always = tuple(always)
        self._source_ids = source_ids
        self._built_revision = int(revision)
        self._failed_key = None
        pending = self._pending_build
        if pending is not None:
            pending.active = False
        self._pending_build = None

    def _ensure_current(
        self,
        artists: Sequence[Artist],
        *,
        revision: int,
        source_ids: tuple[int, ...],
        bounds_provider: BoundsProvider,
    ) -> bool:
        """Build once for the supplied allocation-stable roster identity."""

        revision = int(revision)
        key = revision, source_ids
        current = self._built_revision == revision and self._source_ids is source_ids
        if not current:
            # External callers are allowed to provide an equal tuple rather
            # than the exact cached object.  Manager hot paths always reuse the
            # identical object and avoid this potentially large comparison.
            current = (
                self._built_revision == revision and self._source_ids == source_ids
            )
        if current:
            return True
        pending = self._pending_build
        if pending is not None:
            if pending.matches(
                artists, revision=revision, source_ids=source_ids
            ):
                return False
            pending.active = False
            self._pending_build = None
        if self._failed_key == key:
            return False
        try:
            self._build(
                artists,
                revision=revision,
                source_ids=source_ids,
                bounds_provider=bounds_provider,
            )
        except Exception:
            # Never retain or query a partial/stale snapshot after failure.
            self._built_revision = None
            self._source_ids = ()
            self._cells = {}
            self._always = ()
            self._failed_key = key
            return False
        return True

    def candidate_indices(
        self,
        x: float,
        y: float,
        artists: Sequence[Artist],
        *,
        revision: int,
        bounds_provider: BoundsProvider,
        source_ids: tuple[int, ...] | None = None,
    ) -> Optional[Sequence[int]]:
        """Return conservative source indices, or ``None`` for a full scan."""

        try:
            x, y = float(x), float(y)
        except (TypeError, ValueError):
            return None
        if not np.isfinite(x) or not np.isfinite(y):
            return None

        validated_ids = self._validated_source_ids(artists, source_ids)
        if validated_ids is None:
            return None
        pending = self._pending_build
        if pending is not None and pending.matches(
            artists, revision=revision, source_ids=validated_ids
        ):
            return pending.candidate_indices(x, y)

        if not self._ensure_current(
            artists,
            revision=revision,
            source_ids=validated_ids,
            bounds_provider=bounds_provider,
        ):
            return None

        cell = (floor(x / self.cell_size), floor(y / self.cell_size))
        local = self._cells.get(cell, ())
        if not self._always:
            return local
        if not local:
            return self._always
        # Indexed and always-tested items are disjoint.  Source-order sorting
        # makes the result deterministic and keeps downstream stable ordering.
        return tuple(sorted((*self._always, *local)))

    def candidate_indices_for_bounds(
        self,
        x0: float,
        y0: float,
        x1: float,
        y1: float,
        artists: Sequence[Artist],
        *,
        revision: int,
        bounds_provider: BoundsProvider,
        source_ids: tuple[int, ...] | None = None,
    ) -> Optional[Sequence[int]]:
        """Return a conservative roster subset intersecting a display bbox."""

        try:
            x0, x1 = sorted((float(x0), float(x1)))
            y0, y1 = sorted((float(y0), float(y1)))
        except (TypeError, ValueError):
            return None
        if not np.all(np.isfinite((x0, y0, x1, y1))):
            return None
        validated_ids = self._validated_source_ids(artists, source_ids)
        if validated_ids is None:
            return None
        pending = self._pending_build
        if pending is not None and pending.matches(
            artists, revision=revision, source_ids=validated_ids
        ):
            return pending.candidate_indices_for_bounds(x0, y0, x1, y1)
        if not self._ensure_current(
            artists,
            revision=revision,
            source_ids=validated_ids,
            bounds_provider=bounds_provider,
        ):
            return None

        ix0, iy0 = floor(x0 / self.cell_size), floor(y0 / self.cell_size)
        ix1, iy1 = floor(x1 / self.cell_size), floor(y1 / self.cell_size)
        cell_count = (ix1 - ix0 + 1) * (iy1 - iy0 + 1)
        if cell_count > self.max_query_cells:
            return range(len(artists))
        found = set(self._always)
        for ix in range(ix0, ix1 + 1):
            for iy in range(iy0, iy1 + 1):
                found.update(self._cells.get((ix, iy), ()))
        return tuple(sorted(found))
