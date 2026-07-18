"""Revision-bound display geometry shared by editor interaction consumers.

The manager's interaction revision is the sole dirty-state owner.  A renderer
identity guard prevents reuse if a backend swaps renderers before a draw event,
and an immutable :class:`ArtistRoster` prevents every warm pointer query from
rebuilding large Artist/id tuples.  Geometry failures are cached as ``None``;
callers must treat them as always-test/fail-open rather than as absent paint.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator, Sequence

import numpy as np
from matplotlib.artist import Artist

from .artist_adapters import selection_geometry_snapshot
from .snap import TargetWrapper


Bounds = tuple[float, float, float, float]


@dataclass(frozen=True)
class ArtistRoster:
    """One immutable Artist inventory and its allocation-stable identities."""

    artists: tuple[Artist, ...]
    source_ids: tuple[int, ...]

    @classmethod
    def capture(cls, artists: Sequence[Artist]) -> "ArtistRoster":
        captured = tuple(artists)
        return cls(captured, tuple(id(artist) for artist in captured))


class DisplayGeometryCache:
    """Selection geometry measured at most once per Artist and revision."""

    def __init__(self) -> None:
        self.revision: int | None = None
        self.roster: ArtistRoster | None = None
        self.renderer = None
        self._selection_cache: dict = {}
        self._selection_bounds: dict[int, tuple[Artist, Bounds | None]] = {}

    def bind(
        self,
        *,
        revision: int,
        roster: ArtistRoster,
        renderer,
    ) -> "DisplayGeometryCache":
        """Bind to the manager-owned truth tokens, clearing stale geometry."""

        revision = int(revision)
        if (
            self.revision != revision
            or self.roster is not roster
            or self.renderer is not renderer
        ):
            self.revision = revision
            self.roster = roster
            self.renderer = renderer
            self._selection_cache = {}
            self._selection_bounds = {}
        return self

    def invalidate(self) -> None:
        """Release geometry immediately when the manager advances dirty state."""

        self.revision = None
        self.roster = None
        self.renderer = None
        self._selection_cache = {}
        self._selection_bounds = {}

    @contextmanager
    def snapshot(self) -> Iterator[None]:
        """Make adapter measurements participate in this revision snapshot."""

        with selection_geometry_snapshot(self._selection_cache):
            yield

    def selection_bounds(self, artist: Artist) -> Bounds | None:
        """Return finite visible bounds, or ``None`` for fail-open handling."""

        key = id(artist)
        cached = self._selection_bounds.get(key)
        if cached is not None and cached[0] is artist:
            return cached[1]
        bounds = None
        try:
            with self.snapshot():
                points = np.asarray(
                    TargetWrapper(artist).get_selection_points(), dtype=float
                )
            if points.ndim == 2 and points.shape[1] >= 2:
                points = points[:, :2]
                points = points[np.all(np.isfinite(points), axis=1)]
                if len(points):
                    low = np.min(points, axis=0)
                    high = np.max(points, axis=0)
                    bounds = (
                        float(low[0]),
                        float(low[1]),
                        float(high[0]),
                        float(high[1]),
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
            bounds = None
        self._selection_bounds[key] = artist, bounds
        return bounds


__all__ = ["ArtistRoster", "Bounds", "DisplayGeometryCache"]
