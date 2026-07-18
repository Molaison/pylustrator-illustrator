"""Revision-bound display geometry shared by editor interaction consumers.

The manager's interaction revision is the sole dirty-state owner.  A renderer
identity guard prevents reuse if a backend swaps renderers before a draw event,
and an immutable :class:`ArtistRoster` prevents every warm pointer query from
rebuilding large Artist/id tuples.  Geometry failures are cached as ``None``;
callers must treat them as always-test/fail-open rather than as absent paint.
"""

from __future__ import annotations

from copy import deepcopy
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from types import BuiltinFunctionType, FunctionType, MethodType
from typing import Iterator, Sequence

import numpy as np
import matplotlib.patheffects as mpl_path_effects
from matplotlib.artist import Artist
from matplotlib.backends.backend_agg import RendererAgg
from matplotlib.collections import Collection, PathCollection
from matplotlib.legend import Legend
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from .artist_adapters import selection_geometry_snapshot
from .snap import TargetWrapper


Bounds = tuple[float, float, float, float]
DEFAULT_PAINT_RASTER_BUDGET_BYTES = 64 * 1024 * 1024


class PaintEnvelopeAccuracy(str, Enum):
    """How strongly a :class:`PaintEnvelope` constrains visible paint."""

    EXACT = "exact"
    CONSERVATIVE = "conservative"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class PaintEnvelope:
    """A revision-bound visible-paint observation.

    ``bounds is None`` is exact only when ``accuracy`` is :attr:`EXACT`; in
    that case Agg observed no non-zero alpha.  Consumers which authorize a
    mutation must require :attr:`EXACT`.  Conservative and unavailable results
    are only suitable for fail-open candidate filtering.
    """

    bounds: Bounds | None
    accuracy: PaintEnvelopeAccuracy
    reason: str | None = None

    @property
    def exact(self) -> bool:
        return self.accuracy is PaintEnvelopeAccuracy.EXACT


_AGG_PAINT_ARTIST_TYPES = (Patch, Line2D, Collection)
_EXACT_PATH_EFFECT_TYPES = (
    mpl_path_effects.Normal,
    mpl_path_effects.Stroke,
    mpl_path_effects.withStroke,
    mpl_path_effects.SimplePatchShadow,
    mpl_path_effects.withSimplePatchShadow,
    mpl_path_effects.SimpleLineShadow,
)
_EXACT_DRAW_METHODS = (
    Patch.draw,
    Line2D.draw,
    Collection.draw,
    # PathCollection and PolyCollection inherit this validated
    # _CollectionWithSizes draw wrapper.
    PathCollection.draw,
)


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
    """Revision-bound analytic geometry and opt-in Agg paint observations.

    Selection geometry remains cheap and lazy.  Renderer-faithful measurement
    is deliberately split into :meth:`capture_paint_envelope`, which may draw
    one Artist into a private transparent Agg renderer, and
    :meth:`paint_envelope`, which is a cache-only lookup.  Pointer code must use
    only the latter; a cache miss is conservative and must stay in the native
    hit-test candidate set.
    """

    def __init__(
        self,
        *,
        paint_raster_budget_bytes: int = DEFAULT_PAINT_RASTER_BUDGET_BYTES,
    ) -> None:
        paint_raster_budget_bytes = int(paint_raster_budget_bytes)
        if paint_raster_budget_bytes < 0:
            raise ValueError("paint_raster_budget_bytes must be non-negative")
        self.paint_raster_budget_bytes = paint_raster_budget_bytes
        self.revision: int | None = None
        self.roster: ArtistRoster | None = None
        self.renderer = None
        self._selection_cache: dict = {}
        self._selection_bounds: dict[int, tuple[Artist, Bounds | None]] = {}
        self._paint_envelopes: dict[int, tuple[Artist, PaintEnvelope]] = {}
        self._paint_renderer: RendererAgg | None = None
        self._paint_roster_lookup: dict[int, Artist] | None = None
        self._paint_legend_owner_inventory: (
            tuple[Artist, dict[int, tuple[Artist, Legend]]] | None
        ) = None

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
            self._paint_envelopes = {}
            self._paint_renderer = None
            self._paint_roster_lookup = None
            self._paint_legend_owner_inventory = None
        return self

    def invalidate(self) -> None:
        """Release geometry immediately when the manager advances dirty state."""

        self.revision = None
        self.roster = None
        self.renderer = None
        self._selection_cache = {}
        self._selection_bounds = {}
        self._paint_envelopes = {}
        self._paint_renderer = None
        self._paint_roster_lookup = None
        self._paint_legend_owner_inventory = None

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

    def paint_envelope(self, artist: Artist) -> PaintEnvelope | None:
        """Return a previously published paint envelope without measuring.

        This method never creates a renderer, calls ``Artist.draw``, reads the
        canvas, or falls back to analytic geometry.  ``None`` means "not
        captured for this revision", not "no visible paint".
        """

        cached = self._paint_envelopes.get(id(artist))
        if cached is None or cached[0] is not artist:
            return None
        return cached[1]

    def paint_bounds(self, artist: Artist) -> Bounds | None:
        """Return cached exact/conservative bounds for fail-open consumers."""

        envelope = self.paint_envelope(artist)
        return None if envelope is None else envelope.bounds

    def capture_paint_envelope(self, artist: Artist) -> PaintEnvelope:
        """Rasterize one supported Artist into transparent Agg and cache it.

        This is an explicit warm/capture operation and must not be called from
        a pointer hot path.  The bound renderer is used only as the authoritative
        size/DPI token; the live canvas buffer is neither redrawn nor read.
        Patch, Line2D, and Collection drawing therefore preserves Agg's real
        clipping, stroke cap/join, markers, antialiasing, and path effects.

        Unsupported or failed draws publish a whole-canvas conservative
        envelope when that is provable.  They never masquerade as exact.
        """

        cached = self.paint_envelope(artist)
        if cached is not None:
            return cached

        if self.revision is None or self.roster is None:
            return PaintEnvelope(
                None,
                PaintEnvelopeAccuracy.UNAVAILABLE,
                "cache-unbound",
            )
        if not self._roster_contains(artist):
            return PaintEnvelope(
                None,
                PaintEnvelopeAccuracy.UNAVAILABLE,
                "artist-not-in-roster",
            )
        # A subclass may override drawing primitives or coordinate semantics;
        # recreating it as a base RendererAgg would no longer be renderer-
        # faithful.  Standard Agg/QtAgg canvases expose RendererAgg directly.
        if type(self.renderer) is not RendererAgg:
            result = PaintEnvelope(
                None,
                PaintEnvelopeAccuracy.UNAVAILABLE,
                "renderer-not-agg",
            )
            self._paint_envelopes[id(artist)] = artist, result
            return result

        raster_spec = self._paint_raster_spec()
        if raster_spec is None:
            result = PaintEnvelope(
                None,
                PaintEnvelopeAccuracy.UNAVAILABLE,
                "renderer-metadata-invalid",
            )
            self._paint_envelopes[id(artist)] = artist, result
            return result
        width, height, dpi = raster_spec
        canvas_bounds: Bounds = (0.0, 0.0, float(width), float(height))

        if getattr(artist, "figure", None) is None:
            result = PaintEnvelope(
                None,
                PaintEnvelopeAccuracy.UNAVAILABLE,
                "artist-detached",
            )
        elif not isinstance(artist, _AGG_PAINT_ARTIST_TYPES):
            result = PaintEnvelope(
                canvas_bounds,
                PaintEnvelopeAccuracy.CONSERVATIVE,
                "artist-type-unsupported",
            )
        elif (draw_reason := self._draw_contract_denial_reason(artist)) is not None:
            result = PaintEnvelope(
                canvas_bounds,
                PaintEnvelopeAccuracy.CONSERVATIVE,
                draw_reason,
            )
        elif (denial_reason := self._paint_state_denial_reason(artist)) is not None:
            result = PaintEnvelope(
                canvas_bounds,
                PaintEnvelopeAccuracy.CONSERVATIVE,
                denial_reason,
            )
        elif (clone_reason := self._clone_state_denial_reason(artist)) is not None:
            result = PaintEnvelope(
                canvas_bounds,
                PaintEnvelopeAccuracy.CONSERVATIVE,
                clone_reason,
            )
        else:
            try:
                paint_ancestors = self._paint_ancestors(artist)
            except (AttributeError, TypeError, ValueError, RuntimeError):
                result = PaintEnvelope(
                    canvas_bounds,
                    PaintEnvelopeAccuracy.CONSERVATIVE,
                    "ancestor-state-unavailable",
                )
            else:
                ancestor_state = self._known_ancestor_paint_state(paint_ancestors)
                if ancestor_state is not None:
                    hidden, ancestor_reason = ancestor_state
                    if hidden:
                        result = PaintEnvelope(None, PaintEnvelopeAccuracy.EXACT)
                    else:
                        result = PaintEnvelope(
                            canvas_bounds,
                            PaintEnvelopeAccuracy.CONSERVATIVE,
                            ancestor_reason,
                        )
                elif width * height * 4 > self.paint_raster_budget_bytes:
                    result = PaintEnvelope(
                        canvas_bounds,
                        PaintEnvelopeAccuracy.CONSERVATIVE,
                        "raster-budget-exceeded",
                    )
                else:
                    result = self._capture_agg_paint_envelope(
                        artist,
                        width=width,
                        height=height,
                        dpi=dpi,
                        canvas_bounds=canvas_bounds,
                        paint_ancestors=paint_ancestors,
                    )

        # Publish only after the draw and alpha reduction have completed.  A
        # pointer lookup can therefore observe either the previous immutable
        # result or a miss, never a partially measured envelope.
        self._paint_envelopes[id(artist)] = artist, result
        return result

    def _roster_contains(self, artist: Artist) -> bool:
        roster = self.roster
        if roster is None:
            return False
        lookup = self._paint_roster_lookup
        if lookup is None:
            lookup = dict(zip(roster.source_ids, roster.artists))
            self._paint_roster_lookup = lookup
        return lookup.get(id(artist)) is artist

    @staticmethod
    def _draw_contract_denial_reason(artist: Artist) -> str | None:
        """Allow only audited Matplotlib primitive draw implementations."""

        instance_state = getattr(artist, "__dict__", {})
        artist_type = type(artist)
        if any(
            name == "draw" or name.startswith("get_") for name in instance_state
        ) or not artist_type.__module__.startswith("matplotlib."):
            return "custom-draw-unsupported"
        if getattr(artist_type, "draw", None) not in _EXACT_DRAW_METHODS:
            return "draw-contract-unsupported"
        return None

    @staticmethod
    def _paint_state_denial_reason(artist: Artist) -> str | None:
        """Reject callbacks whose second draw cannot be proven deterministic."""

        try:
            if isinstance(artist, Collection) and artist.get_array() is not None:
                return "scalar-mappable-unsupported"
            if artist.get_agg_filter() is not None:
                return "agg-filter-unsupported"
            path_effects = artist.get_path_effects() or ()
            transform = artist.get_transform()
            stale = artist.stale
        except (AttributeError, TypeError, ValueError, RuntimeError):
            return "artist-paint-state-unavailable"
        if any(type(effect) not in _EXACT_PATH_EFFECT_TYPES for effect in path_effects):
            return "path-effect-unsupported"
        if not type(transform).__module__.startswith("matplotlib."):
            return "transform-unsupported"
        if stale:
            return "artist-stale"
        return None

    @staticmethod
    def _clone_state_denial_reason(artist: Artist) -> str | None:
        """Reject state whose deepcopy could execute untrusted user code."""

        remaining = 512
        seen: set[int] = set()

        def trusted(value) -> bool:
            nonlocal remaining
            remaining -= 1
            if remaining < 0:
                return False
            if value is None or isinstance(
                value,
                (
                    str,
                    bytes,
                    int,
                    float,
                    complex,
                    bool,
                    FunctionType,
                    BuiltinFunctionType,
                    MethodType,
                    type,
                ),
            ):
                return True
            if isinstance(value, np.ndarray):
                return not value.dtype.hasobject
            identity = id(value)
            if identity in seen:
                return True
            if isinstance(value, (list, tuple, set, frozenset)):
                if len(value) > remaining:
                    return False
                seen.add(identity)
                try:
                    return all(trusted(item) for item in value)
                finally:
                    seen.remove(identity)
            if isinstance(value, dict):
                if len(value) * 2 > remaining:
                    return False
                seen.add(identity)
                try:
                    return all(
                        trusted(key) and trusted(item)
                        for key, item in value.items()
                    )
                finally:
                    seen.remove(identity)
            module = type(value).__module__
            return module.startswith(("matplotlib.", "numpy", "weakref"))

        if all(trusted(value) for value in artist.__dict__.values()):
            return None
        return "clone-state-unsupported"

    @staticmethod
    def _reject_custom_artist_callbacks(artist: Artist) -> None:
        artist_type = type(artist)
        if not artist_type.__module__.startswith("matplotlib.") or any(
            name == "draw" or name.startswith("get_")
            for name in getattr(artist, "__dict__", {})
        ):
            raise TypeError("custom Artist callback")
        get_children = getattr(artist_type, "get_children", None)
        if get_children is not None and not getattr(
            get_children, "__module__", ""
        ).startswith("matplotlib."):
            raise TypeError("custom Artist child callback")

    @classmethod
    def _figure_legends_without_callbacks(cls, figure) -> tuple[Legend, ...]:
        """Read the standard Figure/Axes inventories without calling getters."""

        legends: list[Legend] = []
        seen_legends: set[int] = set()
        seen_owners: set[int] = set()
        owners = [figure]
        while owners:
            owner = owners.pop()
            if id(owner) in seen_owners:
                continue
            seen_owners.add(id(owner))
            cls._reject_custom_artist_callbacks(owner)

            def add(candidate) -> None:
                if isinstance(candidate, Legend) and id(candidate) not in seen_legends:
                    cls._reject_custom_artist_callbacks(candidate)
                    seen_legends.add(id(candidate))
                    legends.append(candidate)

            for candidate in getattr(owner, "legends", ()):
                add(candidate)
            for candidate in getattr(owner, "artists", ()):
                add(candidate)
            for axes in getattr(owner, "axes", ()):
                cls._reject_custom_artist_callbacks(axes)
                add(getattr(axes, "legend_", None))
                for candidate in getattr(axes, "_children", ()):
                    add(candidate)
            owners.extend(getattr(owner, "subfigs", ()))
        return tuple(legends)

    @classmethod
    def _legend_managed_artists_without_callbacks(
        cls, legend: Legend
    ) -> tuple[Artist, ...]:
        managed: list[Artist] = []
        seen = {id(legend)}
        stack = list(legend.get_children())
        while stack:
            child = stack.pop()
            if not isinstance(child, Artist) or id(child) in seen:
                continue
            seen.add(id(child))
            managed.append(child)
            cls._reject_custom_artist_callbacks(child)
            stack.extend(child.get_children())
        return tuple(managed)

    def _legend_owner_without_callbacks(
        self,
        artist: Artist,
        figure: Artist,
    ) -> Legend | None:
        cached = self._paint_legend_owner_inventory
        if cached is None or cached[0] is not figure:
            inventory: dict[int, tuple[Artist, Legend]] = {}
            for legend in self._figure_legends_without_callbacks(figure):
                for child in self._legend_managed_artists_without_callbacks(legend):
                    inventory[id(child)] = child, legend
            cached = figure, inventory
            self._paint_legend_owner_inventory = cached
        entry = cached[1].get(id(artist))
        if entry is None or entry[0] is not artist:
            return None
        return entry[1]

    def _paint_ancestors(self, artist: Artist) -> tuple[Artist, ...]:
        """Return live draw owners without populating editor-side caches."""

        ancestors: list[Artist] = []
        seen = {id(artist)}

        def add(candidate) -> None:
            if isinstance(candidate, Artist) and id(candidate) not in seen:
                seen.add(id(candidate))
                ancestors.append(candidate)

        get_figure = getattr(artist, "get_figure", None)
        if callable(get_figure):
            try:
                figure = get_figure(root=True)
            except TypeError:
                figure = get_figure()
        else:
            figure = getattr(artist, "figure", None)
        if isinstance(figure, Artist):
            add(self._legend_owner_without_callbacks(artist, figure))
        add(getattr(artist, "axes", None))
        add(figure)
        return tuple(ancestors)

    @staticmethod
    def _known_ancestor_paint_state(
        ancestors: Sequence[Artist],
    ) -> tuple[bool, str | None] | None:
        """Resolve visibility/effects inherited from known draw owners."""

        for ancestor in ancestors:
            try:
                if not ancestor.get_visible():
                    return True, None
                if ancestor.get_agg_filter() is not None:
                    return False, "ancestor-agg-filter-unsupported"
            except (AttributeError, TypeError, ValueError, RuntimeError):
                return False, "ancestor-state-unavailable"
        return None

    def _paint_raster_spec(self) -> tuple[int, int, float] | None:
        renderer = self.renderer
        try:
            width_value = float(renderer.width)
            height_value = float(renderer.height)
            dpi = float(renderer.dpi)
        except (AttributeError, TypeError, ValueError, OverflowError):
            return None
        if not np.all(np.isfinite((width_value, height_value, dpi))):
            return None
        width = int(width_value)
        height = int(height_value)
        if width <= 0 or height <= 0 or dpi <= 0.0:
            return None
        return width, height, dpi

    def _capture_agg_paint_envelope(
        self,
        artist: Artist,
        *,
        width: int,
        height: int,
        dpi: float,
        canvas_bounds: Bounds,
        paint_ancestors: Sequence[Artist],
    ) -> PaintEnvelope:
        try:
            renderer = self._paint_renderer
            if (
                renderer is None
                or int(renderer.width) != width
                or int(renderer.height) != height
                or float(renderer.dpi) != dpi
            ):
                renderer = RendererAgg(width, height, dpi)
                self._paint_renderer = renderer
            renderer.clear()
            memo = {}
            figure = getattr(artist, "figure", None)
            axes = getattr(artist, "axes", None)
            if figure is not None:
                memo[id(figure)] = figure
            if axes is not None:
                memo[id(axes)] = axes
            for owner in paint_ancestors:
                memo[id(owner)] = owner
            # Ownership callbacks can close over large live container lists;
            # the disposable clone never needs either callback while drawing.
            for callback_name in ("stale_callback", "_remove_method"):
                callback = getattr(artist, callback_name, None)
                if callback is not None:
                    memo[id(callback)] = None
            clone = deepcopy(artist, memo)
            for descendant in clone.findobj():
                descendant.stale_callback = None
                descendant._remove_method = None
            clone.draw(renderer)
            alpha = np.asarray(renderer.buffer_rgba())[..., 3]
            occupied_x = np.flatnonzero(np.any(alpha, axis=0))
            occupied_y = np.flatnonzero(np.any(alpha, axis=1))
            if not len(occupied_x) or not len(occupied_y):
                bounds = None
            else:
                # Agg's RGBA rows run top-to-bottom while Matplotlib display
                # coordinates use a bottom-left origin.  Bounds denote pixel
                # edges, so the upper/right coordinates are exclusive.
                bounds = (
                    float(occupied_x[0]),
                    float(height - occupied_y[-1] - 1),
                    float(occupied_x[-1] + 1),
                    float(height - occupied_y[0]),
                )
            return PaintEnvelope(bounds, PaintEnvelopeAccuracy.EXACT)
        except Exception:
            # Arbitrary third-party draw/path-effect/filter code can fail in a
            # private renderer even after succeeding as part of a composite
            # Figure draw.  Discard potentially unbalanced filter state and
            # retain the only universally safe visible-paint envelope.
            self._paint_renderer = None
            return PaintEnvelope(
                canvas_bounds,
                PaintEnvelopeAccuracy.CONSERVATIVE,
                "artist-draw-failed",
            )


__all__ = [
    "ArtistRoster",
    "Bounds",
    "DEFAULT_PAINT_RASTER_BUDGET_BYTES",
    "DisplayGeometryCache",
    "PaintEnvelope",
    "PaintEnvelopeAccuracy",
]
