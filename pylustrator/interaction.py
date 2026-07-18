"""Editor interaction primitives independent of Matplotlib mutation code.

The classes in this module deliberately separate visual hit ordering from the
policy that turns a hit into a selection.  Matplotlib ownership is supplied by
callbacks; it is not treated as editor grouping by default.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Iterable, Iterator, Optional

from matplotlib.artist import Artist


class SelectionMode(str, Enum):
    """The two selection tools familiar from vector editors."""

    OBJECT = "object"
    DIRECT = "direct"

    @classmethod
    def coerce(cls, value: "SelectionMode | str") -> "SelectionMode":
        if isinstance(value, cls):
            return value
        return cls(str(value).lower())


@dataclass(frozen=True)
class HitCandidate:
    """One visually hit Artist, ordered from front to back by :class:`HitStack`."""

    artist: Artist
    editable: bool
    draw_key: tuple
    registration_index: int


@dataclass(frozen=True)
class HitStack:
    """All Artists under one pointer position in deterministic visual order."""

    candidates: tuple[HitCandidate, ...] = ()

    def __iter__(self) -> Iterator[HitCandidate]:
        return iter(self.candidates)

    def __len__(self) -> int:
        return len(self.candidates)

    @property
    def artists(self) -> tuple[Artist, ...]:
        return tuple(candidate.artist for candidate in self.candidates)


@dataclass(frozen=True)
class HitResolution:
    """One immutable interpretation of a pointer event's visual hit stack.

    ``raw_leaf`` exposes the foreground editable Artist for legacy callers,
    while ``target`` is the only object canvas selection and dragging may use.
    Keeping both values in one result prevents a later raw-Artist lookup from
    bypassing grouping or isolation policy.
    """

    hit_stack: HitStack
    raw_leaf: Optional[Artist]
    candidates: tuple[Artist, ...]
    target: Optional[Artist]
    blocked: bool


class TopHitStatus(str, Enum):
    """Whether a streamed foreground hit is final or needs the full stack."""

    RESOLVED = "resolved"
    NEEDS_FULL_STACK = "needs_full_stack"


@dataclass(frozen=True)
class TopHitDecision:
    """A short-circuitable interpretation of a front-to-back hit stream.

    Direct Selection cannot decide a group-shell hit without knowing whether a
    descendant leaf appears later in the stream.  In that one case ``status``
    requests the existing full-stack resolver instead of returning a partial
    :class:`HitResolution` whose candidate list would be incomplete.
    """

    status: TopHitStatus
    target: Optional[Artist] = None
    raw_leaf: Optional[Artist] = None
    blocked: bool = False


@dataclass(frozen=True)
class SelectionScope:
    """One entered logical group in the isolation stack."""

    root: Artist
    label: str


class SelectionKernel:
    """Resolve hit stacks using tool mode, logical groups, and isolation scope."""

    def __init__(
        self,
        *,
        parent_of: Callable[[Artist], Optional[Artist]],
        is_group: Callable[[Artist], bool],
        label_of: Optional[Callable[[Artist], str]] = None,
    ) -> None:
        self._parent_of = parent_of
        self._is_group = is_group
        self._label_of = label_of or (lambda artist: type(artist).__name__)
        self.mode = SelectionMode.OBJECT
        self._scopes: list[SelectionScope] = []

    @property
    def scopes(self) -> tuple[SelectionScope, ...]:
        return tuple(self._scopes)

    @property
    def scope_root(self) -> Optional[Artist]:
        return self._scopes[-1].root if self._scopes else None

    @property
    def breadcrumbs(self) -> tuple[str, ...]:
        return tuple(scope.label for scope in self._scopes)

    def set_mode(self, mode: SelectionMode | str) -> SelectionMode:
        self.mode = SelectionMode.coerce(mode)
        return self.mode

    def _ancestors(self, artist: Artist) -> Iterator[Artist]:
        current = self._parent_of(artist)
        seen: set[int] = set()
        while current is not None and id(current) not in seen:
            seen.add(id(current))
            yield current
            current = self._parent_of(current)

    def contains(self, root: Artist, artist: Artist) -> bool:
        return artist is root or any(parent is root for parent in self._ancestors(artist))

    def _in_scope(self, artist: Artist) -> bool:
        root = self.scope_root
        return root is None or self.contains(root, artist)

    def object_target(self, artist: Artist) -> Optional[Artist]:
        """Map a leaf hit to the logical object selected by the active tool."""

        if not self._in_scope(artist):
            return None
        if self.mode is SelectionMode.DIRECT:
            return artist

        root = self.scope_root
        target = artist
        if self._is_group(target) and target is not root:
            return target
        for parent in self._ancestors(artist):
            if parent is root:
                break
            if self._is_group(parent):
                target = parent
        return target

    def candidates(self, hit_stack: HitStack) -> tuple[Artist, ...]:
        """Return selectable logical targets from front to back.

        An unsupported foreground Artist remains a barrier for objects below it,
        matching what the user can see instead of allowing clicks to leak into a
        containing Axes.
        """

        raw_candidates = tuple(hit_stack)
        resolved: list[Artist] = []
        seen: set[int] = set()
        root = self.scope_root
        for candidate in raw_candidates:
            if not self._in_scope(candidate.artist):
                continue
            if root is not None and candidate.artist is root:
                continue
            if not candidate.editable:
                break
            if self.mode is SelectionMode.DIRECT and self._is_group(candidate.artist):
                has_leaf_hit = any(
                    other.editable
                    and not self._is_group(other.artist)
                    and self.contains(candidate.artist, other.artist)
                    for other in raw_candidates
                    if other.artist is not candidate.artist
                )
                # Skip a group shell when a leaf inside it was also hit. An
                # otherwise-empty group background remains a foreground barrier
                # so Direct Selection cannot leak through into the owning Axes.
                if has_leaf_hit:
                    continue
                break
            target = self.object_target(candidate.artist)
            if target is None or id(target) in seen:
                continue
            resolved.append(target)
            seen.add(id(target))
        return tuple(resolved)

    def pick(
        self,
        hit_stack: HitStack,
        *,
        cycle_from: Optional[Artist] = None,
        wrap: bool = False,
    ) -> Optional[Artist]:
        candidates = self.candidates(hit_stack)
        return self._pick_candidate(candidates, cycle_from=cycle_from, wrap=wrap)

    @staticmethod
    def _pick_candidate(
        candidates: tuple[Artist, ...],
        *,
        cycle_from: Optional[Artist] = None,
        wrap: bool = False,
    ) -> Optional[Artist]:
        if not candidates:
            return None
        if cycle_from is None or cycle_from not in candidates:
            return candidates[0]
        index = candidates.index(cycle_from) + 1
        if index < len(candidates):
            return candidates[index]
        return candidates[0] if wrap else None

    def resolve(
        self,
        hit_stack: HitStack,
        *,
        cycle_from: Optional[Artist] = None,
        wrap: bool = False,
    ) -> HitResolution:
        """Resolve one already-built stack without performing another hit test."""

        root = self.scope_root
        raw_leaf = None
        raw_barrier = False
        for candidate in hit_stack:
            if not self._in_scope(candidate.artist):
                continue
            if root is not None and candidate.artist is root:
                continue
            if not candidate.editable:
                raw_barrier = True
                break
            raw_leaf = candidate.artist
            break

        candidates = self.candidates(hit_stack)
        target = self._pick_candidate(
            candidates,
            cycle_from=cycle_from,
            wrap=wrap,
        )
        return HitResolution(
            hit_stack=hit_stack,
            raw_leaf=raw_leaf,
            candidates=candidates,
            target=target,
            blocked=bool(raw_barrier or (raw_leaf is not None and not candidates)),
        )

    def resolve_top(
        self,
        candidates: Iterable[HitCandidate],
    ) -> TopHitDecision:
        """Resolve the first conclusive item in a front-to-back hit stream.

        ``candidates`` must contain only actual visual hits in paint order.  The
        iterator is consumed only until selection policy becomes conclusive.
        Callers must build a complete :class:`HitStack` when the returned status
        is :attr:`TopHitStatus.NEEDS_FULL_STACK`.
        """

        root = self.scope_root
        for candidate in candidates:
            artist = candidate.artist
            if not self._in_scope(artist):
                continue
            if root is not None and artist is root:
                continue
            if not candidate.editable:
                return TopHitDecision(
                    status=TopHitStatus.RESOLVED,
                    blocked=True,
                )
            if self.mode is SelectionMode.DIRECT and self._is_group(artist):
                return TopHitDecision(
                    status=TopHitStatus.NEEDS_FULL_STACK,
                    raw_leaf=artist,
                )
            return TopHitDecision(
                status=TopHitStatus.RESOLVED,
                target=self.object_target(artist),
                raw_leaf=artist,
            )

        return TopHitDecision(status=TopHitStatus.RESOLVED)

    def map_artists(self, artists: Iterable[Artist]) -> list[Artist]:
        """Resolve an unordered marquee result through the active tool mode."""

        result: list[Artist] = []
        seen: set[int] = set()
        for artist in artists:
            if self.mode is SelectionMode.DIRECT and self._is_group(artist):
                continue
            target = self.object_target(artist)
            if target is None or id(target) in seen:
                continue
            result.append(target)
            seen.add(id(target))
        return result

    def enter_isolation(self, root: Artist) -> bool:
        if not self._is_group(root):
            return False
        current = self.scope_root
        if current is not None and not self.contains(current, root):
            return False
        if current is root:
            return False
        self._scopes.append(SelectionScope(root, self._label_of(root)))
        return True

    def exit_isolation(self) -> Optional[Artist]:
        if not self._scopes:
            return None
        return self._scopes.pop().root

    def clear_isolation(self) -> None:
        self._scopes.clear()


__all__ = [
    "HitCandidate",
    "HitResolution",
    "HitStack",
    "SelectionKernel",
    "SelectionMode",
    "SelectionScope",
    "TopHitDecision",
    "TopHitStatus",
]
