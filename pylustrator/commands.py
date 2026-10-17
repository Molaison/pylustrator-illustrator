"""Transactional command helpers, stable locators, and replay migrations."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Number
from threading import RLock
from types import MethodType
from typing import Any, Optional

import numpy as np
from matplotlib.artist import Artist
from matplotlib.lines import Line2D

from .editor_model import EditorGroup, EditorScene
from .legend_replay import axes_handles_reproduce_legend
from .source_migration import (
    GENERATED_STATE_VERSION,
    migrate_generated_command,
    migrate_generated_source,
)


_LINE_ENDPOINT_REPLAY_ATTR = "_pylustrator_set_line_endpoints"
_LINE_ENDPOINT_REPLAY_MARKER = "_pylustrator_line_endpoint_replay"
_LINE_ENDPOINT_REPLAY_LOCK = RLock()
_MISSING = object()


class LineEndpointReplayConflictError(RuntimeError):
    """A third-party attribute prevents safe, instance-local line replay."""

    def __init__(self, target: Line2D, *, scope: str, owner: object) -> None:
        self.target = target
        self.scope = scope
        self.owner = owner
        owner_name = (
            f"{owner.__module__}.{owner.__qualname__}"
            if isinstance(owner, type)
            else type(owner).__name__
        )
        super().__init__(
            f"Cannot enable compact Line2D endpoint replay: {scope} "
            f"attribute {_LINE_ENDPOINT_REPLAY_ATTR!r} is already owned by "
            f"{owner_name}"
        )


def _set_line_endpoints(self, keys, values):
    from .artist_adapters import Line2DAdapter, get_artist_adapter

    adapter = get_artist_adapter(self)
    if not isinstance(adapter, Line2DAdapter):
        raise TypeError(
            "_pylustrator_set_line_endpoints requires a supported Line2D"
        )
    adapter.apply_replayed_endpoints(keys, values)
    return self


setattr(_set_line_endpoints, _LINE_ENDPOINT_REPLAY_MARKER, True)


def _owned_line_endpoint_replay(value) -> bool:
    function = getattr(value, "__func__", value)
    if function is _set_line_endpoints:
        return True
    return bool(
        getattr(function, _LINE_ENDPOINT_REPLAY_MARKER, False)
        and getattr(function, "__module__", None) == __name__
        and getattr(function, "__qualname__", "")
        in {
            "_set_line_endpoints",
            "install_line_endpoint_replay_api.<locals>.set_endpoints",
        }
    )


def _line_endpoint_class_attribute(target: Line2D):
    for owner in type(target).__mro__:
        namespace = vars(owner)
        if _LINE_ENDPOINT_REPLAY_ATTR in namespace:
            return owner, namespace[_LINE_ENDPOINT_REPLAY_ATTR]
    return None, None


def line_endpoint_replay_conflict(
    target: Line2D,
) -> LineEndpointReplayConflictError | None:
    """Return the collision blocking instance-local endpoint replay, if any."""

    if not isinstance(target, Line2D):
        raise TypeError("Line endpoint replay can only be installed on Line2D")
    owner, class_value = _line_endpoint_class_attribute(target)
    if owner is not None and not _owned_line_endpoint_replay(class_value):
        return LineEndpointReplayConflictError(
            target,
            scope="class",
            owner=owner,
        )
    instance_value = vars(target).get(_LINE_ENDPOINT_REPLAY_ATTR, _MISSING)
    if instance_value is not _MISSING and not (
        isinstance(instance_value, MethodType)
        and instance_value.__self__ is target
        and _owned_line_endpoint_replay(instance_value)
    ):
        return LineEndpointReplayConflictError(
            target,
            scope="instance",
            owner=target,
        )
    return None


def _install_line_endpoint_replay_bindings(targets) -> None:
    """Validate first, then bind a compact replay method to each target."""

    unique_targets = []
    seen_ids = set()
    for target in targets:
        if id(target) not in seen_ids:
            seen_ids.add(id(target))
            unique_targets.append(target)
    targets = tuple(unique_targets)
    with _LINE_ENDPOINT_REPLAY_LOCK:
        legacy_class_attributes = {}
        pending = []
        for target in targets:
            conflict = line_endpoint_replay_conflict(target)
            if conflict is not None:
                raise conflict
            owner, class_value = _line_endpoint_class_attribute(target)
            if owner is not None:
                legacy_class_attributes[owner] = class_value
            instance_value = vars(target).get(_LINE_ENDPOINT_REPLAY_ATTR, _MISSING)
            if instance_value is _MISSING:
                pending.append(target)

        removed = []
        bound = []
        try:
            for owner, value in legacy_class_attributes.items():
                if vars(owner).get(_LINE_ENDPOINT_REPLAY_ATTR) is not value:
                    raise LineEndpointReplayConflictError(
                        targets[0],
                        scope="class",
                        owner=owner,
                    )
                delattr(owner, _LINE_ENDPOINT_REPLAY_ATTR)
                removed.append((owner, value))
            for target in pending:
                vars(target)[_LINE_ENDPOINT_REPLAY_ATTR] = MethodType(
                    _set_line_endpoints,
                    target,
                )
                bound.append(target)
        except Exception:
            for target in reversed(bound):
                current = vars(target).get(_LINE_ENDPOINT_REPLAY_ATTR)
                if _owned_line_endpoint_replay(current):
                    vars(target).pop(_LINE_ENDPOINT_REPLAY_ATTR, None)
            for owner, value in removed:
                setattr(owner, _LINE_ENDPOINT_REPLAY_ATTR, value)
            raise


def ensure_line_endpoint_replay_api(target: Line2D) -> Line2D:
    """Bind compact endpoint replay to exactly one Line2D instance."""

    _install_line_endpoint_replay_bindings((target,))
    return target


def _figure_line_endpoint_replay_targets(figure):
    """Yield only explicit line inventories with stable generated locators."""

    seen = set()

    def add(target):
        if isinstance(target, Line2D) and id(target) not in seen:
            seen.add(id(target))
            yield target

    owners = [figure]
    for owner in owners:
        owners.extend(getattr(owner, "subfigs", ()))
        for artist in getattr(owner, "artists", ()):
            yield from add(artist)
    for axes in figure.axes:
        for line in axes.lines:
            yield from add(line)



def semantic_equal(
    left: Any,
    right: Any,
    *,
    atol: float = 1e-10,
    rtol: float = 1e-12,
) -> bool:
    """Compare adapter snapshots while ignoring meaningless float noise."""

    if left is right:
        return True
    if isinstance(left, Artist) or isinstance(right, Artist):
        return left is right
    if isinstance(left, Number) and isinstance(right, Number):
        return bool(np.isclose(left, right, atol=atol, rtol=rtol, equal_nan=True))
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        left_masked = isinstance(left, np.ma.MaskedArray)
        right_masked = isinstance(right, np.ma.MaskedArray)
        if left_masked or right_masked:
            if not (left_masked and right_masked):
                return False
            return _masked_array_semantic_equal(
                left,
                right,
                atol=atol,
                rtol=rtol,
            )
        try:
            return bool(
                np.allclose(
                    np.asarray(left, dtype=float),
                    np.asarray(right, dtype=float),
                    atol=atol,
                    rtol=rtol,
                    equal_nan=True,
                )
            )
        except (TypeError, ValueError):
            return False
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(
            semantic_equal(left[key], right[key], atol=atol, rtol=rtol)
            for key in left
        )
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(
            semantic_equal(a, b, atol=atol, rtol=rtol)
            for a, b in zip(left, right)
        )
    matrix_left = getattr(left, "get_matrix", None)
    matrix_right = getattr(right, "get_matrix", None)
    if callable(matrix_left) and callable(matrix_right):
        try:
            return semantic_equal(
                matrix_left(), matrix_right(), atol=atol, rtol=rtol
            )
        except (TypeError, ValueError, RuntimeError):
            pass
    try:
        return bool(left == right)
    except (TypeError, ValueError):
        return False


def _masked_array_semantic_equal(
    left: np.ma.MaskedArray,
    right: np.ma.MaskedArray,
    *,
    atol: float,
    rtol: float,
) -> bool:
    """Compare every lossless part of a masked-array snapshot."""

    left_constant = isinstance(left, type(np.ma.masked))
    right_constant = isinstance(right, type(np.ma.masked))
    if left_constant or right_constant:
        return left_constant and right_constant
    if (
        left.shape != right.shape
        or left.dtype != right.dtype
        or bool(left.hardmask) != bool(right.hardmask)
    ):
        return False

    left_nomask = left.mask is np.ma.nomask
    right_nomask = right.mask is np.ma.nomask
    if left_nomask != right_nomask:
        return False
    if not left_nomask:
        left_mask = np.asarray(left.mask)
        right_mask = np.asarray(right.mask)
        if left_mask.shape != right_mask.shape or not np.array_equal(
            left_mask, right_mask
        ):
            return False

    if not _exact_value_equal(left.fill_value, right.fill_value):
        return False

    left_data = np.asarray(left.data)
    right_data = np.asarray(right.data)
    if left.dtype.kind in "fc":
        try:
            return bool(
                np.allclose(
                    left_data,
                    right_data,
                    atol=atol,
                    rtol=rtol,
                    equal_nan=True,
                )
            )
        except (TypeError, ValueError):
            return False
    return _exact_value_equal(left_data, right_data)


def _exact_value_equal(left: Any, right: Any) -> bool:
    try:
        return bool(np.array_equal(left, right, equal_nan=True))
    except (TypeError, ValueError):
        try:
            return bool(np.array_equal(left, right))
        except (TypeError, ValueError):
            return False


@dataclass(frozen=True)
class ObjectLocator:
    """A versioned locator for an Artist or logical editor group."""

    reference: str
    expected_type: str
    semantic_name: str = ""
    gid: str = ""
    version: int = GENERATED_STATE_VERSION

    @classmethod
    def from_artist(cls, artist: Artist) -> "ObjectLocator":
        if isinstance(artist, EditorGroup):
            reference = f"group:{artist.group_id}"
        else:
            from .change_tracker import getReference

            reference = getReference(artist)
        gid = getattr(artist, "get_gid", lambda: None)() or ""
        semantic_name = getattr(artist, "get_label", lambda: "")() or ""
        if not semantic_name or str(semantic_name).startswith("_"):
            semantic_name = getattr(artist, "get_text", lambda: "")() or ""
        if isinstance(artist, EditorGroup):
            semantic_name = artist.name
        return cls(
            reference,
            type(artist).__name__,
            str(semantic_name),
            str(gid),
        )

    def to_data(self) -> dict:
        return {
            "version": self.version,
            "reference": self.reference,
            "type": self.expected_type,
            "name": self.semantic_name,
            "gid": self.gid,
        }

    @classmethod
    def from_data(cls, data: dict | str) -> "ObjectLocator":
        if isinstance(data, str):
            return cls(data, "")
        return cls(
            str(data.get("reference", "")),
            str(data.get("type", "")),
            str(data.get("name", "")),
            str(data.get("gid", "")),
            int(data.get("version", 0)),
        )

    def resolve(self, scene: EditorScene) -> Optional[Artist]:
        artist = scene.resolve_locator(self.reference)
        if artist is not None and (
            not self.expected_type or type(artist).__name__ == self.expected_type
        ):
            return artist
        candidates = [
            candidate
            for candidate in scene.known_artists
            if not self.expected_type
            or type(candidate).__name__ == self.expected_type
        ]
        if self.gid:
            matches = [
                candidate
                for candidate in candidates
                if (getattr(candidate, "get_gid", lambda: None)() or "") == self.gid
            ]
            if len(matches) == 1:
                return matches[0]
        if self.semantic_name:
            def name(candidate):
                value = getattr(candidate, "get_label", lambda: "")() or ""
                if not value or str(value).startswith("_"):
                    value = getattr(candidate, "get_text", lambda: "")() or ""
                if isinstance(candidate, EditorGroup):
                    value = candidate.name
                return str(value)

            matches = [candidate for candidate in candidates if name(candidate) == self.semantic_name]
            if len(matches) == 1:
                return matches[0]
        return None


def install_legacy_legend_replay_compatibility(figure) -> None:
    """Make old proxy-legend references replay until the block is resaved.

    This is installed per Axes from the generated block's ``_pylustrator_init``
    header, so ordinary Matplotlib code keeps the original API semantics.
    """

    for axes in figure.axes:
        if hasattr(axes, "_pylustrator_original_get_legend_handles_labels"):
            continue
        original = axes.get_legend_handles_labels

        def compatible(self, legend_handler_map=None, _original=original):
            handles, labels = _original(legend_handler_map)
            legend = self.get_legend()
            if legend is None:
                return handles, labels
            reproduces = axes_handles_reproduce_legend(legend, handles, labels)
            if reproduces is not False:
                return handles, labels
            proxies = list(getattr(legend, "legend_handles", []))
            if not proxies:
                return handles, labels
            proxy_labels = [text.get_text() for text in legend.get_texts()]
            return proxies, proxy_labels

        axes._pylustrator_original_get_legend_handles_labels = original
        axes.get_legend_handles_labels = MethodType(compatible, axes)


def install_line_endpoint_replay_api(figure) -> None:
    """Bind endpoint replay to the Figure's explicitly referenceable lines.

    Normal editor attachment uses the O(1) single-target helper instead.  This
    Figure-level compatibility API deliberately walks ``Axes.lines`` and
    direct owner ``artists`` only, never renderer-managed ticks or Legend
    proxy children.
    """

    _install_line_endpoint_replay_bindings(
        _figure_line_endpoint_replay_targets(figure)
    )


@dataclass(frozen=True)
class InteractionState:
    """Selection, tool scope, and non-document transform UI state."""

    mode: str
    selected: tuple[ObjectLocator, ...]
    primary: Optional[ObjectLocator]
    scopes: tuple[ObjectLocator, ...]
    alignment_reference_mode: str = "selection"
    alignment_key: Optional[ObjectLocator] = None
    reference_point: tuple[float, float] = (0.5, 0.5)
    custom_rotation_pivot_inches: Optional[tuple[float, float]] = None


__all__ = [
    "GENERATED_STATE_VERSION",
    "InteractionState",
    "ObjectLocator",
    "migrate_generated_command",
    "migrate_generated_source",
    "install_legacy_legend_replay_compatibility",
    "install_line_endpoint_replay_api",
    "ensure_line_endpoint_replay_api",
    "line_endpoint_replay_conflict",
    "LineEndpointReplayConflictError",
    "semantic_equal",
]
