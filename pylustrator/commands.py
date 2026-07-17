"""Transactional command helpers, stable locators, and replay migrations."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Number
from types import MethodType
from typing import Any, Optional

import numpy as np
from matplotlib.artist import Artist

from .editor_model import EditorGroup, EditorScene
from .legend_replay import axes_handles_reproduce_legend
from .source_migration import (
    GENERATED_STATE_VERSION,
    migrate_generated_command,
    migrate_generated_source,
)



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


@dataclass(frozen=True)
class InteractionState:
    """Selection, tool scope, and non-document alignment UI state."""

    mode: str
    selected: tuple[ObjectLocator, ...]
    primary: Optional[ObjectLocator]
    scopes: tuple[ObjectLocator, ...]
    alignment_reference_mode: str = "selection"
    alignment_key: Optional[ObjectLocator] = None


__all__ = [
    "GENERATED_STATE_VERSION",
    "InteractionState",
    "ObjectLocator",
    "migrate_generated_command",
    "migrate_generated_source",
    "install_legacy_legend_replay_compatibility",
    "semantic_equal",
]
