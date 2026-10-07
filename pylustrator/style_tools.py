"""Scoped selection queries and a deliberately bounded, value-only style clipboard.

No Artist, transform, ownership link, data array, or text content is copied.
Pastes use the property transaction boundary, including its rollback and replay.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

from matplotlib.artist import Artist
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from matplotlib.text import Text

from .property_transactions import (
    PropertyPlan, PropertyPreflightError, TEXT_INDEPENDENT_SETTERS,
    _TEXT_STATE_PROPERTIES,
)
from .replay import replay_literal


# Text uses only fields covered by ChangeTracker's complete Text snapshot.
# In particular, copying alpha/fontfamily through independent setters would be
# lost the next time addNewTextChange replaces that snapshot's recording.
_TEXT_STYLE = ("color", "fontsize", "style", "weight")
_LINE_STYLE = (
    "color", "linewidth", "alpha", "markersize", "markeredgewidth",
)
_STYLE_PROPERTIES = {Text: _TEXT_STYLE, Line2D: _LINE_STYLE}


@dataclass(frozen=True)
class StyleSnapshot:
    """A captured value snapshot, not a live link to a source Artist."""

    artist_type: type[Artist]
    values: tuple[tuple[str, object], ...]

    @classmethod
    def capture(cls, artist: Artist) -> StyleSnapshot:
        properties = _STYLE_PROPERTIES.get(type(artist))
        if properties is None:
            raise PropertyPreflightError(
                "Copy Style supports ordinary Text and Line2D objects only"
            )
        values = []
        for name in properties:
            value = deepcopy(getattr(artist, f"get_{name}")())
            replay_literal(value)
            values.append((name, value))
        return cls(type(artist), tuple(values))


def scalar_color(artist: Artist) -> tuple[float, float, float, float] | None:
    """Normalize a single color; never choose a color from a collection array."""

    getter = getattr(artist, "get_color", None)
    if not callable(getter):
        getter = getattr(artist, "get_facecolor", None)
    if not callable(getter):
        return None
    try:
        color = getter()
        # to_rgba accepts a (1, 4) array too; explicitly reject color arrays.
        import numpy as np

        if np.ndim(color) > 1:
            return None
        return to_rgba(color, artist.get_alpha())
    except (TypeError, ValueError):
        return None


def _available(manager, artist: Artist) -> bool:
    if not manager._is_pick_candidate(artist, explicit=True):
        return False
    kernel = manager._ensure_selection_kernel()
    if artist is kernel.scope_root:
        return False
    # Respect logical group promotion as well as isolation. A command must not
    # silently edit a group member while Object Selection targets its group.
    if kernel.map_artists([artist]) != [artist]:
        return False
    # A logical group can span Axes and be owned by the Figure, bypassing
    # its member's real Axes in the logical chain. Check both ancestor graphs.
    scene = manager._ensure_editor_scene()
    pending = [artist]
    seen = set()
    while pending:
        current = pending.pop()
        if current is None or id(current) in seen:
            continue
        seen.add(id(current))
        if (not current.get_visible() or scene.is_locked(current)
                or scene.is_explicitly_hidden(current)):
            return False
        pending.extend((manager._interaction_parent(current), manager._draw_parent(current)))
    return True


def _primary(manager) -> Artist:
    primary = manager.selected_element
    if primary is None or primary not in manager._selected_artists():
        raise PropertyPreflightError("Select an object first")
    if not _available(manager, primary):
        raise PropertyPreflightError("The selected object is unavailable in this scope")
    return primary


def select_same(manager, criterion: str = "type") -> list[Artist]:
    """Replace the selection with matches to its primary, within current scope."""

    if criterion not in ("type", "color"):
        raise ValueError("Select Same accepts 'type' or 'color'")
    primary = _primary(manager)
    key = type if criterion == "type" else scalar_color
    expected = key(primary)
    if expected is None:
        raise PropertyPreflightError("The selected object has no single scalar color")
    candidates = manager._ensure_selection_kernel().map_artists(
        manager.iter_selectable_artists()
    )
    matches = [
        artist for artist in candidates
        if _available(manager, artist) and key(artist) == expected
    ]
    return manager.select_elements(matches, primary=primary)


def copy_style(manager) -> StyleSnapshot:
    """Capture the primary selection, preserving the old clipboard on failure."""

    snapshot = StyleSnapshot.capture(_primary(manager))
    manager._style_clipboard = snapshot
    return snapshot


def _check_text_recording(manager, artist: Artist) -> None:
    """Reject recording contracts that the Text snapshot cannot safely refresh."""

    if type(artist) is not Text:
        return
    properties = set(_TEXT_STATE_PROPERTIES)
    for canonical, aliases in Text._alias_map.items():
        if canonical in properties or properties.intersection(aliases):
            properties.update((canonical, *aliases))
    # These independent setters are retained by PropertyPlan when it rebuilds
    # the Text snapshot. Unknown/compound commands need their own contract.
    supported = {
        ".set", ".new", *TEXT_INDEPENDENT_SETTERS,
        *(f".set_{name}" for name in properties),
    }
    changes = manager.figure.change_tracker.changes
    for owner, command in changes:
        if owner is artist and command not in supported:
            raise PropertyPreflightError(
                f"Paste Style cannot safely preserve this Text edit: {command}"
            )


def paste_style(manager) -> bool:
    """Paste into a compatible selection in one all-or-nothing history entry."""

    snapshot = getattr(manager, "_style_clipboard", None)
    if not isinstance(snapshot, StyleSnapshot):
        raise PropertyPreflightError("Copy a style in this figure before pasting")
    primary = _primary(manager)
    targets = manager._selected_artists()
    for artist in targets:
        if type(artist) is not snapshot.artist_type:
            raise PropertyPreflightError(
                "Paste Style requires a selection of the copied concrete artist type"
            )
        if not _available(manager, artist):
            raise PropertyPreflightError(
                "Paste Style cannot change locked, hidden, or out-of-scope objects"
            )
        _check_text_recording(manager, artist)
    plan = PropertyPlan.for_selection_changes(
        primary, targets, dict(snapshot.values)
    )
    return plan.commit("Paste Style")


__all__ = ["StyleSnapshot", "copy_style", "paste_style", "scalar_color", "select_same"]
