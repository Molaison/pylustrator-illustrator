"""Logical editor groups and layer state independent of Matplotlib ownership."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Callable, Iterable, Optional

import matplotlib.pyplot as plt
from matplotlib.artist import Artist
from matplotlib.figure import Figure
from matplotlib.legend import Legend
from matplotlib.transforms import Bbox


EDITOR_STATE_VERSION = 2


class EditorGroup(Artist):
    """A non-rendering logical group whose members remain live Artists."""

    _pylustrator_editor_group = True

    def __init__(
        self,
        figure: Figure,
        group_id: str,
        members: Iterable[Artist],
        *,
        name: str,
        owner: Optional[Artist] = None,
    ) -> None:
        super().__init__()
        self.group_id = str(group_id)
        self.members = list(dict.fromkeys(members))
        self.name = str(name)
        self.owner = owner or figure
        self.set_figure(figure)
        self.set_label(self.name)
        self.set_picker(True)
        self._member_visibility: Optional[dict[int, bool]] = None

    def __str__(self) -> str:
        return self.name

    def get_children(self) -> list[Artist]:
        return list(self.members)

    def contains(self, event) -> tuple[bool, dict]:
        for member in reversed(self.members):
            try:
                if member.get_visible() and member.contains(event)[0]:
                    return True, {"member": member}
            except (AttributeError, TypeError, ValueError, RuntimeError):
                continue
        return False, {}

    def get_window_extent(self, renderer=None):
        boxes = []
        for member in self.members:
            try:
                if member.get_visible():
                    boxes.append(member.get_window_extent(renderer))
            except (AttributeError, TypeError, ValueError, RuntimeError):
                continue
        return Bbox.union(boxes) if boxes else Bbox.null()

    def get_zorder(self) -> float:
        if not self.members:
            return super().get_zorder()
        return max(float(member.get_zorder()) for member in self.members)

    def set_zorder(self, level) -> None:
        level = float(level)
        current = self.get_zorder()
        delta = level - current
        for member in self.members:
            member.set_zorder(float(member.get_zorder()) + delta)
        super().set_zorder(level)

    def set_visible(self, visible: bool) -> None:
        visible = bool(visible)
        if not visible and self._member_visibility is None:
            self._member_visibility = {
                id(member): bool(member.get_visible()) for member in self.members
            }
        for member in self.members:
            member.set_visible(
                self._member_visibility.get(id(member), True)
                if visible and self._member_visibility is not None
                else False
            )
        if visible:
            self._member_visibility = None
        super().set_visible(visible)


@dataclass(frozen=True)
class LayerMutation:
    """Serializable before/after editor-layer state."""

    before: dict
    after: dict
    name: str


class EditorScene:
    """Store logical grouping and layer state for one Figure."""

    def __init__(
        self,
        figure: Figure,
        *,
        ownership_parent: Callable[[Artist], Optional[Artist]],
    ) -> None:
        self.figure = figure
        self._ownership_parent = ownership_parent
        self.groups: dict[str, EditorGroup] = {}
        self._logical_parent_by_id: dict[int, EditorGroup] = {}
        self._locked_ids: set[int] = set()
        self._explicitly_hidden_ids: set[int] = set()
        self._known_artists: dict[int, Artist] = {id(figure): figure}
        self._next_group_number = 1

    def register_artist(self, artist: Artist) -> None:
        self._known_artists[id(artist)] = artist

    @property
    def known_artists(self) -> tuple[Artist, ...]:
        return tuple(self._known_artists.values())

    @staticmethod
    def is_group(artist: Artist) -> bool:
        return isinstance(artist, (Legend, EditorGroup))

    def selection_parent(self, artist: Artist) -> Optional[Artist]:
        logical = self._logical_parent_by_id.get(id(artist))
        return logical if logical is not None else self._ownership_parent(artist)

    def contains(self, root: Artist, descendant: Artist) -> bool:
        if root is descendant:
            return True
        current = self.selection_parent(descendant)
        seen: set[int] = set()
        while current is not None and id(current) not in seen:
            if current is root:
                return True
            seen.add(id(current))
            current = self.selection_parent(current)
        return False

    def _effective_flag(self, artist: Artist, values: set[int]) -> bool:
        current: Optional[Artist] = artist
        seen: set[int] = set()
        while current is not None and id(current) not in seen:
            if id(current) in values:
                return True
            seen.add(id(current))
            current = self.selection_parent(current)
        return False

    def is_locked(self, artist: Artist) -> bool:
        return self._effective_flag(artist, self._locked_ids)

    def is_explicitly_hidden(self, artist: Artist) -> bool:
        return self._effective_flag(artist, self._explicitly_hidden_ids)

    def set_locked(self, artists: Iterable[Artist], locked: bool) -> bool:
        changed = False
        for artist in artists:
            self.register_artist(artist)
            key = id(artist)
            if locked and key not in self._locked_ids:
                self._locked_ids.add(key)
                changed = True
            elif not locked and key in self._locked_ids:
                self._locked_ids.remove(key)
                changed = True
        return changed

    def set_visible(self, artists: Iterable[Artist], visible: bool) -> bool:
        changed = False
        for artist in artists:
            self.register_artist(artist)
            key = id(artist)
            if visible:
                if key in self._explicitly_hidden_ids:
                    self._explicitly_hidden_ids.remove(key)
                    artist.set_visible(True)
                    changed = True
            elif key not in self._explicitly_hidden_ids:
                self._explicitly_hidden_ids.add(key)
                artist.set_visible(False)
                changed = True
        return changed

    def _common_owner(self, members: list[Artist]) -> Artist:
        owners = [self._ownership_parent(member) for member in members]
        first = owners[0] if owners else None
        return first if first is not None and all(owner is first for owner in owners) else self.figure

    def _new_group_id(self) -> str:
        while f"pyl-group-{self._next_group_number}" in self.groups:
            self._next_group_number += 1
        result = f"pyl-group-{self._next_group_number}"
        self._next_group_number += 1
        return result

    def create_group(
        self,
        members: Iterable[Artist],
        *,
        name: Optional[str] = None,
        group_id: Optional[str] = None,
        owner: Optional[Artist] = None,
    ) -> EditorGroup:
        members = list(dict.fromkeys(members))
        if len(members) < 2:
            raise ValueError("Select at least two objects to create a group.")
        if any(member is self.figure for member in members):
            raise ValueError("The Figure itself cannot be grouped.")
        for member in members:
            parent = self._logical_parent_by_id.get(id(member))
            if parent is not None:
                raise ValueError(
                    f"{type(member).__name__} already belongs to group {parent.name!r}."
                )
        group_id = group_id or self._new_group_id()
        if group_id in self.groups:
            raise ValueError(f"Duplicate editor group id: {group_id}")
        group = EditorGroup(
            self.figure,
            group_id,
            members,
            name=name or f"Group {len(self.groups) + 1}",
            owner=owner or self._common_owner(members),
        )
        self.groups[group_id] = group
        self.register_artist(group)
        for member in members:
            self.register_artist(member)
            self._logical_parent_by_id[id(member)] = group
        return group

    def remove_group(self, group: EditorGroup) -> list[Artist]:
        if group.group_id not in self.groups:
            return []
        members = list(group.members)
        self.groups.pop(group.group_id)
        for member in members:
            if self._logical_parent_by_id.get(id(member)) is group:
                self._logical_parent_by_id.pop(id(member), None)
        self._locked_ids.discard(id(group))
        self._explicitly_hidden_ids.discard(id(group))
        return members

    def groups_for_owner(self, owner: Artist) -> list[EditorGroup]:
        return [group for group in self.groups.values() if group.owner is owner]

    def tree_children(self, owner: Artist, ordinary_children: Iterable[Artist]) -> list[Artist]:
        groups = self.groups_for_owner(owner)
        grouped_ids = {
            id(member)
            for group in groups
            for member in group.members
        }
        children = [child for child in ordinary_children if id(child) not in grouped_ids]
        children.extend(groups)
        return children

    def tree_parent(self, artist: Artist) -> Optional[Artist]:
        if isinstance(artist, EditorGroup):
            return artist.owner
        return self.selection_parent(artist)

    def _locator(self, artist: Artist) -> dict:
        from .commands import ObjectLocator

        return ObjectLocator.from_artist(artist).to_data()

    def _resolve_locator(self, locator: dict | str) -> Optional[Artist]:
        if isinstance(locator, dict):
            from .commands import ObjectLocator

            return ObjectLocator.from_data(locator).resolve(self)
        if locator.startswith("group:"):
            return self.groups.get(locator.removeprefix("group:"))
        try:
            return eval(locator, {"plt": plt})
        except (AttributeError, IndexError, KeyError, NameError, SyntaxError):
            return None

    def resolve_locator(self, locator: dict | str) -> Optional[Artist]:
        return self._resolve_locator(locator)

    def export_state(self) -> dict:
        return {
            "version": EDITOR_STATE_VERSION,
            "groups": [
                {
                    "id": group.group_id,
                    "name": group.name,
                    "owner": self._locator(group.owner),
                    "members": [self._locator(member) for member in group.members],
                }
                for group in self.groups.values()
            ],
            "locked": [
                self._locator(artist)
                for key in self._locked_ids
                if (artist := self._known_artists.get(key)) is not None
            ],
            "hidden": [
                self._locator(artist)
                for key in self._explicitly_hidden_ids
                if (artist := self._known_artists.get(key)) is not None
            ],
        }

    def apply_state(self, state: Optional[dict]) -> None:
        state = deepcopy(state or {"version": EDITOR_STATE_VERSION})
        for key in list(self._explicitly_hidden_ids):
            artist = self._known_artists.get(key)
            if artist is not None:
                artist.set_visible(True)
        self.groups.clear()
        self._logical_parent_by_id.clear()
        self._locked_ids.clear()
        self._explicitly_hidden_ids.clear()

        pending = list(state.get("groups", []))
        while pending:
            progressed = False
            for spec in list(pending):
                owner = self._resolve_locator(spec.get("owner", "")) or self.figure
                members = [
                    member
                    for locator in spec.get("members", [])
                    if (member := self._resolve_locator(locator)) is not None
                ]
                if len(members) != len(spec.get("members", [])):
                    continue
                self.create_group(
                    members,
                    name=spec.get("name"),
                    group_id=spec.get("id"),
                    owner=owner,
                )
                pending.remove(spec)
                progressed = True
            if not progressed:
                break

        locked = [
            artist
            for locator in state.get("locked", [])
            if (artist := self._resolve_locator(locator)) is not None
        ]
        hidden = [
            artist
            for locator in state.get("hidden", [])
            if (artist := self._resolve_locator(locator)) is not None
        ]
        self.set_locked(locked, True)
        self.set_visible(hidden, False)
        self.figure._pylustrator_editor_state = self.export_state()

    def restore_persisted_state(self) -> None:
        self.apply_state(getattr(self.figure, "_pylustrator_editor_state", None))

    def record_state(self) -> None:
        state = self.export_state()
        self.figure._pylustrator_editor_state = deepcopy(state)
        tracker = getattr(self.figure, "change_tracker", None)
        if tracker is not None:
            command = f"._pylustrator_editor_state = {state!r}"
            try:
                tracker.addChange(
                    self.figure,
                    command,
                    reference_obj=self.figure,
                    reference_command="._pylustrator_editor_state",
                )
            except TypeError:
                # Lightweight test/integration trackers may only implement the
                # historical two-argument surface.
                tracker.addChange(self.figure, command)


__all__ = [
    "EDITOR_STATE_VERSION",
    "EditorGroup",
    "EditorScene",
    "LayerMutation",
]
