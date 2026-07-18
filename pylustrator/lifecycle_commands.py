"""Atomic lifecycle commands for editor-owned Artist selections.

Deleting an object is not merely a call to :meth:`matplotlib.artist.Artist.remove`.
It also changes generated-source bookkeeping and editor selection state.  This
module keeps those three pieces behind one transaction boundary so a multi-
selection delete is one reversible user action.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Iterable

from matplotlib.artist import Artist
from matplotlib.text import Text

from .editor_model import EditorGroup


@dataclass(frozen=True)
class _DeletedArtistState:
    artist: Artist
    visible: bool
    text: str | None
    axis_label: bool
    created_by_editor: bool

    def restore(self) -> None:
        if self.text is not None:
            self.artist.set_text(self.text)
        self.artist.set_visible(self.visible)

    def apply(self) -> None:
        if self.axis_label:
            self.artist.set_text("")
        else:
            self.artist.set_visible(False)


def _unique_lifecycle_targets(targets: Iterable[Artist]) -> tuple[Artist, ...]:
    """Flatten logical groups while retaining deterministic member order."""

    result: list[Artist] = []
    seen: set[int] = set()

    def add(target: Artist) -> None:
        if isinstance(target, EditorGroup):
            for member in target.members:
                add(member)
            return
        if id(target) in seen:
            return
        seen.add(id(target))
        result.append(target)

    for target in targets:
        add(target)
    return tuple(result)


def _is_axis_label(artist: Artist) -> bool:
    figure = getattr(artist, "figure", None)
    if not isinstance(artist, Text) or figure is None:
        return False
    for axes in getattr(figure, "axes", ()):
        for axis_name in ("xaxis", "yaxis", "zaxis"):
            axis = getattr(axes, axis_name, None)
            if axis is not None and artist is getattr(axis, "label", None):
                return True
    return False


def _remove_owned_changes(tracker, artist: Artist) -> None:
    changes = getattr(tracker, "changes", None)
    if isinstance(changes, dict):
        for key in list(changes):
            if isinstance(key, tuple) and key and key[0] is artist:
                del changes[key]
        return
    if isinstance(changes, list):
        changes[:] = [
            item
            for item in changes
            if not (isinstance(item, tuple) and item and item[0] is artist)
        ]
        return
    raise TypeError("Atomic delete requires mutable change recording")


def _created_by_editor(tracker, artist: Artist) -> bool:
    changes = getattr(tracker, "changes", ())
    if isinstance(changes, dict):
        return (artist, ".new") in changes
    return any(
        isinstance(item, tuple)
        and len(item) >= 2
        and item[0] is artist
        and item[1] == ".new"
        for item in changes
    )


def _capture_history(tracker):
    if not hasattr(tracker, "edits") or not hasattr(tracker, "last_edit"):
        return None
    return list(tracker.edits), int(tracker.last_edit)


def _restore_history(tracker, state) -> None:
    if state is None:
        return
    edits, last_edit = state
    tracker.edits = list(edits)
    tracker.last_edit = int(last_edit)


def delete_selection(manager, targets: Iterable[Artist]) -> bool:
    """Delete *targets* as one atomic, selection-preserving history command.

    Editor-created non-Text Artists are hidden in the live session and have
    their creation commands removed.  Keeping their identity alive makes Undo
    exact and preserves paint order; after saving/reloading they disappear
    because no creation command remains.
    """

    requested_targets = tuple(targets)
    artists = _unique_lifecycle_targets(requested_targets)
    if not artists:
        return False
    figure = manager.figure
    tracker = figure.change_tracker
    capture_recording = getattr(tracker, "capture_recording_state", None)
    restore_recording = getattr(tracker, "restore_recording_state", None)
    if not callable(capture_recording) or not callable(restore_recording):
        raise TypeError("Atomic delete requires transactional change recording")

    recording_before = capture_recording()
    history_before = _capture_history(tracker)
    interaction_before = manager.capture_interaction_state()
    states = tuple(
        _DeletedArtistState(
            artist=artist,
            visible=bool(artist.get_visible()),
            text=artist.get_text() if isinstance(artist, Text) else None,
            axis_label=_is_axis_label(artist),
            created_by_editor=_created_by_editor(tracker, artist),
        )
        for artist in artists
    )

    def restore_artists() -> None:
        failures = []
        for state in reversed(states):
            try:
                state.restore()
            except Exception as error:  # restore every earlier target as well
                failures.append((state.artist, error))
        if failures:
            error = RuntimeError("Atomic delete rollback failed")
            error.pylustrator_rollback_failures = tuple(failures)
            raise error

    def apply_artists(*, record: bool) -> None:
        for state in states:
            if record:
                _remove_owned_changes(tracker, state.artist)
            state.apply()
            if not record or (state.created_by_editor and not isinstance(state.artist, Text)):
                continue
            if isinstance(state.artist, Text):
                tracker.addNewTextChange(state.artist)
            else:
                tracker.addChange(state.artist, ".set(visible=False)")

    try:
        apply_artists(record=True)
        # Deleting an active isolation root must not strand the editor inside an
        # invisible scope. Other isolation scopes remain unchanged.
        kernel = manager._ensure_selection_kernel()
        if any(scope.root in requested_targets for scope in kernel.scopes):
            kernel.clear_isolation()
        manager.select_element(None)
        interaction_after = manager.capture_interaction_state()
        recording_after = capture_recording()

        def undo() -> None:
            restore_artists()
            restore_recording(recording_before)
            manager.restore_interaction_state(interaction_before)

        def redo() -> None:
            apply_artists(record=False)
            restore_recording(recording_after)
            manager.restore_interaction_state(interaction_after)

        label = "Delete object" if len(artists) == 1 else f"Delete {len(artists)} objects"
        tracker.addEdit([undo, redo, label])
    except Exception:
        rollback_failures = []
        try:
            restore_artists()
        except Exception as rollback_error:
            rollback_failures.extend(
                getattr(rollback_error, "pylustrator_rollback_failures", ())
            )
        try:
            restore_recording(recording_before)
        except Exception as rollback_error:
            rollback_failures.append((tracker, rollback_error))
        try:
            _restore_history(tracker, history_before)
        except Exception as rollback_error:
            rollback_failures.append((tracker, rollback_error))
        try:
            manager.restore_interaction_state(interaction_before)
        except Exception as rollback_error:
            rollback_failures.append((manager, rollback_error))
        active_error = sys.exc_info()[1]
        if rollback_failures and active_error is not None:
            active_error.pylustrator_rollback_failures = tuple(rollback_failures)
            add_note = getattr(active_error, "add_note", None)
            if callable(add_note):
                details = "; ".join(
                    f"{type(target).__name__}: {error}"
                    for target, error in rollback_failures
                )
                add_note(f"Pylustrator delete rollback failures: {details}")
        raise
    return True


__all__ = ["delete_selection"]
