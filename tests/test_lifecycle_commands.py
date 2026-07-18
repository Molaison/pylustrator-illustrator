from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest
from matplotlib.artist import Artist
from matplotlib.patches import Rectangle

from pylustrator.lifecycle_commands import delete_selection


class _Tracker:
    def __init__(self, *, fail_on_add: int | None = None) -> None:
        self.changes = {}
        self.saved = True
        self.edits = []
        self.last_edit = -1
        self._add_count = 0
        self._fail_on_add = fail_on_add

    def capture_recording_state(self):
        return dict(self.changes), bool(self.saved)

    def restore_recording_state(self, state) -> None:
        changes, saved = state
        self.changes = dict(changes)
        self.saved = bool(saved)

    def _record(self, artist: Artist, command: str) -> None:
        self._add_count += 1
        if self._fail_on_add == self._add_count:
            raise RuntimeError("synthetic recording failure")
        self.changes[artist, command.split("(", 1)[0]] = (artist, command)
        self.saved = False

    def addChange(self, artist: Artist, command: str) -> None:
        self._record(artist, command)

    def addNewTextChange(self, artist: Artist) -> None:
        self._record(artist, ".set(text=...)")

    def addEdit(self, edit) -> None:
        self.edits.append(edit)
        self.last_edit = len(self.edits) - 1


class _Kernel:
    scopes = ()

    def clear_isolation(self) -> None:
        self.scopes = ()


class _Manager:
    def __init__(self, figure, selected) -> None:
        self.figure = figure
        self.selected = list(selected)
        self.selected_element = self.selected[-1] if self.selected else None
        self.kernel = _Kernel()

    def _ensure_selection_kernel(self):
        return self.kernel

    def capture_interaction_state(self):
        return tuple(self.selected), self.selected_element

    def restore_interaction_state(self, state) -> None:
        selected, primary = state
        self.selected = list(selected)
        self.selected_element = primary

    def select_element(self, artist) -> None:
        self.selected = [] if artist is None else [artist]
        self.selected_element = artist


def _scene(*, tracker=None):
    fig, ax = plt.subplots()
    first = ax.add_patch(Rectangle((0.1, 0.1), 0.2, 0.2))
    second = ax.add_patch(Rectangle((0.5, 0.5), 0.2, 0.2))
    fig.change_tracker = tracker or _Tracker()
    manager = _Manager(fig, [first, second])
    fig.figure_dragger = manager
    return fig, manager, first, second


def test_multi_delete_is_one_atomic_selection_preserving_history_item() -> None:
    fig, manager, first, second = _scene()
    tracker = fig.change_tracker
    tracker.changes[first, ".set_alpha"] = (first, ".set_alpha(0.5)")
    tracker.changes[second, ".set_alpha"] = (second, ".set_alpha(0.5)")

    assert delete_selection(manager, [first, second])
    assert not first.get_visible() and not second.get_visible()
    assert manager.selected == []
    assert len(tracker.edits) == 1
    assert tracker.edits[0][2] == "Delete 2 objects"

    tracker.edits[0][0]()
    assert first.get_visible() and second.get_visible()
    assert manager.selected == [first, second]
    assert (first, ".set_alpha") in tracker.changes
    assert (second, ".set_alpha") in tracker.changes

    tracker.edits[0][1]()
    assert not first.get_visible() and not second.get_visible()
    assert manager.selected == []
    plt.close(fig)


def test_delete_editor_created_artist_removes_creation_record_but_remains_undoable() -> None:
    fig, manager, first, _second = _scene()
    manager.selected = [first]
    manager.selected_element = first
    tracker = fig.change_tracker
    tracker.changes[first, ".new"] = (first.axes, ".add_patch(...)")

    assert delete_selection(manager, [first])
    assert not first.get_visible()
    assert all(key[0] is not first for key in tracker.changes)
    assert len(tracker.edits) == 1

    tracker.edits[0][0]()
    assert first.get_visible()
    assert tracker.changes[first, ".new"] == (first.axes, ".add_patch(...)")
    assert manager.selected == [first]

    tracker.edits[0][1]()
    assert not first.get_visible()
    assert all(key[0] is not first for key in tracker.changes)
    plt.close(fig)


def test_delete_recording_failure_rolls_back_every_artist_and_history() -> None:
    tracker = _Tracker(fail_on_add=2)
    fig, manager, first, second = _scene(tracker=tracker)
    before = tracker.capture_recording_state()

    with pytest.raises(RuntimeError, match="synthetic recording failure"):
        delete_selection(manager, [first, second])

    assert first.get_visible() and second.get_visible()
    assert manager.selected == [first, second]
    assert manager.selected_element is second
    assert tracker.capture_recording_state() == before
    assert tracker.edits == []
    assert tracker.last_edit == -1
    plt.close(fig)
