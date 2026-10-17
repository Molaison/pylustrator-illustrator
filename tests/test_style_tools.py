from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.patches import Rectangle
from qtpy import QtWidgets

from pylustrator.QtGuiDrag import PlotWindow
from pylustrator.change_tracker import init_figure
from pylustrator.interaction import SelectionMode
from pylustrator.property_transactions import PropertyPreflightError
from pylustrator.style_tools import StyleSnapshot, scalar_color
from test_selection_indicator import attach_drag_manager
from test_zorder_interaction import install_real_tracker


@pytest.fixture
def scene():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    figures = []

    def make(kind="text"):
        fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
        figures.append(fig)
        if kind == "text":
            source = ax.text(.2, .2, "source", color="red", fontsize=18,
                             style="italic", weight="bold", rotation=32)
            first = ax.text(.4, .5, "first", color="blue", fontsize=10,
                            family=["serif", "sans-serif"])
            second = ax.text(.7, .6, "second", color="green", fontsize=12)
        else:
            source, = ax.plot([0, 1], [0, 1], color="red", linewidth=4,
                              alpha=.4, markersize=9, markeredgewidth=3,
                              linestyle="--", marker="o")
            first, = ax.plot([.1, .6], [.4, .8], color="blue", linewidth=1,
                             linestyle=(2, (1, 3)), marker="x")
            second, = ax.plot([.3, .8], [.8, .2], color="green", linewidth=2,
                              linestyle=":", marker="s")
        fig.canvas.draw()
        init_figure(fig)
        manager = attach_drag_manager(fig)
        tracker = install_real_tracker(fig)
        return fig, ax, manager, tracker, source, first, second

    yield make
    for fig in figures:
        fig.figure_dragger.selection.clear_targets()
        plt.close(fig)
    assert app is not None


def selected(manager):
    return [target.target for target in manager.selection.targets]


def test_select_same_concrete_type_and_normalized_color(scene):
    fig, ax, manager, tracker, source, first, second = scene()
    first.set_color("#ff0000")
    line, = ax.plot([0, 1], [1, 0], color=(1, 0, 0, 1))
    rectangle = ax.add_patch(Rectangle((.1, .3), .1, .1, facecolor="red"))
    annotation = ax.annotate("subclass", (.3, .4), color="red")
    for artist in (line, rectangle, annotation):
        manager.make_draggable(artist)
    manager.select_element(source)
    matches = manager.select_same("type")
    assert source in matches and first in matches and second in matches
    assert annotation not in matches and line not in matches
    assert manager.selected_element is source
    assert set(manager.select_same("color")) == set(selected(manager))
    assert set(selected(manager)) == {source, first, line, rectangle, annotation}
    assert tracker.edits == [] and tracker.changes == {}
    assert fig is manager.figure


def test_scalar_color_normalizes_alpha_and_rejects_collection_arrays(scene):
    _fig, ax, _manager, _tracker, source, first, _second = scene()
    source.set_alpha(.5)
    first.set_color((1, 0, 0, .5))
    assert scalar_color(source) == scalar_color(first)
    assert scalar_color(ax.scatter([.2], [.3], color="red")) is None
    assert scalar_color(ax.scatter([.2, .4], [.3, .5], c=["red", "blue"])) is None


def test_select_same_respects_locks_hidden_ancestors_groups_and_isolation(scene):
    fig, ax, manager, _tracker, source, first, second = scene()
    other_axes = fig.add_axes([.1, .1, .2, .2])
    hidden_child = other_axes.text(.5, .5, "hidden ancestor")
    manager.make_axes_draggable([other_axes])
    other_axes.set_visible(False)
    manager.select_elements([source, first], primary=source)
    group = manager.group_selection("Pair")
    manager.select_element(second)
    assert group not in manager.select_same("type")
    assert source not in selected(manager) and hidden_child not in selected(manager)
    manager.set_selection_mode(SelectionMode.DIRECT)
    manager.enter_isolation(group)
    manager.select_element(source)
    assert set(manager.select_same("type")) == {source, first}
    manager._ensure_editor_scene().set_locked([first], True)
    assert manager.select_same("type") == [source]
    manager._ensure_editor_scene().set_locked([first], False)
    first.set_visible(False)
    assert manager.select_same("type") == [source]
    assert manager.isolation_breadcrumbs == ("Pair",)


@pytest.mark.parametrize("kind", ["text", "line"])
def test_style_snapshot_undo_redo_replay_and_geometry_preservation(scene, kind):
    fig, _ax, manager, tracker, source, first, second = scene(kind)
    manager.select_element(source)
    snapshot = manager.copy_style()
    assert isinstance(snapshot, StyleSnapshot)
    source.set_color("purple")  # Copy is a snapshot, not a source reference.
    manager.select_elements([first, second], primary=second)
    before = [StyleSnapshot.capture(artist) for artist in (first, second)]
    if kind == "text":
        unchanged = [(a.get_position(), a.get_text(), a.get_rotation(),
                      a.get_fontfamily(), a.get_alpha(), a.get_transform())
                     for a in (first, second)]
    else:
        unchanged = [(deepcopy(a.get_data()), deepcopy(a._unscaled_dash_pattern),
                      a.get_marker(), a.get_transform()) for a in (first, second)]
    interaction_before = manager.capture_interaction_state()
    assert manager.paste_style()
    assert len(tracker.edits) == 1 and tracker.edits[0][2] == "Paste Style"
    assert all(StyleSnapshot.capture(a) == snapshot for a in (first, second))
    recording_after = tracker.capture_recording_state()
    commands = tuple(tracker.changes.values())
    assert manager.capture_interaction_state() == interaction_before
    tracker.backEdit()
    assert [StyleSnapshot.capture(a) for a in (first, second)] == before
    assert tracker.changes == {}
    tracker.forwardEdit()
    assert all(StyleSnapshot.capture(a) == snapshot for a in (first, second))
    assert tracker.capture_recording_state() == recording_after
    assert not manager.paste_style()  # Repeating an identical paste is a no-op.
    assert len(tracker.edits) == 1
    tracker.backEdit()
    for target, command in commands:
        exec(f"target{command}", {"target": target, "np": np, "plt": plt})
    fig.canvas.draw()
    assert all(StyleSnapshot.capture(a) == snapshot for a in (first, second))
    for artist, original in zip((first, second), unchanged):
        if kind == "text":
            assert (artist.get_position(), artist.get_text(), artist.get_rotation(),
                    artist.get_fontfamily(), artist.get_alpha(), artist.get_transform()) == original
        else:
            np.testing.assert_array_equal(artist.get_xdata(), original[0][0])
            np.testing.assert_array_equal(artist.get_ydata(), original[0][1])
            assert artist._unscaled_dash_pattern == original[1]
            assert artist.get_marker() == original[2]
            assert artist.get_transform() is original[3]


def test_paste_rejects_mixed_selection_without_mutation(scene):
    _fig, ax, manager, tracker, source, first, _second = scene()
    line, = ax.plot([0, 1], [0, 1])
    manager.make_draggable(line)
    manager.select_element(source)
    manager.copy_style()
    manager.select_elements([first, line], primary=first)
    before = StyleSnapshot.capture(first)
    with pytest.raises(PropertyPreflightError, match="concrete artist type"):
        manager.paste_style()
    assert StyleSnapshot.capture(first) == before
    assert tracker.edits == [] and tracker.changes == {}


@pytest.mark.parametrize("blocked", ["lock", "hide", "isolate", "group"])
def test_paste_rejects_stale_unavailable_selection_atomically(scene, blocked):
    _fig, _ax, manager, tracker, source, first, second = scene()
    manager.select_element(source)
    manager.copy_style()
    manager.select_elements([first, second], primary=first)
    editor = manager._ensure_editor_scene()
    if blocked == "lock":
        editor.set_locked([second], True)
    elif blocked == "hide":
        second.set_visible(False)
    else:
        group = editor.create_group([source, first], name="Scope")
        if blocked == "isolate":
            manager._ensure_selection_kernel().enter_isolation(group)
    before = StyleSnapshot.capture(first)
    with pytest.raises(PropertyPreflightError):
        manager.paste_style()
    assert StyleSnapshot.capture(first) == before
    assert tracker.edits == [] and tracker.changes == {}


@pytest.mark.parametrize("failure", ["setter", "recording", "history"])
def test_paste_failure_rolls_back_all_properties_recording_and_history(scene, failure):
    _fig, _ax, manager, tracker, source, first, second = scene()
    manager.select_element(source)
    manager.copy_style()
    manager.select_elements([first, second], primary=first)
    before = [StyleSnapshot.capture(a) for a in (first, second)]
    recording_before = tracker.capture_recording_state()
    if failure == "setter":
        original = second.set_weight

        def fail(value):
            original(value)
            if value == "bold":
                raise RuntimeError("injected failure")
        second.set_weight = fail
    elif failure == "recording":
        original = tracker.addNewTextChange

        def fail(target):
            original(target)
            if target is second:
                raise RuntimeError("injected failure")
        tracker.addNewTextChange = fail
    else:
        original = tracker.addEdit

        def fail(edit):
            original(edit)
            raise RuntimeError("injected failure")
        tracker.addEdit = fail
    with pytest.raises(RuntimeError, match="injected failure"):
        manager.paste_style()
    assert [StyleSnapshot.capture(a) for a in (first, second)] == before
    assert tracker.capture_recording_state() == recording_before
    assert tracker.edits == [] and tracker.last_edit == -1


def test_unsupported_copy_keeps_clipboard_and_empty_actions_are_non_mutating(scene):
    _fig, ax, manager, tracker, source, first, _second = scene()
    with pytest.raises(PropertyPreflightError, match="Copy a style"):
        manager.paste_style()
    with pytest.raises(PropertyPreflightError, match="Select an object"):
        manager.select_same()
    manager.select_element(source)
    snapshot = manager.copy_style()
    manager.select_element(ax)
    with pytest.raises(PropertyPreflightError, match="ordinary Text and Line2D"):
        manager.copy_style()
    manager.select_element(first)
    assert manager._style_clipboard is snapshot
    assert tracker.edits == [] and tracker.changes == {}


def test_edit_menu_routes_actions_to_current_figure():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = PlotWindow(1)
    calls = []
    dragger = SimpleNamespace(
        select_same=lambda criterion: calls.append(criterion),
        copy_style=lambda: calls.append("copy"),
        paste_style=lambda: calls.append("paste"),
    )
    window.fig = SimpleNamespace(figure_dragger=dragger)
    actions = {action.text(): action for action in window.menu_edit.actions()}
    try:
        for name in ("Select Same Type", "Select Same Color", "Copy Style", "Paste Style"):
            actions[name].trigger()
        assert calls == ["type", "color", "copy", "paste"]
        assert actions["Copy Style"].shortcut().toString() == "Ctrl+Alt+C"
        assert actions["Paste Style"].shortcut().toString() == "Ctrl+Alt+V"
    finally:
        window.fig = None
        window.close()
    assert app is not None


@pytest.mark.parametrize("kind, property_name, copied_value", [
    ("text", "fontsize", 18), ("line", "linewidth", 4),
])
def test_failed_style_undo_redo_keeps_history_and_recording_atomic(
    scene, kind, property_name, copied_value,
):
    _fig, _ax, manager, tracker, source, first, second = scene(kind)
    manager.select_element(source)
    manager.copy_style()
    manager.select_elements([first, second], primary=first)
    before = [StyleSnapshot.capture(a) for a in (first, second)]
    original = getattr(second, f"set_{property_name}")
    old_value = getattr(second, f"get_{property_name}")()
    failure = [None]

    def setter(value):
        original(value)
        if value == failure[0]:
            raise RuntimeError("injected history failure")

    setattr(second, f"set_{property_name}", setter)
    assert manager.paste_style()
    after = [StyleSnapshot.capture(a) for a in (first, second)]
    recording_after = tracker.capture_recording_state()
    failure[0] = old_value
    with pytest.raises(RuntimeError, match="injected history failure"):
        tracker.backEdit()
    assert [StyleSnapshot.capture(a) for a in (first, second)] == after
    assert tracker.capture_recording_state() == recording_after
    assert tracker.last_edit == 0
    failure[0] = None
    tracker.backEdit()
    recording_before = tracker.capture_recording_state()
    failure[0] = copied_value
    with pytest.raises(RuntimeError, match="injected history failure"):
        tracker.forwardEdit()
    assert [StyleSnapshot.capture(a) for a in (first, second)] == before
    assert tracker.capture_recording_state() == recording_before
    assert tracker.last_edit == -1
    failure[0] = None
    tracker.forwardEdit()
    assert [StyleSnapshot.capture(a) for a in (first, second)] == after


@pytest.mark.parametrize("blocked", ["hide", "lock"])
def test_cross_axes_group_cannot_bypass_real_ownership_protection(scene, blocked):
    fig, ax, manager, tracker, source, first, second = scene()
    other_axes = fig.add_axes([.1, .1, .2, .2])
    other = other_axes.text(.5, .5, "other")
    manager.make_axes_draggable([other_axes])
    manager.select_element(source)
    manager.copy_style()
    manager._ensure_editor_scene().create_group([first, other], name="Cross Axes")
    manager.set_selection_mode(SelectionMode.DIRECT)
    manager.select_elements([first, other], primary=other)
    if blocked == "hide":
        ax.set_visible(False)
    else:
        manager._ensure_editor_scene().set_locked([ax], True)
    before = StyleSnapshot.capture(other)
    with pytest.raises(PropertyPreflightError):
        manager.paste_style()
    assert StyleSnapshot.capture(other) == before
    manager.select_element(other)
    assert manager.select_same("type") == [other]
    assert tracker.edits == [] and tracker.changes == {}


def test_paste_rejects_unknown_text_replay_without_losing_prior_edit(scene):
    _fig, _ax, manager, tracker, source, first, second = scene()
    manager.select_element(source)
    manager.copy_style()
    manager.select_elements([first, second], primary=first)
    second.set_clip_on(False)
    tracker.addChange(second, ".set_clip_on(False)")
    before = [StyleSnapshot.capture(a) for a in (first, second)]
    recording = tracker.capture_recording_state()
    with pytest.raises(PropertyPreflightError, match="cannot safely preserve"):
        manager.paste_style()
    assert [StyleSnapshot.capture(a) for a in (first, second)] == before
    assert tracker.capture_recording_state() == recording
    assert tracker.edits == []


def test_paste_preserves_previous_independent_text_edits_on_sorted_replay(scene):
    from pylustrator.property_transactions import PropertyPlan

    fig, _ax, manager, tracker, source, first, _second = scene()
    PropertyPlan.for_targets([first], {
        "alpha": .35, "fontfamily": ["monospace", "sans-serif"],
        "zorder": 7, "label": "retained label",
    }).commit()
    prior_recording = tracker.capture_recording_state()
    manager.select_element(source)
    expected = manager.copy_style()
    manager.select_element(first)
    assert manager.paste_style()
    commands = tracker.sorted_changes()
    assert any(".set_alpha(0.35)" in command for command in commands)
    assert any(".set_fontfamily(" in command for command in commands)
    assert any(".set_zorder(7)" in command for command in commands)
    tracker.backEdit()
    assert tracker.capture_recording_state() == prior_recording
    tracker.forwardEdit()
    first.set_alpha(None)
    first.set_fontfamily(["serif", "sans-serif"])
    first.set_zorder(3)
    first.set_label("")
    for command in commands:
        exec(command, {"plt": plt, "np": np})
    fig.canvas.draw()
    assert StyleSnapshot.capture(first) == expected
    assert first.get_alpha() == .35
    assert first.get_fontfamily() == ["monospace", "sans-serif"]
    assert first.get_zorder() == 7
    assert first.get_label() == "retained label"
