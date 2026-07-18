from __future__ import annotations

from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest
from matplotlib.patches import Rectangle
from qtpy import QtWidgets

from pylustrator.QLinkableWidgets import TextWidget
from pylustrator.change_tracker import ChangeTracker, init_figure
from pylustrator.components.qitem_properties import TextPropertiesWidget
from pylustrator.property_transactions import (
    PropertyPlan,
    PropertyPreflightError,
)


class Signal:
    def __init__(self):
        self.callbacks = []
        self.emissions = 0

    def connect(self, callback):
        self.callbacks.append(callback)

    def emit(self, *args):
        self.emissions += 1
        for callback in tuple(self.callbacks):
            callback(*args)


def install_tracker(fig, *, selected=()):
    init_figure(fig)
    tracker = ChangeTracker.__new__(ChangeTracker)
    tracker.figure = fig
    tracker.changes = {}
    tracker.saved = True
    tracker.edits = []
    tracker.last_edit = -1
    tracker.update_changes_signal = None
    tracker.no_save = False
    fig.change_tracker = tracker
    fig.signals = SimpleNamespace(figure_selection_property_changed=Signal())
    fig.selection = SimpleNamespace(
        targets=[SimpleNamespace(target=target) for target in selected]
    )
    return tracker


@pytest.fixture
def text_and_rectangle():
    fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
    text = ax.text(0.25, 0.5, "text", fontsize=10)
    rectangle = Rectangle((0.5, 0.4), 0.2, 0.2, label="patch")
    ax.add_patch(rectangle)
    fig.canvas.draw()
    tracker = install_tracker(fig, selected=(text, rectangle))
    try:
        yield fig, text, rectangle, tracker
    finally:
        plt.close(fig)


def test_mixed_selection_rejects_missing_setter_before_first_mutation(
    text_and_rectangle,
) -> None:
    _fig, text, rectangle, tracker = text_and_rectangle
    fontsize_before = text.get_fontsize()
    rectangle_label_before = rectangle.get_label()
    recording_before = tracker.capture_recording_state()

    with pytest.raises(PropertyPreflightError, match="Rectangle.*fontsize"):
        PropertyPlan.for_selection(
            text,
            (text, rectangle),
            "fontsize",
            17,
        )

    assert text.get_fontsize() == fontsize_before
    assert rectangle.get_label() == rectangle_label_before
    assert tracker.capture_recording_state() == recording_before
    assert tracker.edits == []
    assert tracker.last_edit == -1


def test_generic_link_disables_mixed_unsupported_property_without_fake_replay(
    text_and_rectangle,
) -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    fig, text, _rectangle, tracker = text_and_rectangle
    parent = QtWidgets.QWidget()
    layout = QtWidgets.QVBoxLayout(parent)
    target_changed = Signal()
    widget = TextWidget(layout, "Text:")
    widget.link("text", target_changed)

    target_changed.emit(text)
    assert not widget.isEnabled()

    widget.setText("must not be committed")
    widget.updateLink()

    assert text.get_text() == "text"
    assert tracker.changes == {}
    assert tracker.edits == []
    assert app is not None


def test_text_property_panel_disables_font_edit_for_mixed_selection(
    text_and_rectangle,
) -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    _fig, text, rectangle, tracker = text_and_rectangle
    parent = QtWidgets.QWidget()
    layout = QtWidgets.QVBoxLayout(parent)
    widget = TextPropertiesWidget(layout)
    before = text.get_fontsize()

    widget.setTarget([text, rectangle])
    assert not widget.font_size.isEnabled()
    widget.changeFontSize(19)

    assert text.get_fontsize() == before
    assert tracker.changes == {}
    assert tracker.edits == []
    assert app is not None


def test_text_property_panel_resolves_axis_label_semantic_owners_atomically() -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    fig, (first_axes, second_axes) = plt.subplots(
        1, 2, figsize=(6, 3), dpi=100
    )
    first_axes.set_ylabel("first")
    second_axes.set_ylabel("second")
    fig.canvas.draw()
    first = first_axes.yaxis.get_label()
    second = second_axes.yaxis.get_label()
    tracker = install_tracker(fig, selected=(first_axes, second_axes))
    before = (first.get_fontsize(), second.get_fontsize())
    parent = QtWidgets.QWidget()
    layout = QtWidgets.QVBoxLayout(parent)
    widget = TextPropertiesWidget(layout)

    widget.setTarget([first, first_axes, second_axes])
    assert widget.font_size.isEnabled()
    widget.changeFontSize(18)

    assert (first.get_fontsize(), second.get_fontsize()) == (18, 18)
    assert len(tracker.edits) == 1
    tracker.backEdit()
    assert (first.get_fontsize(), second.get_fontsize()) == before
    tracker.forwardEdit()
    assert (first.get_fontsize(), second.get_fontsize()) == (18, 18)
    assert app is not None
    plt.close(fig)


def test_setter_failure_rolls_back_every_artist_recording_and_history() -> None:
    fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
    first = ax.text(0.2, 0.5, "first", fontsize=10)
    second = ax.text(0.7, 0.5, "second", fontsize=11)
    fig.canvas.draw()
    tracker = install_tracker(fig, selected=(first, second))
    first_before = first.get_fontsize()
    second_before = second.get_fontsize()
    recording_before = tracker.capture_recording_state()
    original_second_setter = second.set_fontsize

    def fail_requested_value(value):
        if value == 18:
            raise RuntimeError("injected setter failure")
        return original_second_setter(value)

    second.set_fontsize = fail_requested_value
    plan = PropertyPlan.for_targets((first, second), {"fontsize": 18})

    try:
        with pytest.raises(RuntimeError, match="injected setter failure"):
            plan.commit("Change font size")
    finally:
        second.set_fontsize = original_second_setter

    assert first.get_fontsize() == first_before
    assert second.get_fontsize() == second_before
    assert tracker.capture_recording_state() == recording_before
    assert tracker.edits == []
    assert tracker.last_edit == -1
    plt.close(fig)


def test_recording_failure_rolls_back_state_generated_source_and_history() -> None:
    fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
    first = ax.text(0.2, 0.5, "first", fontsize=10)
    second = ax.text(0.7, 0.5, "second", fontsize=11)
    fig.canvas.draw()
    tracker = install_tracker(fig, selected=(first, second))
    before = (first.get_fontsize(), second.get_fontsize())
    recording_before = tracker.capture_recording_state()
    original_record = tracker.addNewTextChange

    def fail_after_second_record(target):
        original_record(target)
        if target is second:
            raise RuntimeError("injected recording failure")

    tracker.addNewTextChange = fail_after_second_record
    plan = PropertyPlan.for_targets((first, second), {"fontsize": 16})

    try:
        with pytest.raises(RuntimeError, match="injected recording failure"):
            plan.commit("Change font size")
    finally:
        tracker.addNewTextChange = original_record

    assert (first.get_fontsize(), second.get_fontsize()) == before
    assert tracker.capture_recording_state() == recording_before
    assert tracker.edits == []
    assert tracker.last_edit == -1
    plt.close(fig)


def test_history_failure_after_add_edit_rolls_back_the_whole_transaction() -> None:
    fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
    text = ax.text(0.5, 0.5, "text", fontsize=10)
    fig.canvas.draw()
    tracker = install_tracker(fig, selected=(text,))
    before = text.get_fontsize()
    recording_before = tracker.capture_recording_state()
    history_before = (list(tracker.edits), tracker.last_edit)
    original_add_edit = tracker.addEdit

    def fail_after_history_mutation(edit):
        original_add_edit(edit)
        raise RuntimeError("injected history failure")

    tracker.addEdit = fail_after_history_mutation
    plan = PropertyPlan.for_targets((text,), {"fontsize": 20})

    try:
        with pytest.raises(RuntimeError, match="injected history failure"):
            plan.commit("Change font size")
    finally:
        tracker.addEdit = original_add_edit

    assert text.get_fontsize() == before
    assert tracker.capture_recording_state() == recording_before
    assert (tracker.edits, tracker.last_edit) == history_before
    plt.close(fig)


def test_common_property_has_one_atomic_undo_redo_and_real_commands(
    text_and_rectangle,
) -> None:
    _fig, text, rectangle, tracker = text_and_rectangle
    before = (text.get_label(), rectangle.get_label())

    plan = PropertyPlan.for_selection(
        text,
        (text, rectangle),
        "label",
        "shared",
    )
    assert plan.commit("Change label")

    assert (text.get_label(), rectangle.get_label()) == ("shared", "shared")
    assert len(tracker.edits) == 1
    assert tracker.last_edit == 0
    commands = [command for _target, command in tracker.changes.values()]
    assert commands == [".set_label('shared')", ".set_label('shared')"]
    assert not any("set_text" in command for command in commands)

    tracker.backEdit()
    assert (text.get_label(), rectangle.get_label()) == before
    assert tracker.changes == {}

    tracker.forwardEdit()
    assert (text.get_label(), rectangle.get_label()) == ("shared", "shared")
    assert [command for _target, command in tracker.changes.values()] == commands


def test_mixed_property_undo_redo_failures_restore_entry_side_atomically(
    text_and_rectangle,
) -> None:
    fig, text, rectangle, tracker = text_and_rectangle
    before = (text.get_label(), rectangle.get_label())
    recording_before = tracker.capture_recording_state()
    original_set_label = rectangle.set_label
    failure_value = {"value": None}

    def conditional_set_label(value):
        if value == failure_value["value"]:
            raise RuntimeError("injected mixed-property history failure")
        return original_set_label(value)

    rectangle.set_label = conditional_set_label
    plan = PropertyPlan.for_selection(
        text,
        (text, rectangle),
        "label",
        "shared",
    )
    assert plan.commit("Change label")
    recording_after = tracker.capture_recording_state()
    original_draw = fig.canvas.draw
    draw_calls = 0

    def counted_draw(*args, **kwargs):
        nonlocal draw_calls
        draw_calls += 1
        return original_draw(*args, **kwargs)

    fig.canvas.draw = counted_draw
    try:
        failure_value["value"] = before[1]
        with pytest.raises(RuntimeError, match="mixed-property history failure"):
            tracker.backEdit()
        assert (text.get_label(), rectangle.get_label()) == ("shared", "shared")
        assert tracker.capture_recording_state() == recording_after
        assert tracker.last_edit == 0
        assert draw_calls == 0

        failure_value["value"] = None
        tracker.backEdit()
        assert (text.get_label(), rectangle.get_label()) == before
        assert tracker.capture_recording_state() == recording_before
        assert tracker.last_edit == -1
        assert draw_calls == 1

        draw_calls = 0
        failure_value["value"] = "shared"
        with pytest.raises(RuntimeError, match="mixed-property history failure"):
            tracker.forwardEdit()
        assert (text.get_label(), rectangle.get_label()) == before
        assert tracker.capture_recording_state() == recording_before
        assert tracker.last_edit == -1
        assert draw_calls == 0

        failure_value["value"] = None
        tracker.forwardEdit()
        assert (text.get_label(), rectangle.get_label()) == ("shared", "shared")
        assert tracker.capture_recording_state() == recording_after
        assert tracker.last_edit == 0
        assert draw_calls == 1
    finally:
        rectangle.set_label = original_set_label
        fig.canvas.draw = original_draw


def test_history_rollback_failure_is_attached_to_original_setter_error() -> None:
    fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
    first = ax.text(0.2, 0.5, "first", fontsize=10)
    second = ax.text(0.7, 0.5, "second", fontsize=11)
    fig.canvas.draw()
    tracker = install_tracker(fig, selected=(first, second))
    before = (first.get_fontsize(), second.get_fontsize())
    original_second_setter = second.set_fontsize
    failure_enabled = {"value": False}

    def fail_destination_and_rollback(value):
        if failure_enabled["value"] and value == before[1]:
            raise RuntimeError("injected undo destination failure")
        if failure_enabled["value"] and value == 20:
            raise RuntimeError("injected undo rollback failure")
        return original_second_setter(value)

    second.set_fontsize = fail_destination_and_rollback
    plan = PropertyPlan.for_targets((first, second), {"fontsize": 20})
    assert plan.commit("Change font size")
    recording_after = tracker.capture_recording_state()
    failure_enabled["value"] = True

    try:
        with pytest.raises(
            RuntimeError, match="injected undo destination failure"
        ) as raised:
            tracker.backEdit()
    finally:
        failure_enabled["value"] = False
        second.set_fontsize = original_second_setter

    assert (first.get_fontsize(), second.get_fontsize()) == (20, 20)
    assert tracker.capture_recording_state() == recording_after
    assert tracker.last_edit == 0
    failures = raised.value.pylustrator_rollback_failures
    assert len(failures) == 1
    assert failures[0][0] is second
    assert "injected undo rollback failure" in str(failures[0][1])
    plt.close(fig)
