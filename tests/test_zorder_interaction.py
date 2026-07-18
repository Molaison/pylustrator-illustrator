from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest
from matplotlib.backend_bases import KeyEvent, MouseEvent
from matplotlib.patches import Rectangle
from qtpy import QtGui, QtWidgets

from pylustrator.change_tracker import ChangeTracker
from pylustrator.commands import semantic_equal
from pylustrator.interaction import SelectionMode
from test_selection_indicator import attach_drag_manager


def paint_order(parent, artists):
    child_order = {
        id(child): index for index, child in enumerate(parent.get_children())
    }
    return sorted(
        artists,
        key=lambda artist: (
            float(artist.get_zorder()),
            child_order[id(artist)],
        ),
    )


def overlapping_rectangles(ax, zorders):
    return tuple(
        ax.add_patch(
            Rectangle(
                (0.2, 0.2),
                0.6,
                0.6,
                zorder=zorder,
                label=f"rectangle-{index}",
            )
        )
        for index, zorder in enumerate(zorders)
    )


def install_real_tracker(fig):
    tracker = ChangeTracker.__new__(ChangeTracker)
    tracker.figure = fig
    tracker.changes = {}
    tracker.saved = True
    tracker.edits = []
    tracker.last_edit = -1
    tracker.update_changes_signal = None
    tracker.no_save = False
    fig.change_tracker = tracker
    return tracker


def test_bring_forward_crosses_next_visible_sibling_and_updates_hit_stack() -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
    bottom, middle, top = overlapping_rectangles(ax, (1, 10, 30))
    hidden = ax.add_patch(
        Rectangle((0.2, 0.2), 0.6, 0.6, zorder=5, visible=False)
    )
    fig.canvas.draw()
    manager = attach_drag_manager(fig)
    manager.select_element(bottom)
    x, y = ax.transData.transform((0.5, 0.5))
    event = MouseEvent("button_press_event", fig.canvas, x, y, button=1)

    assert manager.change_selection_zorder("forward")

    assert paint_order(ax, (bottom, middle, top)) == [middle, bottom, top]
    assert hidden.get_zorder() == 5
    assert manager.get_hit_candidates(event)[:3] == (top, bottom, middle)
    assert [target.target for target in manager.selection.targets] == [bottom]

    manager.select_element(top)
    assert manager.change_selection_zorder("backward")
    assert paint_order(ax, (bottom, middle, top)) == [middle, top, bottom]
    assert manager.get_hit_candidates(event)[:3] == (bottom, top, middle)
    manager.selection.clear_targets()
    plt.close(fig)
    assert app is not None


def test_equal_zorder_uses_child_order_and_undo_redo_each_need_one_draw() -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
    first, second, third = overlapping_rectangles(ax, (3, 3, 3))
    fig.canvas.draw()
    manager = attach_drag_manager(fig)
    manager.select_element(first)
    tracker = fig.change_tracker
    original_draw = fig.canvas.draw
    draw_calls = 0

    def counted_draw(*args, **kwargs):
        nonlocal draw_calls
        draw_calls += 1
        return original_draw(*args, **kwargs)

    fig.canvas.draw = counted_draw
    assert manager.change_selection_zorder("forward")
    assert draw_calls == 1
    assert paint_order(ax, (first, second, third)) == [second, first, third]
    recorded = list(tracker.changes)
    assert recorded
    assert all(command.startswith(".set_zorder(") for _target, command in recorded)

    draw_calls = 0
    tracker.edit[0]()
    assert draw_calls == 0
    fig.canvas.draw()
    assert draw_calls == 1
    assert paint_order(ax, (first, second, third)) == [first, second, third]
    assert tracker.changes == []

    draw_calls = 0
    tracker.edit[1]()
    assert draw_calls == 0
    fig.canvas.draw()
    assert draw_calls == 1
    assert paint_order(ax, (first, second, third)) == [second, first, third]
    assert tracker.changes == recorded

    tracker.edit[0]()
    for target, command in recorded:
        exec(f"target{command}", {"target": target})
    fig.canvas.draw()
    assert paint_order(ax, (first, second, third)) == [second, first, third]
    manager.selection.clear_targets()
    plt.close(fig)
    assert app is not None


def test_multi_selection_keeps_internal_order_key_selection_and_scope() -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
    first, second, third, fourth = overlapping_rectangles(ax, (3, 3, 3, 3))
    fig.canvas.draw()
    manager = attach_drag_manager(fig)
    manager.select_elements([first, third], primary=third)
    group = manager.group_selection("Pair")
    assert manager.enter_isolation(group)
    manager.set_selection_mode(SelectionMode.DIRECT)
    manager.select_elements([first, third], primary=third)
    manager.selection.set_alignment_reference("key_object", key=third)
    interaction_before = manager.capture_interaction_state()

    assert manager.change_selection_zorder("forward")

    assert paint_order(ax, (first, second, third, fourth)) == [
        second,
        first,
        fourth,
        third,
    ]
    assert semantic_equal(manager.capture_interaction_state(), interaction_before)
    assert [target.target for target in manager.selection.targets] == [first, third]
    assert manager.selection.alignment_key is third
    assert manager.isolation_breadcrumbs == ("Pair",)

    fig.change_tracker.edit[0]()
    assert paint_order(ax, (first, second, third, fourth)) == [
        first,
        second,
        third,
        fourth,
    ]
    assert semantic_equal(manager.capture_interaction_state(), interaction_before)
    manager.selection.clear_targets()
    plt.close(fig)
    assert app is not None


def test_front_and_back_stay_within_the_same_paint_parent() -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
    first, selected, third = overlapping_rectangles(ax, (1, 5, 9))
    fig.canvas.draw()
    manager = attach_drag_manager(fig)
    manager.select_element(selected)

    assert manager.change_selection_zorder("front")
    assert paint_order(ax, (first, selected, third)) == [first, third, selected]

    fig.change_tracker.edit[0]()
    assert manager.change_selection_zorder("back")
    assert paint_order(ax, (first, selected, third)) == [selected, first, third]
    manager.selection.clear_targets()
    plt.close(fig)
    assert app is not None


def test_arrange_rejects_cross_axes_selection_without_mutation_or_history() -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    fig, (first_axes, second_axes) = plt.subplots(
        1, 2, figsize=(6, 3), dpi=100
    )
    first = first_axes.add_patch(Rectangle((0.2, 0.2), 0.6, 0.6, zorder=2))
    second = second_axes.add_patch(Rectangle((0.2, 0.2), 0.6, 0.6, zorder=8))
    fig.canvas.draw()
    manager = attach_drag_manager(fig)
    manager.select_elements([first, second], primary=second)
    values_before = (first.get_zorder(), second.get_zorder())
    recording_before = manager.figure.change_tracker.capture_recording_state()
    history_before = list(manager.figure.change_tracker.edits)

    with pytest.raises(ValueError, match="one paint container"):
        manager.change_selection_zorder("front")

    manager.selection.keyPressEvent(
        KeyEvent("key_press_event", fig.canvas, "pageup")
    )
    assert (first.get_zorder(), second.get_zorder()) == values_before
    assert manager.figure.change_tracker.capture_recording_state() == recording_before
    assert manager.figure.change_tracker.edits == history_before
    assert [target.target for target in manager.selection.targets] == [first, second]
    manager.selection.clear_targets()
    plt.close(fig)
    assert app is not None


def test_arrange_failure_rolls_back_artists_recording_history_and_selection() -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
    selected, sibling = overlapping_rectangles(ax, (1, 10))
    fig.canvas.draw()
    manager = attach_drag_manager(fig)
    manager.select_element(selected)
    values_before = (selected.get_zorder(), sibling.get_zorder())
    tracker = fig.change_tracker
    recording_before = tracker.capture_recording_state()
    history_before = list(tracker.edits)
    original_set_zorder = sibling.set_zorder

    def fail_destination(value):
        if value == 1:
            raise RuntimeError("injected z-order failure")
        return original_set_zorder(value)

    sibling.set_zorder = fail_destination
    try:
        with pytest.raises(RuntimeError, match="injected z-order failure"):
            manager.change_selection_zorder("forward")
    finally:
        sibling.set_zorder = original_set_zorder

    assert (selected.get_zorder(), sibling.get_zorder()) == values_before
    assert tracker.capture_recording_state() == recording_before
    assert tracker.edits == history_before
    assert [target.target for target in manager.selection.targets] == [selected]
    manager.selection.clear_targets()
    plt.close(fig)
    assert app is not None


def test_zorder_undo_redo_setter_failures_restore_all_siblings_atomically() -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
    first, second, third = overlapping_rectangles(ax, (1, 10, 20))
    fig.canvas.draw()
    manager = attach_drag_manager(fig)
    tracker = install_real_tracker(fig)
    manager.select_elements([first, second], primary=second)
    original_second_setter = second.set_zorder
    failure_value = {"value": None}

    def conditional_set_zorder(value):
        if value == failure_value["value"]:
            raise RuntimeError("injected z-order history failure")
        return original_second_setter(value)

    second.set_zorder = conditional_set_zorder
    assert manager.change_selection_zorder("forward")
    after = (first.get_zorder(), second.get_zorder(), third.get_zorder())
    assert after == (10, 20, 1)
    recording_after = tracker.capture_recording_state()
    original_draw = fig.canvas.draw
    draw_calls = 0

    def counted_draw(*args, **kwargs):
        nonlocal draw_calls
        draw_calls += 1
        return original_draw(*args, **kwargs)

    fig.canvas.draw = counted_draw
    try:
        failure_value["value"] = 10
        with pytest.raises(RuntimeError, match="z-order history failure"):
            tracker.backEdit()
        assert (first.get_zorder(), second.get_zorder(), third.get_zorder()) == after
        assert tracker.capture_recording_state() == recording_after
        assert tracker.last_edit == 0
        assert draw_calls == 0

        failure_value["value"] = None
        tracker.backEdit()
        assert (first.get_zorder(), second.get_zorder(), third.get_zorder()) == (
            1,
            10,
            20,
        )
        assert tracker.last_edit == -1
        assert draw_calls == 1
        recording_before = tracker.capture_recording_state()

        draw_calls = 0
        failure_value["value"] = 20
        with pytest.raises(RuntimeError, match="z-order history failure"):
            tracker.forwardEdit()
        assert (first.get_zorder(), second.get_zorder(), third.get_zorder()) == (
            1,
            10,
            20,
        )
        assert tracker.capture_recording_state() == recording_before
        assert tracker.last_edit == -1
        assert draw_calls == 0

        failure_value["value"] = None
        tracker.forwardEdit()
        assert (first.get_zorder(), second.get_zorder(), third.get_zorder()) == after
        assert tracker.capture_recording_state() == recording_after
        assert tracker.last_edit == 0
        assert draw_calls == 1
    finally:
        second.set_zorder = original_second_setter
        fig.canvas.draw = original_draw

    assert [target.target for target in manager.selection.targets] == [first, second]
    manager.selection.clear_targets()
    plt.close(fig)
    assert app is not None


def test_redo_action_accepts_both_illustrator_and_legacy_shortcuts() -> None:
    from pylustrator.QtGuiDrag import PlotWindow

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = PlotWindow(1)
    shortcuts = {
        shortcut.toString(QtGui.QKeySequence.PortableText)
        for shortcut in window.redo_act.shortcuts()
    }

    assert shortcuts == {"Ctrl+Y", "Ctrl+Shift+Z"}
    window.deleteLater()
    assert app is not None
