from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.backend_bases import KeyEvent, MouseEvent
from qtpy import QtCore, QtGui, QtWidgets

from pylustrator.interaction import SelectionMode
from test_selection_indicator import attach_drag_manager


def _figure_with_manager():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
    manager = attach_drag_manager(fig)
    return app, fig, ax, manager


def test_enter_starts_draft_only_editor_and_ctrl_enter_commits_one_atomic_edit():
    app, fig, ax, manager = _figure_with_manager()
    text = ax.text(0.3, 0.6, r"before\nliteral")
    fig.canvas.draw()
    manager.select_element(text)

    manager.key_press_event(KeyEvent("key_press_event", fig.canvas, "enter"))

    editor = manager.inline_text_editor
    assert editor.active
    assert editor.target is text
    assert editor.widget.toPlainText() == r"before\nliteral"
    assert text.get_text() == r"before\nliteral"
    assert fig.change_tracker.edits == []
    assert fig.change_tracker.changes == []

    editor.widget.setPlainText("first line\nsecond \\n literal")
    manager.key_press_event(KeyEvent("key_press_event", fig.canvas, "ctrl+enter"))

    assert not editor.active
    assert text.get_text() == "first line\nsecond \\n literal"
    assert len(fig.change_tracker.edits) == 1
    assert fig.change_tracker.edit[2] == "Edit text"
    assert fig.change_tracker.text_change_count == 1

    manager.selection.clear_targets()
    plt.close(fig)
    assert app is not None


def test_escape_cancels_inline_draft_without_clearing_selection_or_switching_tool():
    app, fig, ax, manager = _figure_with_manager()
    text = ax.text(0.25, 0.55, "unchanged")
    fig.canvas.draw()
    manager.set_selection_mode(SelectionMode.DIRECT)
    manager.select_element(text)
    manager.key_press_event(KeyEvent("key_press_event", fig.canvas, "return"))
    editor = manager.inline_text_editor
    editor.widget.setPlainText("discard me")

    manager.key_press_event(KeyEvent("key_press_event", fig.canvas, "v"))
    assert editor.active
    assert manager.selection_mode is SelectionMode.DIRECT
    assert text.get_text() == "unchanged"

    manager.key_press_event(KeyEvent("key_press_event", fig.canvas, "escape"))
    assert not editor.active
    assert text.get_text() == "unchanged"
    assert [target.target for target in manager.selection.targets] == [text]
    assert manager.selected_element is text
    assert fig.change_tracker.edits == []

    manager.selection.clear_targets()
    plt.close(fig)
    assert app is not None


def test_double_click_selected_text_opens_inline_editor_without_starting_drag():
    app, fig, ax, manager = _figure_with_manager()
    text = ax.text(0.45, 0.5, "double click")
    fig.canvas.draw()
    manager.make_draggable(text, ax)
    manager.set_selection_mode(SelectionMode.DIRECT)
    bbox = text.get_window_extent(fig.canvas.get_renderer())
    event = MouseEvent(
        "button_press_event",
        fig.canvas,
        (bbox.x0 + bbox.x1) / 2,
        (bbox.y0 + bbox.y1) / 2,
        button=1,
        dblclick=True,
    )

    manager.button_press_event0(event)

    assert manager.inline_text_editor.active
    assert manager.inline_text_editor.target is text
    assert manager.selected_element is text
    assert not manager.selection.got_artist
    manager.inline_text_editor.cancel()

    manager.selection.clear_targets()
    plt.close(fig)
    assert app is not None


def test_inline_tick_label_commit_uses_axis_transaction_and_survives_draw():
    app, fig, ax, manager = _figure_with_manager()
    ax.set_yticks([0, 1], labels=["zero", "one"])
    fig.canvas.draw()
    label = ax.yaxis.get_major_ticks()[1].label1
    manager.select_element(label)
    manager.key_press_event(KeyEvent("key_press_event", fig.canvas, "enter"))
    editor = manager.inline_text_editor
    assert editor.active
    editor.widget.setPlainText("edited one")

    assert editor.commit()
    fig.canvas.draw()

    assert label.get_text() == "edited one"
    assert len(fig.change_tracker.edits) == 1
    assert fig.change_tracker.edit[2] == "Change tick label text"
    assert fig.change_tracker.change[0] is ax

    manager.selection.clear_targets()
    plt.close(fig)
    assert app is not None


def test_offset_text_is_typed_denied_without_creating_overlay_or_history():
    app, fig, ax, manager = _figure_with_manager()
    ax.plot([0, 1], [1_000_000, 2_000_000])
    fig.canvas.draw()
    offset = ax.yaxis.offsetText
    editor = manager._ensure_inline_text_editor()

    original_show = QtWidgets.QToolTip.showText
    calls = []
    QtWidgets.QToolTip.showText = lambda *args, **kwargs: calls.append(args)
    try:
        assert editor.start(offset) is False
    finally:
        QtWidgets.QToolTip.showText = original_show

    assert not editor.active
    assert calls
    assert fig.change_tracker.edits == []
    assert fig.change_tracker.changes == []

    manager.selection.clear_targets()
    plt.close(fig)
    assert app is not None


def test_inline_commit_failure_keeps_editor_open_and_document_unchanged():
    app, fig, ax, manager = _figure_with_manager()
    text = ax.text(0.35, 0.45, "source")
    fig.canvas.draw()
    manager.select_element(text)
    manager.key_press_event(KeyEvent("key_press_event", fig.canvas, "enter"))
    editor = manager.inline_text_editor
    editor.widget.setPlainText("draft")
    text.set_text("external")

    assert editor.commit() is False
    assert editor.active
    assert text.get_text() == "external"
    assert fig.change_tracker.edits == []
    assert "changed" in editor.widget.toolTip().lower()
    assert not editor._committing
    editor.cancel()

    manager.selection.clear_targets()
    plt.close(fig)
    assert app is not None


def test_inline_commit_blocks_reentrant_focus_or_callback_commit(monkeypatch):
    app, fig, ax, manager = _figure_with_manager()
    text = ax.text(0.35, 0.45, "source")
    fig.canvas.draw()
    manager.select_element(text)
    manager.key_press_event(KeyEvent("key_press_event", fig.canvas, "enter"))
    editor = manager.inline_text_editor
    editor.widget.setPlainText("draft")
    plan_type = type(editor.plan)
    real_commit = plan_type.commit
    entries = []
    nested_results = []

    def reentrant_commit(plan, value):
        entries.append(value)
        nested_results.append(editor.commit())
        return real_commit(plan, value)

    monkeypatch.setattr(plan_type, "commit", reentrant_commit)

    assert editor.commit() is True
    assert entries == ["draft"]
    assert nested_results == [False]
    assert text.get_text() == "draft"
    assert len(fig.change_tracker.edits) == 1
    assert not editor.active
    assert not editor._committing

    manager.selection.clear_targets()
    plt.close(fig)
    assert app is not None


def test_inline_widget_enter_inserts_real_newline_and_escape_never_reaches_document():
    app, fig, ax, manager = _figure_with_manager()
    text = ax.text(0.2, 0.4, "a")
    fig.canvas.draw()
    manager.select_element(text)
    manager.key_press_event(KeyEvent("key_press_event", fig.canvas, "enter"))
    widget = manager.inline_text_editor.widget
    widget.moveCursor(QtGui.QTextCursor.End)
    event = QtGui.QKeyEvent(
        QtCore.QEvent.KeyPress,
        QtCore.Qt.Key_Return,
        QtCore.Qt.NoModifier,
    )

    widget.keyPressEvent(event)

    assert widget.toPlainText() == "a\n"
    assert text.get_text() == "a"
    assert fig.change_tracker.edits == []
    manager.inline_text_editor.cancel()

    manager.selection.clear_targets()
    plt.close(fig)
    assert app is not None
