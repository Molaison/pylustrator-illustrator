from __future__ import annotations

from types import SimpleNamespace

from pylustrator.drag_helper import DragManager


class _Canvas:
    def __init__(self) -> None:
        self.draw_calls = 0

    def draw(self) -> None:
        self.draw_calls += 1


class _Tracker:
    def __init__(self, canvas: _Canvas) -> None:
        self.canvas = canvas
        self.undo_calls = 0
        self.redo_calls = 0

    def backEdit(self) -> None:
        self.undo_calls += 1
        self.canvas.draw()

    def forwardEdit(self) -> None:
        self.redo_calls += 1
        self.canvas.draw()


def _manager():
    canvas = _Canvas()
    tracker = _Tracker(canvas)
    manager = DragManager.__new__(DragManager)
    manager.figure = SimpleNamespace(canvas=canvas, change_tracker=tracker)
    manager.selection = SimpleNamespace(targets=[])
    manager._cancel_active_pointer_transform = lambda: False
    manager._update_interaction_controls = lambda: None
    manager._notify_selected_element_changed = lambda: None
    return manager, canvas, tracker


def test_drag_manager_undo_uses_history_controllers_single_draw() -> None:
    manager, canvas, tracker = _manager()

    manager.undo()

    assert tracker.undo_calls == 1
    assert canvas.draw_calls == 1


def test_drag_manager_redo_uses_history_controllers_single_draw() -> None:
    manager, canvas, tracker = _manager()

    manager.redo()

    assert tracker.redo_calls == 1
    assert canvas.draw_calls == 1
