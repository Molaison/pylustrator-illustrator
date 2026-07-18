from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import _pylab_helpers
from matplotlib.backends.qt_compat import QtWidgets

from pylustrator import QtGuiDrag
from pylustrator.change_tracker import init_figure
from pylustrator.drag_helper import DragManager
from pylustrator.QtGuiDrag import PlotWindow


def _callback_count(canvas, owner) -> int:
    count = 0
    for callbacks in canvas.callbacks.callbacks.values():
        for callback_ref in callbacks.values():
            callback = callback_ref()
            if getattr(callback, "__self__", None) is owner:
                count += 1
    return count


def _close_all_windows() -> None:
    for manager in _pylab_helpers.Gcf.figs.copy().values():
        window = getattr(manager.canvas.figure, "window", None)
        if isinstance(window, PlotWindow):
            window.deactivate()
    plt.close("all")
    _pylab_helpers.Gcf.destroy_all()


def test_show_close_and_reopen_has_one_live_session() -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    original_show = plt.show
    original_no_save = QtGuiDrag.no_save_allowed
    _close_all_windows()
    try:
        QtGuiDrag.initialize(disable_save=True)
        fig, _ax = plt.subplots()
        source_canvas = fig.canvas

        (window,) = QtGuiDrag.show(hide_window=True)
        manager = fig.figure_dragger
        selection = fig.selection
        embedded_canvas = fig.canvas
        callback_ids = (
            manager.c2,
            manager.c3,
            manager.c4,
            manager.c5,
            manager.c6,
            selection.c4,
        )

        (same_window,) = QtGuiDrag.show(hide_window=True)
        same_window.update()
        same_window.update()

        assert same_window is window
        assert fig.figure_dragger is manager
        assert fig.selection is selection
        assert fig.canvas is embedded_canvas
        assert callback_ids == (
            manager.c2,
            manager.c3,
            manager.c4,
            manager.c5,
            manager.c6,
            selection.c4,
        )
        assert manager.on_select.__self__ is manager
        assert _callback_count(embedded_canvas, manager) == 5
        assert _callback_count(embedded_canvas, selection) == 1

        assert window.close()
        app.processEvents()

        assert window._deactivated is True
        assert fig.window is None
        assert fig.signals is None
        assert fig.figure_dragger is None
        assert fig.selection is None
        assert fig.canvas is source_canvas
        assert manager._interaction_active is False
        assert _callback_count(embedded_canvas, manager) == 0
        assert _callback_count(embedded_canvas, selection) == 0

        (new_window,) = QtGuiDrag.show(hide_window=True)
        assert new_window is not window
        assert fig.figure_dragger is not manager
        assert fig.canvas is not embedded_canvas
        assert _callback_count(fig.canvas, fig.figure_dragger) == 5
        assert _callback_count(fig.canvas, fig.selection) == 1
        new_window.deactivate()
    finally:
        _close_all_windows()
        plt.show = original_show
        QtGuiDrag.no_save_allowed = original_no_save


def test_drag_manager_activate_deactivate_is_idempotent() -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    original_no_save = QtGuiDrag.no_save_allowed
    QtGuiDrag.no_save_allowed = True
    fig = plt.figure()
    source_canvas = fig.canvas
    window = PlotWindow()
    try:
        window.setFigure(fig)
        init_figure(fig)
        manager = DragManager(fig, True)
        window.configure_figure_manager(fig)
        window.update()
        canvas = fig.canvas
        selection = manager.selection

        assert manager.activate() is False
        assert _callback_count(canvas, manager) == 5
        assert _callback_count(canvas, selection) == 1

        assert manager.deactivate(redraw=False) is True
        assert manager.deactivate(redraw=False) is False
        assert _callback_count(canvas, manager) == 0
        assert _callback_count(canvas, selection) == 0

        assert manager.activate() is True
        assert manager.activate() is False
        assert _callback_count(canvas, manager) == 5
        assert _callback_count(canvas, selection) == 1

        assert window.deactivate() is True
        assert window.deactivate() is False
        assert fig.canvas is source_canvas
    finally:
        window.deactivate()
        plt.close(fig)
        QtGuiDrag.no_save_allowed = original_no_save
    assert app is not None


def test_pyl_show_reuses_each_figure_canvas_and_callback_set() -> None:
    original_show = plt.show
    original_no_save = QtGuiDrag.no_save_allowed
    _close_all_windows()
    try:
        QtGuiDrag.initialize(disable_save=True)
        fig1, _ax1 = plt.subplots()
        fig2, _ax2 = plt.subplots()
        source_canvases = (fig1.canvas, fig2.canvas)

        window = QtGuiDrag.pyl_show(hide_window=True)
        embedded_canvases = (fig1.canvas, fig2.canvas)
        managers = (fig1.figure_dragger, fig2.figure_dragger)
        selections = (fig1.selection, fig2.selection)

        for _ in range(5):
            assert QtGuiDrag.pyl_show(hide_window=True) is window
            window.setFigure(fig1)
            window.setFigure(fig2)

        assert (fig1.canvas, fig2.canvas) == embedded_canvases
        assert (fig1.figure_dragger, fig2.figure_dragger) == managers
        for canvas, manager, selection in zip(
            embedded_canvases, managers, selections
        ):
            assert _callback_count(canvas, manager) == 5
            assert _callback_count(canvas, selection) == 1

        window.deactivate()
        assert (fig1.canvas, fig2.canvas) == source_canvases
    finally:
        _close_all_windows()
        plt.show = original_show
        QtGuiDrag.no_save_allowed = original_no_save


def test_exception_swallower_install_does_not_wrap_recursively(monkeypatch) -> None:
    from pylustrator import exception_swallower

    class FakeFigure:
        pass

    class FakeAxes:
        def get_legend(self):
            return None

    class FakeAxis:
        def get_minor_ticks(self):
            return []

        def get_major_ticks(self):
            return []

    monkeypatch.setattr(exception_swallower, "Figure", FakeFigure)
    monkeypatch.setattr(exception_swallower, "_AxesBase", FakeAxes)
    monkeypatch.setattr(exception_swallower, "Axis", FakeAxis)
    monkeypatch.setattr(exception_swallower, "_exception_swallower_installed", False)
    exception_swallower.swallow_get_exceptions()
    first = (
        FakeFigure.axes,
        FakeAxes.get_legend,
        FakeAxis.get_minor_ticks,
        FakeAxis.get_major_ticks,
    )
    exception_swallower.swallow_get_exceptions()
    second = (
        FakeFigure.axes,
        FakeAxes.get_legend,
        FakeAxis.get_minor_ticks,
        FakeAxis.get_major_ticks,
    )

    assert all(before is after for before, after in zip(first, second))
