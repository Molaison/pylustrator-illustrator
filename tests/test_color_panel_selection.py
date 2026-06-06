from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from qtpy import QtCore, QtWidgets

from pylustrator.QtGui import ColorChooserWidget
from pylustrator.QtGuiDrag import PlotWindow
from pylustrator.QtShortCuts import QDragableColor
from pylustrator.change_tracker import init_figure
from pylustrator.drag_helper import DragManager
from pylustrator.components.plot_layout import Canvas as PlotCanvas


class SignalBundle(QtCore.QObject):
    figure_changed = QtCore.Signal(object)
    canvas_changed = QtCore.Signal(object)
    figure_selection_update = QtCore.Signal()
    figure_size_changed = QtCore.Signal()
    figure_element_selected = QtCore.Signal(object)
    figure_selection_property_changed = QtCore.Signal()


class FigureCanvas:
    def __init__(self, figure):
        self.figure = figure
        self.geometry_updates = 0
        self.draw_count = 0

    def updateGeometry(self):
        self.geometry_updates += 1

    def draw(self):
        self.draw_count += 1


class ChangeTracker:
    saved = True


def make_colors(count: int) -> list[str]:
    return [
        f"#{(index * 37) % 255:02x}{(index * 71) % 255:02x}{(index * 109) % 255:02x}"
        for index in range(1, count + 1)
    ]


def shown_colors(widget: ColorChooserWidget) -> list[str]:
    return [button.getColor() for button in widget.color_buttons_list]


def canvas_top_left(window: PlotWindow) -> tuple[int, int]:
    point = window.plot_layout.canvas_canvas.mapTo(
        window,
        window.plot_layout.canvas_canvas.rect().topLeft(),
    )
    return point.x(), point.y()


def assert_figure_fits_viewport(canvas: PlotCanvas) -> None:
    figure_width, figure_height = canvas.canvas.get_width_height()
    viewport = canvas.canvas_scroll.viewport().size()

    assert figure_width <= viewport.width()
    assert figure_height <= viewport.height()


def test_color_panel_refresh_replaces_old_qt_widgets() -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    signals = SignalBundle()
    fig, ax = plt.subplots()
    first = ax.plot([0, 1], [0, 1], color="#112233")[0]
    second = ax.plot([0, 1], [1, 0], color="#445566")[0]
    widget = ColorChooserWidget(None, FigureCanvas(fig), signals)

    for _ in range(3):
        signals.figure_element_selected.emit(first)
        assert shown_colors(widget) == ["#112233"]
        assert len(widget.findChildren(QDragableColor)) == 1

        signals.figure_element_selected.emit(second)
        assert shown_colors(widget) == ["#445566"]
        assert len(widget.findChildren(QDragableColor)) == 1

    plt.close(fig)
    assert app is not None


def test_color_panel_shows_all_current_colors_without_truncation() -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    signals = SignalBundle()
    colors = make_colors(25)
    fig, ax = plt.subplots()
    for index, color in enumerate(colors):
        ax.plot([0, 1], [index, index + 1], color=color)

    widget = ColorChooserWidget(None, FigureCanvas(fig), signals)
    signals.figure_element_selected.emit(fig)

    assert shown_colors(widget) == colors
    assert widget.colors_text_widget.toPlainText().splitlines() == colors

    plt.close(fig)
    assert app is not None


def test_plot_window_color_pane_is_resizable() -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = PlotWindow(1)

    assert window.colorWidget.maximumWidth() > 150
    assert window.colorWidget.sizePolicy().horizontalPolicy() != QtWidgets.QSizePolicy.Fixed

    window.close()
    assert app is not None


def test_plot_window_minimum_width_is_not_locked_by_tools_panel() -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = PlotWindow(1)
    tools = window.tools_scroll

    assert tools.minimumSizeHint().width() < 600
    assert window.input_size.minimumSizeHint().width() < 600
    assert window.minimumSizeHint().width() < 800

    window.close()
    assert app is not None


def test_plot_window_selection_does_not_resize_or_move_canvas() -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    fig, ax = plt.subplots(figsize=(5, 4), dpi=100)
    fig.change_tracker = ChangeTracker()
    line = ax.plot([0, 1], [0, 1], color="#112233", marker="o")[0]
    text = ax.text(0.5, 0.5, "label")
    window = PlotWindow(1)
    window.setFigure(fig)
    window.show()
    app.processEvents()
    window.layout_main.setSizes([266, 384, 180])
    app.processEvents()

    initial_size = window.size()
    initial_splitter_sizes = window.layout_main.sizes()
    initial_canvas_pos = canvas_top_left(window)
    for artist in (line, text, ax, fig, line):
        window.signals.figure_element_selected.emit(artist)
        app.processEvents()
        assert window.size() == initial_size
        assert window.layout_main.sizes() == initial_splitter_sizes
        assert canvas_top_left(window) == initial_canvas_pos

    window.close()
    plt.close(fig)
    assert app is not None


def test_plot_window_can_be_resized_narrower_than_initial_hint() -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = PlotWindow(1)

    window.show()
    app.processEvents()
    window.resize(700, 360)
    app.processEvents()

    assert window.size().width() <= 720

    window.close()
    assert app is not None


def test_canvas_fit_without_dpi_change_keeps_figure_size_in_scroll_area() -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    signals = SignalBundle()
    fig = plt.figure(figsize=(12, 8), dpi=100)
    canvas = PlotCanvas(signals)
    canvas.setFigure(fig)
    figure_width, figure_height = canvas.canvas.get_width_height()

    canvas.fitToView(False)

    assert canvas.canvas_canvas.minimumWidth() >= figure_width
    assert canvas.canvas_canvas.minimumHeight() >= figure_height
    assert fig.canvas.get_width_height() == (figure_width, figure_height)

    plt.close(fig)
    assert app is not None


def test_plot_canvas_fits_figure_to_visible_viewport_on_show() -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    signals = SignalBundle()
    fig = plt.figure(figsize=(12, 8), dpi=100)
    canvas = PlotCanvas(signals)
    canvas.resize(360, 260)
    canvas.setFigure(fig)

    canvas.show()
    app.processEvents()

    assert canvas.fitted_to_view is True
    assert_figure_fits_viewport(canvas)

    canvas.close()
    plt.close(fig)
    assert app is not None


def test_plot_canvas_refits_figure_when_fit_view_is_resized() -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    signals = SignalBundle()
    fig = plt.figure(figsize=(12, 8), dpi=100)
    canvas = PlotCanvas(signals)
    canvas.resize(520, 380)
    canvas.setFigure(fig)
    canvas.show()
    app.processEvents()
    first_size = fig.canvas.get_width_height()

    canvas.resize(320, 240)
    app.processEvents()
    QtCore.QThread.msleep(40)
    app.processEvents()

    assert canvas.fitted_to_view is True
    assert_figure_fits_viewport(canvas)
    assert fig.canvas.get_width_height() != first_size

    canvas.close()
    plt.close(fig)
    assert app is not None


def test_plot_canvas_scrolls_large_figure_without_resizing_window() -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    signals = SignalBundle()
    fig = plt.figure(figsize=(12, 8), dpi=100)
    canvas = PlotCanvas(signals)
    canvas.resize(360, 260)
    canvas.setFigure(fig)
    canvas.show()
    app.processEvents()

    fig.set_dpi(100)
    fig.canvas.draw()
    canvas.fitToView(False)
    app.processEvents()

    assert canvas.canvas_scroll.horizontalScrollBar().maximum() > 0
    assert canvas.canvas_scroll.verticalScrollBar().maximum() > 0
    figure_width, figure_height = fig.canvas.get_width_height()
    viewport = canvas.canvas_scroll.viewport().size()
    assert figure_width > viewport.width()
    assert figure_height > viewport.height()
    assert fig.get_dpi() == 100

    plt.close(fig)
    assert app is not None


def test_ctrl_wheel_zoom_leaves_fit_mode_and_syncs_canvas_size() -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    signals = SignalBundle()
    fig, ax = plt.subplots(figsize=(5, 4), dpi=100)
    canvas = PlotCanvas(signals)
    canvas.resize(420, 320)
    canvas.setFigure(fig)
    DragManager(fig, True)
    init_figure(fig)
    canvas.show()
    app.processEvents()
    old_dpi = fig.get_dpi()

    canvas.control_modifier = True

    class Event:
        step = 1
        x = 120
        y = 100

    canvas.scroll_event(Event())
    app.processEvents()
    figure_width, figure_height = fig.canvas.get_width_height()

    assert canvas.fitted_to_view is False
    assert fig.get_dpi() == old_dpi + 10
    assert canvas.canvas.size().width() == figure_width
    assert canvas.canvas.size().height() == figure_height
    assert canvas.canvas_container.size().width() == figure_width
    assert canvas.canvas_container.size().height() == figure_height
    assert ax is not None

    canvas.close()
    plt.close(fig)
    assert app is not None


def test_plot_window_click_selects_text_after_scrollable_layout() -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    fig, ax = plt.subplots(figsize=(5, 4), dpi=100)
    text = ax.text(0.5, 0.5, "pick me", picker=True)
    window = PlotWindow(1)
    window.setFigure(fig)
    DragManager(fig, True)
    init_figure(fig)
    window.update()
    window.show()
    app.processEvents()
    fig.canvas.draw()
    app.processEvents()

    bbox = text.get_window_extent(fig.canvas.get_renderer())
    event = matplotlib.backend_bases.MouseEvent(
        "button_press_event",
        fig.canvas,
        (bbox.x0 + bbox.x1) / 2,
        (bbox.y0 + bbox.y1) / 2,
        button=1,
    )
    fig.figure_dragger.button_press_event0(event)

    assert fig.figure_dragger.selected_element is text

    window.close()
    plt.close(fig)
    assert app is not None
