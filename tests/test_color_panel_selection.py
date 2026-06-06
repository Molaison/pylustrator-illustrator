from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from qtpy import QtCore, QtWidgets

from pylustrator.QtGui import ColorChooserWidget
from pylustrator.QtGuiDrag import PlotWindow
from pylustrator.QtShortCuts import QDragableColor
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


def make_colors(count: int) -> list[str]:
    return [
        f"#{(index * 37) % 255:02x}{(index * 71) % 255:02x}{(index * 109) % 255:02x}"
        for index in range(1, count + 1)
    ]


def shown_colors(widget: ColorChooserWidget) -> list[str]:
    return [button.getColor() for button in widget.color_buttons_list]


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
    tools = window.layout_main.widget(0)

    assert tools.minimumSizeHint().width() < 600
    assert window.input_size.minimumSizeHint().width() < 600
    assert window.minimumSizeHint().width() < 800

    window.close()
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


def test_canvas_fit_without_dpi_change_does_not_lock_window_to_figure_size() -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    signals = SignalBundle()
    fig = plt.figure(figsize=(12, 8), dpi=100)
    canvas = PlotCanvas(signals)
    canvas.setFigure(fig)
    figure_width, figure_height = canvas.canvas.get_width_height()

    canvas.fitToView(False)

    assert canvas.canvas_canvas.minimumWidth() < figure_width
    assert canvas.canvas_canvas.minimumHeight() < figure_height

    plt.close(fig)
    assert app is not None
