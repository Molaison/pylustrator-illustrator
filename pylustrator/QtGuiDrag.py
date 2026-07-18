#!/usr/bin/env python
# -*- coding: utf-8 -*-
# QtGuiDrag.py

# Copyright (c) 2016-2020, Richard Gerum
#
# This file is part of Pylustrator.
#
# Pylustrator is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Pylustrator is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Pylustrator. If not, see <http://www.gnu.org/licenses/>
import sys
import traceback

from matplotlib import _pylab_helpers

import os
import qtawesome as qta
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes._axes import Axes
from matplotlib.backends.qt_compat import QtCore, QtGui, QtWidgets, _version_info

if _version_info[0] == 6:
    QAction = QtGui.QAction
else:
    QAction = QtWidgets.QAction

from .ax_rasterisation import rasterizeAxes, restoreAxes
from .change_tracker import setFigureVariableNames
from .drag_helper import DragManager
from .exception_swallower import swallow_get_exceptions
from .interaction import SelectionMode

from .components.qitem_properties import QItemProperties
from .components.tree_view import MyTreeView
from .components.align import Align
from .components.plot_layout import PlotLayout
from .components.info_dialog import InfoDialog
from .components.qpos_and_size import QPosAndSize
from .change_tracker import init_figure


def my_excepthook(type, value, tback):
    sys.__excepthook__(type, value, tback)


sys.excepthook = my_excepthook

""" Matplotlib overload """
figures = {}
app = None
keys_for_lines = {}

no_save_allowed = False


def initialize(
    use_global_variable_names=False, use_exception_silencer=False, disable_save=False
):
    """
    This will overload the commands ``plt.figure()`` and ``plt.show()``.
    If a figure is created after this command was called (directly or indirectly), a GUI window will be initialized
    that allows to interactively manipulate the figure and generate code in the calling script to define these changes.
    The window will be shown when ``plt.show()`` is called.

    See also :ref:`styling`.

    Parameters
    ---------
    use_global_variable_names : bool, optional
        if used, try to find global variables that reference a figure and use them in the generated code.
    """
    global \
        app, \
        keys_for_lines, \
        old_pltshow, \
        old_pltfigure, \
        setting_use_global_variable_names, \
        no_save_allowed

    # remember line-numbers where texts are created
    def wrap_text_function(text):
        if getattr(text, "_pylustrator_text_wrapper", False):
            return text

        def wrapped_text(*args, **kwargs):
            element = text(
                *args, fontdict=kwargs["fontdict"] if "fontdict" in kwargs else None
            )
            from pylustrator.change_tracker import getReference

            stack_position = traceback.extract_stack()[-2]
            element._pylustrator_reference = dict(
                reference=getReference(element), stack_position=stack_position
            )
            old_args = {}
            properties_to_save = [
                "position",
                "text",
                "ha",
                "va",
                "fontsize",
                "color",
                "style",
                "weight",
                "fontname",
                "rotation",
            ]
            for name in properties_to_save:
                try:
                    old_args[name] = getattr(element, f"get_{name}")()
                except AttributeError:
                    continue
            old_args["position"] = None
            old_args["text"] = None
            old_values = getattr(element, "_pylustrator_old_values", [])
            old_values.append(dict(stack_position=stack_position, old_args=old_args))
            element._pylustrator_old_values = old_values

            if "fontdict" in kwargs:
                del kwargs["fontdict"]
            element.set(**kwargs)
            return element

        wrapped_text._pylustrator_text_wrapper = True
        wrapped_text._pylustrator_original_text = text
        return wrapped_text

    Axes.text = wrap_text_function(Axes.text)
    Figure.text = wrap_text_function(Figure.text)

    setattr(Figure, "_pylustrator_init", init_figure)

    # store write only attribute
    no_save_allowed = disable_save

    # warning for shell session
    stack_pos = traceback.extract_stack()[-2]
    if not stack_pos.filename.endswith(".py") and not stack_pos.filename.startswith(
        "<ipython-input-"
    ):
        print(
            "WARNING: you are using pylustartor in a shell session. Changes cannot be saved to a file. They will just be printed.",
            file=sys.stderr,
        )

    setting_use_global_variable_names = use_global_variable_names

    if use_exception_silencer:
        swallow_get_exceptions()

    if app is None:
        app = QtWidgets.QApplication(sys.argv)
    if plt.show is not show:
        old_pltshow = plt.show
    if plt.figure is not figure:
        old_pltfigure = plt.figure
    plt.show = show
    patchColormapsWithMetaInfo()

    # stack_call_position = traceback.extract_stack()[-2]
    # stack_call_position.filename

    plt.keys_for_lines = keys_for_lines

    # store the last figure save filename
    if not getattr(Figure.savefig, "_pylustrator_savefig_wrapper", False):
        sf = Figure.savefig

        def savefig(self, filename, *args, **kwargs):
            self._last_saved_figure = getattr(self, "_last_saved_figure", []) + [
                (filename, args, kwargs)
            ]
            return sf(self, filename, *args, **kwargs)

        savefig._pylustrator_savefig_wrapper = True
        savefig._pylustrator_original_savefig = sf
        Figure.savefig = savefig


def pyl_show(hide_window: bool = False):
    """the function overloads the matplotlib show function.
    It opens a DragManager window instead of the default matplotlib window.
    """
    global figures, app
    # set an application id, so that windows properly stacks them in the task bar
    if sys.platform[:3] == "win":
        import ctypes

        myappid = "rgerum.pylustrator"  # arbitrary string
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)

    if app is None:
        app = QtWidgets.QApplication(sys.argv)
    # iterate over figures
    window = PlotWindow()
    for figure_number in _pylab_helpers.Gcf.figs.copy():
        fig = _pylab_helpers.Gcf.figs[figure_number].canvas.figure

        # get variable names that point to this figure
        # if setting_use_global_variable_names:
        #    setFigureVariableNames(figure_number)
        # get the window
        # window = _pylab_helpers.Gcf.figs[figure].canvas.window_pylustrator
        # warn about ticks not fitting tick labels
        warnAboutTicks(fig)
        # add dragger
        DragManager(fig, no_save_allowed)
        init_figure(fig)
        window.setFigure(fig)
        window.addFigure(fig)
        window.update()
        # and show it
        if hide_window is False:
            window.show()
    if hide_window is False:
        # execute the application
        app.exec_()


def show(hide_window: bool = False):
    """the function overloads the matplotlib show function.
    It opens a DragManager window instead of the default matplotlib window.
    """
    global figures
    # set an application id, so that windows properly stacks them in the task bar
    if sys.platform[:3] == "win":
        import ctypes

        myappid = "rgerum.pylustrator"  # arbitrary string
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
    # iterate over figures
    for figure in _pylab_helpers.Gcf.figs.copy():
        # get variable names that point to this figure
        if setting_use_global_variable_names:
            setFigureVariableNames(figure)
        # get the window
        # window = _pylab_helpers.Gcf.figs[figure].canvas.window_pylustrator
        window = PlotWindow()
        window.setFigure(_pylab_helpers.Gcf.figs[figure].canvas.figure)
        # warn about ticks not fitting tick labels
        warnAboutTicks(window.fig)
        # add dragger
        DragManager(_pylab_helpers.Gcf.figs[figure].canvas.figure, no_save_allowed)
        init_figure(_pylab_helpers.Gcf.figs[figure].canvas.figure)
        window.update()
        # and show it
        if hide_window is False:
            window.show()
    if hide_window is False:
        # execute the application
        app.exec_()

    plt.show = old_pltshow
    plt.figure = old_pltfigure


class CmapColor(list):
    """a color like object that has the colormap as metadata"""

    def setMeta(self, value, cmap):
        self.value = value
        self.cmap = cmap


def patchColormapsWithMetaInfo():
    """all colormaps now return color with metadata from which colormap the color came from"""
    from matplotlib.colors import Colormap

    if getattr(Colormap.__call__, "_pylustrator_colormap_wrapper", False):
        return
    cm_call = Colormap.__call__

    def new_call(self, *args, **kwargs):
        c = cm_call(self, *args, **kwargs)
        if isinstance(c, (tuple, list)):
            c = CmapColor(c)
            c.setMeta(args[0], self.name)
        return c

    new_call._pylustrator_colormap_wrapper = True
    new_call._pylustrator_original_call = cm_call
    Colormap.__call__ = new_call


def figure(num=None, figsize=None, force_add=False, *args, **kwargs):
    """overloads the matplotlib figure call and wraps the Figure in a PlotWindow"""
    global figures
    # if num is not defined create a new number
    if num is None:
        num = len(_pylab_helpers.Gcf.figs) + 1
    # if number is not defined
    if force_add or num not in _pylab_helpers.Gcf.figs.keys():
        # create a new window and store it
        canvas = PlotWindow(num, figsize, *args, **kwargs).canvas
        canvas.figure.number = num
        canvas.figure.clf()
        canvas.manager.num = num
        _pylab_helpers.Gcf.figs[num] = canvas.manager
    # get the canvas of the figure
    manager = _pylab_helpers.Gcf.figs[num]
    # set the size if it is defined
    if figsize is not None:
        _pylab_helpers.Gcf.figs[num].window.setGeometry(
            100, 100, figsize[0] * 80, figsize[1] * 80
        )
    # set the figure as the active figure
    _pylab_helpers.Gcf.set_active(manager)
    # return the figure
    return manager.canvas.figure


def warnAboutTicks(fig):
    """warn if the tick labels and tick values do not match, to prevent users from accidentally setting wrong tick values"""
    import sys

    for index, ax in enumerate(fig.axes):
        ticks = ax.get_yticks()
        labels = [t.get_text() for t in ax.get_yticklabels()]
        for tick, label in zip(ticks, labels):
            label = label.replace("−", "-")
            if label == "":
                continue
            try:
                label = float(label)
            except ValueError:
                pass
            # if the label is still a string or too far away from the tick value
            if isinstance(label, str) or abs(tick - label) > abs(1e-3 * tick):
                ax_name = ax.get_label()
                if ax_name == "":
                    ax_name = "#%d" % index
                else:
                    ax_name = '"' + ax_name + '"'
                print(
                    "Warning tick and label differ",
                    tick,
                    label,
                    "for axes",
                    ax_name,
                    file=sys.stderr,
                )


""" Window """


class Signals(QtWidgets.QWidget):
    figure_changed = QtCore.Signal(Figure)
    canvas_changed = QtCore.Signal(object)
    figure_size_changed = QtCore.Signal()
    figure_element_selected = QtCore.Signal(object)
    figure_selection_moved = QtCore.Signal()
    figure_selection_property_changed = QtCore.Signal()
    figure_selection_update = QtCore.Signal()
    figure_element_child_created = QtCore.Signal(object)


class PlotWindow(QtWidgets.QWidget):
    fig = None
    update_changes_signal = QtCore.Signal(bool, bool, str, str)

    def setFigure(self, figure):
        if self.fig is not None:
            self.fig.window = None
            self.fig.signals = None
        figure.no_figure_dragger_selection_update = False
        self.fig = figure
        self.fig.window = self
        self.fig.signals = self.signals
        if getattr(figure, "_pylustrator_initial_dpi", None) is None:
            figure._pylustrator_initial_dpi = figure.get_dpi()
        selection = getattr(figure, "selection", None)
        if selection is not None:
            selection.defer_artist_updates = self.fast_drag_preview
        dragger = getattr(figure, "figure_dragger", None)
        if dragger is not None:
            dragger.marquee_select_containers_only = (
                self.marquee_select_containers_only
            )
            dragger.set_selection_mode(self.selection_mode)
        self.updateSelectionControls()
        self.signals.figure_changed.emit(figure)

    def setCanvas(self, canvas):
        self.canvas = canvas
        self.canvas.window_pylustrator = self

    def addFigure(self, figure):
        self.figures.append(figure)

        undo_act = QAction(f"Figure {figure.number}", self)

        def undo():
            self.setFigure(figure)

        undo_act.triggered.connect(undo)
        self.menu_edit.addAction(undo_act)

        # self.preview.addFigure(figure)

    def selectionProperyChanged(self):
        self.fig.selection.update_selection_rectangles()
        self.fig.selection.update_extent()

    def create_menu(self, layout_parent):
        self.menuBar = QtWidgets.QMenuBar()
        file_menu = self.menuBar.addMenu("&File")

        if no_save_allowed is False:
            open_act = QAction("&Save", self)
            open_act.setShortcut("Ctrl+S")
            open_act.triggered.connect(self.actionSave)
            file_menu.addAction(open_act)

        open_act = QAction("Save &Image...", self)
        open_act.setShortcut("Ctrl+I")
        open_act.triggered.connect(self.actionSaveImage)
        file_menu.addAction(open_act)

        open_act = QAction("Exit", self)
        open_act.triggered.connect(self.close)
        open_act.setShortcut("Ctrl+Q")
        file_menu.addAction(open_act)

        file_menu = self.menuBar.addMenu("&Edit")
        self.menu_edit = file_menu

        info_act = QAction("&Info", self)
        info_act.triggered.connect(self.showInfo)

        self.undo_act = QAction("Undo", self)
        self.undo_act.triggered.connect(self.undo)
        self.undo_act.setShortcut("Ctrl+Z")
        file_menu.addAction(self.undo_act)

        self.redo_act = QAction("Redo", self)
        self.redo_act.triggered.connect(self.redo)
        self.redo_act.setShortcuts(["Ctrl+Y", "Ctrl+Shift+Z"])
        file_menu.addAction(self.redo_act)

        delete_act = QAction("Delete", self)
        delete_act.triggered.connect(self.delete_selection)
        delete_act.setShortcuts(["Delete", "Backspace"])
        file_menu.addAction(delete_act)

        def interaction_action(label, shortcut, callback):
            action = QAction(label, self)
            action.setShortcut(shortcut)

            def execute():
                if self.fig is None:
                    return
                try:
                    callback(self.fig.figure_dragger)
                except ValueError as exc:
                    QtWidgets.QMessageBox.warning(self, "Pylustrator", str(exc))

            action.triggered.connect(execute)
            file_menu.addAction(action)
            return action

        interaction_action("Group", "Ctrl+G", lambda dragger: dragger.group_selection())
        interaction_action(
            "Ungroup", "Ctrl+Shift+G", lambda dragger: dragger.ungroup_selection()
        )
        interaction_action(
            "Lock Selection", "Ctrl+2", lambda dragger: dragger.set_selection_locked(True)
        )
        interaction_action("Unlock All", "Ctrl+Alt+2", lambda dragger: dragger.unlock_all())
        interaction_action(
            "Hide Selection", "Ctrl+3", lambda dragger: dragger.set_selection_visible(False)
        )
        interaction_action("Show All", "Ctrl+Alt+3", lambda dragger: dragger.show_all())
        interaction_action(
            "Bring Forward",
            "Ctrl+]",
            lambda dragger: dragger.change_selection_zorder("forward"),
        )
        interaction_action(
            "Send Backward",
            "Ctrl+[",
            lambda dragger: dragger.change_selection_zorder("backward"),
        )
        interaction_action(
            "Bring to Front",
            "Ctrl+Shift+]",
            lambda dragger: dragger.change_selection_zorder("front"),
        )
        interaction_action(
            "Send to Back",
            "Ctrl+Shift+[",
            lambda dragger: dragger.change_selection_zorder("back"),
        )

        self.menuBar.addAction(info_act)

        layout_parent.addWidget(self.menuBar)

    def undo(self):
        self.fig.figure_dragger.undo()

    def redo(self):
        self.fig.figure_dragger.redo()

    def delete_selection(self):
        self.fig.selection.delete_targets()

    def __init__(self, number: int = 0):
        """The main window of pylustrator

        Args:
            number: the id of the figure
            size: the size of the figure
        """
        super().__init__()

        self.figures = []
        self._initial_layout_applied = False
        self.fast_drag_preview = True
        self.marquee_select_containers_only = False
        self.selection_mode = SelectionMode.OBJECT

        self.signals = Signals()
        self.signals.canvas_changed.connect(self.setCanvas)
        self.signals.figure_selection_property_changed.connect(
            self.selectionProperyChanged
        )

        self.plot_layout = PlotLayout(self.signals)

        # widget layout and elements
        self.setWindowTitle("Figure %s - Pylustrator" % number)
        self.setWindowIcon(
            QtGui.QIcon(os.path.join(os.path.dirname(__file__), "icons", "logo.ico"))
        )
        layout_parent = QtWidgets.QVBoxLayout(self)
        layout_parent.setContentsMargins(0, 0, 0, 0)

        # add the menu
        self.create_menu(layout_parent)

        layout_top_bar = QtWidgets.QHBoxLayout()
        layout_parent.addLayout(layout_top_bar)
        layout_top_bar.setContentsMargins(10, 0, 10, 0)

        button_undo = QtWidgets.QPushButton(qta.icon("mdi.undo"), "")
        button_undo.setToolTip("undo")
        button_undo.clicked.connect(self.undo)
        layout_top_bar.addWidget(button_undo)

        button_redo = QtWidgets.QPushButton(qta.icon("mdi.redo"), "")
        button_redo.setToolTip("redo")
        button_redo.clicked.connect(self.redo)
        layout_top_bar.addWidget(button_redo)

        def updateChangesSignal(undo, redo, undo_text, redo_text):
            button_undo.setDisabled(undo)
            self.undo_act.setDisabled(undo)
            if undo_text != "":
                self.undo_act.setText(f"Undo: {undo_text}")
                button_undo.setToolTip(f"Undo: {undo_text}")
            else:
                self.undo_act.setText("Undo")
                button_undo.setToolTip("Undo")
            button_redo.setDisabled(redo)
            self.redo_act.setDisabled(redo)
            if redo_text != "":
                self.redo_act.setText(f"Redo: {redo_text}")
                button_redo.setToolTip(f"Redo: {redo_text}")
            else:
                self.redo_act.setText("Redo")
                button_redo.setToolTip("Redo")

        self.update_changes_signal.connect(updateChangesSignal)

        selection_group = QtWidgets.QButtonGroup(self)
        selection_group.setExclusive(True)
        self.button_object_selection = QtWidgets.QPushButton("V")
        self.button_object_selection.setCheckable(True)
        self.button_object_selection.setChecked(True)
        self.button_object_selection.setFixedWidth(22)
        self.button_object_selection.setToolTip("Object Selection (V)")
        self.button_object_selection.setShortcut("V")
        self.button_object_selection.clicked.connect(
            lambda checked: checked and self.setSelectionMode(SelectionMode.OBJECT)
        )
        selection_group.addButton(self.button_object_selection)

        self.button_direct_selection = QtWidgets.QPushButton("A")
        self.button_direct_selection.setCheckable(True)
        self.button_direct_selection.setFixedWidth(22)
        self.button_direct_selection.setToolTip("Direct Selection (A)")
        self.button_direct_selection.setShortcut("A")
        self.button_direct_selection.clicked.connect(
            lambda checked: checked and self.setSelectionMode(SelectionMode.DIRECT)
        )
        selection_group.addButton(self.button_direct_selection)

        self.selection_scope_label = QtWidgets.QLabel("")
        self.selection_scope_label.setToolTip("Isolation scope")
        self.selection_scope_label.setMinimumWidth(0)
        self.selection_scope_label.setSizePolicy(
            QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Preferred
        )

        self.button_fast_drag = QtWidgets.QPushButton(qta.icon("fa5s.bolt"), "")
        self.button_fast_drag.setCheckable(True)
        self.button_fast_drag.setChecked(self.fast_drag_preview)
        self.button_fast_drag.setToolTip("Fast drag preview")
        self.button_fast_drag.clicked.connect(self.setFastDragPreview)
        layout_top_bar.addWidget(self.button_fast_drag)

        self.button_marquee_containers = QtWidgets.QPushButton(
            qta.icon("fa5s.layer-group"), ""
        )
        self.button_marquee_containers.setCheckable(True)
        self.button_marquee_containers.setChecked(
            self.marquee_select_containers_only
        )
        self.button_marquee_containers.setToolTip("Box select containers only")
        self.button_marquee_containers.clicked.connect(
            self.setMarqueeSelectContainersOnly
        )
        layout_top_bar.addWidget(self.button_marquee_containers)

        self.input_size = QPosAndSize(layout_top_bar, self.signals)

        if 0:
            self.layout_main = QtWidgets.QHBoxLayout()
            self.layout_main.setContentsMargins(0, 0, 0, 0)
            layout_parent.addLayout(self.layout_main)
        else:
            self.layout_main = QtWidgets.QSplitter()
            self.layout_main.setSizePolicy(
                QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding
            )
            layout_parent.addWidget(self.layout_main)

        # self.preview = FigurePreviews(self)
        # self.layout_main.addWidget(self.preview)
        #
        widget = QtWidgets.QWidget()
        self.layout_tools = QtWidgets.QVBoxLayout(widget)
        selection_tools = QtWidgets.QHBoxLayout()
        selection_tools.setContentsMargins(0, 0, 0, 0)
        selection_tools.addWidget(self.button_object_selection)
        selection_tools.addWidget(self.button_direct_selection)
        selection_tools.addWidget(self.selection_scope_label, 1)
        self.layout_tools.addLayout(selection_tools)
        widget.setSizePolicy(
            QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed
        )
        tools_scroll = QtWidgets.QScrollArea()
        tools_scroll.setWidgetResizable(True)
        tools_scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        tools_scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        tools_scroll.setSizePolicy(
            QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Expanding
        )
        tools_scroll.setWidget(widget)
        self.layout_main.addWidget(tools_scroll)
        self.tools_scroll = tools_scroll

        if 0:
            layout_rasterize_buttons = QtWidgets.QHBoxLayout()
            self.layout_tools.addLayout(layout_rasterize_buttons)
            self.button_rasterize = QtWidgets.QPushButton("rasterize")
            layout_rasterize_buttons.addWidget(self.button_rasterize)
            self.button_rasterize.clicked.connect(lambda x: self.rasterize(True))
            self.button_derasterize = QtWidgets.QPushButton("derasterize")
            layout_rasterize_buttons.addWidget(self.button_derasterize)
            self.button_derasterize.clicked.connect(lambda x: self.rasterize(False))
            self.button_derasterize.setDisabled(True)
        elif 0:
            self.button_rasterize = QAction("rasterize", self)
            self.button_rasterize.triggered.connect(lambda x: self.rasterize(True))
            self.menu_edit.addAction(self.button_rasterize)

            self.button_derasterize = QAction("derasterize", self)
            self.button_derasterize.triggered.connect(lambda x: self.rasterize(False))
            self.menu_edit.addAction(self.button_derasterize)
            self.button_derasterize.setDisabled(True)

        self.treeView = MyTreeView(self.signals, self.layout_tools)

        self.input_properties = QItemProperties(self.layout_tools, self.signals)
        self.input_align = Align(self.layout_tools, self.signals)

        # add plot layout
        self.layout_main.addWidget(self.plot_layout)

        from .QtGui import ColorChooserWidget

        self.colorWidget = ColorChooserWidget(self, None, self.signals)
        self.colorWidget.setSizePolicy(
            QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Expanding
        )
        self.color_scroll = QtWidgets.QScrollArea()
        self.color_scroll.setWidgetResizable(True)
        self.color_scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.color_scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.color_scroll.setWidget(self.colorWidget)
        self.layout_main.addWidget(self.color_scroll)

        self.layout_main.setStretchFactor(0, 0)
        self.layout_main.setStretchFactor(1, 1)
        self.layout_main.setStretchFactor(2, 0)

    def _available_geometry(self) -> QtCore.QRect:
        screen = self.screen() or QtWidgets.QApplication.primaryScreen()
        if screen is None:
            return QtCore.QRect(0, 0, 1440, 900)
        return screen.availableGeometry()

    def _figure_design_pixels(self) -> tuple[float, float]:
        if self.fig is None:
            return 800.0, 600.0
        width, height = self.fig.get_size_inches()
        dpi = getattr(self.fig, "_pylustrator_initial_dpi", self.fig.get_dpi())
        return float(width * dpi), float(height * dpi)

    def _initial_side_widths(self, max_width: int) -> tuple[int, int]:
        tools_width = 280
        color_width = 220
        min_canvas_width = 420
        if max_width < tools_width + color_width + min_canvas_width + 40:
            tools_width = 220
            color_width = 180
        if max_width < tools_width + color_width + min_canvas_width + 40:
            tools_width = max(160, int(max_width * 0.25))
            color_width = max(140, int(max_width * 0.20))
        return tools_width, color_width

    def _apply_initial_layout(self):
        if self.fig is None:
            return
        available = self._available_geometry()
        max_width = max(700, int(available.width() * 0.92))
        max_height = max(520, int(available.height() * 0.88))
        tools_width, color_width = self._initial_side_widths(max_width)

        top_chrome = self.height() - self.layout_main.height()
        if top_chrome <= 0:
            top_chrome = (
                self.menuBar.sizeHint().height()
                + self.input_size.sizeHint().height()
                + 12
            )
        plot_chrome = (
            self.plot_layout.height() - self.plot_layout.canvas_canvas.height()
        )
        if plot_chrome <= 0:
            plot_chrome = 72

        figure_width, figure_height = self._figure_design_pixels()
        available_canvas_width = max(max_width - tools_width - color_width - 40, 300)
        available_canvas_height = max(max_height - top_chrome - plot_chrome, 260)

        canvas_width = int(
            min(max(figure_width, min(720, available_canvas_width)), available_canvas_width)
        )
        canvas_height = int(
            min(max(figure_height, min(520, available_canvas_height)), available_canvas_height)
        )
        window_width = min(max_width, tools_width + canvas_width + color_width + 40)
        window_height = min(max_height, canvas_height + top_chrome + plot_chrome)

        x = available.x() + max((available.width() - window_width) // 2, 0)
        y = available.y() + max((available.height() - window_height) // 2, 0)
        self.setGeometry(x, y, int(window_width), int(window_height))
        self.layout_main.setSizes(
            [
                int(tools_width),
                max(int(window_width - tools_width - color_width), 300),
                int(color_width),
            ]
        )
        self.plot_layout.canvas_canvas.fitToView(True)

    def setFastDragPreview(self, enabled: bool):
        self.fast_drag_preview = bool(enabled)
        if self.fig is not None and getattr(self.fig, "selection", None) is not None:
            self.fig.selection.defer_artist_updates = self.fast_drag_preview

    def setSelectionMode(self, mode: SelectionMode | str):
        self.selection_mode = SelectionMode.coerce(mode)
        if (
            self.fig is not None
            and getattr(self.fig, "figure_dragger", None) is not None
        ):
            self.fig.figure_dragger.set_selection_mode(self.selection_mode)
        self.updateSelectionControls()

    def updateSelectionControls(self):
        mode = self.selection_mode
        breadcrumbs = ()
        if (
            self.fig is not None
            and getattr(self.fig, "figure_dragger", None) is not None
        ):
            mode = self.fig.figure_dragger.selection_mode
            breadcrumbs = self.fig.figure_dragger.isolation_breadcrumbs
        self.selection_mode = mode
        for button, checked in (
            (self.button_object_selection, mode is SelectionMode.OBJECT),
            (self.button_direct_selection, mode is SelectionMode.DIRECT),
        ):
            old = button.blockSignals(True)
            button.setChecked(checked)
            button.blockSignals(old)
        self.selection_scope_label.setText(
            " / ".join(breadcrumbs) if breadcrumbs else ""
        )

    def setMarqueeSelectContainersOnly(self, enabled: bool):
        self.marquee_select_containers_only = bool(enabled)
        if (
            self.fig is not None
            and getattr(self.fig, "figure_dragger", None) is not None
        ):
            self.fig.figure_dragger.marquee_select_containers_only = (
                self.marquee_select_containers_only
            )

    def rasterize(self, rasterize: bool):
        """convert the figur elements to an image"""
        if len(self.fig.selection.targets):
            self.fig.figure_dragger.select_element(None)
        if rasterize:
            rasterizeAxes(self.fig)
            self.button_derasterize.setDisabled(False)
        else:
            restoreAxes(self.fig)
            self.button_derasterize.setDisabled(True)
        self.fig.canvas.draw()

    def actionSave(self):
        """Save the editable source document.

        Image export is an explicit, separate action.  Replaying previous
        ``savefig`` calls here both blocked the UI on rendering and caused the
        tracking wrapper to append the replayed requests again, doubling the
        work after every Ctrl+S.
        """
        self.fig.change_tracker.save()

    def actionSaveImage(self):
        """save figure as an image"""
        path = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save Image",
            str(getattr(self.fig, "_last_saved_figure", [(None,)])[0][0]),
            "Images (*.png *.jpg *.pdf)",
        )
        if isinstance(path, tuple):
            path = str(path[0])
        else:
            path = str(path)
        if not path:
            return
        if os.path.splitext(path)[1] == ".pdf":
            self.fig.savefig(path, dpi=300)
        else:
            self.fig.savefig(path)
        print("Saved plot image as", path)

    def showInfo(self):
        """show the info dialog"""
        self.info_dialog = InfoDialog(self)
        self.info_dialog.show()

    def showEvent(self, event: QtCore.QEvent):
        """when the window is shown"""
        super().showEvent(event)
        self.colorWidget.updateColorsText()
        if not self._initial_layout_applied:
            self._initial_layout_applied = True
            self._apply_initial_layout()
            QtCore.QTimer.singleShot(
                0, lambda: self.plot_layout.canvas_canvas.fitToView(True)
            )

    def update(self):
        """update the tree view"""

        # self.input_size.setValue(np.array(self.fig.get_size_inches()) * 2.54)

        def wrap(func):
            def newfunc(element, event=None):
                self.fig.no_figure_dragger_selection_update = True
                self.signals.figure_element_selected.emit(element)
                ret = func(element, event)
                self.fig.no_figure_dragger_selection_update = False
                return ret

            return newfunc

        self.fig.figure_dragger.on_select = wrap(self.fig.figure_dragger.on_select)
        self.fig.change_tracker.update_changes_signal = self.update_changes_signal
        self.update_changes_signal.emit(True, True, "", "")

        def wrap(func):
            def newfunc(*args):
                self.updateTitle()
                return func(*args)

            return newfunc

        self.fig.change_tracker.addChange = wrap(self.fig.change_tracker.addChange)
        self.fig.change_tracker.save = wrap(self.fig.change_tracker.save)
        self.signals.figure_element_selected.emit(self.fig)

    def updateTitle(self):
        """update the title of the window to display if it is saved or not"""
        if self.fig.change_tracker.saved:
            self.setWindowTitle("Figure %s - Pylustrator" % self.fig.number)
        else:
            self.setWindowTitle("Figure %s* - Pylustrator" % self.fig.number)

    def closeEvent(self, event: QtCore.QEvent):
        """when the window is closed, ask the user to save"""
        if self.fig is None:
            return
        if not self.fig.change_tracker.saved and not no_save_allowed:
            reply = QtWidgets.QMessageBox.question(
                self,
                "Warning - Pylustrator",
                "The figure has not been saved. "
                "All data will be lost.\nDo you want to save it?",
                QtWidgets.QMessageBox.Cancel
                | QtWidgets.QMessageBox.No
                | QtWidgets.QMessageBox.Yes,
                QtWidgets.QMessageBox.Yes,
            )

            if reply == QtWidgets.QMessageBox.Cancel:
                event.ignore()
            if reply == QtWidgets.QMessageBox.Yes:
                self.fig.change_tracker.save()
                # app.clipboard().setText("\r\n".join(output))
