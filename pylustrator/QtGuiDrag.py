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
old_pltshow = plt.show
old_pltfigure = plt.figure
setting_use_global_variable_names = False

no_save_allowed = False


def _ensure_application():
    """Return the live Qt application instead of trusting a stale global."""

    global app
    current = QtWidgets.QApplication.instance()
    if current is None:
        current = QtWidgets.QApplication(sys.argv)
    app = current
    return current


def _restore_matplotlib_entry_points() -> None:
    """Undo only pylustrator's own temporary pyplot replacements."""

    if plt.show is show:
        plt.show = old_pltshow
    if plt.figure is figure:
        plt.figure = old_pltfigure


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

    _ensure_application()
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


def _managed_figures():
    return [
        manager.canvas.figure
        for manager in _pylab_helpers.Gcf.figs.copy().values()
    ]


def _live_window(figure):
    window = getattr(figure, "window", None)
    manager = getattr(figure, "figure_dragger", None)
    structure_matches = getattr(manager, "figure_structure_matches", None)
    if (
        isinstance(window, PlotWindow)
        and not getattr(window, "_deactivated", False)
        and window.owns_figure(figure)
        and manager is not None
        and callable(structure_matches)
        and structure_matches()
    ):
        return window
    return None


def _deactivate_stale_figure_windows(managed_figures) -> None:
    """Tear down shared windows before rebuilding any attached Figure.

    A window owns one or more Figure sessions.  Rebuilding one stale Figure
    while iterating that window would deactivate sessions already prepared
    earlier in the same pass, so stale ownership must be resolved up front.
    """

    stale_windows = {
        window
        for figure in managed_figures
        if isinstance((window := getattr(figure, "window", None)), PlotWindow)
        and not getattr(window, "_deactivated", False)
        and _live_window(figure) is None
    }
    for window in stale_windows:
        window.deactivate()


def _prepare_figure(window, figure, *, source_stack_position=None):
    """Attach one Figure to one window exactly once."""

    current_window = getattr(figure, "window", None)
    if not isinstance(current_window, PlotWindow) or getattr(
        current_window, "_deactivated", False
    ):
        current_window = None
    if current_window is not None and current_window is not window:
        current_window.deactivate()

    owns_figure = getattr(window, "owns_figure", lambda _figure: False)
    manager = getattr(figure, "figure_dragger", None)
    if owns_figure(figure) and manager is not None:
        window.setFigure(figure)
        window.update()
        return manager

    window.setFigure(figure)
    init_figure(figure)
    warnAboutTicks(figure)
    manager = DragManager(
        figure,
        no_save_allowed,
        source_stack_position=source_stack_position,
    )
    configure = getattr(window, "configure_figure_manager", None)
    if configure is not None:
        configure(figure)
    window.update()
    return manager


def _set_windows_app_id() -> None:
    if sys.platform[:3] != "win":
        return
    import ctypes

    myappid = "rgerum.pylustrator"  # arbitrary string
    ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)


def pyl_show(hide_window: bool = False):
    """Open all pyplot figures in one reusable Pylustrator window."""

    source_stack_position = traceback.extract_stack()[-2]
    _set_windows_app_id()
    application = _ensure_application()
    managed_figures = _managed_figures()
    if not managed_figures:
        return None

    _deactivate_stale_figure_windows(managed_figures)

    existing_windows = {
        window
        for figure in managed_figures
        if (window := _live_window(figure)) is not None
    }
    window = (
        next(iter(existing_windows))
        if len(existing_windows) == 1
        else PlotWindow()
    )
    for figure in managed_figures:
        _prepare_figure(
            window,
            figure,
            source_stack_position=source_stack_position,
        )
        window.addFigure(figure)
    if not hide_window:
        window.show()
        application.exec_()
    return window


def show(hide_window: bool = False):
    """the function overloads the matplotlib show function.
    It opens a DragManager window instead of the default matplotlib window.
    """
    source_stack_position = traceback.extract_stack()[-2]
    _set_windows_app_id()
    application = _ensure_application()
    windows = []
    try:
        managed_entries = tuple(_pylab_helpers.Gcf.figs.copy().items())
        _deactivate_stale_figure_windows(
            [manager.canvas.figure for _number, manager in managed_entries]
        )
        for figure_number, manager in managed_entries:
            if setting_use_global_variable_names:
                setFigureVariableNames(figure_number)
            figure = manager.canvas.figure
            window = _live_window(figure) or PlotWindow()
            _prepare_figure(
                window,
                figure,
                source_stack_position=source_stack_position,
            )
            if window not in windows:
                windows.append(window)
            if not hide_window:
                window.show()
        if windows and not hide_window:
            application.exec_()
        return windows
    finally:
        _restore_matplotlib_entry_points()


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


class Signals(QtCore.QObject):
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

    def owns_figure(self, figure) -> bool:
        return self.plot_layout.canvas_canvas.has_figure(figure)

    def configure_figure_manager(self, figure) -> None:
        figure.no_figure_dragger_selection_update = False
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

    def setFigure(self, figure):
        if self._deactivated:
            raise RuntimeError("Cannot attach a Figure to a closed PlotWindow")
        already_current = self.fig is figure and self.owns_figure(figure)
        if not already_current:
            self._unbind_current_tracker()
        self.fig = figure
        if figure not in self._owned_figures:
            self._owned_figures.append(figure)
        figure.window = self
        figure.signals = self.signals
        if already_current:
            self.configure_figure_manager(figure)
            self._bind_current_tracker()
            self.updateSelectionControls()
            return
        self.signals.figure_changed.emit(figure)
        self.configure_figure_manager(figure)
        self._bind_current_tracker()
        self.updateSelectionControls()

    def setCanvas(self, canvas):
        self.canvas = canvas
        self.canvas.window_pylustrator = self

    def addFigure(self, figure):
        if figure in self.figures:
            return
        self.figures.append(figure)

        undo_act = QAction(f"Figure {figure.number}", self)

        def undo():
            self.setFigure(figure)
            self.update()

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
        self._owned_figures = []
        self._bound_tracker = None
        self._deactivated = False
        self._initial_layout_applied = False
        self.fast_drag_preview = True
        self.marquee_select_containers_only = False
        self.selection_mode = SelectionMode.OBJECT
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose, True)

        self.signals = Signals(self)
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
            self.updateTitle()
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
        self._bind_current_tracker()
        self.signals.figure_element_selected.emit(self.fig)

    def _unbind_current_tracker(self) -> None:
        tracker = self._bound_tracker
        if tracker is not None:
            tracker.update_changes_signal = None
        self._bound_tracker = None

    def _bind_current_tracker(self) -> bool:
        if self.fig is None:
            return False
        tracker = getattr(self.fig, "change_tracker", None)
        if tracker is None or tracker is self._bound_tracker:
            return False
        self._unbind_current_tracker()
        tracker.update_changes_signal = self.update_changes_signal
        self._bound_tracker = tracker
        refresh = getattr(tracker, "changeCountChanged", None)
        if callable(refresh):
            refresh()
        return True

    def updateTitle(self):
        """update the title of the window to display if it is saved or not"""
        if self.fig is None:
            return
        if self.fig.change_tracker.saved:
            self.setWindowTitle("Figure %s - Pylustrator" % self.fig.number)
        else:
            self.setWindowTitle("Figure %s* - Pylustrator" % self.fig.number)

    def closeEvent(self, event: QtCore.QEvent):
        """when the window is closed, ask the user to save"""
        if self._deactivated:
            event.accept()
            return
        dirty_figures = []
        for figure in self._owned_figures:
            tracker = getattr(figure, "change_tracker", None)
            if (
                tracker is not None
                and not getattr(tracker, "saved", True)
                and not getattr(tracker, "no_save", no_save_allowed)
            ):
                dirty_figures.append(figure)
        for figure in dirty_figures:
            reply = QtWidgets.QMessageBox.question(
                self,
                "Warning - Pylustrator",
                f"Figure {figure.number} has not been saved. "
                "All data will be lost.\nDo you want to save it?",
                QtWidgets.QMessageBox.Cancel
                | QtWidgets.QMessageBox.No
                | QtWidgets.QMessageBox.Yes,
                QtWidgets.QMessageBox.Yes,
            )

            if reply == QtWidgets.QMessageBox.Cancel:
                event.ignore()
                return
            if reply == QtWidgets.QMessageBox.Yes:
                figure.change_tracker.save()
        self.deactivate()
        event.accept()

    def deactivate(self) -> bool:
        """Release every Figure, callback, and Qt canvas owned by this window."""

        if self._deactivated:
            return False
        self._deactivated = True
        self.hide()

        self._unbind_current_tracker()

        owned_figures = tuple(self._owned_figures)
        for figure in owned_figures:
            if getattr(figure, "window", None) is self:
                figure.window = None
            if getattr(figure, "signals", None) is self.signals:
                figure.signals = None
            manager = getattr(figure, "figure_dragger", None)
            if manager is not None:
                dispose = getattr(manager, "dispose", None)
                if callable(dispose):
                    dispose(redraw=False)
                else:
                    manager.deactivate(redraw=False)
                if getattr(figure, "figure_dragger", None) is manager:
                    figure.figure_dragger = None
                if getattr(figure, "selection", None) is getattr(
                    manager, "selection", None
                ):
                    figure.selection = None
            figure._pyl_graphics_scene_snapparent = None

        self.plot_layout.release_figures()
        self.canvas = None
        self.fig = None
        self.figures.clear()
        self._owned_figures.clear()
        return True
