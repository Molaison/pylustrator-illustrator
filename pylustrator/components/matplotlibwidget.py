#!/usr/bin/env python
# -*- coding: utf-8 -*-
# matplotlibwidget.py

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

"""
MatplotlibWidget
================

Example of matplotlib widget for PyQt4

Copyright © 2009 Pierre Raybaut
This software is licensed under the terms of the MIT License

Derived from 'embedding_in_pyqt4.py':
Copyright © 2005 Florent Rougon, 2006 Darren Dale
"""

__version__ = "1.0.0"

import time

import qtawesome as qta
from qtpy import QtWidgets, QtCore

try:  # for matplotlib > 3.0
    from matplotlib.backends.backend_qtagg import (
        FigureCanvas,
        NavigationToolbar2QT as NavigationToolbar,
    )
except ModuleNotFoundError:
    from matplotlib.backends.backend_qt5agg import (
        FigureCanvas,
        NavigationToolbar2QT as NavigationToolbar,
    )
from matplotlib.figure import Figure


class EmbeddedFigureManager:
    """Minimal manager contract for a canvas already owned by another window.

    ``FigureManagerQT`` always creates a top-level ``MainWindow`` and reparents
    the canvas into it.  Pylustrator embeds the canvas in its own ``PlotWindow``,
    so constructing that manager leaves one hidden top-level window per reopen.
    This manager deliberately owns no Qt widgets or default callbacks.  A
    Figure's CallbackRegistry is shared by all of its canvases, so installing
    (or disconnecting) the standard handlers here could mutate the source
    canvas manager's callbacks.
    """

    def __init__(self, canvas, num: int):
        self.canvas = canvas
        self.num = num
        self.toolbar = None
        self.toolmanager = None
        self._window_title = f"Figure {num:d}"
        canvas.manager = self
        self.key_press_handler_id = None
        self.button_press_handler_id = None

    def destroy(self, *args) -> None:
        canvas = self.canvas
        if canvas is None:
            return
        for name in ("key_press_handler_id", "button_press_handler_id"):
            connection = getattr(self, name, None)
            if isinstance(connection, int):
                canvas.mpl_disconnect(connection)
            setattr(self, name, None)
        connection = getattr(self, "_cidgcf", None)
        if isinstance(connection, int):
            canvas.mpl_disconnect(connection)
        if hasattr(self, "_cidgcf"):
            del self._cidgcf
        if getattr(canvas, "manager", None) is self:
            canvas.manager = None
        self.canvas = None
        self.toolbar = None
        self.toolmanager = None

    def show(self) -> None:
        canvas = self.canvas
        if canvas is None:
            return
        window = getattr(canvas, "window_pylustrator", None) or getattr(
            canvas, "window", None
        )
        (window or canvas).show()

    def resize(self, width: int, height: int) -> None:
        if self.canvas is not None:
            self.canvas.resize(width, height)

    def get_window_title(self) -> str:
        return self._window_title

    def set_window_title(self, title: str) -> None:
        self._window_title = str(title)

    def full_screen_toggle(self) -> None:
        canvas = self.canvas
        if canvas is None:
            return
        window = canvas.window()
        if window.isFullScreen():
            window.showNormal()
        else:
            window.showFullScreen()


class MatplotlibWidget(FigureCanvas):
    quick_draw = True

    def __init__(
        self, parent=None, num=1, size=None, dpi=100, figure=None, *args, **kwargs
    ):
        if figure is None:
            self.figure = Figure(figsize=size, dpi=dpi, *args, **kwargs)
        else:
            self.figure = figure

        super().__init__(self.figure)
        self.setParent(parent)

        self.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        self.updateGeometry()

        self.manager = EmbeddedFigureManager(self, num)

        self.timer = QtCore.QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.setInterval(16)
        self.timer.timeout.connect(self.draw)

    timer = None

    def schedule_draw(self):
        if not self.timer.isActive():
            self.timer.start()

    def draw(self):
        self.timer.stop()
        # import traceback
        # print(traceback.print_stack())
        t = time.time()
        super().draw()
        duration = time.time() - t
        # if drawing is slow delay the drawing a bit to create a more smooth experience
        if duration > 0.1:
            self.quick_draw = False
            self.timer.setInterval(min(300, max(50, int(duration * 1000 * 1.5))))
        else:
            self.quick_draw = True
            self.timer.setInterval(16)

    def show(self):
        self.draw()

    def dispose(self) -> None:
        """Break every non-Qt ownership edge before deferred widget deletion."""

        if getattr(self, "_disposed", False):
            return
        self._disposed = True
        timer = self.timer
        if timer is not None:
            timer.stop()
            try:
                timer.timeout.disconnect(self.draw)
            except (TypeError, RuntimeError):
                pass
            timer.deleteLater()
            self.timer = None
        manager = getattr(self, "manager", None)
        if manager is not None:
            manager.destroy()
        self.manager = None
        self.toolbar = None
        self.window_pylustrator = None
        self.pyl_toolbar = None
        self.figure = None

    def sizeHint(self):
        w, h = self.get_width_height()
        return QtCore.QSize(w, h)

    def minimumSizeHint(self):
        return QtCore.QSize(10, 10)


def make_pickelable(cls):
    def __getstate__(self):
        return {}

    def __setstate__(self, state):
        self.__init__()

    cls.__getstate__ = __getstate__
    cls.__setstate__ = __setstate__


try:
    make_pickelable(NavigationToolbar)
    make_pickelable(MatplotlibWidget)
except AttributeError:
    pass


class CanvasWindow(QtWidgets.QWidget):
    signal = QtCore.Signal()

    def __init__(self, num="", *args, **kwargs):
        QtWidgets.QWidget.__init__(self)
        self.setWindowTitle("Figure %s" % num)
        self.setWindowIcon(qta.icon("fa5s.bar-chart"))
        self.layout = QtWidgets.QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(0)
        self.canvas = MatplotlibWidget(self, *args, **kwargs)
        self.canvas.window = self
        self.layout.addWidget(self.canvas)
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.layout.addWidget(self.toolbar)

        self.signal.connect(self.show)

    def scheduleShow(self):
        self.signal.emit()
