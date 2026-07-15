from typing import Optional
from qtpy import QtWidgets
import qtawesome as qta

import matplotlib as mpl
import matplotlib.transforms as transforms
import numpy as np
from matplotlib.figure import Figure
from matplotlib.artist import Artist

try:  # starting from mpl version 3.6.0
    from matplotlib.axes import Axes
except ImportError:
    from matplotlib.axes._subplots import Axes
from pylustrator.helper_functions import changeFigureSize, main_figure
from pylustrator.QLinkableWidgets import DimensionsWidget, ComboWidget
from pylustrator.snap import TargetWrapper


class QPosAndSize(QtWidgets.QWidget):
    element = None
    transform = None
    transform_index = 0
    scale_type = 0

    def __init__(self, layout: QtWidgets.QLayout, signals):
        """a widget that holds all the properties to set and the tree view

        Args:
            layout: the layout to which to add the widget
            fig: the figure
        """
        QtWidgets.QWidget.__init__(self)

        signals.figure_changed.connect(self.setFigure)
        signals.figure_element_selected.connect(self.select_element)
        signals.figure_selection_moved.connect(self.selection_moved)
        self.signals = signals

        layout.addWidget(self)
        self.layout = QtWidgets.QGridLayout(self)
        self.layout.setContentsMargins(4, 0, 4, 0)

        self.input_position = DimensionsWidget(self.layout, "X:", "Y:", "cm")
        self.layout.addWidget(self.input_position, 0, 0)
        self.input_position.valueChangedX.connect(lambda x: self.changePos(x, None))
        self.input_position.valueChangedY.connect(lambda y: self.changePos(None, y))

        self.input_shape = DimensionsWidget(self.layout, "W:", "H:", "cm")
        self.layout.addWidget(self.input_shape, 0, 1)
        self.input_shape.valueChangedX.connect(
            lambda _x: self.changeSize(self.input_shape.value(), changed_axis=0)
        )
        self.input_shape.valueChangedY.connect(
            lambda _y: self.changeSize(self.input_shape.value(), changed_axis=1)
        )

        self.input_transform = ComboWidget(self.layout, "", ["cm", "in", "px", "none"])
        self.layout.addWidget(self.input_transform, 1, 0)
        self.input_transform.editingFinished.connect(self.changeTransform)

        self.input_shape_transform = ComboWidget(
            self.layout, "", ["scale", "bottom right", "top left"]
        )
        self.layout.addWidget(self.input_shape_transform, 1, 1)
        self.input_shape_transform.editingFinished.connect(self.changeTransform2)

        self.input_lock_aspect = QtWidgets.QPushButton(qta.icon("fa5s.lock"), "")
        self.input_lock_aspect.setCheckable(True)
        self.input_lock_aspect.setToolTip("Lock aspect ratio")
        self.input_lock_aspect.setFixedWidth(28)
        self.layout.addWidget(self.input_lock_aspect, 1, 2)
        self.input_lock_aspect.toggled.connect(self.changeLockAspect)

        self.layout.setColumnStretch(3, 1)

    def setFigure(self, figure):
        self.fig = figure
        selection = getattr(figure, "selection", None)
        if selection is not None:
            self.input_lock_aspect.setChecked(
                bool(getattr(selection, "lock_aspect_ratio", False))
            )

    def select_element(self, element):
        """select an element"""
        if element is None:
            self.setElement(self.fig)
        else:
            self.setElement(element)

    def selection_moved(self):
        self.setElement(self.element)

    def changeTransform(self):
        """change the transform and the units of the position and size widgets"""
        name = self.input_transform.text()
        self.transform_index = ["cm", "in", "px", "none"].index(name)  # transform_index
        if name == "none":
            name = ""
        self.input_shape.setUnit(name)
        self.input_position.setUnit(name)
        self.setElement(self.element)

    def changeTransform2(self):  # , state: int, name: str):
        """when the dimension change type is changed from 'scale' to 'bottom right' or 'bottom left'"""
        name = self.input_shape_transform.text()
        self.scale_type = ["scale", "bottom right", "top left"].index(name)
        # self.scale_type = state

    def changeLockAspect(self, state: bool):
        selection = getattr(getattr(self, "fig", None), "selection", None)
        if selection is not None:
            selection.lock_aspect_ratio = bool(state)

    def _lock_aspect_enabled(self) -> bool:
        selection = getattr(getattr(self, "fig", None), "selection", None)
        return bool(
            self.input_lock_aspect.isChecked()
            or getattr(selection, "lock_aspect_ratio", False)
        )

    @staticmethod
    def _locked_size(
        value: list | tuple,
        current_size: tuple[float, float],
        changed_axis: int | None,
    ) -> list[float]:
        new_size = [float(value[0]), float(value[1])]
        current_width, current_height = current_size
        if changed_axis is None or current_width == 0 or current_height == 0:
            return new_size
        if changed_axis == 0:
            new_size[1] = new_size[0] * current_height / current_width
        else:
            new_size[0] = new_size[1] * current_width / current_height
        return new_size

    @staticmethod
    def _axes_position(element: Axes) -> list[float]:
        pos = element.get_position()
        return [pos.x0, pos.y0, pos.width, pos.height]

    def _record_figure_state(self, state, include_layout: bool = True):
        size, axes_positions, text_positions = state
        self.fig.change_tracker.addChange(
            self.fig,
            ".set_size_inches(%f/2.54, %f/2.54, forward=True)"
            % (size[0] * 2.54, size[1] * 2.54),
        )
        if not include_layout:
            return
        for axes, pos in zip(self.fig.axes, axes_positions):
            self.fig.change_tracker.addChange(
                axes,
                ".set_position([%f, %f, %f, %f])" % tuple(pos),
            )
        for text, pos in zip(self.fig.texts, text_positions):
            self.fig.change_tracker.addChange(
                text, ".set_position([%f, %f])" % tuple(pos)
            )

    def _figure_state(self):
        return (
            tuple(float(v) for v in self.fig.get_size_inches()),
            [self._axes_position(axes) for axes in self.fig.axes],
            [tuple(text.get_position()) for text in self.fig.texts],
        )

    def _apply_figure_state(self, state, include_layout: bool = True):
        size, axes_positions, text_positions = state
        self.fig.set_size_inches(size, forward=True)
        for axes, pos in zip(self.fig.axes, axes_positions):
            axes.set_position(pos)
        for text, pos in zip(self.fig.texts, text_positions):
            text.set_position(pos)
        self._record_figure_state(state, include_layout)
        self.fig.selection.update_selection_rectangles()
        self.fig.canvas.draw()
        widget = getattr(self.fig, "widget", None)
        if widget is not None:
            widget.updateGeometry()
        self.signals.figure_size_changed.emit()

    def changePos(self, value_x: float, value_y: float):
        """Move the selection by one display-space delta.

        Native Matplotlib positions can be data, axes, figure, or display
        coordinates.  Assigning one raw X/Y value to a mixed selection corrupts
        every element whose transform differs from the primary element.
        """
        selection = main_figure(self.element).selection
        elements = [target.target for target in selection.targets]
        if self.element not in elements:
            elements.append(self.element)
        wrappers = []
        seen = set()
        for element in elements:
            if id(element) in seen or not TargetWrapper.supports_target(element):
                continue
            seen.add(id(element))
            wrappers.append(TargetWrapper(element))
        if not wrappers:
            return

        primary = TargetWrapper(self.element)
        old_position = self.element.get_position()
        if getattr(old_position, "width", None) is not None:
            desired_native = [old_position.x0, old_position.y0]
        else:
            desired_native = [old_position[0], old_position[1]]
        if value_x is not None:
            desired_native[0] = value_x
        if value_y is not None:
            desired_native[1] = value_y

        current_display = np.asarray(primary.get_positions()[0], dtype=float)
        desired_display = np.asarray(
            primary.transform_points([desired_native])[0], dtype=float
        )
        delta = desired_display - current_display
        if np.allclose(delta, 0):
            return

        old_states = [wrapper.get_restore_state() for wrapper in wrappers]
        for wrapper in wrappers:
            wrapper.translate(delta)
        new_states = [wrapper.get_restore_state() for wrapper in wrappers]

        def apply(states):
            for wrapper, state in zip(wrappers, states):
                wrapper.restore_state(state)
            selection.update_extent()
            selection.update_selection_rectangles()
            self.fig.canvas.draw()

        def redo():
            apply(new_states)

        def undo():
            apply(old_states)

        self.fig.change_tracker.addEdit([undo, redo, "Change position"])
        selection.update_extent()
        selection.update_selection_rectangles()
        self.fig.signals.figure_selection_property_changed.emit()
        self.fig.canvas.draw()

    def changeSize(self, value: list, changed_axis: int | None = None):
        """change the size of an axes or figure"""
        if isinstance(self.element, Figure):
            if self._lock_aspect_enabled():
                current_size = tuple(float(v) for v in self.fig.get_size_inches())
                value = self._locked_size(value, current_size, changed_axis)
            old_state = self._figure_state()
            if self.scale_type == 0:
                include_layout = False
                self.fig.set_size_inches(value)
            else:
                include_layout = True
                if self.scale_type == 1:
                    changeFigureSize(value[0], value[1], fig=self.fig)
                elif self.scale_type == 2:
                    changeFigureSize(
                        value[0],
                        value[1],
                        cut_from_top=True,
                        cut_from_left=True,
                        fig=self.fig,
                    )
            new_state = self._figure_state()

            def undo():
                self._apply_figure_state(old_state, include_layout)

            def redo():
                self._apply_figure_state(new_state, include_layout)

            self._record_figure_state(new_state, include_layout)
            self.fig.change_tracker.addEdit([undo, redo, "Change figure size"])
            self.fig.selection.update_selection_rectangles()
            self.fig.canvas.draw()
            widget = getattr(self.fig, "widget", None)
            if widget is not None:
                widget.updateGeometry()
            self.setElement(self.element)
            self.signals.figure_size_changed.emit()
        else:
            elements = [self.element]
            elements += [
                element.target
                for element in self.element.figure.selection.targets
                if element.target != self.element and isinstance(element.target, Axes)
            ]

            old_positions = []
            new_positions = []
            for element in elements:
                pos = element.get_position()
                old_positions.append([pos.x0, pos.y0, pos.width, pos.height])
                pos = [pos.x0, pos.y0, pos.width, pos.height]
                size = list(value)
                if self._lock_aspect_enabled():
                    size = self._locked_size(size, (pos[2], pos[3]), changed_axis)
                pos[2] = size[0]
                pos[3] = size[1]
                new_positions.append(pos)

            fig = self.fig

            def apply_positions(positions):
                for element, pos in zip(elements, positions):
                    element.set_position(pos)
                    if isinstance(element, Axes):
                        fig.change_tracker.addNewAxesChange(element)
                    else:
                        fig.change_tracker.addChange(
                            element, ".set_position([%f, %f, %f, %f])" % tuple(pos)
                        )

            def redo():
                apply_positions(new_positions)

            def undo():
                apply_positions(old_positions)

            redo()
            self.fig.change_tracker.addEdit([undo, redo, "Change size"])
            self.setElement(self.element)
            self.fig.signals.figure_selection_property_changed.emit()
            self.fig.canvas.draw()

    def getTransform(self, element: Artist) -> Optional[mpl.transforms.Transform]:
        """get the transform of an Artist"""
        if isinstance(element, Figure):
            if self.transform_index == 0:
                return transforms.Affine2D().scale(2.54, 2.54)
            return None
        if isinstance(element, Axes):
            display_transform = TargetWrapper(element).get_transform()
            if self.transform_index == 0:
                return (
                    transforms.Affine2D().scale(2.54, 2.54)
                    + element.figure.dpi_scale_trans.inverted()
                    + display_transform
                )
            if self.transform_index == 1:
                return element.figure.dpi_scale_trans.inverted() + display_transform
            if self.transform_index == 2:
                return display_transform
            return None
        if self.transform_index == 0:
            return (
                transforms.Affine2D().scale(2.54, 2.54)
                + element.figure.dpi_scale_trans.inverted()
                + element.get_transform()
            )
        if self.transform_index == 1:
            return element.figure.dpi_scale_trans.inverted() + element.get_transform()
        if self.transform_index == 2:
            return element.get_transform()
        return None

    def setElement(self, element: Artist):
        """set the target Artist of this widget"""
        # self.label.setText(str(element))
        self.element = element

        self.input_shape_transform.setDisabled(True)
        self.input_transform.setDisabled(True)
        self.input_lock_aspect.setEnabled(True)

        if isinstance(element, Figure):
            pos = element.get_size_inches()
            self.input_shape.setTransform(self.getTransform(element))
            self.input_shape.setValue((pos[0], pos[1]))
            self.input_shape.setEnabled(True)
            self.input_transform.setEnabled(True)
            self.input_shape_transform.setEnabled(True)
        elif isinstance(element, Axes):
            pos = element.get_position()
            self.input_shape.setTransform(self.getTransform(element))
            self.input_shape.setValue((pos.width, pos.height))
            self.input_transform.setEnabled(True)
            self.input_shape.setEnabled(True)

        else:
            self.input_shape.setDisabled(True)

        try:
            pos = element.get_position()
            self.input_position.setTransform(self.getTransform(element))
            try:
                self.input_position.setValue(pos)
            except (ValueError, TypeError):
                self.input_position.setValue((pos.x0, pos.y0))
            self.input_transform.setEnabled(True)
            self.input_position.setEnabled(True)
        except (AttributeError, RuntimeError):
            self.input_position.setDisabled(True)
