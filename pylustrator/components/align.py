import os
from qtpy import QtGui, QtWidgets
import qtawesome as qta

from ..operations import TransformOperation


class Align(QtWidgets.QWidget):
    def __init__(self, layout: QtWidgets.QLayout, signals):
        """A widget that allows to align the elements of a multi selection.

        Args:
            layout: the layout to which to add the widget
            fig: the target figure
        """
        QtWidgets.QWidget.__init__(self)
        layout.addWidget(self)

        signals.figure_changed.connect(self.setFigure)

        self.layout_main = QtWidgets.QGridLayout(self)
        self.layout_main.setContentsMargins(0, 0, 0, 0)

        actions = [
            "left_x",
            "center_x",
            "right_x",
            "distribute_x",
            "top_y",
            "center_y",
            "bottom_y",
            "distribute_y",
            "group",
            "same_width",
            "same_height",
            "same_size",
            "scale_up",
            "scale_down",
            "appearance_up",
            "appearance_down",
            "rotate_left",
            "rotate_right",
        ]
        icons = [
            "left_x.png",
            "center_x.png",
            "right_x.png",
            "distribute_x.png",
            "top_y.png",
            "center_y.png",
            "bottom_y.png",
            "distribute_y.png",
            "group.png",
            "fa5s.arrows-alt-h",
            "fa5s.arrows-alt-v",
            "fa5s.expand-arrows-alt",
            "fa5s.search-plus",
            "fa5s.search-minus",
            None,
            None,
            "mdi.rotate-left",
            "mdi.rotate-right",
        ]
        tooltips = {
            "same_width": "match width to the key object, or first selected object",
            "same_height": "match height to the key object, or first selected object",
            "same_size": "match size to the key object, or first selected object",
            "scale_up": (
                "enlarge selected geometry; font, stroke, and markers stay unchanged"
            ),
            "scale_down": (
                "shrink selected geometry; font, stroke, and markers stay unchanged"
            ),
            "appearance_up": "increase font, stroke, and marker appearance by 10%",
            "appearance_down": (
                "decrease font, stroke, and marker appearance by about 9.1%"
            ),
            "rotate_left": "rotate selected objects 15 degrees counterclockwise",
            "rotate_right": "rotate selected objects 15 degrees clockwise",
        }
        columns = 4
        self.buttons = []
        self.buttons_by_action = {}
        align_group = QtWidgets.QButtonGroup(self)
        for index, act in enumerate(actions):
            icon_name = icons[index]
            if icon_name is None:
                button = QtWidgets.QPushButton(
                    "A+" if act == "appearance_up" else "A−"
                )
            elif icon_name.endswith(".png"):
                icon = QtGui.QIcon(
                    os.path.join(os.path.dirname(__file__), "..", "icons", icon_name)
                )
                button = QtWidgets.QPushButton(icon, "")
            else:
                icon = qta.icon(icon_name)
                button = QtWidgets.QPushButton(icon, "")
            button.setSizePolicy(
                QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
            )
            button.setToolTip(tooltips.get(act, act.replace("_", " ")))
            self.layout_main.addWidget(button, index // columns, index % columns)
            button.clicked.connect(lambda x, act=act: self.execute_action(act))
            self.buttons.append(button)
            self.buttons_by_action[act] = button
            align_group.addButton(button)

        action_rows = (len(actions) + columns - 1) // columns
        self.reference_combo = QtWidgets.QComboBox(self)
        self.reference_combo.addItem("Align to Selection", "selection")
        self.reference_combo.addItem("Align to Key Object", "key_object")
        self.reference_combo.addItem("Align to Artboard", "artboard")
        self.reference_combo.setToolTip(
            "Choose the visible bounds used as the alignment reference"
        )
        self.layout_main.addWidget(
            self.reference_combo, action_rows, 0, 1, columns
        )
        self.reference_combo.currentIndexChanged.connect(
            self.change_reference_mode
        )

        self.spacing_enabled = QtWidgets.QCheckBox("Spacing", self)
        self.spacing_enabled.setToolTip(
            "Use an exact gap around the key object when distributing"
        )
        self.layout_main.addWidget(self.spacing_enabled, action_rows + 1, 0)
        self.spacing_input = QtWidgets.QDoubleSpinBox(self)
        self.spacing_input.setRange(-10000.0, 10000.0)
        self.spacing_input.setDecimals(2)
        self.spacing_input.setSuffix(" px")
        self.spacing_input.setToolTip(
            "Exact display-space gap; negative values overlap objects"
        )
        self.layout_main.addWidget(
            self.spacing_input, action_rows + 1, 1, 1, columns - 1
        )
        self.spacing_enabled.toggled.connect(self.refresh_controls)

        self.fig = None
        self._updating_reference = False
        signals.figure_element_selected.connect(self.selection_changed)
        selection_update = getattr(signals, "figure_selection_update", None)
        if selection_update is not None:
            selection_update.connect(self.refresh_controls)

    def selection_changed(self, _element) -> None:
        self.refresh_controls()

    def refresh_controls(self, *_args) -> None:
        selection = getattr(getattr(self, "fig", None), "selection", None)
        if selection is None:
            self.reference_combo.setEnabled(False)
            self.spacing_enabled.setEnabled(False)
            self.spacing_input.setEnabled(False)
            self.buttons_by_action["appearance_up"].setEnabled(False)
            self.buttons_by_action["appearance_down"].setEnabled(False)
            self.buttons_by_action["rotate_left"].setEnabled(False)
            self.buttons_by_action["rotate_right"].setEnabled(False)
            return
        self.reference_combo.setEnabled(True)
        rotation_enabled = bool(
            getattr(selection, "rotation_handle_supported", lambda: False)()
        )
        self.buttons_by_action["rotate_left"].setEnabled(rotation_enabled)
        self.buttons_by_action["rotate_right"].setEnabled(rotation_enabled)
        appearance_support = selection.operation_support(
            TransformOperation.SCALE_APPEARANCE
        )
        for action in ("appearance_up", "appearance_down"):
            button = self.buttons_by_action[action]
            button.setEnabled(appearance_support.supported)
            base_tooltip = (
                "increase font, stroke, and marker appearance by 10%"
                if action == "appearance_up"
                else (
                    "decrease font, stroke, and marker appearance by about 9.1% "
                    "(inverse of +10%)"
                )
            )
            button.setToolTip(
                base_tooltip
                if appearance_support.supported
                else f"{base_tooltip}\nUnavailable: {appearance_support.reason}"
            )
        mode = getattr(selection, "alignment_reference_mode", "selection")
        index = self.reference_combo.findData(mode)
        self._updating_reference = True
        try:
            if index >= 0:
                self.reference_combo.setCurrentIndex(index)
        finally:
            self._updating_reference = False
        key = getattr(selection, "alignment_key", None)
        key_enabled = (
            mode == "key_object"
            and len(selection.targets) >= 2
            and key is not None
        )
        self.spacing_enabled.setEnabled(key_enabled)
        self.spacing_input.setEnabled(
            key_enabled and self.spacing_enabled.isChecked()
        )
        if mode == "key_object" and key is not None:
            self.reference_combo.setToolTip(
                f"Key object: {type(key).__name__}; click another selected object to change it"
            )
        elif mode == "key_object":
            self.reference_combo.setToolTip(
                "Click a selected object to choose the alignment key"
            )
        else:
            self.reference_combo.setToolTip(
                "Choose the visible bounds used as the alignment reference"
            )

    def change_reference_mode(self, index: int) -> None:
        if self._updating_reference or self.fig is None:
            return
        mode = self.reference_combo.itemData(index)
        try:
            self.fig.selection.set_alignment_reference(mode)
        except ValueError as error:
            QtWidgets.QMessageBox.warning(self, "Pylustrator", str(error))
        self.refresh_controls()

    def execute_action(self, act: str):
        """execute an alignment action"""
        action_handles_redraw = False
        try:
            if act.startswith("same_"):
                self.fig.selection.match_size(act.removeprefix("same_"))
            elif act == "scale_up":
                self.fig.selection.scale_selection(1.1)
            elif act == "scale_down":
                self.fig.selection.scale_selection(1 / 1.1)
            elif act == "appearance_up":
                self.fig.selection.scale_appearance_selection(1.1)
                action_handles_redraw = True
            elif act == "appearance_down":
                self.fig.selection.scale_appearance_selection(1 / 1.1)
                action_handles_redraw = True
            elif act == "rotate_left":
                self.fig.selection.rotate_selection(15)
            elif act == "rotate_right":
                self.fig.selection.rotate_selection(-15)
            else:
                spacing = None
                if (
                    act in ("distribute_x", "distribute_y")
                    and self.spacing_input.isEnabled()
                    and self.spacing_enabled.isChecked()
                ):
                    spacing = self.spacing_input.value()
                self.fig.selection.align_points(act, spacing=spacing)
        except (TypeError, ValueError) as exc:
            QtWidgets.QMessageBox.warning(self, "Pylustrator", str(exc))
            return
        if not action_handles_redraw:
            self.fig.selection.update_selection_rectangles()
            self.fig.canvas.draw()
        self.refresh_controls()

    def setFigure(self, fig):
        self.fig = fig
        self.refresh_controls()
