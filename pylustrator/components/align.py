import os
from qtpy import QtGui, QtWidgets
import qtawesome as qta


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

        self.layout_main = QtWidgets.QHBoxLayout(self)
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
            "mdi.rotate-left",
            "mdi.rotate-right",
        ]
        tooltips = {
            "same_width": "match width to first selected object",
            "same_height": "match height to first selected object",
            "same_size": "match size to first selected object",
            "scale_up": "enlarge selected objects",
            "scale_down": "shrink selected objects",
            "rotate_left": "rotate selected objects 15 degrees counterclockwise",
            "rotate_right": "rotate selected objects 15 degrees clockwise",
        }
        self.buttons = []
        align_group = QtWidgets.QButtonGroup(self)
        for index, act in enumerate(actions):
            icon_name = icons[index]
            if icon_name.endswith(".png"):
                icon = QtGui.QIcon(
                    os.path.join(os.path.dirname(__file__), "..", "icons", icon_name)
                )
            else:
                icon = qta.icon(icon_name)
            button = QtWidgets.QPushButton(
                icon,
                "",
            )
            button.setToolTip(tooltips.get(act, act.replace("_", " ")))
            self.layout_main.addWidget(button)
            button.clicked.connect(lambda x, act=act: self.execute_action(act))
            self.buttons.append(button)
            align_group.addButton(button)
            if index == 3 or index == 7 or index == 8 or index == 11:
                line = QtWidgets.QFrame()
                line.setFrameShape(QtWidgets.QFrame.VLine)
                line.setFrameShadow(QtWidgets.QFrame.Sunken)
                self.layout_main.addWidget(line)
        self.layout_main.addStretch()

    def execute_action(self, act: str):
        """execute an alignment action"""
        try:
            if act.startswith("same_"):
                self.fig.selection.match_size(act.removeprefix("same_"))
            elif act == "scale_up":
                self.fig.selection.scale_selection(1.1)
            elif act == "scale_down":
                self.fig.selection.scale_selection(1 / 1.1)
            elif act == "rotate_left":
                self.fig.selection.rotate_selection(15)
            elif act == "rotate_right":
                self.fig.selection.rotate_selection(-15)
            else:
                self.fig.selection.align_points(act)
        except ValueError as exc:
            QtWidgets.QMessageBox.warning(self, "Pylustrator", str(exc))
            return
        self.fig.selection.update_selection_rectangles()
        self.fig.canvas.draw()

    def setFigure(self, fig):
        self.fig = fig
