from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from qtpy import QtCore, QtWidgets

from pylustrator.drag_helper import GrabbableRectangleSelection


class SelectionView:
    h = 200
    grabber_found = False


def test_multi_selection_has_visible_per_target_indicators() -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    scene = QtWidgets.QGraphicsScene()
    origin = QtWidgets.QGraphicsRectItem()
    origin.view = SelectionView()
    scene.addItem(origin)

    fig, axes = plt.subplots(1, 2, figsize=(4, 2), dpi=100)
    fig._pyl_scene = origin
    selection = GrabbableRectangleSelection(fig, origin)

    selection.add_target(axes[0])
    selection.add_target(axes[1])

    assert len(selection.targets) == 2
    assert len(selection.targets_rects) == 4

    primary_rects = selection.targets_rects[0::2]
    contrast_rects = selection.targets_rects[1::2]
    for rect in primary_rects:
        assert rect.isVisible()
        assert rect.zValue() >= 900
        assert rect.pen().width() >= 3
        assert rect.pen().color().name().lower() == "#1e88e5"
        assert rect.brush().color().alpha() > 0
    for rect in contrast_rects:
        assert rect.isVisible()
        assert rect.zValue() > primary_rects[0].zValue()
        assert rect.pen().style() == QtCore.Qt.DashLine

    selection.clear_targets()
    plt.close(fig)
    assert app is not None
