from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch


def assert_bbox_close(actual, expected) -> None:
    assert np.allclose(actual, expected, atol=1e-9), (actual, expected)


class ChangeTracker:
    def addEdit(self, edit):
        self.edit = edit

    def addNewLegendChange(self, target):
        self.legend = target


class Dragger:
    def make_draggable(self, target):
        self.dragable = target

    def select_element(self, target):
        self.selected = target


class Selection:
    def update_selection_rectangles(self):
        self.updated = True


def attach_figure_helpers(fig) -> None:
    fig.change_tracker = ChangeTracker()
    fig.figure_dragger = Dragger()
    fig.selection = Selection()


def test_point_anchored_legend_move_keeps_anchor_compact_after_transform_change() -> None:
    from pylustrator.snap import TargetWrapper

    fig, ax = plt.subplots(figsize=(3.56, 3.35), dpi=100)
    attach_figure_helpers(fig)
    legend = fig.legend(
        handles=[
            Patch(label="ipTM-oriented"),
            Patch(label="+ pocket-oriented"),
            Patch(label="+ trajectory rescue"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.628, 0.99),
        ncol=3,
        fontsize=5.25,
        handlelength=0.72,
        columnspacing=0.3,
        handletextpad=0.2,
        borderaxespad=0.0,
        labelspacing=0.3,
        borderpad=0.28,
        frameon=False,
    )
    fig.canvas.draw()
    legend.get_transform = lambda: fig.transFigure
    renderer = fig.canvas.get_renderer()
    before = legend.get_window_extent(renderer).bounds
    anchor_before = legend.get_bbox_to_anchor().bounds

    wrapper = TargetWrapper(legend)
    original_positions = wrapper.get_positions()
    moved_positions = [point + [12, -7] for point in original_positions]
    wrapper.set_positions(moved_positions)
    fig.canvas.draw()

    moved = legend.get_window_extent(renderer).bounds
    anchor_after = legend.get_bbox_to_anchor().bounds
    assert anchor_before[2:] == (0.0, 0.0)
    assert anchor_after[2:] == (0.0, 0.0)
    assert_bbox_close((moved[2], moved[3]), (before[2], before[3]))
    assert_bbox_close((moved[0] - before[0], moved[1] - before[1]), (12.0, -7.0))
    plt.close(fig)


def test_proxy_legend_property_change_preserves_anchor_and_contents() -> None:
    from pylustrator.components.qitem_properties import LegendPropertiesWidget

    fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
    attach_figure_helpers(fig)
    legend = fig.legend(
        handles=[Patch(label="proxy A"), Patch(label="proxy B")],
        labels=["proxy A", "proxy B"],
        loc="upper center",
        bbox_to_anchor=(0.625, 0.99),
        frameon=False,
        fontsize=8,
    )
    fig.canvas.draw()
    anchor_before = legend.get_bbox_to_anchor().bounds
    loc_before = legend._loc

    widget = LegendPropertiesWidget.__new__(LegendPropertiesWidget)
    widget.properties = {"fontsize": 8, "frameon": False}
    widget.target = legend

    widget.changePropertiy("fontsize", 9)
    fig.canvas.draw()

    changed = fig.legends[0]
    assert changed.get_bbox_to_anchor().bounds[2:] == (0.0, 0.0)
    assert_bbox_close(changed.get_bbox_to_anchor().bounds, anchor_before)
    assert changed._loc == loc_before
    assert [text.get_text() for text in changed.get_texts()] == ["proxy A", "proxy B"]
    assert len(changed.legend_handles) == 2
    assert int(changed._fontsize) == 9
    plt.close(fig)


def test_legend_children_are_pickable_and_referenceable() -> None:
    from pylustrator.change_tracker import ChangeTracker, getReference
    from pylustrator.drag_helper import DragManager

    fig, ax = plt.subplots()
    legend = fig.legend(handles=[Patch(facecolor="red", label="proxy")], labels=["proxy"])
    text = legend.get_texts()[0]
    handle = legend.legend_handles[0]

    manager = DragManager.__new__(DragManager)
    manager.make_draggable(legend)

    assert text.pickable()
    assert handle.pickable()
    assert getReference(text).endswith(".legends[0].get_texts()[0]")
    assert getReference(handle).endswith(".legends[0].legend_handles[0]")

    tracker = ChangeTracker.__new__(ChangeTracker)
    tracker.figure = fig
    tracker.changes = {}
    tracker.saved = True
    tracker.no_save = False
    tracker.changeCountChanged = lambda: None
    fig.change_tracker = tracker

    text.set_text("edited")
    tracker.addNewTextChange(text)
    handle.set_alpha(0.5)
    tracker.addChange(handle, ".set_alpha(0.500000)")

    assert (text, ".set(text='edited')") in tracker.changes.values()
    assert (handle, ".set_alpha(0.500000)") in tracker.changes.values()
    plt.close(fig)
