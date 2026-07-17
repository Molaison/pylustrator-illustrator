from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.backend_bases import MouseEvent
from matplotlib.collections import PathCollection
from matplotlib.lines import Line2D
from matplotlib.path import Path
from matplotlib.patches import Patch


def assert_bbox_close(actual, expected) -> None:
    assert np.allclose(actual, expected, atol=1e-9), (actual, expected)


class ChangeTracker:
    def addEdit(self, edit):
        self.edit = edit

    def addChange(self, target, command):
        self.command = (target, command)

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

    def update_extent(self):
        self.extent_updated = True


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


def test_frameon_property_preserves_legend_identity_children_and_undo() -> None:
    from pylustrator.change_tracker import getReference
    from pylustrator.components.qitem_properties import LegendPropertiesWidget

    fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
    attach_figure_helpers(fig)
    legend = ax.legend(
        handles=[Patch(label="proxy A"), Patch(label="proxy B")],
        labels=["proxy A", "proxy B"],
        loc="upper center",
        frameon=False,
        fontsize=8,
    )
    fig.canvas.draw()
    children = tuple([*legend.legend_handles, *legend.get_texts()])
    renderer = fig.canvas.get_renderer()
    child_bounds = [child.get_window_extent(renderer).bounds for child in children]

    widget = LegendPropertiesWidget.__new__(LegendPropertiesWidget)
    widget.properties = {"frameon": False}
    widget.target = legend
    widget.changePropertiy("frameon", True)

    assert ax.get_legend() is legend
    assert widget.target is legend
    assert fig.figure_dragger.selected is legend
    assert tuple([*legend.legend_handles, *legend.get_texts()]) == children
    assert [child.get_window_extent(renderer).bounds for child in children] == child_bounds
    assert legend.get_frame_on()
    assert getReference(legend).endswith(".get_legend()")

    undo, redo, _name = fig.change_tracker.edit
    undo()
    assert ax.get_legend() is legend
    assert not legend.get_frame_on()
    redo()
    assert ax.get_legend() is legend
    assert legend.get_frame_on()
    plt.close(fig)


def test_frameon_undo_keeps_live_legend_after_identity_preserving_reflow() -> None:
    from pylustrator.components.qitem_properties import LegendPropertiesWidget

    fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
    attach_figure_helpers(fig)
    legend = ax.legend(
        handles=[Patch(label="proxy A"), Patch(label="proxy B")],
        labels=["proxy A", "proxy B"],
        frameon=False,
        borderpad=0.4,
    )
    fig.canvas.draw()

    widget = LegendPropertiesWidget.__new__(LegendPropertiesWidget)
    widget.properties = {"frameon": False, "borderpad": 0.4}
    widget.target = legend
    widget.changePropertiy("frameon", True)
    frame_undo, frame_redo, _name = fig.change_tracker.edit

    widget.changePropertiy("borderpad", 0.2)
    reflowed = ax.get_legend()
    assert reflowed is legend
    assert reflowed.get_frame_on()

    frame_undo()
    assert not ax.get_legend().get_frame_on()
    frame_redo()
    assert ax.get_legend().get_frame_on()
    plt.close(fig)


def test_legend_selection_bounds_follow_visible_children_outside_layout_frame() -> None:
    from pylustrator.artist_adapters import get_artist_adapter

    fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
    legend = ax.legend(
        handles=[Patch(label="A"), Patch(label="B")],
        labels=["A", "B"],
        loc="upper center",
        frameon=False,
    )
    fig.canvas.draw()
    legend.get_texts()[0].set_position((80.0, -25.0))
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    children = [*legend.legend_handles, *legend.get_texts()]
    extents = np.array(
        [child.get_window_extent(renderer).extents for child in children], dtype=float
    )
    visible_bounds = np.array(
        [
            np.min(extents[:, 0]),
            np.min(extents[:, 1]),
            np.max(extents[:, 2]),
            np.max(extents[:, 3]),
        ]
    )
    points = get_artist_adapter(legend).selection_points()
    selection_bounds = np.array([*points[0], *points[1]])
    layout_bounds = np.array(legend.get_window_extent(renderer).extents)

    assert np.allclose(selection_bounds, visible_bounds)
    assert not np.allclose(selection_bounds, layout_bounds)
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


def test_legend_marker_hit_testing_prefers_handle() -> None:
    from pylustrator.drag_helper import DragManager, get_artist_children

    fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
    ax.plot([0, 1], [0, 1], marker="o", markersize=10, label="line")
    legend = ax.legend()
    fig.canvas.draw()
    handle = legend.legend_handles[0]

    manager = DragManager.__new__(DragManager)
    manager.figure = fig
    manager.make_draggable(legend)
    fig.canvas.draw()

    bbox = handle.get_window_extent(fig.canvas.get_renderer())
    event = MouseEvent(
        "button_press_event",
        fig.canvas,
        (bbox.x0 + bbox.x1) / 2,
        (bbox.y0 + bbox.y1) / 2,
        button=1,
    )
    picked, _finished = manager.get_picked_element(event)

    assert handle in get_artist_children(legend)
    assert handle.contains(event)[0]
    assert picked is handle
    plt.close(fig)


def test_axes_legend_change_description_uses_axes_parent_transform() -> None:
    from pylustrator.change_tracker import ChangeTracker

    fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
    ax.plot([0, 1], [0, 1], label="line")
    legend = ax.legend(loc="upper right", bbox_to_anchor=(0.95, 0.95))
    fig.canvas.draw()

    tracker = ChangeTracker.__new__(ChangeTracker)
    command_parent, command = tracker.get_describtion_string(legend, exclude_default=False)

    assert command_parent is ax
    assert command.startswith(".legend(")
    assert "bbox_to_anchor=" in command
    plt.close(fig)


def test_axes_legend_change_description_preserves_handles_and_labels() -> None:
    from pylustrator.change_tracker import ChangeTracker

    fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
    line1 = ax.plot([0, 1], [0, 1], label="line A")[0]
    line2 = ax.plot([0, 1], [1, 0], label="line B")[0]
    legend = ax.legend(handles=[line1, line2], labels=["line A", "line B"])
    fig.canvas.draw()

    tracker = ChangeTracker.__new__(ChangeTracker)
    command_parent, command = tracker.get_describtion_string(legend, exclude_default=False)

    assert command_parent is ax
    assert "handles=" in command
    assert "labels=['line A', 'line B']" in command
    assert ".get_legend_handles_labels()[0]" in command
    eval("command_parent" + command)

    changed = ax.get_legend()
    assert [text.get_text() for text in changed.get_texts()] == ["line A", "line B"]
    assert len(changed.legend_handles) == 2
    plt.close(fig)


def test_axes_proxy_legend_replay_freezes_handles_without_existing_legend() -> None:
    from pylustrator.change_tracker import ChangeTracker

    fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
    legend = ax.legend(
        handles=[
            Patch(facecolor="red", edgecolor="black", linewidth=2),
            Line2D(
                [],
                [],
                color="blue",
                linestyle=(1, (3, 2)),
                marker="o",
                markersize=7,
            ),
            PathCollection(
                [Path.unit_circle()],
                sizes=[49],
                facecolors=[(0.2, 0.7, 0.3, 1.0)],
                edgecolors=[(0.1, 0.2, 0.1, 1.0)],
            ),
        ],
        labels=["proxy patch", "proxy line", "proxy collection"],
        loc="upper right",
        markerscale=2.0,
    )
    fig.canvas.draw()
    expected_dash_pattern = legend.legend_handles[1]._unscaled_dash_pattern
    assert ax.get_legend_handles_labels() == ([], [])

    tracker = ChangeTracker.__new__(ChangeTracker)
    command_parent, command = tracker.get_describtion_string(
        legend, exclude_default=False
    )

    assert command_parent is ax
    assert ".get_legend().legend_handles" not in command
    assert "mpl.patches.Patch" in command
    assert "mpl.lines.Line2D" in command
    assert "mpl.collections.PathCollection" in command

    plt.close(fig)
    fig2, ax2 = plt.subplots(figsize=(4, 3), dpi=100)
    assert ax2.get_legend() is None
    eval(
        "command_parent" + command,
        {"command_parent": ax2, "mpl": matplotlib, "np": np},
    )
    fig2.canvas.draw()

    changed = ax2.get_legend()
    assert [text.get_text() for text in changed.get_texts()] == [
        "proxy patch",
        "proxy line",
        "proxy collection",
    ]
    assert len(changed.legend_handles) == 3
    assert np.allclose(changed.legend_handles[0].get_facecolor(), (1, 0, 0, 1))
    assert changed.legend_handles[0].get_linewidth() == 2
    assert changed.legend_handles[1].get_color() == "blue"
    assert changed.legend_handles[1].get_marker() == "o"
    actual_dash_pattern = changed.legend_handles[1]._unscaled_dash_pattern
    assert actual_dash_pattern[0] == expected_dash_pattern[0]
    assert np.allclose(actual_dash_pattern[1], expected_dash_pattern[1])
    assert changed.markerscale == 2.0
    assert changed.legend_handles[1].get_markersize() == 14
    assert np.allclose(changed.legend_handles[2].get_sizes(), [196])
    assert np.allclose(
        changed.legend_handles[2].get_facecolors(), [(0.2, 0.7, 0.3, 1.0)]
    )
    plt.close(fig2)


def test_axes_handles_with_matching_labels_but_different_style_are_not_reused() -> None:
    from pylustrator.change_tracker import ChangeTracker

    fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
    ax.plot([0, 1], [0, 1], color="black", label="same label")
    legend = ax.legend(
        handles=[Line2D([], [], color="red", linewidth=4, label="same label")],
        labels=["same label"],
    )
    fig.canvas.draw()

    tracker = ChangeTracker.__new__(ChangeTracker)
    _command_parent, command = tracker.get_describtion_string(
        legend, exclude_default=False
    )

    assert ".get_legend_handles_labels()[0]" not in command
    assert "mpl.lines.Line2D" in command
    assert "color='red'" in command
    plt.close(fig)


def test_explicit_composite_legend_fails_replay_capability_preflight() -> None:
    from pylustrator.artist_adapters import UnsupportedArtistError, get_artist_adapter
    from pylustrator.change_tracker import ChangeTracker
    from pylustrator.legend_replay import UnsupportedLegendEntry

    fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
    errorbar = ax.errorbar([0, 1], [0, 1], yerr=[0.1, 0.2], label="_nolegend_")
    legend = ax.legend(handles=[errorbar], labels=["explicit errorbar"])
    fig.canvas.draw()

    adapter = get_artist_adapter(legend)
    assert adapter.capabilities.can_select
    assert not adapter.capabilities.can_translate
    assert not adapter.capabilities.can_snapshot
    assert not adapter.capabilities.can_serialize
    assert legend.get_frame_on()
    with pytest.raises(UnsupportedArtistError):
        adapter.set_frame_on(False)
    assert legend.get_frame_on()

    tracker = ChangeTracker.__new__(ChangeTracker)
    with pytest.raises(UnsupportedLegendEntry, match="composite"):
        tracker.get_describtion_string(legend, exclude_default=False)

    from pylustrator.components.qitem_properties import LegendPropertiesWidget

    widget = LegendPropertiesWidget.__new__(LegendPropertiesWidget)
    widget.target = legend
    widget.properties = {"fontsize": legend._fontsize}
    entry_children_before = [
        tuple(entry.get_children())
        for entry in legend._legend_handle_box.findobj(
            lambda artist: type(artist).__name__ == "DrawingArea"
        )
    ]
    with pytest.raises(UnsupportedArtistError):
        widget.changePropertiy("fontsize", legend._fontsize + 1)
    entry_children_after = [
        tuple(entry.get_children())
        for entry in legend._legend_handle_box.findobj(
            lambda artist: type(artist).__name__ == "DrawingArea"
        )
    ]
    assert ax.get_legend() is legend
    assert entry_children_after == entry_children_before
    assert legend._fontsize == widget.properties["fontsize"]
    plt.close(fig)


def test_legend_property_rebuild_uses_unscaled_proxy_handles() -> None:
    from pylustrator.components.qitem_properties import LegendPropertiesWidget

    fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
    attach_figure_helpers(fig)
    legend = ax.legend(
        handles=[
            Line2D([], [], marker="o", markersize=7, linestyle="None"),
            PathCollection(
                [Path.unit_circle()],
                sizes=[49],
                facecolors=[(0.2, 0.7, 0.3, 1.0)],
            ),
        ],
        labels=["line proxy", "collection proxy"],
        markerscale=2.0,
    )
    fig.canvas.draw()
    assert legend.legend_handles[0].get_markersize() == 14
    assert np.allclose(legend.legend_handles[1].get_sizes(), [196])

    widget = LegendPropertiesWidget.__new__(LegendPropertiesWidget)
    widget.target = legend
    widget.properties = {
        "loc": legend._loc,
        "borderpad": legend.borderpad,
        "markerscale": legend.markerscale,
    }
    widget.changePropertiy("borderpad", 0.2)
    changed = ax.get_legend()

    assert changed.legend_handles[0].get_markersize() == 14
    assert np.allclose(changed.legend_handles[1].get_sizes(), [196])
    assert changed.markerscale == 2.0
    assert changed.borderpad == 0.2
    plt.close(fig)


def test_semantic_errorbar_composite_reuses_underlying_axes_handle() -> None:
    from pylustrator.artist_adapters import get_artist_adapter
    from pylustrator.change_tracker import ChangeTracker
    from pylustrator.components.qitem_properties import LegendPropertiesWidget

    fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
    attach_figure_helpers(fig)
    ax.errorbar([0, 1], [0, 1], yerr=[0.1, 0.2], label="errorbar")
    legend = ax.legend()
    fig.canvas.draw()

    assert get_artist_adapter(legend).capabilities.can_serialize
    tracker = ChangeTracker.__new__(ChangeTracker)
    _command_parent, command = tracker.get_describtion_string(
        legend, exclude_default=False
    )
    assert ".get_legend_handles_labels()[0]" in command

    def glyph_signature(target):
        return [
            tuple(type(child).__name__ for child in entry.get_children())
            for entry in target._legend_handle_box.findobj(
                lambda artist: type(artist).__name__ == "DrawingArea"
            )
        ]

    before = glyph_signature(legend)
    widget = LegendPropertiesWidget.__new__(LegendPropertiesWidget)
    widget.target = legend
    widget.properties = {
        "loc": legend._loc,
        "borderpad": legend.borderpad,
        "markerscale": legend.markerscale,
    }
    widget.changePropertiy("borderpad", 0.2)
    assert glyph_signature(ax.get_legend()) == before
    plt.close(fig)


def test_legacy_proxy_handle_reference_replays_after_generated_block_init() -> None:
    from pylustrator.change_tracker import init_figure

    fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
    legend = ax.legend(
        handles=[Patch(facecolor="red"), Patch(facecolor="blue")],
        labels=["proxy A", "proxy B"],
    )
    assert ax.get_legend_handles_labels() == ([], [])

    init_figure(fig)

    legacy_handles = ax.get_legend_handles_labels()[0]
    assert legacy_handles == list(legend.legend_handles)
    legacy_handles[0].set_alpha(0.25)
    assert legend.legend_handles[0].get_alpha() == 0.25
    plt.close(fig)


def test_legacy_proxy_compatibility_rejects_mismatched_underlying_handles() -> None:
    from pylustrator.change_tracker import init_figure

    fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
    for index in range(5):
        ax.plot([0, 1], [index, index + 1], label=f"connector {index}")
    legend = ax.legend(
        handles=[
            Patch(facecolor="red"),
            Patch(facecolor="green"),
            Patch(facecolor="blue"),
        ],
        labels=["Binder", "Domain 1", "Domain 2"],
    )
    fig.canvas.draw()

    init_figure(fig)
    handles, labels = ax.get_legend_handles_labels()

    assert handles == list(legend.legend_handles)
    assert labels == ["Binder", "Domain 1", "Domain 2"]
    assert all(isinstance(handle, Patch) for handle in handles)
    plt.close(fig)


def test_generated_source_migration_rewrites_legacy_legend_proxy_locator() -> None:
    from pylustrator.commands import migrate_generated_source

    source = (
        '#% start: automatic generated code from pylustrator\n'
        'plt.figure(1).axes[0].get_legend_handles_labels()[0][1].set_alpha(0.5)\n'
        '#% end: automatic generated code from pylustrator\n'
    )

    migrated = migrate_generated_source(source)

    assert "get_legend_handles_labels" not in migrated
    assert ".get_legend().legend_handles[1]" in migrated


def test_extra_axes_legend_uses_artist_reference_not_current_axes_legend() -> None:
    from pylustrator.change_tracker import ChangeTracker, getReference

    fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
    method = ax.legend(
        handles=[Line2D([0], [0], label="method")],
        labels=["method"],
        loc="upper right",
        bbox_to_anchor=(0.9, 0.9),
        title="method",
    )
    ax.add_artist(method)
    current = ax.legend(
        handles=[Line2D([0], [0], label="current")],
        labels=["current"],
        loc="lower right",
        title="current",
    )
    fig.canvas.draw()

    tracker = ChangeTracker.__new__(ChangeTracker)
    commands = tracker.get_describtion_string(method, exclude_default=False)

    assert getReference(method).endswith(".artists[0]")
    assert getReference(current).endswith(".get_legend()")
    assert commands[0] == [method, "._set_loc(1)"]
    assert commands[1][0] is method
    assert commands[1][1].startswith(".set_bbox_to_anchor(")
    assert "transFigure" in commands[1][1]
    assert "get_legend()" not in commands[1][1]
    assert [method, ".set_frame_on(True)"] in commands
    assert commands[-1][0] is method
    assert commands[-1][1].startswith("._pylustrator_reflow_layout(")
    for command_parent, command in commands:
        eval("command_parent" + command)
    assert ax.get_legend() is current
    assert ax.artists[0] is method
    plt.close(fig)


def test_extra_axes_legend_saved_move_reopens_on_same_artist() -> None:
    from pylustrator.change_tracker import ChangeTracker
    from pylustrator.snap import TargetWrapper

    def make_figure():
        fig, ax = plt.subplots(num=1, clear=True, figsize=(4, 3), dpi=100)
        method = ax.legend(
            handles=[Line2D([0], [0], label="method")],
            labels=["method"],
            loc="upper right",
            bbox_to_anchor=(0.9, 0.9),
            title="method",
        )
        ax.add_artist(method)
        current = ax.legend(
            handles=[Line2D([0], [0], label="current")],
            labels=["current"],
            loc="lower right",
            title="current",
        )
        fig.canvas.draw()
        return fig, ax, method, current

    plt.close("all")
    fig, ax, method, current = make_figure()
    tracker = ChangeTracker.__new__(ChangeTracker)
    tracker.figure = fig
    tracker.changes = {}
    tracker.saved = True
    tracker.no_save = False
    tracker.changeCountChanged = lambda: None
    fig.change_tracker = tracker
    renderer = fig.canvas.get_renderer()
    before = method.get_window_extent(renderer).bounds

    wrapper = TargetWrapper(method)
    wrapper.set_positions([point + [12, -7] for point in wrapper.get_positions()])
    fig.canvas.draw()
    moved = method.get_window_extent(renderer).bounds
    saved_lines = tracker.sorted_changes()

    assert saved_lines == [
        "plt.figure(1).axes[0].artists[0]._set_loc(1)",
        "plt.figure(1).axes[0].artists[0].get_frame().set(linewidth=1.0, edgecolor=(0.8, 0.8, 0.8, 0.8), facecolor=(1.0, 1.0, 1.0, 0.8), alpha=0.8)",
        "plt.figure(1).axes[0].artists[0].set_bbox_to_anchor((0.8525000000000001, 0.7796666666666668), transform=plt.figure(1).transFigure)",
    ]
    assert ax.get_legend() is current
    assert ax.artists[0] is method

    plt.close(fig)
    fig2, ax2, method2, current2 = make_figure()
    for line in saved_lines:
        exec(line)
    fig2.canvas.draw()
    reopened = method2.get_window_extent(fig2.canvas.get_renderer()).bounds

    assert ax2.get_legend() is current2
    assert ax2.artists[0] is method2
    assert [text.get_text() for text in current2.get_texts()] == ["current"]
    assert [text.get_text() for text in method2.get_texts()] == ["method"]
    assert np.allclose((moved[0], moved[1]), (reopened[0], reopened[1]), atol=0.01)
    assert_bbox_close((moved[2], moved[3]), (reopened[2], reopened[3]))
    assert_bbox_close((moved[0] - before[0], moved[1] - before[1]), (12.0, -7.0))
    plt.close(fig2)


def test_extra_axes_legend_replay_after_axes_position_change() -> None:
    from pylustrator.change_tracker import ChangeTracker
    from pylustrator.snap import TargetWrapper

    axes_position = [0.1239, 0.2097, 0.7228, 0.7258]

    def make_figure():
        fig, ax = plt.subplots(num=1, clear=True, figsize=(4, 3), dpi=100)
        method = ax.legend(
            handles=[Line2D([0], [0], label="method")],
            labels=["method"],
            loc="upper right",
            bbox_to_anchor=(0.944, 0.9818),
            title="method",
        )
        ax.add_artist(method)
        current = ax.legend(
            handles=[Line2D([0], [0], label="current")],
            labels=["current"],
            loc="lower right",
            title="current",
        )
        fig.canvas.draw()
        return fig, ax, method, current

    plt.close("all")
    fig, ax, method, current = make_figure()
    tracker = ChangeTracker.__new__(ChangeTracker)
    tracker.figure = fig
    tracker.changes = {}
    tracker.saved = True
    tracker.no_save = False
    tracker.changeCountChanged = lambda: None
    fig.change_tracker = tracker

    ax.set(position=axes_position)
    tracker.addChange(ax, ".set(position=[0.1239, 0.2097, 0.7228, 0.7258])")
    fig.canvas.draw()
    before = method.get_window_extent(fig.canvas.get_renderer()).bounds
    wrapper = TargetWrapper(method)
    wrapper.set_positions([point + [20, -10] for point in wrapper.get_positions()])
    fig.canvas.draw()
    moved = method.get_window_extent(fig.canvas.get_renderer()).bounds
    saved_lines = tracker.sorted_changes()

    axes_index = next(index for index, line in enumerate(saved_lines) if ".set(position=" in line)
    legend_index = next(index for index, line in enumerate(saved_lines) if ".artists[0].set_bbox_to_anchor" in line)
    assert axes_index < legend_index

    plt.close(fig)
    fig2, ax2, method2, current2 = make_figure()
    for line in saved_lines:
        exec(line)
    fig2.canvas.draw()
    reopened = method2.get_window_extent(fig2.canvas.get_renderer()).bounds

    assert ax2.get_legend() is current2
    assert ax2.artists[0] is method2
    assert [text.get_text() for text in current2.get_texts()] == ["current"]
    assert [text.get_text() for text in method2.get_texts()] == ["method"]
    assert np.allclose((moved[0], moved[1]), (reopened[0], reopened[1]), atol=0.1)
    assert_bbox_close((moved[2], moved[3]), (reopened[2], reopened[3]))
    assert_bbox_close((moved[0] - before[0], moved[1] - before[1]), (20.0, -10.0))
    plt.close(fig2)


def test_axes_legend_move_records_change_without_transfigure_error() -> None:
    from pylustrator.change_tracker import ChangeTracker
    from pylustrator.snap import TargetWrapper

    fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
    tracker = ChangeTracker.__new__(ChangeTracker)
    tracker.figure = fig
    tracker.changes = {}
    tracker.saved = True
    tracker.no_save = False
    tracker.changeCountChanged = lambda: None
    fig.change_tracker = tracker
    ax.plot([0, 1], [0, 1], label="line")
    legend = ax.legend(loc="upper right", bbox_to_anchor=(0.95, 0.95))
    fig.canvas.draw()

    wrapper = TargetWrapper(legend)
    moved_positions = [point + [8, -4] for point in wrapper.get_positions()]
    wrapper.set_positions(moved_positions)

    commands = list(tracker.changes.values())
    assert commands[0][0] is ax
    assert commands[0][1].startswith(".legend(")
    assert "bbox_to_anchor=" in commands[0][1]
    plt.close(fig)


def test_axes_legend_frame_style_replays_after_legend_creation() -> None:
    from pylustrator.artist_adapters import get_artist_adapter
    from pylustrator.change_tracker import ChangeTracker

    def make_figure(*, with_legend):
        fig, ax = plt.subplots(num=1, clear=True, figsize=(4, 3), dpi=100)
        ax.plot([0, 1], [0, 1], label="line")
        legend = ax.legend(frameon=True) if with_legend else None
        fig.canvas.draw()
        return fig, ax, legend

    plt.close("all")
    fig, ax, legend = make_figure(with_legend=True)
    tracker = ChangeTracker.__new__(ChangeTracker)
    tracker.figure = fig
    tracker.changes = {}
    tracker.saved = True
    tracker.no_save = False
    tracker.changeCountChanged = lambda: None
    fig.change_tracker = tracker
    legend.get_frame().set(
        linewidth=0.5,
        edgecolor=(0.2, 0.3, 0.4, 0.7),
        facecolor=(0.8, 0.7, 0.6, 0.4),
        alpha=0.6,
    )
    get_artist_adapter(legend).record_changes()
    saved_lines = tracker.sorted_changes()

    legend_index = next(
        index for index, line in enumerate(saved_lines) if ".legend(" in line
    )
    frame_index = next(
        index for index, line in enumerate(saved_lines) if ".get_legend().get_frame()" in line
    )
    assert legend_index < frame_index

    plt.close(fig)
    fig2, ax2, _ = make_figure(with_legend=False)
    for line in saved_lines:
        exec(line)
    fig2.canvas.draw()

    replayed_frame = ax2.get_legend().get_frame()
    assert replayed_frame.get_linewidth() == 0.5
    assert np.allclose(replayed_frame.get_edgecolor(), (0.2, 0.3, 0.4, 0.6))
    assert np.allclose(replayed_frame.get_facecolor(), (0.8, 0.7, 0.6, 0.6))
    assert replayed_frame.get_alpha() == 0.6
    plt.close(fig2)


def test_axes_recording_preserves_logically_owned_legend_commands() -> None:
    from pylustrator.artist_adapters import get_artist_adapter
    from pylustrator.change_tracker import ChangeTracker, init_figure

    def make_figure(*, with_legend):
        fig, ax = plt.subplots(num=1, clear=True, figsize=(4, 3), dpi=100)
        legend = None
        if with_legend:
            legend = ax.legend(
                handles=[Patch(facecolor="red", edgecolor="black")],
                labels=["proxy"],
                frameon=True,
            )
        fig.canvas.draw()
        return fig, ax, legend

    plt.close("all")
    fig, ax, legend = make_figure(with_legend=True)
    init_figure(fig)
    tracker = ChangeTracker.__new__(ChangeTracker)
    tracker.figure = fig
    tracker.changes = {}
    tracker.saved = True
    tracker.no_save = False
    tracker.changeCountChanged = lambda: None
    fig.change_tracker = tracker

    legend.get_frame().set_linewidth(0.5)
    get_artist_adapter(legend).record_changes()
    ax.set_position([0.2, 0.18, 0.65, 0.7])
    get_artist_adapter(ax).record_changes()

    legend_commands = {
        reference_command
        for (reference_obj, reference_command) in tracker.changes
        if reference_obj is legend
    }
    assert {".legend", ".get_frame"} <= legend_commands
    saved_lines = tracker.sorted_changes()
    assert sum(".legend(" in line for line in saved_lines) == 1
    assert sum(".get_legend().get_frame()" in line for line in saved_lines) == 1

    plt.close(fig)
    fig2, ax2, _ = make_figure(with_legend=False)
    namespace = {"plt": plt, "mpl": matplotlib, "np": np}
    for line in saved_lines:
        exec(line, namespace)
    fig2.canvas.draw()

    assert [text.get_text() for text in ax2.get_legend().get_texts()] == ["proxy"]
    assert ax2.get_legend().get_frame().get_linewidth() == 0.5
    assert np.allclose(ax2.get_position().bounds, [0.2, 0.18, 0.65, 0.7])
    plt.close(fig2)


def test_figure_level_legend_saved_move_reopens_without_duplicate_legend() -> None:
    from pylustrator.change_tracker import ChangeTracker
    from pylustrator.snap import TargetWrapper

    def make_figure():
        fig, ax = plt.subplots(num=1, clear=True, figsize=(3.56, 3.35), dpi=100)
        method_handles = [
            Patch(facecolor="red", label="ipTM-oriented"),
            Patch(facecolor="blue", label="+ pocket-oriented"),
            Patch(facecolor="green", label="+ trajectory rescue"),
        ]
        segment_handles = [
            Patch(facecolor="#BDBDBD", label="pocket occupancy > 0.8"),
            Patch(facecolor="#4A4A4A", label="+ ipTM > 0.8"),
        ]
        method = fig.legend(
            handles=method_handles,
            frameon=False,
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
        )
        fig.add_artist(method)
        segment = fig.legend(
            handles=segment_handles,
            frameon=False,
            loc="upper center",
            bbox_to_anchor=(0.628, 0.925),
            ncol=2,
            fontsize=5.25,
            handlelength=0.72,
            columnspacing=0.34,
            handletextpad=0.2,
            borderaxespad=0.0,
            labelspacing=0.3,
            borderpad=0.28,
        )
        fig.canvas.draw()
        return fig, ax, method, segment

    plt.close("all")
    fig, ax, method, segment = make_figure()
    tracker = ChangeTracker.__new__(ChangeTracker)
    tracker.figure = fig
    tracker.changes = {}
    tracker.saved = True
    tracker.no_save = False
    tracker.changeCountChanged = lambda: None
    fig.change_tracker = tracker
    before_count = len(fig.legends)
    before = method.get_window_extent(fig.canvas.get_renderer()).bounds

    wrapper = TargetWrapper(method)
    wrapper.set_positions([point + [10, -6] for point in wrapper.get_positions()])
    fig.canvas.draw()
    moved = method.get_window_extent(fig.canvas.get_renderer()).bounds
    saved_lines = tracker.sorted_changes()

    assert before_count == 2
    assert all(".legend(" not in line for line in saved_lines)

    plt.close(fig)
    fig2, ax2, method2, segment2 = make_figure()
    for line in saved_lines:
        exec(line)
    fig2.canvas.draw()
    reopened = method2.get_window_extent(fig2.canvas.get_renderer()).bounds

    assert len(fig2.legends) == 2
    assert fig2.legends[0] is method2
    assert fig2.legends[1] is segment2
    assert [text.get_text() for text in method2.get_texts()] == [
        "ipTM-oriented",
        "+ pocket-oriented",
        "+ trajectory rescue",
    ]
    assert np.allclose((moved[0], moved[1]), (reopened[0], reopened[1]), atol=0.1)
    assert_bbox_close((moved[2], moved[3]), (reopened[2], reopened[3]))
    assert_bbox_close((moved[0] - before[0], moved[1] - before[1]), (10.0, -6.0))
    plt.close(fig2)


def test_numpy_scalar_values_are_saved_as_plain_python_literals() -> None:
    from pylustrator.change_tracker import kwargs_to_string

    saved = kwargs_to_string({"position": (np.int64(0), np.float64(0.5))})

    assert saved == "position=(0, 0.5)"
