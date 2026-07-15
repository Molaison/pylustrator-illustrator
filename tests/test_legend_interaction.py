from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backend_bases import MouseEvent
from matplotlib.lines import Line2D
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
    eval("command_parent" + command)

    changed = ax.get_legend()
    assert [text.get_text() for text in changed.get_texts()] == ["line A", "line B"]
    assert len(changed.legend_handles) == 2
    plt.close(fig)


def test_axes_proxy_legend_replay_uses_existing_legend_handles() -> None:
    from pylustrator.change_tracker import ChangeTracker

    fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
    legend = ax.legend(
        handles=[Patch(facecolor="red"), Patch(facecolor="blue")],
        labels=["proxy A", "proxy B"],
        loc="upper right",
    )
    fig.canvas.draw()
    assert ax.get_legend_handles_labels() == ([], [])

    tracker = ChangeTracker.__new__(ChangeTracker)
    command_parent, command = tracker.get_describtion_string(
        legend, exclude_default=False
    )

    assert command_parent is ax
    assert ".get_legend().legend_handles" in command
    eval("command_parent" + command)

    changed = ax.get_legend()
    assert [text.get_text() for text in changed.get_texts()] == [
        "proxy A",
        "proxy B",
    ]
    assert len(changed.legend_handles) == 2
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


def test_generated_source_migration_rewrites_legacy_legend_proxy_locator() -> None:
    from pylustrator.commands import migrate_generated_source

    source = (
        'plt.figure(1).axes[0].get_legend_handles_labels()[0][1].set_alpha(0.5)\n'
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
    assert commands[-1] == [method, ".set_frame_on(True)"]
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
