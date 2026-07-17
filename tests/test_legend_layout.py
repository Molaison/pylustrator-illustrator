from __future__ import annotations

from dataclasses import FrozenInstanceError
import subprocess
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.lines import Line2D
from matplotlib.offsetbox import DrawingArea, TextArea, VPacker
from matplotlib.patches import Patch
from qtpy import QtWidgets

from pylustrator.artist_adapters import UnsupportedArtistError, get_artist_adapter
from pylustrator.change_tracker import ChangeTracker, init_figure
from pylustrator.legend_layout import (
    LegendLayoutError,
    LegendLayoutPlan,
    LegendLayoutSpec,
    _matplotlib_only_reflow,
    ensure_legend_layout_baseline,
    legend_layout_replay_bootstrap,
    reflow_legend_layout,
)
from pylustrator.operations import TransformIntent, TransformOperation
from pylustrator.source_doctor import diagnose_generated_source
from pylustrator.transform_engine import TransformPlan, TransformPreflightError


def _make_legend(owner: str, *, markerfirst: bool = True):
    fig, ax = plt.subplots(num=1, clear=True, figsize=(4, 3), dpi=100)
    handles = [
        Line2D([], [], marker="o", color=f"C{index}", label=f"entry {index}")
        for index in range(5)
    ]
    kwargs = dict(
        handles=handles,
        labels=[handle.get_label() for handle in handles],
        title="Legend title",
        ncols=2,
        borderpad=0.4,
        labelspacing=0.5,
        handlelength=2.0,
        handletextpad=0.8,
        columnspacing=2.0,
        markerfirst=markerfirst,
        loc="upper center",
    )
    if owner == "current":
        legend = ax.legend(**kwargs)
    elif owner == "extra":
        legend = ax.legend(**kwargs)
        ax.add_artist(legend)
        ax.legend(handles=[Patch(label="current")], loc="lower right")
    elif owner == "figure":
        legend = fig.legend(**kwargs)
    else:  # pragma: no cover - protects parametrization mistakes
        raise AssertionError(owner)
    fig.canvas.draw()
    return fig, ax, legend


def _leaf_identity(legend):
    drawing_areas = tuple(legend._legend_handle_box.findobj(match=DrawingArea))
    text_areas = tuple(legend._legend_box.findobj(match=TextArea))
    artists = (
        legend,
        legend.get_frame(),
        *legend.legend_handles,
        *legend.get_texts(),
        legend.get_title(),
        *drawing_areas,
        *text_areas,
    )
    return tuple(id(artist) for artist in artists)


def _deep_handle_leaf_identity(legend):
    leaves = []
    for drawing_area in legend._legend_handle_box.findobj(match=DrawingArea):
        leaves.extend(
            drawing_area.findobj(match=lambda artist: not artist.get_children())
        )
    return tuple(id(artist) for artist in leaves)


def _install_tracker(fig) -> ChangeTracker:
    tracker = ChangeTracker.__new__(ChangeTracker)
    tracker.figure = fig
    tracker.changes = {}
    tracker.saved = True
    tracker.edits = []
    tracker.last_edit = -1
    tracker.update_changes_signal = None
    tracker.no_save = False
    fig.change_tracker = tracker
    return tracker


@pytest.mark.parametrize("owner", ["current", "extra", "figure"])
@pytest.mark.parametrize("markerfirst", [True, False])
def test_reflow_preserves_every_leaf_and_matches_standard_layout(
    owner, markerfirst
) -> None:
    fig, _ax, legend = _make_legend(owner, markerfirst=markerfirst)
    renderer = fig.canvas.get_renderer()
    identities = _leaf_identity(legend)
    old_root = legend._legend_box
    old_handle_box = legend._legend_handle_box
    legend.get_texts()[0].set_position((13.0, -7.0))
    legend.legend_handles[0].set_alpha(0.35)
    text_position = legend.get_texts()[0].get_position()

    destination = LegendLayoutSpec.from_legend(legend).with_updates(
        {
            "ncols": 3,
            "borderpad": 0.75,
            "labelspacing": 0.3,
            "handlelength": 1.4,
            "handletextpad": 0.25,
            "columnspacing": 0.6,
        }
    )
    plan = LegendLayoutPlan.preflight(legend, destination)
    with pytest.raises(FrozenInstanceError):
        plan.destination = LegendLayoutSpec.from_legend(legend)
    assert plan.apply()
    fig.canvas.draw()

    assert legend._legend_box is not old_root
    assert legend._legend_handle_box is not old_handle_box
    assert _leaf_identity(legend) == identities
    assert LegendLayoutSpec.from_legend(legend) == destination
    assert legend.get_texts()[0].get_position() == text_position
    assert legend.legend_handles[0].get_alpha() == 0.35
    assert np.all(np.isfinite(legend.get_window_extent(renderer).extents))
    plt.close(fig)


@pytest.mark.parametrize("markerfirst", [True, False])
def test_reflow_geometry_matches_a_native_control_legend(markerfirst) -> None:
    fig, _ax, legend = _make_legend("current", markerfirst=markerfirst)
    destination = {
        "ncols": 3,
        "borderpad": 0.75,
        "labelspacing": 0.3,
        "handlelength": 1.4,
        "handletextpad": 0.25,
        "columnspacing": 0.6,
    }
    assert reflow_legend_layout(legend, **destination)
    fig.canvas.draw()
    reflowed = legend.get_window_extent(fig.canvas.get_renderer()).bounds

    plt.close(fig)
    fig2, ax2 = plt.subplots(num=1, clear=True, figsize=(4, 3), dpi=100)
    handles = [
        Line2D([], [], marker="o", color=f"C{index}", label=f"entry {index}")
        for index in range(5)
    ]
    control = ax2.legend(
        handles=handles,
        labels=[handle.get_label() for handle in handles],
        title="Legend title",
        markerfirst=markerfirst,
        loc="upper center",
        **destination,
    )
    fig2.canvas.draw()
    native = control.get_window_extent(fig2.canvas.get_renderer()).bounds

    assert np.allclose(reflowed, native, atol=0.25, rtol=0.0)
    plt.close(fig2)


@pytest.mark.parametrize(
    "bbox_to_anchor",
    [(0.62, 0.91), (0.2, 0.3, 0.5, 0.4)],
)
def test_reflow_preserves_point_and_bbox_anchors(bbox_to_anchor) -> None:
    fig, ax = plt.subplots(num=1, clear=True, figsize=(4, 3), dpi=100)
    legend = ax.legend(
        handles=[Line2D([], [], label="A"), Line2D([], [], label="B")],
        bbox_to_anchor=bbox_to_anchor,
        loc="upper center",
    )
    fig.canvas.draw()
    anchor = legend.get_bbox_to_anchor()
    before = (anchor.bounds, anchor._transform)

    assert reflow_legend_layout(legend, ncols=2, borderpad=0.8)
    fig.canvas.draw()
    after = legend.get_bbox_to_anchor()

    assert after.bounds == before[0]
    assert after._transform is before[1]
    plt.close(fig)


@pytest.mark.parametrize("entry_count", [0, 2])
def test_reflow_supports_empty_legends_and_more_columns_than_entries(
    entry_count,
) -> None:
    fig, ax = plt.subplots(num=1, clear=True, figsize=(4, 3), dpi=100)
    handles = [Line2D([], [], label=f"entry {index}") for index in range(entry_count)]
    legend = ax.legend(handles=handles, title="Title only" if not handles else "")
    fig.canvas.draw()
    identity = _leaf_identity(legend)

    assert reflow_legend_layout(legend, ncols=8, borderpad=0.7)
    fig.canvas.draw()

    assert legend._ncols == 8
    assert len(legend._legend_handle_box.get_children()) == entry_count
    assert _leaf_identity(legend) == identity
    plt.close(fig)


@pytest.mark.parametrize("owner", ["current", "extra", "figure"])
def test_layout_only_generated_command_replays_in_place_for_every_owner(owner) -> None:
    plt.close("all")
    fig, _ax, legend = _make_legend(owner)
    tracker = _install_tracker(fig)
    adapter = get_artist_adapter(legend)
    before_identity = _leaf_identity(legend)

    assert adapter.reflow_layout(
        {
            "ncols": 3,
            "borderpad": 0.65,
            "labelspacing": 0.2,
            "handlelength": 1.6,
            "handletextpad": 0.3,
            "columnspacing": 0.7,
        }
    )
    fig.canvas.draw()
    expected_spec = LegendLayoutSpec.from_legend(legend)
    expected_bounds = legend.get_window_extent(fig.canvas.get_renderer()).bounds
    assert _leaf_identity(legend) == before_identity
    saved_lines = tracker.sorted_changes()
    assert len(saved_lines) == 1
    assert "._pylustrator_reflow_layout(" in saved_lines[0]
    assert ".legend(" not in saved_lines[0]

    plt.close(fig)
    fig2, _ax2, legend2 = _make_legend(owner)
    init_figure(fig2)
    replay_identity = _leaf_identity(legend2)
    exec(saved_lines[0], {"plt": plt})
    fig2.canvas.draw()

    assert _leaf_identity(legend2) == replay_identity
    assert LegendLayoutSpec.from_legend(legend2) == expected_spec
    assert np.allclose(
        legend2.get_window_extent(fig2.canvas.get_renderer()).bounds,
        expected_bounds,
        atol=1e-9,
    )
    plt.close(fig2)


def test_layout_state_undo_redo_restores_exact_packer_trees_and_recording() -> None:
    fig, _ax, legend = _make_legend("current")
    tracker = _install_tracker(fig)
    adapter = get_artist_adapter(legend)
    before = adapter.layout_state()
    leaf_identity = _leaf_identity(legend)

    plan = adapter.plan_layout_reflow({"ncols": 3, "borderpad": 0.8})
    assert adapter.apply_layout_reflow_plan(plan)
    after = adapter.layout_state()
    assert tracker.sorted_changes() == [
        "plt.figure(1).axes[0].get_legend()" + after.spec.replay_command()
    ]

    adapter.restore_layout_state(before)
    fig.canvas.draw()
    assert legend._legend_box is before.root
    assert legend._legend_handle_box is before.handle_box
    assert LegendLayoutSpec.from_legend(legend) == before.spec
    assert tracker.sorted_changes() == []
    assert _leaf_identity(legend) == leaf_identity

    adapter.restore_layout_state(after)
    fig.canvas.draw()
    assert legend._legend_box is after.root
    assert legend._legend_handle_box is after.handle_box
    assert LegendLayoutSpec.from_legend(legend) == after.spec
    assert _leaf_identity(legend) == leaf_identity
    plt.close(fig)


def test_semantic_transform_plan_reflows_multiple_legends_atomically() -> None:
    fig, axes = plt.subplots(1, 2, num=1, clear=True, figsize=(6, 3), dpi=100)
    legends = []
    for ax in axes:
        handles = [Line2D([], [], label=f"entry {index}") for index in range(4)]
        legends.append(ax.legend(handles=handles, ncols=2))
    fig.canvas.draw()
    tracker = _install_tracker(fig)
    identities = tuple(_leaf_identity(legend) for legend in legends)

    plan = TransformPlan.preflight(
        legends,
        TransformIntent.reflow_layout(
            {"ncols": 3, "borderpad": 0.7, "columnspacing": 0.6}
        ),
    )
    assert len(plan.legend_layout_plans) == 2
    with pytest.raises(ValueError, match="commit and draw"):
        plan.preview_control_points()
    with pytest.raises(ValueError, match="commit and draw"):
        plan.preview_selection_points()
    plan.commit()
    fig.canvas.draw()

    assert all(legend._ncols == 3 for legend in legends)
    assert all(legend.borderpad == 0.7 for legend in legends)
    assert tuple(_leaf_identity(legend) for legend in legends) == identities
    assert len(tracker.sorted_changes()) == 2
    plt.close(fig)


def test_semantic_transform_preflight_rejects_legend_descendant_selection() -> None:
    fig, _ax, legend = _make_legend("current")
    root = legend._legend_box

    with pytest.raises(TransformPreflightError) as caught:
        TransformPlan.preflight(
            [legend, legend.get_texts()[0]],
            TransformIntent.reflow_layout({"borderpad": 0.8}),
        )

    assert "Deselect Legend descendants" in str(caught.value)
    assert legend._legend_box is root
    plt.close(fig)


def test_semantic_transform_recording_failure_rolls_back_all_legends() -> None:
    class FailingSecondTracker:
        def __init__(self):
            self.changes = {}
            self.saved = True
            self.calls = 0

        def capture_recording_state(self):
            return dict(self.changes), self.saved

        def restore_recording_state(self, state):
            self.changes, self.saved = dict(state[0]), bool(state[1])

        def addNewLegendLayoutChange(self, target):
            self.calls += 1
            self.changes[(target, "layout")] = (target, ".layout()")
            self.saved = False
            if self.calls == 2:
                raise RuntimeError("simulated second layout recording failure")

    fig, axes = plt.subplots(1, 2, num=1, clear=True, figsize=(6, 3), dpi=100)
    legends = [ax.legend(handles=[Line2D([], [], label="entry")]) for ax in axes]
    fig.canvas.draw()
    tracker = FailingSecondTracker()
    fig.change_tracker = tracker
    adapters = [get_artist_adapter(legend) for legend in legends]
    states = [adapter.layout_state() for adapter in adapters]
    plan = TransformPlan.preflight(
        legends,
        TransformIntent.reflow_layout({"borderpad": 0.8}),
    )

    with pytest.raises(RuntimeError, match="simulated second layout recording failure"):
        plan.commit()

    for legend, state in zip(legends, states):
        assert legend._legend_box is state.root
        assert LegendLayoutSpec.from_legend(legend) == state.spec
    assert tracker.changes == {}
    assert tracker.saved
    plt.close(fig)


def test_reflow_rejects_expand_unknown_packer_and_parent_child_selection() -> None:
    fig, ax, legend = _make_legend("current")
    adapter = get_artist_adapter(legend)
    original = adapter.layout_state()
    text = legend.get_texts()[0]

    with pytest.raises(UnsupportedArtistError, match="Deselect Legend descendants"):
        adapter.plan_layout_reflow({"borderpad": 0.7}, selected_artists=[legend, text])
    assert legend._legend_box is original.root
    assert LegendLayoutSpec.from_legend(legend) == original.spec

    legend._mode = "expand"
    support = adapter.operation_support(TransformOperation.REFLOW_LAYOUT)
    assert not support.supported
    assert "expand" in support.reason
    legend._mode = None

    class CustomVPacker(VPacker):
        pass

    custom = CustomVPacker(
        pad=original.root.pad,
        sep=original.root.sep,
        align=original.root.align,
        mode="fixed",
        children=list(original.root.get_children()),
    )
    custom.set_offset(original.root._offset)
    custom.set_figure(fig)
    custom.axes = ax
    legend._legend_box = custom
    support = adapter.operation_support(TransformOperation.REFLOW_LAYOUT)
    assert not support.supported
    assert "unsupported packer" in support.reason
    plt.close(fig)


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"ncols": 0}, "at least 1"),
        ({"ncols": 1.5}, "integer"),
        ({"borderpad": -0.1}, "non-negative"),
        ({"labelspacing": float("nan")}, "finite"),
        ({"handlelength": float("inf")}, "finite"),
        ({"borderpad": "0.8"}, "numeric"),
        ({"unknown": 1}, "Unsupported"),
    ],
)
def test_invalid_destination_is_side_effect_free(updates, message) -> None:
    fig, _ax, legend = _make_legend("current")
    before = _leaf_identity(legend)
    root = legend._legend_box
    spec = LegendLayoutSpec.from_legend(legend)

    with pytest.raises(LegendLayoutError, match=message):
        LegendLayoutPlan.preflight(legend, updates)

    assert legend._legend_box is root
    assert _leaf_identity(legend) == before
    assert LegendLayoutSpec.from_legend(legend) == spec
    plt.close(fig)


def test_destination_is_canonicalized_before_it_enters_the_plan() -> None:
    fig, _ax, legend = _make_legend("current")
    plan = LegendLayoutPlan.preflight(
        legend,
        LegendLayoutSpec(
            np.int64(3),
            np.float64(0.7),
            np.float32(0.2),
            np.float64(1.4),
            np.float64(0.3),
            np.float64(0.6),
        ),
    )

    assert type(plan.destination.ncols) is int
    assert all(
        type(getattr(plan.destination, name)) is float
        for name in (
            "borderpad",
            "labelspacing",
            "handlelength",
            "handletextpad",
            "columnspacing",
        )
    )
    plt.close(fig)


def test_stale_plan_cannot_overwrite_a_newer_layout() -> None:
    fig, _ax, legend = _make_legend("current")
    stale = LegendLayoutPlan.preflight(legend, {"borderpad": 0.7})
    assert reflow_legend_layout(legend, ncols=3)
    newer_root = legend._legend_box
    newer_spec = LegendLayoutSpec.from_legend(legend)

    with pytest.raises(LegendLayoutError, match="changed after preflight"):
        stale.apply()

    assert legend._legend_box is newer_root
    assert LegendLayoutSpec.from_legend(legend) == newer_spec
    plt.close(fig)


def test_native_draggable_is_rebound_across_reflow_undo_and_redo() -> None:
    fig, _ax, legend = _make_legend("current")
    draggable = legend.set_draggable(True)
    adapter = get_artist_adapter(legend)
    before = adapter.layout_state()

    assert draggable.offsetbox is before.root
    assert adapter.reflow_layout({"ncols": 3}, record_changes=False)
    after = adapter.layout_state()
    assert draggable.offsetbox is legend._legend_box is after.root

    adapter.restore_layout_state(before, record_changes=False)
    assert draggable.offsetbox is legend._legend_box is before.root
    adapter.restore_layout_state(after, record_changes=False)
    assert draggable.offsetbox is legend._legend_box is after.root
    legend.set_draggable(False)
    plt.close(fig)


@pytest.mark.parametrize(
    "customize",
    [
        lambda box: box.set_visible(False),
        lambda box: box.set_gid("custom-row"),
        lambda box: box.set_picker(True),
        lambda box: box.add_callback(lambda _artist: None),
    ],
)
def test_reflow_rejects_replaceable_packers_with_custom_artist_state(
    customize,
) -> None:
    fig, _ax, legend = _make_legend("current")
    row = legend._legend_handle_box.get_children()[0].get_children()[0]
    customize(row)
    adapter = get_artist_adapter(legend)
    root = legend._legend_box

    support = adapter.operation_support(TransformOperation.REFLOW_LAYOUT)
    assert not support.supported
    assert "custom" in support.reason
    with pytest.raises(UnsupportedArtistError, match="custom"):
        adapter.plan_layout_reflow({"borderpad": 0.8})

    assert legend._legend_box is root
    plt.close(fig)


def test_layout_widget_history_failure_rolls_back_the_entire_transaction() -> None:
    from pylustrator.components.qitem_properties import LegendPropertiesWidget

    class SelectionRefresh:
        def __init__(self):
            self.calls = 0

        def refresh_selection_geometry(self):
            self.calls += 1

    class Signal:
        def __init__(self):
            self.calls = 0

        def emit(self):
            self.calls += 1

    fig, _ax, legend = _make_legend("current")
    tracker = _install_tracker(fig)
    tracker.changes[("existing", "change")] = ("existing", ".change()")
    tracker.saved = True
    tracker.edits = [[lambda: None, lambda: None, "Existing edit"]]
    tracker.last_edit = 0
    recording_before = tracker.capture_recording_state()
    edits_before = list(tracker.edits)
    root = legend._legend_box
    spec = LegendLayoutSpec.from_legend(legend)
    identity = _leaf_identity(legend)
    fig.figure_dragger = SelectionRefresh()
    fig.signals = type(
        "Signals", (), {"figure_selection_moved": Signal()}
    )()

    original_add_edit = tracker.addEdit

    def fail_after_inserting_history(edit):
        original_add_edit(edit)
        raise RuntimeError("simulated history failure")

    tracker.addEdit = fail_after_inserting_history
    widget = LegendPropertiesWidget.__new__(LegendPropertiesWidget)
    widget.target = legend
    widget.properties = {
        "ncol": legend._ncols,
        "borderpad": legend.borderpad,
        "labelspacing": legend.labelspacing,
        "handlelength": legend.handlelength,
        "handletextpad": legend.handletextpad,
        "columnspacing": legend.columnspacing,
    }

    with pytest.raises(RuntimeError, match="simulated history failure"):
        widget.changePropertiy("borderpad", 0.8)

    assert legend._legend_box is root
    assert LegendLayoutSpec.from_legend(legend) == spec
    assert _leaf_identity(legend) == identity
    assert tracker.capture_recording_state() == recording_before
    assert tracker.edits == edits_before
    assert tracker.last_edit == 0
    assert widget.target is legend
    assert widget.properties["borderpad"] == spec.borderpad
    assert fig.figure_dragger.calls == 2
    assert fig.signals.figure_selection_moved.calls == 2
    plt.close(fig)


def test_matplotlib_only_bootstrap_replays_without_importing_pylustrator() -> None:
    bootstrap = legend_layout_replay_bootstrap()
    source = f"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
fig, ax = plt.subplots()
legend = ax.legend(handles=[Line2D([], [], label='A'), Line2D([], [], label='B')])
fig.canvas.draw()
identities = tuple(map(id, (legend, legend.get_frame(), *legend.legend_handles, *legend.get_texts())))
{bootstrap}
legend._pylustrator_reflow_layout(ncols=2, borderpad=0.9, labelspacing=0.2, handlelength=1.4, handletextpad=0.3, columnspacing=0.7)
fig.canvas.draw()
assert identities == tuple(map(id, (legend, legend.get_frame(), *legend.legend_handles, *legend.get_texts())))
assert legend.borderpad == 0.9 and legend._ncols == 2
assert isinstance(legend._pylustrator_original_layout_spec, tuple)
"""
    result = subprocess.run(
        [sys.executable, "-c", source],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_saved_block_embeds_matplotlib_only_reflow_bootstrap(
    monkeypatch,
) -> None:
    fig, _ax, legend = _make_legend("current")
    tracker = _install_tracker(fig)
    tracker.get_reference_cached = {}
    adapter = get_artist_adapter(legend)
    assert adapter.reflow_layout({"ncols": 3, "borderpad": 0.9})
    saved = {}

    import pylustrator.change_tracker as change_tracker_module

    monkeypatch.setattr(change_tracker_module, "getTextFromFile", lambda *_args: [])
    monkeypatch.setattr(
        change_tracker_module, "stack_position", object(), raising=False
    )

    def capture_output(output, *_args):
        saved["source"] = "\n".join(output)

    monkeypatch.setattr(change_tracker_module, "insertTextToFile", capture_output)
    tracker.save()
    generated = saved["source"]

    assert "def _matplotlib_only_reflow(" in generated
    assert "from matplotlib.legend import Legend as _PylustratorLegend" in generated
    assert "import pylustrator" not in generated
    assert generated.index("def _matplotlib_only_reflow(") < generated.index(
        "._pylustrator_reflow_layout("
    )
    doctor_report = diagnose_generated_source(generated, filename="generated.py")
    assert not doctor_report.has_errors, doctor_report.diagnostics

    source = f"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
fig, ax = plt.subplots(num=1, clear=True, figsize=(4, 3), dpi=100)
handles = [Line2D([], [], marker='o', color=f'C{{index}}', label=f'entry {{index}}') for index in range(5)]
legend = ax.legend(handles=handles, labels=[handle.get_label() for handle in handles], title='Legend title', ncols=2, borderpad=0.4, labelspacing=0.5, handlelength=2.0, handletextpad=0.8, columnspacing=2.0, loc='upper center')
fig.canvas.draw()
identities = tuple(map(id, (legend, legend.get_frame(), *legend.legend_handles, *legend.get_texts())))
{generated}
fig.canvas.draw()
assert identities == tuple(map(id, (legend, legend.get_frame(), *legend.legend_handles, *legend.get_texts())))
assert legend._ncols == 3 and legend.borderpad == 0.9
"""
    result = subprocess.run(
        [sys.executable, "-c", source],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    plt.close(fig)


def test_replay_before_init_preserves_the_source_layout_baseline() -> None:
    fig, _ax, legend = _make_legend("current")
    source_spec = LegendLayoutSpec.from_legend(legend)

    _matplotlib_only_reflow(
        legend,
        ncols=3,
        borderpad=0.9,
        labelspacing=0.2,
        handlelength=1.4,
        handletextpad=0.3,
        columnspacing=0.7,
    )
    init_figure(fig)

    assert ensure_legend_layout_baseline(legend) == source_spec
    assert LegendLayoutSpec.from_legend(legend) != source_spec
    plt.close(fig)


def test_change_tracker_load_ignores_bootstrap_and_recovers_layout_command(
    monkeypatch,
) -> None:
    from pylustrator.commands import GENERATED_STATE_VERSION

    fig, _ax, legend = _make_legend("current")
    tracker = _install_tracker(fig)
    tracker.get_reference_cached = {}
    adapter = get_artist_adapter(legend)
    assert adapter.reflow_layout({"ncols": 3, "borderpad": 0.9})
    expected = tracker.sorted_changes()
    assert len(expected) == 1
    fig._pylustrator_generated_version = GENERATED_STATE_VERSION
    block = [
        "import matplotlib as mpl",
        "import numpy as np",
        *legend_layout_replay_bootstrap().splitlines(),
        f"plt.figure(1)._pylustrator_generated_version = {GENERATED_STATE_VERSION}",
        expected[0],
    ]

    import pylustrator.change_tracker as change_tracker_module

    monkeypatch.setattr(
        change_tracker_module,
        "getTextFromFile",
        lambda *_args: (block, 0),
    )
    monkeypatch.setattr(
        change_tracker_module, "stack_position", object(), raising=False
    )
    loaded = ChangeTracker.__new__(ChangeTracker)
    loaded.figure = fig
    loaded.changes = {}
    loaded.saved = True
    loaded.no_save = False
    loaded.changeCountChanged = lambda: None
    fig.change_tracker = loaded

    loaded.load()

    assert loaded.sorted_changes() == expected
    assert len(loaded.changes) == 1
    plt.close(fig)


def test_recording_failure_restores_exact_layout_and_tracker_state() -> None:
    class FailingTracker:
        def __init__(self):
            self.changes = {("existing", "change"): ("existing", ".change()")}
            self.saved = False

        def capture_recording_state(self):
            return dict(self.changes), self.saved

        def restore_recording_state(self, state):
            self.changes, self.saved = dict(state[0]), bool(state[1])

        def addNewLegendLayoutChange(self, _target):
            self.changes[("partial", "layout")] = ("partial", ".layout()")
            raise RuntimeError("simulated layout recording failure")

    fig, _ax, legend = _make_legend("current")
    tracker = FailingTracker()
    fig.change_tracker = tracker
    adapter = get_artist_adapter(legend)
    before = adapter.layout_state()
    identity = _leaf_identity(legend)
    tracker_before = tracker.capture_recording_state()
    plan = adapter.plan_layout_reflow({"ncols": 3, "borderpad": 0.9})

    with pytest.raises(RuntimeError, match="simulated layout recording failure"):
        adapter.apply_layout_reflow_plan(plan)

    assert legend._legend_box is before.root
    assert legend._legend_handle_box is before.handle_box
    assert LegendLayoutSpec.from_legend(legend) == before.spec
    assert _leaf_identity(legend) == identity
    assert tracker.capture_recording_state() == tracker_before
    plt.close(fig)


def test_explicit_composite_legend_can_reflow_without_full_reconstruction() -> None:
    fig, ax = plt.subplots(num=1, clear=True, figsize=(4, 3), dpi=100)
    errorbar = ax.errorbar([0, 1], [0, 1], yerr=[0.1, 0.2], label="_nolegend_")
    legend = ax.legend(handles=[errorbar], labels=["explicit errorbar"])
    fig.canvas.draw()
    tracker = _install_tracker(fig)
    adapter = get_artist_adapter(legend)
    identity = _leaf_identity(legend)
    deep_identity = _deep_handle_leaf_identity(legend)

    assert not adapter.operation_support(TransformOperation.SERIALIZE).supported
    assert adapter.operation_support(TransformOperation.REFLOW_LAYOUT).supported
    assert adapter.reflow_layout({"borderpad": 0.75, "handlelength": 1.5})
    fig.canvas.draw()

    assert _leaf_identity(legend) == identity
    assert _deep_handle_leaf_identity(legend) == deep_identity
    assert tracker.sorted_changes() == [
        "plt.figure(1).axes[0].get_legend()"
        + LegendLayoutSpec.from_legend(legend).replay_command()
    ]
    plt.close(fig)


@pytest.mark.parametrize("owner", ["current", "extra", "figure"])
def test_six_layout_properties_never_call_legend_reconstruction(
    owner, monkeypatch
) -> None:
    from pylustrator.components.qitem_properties import LegendPropertiesWidget

    fig, ax, legend = _make_legend(owner)
    tracker = _install_tracker(fig)
    current_sibling = ax.get_legend()
    identity = _leaf_identity(legend)
    widget = LegendPropertiesWidget.__new__(LegendPropertiesWidget)
    widget.target = legend
    widget.properties = {
        "ncols": legend._ncols,
        "borderpad": legend.borderpad,
        "labelspacing": legend.labelspacing,
        "handlelength": legend.handlelength,
        "handletextpad": legend.handletextpad,
        "columnspacing": legend.columnspacing,
    }

    def fail_reconstruction(*_args, **_kwargs):
        raise AssertionError("Legend reconstruction was called")

    monkeypatch.setattr(ax, "legend", fail_reconstruction)
    monkeypatch.setattr(fig, "legend", fail_reconstruction)
    for name, value in {
        "ncols": 3,
        "borderpad": 0.75,
        "labelspacing": 0.2,
        "handlelength": 1.4,
        "handletextpad": 0.3,
        "columnspacing": 0.6,
    }.items():
        widget.changePropertiy(name, value)
        assert widget.target is legend
        assert _leaf_identity(legend) == identity
        if owner == "extra":
            assert ax.get_legend() is current_sibling
            assert legend in ax.artists

    assert len(tracker.edits) == 6
    assert LegendLayoutSpec.from_legend(legend) == LegendLayoutSpec(
        3, 0.75, 0.2, 1.4, 0.3, 0.6
    )
    plt.close(fig)


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("ncols", 3),
        ("borderpad", 0.75),
        ("labelspacing", 0.2),
        ("handlelength", 1.4),
        ("handletextpad", 0.3),
        ("columnspacing", 0.6),
    ],
)
def test_real_qt_layout_controls_use_identity_preserving_path(
    name, value, monkeypatch
) -> None:
    from pylustrator.components.qitem_properties import LegendPropertiesWidget

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    fig, ax, legend = _make_legend("current")
    tracker = _install_tracker(fig)
    identity = _leaf_identity(legend)
    container = QtWidgets.QWidget()
    layout = QtWidgets.QVBoxLayout(container)
    widget = LegendPropertiesWidget(layout)
    widget.setTarget(legend)

    def fail_reconstruction(*_args, **_kwargs):
        raise AssertionError("Legend reconstruction was called")

    monkeypatch.setattr(ax, "legend", fail_reconstruction)
    monkeypatch.setattr(fig, "legend", fail_reconstruction)
    widget.widgets[name].setValue(value)

    normalized = "ncols" if name == "ncol" else name
    assert getattr(LegendLayoutSpec.from_legend(legend), normalized) == value
    assert _leaf_identity(legend) == identity
    assert len(tracker.edits) == 1
    container.deleteLater()
    plt.close(fig)
    assert app is not None


def test_qt_rejection_warns_without_escaping_event_loop_or_mutating(
    monkeypatch,
) -> None:
    from pylustrator.components.qitem_properties import LegendPropertiesWidget
    from pylustrator.snap import TargetWrapper

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    fig, _ax, legend = _make_legend("current")
    tracker = _install_tracker(fig)
    text = legend.get_texts()[0]
    fig.selection = type(
        "SelectedLegendAndChild",
        (),
        {"targets": [TargetWrapper(legend), TargetWrapper(text)]},
    )()
    root = legend._legend_box
    spec = LegendLayoutSpec.from_legend(legend)
    identity = _leaf_identity(legend)
    container = QtWidgets.QWidget()
    layout = QtWidgets.QVBoxLayout(container)
    widget = LegendPropertiesWidget(layout)
    widget.setTarget(legend)
    warnings = []
    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "warning",
        lambda *_args: warnings.append(_args[-1]),
    )

    widget.widgets["borderpad"].setValue(0.75)

    assert len(warnings) == 1
    assert "Deselect Legend descendants" in warnings[0]
    assert legend._legend_box is root
    assert LegendLayoutSpec.from_legend(legend) == spec
    assert _leaf_identity(legend) == identity
    assert tracker.edits == []
    container.deleteLater()
    plt.close(fig)
    assert app is not None
