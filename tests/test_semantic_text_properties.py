from __future__ import annotations

from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from qtpy import QtWidgets

from pylustrator.QLinkableWidgets import TextWidget
from pylustrator.change_tracker import (
    ChangeTracker,
    UndoRedo,
    getReference,
    init_figure,
)
from pylustrator.property_adapters import (
    StaleTextContentPlanError,
    TextContentKind,
    TextContentPlan,
    TextContentPreflightError,
    edit_text_content_if_axis_managed,
    text_content_support,
)


class Signal:
    def __init__(self):
        self.callbacks = []
        self.emissions = 0

    def connect(self, callback):
        self.callbacks.append(callback)

    def emit(self, *args):
        self.emissions += 1
        for callback in list(self.callbacks):
            callback(*args)


def install_tracker(fig, *, selected=()):
    init_figure(fig)
    tracker = ChangeTracker.__new__(ChangeTracker)
    tracker.figure = fig
    tracker.changes = {}
    tracker.saved = True
    tracker.edits = []
    tracker.last_edit = -1
    tracker.update_changes_signal = None
    tracker.no_save = False
    fig.change_tracker = tracker
    fig.signals = SimpleNamespace(figure_selection_property_changed=Signal())
    fig.selection = SimpleNamespace(
        targets=[SimpleNamespace(target=target) for target in selected]
    )
    return tracker


def replay_changes(changes):
    namespace = {"mpl": matplotlib, "np": np, "plt": plt}
    for command_target, command in changes:
        exec(f"{getReference(command_target)}{command}", namespace)


def test_empty_axis_label_font_edit_has_lossless_undo_redo_and_replay() -> None:
    fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
    label = ax.yaxis.get_label()
    label.set_fontsize(7.6)
    fig.canvas.draw()
    tracker = install_tracker(fig, selected=(label,))

    try:
        assert label.get_text() == ""
        with UndoRedo([label], "Change font size"):
            label.set_fontsize(9.6)

        assert label.get_fontsize() == pytest.approx(9.6)
        generated = list(tracker.changes.values())
        assert len(generated) == 1
        assert generated[0][0] is label
        assert "fontsize=9.6" in generated[0][1]

        tracker.backEdit()
        assert label.get_text() == ""
        assert label.get_fontsize() == pytest.approx(7.6)
        tracker.forwardEdit()
        assert label.get_fontsize() == pytest.approx(9.6)

        tracker.backEdit()
        replay_changes(generated)
        fig.canvas.draw()
        assert label.get_text() == ""
        assert label.get_fontsize() == pytest.approx(9.6)
    finally:
        plt.close(fig)


def test_init_figure_registers_matplotlib_managed_text_defaults() -> None:
    fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
    ax.set_yticks([0, 1], labels=["first", "second"])
    ax.tick_params(top=True, labeltop=True)
    fig.canvas.draw()
    tick = ax.yaxis.get_major_ticks()[0]

    try:
        init_figure(fig)
        for text in (
            tick.label1,
            tick.label2,
            ax.xaxis.get_offset_text(),
            ax.yaxis.get_offset_text(),
            ax.xaxis.get_label(),
            ax.yaxis.get_label(),
        ):
            assert hasattr(text, "_pylustrator_old_args")
            assert "fontsize" in text._pylustrator_old_args
    finally:
        plt.close(fig)


def test_tick_label_font_edit_survives_draw_undo_redo_and_replay() -> None:
    fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
    ax.set_yticks([0, 1], labels=["first", "second"])
    fig.canvas.draw()
    label = ax.yaxis.get_major_ticks()[0].label1
    tracker = install_tracker(fig, selected=(label,))
    original_size = label.get_fontsize()

    try:
        with UndoRedo([label], "Change font size"):
            label.set_fontsize(original_size + 3)
        fig.canvas.draw()

        assert label.get_fontsize() == pytest.approx(original_size + 3)
        generated = list(tracker.changes.values())
        assert len(generated) == 1
        assert generated[0][0] is label
        assert "fontsize=" in generated[0][1]

        tracker.backEdit()
        assert label.get_fontsize() == pytest.approx(original_size)
        tracker.forwardEdit()
        assert label.get_fontsize() == pytest.approx(original_size + 3)

        tracker.backEdit()
        replay_changes(generated)
        fig.canvas.draw()
        assert label.get_fontsize() == pytest.approx(original_size + 3)
    finally:
        plt.close(fig)


def test_tick_label_text_widget_edits_formatter_atomically_and_replays() -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
    ax.set_ylim(-0.25, 1.25)
    ax.set_yticks([0, 1], labels=["first", "second"])
    fig.canvas.draw()
    tick = ax.yaxis.get_major_ticks()[0]
    label = tick.label1
    other = ax.yaxis.get_major_ticks()[1].label1
    tracker = install_tracker(fig, selected=(label,))
    locator_before = ax.yaxis.major.locator
    formatter_before = ax.yaxis.major.formatter
    limits_before = ax.get_ylim()
    target_signal = Signal()
    parent = QtWidgets.QWidget()
    layout = QtWidgets.QVBoxLayout(parent)
    widget = TextWidget(layout, "Text:")
    widget.link("text", target_signal)
    target_signal.emit(label)

    try:
        widget.setText("edited\nlabel", signal=True)
        assert tick.label1 is label
        assert label.get_text() == "edited\nlabel"
        assert other.get_text() == "second"
        assert ax.get_ylim() == pytest.approx(limits_before)
        fig.canvas.draw()
        assert label.get_text() == "edited\nlabel"
        assert len(tracker.edits) == 1

        generated = list(tracker.changes.values())
        assert len(generated) == 1
        assert generated[0][0] is ax
        assert ".set_yticks(" in generated[0][1]
        assert "edited\\nlabel" in generated[0][1]

        tracker.backEdit()
        assert ax.yaxis.major.locator is locator_before
        assert ax.yaxis.major.formatter is formatter_before
        assert label.get_text() == "first"
        assert ax.get_ylim() == pytest.approx(limits_before)

        tracker.forwardEdit()
        assert tick.label1 is label
        assert label.get_text() == "edited\nlabel"
        assert other.get_text() == "second"
        assert ax.get_ylim() == pytest.approx(limits_before)

        tracker.backEdit()
        replay_changes(generated)
        fig.canvas.draw()
        assert tick.label1 is label
        assert label.get_text() == "edited\nlabel"
        assert other.get_text() == "second"
        assert ax.get_ylim() == pytest.approx(limits_before)
    finally:
        parent.close()
        plt.close(fig)
    assert app is not None


@pytest.mark.parametrize("axis_name", ["x", "y"])
@pytest.mark.parametrize("minor", [False, True])
@pytest.mark.parametrize("side", ["label1", "label2"])
def test_tick_label_content_covers_axis_level_and_secondary_side(
    axis_name: str,
    minor: bool,
    side: str,
) -> None:
    fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
    ax.tick_params(
        top=True,
        labeltop=True,
        right=True,
        labelright=True,
        which="both",
    )
    setter = getattr(ax, f"set_{axis_name}ticks")
    setter([0.25, 0.75], ["first", "second"], minor=minor)
    fig.canvas.draw()
    axis = getattr(ax, f"{axis_name}axis")
    ticks = axis.get_minor_ticks() if minor else axis.get_major_ticks()
    label = getattr(ticks[0], side)
    tracker = install_tracker(fig, selected=(label,))
    limits_before = getattr(ax, f"get_{axis_name}lim")()

    try:
        assert edit_text_content_if_axis_managed(label, "edited", (label,))
        fig.canvas.draw()
        assert ticks[0].label1.get_text() == "edited"
        assert ticks[0].label2.get_text() == "edited"
        assert ticks[1].label1.get_text() == "second"
        assert getattr(ax, f"get_{axis_name}lim")() == pytest.approx(limits_before)
        assert len(tracker.edits) == 1

        tracker.backEdit()
        assert ticks[0].label1.get_text() == "first"
        tracker.forwardEdit()
        assert ticks[0].label2.get_text() == "edited"
    finally:
        plt.close(fig)


def test_tick_label_content_mixed_selection_is_one_atomic_edit() -> None:
    fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
    ax.set_yticks([0, 1], labels=["first", "second"])
    ordinary = ax.text(0.5, 0.5, "ordinary", transform=ax.transAxes)
    fig.canvas.draw()
    tick_label = ax.yaxis.get_major_ticks()[0].label1
    tracker = install_tracker(fig, selected=(tick_label, ordinary))

    try:
        assert edit_text_content_if_axis_managed(
            tick_label,
            "shared",
            (tick_label, ordinary),
        )
        fig.canvas.draw()
        assert tick_label.get_text() == "shared"
        assert ordinary.get_text() == "shared"
        assert len(tracker.edits) == 1
        assert len(tracker.changes) == 2

        tracker.backEdit()
        assert tick_label.get_text() == "first"
        assert ordinary.get_text() == "ordinary"
        tracker.forwardEdit()
        assert tick_label.get_text() == "shared"
        assert ordinary.get_text() == "shared"
    finally:
        plt.close(fig)


def test_tick_label_content_failure_rolls_back_geometry_recording_and_history() -> None:
    fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
    ax.set_yticks([0, 1], labels=["first", "second"])
    fig.canvas.draw()
    label = ax.yaxis.get_major_ticks()[0].label1
    tracker = install_tracker(fig, selected=(label,))
    locator_before = ax.yaxis.major.locator
    formatter_before = ax.yaxis.major.formatter
    recording_before = tracker.capture_recording_state()
    add_change = tracker.addChange

    def fail_after_recording(*args, **kwargs):
        add_change(*args, **kwargs)
        raise RuntimeError("injected recording failure")

    tracker.addChange = fail_after_recording
    try:
        with pytest.raises(RuntimeError, match="injected recording failure"):
            edit_text_content_if_axis_managed(label, "edited", (label,))
        assert label.get_text() == "first"
        assert ax.yaxis.major.locator is locator_before
        assert ax.yaxis.major.formatter is formatter_before
        assert tracker.capture_recording_state() == recording_before
        assert tracker.edits == []
        assert tracker.last_edit == -1
    finally:
        plt.close(fig)


def test_tick_label_content_noop_does_not_dirty_document() -> None:
    fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
    ax.set_yticks([0], labels=["same"])
    fig.canvas.draw()
    label = ax.yaxis.get_major_ticks()[0].label1
    tracker = install_tracker(fig, selected=(label,))

    try:
        assert edit_text_content_if_axis_managed(label, "same", (label,))
        assert tracker.changes == {}
        assert tracker.edits == []
        assert tracker.saved
    finally:
        plt.close(fig)


@pytest.mark.parametrize(
    "target_name",
    [
        "axes text",
        "annotation",
        "axes title",
        "axis label",
        "figure title",
        "legend entry",
        "legend title",
    ],
)
def test_inline_text_plan_supports_replayable_text_owners(
    target_name: str,
) -> None:
    fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
    line = ax.plot([0, 1], label="legend entry")[0]
    legend = ax.legend(handles=[line], title="legend title")
    targets = {
        "axes text": ax.text(0.25, 0.5, "axes text"),
        "annotation": ax.annotate("annotation", (0.5, 0.5)),
        "axes title": ax.set_title("axes title"),
        "axis label": ax.set_ylabel("axis label"),
        "figure title": fig.suptitle("figure title"),
        "legend entry": legend.get_texts()[0],
        "legend title": legend.get_title(),
    }
    target = targets[target_name]
    fig.canvas.draw()
    tracker = install_tracker(fig, selected=(target,))
    edited = f"edited {target_name}"

    try:
        support = text_content_support(target)
        assert support.supported
        assert support.kind is TextContentKind.ORDINARY

        plan = TextContentPlan.preflight(target)
        assert plan.kind is TextContentKind.ORDINARY
        assert plan.commit(edited)
        fig.canvas.draw()

        assert target.get_text() == edited
        assert len(tracker.edits) == 1
        tracker.backEdit()
        assert target.get_text() == target_name
        tracker.forwardEdit()
        assert target.get_text() == edited
    finally:
        plt.close(fig)


def test_inline_ordinary_text_commit_draw_undo_redo_and_replay() -> None:
    fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
    target = ax.text(0.25, 0.5, "before")
    fig.canvas.draw()
    tracker = install_tracker(fig, selected=(target,))

    try:
        plan = TextContentPlan.preflight(target)
        assert plan.commit("after\nline")
        assert target.get_text() == "after\nline"
        assert fig.signals.figure_selection_property_changed.emissions == 1

        generated = list(tracker.changes.values())
        assert len(generated) == 1
        assert generated[0][0] is target
        assert "after\\nline" in generated[0][1]

        tracker.backEdit()
        assert target.get_text() == "before"
        tracker.forwardEdit()
        assert target.get_text() == "after\nline"

        tracker.backEdit()
        replay_changes(generated)
        fig.canvas.draw()
        assert target.get_text() == "after\nline"
    finally:
        plt.close(fig)


def test_inline_tick_plan_preserves_identity_limits_and_replays() -> None:
    fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
    ax.set_ylim(-0.25, 1.25)
    ax.set_yticks([0, 1], labels=["first", "second"])
    fig.canvas.draw()
    ticks = ax.yaxis.get_major_ticks()
    target = ticks[0].label1
    other = ticks[1].label1
    tracker = install_tracker(fig, selected=(target,))
    locator_before = ax.yaxis.major.locator
    formatter_before = ax.yaxis.major.formatter
    limits_before = ax.get_ylim()

    try:
        plan = TextContentPlan.preflight(target)
        assert plan.kind is TextContentKind.AXIS_TICK_LABEL
        assert plan.commit("edited")
        fig.canvas.draw()

        assert ticks[0].label1 is target
        assert target.get_text() == "edited"
        assert other.get_text() == "second"
        assert ax.get_ylim() == pytest.approx(limits_before)
        assert len(tracker.edits) == 1
        generated = list(tracker.changes.values())
        assert len(generated) == 1
        assert generated[0][0] is ax
        assert ".set_yticks(" in generated[0][1]

        tracker.backEdit()
        assert ax.yaxis.major.locator is locator_before
        assert ax.yaxis.major.formatter is formatter_before
        assert target.get_text() == "first"
        assert ax.get_ylim() == pytest.approx(limits_before)
        tracker.forwardEdit()
        assert ticks[0].label1 is target
        assert target.get_text() == "edited"

        tracker.backEdit()
        replay_changes(generated)
        fig.canvas.draw()
        assert ticks[0].label1 is target
        assert target.get_text() == "edited"
        assert other.get_text() == "second"
        assert ax.get_ylim() == pytest.approx(limits_before)
    finally:
        plt.close(fig)


def test_inline_tick_plan_rejects_reordered_inventory_without_mutation() -> None:
    fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
    ax.set_yticks([0, 1], labels=["first", "second"])
    fig.canvas.draw()
    target = ax.yaxis.majorTicks[0].label1
    tracker = install_tracker(fig, selected=(target,))
    plan = TextContentPlan.preflight(target)
    ax.yaxis.majorTicks[0], ax.yaxis.majorTicks[1] = (
        ax.yaxis.majorTicks[1],
        ax.yaxis.majorTicks[0],
    )

    try:
        with pytest.raises(StaleTextContentPlanError, match="ownership changed"):
            plan.commit("must not apply")
        assert target.get_text() == "first"
        assert tracker.changes == {}
        assert tracker.edits == []
        assert tracker.last_edit == -1
    finally:
        plt.close(fig)


def test_inline_ordinary_plan_rejects_changed_replay_address() -> None:
    fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
    target = ax.text(0.25, 0.5, "first")
    ax.text(0.75, 0.5, "second")
    fig.canvas.draw()
    tracker = install_tracker(fig, selected=(target,))
    plan = TextContentPlan.preflight(target)
    target.remove()
    ax.add_artist(target)

    try:
        with pytest.raises(
            StaleTextContentPlanError,
            match="generated-source address changed",
        ):
            plan.commit("must not apply")
        assert target.get_text() == "first"
        assert tracker.changes == {}
        assert tracker.edits == []
    finally:
        plt.close(fig)


def test_inline_axis_offset_text_is_a_typed_preflight_denial() -> None:
    fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
    ax.plot([1_000_000, 1_000_001], [0, 1])
    fig.canvas.draw()
    target = ax.xaxis.get_offset_text()
    tracker = install_tracker(fig, selected=(target,))

    try:
        support = text_content_support(target)
        assert not support.supported
        assert support.kind is TextContentKind.AXIS_OFFSET_TEXT
        assert "formatter output" in support.reason
        with pytest.raises(TextContentPreflightError) as raised:
            TextContentPlan.preflight(target)
        assert raised.value.support.kind is TextContentKind.AXIS_OFFSET_TEXT
        assert tracker.changes == {}
        assert tracker.edits == []
        assert tracker.last_edit == -1
    finally:
        plt.close(fig)


def test_inline_secondary_axis_managed_text_is_explicitly_denied() -> None:
    fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
    secondary = ax.secondary_xaxis("top")
    secondary.set_xlabel("secondary")
    fig.canvas.draw()
    tracker = install_tracker(fig)
    targets = (
        secondary.xaxis.get_label(),
        secondary.xaxis.get_major_ticks()[0].label1,
    )

    try:
        for target in targets:
            support = text_content_support(target)
            assert not support.supported
            assert "secondary, inset, or z-axis" in support.reason
            with pytest.raises(TextContentPreflightError):
                TextContentPlan.preflight(target)
        assert tracker.changes == {}
        assert tracker.edits == []
    finally:
        plt.close(fig)


def test_inline_z_axis_managed_text_is_explicitly_denied() -> None:
    fig = plt.figure(figsize=(4, 3), dpi=100)
    ax = fig.add_subplot(projection="3d")
    ax.set_zlabel("depth")
    fig.canvas.draw()
    tracker = install_tracker(fig)
    targets = (ax.zaxis.get_label(), ax.zaxis.get_major_ticks()[0].label1)

    try:
        for target in targets:
            support = text_content_support(target)
            assert not support.supported
            with pytest.raises(TextContentPreflightError):
                TextContentPlan.preflight(target)
        assert tracker.changes == {}
        assert tracker.edits == []
    finally:
        plt.close(fig)


def test_inline_unregistered_text_is_denied_without_side_effects() -> None:
    fig, _ax = plt.subplots(figsize=(4, 3), dpi=100)
    target = matplotlib.text.Text(0, 0, "unregistered")
    target.set_figure(fig)
    tracker = install_tracker(fig, selected=(target,))
    recording_before = tracker.capture_recording_state()

    try:
        support = text_content_support(target)
        assert not support.supported
        assert support.kind is TextContentKind.ORDINARY
        assert "stable generated-source reference" in support.reason
        with pytest.raises(TextContentPreflightError):
            TextContentPlan.preflight(target)
        assert tracker.capture_recording_state() == recording_before
        assert tracker.edits == []
        assert tracker.last_edit == -1
    finally:
        plt.close(fig)


def test_inline_text_noop_does_not_draw_record_or_create_history() -> None:
    fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
    target = ax.text(0.5, 0.5, "same")
    fig.canvas.draw()
    tracker = install_tracker(fig, selected=(target,))
    original_draw = fig.canvas.draw
    draw_calls = 0

    def counted_draw(*args, **kwargs):
        nonlocal draw_calls
        draw_calls += 1
        return original_draw(*args, **kwargs)

    fig.canvas.draw = counted_draw
    try:
        assert not TextContentPlan.preflight(target).commit("same")
        assert draw_calls == 0
        assert tracker.changes == {}
        assert tracker.edits == []
        assert tracker.saved
    finally:
        fig.canvas.draw = original_draw
        plt.close(fig)


def test_inline_tick_text_noop_returns_false_without_recording() -> None:
    fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
    ax.set_yticks([0], labels=["same"])
    fig.canvas.draw()
    target = ax.yaxis.get_major_ticks()[0].label1
    tracker = install_tracker(fig, selected=(target,))

    try:
        assert not TextContentPlan.preflight(target).commit("same")
        assert tracker.changes == {}
        assert tracker.edits == []
        assert tracker.saved
    finally:
        plt.close(fig)


def test_inline_text_detects_external_content_drift_before_commit() -> None:
    fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
    target = ax.text(0.5, 0.5, "initial")
    fig.canvas.draw()
    tracker = install_tracker(fig, selected=(target,))
    plan = TextContentPlan.preflight(target)
    target.set_text("external")

    try:
        with pytest.raises(StaleTextContentPlanError, match="content changed"):
            plan.commit("inline")
        assert target.get_text() == "external"
        assert tracker.changes == {}
        assert tracker.edits == []
    finally:
        plt.close(fig)


def test_inline_text_recording_failure_rolls_back_state_and_history() -> None:
    fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
    target = ax.text(0.5, 0.5, "before")
    fig.canvas.draw()
    tracker = install_tracker(fig, selected=(target,))
    plan = TextContentPlan.preflight(target)
    recording_before = tracker.capture_recording_state()
    original_record = tracker.addNewTextChange

    def fail_after_recording(text):
        original_record(text)
        raise RuntimeError("injected inline recording failure")

    tracker.addNewTextChange = fail_after_recording
    try:
        with pytest.raises(RuntimeError, match="inline recording failure"):
            plan.commit("after")
        assert target.get_text() == "before"
        assert tracker.capture_recording_state() == recording_before
        assert tracker.edits == []
        assert tracker.last_edit == -1
    finally:
        tracker.addNewTextChange = original_record
        plt.close(fig)
