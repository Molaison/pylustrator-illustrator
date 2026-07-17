"""Semantic property edits for Matplotlib-owned Artists.

The legacy property panel assumes that every displayed object owns writable
``get_*``/``set_*`` values.  Matplotlib tick labels violate that assumption:
their text is output produced by an Axis formatter, so ``Text.set_text`` is
silently overwritten on the next draw.  This module resolves such ownership
before mutation and commits one reversible, replayable semantic edit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
from matplotlib.axes import Axes
from matplotlib.text import Text

from .helper_functions import main_figure
from .replay import replay_literal


@dataclass(frozen=True)
class AxisTickLabelReference:
    """Stable semantic address of one currently rendered tick-label Text."""

    axes: Axes
    axis_name: str
    minor: bool
    index: int
    side: str

    @property
    def axis(self):
        return getattr(self.axes, f"{self.axis_name}axis")

    @property
    def group_key(self) -> tuple[Axes, str, bool]:
        return self.axes, self.axis_name, self.minor


@dataclass(frozen=True)
class AxisTickGroupState:
    """Exact locator/formatter state needed for atomic undo and redo."""

    axes: Axes
    axis_name: str
    minor: bool
    locator: object
    formatter: object
    limits: tuple[float, float]

    @classmethod
    def capture(cls, reference: AxisTickLabelReference):
        axis = reference.axis
        level = axis.minor if reference.minor else axis.major
        limits = getattr(reference.axes, f"get_{reference.axis_name}lim")()
        return cls(
            axes=reference.axes,
            axis_name=reference.axis_name,
            minor=reference.minor,
            locator=level.locator,
            formatter=level.formatter,
            limits=tuple(float(value) for value in limits),
        )

    def restore(self) -> None:
        axis = getattr(self.axes, f"{self.axis_name}axis")
        if self.minor:
            axis.set_minor_locator(self.locator)
            axis.set_minor_formatter(self.formatter)
        else:
            axis.set_major_locator(self.locator)
            axis.set_major_formatter(self.formatter)
        getattr(self.axes, f"set_{self.axis_name}lim")(self.limits)


def _axis_items(axes: Axes):
    seen = set()
    for name in ("x", "y", "z"):
        axis = getattr(axes, f"{name}axis", None)
        if axis is not None and id(axis) not in seen:
            seen.add(id(axis))
            yield name, axis


def axis_tick_label_reference(target: Text) -> AxisTickLabelReference | None:
    """Return the Axis-owned identity of *target*, or ``None``."""

    figure = getattr(target, "figure", None)
    if figure is None:
        return None
    target_axes = getattr(target, "axes", None)
    axes_candidates = []
    if isinstance(target_axes, Axes):
        axes_candidates.append(target_axes)
    axes_candidates.extend(
        axes for axes in figure.axes if axes is not target_axes
    )
    for axes in axes_candidates:
        for axis_name, axis in _axis_items(axes):
            for minor, ticks in (
                (False, getattr(axis, "majorTicks", ())),
                (True, getattr(axis, "minorTicks", ())),
            ):
                for index, tick in enumerate(ticks):
                    for side in ("label1", "label2"):
                        if target is getattr(tick, side, None):
                            return AxisTickLabelReference(
                                axes=axes,
                                axis_name=axis_name,
                                minor=minor,
                                index=index,
                                side=side,
                            )
    return None


def _unique_text_targets(primary: Text, selected: Iterable[object]) -> list[Text]:
    targets = []
    seen = set()
    for target in (primary, *selected):
        if not isinstance(target, Text) or id(target) in seen:
            continue
        seen.add(id(target))
        targets.append(target)
    return targets


def _tick_locations_and_objects(reference: AxisTickLabelReference):
    axis = reference.axis
    locations = np.asarray(
        axis.get_minorticklocs() if reference.minor else axis.get_majorticklocs(),
        dtype=float,
    )
    if locations.ndim != 1 or not np.all(np.isfinite(locations)):
        raise ValueError("Tick-label content requires finite one-dimensional ticks")
    ticks = (
        axis.get_minor_ticks(len(locations))
        if reference.minor
        else axis.get_major_ticks(len(locations))
    )
    return locations, ticks


def _displayed_tick_text(tick) -> str:
    label1 = tick.label1
    label2 = tick.label2
    if label1.get_visible() or not label2.get_visible():
        return label1.get_text()
    return label2.get_text()


def _apply_tick_group(
    reference: AxisTickLabelReference,
    updates: dict[int, str],
) -> tuple[list[float], list[str]]:
    locations, ticks = _tick_locations_and_objects(reference)
    if any(index >= len(ticks) for index in updates):
        raise RuntimeError("Tick inventory changed during property preflight")
    labels = [_displayed_tick_text(tick) for tick in ticks]
    for index, value in updates.items():
        labels[index] = value

    limits = tuple(
        float(value)
        for value in getattr(reference.axes, f"get_{reference.axis_name}lim")()
    )
    setter = getattr(reference.axes, f"set_{reference.axis_name}ticks")
    setter(locations.tolist(), labels, minor=reference.minor)
    # Matplotlib intentionally expands view limits in set_ticks.  Content
    # editing must not move or resize the selected artwork, so restore them.
    getattr(reference.axes, f"set_{reference.axis_name}lim")(limits)

    current_ticks = (
        reference.axis.get_minor_ticks(len(locations))
        if reference.minor
        else reference.axis.get_major_ticks(len(locations))
    )
    for index in updates:
        original = ticks[index]
        current = current_ticks[index]
        if getattr(original, reference.side) is not getattr(current, reference.side):
            raise RuntimeError("Tick-label identity changed during semantic edit")
    return [float(value) for value in locations], labels


def _restore_recording(tracker, state) -> None:
    restore = getattr(tracker, "restore_recording_state", None)
    if state is not None and restore is not None:
        restore(state)


def edit_text_content_if_axis_managed(
    primary: Text,
    value: str,
    selected_targets: Sequence[object] = (),
) -> bool:
    """Edit selected text atomically when any target is a tick label.

    Returns ``False`` when the ordinary Text path should handle the request.
    A tick label's content is materialized as an explicit Axis tick/label
    mapping; this is the only public Matplotlib representation that survives a
    draw and can be replayed without mutating a formatter's private state.
    """

    targets = _unique_text_targets(primary, selected_targets)
    references = {target: axis_tick_label_reference(target) for target in targets}
    if not any(reference is not None for reference in references.values()):
        return False

    value = str(value)
    changed_targets = [target for target in targets if target.get_text() != value]
    if not changed_targets:
        return True

    figure = main_figure(primary)
    tracker = figure.change_tracker
    capture = getattr(tracker, "capture_recording_state", None)
    recording_before = capture() if capture is not None else None
    edit_history_before = None
    if hasattr(tracker, "edits") and hasattr(tracker, "last_edit"):
        edit_history_before = (list(tracker.edits), int(tracker.last_edit))

    ordinary_before = {
        target: target.get_text()
        for target in changed_targets
        if references[target] is None
    }
    group_references: dict[
        tuple[Axes, str, bool], AxisTickLabelReference
    ] = {}
    group_updates: dict[tuple[Axes, str, bool], dict[int, str]] = {}
    for target in changed_targets:
        reference = references[target]
        if reference is None:
            continue
        group_references.setdefault(reference.group_key, reference)
        group_updates.setdefault(reference.group_key, {})[reference.index] = value
    states_before = {
        key: AxisTickGroupState.capture(reference)
        for key, reference in group_references.items()
    }

    def restore_geometry(states, ordinary_values) -> None:
        for state in states.values():
            state.restore()
        for target, text in ordinary_values.items():
            target.set_text(text)

    try:
        for target in ordinary_before:
            target.set_text(value)

        serialized_groups = {}
        for key, reference in group_references.items():
            serialized_groups[key] = _apply_tick_group(
                reference, group_updates[key]
            )

        # Newly materialized ticks normally reuse their Text instances.  If a
        # backend created one lazily, give it a property baseline immediately.
        from .change_tracker import add_text_default

        for reference in group_references.values():
            _locations, ticks = _tick_locations_and_objects(reference)
            for tick in ticks:
                add_text_default(tick.label1)
                add_text_default(tick.label2)

        for target in ordinary_before:
            tracker.addNewTextChange(target)
        from .change_tracker import getReference

        for key, reference in group_references.items():
            locations, labels = serialized_groups[key]
            minor = ", minor=True" if reference.minor else ""
            limits = states_before[key].limits
            command = (
                f".set_{reference.axis_name}ticks("
                f"{replay_literal(locations)}, {replay_literal(labels)}{minor}), "
                f"{getReference(reference.axes)}.set_{reference.axis_name}lim("
                f"{replay_literal(limits)})"
            )
            reference_command = f".set_{reference.axis_name}ticks"
            if reference.minor:
                reference_command += "_minor"
            tracker.addChange(
                reference.axes,
                command,
                reference.axes,
                reference_command,
            )

        states_after = {
            key: AxisTickGroupState.capture(reference)
            for key, reference in group_references.items()
        }
        ordinary_after = {target: target.get_text() for target in ordinary_before}
        recording_after = capture() if capture is not None else None

        def undo():
            restore_geometry(states_before, ordinary_before)
            _restore_recording(tracker, recording_before)

        def redo():
            restore_geometry(states_after, ordinary_after)
            _restore_recording(tracker, recording_after)

        figure.canvas.draw()
        signal = getattr(
            getattr(figure, "signals", None),
            "figure_selection_property_changed",
            None,
        )
        if signal is not None:
            signal.emit()
        tracker.addEdit([undo, redo, "Change tick label text"])
    except Exception:
        restore_geometry(states_before, ordinary_before)
        _restore_recording(tracker, recording_before)
        if edit_history_before is not None:
            tracker.edits, tracker.last_edit = edit_history_before
        figure.canvas.draw()
        raise
    return True


__all__ = [
    "AxisTickGroupState",
    "AxisTickLabelReference",
    "axis_tick_label_reference",
    "edit_text_content_if_axis_managed",
]
