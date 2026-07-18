"""Atomic transactions for property-panel edits.

Property controls used to discover support while mutating the selection: the
primary Artist was changed first and missing setters on later Artists were
silently ignored.  Besides producing mixed states, an exception in that loop
left generated source and Undo history out of sync with the canvas.

This module deliberately separates planning from execution.  A
``PropertyPlan`` resolves every semantic target and every getter/setter before
the first mutation.  Commit then treats Artist state, generated recording, and
Undo history as one transaction.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np
from matplotlib.artist import Artist
from matplotlib.axes import Axes
from matplotlib.text import Text

from .helper_functions import main_figure
from .replay import replay_literal


class PropertyPreflightError(ValueError):
    """Raised before mutation when a selection has no common property edit."""


_TEXT_STATE_PROPERTIES = frozenset(
    {
        "position",
        "text",
        "ha",
        "va",
        "fontsize",
        "color",
        "style",
        "weight",
        "fontname",
        "rotation",
        "visible",
    }
)
_AXES_STATE_PROPERTIES = frozenset(
    {
        "xlim",
        "ylim",
        "xlabel",
        "ylabel",
        "xticks",
        "yticks",
        "xticklabels",
        "yticklabels",
        "xscale",
        "yscale",
    }
)


def _copy_value(value):
    """Copy getter output without requiring every Matplotlib value to pickle."""

    try:
        return deepcopy(value)
    except Exception:
        copy_method = getattr(value, "copy", None)
        if callable(copy_method):
            try:
                return copy_method()
            except Exception:
                pass
        return value


def _values_equal(left, right) -> bool:
    if left is right:
        return True
    try:
        comparison = np.asarray(left) == np.asarray(right)
        return bool(np.all(comparison))
    except (TypeError, ValueError):
        try:
            return bool(left == right)
        except (TypeError, ValueError):
            return False


def _unwrap_target(candidate) -> Artist | None:
    candidate = getattr(candidate, "target", candidate)
    return candidate if isinstance(candidate, Artist) else None


def _unique_targets(candidates: Iterable[object]) -> tuple[Artist, ...]:
    result = []
    seen = set()
    for candidate in candidates:
        target = _unwrap_target(candidate)
        if target is None or id(target) in seen:
            continue
        seen.add(id(target))
        result.append(target)
    return tuple(result)


def _axis_label_kind(primary: Artist, targets: Sequence[Artist]) -> str | None:
    if not isinstance(primary, Text):
        return None
    for target in targets:
        if not isinstance(target, Axes):
            continue
        if primary is target.xaxis.get_label():
            return "x"
        if primary is target.yaxis.get_label():
            return "y"
    return None


def _semantic_targets(
    primary: Artist, selected_targets: Iterable[object]
) -> tuple[tuple[Artist, Artist], ...]:
    """Return ``(logical owner, writable target)`` pairs for a selection."""

    logical_targets = _unique_targets((primary, *selected_targets))
    label_kind = _axis_label_kind(primary, logical_targets)
    result = []
    seen = set()
    for logical_target in logical_targets:
        writable_target = logical_target
        if label_kind is not None and isinstance(logical_target, Axes):
            writable_target = getattr(logical_target, f"{label_kind}axis").get_label()
        if id(writable_target) in seen:
            continue
        seen.add(id(writable_target))
        result.append((logical_target, writable_target))
    return tuple(result)


def _history_state(tracker):
    if hasattr(tracker, "edits") and hasattr(tracker, "last_edit"):
        return list(tracker.edits), int(tracker.last_edit)
    return None


def _restore_history(tracker, state) -> None:
    if state is not None:
        tracker.edits, tracker.last_edit = list(state[0]), state[1]


def _recording_state(tracker):
    capture = getattr(tracker, "capture_recording_state", None)
    return capture() if callable(capture) else None


def _restore_recording(tracker, state) -> None:
    restore = getattr(tracker, "restore_recording_state", None)
    if state is not None and callable(restore):
        restore(state)


def _annotate_rollback_failures(error: Exception, failures) -> None:
    failures = tuple(failures)
    if not failures:
        return
    try:
        error.pylustrator_rollback_failures = failures
    except (AttributeError, TypeError):
        pass
    add_note = getattr(error, "add_note", None)
    if callable(add_note):
        details = "; ".join(
            f"{type(target).__name__}: {rollback_error}"
            for target, rollback_error in failures
        )
        add_note(f"Pylustrator rollback failures: {details}")


@dataclass(frozen=True)
class PropertyOperation:
    """One fully resolved property mutation in a larger atomic plan."""

    owner: Artist
    target: Artist
    property_name: str
    getter: Callable[[], object]
    setter: Callable[[object], object]
    requested_value: object
    command_factory: Callable[[object], str] | None = None

    @classmethod
    def for_setter(
        cls,
        owner: Artist,
        target: Artist,
        property_name: str,
        value,
    ) -> "PropertyOperation":
        getter_name = f"get_{property_name}"
        setter_name = f"set_{property_name}"
        getter = getattr(target, getter_name, None)
        setter = getattr(target, setter_name, None)
        if not callable(getter) or not callable(setter):
            missing = []
            if not callable(getter):
                missing.append(getter_name)
            if not callable(setter):
                missing.append(setter_name)
            detail = " and ".join(missing)
            raise PropertyPreflightError(
                f"{type(target).__name__} does not support {property_name!r} "
                f"({detail} is missing)"
            )

        # Getter execution and replay serialization are preflight too.  The
        # latter prevents a successful canvas mutation followed by generation
        # of an invalid or invented command.
        try:
            getter()
        except Exception as exc:
            raise PropertyPreflightError(
                f"Cannot read {property_name!r} from {type(target).__name__}: {exc}"
            ) from exc
        if not (
            isinstance(target, Text) and property_name in _TEXT_STATE_PROPERTIES
        ) and not (
            isinstance(target, Axes) and property_name in _AXES_STATE_PROPERTIES
        ):
            try:
                replay_literal(value)
            except (TypeError, ValueError) as exc:
                raise PropertyPreflightError(
                    f"{property_name!r} cannot be represented in generated source: {exc}"
                ) from exc

        return cls(
            owner=owner,
            target=target,
            property_name=property_name,
            getter=getter,
            setter=setter,
            requested_value=_copy_value(value),
        )

    def capture(self):
        return _copy_value(self.getter())

    def apply(self, value) -> None:
        self.setter(_copy_value(value))

    def generated_command(self, value) -> str:
        if self.command_factory is not None:
            return self.command_factory(value)
        return f".set_{self.property_name}({replay_literal(value)})"


class PropertyPlan:
    """A preflighted, atomic edit spanning one or more Artists/properties."""

    def __init__(self, figure, operations: Sequence[PropertyOperation]):
        operations = tuple(operations)
        if not operations:
            raise PropertyPreflightError("A property edit needs at least one Artist")
        tracker = getattr(figure, "change_tracker", None)
        if tracker is None:
            raise PropertyPreflightError(
                "Property editing requires an initialized change tracker"
            )
        self.figure = figure
        self.tracker = tracker
        self.operations = operations

    @classmethod
    def for_selection(
        cls,
        primary: Artist,
        selected_targets: Iterable[object],
        property_name: str,
        value,
    ) -> "PropertyPlan":
        return cls.for_selection_changes(
            primary,
            selected_targets,
            {property_name: value},
        )

    @classmethod
    def for_selection_changes(
        cls,
        primary: Artist,
        selected_targets: Iterable[object],
        changes: Mapping[str, object],
    ) -> "PropertyPlan":
        pairs = _semantic_targets(primary, selected_targets)
        return cls._from_pairs(pairs, changes)

    @classmethod
    def for_targets(
        cls,
        targets: Iterable[object],
        changes: Mapping[str, object],
    ) -> "PropertyPlan":
        targets = _unique_targets(targets)
        return cls._from_pairs(((target, target) for target in targets), changes)

    @classmethod
    def _from_pairs(
        cls,
        pairs: Iterable[tuple[Artist, Artist]],
        changes: Mapping[str, object],
    ) -> "PropertyPlan":
        pairs = tuple(pairs)
        if not pairs:
            raise PropertyPreflightError("A property edit needs at least one Artist")
        if not changes:
            raise PropertyPreflightError("A property edit needs at least one value")

        figure = main_figure(pairs[0][1])
        operations = []
        # This nested loop is intentionally completed in full before any
        # setter is called.  A mixed selection therefore either has a common
        # operation contract or is rejected without touching its first item.
        for owner, target in pairs:
            if main_figure(target) is not figure:
                raise PropertyPreflightError(
                    "A property transaction cannot span multiple figures"
                )
            for property_name, value in changes.items():
                operations.append(
                    PropertyOperation.for_setter(
                        owner, target, property_name, value
                    )
                )
        return cls(figure, operations)

    def _capture(self) -> tuple[object, ...]:
        return tuple(operation.capture() for operation in self.operations)

    def _apply_state(self, state: Sequence[object]) -> None:
        for operation, value in zip(self.operations, state):
            operation.apply(value)

    def _apply_state_atomically(
        self,
        destination: Sequence[object],
        destination_recording,
    ) -> None:
        """Restore one history side without exposing a half-applied closure.

        ``ChangeTracker`` advances its history pointer and redraws only after
        an Undo/Redo closure returns successfully.  Keep that contract: this
        helper changes neither the pointer nor the canvas.  If any setter (or
        recording restore) fails, every operation is returned to the state at
        closure entry and the original exception is re-raised with any
        rollback failures attached.
        """

        current = self._capture()
        current_recording = _recording_state(self.tracker)
        try:
            self._apply_state(destination)
            _restore_recording(self.tracker, destination_recording)
        except Exception as error:
            rollback_failures = []
            for operation, value in reversed(tuple(zip(self.operations, current))):
                try:
                    operation.apply(value)
                except Exception as rollback_error:
                    rollback_failures.append((operation.target, rollback_error))
            try:
                _restore_recording(self.tracker, current_recording)
            except Exception as rollback_error:
                rollback_failures.append((self.tracker, rollback_error))
            _annotate_rollback_failures(error, rollback_failures)
            raise

    def _record_changes(
        self, before: Sequence[object], after: Sequence[object]
    ) -> None:
        changed = [
            not _values_equal(old, new) for old, new in zip(before, after)
        ]
        recorded_text = set()
        recorded_axes = set()
        for operation, value, was_changed in zip(self.operations, after, changed):
            if not was_changed:
                continue
            target = operation.target
            if (
                isinstance(target, Text)
                and operation.property_name in _TEXT_STATE_PROPERTIES
            ):
                if id(target) not in recorded_text:
                    self.tracker.addNewTextChange(target)
                    recorded_text.add(id(target))
                continue
            if (
                isinstance(target, Axes)
                and operation.property_name in _AXES_STATE_PROPERTIES
            ):
                if id(target) not in recorded_axes:
                    self.tracker.addNewAxesChange(target)
                    recorded_axes.add(id(target))
                continue
            self.tracker.addChange(target, operation.generated_command(value))

    def _notify(self) -> None:
        signal = getattr(
            getattr(self.figure, "signals", None),
            "figure_selection_property_changed",
            None,
        )
        emit = getattr(signal, "emit", None)
        if callable(emit):
            emit()

    def commit(self, name: str = "Change property") -> bool:
        """Apply the plan once and install lossless Undo/Redo closures."""

        recording_before = _recording_state(self.tracker)
        history_before = _history_state(self.tracker)
        before = self._capture()
        requested = tuple(
            _copy_value(operation.requested_value) for operation in self.operations
        )
        if all(_values_equal(old, new) for old, new in zip(before, requested)):
            return False

        try:
            self._apply_state(requested)
            after = self._capture()
            if all(_values_equal(old, new) for old, new in zip(before, after)):
                return False
            self._record_changes(before, after)
            recording_after = _recording_state(self.tracker)

            def undo():
                self._apply_state_atomically(before, recording_before)

            def redo():
                self._apply_state_atomically(after, recording_after)

            self.figure.canvas.draw()
            self._notify()
            self.tracker.addEdit([undo, redo, name])
        except Exception as error:
            rollback_failures = []
            for operation, value in reversed(tuple(zip(self.operations, before))):
                try:
                    operation.apply(value)
                except Exception as rollback_error:
                    rollback_failures.append((operation.target, rollback_error))
            try:
                _restore_recording(self.tracker, recording_before)
            except Exception as rollback_error:
                rollback_failures.append((self.tracker, rollback_error))
            try:
                _restore_history(self.tracker, history_before)
            except Exception as rollback_error:
                rollback_failures.append((self.tracker, rollback_error))
            try:
                self.figure.canvas.draw()
            except Exception as rollback_error:
                rollback_failures.append((self.figure, rollback_error))
            _annotate_rollback_failures(error, rollback_failures)
            raise
        return True


__all__ = [
    "PropertyOperation",
    "PropertyPlan",
    "PropertyPreflightError",
]
