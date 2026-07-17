"""Atomic preflight and execution of semantic transform intents."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from matplotlib.artist import Artist

from .artist_adapters import (
    AppearanceScalePlan,
    ArtistAdapter,
    RigidRotationPlan,
    get_artist_adapter,
    suspend_change_recording,
)
from .operations import OperationSupport, TransformIntent, TransformOperation


class TransformPreflightError(ValueError):
    def __init__(self, failures: Iterable[tuple[Artist, OperationSupport]]):
        self.failures = tuple(failures)
        details = ", ".join(
            f"{type(artist).__name__}: {support.reason}"
            for artist, support in self.failures
        )
        super().__init__(details or "Transform is not supported")


@dataclass(frozen=True)
class TransformPlan:
    intent: TransformIntent
    adapters: tuple[ArtistAdapter, ...]
    rigid_rotation_plans: tuple[RigidRotationPlan, ...] = ()
    appearance_scale_plans: tuple[AppearanceScalePlan, ...] = ()

    @classmethod
    def preflight(
        cls, targets: Iterable[Artist], intent: TransformIntent
    ) -> "TransformPlan":
        adapters = tuple(get_artist_adapter(target) for target in targets)
        failures = []
        rigid_rotation_plans = []
        appearance_scale_plans = []
        for adapter in adapters:
            support = adapter.operation_support(intent.operation)
            if not support.supported:
                failures.append((adapter.target, support))
                continue
            try:
                if intent.operation is TransformOperation.TRANSLATE:
                    adapter.preflight_translation(intent.delta)
                elif intent.operation is TransformOperation.RESIZE_GEOMETRY:
                    adapter.preflight_resize(
                        np.asarray(intent.matrix, dtype=float)
                    )
                elif intent.operation is TransformOperation.RIGID_ROTATE:
                    rigid_rotation_plans.append(
                        adapter.plan_rigid_rotation(
                            float(intent.angle_degrees), intent.pivot
                        )
                    )
                elif intent.operation is TransformOperation.SCALE_APPEARANCE:
                    appearance_scale_plans.append(
                        adapter._plan_preflighted_appearance_scale(
                            float(intent.factor)
                        )
                    )
            except (TypeError, ValueError) as error:
                failures.append(
                    (
                        adapter.target,
                        OperationSupport.denied(intent.operation, str(error)),
                    )
                )
        if failures:
            raise TransformPreflightError(failures)
        return cls(
            intent,
            adapters,
            tuple(rigid_rotation_plans),
            tuple(appearance_scale_plans),
        )

    def preview_control_points(self) -> tuple[np.ndarray, ...]:
        operation = self.intent.operation
        if operation is TransformOperation.TRANSLATE:
            delta = np.asarray(self.intent.delta, dtype=float)
            return tuple(
                adapter.point_array(adapter.control_points()) + delta
                for adapter in self.adapters
            )
        if operation is TransformOperation.RESIZE_GEOMETRY:
            matrix = np.asarray(self.intent.matrix, dtype=float)
            return tuple(
                adapter.preview_resize_control_points(matrix)
                for adapter in self.adapters
            )
        if operation is TransformOperation.RIGID_ROTATE:
            return tuple(
                plan.control_array() for plan in self.rigid_rotation_plans
            )
        return tuple(
            np.asarray(adapter.control_points(), dtype=float)
            for adapter in self.adapters
        )

    def preview_selection_points(self) -> tuple[np.ndarray, ...]:
        """Return the visible destination carried by the immutable plan."""

        operation = self.intent.operation
        if operation is TransformOperation.TRANSLATE:
            delta = np.asarray(self.intent.delta, dtype=float)
            return tuple(
                adapter.preview_translation_selection_points(delta)
                for adapter in self.adapters
            )
        if operation is TransformOperation.RESIZE_GEOMETRY:
            matrix = np.asarray(self.intent.matrix, dtype=float)
            return tuple(adapter.preflight_resize(matrix) for adapter in self.adapters)
        if operation is TransformOperation.RIGID_ROTATE:
            return tuple(
                plan.selection_array() for plan in self.rigid_rotation_plans
            )
        if operation is TransformOperation.SCALE_APPEARANCE:
            return tuple(
                plan.selection_array() for plan in self.appearance_scale_plans
            )
        return tuple(
            np.asarray(adapter.selection_points(), dtype=float)
            for adapter in self.adapters
        )

    def commit(self) -> None:
        appearance_only = (
            self.intent.operation is TransformOperation.SCALE_APPEARANCE
        )
        snapshots = [
            adapter.appearance_state() if appearance_only else adapter.snapshot()
            for adapter in self.adapters
        ]
        tracker_states = []
        seen_trackers = set()
        for adapter in self.adapters:
            try:
                tracker = adapter.change_tracker()
            except AttributeError:
                continue
            if id(tracker) in seen_trackers:
                continue
            capture = getattr(tracker, "capture_recording_state", None)
            restore = getattr(tracker, "restore_recording_state", None)
            if not callable(capture) or not callable(restore):
                continue
            seen_trackers.add(id(tracker))
            tracker_states.append((tracker, capture()))
        try:
            for index, adapter in enumerate(self.adapters):
                if self.intent.operation is TransformOperation.TRANSLATE:
                    adapter.translate(self.intent.delta)
                elif self.intent.operation is TransformOperation.RESIZE_GEOMETRY:
                    adapter.resize(np.asarray(self.intent.matrix, dtype=float))
                elif self.intent.operation is TransformOperation.ROTATE:
                    adapter.set_rotation(
                        adapter.rotation() + float(self.intent.angle_degrees)
                    )
                elif self.intent.operation is TransformOperation.RIGID_ROTATE:
                    adapter.apply_rigid_rotation_plan(
                        self.rigid_rotation_plans[index]
                    )
                elif self.intent.operation is TransformOperation.SCALE_APPEARANCE:
                    adapter._apply_preflighted_appearance_scale_plan(
                        self.appearance_scale_plans[index]
                    )
                else:
                    raise TransformPreflightError(
                        [
                            (
                                adapter.target,
                                OperationSupport.denied(
                                    self.intent.operation,
                                    "No executor is registered for this operation",
                                ),
                            )
                        ]
                    )
            if (
                self.intent.operation is TransformOperation.SCALE_APPEARANCE
                and float(self.intent.factor) != 1.0
            ):
                for adapter in self.adapters:
                    adapter._record_change_records(
                        adapter.serialize_appearance_changes()
                    )
        except Exception as error:
            rollback_failures = []
            with suspend_change_recording():
                for adapter, state in zip(
                    reversed(self.adapters), reversed(snapshots)
                ):
                    try:
                        if appearance_only:
                            adapter.restore_appearance_state(
                                state, record_changes=False
                            )
                        else:
                            adapter.restore(state)
                    except Exception as rollback_error:
                        # Continue restoring earlier targets even when the adapter
                        # that failed the commit also cannot restore itself.
                        rollback_failures.append(
                            (adapter.target, rollback_error)
                        )
            for tracker, state in tracker_states:
                try:
                    tracker.restore_recording_state(state)
                except Exception as rollback_error:
                    rollback_failures.append((tracker, rollback_error))
            ArtistAdapter.annotate_rollback_failures(error, rollback_failures)
            raise


__all__ = ["TransformPlan", "TransformPreflightError"]
