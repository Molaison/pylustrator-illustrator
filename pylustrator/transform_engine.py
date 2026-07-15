"""Atomic preflight and execution of semantic transform intents."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from matplotlib.artist import Artist

from .artist_adapters import (
    ArtistAdapter,
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


@dataclass
class TransformPlan:
    intent: TransformIntent
    adapters: tuple[ArtistAdapter, ...]

    @classmethod
    def preflight(
        cls, targets: Iterable[Artist], intent: TransformIntent
    ) -> "TransformPlan":
        adapters = tuple(get_artist_adapter(target) for target in targets)
        failures = []
        for adapter in adapters:
            support = adapter.operation_support(intent.operation)
            if not support.supported:
                failures.append((adapter.target, support))
        if failures:
            raise TransformPreflightError(failures)
        return cls(intent, adapters)

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
        return tuple(
            np.asarray(adapter.control_points(), dtype=float)
            for adapter in self.adapters
        )

    def commit(self) -> None:
        snapshots = [adapter.snapshot() for adapter in self.adapters]
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
            for adapter in self.adapters:
                if self.intent.operation is TransformOperation.TRANSLATE:
                    adapter.translate(self.intent.delta)
                elif self.intent.operation is TransformOperation.RESIZE_GEOMETRY:
                    adapter.resize(np.asarray(self.intent.matrix, dtype=float))
                elif self.intent.operation is TransformOperation.ROTATE:
                    adapter.set_rotation(
                        adapter.rotation() + float(self.intent.angle_degrees)
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
        except Exception:
            with suspend_change_recording():
                for adapter, state in zip(
                    reversed(self.adapters), reversed(snapshots)
                ):
                    try:
                        adapter.restore(state)
                    except Exception:
                        # Continue restoring earlier targets even when the adapter
                        # that failed the commit also cannot restore itself.
                        continue
            for tracker, state in tracker_states:
                tracker.restore_recording_state(state)
            raise


__all__ = ["TransformPlan", "TransformPreflightError"]
