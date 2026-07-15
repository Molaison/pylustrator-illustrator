"""Atomic preflight and execution of semantic transform intents."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from matplotlib.artist import Artist

from .artist_adapters import ArtistAdapter, get_artist_adapter
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
                np.asarray(adapter.control_points(), dtype=float) + delta
                for adapter in self.adapters
            )
        if operation is TransformOperation.RESIZE_GEOMETRY:
            matrix = np.asarray(self.intent.matrix, dtype=float)
            result = []
            for adapter in self.adapters:
                points = np.asarray(adapter.control_points(), dtype=float)
                homogeneous = np.column_stack((points, np.ones(len(points))))
                result.append((matrix @ homogeneous.T).T[:, :2])
            return tuple(result)
        return tuple(
            np.asarray(adapter.control_points(), dtype=float)
            for adapter in self.adapters
        )

    def commit(self) -> None:
        snapshots = [adapter.snapshot() for adapter in self.adapters]
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
            for adapter, state in zip(reversed(self.adapters), reversed(snapshots)):
                try:
                    adapter.restore(state)
                except Exception:
                    # Continue restoring earlier targets even when the adapter
                    # that failed the commit also cannot restore itself.
                    continue
            raise


__all__ = ["TransformPlan", "TransformPreflightError"]
