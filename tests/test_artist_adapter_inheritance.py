from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection

from pylustrator.artist_adapters import (
    UnsupportedArtistError,
    UnsupportedSubclassAdapter,
    get_artist_adapter,
)
from pylustrator.operations import TransformIntent, TransformOperation
from pylustrator.snap import TargetWrapper
from pylustrator.transform_engine import TransformPlan, TransformPreflightError


def _make_3d_target(ax, kind: str):
    if kind == "Line3D":
        return ax.plot([0.1, 0.9], [0.2, 0.8], [0.3, 0.7])[0]
    if kind == "Text3D":
        return ax.text(0.2, 0.3, 0.4, "3d text")
    if kind == "Path3DCollection":
        return ax.scatter([0.2, 0.8], [0.3, 0.7], [0.4, 0.6])
    if kind == "Line3DCollection":
        target = Line3DCollection([[(0.1, 0.2, 0.3), (0.8, 0.7, 0.6)]])
        ax.add_collection3d(target)
        return target
    if kind == "Poly3DCollection":
        target = Poly3DCollection([[(0.1, 0.2, 0.3), (0.8, 0.2, 0.3), (0.4, 0.7, 0.6)]])
        ax.add_collection3d(target)
        return target
    raise AssertionError(f"Unknown 3D QA target: {kind}")


def _native_3d_state(target) -> np.ndarray:
    kind = type(target).__name__
    if kind == "Line3D":
        return np.column_stack(target.get_data_3d()).copy()
    if kind == "Text3D":
        return np.asarray(target.get_position_3d(), dtype=float).copy()
    if kind == "Path3DCollection":
        return np.column_stack(
            [np.asarray(values) for values in target._offsets3d]
        ).copy()
    if kind == "Line3DCollection":
        return np.asarray(target._segments3d, dtype=float).copy()
    if kind == "Poly3DCollection":
        return np.asarray(target._vec, dtype=float).copy()
    raise AssertionError(f"Unknown 3D QA target: {kind}")


@pytest.mark.parametrize(
    ("kind", "blocked_parent"),
    [
        ("Line3D", "Line2D"),
        ("Text3D", "Text"),
        ("Path3DCollection", "PathCollection"),
        ("Line3DCollection", "LineCollection"),
        ("Poly3DCollection", "PolyCollection"),
    ],
)
def test_3d_semantic_subclasses_fail_closed_without_preview_or_mutation(
    kind, blocked_parent
) -> None:
    fig = plt.figure(figsize=(4, 3), dpi=100)
    ax = fig.add_subplot(projection="3d")
    target = _make_3d_target(ax, kind)
    fig.canvas.draw()
    before = _native_3d_state(target)
    adapter = get_artist_adapter(target)

    try:
        assert type(target).__name__ == kind
        assert isinstance(adapter, UnsupportedSubclassAdapter)
        assert not TargetWrapper.supports_target(target)

        for operation in (
            TransformOperation.SELECT,
            TransformOperation.TRANSLATE,
            TransformOperation.RESIZE_GEOMETRY,
            TransformOperation.SNAPSHOT,
            TransformOperation.SERIALIZE,
        ):
            support = adapter.operation_support(operation)
            assert not support.supported
            assert kind in support.reason
            assert blocked_parent in support.reason
            assert "exact-only" in support.reason

        with pytest.raises(UnsupportedArtistError, match="exact-only"):
            adapter.translate((11.0, -7.0))
        with pytest.raises(TransformPreflightError) as error:
            TransformPlan.preflight([target], TransformIntent.translate((11.0, -7.0)))
        assert "exact-only" in error.value.failures[0][1].reason
        np.testing.assert_array_equal(_native_3d_state(target), before)
    finally:
        plt.close(fig)
