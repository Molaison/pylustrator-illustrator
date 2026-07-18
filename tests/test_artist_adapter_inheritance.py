from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.patches import Arc, Circle, CirclePolygon
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection

from pylustrator.artist_adapters import (
    AdapterInheritancePolicy,
    ArcAdapter,
    CircleAdapter,
    CirclePolygonAdapter,
    UnsupportedArtistError,
    UnsupportedSubclassAdapter,
    artist_adapter_registry,
    get_artist_adapter,
)
from pylustrator.operations import TransformIntent, TransformOperation
from pylustrator.snap import TargetWrapper
from pylustrator.transform_engine import TransformPlan, TransformPreflightError


class _RecordingTracker:
    def __init__(self):
        self.calls = []

    def addChange(self, target, command):
        self.calls.append((target, command))


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


def _make_semantic_patch(ax, kind: str):
    if kind == "Arc":
        return ax.add_patch(
            Arc(
                (0.45, 0.52),
                0.34,
                0.22,
                angle=19.0,
                theta1=15.0,
                theta2=285.0,
            )
        )
    if kind == "Circle":
        return ax.add_patch(Circle((0.45, 0.52), radius=0.15))
    if kind == "CirclePolygon":
        return ax.add_patch(
            CirclePolygon((0.45, 0.52), radius=0.15, resolution=12)
        )
    raise AssertionError(f"Unknown semantic patch: {kind}")


def _semantic_position(target) -> np.ndarray:
    value = target.xy if isinstance(target, CirclePolygon) else target.center
    return np.asarray(value, dtype=float).copy()


def _semantic_shape_state(target):
    if isinstance(target, Arc):
        return (
            float(target.width),
            float(target.height),
            float(target.angle),
            float(target.theta1),
            float(target.theta2),
        )
    if isinstance(target, Circle):
        return (float(target.radius), float(target.width), float(target.height))
    return (
        float(target.radius),
        float(target.orientation),
        int(target.numvertices),
    )


def _semantic_clone(target):
    if isinstance(target, Arc):
        return Arc(
            (0.0, 0.0),
            target.width,
            target.height,
            angle=target.angle,
            theta1=target.theta1,
            theta2=target.theta2,
        )
    if isinstance(target, Circle):
        return Circle((0.0, 0.0), radius=target.radius)
    return CirclePolygon(
        (0.0, 0.0),
        radius=target.radius,
        resolution=target.numvertices,
    )


@pytest.mark.parametrize(
    ("kind", "adapter_type"),
    [
        ("Arc", ArcAdapter),
        ("Circle", CircleAdapter),
        ("CirclePolygon", CirclePolygonAdapter),
    ],
)
def test_semantic_patch_exact_contract_translates_undoes_and_replays(
    kind, adapter_type
) -> None:
    fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
    fig.change_tracker = _RecordingTracker()
    target = _make_semantic_patch(ax, kind)
    fig.canvas.draw()
    adapter = get_artist_adapter(target)
    snapshot = adapter.snapshot()
    shape_before = _semantic_shape_state(target)
    selection_before = np.asarray(adapter.selection_points(), dtype=float)
    delta = np.array([11.0, -7.0])
    preview = adapter.preview_translation_selection_points(delta)

    try:
        assert type(adapter) is adapter_type
        assert adapter.operation_support(TransformOperation.SELECT).supported
        assert adapter.operation_support(TransformOperation.TRANSLATE).supported
        for operation in (
            TransformOperation.RESIZE_GEOMETRY,
            TransformOperation.ROTATE,
            TransformOperation.RIGID_ROTATE,
        ):
            support = adapter.operation_support(operation)
            assert not support.supported
            assert support.reason.strip()

        adapter.translate(delta)
        fig.canvas.draw()
        np.testing.assert_allclose(
            adapter.selection_points(), preview, atol=0.25, rtol=0.0
        )
        assert _semantic_shape_state(target) == shape_before
        records = adapter.serialize_changes()
        assert len(records) == 1
        assert len(fig.change_tracker.calls) == 1

        clone = _semantic_clone(target)
        exec("clone" + records[0].command, {"clone": clone})
        np.testing.assert_allclose(
            _semantic_position(clone), _semantic_position(target), atol=0, rtol=0
        )
        assert _semantic_shape_state(clone) == shape_before

        adapter.restore(snapshot)
        fig.canvas.draw()
        np.testing.assert_allclose(
            adapter.selection_points(), selection_before, atol=0.25, rtol=0.0
        )
        assert _semantic_shape_state(target) == shape_before
        assert len(fig.change_tracker.calls) == 2
    finally:
        plt.close(fig)


@pytest.mark.parametrize(
    ("artist_type", "adapter_type"),
    [
        (Arc, ArcAdapter),
        (Circle, CircleAdapter),
        (CirclePolygon, CirclePolygonAdapter),
    ],
)
def test_semantic_patch_exact_registration_stays_on_cached_hot_path(
    monkeypatch, artist_type, adapter_type
) -> None:
    registration = next(
        item
        for item in artist_adapter_registry.registrations()
        if item.artist_type is artist_type
    )
    assert registration.inheritance_policy is AdapterInheritancePolicy.EXACT
    assert artist_adapter_registry.resolve_type(artist_type) is adapter_type
    assert artist_adapter_registry._cache[artist_type] is adapter_type

    def fail_uncached(_concrete):
        raise AssertionError("cached exact registration fell back to MRO resolution")

    monkeypatch.setattr(
        artist_adapter_registry, "_resolve_uncached", fail_uncached
    )
    for _ in range(10_000):
        assert artist_adapter_registry.resolve_type(artist_type) is adapter_type


@pytest.mark.parametrize("kind", ["Arc", "Circle", "CirclePolygon"])
def test_semantic_patch_non_affine_translation_is_rejected_without_mutation(
    kind,
) -> None:
    fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
    ax.set_xscale("log")
    ax.set_xlim(0.1, 10.0)
    target = _make_semantic_patch(ax, kind)
    fig.canvas.draw()
    adapter = get_artist_adapter(target)
    position_before = _semantic_position(target)
    shape_before = _semantic_shape_state(target)

    try:
        assert adapter.operation_support(TransformOperation.SELECT).supported
        assert adapter.operation_support(TransformOperation.SERIALIZE).supported
        support = adapter.operation_support(TransformOperation.TRANSLATE)
        assert not support.supported
        with pytest.raises(UnsupportedArtistError, match="translation"):
            adapter.translate((11.0, -7.0))
        np.testing.assert_array_equal(_semantic_position(target), position_before)
        assert _semantic_shape_state(target) == shape_before
    finally:
        plt.close(fig)
