from __future__ import annotations

from time import perf_counter

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.patches import Rectangle
from matplotlib.transforms import IdentityTransform

from pylustrator.artist_adapters import (
    _DisplayBboxContainsPatch,
    get_artist_adapter,
)
from pylustrator.transform_engine import PointEditPlan, PointEditSource


_CONNECTION_STYLES = (
    "arc3",
    "arc3,rad=0.35",
    "angle3,angleA=0,angleB=90",
    "angle,angleA=0,angleB=90,rad=5",
    "bar,fraction=0.25",
)


def _annotation_pair(*, arrowprops, rotation=0.0):
    fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
    kwargs = dict(
        xy=(0.25, 0.3),
        xycoords="data",
        xytext=(0.72, 0.78),
        textcoords="axes fraction",
        annotation_clip=False,
        rotation=rotation,
    )
    previewed = ax.annotate("reference", arrowprops=dict(arrowprops), **kwargs)
    oracle = ax.annotate("reference", arrowprops=dict(arrowprops), **kwargs)
    fig.canvas.draw()
    return fig, previewed, oracle


def _assert_preview_arrow_matches_live_oracle(
    source,
    oracle,
    oracle_adapter,
    *,
    key,
    destination,
):
    plan = PointEditPlan.preview(source, key, destination)
    native = oracle_adapter.display_to_native(plan.control_array())
    oracle_adapter._apply_native_control_points(native)
    oracle.update_positions(oracle.figure.canvas.get_renderer())

    preview_path = source.preview_context.target.arrow_patch.get_path()
    oracle_path = oracle.arrow_patch.get_path()
    np.testing.assert_array_equal(preview_path.codes, oracle_path.codes)
    np.testing.assert_array_equal(preview_path.vertices, oracle_path.vertices)
    return source.preview_context.target.arrow_patch.patchA


@pytest.mark.parametrize(
    ("connectionstyle", "rotation"),
    zip(_CONNECTION_STYLES, (0.0, 37.0, 90.0, 37.0, 0.0)),
)
def test_isolated_auto_patch_proxy_matches_live_arrow_path_at_20_positions(
    connectionstyle,
    rotation,
):
    fig, previewed, oracle = _annotation_pair(
        arrowprops={"arrowstyle": "->", "connectionstyle": connectionstyle},
        rotation=rotation,
    )
    try:
        source = PointEditSource.capture(previewed)
        oracle_adapter = get_artist_adapter(oracle)
        proxy = None
        angles = np.linspace(0.0, 2.0 * np.pi, 20, endpoint=False)
        for index, angle in enumerate(angles):
            key = index % 2
            destination = source.handle_model.positions_array()[key] + np.array(
                (12.0 * np.cos(angle), 9.0 * np.sin(angle))
            )
            current = _assert_preview_arrow_matches_live_oracle(
                source,
                oracle,
                oracle_adapter,
                key=key,
                destination=destination,
            )
            assert isinstance(current, _DisplayBboxContainsPatch)
            proxy = current if proxy is None else proxy
            assert current is proxy
            assert isinstance(oracle.arrow_patch.patchA, Rectangle)
            assert not isinstance(
                oracle.arrow_patch.patchA,
                _DisplayBboxContainsPatch,
            )
    finally:
        plt.close(fig)


@pytest.mark.parametrize("rotation", (0.0, 37.0, 90.0))
def test_legacy_annotation_arrow_proxy_matches_live_path_for_both_anchors(rotation):
    fig, previewed, oracle = _annotation_pair(
        arrowprops={
            "width": 2.0,
            "headwidth": 8.0,
            "headlength": 10.0,
            "shrink": 0.05,
            "connectionstyle": "arc3",
        },
        rotation=rotation,
    )
    try:
        source = PointEditSource.capture(previewed)
        oracle_adapter = get_artist_adapter(oracle)
        for key, delta in ((0, (7.0, -5.0)), (1, (-9.0, 6.0))):
            patch = _assert_preview_arrow_matches_live_oracle(
                source,
                oracle,
                oracle_adapter,
                key=key,
                destination=source.handle_model.positions_array()[key] + delta,
            )
            assert isinstance(patch, _DisplayBboxContainsPatch)
    finally:
        plt.close(fig)


def test_explicit_annotation_patch_a_is_never_replaced_on_preview_clone():
    custom_patch = Rectangle(
        (10.0, 20.0),
        30.0,
        40.0,
        transform=IdentityTransform(),
    )
    fig, previewed, _oracle = _annotation_pair(
        arrowprops={"arrowstyle": "->", "patchA": custom_patch},
    )
    try:
        source = PointEditSource.capture(previewed)
        destination = source.handle_model.positions_array()[1] + (8.0, -4.0)
        PointEditPlan.preview(source, 1, destination)

        assert previewed.arrow_patch.patchA is custom_patch
        assert source.preview_context.target.arrow_patch.patchA is custom_patch
        assert not isinstance(custom_patch, _DisplayBboxContainsPatch)
    finally:
        plt.close(fig)


@pytest.mark.parametrize(
    ("text", "bbox"),
    (("", None), ("reference", {"boxstyle": "round", "facecolor": "white"})),
)
def test_auto_patch_proxy_guard_skips_empty_text_and_bbox_annotations(text, bbox):
    fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
    annotation = ax.annotate(
        text,
        xy=(0.25, 0.3),
        xytext=(0.72, 0.78),
        arrowprops={"arrowstyle": "->"},
        bbox=bbox,
        annotation_clip=False,
    )
    fig.canvas.draw()
    try:
        clone_adapter = get_artist_adapter(annotation)._point_edit_preview_adapter()
        clone_adapter.selection_points()
        patch = clone_adapter.target.arrow_patch.patchA

        assert not isinstance(patch, _DisplayBboxContainsPatch)
        if bbox is None:
            assert patch is None
        else:
            assert patch is clone_adapter.target.get_bbox_patch()
    finally:
        plt.close(fig)


def test_annotation_auto_patch_warm_preview_p95_stays_below_four_ms():
    fig, previewed, _oracle = _annotation_pair(
        arrowprops={"arrowstyle": "->", "connectionstyle": "arc3"},
    )
    try:
        source = PointEditSource.capture(previewed)
        destination = source.handle_model.positions_array()[1] + (12.0, -7.0)
        for offset in np.linspace(0.0, 1.0, 20):
            PointEditPlan.preview(source, 1, destination + (offset, 0.0))

        samples = []
        for offset in np.linspace(0.0, 1.0, 200):
            started = perf_counter()
            PointEditPlan.preview(source, 1, destination + (offset, 0.0))
            samples.append((perf_counter() - started) * 1000.0)

        assert np.percentile(samples, 95) < 4.0
    finally:
        plt.close(fig)
