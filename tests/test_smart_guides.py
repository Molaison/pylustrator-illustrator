from __future__ import annotations

from dataclasses import FrozenInstanceError
import math
from time import perf_counter

import numpy as np
import pytest

from pylustrator.smart_guides import (
    Axis,
    DisplayBounds,
    EqualGapOverlay,
    ExplicitAnchor,
    GuideCandidateIndex,
    GuideKind,
    GuideLine,
    GuideObject,
    GuideSnapshot,
    MovingGeometry,
    SnapPlan,
    StaleGuideSnapshotError,
    _nearest_gap_pairs_sweep,
    _nearest_gap_pairs_vectorized,
)


def _source(
    stable_id: str,
    bounds: tuple[float, float, float, float],
    **kwargs,
) -> GuideObject:
    return GuideObject(stable_id, DisplayBounds(*bounds), **kwargs)


def _index(
    *sources: GuideObject,
    selected_ids: tuple[str, ...] = (),
    revision: str | int | None = "test",
) -> GuideCandidateIndex:
    return GuideCandidateIndex(
        GuideSnapshot.capture(
            sources,
            selected_ids=selected_ids,
            revision=revision,
        )
    )


def _hit(plan: SnapPlan, axis: Axis):
    return next(hit for hit in plan.hits if hit.axis is axis)


def test_snapshot_filters_selected_hidden_locked_and_nonfinite_sources() -> None:
    selected = _source("selected", (0, 0, 10, 10))
    visible = _source("visible", (20, 20, 30, 30))
    hidden = _source("hidden", (40, 40, 50, 50), visible=False)
    locked = _source("locked", (60, 60, 70, 70), locked=True)
    nonfinite_bounds = _source("nan-bounds", (math.nan, 0, 1, 1))
    nonfinite_anchor = _source(
        "nan-anchor",
        (80, 80, 90, 90),
        anchors=(ExplicitAnchor(math.inf, 85),),
    )

    snapshot = GuideSnapshot.capture(
        [selected, visible, hidden, locked, nonfinite_bounds, nonfinite_anchor],
        selected_ids=("selected",),
        revision=7,
    )

    assert tuple(source.stable_id for source in snapshot.objects) == ("visible",)
    assert snapshot.selected_ids == frozenset({"selected"})
    with pytest.raises(FrozenInstanceError):
        snapshot.fingerprint = "changed"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        visible.order = 999  # type: ignore[misc]


def test_snapshot_copies_anchor_sequence_and_fingerprint_is_input_order_independent() -> None:
    anchors = [ExplicitAnchor(2, 3, "text")]
    first = _source("a", (0, 0, 10, 10), anchors=anchors)
    anchors.append(ExplicitAnchor(8, 9, "late mutation"))
    second = _source("b", (20, 20, 30, 30))

    forward = GuideSnapshot.capture([first, second], revision="r1")
    reverse = GuideSnapshot.capture([second, first], revision="r1")

    assert len(first.anchors) == 1
    assert forward.objects == reverse.objects
    assert forward.fingerprint == reverse.fingerprint


def test_selected_target_cannot_snap_to_itself() -> None:
    target = _source("target", (100, 100, 120, 120))
    index = _index(target, selected_ids=("target",))

    plan = index.query(MovingGeometry.from_object(target), tolerance_px=10)

    assert not plan.has_snap
    assert plan.delta_px == (0.0, 0.0)
    assert index.source_count == 0


def test_edge_snap_returns_one_reusable_preview_and_commit_plan() -> None:
    index = _index(_source("source", (100, 100, 140, 140)))
    moving = MovingGeometry(DisplayBounds(78, 200, 98, 220))

    plan = index.query(moving, tolerance_px=3)

    assert plan.delta_px == (2.0, 0.0)
    assert _hit(plan, Axis.X).kind is GuideKind.EDGE
    assert _hit(plan, Axis.X).source_ids == ("source",)
    assert plan.apply_to_bounds(moving.bounds) == DisplayBounds(80, 200, 100, 220)
    assert plan.apply_to_point((10, 20)) == (12.0, 20.0)
    overlay = next(item for item in plan.overlays if item.axis is Axis.X)
    assert overlay == GuideLine(
        Axis.X,
        100.0,
        (100.0, 220.0),
        GuideKind.EDGE,
        ("source",),
    )
    assert not hasattr(plan, "candidates")
    assert not hasattr(plan, "__dict__")
    assert len(plan.hits) <= 2
    with pytest.raises(FrozenInstanceError):
        plan.delta_px = (99, 99)  # type: ignore[misc]


def test_center_baseline_and_explicit_anchor_are_semantic_guides() -> None:
    center_index = _index(_source("center", (100, 100, 140, 140)))
    center_plan = center_index.query(
        MovingGeometry(DisplayBounds(107, 200, 127, 220)), tolerance_px=4
    )
    assert center_plan.delta_px == (3.0, 0.0)
    assert _hit(center_plan, Axis.X).kind is GuideKind.CENTER

    baseline_index = _index(
        _source("baseline", (300, 30, 340, 60), baseline_y=72)
    )
    baseline_plan = baseline_index.query(
        MovingGeometry(DisplayBounds(100, 180, 130, 210), baseline_y=69),
        tolerance_px=4,
    )
    assert baseline_plan.delta_px == (0.0, 3.0)
    assert _hit(baseline_plan, Axis.Y).kind is GuideKind.BASELINE

    anchor_index = _index(
        _source(
            "anchor",
            (300, 300, 320, 320),
            anchors=(ExplicitAnchor(80, 120, "insertion"),),
        )
    )
    anchor_plan = anchor_index.query(
        MovingGeometry(
            DisplayBounds(200, 200, 220, 220),
            anchors=(ExplicitAnchor(78, 117, "insertion"),),
        ),
        tolerance_px=4,
    )
    assert anchor_plan.delta_px == (2.0, 3.0)
    assert _hit(anchor_plan, Axis.X).kind is GuideKind.ANCHOR
    assert _hit(anchor_plan, Axis.Y).kind is GuideKind.ANCHOR


def test_cross_type_edge_to_center_and_anchor_to_edge_snapping() -> None:
    center_index = _index(_source("wide", (0, 0, 200, 10)))
    edge_to_center = center_index.query(
        MovingGeometry(DisplayBounds(98, 100, 98, 110)), tolerance_px=2
    )
    center_hit = _hit(edge_to_center, Axis.X)
    assert center_hit.kind is GuideKind.CENTER
    assert center_hit.source_feature == "x_center"
    assert center_hit.target_feature == "x_min_edge"
    assert center_hit.delta_px == 2.0

    edge_index = _index(_source("edge", (100, 0, 120, 10)))
    anchor_to_edge = edge_index.query(
        MovingGeometry(
            DisplayBounds(300, 300, 320, 320),
            anchors=(ExplicitAnchor(98, 500, "text"),),
        ),
        tolerance_px=2,
    )
    edge_hit = _hit(anchor_to_edge, Axis.X)
    assert edge_hit.kind is GuideKind.EDGE
    assert edge_hit.source_feature == "x_min_edge"
    assert edge_hit.target_feature == "anchor:text:x"
    assert edge_hit.delta_px == 2.0


def test_baseline_cross_type_competes_with_ordinary_source_features() -> None:
    index = _index(
        _source("baseline", (300, 20, 320, 40), baseline_y=100),
        _source("ordinary-edge", (340, 100, 360, 120)),
    )
    competing = index.query(
        MovingGeometry(DisplayBounds(0, 200, 10, 210), baseline_y=99),
        tolerance_px=2,
    )
    winner = _hit(competing, Axis.Y)
    assert winner.delta_px == 1.0
    assert winner.kind is GuideKind.BASELINE
    assert winner.source_ids == ("baseline",)
    assert winner.target_feature == "baseline"

    baseline_only = _index(
        _source("baseline", (300, 20, 320, 40), baseline_y=100)
    )
    ordinary_edge_to_baseline = baseline_only.query(
        MovingGeometry(DisplayBounds(0, 98, 10, 98)), tolerance_px=2
    )
    cross_hit = _hit(ordinary_edge_to_baseline, Axis.Y)
    assert cross_hit.kind is GuideKind.BASELINE
    assert cross_hit.source_feature == "baseline"
    assert cross_hit.target_feature == "y_min_edge"


def test_ties_use_guide_type_then_z_order_then_order_then_stable_id() -> None:
    semantic_index = _index(_source("semantic", (100, 200, 110, 220)))
    semantic_plan = semantic_index.query(
        MovingGeometry(DisplayBounds(99, 300, 109, 310)), tolerance_px=2
    )
    assert _hit(semantic_plan, Axis.X).kind is GuideKind.EDGE

    tie_index = _index(
        _source("low-z", (100, 10, 120, 20), z_order=2, order=99),
        _source("low-order", (100, 30, 120, 40), z_order=3, order=1),
        _source("d", (100, 50, 120, 60), z_order=3, order=2),
        _source("c", (100, 70, 120, 80), z_order=3, order=2),
    )
    tie_plan = tie_index.query(
        MovingGeometry(DisplayBounds(99, 300, 109, 310)), tolerance_px=2
    )
    assert _hit(tie_plan, Axis.X).source_ids == ("c",)


def test_negative_coordinates_and_inclusive_display_pixel_tolerance() -> None:
    negative_index = _index(_source("negative", (-100, 100, -90, 110)))
    negative_plan = negative_index.query(
        MovingGeometry(DisplayBounds(-112, 200, -102, 210)), tolerance_px=2
    )
    assert negative_plan.delta_px == (2.0, 0.0)

    pixel_index = _index(_source("pixel", (100, 100, 110, 110)))
    moving = MovingGeometry(DisplayBounds(85.5, 200, 95.5, 210))
    outside = pixel_index.query(moving, tolerance_px=4.49)
    boundary = pixel_index.query(moving, tolerance_px=4.5)
    assert not any(hit.axis is Axis.X for hit in outside.hits)
    assert _hit(boundary, Axis.X).delta_px == 4.5


def test_hidpi_geometry_still_uses_unscaled_display_pixel_tolerance() -> None:
    normal_index = _index(_source("normal", (100, 100, 110, 110)))
    normal_moving = MovingGeometry(DisplayBounds(85.5, 200, 95.5, 210))
    assert _hit(normal_index.query(normal_moving, tolerance_px=4.5), Axis.X)

    # A renderer reporting twice as many display pixels also reports twice the
    # coordinate distance.  The kernel never silently converts tolerance to
    # points, inches, logical pixels, or a device-pixel ratio.
    hidpi_index = _index(_source("hidpi", (200, 200, 220, 220)))
    hidpi_moving = MovingGeometry(DisplayBounds(171, 400, 191, 420))
    no_hit = hidpi_index.query(hidpi_moving, tolerance_px=4.5)
    hit = hidpi_index.query(hidpi_moving, tolerance_px=9)
    assert not any(candidate.axis is Axis.X for candidate in no_hit.hits)
    assert _hit(hit, Axis.X).delta_px == 9.0


def test_scope_prevents_unrelated_display_surfaces_from_interacting() -> None:
    index = _index(
        _source("canvas-a", (100, 100, 110, 110), scope_id="a"),
        _source("canvas-b", (101, 100, 111, 110), scope_id="b"),
    )
    plan = index.query(
        MovingGeometry(DisplayBounds(90, 200, 99, 209), scope_id="a"),
        tolerance_px=3,
    )
    assert _hit(plan, Axis.X).source_ids == ("canvas-a",)
    assert _hit(plan, Axis.X).delta_px == 1.0


def test_equal_gap_after_uses_preindexed_pair_and_drawable_intervals() -> None:
    index = _index(
        _source("a", (0, 0, 10, 10)),
        _source("b", (20, 0, 30, 10)),
    )

    plan = index.query(
        MovingGeometry(DisplayBounds(42, 0, 50, 10)), tolerance_px=3
    )

    x_hit = _hit(plan, Axis.X)
    assert x_hit.kind is GuideKind.EQUAL_GAP
    assert x_hit.delta_px == -2.0
    assert x_hit.gap_px == 10.0
    overlay = next(
        item for item in plan.overlays if isinstance(item, EqualGapOverlay)
    )
    assert overlay.axis is Axis.X
    assert overlay.intervals_px == ((10.0, 20.0), (30.0, 40.0))
    assert overlay.gap_px == 10.0


def test_equal_gap_sweep_ignores_orthogonally_disjoint_shadowing_object() -> None:
    index = _index(
        _source("a", (0, 0, 10, 10)),
        _source("shadow-in-another-row", (20, 20, 30, 30)),
        _source("b", (40, 0, 50, 10)),
    )

    plan = index.query(
        MovingGeometry(DisplayBounds(82, 0, 90, 10)), tolerance_px=3
    )

    hit = _hit(plan, Axis.X)
    assert hit.kind is GuideKind.EQUAL_GAP
    assert hit.source_ids == ("a", "b")
    assert hit.gap_px == 30.0
    assert hit.delta_px == -2.0


def test_vectorized_gap_neighbors_exactly_match_sweep_random_scenes() -> None:
    rng = np.random.default_rng(20260717)
    for count in (2, 7, 31, 97):
        for repetition in range(6):
            sources = []
            for index in range(count):
                x0, y0 = rng.uniform(-100, 100, size=2)
                width, height = rng.uniform(0, 30, size=2)
                sources.append(
                    _source(
                        f"{count}:{repetition}:{index}",
                        (x0, y0, x0 + width, y0 + height),
                        z_order=float(rng.integers(-3, 6)),
                        order=int(rng.integers(0, 20)),
                    )
                )
            for axis in (Axis.X, Axis.Y):
                sweep = _nearest_gap_pairs_sweep(sources, axis)
                vectorized = _nearest_gap_pairs_vectorized(sources, axis)
                assert [
                    (first.stable_id, second.stable_id)
                    for first, second in vectorized
                ] == [
                    (first.stable_id, second.stable_id)
                    for first, second in sweep
                ]


def test_axis_restricted_query_recomputes_overlay_from_restricted_delta() -> None:
    index = _index(_source("source", (100, 100, 110, 110)))
    moving = MovingGeometry(DisplayBounds(89, 89, 99, 99))

    full = index.query(moving, tolerance_px=2)
    x_only = index.query(
        moving,
        tolerance_px=2,
        axes=frozenset((Axis.X,)),
    )

    assert full.delta_px == (1.0, 1.0)
    assert x_only.delta_px == (1.0, 0.0)
    assert {hit.axis for hit in x_only.hits} == {Axis.X}
    overlay = next(item for item in x_only.overlays if item.axis is Axis.X)
    assert isinstance(overlay, GuideLine)
    assert overlay.span_px == (89.0, 110.0)


def test_equal_gap_between_accounts_for_moving_size() -> None:
    index = _index(
        _source("a", (0, 0, 10, 10)),
        _source("b", (30, 0, 40, 10)),
    )

    plan = index.query(
        MovingGeometry(DisplayBounds(16, 0, 26, 10)), tolerance_px=2
    )

    x_hit = _hit(plan, Axis.X)
    assert x_hit.kind is GuideKind.EQUAL_GAP
    assert x_hit.delta_px == -1.0
    assert x_hit.gap_px == 5.0
    overlay = next(
        item for item in plan.overlays if isinstance(item, EqualGapOverlay)
    )
    assert overlay.intervals_px == ((10.0, 15.0), (25.0, 30.0))


def test_equal_gap_is_symmetric_and_works_on_the_y_axis() -> None:
    horizontal = _index(
        _source("a", (20, 0, 30, 10)),
        _source("b", (40, 0, 50, 10)),
    )
    before = horizontal.query(
        MovingGeometry(DisplayBounds(2, 0, 12, 10)), tolerance_px=3
    )
    assert _hit(before, Axis.X).kind is GuideKind.EQUAL_GAP
    assert _hit(before, Axis.X).delta_px == -2.0

    vertical = _index(
        _source("top", (0, 0, 10, 10)),
        _source("bottom", (0, 20, 10, 30)),
    )
    after = vertical.query(
        MovingGeometry(DisplayBounds(0, 42, 10, 50)), tolerance_px=3
    )
    assert _hit(after, Axis.Y).kind is GuideKind.EQUAL_GAP
    assert _hit(after, Axis.Y).delta_px == -2.0


def test_equal_gap_can_be_disabled_without_rebuilding_the_index() -> None:
    index = _index(
        _source("a", (0, 0, 10, 10)),
        _source("b", (20, 0, 30, 10)),
    )
    moving = MovingGeometry(DisplayBounds(42, 0, 50, 10))

    plan = index.query(moving, tolerance_px=3, include_equal_gaps=False)

    assert not any(hit.axis is Axis.X for hit in plan.hits)


def test_no_hit_has_zero_delta_and_no_overlay() -> None:
    index = _index(_source("far", (0, 0, 10, 10)))

    plan = index.query(
        MovingGeometry(DisplayBounds(100, 100, 110, 110)), tolerance_px=3
    )

    assert plan.delta_px == (0.0, 0.0)
    assert plan.hits == ()
    assert plan.overlays == ()
    assert not plan.has_snap


def test_stale_snapshot_is_rejected_at_query_and_before_commit() -> None:
    source = _source("source", (100, 100, 110, 110))
    original = GuideSnapshot.capture([source], revision=1)
    changed = GuideSnapshot.capture([source], revision=2)
    index = GuideCandidateIndex(original)
    moving = MovingGeometry(DisplayBounds(90, 200, 99, 209))

    with pytest.raises(StaleGuideSnapshotError, match="changed"):
        index.query(moving, expected_fingerprint=changed.fingerprint)

    plan = index.query(moving, expected_fingerprint=original.fingerprint)
    plan.require_fingerprint(original.fingerprint)
    with pytest.raises(StaleGuideSnapshotError, match="older"):
        plan.require_fingerprint(changed.fingerprint)


def _performance_index(target_candidate_count: int) -> GuideCandidateIndex:
    source_count = math.ceil(target_candidate_count / 6)
    sources = [
        _source(
            f"source-{index:05d}",
            (
                index * 20.0,
                index * 13.0,
                index * 20.0 + 5.0,
                index * 13.0 + 5.0,
            ),
            order=index,
        )
        for index in range(source_count)
    ]
    return _index(*sources)


def _timed_row_build(
    target_candidate_count: int,
) -> tuple[GuideCandidateIndex, float]:
    source_count = math.ceil(target_candidate_count / 6)
    sources = [
        _source(
            f"row-{index:05d}",
            (index * 10.0, 0.0, index * 10.0 + 5.0, 5.0),
            order=index,
        )
        for index in range(source_count)
    ]
    start = perf_counter()
    index = _index(*sources)
    return index, perf_counter() - start


def _best_query_time(index: GuideCandidateIndex, repetitions: int = 250) -> float:
    middle = index.source_count // 2
    moving = MovingGeometry(
        DisplayBounds(
            middle * 20.0 - 7.0,
            middle * 13.0 - 7.0,
            middle * 20.0 - 2.0,
            middle * 13.0 - 2.0,
        )
    )
    index.query(moving, tolerance_px=4)
    samples = []
    for _ in range(3):
        start = perf_counter()
        for _ in range(repetitions):
            index.query(moving, tolerance_px=4)
        samples.append(perf_counter() - start)
    return min(samples)


def test_query_microbenchmark_is_indexed_at_1k_and_10k_candidates() -> None:
    one_thousand = _performance_index(1_000)
    ten_thousand = _performance_index(10_000)
    assert one_thousand.alignment_candidate_count >= 1_000
    assert ten_thousand.alignment_candidate_count >= 10_000

    small_moving = MovingGeometry(DisplayBounds(1658, 1071, 1663, 1076))
    large_moving = MovingGeometry(DisplayBounds(16658, 10821, 16663, 10826))
    small_plan = one_thousand.query(small_moving, tolerance_px=4)
    large_plan = ten_thousand.query(large_moving, tolerance_px=4)
    assert small_plan.examined_candidate_count <= 12
    assert large_plan.examined_candidate_count <= 12

    small_time = _best_query_time(one_thousand)
    large_time = _best_query_time(ten_thousand)
    # A tenfold candidate increase should remain near-flat after indexing.  The
    # additive allowance avoids false failures on very fast/noisy CI workers.
    assert large_time <= small_time * 4.0 + 0.003


def test_gap_sweep_build_microbenchmark_at_1k_and_10k_candidates() -> None:
    one_thousand, small_time = _timed_row_build(1_000)
    ten_thousand, large_time = _timed_row_build(10_000)

    assert one_thousand.alignment_candidate_count >= 1_000
    assert ten_thousand.alignment_candidate_count >= 10_000
    assert one_thousand.gap_reference_count == (one_thousand.source_count - 1) * 3
    assert ten_thousand.gap_reference_count == (ten_thousand.source_count - 1) * 3
    # Ten times more objects plus O(n log n) neighbour sweeps should remain
    # well below quadratic growth.  Keep a broad additive CI allowance.
    assert large_time <= small_time * 25.0 + 0.05


@pytest.mark.parametrize("tolerance", [-1, math.inf, math.nan])
def test_invalid_tolerance_is_rejected(tolerance: float) -> None:
    index = _index(_source("source", (0, 0, 10, 10)))
    with pytest.raises(ValueError, match="display-pixel"):
        index.query(
            MovingGeometry(DisplayBounds(20, 20, 30, 30)),
            tolerance_px=tolerance,
        )
