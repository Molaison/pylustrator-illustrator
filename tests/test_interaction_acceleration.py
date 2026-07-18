from __future__ import annotations

import tracemalloc
from time import perf_counter

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.artist import Artist
from matplotlib.backend_bases import MouseEvent
from matplotlib.patches import Rectangle
from matplotlib.text import Text
from matplotlib.transforms import Bbox

from pylustrator import artist_adapters
from pylustrator.drag_helper import DragManager, GrabbableRectangleSelection
from pylustrator.editor_model import EditorScene
from pylustrator.interaction import SelectionKernel
from pylustrator.interaction_index import DisplaySpaceHitIndex
from pylustrator.snap import TargetWrapper


class _FullScanIndex:
    def invalidate(self) -> None:
        pass

    def candidate_indices(self, *_args, **_kwargs):
        return None


class _ForegroundBlocker(Artist):
    """An unsupported custom hit contract that must remain always-tested."""

    def __init__(self, bounds) -> None:
        super().__init__()
        self.bounds = tuple(float(value) for value in bounds)
        self.set_zorder(100)

    def contains(self, event):
        return Bbox.from_extents(*self.bounds).contains(event.x, event.y), {}

    def get_window_extent(self, renderer=None):
        return Bbox.from_extents(*self.bounds)


def _manager_for_figure(fig) -> DragManager:
    manager = DragManager.__new__(DragManager)
    manager.figure = fig
    manager.selected_element = None
    manager.grab_element = None
    manager._selectable_artists = []
    manager._selectable_artist_ids = set()
    manager._uneditable_artists = []
    manager._uneditable_artist_ids = set()
    manager._interaction_artists = []
    manager._interaction_artist_ids = set()
    manager._selection_parent_by_id = {}
    manager._draw_child_orders = {}
    manager.editor_scene = EditorScene(fig, ownership_parent=manager._draw_parent)
    manager.selection_kernel = SelectionKernel(
        parent_of=manager._interaction_parent,
        is_group=manager._interaction_is_group,
        label_of=manager._interaction_label,
    )
    manager.make_figure_draggable(fig)
    manager.make_axes_draggable(fig.axes)
    fig.figure_dragger = manager
    return manager


def _event(fig, x: float, y: float) -> MouseEvent:
    return MouseEvent("button_press_event", fig.canvas, x, y, button=1)


def _full_scan_stack(manager: DragManager, event: MouseEvent):
    index = manager._interaction_index
    try:
        manager._interaction_index = _FullScanIndex()
        return manager.get_hit_stack(event)
    finally:
        manager._interaction_index = index


def test_display_index_build_failure_is_atomic_and_fails_open() -> None:
    artists = [Artist(), Artist(), Artist()]
    index = DisplaySpaceHitIndex(cell_size=10)
    calls = 0

    def broken_bounds(artist):
        nonlocal calls
        calls += 1
        if artist is artists[1]:
            raise RuntimeError("synthetic build failure")
        return (0, 0, 5, 5)

    assert (
        index.candidate_indices(
            2,
            2,
            artists,
            revision=1,
            bounds_provider=broken_bounds,
        )
        is None
    )
    assert index.built_revision is None
    first_calls = calls

    # A failed revision never publishes or repeatedly rebuilds a partial map.
    assert (
        index.candidate_indices(
            2,
            2,
            artists,
            revision=1,
            bounds_provider=broken_bounds,
        )
        is None
    )
    assert calls == first_calls

    index.invalidate()
    assert index.candidate_indices(
        2,
        2,
        artists,
        revision=2,
        bounds_provider=lambda _artist: (0, 0, 5, 5),
    ) == (0, 1, 2)


def test_incremental_index_is_conservative_without_pointer_side_measurement() -> None:
    artists = tuple(Artist() for _ in range(4))
    source_ids = tuple(id(artist) for artist in artists)
    index = DisplaySpaceHitIndex(cell_size=10)
    build = index.begin_incremental_build(
        artists, revision=7, source_ids=source_ids
    )
    assert build is not None
    assert build.add_bounds((0, 0, 5, 5))
    assert build.add_bounds((20, 20, 25, 25))

    provider_calls = 0

    def pointer_must_not_measure(_artist):
        nonlocal provider_calls
        provider_calls += 1
        raise AssertionError("pointer query performed a synchronous scene build")

    # Measured remote entries may be pruned, while every not-yet-measured
    # entry remains a conservative native-hit candidate.
    assert tuple(
        index.candidate_indices(
            2,
            2,
            artists,
            revision=7,
            bounds_provider=pointer_must_not_measure,
            source_ids=source_ids,
        )
    ) == (0, 2, 3)
    assert provider_calls == 0

    assert build.add_bounds(None)
    assert build.add_bounds((0, 0, 5, 5))
    assert build.finish()
    assert index.is_current(revision=7, source_ids=source_ids)
    assert tuple(
        index.candidate_indices(
            2,
            2,
            artists,
            revision=7,
            bounds_provider=pointer_must_not_measure,
            source_ids=source_ids,
        )
    ) == (0, 2, 3)
    assert provider_calls == 0


def test_incremental_index_cancels_stale_revision_and_roster_builds() -> None:
    original = tuple(Artist() for _ in range(3))
    original_ids = tuple(id(artist) for artist in original)
    index = DisplaySpaceHitIndex(cell_size=10)
    invalidated = index.begin_incremental_build(
        original, revision=1, source_ids=original_ids
    )
    assert invalidated is not None
    assert invalidated.add_bounds((0, 0, 5, 5))

    index.invalidate()
    assert not invalidated.active
    assert not invalidated.add_bounds((10, 10, 15, 15))
    assert not invalidated.finish()
    assert index.built_revision is None
    assert index._cells == {}

    superseded = index.begin_incremental_build(
        original, revision=2, source_ids=original_ids
    )
    assert superseded is not None
    assert superseded.add_bounds((50, 50, 55, 55))
    replacement = tuple(Artist() for _ in range(2))
    replacement_ids = tuple(id(artist) for artist in replacement)
    current = index.begin_incremental_build(
        replacement, revision=3, source_ids=replacement_ids
    )
    assert current is not None
    assert not superseded.active
    assert not superseded.finish()

    provider_calls = 0

    def stale_sync_build_must_not_run(_artist):
        nonlocal provider_calls
        provider_calls += 1
        raise AssertionError("replacement query reused stale staging cells")

    # The old three-item roster cannot leak through the replacement's partial
    # query; both not-yet-measured replacement Artists remain conservative.
    assert tuple(
        index.candidate_indices(
            2,
            2,
            replacement,
            revision=3,
            bounds_provider=stale_sync_build_must_not_run,
            source_ids=replacement_ids,
        )
    ) == (0, 1)
    assert provider_calls == 0

    assert current.add_bounds((0, 0, 5, 5))
    assert current.add_bounds((20, 20, 25, 25))
    assert current.finish()
    assert index.is_current(revision=3, source_ids=replacement_ids)
    assert not index.is_current(revision=2, source_ids=original_ids)
    assert tuple(
        index.candidate_indices(
            2,
            2,
            replacement,
            revision=3,
            bounds_provider=stale_sync_build_must_not_run,
            source_ids=replacement_ids,
        )
    ) == (0,)
    assert provider_calls == 0


def test_bbox_query_is_conservative_and_keeps_unknown_envelopes() -> None:
    artists = tuple(Artist() for _ in range(6))
    bounds = (
        (0, 0, 5, 5),
        (20, 20, 25, 25),
        None,
        (9, 9, 11, 11),
        (-20, -20, -10, -10),
        (100, 100, 120, 120),
    )
    index = DisplaySpaceHitIndex(cell_size=8)
    source_ids = tuple(id(artist) for artist in artists)

    candidates = index.candidate_indices_for_bounds(
        8,
        8,
        22,
        22,
        artists,
        revision=3,
        bounds_provider=lambda artist: bounds[artists.index(artist)],
        source_ids=source_ids,
    )

    assert candidates is not None
    candidate_set = set(candidates)
    assert {1, 2, 3}.issubset(candidate_set)
    assert 0 not in candidate_set
    assert 4 not in candidate_set
    assert 5 not in candidate_set
    assert index._source_ids is source_ids


def test_100k_warm_queries_reuse_source_ids_without_large_allocations() -> None:
    artists = tuple(Artist() for _ in range(100_000))
    source_ids = tuple(id(artist) for artist in artists)
    index = DisplaySpaceHitIndex(cell_size=16)

    def provider(_artist):
        return (0, 0, 1, 1)

    assert index.candidate_indices(
        0.5,
        0.5,
        artists,
        revision=1,
        bounds_provider=provider,
        source_ids=source_ids,
    ) is not None

    tracemalloc.start()
    for _ in range(20):
        assert index.candidate_indices(
            0.5,
            0.5,
            artists,
            revision=1,
            bounds_provider=provider,
            source_ids=source_ids,
        ) is not None
    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    assert index._source_ids is source_ids
    assert peak < 1_000_000


def test_indexed_hit_stack_matches_full_scan_grid_and_artist_centers() -> None:
    fig, ax = plt.subplots(figsize=(5, 4), dpi=100)
    lower = ax.add_patch(Rectangle((0.15, 0.15), 0.7, 0.7, zorder=2, label="lower"))
    upper = ax.add_patch(Rectangle((0.3, 0.3), 0.4, 0.4, zorder=8, label="upper"))
    (line,) = ax.plot(
        [0.05, 0.95],
        [0.75, 0.25],
        linewidth=0.5,
        pickradius=10,
        zorder=12,
        label="line",
    )
    text = ax.text(0.52, 0.52, "center", zorder=14)
    fig.canvas.draw()
    blocker = _ForegroundBlocker((230, 170, 280, 220))
    fig.add_artist(blocker)
    manager = _manager_for_figure(fig)

    xs = np.linspace(0, fig.bbox.width, 17)
    ys = np.linspace(0, fig.bbox.height, 13)
    positions = [(float(x), float(y)) for x in xs for y in ys]
    for artist in (lower, upper, line, text, ax, blocker):
        bbox = artist.get_window_extent(fig.canvas.get_renderer())
        if np.all(np.isfinite(bbox.extents)):
            positions.append(
                (float((bbox.x0 + bbox.x1) / 2), float((bbox.y0 + bbox.y1) / 2))
            )

    for x, y in positions:
        event = _event(fig, x, y)
        expected = _full_scan_stack(manager, event)
        actual = manager.get_hit_stack(event)
        assert actual == expected

    # The unsupported foreground hit remains a non-editable barrier and is
    # never dropped merely because its custom contract cannot be indexed.
    blocked = manager.get_hit_stack(_event(fig, 250, 190))
    assert blocker in blocked.artists
    assert next(item for item in blocked if item.artist is blocker).editable is False
    plt.close(fig)


def test_picker_radius_is_inside_conservative_index_envelope() -> None:
    fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
    (line,) = ax.plot(
        [0.1, 0.9],
        [0.5, 0.5],
        linewidth=0.1,
        pickradius=10,
    )
    fig.canvas.draw()
    manager = _manager_for_figure(fig)
    # Registration makes an otherwise-unpickable Line2D pickable.  Restore a
    # deliberately large numeric picker to exercise the indexed tolerance.
    line.set_picker(10)
    x, y = ax.transData.transform((0.5, 0.5))
    # Eight pixels is outside the index's fixed five-pixel AA fallback but
    # comfortably inside a 10pt picker even after MouseEvent integer rounding.
    event = _event(fig, float(x), float(y + 8))

    assert line.contains(event)[0]
    assert line in manager.get_hit_stack(event).artists
    assert manager.get_hit_stack(event) == _full_scan_stack(manager, event)
    plt.close(fig)


def test_invisible_text_bbox_native_hit_remains_in_index_envelope() -> None:
    fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
    text = ax.text(
        0.5,
        0.5,
        "x",
        bbox={"boxstyle": "square,pad=4", "facecolor": "none", "edgecolor": "none"},
    )
    fig.canvas.draw()
    manager = _manager_for_figure(fig)
    patch_bbox = text.get_bbox_patch().get_window_extent(fig.canvas.get_renderer())
    event = _event(fig, patch_bbox.x0 + 2, (patch_bbox.y0 + patch_bbox.y1) / 2)

    assert text.contains(event)[0]
    assert text in manager.get_hit_stack(event).artists
    assert manager.get_hit_stack(event) == _full_scan_stack(manager, event)
    plt.close(fig)


def test_annotation_arrow_hit_far_from_text_matches_full_scan() -> None:
    fig, ax = plt.subplots(figsize=(5, 4), dpi=100)
    annotation = ax.annotate(
        "far text",
        xy=(0.12, 0.18),
        xytext=(0.86, 0.86),
        arrowprops={"arrowstyle": "->", "linewidth": 3},
        bbox={"boxstyle": "round", "facecolor": "white"},
    )
    fig.canvas.draw()
    manager = _manager_for_figure(fig)
    renderer = fig.canvas.get_renderer()
    text_bbox = Text.get_window_extent(annotation, renderer)
    arrow = annotation.arrow_patch
    display_path = arrow.get_transform().transform_path(arrow.get_path())
    vertices = np.asarray(display_path.vertices, dtype=float)
    samples = list(vertices)
    for start, end in zip(vertices[:-1], vertices[1:]):
        samples.extend(
            start * (1.0 - fraction) + end * fraction
            for fraction in np.linspace(0.0, 1.0, 11)
        )

    arrow_hits = []
    for point in samples:
        event = _event(fig, float(point[0]), float(point[1]))
        if not arrow.contains(event)[0] or text_bbox.contains(event.x, event.y):
            continue
        dx = max(text_bbox.x0 - event.x, 0.0, event.x - text_bbox.x1)
        dy = max(text_bbox.y0 - event.y, 0.0, event.y - text_bbox.y1)
        arrow_hits.append((float(np.hypot(dx, dy)), event))
    assert arrow_hits
    distance, event = max(arrow_hits, key=lambda item: item[0])
    assert distance > 100
    assert annotation.contains(event)[0]

    artists = list(manager._interaction_artists)
    indices = manager._interaction_candidate_indices(event, artists)
    assert artists.index(annotation) in indices
    assert annotation in manager.get_hit_stack(event).artists
    assert manager.get_hit_stack(event) == _full_scan_stack(manager, event)
    plt.close(fig)


def test_custom_composite_child_hit_contract_is_always_tested() -> None:
    fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
    annotation = ax.annotate(
        "custom",
        xy=(0.2, 0.2),
        xytext=(0.8, 0.8),
        arrowprops={"arrowstyle": "->"},
    )
    fig.canvas.draw()
    annotation.arrow_patch.contains = lambda _event: (True, {})
    manager = _manager_for_figure(fig)
    event = _event(fig, 2, 2)

    artists = list(manager._interaction_artists)
    indices = manager._interaction_candidate_indices(event, artists)
    assert artists.index(annotation) in indices
    assert manager._interaction_index.always_count >= 1
    assert annotation in manager.get_hit_stack(event).artists
    assert manager.get_hit_stack(event) == _full_scan_stack(manager, event)
    plt.close(fig)


def test_warm_query_reduces_native_tests_and_defers_capabilities_to_hits(
    monkeypatch,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 5), dpi=100)
    rectangles = []
    count_x, count_y = 24, 18
    for ix in range(count_x):
        for iy in range(count_y):
            rectangles.append(
                ax.add_patch(
                    Rectangle(
                        (ix / count_x, iy / count_y),
                        0.7 / count_x,
                        0.7 / count_y,
                        linewidth=0.2,
                    )
                )
            )
    fig.canvas.draw()
    manager = _manager_for_figure(fig)
    x, y = ax.transData.transform((0.51, 0.51))
    event = _event(fig, float(x), float(y))

    # Build once before instrumenting the authoritative final tests.
    manager.get_hit_stack(event)
    native_calls = 0
    original_hit = manager._is_interaction_hit

    def counted_hit(*args, **kwargs):
        nonlocal native_calls
        native_calls += 1
        return original_hit(*args, **kwargs)

    capability_calls = 0
    registry = artist_adapters.artist_adapter_registry
    original_capabilities = registry.capabilities_for

    def counted_capabilities(target):
        nonlocal capability_calls
        capability_calls += 1
        return original_capabilities(target)

    monkeypatch.setattr(manager, "_is_interaction_hit", counted_hit)
    monkeypatch.setattr(registry, "capabilities_for", counted_capabilities)
    stack = manager.get_hit_stack(event)

    assert len(manager._interaction_artists) > 400
    assert native_calls < len(manager._interaction_artists) // 5
    # Registration already established adapter support.  Dynamic selection
    # geometry/capability validation is paid only for actual visual hits.
    assert capability_calls <= len(stack)
    assert capability_calls < 10
    assert any(rectangle in stack.artists for rectangle in rectangles)
    plt.close(fig)


def test_fig2_scale_warm_hit_p95_stays_below_four_milliseconds() -> None:
    fig, ax = plt.subplots(figsize=(7, 5), dpi=100)
    for ix in range(24):
        for iy in range(18):
            ax.add_patch(
                Rectangle(
                    (ix / 24, iy / 18),
                    0.7 / 24,
                    0.7 / 18,
                    linewidth=0.2,
                )
            )
    fig.canvas.draw()
    manager = _manager_for_figure(fig)
    x, y = ax.transData.transform((0.51, 0.51))
    event = _event(fig, float(x), float(y))
    manager.get_hit_stack(event)
    roster, _editable, _order = manager._interaction_roster_snapshot()

    samples = []
    for _ in range(80):
        start = perf_counter()
        manager.get_hit_stack(event)
        samples.append(perf_counter() - start)

    assert np.percentile(samples, 95) < 0.004
    assert manager._interaction_index._source_ids is roster.source_ids
    assert manager._interaction_roster_snapshot()[0] is roster

    manager.invalidate_geometry_cache()
    assert manager._interaction_roster_snapshot()[0] is roster
    rectangle = ax.add_patch(Rectangle((0.2, 0.2), 0.01, 0.01))
    manager.make_draggable(rectangle, ax)
    assert manager._interaction_roster_snapshot()[0] is not roster
    plt.close(fig)


def test_draw_and_inventory_changes_advance_index_revision() -> None:
    fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
    fig.canvas.draw()
    manager = _manager_for_figure(fig)
    event = _event(fig, *ax.transData.transform((0.5, 0.5)))
    manager.get_hit_stack(event)
    built_revision = manager._interaction_index.built_revision
    assert built_revision is not None
    assert built_revision == manager._interaction_revision

    manager.invalidate_geometry_cache()
    assert manager._interaction_revision > built_revision
    assert manager._interaction_index.built_revision is None

    before_registration = manager._interaction_revision
    rectangle = ax.add_patch(Rectangle((0.2, 0.2), 0.1, 0.1))
    manager.make_draggable(rectangle, ax)
    assert manager._interaction_revision > before_registration
    assert rectangle in manager._interaction_artists
    plt.close(fig)


def test_preview_positions_use_owned_contiguous_arrays_for_large_lines() -> None:
    fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
    x = np.linspace(0.0, 1.0, 100_000)
    (line,) = ax.plot(x, np.sin(x))
    fig.canvas.draw()
    wrapper = TargetWrapper(line)
    selection = GrabbableRectangleSelection.__new__(GrabbableRectangleSelection)
    points = np.column_stack((x, x + 3.0))
    selection_points = np.array([[2.0, 3.0], [5.0, 7.0]])
    expected = points.copy()

    selection._set_preview_positions(wrapper, points, selection_points)
    stored = line._pylustrator_preview_positions
    stored_selection = line._pylustrator_preview_selection_points

    assert isinstance(stored, np.ndarray)
    assert stored.shape == (100_000, 2)
    assert stored.dtype == float
    assert stored.flags.c_contiguous and stored.flags.owndata
    assert stored.nbytes == expected.nbytes
    assert isinstance(stored_selection, np.ndarray)
    assert stored_selection.flags.c_contiguous and stored_selection.flags.owndata
    assert not np.shares_memory(stored, points)
    points[:] = -999
    assert np.array_equal(stored, expected)
    assert np.array_equal(wrapper.get_positions(), expected)

    selection._clear_preview(wrapper)
    assert not hasattr(line, "_pylustrator_preview_positions")
    assert not hasattr(line, "_pylustrator_preview_selection_points")
    plt.close(fig)
