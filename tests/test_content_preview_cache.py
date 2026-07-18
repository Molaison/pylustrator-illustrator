from __future__ import annotations

from copy import deepcopy
from time import perf_counter
import tracemalloc
import weakref

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.artist import Artist
from matplotlib.backends.backend_agg import RendererAgg as MatplotlibRendererAgg
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from qtpy import QtCore, QtWidgets

from pylustrator.content_preview_cache import (
    ContentPreviewCache,
    ContentPreviewToken,
    ContentPreviewUnavailable,
    artist_source_fingerprint,
    close_content_preview_cache,
    ensure_content_preview_cache,
    invalidate_content_preview_cache,
)
from pylustrator import content_preview_cache as preview_cache_module
from pylustrator.components.plot_layout import selection_scene_transform
from pylustrator.drag_helper import (
    DIR_X0,
    DIR_X1,
    DIR_Y0,
    DIR_Y1,
    DragManager,
    GrabbableRectangleSelection,
)
from pylustrator.editor_model import EditorGroup
from pylustrator.snap import TargetWrapper


class _Signal:
    def __init__(self) -> None:
        self.calls = []

    def emit(self, *args) -> None:
        self.calls.append(args)


class _Signals:
    def __init__(self) -> None:
        self.figure_selection_moved = _Signal()
        self.figure_selection_update = _Signal()
        self.figure_selection_property_changed = _Signal()


class _Tracker:
    def __init__(self) -> None:
        self.edits = []
        self.last_edit = -1
        self.changes = []
        self.saved = True

    def addEdit(self, edit) -> None:
        self.edits.append(edit)
        self.last_edit = len(self.edits) - 1

    def capture_recording_state(self):
        return list(self.changes), bool(self.saved)

    def restore_recording_state(self, state) -> None:
        self.changes, self.saved = list(state[0]), bool(state[1])

    def addChange(self, target, command) -> None:
        self.changes.append((target, command))

    def addNewAxesChange(self, target) -> None:
        self.changes.append((target, "axes"))

    def addNewLegendChange(self, target) -> None:
        self.changes.append((target, "legend"))

    def addNewTextChange(self, target) -> None:
        self.changes.append((target, "text"))


class _Manager:
    _draw_parent = DragManager._draw_parent
    _paint_order_key = DragManager._paint_order_key

    def __init__(self, figure) -> None:
        self.figure = figure
        self._interaction_revision = 0
        self.selected_element = None
        self._selection_refresh_on_draw = False
        self._selection_parent_by_id = {
            id(axes): figure for axes in getattr(figure, "axes", ())
        }
        self._interaction_artists = []
        self._draw_child_orders = {}

    def _invalidate_interaction_index(self) -> None:
        self._interaction_revision += 1
        invalidate_content_preview_cache(self, "test-revision")


class _SceneView:
    h = 300
    device_pixel_ratio = 1.0


def _selection_for(figure, artist):
    scene = QtWidgets.QGraphicsScene()
    origin = QtWidgets.QGraphicsRectItem()
    origin.view = _SceneView()
    scene.addItem(origin)
    figure._pyl_scene = origin
    figure._preview_test_scene = scene
    figure.signals = _Signals()
    figure.change_tracker = _Tracker()
    manager = _Manager(figure)
    selection = GrabbableRectangleSelection(figure, origin)
    manager.selection = selection
    figure.figure_dragger = manager
    figure.selection = selection
    manager.selected_element = artist
    owner = getattr(artist, "axes", None) or getattr(artist, "figure", None)
    if owner is not None and owner is not artist:
        manager._selection_parent_by_id[id(artist)] = owner
    manager._interaction_artists = [
        figure,
        *getattr(figure, "axes", ()),
        artist,
    ]
    selection.add_target(artist)
    return manager, selection


@pytest.fixture
def qapp():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture
def line_selection(qapp):
    fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
    line = ax.plot([0.1, 0.5, 0.9], [0.2, 0.8, 0.3], marker="o")[0]
    line.set_clip_on(False)
    fig.canvas.draw()
    manager, selection = _selection_for(fig, line)
    yield fig, line, manager, selection
    selection.clear_targets()
    close_content_preview_cache(manager)
    plt.close(fig)
    assert qapp is not None


def _mapped_item_bounds(entry) -> np.ndarray:
    bounds = entry.item.mapRectToParent(entry.item.boundingRect())
    return np.array(
        [bounds.left(), bounds.top(), bounds.right(), bounds.bottom()], dtype=float
    )


def _mapped_root_bounds(entry) -> np.ndarray:
    bounds = entry.root.mapRectToParent(
        entry.item.mapRectToParent(entry.item.boundingRect())
    )
    return np.array(
        [bounds.left(), bounds.top(), bounds.right(), bounds.bottom()], dtype=float
    )


def test_cache_token_binds_revision_renderer_selection_and_in_place_source(
    line_selection,
) -> None:
    fig, line, manager, selection = line_selection
    renderer = fig.canvas.get_renderer()
    artists = (line,)
    token = ContentPreviewToken.capture(
        manager,
        artists,
        renderer,
        source_byte_limit=selection.content_preview_source_budget_bytes,
    )
    assert token.is_current(
        manager,
        artists,
        renderer,
        source_byte_limit=selection.content_preview_source_budget_bytes,
    )

    manager._interaction_revision += 1
    assert not token.is_current(
        manager,
        artists,
        renderer,
        source_byte_limit=selection.content_preview_source_budget_bytes,
    )
    manager._interaction_revision -= 1
    assert not token.is_current(
        manager,
        artists,
        object(),
        source_byte_limit=selection.content_preview_source_budget_bytes,
    )

    other = Line2D([0, 1], [1, 0])
    assert not token.is_current(
        manager,
        (other,),
        renderer,
        source_byte_limit=selection.content_preview_source_budget_bytes,
    )

    raw_y = line.get_ydata(orig=True)
    assert not line.stale
    raw_y[1] += 0.125
    assert not line.stale
    assert not token.is_current(
        manager,
        artists,
        renderer,
        source_byte_limit=selection.content_preview_source_budget_bytes,
    )


def test_budget_rejection_happens_before_renderer_allocation(
    line_selection, monkeypatch
) -> None:
    _fig, _line, manager, selection = line_selection
    selection.content_preview_memory_budget_bytes = 1
    cache = ContentPreviewCache(memory_budget_bytes=1)

    def forbidden_renderer(*_args, **_kwargs):
        raise AssertionError("budget fallback allocated a renderer")

    monkeypatch.setattr(
        "pylustrator.content_preview_cache.RendererAgg", forbidden_renderer
    )
    assert not cache.warm_now(manager, selection)
    assert cache.last_fallback_reason == "memory-budget:renderer"
    assert cache.entry is None


def test_large_artist_falls_back_in_bounded_memory_before_hash_or_render(
    qapp, monkeypatch
) -> None:
    x = np.linspace(0.0, 1.0, 100_000)
    line = Line2D(x, np.sin(x))
    line.stale = False
    tracemalloc.start()
    with pytest.raises(ContentPreviewUnavailable, match="source-budget"):
        artist_source_fingerprint(line, byte_limit=512 * 1024)
    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert peak < 256 * 1024

    fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
    selected = ax.add_line(line)
    selected.set_clip_on(False)
    fig.canvas.draw()
    manager, selection = _selection_for(fig, selected)

    def forbidden_renderer(*_args, **_kwargs):
        raise AssertionError("large-Artist fallback rendered")

    monkeypatch.setattr(
        "pylustrator.content_preview_cache.RendererAgg", forbidden_renderer
    )
    cache = ensure_content_preview_cache(manager)
    assert not cache.warm_now(manager, selection)
    assert cache.last_fallback_reason.startswith("source-budget:")
    selection.clear_targets()
    close_content_preview_cache(manager)
    plt.close(fig)
    assert qapp is not None


def test_clone_accounting_rejects_near_budget_line_before_renderer(
    qapp, monkeypatch
) -> None:
    fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
    x = np.linspace(0.1, 0.9, 30_000)
    line = ax.plot(x, 0.5 + 0.2 * np.sin(x * 20.0))[0]
    line.set_clip_on(False)
    fig.canvas.draw()
    manager, selection = _selection_for(fig, line)
    selection.content_preview_memory_budget_bytes = 3 * 1024 * 1024

    def forbidden_renderer(*_args, **_kwargs):
        raise AssertionError("near-budget clone reached renderer allocation")

    monkeypatch.setattr(
        "pylustrator.content_preview_cache.RendererAgg", forbidden_renderer
    )
    cache = ensure_content_preview_cache(manager)
    tracemalloc.start()
    assert not cache.warm_now(manager, selection)
    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert cache.last_fallback_reason == "memory-budget:renderer"
    assert peak < selection.content_preview_memory_budget_bytes
    assert cache.entry is None
    selection.clear_targets()
    close_content_preview_cache(manager)
    plt.close(fig)
    assert qapp is not None


def test_nested_scalar_budget_never_builds_unbounded_repr_temporaries() -> None:
    huge = "x" * (4 * 1024 * 1024)
    builder = preview_cache_module._FingerprintBuilder(512 * 1024)
    tracemalloc.start()
    with pytest.raises(ContentPreviewUnavailable, match="source-budget"):
        builder.scalar("font-family", [huge])
    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert peak < 256 * 1024
    assert builder.source_bytes <= builder.byte_limit

    called = False

    class DangerousRepr:
        def __repr__(self):
            nonlocal called
            called = True
            raise AssertionError("custom repr must never run")

    builder = preview_cache_module._FingerprintBuilder(512 * 1024)
    with pytest.raises(ContentPreviewUnavailable, match="opaque-source"):
        builder.scalar("nested", {"value": [DangerousRepr()]})
    assert not called


def test_scalar_mappable_collection_falls_back_without_live_state_mutation(
    qapp, monkeypatch
) -> None:
    fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
    collection = ax.scatter(
        [0.2, 0.5, 0.8],
        [0.3, 0.7, 0.4],
        c=[0.0, 0.5, 1.0],
        cmap="viridis",
    )
    collection.set_clip_on(False)
    fig.canvas.draw()
    collection.set_clim(0.0, 2.0)
    manager, selection = _selection_for(fig, collection)
    facecolors = collection.get_facecolors().copy()
    edgecolors = collection.get_edgecolors().copy()
    clim = collection.get_clim()
    stale_state = (
        collection.stale,
        ax.stale,
        fig.stale,
    )

    def forbidden_renderer(*_args, **_kwargs):
        raise AssertionError("unsafe scalar mapping reached private draw")

    monkeypatch.setattr(
        "pylustrator.content_preview_cache.RendererAgg", forbidden_renderer
    )
    cache = ensure_content_preview_cache(manager)
    assert not cache.warm_now(manager, selection)
    assert cache.last_fallback_reason == "scalar-mappable-unsupported"
    assert np.array_equal(collection.get_facecolors(), facecolors)
    assert np.array_equal(collection.get_edgecolors(), edgecolors)
    assert collection.get_clim() == clim
    assert (collection.stale, ax.stale, fig.stale) == stale_state
    selection.clear_targets()
    close_content_preview_cache(manager)
    plt.close(fig)
    assert qapp is not None


def test_unsupported_no_application_and_capture_exception_all_fail_open(
    line_selection, monkeypatch
) -> None:
    _fig, line, manager, selection = line_selection

    class CustomArtist(Artist):
        def draw(self, renderer):
            raise AssertionError

    with pytest.raises(ContentPreviewUnavailable, match="unsupported-artist"):
        artist_source_fingerprint(CustomArtist())

    cache = ContentPreviewCache()
    monkeypatch.setattr(
        "pylustrator.content_preview_cache._qt_application_ready", lambda: False
    )
    assert not cache.warm_now(manager, selection)
    assert cache.last_fallback_reason == "no-qapplication"

    monkeypatch.setattr(
        "pylustrator.content_preview_cache._qt_application_ready", lambda: True
    )

    called = False

    def broken_draw(_renderer):
        nonlocal called
        called = True
        raise RuntimeError("synthetic draw failure")

    line.draw = broken_draw
    assert not cache.warm_now(manager, selection)
    assert cache.last_fallback_reason == "unsupported-artist"
    assert not called
    del line.draw

    class BrokenRenderer(MatplotlibRendererAgg):
        def draw_path(self, *_args, **_kwargs):
            raise RuntimeError("synthetic renderer failure")

    monkeypatch.setattr(
        "pylustrator.content_preview_cache.RendererAgg", BrokenRenderer
    )
    live_state = (
        id(getattr(line, "_transformed_path", None)),
        line.get_path().vertices.copy(),
        line.stale,
        manager.figure.stale,
    )
    assert not cache.warm_now(manager, selection)
    assert cache.last_fallback_reason == "capture-failed"
    assert cache.entry is None
    assert id(getattr(line, "_transformed_path", None)) == live_state[0]
    assert np.array_equal(line.get_path().vertices, live_state[1])
    assert line.stale is live_state[2]
    assert manager.figure.stale is live_state[3]


def test_custom_class_and_instance_getters_never_execute_user_callbacks(
    line_selection,
) -> None:
    _fig, line, manager, selection = line_selection
    called = []

    class CustomLine(Line2D):
        def get_color(self):
            called.append("class-getter")
            return super().get_color()

    custom = CustomLine([0.0, 1.0], [0.0, 1.0])
    custom.stale = False
    with pytest.raises(ContentPreviewUnavailable, match="unsupported-artist"):
        artist_source_fingerprint(custom)
    assert called == []

    def instance_getter():
        called.append("instance-getter")
        return "black"

    line.get_color = instance_getter
    cache = ensure_content_preview_cache(manager)
    assert not cache.warm_now(manager, selection)
    assert cache.last_fallback_reason == "instance-method-override"
    assert called == []
    del line.get_color

    class DeepcopyBomb:
        def __deepcopy__(self, memo):
            called.append("deepcopy")
            raise AssertionError("user __deepcopy__ must not execute")

    line.user_metadata = {"payload": DeepcopyBomb()}
    assert cache.warm_now(manager, selection), cache.last_fallback_reason
    assert called == []
    del line.user_metadata


def test_capture_reuses_one_scratch_renderer_and_releases_shallow_clones(
    line_selection, monkeypatch
) -> None:
    _fig, _line, manager, selection = line_selection
    renderer_allocations = 0
    original_renderer = preview_cache_module.RendererAgg
    original_clone = preview_cache_module._disposable_draw_clone
    clone_refs = []

    class CountingRenderer(original_renderer):
        def __init__(self, *args, **kwargs):
            nonlocal renderer_allocations
            renderer_allocations += 1
            super().__init__(*args, **kwargs)

    def recorded_clone(artist):
        clone = original_clone(artist)
        clone_refs.append(weakref.ref(clone))
        return clone

    monkeypatch.setattr(preview_cache_module, "RendererAgg", CountingRenderer)
    monkeypatch.setattr(
        preview_cache_module, "_disposable_draw_clone", recorded_clone
    )
    cache = ensure_content_preview_cache(manager)
    roots = []
    for _index in range(12):
        assert cache.warm_now(manager, selection), cache.last_fallback_reason
        roots.append(cache.entry.root)
        assert all(reference() is None for reference in clone_refs)

    assert renderer_allocations == 1
    assert all(root.scene() is None for root in roots[:-1])
    assert roots[-1].scene() is not None


def test_large_canvas_idle_budget_falls_back_before_clone_or_private_renderer(
    qapp, monkeypatch
) -> None:
    fig = plt.figure(figsize=(20, 15), dpi=100)
    text = fig.text(0.5, 0.5, "large canvas")
    fig.canvas.draw()
    manager, selection = _selection_for(fig, text)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("idle-work fallback reached expensive capture work")

    monkeypatch.setattr(
        preview_cache_module, "_disposable_draw_clone", forbidden
    )
    monkeypatch.setattr(preview_cache_module, "RendererAgg", forbidden)
    cache = ensure_content_preview_cache(manager)
    assert not cache.warm_now(manager, selection)
    assert cache.last_fallback_reason == "idle-work-budget"
    selection.clear_targets()
    close_content_preview_cache(manager)
    plt.close(fig)
    assert qapp is not None


def test_pending_line_shallow_capture_preserves_every_live_derived_cache(
    qapp,
) -> None:
    fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
    line = ax.plot([0.1, 0.5, 0.9], [0.2, 0.8, 0.3], marker="o")[0]
    line.set_clip_on(False)
    fig.canvas.draw()
    line.set_ydata([0.7, 0.15, 0.65])
    manager, selection = _selection_for(fig, line)
    # Selection geometry is allowed to finalize a pending Line.  Capture must
    # preserve the state that exists once the idle job itself begins.
    line.set_ydata([0.65, 0.2, 0.6])
    before = {
        "invalidx": line._invalidx,
        "invalidy": line._invalidy,
        "xy_id": id(line._xy),
        "xy": line._xy.copy(),
        "path_id": id(line._path),
        "path": line._path.vertices.copy(),
        "transformed_path_id": id(line._transformed_path),
        "stale": line.stale,
        "axes_stale": ax.stale,
        "figure_stale": fig.stale,
    }
    cache = ensure_content_preview_cache(manager)
    assert cache.warm_now(manager, selection), cache.last_fallback_reason
    assert line._invalidx is before["invalidx"]
    assert line._invalidy is before["invalidy"]
    assert id(line._xy) == before["xy_id"]
    assert np.array_equal(line._xy, before["xy"])
    assert id(line._path) == before["path_id"]
    assert np.array_equal(line._path.vertices, before["path"])
    assert id(line._transformed_path) == before["transformed_path_id"]
    assert line.stale is before["stale"]
    assert ax.stale is before["axes_stale"]
    assert fig.stale is before["figure_stale"]
    selection.clear_targets()
    close_content_preview_cache(manager)
    plt.close(fig)
    assert qapp is not None


def test_shallow_legend_leaf_replay_is_pixel_identical_to_standard_draw(
    qapp,
) -> None:
    fig, ax = plt.subplots(figsize=(6, 4), dpi=100)
    ax.plot([0.1, 0.9], [0.2, 0.8], marker="o", label="line")
    ax.scatter([0.25, 0.75], [0.7, 0.3], label="points")
    ax.bar([0.45], [0.4], width=0.12, label="bar")
    legend = ax.legend(frameon=True, title="Kinds", ncols=3)
    fig.canvas.draw()
    manager, selection = _selection_for(fig, legend)
    cache = ensure_content_preview_cache(manager)
    assert cache.warm_now(manager, selection), cache.last_fallback_reason
    observed = np.asarray(cache._capture_renderer.buffer_rgba()).copy()

    renderer = fig.canvas.get_renderer()
    reference_renderer = MatplotlibRendererAgg(
        int(renderer.width), int(renderer.height), float(renderer.dpi)
    )
    reference_renderer.clear()
    memo = {id(fig): fig, id(ax): ax}
    if legend.get_draggable() is not None:
        memo[id(legend.get_draggable())] = None
    reference = deepcopy(legend, memo)
    for descendant in reference.findobj():
        descendant.stale_callback = None
        descendant._remove_method = None
    reference.draw(reference_renderer)
    expected = np.asarray(reference_renderer.buffer_rgba())

    assert np.array_equal(observed[..., 3], expected[..., 3])
    observed_premultiplied = (
        observed[..., :3].astype(np.uint16)
        * observed[..., 3, None].astype(np.uint16)
    )
    expected_premultiplied = (
        expected[..., :3].astype(np.uint16)
        * expected[..., 3, None].astype(np.uint16)
    )
    assert np.array_equal(observed_premultiplied, expected_premultiplied)
    selection.clear_targets()
    close_content_preview_cache(manager)
    plt.close(fig)
    assert qapp is not None


def test_idle_capture_preserves_artist_state_and_pixmap_orientation(
    line_selection,
) -> None:
    _fig, line, manager, selection = line_selection
    cache = ensure_content_preview_cache(manager)
    stale_before = line.stale
    assert cache.warm_now(manager, selection), cache.last_fallback_reason
    assert line.stale is stale_before
    entry = cache.entry
    assert entry is not None
    assert entry.peak_bytes <= cache.memory_budget_bytes
    assert entry.retained_bytes <= cache.memory_budget_bytes
    assert np.allclose(_mapped_item_bounds(entry), entry.display_bounds, atol=0.0)
    assert not entry.root.isVisible()


def test_successful_clone_capture_does_not_mutate_live_derived_state(qapp) -> None:
    fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
    annotation = ax.annotate(
        "safe clone",
        xy=(0.75, 0.8),
        xytext=(0.3, 0.45),
        annotation_clip=False,
        arrowprops={"arrowstyle": "->", "color": "black"},
    )
    fig.canvas.draw()
    manager, selection = _selection_for(fig, annotation)
    arrow = annotation.arrow_patch
    before = {
        "annotation_stale": annotation.stale,
        "arrow_stale": arrow.stale,
        "axes_stale": ax.stale,
        "figure_stale": fig.stale,
        "arrow_positions": tuple(np.asarray(item).copy() for item in arrow._posA_posB),
        "arrow_path": arrow.get_path().vertices.copy(),
        "annotation_position": annotation.get_position(),
    }
    cache = ensure_content_preview_cache(manager)
    assert cache.warm_now(manager, selection), cache.last_fallback_reason
    assert annotation.stale is before["annotation_stale"]
    assert arrow.stale is before["arrow_stale"]
    assert ax.stale is before["axes_stale"]
    assert fig.stale is before["figure_stale"]
    assert annotation.get_position() == before["annotation_position"]
    assert all(
        np.array_equal(current, original)
        for current, original in zip(arrow._posA_posB, before["arrow_positions"])
    )
    assert np.array_equal(arrow.get_path().vertices, before["arrow_path"])
    selection.clear_targets()
    close_content_preview_cache(manager)
    plt.close(fig)
    assert qapp is not None


def test_standard_legend_and_editor_group_receive_content_ghosts(qapp) -> None:
    fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
    first = ax.plot([0, 1], [0.2, 0.8], marker="o", label="first")[0]
    second = ax.plot([0, 1], [0.8, 0.2], marker="s", label="second")[0]
    legend = ax.legend(frameon=True)
    fig.canvas.draw()
    manager, selection = _selection_for(fig, legend)
    cache = ensure_content_preview_cache(manager)
    assert cache.warm_now(manager, selection), cache.last_fallback_reason
    assert cache.entry is not None and cache.entry.retained_bytes > 0

    # A child source change invalidates the composite token even if the
    # selected identity remains the Legend itself.
    legend.get_texts()[0].set_text("changed")
    assert not cache.activate(manager, selection)
    assert cache.last_fallback_reason == "stale-token"
    selection.clear_targets()
    close_content_preview_cache(manager)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
    red = ax.add_patch(Rectangle((0.15, 0.2), 0.25, 0.3, color="red"))
    blue = ax.add_patch(Rectangle((0.55, 0.3), 0.25, 0.3, color="blue"))
    red.set_clip_on(False)
    blue.set_clip_on(False)
    fig.canvas.draw()
    group = EditorGroup(fig, "preview-group", [red, blue], name="Pair", owner=ax)
    manager, selection = _selection_for(fig, group)
    manager._selection_parent_by_id.update({id(red): ax, id(blue): ax})
    manager._interaction_artists.extend([red, blue])
    cache = ensure_content_preview_cache(manager)
    assert group._transform is None
    assert cache.warm_now(manager, selection), cache.last_fallback_reason
    assert group._transform is None
    assert cache.entry is not None and cache.entry.retained_bytes > 0
    selection.start_move()
    selection.addOffset((6.0, -4.0), DIR_X0 | DIR_X1 | DIR_Y0 | DIR_Y1)
    assert cache.entry.root.isVisible()
    selection.has_moved = False
    selection.end_move()
    selection.clear_targets()
    close_content_preview_cache(manager)
    plt.close(fig)
    assert first is not second
    assert qapp is not None


def test_first_annotation_selection_warms_even_if_geometry_marks_it_stale(
    qapp,
) -> None:
    fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
    annotation = ax.annotate(
        "peak",
        xy=(0.7, 0.8),
        xytext=(0.35, 0.45),
        arrowprops={"arrowstyle": "->", "color": "black"},
        annotation_clip=False,
    )
    fig.canvas.draw()
    assert not annotation.stale
    manager, selection = _selection_for(fig, annotation)
    # Annotation selection geometry may refresh its arrow patch and propagate
    # stale=True even though renderer/revision/source are still coherent.
    cache = ensure_content_preview_cache(manager)
    assert cache.warm_now(manager, selection), cache.last_fallback_reason
    assert cache.entry is not None and cache.entry.retained_bytes > 0
    assert cache.activate(manager, selection)
    selection.clear_targets()
    close_content_preview_cache(manager)
    plt.close(fig)
    assert qapp is not None


@pytest.mark.parametrize("annotation_clip", [None, True])
def test_implicit_annotation_clip_falls_back_before_renderer(
    qapp, monkeypatch, annotation_clip
) -> None:
    fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
    annotation = ax.annotate(
        "boundary",
        xy=(0.8, 0.8),
        xytext=(0.45, 0.5),
        xycoords="data",
        annotation_clip=annotation_clip,
        arrowprops={"arrowstyle": "->"},
    )
    fig.canvas.draw()
    manager, selection = _selection_for(fig, annotation)

    def forbidden_renderer(*_args, **_kwargs):
        raise AssertionError("implicit Annotation clip reached raster allocation")

    monkeypatch.setattr(
        "pylustrator.content_preview_cache.RendererAgg", forbidden_renderer
    )
    cache = ensure_content_preview_cache(manager)
    assert not cache.warm_now(manager, selection)
    assert cache.last_fallback_reason == "annotation-clip"
    selection.clear_targets()
    close_content_preview_cache(manager)
    plt.close(fig)
    assert qapp is not None


@pytest.mark.parametrize("clip_kind", ["rectangular", "path"])
def test_fixed_scene_clips_use_analytic_preview_without_raster_allocation(
    qapp, monkeypatch, clip_kind
) -> None:
    from matplotlib.patches import Circle
    fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
    line = ax.plot([-0.4, 0.5, 1.4], [0.3, 0.7, 0.4], linewidth=5)[0]
    if clip_kind == "rectangular":
        line.set_clip_box(ax.bbox)
    else:
        line.set_clip_path(Circle((0.5, 0.5), 0.35, transform=ax.transData))
    line.set_clip_on(True)
    fig.canvas.draw()
    manager, selection = _selection_for(fig, line)

    def forbidden_renderer(*_args, **_kwargs):
        raise AssertionError("fixed clip reached raster allocation")

    monkeypatch.setattr(
        "pylustrator.content_preview_cache.RendererAgg", forbidden_renderer
    )
    cache = ensure_content_preview_cache(manager)
    assert not cache.warm_now(manager, selection)
    assert cache.last_fallback_reason == "active-clip"

    before = TargetWrapper(line).get_positions().copy()
    selection.start_move()
    selection.addOffset((7.0, -4.0), DIR_X0 | DIR_X1 | DIR_Y0 | DIR_Y1)
    selection.has_moved = True
    selection.end_move()
    after = TargetWrapper(line).get_positions()
    assert np.allclose(after - before, [7.0, -4.0], atol=1e-9)
    selection.clear_targets()
    close_content_preview_cache(manager)
    plt.close(fig)
    assert qapp is not None


def test_canvas_edge_source_and_destination_fall_back_to_analytic(qapp) -> None:
    fig = plt.figure(figsize=(4, 3), dpi=100)
    edge = fig.text(0.0, 0.0, "edge", fontsize=14)
    fig.canvas.draw()
    manager, selection = _selection_for(fig, edge)
    cache = ensure_content_preview_cache(manager)
    assert not cache.warm_now(manager, selection)
    assert cache.last_fallback_reason in {
        "canvas-edge-source",
        "canvas-edge-paint",
    }
    selection.clear_targets()
    close_content_preview_cache(manager)
    plt.close(fig)

    fig = plt.figure(figsize=(4, 3), dpi=100)
    interior = fig.text(0.5, 0.5, "inside", fontsize=14)
    fig.canvas.draw()
    manager, selection = _selection_for(fig, interior)
    cache = ensure_content_preview_cache(manager)
    assert cache.warm_now(manager, selection), cache.last_fallback_reason
    assert cache.activate(manager, selection)
    assert not cache.update_transform(
        np.array([[1.0, 0.0, -250.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    )
    assert cache.last_fallback_reason == "canvas-edge-destination"
    assert not cache.entry.root.isVisible()
    selection.clear_targets()
    close_content_preview_cache(manager)
    plt.close(fig)
    assert qapp is not None


def test_resize_with_fixed_appearance_never_scales_cached_bitmap(qapp) -> None:
    fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
    patch = ax.add_patch(
        Rectangle(
            (0.25, 0.3),
            0.35,
            0.3,
            facecolor="none",
            edgecolor="black",
            linewidth=12.0,
        )
    )
    patch.set_clip_on(False)
    fig.canvas.draw()
    manager, selection = _selection_for(fig, patch)
    cache = ensure_content_preview_cache(manager)
    assert cache.warm_now(manager, selection), cache.last_fallback_reason
    assert cache.entry is not None
    selection.start_move()
    selection.addOffset((12.0, 8.0), DIR_X1 | DIR_Y1, keep_aspect_ratio=False)
    assert cache.last_fallback_reason == "translation-only"
    assert not cache.entry.root.isVisible()
    selection.has_moved = False
    selection.end_move()
    selection.clear_targets()
    close_content_preview_cache(manager)
    plt.close(fig)
    assert qapp is not None


def test_same_z_multi_selection_uses_authoritative_paint_order(qapp) -> None:
    fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
    red = ax.add_patch(
        Rectangle((0.2, 0.2), 0.6, 0.6, facecolor="red", edgecolor="none")
    )
    blue = ax.add_patch(
        Rectangle((0.2, 0.2), 0.6, 0.6, facecolor="blue", edgecolor="none")
    )
    red.set_zorder(5)
    blue.set_zorder(5)
    red.set_clip_on(False)
    blue.set_clip_on(False)
    fig.canvas.draw()
    # Deliberately reverse selection order relative to Axes paint order.
    manager, selection = _selection_for(fig, blue)
    manager._selection_parent_by_id[id(red)] = ax
    manager._interaction_artists.append(red)
    selection.add_target(red)
    cache = ensure_content_preview_cache(manager)
    assert cache.warm_now(manager, selection), cache.last_fallback_reason
    image = cache.entry.pixmap.toImage()
    color = image.pixelColor(image.width() // 2, image.height() // 2)
    assert color.blue() > 240
    assert color.red() < 15
    selection.clear_targets()
    close_content_preview_cache(manager)
    plt.close(fig)
    assert qapp is not None


def test_large_legend_falls_back_before_fingerprint_or_renderer_allocation(
    qapp, monkeypatch
) -> None:
    fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
    for index in range(150):
        ax.plot(
            [0.0, 1.0],
            [float(index), float(index) + 0.5],
            label=f"case {index}",
        )
    legend = ax.legend(ncol=5, frameon=True)
    fig.canvas.draw()
    manager, selection = _selection_for(fig, legend)

    def forbidden_renderer(*_args, **_kwargs):
        raise AssertionError("oversized Legend reached renderer allocation")

    monkeypatch.setattr(
        "pylustrator.content_preview_cache.RendererAgg", forbidden_renderer
    )
    cache = ensure_content_preview_cache(manager)
    assert not cache.warm_now(manager, selection)
    assert cache.last_fallback_reason in {
        "artist-count-budget",
        "composite-node-budget",
    }
    assert cache.entry is None
    selection.clear_targets()
    close_content_preview_cache(manager)
    plt.close(fig)
    assert qapp is not None


def test_pointer_press_and_motion_never_capture_or_render(
    line_selection, monkeypatch
) -> None:
    _fig, _line, manager, selection = line_selection
    cache = ensure_content_preview_cache(manager)
    assert cache.warm_now(manager, selection)
    captures = cache.capture_count

    def forbidden_capture(*_args, **_kwargs):
        raise AssertionError("pointer path attempted a capture")

    monkeypatch.setattr(cache, "_capture_entry", forbidden_capture)
    monkeypatch.setattr(
        "pylustrator.content_preview_cache.RendererAgg", forbidden_capture
    )
    selection.start_move()
    assert cache.active
    selection.addOffset(
        (13.0, -7.0), DIR_X0 | DIR_X1 | DIR_Y0 | DIR_Y1
    )
    assert cache.capture_count == captures
    assert cache.motion_update_count == 1
    assert cache.entry is not None and cache.entry.root.isVisible()
    selection.has_moved = False
    selection.end_move()


def test_cached_content_follows_translation_and_resize_falls_back(
    line_selection,
) -> None:
    _fig, _line, manager, selection = line_selection
    cache = ensure_content_preview_cache(manager)
    assert cache.warm_now(manager, selection)
    entry = cache.entry
    assert entry is not None
    original = np.asarray(entry.display_bounds, dtype=float)

    selection.start_move()
    selection.addOffset(
        (11.0, -6.0), DIR_X0 | DIR_X1 | DIR_Y0 | DIR_Y1
    )
    assert np.allclose(
        _mapped_root_bounds(entry),
        original + np.array([11.0, -6.0, 11.0, -6.0]),
        atol=1e-9,
    )
    selection.has_moved = False
    selection.end_move()

    assert cache.warm_now(manager, selection)
    entry = cache.entry
    assert entry is not None
    assert cache.activate(manager, selection)
    x0, y0, x1, y1 = selection.positions
    transform = np.array(
        [
            [(x1 - x0 + 9.0) / (x1 - x0), 0.0, 0.0],
            [0.0, (y1 - y0 + 5.0) / (y1 - y0), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    transform[0, 2] = x0 - transform[0, 0] * x0
    transform[1, 2] = y0 - transform[1, 1] * y0
    assert not cache.update_transform(transform)
    assert cache.last_fallback_reason == "translation-only"
    assert not entry.root.isVisible()


def test_commit_and_undo_geometry_are_unchanged_and_cache_is_disposable(
    qapp,
) -> None:
    fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
    patch = ax.add_patch(Rectangle((0.2, 0.3), 0.25, 0.2))
    patch.set_clip_on(False)
    fig.canvas.draw()
    manager, selection = _selection_for(fig, patch)
    cache = ensure_content_preview_cache(manager)
    assert cache.warm_now(manager, selection)
    entry = cache.entry
    assert entry is not None
    root = entry.root
    before = TargetWrapper(patch).get_selection_points().copy()

    selection.start_move()
    selection.addOffset(
        (8.0, -3.0), DIR_X0 | DIR_X1 | DIR_Y0 | DIR_Y1
    )
    preview = np.asarray(
        patch._pylustrator_preview_selection_points, dtype=float
    ).copy()
    selection.has_moved = True
    selection.end_move()
    committed = TargetWrapper(patch).get_selection_points()
    assert np.max(np.abs(committed - preview)) < 0.25
    assert np.allclose(committed - before, [8.0, -3.0], atol=1e-9)
    assert cache.entry is None
    assert root.scene() is None

    edit = fig.change_tracker.edits[-1]
    edit[0]()
    restored = TargetWrapper(patch).get_selection_points()
    assert np.max(np.abs(restored - before)) < 0.25
    selection.clear_targets()
    close_content_preview_cache(manager)
    plt.close(fig)
    assert qapp is not None


def test_clear_cancel_and_close_release_scene_items(line_selection) -> None:
    fig, _line, manager, selection = line_selection
    cache = ensure_content_preview_cache(manager)
    assert cache.warm_now(manager, selection)
    cancelled_root = cache.entry.root
    selection.start_move()
    selection._restore_move_start()
    selection._clear_move_transaction()
    assert cache.entry is not None
    assert cancelled_root.scene() is not None
    assert not cancelled_root.isVisible()

    cleared_root = cache.entry.root
    selection.clear_targets()
    assert cache.entry is None
    assert cleared_root.scene() is None

    selection.add_target(_line)
    assert cache.warm_now(manager, selection)
    closed_root = cache.entry.root
    close_content_preview_cache(manager)
    assert cache.entry is None
    assert closed_root.scene() is None


def test_noop_release_rewarms_content_cache_without_waiting_for_draw(
    line_selection, qapp
) -> None:
    _fig, _line, manager, selection = line_selection
    cache = ensure_content_preview_cache(manager)
    assert cache.warm_now(manager, selection)
    capture_count = cache.capture_count
    entry = cache.entry
    selection.start_move()
    selection.has_moved = False
    selection.releasedEvent(None)
    assert cache.entry is entry
    assert cache.capture_count == capture_count
    assert not entry.root.isVisible()


def test_ready_first_frame_and_warm_motion_stay_below_interaction_budgets(
    line_selection,
) -> None:
    _fig, _line, manager, selection = line_selection
    cache = ensure_content_preview_cache(manager)
    assert cache.warm_now(manager, selection)

    started = perf_counter()
    assert cache.activate(manager, selection)
    assert cache.update_transform(np.eye(3))
    first_frame_ms = (perf_counter() - started) * 1000.0
    assert first_frame_ms < 16.7

    samples = []
    for index in range(200):
        matrix = np.array(
            [[1.0, 0.0, index * 0.1], [0.0, 1.0, -index * 0.05], [0, 0, 1]]
        )
        started = perf_counter()
        assert cache.update_transform(matrix)
        samples.append((perf_counter() - started) * 1000.0)
    assert np.percentile(samples, 95) < 4.0


def test_hidpi_scene_mapping_uses_physical_pixmap_pixels_exactly_once(
    line_selection,
) -> None:
    _fig, _line, manager, selection = line_selection
    cache = ensure_content_preview_cache(manager)
    assert cache.warm_now(manager, selection)
    entry = cache.entry
    assert entry is not None
    physical_height = entry.canvas_bounds[3]
    logical_height = physical_height / 2.0
    selection.graphics_scene.setTransform(
        selection_scene_transform(2.0, logical_height)
    )
    assert entry.pixmap.devicePixelRatio() == 1.0

    physical = QtCore.QPointF(120.0, 80.0)
    scene_point = entry.root.mapToScene(physical)
    assert np.allclose(
        [scene_point.x(), scene_point.y()],
        [physical.x() / 2.0, logical_height - physical.y() / 2.0],
        atol=1e-12,
    )
    assert cache.activate(manager, selection)
    before = entry.item.sceneBoundingRect()
    assert cache.update_transform(
        np.array([[1.0, 0.0, 10.0], [0.0, 1.0, -6.0], [0.0, 0.0, 1.0]])
    )
    after = entry.item.sceneBoundingRect()
    assert np.allclose(
        [after.left() - before.left(), after.top() - before.top()],
        [5.0, 3.0],
        atol=1e-12,
    )


def test_nontranslation_affine_is_never_shown_as_a_scaled_bitmap(
    line_selection,
) -> None:
    _fig, _line, manager, selection = line_selection
    cache = ensure_content_preview_cache(manager)
    assert cache.warm_now(manager, selection)
    assert cache.activate(manager, selection)
    matrix = np.array(
        [[1.25, 0.2, 7.0], [-0.1, 0.8, -4.0], [0.0, 0.0, 1.0]]
    )
    assert not cache.update_transform(matrix)
    assert cache.last_fallback_reason == "translation-only"
    assert not cache.entry.root.isVisible()
