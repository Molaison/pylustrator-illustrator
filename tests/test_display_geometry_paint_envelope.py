from __future__ import annotations

from time import perf_counter

import matplotlib.patheffects as path_effects
import numpy as np
import pytest
from matplotlib.backends.backend_agg import FigureCanvasAgg, RendererAgg
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Arc, Circle, PathPatch, Rectangle
from matplotlib.path import Path
from matplotlib.text import Text
from matplotlib.transforms import Bbox, IdentityTransform

from pylustrator.display_geometry import (
    ArtistRoster,
    DisplayGeometryCache,
    PaintEnvelopeAccuracy,
)


def _figure_renderer(*, width: int = 400, height: int = 300, dpi: int = 100):
    figure = Figure(figsize=(width / dpi, height / dpi), dpi=dpi)
    FigureCanvasAgg(figure)
    figure.canvas.draw()
    return figure, figure.canvas.get_renderer()


def _attach(figure: Figure, artist, *, draw: bool = True):
    figure.add_artist(artist)
    if draw:
        figure.canvas.draw()
    return artist


def _cache(artist, renderer, *, revision: int = 1, budget=None):
    kwargs = {} if budget is None else {"paint_raster_budget_bytes": budget}
    cache = DisplayGeometryCache(**kwargs)
    roster = ArtistRoster.capture((artist,))
    cache.bind(revision=revision, roster=roster, renderer=renderer)
    return cache, roster


def _independent_alpha_bounds(artist, renderer):
    """Small test oracle using the pixel coordinates, not production helpers."""

    probe = RendererAgg(renderer.width, renderer.height, renderer.dpi)
    artist.draw(probe)
    alpha = np.asarray(probe.buffer_rgba())[..., 3]
    pixels = np.argwhere(alpha != 0)
    if not len(pixels):
        return None
    row_min, col_min = pixels.min(axis=0)
    row_max, col_max = pixels.max(axis=0)
    return (
        float(col_min),
        float(alpha.shape[0] - row_max - 1),
        float(col_max + 1),
        float(alpha.shape[0] - row_min),
    )


def _line_state(line: Line2D) -> tuple:
    return (
        frozenset(line.__dict__),
        line._invalidx,
        line._invalidy,
        id(line._xy),
        line._xy.copy(),
        id(line._path),
        line._path.vertices.copy(),
        id(line._transformed_path),
        line._stale,
        id(line.stale_callback),
        id(line._remove_method),
    )


def _assert_line_state_unchanged(line: Line2D, before: tuple) -> None:
    after = _line_state(line)
    assert after[:4] == before[:4]
    assert np.array_equal(after[4], before[4])
    assert after[5] == before[5]
    assert np.array_equal(after[6], before[6])
    assert after[7:] == before[7:]


def _owner_state(owner) -> tuple:
    values = []
    for name, value in sorted(owner.__dict__.items()):
        if isinstance(value, np.ndarray):
            token = ("array", id(value), value.shape, value.dtype.str, value.tobytes())
        elif isinstance(value, (str, bytes, int, float, bool, type(None))):
            token = ("scalar", repr(value))
        elif isinstance(value, (list, tuple)):
            token = ("sequence", id(value), tuple(id(item) for item in value))
        elif isinstance(value, dict):
            token = (
                "mapping",
                id(value),
                tuple(sorted((id(key), id(item)) for key, item in value.items())),
            )
        else:
            token = ("identity", id(value))
        values.append((name, token))
    return tuple(values)


@pytest.mark.parametrize("capstyle", ["butt", "round", "projecting"])
def test_agg_envelope_matches_stroke_cap_pixels(capstyle: str) -> None:
    figure, renderer = _figure_renderer()
    line = _attach(
        figure,
        Line2D(
            [100.0, 250.0],
            [140.0, 140.0],
            linewidth=18.0,
            color="black",
            solid_capstyle=capstyle,
            transform=IdentityTransform(),
        ),
    )
    cache, _roster = _cache(line, renderer)

    envelope = cache.capture_paint_envelope(line)

    assert envelope.accuracy is PaintEnvelopeAccuracy.EXACT
    assert envelope.bounds == _independent_alpha_bounds(line, renderer)


@pytest.mark.parametrize("joinstyle", ["bevel", "round", "miter"])
def test_agg_envelope_matches_stroke_join_pixels(joinstyle: str) -> None:
    figure, renderer = _figure_renderer()
    path = Path(
        np.asarray(((80.0, 60.0), (200.0, 245.0), (230.0, 60.0))),
        (Path.MOVETO, Path.LINETO, Path.LINETO),
    )
    patch = _attach(
        figure,
        PathPatch(
            path,
            fill=False,
            edgecolor="black",
            linewidth=22.0,
            joinstyle=joinstyle,
            capstyle="butt",
            transform=IdentityTransform(),
        ),
    )
    cache, _roster = _cache(patch, renderer)

    envelope = cache.capture_paint_envelope(patch)

    assert envelope.accuracy is PaintEnvelopeAccuracy.EXACT
    assert envelope.bounds == _independent_alpha_bounds(patch, renderer)


@pytest.mark.parametrize(
    ("facecolor", "edgecolor", "edgewidth"),
    [
        ("tab:blue", "none", 0.0),
        ("none", "tab:red", 5.0),
        ("tab:blue", "tab:red", 5.0),
    ],
)
def test_agg_envelope_matches_marker_fill_and_edge_pixels(
    facecolor: str, edgecolor: str, edgewidth: float
) -> None:
    figure, renderer = _figure_renderer()
    line = _attach(
        figure,
        Line2D(
            [90.0, 205.0, 310.0],
            [80.0, 210.0, 105.0],
            linestyle="none",
            marker="o",
            markersize=24.0,
            markerfacecolor=facecolor,
            markeredgecolor=edgecolor,
            markeredgewidth=edgewidth,
            transform=IdentityTransform(),
        ),
    )
    cache, _roster = _cache(line, renderer)

    envelope = cache.capture_paint_envelope(line)

    assert envelope.accuracy is PaintEnvelopeAccuracy.EXACT
    assert envelope.bounds == _independent_alpha_bounds(line, renderer)


def test_agg_envelope_respects_rectangular_clip() -> None:
    figure, renderer = _figure_renderer()
    rectangle = _attach(
        figure,
        Rectangle(
            (55.0, 45.0),
            300.0,
            210.0,
            facecolor="tab:blue",
            edgecolor="black",
            linewidth=14.0,
            transform=IdentityTransform(),
        ),
    )
    rectangle.set_clip_box(Bbox.from_extents(120.0, 90.0, 280.0, 195.0))
    rectangle.set_clip_on(True)
    figure.canvas.draw()
    cache, _roster = _cache(rectangle, renderer)

    envelope = cache.capture_paint_envelope(rectangle)

    assert envelope.accuracy is PaintEnvelopeAccuracy.EXACT
    assert envelope.bounds == _independent_alpha_bounds(rectangle, renderer)
    assert envelope.bounds == (120.0, 90.0, 280.0, 195.0)


def test_agg_envelope_respects_nonrectangular_clip() -> None:
    figure, renderer = _figure_renderer()
    rectangle = _attach(
        figure,
        Rectangle(
            (40.0, 30.0),
            320.0,
            240.0,
            facecolor="tab:blue",
            edgecolor="none",
            transform=IdentityTransform(),
        ),
    )
    clip = Circle((205.0, 145.0), 58.0, transform=IdentityTransform())
    rectangle.set_clip_path(clip)
    rectangle.set_clip_on(True)
    figure.canvas.draw()
    cache, _roster = _cache(rectangle, renderer)

    envelope = cache.capture_paint_envelope(rectangle)

    assert envelope.accuracy is PaintEnvelopeAccuracy.EXACT
    assert envelope.bounds == _independent_alpha_bounds(rectangle, renderer)
    assert envelope.bounds is not None
    assert envelope.bounds[0] >= 146.0
    assert envelope.bounds[2] <= 264.0


@pytest.mark.parametrize("effect", ["stroke", "shadow"])
def test_agg_envelope_includes_reliably_rendered_path_effects(effect: str) -> None:
    figure, renderer = _figure_renderer()
    if effect == "stroke":
        artist = Line2D(
            [95.0, 290.0],
            [105.0, 190.0],
            linewidth=3.0,
            color="white",
            transform=IdentityTransform(),
            path_effects=[path_effects.withStroke(linewidth=21.0, foreground="black")],
        )
    else:
        artist = Rectangle(
            (105.0, 95.0),
            135.0,
            80.0,
            facecolor="tab:blue",
            edgecolor="black",
            transform=IdentityTransform(),
            path_effects=[
                path_effects.SimplePatchShadow(offset=(14.0, -9.0), alpha=0.7),
                path_effects.Normal(),
            ],
        )
    _attach(figure, artist)
    cache, _roster = _cache(artist, renderer)

    envelope = cache.capture_paint_envelope(artist)

    assert envelope.accuracy is PaintEnvelopeAccuracy.EXACT
    assert envelope.bounds == _independent_alpha_bounds(artist, renderer)


def test_agg_envelope_matches_path_collection_pixels() -> None:
    figure, renderer = _figure_renderer()
    axes = figure.add_axes((0.0, 0.0, 1.0, 1.0), frameon=False)
    axes.set_xlim(0.0, 4.0)
    axes.set_ylim(0.0, 3.0)
    collection = axes.scatter(
        [0.8, 2.0, 3.25],
        [0.65, 2.15, 1.05],
        s=[500.0, 1100.0, 700.0],
        marker="s",
        facecolors=["tab:blue", "none", "tab:green"],
        edgecolors="tab:red",
        linewidths=[2.0, 5.0, 8.0],
    )
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    cache, _roster = _cache(collection, renderer)

    envelope = cache.capture_paint_envelope(collection)

    assert envelope.accuracy is PaintEnvelopeAccuracy.EXACT
    assert envelope.bounds == _independent_alpha_bounds(collection, renderer)


def test_paint_cache_reuses_capture_and_lookup_never_draws(monkeypatch) -> None:
    figure, renderer = _figure_renderer()
    line = _attach(
        figure,
        Line2D(
            [70.0, 300.0],
            [90.0, 210.0],
            linewidth=9.0,
            color="black",
            transform=IdentityTransform(),
        ),
    )
    cache, _roster = _cache(line, renderer)
    original_capture = cache._capture_agg_paint_envelope
    capture_count = 0

    def counted_capture(*args, **kwargs):
        nonlocal capture_count
        capture_count += 1
        return original_capture(*args, **kwargs)

    monkeypatch.setattr(cache, "_capture_agg_paint_envelope", counted_capture)
    monkeypatch.setattr(
        figure.canvas,
        "draw",
        lambda: pytest.fail("paint capture must not draw the canvas"),
    )
    monkeypatch.setattr(
        figure.canvas,
        "draw_idle",
        lambda: pytest.fail("paint capture must not schedule a canvas draw"),
    )

    assert cache.paint_envelope(line) is None
    assert cache.paint_bounds(line) is None
    assert capture_count == 0

    first = cache.capture_paint_envelope(line)
    assert capture_count == 1
    assert cache.capture_paint_envelope(line) is first
    assert cache.paint_envelope(line) is first
    assert cache.paint_bounds(line) == first.bounds
    assert capture_count == 1


def test_paint_capture_preserves_artist_stale_lifecycle() -> None:
    figure, renderer = _figure_renderer()
    line = _attach(
        figure,
        Line2D(
            [70.0, 300.0],
            [90.0, 210.0],
            linewidth=9.0,
            color="black",
            transform=IdentityTransform(),
        ),
    )
    cache, _roster = _cache(line, renderer)
    line.stale = True

    envelope = cache.capture_paint_envelope(line)

    assert envelope.accuracy is PaintEnvelopeAccuracy.CONSERVATIVE
    assert envelope.reason == "artist-stale"
    assert line.stale is True
    assert cache._paint_renderer is None


def test_collection_capture_does_not_dirty_artist_axes_or_figure() -> None:
    figure, _renderer = _figure_renderer()
    axes = figure.add_axes((0.0, 0.0, 1.0, 1.0), frameon=False)
    collection = axes.scatter([0.8, 2.0], [0.65, 2.15], s=[500.0, 1100.0])
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    cache, _roster = _cache(collection, renderer)
    callback = collection.stale_callback
    collection._stale = True
    axes._stale = False
    figure._stale = False

    envelope = cache.capture_paint_envelope(collection)

    assert envelope.accuracy is PaintEnvelopeAccuracy.CONSERVATIVE
    assert envelope.reason == "artist-stale"
    assert collection._stale is True
    assert collection.stale_callback is callback
    assert axes.stale is False
    assert figure.stale is False


@pytest.mark.parametrize("mutation", ["set-clim", "cmap-in-place", "norm-in-place"])
def test_scalar_mappable_collection_fails_closed_without_derived_color_leak(
    mutation: str,
) -> None:
    figure, _renderer = _figure_renderer()
    axes = figure.add_axes((0.0, 0.0, 1.0, 1.0), frameon=False)
    collection = axes.scatter(
        [0.2, 0.5, 0.8],
        [0.3, 0.7, 0.4],
        c=[0.0, 0.5, 1.0],
        cmap="viridis",
    )
    figure.canvas.draw()
    if mutation == "set-clim":
        collection.set_clim(0.0, 2.0)
    elif mutation == "cmap-in-place":
        collection.cmap.set_bad("tab:red", alpha=0.35)
    else:
        collection.norm.vmax = 2.0
    renderer = figure.canvas.get_renderer()
    cache, _roster = _cache(collection, renderer)
    facecolors = collection.get_facecolors().copy()
    color_state = (
        collection.cmap._rgba_bad,
        collection.norm.vmin,
        collection.norm.vmax,
        collection.norm.clip,
    )
    invalid_state = (
        getattr(collection, "_invalidx", None),
        getattr(collection, "_invalidy", None),
        collection.stale,
        axes.stale,
        figure.stale,
    )

    envelope = cache.capture_paint_envelope(collection)

    assert envelope.accuracy is PaintEnvelopeAccuracy.CONSERVATIVE
    assert envelope.reason == "scalar-mappable-unsupported"
    assert np.array_equal(collection.get_facecolors(), facecolors)
    assert (
        collection.cmap._rgba_bad,
        collection.norm.vmin,
        collection.norm.vmax,
        collection.norm.clip,
    ) == color_state
    assert (
        getattr(collection, "_invalidx", None),
        getattr(collection, "_invalidy", None),
        collection.stale,
        axes.stale,
        figure.stale,
    ) == invalid_state


def test_pending_line_capture_is_conservative_without_cache_leak() -> None:
    figure, renderer = _figure_renderer()
    line = _attach(
        figure,
        Line2D(
            [70.0, 300.0],
            [90.0, 210.0],
            linewidth=9.0,
            color="black",
            transform=IdentityTransform(),
        ),
    )
    figure.canvas.draw()
    line.set_data([85.0, 280.0], [105.0, 190.0])
    cache, _roster = _cache(line, renderer, revision=2)
    live_state = (
        line._invalidx,
        line._invalidy,
        id(line._xy),
        line._xy.copy(),
        id(line._path),
        line._path.vertices.copy(),
        line.stale,
        figure.stale,
    )

    envelope = cache.capture_paint_envelope(line)

    assert envelope.accuracy is PaintEnvelopeAccuracy.CONSERVATIVE
    assert envelope.reason == "artist-stale"
    assert cache._paint_renderer is None
    assert (line._invalidx, line._invalidy) == live_state[:2]
    assert id(line._xy) == live_state[2]
    assert np.array_equal(line._xy, live_state[3])
    assert id(line._path) == live_state[4]
    assert np.array_equal(line._path.vertices, live_state[5])
    assert line.stale is live_state[6]
    assert figure.stale is live_state[7]


def test_successful_capture_does_not_mutate_legend_handle_or_live_owners() -> None:
    figure, _renderer = _figure_renderer()
    axes = figure.add_axes((0.0, 0.0, 1.0, 1.0))
    axes.plot([0.2, 0.8], [0.3, 0.7], marker="o", label="line")
    legend = axes.legend()
    figure.canvas.draw()
    handle = legend.legend_handles[0]
    renderer = figure.canvas.get_renderer()
    cache, _roster = _cache(handle, renderer)
    live_line_state = _line_state(handle)
    owner_state = tuple(_owner_state(owner) for owner in (legend, axes, figure))
    assert "_pylustrator_legend_owner" not in handle.__dict__
    assert "_pylustrator_legend_owner_inventory" not in figure.__dict__

    envelope = cache.capture_paint_envelope(handle)

    assert envelope.exact
    _assert_line_state_unchanged(handle, live_line_state)
    assert tuple(_owner_state(owner) for owner in (legend, axes, figure)) == owner_state
    assert "_pylustrator_legend_owner" not in handle.__dict__
    assert "_pylustrator_legend_owner_inventory" not in figure.__dict__


def test_successful_capture_does_not_mutate_clip_dependency() -> None:
    figure, renderer = _figure_renderer()
    rectangle = _attach(
        figure,
        Rectangle(
            (40.0, 30.0),
            320.0,
            240.0,
            facecolor="tab:blue",
            edgecolor="none",
            transform=IdentityTransform(),
        ),
    )
    clip = Circle((205.0, 145.0), 58.0, transform=IdentityTransform())
    rectangle.set_clip_path(clip)
    rectangle.set_clip_on(True)
    figure.canvas.draw()
    cache, _roster = _cache(rectangle, renderer)
    live_state = (_owner_state(rectangle), _owner_state(clip), _owner_state(figure))

    envelope = cache.capture_paint_envelope(rectangle)

    assert envelope.exact
    assert (
        _owner_state(rectangle),
        _owner_state(clip),
        _owner_state(figure),
    ) == live_state


def test_failed_clone_draw_does_not_mutate_artist_or_live_owners(monkeypatch) -> None:
    figure, _renderer = _figure_renderer()
    axes = figure.add_axes((0.0, 0.0, 1.0, 1.0))
    (line,) = axes.plot([0.2, 0.8], [0.3, 0.7], linewidth=7.0)
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    cache, _roster = _cache(line, renderer)
    live_line_state = _line_state(line)
    owner_state = tuple(_owner_state(owner) for owner in (axes, figure))

    def fail_draw_path(*_args, **_kwargs):
        raise RuntimeError("synthetic private draw failure")

    monkeypatch.setattr(RendererAgg, "draw_path", fail_draw_path)
    envelope = cache.capture_paint_envelope(line)

    assert envelope.accuracy is PaintEnvelopeAccuracy.CONSERVATIVE
    assert envelope.reason == "artist-draw-failed"
    _assert_line_state_unchanged(line, live_line_state)
    assert tuple(_owner_state(owner) for owner in (axes, figure)) == owner_state


def test_instance_getter_callback_is_rejected_before_owner_lookup_or_draw() -> None:
    figure, renderer = _figure_renderer()
    line = _attach(
        figure,
        Line2D([50.0, 350.0], [150.0, 150.0], transform=IdentityTransform()),
    )
    callback_called = False

    def custom_get_figure(*_args, **_kwargs):
        nonlocal callback_called
        callback_called = True
        raise AssertionError("custom getter executed")

    line.get_figure = custom_get_figure
    cache, _roster = _cache(line, renderer)

    envelope = cache.capture_paint_envelope(line)

    assert not callback_called
    assert envelope.accuracy is PaintEnvelopeAccuracy.CONSERVATIVE
    assert envelope.reason == "custom-draw-unsupported"
    assert cache._paint_renderer is None


def test_custom_deepcopy_state_is_rejected_without_execution() -> None:
    callback_called = False

    class CustomState:
        def __deepcopy__(self, _memo):
            nonlocal callback_called
            callback_called = True
            raise AssertionError("custom deepcopy executed")

    figure, renderer = _figure_renderer()
    line = _attach(
        figure,
        Line2D([50.0, 350.0], [150.0, 150.0], transform=IdentityTransform()),
    )
    line.custom_state = CustomState()
    cache, _roster = _cache(line, renderer)

    envelope = cache.capture_paint_envelope(line)

    assert not callback_called
    assert envelope.accuracy is PaintEnvelopeAccuracy.CONSERVATIVE
    assert envelope.reason == "clone-state-unsupported"
    assert cache._paint_renderer is None


def test_custom_owner_getter_is_rejected_without_execution() -> None:
    callback_called = False
    figure, _renderer = _figure_renderer()
    axes = figure.add_axes((0.0, 0.0, 1.0, 1.0))
    (line,) = axes.plot([0.2, 0.8], [0.3, 0.7])
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()

    def custom_get_legend():
        nonlocal callback_called
        callback_called = True
        raise AssertionError("custom owner getter executed")

    axes.get_legend = custom_get_legend
    cache, _roster = _cache(line, renderer)

    envelope = cache.capture_paint_envelope(line)

    assert not callback_called
    assert envelope.accuracy is PaintEnvelopeAccuracy.CONSERVATIVE
    assert envelope.reason == "ancestor-state-unavailable"
    assert cache._paint_renderer is None


def test_hidden_axes_and_legend_owners_produce_exact_empty_without_draw() -> None:
    figure, _renderer = _figure_renderer()
    axes = figure.add_axes((0.0, 0.0, 1.0, 1.0))
    (line,) = axes.plot([0.2, 0.8], [0.3, 0.7], label="line")
    legend = axes.legend()
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()

    axes.set_visible(False)
    figure.canvas.draw()
    axes_cache, _axes_roster = _cache(line, renderer)
    assert axes_cache.capture_paint_envelope(line).exact
    assert axes_cache.paint_envelope(line).bounds is None
    assert axes_cache._paint_renderer is None

    axes.set_visible(True)
    legend.set_visible(False)
    figure.canvas.draw()
    handle = legend.legend_handles[0]
    legend_cache, _legend_roster = _cache(handle, renderer)
    assert legend_cache.capture_paint_envelope(handle).exact
    assert legend_cache.paint_envelope(handle).bounds is None
    assert legend_cache._paint_renderer is None


def test_ancestor_agg_filter_is_conservative_without_redrawing() -> None:
    figure, _renderer = _figure_renderer()
    axes = figure.add_axes((0.0, 0.0, 1.0, 1.0))
    (line,) = axes.plot([0.2, 0.8], [0.3, 0.7])
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    axes.set_agg_filter(lambda image, _dpi: (image, 0.0, 0.0))
    cache, _roster = _cache(line, renderer)

    envelope = cache.capture_paint_envelope(line)

    assert envelope.accuracy is PaintEnvelopeAccuracy.CONSERVATIVE
    assert envelope.bounds == (0.0, 0.0, 400.0, 300.0)
    assert envelope.reason == "ancestor-agg-filter-unsupported"
    assert cache._paint_renderer is None


def test_revision_renderer_roster_and_mutations_invalidate_paint() -> None:
    figure, renderer = _figure_renderer()
    line = _attach(
        figure,
        Line2D(
            [80.0, 230.0],
            [110.0, 110.0],
            linewidth=4.0,
            color="black",
            transform=IdentityTransform(),
        ),
    )
    cache, roster = _cache(line, renderer)
    first = cache.capture_paint_envelope(line)
    assert first.bounds is not None

    line.set_linewidth(24.0)
    assert cache.capture_paint_envelope(line) is first
    figure.canvas.draw()
    cache.bind(revision=2, roster=roster, renderer=renderer)
    assert cache.paint_envelope(line) is None
    wider = cache.capture_paint_envelope(line)
    assert wider.bounds is not None
    assert wider.bounds[1] < first.bounds[1]
    assert wider.bounds[3] > first.bounds[3]

    line.set_data([145.0, 295.0], [165.0, 165.0])
    figure.canvas.draw()
    cache.bind(revision=3, roster=roster, renderer=renderer)
    moved = cache.capture_paint_envelope(line)
    assert moved.bounds is not None
    assert moved.bounds[0] > wider.bounds[0]

    line.set_clip_box(Bbox.from_extents(180.0, 120.0, 250.0, 200.0))
    line.set_clip_on(True)
    figure.canvas.draw()
    cache.bind(revision=4, roster=roster, renderer=renderer)
    clipped = cache.capture_paint_envelope(line)
    assert clipped.bounds is not None
    assert clipped.bounds[0] >= 180.0
    assert clipped.bounds[2] <= 250.0

    replacement_renderer = RendererAgg(renderer.width, renderer.height, renderer.dpi)
    cache.bind(revision=4, roster=roster, renderer=replacement_renderer)
    assert cache.paint_envelope(line) is None
    assert cache.capture_paint_envelope(line).exact

    replacement_roster = ArtistRoster.capture((line,))
    cache.bind(
        revision=4,
        roster=replacement_roster,
        renderer=replacement_renderer,
    )
    assert cache.paint_envelope(line) is None


def test_empty_agg_paint_is_distinct_from_uncaptured() -> None:
    figure, renderer = _figure_renderer()
    line = _attach(
        figure,
        Line2D(
            [80.0, 230.0],
            [110.0, 110.0],
            linewidth=12.0,
            color=(0.0, 0.0, 0.0, 0.0),
            transform=IdentityTransform(),
        ),
    )
    cache, _roster = _cache(line, renderer)
    assert cache.paint_envelope(line) is None

    envelope = cache.capture_paint_envelope(line)

    assert envelope.exact
    assert envelope.bounds is None
    assert cache.paint_envelope(line) is envelope


def test_unsupported_artist_and_budget_exhaustion_are_conservative() -> None:
    figure, renderer = _figure_renderer()
    text = _attach(figure, Text(0.5, 0.5, "unsupported composite text"))
    text_cache, _text_roster = _cache(text, renderer)

    text_envelope = text_cache.capture_paint_envelope(text)

    assert text_envelope.accuracy is PaintEnvelopeAccuracy.CONSERVATIVE
    assert text_envelope.bounds == (0.0, 0.0, 400.0, 300.0)
    assert text_envelope.reason == "artist-type-unsupported"

    line = _attach(
        figure,
        Line2D(
            [50.0, 350.0],
            [150.0, 150.0],
            transform=IdentityTransform(),
        ),
    )
    line_cache, _line_roster = _cache(line, renderer, budget=0)

    budget_envelope = line_cache.capture_paint_envelope(line)

    assert budget_envelope.accuracy is PaintEnvelopeAccuracy.CONSERVATIVE
    assert budget_envelope.bounds == (0.0, 0.0, 400.0, 300.0)
    assert budget_envelope.reason == "raster-budget-exceeded"
    assert line_cache._paint_renderer is None


def test_arbitrary_agg_filter_is_conservative_without_redrawing() -> None:
    figure, renderer = _figure_renderer()
    line = _attach(
        figure,
        Line2D(
            [50.0, 350.0],
            [150.0, 150.0],
            linewidth=8.0,
            transform=IdentityTransform(),
        ),
    )
    line.set_agg_filter(lambda image, _dpi: (image, 0.0, 0.0))
    cache, _roster = _cache(line, renderer)

    envelope = cache.capture_paint_envelope(line)

    assert envelope.accuracy is PaintEnvelopeAccuracy.CONSERVATIVE
    assert envelope.bounds == (0.0, 0.0, 400.0, 300.0)
    assert envelope.reason == "agg-filter-unsupported"
    assert cache._paint_renderer is None


def test_third_party_path_effect_is_conservative_without_redrawing() -> None:
    class CustomEffect(path_effects.AbstractPathEffect):
        def draw_path(self, renderer, gc, tpath, affine, rgb_face=None):
            renderer.draw_path(gc, tpath, affine, rgb_face)

    figure, renderer = _figure_renderer()
    line = _attach(
        figure,
        Line2D(
            [50.0, 350.0],
            [150.0, 150.0],
            linewidth=8.0,
            transform=IdentityTransform(),
            path_effects=[CustomEffect()],
        ),
    )
    cache, _roster = _cache(line, renderer)

    envelope = cache.capture_paint_envelope(line)

    assert envelope.accuracy is PaintEnvelopeAccuracy.CONSERVATIVE
    assert envelope.bounds == (0.0, 0.0, 400.0, 300.0)
    assert envelope.reason == "path-effect-unsupported"
    assert cache._paint_renderer is None


def test_unbudgeted_builtin_path_effect_is_conservative_without_redrawing() -> None:
    figure, renderer = _figure_renderer()
    line = _attach(
        figure,
        Line2D(
            [50.0, 350.0],
            [150.0, 150.0],
            linewidth=8.0,
            transform=IdentityTransform(),
            path_effects=[path_effects.TickedStroke()],
        ),
    )
    cache, _roster = _cache(line, renderer)

    envelope = cache.capture_paint_envelope(line)

    assert envelope.accuracy is PaintEnvelopeAccuracy.CONSERVATIVE
    assert envelope.bounds == (0.0, 0.0, 400.0, 300.0)
    assert envelope.reason == "path-effect-unsupported"
    assert cache._paint_renderer is None


def test_custom_subclass_draw_is_conservative_without_execution() -> None:
    draw_called = False

    class CustomLine(Line2D):
        def draw(self, renderer):
            nonlocal draw_called
            draw_called = True
            return super().draw(renderer)

    figure, renderer = _figure_renderer()
    line = _attach(
        figure,
        CustomLine(
            [50.0, 350.0],
            [150.0, 150.0],
            linewidth=8.0,
            transform=IdentityTransform(),
        ),
        draw=False,
    )
    cache, _roster = _cache(line, renderer)

    envelope = cache.capture_paint_envelope(line)

    assert draw_called is False
    assert envelope.accuracy is PaintEnvelopeAccuracy.CONSERVATIVE
    assert envelope.bounds == (0.0, 0.0, 400.0, 300.0)
    assert envelope.reason == "custom-draw-unsupported"
    assert cache._paint_renderer is None


def test_instance_draw_override_is_conservative_without_execution() -> None:
    figure, renderer = _figure_renderer()
    line = _attach(
        figure,
        Line2D(
            [50.0, 350.0],
            [150.0, 150.0],
            linewidth=8.0,
            transform=IdentityTransform(),
        ),
    )
    draw_called = False

    def custom_draw(_renderer):
        nonlocal draw_called
        draw_called = True

    line.draw = custom_draw
    cache, _roster = _cache(line, renderer)

    envelope = cache.capture_paint_envelope(line)

    assert draw_called is False
    assert envelope.accuracy is PaintEnvelopeAccuracy.CONSERVATIVE
    assert envelope.bounds == (0.0, 0.0, 400.0, 300.0)
    assert envelope.reason == "custom-draw-unsupported"
    assert cache._paint_renderer is None


def test_unvalidated_matplotlib_draw_override_is_conservative() -> None:
    figure, renderer = _figure_renderer()
    arc = _attach(
        figure,
        Arc(
            (200.0, 150.0),
            180.0,
            100.0,
            theta1=20.0,
            theta2=320.0,
            transform=IdentityTransform(),
        ),
    )
    cache, _roster = _cache(arc, renderer)

    envelope = cache.capture_paint_envelope(arc)

    assert envelope.accuracy is PaintEnvelopeAccuracy.CONSERVATIVE
    assert envelope.bounds == (0.0, 0.0, 400.0, 300.0)
    assert envelope.reason == "draw-contract-unsupported"
    assert cache._paint_renderer is None


def test_non_agg_renderer_and_detached_artist_are_unavailable() -> None:
    figure, renderer = _figure_renderer()
    line = _attach(
        figure,
        Line2D([50.0, 350.0], [150.0, 150.0], transform=IdentityTransform()),
    )
    cache, roster = _cache(line, renderer)
    cache.bind(revision=2, roster=roster, renderer=object())
    non_agg = cache.capture_paint_envelope(line)
    assert non_agg.accuracy is PaintEnvelopeAccuracy.UNAVAILABLE
    assert non_agg.reason == "renderer-not-agg"

    detached = Line2D([50.0, 350.0], [150.0, 150.0], transform=IdentityTransform())
    detached_cache, _detached_roster = _cache(detached, renderer)
    result = detached_cache.capture_paint_envelope(detached)
    assert result.accuracy is PaintEnvelopeAccuracy.UNAVAILABLE
    assert result.reason == "artist-detached"


def test_private_renderer_failure_is_full_canvas_conservative(monkeypatch) -> None:
    figure, renderer = _figure_renderer()
    line = _attach(
        figure,
        Line2D([50.0, 350.0], [150.0, 150.0], transform=IdentityTransform()),
    )
    cache, _roster = _cache(line, renderer)

    def fail_clear(_renderer):
        raise RuntimeError("private renderer failed")

    monkeypatch.setattr(RendererAgg, "clear", fail_clear)
    envelope = cache.capture_paint_envelope(line)

    assert envelope.accuracy is PaintEnvelopeAccuracy.CONSERVATIVE
    assert envelope.bounds == (0.0, 0.0, 400.0, 300.0)
    assert envelope.reason == "artist-draw-failed"
    assert cache.paint_envelope(line) is envelope


def test_100k_line_capture_has_canvas_bounded_scratch_and_fast_cached_lookup() -> None:
    figure, renderer = _figure_renderer(width=480, height=320)
    count = 100_000
    x = np.linspace(20.0, 460.0, count)
    y = 160.0 + 95.0 * np.sin(np.linspace(0.0, 80.0, count))
    line = _attach(
        figure,
        Line2D(x, y, linewidth=1.5, color="black", transform=IdentityTransform()),
    )
    cache, _roster = _cache(line, renderer)

    capture_start = perf_counter()
    envelope = cache.capture_paint_envelope(line)
    capture_elapsed = perf_counter() - capture_start

    assert envelope.exact
    assert envelope.bounds is not None
    assert capture_elapsed < 1.0
    scratch = cache._paint_renderer
    assert scratch is not None
    assert np.asarray(scratch.buffer_rgba()).nbytes == 480 * 320 * 4
    assert len(cache._paint_envelopes) == 1

    lookup_start = perf_counter()
    for _ in range(100_000):
        assert cache.paint_envelope(line) is envelope
    lookup_elapsed = perf_counter() - lookup_start
    assert lookup_elapsed < 0.25
