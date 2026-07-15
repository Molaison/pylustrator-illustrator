from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.artist import Artist
from matplotlib.collections import LineCollection, PathCollection, PolyCollection
from matplotlib.image import AxesImage
from matplotlib.legend import Legend
from matplotlib.lines import Line2D
from matplotlib.path import Path
from matplotlib.patches import (
    ConnectionPatch,
    Ellipse,
    FancyArrowPatch,
    FancyBboxPatch,
    PathPatch,
    Polygon,
    Rectangle,
    RegularPolygon,
    Wedge,
)
from matplotlib.text import Annotation, Text
from matplotlib.transforms import IdentityTransform

from pylustrator.artist_adapters import (
    AnnotationAdapter,
    ArtistAdapter,
    ArtistCapabilities,
    AxesAdapter,
    AxesImageAdapter,
    ConnectionPatchAdapter,
    EditorGroupAdapter,
    EllipseAdapter,
    FancyArrowPatchAdapter,
    FancyBboxPatchAdapter,
    LegendAdapter,
    Line2DAdapter,
    LineCollectionAdapter,
    PathCollectionAdapter,
    PathPatchAdapter,
    PolyCollectionAdapter,
    PolygonAdapter,
    RectangleAdapter,
    RegularPolygonAdapter,
    TextAdapter,
    UnsupportedArtistError,
    WedgeAdapter,
    artist_adapter_registry,
    get_artist_adapter,
)
from pylustrator.editor_model import EditorGroup
from pylustrator.commands import semantic_equal
from pylustrator.operations import TransformIntent, TransformOperation
from pylustrator.snap import TargetWrapper
from pylustrator.transform_engine import TransformPlan, TransformPreflightError


PIXEL_TOLERANCE = 0.25
TRANSLATION = np.array([13.0, -7.0])


class RecordingChangeTracker:
    """Small tracker double that preserves generated-change key semantics."""

    def __init__(self) -> None:
        self.changes = {}
        self.saved = True
        self.calls = []

    def _store(self, kind, target, command=None) -> None:
        self.calls.append((kind, target, command))
        key = (target, kind if command is None else command.split("(", 1)[0])
        self.changes[key] = (target, command)
        self.saved = False

    def addChange(self, target, command) -> None:
        self._store("command", target, command)

    def addNewTextChange(self, target) -> None:
        self._store("text", target)

    def addNewLegendChange(self, target) -> None:
        self._store("legend", target)

    def addNewAxesChange(self, target) -> None:
        self._store("axes", target)

    def capture_recording_state(self):
        return dict(self.changes), self.saved

    def restore_recording_state(self, state) -> None:
        self.changes, self.saved = dict(state[0]), bool(state[1])


CapabilitiesTuple = tuple[bool, bool, bool, bool, bool, bool, bool]
Builder = Callable[[plt.Figure, plt.Axes], Artist]


@dataclass(frozen=True)
class ArtistCase:
    name: str
    artist_type: type[Artist]
    adapter_type: type[ArtistAdapter]
    capabilities: CapabilitiesTuple
    builder: Builder


@dataclass
class BuiltArtistCase:
    spec: ArtistCase
    figure: plt.Figure
    axes: plt.Axes
    target: Artist
    adapter: ArtistAdapter
    tracker: RecordingChangeTracker
    sentinel: Text


def _fallback(fig, _ax):
    target = Artist()
    fig.add_artist(target)
    return target


def _editor_group(fig, ax):
    first = ax.add_patch(Rectangle((0.15, 0.2), 0.18, 0.22, label="qa-first"))
    second = ax.add_patch(
        Polygon(
            [[0.55, 0.25], [0.78, 0.3], [0.68, 0.58]],
            closed=True,
            label="qa-second",
        )
    )
    return EditorGroup(
        fig,
        "qa-group",
        [first, second],
        name="QA Group",
        owner=ax,
    )


def _axes(_fig, ax):
    return ax


def _text(_fig, ax):
    return ax.text(0.28, 0.72, "QA text", transform=ax.transAxes)


def _annotation(_fig, ax):
    return ax.annotate(
        "QA note",
        xy=(0.28, 0.32),
        xycoords="data",
        xytext=(0.7, 0.78),
        textcoords="axes fraction",
        arrowprops={"arrowstyle": "->"},
    )


def _legend(_fig, ax):
    line = ax.plot([0.15, 0.8], [0.2, 0.7], label="QA line")[0]
    return ax.legend(handles=[line], loc="upper right", frameon=False)


def _line(_fig, ax):
    return ax.plot([0.18, 0.45, 0.82], [0.25, 0.78, 0.42], label="qa-line")[0]


def _image(_fig, ax):
    target = ax.imshow(
        np.arange(16).reshape(4, 4),
        extent=(0.18, 0.62, 0.22, 0.66),
        interpolation="nearest",
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    return target


def _rectangle(_fig, ax):
    return ax.add_patch(Rectangle((0.18, 0.22), 0.34, 0.28, label="qa-rectangle"))


def _ellipse(_fig, ax):
    return ax.add_patch(Ellipse((0.42, 0.52), 0.36, 0.24, label="qa-ellipse"))


def _arrow(_fig, ax):
    return ax.add_patch(
        FancyArrowPatch(
            (0.18, 0.25),
            (0.78, 0.72),
            arrowstyle="-|>",
            mutation_scale=13,
        )
    )


def _connection(_fig, ax):
    target = ConnectionPatch(
        (0.2, 0.25),
        (0.75, 0.8),
        coordsA="data",
        coordsB="axes fraction",
        axesA=ax,
        axesB=ax,
        arrowstyle="->",
    )
    ax.add_artist(target)
    return target


def _fancy_bbox(_fig, ax):
    return ax.add_patch(
        FancyBboxPatch(
            (0.2, 0.25),
            0.42,
            0.3,
            boxstyle="round,pad=0.03",
            label="qa-fancy-bbox",
        )
    )


def _regular_polygon(_fig, ax):
    return ax.add_patch(RegularPolygon((0.48, 0.48), 6, radius=0.22))


def _wedge(_fig, ax):
    return ax.add_patch(Wedge((0.48, 0.48), 0.24, 20, 285))


def _polygon(_fig, ax):
    return ax.add_patch(
        Polygon(
            [[0.18, 0.2], [0.72, 0.28], [0.62, 0.74], [0.28, 0.66]],
            closed=True,
            label="qa-polygon",
        )
    )


def _path_patch(_fig, ax):
    path = Path(
        [[0.18, 0.2], [0.72, 0.22], [0.64, 0.72], [0.18, 0.2]],
        [Path.MOVETO, Path.LINETO, Path.LINETO, Path.CLOSEPOLY],
    )
    return ax.add_patch(PathPatch(path, label="qa-path-patch"))


def _path_collection(_fig, ax):
    return ax.scatter(
        [0.2, 0.48, 0.78],
        [0.28, 0.72, 0.42],
        s=[36, 64, 100],
        linewidths=1.25,
    )


def _line_collection(_fig, ax):
    target = LineCollection(
        [
            [[0.18, 0.22], [0.42, 0.68]],
            [[0.52, 0.3], [0.78, 0.74]],
        ],
        linewidths=[1.0, 2.0],
    )
    ax.add_collection(target)
    return target


def _poly_collection(_fig, ax):
    target = PolyCollection(
        [
            [[0.16, 0.2], [0.38, 0.24], [0.3, 0.5]],
            [[0.56, 0.3], [0.82, 0.35], [0.72, 0.68]],
        ],
        linewidths=[1.0, 2.0],
    )
    ax.add_collection(target)
    return target


# Capability tuple order follows ArtistCapabilities field order:
# select, translate, resize, snapshot, serialize, fixed_aspect, rotate.
ARTIST_CASES = (
    ArtistCase(
        "Artist fallback",
        Artist,
        ArtistAdapter,
        (False, False, False, False, False, False, False),
        _fallback,
    ),
    ArtistCase(
        "EditorGroup",
        EditorGroup,
        EditorGroupAdapter,
        (True, True, True, True, True, False, False),
        _editor_group,
    ),
    ArtistCase(
        "Axes",
        plt.Axes,
        AxesAdapter,
        (True, True, True, True, True, False, False),
        _axes,
    ),
    ArtistCase(
        "Text",
        Text,
        TextAdapter,
        (True, True, False, True, True, False, True),
        _text,
    ),
    ArtistCase(
        "Annotation",
        Annotation,
        AnnotationAdapter,
        (True, True, False, True, True, False, True),
        _annotation,
    ),
    ArtistCase(
        "Legend",
        Legend,
        LegendAdapter,
        (True, True, False, True, True, False, False),
        _legend,
    ),
    ArtistCase(
        "Line2D",
        Line2D,
        Line2DAdapter,
        (True, True, False, True, True, False, False),
        _line,
    ),
    ArtistCase(
        "AxesImage",
        AxesImage,
        AxesImageAdapter,
        (True, True, True, True, True, False, False),
        _image,
    ),
    ArtistCase(
        "Rectangle",
        Rectangle,
        RectangleAdapter,
        (True, True, True, True, True, False, True),
        _rectangle,
    ),
    ArtistCase(
        "Ellipse",
        Ellipse,
        EllipseAdapter,
        (True, True, True, True, True, False, True),
        _ellipse,
    ),
    ArtistCase(
        "FancyArrowPatch",
        FancyArrowPatch,
        FancyArrowPatchAdapter,
        (True, True, False, True, True, False, False),
        _arrow,
    ),
    ArtistCase(
        "ConnectionPatch",
        ConnectionPatch,
        ConnectionPatchAdapter,
        (False, False, False, False, False, False, False),
        _connection,
    ),
    ArtistCase(
        "FancyBboxPatch",
        FancyBboxPatch,
        FancyBboxPatchAdapter,
        (True, True, False, True, True, False, False),
        _fancy_bbox,
    ),
    ArtistCase(
        "RegularPolygon",
        RegularPolygon,
        RegularPolygonAdapter,
        (True, True, False, True, True, False, False),
        _regular_polygon,
    ),
    ArtistCase(
        "Wedge",
        Wedge,
        WedgeAdapter,
        (True, True, False, True, True, False, False),
        _wedge,
    ),
    ArtistCase(
        "Polygon",
        Polygon,
        PolygonAdapter,
        (True, True, True, True, True, False, False),
        _polygon,
    ),
    ArtistCase(
        "PathPatch",
        PathPatch,
        PathPatchAdapter,
        (True, True, True, True, True, False, False),
        _path_patch,
    ),
    ArtistCase(
        "PathCollection",
        PathCollection,
        PathCollectionAdapter,
        (True, True, False, True, True, False, False),
        _path_collection,
    ),
    ArtistCase(
        "LineCollection",
        LineCollection,
        LineCollectionAdapter,
        (True, True, False, True, True, False, False),
        _line_collection,
    ),
    ArtistCase(
        "PolyCollection",
        PolyCollection,
        PolyCollectionAdapter,
        (True, True, False, True, True, False, False),
        _poly_collection,
    ),
)


def _build_case(spec: ArtistCase) -> BuiltArtistCase:
    fig, ax = plt.subplots(figsize=(5, 4), dpi=100)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    sentinel = fig.text(0.02, 0.98, "sentinel", va="top")
    target = spec.builder(fig, ax)
    tracker = RecordingChangeTracker()
    fig.change_tracker = tracker
    fig.canvas.draw()
    return BuiltArtistCase(
        spec,
        fig,
        ax,
        target,
        get_artist_adapter(target),
        tracker,
        sentinel,
    )


@pytest.fixture(params=ARTIST_CASES, ids=lambda case: case.name)
def artist_case(request):
    built = _build_case(request.param)
    try:
        yield built
    finally:
        plt.close(built.figure)


def _capabilities_tuple(capabilities: ArtistCapabilities) -> CapabilitiesTuple:
    return (
        capabilities.can_select,
        capabilities.can_translate,
        capabilities.can_resize,
        capabilities.can_snapshot,
        capabilities.can_serialize,
        capabilities.fixed_aspect,
        capabilities.can_rotate,
    )


def _bounds(points) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    assert points.ndim == 2 and points.shape[1] == 2
    assert len(points) > 0
    assert np.all(np.isfinite(points))
    return np.array(
        [
            np.min(points[:, 0]),
            np.min(points[:, 1]),
            np.max(points[:, 0]),
            np.max(points[:, 1]),
        ]
    )


def _transform_points(matrix, points) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    homogeneous = np.column_stack((points, np.ones(len(points))))
    return (np.asarray(matrix, dtype=float) @ homogeneous.T).T[:, :2]


def _assert_px_close(actual, expected, *, atol=PIXEL_TOLERANCE) -> None:
    np.testing.assert_allclose(actual, expected, atol=atol, rtol=0)


def _axis_position(ax) -> np.ndarray:
    return np.asarray(ax.get_position().bounds, dtype=float)


def test_registry_inventory_and_advertised_capabilities_match_contract(
    artist_case,
) -> None:
    expected_types = {case.artist_type for case in ARTIST_CASES}
    registered_types = {
        registration.artist_type
        for registration in artist_adapter_registry.registrations()
    }

    assert expected_types == registered_types
    assert type(artist_case.adapter) is artist_case.spec.adapter_type
    assert _capabilities_tuple(artist_case.adapter.capabilities) == (
        artist_case.spec.capabilities
    )


OPERATION_CAPABILITY = {
    TransformOperation.SELECT: "can_select",
    TransformOperation.TRANSLATE: "can_translate",
    TransformOperation.RESIZE_GEOMETRY: "can_resize",
    TransformOperation.ROTATE: "can_rotate",
    TransformOperation.SNAPSHOT: "can_snapshot",
    TransformOperation.SERIALIZE: "can_serialize",
}


@pytest.mark.parametrize("operation", tuple(TransformOperation))
def test_operation_support_agrees_with_capabilities(artist_case, operation) -> None:
    support = artist_case.adapter.operation_support(operation)
    capability_name = OPERATION_CAPABILITY.get(operation)
    expected = (
        bool(getattr(artist_case.adapter.capabilities, capability_name))
        if capability_name is not None
        else False
    )

    assert support.operation is operation
    assert support.supported is expected
    if support.supported:
        assert support.reason == ""
    else:
        assert support.reason.strip()


def test_selectable_artist_has_finite_visible_selection_bounds(artist_case) -> None:
    if not artist_case.adapter.capabilities.can_select:
        pytest.skip("selection is explicitly unsupported")

    points = artist_case.adapter.selection_points()

    bounds = _bounds(points)
    assert bounds[2] >= bounds[0]
    assert bounds[3] >= bounds[1]


def test_display_translate_matches_preview_and_moves_only_the_target(
    artist_case,
) -> None:
    adapter = artist_case.adapter
    if not adapter.capabilities.can_translate:
        pytest.skip("translation is explicitly unsupported")

    selection_before = np.asarray(adapter.selection_points(), dtype=float)
    bounds_before = _bounds(selection_before)
    controls_before = np.asarray(adapter.control_points(), dtype=float)
    axes_before = _axis_position(artist_case.axes)
    sentinel_before = artist_case.sentinel.get_window_extent(
        artist_case.figure.canvas.get_renderer()
    ).extents.copy()
    figure_size_before = artist_case.figure.get_size_inches().copy()
    target_id = id(artist_case.target)
    plan = TransformPlan.preflight(
        [artist_case.target], TransformIntent.translate(TRANSLATION)
    )

    preview_controls = plan.preview_control_points()[0]
    _assert_px_close(preview_controls, controls_before + TRANSLATION)
    plan.commit()
    artist_case.figure.canvas.draw()

    assert id(artist_case.target) == target_id
    _assert_px_close(adapter.control_points(), preview_controls)
    _assert_px_close(
        _bounds(adapter.selection_points()),
        bounds_before + np.tile(TRANSLATION, 2),
    )
    _assert_px_close(
        artist_case.sentinel.get_window_extent(
            artist_case.figure.canvas.get_renderer()
        ).extents,
        sentinel_before,
    )
    np.testing.assert_allclose(
        artist_case.figure.get_size_inches(), figure_size_before, atol=0, rtol=0
    )
    if artist_case.target is not artist_case.axes:
        np.testing.assert_allclose(
            _axis_position(artist_case.axes), axes_before, atol=0, rtol=0
        )
    if isinstance(artist_case.target, Legend):
        assert artist_case.axes.get_legend() is artist_case.target


def test_display_resize_matches_preview_and_rendered_bounds(artist_case) -> None:
    adapter = artist_case.adapter
    if not adapter.capabilities.can_resize:
        pytest.skip("geometry resize is explicitly unsupported")

    selection_before = np.asarray(adapter.selection_points(), dtype=float)
    bounds_before = _bounds(selection_before)
    controls_before = np.asarray(adapter.control_points(), dtype=float)
    center = np.array(
        [
            (bounds_before[0] + bounds_before[2]) / 2,
            (bounds_before[1] + bounds_before[3]) / 2,
        ]
    )
    scale = 1.12
    matrix = np.array(
        [
            [scale, 0.0, center[0] * (1 - scale)],
            [0.0, scale, center[1] * (1 - scale)],
            [0.0, 0.0, 1.0],
        ]
    )
    axes_before = _axis_position(artist_case.axes)
    limits_before = (artist_case.axes.get_xlim(), artist_case.axes.get_ylim())
    plan = TransformPlan.preflight(
        [artist_case.target], TransformIntent.resize(matrix)
    )

    preview_controls = plan.preview_control_points()[0]
    _assert_px_close(preview_controls, _transform_points(matrix, controls_before))
    plan.commit()
    artist_case.figure.canvas.draw()

    _assert_px_close(adapter.control_points(), preview_controls)
    _assert_px_close(
        _bounds(adapter.selection_points()),
        _bounds(_transform_points(matrix, selection_before)),
    )
    if artist_case.target is not artist_case.axes:
        np.testing.assert_allclose(
            _axis_position(artist_case.axes), axes_before, atol=0, rtol=0
        )
    if isinstance(artist_case.target, AxesImage):
        assert artist_case.axes.get_xlim() == limits_before[0]
        assert artist_case.axes.get_ylim() == limits_before[1]


def test_native_rotation_is_applied_to_the_selected_artist_only(artist_case) -> None:
    adapter = artist_case.adapter
    if not adapter.capabilities.can_rotate:
        pytest.skip("rotation is explicitly unsupported")

    axes_before = _axis_position(artist_case.axes)
    old_rotation = adapter.rotation()
    plan = TransformPlan.preflight(
        [artist_case.target], TransformIntent.rotate(17.0)
    )

    assert plan.preview_control_points()[0].shape == np.asarray(
        adapter.control_points()
    ).shape
    plan.commit()
    artist_case.figure.canvas.draw()

    assert adapter.rotation() == pytest.approx(old_rotation + 17.0)
    visible = artist_case.target.get_window_extent(
        artist_case.figure.canvas.get_renderer()
    )
    selection = _bounds(adapter.selection_points())
    assert selection[0] <= visible.x0 + PIXEL_TOLERANCE
    assert selection[1] <= visible.y0 + PIXEL_TOLERANCE
    assert selection[2] >= visible.x1 - PIXEL_TOLERANCE
    assert selection[3] >= visible.y1 - PIXEL_TOLERANCE
    np.testing.assert_allclose(
        _axis_position(artist_case.axes), axes_before, atol=0, rtol=0
    )


def test_native_rotation_undo_redo_restores_angle_and_bookkeeping(
    artist_case,
) -> None:
    adapter = artist_case.adapter
    if not adapter.capabilities.can_rotate:
        pytest.skip("rotation is explicitly unsupported")
    tracker = _install_real_change_tracker(artist_case.figure)
    old_rotation = adapter.rotation()
    new_rotation = old_rotation + 17.0

    def apply(value):
        adapter.set_rotation(value)

    def undo():
        apply(old_rotation)

    def redo():
        apply(new_rotation)

    redo()
    recording_after = tracker.capture_recording_state()
    tracker.addEdit([undo, redo, "QA Rotate"])
    tracker.backEdit()

    assert adapter.rotation() == pytest.approx(old_rotation)
    assert tracker.last_edit == -1

    tracker.forwardEdit()

    assert adapter.rotation() == pytest.approx(new_rotation)
    assert tracker.last_edit == 0
    assert tracker.capture_recording_state()[0] == recording_after[0]


def test_snapshot_restore_round_trip_and_generated_change_replay(
    artist_case,
) -> None:
    adapter = artist_case.adapter
    if not adapter.capabilities.can_snapshot:
        pytest.skip("snapshots are explicitly unsupported")

    wrapper = TargetWrapper(artist_case.target)
    before_state = wrapper.get_restore_state()
    before_bounds = _bounds(adapter.selection_points())
    recording_before = artist_case.tracker.capture_recording_state()

    wrapper.translate(TRANSLATION)
    artist_case.figure.canvas.draw()
    after_state = wrapper.get_restore_state()
    after_bounds = _bounds(adapter.selection_points())
    recording_after = artist_case.tracker.capture_recording_state()

    assert not np.allclose(after_bounds, before_bounds, atol=PIXEL_TOLERANCE)
    assert recording_after != recording_before

    wrapper.restore_state(before_state, record_changes=False)
    artist_case.tracker.restore_recording_state(recording_before)
    artist_case.figure.canvas.draw()
    _assert_px_close(_bounds(adapter.selection_points()), before_bounds)
    assert artist_case.tracker.capture_recording_state() == recording_before

    wrapper.restore_state(after_state, record_changes=False)
    artist_case.tracker.restore_recording_state(recording_after)
    artist_case.figure.canvas.draw()
    _assert_px_close(_bounds(adapter.selection_points()), after_bounds)
    assert artist_case.tracker.capture_recording_state() == recording_after


def test_serialization_records_are_nonempty_and_target_owned(artist_case) -> None:
    adapter = artist_case.adapter
    if not adapter.capabilities.can_serialize:
        pytest.skip("serialization is explicitly unsupported")

    records = adapter.serialize_changes()

    assert records
    if isinstance(artist_case.target, EditorGroup):
        assert {record.target for record in records} == set(artist_case.target.members)
    else:
        assert artist_case.target in {record.target for record in records}
    before_calls = len(artist_case.tracker.calls)
    adapter.record_changes()
    assert len(artist_case.tracker.calls) > before_calls


def _install_real_change_tracker(fig):
    from pylustrator.change_tracker import ChangeTracker, init_figure

    init_figure(fig)
    tracker = ChangeTracker.__new__(ChangeTracker)
    tracker.figure = fig
    tracker.changes = {}
    tracker.saved = True
    tracker.edits = []
    tracker.last_edit = -1
    tracker.update_changes_signal = None
    tracker.no_save = False
    fig.change_tracker = tracker
    return tracker


def test_generated_commands_replay_the_translated_rendered_bounds(
    artist_case,
) -> None:
    adapter = artist_case.adapter
    if not adapter.capabilities.can_serialize:
        pytest.skip("serialization is explicitly unsupported")
    tracker = _install_real_change_tracker(artist_case.figure)
    wrapper = TargetWrapper(artist_case.target)
    before_state = wrapper.get_restore_state()

    wrapper.translate(TRANSLATION)
    artist_case.figure.canvas.draw()
    moved_bounds = _bounds(adapter.selection_points())
    generated = list(tracker.changes.values())

    assert generated
    assert not tracker.saved
    wrapper.restore_state(before_state, record_changes=False)
    artist_case.figure.canvas.draw()
    namespace = {"mpl": matplotlib, "np": np, "plt": plt}
    from pylustrator.change_tracker import getReference

    for command_target, command in generated:
        exec(f"{getReference(command_target)}{command}", namespace)
    artist_case.figure.canvas.draw()
    replayed_target = (
        artist_case.axes.get_legend()
        if isinstance(artist_case.target, Legend)
        else artist_case.target
    )

    _assert_px_close(
        _bounds(get_artist_adapter(replayed_target).selection_points()),
        moved_bounds,
    )


def test_line_nonfinite_coordinates_serialize_with_qualified_replay_literals() -> None:
    fig, ax = plt.subplots(figsize=(5, 4), dpi=100)
    fig.change_tracker = RecordingChangeTracker()
    x = np.array([0.2, np.nan, np.inf, -np.inf, 0.8])
    y = np.array([0.3, np.nan, np.inf, -np.inf, 0.7])
    target = ax.plot(x, y)[0]
    fig.canvas.draw()
    adapter = get_artist_adapter(target)
    initial_command = adapter.serialize_changes()[0].command

    try:
        assert "np.nan" in initial_command
        assert "np.inf" in initial_command
        assert "-np.inf" in initial_command

        adapter.translate(TRANSLATION)
        moved = np.asarray(target.get_xydata(), dtype=float).copy()
        command = adapter.serialize_changes()[0].command
        target.set_data(x, y)
        exec(f"target{command}", {"target": target, "np": np})
        np.testing.assert_allclose(
            target.get_xydata(), moved, rtol=1e-12, atol=1e-15, equal_nan=True
        )
    finally:
        plt.close(fig)


def test_replay_literals_preserve_exact_finite_scale_and_qualify_nonfinite() -> None:
    from pylustrator.replay import replay_literal

    narrow_axis_value = 1.0000000000000002
    assert float(replay_literal(narrow_axis_value)) == narrow_axis_value
    assert float(replay_literal(1.2345678901234567e-12)) == (
        1.2345678901234567e-12
    )
    assert replay_literal(np.nan) == "np.nan"
    assert replay_literal(np.inf) == "np.inf"
    assert replay_literal(-np.inf) == "-np.inf"


def test_narrow_axis_line_translate_replays_without_precision_amplification() -> None:
    fig, ax = plt.subplots(figsize=(5, 4), dpi=100)
    ax.set_xlim(1.0, 1.0 + 1e-12)
    ax.set_ylim(0.0, 1.0)
    x = np.array([1.0 + 2e-13, 1.0 + 8e-13])
    y = np.array([0.3, 0.7])
    target = ax.plot(x, y, linewidth=3)[0]
    fig.change_tracker = RecordingChangeTracker()
    fig.canvas.draw()
    adapter = get_artist_adapter(target)

    try:
        adapter.translate(TRANSLATION)
        fig.canvas.draw()
        moved_bounds = _bounds(adapter.selection_points())
        command = adapter.serialize_changes()[0].command

        target.set_data(x, y)
        exec(f"target{command}", {"target": target, "np": np})
        fig.canvas.draw()
        _assert_px_close(_bounds(adapter.selection_points()), moved_bounds)
    finally:
        plt.close(fig)


def test_saved_generated_block_imports_numpy_for_nonfinite_literals(
    monkeypatch,
) -> None:
    fig, ax = plt.subplots(figsize=(5, 4), dpi=100)
    target = ax.plot([0.2, np.nan, 0.8], [0.3, np.nan, 0.7])[0]
    fig.canvas.draw()
    tracker = _install_real_change_tracker(fig)
    tracker.get_reference_cached = {}
    get_artist_adapter(target).record_changes()
    saved = {}

    import pylustrator.change_tracker as change_tracker_module

    monkeypatch.setattr(change_tracker_module, "getTextFromFile", lambda *_args: [])
    monkeypatch.setattr(
        change_tracker_module, "stack_position", object(), raising=False
    )

    def capture_output(output, *_args):
        saved["lines"] = list(output)

    monkeypatch.setattr(change_tracker_module, "insertTextToFile", capture_output)

    try:
        tracker.save()
        generated = "\n".join(saved["lines"])
        assert "import numpy as np" in generated
        assert generated.index("import numpy as np") < generated.index("np.nan")

        target.set_data([0.1, 0.9], [0.1, 0.9])
        namespace = {"plt": plt}
        exec(generated, namespace)
        assert "np" in namespace
        assert np.isnan(target.get_xydata()[1]).all()
    finally:
        plt.close(fig)


@pytest.mark.parametrize("kind", ["rectangle", "ellipse", "text"])
def test_tiny_log_coordinates_translate_serialize_and_replay_losslessly(kind) -> None:
    fig, ax = plt.subplots(figsize=(5, 4), dpi=100)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1e-12, 1e-6)
    ax.set_ylim(1e-12, 1e-6)
    if kind == "rectangle":
        target = ax.add_patch(
            Rectangle(
                (2e-10, 3e-10),
                1e-10,
                2e-10,
                facecolor="none",
                edgecolor="black",
                label="_rect-data",
            )
        )
    elif kind == "ellipse":
        target = ax.add_patch(
            Ellipse(
                (3e-10, 4e-10),
                1e-10,
                2e-10,
                facecolor="none",
                edgecolor="black",
                label="qa-tiny-ellipse",
            )
        )
    else:
        target = ax.text(3e-10, 4e-10, "tiny log text")
    fig.canvas.draw()
    tracker = _install_real_change_tracker(fig)
    wrapper = TargetWrapper(target)
    before_state = wrapper.get_restore_state()

    try:
        wrapper.translate(TRANSLATION)
        fig.canvas.draw()
        moved_bounds = _bounds(wrapper.get_selection_points())
        generated = list(tracker.changes.values())
        assert generated
        assert "e-" in " ".join(command for _, command in generated)

        wrapper.restore_state(before_state, record_changes=False)
        namespace = {"mpl": matplotlib, "np": np, "plt": plt}
        from pylustrator.change_tracker import getReference

        for command_target, command in generated:
            exec(f"{getReference(command_target)}{command}", namespace)
        fig.canvas.draw()
        _assert_px_close(_bounds(wrapper.get_selection_points()), moved_bounds)
    finally:
        plt.close(fig)


def _observable_state(built: BuiltArtistCase) -> tuple:
    adapter = built.adapter
    try:
        controls = np.asarray(adapter.control_points(), dtype=float).copy()
    except (AttributeError, TypeError, ValueError, RuntimeError):
        controls = np.empty((0, 2), dtype=float)
    try:
        selection = np.asarray(adapter.selection_points(), dtype=float).copy()
    except (AttributeError, TypeError, ValueError, RuntimeError):
        selection = np.empty((0, 2), dtype=float)
    return (
        controls,
        selection,
        _axis_position(built.axes),
        built.figure.get_size_inches().copy(),
        bool(built.target.get_visible()),
        float(built.target.get_zorder()),
        built.tracker.capture_recording_state(),
    )


def _assert_observable_state_equal(actual, expected) -> None:
    for actual_array, expected_array in zip(actual[:4], expected[:4]):
        np.testing.assert_allclose(actual_array, expected_array, atol=0, rtol=0)
    assert actual[4:] == expected[4:]


@pytest.mark.parametrize(
    ("operation", "method", "argument"),
    [
        (TransformOperation.TRANSLATE, "translate", TRANSLATION),
        (
            TransformOperation.RESIZE_GEOMETRY,
            "resize",
            np.array([[1.1, 0, 0], [0, 1.1, 0], [0, 0, 1]]),
        ),
        (TransformOperation.ROTATE, "set_rotation", 19.0),
        (TransformOperation.SNAPSHOT, "snapshot", None),
    ],
)
def test_unsupported_operation_is_rejected_without_mutation(
    artist_case, operation, method, argument
) -> None:
    adapter = artist_case.adapter
    if adapter.operation_support(operation).supported:
        pytest.skip(f"{operation.value} is supported")
    if (
        artist_case.spec.name == "Artist fallback"
        and operation is TransformOperation.TRANSLATE
    ):
        pytest.skip("covered by dedicated adapter-contract error tests")
    before = _observable_state(artist_case)

    with pytest.raises(UnsupportedArtistError):
        if argument is None:
            getattr(adapter, method)()
        else:
            getattr(adapter, method)(argument)

    _assert_observable_state_equal(_observable_state(artist_case), before)


def test_fallback_translate_rejects_with_adapter_contract_error() -> None:
    built = _build_case(next(case for case in ARTIST_CASES if case.name == "Artist fallback"))
    before = _observable_state(built)

    try:
        with pytest.raises(UnsupportedArtistError):
            built.adapter.translate(TRANSLATION)
        _assert_observable_state_equal(_observable_state(built), before)
    finally:
        plt.close(built.figure)


def test_fallback_display_transform_rejects_with_adapter_contract_error() -> None:
    built = _build_case(next(case for case in ARTIST_CASES if case.name == "Artist fallback"))
    before = _observable_state(built)

    try:
        with pytest.raises(UnsupportedArtistError):
            built.adapter.apply_display_transform(np.eye(3))
        _assert_observable_state_equal(_observable_state(built), before)
    finally:
        plt.close(built.figure)


def test_display_transform_rejects_non_translation_matrix_without_mutation() -> None:
    built = _build_case(next(case for case in ARTIST_CASES if case.name == "Line2D"))
    before = _observable_state(built)
    scale = np.array(
        [[1.2, 0.0, -10.0], [0.0, 0.8, 12.0], [0.0, 0.0, 1.0]]
    )

    try:
        assert not built.adapter.supports_operation(
            TransformOperation.RESIZE_GEOMETRY
        )
        with pytest.raises(UnsupportedArtistError, match="semantic resize"):
            built.adapter.apply_display_transform(scale)
        _assert_observable_state_equal(_observable_state(built), before)
    finally:
        plt.close(built.figure)


def test_display_transform_preserves_legacy_pure_translation() -> None:
    built = _build_case(next(case for case in ARTIST_CASES if case.name == "Line2D"))
    bounds_before = _bounds(built.adapter.selection_points())
    matrix = np.array(
        [
            [1.0, 0.0, TRANSLATION[0]],
            [0.0, 1.0, TRANSLATION[1]],
            [0.0, 0.0, 1.0],
        ]
    )

    try:
        built.adapter.apply_display_transform(matrix)
        built.figure.canvas.draw()
        _assert_px_close(
            _bounds(built.adapter.selection_points()),
            bounds_before + np.tile(TRANSLATION, 2),
        )
    finally:
        plt.close(built.figure)


@pytest.mark.parametrize(
    "kind",
    [
        "line",
        "path_collection",
        "line_collection",
        "poly_collection",
        "path_patch",
        "polygon",
    ],
)
def test_empty_geometry_adapters_deny_operations_without_array_errors(kind) -> None:
    fig, ax = plt.subplots(figsize=(5, 4), dpi=100)
    tracker = RecordingChangeTracker()
    fig.change_tracker = tracker
    if kind == "line":
        target = ax.plot([], [])[0]
    elif kind == "path_collection":
        target = ax.scatter([], [])
    elif kind == "line_collection":
        target = LineCollection([])
        ax.add_collection(target)
    elif kind == "poly_collection":
        target = PolyCollection([])
        ax.add_collection(target)
    elif kind == "path_patch":
        target = ax.add_patch(PathPatch(Path(np.empty((0, 2)))))
    else:
        target = ax.add_patch(Polygon(np.empty((0, 2))))
    fig.canvas.draw()
    adapter = get_artist_adapter(target)

    try:
        assert not adapter.capabilities.editable
        assert not adapter.operation_support(TransformOperation.SELECT).supported
        assert not adapter.operation_support(TransformOperation.TRANSLATE).supported
        assert not adapter.operation_support(TransformOperation.RESIZE_GEOMETRY).supported
        assert not adapter.operation_support(TransformOperation.SNAPSHOT).supported
        assert not adapter.operation_support(TransformOperation.SERIALIZE).supported
        assert adapter.selection_points().shape == (0, 2)
        with pytest.raises(UnsupportedArtistError):
            adapter.translate(TRANSLATION)
        with pytest.raises(UnsupportedArtistError):
            adapter.apply_display_transform(np.eye(3))
        with pytest.raises(UnsupportedArtistError):
            adapter.resize(np.eye(3))
        with pytest.raises(UnsupportedArtistError):
            adapter.snapshot()
        assert tracker.calls == []
        assert tracker.changes == {}
    finally:
        plt.close(fig)


def test_axis_labels_translate_in_display_space_and_restore_labelpad() -> None:
    fig, ax = plt.subplots(figsize=(5, 4), dpi=100)
    tracker = RecordingChangeTracker()
    fig.change_tracker = tracker
    x_label = ax.set_xlabel("QA x label", labelpad=11)
    y_label = ax.set_ylabel("QA y label", labelpad=13)
    fig.canvas.draw()
    axes_before = _axis_position(ax)

    try:
        for label in (x_label, y_label):
            wrapper = TargetWrapper(label)
            state = wrapper.get_restore_state()
            bounds_before = _bounds(wrapper.get_selection_points())
            labelpad_before = (
                ax.xaxis.labelpad if label is x_label else ax.yaxis.labelpad
            )

            wrapper.translate(TRANSLATION)
            fig.canvas.draw()
            _assert_px_close(
                _bounds(wrapper.get_selection_points()),
                bounds_before + np.tile(TRANSLATION, 2),
            )
            np.testing.assert_allclose(_axis_position(ax), axes_before, atol=0, rtol=0)

            wrapper.restore_state(state, record_changes=False)
            fig.canvas.draw()
            _assert_px_close(_bounds(wrapper.get_selection_points()), bounds_before)
            assert (
                ax.xaxis.labelpad if label is x_label else ax.yaxis.labelpad
            ) == pytest.approx(labelpad_before)
    finally:
        plt.close(fig)


@pytest.mark.parametrize("axis_name", ["x", "y"])
def test_axis_label_generated_commands_replay_display_position(axis_name) -> None:
    fig, ax = plt.subplots(figsize=(5, 4), dpi=100)
    label = (
        ax.set_xlabel("QA x label", labelpad=11)
        if axis_name == "x"
        else ax.set_ylabel("QA y label", labelpad=13)
    )
    fig.canvas.draw()
    tracker = _install_real_change_tracker(fig)
    wrapper = TargetWrapper(label)
    before_state = wrapper.get_restore_state()

    try:
        wrapper.translate(TRANSLATION)
        fig.canvas.draw()
        moved_bounds = _bounds(wrapper.get_selection_points())
        generated = list(tracker.changes.values())
        assert generated

        wrapper.restore_state(before_state, record_changes=False)
        namespace = {"mpl": matplotlib, "np": np, "plt": plt}
        from pylustrator.change_tracker import getReference

        for command_target, command in generated:
            exec(f"{getReference(command_target)}{command}", namespace)
        fig.canvas.draw()
        _assert_px_close(_bounds(wrapper.get_selection_points()), moved_bounds)
    finally:
        plt.close(fig)


@pytest.mark.parametrize("frameon", [False, True])
def test_legend_selection_bounds_cover_visible_children_and_optional_frame(
    frameon,
) -> None:
    fig, ax = plt.subplots(figsize=(5, 4), dpi=100)
    fig.change_tracker = RecordingChangeTracker()
    handles = [
        ax.plot([0.1, 0.8], [0.2, 0.7], label="first")[0],
        ax.plot([0.1, 0.8], [0.7, 0.3], label="second")[0],
    ]
    legend = ax.legend(handles=handles, frameon=frameon, title="QA legend")
    fig.canvas.draw()
    legend.get_texts()[0].set_position((35.0, -18.0))
    fig.canvas.draw()
    visible_children = [
        *legend.legend_handles,
        *legend.get_texts(),
        legend.get_title(),
    ]
    expected_points = []
    for child in visible_children:
        if child.get_visible():
            expected_points.extend(
                np.asarray(get_artist_adapter(child).selection_points(), dtype=float)
            )
    if frameon:
        expected_points.extend(
            get_artist_adapter(legend.get_frame()).selection_points()
        )

    try:
        _assert_px_close(
            _bounds(get_artist_adapter(legend).selection_points()),
            _bounds(expected_points),
        )
    finally:
        plt.close(fig)


@pytest.mark.parametrize("kind", ["text", "annotation"])
def test_text_bbox_selection_bounds_include_visible_edge_stroke(kind) -> None:
    fig, ax = plt.subplots(figsize=(5, 4), dpi=100)
    fig.change_tracker = RecordingChangeTracker()
    bbox_style = {
        "facecolor": "none",
        "edgecolor": "black",
        "linewidth": 20,
        "pad": 10,
    }
    if kind == "text":
        target = ax.text(0.3, 0.6, "QA bbox", bbox=bbox_style)
    else:
        target = ax.annotate(
            "QA bbox",
            xy=(0.7, 0.3),
            xytext=(0.3, 0.6),
            bbox=bbox_style,
        )
    fig.canvas.draw()
    target.update_bbox_position_size(fig.canvas.get_renderer())
    bbox_patch = target.get_bbox_patch()
    raw = bbox_patch.get_window_extent(fig.canvas.get_renderer()).extents
    radius = bbox_patch.get_linewidth() * fig.dpi / 72.0 / 2
    expected = raw + np.array([-radius, -radius, radius, radius])

    try:
        _assert_px_close(
            _bounds(get_artist_adapter(target).selection_points()), expected
        )
    finally:
        plt.close(fig)


def test_annotation_selection_bounds_include_arrow_stroke_and_translate() -> None:
    fig, ax = plt.subplots(figsize=(5, 4), dpi=100)
    fig.change_tracker = RecordingChangeTracker()
    target = ax.annotate(
        "thick arrow",
        xy=(0.75, 0.25),
        xytext=(0.25, 0.7),
        arrowprops={"arrowstyle": "->", "linewidth": 20, "color": "black"},
    )
    fig.canvas.draw()
    adapter = get_artist_adapter(target)
    renderer = fig.canvas.get_renderer()
    raw = target.get_window_extent(renderer).get_points()
    arrow = get_artist_adapter(target.arrow_patch).selection_points()
    expected = _bounds(np.concatenate((raw, arrow)))
    before = _bounds(adapter.selection_points())

    try:
        _assert_px_close(before, expected)
        adapter.translate(TRANSLATION)
        fig.canvas.draw()
        _assert_px_close(
            _bounds(adapter.selection_points()),
            before + np.tile(TRANSLATION, 2),
        )
    finally:
        plt.close(fig)


def test_legend_frame_selection_bounds_include_visible_edge_stroke() -> None:
    fig, ax = plt.subplots(figsize=(5, 4), dpi=100)
    fig.change_tracker = RecordingChangeTracker()
    ax.plot([0, 1], [0, 1], label="QA legend")
    legend = ax.legend(frameon=True)
    legend.get_frame().set_linewidth(20)
    fig.canvas.draw()
    frame_bounds = _bounds(
        get_artist_adapter(legend.get_frame()).selection_points()
    )

    try:
        selection = _bounds(get_artist_adapter(legend).selection_points())
        assert selection[0] <= frame_bounds[0] + PIXEL_TOLERANCE
        assert selection[1] <= frame_bounds[1] + PIXEL_TOLERANCE
        assert selection[2] >= frame_bounds[2] - PIXEL_TOLERANCE
        assert selection[3] >= frame_bounds[3] - PIXEL_TOLERANCE
    finally:
        plt.close(fig)


def test_rectangle_selection_bounds_include_visible_stroke_width() -> None:
    fig, ax = plt.subplots(figsize=(5, 4), dpi=100)
    fig.change_tracker = RecordingChangeTracker()
    target = ax.add_patch(
        Rectangle(
            (0.2, 0.25),
            0.4,
            0.3,
            facecolor="none",
            edgecolor="black",
            linewidth=12,
        )
    )
    fig.canvas.draw()
    geometry = target.get_window_extent(fig.canvas.get_renderer()).extents
    stroke_radius = target.get_linewidth() * fig.dpi / 72.0 / 2.0
    expected = geometry + np.array(
        [-stroke_radius, -stroke_radius, stroke_radius, stroke_radius]
    )

    try:
        _assert_px_close(
            _bounds(get_artist_adapter(target).selection_points()), expected
        )
    finally:
        plt.close(fig)


def test_thick_patch_resize_keeps_stroke_fixed_and_preview_matches_commit() -> None:
    fig, ax = plt.subplots(figsize=(5, 4), dpi=100)
    fig.change_tracker = RecordingChangeTracker()
    target = ax.add_patch(
        Rectangle(
            (0.2, 0.25),
            0.4,
            0.3,
            facecolor="none",
            edgecolor="black",
            linewidth=18,
        )
    )
    fig.canvas.draw()
    adapter = get_artist_adapter(target)
    visible_before = np.asarray(adapter.selection_points(), dtype=float)
    controls_before = np.asarray(adapter.control_points(), dtype=float)
    matrix = np.array(
        [[1.2, 0.0, -25.0], [0.0, 0.8, 20.0], [0.0, 0.0, 1.0]],
        dtype=float,
    )
    preview_visible = adapter.preview_resize_selection_points(matrix)
    preview_controls = adapter.preview_resize_control_points(matrix)

    try:
        assert not np.allclose(
            preview_controls, adapter._transform_points(matrix, controls_before)
        )
        adapter.resize(matrix)
        fig.canvas.draw()
        _assert_px_close(adapter.control_points(), preview_controls)
        _assert_px_close(adapter.selection_points(), preview_visible)
        assert target.get_linewidth() == pytest.approx(18)
        _assert_px_close(
            _bounds(preview_visible),
            _bounds(adapter._transform_points(matrix, visible_before)),
        )
    finally:
        plt.close(fig)


def test_thick_patch_resize_clamps_before_fixed_stroke_would_invert() -> None:
    fig, ax = plt.subplots(figsize=(5, 4), dpi=100)
    fig.change_tracker = RecordingChangeTracker()
    target = ax.add_patch(
        Rectangle(
            (0.2, 0.25),
            0.4,
            0.3,
            facecolor="none",
            edgecolor="black",
            linewidth=30,
            label="qa-collapse",
        )
    )
    fig.canvas.draw()
    adapter = get_artist_adapter(target)
    visible = _bounds(adapter.selection_points())
    desired_width = 5.0
    scale_x = desired_width / (visible[2] - visible[0])
    matrix = np.array(
        [
            [scale_x, 0.0, visible[0] * (1.0 - scale_x)],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    preview = _bounds(adapter.preview_resize_selection_points(matrix))
    minimum_width = target.get_linewidth() * fig.dpi / 72.0

    try:
        assert preview[0] == pytest.approx(visible[0])
        assert preview[2] - preview[0] == pytest.approx(
            minimum_width, abs=PIXEL_TOLERANCE
        )
        adapter.resize(matrix)
        fig.canvas.draw()
        _assert_px_close(_bounds(adapter.selection_points()), preview)
        assert target.get_width() == pytest.approx(0.0, abs=1e-12)
    finally:
        plt.close(fig)


def test_curved_path_patch_resize_preview_uses_rendered_path_not_control_hull() -> None:
    fig, ax = plt.subplots(figsize=(5, 4), dpi=100)
    fig.change_tracker = RecordingChangeTracker()
    path = Path(
        [
            [0.2, 0.2],
            [0.2, 1.0],
            [0.8, 1.0],
            [0.8, 0.2],
            [0.2, 0.2],
        ],
        [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4, Path.CLOSEPOLY],
    )
    target = ax.add_patch(
        PathPatch(
            path,
            facecolor="none",
            edgecolor="black",
            linewidth=18,
            label="qa-curved-path",
        )
    )
    fig.canvas.draw()
    adapter = get_artist_adapter(target)
    visible_before = np.asarray(adapter.selection_points(), dtype=float)
    matrix = np.array(
        [[1.18, 0.0, -22.0], [0.0, 0.72, 35.0], [0.0, 0.0, 1.0]],
        dtype=float,
    )
    preview_visible = adapter.preview_resize_selection_points(matrix)
    preview_controls = adapter.preview_resize_control_points(matrix)

    try:
        adapter.resize(matrix)
        fig.canvas.draw()
        _assert_px_close(adapter.control_points(), preview_controls)
        _assert_px_close(adapter.selection_points(), preview_visible)
        _assert_px_close(
            _bounds(preview_visible),
            _bounds(adapter._transform_points(matrix, visible_before)),
        )
    finally:
        plt.close(fig)


def test_line_selection_bounds_include_visible_stroke_width() -> None:
    fig, ax = plt.subplots(figsize=(5, 4), dpi=100)
    fig.change_tracker = RecordingChangeTracker()
    target = ax.plot(
        [0.3, 0.7],
        [0.5, 0.5],
        linewidth=20,
    )[0]
    fig.canvas.draw()
    bounds = _bounds(get_artist_adapter(target).selection_points())
    minimum_stroke_width = target.get_linewidth() * fig.dpi / 72.0

    try:
        assert bounds[3] - bounds[1] >= minimum_stroke_width - PIXEL_TOLERANCE
    finally:
        plt.close(fig)


def test_line_marker_selection_bounds_include_visible_edge_stroke() -> None:
    fig, ax = plt.subplots(figsize=(5, 4), dpi=100)
    fig.change_tracker = RecordingChangeTracker()
    target = ax.plot(
        [0.3, 0.7],
        [0.5, 0.5],
        linestyle="none",
        marker="o",
        markersize=20,
        markerfacecolor="none",
        markeredgecolor="black",
        markeredgewidth=12,
    )[0]
    fig.canvas.draw()
    bounds = _bounds(get_artist_adapter(target).selection_points())
    expected_height = (20 + 12) * fig.dpi / 72.0

    try:
        assert bounds[3] - bounds[1] == pytest.approx(
            expected_height, abs=PIXEL_TOLERANCE
        )
    finally:
        plt.close(fig)


def test_path_collection_selection_bounds_use_each_marker_size() -> None:
    built = _build_case(
        next(case for case in ARTIST_CASES if case.name == "PathCollection")
    )
    target = built.target
    centers = np.asarray(
        target.get_offset_transform().transform(target.get_offsets()), dtype=float
    )
    sizes = np.asarray(target.get_sizes(), dtype=float)
    linewidths = np.asarray(target.get_linewidths(), dtype=float)
    if len(linewidths) == 1:
        linewidths = np.repeat(linewidths, len(sizes))
    radii = (
        np.sqrt(sizes) * built.figure.dpi / 72.0 / 2.0
        + linewidths * built.figure.dpi / 72.0 / 2.0
    )
    expected = np.array(
        [
            np.min(centers[:, 0] - radii),
            np.min(centers[:, 1] - radii),
            np.max(centers[:, 0] + radii),
            np.max(centers[:, 1] + radii),
        ]
    )

    try:
        _assert_px_close(_bounds(built.adapter.selection_points()), expected)
    finally:
        plt.close(built.figure)


def test_path_collection_style_arrays_do_not_create_phantom_items() -> None:
    fig, ax = plt.subplots(figsize=(5, 4), dpi=100)
    fig.change_tracker = RecordingChangeTracker()
    target = ax.scatter(
        [0.5],
        [0.5],
        s=[100.0],
        facecolors="none",
        edgecolors=["black", "black"],
        linewidths=[1.0, 30.0],
    )
    fig.canvas.draw()
    adapter = get_artist_adapter(target)
    rendered_groups = adapter.display_groups()
    padding = target.get_linewidths()[0] * fig.dpi / 72.0 / 2.0
    expected = _bounds(rendered_groups[0]) + np.array(
        [-padding, -padding, padding, padding]
    )

    try:
        assert len(rendered_groups) == 1
        _assert_px_close(_bounds(adapter.selection_points()), expected)
    finally:
        plt.close(fig)


def test_masked_path_collection_skips_invalid_items_and_translates_finite_bounds() -> None:
    fig, ax = plt.subplots(figsize=(5, 4), dpi=100)
    fig.change_tracker = RecordingChangeTracker()
    x = np.ma.array([0.2, 0.5, 0.8], mask=[False, True, False])
    y = np.ma.array([0.3, 0.6, 0.7], mask=[False, True, False])
    target = ax.scatter(
        x,
        y,
        s=np.ma.array([25.0, 400.0, 100.0], mask=[False, True, False]),
        linewidths=np.ma.array([1.0, 20.0, 3.0], mask=[False, True, False]),
    )
    with pytest.warns(UserWarning, match="converting a masked element"):
        fig.canvas.draw()
    adapter = get_artist_adapter(target)
    before = _bounds(adapter.selection_points())
    offsets_before = np.ma.asarray(target.get_offsets()).copy()

    try:
        assert np.all(np.isfinite(before))
        adapter.translate(TRANSLATION)
        with pytest.warns(UserWarning, match="converting a masked element"):
            fig.canvas.draw()
        moved_offsets = adapter.point_array(target.get_offsets())
        command = adapter.serialize_changes()[0].command
        assert "np.nan" in command
        _assert_px_close(
            _bounds(adapter.selection_points()),
            before + np.tile(TRANSLATION, 2),
        )

        target.set_offsets(offsets_before)
        exec(f"target{command}", {"target": target, "np": np})
        np.testing.assert_allclose(
            adapter.point_array(target.get_offsets()),
            moved_offsets,
            rtol=1e-12,
            atol=1e-15,
            equal_nan=True,
        )
    finally:
        plt.close(fig)


def test_line_collection_selection_bounds_use_each_segment_linewidth() -> None:
    fig, ax = plt.subplots(figsize=(5, 4), dpi=100)
    fig.change_tracker = RecordingChangeTracker()
    target = LineCollection(
        [
            [[0.2, 0.3], [0.4, 0.3]],
            [[0.6, 0.7], [0.8, 0.7]],
        ],
        linewidths=[2.0, 10.0],
    )
    ax.add_collection(target)
    fig.canvas.draw()
    adapter = get_artist_adapter(target)
    groups = adapter.display_groups()
    radii = np.asarray(target.get_linewidths()) * fig.dpi / 72.0 / 2.0
    expected_y = np.array(
        [
            min(np.min(group[:, 1]) - radius for group, radius in zip(groups, radii)),
            max(np.max(group[:, 1]) + radius for group, radius in zip(groups, radii)),
        ]
    )
    bounds = _bounds(adapter.selection_points())

    try:
        _assert_px_close(bounds[[1, 3]], expected_y)
    finally:
        plt.close(fig)


def test_line_collection_transform_and_replay_preserve_nan_path_breaks() -> None:
    fig, ax = plt.subplots(figsize=(5, 4), dpi=100)
    fig.change_tracker = RecordingChangeTracker()
    vertices = np.array(
        [
            [0.1, 0.1],
            [0.2, 0.2],
            [np.nan, np.nan],
            [0.8, 0.8],
            [0.9, 0.9],
        ]
    )
    target = LineCollection([vertices], linewidths=[4.0])
    ax.add_collection(target)
    fig.canvas.draw()
    adapter = get_artist_adapter(target)
    vertices_before = target.get_paths()[0].vertices.copy()
    bounds_before = _bounds(adapter.selection_points())

    try:
        adapter.translate(np.zeros(2))
        np.testing.assert_allclose(
            target.get_paths()[0].vertices,
            vertices_before,
            rtol=0,
            atol=0,
            equal_nan=True,
        )

        adapter.translate(TRANSLATION)
        fig.canvas.draw()
        moved_vertices = target.get_paths()[0].vertices.copy()
        command = adapter.serialize_changes()[0].command
        assert moved_vertices.shape == vertices_before.shape
        assert np.isnan(moved_vertices[2]).all()
        assert "np.nan" in command
        _assert_px_close(
            _bounds(adapter.selection_points()),
            bounds_before + np.tile(TRANSLATION, 2),
        )

        target.set_segments([vertices_before])
        exec(f"target{command}", {"target": target, "np": np})
        np.testing.assert_allclose(
            target.get_paths()[0].vertices,
            moved_vertices,
            rtol=1e-12,
            atol=1e-15,
            equal_nan=True,
        )
    finally:
        plt.close(fig)


@pytest.mark.parametrize("kind", ["line", "poly"])
def test_offset_line_and_poly_collections_follow_renderer_items_and_replay(kind) -> None:
    fig, ax = plt.subplots(figsize=(5, 4), dpi=100)
    tracker = RecordingChangeTracker()
    fig.change_tracker = tracker
    kwargs = {
        "offsets": [[0.2, 0.2], [0.8, 0.8]],
        "transOffset": ax.transData,
        "transform": IdentityTransform(),
    }
    if kind == "line":
        target = LineCollection(
            [[[-10.0, 0.0], [10.0, 0.0]]], linewidths=[4.0], **kwargs
        )
    else:
        target = PolyCollection(
            [[[-10.0, -5.0], [10.0, -5.0], [10.0, 8.0], [-10.0, 8.0]]],
            **kwargs,
        )
    ax.add_collection(target)
    fig.canvas.draw()
    adapter = get_artist_adapter(target)
    paths_before = [path.vertices.copy() for path in target.get_paths()]
    offsets_before = np.asarray(target.get_offsets(), dtype=float).copy()
    controls_before = np.asarray(adapter.control_points(), dtype=float)
    groups = adapter.display_groups()
    paddings = adapter.selection_paddings(len(groups))
    expected = _bounds(
        np.concatenate(
            [adapter.bounds_points(group, float(padding)) for group, padding in zip(groups, paddings)]
        )
    )
    bounds_before = _bounds(adapter.selection_points())

    try:
        assert adapter.capabilities.editable
        assert len(groups) == len(offsets_before)
        _assert_px_close(bounds_before, expected)

        adapter.translate(TRANSLATION)
        fig.canvas.draw()
        moved_offsets = np.asarray(target.get_offsets(), dtype=float).copy()
        moved_bounds = _bounds(adapter.selection_points())
        records = adapter.serialize_changes()
        _assert_px_close(adapter.control_points(), controls_before + TRANSLATION)
        _assert_px_close(moved_bounds, bounds_before + np.tile(TRANSLATION, 2))
        assert any(record.command.startswith(".set_offsets") for record in records)
        for actual, expected in zip(target.get_paths(), paths_before):
            np.testing.assert_array_equal(actual.vertices, expected)

        target.set_offsets(offsets_before)
        for record in records:
            exec(f"target{record.command}", {"target": target, "np": np})
        fig.canvas.draw()
        np.testing.assert_allclose(target.get_offsets(), moved_offsets)
        _assert_px_close(_bounds(adapter.selection_points()), moved_bounds)
    finally:
        plt.close(fig)


def test_poly_collection_selection_bounds_use_each_polygon_linewidth() -> None:
    fig, ax = plt.subplots(figsize=(5, 4), dpi=100)
    fig.change_tracker = RecordingChangeTracker()
    target = PolyCollection(
        [
            [[0.2, 0.2], [0.4, 0.2], [0.4, 0.35], [0.2, 0.35]],
            [[0.6, 0.65], [0.8, 0.65], [0.8, 0.8], [0.6, 0.8]],
        ],
        linewidths=[2.0, 10.0],
        edgecolors="black",
    )
    ax.add_collection(target)
    fig.canvas.draw()
    adapter = get_artist_adapter(target)
    groups = adapter.display_groups()
    radii = np.asarray(target.get_linewidths()) * fig.dpi / 72.0 / 2.0
    expected_y = np.array(
        [
            min(np.min(group[:, 1]) - radius for group, radius in zip(groups, radii)),
            max(np.max(group[:, 1]) + radius for group, radius in zip(groups, radii)),
        ]
    )
    bounds = _bounds(adapter.selection_points())

    try:
        _assert_px_close(bounds[[1, 3]], expected_y)
    finally:
        plt.close(fig)


def test_poly_collection_invisible_edges_add_no_selection_padding() -> None:
    fig, ax = plt.subplots(figsize=(5, 4), dpi=100)
    fig.change_tracker = RecordingChangeTracker()
    target = PolyCollection(
        [
            [[0.2, 0.2], [0.4, 0.2], [0.4, 0.35], [0.2, 0.35]],
            [[0.6, 0.65], [0.8, 0.65], [0.8, 0.8], [0.6, 0.8]],
        ],
        linewidths=[2.0, 30.0],
        edgecolors="none",
    )
    ax.add_collection(target)
    fig.canvas.draw()
    adapter = get_artist_adapter(target)
    geometry = adapter.bounds_points(np.concatenate(adapter.display_groups()))

    try:
        _assert_px_close(adapter.selection_points(), geometry)
    finally:
        plt.close(fig)


@pytest.mark.parametrize("operation", ["translate", "resize"])
def test_editor_group_records_each_member_change_once(operation) -> None:
    built = _build_case(
        next(case for case in ARTIST_CASES if case.name == "EditorGroup")
    )
    expected_calls = sum(
        len(get_artist_adapter(member).serialize_changes())
        for member in built.target.members
    )

    try:
        if operation == "translate":
            built.adapter.translate(TRANSLATION)
        else:
            built.adapter.resize(
                np.array(
                    [[1.05, 0.0, -5.0], [0.0, 0.95, 7.0], [0.0, 0.0, 1.0]]
                )
            )
        assert len(built.tracker.calls) == expected_calls
    finally:
        plt.close(built.figure)


def test_editor_group_resize_reapplies_each_members_fixed_stroke_outset() -> None:
    fig, ax = plt.subplots(figsize=(5, 4), dpi=100)
    tracker = RecordingChangeTracker()
    fig.change_tracker = tracker
    first = ax.add_patch(
        Rectangle(
            (0.15, 0.2),
            0.18,
            0.22,
            facecolor="none",
            edgecolor="black",
            linewidth=4,
            label="qa-group-first",
        )
    )
    second = ax.add_patch(
        Rectangle(
            (0.55, 0.3),
            0.2,
            0.26,
            facecolor="none",
            edgecolor="black",
            linewidth=18,
            label="qa-group-second",
        )
    )
    group = EditorGroup(fig, "qa-thick-group", [first, second], name="QA Thick")
    fig.canvas.draw()
    adapter = get_artist_adapter(group)
    visible_before = np.asarray(adapter.selection_points(), dtype=float)
    matrix = np.array(
        [[1.15, 0.0, -20.0], [0.0, 0.85, 18.0], [0.0, 0.0, 1.0]],
        dtype=float,
    )
    preview_visible = adapter.preview_resize_selection_points(matrix)
    preview_controls = adapter.preview_resize_control_points(matrix)
    expected_calls = sum(
        len(get_artist_adapter(member).serialize_changes())
        for member in group.members
    )

    try:
        adapter.resize(matrix)
        fig.canvas.draw()
        _assert_px_close(adapter.control_points(), preview_controls)
        _assert_px_close(
            _bounds(adapter.selection_points()), _bounds(preview_visible)
        )
        _assert_px_close(
            _bounds(preview_visible),
            _bounds(adapter._transform_points(matrix, visible_before)),
        )
        assert len(tracker.calls) == expected_calls
        assert first.get_linewidth() == pytest.approx(4)
        assert second.get_linewidth() == pytest.approx(18)
    finally:
        plt.close(fig)


def test_editor_group_selection_bounds_exclude_hidden_members() -> None:
    fig, ax = plt.subplots(figsize=(5, 4), dpi=100)
    fig.change_tracker = RecordingChangeTracker()
    visible = ax.add_patch(
        Rectangle((0.1, 0.2), 0.2, 0.25, label="qa-visible-member")
    )
    hidden = ax.add_patch(
        Rectangle((0.7, 0.65), 0.2, 0.2, label="qa-hidden-member")
    )
    hidden.set_visible(False)
    group = EditorGroup(fig, "qa-visible-group", [visible, hidden], name="QA Visible")
    fig.canvas.draw()

    try:
        _assert_px_close(
            _bounds(get_artist_adapter(group).selection_points()),
            _bounds(get_artist_adapter(visible).selection_points()),
        )
    finally:
        plt.close(fig)


def test_axes_image_translation_does_not_move_the_viewport() -> None:
    built = _build_case(next(case for case in ARTIST_CASES if case.name == "AxesImage"))
    limits_before = (built.axes.get_xlim(), built.axes.get_ylim())
    position_before = _axis_position(built.axes)

    try:
        built.adapter.translate(TRANSLATION)
        built.figure.canvas.draw()
        assert built.axes.get_xlim() == limits_before[0]
        assert built.axes.get_ylim() == limits_before[1]
        np.testing.assert_allclose(
            _axis_position(built.axes), position_before, atol=0, rtol=0
        )
    finally:
        plt.close(built.figure)


@pytest.mark.parametrize(
    "kind",
    ["text-data", "text-axes", "text-figure", "text-display", "line-log"],
)
def test_coordinate_system_variants_translate_in_display_space(kind) -> None:
    fig, ax = plt.subplots(figsize=(5, 4), dpi=100)
    fig.change_tracker = RecordingChangeTracker()
    ax.set_xlim(0.1, 10)
    ax.set_ylim(0.1, 10)
    if kind == "text-data":
        target = ax.text(1.8, 3.0, "data", transform=ax.transData)
    elif kind == "text-axes":
        target = ax.text(0.3, 0.7, "axes", transform=ax.transAxes)
    elif kind == "text-figure":
        target = fig.text(0.3, 0.7, "figure", transform=fig.transFigure)
    elif kind == "text-display":
        from matplotlib.transforms import IdentityTransform

        target = ax.text(180, 250, "display", transform=IdentityTransform())
    else:
        ax.set_xscale("log")
        ax.set_yscale("log")
        target = ax.plot([0.2, 1.5, 7.0], [0.3, 3.0, 8.0])[0]
    fig.canvas.draw()
    wrapper = TargetWrapper(target)
    bounds_before = _bounds(wrapper.get_selection_points())
    axes_before = _axis_position(ax)

    try:
        wrapper.translate(TRANSLATION)
        fig.canvas.draw()
        _assert_px_close(
            _bounds(wrapper.get_selection_points()),
            bounds_before + np.tile(TRANSLATION, 2),
        )
        np.testing.assert_allclose(_axis_position(ax), axes_before, atol=0, rtol=0)
    finally:
        plt.close(fig)


@pytest.mark.parametrize("kind", ["path", "line", "poly"])
def test_collection_non_affine_data_transforms_translate_in_display_space(
    kind,
) -> None:
    fig, ax = plt.subplots(figsize=(5, 4), dpi=100)
    fig.change_tracker = RecordingChangeTracker()
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.1, 10)
    ax.set_ylim(0.1, 10)
    if kind == "path":
        target = ax.scatter([0.2, 1.0, 5.0], [0.3, 2.0, 7.0])
    elif kind == "line":
        target = LineCollection(
            [
                [[0.2, 0.3], [1.0, 2.0]],
                [[2.0, 0.4], [7.0, 5.0]],
            ]
        )
        ax.add_collection(target)
    else:
        target = PolyCollection(
            [
                [[0.2, 0.3], [1.0, 0.4], [0.5, 2.0]],
                [[2.0, 0.4], [7.0, 0.5], [4.0, 5.0]],
            ]
        )
        ax.add_collection(target)
    fig.canvas.draw()
    wrapper = TargetWrapper(target)
    bounds_before = _bounds(wrapper.get_selection_points())

    try:
        wrapper.translate(TRANSLATION)
        fig.canvas.draw()
        _assert_px_close(
            _bounds(wrapper.get_selection_points()),
            bounds_before + np.tile(TRANSLATION, 2),
        )
    finally:
        plt.close(fig)


def test_non_affine_fancy_bbox_is_explicitly_blocked_without_mutation() -> None:
    fig, ax = plt.subplots(figsize=(5, 4), dpi=100)
    fig.change_tracker = RecordingChangeTracker()
    ax.set_xscale("log")
    ax.set_xlim(0.1, 10)
    target = ax.add_patch(
        FancyBboxPatch((0.2, 0.25), 0.8, 0.3, boxstyle="round,pad=0.03")
    )
    fig.canvas.draw()
    adapter = get_artist_adapter(target)
    bounds_before = target.get_bbox().bounds

    try:
        assert type(adapter) is FancyBboxPatchAdapter
        assert not adapter.capabilities.editable
        with pytest.raises(TransformPreflightError):
            TransformPlan.preflight([target], TransformIntent.translate(TRANSLATION))
        assert target.get_bbox().bounds == bounds_before
    finally:
        plt.close(fig)


@pytest.mark.parametrize("patch_type", [Rectangle, Ellipse])
def test_rotated_patch_translates_exactly_and_blocks_geometry_resize(
    patch_type,
) -> None:
    fig, ax = plt.subplots(figsize=(5, 4), dpi=100)
    fig.change_tracker = RecordingChangeTracker()
    if patch_type is Rectangle:
        target = ax.add_patch(Rectangle((0.2, 0.25), 0.3, 0.2, angle=27))
    else:
        target = ax.add_patch(Ellipse((0.5, 0.5), 0.3, 0.2, angle=27))
    fig.canvas.draw()
    wrapper = TargetWrapper(target)
    bounds_before = _bounds(wrapper.get_selection_points())

    try:
        assert not wrapper.supports_operation(TransformOperation.RESIZE_GEOMETRY)
        wrapper.translate(TRANSLATION)
        fig.canvas.draw()
        _assert_px_close(
            _bounds(wrapper.get_selection_points()),
            bounds_before + np.tile(TRANSLATION, 2),
        )
        with pytest.raises(UnsupportedArtistError):
            wrapper.resize(np.eye(3))
    finally:
        plt.close(fig)


def test_half_turn_rectangle_resize_is_denied_for_xy_rotation_anchor() -> None:
    fig, ax = plt.subplots(figsize=(5, 4), dpi=100)
    tracker = RecordingChangeTracker()
    fig.change_tracker = tracker
    target = ax.add_patch(
        Rectangle(
            (0.2, 0.25),
            0.3,
            0.2,
            angle=180,
            facecolor="none",
            edgecolor="black",
            linewidth=12,
            label="qa-half-turn",
        )
    )
    fig.canvas.draw()
    adapter = get_artist_adapter(target)
    state_before = adapter.snapshot()
    tracker_before = tracker.capture_recording_state()

    try:
        assert not adapter.operation_support(
            TransformOperation.RESIZE_GEOMETRY
        ).supported
        with pytest.raises(TransformPreflightError):
            TransformPlan.preflight(
                [target],
                TransformIntent.resize(
                    np.array(
                        [[1.2, 0.0, -20.0], [0.0, 0.8, 15.0], [0.0, 0.0, 1.0]]
                    )
                ),
            )
        assert semantic_equal(adapter.snapshot(), state_before)
        assert tracker.capture_recording_state() == tracker_before
    finally:
        plt.close(fig)


@pytest.mark.parametrize(("scale_x", "scale_y"), [(1.1, 1.1), (1.3, 0.7)])
def test_fixed_aspect_axes_preview_and_commit_share_native_constraint(
    scale_x, scale_y
) -> None:
    fig, ax = plt.subplots(figsize=(5, 4), dpi=100)
    fig.change_tracker = RecordingChangeTracker()
    ax.set_position([0.2, 0.2, 0.45, 0.45])
    ax.set_aspect("equal", adjustable="box")
    fig.canvas.draw()
    adapter = get_artist_adapter(ax)
    controls_before = np.asarray(adapter.control_points(), dtype=float)
    position_before = ax.get_position()
    aspect_before = position_before.height / position_before.width
    center = np.mean(controls_before, axis=0)
    matrix = np.array(
        [
            [scale_x, 0.0, center[0] * (1 - scale_x)],
            [0.0, scale_y, center[1] * (1 - scale_y)],
            [0.0, 0.0, 1.0],
        ]
    )
    support = adapter.operation_support(TransformOperation.RESIZE_GEOMETRY)
    plan = TransformPlan.preflight([ax], TransformIntent.resize(matrix))
    preview_controls = plan.preview_control_points()[0]
    preview_selection = adapter.preview_resize_selection_points(matrix)

    try:
        assert adapter.capabilities.fixed_aspect
        assert support.constraints == ("fixed_aspect",)
        plan.commit()
        fig.canvas.draw()
        _assert_px_close(adapter.control_points(), preview_controls)
        _assert_px_close(adapter.selection_points(), preview_selection)
        position_after = ax.get_position()
        assert position_after.height / position_after.width == pytest.approx(
            aspect_before
        )
    finally:
        plt.close(fig)


class AtomicQAArtist(Artist):
    def __init__(self, position, *, fail=False):
        super().__init__()
        self.position = np.asarray(position, dtype=float)
        self.fail = fail

    def get_window_extent(self, renderer=None):
        from matplotlib.transforms import Bbox

        return Bbox.from_bounds(*self.position, 2.0, 2.0)


class AtomicQAAdapter(ArtistAdapter):
    default_capabilities = ArtistCapabilities(
        can_select=True,
        can_translate=True,
        can_snapshot=True,
        can_serialize=True,
    )

    def get_transform(self):
        from matplotlib.transforms import IdentityTransform

        return IdentityTransform()

    def native_control_points(self):
        return [self.target.position.copy()]

    def _apply_native_control_points(self, points) -> None:
        if self.target.fail:
            raise RuntimeError("QA planned group failure")
        self.target.position = np.asarray(points[0], dtype=float)

    def serialize_changes(self):
        from pylustrator.artist_adapters import ChangeRecord

        return (
            ChangeRecord.command_change(
                self.target, f".position = np.array({self.target.position.tolist()!r})"
            ),
        )


@contextmanager
def _registered_atomic_qa_adapter():
    artist_adapter_registry.register(AtomicQAArtist, AtomicQAAdapter)
    try:
        yield
    finally:
        artist_adapter_registry.unregister(AtomicQAArtist, AtomicQAAdapter)


def test_logical_group_failure_rolls_back_every_member_geometry() -> None:
    fig = plt.figure(figsize=(3, 2), dpi=100)
    fig.change_tracker = RecordingChangeTracker()
    first = AtomicQAArtist((20, 30))
    second = AtomicQAArtist((50, 60), fail=True)
    fig.add_artist(first)
    fig.add_artist(second)
    group = EditorGroup(fig, "qa-atomic", [first, second], name="Atomic QA")
    before = (first.position.copy(), second.position.copy())

    try:
        with _registered_atomic_qa_adapter():
            plan = TransformPlan.preflight(
                [group], TransformIntent.translate(TRANSLATION)
            )
            with pytest.raises(RuntimeError, match="QA planned group failure"):
                plan.commit()
            np.testing.assert_allclose(first.position, before[0], atol=0, rtol=0)
            np.testing.assert_allclose(second.position, before[1], atol=0, rtol=0)
    finally:
        plt.close(fig)


def test_logical_group_failure_restores_generated_change_bookkeeping() -> None:
    fig = plt.figure(figsize=(3, 2), dpi=100)
    tracker = RecordingChangeTracker()
    fig.change_tracker = tracker
    first = AtomicQAArtist((20, 30))
    second = AtomicQAArtist((50, 60), fail=True)
    fig.add_artist(first)
    fig.add_artist(second)
    group = EditorGroup(fig, "qa-atomic-records", [first, second], name="Atomic QA")
    recording_before = tracker.capture_recording_state()

    try:
        with _registered_atomic_qa_adapter():
            plan = TransformPlan.preflight(
                [group], TransformIntent.translate(TRANSLATION)
            )
            with pytest.raises(RuntimeError, match="QA planned group failure"):
                plan.commit()
            assert tracker.capture_recording_state() == recording_before
    finally:
        plt.close(fig)


def test_failed_multi_artist_rotation_rolls_back_native_angles() -> None:
    class FailingRotationRectangle(Rectangle):
        fail_rotation = False

        def set_angle(self, angle):
            if self.fail_rotation:
                raise RuntimeError("QA planned rotation failure")
            return super().set_angle(angle)

    fig, ax = plt.subplots(figsize=(5, 4), dpi=100)
    fig.change_tracker = RecordingChangeTracker()
    first = ax.add_patch(Rectangle((0.2, 0.25), 0.2, 0.3))
    second = ax.add_patch(FailingRotationRectangle((0.6, 0.25), 0.2, 0.3))
    second.fail_rotation = True
    fig.canvas.draw()

    try:
        plan = TransformPlan.preflight(
            [first, second], TransformIntent.rotate(17.0)
        )
        with pytest.raises(RuntimeError, match="QA planned rotation failure"):
            plan.commit()
        assert first.get_angle() == pytest.approx(0.0)
        assert second.get_angle() == pytest.approx(0.0)
    finally:
        plt.close(fig)


def test_rotatable_snapshot_restore_includes_rotation_state(artist_case) -> None:
    adapter = artist_case.adapter
    if not adapter.capabilities.can_rotate:
        pytest.skip("rotation is explicitly unsupported")
    state = adapter.snapshot()
    old_rotation = adapter.rotation()

    adapter.set_rotation(old_rotation + 17.0)
    adapter.restore(state)

    assert adapter.rotation() == pytest.approx(old_rotation)
