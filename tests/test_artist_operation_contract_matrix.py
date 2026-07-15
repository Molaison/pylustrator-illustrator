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
    _assert_px_close(_bounds(adapter.selection_points()), visible.extents)
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
        pytest.skip("covered by strict-xfail minimum reproduction")
    before = _observable_state(artist_case)

    with pytest.raises(UnsupportedArtistError):
        if argument is None:
            getattr(adapter, method)()
        else:
            getattr(adapter, method)(argument)

    _assert_observable_state_equal(_observable_state(artist_case), before)


@pytest.mark.xfail(
    reason=(
        "confirmed product defect: fallback translation reaches NumPy broadcasting "
        "instead of rejecting through UnsupportedArtistError"
    ),
    strict=True,
)
def test_fallback_translate_rejects_with_adapter_contract_error() -> None:
    built = _build_case(next(case for case in ARTIST_CASES if case.name == "Artist fallback"))
    before = _observable_state(built)

    try:
        with pytest.raises(UnsupportedArtistError):
            built.adapter.translate(TRANSLATION)
        _assert_observable_state_equal(_observable_state(built), before)
    finally:
        plt.close(built.figure)


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
    renderer = fig.canvas.get_renderer()
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
        expected_points.extend(legend.get_window_extent(renderer).get_points())

    try:
        _assert_px_close(
            _bounds(get_artist_adapter(legend).selection_points()),
            _bounds(expected_points),
        )
    finally:
        plt.close(fig)


@pytest.mark.xfail(
    reason=(
        "confirmed product defect: generic Patch selection bounds omit visible "
        "stroke width while the adapter contract calls them visible bounds"
    ),
    strict=True,
)
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


@pytest.mark.xfail(
    reason=(
        "confirmed product defect: Line2D selection bounds include markers but "
        "omit the visible line stroke width"
    ),
    strict=True,
)
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


@pytest.mark.xfail(
    reason=(
        "confirmed product defect: PathCollection uses one global maximum marker "
        "padding instead of the per-item visible envelope"
    ),
    strict=True,
)
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


@pytest.mark.xfail(
    reason=(
        "confirmed product defect: LineCollection applies the global maximum "
        "linewidth padding to every segment instead of a per-segment envelope"
    ),
    strict=True,
)
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


@pytest.mark.xfail(
    reason=(
        "confirmed product defect: PolyCollection applies the global maximum "
        "linewidth padding to every polygon instead of a per-item envelope"
    ),
    strict=True,
)
def test_poly_collection_selection_bounds_use_each_polygon_linewidth() -> None:
    fig, ax = plt.subplots(figsize=(5, 4), dpi=100)
    fig.change_tracker = RecordingChangeTracker()
    target = PolyCollection(
        [
            [[0.2, 0.2], [0.4, 0.2], [0.4, 0.35], [0.2, 0.35]],
            [[0.6, 0.65], [0.8, 0.65], [0.8, 0.8], [0.6, 0.8]],
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


@pytest.mark.xfail(
    reason=(
        "confirmed product defect: EditorGroup translates members through their "
        "adapters and then records the same member changes a second time"
    ),
    strict=True,
)
def test_editor_group_records_each_member_change_once() -> None:
    built = _build_case(
        next(case for case in ARTIST_CASES if case.name == "EditorGroup")
    )
    expected_calls = sum(
        len(get_artist_adapter(member).serialize_changes())
        for member in built.target.members
    )

    try:
        built.adapter.translate(TRANSLATION)
        assert len(built.tracker.calls) == expected_calls
    finally:
        plt.close(built.figure)


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


def test_fixed_aspect_axes_advertise_constraint_and_match_uniform_preview() -> None:
    fig, ax = plt.subplots(figsize=(5, 4), dpi=100)
    fig.change_tracker = RecordingChangeTracker()
    ax.set_position([0.2, 0.2, 0.45, 0.45])
    ax.set_aspect("equal", adjustable="box")
    fig.canvas.draw()
    adapter = get_artist_adapter(ax)
    controls_before = np.asarray(adapter.control_points(), dtype=float)
    center = np.mean(controls_before, axis=0)
    scale = 1.1
    matrix = np.array(
        [
            [scale, 0.0, center[0] * (1 - scale)],
            [0.0, scale, center[1] * (1 - scale)],
            [0.0, 0.0, 1.0],
        ]
    )
    support = adapter.operation_support(TransformOperation.RESIZE_GEOMETRY)
    plan = TransformPlan.preflight([ax], TransformIntent.resize(matrix))
    preview = plan.preview_control_points()[0]

    try:
        assert adapter.capabilities.fixed_aspect
        assert support.constraints == ("fixed_aspect",)
        plan.commit()
        fig.canvas.draw()
        _assert_px_close(adapter.control_points(), preview)
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


@pytest.mark.xfail(
    reason=(
        "confirmed product defect: TransformPlan rollback restores geometry but does "
        "not restore generated-change bookkeeping"
    ),
    strict=True,
)
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


@pytest.mark.xfail(
    reason=(
        "confirmed product defect: rotatable adapter snapshots omit native rotation, "
        "so a later failure leaves earlier targets rotated"
    ),
    strict=True,
)
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


@pytest.mark.xfail(
    reason=(
        "confirmed product defect: rotatable adapter snapshots omit native rotation"
    ),
    strict=True,
)
def test_rotatable_snapshot_restore_includes_rotation_state(artist_case) -> None:
    adapter = artist_case.adapter
    if not adapter.capabilities.can_rotate:
        pytest.skip("rotation is explicitly unsupported")
    state = adapter.snapshot()
    old_rotation = adapter.rotation()

    adapter.set_rotation(old_rotation + 17.0)
    adapter.restore(state)

    assert adapter.rotation() == pytest.approx(old_rotation)
