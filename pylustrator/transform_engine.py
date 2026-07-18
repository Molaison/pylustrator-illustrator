"""Atomic preflight and execution of semantic transform intents."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Iterable

import numpy as np
from matplotlib.artist import Artist

from .artist_adapters import (
    AppearanceScalePlan,
    ArtistAdapter,
    ChangeRecord,
    RigidRotationPlan,
    UnsupportedArtistError,
    get_artist_adapter,
    legend_owner_snapshot,
    selection_geometry_snapshot,
    suspend_change_recording,
)
from .legend_layout import LegendLayoutPlan
from .operations import OperationSupport, TransformIntent, TransformOperation


class TransformPreflightError(ValueError):
    def __init__(self, failures: Iterable[tuple[Artist, OperationSupport]]):
        self.failures = tuple(failures)
        details = ", ".join(
            f"{type(artist).__name__}: {support.reason}"
            for artist, support in self.failures
        )
        super().__init__(details or "Transform is not supported")


class StaleTransformPlanError(TransformPreflightError):
    """Raised when live source geometry no longer matches a frozen plan."""


@dataclass(frozen=True)
class GeometryTransformPlan:
    """One immutable absolute translation/resize destination.

    The source is retained as a compact content/context fingerprint rather
    than a second control-point array.  This keeps a 100k-point Line2D plan to
    one display destination plus one native destination while still rejecting
    in-place source, transform, limits, and layout changes before mutation.
    """

    target: Artist
    operation: TransformOperation
    source_fingerprint: object
    is_noop: bool
    control_points: np.ndarray
    native_control_points: np.ndarray
    selection_points: np.ndarray

    @staticmethod
    def _immutable_points(points) -> np.ndarray:
        values = np.asarray(points, dtype=float)
        if values.size == 0:
            values = np.empty((0, 2), dtype=float)
        if values.ndim != 2 or values.shape[1] != 2:
            raise ValueError("Geometry-plan points must have shape (N, 2)")
        values = np.ascontiguousarray(values, dtype=float)
        return np.frombuffer(values.tobytes(), dtype=float).reshape(values.shape)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "control_points", self._immutable_points(self.control_points)
        )
        object.__setattr__(
            self,
            "native_control_points",
            self._immutable_points(self.native_control_points),
        )
        object.__setattr__(
            self,
            "selection_points",
            self._immutable_points(self.selection_points),
        )

    def control_array(self) -> np.ndarray:
        return self.control_points

    def native_array(self) -> np.ndarray:
        return self.native_control_points

    def selection_array(self) -> np.ndarray:
        return self.selection_points


@dataclass(frozen=True)
class NativeRotationPlan:
    """Frozen absolute native-angle destination for one Artist."""

    target: Artist
    source_fingerprint: object
    source_value: float
    destination_value: float
    is_noop: bool
    control_points: np.ndarray
    selection_points: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "control_points",
            GeometryTransformPlan._immutable_points(self.control_points),
        )
        object.__setattr__(
            self,
            "selection_points",
            GeometryTransformPlan._immutable_points(self.selection_points),
        )

    def control_array(self) -> np.ndarray:
        return self.control_points

    def selection_array(self) -> np.ndarray:
        return self.selection_points


def _array_digest(points) -> tuple[tuple[int, ...], str, bytes]:
    values = np.ascontiguousarray(np.asarray(points, dtype=float))
    hasher = hashlib.sha256()
    if values.size:
        hasher.update(memoryview(values).cast("B"))
    return (
        tuple(int(value) for value in values.shape),
        values.dtype.str,
        hasher.digest()[:16],
    )


def _finite_tuple(values) -> tuple:
    try:
        array = np.asarray(values, dtype=float).ravel()
    except (TypeError, ValueError):
        return ()
    result = []
    for value in array:
        value = float(value)
        if np.isnan(value):
            result.append("nan")
        elif np.isposinf(value):
            result.append("+inf")
        elif np.isneginf(value):
            result.append("-inf")
        else:
            result.append(value)
    return tuple(result)


def _geometry_context_fingerprint(adapter: ArtistAdapter) -> tuple:
    """Capture transform, viewport, and layout state that controls geometry."""

    target = adapter.target
    try:
        transform = adapter.get_transform()
        transform_type = (
            type(transform).__module__,
            type(transform).__qualname__,
        )
        affine_matrix = _finite_tuple(transform.get_affine().get_matrix())
        transform_flags = (
            bool(getattr(transform, "is_affine", False)),
            bool(getattr(transform, "has_inverse", True)),
        )
    except (AttributeError, TypeError, ValueError, RuntimeError):
        transform_type = ()
        affine_matrix = ()
        transform_flags = ()

    axes = target if hasattr(target, "get_xlim") and hasattr(target, "get_ylim") else (
        getattr(target, "axes", None)
    )
    if axes is None:
        axes_state = ()
    else:
        try:
            position = axes.get_position().bounds
        except (AttributeError, TypeError, ValueError, RuntimeError):
            position = ()
        axes_state = (
            id(axes),
            _finite_tuple(getattr(axes, "get_xlim", lambda: ())()),
            _finite_tuple(getattr(axes, "get_ylim", lambda: ())()),
            _finite_tuple(position),
            str(getattr(axes, "get_xscale", lambda: "")()),
            str(getattr(axes, "get_yscale", lambda: "")()),
            bool(getattr(axes, "get_in_layout", lambda: True)()),
        )

    figure = getattr(target, "figure", None)
    if figure is None:
        figure_state = ()
    else:
        try:
            root = target.get_figure(root=True)
        except (AttributeError, TypeError, ValueError):
            root = figure
        get_engine = getattr(root, "get_layout_engine", None)
        engine = get_engine() if callable(get_engine) else None
        params = getattr(engine, "_params", None)
        if isinstance(params, dict):
            engine_params = tuple(
                sorted(
                    (str(key), _finite_tuple([value]))
                    for key, value in params.items()
                    if isinstance(value, (int, float, np.integer, np.floating))
                )
            )
        else:
            engine_params = ()
        figure_state = (
            id(root),
            _finite_tuple(getattr(root, "get_size_inches", lambda: ())()),
            _finite_tuple([getattr(root, "dpi", np.nan)]),
            None
            if engine is None
            else (type(engine).__module__, type(engine).__qualname__),
            engine_params,
        )

    return (
        (type(target).__module__, type(target).__qualname__),
        transform_type,
        transform_flags,
        affine_matrix,
        axes_state,
        figure_state,
        bool(getattr(target, "get_visible", lambda: True)()),
        bool(getattr(target, "get_in_layout", lambda: True)()),
        bool(getattr(target, "_autopos", False)),
        bool(getattr(axes, "_autotitlepos", False)) if axes is not None else False,
    )


def _geometry_source_fingerprint(
    adapter: ArtistAdapter,
    control_points,
    selection_points,
) -> tuple:
    target = adapter.target
    adapter_token = adapter.rigid_rotation_source_fingerprint()
    member_tokens = ()
    members = getattr(target, "members", None)
    if members is not None:
        member_tokens = tuple(
            (
                id(member),
                type(member).__module__,
                type(member).__qualname__,
                get_artist_adapter(member).rigid_rotation_source_fingerprint(),
            )
            for member in members
        )
    return (
        None if adapter_token is not None else _array_digest(control_points),
        _array_digest(selection_points),
        _geometry_context_fingerprint(adapter),
        adapter_token,
        member_tokens,
    )


def _validate_display_round_trip(expected, actual, *, operation: str) -> None:
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    expected_finite = np.isfinite(expected)
    actual_finite = np.isfinite(actual)
    structure_matches = bool(
        expected.shape == actual.shape
        and np.array_equal(expected_finite, actual_finite)
        and np.array_equal(np.isnan(expected), np.isnan(actual))
        and np.array_equal(np.isposinf(expected), np.isposinf(actual))
        and np.array_equal(np.isneginf(expected), np.isneginf(actual))
    )
    if expected.shape == actual.shape and np.any(expected_finite):
        error = float(
            np.max(np.abs(expected[expected_finite] - actual[expected_finite]))
        )
    elif structure_matches:
        error = 0.0
    else:
        error = float("inf")
    if not structure_matches or error > 0.25:
        raise UnsupportedArtistError(
            f"{operation} destination cannot round-trip through native "
            f"coordinates within 0.25 px (error {error:.6g} px)"
        )


def _plan_geometry_transform(
    adapter: ArtistAdapter, intent: TransformIntent
) -> GeometryTransformPlan:
    source_control = adapter.point_array(adapter.control_points())
    source_selection = adapter.point_array(adapter.selection_points())
    source_fingerprint = _geometry_source_fingerprint(
        adapter, source_control, source_selection
    )

    if intent.operation is TransformOperation.TRANSLATE:
        delta = np.asarray(intent.delta, dtype=float)
        is_noop = bool(np.array_equal(delta, np.zeros(2, dtype=float)))
        requested_control = source_control + delta
        destination_selection = adapter.clip_selection_points(
            source_selection + delta
        )
        adapter.preflight_translation(
            delta,
            control_points=source_control,
            selection_points=source_selection,
            destination_selection_points=destination_selection,
        )
    elif intent.operation is TransformOperation.RESIZE_GEOMETRY:
        matrix = np.asarray(intent.matrix, dtype=float)
        is_noop = bool(np.array_equal(matrix, np.eye(3, dtype=float)))
        destination_selection = adapter.preflight_resize(
            matrix,
            control_points=source_control,
            selection_points=source_selection,
        )
        requested_control = adapter.preview_resize_control_points(
            matrix,
            control_points=source_control,
            selection_points=source_selection,
        )
    else:  # pragma: no cover - caller restricts geometry operations
        raise ValueError(f"No frozen geometry planner for {intent.operation.value}")

    try:
        native = adapter.point_array(adapter.display_to_native(requested_control))
        native = adapter.point_array(
            adapter.canonicalize_native_control_points(native)
        )
        representable = adapter.point_array(adapter.native_to_display(native))
    except UnsupportedArtistError:
        raise
    except (
        AttributeError,
        TypeError,
        ValueError,
        NotImplementedError,
        RuntimeError,
        np.linalg.LinAlgError,
    ) as error:
        raise UnsupportedArtistError(
            f"{type(adapter.target).__name__} {intent.operation.value} cannot "
            "convert its display destination through native coordinates"
        ) from error
    _validate_display_round_trip(
        requested_control,
        representable,
        operation=f"{type(adapter.target).__name__} {intent.operation.value}",
    )
    if is_noop:
        # Preserve the exact frozen source display truth for strict semantic
        # no-ops; an inverse/forward floating round trip must not manufacture
        # a change record for zero translation or identity resize.
        representable = source_control
    return GeometryTransformPlan(
        target=adapter.target,
        operation=intent.operation,
        source_fingerprint=source_fingerprint,
        is_noop=is_noop,
        control_points=representable,
        native_control_points=native,
        selection_points=destination_selection,
    )


def _revalidate_destination_selection(
    adapter: ArtistAdapter,
    intent: TransformIntent,
    source_control,
    source_selection,
) -> np.ndarray:
    """Re-run destination visibility preflight against current clip semantics."""

    if intent.operation is TransformOperation.TRANSLATE:
        delta = np.asarray(intent.delta, dtype=float)
        destination = adapter.clip_selection_points(source_selection + delta)
        adapter.preflight_translation(
            delta,
            control_points=source_control,
            selection_points=source_selection,
            destination_selection_points=destination,
        )
        return adapter.point_array(destination)
    if intent.operation is TransformOperation.RESIZE_GEOMETRY:
        return adapter.point_array(
            adapter.preflight_resize(
                np.asarray(intent.matrix, dtype=float),
                control_points=source_control,
                selection_points=source_selection,
            )
        )
    raise ValueError(
        f"No destination visibility preflight for {intent.operation.value}"
    )


def _plan_native_rotation(
    adapter: ArtistAdapter, intent: TransformIntent
) -> NativeRotationPlan:
    source_control = adapter.point_array(adapter.control_points())
    source_selection = adapter.point_array(adapter.selection_points())
    source_value = float(adapter.rotation())
    delta = float(intent.angle_degrees)
    if not np.isfinite(source_value) or not np.isfinite(delta):
        raise ValueError("Native rotation requires finite source and delta angles")
    is_noop = bool(np.isclose(delta % 360.0, 0.0, atol=1e-12, rtol=0.0))
    destination = source_value if is_noop else source_value + delta
    destination_selection = adapter.preview_native_rotation_selection_points(
        destination
    )
    if not len(adapter.finite_points(destination_selection)):
        raise UnsupportedArtistError(
            f"{type(adapter.target).__name__} native rotation would leave no "
            "visible geometry inside the active clip region"
        )
    return NativeRotationPlan(
        target=adapter.target,
        source_fingerprint=_geometry_source_fingerprint(
            adapter, source_control, source_selection
        ),
        source_value=source_value,
        destination_value=destination,
        is_noop=is_noop,
        control_points=source_control,
        selection_points=destination_selection,
    )


@dataclass(frozen=True)
class TransformPlan:
    intent: TransformIntent
    adapters: tuple[ArtistAdapter, ...]
    rigid_rotation_plans: tuple[RigidRotationPlan, ...] = ()
    appearance_scale_plans: tuple[AppearanceScalePlan, ...] = ()
    legend_layout_plans: tuple[LegendLayoutPlan, ...] = ()
    geometry_plans: tuple[GeometryTransformPlan, ...] = ()
    native_rotation_plans: tuple[NativeRotationPlan, ...] = ()

    @classmethod
    @selection_geometry_snapshot()
    def preflight(
        cls, targets: Iterable[Artist], intent: TransformIntent
    ) -> "TransformPlan":
        targets = tuple(targets)
        adapters = tuple(get_artist_adapter(target) for target in targets)
        failures = []
        rigid_rotation_plans = []
        appearance_scale_plans = []
        legend_layout_plans = []
        geometry_plans = []
        native_rotation_plans = []
        for adapter in adapters:
            support = adapter.operation_support(intent.operation)
            if not support.supported:
                failures.append((adapter.target, support))
                continue
            try:
                if intent.operation in {
                    TransformOperation.TRANSLATE,
                    TransformOperation.RESIZE_GEOMETRY,
                }:
                    geometry_plans.append(
                        _plan_geometry_transform(adapter, intent)
                    )
                elif intent.operation is TransformOperation.ROTATE:
                    native_rotation_plans.append(
                        _plan_native_rotation(adapter, intent)
                    )
                elif intent.operation is TransformOperation.RIGID_ROTATE:
                    rigid_rotation_plans.append(
                        adapter.plan_rigid_rotation(
                            float(intent.angle_degrees), intent.pivot
                        )
                    )
                elif intent.operation is TransformOperation.SCALE_APPEARANCE:
                    appearance_scale_plans.append(
                        adapter._plan_preflighted_appearance_scale(
                            float(intent.factor)
                        )
                    )
                elif intent.operation is TransformOperation.REFLOW_LAYOUT:
                    legend_layout_plans.append(
                        adapter.plan_layout_reflow(
                            intent.layout_spec,
                            selected_artists=targets,
                        )
                    )
            except (TypeError, ValueError) as error:
                failures.append(
                    (
                        adapter.target,
                        OperationSupport.denied(intent.operation, str(error)),
                    )
                )
        if failures:
            raise TransformPreflightError(failures)
        return cls(
            intent,
            adapters,
            tuple(rigid_rotation_plans),
            tuple(appearance_scale_plans),
            tuple(legend_layout_plans),
            tuple(geometry_plans),
            tuple(native_rotation_plans),
        )

    def preview_control_points(self) -> tuple[np.ndarray, ...]:
        operation = self.intent.operation
        if operation in {
            TransformOperation.TRANSLATE,
            TransformOperation.RESIZE_GEOMETRY,
        }:
            return tuple(plan.control_array() for plan in self.geometry_plans)
        if operation is TransformOperation.ROTATE:
            return tuple(
                plan.control_array() for plan in self.native_rotation_plans
            )
        if operation is TransformOperation.RIGID_ROTATE:
            return tuple(
                plan.control_array() for plan in self.rigid_rotation_plans
            )
        if operation is TransformOperation.REFLOW_LAYOUT:
            raise ValueError(
                "Legend layout has no geometry control-point preview; commit and draw"
            )
        return tuple(
            np.asarray(adapter.control_points(), dtype=float)
            for adapter in self.adapters
        )

    def preview_selection_points(self) -> tuple[np.ndarray, ...]:
        """Return the visible destination carried by the immutable plan."""

        operation = self.intent.operation
        if operation in {
            TransformOperation.TRANSLATE,
            TransformOperation.RESIZE_GEOMETRY,
        }:
            return tuple(plan.selection_array() for plan in self.geometry_plans)
        if operation is TransformOperation.ROTATE:
            return tuple(
                plan.selection_array() for plan in self.native_rotation_plans
            )
        if operation is TransformOperation.RIGID_ROTATE:
            return tuple(
                plan.selection_array() for plan in self.rigid_rotation_plans
            )
        if operation is TransformOperation.SCALE_APPEARANCE:
            return tuple(
                plan.selection_array() for plan in self.appearance_scale_plans
            )
        if operation is TransformOperation.REFLOW_LAYOUT:
            raise ValueError(
                "Legend layout bounds are finalized by Matplotlib after commit and draw"
            )
        return tuple(
            np.asarray(adapter.selection_points(), dtype=float)
            for adapter in self.adapters
        )

    @selection_geometry_snapshot()
    def _revalidate_geometry_sources(self) -> tuple[bool, ...]:
        if len(self.geometry_plans) != len(self.adapters):
            raise StaleTransformPlanError(
                [
                    (
                        adapter.target,
                        OperationSupport.denied(
                            self.intent.operation,
                            "Frozen geometry-plan membership no longer matches "
                            "the target selection",
                        ),
                    )
                    for adapter in self.adapters
                ]
            )

        failures = []
        changed = []
        for adapter, plan in zip(self.adapters, self.geometry_plans):
            if plan.target is not adapter.target or plan.operation is not (
                self.intent.operation
            ):
                failures.append(
                    (
                        adapter.target,
                        OperationSupport.denied(
                            self.intent.operation,
                            "Frozen geometry plan belongs to a different target "
                            "or operation",
                        ),
                    )
                )
                changed.append(False)
                continue
            support = adapter.operation_support(self.intent.operation)
            if not support.supported:
                failures.append(
                    (
                        adapter.target,
                        OperationSupport.denied(
                            self.intent.operation,
                            "Transform plan became stale after preflight: "
                            f"{support.reason}",
                        ),
                    )
                )
                changed.append(False)
                continue
            try:
                current_control = adapter.point_array(adapter.control_points())
                current_selection = adapter.point_array(
                    adapter.selection_points()
                )
                fingerprint = _geometry_source_fingerprint(
                    adapter, current_control, current_selection
                )
            except (
                AttributeError,
                TypeError,
                ValueError,
                RuntimeError,
                np.linalg.LinAlgError,
            ) as error:
                failures.append(
                    (
                        adapter.target,
                        OperationSupport.denied(
                            self.intent.operation,
                            "Transform plan became stale because source geometry "
                            f"cannot be revalidated: {error}",
                        ),
                    )
                )
                changed.append(False)
                continue
            if fingerprint != plan.source_fingerprint:
                failures.append(
                    (
                        adapter.target,
                        OperationSupport.denied(
                            self.intent.operation,
                            "Transform plan is stale: source geometry, transform, "
                            "viewport, or layout changed after preflight",
                        ),
                    )
                )
                changed.append(False)
                continue
            try:
                live_selection = _revalidate_destination_selection(
                    adapter,
                    self.intent,
                    current_control,
                    current_selection,
                )
                _validate_display_round_trip(
                    plan.selection_array(),
                    live_selection,
                    operation=(
                        f"{type(adapter.target).__name__} stale-plan visible "
                        f"{self.intent.operation.value}"
                    ),
                )
                adapter.validate_native_control_points(plan.native_array())
                live_destination = adapter.point_array(
                    adapter.native_to_display(plan.native_array())
                )
                _validate_display_round_trip(
                    plan.control_array(),
                    live_destination,
                    operation=(
                        f"{type(adapter.target).__name__} stale-plan "
                        f"{self.intent.operation.value}"
                    ),
                )
            except (
                AttributeError,
                TypeError,
                ValueError,
                NotImplementedError,
                RuntimeError,
                np.linalg.LinAlgError,
            ) as error:
                failures.append(
                    (
                        adapter.target,
                        OperationSupport.denied(
                            self.intent.operation,
                            "Transform plan is stale: its frozen native "
                            "destination no longer maps to the accepted display "
                            f"preview within 0.25 px ({error})",
                        ),
                    )
                )
                changed.append(False)
                continue
            destination = plan.control_array()
            changed.append(
                False
                if plan.is_noop
                else not (
                    current_control.shape == destination.shape
                    and np.array_equal(
                        current_control, destination, equal_nan=True
                    )
                )
            )
        if failures:
            raise StaleTransformPlanError(failures)
        return tuple(changed)

    @selection_geometry_snapshot()
    def _revalidate_native_rotation_sources(self) -> tuple[bool, ...]:
        if len(self.native_rotation_plans) != len(self.adapters):
            raise StaleTransformPlanError(
                [
                    (
                        adapter.target,
                        OperationSupport.denied(
                            self.intent.operation,
                            "Frozen native-rotation membership no longer matches "
                            "the target selection",
                        ),
                    )
                    for adapter in self.adapters
                ]
            )
        failures = []
        changed = []
        for adapter, plan in zip(self.adapters, self.native_rotation_plans):
            support = adapter.operation_support(TransformOperation.ROTATE)
            if plan.target is not adapter.target or not support.supported:
                reason = (
                    support.reason
                    if not support.supported
                    else "plan belongs to another target"
                )
                failures.append(
                    (
                        adapter.target,
                        OperationSupport.denied(
                            TransformOperation.ROTATE,
                            "Native-rotation plan became stale after preflight: "
                            f"{reason}",
                        ),
                    )
                )
                changed.append(False)
                continue
            try:
                current_control = adapter.point_array(adapter.control_points())
                current_selection = adapter.point_array(
                    adapter.selection_points()
                )
                current_value = float(adapter.rotation())
                fingerprint = _geometry_source_fingerprint(
                    adapter, current_control, current_selection
                )
            except (
                AttributeError,
                TypeError,
                ValueError,
                RuntimeError,
                np.linalg.LinAlgError,
            ) as error:
                failures.append(
                    (
                        adapter.target,
                        OperationSupport.denied(
                            TransformOperation.ROTATE,
                            "Native-rotation plan became stale because its source "
                            f"cannot be revalidated: {error}",
                        ),
                    )
                )
                changed.append(False)
                continue
            if (
                not np.isfinite(current_value)
                or current_value != plan.source_value
                or fingerprint != plan.source_fingerprint
            ):
                failures.append(
                    (
                        adapter.target,
                        OperationSupport.denied(
                            TransformOperation.ROTATE,
                            "Native-rotation plan is stale: source angle, geometry, "
                            "transform, viewport, or layout changed after preflight",
                        ),
                    )
                )
                changed.append(False)
                continue
            try:
                live_destination = (
                    adapter.preview_native_rotation_selection_points(
                        plan.destination_value
                    )
                )
                _validate_display_round_trip(
                    plan.selection_array(),
                    live_destination,
                    operation=(
                        f"{type(adapter.target).__name__} stale native-rotation "
                        "visible destination"
                    ),
                )
            except (
                AttributeError,
                TypeError,
                ValueError,
                NotImplementedError,
                RuntimeError,
                np.linalg.LinAlgError,
            ) as error:
                failures.append(
                    (
                        adapter.target,
                        OperationSupport.denied(
                            TransformOperation.ROTATE,
                            "Native-rotation plan is stale: its destination "
                            "visible geometry or clip changed after preflight "
                            f"({error})",
                        ),
                    )
                )
                changed.append(False)
                continue
            changed.append(
                not plan.is_noop and current_value != plan.destination_value
            )
        if failures:
            raise StaleTransformPlanError(failures)
        return tuple(changed)

    @legend_owner_snapshot()
    def commit(self) -> None:
        geometry_only = self.intent.operation in {
            TransformOperation.TRANSLATE,
            TransformOperation.RESIZE_GEOMETRY,
        }
        geometry_changed = (
            self._revalidate_geometry_sources() if geometry_only else ()
        )
        if geometry_only and not any(geometry_changed):
            return
        native_rotation_only = self.intent.operation is TransformOperation.ROTATE
        native_rotation_changed = (
            self._revalidate_native_rotation_sources()
            if native_rotation_only
            else ()
        )
        if native_rotation_only and not any(native_rotation_changed):
            return
        appearance_only = (
            self.intent.operation is TransformOperation.SCALE_APPEARANCE
        )
        layout_only = self.intent.operation is TransformOperation.REFLOW_LAYOUT
        snapshots = [
            (
                adapter.appearance_state()
                if appearance_only
                else (
                    adapter.layout_state()
                    if layout_only
                    else adapter.snapshot()
                )
            )
            for adapter in self.adapters
        ]
        tracker_states = []
        seen_trackers = set()
        for adapter in self.adapters:
            try:
                tracker = adapter.change_tracker()
            except AttributeError:
                continue
            if id(tracker) in seen_trackers:
                continue
            capture = getattr(tracker, "capture_recording_state", None)
            restore = getattr(tracker, "restore_recording_state", None)
            if not callable(capture) or not callable(restore):
                continue
            seen_trackers.add(id(tracker))
            tracker_states.append((tracker, capture()))
        try:
            for index, adapter in enumerate(self.adapters):
                if geometry_only:
                    if geometry_changed[index]:
                        adapter.apply_native_control_points(
                            self.geometry_plans[index].native_array()
                        )
                elif native_rotation_only:
                    if native_rotation_changed[index]:
                        adapter.set_rotation(
                            self.native_rotation_plans[index].destination_value
                        )
                elif self.intent.operation is TransformOperation.RIGID_ROTATE:
                    adapter.apply_rigid_rotation_plan(
                        self.rigid_rotation_plans[index]
                    )
                elif self.intent.operation is TransformOperation.SCALE_APPEARANCE:
                    adapter._apply_preflighted_appearance_scale_plan(
                        self.appearance_scale_plans[index]
                    )
                elif self.intent.operation is TransformOperation.REFLOW_LAYOUT:
                    adapter.apply_layout_reflow_plan(
                        self.legend_layout_plans[index],
                        record_changes=False,
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
            if (
                self.intent.operation is TransformOperation.SCALE_APPEARANCE
                and float(self.intent.factor) != 1.0
            ):
                for adapter in self.adapters:
                    adapter._record_change_records(
                        adapter.serialize_appearance_changes()
                    )
            if self.intent.operation is TransformOperation.REFLOW_LAYOUT and any(
                plan.destination != plan.source_spec
                for plan in self.legend_layout_plans
            ):
                for adapter in self.adapters:
                    adapter._record_change_records(
                        (ChangeRecord.legend_layout_change(adapter.target),)
                    )
        except Exception as error:
            rollback_failures = []
            with suspend_change_recording():
                for adapter, state in zip(
                    reversed(self.adapters), reversed(snapshots)
                ):
                    try:
                        if appearance_only:
                            adapter.restore_appearance_state(
                                state, record_changes=False
                            )
                        elif layout_only:
                            adapter.restore_layout_state(
                                state, record_changes=False
                            )
                        else:
                            adapter.restore(state)
                    except Exception as rollback_error:
                        # Continue restoring earlier targets even when the adapter
                        # that failed the commit also cannot restore itself.
                        rollback_failures.append(
                            (adapter.target, rollback_error)
                        )
            for tracker, state in tracker_states:
                try:
                    tracker.restore_recording_state(state)
                except Exception as rollback_error:
                    rollback_failures.append((tracker, rollback_error))
            ArtistAdapter.annotate_rollback_failures(error, rollback_failures)
            raise


__all__ = [
    "GeometryTransformPlan",
    "NativeRotationPlan",
    "StaleTransformPlanError",
    "TransformPlan",
    "TransformPreflightError",
]
