"""Semantic editor operations and capability descriptors."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Sequence


class TransformOperation(str, Enum):
    SELECT = "select"
    TRANSLATE = "translate"
    RESIZE_GEOMETRY = "resize_geometry"
    SCALE_APPEARANCE = "scale_appearance"
    REFLOW_LAYOUT = "reflow_layout"
    ROTATE = "rotate"
    RIGID_ROTATE = "rigid_rotate"
    EDIT_POINTS = "edit_points"
    SNAPSHOT = "snapshot"
    SERIALIZE = "serialize"

    @classmethod
    def coerce(cls, value: "TransformOperation | str") -> "TransformOperation":
        if isinstance(value, cls):
            return value
        return cls(str(value).lower())


@dataclass(frozen=True)
class OperationSupport:
    """Whether one semantic operation is exact and how it is constrained."""

    operation: TransformOperation
    supported: bool
    reason: str = ""
    constraints: tuple[str, ...] = ()
    preview_strategy: str = "control_points"

    @classmethod
    def allowed(
        cls,
        operation: TransformOperation | str,
        *,
        constraints: Sequence[str] = (),
        preview_strategy: str = "control_points",
    ) -> "OperationSupport":
        return cls(
            TransformOperation.coerce(operation),
            True,
            constraints=tuple(constraints),
            preview_strategy=preview_strategy,
        )

    @classmethod
    def denied(
        cls, operation: TransformOperation | str, reason: str
    ) -> "OperationSupport":
        return cls(TransformOperation.coerce(operation), False, str(reason))


@dataclass(frozen=True)
class TransformIntent:
    """One user-level transform request, independent of any Artist type."""

    operation: TransformOperation
    matrix: Optional[tuple[tuple[float, float, float], ...]] = None
    delta: Optional[tuple[float, float]] = None
    angle_degrees: Optional[float] = None
    pivot: Optional[tuple[float, float]] = None
    label: str = "Transform"

    @classmethod
    def translate(cls, delta, *, label: str = "Move") -> "TransformIntent":
        return cls(
            TransformOperation.TRANSLATE,
            delta=(float(delta[0]), float(delta[1])),
            label=label,
        )

    @classmethod
    def resize(cls, matrix, *, label: str = "Resize") -> "TransformIntent":
        return cls(
            TransformOperation.RESIZE_GEOMETRY,
            matrix=tuple(tuple(float(value) for value in row) for row in matrix),
            label=label,
        )

    @classmethod
    def rotate(cls, angle_degrees, *, label: str = "Rotate") -> "TransformIntent":
        return cls(
            TransformOperation.ROTATE,
            angle_degrees=float(angle_degrees),
            label=label,
        )

    @classmethod
    def rigid_rotate(
        cls, angle_degrees, pivot, *, label: str = "Rotate"
    ) -> "TransformIntent":
        """Rotate complete display geometry around one shared pivot."""

        return cls(
            TransformOperation.RIGID_ROTATE,
            angle_degrees=float(angle_degrees),
            pivot=(float(pivot[0]), float(pivot[1])),
            label=label,
        )


__all__ = ["OperationSupport", "TransformIntent", "TransformOperation"]
