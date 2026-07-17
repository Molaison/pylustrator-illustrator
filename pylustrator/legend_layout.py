"""Identity-preserving layout reflow for standard Matplotlib Legends.

Matplotlib normally changes Legend layout by constructing a new Legend.  That
also constructs new handle and Text artists, which breaks direct-selection,
Undo/Redo, and generated-command references.  This module instead treats the
standard OffsetBox tree as layout structure: leaf DrawingArea/TextArea objects
remain authoritative while only HPacker/VPacker nodes are replaced.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import inspect
import math
from numbers import Integral, Real
from textwrap import dedent
from typing import Iterable, Mapping

from matplotlib.artist import Artist
from matplotlib.figure import Figure
from matplotlib.legend import Legend
from matplotlib.offsetbox import DrawingArea, HPacker, TextArea, VPacker
from matplotlib.transforms import IdentityTransform


LEGEND_LAYOUT_FIELDS = (
    "ncols",
    "borderpad",
    "labelspacing",
    "handlelength",
    "handletextpad",
    "columnspacing",
)


class LegendLayoutError(ValueError):
    """Raised when a Legend cannot be reflowed without replacing leaf artists."""


@dataclass(frozen=True)
class LegendLayoutSpec:
    """The deliberately small, replayable v1 Legend layout contract."""

    ncols: int
    borderpad: float
    labelspacing: float
    handlelength: float
    handletextpad: float
    columnspacing: float

    @classmethod
    def from_legend(cls, legend: Legend) -> "LegendLayoutSpec":
        ncols = getattr(legend, "_ncols", getattr(legend, "_ncol", 1))
        return cls(
            int(ncols),
            float(legend.borderpad),
            float(legend.labelspacing),
            float(legend.handlelength),
            float(legend.handletextpad),
            float(legend.columnspacing),
        )

    def with_updates(self, updates: Mapping[str, object]) -> "LegendLayoutSpec":
        normalized = dict(updates)
        if "ncol" in normalized:
            if "ncols" in normalized:
                raise LegendLayoutError("Specify only one of ncol and ncols")
            normalized["ncols"] = normalized.pop("ncol")
        unknown = sorted(set(normalized) - set(LEGEND_LAYOUT_FIELDS))
        if unknown:
            raise LegendLayoutError(
                "Unsupported Legend layout field(s): " + ", ".join(unknown)
            )
        return replace(self, **normalized)

    def replay_command(self) -> str:
        values = ", ".join(
            f"{name}={getattr(self, name)!r}" for name in LEGEND_LAYOUT_FIELDS
        )
        return f"._pylustrator_reflow_layout({values})"


@dataclass(frozen=True)
class _LegendEntry:
    drawing_area: DrawingArea
    text_area: TextArea
    children: tuple[object, object]


@dataclass(frozen=True)
class _LegendStructure:
    root: VPacker
    handle_box: HPacker
    title_box: TextArea
    entries: tuple[_LegendEntry, ...]
    root_align: str
    handle_align: str
    column_align: str
    item_align: str
    root_offset: object
    signature: tuple[object, ...]


@dataclass(frozen=True)
class LegendLayoutState:
    """Exact restorable packer state used by layout-only Undo/Redo."""

    target: Legend
    spec: LegendLayoutSpec
    root: VPacker
    handle_box: HPacker
    title_box: TextArea
    drawing_widths: tuple[tuple[DrawingArea, float], ...]


@dataclass(frozen=True)
class LegendLayoutPlan:
    """Frozen destination for one identity-preserving Legend reflow."""

    target: Legend
    source_spec: LegendLayoutSpec
    destination: LegendLayoutSpec
    structure: _LegendStructure

    @classmethod
    def preflight(
        cls,
        legend: Legend,
        destination: LegendLayoutSpec | Mapping[str, object],
        *,
        selected_artists: Iterable[Artist] = (),
    ) -> "LegendLayoutPlan":
        if not isinstance(legend, Legend):
            raise LegendLayoutError("Legend reflow requires a Matplotlib Legend")
        _validate_attached_legend(legend)
        source_spec = LegendLayoutSpec.from_legend(legend)
        if isinstance(destination, Mapping):
            destination = source_spec.with_updates(destination)
        if not isinstance(destination, LegendLayoutSpec):
            raise LegendLayoutError("Legend layout destination must be a layout spec")
        destination = _canonical_spec(destination)
        _validate_spec(destination, legend)
        _validate_selection(legend, selected_artists)
        structure = _inspect_standard_structure(legend, source_spec)
        return cls(legend, source_spec, destination, structure)

    def apply(self) -> bool:
        """Apply the plan atomically without recording generated changes."""

        if self.destination == self.source_spec:
            return False
        before = capture_legend_layout_state(self.target)
        try:
            _apply_plan(self)
        except Exception:
            restore_legend_layout_state(self.target, before)
            raise
        return True


def _canonical_spec(spec: LegendLayoutSpec) -> LegendLayoutSpec:
    if isinstance(spec.ncols, bool) or not isinstance(spec.ncols, Integral):
        raise LegendLayoutError("Legend ncols must be an integer")
    ncols = int(spec.ncols)
    if ncols < 1:
        raise LegendLayoutError("Legend ncols must be at least 1")

    values = {}
    for name in LEGEND_LAYOUT_FIELDS[1:]:
        value = getattr(spec, name)
        if isinstance(value, bool) or not isinstance(value, Real):
            raise LegendLayoutError(f"Legend {name} must be numeric")
        values[name] = float(value)
    return LegendLayoutSpec(ncols=ncols, **values)


def _validate_spec(spec: LegendLayoutSpec, legend: Legend) -> None:
    fontsize = float(getattr(legend, "_fontsize", math.nan))
    if not math.isfinite(fontsize) or fontsize <= 0.0:
        raise LegendLayoutError("Legend fontsize must be finite and positive")
    for name in LEGEND_LAYOUT_FIELDS[1:]:
        value = float(getattr(spec, name))
        if not math.isfinite(value) or value < 0.0:
            raise LegendLayoutError(f"Legend {name} must be finite and non-negative")
        if not math.isfinite(value * fontsize):
            raise LegendLayoutError(f"Legend {name} is too large to lay out")


def _validate_attached_legend(legend: Legend) -> None:
    try:
        figure = legend.get_figure(root=False)
    except TypeError:  # pragma: no cover - old Matplotlib compatibility
        figure = legend.figure
    if figure is None:
        raise LegendLayoutError("Legend is detached from its Figure")
    if legend.axes is not None:
        if legend.axes.get_legend() is legend or legend in legend.axes.artists:
            return
        raise LegendLayoutError(
            "Axes Legend is neither the current Legend nor a retained extra Legend"
        )
    parent = getattr(legend, "parent", None)
    if isinstance(parent, Figure) and legend in parent.legends:
        return
    raise LegendLayoutError(
        "Only current, retained extra, and Figure-level Legends can be reflowed"
    )


def _iter_descendants(legend: Legend) -> tuple[Artist, ...]:
    descendants = []
    seen = {id(legend)}
    stack = list(legend.get_children())
    while stack:
        child = stack.pop()
        if not isinstance(child, Artist) or id(child) in seen:
            continue
        seen.add(id(child))
        descendants.append(child)
        try:
            stack.extend(child.get_children())
        except (AttributeError, TypeError, RuntimeError):
            pass
    return tuple(descendants)


def _validate_selection(legend: Legend, selected_artists: Iterable[Artist]) -> None:
    selected = tuple(selected_artists)
    if not any(target is legend for target in selected):
        return
    descendant_ids = {id(child) for child in _iter_descendants(legend)}
    conflicts = [
        target
        for target in selected
        if target is not legend and id(target) in descendant_ids
    ]
    if conflicts:
        names = ", ".join(type(target).__name__ for target in conflicts[:3])
        raise LegendLayoutError(
            "Deselect Legend descendants before reflowing the Legend"
            + (f" ({names})" if names else "")
        )


def _close(actual: object, expected: float) -> bool:
    try:
        return math.isclose(
            float(actual), float(expected), rel_tol=1e-12, abs_tol=1e-12
        )
    except (TypeError, ValueError):
        return False


def _validate_replaceable_packer_state(box, label: str) -> None:
    """Reject Artist customizations that a fresh standard packer would lose."""

    defaults = (
        (box.get_visible(), True, "visibility"),
        (box.get_alpha(), None, "alpha"),
        (box.get_animated(), False, "animation"),
        (box.get_clip_on(), False, "clip_on"),
        (box.get_clip_box(), None, "clip box"),
        (box.get_clip_path(), None, "clip path"),
        (box.get_agg_filter(), None, "Agg filter"),
        (box.get_gid(), None, "gid"),
        (box.get_picker(), None, "picker"),
        (box.get_rasterized(), False, "rasterization"),
        (box.get_sketch_params(), None, "sketch parameters"),
        (box.get_snap(), None, "snap state"),
        (box.get_zorder(), 0, "zorder"),
        (box.get_url(), None, "URL"),
        (box.get_label(), "", "label"),
        (box.mouseover, False, "mouseover state"),
    )
    for actual, expected, state_name in defaults:
        if actual != expected:
            raise LegendLayoutError(
                f"Legend {label} has custom {state_name} that reflow cannot preserve"
            )
    if box.get_path_effects():
        raise LegendLayoutError(
            f"Legend {label} has path effects that reflow cannot preserve"
        )
    callback_registry = getattr(box, "_callbacks", None)
    callbacks = getattr(callback_registry, "callbacks", {})
    if any(callbacks.values()):
        raise LegendLayoutError(
            f"Legend {label} has custom callbacks that reflow cannot preserve"
        )
    if box.get_transform() != IdentityTransform():
        raise LegendLayoutError(
            f"Legend {label} has a custom transform that reflow cannot preserve"
        )


def _require_standard_packer(
    box,
    expected_type,
    label: str,
    *,
    pad: float,
    sep: float,
    align: str,
) -> None:
    if type(box) is not expected_type:
        raise LegendLayoutError(
            f"Legend {label} uses unsupported packer {type(box).__name__}"
        )
    _validate_replaceable_packer_state(box, label)
    if getattr(box, "mode", None) != "fixed":
        raise LegendLayoutError(f"Legend {label} must use fixed packing")
    if (
        getattr(box, "width", None) is not None
        or getattr(box, "height", None) is not None
    ):
        raise LegendLayoutError(f"Legend {label} has custom fixed dimensions")
    if not _close(getattr(box, "pad", None), pad):
        raise LegendLayoutError(f"Legend {label} has a custom pad")
    if not _close(getattr(box, "sep", None), sep):
        raise LegendLayoutError(f"Legend {label} has custom spacing")
    if getattr(box, "align", None) != align:
        raise LegendLayoutError(f"Legend {label} has custom alignment")


def _column_lengths(entry_count: int, ncols: int) -> tuple[int, ...]:
    if entry_count == 0:
        return ()
    actual_columns = min(int(ncols), entry_count)
    base, extra = divmod(entry_count, actual_columns)
    return tuple(base + (index < extra) for index in range(actual_columns))


def _contains_identity(root: Artist, target: Artist) -> bool:
    if root is target:
        return True
    stack = list(root.get_children())
    seen = {id(root)}
    while stack:
        child = stack.pop()
        if child is target:
            return True
        if not isinstance(child, Artist) or id(child) in seen:
            continue
        seen.add(id(child))
        stack.extend(child.get_children())
    return False


def _inspect_standard_structure(
    legend: Legend, spec: LegendLayoutSpec | None = None
) -> _LegendStructure:
    if spec is None:
        spec = LegendLayoutSpec.from_legend(legend)
    _validate_spec(spec, legend)
    if getattr(legend, "_mode", None) == "expand":
        raise LegendLayoutError("Legend mode='expand' cannot be reflowed losslessly")

    fontsize = float(legend._fontsize)
    root = getattr(legend, "_legend_box", None)
    handle_box = getattr(legend, "_legend_handle_box", None)
    title_box = getattr(legend, "_legend_title_box", None)
    root_align = getattr(legend, "_alignment", None)
    draggable = getattr(legend, "_draggable", None)
    if draggable is not None:
        if bool(getattr(draggable, "got_artist", False)):
            raise LegendLayoutError("Legend cannot reflow during an active native drag")
        if getattr(draggable, "offsetbox", None) is not root:
            raise LegendLayoutError(
                "Legend native draggable state points to a stale layout box"
            )
    _require_standard_packer(
        root,
        VPacker,
        "root",
        pad=spec.borderpad * fontsize,
        sep=spec.labelspacing * fontsize,
        align=root_align,
    )
    root_children = tuple(root.get_children())
    if root_children != (title_box, handle_box) or not isinstance(title_box, TextArea):
        raise LegendLayoutError(
            "Legend root does not have the standard title/handle tree"
        )
    if getattr(title_box, "_text", None) is not legend.get_title():
        raise LegendLayoutError("Legend title TextArea no longer owns the live title")

    _require_standard_packer(
        handle_box,
        HPacker,
        "handle box",
        pad=0.0,
        sep=spec.columnspacing * fontsize,
        align="baseline",
    )
    columns = tuple(handle_box.get_children())
    actual_lengths = tuple(len(column.get_children()) for column in columns)
    texts = tuple(legend.get_texts())
    handles = tuple(legend.legend_handles)
    if actual_lengths != _column_lengths(len(texts), int(spec.ncols)):
        raise LegendLayoutError("Legend entries use a non-standard column partition")

    entries = []
    column_align = None
    item_align = None
    for column_index, column in enumerate(columns):
        current_column_align = getattr(column, "align", None)
        if column_align is None:
            column_align = current_column_align
        if current_column_align != column_align or column_align not in {
            "baseline",
            "right",
        }:
            raise LegendLayoutError("Legend columns use inconsistent alignment")
        _require_standard_packer(
            column,
            VPacker,
            f"column {column_index}",
            pad=0.0,
            sep=spec.labelspacing * fontsize,
            align=column_align,
        )
        for item_index, item in enumerate(column.get_children()):
            current_item_align = getattr(item, "align", None)
            if item_align is None:
                item_align = current_item_align
            if current_item_align != item_align or item_align != "baseline":
                raise LegendLayoutError("Legend items use inconsistent alignment")
            _require_standard_packer(
                item,
                HPacker,
                f"item {column_index}:{item_index}",
                pad=0.0,
                sep=spec.handletextpad * fontsize,
                align=item_align,
            )
            children = tuple(item.get_children())
            if len(children) != 2:
                raise LegendLayoutError(
                    "Legend item does not contain exactly two leaves"
                )
            drawing_areas = [
                child for child in children if isinstance(child, DrawingArea)
            ]
            text_areas = [child for child in children if isinstance(child, TextArea)]
            if len(drawing_areas) != 1 or len(text_areas) != 1:
                raise LegendLayoutError(
                    "Legend item must contain one DrawingArea and one TextArea"
                )
            drawing_area = drawing_areas[0]
            text_area = text_areas[0]
            if not _close(drawing_area.width, spec.handlelength * fontsize):
                raise LegendLayoutError("Legend handle DrawingArea has a custom width")
            entries.append(_LegendEntry(drawing_area, text_area, children))

    if column_align is None:
        column_align = "baseline"
    if item_align is None:
        item_align = "baseline"
    if len(entries) != len(texts) or len(entries) != len(handles):
        raise LegendLayoutError("Legend packer leaves do not match its public entries")
    if tuple(entry.text_area._text for entry in entries) != texts:
        raise LegendLayoutError("Legend TextArea order differs from get_texts()")
    for entry, handle in zip(entries, handles):
        if handle is None or not _contains_identity(entry.drawing_area, handle):
            raise LegendLayoutError(
                "Legend DrawingArea does not contain its public handle identity"
            )

    signature = (
        id(root),
        id(handle_box),
        id(title_box),
        tuple(
            (
                id(entry.drawing_area),
                id(entry.text_area),
                tuple(id(child) for child in entry.children),
            )
            for entry in entries
        ),
        spec,
    )
    return _LegendStructure(
        root,
        handle_box,
        title_box,
        tuple(entries),
        root_align,
        "baseline",
        column_align,
        item_align,
        getattr(root, "_offset", (0.0, 0.0)),
        signature,
    )


def capture_legend_layout_state(legend: Legend) -> LegendLayoutState:
    spec = LegendLayoutSpec.from_legend(legend)
    structure = _inspect_standard_structure(legend, spec)
    return LegendLayoutState(
        legend,
        spec,
        structure.root,
        structure.handle_box,
        structure.title_box,
        tuple(
            (entry.drawing_area, float(entry.drawing_area.width))
            for entry in structure.entries
        ),
    )


def _set_spec(legend: Legend, spec: LegendLayoutSpec) -> None:
    if hasattr(legend, "_ncols"):
        legend._ncols = int(spec.ncols)
    else:  # pragma: no cover - retained for old Matplotlib compatibility
        legend._ncol = int(spec.ncols)
    for name in LEGEND_LAYOUT_FIELDS[1:]:
        setattr(legend, name, float(getattr(spec, name)))


def _attach_root(legend: Legend, root: VPacker) -> None:
    try:
        figure = legend.get_figure(root=False)
    except TypeError:  # pragma: no cover - old Matplotlib compatibility
        figure = legend.figure
    root.set_figure(figure)
    root.axes = legend.axes
    draggable = getattr(legend, "_draggable", None)
    if draggable is not None:
        draggable.offsetbox = root
    legend.stale = True


def restore_legend_layout_state(legend: Legend, state: LegendLayoutState) -> None:
    if state.target is not legend:
        raise LegendLayoutError("Legend layout state belongs to another object")
    _set_spec(legend, state.spec)
    for drawing_area, width in state.drawing_widths:
        drawing_area.set_width(width)
    legend._legend_title_box = state.title_box
    legend._legend_handle_box = state.handle_box
    legend._legend_box = state.root
    _attach_root(legend, state.root)


def _matplotlib_only_reflow(
    legend,
    *,
    ncols,
    borderpad,
    labelspacing,
    handlelength,
    handletextpad,
    columnspacing,
):
    """Self-contained replay primitive; its source is embedded when saving."""

    import math
    from matplotlib.offsetbox import DrawingArea, HPacker, TextArea, VPacker

    if isinstance(ncols, bool) or not isinstance(ncols, int) or ncols < 1:
        raise ValueError("Legend ncols must be a positive integer")
    values = (
        float(borderpad),
        float(labelspacing),
        float(handlelength),
        float(handletextpad),
        float(columnspacing),
    )
    if any(not math.isfinite(value) or value < 0.0 for value in values):
        raise ValueError("Legend layout dimensions must be finite and non-negative")
    fontsize = float(legend._fontsize)
    if not math.isfinite(fontsize) or fontsize <= 0.0:
        raise ValueError("Legend fontsize must be finite and positive")
    if getattr(legend, "_mode", None) == "expand":
        raise ValueError("Legend mode='expand' cannot be reflowed losslessly")

    root = legend._legend_box
    handle_box = legend._legend_handle_box
    title_box = legend._legend_title_box
    if type(root) is not VPacker or type(handle_box) is not HPacker:
        raise ValueError("Legend does not use the standard root packers")
    if tuple(root.get_children()) != (title_box, handle_box):
        raise ValueError("Legend does not use the standard title/handle tree")

    entries = []
    column_align = None
    item_align = None
    for column in handle_box.get_children():
        if type(column) is not VPacker:
            raise ValueError("Legend contains an unsupported column packer")
        if column_align is None:
            column_align = column.align
        if column.align != column_align:
            raise ValueError("Legend columns use inconsistent alignment")
        for item in column.get_children():
            if type(item) is not HPacker:
                raise ValueError("Legend contains an unsupported item packer")
            if item_align is None:
                item_align = item.align
            if item.align != item_align:
                raise ValueError("Legend items use inconsistent alignment")
            children = tuple(item.get_children())
            drawing_areas = [
                child for child in children if isinstance(child, DrawingArea)
            ]
            text_areas = [child for child in children if isinstance(child, TextArea)]
            if len(children) != 2 or len(drawing_areas) != 1 or len(text_areas) != 1:
                raise ValueError("Legend item does not have two standard leaves")
            entries.append((drawing_areas[0], text_areas[0], children))
    if column_align is None:
        column_align = "baseline"
    if item_align is None:
        item_align = "baseline"

    baseline_missing = not hasattr(legend, "_pylustrator_original_layout_spec")
    source_layout = None
    if baseline_missing:
        old_ncols = getattr(legend, "_ncols", getattr(legend, "_ncol", 1))
        source_layout = (
            int(old_ncols),
            float(legend.borderpad),
            float(legend.labelspacing),
            float(legend.handlelength),
            float(legend.handletextpad),
            float(legend.columnspacing),
        )

    item_boxes = [
        HPacker(
            pad=0.0,
            sep=values[3] * fontsize,
            align=item_align,
            mode="fixed",
            children=list(children),
        )
        for _drawing_area, _text_area, children in entries
    ]
    columns = []
    if item_boxes:
        actual_columns = min(ncols, len(item_boxes))
        base, extra = divmod(len(item_boxes), actual_columns)
        offset = 0
        for index in range(actual_columns):
            length = base + (index < extra)
            columns.append(
                VPacker(
                    pad=0.0,
                    sep=values[1] * fontsize,
                    align=column_align,
                    mode="fixed",
                    children=item_boxes[offset : offset + length],
                )
            )
            offset += length
    new_handle_box = HPacker(
        pad=0.0,
        sep=values[4] * fontsize,
        align=handle_box.align,
        mode="fixed",
        children=columns,
    )
    new_root = VPacker(
        pad=values[0] * fontsize,
        sep=values[1] * fontsize,
        align=root.align,
        mode="fixed",
        children=[title_box, new_handle_box],
    )
    new_root.set_offset(getattr(root, "_offset", (0.0, 0.0)))

    if hasattr(legend, "_ncols"):
        legend._ncols = ncols
    else:
        legend._ncol = ncols
    (
        legend.borderpad,
        legend.labelspacing,
        legend.handlelength,
        legend.handletextpad,
        legend.columnspacing,
    ) = values
    for drawing_area, _text_area, _children in entries:
        drawing_area.set_width(values[2] * fontsize)
    legend._legend_title_box = title_box
    legend._legend_handle_box = new_handle_box
    legend._legend_box = new_root
    try:
        figure = legend.get_figure(root=False)
    except TypeError:
        figure = legend.figure
    new_root.set_figure(figure)
    new_root.axes = legend.axes
    draggable = getattr(legend, "_draggable", None)
    if draggable is not None:
        draggable.offsetbox = new_root
    if baseline_missing:
        legend._pylustrator_original_layout_spec = source_layout
    legend.stale = True
    return True


def _apply_plan(plan: LegendLayoutPlan) -> None:
    current = _inspect_standard_structure(plan.target)
    if current.signature != plan.structure.signature:
        raise LegendLayoutError(
            "Legend layout changed after preflight; build a fresh reflow plan"
        )
    _matplotlib_only_reflow(
        plan.target,
        **{name: getattr(plan.destination, name) for name in LEGEND_LAYOUT_FIELDS},
    )


def ensure_legend_layout_baseline(legend: Legend) -> LegendLayoutSpec:
    """Capture source-authored layout once, before generated replay runs."""

    baseline = getattr(legend, "_pylustrator_original_layout_spec", None)
    if isinstance(baseline, (tuple, list)) and len(baseline) == len(
        LEGEND_LAYOUT_FIELDS
    ):
        baseline = _canonical_spec(LegendLayoutSpec(*baseline))
        legend._pylustrator_original_layout_spec = baseline
    if not isinstance(baseline, LegendLayoutSpec):
        baseline = LegendLayoutSpec.from_legend(legend)
        legend._pylustrator_original_layout_spec = baseline
    return baseline


def legend_layout_replay_bootstrap() -> str:
    """Return Matplotlib-only source that installs the generated replay hook."""

    function_source = dedent(inspect.getsource(_matplotlib_only_reflow)).rstrip()
    return (
        function_source
        + "\nfrom matplotlib.legend import Legend as _PylustratorLegend"
        + "\nsetattr(_PylustratorLegend, "
        + repr("_pylustrator_reflow_layout")
        + ", _matplotlib_only_reflow)"
    )


def plan_legend_layout(
    legend: Legend,
    updates: LegendLayoutSpec | Mapping[str, object],
    *,
    selected_artists: Iterable[Artist] = (),
) -> LegendLayoutPlan:
    return LegendLayoutPlan.preflight(
        legend, updates, selected_artists=selected_artists
    )


def reflow_legend_layout(
    legend: Legend,
    *,
    ncols: int | None = None,
    borderpad: float | None = None,
    labelspacing: float | None = None,
    handlelength: float | None = None,
    handletextpad: float | None = None,
    columnspacing: float | None = None,
) -> bool:
    """Reflow one standard Legend while preserving every persistent leaf."""

    updates = {
        name: value
        for name, value in {
            "ncols": ncols,
            "borderpad": borderpad,
            "labelspacing": labelspacing,
            "handlelength": handlelength,
            "handletextpad": handletextpad,
            "columnspacing": columnspacing,
        }.items()
        if value is not None
    }
    return plan_legend_layout(legend, updates).apply()


def _legend_reflow_method(self: Legend, **kwargs) -> bool:
    return reflow_legend_layout(self, **kwargs)


def install_legend_layout_api() -> None:
    """Install the replay hook used by generated source blocks."""

    current = getattr(Legend, "_pylustrator_reflow_layout", None)
    if current is not _legend_reflow_method:
        Legend._pylustrator_reflow_layout = _legend_reflow_method


install_legend_layout_api()


__all__ = [
    "LEGEND_LAYOUT_FIELDS",
    "LegendLayoutError",
    "LegendLayoutPlan",
    "LegendLayoutSpec",
    "LegendLayoutState",
    "capture_legend_layout_state",
    "ensure_legend_layout_baseline",
    "install_legend_layout_api",
    "legend_layout_replay_bootstrap",
    "plan_legend_layout",
    "reflow_legend_layout",
    "restore_legend_layout_state",
]
