"""Budgeted, renderer-faithful content ghosts for deferred transforms.

This module is deliberately a disposable visual layer.  It never plans or
commits geometry: adapters and ``TransformPlan`` remain the sole source of
truth for preview coordinates, document mutation, and Undo/Redo.  A cache miss
therefore means only that the existing analytic selection outline is shown.

Captures run from a Qt idle callback.  Pointer press only validates an already
published token and pointer motion only updates one ``QGraphicsItem`` affine
transform; neither path renders an Artist or measures the scene.
"""

from __future__ import annotations

from copy import copy
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import math
import weakref
from typing import Sequence

import numpy as np
from matplotlib.artist import Artist
from matplotlib.backends.backend_agg import RendererAgg
from matplotlib.collections import Collection, _CollectionWithSizes
from matplotlib.image import AxesImage
from matplotlib.legend import Legend
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AuxTransformBox, DrawingArea, OffsetBox, TextArea
from matplotlib.patches import Patch
from matplotlib.path import Path
from matplotlib.text import Annotation, Text
from matplotlib.transforms import Transform
from qtpy import QtCore, QtGui, QtWidgets

from .editor_model import EditorGroup


DEFAULT_MEMORY_BUDGET_BYTES = 32 * 1024 * 1024
DEFAULT_SOURCE_FINGERPRINT_BUDGET_BYTES = 512 * 1024
DEFAULT_MAX_ARTISTS = 64
DEFAULT_GHOST_OPACITY = 0.78
DEFAULT_IDLE_WORK_PIXEL_BUDGET = 2_500_000
_IDLE_WORK_PIXELS_PER_PAINT_LEAF = 50_000

_AUDITED_INSTANCE_METHODS = frozenset(
    {
        "draw",
        "findobj",
        "get_children",
        "get_window_extent",
        "get_transform",
        "get_clip_box",
        "get_clip_path",
        "get_visible",
        "get_alpha",
        "get_zorder",
        "get_snap",
        "get_clip_on",
        "get_antialiased",
        "get_rasterized",
        "get_agg_filter",
        "get_path_effects",
        "get_sketch_params",
        "get_data",
        "get_path",
        "get_fontproperties",
        "get_bbox_patch",
        "get_array",
        "get_paths",
        "get_offsets",
        "get_transforms",
        "get_facecolors",
        "get_edgecolors",
        "get_linewidths",
        "get_linestyles",
        "get_sizes",
        "get_bbox_to_anchor",
        "get_frame_on",
        "get_alignment",
        "get_ncols",
    }
)


class ContentPreviewUnavailable(RuntimeError):
    """A typed, behavior-preserving request to use the analytic fallback."""

    def __init__(self, reason: str):
        self.reason = str(reason)
        super().__init__(self.reason)


class _FingerprintBuilder:
    """Hash bounded Artist source state without retaining geometry copies."""

    def __init__(self, byte_limit: int):
        self.byte_limit = max(int(byte_limit), 0)
        self.source_bytes = 0
        self._digest = hashlib.blake2b(digest_size=16)

    def _write(self, label: str, payload: bytes | memoryview) -> None:
        byte_count = len(payload)
        if byte_count > self.byte_limit - self.source_bytes:
            raise ContentPreviewUnavailable(f"source-budget:{label}")
        self.source_bytes += byte_count
        self._digest.update(payload)

    def _write_text(self, label: str, value: str) -> None:
        # Encode in small chunks so a nested multi-megabyte string can never
        # allocate a same-sized temporary before the budget rejects it.
        for start in range(0, len(value), 4096):
            chunk = value[start : start + 4096].encode(
                "utf-8", "backslashreplace"
            )
            self._write(label, chunk)

    def _feed_value(
        self, label: str, value: object, seen: set[int], depth: int
    ) -> None:
        if depth > 32:
            raise ContentPreviewUnavailable(f"source-depth:{label}")
        if value is None:
            self._write(label, b"none")
            return
        if isinstance(value, bool):
            self._write(label, b"bool:1" if value else b"bool:0")
            return
        if isinstance(value, (int, np.integer)):
            integer = int(value)
            if integer.bit_length() // 3 > self.byte_limit - self.source_bytes:
                raise ContentPreviewUnavailable(f"source-budget:{label}")
            try:
                encoded = str(integer).encode("ascii")
            except (ValueError, OverflowError) as error:
                raise ContentPreviewUnavailable(f"source-integer:{label}") from error
            self._write(label, b"int:" + encoded)
            return
        if isinstance(value, (float, complex, np.floating, np.complexfloating)):
            self._write(label, f"number:{value!r}".encode("ascii"))
            return
        if isinstance(value, str):
            self._write(label, b"str:")
            self._write_text(label, value)
            return
        if isinstance(value, (bytes, bytearray, memoryview)):
            view = memoryview(value).cast("B")
            self._write(label, b"bytes:")
            self._write(label, view)
            return
        if isinstance(value, Enum):
            self._write_text(
                label,
                f"enum:{type(value).__module__}.{type(value).__qualname__}:",
            )
            self._feed_value(label, value.value, seen, depth + 1)
            return
        if isinstance(value, np.generic):
            self._write_text(label, f"numpy:{value.dtype.str}:")
            self._write(label, value.tobytes())
            return
        if isinstance(value, (np.ndarray, np.ma.MaskedArray)):
            self.array(f"{label}.array", value)
            return
        if isinstance(value, Path):
            if type(value).__module__ != "matplotlib.path":
                raise ContentPreviewUnavailable(f"custom-path:{label}")
            self._write(label, b"path:")
            self.array(f"{label}.vertices", value.vertices)
            self.array(f"{label}.codes", value.codes)
            return
        if isinstance(value, Transform):
            if not type(value).__module__.startswith("matplotlib."):
                raise ContentPreviewUnavailable(f"custom-transform:{label}")
            self._write_text(
                label,
                f"transform:{type(value).__module__}.{type(value).__qualname__}:",
            )
            self.array(f"{label}.matrix", value.get_matrix())
            return
        if isinstance(value, Artist):
            # An Artist coordinate descriptor depends on another open visual
            # object graph; identity alone cannot make the token exact.
            raise ContentPreviewUnavailable(f"artist-coordinate:{label}")
        if isinstance(value, slice):
            self._write(label, b"slice:")
            for part in (value.start, value.stop, value.step):
                self._feed_value(label, part, seen, depth + 1)
            return
        if isinstance(value, range):
            self._write(label, b"range:")
            for part in (value.start, value.stop, value.step):
                self._feed_value(label, part, seen, depth + 1)
            return
        if isinstance(value, (list, tuple, dict)):
            identity = id(value)
            if identity in seen:
                raise ContentPreviewUnavailable(f"source-cycle:{label}")
            seen.add(identity)
            try:
                self._write(
                    label,
                    ("dict:" if isinstance(value, dict) else "sequence:").encode(
                        "ascii"
                    ),
                )
                self._feed_value(label, len(value), seen, depth + 1)
                items = value.items() if isinstance(value, dict) else enumerate(value)
                for key, item in items:
                    self._feed_value(label, key, seen, depth + 1)
                    self._feed_value(label, item, seen, depth + 1)
            finally:
                seen.remove(identity)
            return
        if isinstance(value, (set, frozenset)):
            raise ContentPreviewUnavailable(f"unordered-source:{label}")
        function = getattr(value, "__func__", None)
        owner = getattr(value, "__self__", None)
        module = getattr(function, "__module__", "")
        if function is not None and owner is not None and module.startswith(
            "matplotlib."
        ):
            self._write_text(
                label,
                f"method:{module}.{getattr(function, '__qualname__', '')}:",
            )
            self._feed_value(label, id(owner), seen, depth + 1)
            return
        # Never invoke an arbitrary repr: it can allocate without bound, run
        # user code, or conceal mutable visual state.
        raise ContentPreviewUnavailable(f"opaque-source:{label}")

    def scalar(self, label: str, value: object) -> None:
        label_bytes = label.encode("utf-8", "backslashreplace")
        self._write(label, label_bytes)
        self._write(label, b"\0")
        self._feed_value(label, value, set(), 0)
        self._write(label, b"\0")

    def _preflight_sequence(self, values, label: str) -> None:
        if isinstance(values, (np.ndarray, np.ma.MaskedArray)):
            return
        try:
            length = len(values)
        except TypeError:
            return
        # Creating an ndarray from a large Python list would itself violate the
        # bounded-source contract.  Eight bytes per scalar is a conservative
        # early gate for the numeric geometry accepted below.
        estimate = int(length) * 8
        if estimate > self.byte_limit - self.source_bytes:
            raise ContentPreviewUnavailable(f"source-budget:{label}")

    def array(self, label: str, values) -> None:
        if values is None:
            self.scalar(label, None)
            return
        if np.ma.isMaskedArray(values):
            masked = np.ma.asarray(values)
            self.array(f"{label}.data", masked.data)
            self.array(f"{label}.mask", np.ma.getmaskarray(masked))
            self.scalar(f"{label}.fill", masked.fill_value)
            return
        self._preflight_sequence(values, label)
        try:
            array = np.asarray(values)
        except (TypeError, ValueError, MemoryError) as error:
            raise ContentPreviewUnavailable(f"source-array:{label}") from error
        if array.dtype.hasobject:
            raise ContentPreviewUnavailable(f"object-source:{label}")
        byte_count = int(array.nbytes)
        if byte_count > self.byte_limit - self.source_bytes:
            raise ContentPreviewUnavailable(f"source-budget:{label}")
        self.source_bytes += byte_count
        self.scalar(f"{label}.dtype", array.dtype.str)
        self.scalar(f"{label}.shape", tuple(int(value) for value in array.shape))
        self._digest.update(label.encode("utf-8", "backslashreplace"))
        self._digest.update(b"\0")
        if array.flags.c_contiguous:
            try:
                self._digest.update(memoryview(array).cast("B"))
            except (TypeError, ValueError):
                self._digest.update(array.tobytes(order="C"))
        else:
            # Non-contiguous sources are accepted only below the explicit
            # source budget, so this bounded temporary cannot scale with a
            # 100k-point fallback Artist.
            self._digest.update(array.tobytes(order="C"))
        self._digest.update(b"\0")

    def finish(self) -> bytes:
        return self._digest.digest()


def _call(artist: Artist, name: str, default=None):
    getter = getattr(artist, name, None)
    if not callable(getter):
        return default
    return getter()


def _require_no_instance_method_overrides(artist: Artist) -> None:
    state = getattr(artist, "__dict__", {})
    if any(
        name in _AUDITED_INSTANCE_METHODS or name.startswith("get_")
        for name in state
    ):
        raise ContentPreviewUnavailable("instance-method-override")


def _feed_common_source(builder: _FingerprintBuilder, artist: Artist) -> None:
    _require_no_instance_method_overrides(artist)
    builder.scalar("type", (type(artist).__module__, type(artist).__qualname__))
    for name in (
        "get_visible",
        "get_alpha",
        "get_zorder",
        "get_snap",
        "get_clip_on",
        "get_antialiased",
        "get_rasterized",
    ):
        try:
            builder.scalar(name, _call(artist, name))
        except (AttributeError, TypeError, ValueError, RuntimeError) as error:
            raise ContentPreviewUnavailable(f"source-property:{name}") from error

    for name, reason in (
        ("get_agg_filter", "agg-filter"),
        ("get_path_effects", "path-effects"),
        ("get_sketch_params", "sketch"),
    ):
        try:
            value = _call(artist, name)
        except (AttributeError, TypeError, ValueError, RuntimeError) as error:
            raise ContentPreviewUnavailable(reason) from error
        if value not in (None, (), []):
            raise ContentPreviewUnavailable(reason)

    try:
        transform = getattr(artist, "_transform", None)
        if transform is None:
            # Artist.get_transform() lazily installs IdentityTransform.  The
            # semantic value is known, so fingerprint it without mutating the
            # live Artist merely to observe that default.
            builder.scalar("transform", "implicit-identity")
        else:
            if not type(transform).__module__.startswith("matplotlib."):
                raise ContentPreviewUnavailable("custom-transform")
            builder.array("transform", transform.get_matrix())
    except (AttributeError, TypeError, ValueError, RuntimeError) as error:
        raise ContentPreviewUnavailable("transform") from error

    try:
        clip_box = artist.get_clip_box()
        if clip_box is not None:
            builder.array("clip-box", clip_box.get_points())
        clip_path = artist.get_clip_path()
        if clip_path is not None:
            path, affine = clip_path.get_transformed_path_and_affine()
            if type(path).__module__ != "matplotlib.path":
                raise ContentPreviewUnavailable("custom-clip-path")
            if not type(affine).__module__.startswith("matplotlib."):
                raise ContentPreviewUnavailable("custom-clip-transform")
            builder.array("clip-path.vertices", path.vertices)
            builder.array("clip-path.codes", path.codes)
            builder.array("clip-path.affine", affine.get_matrix())
    except (AttributeError, TypeError, ValueError, RuntimeError) as error:
        raise ContentPreviewUnavailable("clip") from error


def _feed_line_source(builder: _FingerprintBuilder, artist: Line2D) -> None:
    xdata, ydata = artist.get_data(orig=True)
    builder.array("line.x", xdata)
    builder.array("line.y", ydata)
    for name in (
        "get_color",
        "get_linestyle",
        "get_linewidth",
        "get_drawstyle",
        "get_marker",
        "get_markersize",
        "get_markeredgewidth",
        "get_markeredgecolor",
        "get_markerfacecolor",
        "get_markerfacecoloralt",
        "get_fillstyle",
        "get_markevery",
        "get_solid_capstyle",
        "get_solid_joinstyle",
        "get_dash_capstyle",
        "get_dash_joinstyle",
        "get_gapcolor",
    ):
        try:
            builder.scalar(name, _call(artist, name))
        except (AttributeError, TypeError, ValueError, RuntimeError) as error:
            raise ContentPreviewUnavailable(f"line-property:{name}") from error
    builder.scalar("line.dash-pattern", getattr(artist, "_dash_pattern", None))


def _feed_patch_source(builder: _FingerprintBuilder, artist: Patch) -> None:
    try:
        path = artist.get_path()
        if type(path).__module__ != "matplotlib.path":
            raise ContentPreviewUnavailable("custom-patch-path")
        builder.array("patch.vertices", path.vertices)
        builder.array("patch.codes", path.codes)
    except (AttributeError, TypeError, ValueError, RuntimeError) as error:
        raise ContentPreviewUnavailable("patch-path") from error
    for name in (
        "get_facecolor",
        "get_edgecolor",
        "get_linewidth",
        "get_linestyle",
        "get_fill",
        "get_hatch",
        "get_hatch_linewidth",
        "get_capstyle",
        "get_joinstyle",
    ):
        try:
            builder.scalar(name, _call(artist, name))
        except (AttributeError, TypeError, ValueError, RuntimeError) as error:
            raise ContentPreviewUnavailable(f"patch-property:{name}") from error


def _feed_text_source(builder: _FingerprintBuilder, artist: Text) -> None:
    for name in (
        "get_text",
        "get_position",
        "get_color",
        "get_rotation",
        "get_rotation_mode",
        "get_horizontalalignment",
        "get_verticalalignment",
        "get_linespacing",
        "get_multialignment",
        "get_wrap",
        "get_usetex",
        "get_parse_math",
        "get_transform_rotates_text",
    ):
        try:
            builder.scalar(name, _call(artist, name))
        except (AttributeError, TypeError, ValueError, RuntimeError) as error:
            raise ContentPreviewUnavailable(f"text-property:{name}") from error
    try:
        font = artist.get_fontproperties()
        if not type(font).__module__.startswith("matplotlib."):
            raise ContentPreviewUnavailable("custom-font-properties")
        _require_no_instance_method_overrides(font)
        for name in (
            "get_family",
            "get_style",
            "get_variant",
            "get_weight",
            "get_stretch",
            "get_size_in_points",
            "get_file",
            "get_math_fontfamily",
        ):
            value = getattr(font, name, None)
            builder.scalar(f"font.{name}", value() if callable(value) else None)
    except (AttributeError, TypeError, ValueError, RuntimeError) as error:
        raise ContentPreviewUnavailable("font") from error

    bbox_patch = artist.get_bbox_patch()
    if bbox_patch is not None:
        _feed_common_source(builder, bbox_patch)
        _feed_patch_source(builder, bbox_patch)
    if isinstance(artist, Annotation):
        builder.scalar("annotation.xy", artist.xy)
        builder.scalar("annotation.xycoords", artist.xycoords)
        builder.scalar("annotation.anncoords", artist.anncoords)
        builder.scalar("annotation.clip", artist.get_annotation_clip())
        builder.scalar(
            "annotation.arrow-relpos", getattr(artist, "_arrow_relpos", None)
        )
        if artist.arrow_patch is not None:
            _feed_common_source(builder, artist.arrow_patch)
            _feed_patch_source(builder, artist.arrow_patch)


def _feed_collection_source(
    builder: _FingerprintBuilder, artist: Collection
) -> None:
    try:
        if artist.get_array() is not None:
            # Mutable Norm/Colormap object graphs are intentionally outside
            # the v1 exact-safe token contract.  Fixed-color collections are
            # still renderer-faithful and cover ordinary legend handles.
            raise ContentPreviewUnavailable("scalar-mappable-unsupported")
        for index, path in enumerate(artist.get_paths()):
            if type(path).__module__ != "matplotlib.path":
                raise ContentPreviewUnavailable("custom-collection-path")
            builder.array(f"collection.path.{index}.vertices", path.vertices)
            builder.array(f"collection.path.{index}.codes", path.codes)
        for name in (
            "get_offsets",
            "get_transforms",
            "get_facecolors",
            "get_edgecolors",
            "get_linewidths",
            "get_sizes",
        ):
            value = _call(artist, name)
            if value is not None:
                builder.array(f"collection.{name}", value)
        for index, style in enumerate(artist.get_linestyles()):
            offset, dashes = style
            builder.scalar(f"collection.linestyle.{index}.offset", offset)
            builder.array(f"collection.linestyle.{index}.dashes", dashes)
        builder.scalar("collection.capstyle", _call(artist, "get_capstyle"))
        builder.scalar("collection.joinstyle", _call(artist, "get_joinstyle"))
    except ContentPreviewUnavailable:
        raise
    except (AttributeError, TypeError, ValueError, RuntimeError) as error:
        raise ContentPreviewUnavailable("collection-source") from error


def _feed_image_source(builder: _FingerprintBuilder, artist: AxesImage) -> None:
    try:
        builder.array("image.array", artist.get_array())
        for name in (
            "get_clim",
            "get_interpolation",
            "get_interpolation_stage",
            "get_resample",
            "get_filternorm",
            "get_filterrad",
            "get_extent",
        ):
            builder.scalar(name, _call(artist, name))
        builder.scalar("image.origin", artist.origin)
        cmap = artist.get_cmap()
        builder.scalar("image.cmap", (id(cmap), getattr(cmap, "name", None)))
        norm = artist.norm
        builder.scalar(
            "image.norm",
            (
                id(norm),
                type(norm).__module__,
                type(norm).__qualname__,
                getattr(norm, "vmin", None),
                getattr(norm, "vmax", None),
                getattr(norm, "clip", None),
            ),
        )
    except ContentPreviewUnavailable:
        raise
    except (AttributeError, TypeError, ValueError, RuntimeError) as error:
        raise ContentPreviewUnavailable("image-source") from error


def _class_draw_contract(artist: Artist):
    draw = type(artist).draw
    bound = getattr(artist, "draw", None)
    # An instance attribute can shadow an otherwise trusted class method.  It
    # must fail open before invocation because its side effects are unknowable.
    if getattr(bound, "__func__", None) is not draw:
        return None
    return draw


def _supported_draw_contract(artist: Artist) -> bool:
    if not type(artist).__module__.startswith("matplotlib."):
        return False
    draw = _class_draw_contract(artist)
    if draw is None:
        return False
    if isinstance(artist, Legend):
        return draw is Legend.draw
    if isinstance(artist, Annotation):
        return draw is Annotation.draw
    if isinstance(artist, Text):
        return draw is Text.draw
    if isinstance(artist, Line2D):
        return draw is Line2D.draw
    if isinstance(artist, Patch):
        return draw is Patch.draw
    if isinstance(artist, Collection):
        return draw in {Collection.draw, _CollectionWithSizes.draw}
    # AxesImage always delegates paint to mutable Norm/Colormap state whose
    # open object graph cannot be completely fingerprinted in constant time.
    if isinstance(artist, AxesImage):
        return False
    return False


def _feed_leaf_source(builder: _FingerprintBuilder, artist: Artist) -> None:
    if not _supported_draw_contract(artist):
        raise ContentPreviewUnavailable("unsupported-artist")
    _feed_common_source(builder, artist)
    if not bool(artist.get_visible()):
        return
    if isinstance(artist, Annotation):
        _feed_text_source(builder, artist)
    elif isinstance(artist, Text):
        _feed_text_source(builder, artist)
    elif isinstance(artist, Line2D):
        _feed_line_source(builder, artist)
    elif isinstance(artist, Patch):
        _feed_patch_source(builder, artist)
    elif isinstance(artist, Collection):
        _feed_collection_source(builder, artist)
    elif isinstance(artist, AxesImage):
        _feed_image_source(builder, artist)


def _feed_legend_source(
    builder: _FingerprintBuilder, legend: Legend, seen: set[int]
) -> None:
    _feed_common_source(builder, legend)
    for name in (
        "get_frame_on",
        "get_alignment",
        "get_ncols",
        "get_draggable",
    ):
        try:
            builder.scalar(f"legend.{name}", _call(legend, name))
        except (AttributeError, TypeError, ValueError, RuntimeError) as error:
            raise ContentPreviewUnavailable(f"legend-property:{name}") from error
    for name in (
        "_loc",
        "_mode",
        "_ncols",
        "_fontsize",
        "borderpad",
        "labelspacing",
        "handlelength",
        "handleheight",
        "handletextpad",
        "borderaxespad",
        "columnspacing",
        "markerscale",
        "numpoints",
        "scatterpoints",
    ):
        builder.scalar(f"legend.{name}", getattr(legend, name, None))
    try:
        bbox = legend.get_bbox_to_anchor()
        if bbox is not None:
            builder.array("legend.bbox", bbox.get_points())
    except (AttributeError, TypeError, ValueError, RuntimeError) as error:
        raise ContentPreviewUnavailable("legend-bbox") from error

    try:
        descendants = legend.findobj()
    except (AttributeError, TypeError, ValueError, RuntimeError) as error:
        raise ContentPreviewUnavailable("legend-children") from error
    for descendant in descendants:
        if descendant is legend or id(descendant) in seen:
            continue
        if isinstance(descendant, OffsetBox):
            # These are layout-only composite nodes; their visible source is
            # represented by the standard leaf Artists and their transforms.
            draw = _class_draw_contract(descendant)
            if draw not in {
                OffsetBox.draw,
                DrawingArea.draw,
                TextArea.draw,
                AuxTransformBox.draw,
            }:
                raise ContentPreviewUnavailable("unsupported-legend-layout")
            _require_no_instance_method_overrides(descendant)
            seen.add(id(descendant))
            builder.scalar("legend.layout-id", id(descendant))
            for name in (
                "_offset",
                "pad",
                "sep",
                "width",
                "height",
                "xdescent",
                "ydescent",
                "align",
                "mode",
            ):
                builder.scalar(
                    f"legend.layout.{name}", getattr(descendant, name, None)
                )
            builder.scalar(
                "legend.layout.children",
                tuple(id(child) for child in descendant.get_children()),
            )
            continue
        seen.add(id(descendant))
        builder.scalar("legend.child-id", id(descendant))
        _feed_leaf_source(builder, descendant)


def _feed_artist_source(
    builder: _FingerprintBuilder, artist: Artist, seen: set[int]
) -> None:
    if id(artist) in seen:
        return
    seen.add(id(artist))
    builder.scalar("artist-id", id(artist))
    if isinstance(artist, EditorGroup):
        _feed_common_source(builder, artist)
        builder.scalar("group-id", artist.group_id)
        builder.scalar("group-name", artist.name)
        builder.scalar("group-owner", id(artist.owner))
        builder.scalar("group-members", tuple(id(item) for item in artist.members))
        for member in artist.members:
            _feed_artist_source(builder, member, seen)
        return
    if isinstance(artist, Legend):
        if not _supported_draw_contract(artist):
            raise ContentPreviewUnavailable("unsupported-artist")
        _feed_legend_source(builder, artist, seen)
        return
    _feed_leaf_source(builder, artist)


def artist_source_fingerprint(
    artist: Artist, *, byte_limit: int = DEFAULT_SOURCE_FINGERPRINT_BUDGET_BYTES
) -> tuple[bytes, int]:
    """Return a compact bounded visual-source fingerprint for one selection.

    Standard leaves, Legends, and logical editor groups are supported.
    Unknown/custom drawing contracts fail open to the analytic preview.  Large
    numeric sources are rejected before hashing, which keeps a 100k-point
    Artist out of both the cache and the pointer validation path.
    """

    if not bool(artist.get_visible()):
        raise ContentPreviewUnavailable("invisible-artist")
    builder = _FingerprintBuilder(byte_limit)
    _feed_artist_source(builder, artist, set())
    return builder.finish(), builder.source_bytes


def _expanded_draw_artists(artists: Sequence[Artist]) -> tuple[Artist, ...]:
    result: list[Artist] = []
    seen: set[int] = set()

    def add(artist: Artist) -> None:
        if id(artist) in seen:
            return
        seen.add(id(artist))
        if isinstance(artist, EditorGroup):
            for member in artist.members:
                add(member)
            return
        if bool(artist.get_visible()):
            result.append(artist)

    for artist in artists:
        add(artist)
    return tuple(result)


def _legend_paint_artists(legend: Legend) -> tuple[Artist, ...]:
    """Flatten one already-laid-out standard Legend in its paint order.

    ``Legend.draw`` updates a large cyclic OffsetBox graph before recursively
    drawing a small number of primitive leaves.  The live Figure draw has
    already finalized that layout.  Replaying the visible frame and leaves is
    therefore renderer-equivalent for the v1 standard-Legend contract while
    avoiding a deepcopy of the Figure/Axes/OffsetBox ownership graph.
    """

    if bool(getattr(legend, "shadow", False)):
        raise ContentPreviewUnavailable("legend-shadow-unsupported")

    result: list[Artist] = []
    seen: set[int] = set()

    def add(artist: Artist) -> None:
        if id(artist) in seen:
            return
        seen.add(id(artist))
        if bool(artist.get_visible()):
            result.append(artist)

    frame = legend.get_frame()
    if legend.get_frame_on() and frame is not None:
        add(frame)

    def walk(node) -> None:
        if isinstance(node, OffsetBox):
            if isinstance(node, DrawingArea) and bool(
                getattr(node, "_clip_children", False)
            ):
                raise ContentPreviewUnavailable("legend-layout-clip")
            try:
                children = node.get_children()
            except (AttributeError, TypeError, ValueError, RuntimeError) as error:
                raise ContentPreviewUnavailable("legend-children") from error
            for child in children:
                walk(child)
            return
        if not isinstance(node, Artist) or not _supported_draw_contract(node):
            raise ContentPreviewUnavailable("unsupported-legend-leaf")
        add(node)

    legend_box = getattr(legend, "_legend_box", None)
    if not isinstance(legend_box, OffsetBox):
        raise ContentPreviewUnavailable("unsupported-legend-layout")
    walk(legend_box)
    return tuple(result)


def _require_bounded_composite_complexity(
    artists: Sequence[Artist], *, max_paint_artists: int
) -> tuple[int, int, tuple[Artist, ...]]:
    """Bound actual composite leaves before fingerprinting or raster allocation."""

    max_paint_artists = max(int(max_paint_artists), 0)
    max_nodes = max(max_paint_artists * 8, max_paint_artists)
    stack = list(reversed(artists))
    seen: set[int] = set()
    paint_count = 0
    node_count = 0
    paint_artists: list[Artist] = []
    while stack:
        artist = stack.pop()
        if id(artist) in seen:
            continue
        seen.add(id(artist))
        node_count += 1
        if node_count > max_nodes:
            raise ContentPreviewUnavailable("composite-node-budget")

        if isinstance(artist, EditorGroup):
            stack.extend(reversed(artist.members))
            continue
        if isinstance(artist, (Legend, OffsetBox)):
            if not (
                type(artist).__module__.startswith("matplotlib.")
                and _class_draw_contract(artist) is not None
            ):
                raise ContentPreviewUnavailable("unsupported-composite")
            _require_no_instance_method_overrides(artist)
            try:
                children = artist.get_children()
            except (AttributeError, TypeError, ValueError, RuntimeError) as error:
                raise ContentPreviewUnavailable("composite-children") from error
            stack.extend(reversed(children))
            continue

        paint_count += 1
        if paint_count > max_paint_artists:
            raise ContentPreviewUnavailable("artist-count-budget")
        paint_artists.append(artist)
        if isinstance(artist, Text):
            bbox_patch = artist.get_bbox_patch()
            if bbox_patch is not None:
                stack.append(bbox_patch)
        if isinstance(artist, Annotation) and artist.arrow_patch is not None:
            stack.append(artist.arrow_patch)
    return paint_count, node_count, tuple(paint_artists)


def _require_unclipped_components(artists: Sequence[Artist]) -> None:
    """Reject fixed scene clips; v1 ghosts transform already-clipped pixels."""

    for artist in artists:
        try:
            if isinstance(artist, Annotation):
                annotation_clip = artist.get_annotation_clip()
                if annotation_clip is True or (
                    annotation_clip is None and artist.xycoords == "data"
                ):
                    raise ContentPreviewUnavailable("annotation-clip")
            if bool(artist.get_clip_on()) and (
                artist.get_clip_box() is not None
                or artist.get_clip_path() is not None
            ):
                raise ContentPreviewUnavailable("active-clip")
        except ContentPreviewUnavailable:
            raise
        except (AttributeError, TypeError, ValueError, RuntimeError) as error:
            raise ContentPreviewUnavailable("clip-state") from error


def _require_clone_inside_canvas(
    artist: Artist, renderer, *, width: int, height: int
) -> None:
    """Prove the source has no paint hidden beyond the renderer boundary."""

    if isinstance(artist, Collection):
        raise ContentPreviewUnavailable("collection-envelope-unsupported")
    if isinstance(artist, Legend) and bool(getattr(artist, "shadow", False)):
        raise ContentPreviewUnavailable("legend-shadow-unsupported")
    try:
        bounds = np.asarray(artist.get_window_extent(renderer).extents, dtype=float)
    except (AttributeError, TypeError, ValueError, RuntimeError) as error:
        raise ContentPreviewUnavailable("source-envelope") from error
    if bounds.shape != (4,) or not np.all(np.isfinite(bounds)):
        raise ContentPreviewUnavailable("source-envelope")
    x0, y0, x1, y1 = (float(value) for value in bounds)
    if x0 <= 0.0 or y0 <= 0.0 or x1 >= float(width) or y1 >= float(height):
        raise ContentPreviewUnavailable("canvas-edge-source")


def _paint_order(manager, artists: Sequence[Artist]) -> tuple[Artist, ...]:
    key = getattr(manager, "_paint_order_key", None)
    if callable(key):
        try:
            return tuple(
                artist
                for _index, artist in sorted(
                    enumerate(artists),
                    key=lambda pair: key(pair[1], fallback=pair[0]),
                )
            )
        except (AttributeError, TypeError, ValueError, RuntimeError):
            pass
    return tuple(
        artist
        for _index, artist in sorted(
            enumerate(artists),
            key=lambda pair: (float(pair[1].get_zorder()), pair[0]),
        )
    )


def selection_source_fingerprints(
    artists: Sequence[Artist], *, byte_limit: int
) -> tuple[tuple[bytes, ...], int]:
    remaining = max(int(byte_limit), 0)
    fingerprints: list[bytes] = []
    consumed = 0
    for artist in artists:
        fingerprint, source_bytes = artist_source_fingerprint(
            artist, byte_limit=remaining
        )
        fingerprints.append(fingerprint)
        consumed += int(source_bytes)
        remaining -= int(source_bytes)
    return tuple(fingerprints), consumed


def _renderer_shape(renderer) -> tuple[int, int, float]:
    try:
        width = int(math.ceil(float(renderer.width)))
        height = int(math.ceil(float(renderer.height)))
        dpi = float(renderer.dpi)
    except (AttributeError, TypeError, ValueError, OverflowError) as error:
        raise ContentPreviewUnavailable("renderer-shape") from error
    if width <= 0 or height <= 0 or not np.isfinite(dpi) or dpi <= 0:
        raise ContentPreviewUnavailable("renderer-shape")
    return width, height, dpi


@dataclass(frozen=True)
class ContentPreviewToken:
    revision: int
    renderer_id: int
    renderer: object = field(repr=False, compare=False)
    renderer_shape: tuple[int, int, float]
    artist_ids: tuple[int, ...]
    source_fingerprints: tuple[bytes, ...]
    source_bytes: int

    @classmethod
    def capture(
        cls,
        manager,
        artists: Sequence[Artist],
        renderer,
        *,
        source_byte_limit: int,
    ) -> "ContentPreviewToken":
        fingerprints, source_bytes = selection_source_fingerprints(
            artists, byte_limit=source_byte_limit
        )
        return cls(
            revision=int(getattr(manager, "_interaction_revision", 0)),
            renderer_id=id(renderer),
            renderer=renderer,
            renderer_shape=_renderer_shape(renderer),
            artist_ids=tuple(id(artist) for artist in artists),
            source_fingerprints=fingerprints,
            source_bytes=int(source_bytes),
        )

    def is_current(
        self,
        manager,
        artists: Sequence[Artist],
        renderer,
        *,
        source_byte_limit: int,
    ) -> bool:
        if int(getattr(manager, "_interaction_revision", 0)) != self.revision:
            return False
        if renderer is not self.renderer or id(renderer) != self.renderer_id:
            return False
        if tuple(id(artist) for artist in artists) != self.artist_ids:
            return False
        try:
            current = type(self).capture(
                manager,
                artists,
                renderer,
                source_byte_limit=source_byte_limit,
            )
        except ContentPreviewUnavailable:
            return False
        return (
            current.renderer_shape == self.renderer_shape
            and current.source_fingerprints == self.source_fingerprints
        )


@dataclass
class ContentPreviewEntry:
    token: ContentPreviewToken
    root: QtWidgets.QGraphicsRectItem
    item: QtWidgets.QGraphicsPixmapItem
    pixmap: QtGui.QPixmap
    display_bounds: tuple[float, float, float, float]
    canvas_bounds: tuple[float, float, float, float]
    retained_bytes: int
    peak_bytes: int

    def remove(self) -> None:
        try:
            scene = self.root.scene()
            self.root.setVisible(False)
            self.root.setParentItem(None)
            if scene is not None:
                scene.removeItem(self.root)
        except RuntimeError:
            pass


def _qimage_rgba8888_format():
    direct = getattr(QtGui.QImage, "Format_RGBA8888", None)
    if direct is not None:
        return direct
    return QtGui.QImage.Format.Format_RGBA8888


def _qt_application_ready() -> bool:
    app = QtWidgets.QApplication.instance()
    return app is not None and QtCore.QThread.currentThread() is app.thread()


def _selected_artists(selection) -> tuple[Artist, ...]:
    return tuple(target.target for target in getattr(selection, "targets", ()))


def _disposable_draw_clone(artist: Artist) -> Artist:
    """Create an audited shallow visual clone for one primitive draw.

    Standard Matplotlib primitive draw methods mutate only top-level derived
    caches.  A shallow Artist copy isolates those assignments without copying
    the cyclic Figure/Axes/Transform graph.  The few nested mutable paint
    helpers which draw mutates (Text bbox and Annotation arrow patches) are
    copied explicitly.  Unknown classes never reach this function because the
    source/draw contract rejects them first.
    """

    if isinstance(artist, Legend) or not _supported_draw_contract(artist):
        raise ContentPreviewUnavailable("unsupported-clone")
    try:
        clone = copy(artist)
    except Exception as error:
        raise ContentPreviewUnavailable("clone-failed") from error
    if clone is artist:
        raise ContentPreviewUnavailable("clone-failed")

    clone.stale_callback = None
    clone._remove_method = None

    if isinstance(clone, Text):
        bbox_patch = artist.get_bbox_patch()
        if bbox_patch is not None:
            try:
                clone._bbox_patch = copy(bbox_patch)
            except Exception as error:
                raise ContentPreviewUnavailable("clone-failed") from error
            clone._bbox_patch.stale_callback = None
            clone._bbox_patch._remove_method = None

    if isinstance(clone, Annotation) and artist.arrow_patch is not None:
        try:
            clone.arrow_patch = copy(artist.arrow_patch)
        except Exception as error:
            raise ContentPreviewUnavailable("clone-failed") from error
        clone.arrow_patch.stale_callback = None
        clone.arrow_patch._remove_method = None
        positions = getattr(artist.arrow_patch, "_posA_posB", None)
        if positions is not None:
            clone.arrow_patch._posA_posB = [
                np.asarray(position, dtype=float).copy()
                for position in positions
            ]

    if isinstance(clone, Line2D):
        # TransformedPath lazily owns mutable cached paths.  Rebuild it on the
        # clone while sharing only the immutable/raw path and coordinate data.
        clone._transformed_path = None

    return clone


def _matrix_to_qtransform(matrix: np.ndarray) -> QtGui.QTransform:
    matrix = np.asarray(matrix, dtype=float)
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        raise ContentPreviewUnavailable("preview-transform")
    if not np.allclose(matrix[2], [0.0, 0.0, 1.0], atol=1e-12, rtol=0.0):
        raise ContentPreviewUnavailable("preview-transform")
    return QtGui.QTransform(
        float(matrix[0, 0]),
        float(matrix[1, 0]),
        float(matrix[0, 1]),
        float(matrix[1, 1]),
        float(matrix[0, 2]),
        float(matrix[1, 2]),
    )


class ContentPreviewCache:
    """One-selection cache with an explicit retained and peak memory budget."""

    def __init__(
        self,
        *,
        memory_budget_bytes: int = DEFAULT_MEMORY_BUDGET_BYTES,
        source_fingerprint_budget_bytes: int = (
            DEFAULT_SOURCE_FINGERPRINT_BUDGET_BYTES
        ),
        max_artists: int = DEFAULT_MAX_ARTISTS,
        opacity: float = DEFAULT_GHOST_OPACITY,
    ) -> None:
        self.memory_budget_bytes = max(int(memory_budget_bytes), 0)
        self.source_fingerprint_budget_bytes = max(
            int(source_fingerprint_budget_bytes), 0
        )
        self.max_artists = max(int(max_artists), 0)
        self.opacity = float(np.clip(opacity, 0.0, 1.0))
        self.entry: ContentPreviewEntry | None = None
        self.active = False
        self.closed = False
        self.generation = 0
        self.capture_count = 0
        self.activation_count = 0
        self.motion_update_count = 0
        self.last_fallback_reason: str | None = None
        self._capture_renderer: RendererAgg | None = None
        self._capture_renderer_spec: tuple[int, int, float] | None = None

    def configure_from_selection(self, selection) -> None:
        budget = max(
            int(
                getattr(
                    selection,
                    "content_preview_memory_budget_bytes",
                    DEFAULT_MEMORY_BUDGET_BYTES,
                )
            ),
            0,
        )
        source_budget = max(
            int(
                getattr(
                    selection,
                    "content_preview_source_budget_bytes",
                    DEFAULT_SOURCE_FINGERPRINT_BUDGET_BYTES,
                )
            ),
            0,
        )
        max_artists = max(
            int(
                getattr(
                    selection,
                    "content_preview_max_artists",
                    DEFAULT_MAX_ARTISTS,
                )
            ),
            0,
        )
        changed = (
            budget != self.memory_budget_bytes
            or source_budget != self.source_fingerprint_budget_bytes
            or max_artists != self.max_artists
        )
        self.memory_budget_bytes = budget
        self.source_fingerprint_budget_bytes = source_budget
        self.max_artists = max_artists
        if changed:
            self.invalidate("budget-changed")
            self._release_capture_renderer()

    def invalidate(self, reason: str = "invalidated") -> None:
        self.generation += 1
        self.active = False
        if self.entry is not None:
            self.entry.remove()
            self.entry = None
        self.last_fallback_reason = str(reason)

    def close(self) -> None:
        if self.closed:
            return
        self.invalidate("closed")
        self._release_capture_renderer()
        self.closed = True

    def _release_capture_renderer(self) -> None:
        self._capture_renderer = None
        self._capture_renderer_spec = None

    def _scratch_renderer(
        self, width: int, height: int, dpi: float
    ) -> RendererAgg:
        spec = (int(width), int(height), float(dpi))
        renderer = self._capture_renderer
        if renderer is not None and self._capture_renderer_spec == spec:
            renderer.clear()
            return renderer
        # Drop a differently sized buffer before allocating its replacement so
        # the cache never transiently retains two full-canvas rasters.
        self._release_capture_renderer()
        try:
            renderer = RendererAgg(*spec)
        except (MemoryError, OverflowError, TypeError, ValueError) as error:
            raise ContentPreviewUnavailable("renderer-allocation") from error
        renderer.clear()
        self._capture_renderer = renderer
        self._capture_renderer_spec = spec
        return renderer

    def _capture_entry(self, manager, selection) -> ContentPreviewEntry:
        if self.closed:
            raise ContentPreviewUnavailable("closed")
        if not bool(getattr(selection, "content_preview_enabled", True)):
            raise ContentPreviewUnavailable("disabled")
        if not bool(getattr(selection, "defer_artist_updates", False)):
            raise ContentPreviewUnavailable("analytic-preview-disabled")
        if not _qt_application_ready():
            raise ContentPreviewUnavailable("no-qapplication")
        if bool(getattr(selection, "defer_current_move", False)):
            raise ContentPreviewUnavailable("gesture-active")
        artists = _selected_artists(selection)
        if not artists:
            raise ContentPreviewUnavailable("empty-selection")
        (
            paint_complexity,
            _node_complexity,
            paint_components,
        ) = _require_bounded_composite_complexity(
            artists, max_paint_artists=self.max_artists
        )
        _require_unclipped_components(paint_components)
        draw_artists = _expanded_draw_artists(artists)
        if not draw_artists:
            raise ContentPreviewUnavailable("empty-selection")
        if len(artists) > self.max_artists or len(draw_artists) > self.max_artists:
            raise ContentPreviewUnavailable("artist-count-budget")
        figure = getattr(manager, "figure", None)
        canvas = getattr(figure, "canvas", None)
        scene_parent = getattr(figure, "_pyl_scene", None)
        if canvas is None or scene_parent is None:
            raise ContentPreviewUnavailable("no-scene")
        try:
            renderer = canvas.get_renderer()
        except (AttributeError, TypeError, ValueError, RuntimeError) as error:
            raise ContentPreviewUnavailable("no-renderer") from error

        token = ContentPreviewToken.capture(
            manager,
            artists,
            renderer,
            source_byte_limit=self.source_fingerprint_budget_bytes,
        )
        width, height, dpi = token.renderer_shape
        full_bytes = width * height * 4
        scan_bytes = (width + height) * (1 + np.dtype(np.intp).itemsize)
        idle_work_pixels = (
            width * height
            + paint_complexity * _IDLE_WORK_PIXELS_PER_PAINT_LEAF
        )
        if idle_work_pixels > DEFAULT_IDLE_WORK_PIXEL_BUDGET:
            raise ContentPreviewUnavailable("idle-work-budget")
        clone_reserve_bytes = max(
            int(token.source_bytes) * 8,
            paint_complexity * 128 * 1024,
        )
        # RendererAgg must fit before any allocation.  Later, the exact alpha
        # crop is checked for QImage/QPixmap conversion copies as well.
        if (
            full_bytes + scan_bytes + clone_reserve_bytes
            > self.memory_budget_bytes
        ):
            raise ContentPreviewUnavailable("memory-budget:renderer")

        ordered_artists = _paint_order(manager, draw_artists)
        capture_renderer = self._scratch_renderer(width, height, dpi)
        extent_clone: Artist | None = None
        draw_clone: Artist | None = None
        try:
            for artist in ordered_artists:
                if isinstance(artist, Legend):
                    try:
                        extent_clone = copy(artist)
                    except Exception as error:
                        raise ContentPreviewUnavailable("clone-failed") from error
                    extent_clone.stale_callback = None
                    extent_clone._remove_method = None
                    paint_artists = _legend_paint_artists(artist)
                else:
                    extent_clone = _disposable_draw_clone(artist)
                    paint_artists = (artist,)
                _require_clone_inside_canvas(
                    extent_clone, renderer, width=width, height=height
                )
                extent_clone = None
                for paint_artist in paint_artists:
                    draw_clone = _disposable_draw_clone(paint_artist)
                    draw_clone.draw(capture_renderer)
                    draw_clone = None
        except ContentPreviewUnavailable:
            raise
        except Exception as error:
            raise ContentPreviewUnavailable("capture-failed") from error
        finally:
            # Do not retain even a primitive clone across the token recheck;
            # this also prevents cyclic Matplotlib helper state accumulating
            # until a generation-2 GC pause.
            extent_clone = None
            draw_clone = None

        # A custom draw path must not be able to publish pixels for a source it
        # changed while rendering.  Standard Artists should compare equal.
        after = ContentPreviewToken.capture(
            manager,
            artists,
            renderer,
            source_byte_limit=self.source_fingerprint_budget_bytes,
        )
        if (
            after.artist_ids != token.artist_ids
            or after.source_fingerprints != token.source_fingerprints
            or after.revision != token.revision
            or after.renderer is not token.renderer
        ):
            raise ContentPreviewUnavailable("source-changed-during-capture")

        rgba = np.asarray(capture_renderer.buffer_rgba())
        alpha = rgba[:, :, 3]
        rows = np.flatnonzero(np.any(alpha != 0, axis=1))
        columns = np.flatnonzero(np.any(alpha != 0, axis=0))
        if not len(rows) or not len(columns):
            raise ContentPreviewUnavailable("empty-paint")
        left = int(columns[0])
        right = int(columns[-1]) + 1
        top = int(rows[0])
        bottom = int(rows[-1]) + 1
        if left <= 0 or top <= 0 or right >= width or bottom >= height:
            raise ContentPreviewUnavailable("canvas-edge-paint")
        crop_width = right - left
        crop_height = bottom - top
        crop_bytes = crop_width * crop_height * 4

        # Accounting envelope: one full Agg buffer, a deliberately amplified
        # source/clone reserve, one cropped QImage, the retained pixmap, and a
        # conversion temporary.  The amplified reserve covers Python object
        # graph overhead that raw ndarray byte counts do not expose.
        peak_bytes = (
            full_bytes
            + scan_bytes
            + clone_reserve_bytes
            + 3 * crop_bytes
        )
        if peak_bytes > self.memory_budget_bytes:
            raise ContentPreviewUnavailable("memory-budget:crop")

        try:
            image = QtGui.QImage(  # ty: ignore[no-matching-overload]
                memoryview(capture_renderer.buffer_rgba()),
                width,
                height,
                width * 4,
                _qimage_rgba8888_format(),
            )
            cropped = image.copy(left, top, crop_width, crop_height)
            pixmap = QtGui.QPixmap.fromImage(cropped)
        except (MemoryError, RuntimeError, TypeError, ValueError) as error:
            raise ContentPreviewUnavailable("qt-image") from error
        if pixmap.isNull():
            raise ContentPreviewUnavailable("qt-image")

        display_y0 = float(height - bottom)
        root = QtWidgets.QGraphicsRectItem(0.0, 0.0, 0.0, 0.0, scene_parent)
        root.setAcceptedMouseButtons(QtCore.Qt.NoButton)
        root.setZValue(850.0)
        item = QtWidgets.QGraphicsPixmapItem(pixmap, root)
        item.setAcceptedMouseButtons(QtCore.Qt.NoButton)
        item.setOpacity(self.opacity)
        item.setPos(float(left), display_y0)
        # Agg row zero is the canvas top.  The scene parent is y-up, so flip
        # the pixmap locally about its crop height before the parent transform.
        item.setTransform(
            QtGui.QTransform(
                1.0, 0.0, 0.0, -1.0, 0.0, float(crop_height)
            )
        )
        root.setVisible(False)
        return ContentPreviewEntry(
            token=token,
            root=root,
            item=item,
            pixmap=pixmap,
            display_bounds=(
                float(left),
                display_y0,
                float(right),
                float(height - top),
            ),
            canvas_bounds=(0.0, 0.0, float(width), float(height)),
            retained_bytes=full_bytes + crop_bytes,
            peak_bytes=peak_bytes,
        )

    def warm_now(self, manager, selection) -> bool:
        """Build from an idle callback (public mainly for deterministic tests)."""

        self.configure_from_selection(selection)
        if self.closed:
            return False
        try:
            entry = self._capture_entry(manager, selection)
        except ContentPreviewUnavailable as error:
            self.invalidate(error.reason)
            return False
        except Exception:
            self.invalidate("capture-failed")
            return False
        old = self.entry
        self.entry = entry
        if old is not None:
            old.remove()
        self.active = False
        self.capture_count += 1
        self.last_fallback_reason = None
        return True

    def activate(self, manager, selection) -> bool:
        """Consume a ready cache without capture, rendering, or measurement."""

        self.configure_from_selection(selection)
        if (
            not bool(getattr(selection, "content_preview_enabled", True))
            or not bool(getattr(selection, "defer_artist_updates", False))
        ):
            self.invalidate("disabled")
            return False
        entry = self.entry
        if self.closed or entry is None:
            self.active = False
            return False
        artists = _selected_artists(selection)
        try:
            renderer = manager.figure.canvas.get_renderer()
        except (AttributeError, TypeError, ValueError, RuntimeError):
            self.invalidate("no-renderer")
            return False
        if not entry.token.is_current(
            manager,
            artists,
            renderer,
            source_byte_limit=self.source_fingerprint_budget_bytes,
        ):
            self.invalidate("stale-token")
            return False
        try:
            entry.root.setTransform(QtGui.QTransform())
            entry.root.setVisible(False)
        except RuntimeError:
            self.invalidate("scene-destroyed")
            return False
        self.active = True
        self.activation_count += 1
        return True

    def update_transform(self, matrix: np.ndarray) -> bool:
        entry = self.entry
        if not self.active or entry is None:
            return False
        try:
            matrix = np.asarray(matrix, dtype=float)
            if matrix.shape != (3, 3) or not np.allclose(
                matrix[:2, :2], np.eye(2), atol=1e-12, rtol=0.0
            ):
                self.deactivate()
                self.last_fallback_reason = "translation-only"
                return False
            x0, y0, x1, y1 = entry.display_bounds
            dx, dy = float(matrix[0, 2]), float(matrix[1, 2])
            cx0, cy0, cx1, cy1 = entry.canvas_bounds
            if (
                x0 + dx <= cx0
                or y0 + dy <= cy0
                or x1 + dx >= cx1
                or y1 + dy >= cy1
            ):
                self.deactivate()
                self.last_fallback_reason = "canvas-edge-destination"
                return False
            transform = _matrix_to_qtransform(matrix)
            entry.root.setTransform(transform)
            entry.root.setVisible(True)
        except (ContentPreviewUnavailable, RuntimeError):
            self.invalidate("preview-transform")
            return False
        self.motion_update_count += 1
        return True

    def deactivate(self) -> None:
        self.active = False
        entry = self.entry
        if entry is not None:
            try:
                entry.root.setVisible(False)
                entry.root.setTransform(QtGui.QTransform())
            except RuntimeError:
                self.invalidate("scene-destroyed")


def ensure_content_preview_cache(manager) -> ContentPreviewCache:
    cache = getattr(manager, "_content_preview_cache", None)
    if cache is None or bool(getattr(cache, "closed", False)):
        cache = ContentPreviewCache()
        manager._content_preview_cache = cache
    return cache


def schedule_content_preview_warmup(manager) -> bool:
    """Schedule one generation-checked capture after the current Qt event."""

    selection = getattr(manager, "selection", None)
    if selection is None or not _qt_application_ready():
        return False
    cache = ensure_content_preview_cache(manager)
    cache.configure_from_selection(selection)
    cache.invalidate("selection-or-draw-changed")
    if (
        cache.closed
        or not bool(getattr(selection, "content_preview_enabled", True))
        or not bool(getattr(selection, "defer_artist_updates", False))
        or not getattr(selection, "targets", None)
    ):
        return False
    generation = cache.generation
    manager_ref = weakref.ref(manager)
    selection_ref = weakref.ref(selection)

    def warm() -> None:
        current_manager = manager_ref()
        current_selection = selection_ref()
        if current_manager is None or current_selection is None:
            return
        current_cache = getattr(current_manager, "_content_preview_cache", None)
        if (
            current_cache is not cache
            or cache.closed
            or cache.generation != generation
        ):
            return
        cache.warm_now(current_manager, current_selection)

    QtCore.QTimer.singleShot(0, warm)
    return True


def invalidate_content_preview_cache(manager, reason: str = "invalidated") -> None:
    cache = getattr(manager, "_content_preview_cache", None)
    if cache is not None:
        cache.invalidate(reason)


def activate_content_preview(selection) -> bool:
    manager = getattr(getattr(selection, "figure", None), "figure_dragger", None)
    if manager is None:
        return False
    return ensure_content_preview_cache(manager).activate(manager, selection)


def update_content_preview(selection, matrix: np.ndarray) -> bool:
    manager = getattr(getattr(selection, "figure", None), "figure_dragger", None)
    cache = getattr(manager, "_content_preview_cache", None) if manager else None
    return False if cache is None else cache.update_transform(matrix)


def deactivate_content_preview(selection) -> None:
    manager = getattr(getattr(selection, "figure", None), "figure_dragger", None)
    cache = getattr(manager, "_content_preview_cache", None) if manager else None
    if cache is not None:
        cache.deactivate()


def close_content_preview_cache(manager) -> None:
    cache = getattr(manager, "_content_preview_cache", None)
    if cache is not None:
        cache.close()


__all__ = [
    "ContentPreviewCache",
    "ContentPreviewEntry",
    "ContentPreviewToken",
    "ContentPreviewUnavailable",
    "DEFAULT_MEMORY_BUDGET_BYTES",
    "DEFAULT_MAX_ARTISTS",
    "DEFAULT_SOURCE_FINGERPRINT_BUDGET_BYTES",
    "activate_content_preview",
    "artist_source_fingerprint",
    "close_content_preview_cache",
    "deactivate_content_preview",
    "ensure_content_preview_cache",
    "invalidate_content_preview_cache",
    "schedule_content_preview_warmup",
    "selection_source_fingerprints",
    "update_content_preview",
]
