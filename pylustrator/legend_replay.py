"""Self-contained replay specifications for Legend entry glyphs.

Legend ``legend_handles`` retain only one representative Artist per entry.
That is insufficient for composite handlers such as error bars, so replay is
derived from each entry's DrawingArea and is allowed only when the entry is a
single, registered glyph.  Unsupported composites fail before generated code
is written instead of silently losing parts of their appearance.
"""

from __future__ import annotations

from collections.abc import Callable

import matplotlib as mpl
import numpy as np
from matplotlib.artist import Artist
from matplotlib.collections import LineCollection, PathCollection
from matplotlib.legend import Legend
from matplotlib.lines import Line2D
from matplotlib.offsetbox import DrawingArea
from matplotlib.patches import Patch
from matplotlib.path import Path

from .replay import replay_literal


class UnsupportedLegendEntry(TypeError):
    """Raised when a Legend entry cannot be reconstructed losslessly."""


class _Code(str):
    """A generated expression that must not be quoted as a string literal."""


LegendEntrySerializer = Callable[[Artist, float], str]
_SERIALIZERS: list[tuple[type[Artist], LegendEntrySerializer]] = []


def _value_code(value) -> str:
    if isinstance(value, _Code):
        return str(value)
    return replay_literal(value)


def _kwargs_code(kwargs: dict) -> str:
    return ", ".join(f"{name}={_value_code(value)}" for name, value in kwargs.items())


def _path_code(path: Path) -> _Code:
    vertices = np.asarray(path.vertices, dtype=float)
    codes = None if path.codes is None else np.asarray(path.codes, dtype=int)
    return _Code(
        f"mpl.path.Path({replay_literal(vertices)}, {replay_literal(codes)})"
    )


def _style_name(value):
    if value is None or isinstance(value, str):
        return value
    return getattr(value, "name", str(value))


def _line2d_code(handle: Line2D, marker_scale: float) -> str:
    marker = handle.get_marker()
    if isinstance(marker, Path):
        marker = _path_code(marker)
    linestyle = handle.get_linestyle()
    dash_pattern = getattr(handle, "_unscaled_dash_pattern", None)
    if handle.is_dashed() and dash_pattern is not None:
        linestyle = dash_pattern
    kwargs = {
        "label": handle.get_label(),
        "color": handle.get_color(),
        "linewidth": float(handle.get_linewidth()),
        "linestyle": linestyle,
        "marker": marker,
        "markersize": float(handle.get_markersize()) / marker_scale,
        "markeredgewidth": float(handle.get_markeredgewidth()),
        "markeredgecolor": handle.get_markeredgecolor(),
        "markerfacecolor": handle.get_markerfacecolor(),
        "markerfacecoloralt": handle.get_markerfacecoloralt(),
        "fillstyle": handle.get_fillstyle(),
        "antialiased": bool(handle.get_antialiased()),
        "drawstyle": handle.get_drawstyle(),
        "markevery": handle.get_markevery(),
        "dash_capstyle": _style_name(handle.get_dash_capstyle()),
        "solid_capstyle": _style_name(handle.get_solid_capstyle()),
        "dash_joinstyle": _style_name(handle.get_dash_joinstyle()),
        "solid_joinstyle": _style_name(handle.get_solid_joinstyle()),
        "alpha": handle.get_alpha(),
    }
    gapcolor = handle.get_gapcolor()
    if gapcolor is not None:
        kwargs["gapcolor"] = gapcolor
    return f"mpl.lines.Line2D([], [], {_kwargs_code(kwargs)})"


def _patch_code(handle: Patch, _marker_scale: float) -> str:
    kwargs = {
        "label": handle.get_label(),
        "facecolor": handle.get_facecolor(),
        "edgecolor": handle.get_edgecolor(),
        "linewidth": float(handle.get_linewidth()),
        "linestyle": handle.get_linestyle(),
        "antialiased": bool(handle.get_antialiased()),
        "hatch": handle.get_hatch(),
        "fill": bool(handle.get_fill()),
        "capstyle": _style_name(handle.get_capstyle()),
        "joinstyle": _style_name(handle.get_joinstyle()),
        "alpha": handle.get_alpha(),
    }
    return f"mpl.patches.Patch({_kwargs_code(kwargs)})"


def _path_collection_code(handle: PathCollection, marker_scale: float) -> str:
    paths = _Code("[" + ", ".join(_path_code(path) for path in handle.get_paths()) + "]")
    kwargs = {
        "sizes": np.asarray(handle.get_sizes(), dtype=float) / marker_scale**2,
        "facecolors": np.asarray(handle.get_facecolors(), dtype=float),
        "edgecolors": np.asarray(handle.get_edgecolors(), dtype=float),
        "linewidths": np.asarray(handle.get_linewidths(), dtype=float),
        "linestyles": getattr(handle, "_us_linestyles", handle.get_linestyles()),
        "antialiaseds": np.asarray(handle.get_antialiaseds(), dtype=bool),
        "hatch": handle.get_hatch(),
        "alpha": handle.get_alpha(),
        "label": handle.get_label(),
    }
    return f"mpl.collections.PathCollection({paths}, {_kwargs_code(kwargs)})"


def _line_collection_code(handle: LineCollection, _marker_scale: float) -> str:
    segments = [np.asarray(path.vertices, dtype=float) for path in handle.get_paths()]
    kwargs = {
        "colors": np.asarray(handle.get_colors(), dtype=float),
        "linewidths": np.asarray(handle.get_linewidths(), dtype=float),
        "linestyles": getattr(handle, "_us_linestyles", handle.get_linestyles()),
        "antialiaseds": np.asarray(handle.get_antialiaseds(), dtype=bool),
        "alpha": handle.get_alpha(),
        "label": handle.get_label(),
    }
    return (
        f"mpl.collections.LineCollection({replay_literal(segments)}, "
        f"{_kwargs_code(kwargs)})"
    )


def register_legend_entry_serializer(
    artist_type: type[Artist], serializer: LegendEntrySerializer, *, prepend=False
) -> None:
    """Register ``serializer(handler_artist, markerscale) -> expression``."""

    if not isinstance(artist_type, type) or not issubclass(artist_type, Artist):
        raise TypeError("artist_type must be an Artist subclass")
    if not callable(serializer):
        raise TypeError("serializer must be callable")
    _SERIALIZERS[:] = [item for item in _SERIALIZERS if item[0] is not artist_type]
    item = (artist_type, serializer)
    if prepend:
        _SERIALIZERS.insert(0, item)
    else:
        _SERIALIZERS.append(item)


def _drawing_area_entries(legend) -> list[tuple[Artist, ...]]:
    entries = []

    def visit(box) -> None:
        if isinstance(box, DrawingArea):
            entries.append(tuple(box.get_children()))
            return
        getter = getattr(box, "get_children", None)
        if getter is not None:
            for child in getter():
                visit(child)

    visit(legend._legend_handle_box)
    labels = legend.get_texts()
    if len(entries) != len(labels):
        raise UnsupportedLegendEntry(
            "Legend handler layout does not expose exactly one DrawingArea per entry"
        )
    return entries


def _serializer_for(handle: Artist) -> LegendEntrySerializer:
    for artist_type, serializer in _SERIALIZERS:
        if isinstance(handle, artist_type):
            return serializer
    raise UnsupportedLegendEntry(
        f"Legend entry glyph {type(handle).__name__} has no replay serializer"
    )


def _entry_glyph_signature(legend) -> tuple[tuple[str, ...], ...]:
    marker_scale = float(legend.markerscale)
    if not np.isfinite(marker_scale) or marker_scale <= 0:
        raise UnsupportedLegendEntry(
            "Legend markerscale must be finite and positive for replay comparison"
        )
    signature = []
    for index, artists in enumerate(_drawing_area_entries(legend)):
        entry = []
        for handle in artists:
            try:
                entry.append(_serializer_for(handle)(handle, marker_scale))
            except UnsupportedLegendEntry:
                raise
            except (AttributeError, TypeError, ValueError) as exc:
                raise UnsupportedLegendEntry(
                    f"Legend entry {index} cannot be represented exactly: {exc}"
                ) from exc
        signature.append(tuple(entry))
    return tuple(signature)


def frozen_legend_handles_code(legend) -> str:
    """Return a self-contained handle list for a single-glyph Legend."""

    codes = []
    for index, entry in enumerate(_entry_glyph_signature(legend)):
        if len(entry) != 1:
            raise UnsupportedLegendEntry(
                f"Legend entry {index} is composite ({len(entry)} glyphs); "
                "a full handler specification is required for lossless replay"
            )
        codes.append(entry[0])
    return "[" + ", ".join(codes) + "]"


def axes_handles_reproduce_legend(legend, handles, labels) -> bool | None:
    """Whether Axes handles reproduce the glyphs, or ``None`` if unknown."""

    current_labels = [text.get_text() for text in legend.get_texts()]
    if len(handles) != len(current_labels) or list(labels) != current_labels:
        return False
    try:
        candidate = Legend(
            legend.axes,
            handles,
            labels,
            numpoints=legend.numpoints,
            markerscale=legend.markerscale,
            scatterpoints=legend.scatterpoints,
            fontsize=legend._fontsize,
            handler_map=getattr(legend, "_custom_handler_map", None),
        )
        return _entry_glyph_signature(candidate) == _entry_glyph_signature(legend)
    except (AttributeError, TypeError, ValueError, UnsupportedLegendEntry):
        return None


def replayable_legend_handles(legend) -> list[Artist]:
    """Return unscaled semantic handles suitable for Legend reconstruction."""

    if legend.axes is not None and legend.axes.get_legend() is legend:
        handles, labels = original_axes_legend_handles_labels(legend.axes)
        if axes_handles_reproduce_legend(legend, handles, labels) is True:
            return list(handles)
    code = frozen_legend_handles_code(legend)
    return eval(code, {"__builtins__": {}, "mpl": mpl, "np": np})


def original_axes_legend_handles_labels(axes):
    """Read semantic Axes handles without the legacy proxy compatibility shim."""

    getter = getattr(
        axes,
        "_pylustrator_original_get_legend_handles_labels",
        axes.get_legend_handles_labels,
    )
    return getter()


register_legend_entry_serializer(Line2D, _line2d_code)
register_legend_entry_serializer(Patch, _patch_code)
register_legend_entry_serializer(PathCollection, _path_collection_code)
register_legend_entry_serializer(LineCollection, _line_collection_code)


__all__ = [
    "UnsupportedLegendEntry",
    "axes_handles_reproduce_legend",
    "frozen_legend_handles_code",
    "original_axes_legend_handles_labels",
    "replayable_legend_handles",
    "register_legend_entry_serializer",
]
