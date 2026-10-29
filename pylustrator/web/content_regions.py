"""Non-destructive panel regions. No editor, store, network or model dependency.

Bounds use fractions of the *complete* panel, with a top-left origin. They never
change an image, dataset, crop, or plot. Raster suggestions require confirmation.
"""

from __future__ import annotations

from copy import deepcopy
from io import BytesIO

import numpy as np
from PIL import Image

from .document import DocumentError, identifier, number, validate_plot

SPACE = "normalized-panel"


def validate_bounds(value):
    if not isinstance(value, dict):
        raise DocumentError("A region needs x, y, width and height.")
    result = {k: number(value.get(k), f"region.{k}", 0 if k in {"x", "y"} else 1e-6, 1)
              for k in ("x", "y", "width", "height")}
    if result["x"] + result["width"] > 1 + 1e-9 or result["y"] + result["height"] > 1 + 1e-9:
        raise DocumentError("The region must fit inside the original panel.")
    return result


def source_for(element):
    if element["type"] == "image":
        return {"type": "image", "assetId": element["assetId"]}
    if element["type"] == "plot":
        return {"type": "plot", "plot": deepcopy(element["plot"]),
                "width": element["width"], "height": element["height"]}
    raise DocumentError("Content regions apply to image and plot panels.")


def validate_regions(value):
    if not isinstance(value, dict) or type(value.get("version")) is not int or value["version"] != 1:
        raise DocumentError("Unsupported content region version.")
    if value.get("coordinateSpace") != SPACE:
        raise DocumentError("Content regions must use normalized-panel coordinates.")
    source = value.get("source")
    if not isinstance(source, dict):
        raise DocumentError("Content regions need their source identity.")
    if source.get("type") == "image":
        source = {"type": "image", "assetId": identifier(source.get("assetId"))}
    elif source.get("type") == "plot":
        source = {"type": "plot", "plot": validate_plot(source.get("plot")),
                  "width": number(source.get("width"), "region source width", .01, 8192),
                  "height": number(source.get("height"), "region source height", .01, 8192)}
    else:
        raise DocumentError("Unsupported content region source.")
    result = {"version": 1, "coordinateSpace": SPACE, "source": source}
    for kind in ("plot", "content"):
        region = value.get(kind)
        if region is None:
            result[kind] = None
            continue
        if not isinstance(region, dict) or region.get("method") not in {"matplotlib", "raster-spines", "background", "manual"}:
            raise DocumentError("Unsupported region detection method.")
        if not isinstance(region.get("confirmed"), bool):
            raise DocumentError("Region confirmation must be boolean.")
        result[kind] = {"bounds": validate_bounds(region.get("bounds")), "method": region["method"],
                        "confidence": number(region.get("confidence"), "confidence", 0, 1),
                        "confirmed": region["confirmed"]}
    warnings = value.get("warnings", [])
    if not isinstance(warnings, list) or len(warnings) > 10 or any(not isinstance(x, str) or len(x) > 500 for x in warnings):
        raise DocumentError("Region warnings must be short text messages.")
    result["warnings"] = warnings[:]
    return result


def _bounds(x0, y0, x1, y1, width, height):
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(width, x1), min(height, y1)
    if x1 <= x0 or y1 <= y0:
        return None
    return {"x": x0 / width, "y": y0 / height, "width": (x1 - x0) / width, "height": (y1 - y0) / height}


def _region(bounds, method, confidence, confirmed=False):
    return None if bounds is None else {"bounds": bounds, "method": method, "confidence": confidence, "confirmed": confirmed}


def matplotlib_regions(figure):
    """Read actual Axes geometry; callers own the render lock and figure lifetime.

    Returns one record per visible Axes, never a guessed union of multiple plots.
    Includes Axes titles, tick labels, annotations and legends in content bounds.
    """
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    width, height = figure.bbox.size
    records = []
    for ax in figure.axes:
        if not ax.get_visible() or not ax.axison:
            continue
        plot = ax.get_window_extent(renderer)
        content = ax.get_tightbbox(renderer)
        if content is None:
            content = plot

        def convert(b):
            return _bounds(b.x0, height - b.y1, b.x1, height - b.y0, width, height)
        warnings = []
        if content.x0 < 0 or content.y0 < 0 or content.x1 > width or content.y1 > height:
            warnings.append("Some labels already extend outside the source panel; enlarge its source before export.")
        records.append({"plot": _region(convert(plot), "matplotlib", 1, True),
                        "content": _region(convert(content), "matplotlib", 1, True), "warnings": warnings})
    return records


def _segments(mask, horizontal):
    """Long contiguous dark runs, collapsed across the thickness of a spine."""
    data = mask if horizontal else mask.T
    minimum = max(30, int(data.shape[1] * .22))
    lines = []
    for at, row in enumerate(data):
        edges = np.diff(np.r_[False, row, False].astype(np.int8))
        for start, stop in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
            if stop - start >= minimum:
                lines.append((at, int(start), int(stop - 1)))
    groups = []
    for line in lines:
        prior = next((g for g in reversed(groups[-8:]) if line[0] - g[-1][0] <= 3
                      and abs(line[1] - g[-1][1]) <= 3 and abs(line[2] - g[-1][2]) <= 3), None)
        if prior is None:
            groups.append([line])
        else:
            prior.append(line)
    return [tuple(float(np.median([line[i] for line in group])) for i in range(3)) for group in groups]


def raster_regions(data):
    """Bounded CPU heuristic for a single Cartesian panel on uniform background.

    Pixel foreground is NOT a data plot. A plot suggestion additionally requires
    a left/bottom spine junction. Multiple junctions are explicitly ambiguous.
    """
    if not isinstance(data, bytes) or len(data) > 32 * 1024 * 1024:
        raise DocumentError("Use PNG bytes of at most 32 MiB.")
    with Image.open(BytesIO(data)) as original:
        if original.format != "PNG" or original.width * original.height > 32_000_000:
            raise DocumentError("Use a PNG with at most 32 megapixels.")
        image = original.convert("RGBA")
        image.thumbnail((1400, 1400), Image.Resampling.LANCZOS)
        pixels = np.asarray(image).astype(np.float32)
    height, width = pixels.shape[:2]
    alpha = pixels[:, :, 3] / 255
    rgb = pixels[:, :, :3] * alpha[:, :, None] + 255 * (1 - alpha[:, :, None])
    border = np.concatenate((rgb[0], rgb[-1], rgb[:, 0], rgb[:, -1]))
    background = np.median(border, axis=0)
    uniform = np.mean(np.max(np.abs(border - background), axis=1) < 22) >= .95
    foreground = (np.max(np.abs(rgb - background), axis=2) > 22) & (alpha > .06)
    result = {"plot": None, "content": None, "warnings": []}
    if not uniform:
        result["warnings"].append("Nonuniform border/background: set the content and plot bounds manually.")
        return result
    ys, xs = np.nonzero(foreground)
    if not len(xs):
        result["warnings"].append("No visible foreground detected. Set bounds manually if this is a faint plot.")
        return result
    result["content"] = _region(_bounds(xs.min(), ys.min(), xs.max() + 1, ys.max() + 1, width, height), "background", .8)
    # Neutral, dark strokes exclude colored curves and most pale grid lines.
    dark = (rgb.max(axis=2) < 155) & (np.ptp(rgb, axis=2) < 35) & foreground
    horizontal, vertical = _segments(dark, True), _segments(dark, False)
    candidates = []
    for bottom, left, right in horizontal:
        for x, top, end in vertical:
            # Outward ticks can extend the runs beyond the spine intersection.
            if abs(x - left) <= 8 and abs(bottom - end) <= 8 and right - x > width * .22 and bottom - top > height * .22:
                bounds = _bounds(x, top, right, bottom, width, height)
                if bounds and not any(max(abs(bounds[k] - other[k]) for k in bounds) < .015 for other in candidates):
                    candidates.append(bounds)
    if len(candidates) == 1:
        result["plot"] = _region(candidates[0], "raster-spines", .8)
        result["warnings"].append("Raster axis-line suggestion: inspect the outline and confirm or correct before alignment.")
    elif candidates:
        result["warnings"].append("Multiple axis junctions detected (subplots/grid/frame). Select the intended plot bounds manually.")
    else:
        result["warnings"].append("No unambiguous left/bottom axis junction. Borderless, pale, rotated or photographic panels need manual plot bounds.")
    return result


def detect_element_regions(element, dataset_resolver, asset_resolver):
    source = source_for(element)
    if element["type"] == "image":
        result = raster_regions(asset_resolver(element["assetId"]))
    else:
        # Reuse the renderer's actual layout, not a second copy of its margins.
        import matplotlib as mpl
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure

        from .render import _RC, RENDER_LOCK, _draw_plot, _plot_axes
        width, height = element["width"], element["height"]
        if width * height > 32_000_000:
            raise DocumentError("Plot exceeds 32 megapixels.")
        with RENDER_LOCK, mpl.rc_context(_RC):
            fig = Figure(figsize=(width / 96, height / 96), dpi=96)
            FigureCanvasAgg(fig)
            ax = _plot_axes(fig, 0, 0, width, height, width, height, 0)
            _draw_plot(ax, element["plot"], dataset_resolver(element["plot"]["datasetId"]))
            result = matplotlib_regions(fig)[0]
    return validate_regions({"version": 1, "coordinateSpace": SPACE, "source": source, **result})
