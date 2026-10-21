"""Agg scientific plotting and publication export, without Qt or pyplot state."""

from __future__ import annotations

import base64
from io import BytesIO
import json
import math
from threading import RLock

import matplotlib as mpl
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse, Polygon, Rectangle
from matplotlib.transforms import Affine2D
import numpy as np
from PIL import Image

from .document import DocumentError, MAX_PIXELS, number, validate_document, validate_plot

RENDER_LOCK = RLock()
_RC = {"text.usetex": False, "text.parse_math": False, "svg.fonttype": "none",
       "font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
       "axes.spines.right": False, "axes.unicode_minus": False}
_FORMATS = {"svg", "png", "pdf"}


def _rows(value):
    value = value.get("rows") if isinstance(value, dict) else value
    if not isinstance(value, list) or not all(isinstance(row, dict) for row in value):
        raise DocumentError("The dataset resolver must return a list of row objects.")
    return value


def _asset_bytes(value):
    value = value.get("data") if isinstance(value, dict) else value
    if not isinstance(value, bytes):
        raise DocumentError("The asset resolver must return image bytes.")
    if len(value) > 25_000_000:
        raise DocumentError("Image asset exceeds 25 MB.")
    return value


def _image(value):
    try:
        image = Image.open(BytesIO(_asset_bytes(value)))
        if image.width * image.height > MAX_PIXELS:
            raise DocumentError("Image asset exceeds 32 megapixels.")
        return np.asarray(image.convert("RGBA"))
    except (OSError, ValueError, Image.DecompressionBombError) as exc:
        raise DocumentError("The asset is not a supported raster image.") from exc


def _numeric(value):
    if value is None or value == "" or isinstance(value, bool):
        return None
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def _draw_plot(ax, plot, rows):
    rows = _rows(rows)
    xkey, ykey, kind = plot["x"], plot["y"], plot["kind"]
    if rows and (xkey not in rows[0] or (kind != "histogram" and ykey not in rows[0])):
        raise DocumentError("The selected plot column does not exist in the dataset.")
    color = plot["color"]
    if kind == "histogram":
        values = [n for row in rows if (n := _numeric(row.get(xkey))) is not None]
        if not values:
            raise DocumentError("The histogram column contains no finite numeric values.")
        ax.hist(values, bins=plot.get("bins", min(20, max(1, int(math.sqrt(len(values)))))),
                color=color, edgecolor="white", linewidth=0.5)
    elif kind == "bar":
        pairs = [(str(row[xkey]), y) for row in rows
                 if row.get(xkey) is not None and (y := _numeric(row.get(ykey))) is not None]
        if not pairs:
            raise DocumentError("The selected columns contain no valid bar values.")
        if len(pairs) > 1000:
            raise DocumentError("Bar charts support at most 1000 categories; aggregate the dataset first.")
        # Use explicit positions: repeated category names remain separate observations.
        ax.bar(range(len(pairs)), [y for _, y in pairs], color=color, width=0.72)
        ax.set_xticks(range(len(pairs)), [x for x, _ in pairs])
        if len(pairs) > 12:
            ax.tick_params(axis="x", labelrotation=45)
    else:
        pairs = [(x, y) for row in rows if (x := _numeric(row.get(xkey))) is not None
                 and (y := _numeric(row.get(ykey))) is not None]
        if not pairs:
            raise DocumentError("The selected columns contain no paired finite numeric values.")
        x, y = zip(*pairs)
        if kind == "line":
            ax.plot(x, y, color=color, linewidth=1.5)
        else:
            ax.scatter(x, y, color=color, s=20, alpha=0.85, edgecolors="none")
    ax.set_title(plot["title"], loc="left", fontsize=11, fontweight="bold", pad=10)
    ax.set_xlabel(plot["xlabel"])
    ax.set_ylabel(plot["ylabel"] or ("Count" if kind == "histogram" else ""))
    ax.tick_params(labelsize=8, length=3, colors="#374151")
    for spine in ax.spines.values():
        spine.set_linewidth(0.65)
        spine.set_color("#6b7280")
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#e5e7eb", linewidth=0.5)
    ax.margins(x=0.05)


def _plot_axes(fig, x, y, w, h, W, H, zorder):
    # Pixel-based margins remain stable between preview and document export.
    left = min(58, w * 0.22)
    right = min(16, w * 0.06)
    top = min(34, h * 0.16)
    bottom = min(46, h * 0.22)
    return fig.add_axes([(x + left) / W, (H - y - h + bottom) / H,
                         (w - left - right) / W, (h - top - bottom) / H],
                        zorder=zorder, facecolor="none")


def _save(fig, format):
    if not isinstance(format, str) or format not in _FORMATS:
        raise DocumentError("Export format must be svg, png, pdf, or python.")
    output = BytesIO()
    metadata = {"Date": None} if format == "svg" else ({"CreationDate": None, "ModDate": None} if format == "pdf" else {})
    fig.savefig(output, format=format, dpi=96, facecolor=fig.get_facecolor(), metadata=metadata)
    return output.getvalue()


def render_plot(plot, rows, width=480, height=320, format="svg"):
    """Render a real data-bound plot into a bounded, vector-capable panel."""
    plot = validate_plot(plot)
    width = number(width, "width", 32, 8192)
    height = number(height, "height", 32, 8192)
    if width * height > MAX_PIXELS:
        raise DocumentError("Plot exceeds 32 megapixels.")
    with RENDER_LOCK, mpl.rc_context(_RC):
        fig = Figure(figsize=(width / 96, height / 96), dpi=96, facecolor="white")
        FigureCanvasAgg(fig)
        ax = _plot_axes(fig, 0, 0, width, height, width, height, 0)
        _draw_plot(ax, plot, rows)
        return _save(fig, format)


def build_figure(document, dataset_resolver, asset_resolver):
    """Build the portable document as a Matplotlib Figure.

    Resolvers are supplied by the project store; this module never opens arbitrary
    filesystem paths or network URLs embedded in a document.
    """
    doc = validate_document(document)
    W, H = doc["width"], doc["height"]
    fig = Figure(figsize=(W / 96, H / 96), dpi=96, facecolor=doc["background"])
    FigureCanvasAgg(fig)
    for order, e in enumerate(doc["elements"]):
        if not e["visible"]:
            continue
        kind, x, y, w, h = e["type"], e["x"], e["y"], e["width"], e["height"]
        style, rotation = e["style"], e["rotation"]
        zorder = order + 1
        if kind == "plot" and not rotation:
            ax = _plot_axes(fig, x, y, w, h, W, H, zorder)
            _draw_plot(ax, e["plot"], dataset_resolver(e["plot"]["datasetId"]))
            ax.set_alpha(style["opacity"])
            for artist in ax.findobj():
                artist.set_alpha(style["opacity"])
            continue
        # A separate transparent axes per item preserves paint order across plots,
        # images and labels, without placing new objects in pyplot's global registry.
        ax = fig.add_axes([0, 0, 1, 1], zorder=zorder, facecolor="none")
        ax.set(xlim=(0, W), ylim=(H, 0))
        ax.set_axis_off()
        transform = Affine2D().rotate_deg_around(x + w / 2, y + h / 2, rotation) + ax.transData
        common = {"transform": transform, "alpha": style["opacity"], "clip_on": True}
        paint = {"facecolor": style["fill"], "edgecolor": style["stroke"],
                 "linewidth": style["strokeWidth"] * 0.75, **common}
        if kind == "rect":
            ax.add_patch(Rectangle((x, y), w, h, **paint))
        elif kind == "ellipse":
            ax.add_patch(Ellipse((x + w / 2, y + h / 2), w, h, **paint))
        elif kind in {"line", "path"}:
            pts = np.array(e["points"], dtype=float) + (x, y)
            if kind == "path" and e.get("closed"):
                ax.add_patch(Polygon(pts, closed=True, **paint))
            else:
                ax.add_line(Line2D(pts[:, 0], pts[:, 1], color=style["stroke"],
                                   linewidth=style["strokeWidth"] * 0.75, **common))
        elif kind == "text":
            align = style["textAlign"]
            tx = x + {"left": 0, "center": w / 2, "right": w}[align]
            # Same baseline contract as the SVG editor: first baseline at y+fontSize.
            ax.text(tx, y + style["fontSize"], e["text"], ha=align, va="baseline", color=style["fill"],
                    fontsize=style["fontSize"] * 0.75,
                    fontfamily=[family.strip() for family in style["fontFamily"].split(",")],
                    fontweight=style["fontWeight"], parse_math=False,
                    rotation=-rotation, rotation_mode="anchor", linespacing=1.25, **common)
        elif kind in {"image", "plot"}:
            if kind == "image":
                pixels = _image(asset_resolver(e["assetId"]))
            else:
                # Matplotlib axes are axis-aligned; rotated scientific panels are
                # deliberately rasterized, preserving the whole plot and its labels.
                pixels = _image(render_plot(e["plot"], dataset_resolver(e["plot"]["datasetId"]), w, h, "png"))
            ax.imshow(pixels, extent=(x, x + w, y + h, y), origin="upper", aspect="auto", **common)
    return fig


def _python_source(document, dataset_resolver, asset_resolver):
    doc = validate_document(document)
    dataset_ids = {e["plot"]["datasetId"] for e in doc["elements"] if e["type"] == "plot"}
    asset_ids = {e["assetId"] for e in doc["elements"] if e["type"] == "image"}
    datasets = {key: _rows(dataset_resolver(key)) for key in sorted(dataset_ids)}
    assets = {key: base64.b64encode(_asset_bytes(asset_resolver(key))).decode("ascii") for key in sorted(asset_ids)}
    payload = json.dumps({"document": doc, "datasets": datasets, "assets": assets}, ensure_ascii=True, allow_nan=False)
    # repr only quotes a JSON string; no uploaded text is interpolated as code.
    source = (
        '# Generated by Pylustrator Forge Web. No Qt is required.\n'
        '# Requires the pylustrator package used to create this document.\n'
        '# Original data and raster assets are embedded for reproducibility.\n'
        'import base64\nimport json\nfrom pathlib import Path\n'
        'from pylustrator.web.render import render_document\n\n'
        f'PAYLOAD = json.loads({payload!r})\n\n'
        'def dataset(asset_id):\n    return PAYLOAD["datasets"][asset_id]\n\n'
        'def image(asset_id):\n    return base64.b64decode(PAYLOAD["assets"][asset_id])\n\n'
        'if __name__ == "__main__":\n'
        '    for output_format in ("svg", "png", "pdf"):\n'
        '        Path("figure." + output_format).write_bytes(\n'
        '            render_document(PAYLOAD["document"], dataset, image, output_format)\n'
        '        )\n'
    )
    return source.encode("utf-8")


def render_document(document, dataset_resolver=None, asset_resolver=None, format="svg"):
    """Export a validated document as SVG/PNG/PDF or self-contained replay Python."""
    def missing(key):
        raise DocumentError(f"Referenced project asset is unavailable: {key}")
    dataset_resolver = dataset_resolver or missing
    asset_resolver = asset_resolver or missing
    if format == "python":
        return _python_source(document, dataset_resolver, asset_resolver)
    with RENDER_LOCK, mpl.rc_context(_RC):
        return _save(build_figure(document, dataset_resolver, asset_resolver), format)
