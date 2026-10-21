"""Validated, JSON-only scientific figure documents.

The document is the interchange boundary, not a serialization of a Python Figure.
All coordinates are CSS pixels at 96 DPI and have a top-left origin.
"""

from __future__ import annotations

from copy import deepcopy
import json
import math
import re
from uuid import uuid4

from matplotlib.colors import is_color_like

SCHEMA_VERSION = 1
MAX_ELEMENTS = 2000
MAX_DIMENSION = 8192
MAX_PIXELS = 32_000_000
ELEMENT_TYPES = frozenset({"text", "rect", "ellipse", "line", "path", "plot", "image"})
PLOT_TYPES = frozenset({"line", "scatter", "bar", "histogram"})
_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")


class DocumentError(ValueError):
    """A document violates the portable rendering contract."""


def identifier(value, field="id") -> str:
    if not isinstance(value, str) or not _ID.fullmatch(value):
        raise DocumentError(f"{field} must be a nonempty identifier (letters, digits, . _ : -).")
    return value


def number(value, field, minimum=-1_000_000, maximum=1_000_000) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DocumentError(f"{field} must be a number.")
    value = float(value)
    if not math.isfinite(value) or not minimum <= value <= maximum:
        raise DocumentError(f"{field} must be finite and between {minimum} and {maximum}.")
    return value


def _text(value, field, limit=10000):
    if not isinstance(value, str) or len(value) > limit:
        raise DocumentError(f"{field} must be text of at most {limit} characters.")
    return value


def _color(value, field):
    if not isinstance(value, str) or len(value) > 64:
        raise DocumentError(f"{field} must be a CSS color.")
    if value == "transparent":
        return "none"
    if not is_color_like(value):
        raise DocumentError(f"{field} is not a supported color.")
    return value


def new_document(name="Figure 1", width=960, height=640):
    return validate_document({
        "schemaVersion": SCHEMA_VERSION, "id": "figure-" + uuid4().hex,
        "name": name, "width": width, "height": height,
        "background": "#ffffff", "groups": [], "elements": [],
    })


def validate_plot(plot):
    if not isinstance(plot, dict):
        raise DocumentError("plot must be an object.")
    result = {
        "datasetId": identifier(plot.get("datasetId"), "plot.datasetId"),
        "kind": plot.get("kind", "scatter"),
        "x": _text(plot.get("x", ""), "plot.x", 500),
        "y": _text(plot.get("y", ""), "plot.y", 500),
        "title": _text(plot.get("title", ""), "plot.title", 1000),
        "xlabel": _text(plot.get("xlabel", plot.get("x", "")), "plot.xlabel", 1000),
        "ylabel": _text(plot.get("ylabel", plot.get("y", "")), "plot.ylabel", 1000),
        "color": _color(plot.get("color", "#2563eb"), "plot.color"),
    }
    if not isinstance(result["kind"], str) or result["kind"] not in PLOT_TYPES:
        raise DocumentError("Unsupported plot type.")
    if not result["x"] or (result["kind"] != "histogram" and not result["y"]):
        raise DocumentError("A plot needs X and Y columns (histograms need X only).")
    if "bins" in plot:
        result["bins"] = int(number(plot["bins"], "plot.bins", 1, 200))
    return result


def validate_document(document):
    """Return an independent normalized document or raise :class:`DocumentError`."""
    if not isinstance(document, dict):
        raise DocumentError("The figure document must be an object.")
    if isinstance(document.get("schemaVersion"), bool) or document.get("schemaVersion") != SCHEMA_VERSION:
        raise DocumentError(f"Unsupported document schemaVersion; expected {SCHEMA_VERSION}.")
    # Reject Python values, non-finite metadata and oversized nested payloads at the boundary.
    try:
        encoded = json.dumps(document, allow_nan=False)
    except (ValueError, TypeError, RecursionError) as exc:
        raise DocumentError("The document must contain finite JSON values only.") from exc
    if len(encoded) > 16_000_000:
        raise DocumentError("The document exceeds the 16 MB size limit.")
    doc = {
        "schemaVersion": SCHEMA_VERSION,
        "id": identifier(document.get("id"), "document.id"),
        "name": _text(document.get("name", "Figure 1"), "document.name", 500),
        "width": number(document.get("width", 960), "document.width", 32, MAX_DIMENSION),
        "height": number(document.get("height", 640), "document.height", 32, MAX_DIMENSION),
        "background": _color(document.get("background", "#ffffff"), "background"),
        "groups": [], "elements": [],
    }
    if doc["width"] * doc["height"] > MAX_PIXELS:
        raise DocumentError("The figure exceeds the 32 megapixel limit.")
    groups = document.get("groups", [])
    elements = document.get("elements", [])
    if not isinstance(groups, list) or not isinstance(elements, list):
        raise DocumentError("groups and elements must be arrays.")
    if len(groups) > MAX_ELEMENTS or len(elements) > MAX_ELEMENTS:
        raise DocumentError(f"A figure supports at most {MAX_ELEMENTS} elements/groups.")
    group_ids = set()
    for group in groups:
        if not isinstance(group, dict):
            raise DocumentError("Each group must be an object.")
        gid = identifier(group.get("id"), "group.id")
        if gid in group_ids:
            raise DocumentError(f"Duplicate group ID: {gid}.")
        group_ids.add(gid)
        doc["groups"].append({"id": gid, "name": _text(group.get("name", "Group"), "group.name", 500)})
    ids = set()
    point_count = 0
    for item in elements:
        if not isinstance(item, dict):
            raise DocumentError("Each element must be an object.")
        eid = identifier(item.get("id"), "element.id")
        if eid in ids:
            raise DocumentError(f"Duplicate element ID: {eid}.")
        ids.add(eid)
        kind = item.get("type")
        if not isinstance(kind, str) or kind not in ELEMENT_TYPES:
            raise DocumentError(f"Unsupported element type: {kind!r}.")
        e = {
            "id": eid, "type": kind,
            "name": _text(item.get("name", kind.title()), "element.name", 500),
            "x": number(item.get("x", 0), "element.x"),
            "y": number(item.get("y", 0), "element.y"),
            "width": number(item.get("width", 100), "element.width", 0.01, MAX_DIMENSION),
            "height": number(item.get("height", 100), "element.height", 0.01, MAX_DIMENSION),
            "rotation": number(item.get("rotation", 0), "element.rotation", -36000, 36000),
            "visible": item.get("visible", True), "locked": item.get("locked", False),
            "groupId": item.get("groupId"),
        }
        if not isinstance(e["visible"], bool) or not isinstance(e["locked"], bool):
            raise DocumentError("visible and locked must be booleans.")
        if e["groupId"] is not None and (not isinstance(e["groupId"], str) or e["groupId"] not in group_ids):
            raise DocumentError(f"Element {eid} refers to a missing group.")
        style = item.get("style", {})
        if not isinstance(style, dict):
            raise DocumentError("element.style must be an object.")
        e["style"] = {
            "fill": _color(style.get("fill", "#1f2937" if kind == "text" else "none"), "style.fill"),
            "stroke": _color(style.get("stroke", "none" if kind == "text" else "#374151"), "style.stroke"),
            "strokeWidth": number(style.get("strokeWidth", 1), "style.strokeWidth", 0, 200),
            "opacity": number(style.get("opacity", 1), "style.opacity", 0, 1),
            "fontSize": number(style.get("fontSize", 20), "style.fontSize", 1, 1000),
            "fontFamily": _text(style.get("fontFamily", "DejaVu Sans"), "style.fontFamily", 100),
            "textAlign": style.get("textAlign", "left"),
            "fontWeight": style.get("fontWeight", "normal"),
        }
        if not isinstance(e["style"]["textAlign"], str) or e["style"]["textAlign"] not in {"left", "center", "right"}:
            raise DocumentError("textAlign must be left, center, or right.")
        if not isinstance(e["style"]["fontWeight"], str) or e["style"]["fontWeight"] not in {"normal", "bold"}:
            raise DocumentError("fontWeight must be normal or bold.")
        if kind == "text":
            e["text"] = _text(item.get("text", ""), "element.text")
        elif kind in {"line", "path"}:
            points = item.get("points", [[0, 0], [e["width"], e["height"]]])
            if not isinstance(points, list) or not 2 <= len(points) <= 100000:
                raise DocumentError("Paths need between 2 and 100000 points.")
            e["points"] = []
            for point in points:
                if not isinstance(point, (list, tuple)) or len(point) != 2:
                    raise DocumentError("Each path point must be [x, y].")
                e["points"].append([number(point[0], "point.x"), number(point[1], "point.y")])
            e["closed"] = item.get("closed", False)
            if not isinstance(e["closed"], bool):
                raise DocumentError("closed must be a boolean.")
            point_count += len(points)
        elif kind == "plot":
            e["plot"] = validate_plot(item.get("plot"))
        elif kind == "image":
            e["assetId"] = identifier(item.get("assetId"), "element.assetId")
        doc["elements"].append(e)
    if point_count > 500000:
        raise DocumentError("The figure exceeds the 500000 point budget.")
    return deepcopy(doc)
