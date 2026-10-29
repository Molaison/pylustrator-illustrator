"""Detection uses independent real plotting geometry, not guessed ideal rectangles."""

from copy import deepcopy
from io import BytesIO

import numpy as np
import pytest
from fastapi.testclient import TestClient
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from PIL import Image

from pylustrator.web.app import create_app
from pylustrator.web.content_regions import (
    detect_element_regions,
    matplotlib_regions,
    raster_regions,
)
from pylustrator.web.document import DocumentError, new_document, validate_document
from scripts.demo_content_regions import plotting_products


def test_real_rasters_find_data_axes_not_label_or_panel_bounds():
    for item in plotting_products():
        result = raster_regions(item["png"])
        assert result["plot"] is not None, result
        detected = result["plot"]["bounds"]
        for key, expected in item["plotBounds"].items():
            dimension = item["width"] if key in {"x", "width"} else item["height"]
            assert abs(detected[key] - expected) * dimension <= 2
        assert not result["plot"]["confirmed"]
        assert result["content"]["bounds"]["x"] < detected["x"]
        assert result["content"]["bounds"]["width"] > detected["width"]


def test_transparent_background_and_explicit_failures():
    item = plotting_products()[0]
    pixels = np.array(Image.open(BytesIO(item["png"])).convert("RGBA"))
    pixels[np.all(pixels[:, :, :3] > 250, axis=2), 3] = 0
    stream = BytesIO()
    Image.fromarray(pixels).save(stream, format="PNG")
    assert raster_regions(stream.getvalue())["plot"] is not None
    for mode, color in [("RGBA", (0, 0, 0, 0)), ("RGB", "white")]:
        blank = BytesIO()
        Image.new(mode, (180, 120), color).save(blank, format="PNG")
        result = raster_regions(blank.getvalue())
        assert result["plot"] is None and result["content"] is None
        assert result["warnings"]
    # A real two-Axes plot must never silently choose one subpanel.
    fig = Figure(figsize=(8, 3), dpi=100)
    FigureCanvasAgg(fig)
    for bounds in ([.08, .2, .36, .6], [.57, .2, .36, .6]):
        ax = fig.add_axes(bounds)
        ax.plot([0, 1, 2], [1, 2, 1])
    stream = BytesIO()
    fig.savefig(stream, format="png")
    result = raster_regions(stream.getvalue())
    assert result["plot"] is None
    assert any("Multiple" in w for w in result["warnings"])
    # A gradient touching the border cannot establish a uniform background.
    gradient = np.repeat(np.arange(256, dtype=np.uint8)[None, :, None], 3, axis=2)
    stream = BytesIO()
    Image.fromarray(np.repeat(gradient, 120, axis=0)).save(stream, format="PNG")
    result = raster_regions(stream.getvalue())
    assert result["plot"] is None and result["content"] is None
    assert "Nonuniform" in result["warnings"][0]
    # A real plot with hidden spines may still contain foreground, but has no
    # defensible axis rectangle. It must not align by a crop of its data marks.
    fig = Figure(figsize=(4, 3), dpi=100)
    FigureCanvasAgg(fig)
    ax = fig.add_subplot()
    ax.scatter([1, 2, 3], [2, 1, 4])
    ax.spines[:].set_visible(False)
    stream = BytesIO()
    fig.savefig(stream, format="png")
    result = raster_regions(stream.getvalue())
    assert result["content"] is not None and result["plot"] is None


def test_native_axes_exact_bounds_and_distinct_decorations():
    fig = Figure(figsize=(6, 4), dpi=100)
    FigureCanvasAgg(fig)
    ax = fig.add_axes([.2, .2, .6, .65])
    ax.plot([1, 2], [3, 4], label="Kept legend")
    ax.set_ylabel("Long scientific label")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1))
    region = matplotlib_regions(fig)[0]
    assert region["plot"]["bounds"] == pytest.approx({"x": .2, "y": .15, "width": .6, "height": .65})
    assert region["content"]["bounds"]["x"] < .2
    assert region["content"]["bounds"]["width"] > .6
    assert region["plot"]["confirmed"]


def test_composition_plot_uses_actual_renderer_axes():
    doc = new_document()
    doc["elements"] = [{"id": "p", "type": "plot", "width": 480, "height": 320,
                        "plot": {"datasetId": "d", "x": "x", "y": "y"}}]
    element = validate_document(doc)["elements"][0]
    result = detect_element_regions(element, lambda _: [{"x": 0, "y": 1}, {"x": 1, "y": 3}], lambda _: None)
    assert result["plot"]["method"] == "matplotlib"
    assert result["plot"]["bounds"]["x"] == pytest.approx(58 / 480)
    assert result["source"]["width"] == 480


def test_detection_read_only_project_scoped_and_persistent(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        store = client.app.state.store
        project = store.create_project("Regions")
        pid = project["id"]
        original = plotting_products()[0]["png"]
        added = store.add_asset(pid, "Original", original, mime_type="image/png")
        element = {"id": "image", "type": "image", "assetId": added["asset"]["id"], "width": 620, "height": 340}
        before = store.get_project(pid)
        response = client.post(f"/api/projects/{pid}/content-regions/detect", json={"element": element})
        assert response.status_code == 200, response.text
        assert store.get_project(pid) == before
        foreign = store.create_project("Other")
        assert client.post(f"/api/projects/{foreign['id']}/content-regions/detect", json={"element": element}).status_code == 404
        element["contentRegions"] = response.json()["regions"]
        element["contentRegions"]["plot"].update(method="manual", confirmed=True)
        doc = before["document"]
        doc["elements"] = [element]
        saved = client.put(f"/api/projects/{pid}", json={"revision": before["revision"], "document": doc})
        assert saved.status_code == 200, saved.text
        region = saved.json()["document"]["elements"][0]["contentRegions"]
        imported = store.import_bundle(store.export_bundle(pid))
        assert imported["document"]["elements"][0]["contentRegions"] == region
        assert store.get_asset(pid, element["assetId"])[0] == original
        for invalid in [float("nan"), -1, True, 2]:
            broken = deepcopy(doc)
            broken["elements"][0]["contentRegions"]["plot"]["bounds"]["x"] = invalid
            with pytest.raises(DocumentError):
                validate_document(broken)
