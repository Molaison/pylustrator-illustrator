"""The browser's JSON boundary and real scientific publication exports."""

from copy import deepcopy
from io import BytesIO
import os
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

from PIL import Image
import pytest

from pylustrator.web.document import DocumentError, new_document, validate_document
from pylustrator.web.render import render_document, render_plot


def example():
    doc = new_document(width=640, height=480)
    doc["elements"] = [
        {"id": "plot", "type": "plot", "x": 20, "y": 70, "width": 450, "height": 320,
         "plot": {"datasetId": "source", "kind": "scatter", "x": "dose", "y": "response", "title": "Measured response"}},
        {"id": "label", "type": "text", "x": 20, "y": 20, "width": 300, "height": 30,
         "text": "Panel A <not markup> & raw", "style": {"fontSize": 22}},
    ]
    return doc


ROWS = [{"dose": 0, "response": 1}, {"dose": 1, "response": 3}, {"dose": 2, "response": 2}]


def test_package_import_never_loads_qt():
    script = """import sys
import importlib.abc
class NoQt(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith(('PyQt', 'PySide', 'qtpy', 'qtawesome')):
            raise AssertionError('Qt import: ' + fullname)
sys.meta_path.insert(0, NoQt())
import pylustrator
from pylustrator.web.session import FigureSession
from pylustrator.web.render import render_document
assert not any(m.startswith(('PyQt', 'PySide', 'qtpy')) for m in sys.modules)
"""
    subprocess.run([sys.executable, "-c", script], check=True,
                   env={**os.environ, "MPLBACKEND": "Agg"}, capture_output=True)


@pytest.mark.parametrize("mutation", [
    lambda d: d.update(schemaVersion=2),
    lambda d: d.update(width=float("nan")),
    lambda d: d.update(width=8192, height=8192),
    lambda d: d["elements"].append(deepcopy(d["elements"][0])),
    lambda d: d["elements"][0].update(groupId="missing"),
    lambda d: d["elements"][0].update(type="script"),
    lambda d: d["elements"][0].update(visible="false"),
    lambda d: d["elements"][0]["plot"].update(kind="eval"),
    lambda d: d["elements"][0]["plot"].update(datasetId="../../secret"),
    lambda d: d["elements"][1].update(style={"fill": "url(file:///secret)"}),
])
def test_invalid_documents_rejected_before_render(mutation):
    doc = example()
    mutation(doc)
    with pytest.raises(DocumentError):
        validate_document(doc)


def test_normalization_is_independent_and_preserves_real_data():
    doc = example()
    validated = validate_document(doc)
    validated["elements"][1]["text"] = "Different"
    assert doc["elements"][1]["text"].startswith("Panel A")
    svg = render_document(doc, lambda _: ROWS).decode()
    ET.fromstring(svg)
    assert "Measured response" in svg
    assert "&lt;not markup&gt;" in svg
    assert "Panel A" in svg
    changed = render_document(doc, lambda _: [{"dose": 0, "response": 1}, {"dose": 2, "response": 90}])
    assert changed != svg.encode()


@pytest.mark.parametrize("format,signature", [("png", b"\x89PNG"), ("pdf", b"%PDF"), ("svg", b"<?xml")])
def test_publication_exports(format, signature):
    output = render_document(example(), lambda _: ROWS, format=format)
    assert output.startswith(signature)
    if format == "png":
        image = Image.open(BytesIO(output)).convert("RGB")
        assert image.size == (640, 480)
        assert image.getextrema() != ((255, 255),) * 3


@pytest.mark.parametrize("kind", ["line", "scatter", "bar", "histogram"])
def test_plot_kinds_render_real_numeric_observations(kind):
    svg = render_plot({"datasetId": "data", "kind": kind, "x": "dose", "y": "response"}, ROWS)
    assert len(svg) > 2000


def test_missing_columns_and_non_numeric_data_have_explanations():
    plot = {"datasetId": "data", "kind": "scatter", "x": "dose", "y": "response"}
    with pytest.raises(DocumentError, match="column"):
        render_plot(plot, [{"dose": 1, "wrong": 2}])
    with pytest.raises(DocumentError, match="paired finite"):
        render_plot(plot, [{"dose": 1, "response": None}])


def test_exported_python_replays_data_without_user_text_execution(tmp_path):
    doc = example()
    doc["elements"][1]["text"] = "__import__('os').system('echo NOT_EXECUTED')"
    script = render_document(doc, lambda _: ROWS, format="python")
    source = tmp_path / "replay.py"
    source.write_bytes(script)
    env = {**os.environ, "PYTHONPATH": str(Path(__file__).parents[1]), "MPLBACKEND": "Agg"}
    result = subprocess.run([sys.executable, str(source)], cwd=tmp_path, env=env, capture_output=True, check=True)
    assert b"NOT_EXECUTED" not in result.stdout
    assert (tmp_path / "figure.png").read_bytes().startswith(b"\x89PNG")
    assert (tmp_path / "figure.pdf").read_bytes().startswith(b"%PDF")


def test_hidden_items_do_not_resolve_missing_assets_and_images_are_scoped():
    doc = new_document()
    doc["elements"] = [{"id": "image", "type": "image", "assetId": "foreign", "visible": False}]
    render_document(doc)
    doc["elements"][0]["visible"] = True
    with pytest.raises(DocumentError, match="unavailable"):
        render_document(doc)


def test_rotated_plot_and_raster_image_export():
    doc = example()
    doc["elements"][0]["rotation"] = 15
    image = Image.new("RGB", (20, 20), "red")
    data = BytesIO()
    image.save(data, format="PNG")
    doc["elements"].append({"id": "image", "type": "image", "assetId": "img", "x": 500, "y": 20,
                             "width": 60, "height": 60, "rotation": 30})
    output = render_document(doc, lambda _: ROWS, lambda _: data.getvalue())
    assert b"image/png;base64" in output
