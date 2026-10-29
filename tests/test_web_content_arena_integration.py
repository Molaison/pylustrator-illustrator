"""Optional combined-branch contract check; Arena remains a separate feature.

Run this in a local worktree combining content bounds with the Arena branch.
The standalone content branch deliberately skips it when Arena is not present.
"""

from copy import deepcopy

import pytest
from fastapi.testclient import TestClient

from pylustrator.web.app import create_app
from pylustrator.web.content_regions import matplotlib_regions
from pylustrator.web.render import RENDER_LOCK
from pylustrator.web.session import FigureSession
from scripts.demo_content_regions import plotting_products


def test_content_documents_and_native_arena_coexist_without_history_or_coordinate_leakage(tmp_path):
    arena_module = pytest.importorskip("pylustrator.web.arena")
    from pylustrator.web.arena_demo import example_figure

    session = FigureSession(example_figure())
    app = arena_module.install_arena(create_app(tmp_path, sessions=[session]))
    store = app.state.store
    try:
        with TestClient(app) as client:
            project = store.create_project("Content and Arena integration")
            product = plotting_products()[0]
            imported = store.add_asset(project["id"], "Panel", product["png"], mime_type="image/png")
            element = {"id": "panel", "type": "image", "assetId": imported["asset"]["id"],
                       "x": 120, "y": 100, "width": 620, "height": 340}
            base = f"/api/arena/sessions/{session.id}"
            before = client.get(base + "/observe").json()
            response = client.post(f"/api/projects/{project['id']}/content-regions/detect", json={"element": element})
            assert response.status_code == 200, response.text
            assert session.revision == before["revision"]
            element["contentRegions"] = response.json()["regions"]
            element["contentRegions"]["plot"].update(method="manual", confirmed=True)
            project = store.get_project(project["id"])
            project["document"]["elements"] = [element]
            saved = store.update_project(project["id"], project["revision"], document=project["document"])
            content_snapshot = deepcopy(saved)
            # Both contracts use top-left coordinates. Native Arena pixels are
            # normalized by its own Figure dimensions, not by browser view zoom.
            with session.lock, RENDER_LOCK:
                regions = matplotlib_regions(session.figure)
            objects = [e for e in before["objects"] if e["basis"] == "plot_area"]
            for region, item in zip(regions, objects):
                b = region["plot"]["bounds"]
                width, height = before["canvas"]["width"], before["canvas"]["height"]
                assert item["bounds"] == pytest.approx([b["x"] * width, b["y"] * height, b["width"] * width, b["height"] * height])
            action = {"requestId": "integration-move", "revision": before["revision"], "operation": "move",
                      "target": objects[1]["id"], "dx": -20, "dy": -10}
            moved = client.post(base + "/actions", json=action)
            assert moved.status_code == 200, moved.text
            revision = moved.json()["observation"]["revision"]
            undone = client.post(base + "/actions", json={"requestId": "integration-undo", "revision": revision, "operation": "undo"})
            assert undone.status_code == 200, undone.text
            restored = [e for e in undone.json()["observation"]["objects"] if e["basis"] == "plot_area"]
            assert restored[1]["bounds"] == pytest.approx(objects[1]["bounds"])
            assert store.get_project(project["id"]) == content_snapshot
            assert store.get_asset(project["id"], element["assetId"])[0] == product["png"]
            # Native object IDs do not stand in for composition panel IDs.
            rejected = client.post(base + "/actions", json={**action, "requestId": "wrong-domain",
                                    "revision": undone.json()["observation"]["revision"], "target": element["id"]})
            assert rejected.status_code == 409
    finally:
        session.close()
