"""Cross-domain API checks: data → plot → version → structure → export."""

import base64
import io
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
pytest.importorskip("gemmi")

from fastapi.testclient import TestClient
from matplotlib.figure import Figure
from PIL import Image

from pylustrator.web.app import create_app
from pylustrator.web.session import FigureSession


@pytest.fixture
def client(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        yield client


def good(response):
    assert response.status_code == 200, response.text
    return response.json()


def test_complete_data_plot_checkpoint_workflow(client):
    project = good(client.post("/api/projects", json={"name": "Research α"}))
    pid = project["id"]
    project = good(client.post(f"/api/projects/{pid}/datasets", json={
        "name": "kinetics.csv", "format": "csv", "content": "x,y\n0,1\n1,\n2,4\n3,8\n", "revision": project["revision"]}))
    did = project["datasets"][0]["id"]
    original = good(client.get(f"/api/projects/{pid}/datasets/{did}"))
    plan = good(client.post(f"/api/projects/{pid}/datasets/{did}/ai-plan", json={"provider": "local", "prompt": "fill missing y"}))
    preview = good(client.post(f"/api/projects/{pid}/datasets/{did}/preview", json={"steps": plan["steps"]}))
    assert preview["rowCount"] == 4
    project = good(client.post(f"/api/projects/{pid}/datasets/{did}/transform", json={
        "steps": plan["steps"], "expectedHash": plan["inputHash"], "revision": project["revision"], "provenance": plan}))
    assert good(client.get(f"/api/projects/{pid}/datasets/{did}"))["hash"] == original["hash"]
    derived = project["datasets"][-1]["id"]
    document = project["document"]
    document["elements"] = [{"id": "plot-1", "type": "plot", "x": 20, "y": 20, "width": 600, "height": 400,
                              "plot": {"datasetId": derived, "kind": "scatter", "x": "x", "y": "y"}}]
    project = good(client.put(f"/api/projects/{pid}", json={"revision": project["revision"], "document": document}))
    checkpoint = good(client.post(f"/api/projects/{pid}/versions", json={"name": "Figure submitted", "revision": project["revision"]}))
    for format, signature in [("svg", b"<svg"), ("png", b"\x89PNG"), ("pdf", b"%PDF"), ("python", b"pylustrator")]:
        response = client.post(f"/api/projects/{pid}/export", json={"format": format})
        assert response.status_code == 200, response.text
        assert signature in response.content
    changed = good(client.put(f"/api/projects/{pid}", json={"revision": project["revision"], "name": "Changed"}))
    restored = good(client.post(f"/api/projects/{pid}/versions/{checkpoint['id']}/restore", json={"revision": changed["revision"]}))
    assert restored["name"] == "Research α"
    assert len(good(client.get(f"/api/projects/{pid}/versions"))["versions"]) >= 2
    bundle = good(client.get(f"/api/projects/{pid}/bundle"))
    imported = good(client.post("/api/projects/import", json=bundle))
    assert imported["id"] != pid
    assert len(imported["datasets"]) == 2
    assert client.post(f"/api/projects/{imported['id']}/export", json={"format": "svg"}).status_code == 200


def test_stale_save_and_cross_project_asset_rejected(client):
    first = good(client.post("/api/projects", json={"name": "First"}))
    second = good(client.post("/api/projects", json={"name": "Second"}))
    pid = first["id"]
    good(client.put(f"/api/projects/{pid}", json={"revision": 0, "name": "Saved"}))
    conflict = client.put(f"/api/projects/{pid}", json={"revision": 0, "name": "Lost update"})
    assert conflict.status_code == 409
    assert conflict.json()["error"]["code"] == "revision_conflict"
    png = io.BytesIO()
    Image.new("RGB", (12, 12), "blue").save(png, format="PNG")
    result = good(client.post(f"/api/projects/{pid}/assets", json={"name": "snapshot", "data": base64.b64encode(png.getvalue()).decode()}))
    aid = result["asset"]["id"]
    assert client.get(f"/api/projects/{second['id']}/assets/{aid}").status_code == 404
    assert client.get(f"/api/projects/{pid}/assets/{aid}").content == png.getvalue()
    assert client.post(f"/api/projects/{pid}/assets", json={"data": "ZmFrZQ=="}).status_code == 400


def test_structure_round_trip_and_view(client):
    project = good(client.post("/api/projects", json={"name": "Protein"}))
    pid = project["id"]
    source = (Path(__file__).parents[1] / "pylustrator/web/examples/1crn.pdb").read_text()
    project = good(client.post(f"/api/projects/{pid}/structures", json={"name": "Crambin", "content": source, "format": "pdb"}))
    sid = project["structures"][0]["id"]
    structure = good(client.get(f"/api/projects/{pid}/structures/{sid}"))
    assert structure["content"] == source
    project = good(client.put(f"/api/projects/{pid}/structures/{sid}", json={
        "revision": project["revision"], "view": {"representation": "stick", "chain": "A", "camera": [0, 0, 0, -50, 0, 0, 0, 1]}}))
    assert project["structures"][0]["view"]["representation"] == "stick"


def test_same_origin_boundary_and_optional_auth(tmp_path, client):
    denied = client.post("/api/projects", json={"name": "Injected"}, headers={"Origin": "https://unrelated.example"})
    assert denied.status_code == 403
    assert client.post("/api/projects", content='{"name":"x"}', headers={"Content-Type": "text/plain"}).status_code == 415
    assert client.get("/api/health", headers={"Host": "unrelated.example"}).status_code == 400
    assert client.get("/api/health").headers["content-security-policy"].startswith("default-src 'self'")
    secret = "a-long-explicit-test-token-for-workspace"
    with TestClient(create_app(tmp_path, token=secret)) as locked:
        assert locked.get("/api/projects").status_code == 401
        assert locked.post("/api/auth", json={"token": "incorrect"}).status_code == 401
        assert locked.post("/api/auth", json={"token": secret}).status_code == 200
        assert locked.get("/api/projects").status_code == 200
        assert secret not in locked.get("/api/providers").text


def test_native_figure_api_retains_semantics(tmp_path):
    figure = Figure(figsize=(6, 4), dpi=96)
    label = figure.text(0.2, 0.3, "Native text")
    session = FigureSession(figure)
    with TestClient(create_app(tmp_path, sessions=[session])) as client:
        state = good(client.get(f"/api/sessions/{session.id}"))
        artist = next(artist for artist in state["artists"] if artist.get("text") == "Native text")
        moved = good(client.post(f"/api/sessions/{session.id}/commands", json={"revision": 0, "operation": "translate", "targets": [artist["id"]], "dx": 20, "dy": 10}))
        assert moved["revision"] == 1
        assert label.get_position() != (0.2, 0.3)
        changes = client.get(f"/api/sessions/{session.id}/changes.py")
        assert changes.status_code == 200
        assert "attachment" in changes.headers["content-disposition"]
        replay = Figure(figsize=(6, 4), dpi=96)
        replay_label = replay.text(0.2, 0.3, "Native text")
        exec(compile(changes.text, "downloaded_changes.py", "exec"), {"fig": replay})
        assert replay_label.get_position() == pytest.approx(label.get_position())
        assert client.post(f"/api/sessions/{session.id}/commands", json={"revision": 0, "operation": "undo"}).status_code == 409
        good(client.post(f"/api/sessions/{session.id}/commands", json={"revision": 1, "operation": "undo"}))
        assert label.get_position() == pytest.approx((0.2, 0.3))
        assert b"<svg" in client.get(f"/api/sessions/{session.id}/render.svg").content


def test_example_is_real_data_bound_figure(client):
    project = good(client.post("/api/projects", json={"name": "Example", "example": True}))
    assert len(project["datasets"]) == 1
    assert any(element["type"] == "plot" for element in project["document"]["elements"])
    result = client.post(f"/api/projects/{project['id']}/export", json={"format": "png"})
    assert result.status_code == 200, result.text
    assert result.content.startswith(b"\x89PNG")
