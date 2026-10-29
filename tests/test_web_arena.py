"""Arena outcomes against real FigureSession geometry, pixels and HTTP routes."""

import base64
import json
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO

import matplotlib as mpl
import numpy as np
import pytest
from matplotlib.figure import Figure
from PIL import Image

from pylustrator.web.arena import ArenaError, FigureArena, install_arena
from pylustrator.web.arena_demo import example_figure, run_demo
from pylustrator.web.session import FigureSession


@pytest.fixture
def arena():
    session = FigureSession(example_figure())
    yield FigureArena(session)
    session.close()


def panel(observation, index=1):
    return [e for e in observation["objects"] if e["basis"] == "plot_area"][index]


def move(arena, observation=None, **changes):
    observation = observation or arena.observe()
    return {"operation": "move", "requestId": "move-1", "revision": observation["revision"],
            "target": panel(observation)["id"], "dx": -30, "dy": -20, **changes}


def test_http_example_produces_real_before_after_and_replay(tmp_path):
    pytest.importorskip("gemmi")
    from pylustrator.web.app import create_app

    session = FigureSession(example_figure())
    try:
        app = install_arena(create_app(tmp_path / "store", sessions=[session]))
        assert install_arena(app) is app
        checks = run_demo(app, session, tmp_path / "evidence")
        assert checks["alignedBounds"][1:] == pytest.approx([140, 320, 268.8])
        before = np.asarray(Image.open(tmp_path / "evidence/00-before.png"))
        after = np.asarray(Image.open(tmp_path / "evidence/03-align.png"))
        assert before.shape == after.shape == (560, 1000, 4)
        assert np.count_nonzero(before != after) > 10000
        assert (tmp_path / "evidence/03-align.png").read_bytes() == (tmp_path / "evidence/05-redo.png").read_bytes()
        replay = example_figure()
        code = (tmp_path / "evidence/native-changes.py").read_text()
        exec(compile(code, "native-changes.py", "exec"), {"fig": replay})  # noqa: S102 - trusted fixture's generated replay
        for original, actual in zip(session.figure.axes, replay.axes):
            assert actual.get_position().bounds == pytest.approx(original.get_position().bounds)
            np.testing.assert_array_equal(actual.lines[0].get_ydata(), original.lines[0].get_ydata())
    finally:
        session.close()


def test_observation_geometry_uses_full_canvas_even_with_tight_export_rc(arena):
    with mpl.rc_context({"savefig.bbox": "tight", "savefig.pad_inches": .5}):
        observed = arena.observe()
    image = Image.open(BytesIO(base64.b64decode(observed["preview"]["data"])))
    assert image.size == (observed["canvas"]["width"], observed["canvas"]["height"]) == (1000, 560)
    assert panel(observed, 0)["bounds"] == pytest.approx([100, 140, 320, 268.8])


def test_duplicate_request_returns_original_receipt_and_current_state(arena):
    start = arena.observe()
    request = move(arena, start)
    applied = arena.act(request)
    later = arena.act(move(arena, applied["observation"], requestId="move-2", dx=-10))
    retry = arena.act(json.loads(json.dumps(request)))
    assert retry["replayed"] and retry["receipt"] == applied["receipt"]
    assert retry["observation"] == later["observation"]
    with pytest.raises(ArenaError, match="different payload") as exc:
        arena.act({**request, "dx": -31})
    assert exc.value.status == 409 and arena.session.revision == 2


@pytest.mark.parametrize("changes", [
    {"dx": float("nan")}, {"dy": float("inf")}, {"dx": True}, {"dy": "12"},
    {"dx": 10**310}, {"dx": 1_000_001}, {"dx": 900},
    {"revision": True}, {"revision": 0.0}, {"revision": -1},
    {"target": []}, {"requestId": ""}, {"operation": "points"}, {"dxx": 1},
    {"operation": "scale", "factorX": 0},
])
def test_invalid_input_never_mutates_geometry_or_history(arena, changes):
    before = arena.observe()
    with pytest.raises(ArenaError):
        arena.act(move(arena, before, **changes))
    after = arena.observe()
    assert after == before


def test_stale_revision_and_removed_or_cross_session_target_are_rejected(arena):
    before = arena.observe()
    arena.act(move(arena, before))
    with pytest.raises(ArenaError) as stale:
        arena.act(move(arena, before, requestId="stale"))
    assert stale.value.code == "revision_conflict"
    victim = arena.session.figure.axes[1]
    victim.remove()
    # Even a new object with identical label and bounds cannot inherit an ID.
    arena.session.figure.add_axes([.6, .18, .24, .32], label="Panel B · treated")
    current = arena.observe()
    assert panel(current)["id"] != panel(before)["id"]
    with pytest.raises(ArenaError) as missing:
        arena.act(move(arena, before, requestId="removed", revision=current["revision"]))
    assert missing.value.code == "stale_object"
    other = FigureSession(example_figure())
    try:
        with pytest.raises(ArenaError) as wrong:
            FigureArena(other).act(move(arena, before, requestId="wrong-figure"))
        assert wrong.value.code == "stale_object"
    finally:
        other.close()


def test_simultaneous_agents_cannot_both_apply_the_same_revision(arena):
    before = arena.observe()

    def attempt(request):
        try:
            return arena.act(request)["receipt"]["appliedRevision"]
        except ArenaError as exc:
            return exc.code

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(attempt, [move(arena, before, requestId="one"), move(arena, before, requestId="two")]))
    assert sorted(map(str, outcomes)) == ["1", "revision_conflict"]
    np.testing.assert_allclose(np.array(panel(arena.observe())["bounds"]) - panel(before)["bounds"], [-30, -20, 0, 0], atol=1e-8)


def test_lost_render_after_commit_has_readable_receipt_and_safe_retry(arena, monkeypatch):
    before = arena.observe()
    request = move(arena, before)
    real_render = arena.session.render
    monkeypatch.setattr(arena.session, "render", lambda *args: (_ for _ in ()).throw(RuntimeError("render unavailable")))
    with pytest.raises(ArenaError) as failure:
        arena.act(request)
    assert failure.value.status == 503 and failure.value.receipt["appliedRevision"] == 1
    assert arena.receipt(request["requestId"]) == failure.value.receipt
    monkeypatch.setattr(arena.session, "render", real_render)
    retry = arena.act(request)
    assert retry["replayed"] and retry["observation"]["revision"] == 1
    np.testing.assert_allclose(np.array(panel(retry["observation"])["bounds"]) - panel(before)["bounds"], [-30, -20, 0, 0], atol=1e-8)


def test_session_postcommit_failure_does_not_repeat_move(arena, monkeypatch):
    request = move(arena)
    real_command = arena.session.command

    def failing_command(payload):
        real_command(payload)
        raise RuntimeError("native post-commit drawing failed")

    monkeypatch.setattr(arena.session, "command", failing_command)
    with pytest.raises(ArenaError) as failure:
        arena.act(request)
    assert failure.value.receipt["changed"] and arena.session.revision == 1
    assert arena.act(request)["replayed"]
    assert arena.session.revision == 1


def test_noop_does_not_destroy_redo_and_new_edit_does(arena):
    start = arena.observe()
    arena.act(move(arena, start))
    undone = arena.act({"requestId": "undo-1", "revision": 1, "operation": "undo"})["observation"]
    noop = arena.act(move(arena, undone, requestId="noop", dx=0, dy=0))
    assert not noop["receipt"]["changed"] and noop["observation"]["history"]["canRedo"]
    new = arena.act(move(arena, undone, requestId="new", dx=-4))
    assert not new["observation"]["history"]["canRedo"]
    assert panel(new["observation"])["id"] == panel(start)["id"]


def test_external_native_edit_uses_same_revision_and_is_a_history_boundary(arena):
    state = arena.act(move(arena))["observation"]
    text = arena.session.figure.texts[0]
    native_id = arena.session.artist_id(text)
    arena.session.command({"operation": "text", "targets": [native_id], "revision": 1, "text": "Native edit"})
    with pytest.raises(ArenaError) as conflict:
        arena.act(move(arena, state, requestId="stale-native"))
    assert conflict.value.code == "revision_conflict"
    current = arena.observe()
    assert not current["history"]["canUndo"]
    with pytest.raises(ArenaError) as boundary:
        arena.act({"operation": "undo", "revision": 2, "requestId": "undo-native"})
    assert boundary.value.code == "history_boundary" and text.get_text() == "Native edit"
    arena.act(move(arena, current, requestId="after-native"))
    arena.act({"operation": "undo", "revision": 3, "requestId": "undo-own"})
    assert text.get_text() == "Native edit"


def test_scale_semantics_and_locked_or_coupled_panels_fail_closed(arena):
    before = arena.observe()
    target = panel(before)
    request = {"requestId": "scale", "revision": 0, "operation": "scale", "target": target["id"],
               "factorX": 1.2, "factorY": 1.2, "basis": "plot_area"}
    for changes in [{"basis": "whole_panel"}, {"factorX": -1}, {"factorY": True}, {"anchor": "unknown"}, {"factorX": 20}]:
        with pytest.raises(ArenaError):
            arena.act({**request, **changes})
        assert arena.session.revision == 0
    arena.session.command({"revision": 0, "operation": "lock", "targets": [target["nativeId"]], "locked": True})
    with pytest.raises(ArenaError, match="Unlock"):
        arena.act({**request, "revision": 1})
    assert not panel(arena.observe())["capabilities"]["scale"]
    arena.session.figure.axes[0].twinx()
    assert "Twinned" in panel(arena.observe(), 0)["restriction"]


def test_data_artists_are_not_targets_and_fig_text_keeps_font_size(arena):
    start = arena.observe()
    assert {e["basis"] for e in start["objects"]} == {"plot_area", "text"}
    line = arena.session.figure.axes[0].lines[0]
    with pytest.raises(ArenaError) as error:
        arena.act(move(arena, start, target=arena.session.artist_id(line)))
    assert error.value.code == "stale_object"
    text = next(e for e in start["objects"] if e["basis"] == "text")
    size = arena.session.figure.texts[0].get_fontsize()
    after = arena.act(move(arena, start, target=text["id"], dx=8, dy=3))["observation"]
    assert arena.session.figure.texts[0].get_fontsize() == size
    assert next(e for e in after["objects"] if e["id"] == text["id"])["bounds"][0] == pytest.approx(text["bounds"][0] + 8)


def test_request_budget_never_evicts_receipts_for_noops(arena, monkeypatch):
    monkeypatch.setattr(arena, "MAX_REQUESTS", 1)
    request = move(arena, dx=0, dy=0)
    result = arena.act(request)
    with pytest.raises(ArenaError) as limit:
        arena.act({**request, "requestId": "second"})
    assert limit.value.code == "request_limit"
    assert arena.act(request)["replayed"]
    assert arena.receipt(request["requestId"]) == result["receipt"]


def test_http_auth_origin_and_existing_editor_routes_are_shared(tmp_path):
    pytest.importorskip("gemmi")
    from fastapi.testclient import TestClient

    from pylustrator.web.app import create_app

    session = FigureSession(Figure(figsize=(4, 3), dpi=100))
    session.figure.add_axes([.2, .2, .6, .6])
    try:
        app = install_arena(create_app(tmp_path, sessions=[session], token="test-token"))
        base = f"/api/arena/sessions/{session.id}"
        with TestClient(app) as client:
            assert client.get(base + "/observe").status_code == 401
            client.headers["Authorization"] = "Bearer test-token"
            observation = client.get(base + "/observe").json()
            request = {"requestId": "http-move", "revision": 0, "operation": "move",
                       "target": observation["objects"][0]["id"], "dx": 5, "dy": 0}
            assert client.post(base + "/actions", json=request, headers={"Origin": "https://other.invalid"}).status_code == 403
            result = client.post(base + "/actions", json=request)
            assert result.status_code == 200 and result.headers["cache-control"] == "no-store"
            receipt = client.get(base + "/requests/http-move").json()
            assert receipt["appliedRevision"] == 1
            native = client.get(f"/api/sessions/{session.id}").json()
            assert native["revision"] == 1
            assert native["artists"] and client.get("/").status_code == 200
            assert client.get("/api/arena/sessions/missing/observe").status_code == 404
            assert client.get(base + "/requests/missing").status_code == 404
            bad = client.post(base + "/actions", json={**request, "requestId": "stale"})
            assert bad.status_code == 409 and bad.json()["error"]["code"] == "revision_conflict"
    finally:
        session.close()
