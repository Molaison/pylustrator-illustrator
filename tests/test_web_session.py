"""Native scientific semantics remain usable with no QApplication or display."""

from matplotlib.figure import Figure
import numpy as np
import pytest

from pylustrator.web.document import DocumentError
from pylustrator.web.session import FigureSession, SessionConflict


@pytest.fixture
def session():
    figure = Figure(figsize=(6, 4), dpi=100)
    ax = figure.add_subplot()
    line, = ax.plot([0, 1, 2], [1, 3, 2], label="observations")
    text = figure.text(.2, .9, "Before")
    native = FigureSession(figure)
    yield native, figure, ax, line, text
    native.close()


def artist_id(session, target):
    return next(key for key, artist in session._artists.items() if artist is target)


def test_native_move_undo_redo_and_noop(session):
    native, _, _, _, text = session
    target = artist_id(native, text)
    state = native.command({"revision": 0, "operation": "translate", "targets": [target], "dx": 12, "dy": 20})
    assert state["revision"] == 1 and state["canUndo"]
    assert text.get_position() == pytest.approx((.22, .85))
    assert native.undo(1)["revision"] == 2
    assert text.get_position() == pytest.approx((.2, .9))
    assert native.redo(2)["revision"] == 3
    assert text.get_position() == pytest.approx((.22, .85))
    state = native.command({"revision": 3, "operation": "translate", "targets": [target], "dx": 0, "dy": 0})
    assert state["revision"] == 3


def test_revision_failure_and_locked_artist_never_mutate(session):
    native, _, _, _, text = session
    target = artist_id(native, text)
    with pytest.raises(SessionConflict):
        native.command({"revision": 50, "operation": "translate", "targets": [target], "dx": 20})
    with pytest.raises(DocumentError, match="revision"):
        native.command({"operation": "translate", "targets": [target], "dx": 20})
    native.command({"revision": 0, "operation": "lock", "targets": [target], "locked": True})
    with pytest.raises(DocumentError, match="Unlock"):
        native.command({"revision": 1, "operation": "translate", "targets": [target], "dx": 20})
    assert text.get_position() == (.2, .9)


def test_native_direct_endpoint_edit_preserves_interior_and_raw_dtype(session):
    native, _, _, line, _ = session
    target = artist_id(native, line)
    info = next(a for a in native.describe()["artists"] if a["id"] == target)
    point = info["points"][0]
    point["x"] += 10
    before = line.get_xdata().copy()
    native.command({"revision": 0, "operation": "points", "targets": [target], "points": [point]})
    assert line.get_xdata()[0] != before[0]
    np.testing.assert_array_equal(line.get_xdata()[1:], before[1:])
    native.undo(1)
    np.testing.assert_array_equal(line.get_xdata(), before)
    assert line.get_xdata().dtype == before.dtype


def test_native_semantic_text_and_tick_history(session):
    native, _, axes, _, text = session
    target = artist_id(native, text)
    native.command({"revision": 0, "operation": "text", "targets": [target], "text": "After"})
    assert text.get_text() == "After"
    native.undo(1)
    assert text.get_text() == "Before"
    ticks = axes.get_xticklabels()
    tick = ticks[1]
    old = tick.get_text()
    target = artist_id(native, tick)
    native.command({"revision": 2, "operation": "text", "targets": [target], "text": "Control"})
    native.figure.canvas.draw()
    assert axes.get_xticklabels()[1].get_text() == "Control"
    native.undo(3)
    assert axes.get_xticklabels()[1].get_text() == old


def test_failed_mixed_style_is_atomic(session):
    native, _, axes, line, text = session
    old = line.get_linewidth()
    with pytest.raises(ValueError):
        native.command({"revision": 0, "operation": "style", "targets": [artist_id(native, line), artist_id(native, text)],
                        "properties": {"linewidth": 9}})
    assert line.get_linewidth() == old
    assert native.revision == 0
    with pytest.raises(DocumentError, match="container"):
        native.command({"revision": 0, "operation": "translate", "targets": [artist_id(native, axes), artist_id(native, line)], "dx": 10})


def test_render_and_close_restore_canvas(session):
    native, figure, _, _, _ = session
    old = native._old_canvas
    assert b"<svg" in native.render("svg")
    assert native.render("png").startswith(b"\x89PNG")
    assert native.render("pdf").startswith(b"%PDF")
    native.close()
    assert figure.canvas is old
    with pytest.raises(DocumentError, match="closed"):
        native.describe()


def test_existing_tracker_and_identity_survive_native_commands(session):
    native, figure, axes, line, text = session
    previous_tracker = object()
    figure.change_tracker = previous_tracker
    figure._variable_name = "original_figure"
    native.command({"revision": 0, "operation": "text", "targets": [artist_id(native, text)], "text": "Changed"})
    assert figure.change_tracker is previous_tracker
    assert figure._variable_name == "original_figure"
    with pytest.raises(ValueError):
        native.command({"revision": 1, "operation": "style", "targets": [artist_id(native, line)], "properties": {"fontsize": 17}})
    assert figure.change_tracker is previous_tracker
    legend = axes.legend()
    native.describe()
    before = legend.get_bbox_to_anchor().bounds
    native.command({"revision": 1, "operation": "translate", "targets": [artist_id(native, legend)], "dx": 15, "dy": 10})
    native.undo(2)
    assert legend.get_bbox_to_anchor().bounds == pytest.approx(before)
    assert figure.change_tracker is previous_tracker


@pytest.mark.parametrize("payload", [
    {"revision": 0, "operation": []},
    {"revision": 0, "operation": "translate", "targets": [{}]},
    {"revision": True, "operation": "translate", "targets": []},
])
def test_malformed_native_request_is_a_validation_error(session, payload):
    native, _, _, _, text = session
    with pytest.raises(DocumentError):
        native.command(payload)
    assert text.get_position() == (.2, .9)
    assert native.revision == 0


def test_source_snippet_replays_native_moves_endpoints_and_active_undo(session):
    native, _, _, line, text = session
    tid, lid = artist_id(native, text), artist_id(native, line)
    native.command({"revision": 0, "operation": "translate", "targets": [tid], "dx": 19, "dy": 7})
    point = next(a for a in native.describe()["artists"] if a["id"] == lid)["points"][0]
    point["x"] += 13
    point["y"] -= 8
    native.command({"revision": 1, "operation": "points", "targets": [lid], "points": [point]})
    native.command({"revision": 2, "operation": "text", "targets": [tid], "text": "Undone edit"})
    native.undo(3)
    code = native.replay_source()
    assert "Undone edit" not in code
    replay = Figure(figsize=(6, 4), dpi=100)
    ax = replay.add_subplot()
    replay_line, = ax.plot([0, 1, 2], [1, 3, 2], label="observations")
    replay_text = replay.text(.2, .9, "Before")
    original_canvas = replay.canvas
    exec(compile(code, "native_changes.py", "exec"), {"fig": replay})
    assert replay.canvas is original_canvas
    assert replay_text.get_position() == pytest.approx(text.get_position())
    assert replay_text.get_text() == "Before"
    np.testing.assert_allclose(replay_line.get_xdata(), line.get_xdata())
    np.testing.assert_allclose(replay_line.get_ydata(), line.get_ydata())
    native.redo(4)
    assert "Undone edit" in native.replay_source()


def test_source_snippet_requires_original_pixel_geometry(session):
    native, _, _, _, text = session
    native.command({"revision": 0, "operation": "translate", "targets": [artist_id(native, text)], "dx": 10})
    with pytest.raises(ValueError, match="original figure size"):
        exec(compile(native.replay_source(), "changes.py", "exec"), {"fig": Figure(figsize=(4, 4), dpi=72)})
