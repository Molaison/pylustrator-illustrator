"""Reproducible HTTP caller and visible evidence for the native Figure Arena.

python -m pylustrator.web.arena_demo --output /tmp/arena-demo [--serve]
No model, API key, Qt application, network provider or scientific computation.
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import tempfile
from html import escape
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure

from .arena import install_arena
from .session import FigureSession


def example_figure():
    """Load illustrative CSV values into two real, independently editable panels."""
    source = Path(__file__).with_name("examples") / "arena_observations.csv"
    with source.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    values = {key: np.array([float(row[key]) for row in rows]) for key in rows[0]}
    figure = Figure(figsize=(10, 5.6), dpi=100)
    first = figure.add_axes([.10, .27, .32, .48], label="Panel A · control")
    second = figure.add_axes([.60, .18, .24, .32], label="Panel B · treated")
    for ax, column, color, title in [
        (first, "control", "#2874a6", "A   Control"),
        (second, "treated", "#bc6344", "B   Treated"),
    ]:
        ax.plot(values["time"], values[column], "o-", color=color, label=column, linewidth=2)
        ax.set(xlabel="Time (min)", ylabel="Response (a.u.)", xlim=(-.3, 6.3), ylim=(0, 1.05))
        ax.set_title(title, loc="left", fontsize=13, pad=12)
        ax.legend(loc="lower right", fontsize=9, frameon=False)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=.15)
    figure.text(.08, .94, "Figure Arena  /  illustrative observations", fontsize=18, color="#213547")
    figure.text(.10, .055, "Source values, axis limits and font sizes stay unchanged. Bounds refer to the plot area.",
                fontsize=10, color="#536475")
    return figure


def _good(response):
    if response.status_code != 200:
        raise RuntimeError(f"HTTP {response.status_code}: {response.text}")
    return response.json()


def _bounds(observation, key):
    return next(entry["bounds"] for entry in observation["objects"] if entry["id"] == key)


def run_demo(app, session, output: Path):
    """Exercise the actual HTTP routes; write images, receipts and a local report."""
    from fastapi.testclient import TestClient

    output.mkdir(parents=True, exist_ok=True)
    axes = session.figure.axes
    original = [(line, line.get_xdata().copy(), line.get_ydata().copy())
                for ax in axes for line in ax.lines]
    limits = [(ax.get_xlim(), ax.get_ylim()) for ax in axes]
    fonts = [(text, text.get_fontsize()) for text in session.figure.findobj()
             if hasattr(text, "get_fontsize")]
    stages, receipts = [], []

    def save(name, observation):
        (output / f"{name}.png").write_bytes(base64.b64decode(observation["preview"]["data"]))
        (output / f"{name}.json").write_text(json.dumps(observation, indent=2, allow_nan=False) + "\n")
        stages.append((name, observation))
        return observation

    base = f"/api/arena/sessions/{session.id}"
    with TestClient(app) as client:
        before = save("00-before", _good(client.get(base + "/observe")))
        a, b = [entry["id"] for entry in before["objects"] if entry["basis"] == "plot_area"]
        state = before

        def act(name, **fields):
            nonlocal state
            request = {"requestId": name, "revision": state["revision"], **fields}
            result = _good(client.post(base + "/actions", json=request))
            receipts.append({"request": request, "receipt": result["receipt"]})
            state = save(name, result["observation"])
            return request

        move = act("01-move", operation="move", target=b, dx=-45, dy=-40)
        np.testing.assert_allclose(np.array(_bounds(state, b)) - _bounds(before, b), [-45, -40, 0, 0], atol=1e-8)
        moved = state
        act("02-scale", operation="scale", target=b, factorX=4 / 3, factorY=1.5, basis="plot_area")
        np.testing.assert_allclose(_bounds(state, b)[2:], _bounds(before, a)[2:])
        scaled = state
        act("03-align", operation="align", target=b, reference=a, edge="top", basis="plot_area")
        aligned = state
        np.testing.assert_allclose(_bounds(state, b)[1], _bounds(state, a)[1])
        # A delayed retry returns the original receipt plus CURRENT pixels/state.
        retry = _good(client.post(base + "/actions", json=move))
        assert retry["replayed"] and retry["receipt"]["appliedRevision"] == moved["revision"]
        assert retry["observation"]["revision"] == aligned["revision"]
        act("04-undo", operation="undo")
        np.testing.assert_allclose(_bounds(state, b), _bounds(scaled, b))
        act("05-redo", operation="redo")
        np.testing.assert_allclose(_bounds(state, b), _bounds(aligned, b))
        observed = _good(client.get(base + "/observe"))
        assert observed["preview"]["data"] == state["preview"]["data"]
        assert [e["id"] for e in observed["objects"]] == [e["id"] for e in before["objects"]]

    for line, x, y in original:
        np.testing.assert_array_equal(line.get_xdata(), x)
        np.testing.assert_array_equal(line.get_ydata(), y)
    assert [(ax.get_xlim(), ax.get_ylim()) for ax in axes] == limits
    assert all(text.get_fontsize() == size for text, size in fonts)
    assert stages[0][1]["preview"]["data"] != aligned["preview"]["data"]
    assert stages[3][1]["preview"]["data"] == stages[5][1]["preview"]["data"]
    (output / "requests.json").write_text(json.dumps(receipts, indent=2) + "\n")
    (output / "native-changes.py").write_text(session.replay_source())
    checks = {
        "source": "Bundled illustrative CSV; no research data modified",
        "panelA": a, "panelB": b, "beforeBounds": _bounds(before, b),
        "movedBounds": _bounds(moved, b), "scaledBounds": _bounds(scaled, b),
        "alignedBounds": _bounds(aligned, b), "finalRevision": state["revision"],
        "sourceValuesUnchanged": True, "axisLimitsUnchanged": True, "fontSizesUnchanged": True,
        "objectIdsStable": True, "retryAppliedOnce": True, "undoRestoresScaledGeometry": True,
        "redoRestoresActualPixels": True,
    }
    (output / "checks.json").write_text(json.dumps(checks, indent=2) + "\n")
    cards = []
    for name, observation in stages:
        rows = "".join(f"<tr><td>{escape(e['name'])}</td><td><code>{escape(e['id'])}</code></td>"
                       f"<td>{', '.join(f'{v:.1f}' for v in e['bounds'])}</td></tr>"
                       for e in observation["objects"] if e["basis"] == "plot_area")
        cards.append(f"<section><h2>{escape(name)} · revision {observation['revision']}</h2>"
                     f"<img alt='{escape(name)}: actual rendered figure' src='data:image/png;base64,{observation['preview']['data']}'>"
                     f"<table><thead><tr><th>Panel</th><th>Stable ID</th><th>x, y, width, height (px)</th></tr></thead><tbody>{rows}</tbody></table></section>")
    report = """<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Figure Arena · observed results</title><style>
body{margin:0;background:#eef1f3;color:#213547;font:16px system-ui,sans-serif}main{max-width:1100px;margin:auto;padding:32px}
h1{font-size:32px}h2{font-size:20px}p{line-height:1.6}section{background:white;padding:24px;margin:24px 0;border-radius:10px}
img{width:100%;height:auto}table{width:100%;font-size:12px;border-collapse:collapse}td,th{padding:8px;border-bottom:1px solid #ddd;text-align:left}code{overflow-wrap:anywhere}
</style><main><h1>Observe → act → observe</h1><p>Real HTTP actions on the existing FigureSession, followed by its PNG renderer.
Panel B moves 45 px left and 40 px up, matches Panel A’s plot-area size, and aligns to its top edge.
Undo restores the previous geometry; redo restores the aligned pixels. The CSV values, axis limits and font sizes are unchanged.</p>
<p>Coordinates use the preview’s top-left origin. Bounds are <strong>plot areas</strong>, not detected content or label extents.
Source and verification records: <code>requests.json</code>, <code>checks.json</code>, per-stage JSON/PNG, <code>native-changes.py</code>.</p>
""" + "\n".join(cards) + "</main></html>\n"
    (output / "index.html").write_text(report)
    return checks


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("arena-demo-output"))
    parser.add_argument("--serve", action="store_true", help="After the demo, open its live Figure in the existing editor on loopback.")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args(argv)
    from .app import create_app

    session = FigureSession(example_figure(), name="Arena demonstration")
    try:
        with tempfile.TemporaryDirectory(prefix="figure-arena-") as root:
            app = install_arena(create_app(root, sessions=[session]))
            checks = run_demo(app, session, args.output)
            print(json.dumps(checks, indent=2))
            print(f"Visible report: {args.output.resolve() / 'index.html'}")
            if args.serve:
                import uvicorn

                print(f"Native editor: http://127.0.0.1:{args.port}  |  session: {session.id}")
                uvicorn.run(app, host="127.0.0.1", port=args.port, proxy_headers=False)
    finally:
        session.close()


if __name__ == "__main__":
    main()
