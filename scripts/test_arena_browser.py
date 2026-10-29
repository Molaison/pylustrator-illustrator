"""Verify Arena edits in the existing browser editor of arena_demo --serve.

Only the session named 'Arena demonstration' is accepted. Install [browser-test]
and Chromium first, or pass --chromium /path/to/an/existing/chromium.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from uuid import uuid4


def run(url, output, chromium=None):
    import httpx
    from playwright.sync_api import expect, sync_playwright

    output.mkdir(parents=True, exist_ok=True)
    with httpx.Client(base_url=url, timeout=30) as client:
        sessions = client.get("/api/sessions").json()["sessions"]
        demos = [s for s in sessions if s["name"] == "Arena demonstration"]
        if len(demos) != 1:
            raise ValueError("Start exactly one arena_demo --serve; this script edits only that illustrative session.")
        session_id = demos[0]["id"]
        base = f"/api/arena/sessions/{session_id}"
        observation = client.get(base + "/observe").json()
        receipts = []

        def action(**fields):
            nonlocal observation
            request = {"requestId": uuid4().hex, "revision": observation["revision"], **fields}
            response = client.post(base + "/actions", json=request)
            response.raise_for_status()
            result = response.json()
            observation = result["observation"]
            receipts.append({"request": request, "receipt": result["receipt"]})

        # The demo has already run three geometry actions. Restore its input.
        for _ in range(3):
            action(operation="undo")
        panels = [e for e in observation["objects"] if e["basis"] == "plot_area"]
        a, b = panels
        assert all(abs(actual - expected) < 1e-8 for actual, expected in zip(b["bounds"], [600, 280, 240, 179.2]))
        with sync_playwright() as playwright:
            options = {"headless": True, "args": ["--no-sandbox"]}
            if chromium:
                options["executable_path"] = chromium
            browser = playwright.chromium.launch(**options)
            try:
                page = browser.new_page(viewport={"width": 1600, "height": 1000}, device_scale_factor=1)
                failures = []
                page.on("pageerror", lambda error: failures.append(str(error)))
                page.goto(url)
                page.locator('[data-view="native"]').click()
                expect(page.locator("#native-session")).to_have_value(session_id)
                expect(page.locator("#native-status")).to_contain_text("Revision")

                def capture(name):
                    page.locator("#native-refresh").click()
                    expect(page.locator("#native-status")).to_contain_text(f"Revision {observation['revision']} ·")
                    expect(page.locator("#native-image")).to_have_attribute(
                        "src", f"/api/sessions/{session_id}/render.svg?revision={observation['revision']}"
                    )
                    page.locator("#native-image").evaluate("img => img.decode()")
                    page.screenshot(path=str(output / f"{name}.png"), full_page=True)
                    native = client.get(f"/api/sessions/{session_id}").json()
                    target = next(e for e in native["artists"] if e["id"] == b["nativeId"])
                    observed = next(e for e in observation["objects"] if e["id"] == b["id"])
                    assert target["bounds"] == observed["bounds"]

                capture("editor-before")
                action(operation="move", target=b["id"], dx=-45, dy=-40)
                action(operation="scale", target=b["id"], factorX=4 / 3, factorY=1.5, basis="plot_area")
                action(operation="align", target=b["id"], reference=a["id"], edge="top", basis="plot_area")
                capture("editor-after")
                action(operation="undo")
                capture("editor-undo")
                action(operation="redo")
                capture("editor-redo")
                assert not failures, failures
            finally:
                browser.close()
        (output / "browser-actions.json").write_text(json.dumps(receipts, indent=2) + "\n")
        print(f"PASS: native editor renders Arena move/scale/align/undo/redo, no page errors; {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8765")
    parser.add_argument("--output", type=Path, default=Path("/tmp/arena-browser"))
    parser.add_argument("--chromium")
    args = parser.parse_args()
    run(args.url, args.output, args.chromium)


if __name__ == "__main__":
    main()
