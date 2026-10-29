"""Exercise the real editor and record before/after images from Chromium.

python scripts/test_content_regions_browser.py --output /tmp/content-regions-evidence
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

def run(output):
    from playwright.sync_api import expect, sync_playwright
    from scripts.demo_content_regions import build_demo

    output.mkdir(parents=True, exist_ok=True)
    project, truth = build_demo(output / "demo")
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    url = f"http://127.0.0.1:{port}"
    env = {**os.environ, "MPLCONFIGDIR": str(output / "mpl-cache")}
    with (output / "server.log").open("w") as log:
        process = subprocess.Popen([sys.executable, "-m", "pylustrator.web", "--data-dir", str(output / "demo/workspace"),
                                    "--port", str(port)], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        try:
            for _ in range(150):
                if process.poll() is not None:
                    raise RuntimeError("Demo server failed; see server.log")
                try:
                    urllib.request.urlopen(url + "/api/health", timeout=1).close()
                    break
                except (urllib.error.URLError, TimeoutError):
                    time.sleep(.1)
            with sync_playwright() as playwright:
                executable = os.environ.get("CHROMIUM_PATH")
                if not executable and Path("/usr/bin/chromium").exists():
                    executable = "/usr/bin/chromium"
                browser = playwright.chromium.launch(headless=True, executable_path=executable, args=["--no-sandbox"])
                context = browser.new_context(viewport={"width": 1600, "height": 1200}, device_scale_factor=1)
                page = context.new_page()
                errors = []
                page.on("pageerror", lambda error: errors.append(str(error)))
                page.goto(url)
                page.locator("#project-select").select_option(project["id"])
                expect(page.locator("#layer-count")).to_have_text("2")

                def select_both():
                    page.locator('.layer-select[title^="A ·"]').click()
                    page.locator('.layer-select[title^="B ·"]').click(modifiers=["Shift"])

                def persisted():
                    page.locator("#save-project").click()
                    expect(page.locator("#save-state")).to_have_text("Saved")
                    return context.request.get(url + f"/api/projects/{project['id']}").json()["document"]

                def panel_x():
                    return page.locator('#artwork > [data-element="panel-b"]').get_attribute("transform")

                select_both()
                page.locator("#region-detect").click()
                expect(page.locator("#region-status")).to_have_text("Suggestion · review outline")
                # Unreviewed raster guesses cannot silently drive a transform.
                initial_transform = panel_x()
                page.locator('[data-region-align="left"]').click()
                expect(page.locator("#notice-text")).to_contain_text("confirm")
                assert panel_x() == initial_transform
                page.locator("#region-confirm").click()
                page.locator("#region-panel").select_option("panel-b")
                page.locator("#region-confirm").click()
                expect(page.locator("#region-status")).to_have_text("Confirmed bounds")
                before = persisted()
                page.locator("#zoom-fit").click()
                page.locator(".inspector-panel").evaluate("el => el.scrollTop = 0")
                assert page.locator(".inspector-panel").evaluate("el => el.scrollWidth <= el.clientWidth")
                page.screenshot(path=str(output / "before.png"), full_page=True)
                page.locator("#artboard").screenshot(path=str(output / "before-figure.png"))
                page.locator('[data-region-align="left"]').click()
                aligned_transform = panel_x()
                assert aligned_transform != initial_transform
                after = persisted()

                def lefts(document, real):
                    return [e["x"] + e["width"] * (truth[i]["plotBounds"]["x"] if real else e["contentRegions"]["plot"]["bounds"]["x"])
                            for i, e in enumerate(document["elements"])]

                before_lefts, after_lefts = lefts(before, True), lefts(after, True)
                assert abs(before_lefts[1] - before_lefts[0]) > 100
                assert abs(after_lefts[1] - after_lefts[0]) <= 1
                measured_lefts = lefts(after, False)
                assert abs(measured_lefts[1] - measured_lefts[0]) < 1e-8
                for old, new in zip(before["elements"], after["elements"]):
                    assert {**new, "x": old["x"]} == old
                page.locator(".inspector-panel").evaluate("el => el.scrollTop = 0")
                page.screenshot(path=str(output / "after.png"), full_page=True)
                page.locator("#artboard").screenshot(path=str(output / "after-figure.png"))
                page.locator("#undo").click()
                assert panel_x() == initial_transform
                page.locator("#redo").click()
                assert panel_x() == aligned_transform

                # Manual numeric correction previews independently and commits once.
                original_x = float(page.locator('#region-form [name="x"]').input_value())
                page.locator('#region-form [name="x"]').fill(str(original_x + 1))
                page.screenshot(path=str(output / "manual-preview.png"), full_page=True)
                page.get_by_role("button", name="Apply correction", exact=True).click()
                expect(page.locator("#region-status")).to_have_text("Manual bounds · confirmed")
                page.locator("#undo").click()
                assert abs(float(page.locator('#region-form [name="x"]').input_value()) - original_x) < 1e-5
                page.locator("#redo").click()
                expect(page.locator("#region-status")).to_have_text("Manual bounds · confirmed")
                page.locator("#undo").click()
                # Export the actual aligned composition, including complete images.
                response = context.request.post(url + f"/api/projects/{project['id']}/export", data={"format": "png"})
                assert response.ok
                (output / "aligned-export.png").write_bytes(response.body())
                for i, element in enumerate(after["elements"]):
                    data = context.request.get(url + f"/api/projects/{project['id']}/assets/{element['assetId']}").body()
                    assert data == (output / "demo" / truth[i]["file"]).read_bytes()
                bundle = context.request.get(url + f"/api/projects/{project['id']}/bundle").json()
                (output / "aligned-project.json").write_text(json.dumps(bundle, indent=2) + "\n")
                page.reload()
                expect(page.locator("#layer-count")).to_have_text("2")
                select_both()
                page.locator("#region-panel").select_option("panel-b")
                expect(page.locator("#region-status")).to_have_text("Confirmed bounds")
                assert panel_x() == aligned_transform
                # Project/selection changes during detection must discard late results.
                def delayed_selection_change(route):
                    response = route.fetch()
                    page.locator('.layer-select[title^="A ·"]').click()
                    route.fulfill(response=response)

                page.route("**/content-regions/detect", delayed_selection_change)
                page.locator("#region-detect").click()
                expect(page.locator("#notice-text")).to_contain_text("changed")
                expect(page.locator("#region-status")).to_have_text("Confirmed bounds")
                page.unroute("**/content-regions/detect")
                select_both()
                page.locator("#region-panel").select_option("panel-b")
                # Narrow viewport: controls remain reachable in the existing drawer.
                page.set_viewport_size({"width": 390, "height": 844})
                page.locator("#toggle-properties").click()
                expect(page.locator("#region-kind")).to_be_visible()
                page.locator('#region-form [name="x"]').fill(str(original_x + .5))
                page.get_by_role("button", name="Apply correction", exact=True).click()
                expect(page.locator("#region-status")).to_have_text("Manual bounds · confirmed")
                assert page.locator(".inspector-panel").evaluate("el => el.scrollWidth <= el.clientWidth")
                page.screenshot(path=str(output / "narrow.png"), full_page=True)
                # Bound plots use the renderer's exact Axes, and resizing forces
                # re-detection instead of reusing obsolete pixel-based margins.
                page.set_viewport_size({"width": 1600, "height": 1200})
                page.locator("#new-project").click()
                page.locator('#new-project-form input[name="name"]').fill("Native geometry verification")
                page.locator('#new-project-form input[name="example"]').check()
                page.locator('#new-project-form button[type="submit"]').click()
                expect(page.locator("#layer-count")).to_have_text("4")
                page.locator('.layer-select[title^="Response over time"]').click()
                page.locator("#region-detect").click()
                expect(page.locator("#region-status")).to_have_text("Confirmed bounds")
                expected_x = 58 / 790 * 100
                assert abs(float(page.locator('#region-form [name="x"]').input_value()) - expected_x) < 1e-5
                page.locator('[data-prop="width"]').fill("700")
                page.locator('[data-prop="width"]').press("Tab")
                expect(page.locator("#region-status")).to_have_text("Detection needed")
                page.locator("#undo").click()
                expect(page.locator("#region-status")).to_have_text("Confirmed bounds")
                assert not errors, errors
                report = {"browser": browser.version, "source": "scripts/demo_content_regions.py", "beforeTruePlotLeftPx": before_lefts,
                          "coordinateSpace": "artboard", "units": "document-px", "viewportZoomApplied": False,
                          "afterTruePlotLeftPx": after_lefts, "beforeGapPx": abs(before_lefts[1] - before_lefts[0]),
                          "afterGapPx": abs(after_lefts[1] - after_lefts[0]), "originalPngBytesPreserved": True,
                          "undoRedoAlignment": True, "undoRedoCorrection": True, "savedReload": True, "narrowCorrection": True,
                          "lateDetectionDiscarded": True, "controlsFitDrawer": True, "nativePlotDetection": True,
                          "resizedPlotInvalidation": True, "pageErrors": errors}
                (output / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
                print(json.dumps(report, indent=2))
                browser.close()
        finally:
            process.terminate()
            process.wait(timeout=15)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("/tmp/content-regions-evidence"))
    run(parser.parse_args().output.resolve())
