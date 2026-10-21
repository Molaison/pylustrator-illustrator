"""Exercise a real scientific workflow in Chromium against an isolated server.

Install .[web,browser-test] and run `python -m playwright install chromium` first.
No external service, credentials, or experiment data are used. The bundled 1CRN
structure exercises actual WebGL rendering. Use --artifacts DIR for screenshots.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
SERVER = """
import sys
import uvicorn
from matplotlib.figure import Figure
from pylustrator.web.app import create_app
from pylustrator.web.session import FigureSession
fig = Figure(figsize=(6, 4), dpi=96)
ax = fig.add_subplot()
ax.plot([0, 1, 2], [2, 1, 3], label='Signal')
fig.text(.2, .9, 'Browser native text')
session = FigureSession(fig, name='Browser native figure')
other = Figure(figsize=(4, 3), dpi=96)
other.text(.3, .7, 'Second native figure')
app = create_app(sys.argv[1], sessions=[session, FigureSession(other, name='Second native figure')])
uvicorn.run(app, host='127.0.0.1', port=int(sys.argv[2]), log_level='warning')
"""


def wait_server(url, process):
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError("The test server exited before startup.")
        try:
            with urllib.request.urlopen(url + "/api/health", timeout=1) as response:
                if response.status == 200:
                    return
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            time.sleep(0.1)
    raise RuntimeError("The isolated Web test server did not become healthy.")


def run_browser(url: str, artifacts: Path):
    from playwright.sync_api import expect, sync_playwright

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True, args=["--no-sandbox", "--enable-unsafe-swiftshader"])
        context = browser.new_context(viewport={"width": 1440, "height": 1000}, accept_downloads=True)
        # Slow the first Figure detail response to reproduce an early user switch.
        quick = context.new_page()
        quick.add_init_script("""
          const originalFetch = window.fetch.bind(window);
          window.__listRequests = 0;
          window.__detailRequests = 0;
          window.fetch = async (...args) => {
            const path = String(args[0]);
            if (path === "/api/sessions") window.__listRequests++;
            if (path.startsWith("/api/sessions/") && path.split("/").length === 4) {
              window.__detailRequests++;
              if (window.__detailRequests === 1)
                await new Promise(resolve => setTimeout(resolve, 1500));
            }
            return originalFetch(...args);
          };
        """)
        quick.goto(url)
        select = quick.locator("#native-session")
        select.wait_for()
        quick.wait_for_function("window.__detailRequests > 0")
        selected_id = select.select_option(label="Second native figure")[0]
        expect(quick.locator("#native-status")).to_contain_text("Revision 0")
        expect(select).to_have_value(selected_id)
        quick.wait_for_timeout(1800)  # original delayed response has now settled
        expect(select).to_have_value(selected_id)
        expect(quick.get_by_role("button", name="Second native figure · Text", exact=True)).to_be_visible()
        quick.locator('[data-view="figure"]').click()
        quick.locator('[data-view="native"]').click()
        expect(select).to_have_value(selected_id)
        assert quick.evaluate("window.__listRequests") == 1
        quick.close()
        print("PASS lightweight discovery and latest-choice-wins during early Figure switching", flush=True)

        page = context.new_page()
        page.set_default_timeout(20000)
        failures = []
        page.on("pageerror", lambda exc: failures.append(str(exc)))
        page.goto(url)

        # Existing native figures are editable before any composition project exists.
        page.locator('[data-view="native"]').click()
        page.get_by_role("button", name="Browser native text · Text", exact=True).click()
        page.locator('#native-property-form input[name="dx"]').fill("12")
        page.get_by_role("button", name="Move artist", exact=True).click()
        expect(page.locator("#native-status")).to_contain_text("Revision 1")
        page.locator("#native-undo").click()
        expect(page.locator("#native-status")).to_contain_text("Revision 2")
        page.locator("#native-refresh").focus()
        page.keyboard.press("Control+Shift+z")
        expect(page.locator("#native-status")).to_contain_text("Revision 3")
        page.keyboard.press("Control+z")
        expect(page.locator("#native-status")).to_contain_text("Revision 4")
        with page.expect_download() as downloaded:
            page.keyboard.press("Control+s")
        changes_path = artifacts / "browser-native-changes.py"
        downloaded.value.save_as(changes_path)
        assert "FigureSession" in changes_path.read_text()
        print("PASS native Figure semantic edits, keyboard history and Python changes export", flush=True)

        page.locator("#new-project").click()
        page.locator('#new-project-form input[name="name"]').fill("Browser verification")
        page.locator('#new-project-form input[name="example"]').check()
        page.locator('#new-project-form button[type="submit"]').click()
        expect(page.locator("#save-state")).to_have_text("Saved")
        page.locator('[data-view="data"]').click()
        page.locator("#dataset-file").set_input_files({"name": "measurement.csv", "mimeType": "text/csv",
                                                      "buffer": b"x,y\n0,1\n1,\n2,4\n3,8\n"})
        expect(page.locator("#dataset-content h2")).to_have_text("measurement.csv")
        page.locator("#ai-provider").select_option("local")
        page.locator("#ai-prompt").fill("Fill missing y with the mean")
        page.locator('#ai-form button[type="submit"]').click()
        expect(page.locator("#transform-status")).to_contain_text("Preview ready")
        expect(page.locator("#ai-explanation")).to_contain_text("deterministic")
        page.locator("#apply-recipe").click()
        expect(page.locator("#dataset-content h2")).to_contain_text("prepared")
        expect(page.locator("#dataset-list button")).to_have_count(3)
        page.locator('#plot-form [name="x"]').select_option("x")
        page.locator('#plot-form [name="y"]').select_option("y")
        page.locator('#plot-form [name="title"]').fill("Measured response")
        page.locator('#plot-form button[type="submit"]').click()
        expect(page.locator("#view-figure")).to_be_visible()
        page.locator("#save-project").click()
        expect(page.locator("#save-state")).to_have_text("Saved")
        print("PASS source preview, local proposal, derived data, bound plot and save", flush=True)

        # Keyboard commands and numerical geometry use the same document history.
        previous_count = int(page.locator("#layer-count").inner_text())
        page.locator('[data-add="rect"]').click()
        expect(page.locator("#layer-count")).to_have_text(str(previous_count + 1))
        page.locator('[data-prop="x"]').fill("620")
        page.locator('[data-prop="x"]').press("Tab")
        page.locator("#canvas-viewport").focus()
        page.keyboard.press("Control+d")
        expect(page.locator("#layer-count")).to_have_text(str(previous_count + 2))
        page.keyboard.press("Control+z")
        expect(page.locator("#layer-count")).to_have_text(str(previous_count + 1))
        page.locator("#save-project").click()
        expect(page.locator("#save-state")).to_have_text("Saved")
        print("PASS numeric geometry, duplicate and gesture history", flush=True)

        page.locator('[data-view="structures"]').click()
        page.locator("#structure-file").set_input_files(str(ROOT / "pylustrator/web/examples/1crn.pdb"))
        expect(page.locator("#structure-heading h2")).to_have_text("1crn.pdb")
        expect(page.locator("#protein-viewer canvas").first).to_be_visible()
        expect(page.locator("#structure-metadata")).to_contain_text("327 atoms")
        page.locator("#structure-style").select_option("stick")
        page.locator("#save-structure-view").click()
        expect(page.locator("#structure-status")).to_contain_text("saved")
        page.screenshot(path=str(artifacts / "web-structures.png"), full_page=True)
        page.locator("#capture-structure").click()
        expect(page.locator("#view-figure")).to_be_visible()
        expect(page.locator("#asset-summary")).to_contain_text("1 images")
        page.locator("#save-project").click()
        expect(page.locator("#save-state")).to_have_text("Saved")
        print("PASS real PDB WebGL view, persistent camera and provenance snapshot", flush=True)

        page.locator('[data-view="versions"]').click()
        page.locator('#checkpoint-form input[name="name"]').fill("Browser verified checkpoint")
        page.locator('#checkpoint-form button[type="submit"]').click()
        expect(page.locator("#version-list")).to_contain_text("Browser verified checkpoint")
        page.locator('[data-view="figure"]').click()
        page.locator("#export-format").select_option("svg")
        with page.expect_download() as downloaded:
            page.locator("#export-button").click()
        svg_path = artifacts / "browser-export.svg"
        downloaded.value.save_as(svg_path)
        assert b"<svg" in svg_path.read_bytes()
        page.reload()
        expect(page.locator("#save-state")).to_have_text("Saved")
        expect(page.locator("#asset-summary")).to_contain_text("1 structures")
        # Wait for plot responses before recording the final layout.
        expect(page.locator("#artwork")).not_to_contain_text("Rendering plot")
        page.screenshot(path=str(artifacts / "web-figure.png"), full_page=True)
        print("PASS named checkpoint, vector export and persisted reload", flush=True)

        # A stale second tab must retain its edit and surface an actual conflict.
        stale = context.new_page()
        stale.goto(url)
        expect(stale.locator("#save-state")).to_have_text("Saved")
        page.locator('[data-add="ellipse"]').click()
        page.locator("#save-project").click()
        expect(page.locator("#save-state")).to_have_text("Saved")
        stale.locator('[data-add="line"]').click()
        stale.locator("#save-project").click()
        expect(stale.locator("#notice-text")).to_contain_text("another tab")
        expect(stale.locator("#save-state")).to_have_text("Save failed")
        stale.close(run_before_unload=False)
        print("PASS real two-tab optimistic revision conflict", flush=True)

        page.set_viewport_size({"width": 390, "height": 844})
        page.locator('[data-view="data"]').click()
        expect(page.locator("#dataset-content table")).to_be_visible()
        assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth")
        expect(page.locator("#project-menu-button")).to_be_visible()
        page.screenshot(path=str(artifacts / "web-mobile.png"), full_page=True)
        assert not failures, failures
        print("PASS narrow viewport, accessible project actions and zero page errors", flush=True)
        browser.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="forge-browser-") as scratch:
        scratch = Path(scratch)
        artifacts = args.artifacts or scratch / "artifacts"
        artifacts.mkdir(parents=True, exist_ok=True)
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
        env = {**os.environ, "MPLBACKEND": "Agg", "MPLCONFIGDIR": str(scratch / "mpl")}
        with (scratch / "server.log").open("w+") as log:
            process = subprocess.Popen([sys.executable, "-c", SERVER, str(scratch / "projects"), str(port)],
                                       cwd=ROOT, env=env, stdout=log, stderr=log)
            try:
                url = f"http://127.0.0.1:{port}"
                wait_server(url, process)
                run_browser(url, artifacts)
            except BaseException:
                log.seek(0)
                print(log.read()[-6000:], file=sys.stderr)
                raise
            finally:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()


if __name__ == "__main__":
    main()
