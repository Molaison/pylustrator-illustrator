"""Opt-in real-browser workflow check against an already running local server.

Run: FORGE_BASE_URL=http://127.0.0.1:8124 python tests/web/smoke_ui.py
Requires Playwright and a Chromium installation. Creates an explicitly named QA project.
"""
import os
from pathlib import Path

from playwright.sync_api import expect, sync_playwright


def main():
    base = os.environ.get("FORGE_BASE_URL", "http://127.0.0.1:8124")
    errors = []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(args=["--no-sandbox", "--enable-unsafe-swiftshader"])
        page = browser.new_page(viewport={"width": 1440, "height": 980}, device_scale_factor=1)
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto(base)
        page.get_by_role("button", name="New project", exact=True).click()
        page.locator("#new-project-form input[name=name]").fill("QA · browser workflow")
        page.locator("#new-project-form input[name=example]").check()
        page.locator("#new-project-form button[type=submit]").click()
        page.locator("#new-project-dialog").wait_for(state="hidden")
        page.locator('#artwork [data-element="example-plot"] image').wait_for()
        page.locator('[data-add="text"]').click()
        page.locator('#inspector textarea[data-prop="text"]').fill("Experiment α — annotation")
        page.locator('#inspector textarea[data-prop="text"]').press("Tab")
        page.locator("#save-project").click()
        expect(page.locator("#save-state")).to_have_text("Saved")
        page.locator('[data-view="data"]').click()
        page.locator("#dataset-content table").wait_for()
        page.locator("#transform-op").select_option("normalize")
        page.locator("#transform-form button[type=submit]").click()
        page.locator("#processed-data-tab").wait_for()
        page.locator("#apply-recipe").click()
        expect(page.locator("#dataset-content h2")).to_contain_text("prepared")
        page.locator("#ai-prompt").fill("Normalize response")
        page.locator("#ai-form button[type=submit]").click()
        expect(page.locator("#ai-status")).to_contain_text("deterministic")
        page.locator("#plot-form button[type=submit]").click()
        page.locator("#view-figure").wait_for(state="visible")
        page.locator('[data-view="structures"]').click()
        sample = Path(__file__).parents[2] / "pylustrator/web/examples/1crn.pdb"
        page.locator("#structure-file").set_input_files(sample)
        page.locator("#protein-viewer canvas").wait_for(timeout=30000)
        page.locator("#structure-style").select_option("stick")
        page.locator("#capture-structure").click()
        page.locator("#view-figure").wait_for(state="visible", timeout=30000)
        page.locator("#save-project").click()
        expect(page.locator("#save-state")).to_have_text("Saved")
        page.locator('[data-view="versions"]').click()
        page.locator("#checkpoint-form input[name=name]").fill("Browser-verified result")
        page.locator("#checkpoint-form button[type=submit]").click()
        page.get_by_role("heading", name="Browser-verified result").wait_for()
        page.locator('[data-view="figure"]').click()
        with page.expect_download() as info:
            page.locator("#export-button").click()
        download = info.value
        assert download.suggested_filename.endswith(".svg")
        page.screenshot(path="/tmp/forge-ui-desktop.png", full_page=True)
        page.reload()
        page.locator("#artwork image").first.wait_for()
        page.set_viewport_size({"width": 390, "height": 844})
        page.screenshot(path="/tmp/forge-ui-mobile.png", full_page=True)
        assert page.evaluate("() => document.documentElement.scrollWidth <= innerWidth"), "Horizontal page overflow"
        assert not errors, errors
        print("PASS: project → data preview/transform → AI plan → plot → protein view → checkpoint → SVG → reload; no page errors; narrow layout fits")
        browser.close()


if __name__ == "__main__":
    main()
