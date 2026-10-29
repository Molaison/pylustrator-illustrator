import test from "node:test";
import assert from "node:assert/strict";
import { EditorState, makeElement } from "../../pylustrator/web/static/state.js";
import { alignRegions, confirmRegion, manualRegion, regionBounds, regionsCurrent, sourceFor, worldBounds } from "../../pylustrator/web/static/content-region-geometry.js";

function state() {
  const s = new EditorState();
  s.load({ width: 1000, height: 900, groups: [], elements: [
    makeElement("image", { id: "a", assetId: "asset-a", x: 200, y: 30, width: 600, height: 300 }),
    makeElement("image", { id: "b", assetId: "asset-b", x: 200, y: 420, width: 600, height: 350 }),
  ] });
  manualRegion(s, "a", "plot", { x: .15, y: .2, width: .65, height: .6 });
  manualRegion(s, "b", "plot", { x: .3, y: .15, width: .5, height: .65 });
  s.select(["a", "b"]);
  return s;
}
const box = (e) => worldBounds(e, regionBounds(e, "plot"));

test("different white margins align by data axes, preserving entire sources, with one undo", () => {
  const s = state(), before = structuredClone(s.document), count = s.undoStack.length;
  assert.notEqual(box(before.elements[0]).x, box(before.elements[1]).x);
  alignRegions(s, { kind: "plot", mode: "left", referenceId: "a" });
  const [a, b] = s.document.elements;
  assert.equal(box(a).x, box(b).x);
  assert.equal(b.x, 110);
  assert.deepEqual(a, before.elements[0]);
  assert.deepEqual({ ...b, x: before.elements[1].x }, before.elements[1]);
  assert.equal(s.undoStack.length, count + 1);
  const after = structuredClone(s.document);
  s.undo(); assert.deepEqual(s.document, before);
  s.redo(); assert.deepEqual(s.document, after);
});

test("manual correction and confirmation have exact independent undo", () => {
  const s = state(), before = structuredClone(s.document);
  manualRegion(s, "b", "plot", { x: .32, y: .1, width: .5, height: .7 });
  assert.equal(s.document.elements[1].contentRegions.plot.bounds.x, .32);
  s.undo(); assert.deepEqual(s.document, before);
  s.document.elements[1].contentRegions.plot.confirmed = false;
  assert.throws(() => alignRegions(s, { kind: "plot", mode: "left", referenceId: "a" }), /confirm/);
  confirmRegion(s, "b", "plot");
  assert.equal(s.document.elements[1].contentRegions.plot.confirmed, true);
  s.undo(); assert.equal(s.document.elements[1].contentRegions.plot.confirmed, false);
});

test("preflight rejects clipping, locks, groups and missing regions atomically", () => {
  for (const change of [
    (s) => { s.document.elements[0].x = 0; },
    (s) => { s.document.elements[1].locked = true; },
    (s) => { s.document.elements[1].groupId = "g"; },
    (s) => { delete s.document.elements[1].contentRegions; },
  ]) {
    const s = state(); change(s);
    const before = structuredClone(s.document), count = s.undoStack.length;
    assert.throws(() => alignRegions(s, { kind: "plot", mode: "left", referenceId: "a" }));
    assert.deepEqual(s.document, before);
    assert.equal(s.undoStack.length, count);
  }
});

test("normalized image bounds follow resizing; plot layout edits invalidate old regions", () => {
  const s = state(), e = s.document.elements[0];
  e.width *= .5;
  assert.equal(regionsCurrent(e), true);
  assert.equal(box(e).x, 245);
  e.assetId = "new-asset";
  assert.equal(regionsCurrent(e), false);
  const p = makeElement("plot", { id: "p", plot: { datasetId: "d", x: "x", y: "y" } });
  p.contentRegions = { source: structuredClone(sourceFor(p)) };
  assert.equal(regionsCurrent(p), true);
  p.width += 1;
  assert.equal(regionsCurrent(p), false);
  s.document.elements.push(p);
  manualRegion(s, "p", "plot", { x: .1, y: .1, width: .7, height: .7 });
  p.plot.x = "different-column";
  assert.equal(regionsCurrent(p), false);
});

test("rotated local regions map through the panel center, with safe alignment", () => {
  const s = state();
  s.document.elements[0].rotation = 10;
  s.document.elements[0].y = 100;
  s.document.elements[1].rotation = -10;
  alignRegions(s, { kind: "plot", mode: "left", referenceId: "a" });
  assert.ok(Math.abs(box(s.document.elements[0]).x - box(s.document.elements[1]).x) < 1e-9);
});

test("invalid manual corrections never mutate the panel or create history", () => {
  const s = state(), before = structuredClone(s.document), count = s.undoStack.length;
  for (const b of [{ x: .8, y: 0, width: .3, height: .4 }, { x: NaN, y: 0, width: .3, height: .4 }])
    assert.throws(() => manualRegion(s, "a", "plot", b));
  assert.deepEqual(s.document, before);
  assert.equal(s.undoStack.length, count);
});
