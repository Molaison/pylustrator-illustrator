import test from "node:test";
import assert from "node:assert/strict";
import {
  EditorState,
  makeElement,
  rectOf,
} from "../../pylustrator/web/static/state.js";

function state(elements = []) {
  const s = new EditorState();
  s.load({
    schemaVersion: 1,
    id: "figure",
    name: "Test",
    width: 960,
    height: 640,
    background: "#fff",
    groups: [],
    elements,
  });
  return s;
}
test("one pointer gesture records one undo item and cancellation restores exact geometry", () => {
  const s = state([makeElement("rect", { id: "a", x: 10 })]);
  s.beginGesture("Move");
  for (let i = 1; i <= 50; i++) s.preview((d) => (d.elements[0].x = 10 + i));
  s.endGesture();
  assert.equal(s.undoStack.length, 1);
  assert.equal(s.document.elements[0].x, 60);
  s.undo();
  assert.equal(s.document.elements[0].x, 10);
  s.redo();
  assert.equal(s.document.elements[0].x, 60);
  s.beginGesture("Move");
  s.preview((d) => (d.elements[0].x = 800));
  s.endGesture(true);
  assert.equal(s.document.elements[0].x, 60);
  assert.equal(s.undoStack.length, 1);
});
test("locked elements survive delete and transform while unlocked elements change", () => {
  const s = state([
    makeElement("rect", { id: "a", locked: true }),
    makeElement("rect", { id: "b" }),
  ]);
  s.select(["a", "b"]);
  s.updateSelection({ x: 200 });
  assert.equal(s.document.elements[0].x, 80);
  assert.equal(s.document.elements[1].x, 200);
  s.remove();
  assert.equal(s.document.elements.length, 1);
  assert.equal(s.document.elements[0].id, "a");
  s.undo();
  assert.equal(s.document.elements.length, 2);
});
test("changing line dimensions scales its actual geometry and undo restores source points", () => {
  const s = state([makeElement("line", { id: "a" })]);
  s.select(["a"]);
  s.updateSelection({ width: 360, height: 50 });
  assert.deepEqual(s.document.elements[0].points, [
    [0, 0],
    [360, 50],
  ]);
  s.undo();
  assert.deepEqual(s.document.elements[0].points, [
    [0, 0],
    [180, 100],
  ]);
});
test("group selection expands without rewriting parent ownership and duplicate uses fresh IDs", () => {
  const s = state([
    makeElement("rect", { id: "a" }),
    makeElement("text", { id: "b" }),
  ]);
  s.select(["a", "b"]);
  s.group();
  const group = s.document.elements[0].groupId;
  assert.ok(group);
  s.select(["a"]);
  assert.equal(s.selection.size, 2);
  s.duplicate();
  assert.equal(s.document.elements.length, 4);
  assert.equal(new Set(s.document.elements.map((e) => e.id)).size, 4);
  assert.ok(s.selected().every((e) => e.groupId === null));
  s.undo();
  assert.equal(s.document.elements.length, 2);
  assert.equal(s.document.elements[1].groupId, group);
});
test("version save marker does not erase later edits or editing history", () => {
  const s = state([makeElement("text", { id: "a" })]);
  s.select(["a"]);
  s.updateSelection({ text: "Saved text" });
  const submitted = structuredClone(s.document);
  s.updateSelection({ text: "Later edit" });
  s.markSaved(submitted);
  assert.equal(s.dirty, true);
  s.undo();
  assert.equal(s.dirty, false);
  assert.equal(s.undoStack.length, 1);
});
test("rotated painted rectangle bounds and alignment use visible geometry", () => {
  const a = makeElement("rect", {
    id: "a",
    x: 20,
    y: 30,
    width: 100,
    height: 40,
    rotation: 90,
  });
  const b = rectOf(a);
  assert.ok(Math.abs(b.width - 40) < 1e-8);
  assert.ok(Math.abs(b.height - 100) < 1e-8);
  const s = state([a]);
  s.select(["a"]);
  s.align("left", true);
  assert.ok(Math.abs(rectOf(s.document.elements[0]).x) < 1e-8);
});
test("direct point edits outside the nominal box update painted bounds", () => {
  const a = makeElement("path", {
    x: 10,
    y: 20,
    width: 100,
    height: 100,
    points: [
      [-40, -20],
      [180, 130],
    ],
  });
  assert.deepEqual(rectOf(a), { x: -30, y: 0, width: 220, height: 150 });
  const s = state([a]);
  s.select([a.id]);
  s.align("left", true);
  assert.equal(rectOf(s.document.elements[0]).x, 0);
});
