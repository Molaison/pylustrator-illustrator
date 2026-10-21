import { uid } from "./api.js";
export const clone = (value) => JSON.parse(JSON.stringify(value));
export function rectOf(element) {
  const { x, y, width: w, height: h } = element,
    a = ((element.rotation || 0) * Math.PI) / 180;
  const cx = x + w / 2,
    cy = y + h / 2;
  const local =
    ["line", "path"].includes(element.type) && element.points?.length
      ? element.points.map(([px, py]) => [px - w / 2, py - h / 2])
      : [
          [-w / 2, -h / 2],
          [w / 2, -h / 2],
          [w / 2, h / 2],
          [-w / 2, h / 2],
        ];
  const corners = local.map(([px, py]) => [
    cx + px * Math.cos(a) - py * Math.sin(a),
    cy + px * Math.sin(a) + py * Math.cos(a),
  ]);
  const xs = corners.map((p) => p[0]),
    ys = corners.map((p) => p[1]);
  return {
    x: Math.min(...xs),
    y: Math.min(...ys),
    width: Math.max(...xs) - Math.min(...xs),
    height: Math.max(...ys) - Math.min(...ys),
  };
}
export function union(elements) {
  if (!elements.length) return null;
  const bounds = elements.map(rectOf),
    x = Math.min(...bounds.map((b) => b.x)),
    y = Math.min(...bounds.map((b) => b.y));
  return {
    x,
    y,
    width: Math.max(...bounds.map((b) => b.x + b.width)) - x,
    height: Math.max(...bounds.map((b) => b.y + b.height)) - y,
  };
}
export function makeElement(type, extras = {}) {
  const defaults = {
    id: uid("element"),
    type,
    name: {
      text: "Text",
      rect: "Rectangle",
      ellipse: "Ellipse",
      line: "Line",
      path: "Path",
      plot: "Plot",
      image: "Image",
    }[type],
    x: 80,
    y: 80,
    width: 180,
    height: 100,
    rotation: 0,
    visible: true,
    locked: false,
    groupId: null,
    style: {
      fill: "#dbeafe",
      stroke: "#274c77",
      strokeWidth: 1.5,
      opacity: 1,
      fontSize: 24,
    },
  };
  if (type === "text") {
    defaults.text = "Panel label";
    defaults.height = 40;
    defaults.style.fill = "#1f2937";
    defaults.style.stroke = "none";
  }
  if (type === "line") {
    defaults.points = [
      [0, 0],
      [180, 100],
    ];
    defaults.style.fill = "none";
  }
  if (type === "path") {
    defaults.points = [
      [0, 100],
      [70, 0],
      [180, 100],
    ];
    defaults.style.fill = "none";
  }
  if (type === "plot") {
    defaults.width = 480;
    defaults.height = 320;
  }
  return {
    ...defaults,
    ...extras,
    style: { ...defaults.style, ...extras.style },
  };
}

export class EditorState {
  constructor() {
    this.document = null;
    this.selection = new Set();
    this.undoStack = [];
    this.redoStack = [];
    this.listeners = new Set();
    this.saved = "";
    this.gesture = null;
  }
  subscribe(fn) {
    this.listeners.add(fn);
    return () => this.listeners.delete(fn);
  }
  emit(kind = "change") {
    for (const fn of this.listeners) fn(kind);
  }
  load(document) {
    this.document = clone(document);
    this.document.groups ||= [];
    this.selection.clear();
    this.undoStack = [];
    this.redoStack = [];
    this.gesture = null;
    this.saved = JSON.stringify(this.document);
    this.emit("load");
  }
  markSaved(document = this.document) {
    this.saved = JSON.stringify(document);
    this.emit("status");
  }
  get dirty() {
    return (
      Boolean(this.document) && JSON.stringify(this.document) !== this.saved
    );
  }
  selected({ editable = false } = {}) {
    return (this.document?.elements || []).filter(
      (e) => this.selection.has(e.id) && (!editable || !e.locked),
    );
  }
  select(ids, { toggle = false, group = true } = {}) {
    const found = new Set(ids);
    if (group)
      for (const element of this.document?.elements || [])
        if (found.has(element.id) && element.groupId)
          for (const peer of this.document.elements)
            if (peer.groupId === element.groupId) found.add(peer.id);
    if (toggle)
      for (const id of found)
        this.selection.has(id)
          ? this.selection.delete(id)
          : this.selection.add(id);
    else this.selection = found;
    this.emit("selection");
  }
  mutate(label, fn) {
    if (!this.document) return;
    const before = clone(this.document);
    fn(this.document);
    if (JSON.stringify(before) === JSON.stringify(this.document)) return;
    this.undoStack.push({ label, document: before });
    if (this.undoStack.length > 80) this.undoStack.shift();
    this.redoStack = [];
    this.pruneSelection();
    this.emit("change");
  }
  beginGesture(label) {
    if (!this.gesture) this.gesture = { label, document: clone(this.document) };
  }
  preview(fn) {
    if (!this.gesture) return;
    this.document = clone(this.gesture.document);
    fn(this.document);
    this.emit("preview");
  }
  endGesture(cancel = false) {
    if (!this.gesture) return;
    const { label, document } = this.gesture;
    this.gesture = null;
    if (cancel) this.document = document;
    else if (JSON.stringify(document) !== JSON.stringify(this.document)) {
      this.undoStack.push({ label, document });
      if (this.undoStack.length > 80) this.undoStack.shift();
      this.redoStack = [];
    }
    this.emit("change");
  }
  pruneSelection() {
    const ids = new Set(this.document.elements.map((e) => e.id));
    this.selection = new Set([...this.selection].filter((id) => ids.has(id)));
  }
  undo() {
    if (this.gesture) this.endGesture(true);
    const item = this.undoStack.pop();
    if (!item) return;
    this.redoStack.push({ label: item.label, document: clone(this.document) });
    this.document = item.document;
    this.pruneSelection();
    this.emit("change");
  }
  redo() {
    const item = this.redoStack.pop();
    if (!item) return;
    this.undoStack.push({ label: item.label, document: clone(this.document) });
    this.document = item.document;
    this.pruneSelection();
    this.emit("change");
  }
  add(type, extras) {
    const el = makeElement(type, extras);
    this.mutate(`Add ${el.name}`, (doc) => doc.elements.push(el));
    this.select([el.id]);
    return el;
  }
  updateSelection(properties, label = "Edit properties") {
    this.mutate(label, (doc) => {
      for (const el of doc.elements)
        if (this.selection.has(el.id) && !el.locked) {
          if (
            el.points &&
            (properties.width !== undefined || properties.height !== undefined)
          ) {
            const sx =
                properties.width === undefined
                  ? 1
                  : properties.width / el.width,
              sy =
                properties.height === undefined
                  ? 1
                  : properties.height / el.height;
            el.points = el.points.map(([x, y]) => [x * sx, y * sy]);
          }
          Object.assign(
            el,
            properties,
            properties.style
              ? { style: { ...el.style, ...properties.style } }
              : {},
          );
        }
    });
  }
  remove() {
    this.mutate("Delete selection", (doc) => {
      doc.elements = doc.elements.filter(
        (e) => !this.selection.has(e.id) || e.locked,
      );
      doc.groups = doc.groups.filter((g) =>
        doc.elements.some((e) => e.groupId === g.id),
      );
    });
  }
  duplicate() {
    const copies = this.selected().map((e) => ({
      ...clone(e),
      id: uid("element"),
      name: `${e.name} copy`,
      x: e.x + 16,
      y: e.y + 16,
      locked: false,
      groupId: null,
    }));
    this.mutate("Duplicate selection", (doc) => doc.elements.push(...copies));
    this.select(copies.map((e) => e.id));
  }
  group() {
    const ids = this.selected({ editable: true }).map((e) => e.id);
    if (ids.length < 2) return;
    const id = uid("group");
    this.mutate("Group selection", (doc) => {
      doc.groups.push({ id, name: "Group" });
      for (const e of doc.elements) if (ids.includes(e.id)) e.groupId = id;
      doc.groups = doc.groups.filter((g) =>
        doc.elements.some((e) => e.groupId === g.id),
      );
    });
  }
  ungroup() {
    const groups = new Set(this.selected().map((e) => e.groupId));
    this.mutate("Ungroup selection", (doc) => {
      for (const e of doc.elements)
        if (groups.has(e.groupId) && !e.locked) e.groupId = null;
      doc.groups = doc.groups.filter((g) =>
        doc.elements.some((e) => e.groupId === g.id),
      );
    });
  }
  order(direction) {
    this.mutate(direction > 0 ? "Bring forward" : "Send backward", (doc) => {
      const list = doc.elements,
        indices = list
          .map((e, i) => (this.selection.has(e.id) && !e.locked ? i : -1))
          .filter((i) => i >= 0);
      if (direction > 0) indices.reverse();
      for (const i of indices) {
        const j = i + direction;
        if (j >= 0 && j < list.length && !this.selection.has(list[j].id))
          [list[i], list[j]] = [list[j], list[i]];
      }
    });
  }
  align(mode, toArtboard = false) {
    const items = this.selected({ editable: true });
    if (!items.length) return;
    const box = toArtboard
      ? { x: 0, y: 0, width: this.document.width, height: this.document.height }
      : union(items);
    this.mutate(`Align ${mode}`, (doc) => {
      for (const e of doc.elements)
        if (items.some((i) => i.id === e.id)) {
          const b = rectOf(e);
          if (mode === "left") e.x += box.x - b.x;
          if (mode === "center")
            e.x += box.x + box.width / 2 - b.x - b.width / 2;
          if (mode === "right") e.x += box.x + box.width - b.x - b.width;
          if (mode === "top") e.y += box.y - b.y;
          if (mode === "middle")
            e.y += box.y + box.height / 2 - b.y - b.height / 2;
          if (mode === "bottom") e.y += box.y + box.height - b.y - b.height;
        }
    });
  }
  distribute(axis) {
    const items = this.selected({ editable: true }).sort(
      (a, b) => rectOf(a)[axis] - rectOf(b)[axis],
    );
    if (items.length < 3) return;
    const dim = axis === "x" ? "width" : "height",
      first = rectOf(items[0]),
      last = rectOf(items.at(-1)),
      total = items.reduce((sum, e) => sum + rectOf(e)[dim], 0),
      gap = (last[axis] + last[dim] - first[axis] - total) / (items.length - 1);
    let pos = first[axis];
    const offsets = new Map();
    for (const e of items) {
      const b = rectOf(e);
      offsets.set(e.id, pos - b[axis]);
      pos += b[dim] + gap;
    }
    this.mutate("Distribute selection", (doc) => {
      for (const e of doc.elements)
        if (offsets.has(e.id)) e[axis] += offsets.get(e.id);
    });
  }
}
