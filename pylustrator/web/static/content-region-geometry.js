// Pure geometry contract shared by the UI and future consumers. No Agent API.
export function sourceFor(element) {
  if (element.type === "image") return { type: "image", assetId: element.assetId };
  // Match document.validate_plot defaults so unsaved new plots and normalized
  // persisted plots have the same source identity. Never alias the live plot.
  const p = element.plot || {};
  const plot = { datasetId: p.datasetId, kind: p.kind ?? "scatter", x: p.x ?? "", y: p.y ?? "",
    title: p.title ?? "", xlabel: p.xlabel ?? p.x ?? "", ylabel: p.ylabel ?? p.y ?? "",
    color: p.color === "transparent" ? "none" : p.color ?? "#2563eb" };
  if ("bins" in p) plot.bins = Math.trunc(p.bins);
  return { type: "plot", plot, width: element.width, height: element.height };
}

const same = (a, b) => {
  if (a === b) return true;
  if (!a || !b || typeof a !== "object" || typeof b !== "object") return false;
  const keys = Object.keys(a);
  return keys.length === Object.keys(b).length && keys.every((k) => same(a[k], b[k]));
};
export const regionsCurrent = (element) =>
  Boolean(element.contentRegions && same(element.contentRegions.source, sourceFor(element)));

export function validBounds(b) {
  if (!b || ![b.x, b.y, b.width, b.height].every((v) => typeof v === "number" && Number.isFinite(v)) ||
      b.x < 0 || b.y < 0 || b.width < 1e-6 || b.height < 1e-6 || b.x + b.width > 1 + 1e-9 || b.y + b.height > 1 + 1e-9)
    throw new Error("Bounds must fit inside the panel (0–100%, with positive width and height).");
  return b;
}

export function regionBounds(element, kind, requireConfirmed = true) {
  if (kind === "panel") return { x: 0, y: 0, width: 1, height: 1 };
  if (!regionsCurrent(element)) throw new Error(`${element.name}: detect again after changing the source or plot size.`);
  const region = element.contentRegions[kind];
  if (!region) throw new Error(`${element.name}: set ${kind} bounds manually or detect first.`);
  if (requireConfirmed && !region.confirmed) throw new Error(`${element.name}: inspect and confirm the ${kind} outline first.`);
  return validBounds(region.bounds);
}

export function regionCorners(element, bounds) {
  const b = validBounds(bounds), w = element.width, h = element.height;
  const angle = (element.rotation || 0) * Math.PI / 180;
  return [[b.x, b.y], [b.x + b.width, b.y], [b.x + b.width, b.y + b.height], [b.x, b.y + b.height]].map(([u, v]) => {
    const x = (u - .5) * w, y = (v - .5) * h;
    return [element.x + w / 2 + x * Math.cos(angle) - y * Math.sin(angle),
      element.y + h / 2 + x * Math.sin(angle) + y * Math.cos(angle)];
  });
}

export function worldBounds(element, bounds) {
  const points = regionCorners(element, bounds), xs = points.map((p) => p[0]), ys = points.map((p) => p[1]);
  return { x: Math.min(...xs), y: Math.min(...ys), width: Math.max(...xs) - Math.min(...xs), height: Math.max(...ys) - Math.min(...ys) };
}

function requireEditable(state, items) {
  if (state.gesture) throw new Error("Finish the current drag before editing regions.");
  if (items.some((e) => e.locked || !e.visible || !["plot", "image"].includes(e.type)))
    throw new Error("Select visible, unlocked image or plot panels only.");
}

export function manualRegion(state, id, kind, bounds) {
  if (!["plot", "content"].includes(kind)) throw new Error("Choose plot area or all content to correct bounds.");
  validBounds(bounds);
  const element = state.document.elements.find((e) => e.id === id);
  if (!element) throw new Error("The panel is no longer available.");
  requireEditable(state, [element]);
  const regions = regionsCurrent(element) ? structuredClone(element.contentRegions) : {
    version: 1, coordinateSpace: "normalized-panel", source: sourceFor(element), plot: null, content: null, warnings: [],
  };
  regions[kind] = { bounds: { ...bounds }, method: "manual", confidence: 1, confirmed: true };
  state.mutate(`Correct ${kind} bounds`, () => { element.contentRegions = regions; });
}

export function confirmRegion(state, id, kind) {
  const element = state.document.elements.find((e) => e.id === id);
  if (!element) throw new Error("The panel is no longer available.");
  requireEditable(state, [element]);
  regionBounds(element, kind, false);
  state.mutate(`Confirm ${kind} bounds`, () => { element.contentRegions[kind].confirmed = true; });
}

export function alignRegions(state, { kind = "plot", mode, referenceId }) {
  const items = state.selected();
  requireEditable(state, items);
  if (items.length < 2) throw new Error("Select at least two panels to align.");
  if (items.some((e) => e.groupId)) throw new Error("Ungroup panels before aligning their individual regions.");
  const rules = { left: ["x", "width", 0], center: ["x", "width", .5], right: ["x", "width", 1],
    top: ["y", "height", 0], middle: ["y", "height", .5], bottom: ["y", "height", 1] };
  if (!rules[mode]) throw new Error("Choose a valid alignment edge.");
  const reference = items.find((e) => e.id === referenceId);
  if (!reference) throw new Error("Choose a selected reference panel.");
  const [axis, size, fraction] = rules[mode];
  const boxes = new Map(items.map((e) => [e.id, worldBounds(e, regionBounds(e, kind))]));
  const anchor = boxes.get(reference.id), position = anchor[axis] + anchor[size] * fraction;
  const moves = items.map((e) => {
    const b = boxes.get(e.id), delta = position - b[axis] - b[size] * fraction;
    const moved = { ...e, [axis]: e[axis] + delta };
    const full = worldBounds(moved, regionBounds(moved, "panel"));
    if (full.x < -1e-7 || full.y < -1e-7 || full.x + full.width > state.document.width + 1e-7 || full.y + full.height > state.document.height + 1e-7)
      throw new Error(`${e.name}: alignment would put part of the original panel outside the artboard. Move the reference inward or enlarge the artboard.`);
    return { id: e.id, value: moved[axis] };
  });
  state.mutate(`Align ${kind} ${mode}`, (doc) => {
    for (const move of moves) doc.elements.find((e) => e.id === move.id)[axis] = move.value;
  });
}
