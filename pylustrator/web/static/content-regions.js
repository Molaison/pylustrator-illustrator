import { api, projectPath, escapeHTML as esc } from "./api.js";
import { alignRegions, confirmRegion, manualRegion, regionBounds, regionCorners, regionsCurrent, validBounds } from "./content-region-geometry.js";

const NS = "http://www.w3.org/2000/svg";
const rect = { x: .15, y: .1, width: .7, height: .7 };

// Additive integration: uses EditorState's subscription and one mutate per action.
// Owns neither pointer gestures, layer ordering, nor undo/redo stacks.
export class ContentRegionsPanel {
  constructor(state, hooks) {
    this.state = state;
    this.hooks = hooks;
    this.kind = "plot";
    this.epoch = 0;
    this.pending = false;
    this.detection = null;
    this.target = document.createElement("section");
    this.target.className = "panel-section content-regions";
    this.target.id = "content-regions";
    document.querySelector("#inspector").before(this.target);
    this.overlay = document.createElementNS(NS, "g");
    this.overlay.id = "content-region-overlay";
    this.overlay.setAttribute("pointer-events", "none");
    this.overlay.setAttribute("aria-hidden", "true");
    document.querySelector("#artboard").append(this.overlay);
    state.subscribe((kind) => {
      if (kind === "status") return;
      this.epoch++;
      this.cancelDetection("Selection or document changed; detection was cancelled. Run it again for the current panels.");
      this.preview = null;
      if (kind === "load") this.panelId = this.referenceId = null;
      if (kind !== "preview") this.render();
      else this.draw();
    });
    this.render();
  }
  attempt(fn, message) {
    try { fn(); if (message) this.hooks.notify(message); } catch (error) { this.hooks.notify(error.message, "error"); }
  }
  cancelDetection(message = "Detection cancelled. No bounds changed; retry or set bounds manually.") {
    if (!this.detection) return;
    this.detection.reason = message;
    this.detection.controller.abort();
  }
  render() {
    const items = this.state.selected(), panels = items.filter((e) => ["image", "plot"].includes(e.type));
    this.target.hidden = !panels.length;
    if (!panels.length) { this.draw(); return; }
    this.panelId = panels.some((e) => e.id === this.panelId) ? this.panelId : panels[0].id;
    this.referenceId = panels.some((e) => e.id === this.referenceId) ? this.referenceId : panels[0].id;
    const panel = panels.find((e) => e.id === this.panelId), current = regionsCurrent(panel);
    const region = current ? panel.contentRegions[this.kind] : null;
    const bounds = region?.bounds || rect;
    const options = (id) => panels.map((e) => `<option value="${esc(e.id)}" ${e.id === id ? "selected" : ""}>${esc(e.name)}</option>`).join("");
    const unsupported = items.some((e) => e.locked || !e.visible || !["image", "plot"].includes(e.type));
    const disabled = this.pending || unsupported;
    const status = this.kind === "panel" ? "Original panel boundary" : !current ?
      (panel.contentRegions ? "Source changed · detect again or set new bounds" : "Detection needed") : !region ? "Manual bounds needed" :
      region.confirmed ? (region.method === "manual" ? "Manual bounds · confirmed" : "Confirmed bounds") : "Suggestion · review outline";
    this.target.innerHTML = `<h3>Content alignment</h3>
      <p class="small-note">Move complete panels by their plot area. Labels, legends and original images stay intact.</p>
      <label>Align using<select id="region-kind"><option value="plot">Data plot area</option><option value="content">All content (labels + legend)</option><option value="panel">Original panel</option></select></label>
      <button id="region-detect" ${disabled ? "disabled" : ""}>${this.pending ? "Detecting…" : "Detect selected panels"}</button>
      ${this.pending ? '<button id="region-cancel-detection">Cancel detection</button>' : ""}
      ${unsupported ? '<p class="small-note">Select visible, unlocked images or plots only. Remove other objects from the selection to edit regions.</p>' : ""}
      <label>Inspect panel<select id="region-panel">${options(this.panelId)}</select></label>
      <p id="region-status" role="status" class="small-note">${esc(status)}</p>
      ${this.kind !== "panel" ? `<form id="region-form"><div class="field-pair">${["x", "y", "width", "height"].map((key) =>
        `<label>${key[0].toUpperCase() + key.slice(1)}, %<input name="${key}" type="number" min="0" max="100" step="any" value="${Number((bounds[key] * 100).toFixed(5))}" required ${disabled ? "disabled" : ""}></label>`).join("")}</div>
        <div class="field-pair"><button type="submit" ${disabled ? "disabled" : ""}>Apply correction</button><button type="button" id="region-confirm" ${disabled || !region || region.confirmed ? "disabled" : ""}>Confirm outline</button></div>
        <button type="button" id="region-discard" disabled>Discard correction</button>
        <p id="region-draft-status" class="small-note" role="status">Percentages refer to the full source panel, before rotation or view zoom.</p></form>
        ${current ? (panel.contentRegions.warnings || []).filter((w) =>
          !(region?.confirmed && w.startsWith("Raster axis-line suggestion")) &&
          (region?.method !== "manual" || w.startsWith("Some labels already"))).map((w) => `<p class="small-note">${esc(w)}</p>`).join("") : ""}` : ""}
      <label>Keep reference fixed<select id="region-reference">${options(this.referenceId)}</select></label>
      <div class="align-grid">${["left", "center", "right", "top", "middle", "bottom"].map((mode) => `<button data-region-align="${mode}" ${disabled || panels.length < 2 ? "disabled" : ""}>${mode[0].toUpperCase() + mode.slice(1)}</button>`).join("")}</div>
      <p class="small-note">Cyan: confirmed · amber: suggestion · blue: correction preview. Rotated regions align by their outer bounds. Each action supports Undo.</p>`;
    const $ = (selector) => this.target.querySelector(selector);
    $("#region-kind").value = this.kind;
    $("#region-kind").onchange = (e) => { this.kind = e.target.value; this.preview = null; this.render(); };
    $("#region-panel").onchange = (e) => { this.panelId = e.target.value; this.preview = null; this.render(); };
    $("#region-reference").onchange = (e) => { this.referenceId = e.target.value; };
    $("#region-detect").onclick = () => this.detect();
    if ($("#region-cancel-detection")) $("#region-cancel-detection").onclick = () => this.cancelDetection();
    if ($("#region-form")) {
      const read = () => Object.fromEntries(["x", "y", "width", "height"].map((k) => [k, Number($("#region-form").elements[k].value) / 100]));
      const discard = () => { this.preview = null; this.render(); };
      $("#region-form").oninput = () => {
        let message = "Correction preview only. Apply it to save these bounds, or discard it to review the detected outline.";
        try { this.preview = validBounds(read()); } catch (error) { this.preview = null; message = error.message; }
        $("#region-draft-status").textContent = message;
        $("#region-confirm").disabled = true;
        $("#region-detect").disabled = true;
        $("#region-discard").disabled = false;
        // Aligning the previous stored bounds while a different outline is being
        // previewed would misrepresent the user's intended correction.
        this.target.querySelectorAll("[data-region-align]").forEach((b) => { b.disabled = true; });
        this.draw();
      };
      $("#region-discard").onclick = discard;
      $("#region-form").onkeydown = (e) => {
        if (e.key === "Escape") { e.preventDefault(); e.stopPropagation(); discard(); }
      };
      $("#region-form").onsubmit = (e) => { e.preventDefault(); this.attempt(() => manualRegion(this.state, this.panelId, this.kind, read()), "Correction applied. Undo restores the previous bounds."); };
      $("#region-confirm").onclick = () => this.attempt(() => confirmRegion(this.state, this.panelId, this.kind), "Outline confirmed. Original panel and labels preserved.");
    }
    this.target.querySelectorAll("[data-region-align]").forEach((b) => { b.onclick = () => this.attempt(() =>
      alignRegions(this.state, { kind: this.kind, mode: b.dataset.regionAlign, referenceId: this.referenceId }), "Panels aligned. Original images, labels and legends preserved."); });
    this.draw();
  }
  async detect() {
    if (this.pending || this.state.gesture) return;
    const items = this.state.selected();
    if (!items.length || items.some((e) => e.locked || !e.visible || !["image", "plot"].includes(e.type))) return;
    const epoch = this.epoch, project = this.hooks.getProject(), captured = items.map((e) => structuredClone(e));
    const detection = { controller: new AbortController(), reason: "" };
    this.detection = detection;
    this.pending = true;
    this.render();
    try {
      // Serial, bounded requests: a large selection cannot start unbounded renders.
      const results = [];
      for (const element of captured) {
        if (epoch !== this.epoch || project?.id !== this.hooks.getProject()?.id) throw new Error("Selection or document changed; run detection again.");
        try {
          results.push(await api.post(projectPath(project, "/content-regions/detect"), { element }, { signal: detection.controller.signal }));
        } catch (error) {
          if (error.name === "AbortError") throw error;
          throw new Error(`${element.name}: ${error.message} No selected bounds changed; retry detection or set bounds manually.`);
        }
      }
      if (epoch !== this.epoch || project?.id !== this.hooks.getProject()?.id) throw new Error("Document changed while detecting; results were not applied.");
      this.detection = null;
      this.pending = false;
      this.state.mutate("Detect content regions", (doc) => {
        for (const result of results) doc.elements.find((e) => e.id === result.elementId).contentRegions = result.regions;
      });
      this.hooks.notify("Detection complete. Inspect each panel; confirm raster outlines or apply a correction before alignment.");
    } catch (error) { this.hooks.notify(detection.reason || error.message, detection.reason ? "info" : "error"); }
    finally { if (this.detection === detection) this.detection = null; this.pending = false; this.render(); }
  }
  draw() {
    this.overlay.replaceChildren();
    for (const panel of this.state.selected()) {
      if (!panel.visible || !["image", "plot"].includes(panel.type)) continue;
      let bounds;
      const preview = this.preview && panel.id === this.panelId;
      try { bounds = preview ? this.preview : regionBounds(panel, this.kind, false); } catch { continue; }
      const shape = document.createElementNS(NS, "polygon");
      shape.dataset.regionElement = panel.id;
      shape.setAttribute("points", regionCorners(panel, bounds).map((p) => p.join(",")).join(" "));
      shape.setAttribute("fill", "none");
      shape.setAttribute("stroke", preview ? "#2563eb" : this.kind === "panel" || panel.contentRegions[this.kind].confirmed ? "#008b9a" : "#b86e00");
      shape.setAttribute("stroke-width", "2");
      shape.setAttribute("stroke-dasharray", "6 3");
      shape.setAttribute("vector-effect", "non-scaling-stroke");
      this.overlay.append(shape);
    }
  }
}
