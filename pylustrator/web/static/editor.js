import { api, projectPath, escapeHTML as esc } from "./api.js";
import { clone, rectOf, union } from "./state.js";

const NS = "http://www.w3.org/2000/svg";
const svg = (tag, attrs = {}, text) => {
  const node = document.createElementNS(NS, tag);
  for (const [key, value] of Object.entries(attrs))
    if (value !== undefined) node.setAttribute(key, value);
  if (text !== undefined) node.textContent = text;
  return node;
};
const val = (n) => Math.round(n * 100) / 100;
const activeField = () =>
  ["INPUT", "TEXTAREA", "SELECT"].includes(document.activeElement?.tagName) ||
  document.activeElement?.isContentEditable;
export { activeField };
function literalColor(value) {
  if (value === "none") return value;
  if (
    !CSS.supports("color", value) ||
    /currentcolor|inherit|initial|unset|revert|var\(|env\(/i.test(value)
  )
    throw new Error(
      "Use a fixed color such as #175ca4, a named color, or none.",
    );
  const canvas = document.createElement("canvas");
  canvas.width = canvas.height = 1;
  const context = canvas.getContext("2d");
  context.fillStyle = value;
  context.fillRect(0, 0, 1, 1);
  const pixel = context.getImageData(0, 0, 1, 1).data;
  return (
    "#" +
    Array.from(pixel)
      .slice(0, pixel[3] === 255 ? 3 : 4)
      .map((channel) => channel.toString(16).padStart(2, "0"))
      .join("")
  );
}

export class FigureEditor {
  constructor(state, hooks) {
    this.state = state;
    this.hooks = hooks;
    this.project = null;
    this.mode = "select";
    this.scale = 1;
    this.offset = { x: 40, y: 40 };
    this.space = false;
    this.drag = null;
    this.guides = [];
    this.plots = new Map();
    this.pendingPlots = new Set();
    this.clipboard = [];
    this.viewport = document.querySelector("#canvas-viewport");
    this.artboard = document.querySelector("#artboard");
    this.artwork = document.querySelector("#artwork");
    this.overlay = document.querySelector("#selection-overlay");
    this.state.subscribe((kind) => this.render(kind));
    this.viewport.addEventListener("pointerdown", (e) => this.pointerDown(e));
    this.viewport.addEventListener("pointermove", (e) => this.pointerMove(e));
    this.viewport.addEventListener("pointerup", (e) => this.pointerUp(e));
    this.viewport.addEventListener("pointercancel", () => this.cancel());
    this.viewport.addEventListener("dblclick", (e) => {
      const element = this.state.document?.elements.find(
        (x) => x.id === e.target.closest("[data-element]")?.dataset.element,
      );
      if (element?.type === "text" && !element.locked) this.editText(element);
    });
    this.viewport.addEventListener(
      "wheel",
      (e) => {
        e.preventDefault();
        if (e.ctrlKey || e.metaKey)
          this.zoom(this.scale * Math.exp(-e.deltaY * 0.002), {
            x: e.clientX,
            y: e.clientY,
          });
        else {
          this.offset.x -= e.deltaX;
          this.offset.y -= e.deltaY;
          this.position();
        }
      },
      { passive: false },
    );
    document.addEventListener("keydown", (e) => this.keyDown(e));
    document.addEventListener("keyup", (e) => {
      if (e.code === "Space") {
        this.space = false;
        this.viewport.classList.toggle("panning", this.mode === "pan");
      }
    });
    window.addEventListener("blur", () => {
      this.space = false;
      this.cancel();
    });
    document
      .querySelectorAll("[data-tool]")
      .forEach((b) =>
        b.addEventListener("click", () => this.setMode(b.dataset.tool)),
      );
    document.querySelectorAll("[data-add]").forEach((b) =>
      b.addEventListener("click", () => {
        if (this.state.document)
          this.state.add(b.dataset.add, {
            x: 60 + ((this.state.document.elements.length * 8) % 120),
            y: 60 + ((this.state.document.elements.length * 8) % 120),
          });
      }),
    );
    document
      .querySelectorAll("[data-command]")
      .forEach((b) =>
        b.addEventListener("click", () => this.command(b.dataset.command)),
      );
    document
      .querySelector("#zoom-fit")
      .addEventListener("click", () => this.fit());
    document
      .querySelector("#zoom-in")
      .addEventListener("click", () => this.zoom(this.scale * 1.2));
    document
      .querySelector("#zoom-out")
      .addEventListener("click", () => this.zoom(this.scale / 1.2));
    for (const [id, cls] of [
      ["toggle-layers", "left-open"],
      ["toggle-properties", "right-open"],
    ])
      document.getElementById(id).addEventListener("click", (e) => {
        const open = document
          .querySelector(".figure-workspace")
          .classList.toggle(cls);
        e.currentTarget.setAttribute("aria-expanded", String(open));
      });
    new ResizeObserver(() => this.position()).observe(this.viewport);
  }
  setProject(project) {
    this.cancel();
    this.project = project;
    this.state.load(project.document);
    this.pendingFit = true;
    requestAnimationFrame(() => this.fit());
  }
  setMode(mode) {
    this.mode = mode;
    document.querySelectorAll("[data-tool]").forEach((b) => {
      b.classList.toggle("active", b.dataset.tool === mode);
      b.setAttribute("aria-pressed", String(b.dataset.tool === mode));
    });
    this.viewport.classList.toggle("panning", mode === "pan");
    this.render("selection");
  }
  fit() {
    if (!this.state.document) return;
    const box = this.viewport.getBoundingClientRect();
    if (box.width < 1 || box.height < 1) {
      this.pendingFit = true;
      return;
    }
    this.pendingFit = false;
    this.scale = Math.max(
      0.1,
      Math.min(
        1.5,
        (box.width - 64) / this.state.document.width,
        (box.height - 64) / this.state.document.height,
      ),
    );
    this.offset = {
      x: (box.width - this.state.document.width * this.scale) / 2,
      y: (box.height - this.state.document.height * this.scale) / 2,
    };
    this.position();
  }
  position() {
    if (this.pendingFit && this.viewport.getBoundingClientRect().width > 0) {
      this.fit();
      return;
    }
    this.artboard.style.transform = `translate(${this.offset.x}px,${this.offset.y}px) scale(${this.scale})`;
    document.querySelector("#zoom-value").textContent =
      `${Math.round(this.scale * 100)}%`;
    this.renderOverlay();
  }
  zoom(scale, client) {
    if (!this.state.document) return;
    scale = Math.max(0.1, Math.min(5, scale));
    const box = this.viewport.getBoundingClientRect(),
      cx = (client?.x ?? box.left + box.width / 2) - box.left,
      cy = (client?.y ?? box.top + box.height / 2) - box.top;
    this.offset.x = cx - ((cx - this.offset.x) * scale) / this.scale;
    this.offset.y = cy - ((cy - this.offset.y) * scale) / this.scale;
    this.scale = scale;
    this.position();
  }
  point(e) {
    const box = this.viewport.getBoundingClientRect();
    return {
      x: (e.clientX - box.left - this.offset.x) / this.scale,
      y: (e.clientY - box.top - this.offset.y) / this.scale,
    };
  }
  render(kind) {
    const doc = this.state.document;
    document.querySelector("#figure-empty").hidden = Boolean(doc);
    this.artboard.hidden = !doc;
    if (!doc) return;
    this.artboard.setAttribute("width", doc.width);
    this.artboard.setAttribute("height", doc.height);
    this.artboard.setAttribute("viewBox", `0 0 ${doc.width} ${doc.height}`);
    this.artboard.style.background = doc.background || "#fff";
    if (kind !== "selection" && kind !== "status") this.renderArtwork();
    this.renderOverlay();
    if (kind !== "preview") {
      this.renderLayers();
      this.renderInspector();
    }
    const selected = this.state.selected(),
      text = selected.length
        ? `${selected.length} selected${selected.some((e) => e.locked) ? " · locked objects excluded from edits" : ""}`
        : `${doc.name} · ${doc.width} × ${doc.height} px`;
    document.querySelector("#selection-status").textContent = text;
    document.querySelector("#undo").disabled = !this.state.undoStack.length;
    document.querySelector("#redo").disabled = !this.state.redoStack.length;
    document.querySelector("#undo").title =
      `Undo${this.state.undoStack.length ? `: ${this.state.undoStack.at(-1).label}` : ""} (Ctrl/⌘+Z)`;
  }
  renderArtwork() {
    this.artwork.replaceChildren();
    for (const el of this.state.document.elements) {
      if (!el.visible) continue;
      const g = svg("g", {
        "data-element": el.id,
        transform: `translate(${el.x} ${el.y}) rotate(${el.rotation || 0} ${el.width / 2} ${el.height / 2})`,
        opacity: el.style?.opacity ?? 1,
      });
      if (el.locked) g.classList.add("locked");
      const style = el.style || {},
        attrs = {
          fill: style.fill || "none",
          stroke: style.stroke || "none",
          "stroke-width": style.strokeWidth ?? 1,
        };
      if (el.type === "rect")
        g.append(svg("rect", { width: el.width, height: el.height, ...attrs }));
      if (el.type === "ellipse")
        g.append(
          svg("ellipse", {
            cx: el.width / 2,
            cy: el.height / 2,
            rx: el.width / 2,
            ry: el.height / 2,
            ...attrs,
          }),
        );
      if (el.type === "line" || el.type === "path") {
        const points = el.points || [
          [0, 0],
          [el.width, el.height],
        ];
        g.append(
          svg(el.closed ? "polygon" : "polyline", {
            points: points.map((p) => p.join(",")).join(" "),
            ...attrs,
            "stroke-linejoin": "round",
            "stroke-linecap": "round",
          }),
        );
        g.append(
          svg("polyline", {
            points: points.map((p) => p.join(",")).join(" "),
            fill: "none",
            stroke: "transparent",
            "stroke-width": Math.max(10 / this.scale, style.strokeWidth || 1),
          }),
        );
      }
      if (el.type === "text") {
        const size = style.fontSize || 24,
          align = style.textAlign || "left",
          x =
            align === "center"
              ? el.width / 2
              : align === "right"
                ? el.width
                : 0;
        const text = svg("text", {
          x,
          y: size,
          fill: style.fill || "#1f2937",
          "font-size": size,
          "font-family": style.fontFamily || "DejaVu Sans, sans-serif",
          "font-weight": style.fontWeight || "normal",
          "text-anchor":
            align === "center" ? "middle" : align === "right" ? "end" : "start",
        });
        String(el.text || "")
          .split("\n")
          .forEach((line, i) =>
            text.append(
              svg("tspan", { x, dy: i ? size * 1.25 : 0 }, line || " "),
            ),
          );
        g.append(
          svg("rect", {
            width: el.width,
            height: el.height,
            fill: "transparent",
          }),
          text,
        );
      }
      if (el.type === "image")
        g.append(
          svg("image", {
            width: el.width,
            height: el.height,
            href: projectPath(
              this.project,
              `/assets/${encodeURIComponent(el.assetId)}`,
            ),
            preserveAspectRatio: "none",
          }),
        );
      if (el.type === "plot") {
        const original = this.state.gesture?.document.elements.find(
          (item) => item.id === el.id,
        );
        const key = this.plotKey(original || el),
          entry = this.plots.get(key);
        g.append(
          svg("rect", { width: el.width, height: el.height, fill: "#fff" }),
        );
        if (entry?.url)
          g.append(
            svg("image", {
              width: el.width,
              height: el.height,
              href: entry.url,
              preserveAspectRatio: "none",
            }),
          );
        else {
          g.append(
            svg("rect", {
              width: el.width,
              height: el.height,
              fill: "#f6f8fb",
              stroke: "#d6dce4",
            }),
            svg(
              "text",
              {
                x: 16,
                y: 28,
                fill: entry?.error ? "#9e2c25" : "#5d6b7d",
                "font-size": 13,
              },
              entry?.error
                ? "Plot unavailable — check the data mapping"
                : "Rendering plot…",
            ),
          );
          if (!entry?.error) this.fetchPlot(original || el, key);
        }
      }
      this.artwork.append(g);
    }
  }
  plotKey(el) {
    return JSON.stringify([
      this.project.id,
      el.plot,
      val(el.width),
      val(el.height),
      this.project.datasets?.find((d) => d.id === el.plot?.datasetId)?.hash,
    ]);
  }
  async fetchPlot(el, key) {
    if (this.pendingPlots.has(key)) return;
    this.pendingPlots.add(key);
    const projectId = this.project.id;
    try {
      const blob = await api.post(
        "/api/render/plot",
        {
          projectId,
          plot: el.plot,
          width: Math.round(el.width),
          height: Math.round(el.height),
        },
        { binary: true },
      );
      this.plots.set(key, { url: URL.createObjectURL(blob) });
      if (this.plots.size > 40) {
        const [old, item] = this.plots.entries().next().value;
        if (item.url) URL.revokeObjectURL(item.url);
        this.plots.delete(old);
      }
    } catch (error) {
      this.plots.set(key, { error: error.message });
      if (this.project?.id === projectId)
        this.hooks.notify(`Plot could not render: ${error.message}`, "error");
    } finally {
      this.pendingPlots.delete(key);
      if (this.project?.id === projectId) this.renderArtwork();
    }
  }
  renderOverlay() {
    this.overlay.replaceChildren();
    const selected = this.state.selected(),
      editable = selected.filter((e) => !e.locked && e.visible);
    for (const el of selected) {
      if (!el.visible) continue;
      const b = rectOf(el);
      this.overlay.append(
        svg("rect", {
          ...b,
          class: "selection-box",
          "stroke-dasharray": el.locked ? "4 3" : "none",
        }),
      );
    }
    const box = union(editable);
    if (box) {
      const radius = 4 / this.scale;
      if (
        this.mode === "direct" &&
        editable.length === 1 &&
        ["line", "path"].includes(editable[0].type)
      ) {
        const el = editable[0],
          a = ((el.rotation || 0) * Math.PI) / 180;
        for (const [index, [px, py]] of (el.points || []).entries()) {
          const x =
              el.x +
              el.width / 2 +
              (px - el.width / 2) * Math.cos(a) -
              (py - el.height / 2) * Math.sin(a),
            y =
              el.y +
              el.height / 2 +
              (px - el.width / 2) * Math.sin(a) +
              (py - el.height / 2) * Math.cos(a);
          this.overlay.append(
            svg("circle", {
              cx: x,
              cy: y,
              r: radius + 1 / this.scale,
              "data-point": index,
              class: "point-handle",
            }),
          );
        }
      } else {
        for (const [handle, x, y] of [
          ["nw", box.x, box.y],
          ["ne", box.x + box.width, box.y],
          ["sw", box.x, box.y + box.height],
          ["se", box.x + box.width, box.y + box.height],
        ])
          this.overlay.append(
            svg("rect", {
              x: x - radius,
              y: y - radius,
              width: radius * 2,
              height: radius * 2,
              "data-handle": handle,
              class: "selection-handle",
            }),
          );
        const x = box.x + box.width / 2,
          y = box.y - 24 / this.scale;
        this.overlay.append(
          svg("line", {
            x1: x,
            y1: box.y,
            x2: x,
            y2: y,
            class: "selection-box",
          }),
          svg("circle", {
            cx: x,
            cy: y,
            r: radius + 1 / this.scale,
            "data-handle": "rotate",
            class: "selection-handle rotation-handle",
          }),
        );
      }
    }
    if (this.drag?.kind === "marquee") {
      const { start, current } = this.drag;
      if (current)
        this.overlay.append(
          svg("rect", {
            x: Math.min(start.x, current.x),
            y: Math.min(start.y, current.y),
            width: Math.abs(current.x - start.x),
            height: Math.abs(current.y - start.y),
            class: "marquee",
          }),
        );
    }
    const guides = document.querySelector("#guides");
    guides.replaceChildren();
    for (const guide of this.guides)
      guides.append(
        svg(
          "line",
          guide.axis === "x"
            ? {
                x1: guide.value,
                x2: guide.value,
                y1: 0,
                y2: this.state.document.height,
                class: "guide-line",
              }
            : {
                x1: 0,
                x2: this.state.document.width,
                y1: guide.value,
                y2: guide.value,
                class: "guide-line",
              },
        ),
      );
  }
  renderLayers() {
    const list = document.querySelector("#layers");
    list.replaceChildren();
    document.querySelector("#layer-count").textContent = String(
      this.state.document.elements.length,
    );
    if (!this.state.document.elements.length) {
      const p = document.createElement("p");
      p.className = "muted";
      p.textContent =
        "Add text, a shape, or a data plot. Layers appear here in paint order.";
      p.style.padding = "8px";
      list.append(p);
    }
    for (const el of [...this.state.document.elements].reverse()) {
      const row = document.createElement("div");
      row.className = `layer-row${this.state.selection.has(el.id) ? " selected" : ""}${el.visible ? "" : " hidden-layer"}`;
      const select = document.createElement("button");
      select.className = "layer-select";
      select.title = `${el.name} · ${el.type}${el.groupId ? " · grouped" : ""}`;
      select.setAttribute(
        "aria-pressed",
        String(this.state.selection.has(el.id)),
      );
      select.innerHTML = `<span class="layer-type" aria-hidden="true">${{ text: "T", rect: "▢", ellipse: "○", line: "╱", path: "⌁", plot: "▥", image: "▧" }[el.type] || "·"}</span><span>${esc(el.name)}</span>`;
      select.addEventListener("click", (e) =>
        this.state.select([el.id], {
          toggle: e.shiftKey,
          group: this.mode !== "direct",
        }),
      );
      row.append(select);
      for (const [key, on, off] of [
        ["visible", "Hide", "Show"],
        ["locked", "Unlock", "Lock"],
      ]) {
        const b = document.createElement("button");
        b.className = "layer-toggle";
        b.title = `${el[key] ? on : off} ${el.name}`;
        b.setAttribute("aria-label", b.title);
        b.textContent =
          key === "visible" ? (el.visible ? "●" : "○") : el.locked ? "L" : "–";
        b.addEventListener("click", () =>
          this.state.mutate(`${el[key] ? on : off} ${el.name}`, () => {
            el[key] = !el[key];
          }),
        );
        row.append(b);
      }
      list.append(row);
    }
  }
  renderInspector() {
    const target = document.querySelector("#inspector"),
      selected = this.state.selected(),
      editable = selected.filter((e) => !e.locked),
      doc = this.state.document;
    if (!selected.length) {
      target.innerHTML = `<div class="inspector-content"><h3>Artboard</h3><label>Figure name<input data-doc="name" value="${esc(doc.name)}" maxlength="200"></label><div class="field-pair"><label>Width, px<input data-doc="width" type="number" min="64" max="8192" value="${doc.width}"></label><label>Height, px<input data-doc="height" type="number" min="64" max="8192" value="${doc.height}"></label></div><label>Background<input data-doc="background" type="color" value="${esc(doc.background || "#ffffff")}"></label><p class="small-note">Select an object to change its geometry and appearance. Shift-click selects multiple objects.</p><h3>Data-driven figures</h3><p class="small-note">Import data, inspect the source, then map columns to a plot.</p><button data-to-data>Open Data workspace</button></div>`;
      target.querySelector("[data-to-data]").onclick = () =>
        this.hooks.go("data");
      target.querySelectorAll("[data-doc]").forEach((input) =>
        input.addEventListener("change", () => {
          let value = input.value;
          if (["width", "height"].includes(input.dataset.doc)) {
            value = Number(value);
            if (!Number.isFinite(value) || value < 64 || value > 8192) {
              this.hooks.notify(
                "Artboard dimensions must be between 64 and 8192 pixels.",
                "error",
              );
              this.renderInspector();
              return;
            }
          }
          this.state.mutate(
            "Edit artboard",
            (d) => (d[input.dataset.doc] = value),
          );
        }),
      );
      return;
    }
    const first = selected[0],
      mixed = (key) =>
        selected.every((e) => e[key] === first[key]) ? val(first[key]) : "",
      styleMixed = (key) =>
        selected.every((e) => e.style?.[key] === first.style?.[key])
          ? (first.style?.[key] ?? "")
          : "";
    target.innerHTML = `<div class="inspector-content"><h3>${selected.length === 1 ? esc(first.type[0].toUpperCase() + first.type.slice(1)) : `${selected.length} objects`}</h3>${editable.length ? "" : `<p class="small-note">The selection is locked. Unlock it in Layers to edit.</p>`}${selected.length === 1 ? `<label>Layer name<input data-prop="name" value="${esc(first.name)}" maxlength="200" ${first.locked ? "disabled" : ""}></label>` : ""}<div class="field-pair">${[
      ["x", "X, px"],
      ["y", "Y, px"],
      [
        "width",
        ["line", "path"].includes(first.type) ? "Frame width, px" : "Width, px",
      ],
      [
        "height",
        ["line", "path"].includes(first.type)
          ? "Frame height, px"
          : "Height, px",
      ],
    ]
      .map(
        ([key, label]) =>
          `<label>${label}<input type="number" step="0.1" data-prop="${key}" value="${mixed(key)}" placeholder="Mixed" ${!editable.length ? "disabled" : ""}></label>`,
      )
      .join(
        "",
      )}</div><label>Rotation, degrees<input type="number" step="1" data-prop="rotation" value="${mixed("rotation")}" placeholder="Mixed" ${!editable.length ? "disabled" : ""}></label>${selected.length === 1 && first.type === "text" ? `<label>Text<textarea data-prop="text" rows="3" ${first.locked ? "disabled" : ""}>${esc(first.text)}</textarea></label>` : ""}<h3>Appearance</h3><div class="field-pair"><label>Fill<input data-style="fill" value="${esc(styleMixed("fill"))}" placeholder="Mixed or none" ${!editable.length ? "disabled" : ""}></label><label>Stroke<input data-style="stroke" value="${esc(styleMixed("stroke"))}" placeholder="Mixed or none" ${!editable.length ? "disabled" : ""}></label><label>Stroke, px<input type="number" min="0" max="100" step="0.5" data-style="strokeWidth" value="${esc(styleMixed("strokeWidth"))}" ${!editable.length ? "disabled" : ""}></label><label>Opacity<input type="number" min="0" max="1" step="0.1" data-style="opacity" value="${esc(styleMixed("opacity"))}" ${!editable.length ? "disabled" : ""}></label></div>${selected.some((e) => e.type === "text") ? `<label>Font size, px<input type="number" min="1" max="300" data-style="fontSize" value="${esc(styleMixed("fontSize"))}" ${!editable.length ? "disabled" : ""}></label>` : ""}<h3>Align & distribute</h3><label class="check"><input type="checkbox" id="align-artboard"> Align to artboard</label><div class="align-grid">${["left", "center", "right", "top", "middle", "bottom"].map((mode) => `<button data-align="${mode}" ${!editable.length ? "disabled" : ""}>${mode[0].toUpperCase() + mode.slice(1)}</button>`).join("")}</div><div class="field-pair"><button data-distribute="x" title="Select at least three unlocked objects" ${editable.length < 3 ? "disabled" : ""}>Space X</button><button data-distribute="y" title="Select at least three unlocked objects" ${editable.length < 3 ? "disabled" : ""}>Space Y</button></div>${first.type === "plot" ? `<p class="small-note">Source: ${esc(this.project.datasets?.find((d) => d.id === first.plot?.datasetId)?.name || first.plot?.datasetId)}<br>${esc(first.plot?.kind)} · ${esc(first.plot?.x || "")} / ${esc(first.plot?.y || "")}</p>` : ""}${this.mode === "direct" && !selected.some((e) => ["line", "path"].includes(e.type)) ? '<p class="small-note">Point editing applies to line and path vertices. Use Select for this object.</p>' : ""}</div>`;
    target.querySelectorAll("[data-prop],[data-style]").forEach((input) =>
      input.addEventListener("change", () => {
        const key = input.dataset.prop || input.dataset.style;
        let value = input.type === "number" ? Number(input.value) : input.value;
        if (input.type === "number" && !Number.isFinite(value)) return;
        if (["width", "height"].includes(key) && value <= 0) {
          this.hooks.notify("Width and height must be positive.", "error");
          this.renderInspector();
          return;
        }
        if (input.dataset.style && ["fill", "stroke"].includes(key)) {
          try {
            value = literalColor(value);
          } catch (error) {
            this.hooks.notify(error.message, "error");
            return;
          }
        }
        this.state.updateSelection(
          input.dataset.style ? { style: { [key]: value } } : { [key]: value },
        );
      }),
    );
    if (
      selected.length === 1 &&
      ["line", "path"].includes(first.type) &&
      this.mode === "direct"
    ) {
      this.pointIndex = Math.max(
        0,
        Math.min(first.points.length - 1, this.pointIndex || 0),
      );
      const point = first.points[this.pointIndex],
        section = document.createElement("section");
      section.className = "stack";
      section.innerHTML = `<h3>Direct point editing</h3><label>Point number<input id="point-index" type="number" min="1" max="${first.points.length}" step="1" value="${this.pointIndex + 1}"></label><div class="field-pair"><label>Point X, px<input data-point-coordinate="0" type="number" step="0.1" value="${val(point[0])}" ${first.locked ? "disabled" : ""}></label><label>Point Y, px<input data-point-coordinate="1" type="number" step="0.1" value="${val(point[1])}" ${first.locked ? "disabled" : ""}></label></div><p class="small-note">Coordinates are local to the path frame. Points may extend beyond it; selection and alignment use the actual points.</p>`;
      target.querySelector(".inspector-content").append(section);
      section.querySelector("#point-index").onchange = (e) => {
        this.pointIndex = Math.max(
          0,
          Math.min(
            first.points.length - 1,
            Math.floor(Number(e.target.value) || 1) - 1,
          ),
        );
        this.renderInspector();
      };
      section.querySelectorAll("[data-point-coordinate]").forEach(
        (input) =>
          (input.onchange = () => {
            const value = Number(input.value);
            if (!Number.isFinite(value) || Math.abs(value) > 1000000) {
              this.hooks.notify(
                "Point coordinates must be finite and within one million pixels.",
                "error",
              );
              return;
            }
            this.state.mutate(
              "Edit point coordinates",
              (document) =>
                (document.elements.find((el) => el.id === first.id).points[
                  this.pointIndex
                ][Number(input.dataset.pointCoordinate)] = value),
            );
          }),
      );
    }
    if (selected.every((element) => ["plot", "image"].includes(element.type)))
      target
        .querySelectorAll(
          '[data-style="fill"],[data-style="stroke"],[data-style="strokeWidth"]',
        )
        .forEach((input) => {
          input.disabled = true;
          input.title =
            "This embedded content uses its own colors. Opacity and geometry remain editable.";
        });
    target
      .querySelectorAll("[data-align]")
      .forEach(
        (b) =>
          (b.onclick = () =>
            this.state.align(
              b.dataset.align,
              target.querySelector("#align-artboard").checked,
            )),
      );
    target
      .querySelectorAll("[data-distribute]")
      .forEach(
        (b) => (b.onclick = () => this.state.distribute(b.dataset.distribute)),
      );
  }
  pointerDown(e) {
    if (!this.state.document || e.button > 1) return;
    e.preventDefault();
    this.viewport.focus({ preventScroll: true });
    const p = this.point(e);
    this.viewport.setPointerCapture(e.pointerId);
    if (this.space || this.mode === "pan" || e.button === 1) {
      this.drag = {
        kind: "pan",
        client: { x: e.clientX, y: e.clientY },
        offset: { ...this.offset },
      };
      this.viewport.classList.add("dragging");
      return;
    }
    const handle = e.target.dataset.handle,
      point = e.target.dataset.point;
    if (handle || point !== undefined) {
      if (point !== undefined) this.pointIndex = Number(point);
      const items = this.state
        .selected({ editable: true })
        .filter((x) => x.visible);
      if (!items.length) return;
      this.drag = {
        kind:
          point !== undefined
            ? "point"
            : handle === "rotate"
              ? "rotate"
              : "resize",
        start: p,
        items: clone(items),
        box: union(items),
        handle,
        index: Number(point),
      };
      this.state.beginGesture(
        point !== undefined
          ? "Edit point"
          : handle === "rotate"
            ? "Rotate selection"
            : "Resize selection",
      );
      return;
    }
    let id = e.target.closest("[data-element]")?.dataset.element;
    if (e.altKey) {
      const hits = this.state.document.elements
        .filter((el) => {
          if (!el.visible || el.locked) return false;
          const b = rectOf(el);
          return (
            p.x >= b.x &&
            p.x <= b.x + b.width &&
            p.y >= b.y &&
            p.y <= b.y + b.height
          );
        })
        .reverse();
      const current = hits.findIndex((el) => this.state.selection.has(el.id));
      id = hits[(current + 1) % hits.length]?.id;
    }
    const el = this.state.document.elements.find((x) => x.id === id);
    if (el) {
      if (el.locked) {
        this.state.select([id], { toggle: e.shiftKey, group: false });
        return;
      }
      if (e.shiftKey)
        this.state.select([id], {
          toggle: true,
          group: this.mode !== "direct",
        });
      else if (!this.state.selection.has(id))
        this.state.select([id], { group: this.mode !== "direct" });
      const items = this.state
        .selected({ editable: true })
        .filter((x) => x.visible);
      this.drag = {
        kind: "move",
        start: p,
        items: clone(items),
        box: union(items),
      };
      this.state.beginGesture("Move selection");
    } else {
      const initial = e.shiftKey ? [...this.state.selection] : [];
      if (!e.shiftKey) this.state.select([]);
      this.drag = { kind: "marquee", start: p, current: p, initial };
    }
  }
  pointerMove(e) {
    if (!this.drag) return;
    const drag = this.drag,
      p = this.point(e);
    if (drag.kind === "pan") {
      this.offset = {
        x: drag.offset.x + e.clientX - drag.client.x,
        y: drag.offset.y + e.clientY - drag.client.y,
      };
      this.position();
      return;
    }
    if (drag.kind === "marquee") {
      drag.current = p;
      this.renderOverlay();
      return;
    }
    const dx = p.x - drag.start.x,
      dy = p.y - drag.start.y;
    if (drag.kind === "move") {
      let offset = { x: dx, y: dy };
      if (e.shiftKey)
        offset =
          Math.abs(dx) > Math.abs(dy) ? { x: dx, y: 0 } : { x: 0, y: dy };
      offset = this.snap(
        drag.box,
        offset,
        drag.items.map((x) => x.id),
      );
      this.state.preview((doc) => {
        for (const original of drag.items) {
          const el = doc.elements.find((x) => x.id === original.id);
          el.x = original.x + offset.x;
          el.y = original.y + offset.y;
        }
      });
    }
    if (drag.kind === "resize") {
      const b = drag.box,
        left = drag.handle.includes("w"),
        top = drag.handle.includes("n"),
        anchor = {
          x: left ? b.x + b.width : b.x,
          y: top ? b.y + b.height : b.y,
        };
      let sx = Math.max(
          0.02,
          (b.width + (left ? -dx : dx)) / Math.max(1, b.width),
        ),
        sy = Math.max(
          0.02,
          (b.height + (top ? -dy : dy)) / Math.max(1, b.height),
        );
      if (e.shiftKey) sx = sy = Math.max(sx, sy);
      this.state.preview((doc) => {
        for (const original of drag.items) {
          const el = doc.elements.find((x) => x.id === original.id);
          el.x = anchor.x + (original.x - anchor.x) * sx;
          el.y = anchor.y + (original.y - anchor.y) * sy;
          el.width = Math.max(1, original.width * sx);
          el.height = Math.max(1, original.height * sy);
          if (el.points)
            el.points = original.points.map(([x, y]) => [x * sx, y * sy]);
        }
      });
    }
    if (drag.kind === "rotate") {
      const center = {
          x: drag.box.x + drag.box.width / 2,
          y: drag.box.y + drag.box.height / 2,
        },
        angle =
          ((Math.atan2(p.y - center.y, p.x - center.x) -
            Math.atan2(drag.start.y - center.y, drag.start.x - center.x)) *
            180) /
          Math.PI,
        delta = e.shiftKey ? Math.round(angle / 15) * 15 : angle,
        a = (delta * Math.PI) / 180;
      this.state.preview((doc) => {
        for (const original of drag.items) {
          const el = doc.elements.find((x) => x.id === original.id),
            cx = original.x + original.width / 2 - center.x,
            cy = original.y + original.height / 2 - center.y;
          el.x =
            center.x + cx * Math.cos(a) - cy * Math.sin(a) - original.width / 2;
          el.y =
            center.y +
            cx * Math.sin(a) +
            cy * Math.cos(a) -
            original.height / 2;
          el.rotation = (original.rotation || 0) + delta;
        }
      });
    }
    if (drag.kind === "point") {
      const original = drag.items[0],
        a = (-(original.rotation || 0) * Math.PI) / 180,
        cx = p.x - original.x - original.width / 2,
        cy = p.y - original.y - original.height / 2;
      this.state.preview((doc) => {
        const el = doc.elements.find((x) => x.id === original.id);
        el.points[drag.index] = [
          cx * Math.cos(a) - cy * Math.sin(a) + original.width / 2,
          cx * Math.sin(a) + cy * Math.cos(a) + original.height / 2,
        ];
      });
    }
  }
  snap(box, offset, ids) {
    this.guides = [];
    if (!box || !document.querySelector("#snap-enabled").checked) return offset;
    const doc = this.state.document,
      targets = {
        x: [0, doc.width / 2, doc.width],
        y: [0, doc.height / 2, doc.height],
      };
    for (const el of doc.elements)
      if (el.visible && !ids.includes(el.id)) {
        const b = rectOf(el);
        targets.x.push(b.x, b.x + b.width / 2, b.x + b.width);
        targets.y.push(b.y, b.y + b.height / 2, b.y + b.height);
      }
    for (const axis of ["x", "y"]) {
      const dim = axis === "x" ? "width" : "height",
        points = [
          box[axis] + offset[axis],
          box[axis] + offset[axis] + box[dim] / 2,
          box[axis] + offset[axis] + box[dim],
        ],
        candidate = targets[axis]
          .flatMap((value) =>
            points.map((point) => ({ value, diff: value - point })),
          )
          .sort((a, b) => Math.abs(a.diff) - Math.abs(b.diff))[0];
      if (candidate && Math.abs(candidate.diff) < 5 / this.scale) {
        offset[axis] += candidate.diff;
        this.guides.push({ axis, value: candidate.value });
      }
    }
    return offset;
  }
  pointerUp(e) {
    if (!this.drag) return;
    const drag = this.drag;
    if (drag.kind === "marquee") {
      const p = this.point(e),
        x = Math.min(p.x, drag.start.x),
        y = Math.min(p.y, drag.start.y),
        right = Math.max(p.x, drag.start.x),
        bottom = Math.max(p.y, drag.start.y);
      const ids = this.state.document.elements
        .filter((el) => {
          const b = rectOf(el);
          return (
            el.visible &&
            !el.locked &&
            b.x >= x &&
            b.y >= y &&
            b.x + b.width <= right &&
            b.y + b.height <= bottom
          );
        })
        .map((el) => el.id);
      this.state.select([...new Set([...drag.initial, ...ids])]);
    }
    this.drag = null;
    this.guides = [];
    this.viewport.classList.remove("dragging");
    this.state.endGesture();
    this.renderOverlay();
  }
  cancel() {
    if (this.drag) {
      this.drag = null;
      this.guides = [];
      this.state.endGesture(true);
      this.viewport.classList.remove("dragging");
      this.renderOverlay();
    }
  }
  command(name) {
    const commands = {
      undo: () => this.state.undo(),
      redo: () => this.state.redo(),
      group: () => this.state.group(),
      ungroup: () => this.state.ungroup(),
      duplicate: () => this.state.duplicate(),
      delete: () => this.state.remove(),
      forward: () => this.state.order(1),
      backward: () => this.state.order(-1),
    };
    commands[name]?.();
  }
  keyDown(e) {
    if (
      !this.state.document ||
      document.querySelector("#main").inert ||
      document.querySelector("#view-figure").hidden ||
      document.querySelector("dialog[open]") ||
      activeField()
    )
      return;
    const mod = e.ctrlKey || e.metaKey,
      key = e.key.toLowerCase();
    if (e.code === "Space") {
      e.preventDefault();
      this.space = true;
      this.viewport.classList.add("panning");
      return;
    }
    if (e.key === "Escape") {
      this.cancel();
      this.state.select([]);
      return;
    }
    if (mod && key === "s") {
      e.preventDefault();
      this.hooks.save();
      return;
    }
    if (mod && key === "z") {
      e.preventDefault();
      e.shiftKey ? this.state.redo() : this.state.undo();
      return;
    }
    if (mod && key === "y") {
      e.preventDefault();
      this.state.redo();
      return;
    }
    if (mod && key === "g") {
      e.preventDefault();
      e.shiftKey ? this.state.ungroup() : this.state.group();
      return;
    }
    if (mod && key === "d") {
      e.preventDefault();
      this.state.duplicate();
      return;
    }
    if (mod && key === "a") {
      e.preventDefault();
      this.state.select(
        this.state.document.elements
          .filter((x) => x.visible && !x.locked)
          .map((x) => x.id),
      );
      return;
    }
    if (mod && key === "c") {
      if (this.state.selection.size) {
        e.preventDefault();
        this.clipboard = clone(this.state.selected());
      }
      return;
    }
    if (mod && key === "v") {
      if (this.clipboard.length) {
        e.preventDefault();
        const copies = [];
        this.state.mutate("Paste selection", (doc) => {
          for (const original of this.clipboard) {
            const el = clone(original);
            el.id = crypto.randomUUID();
            el.x += 24;
            el.y += 24;
            el.groupId = null;
            el.locked = false;
            doc.elements.push(el);
            copies.push(el.id);
          }
        });
        this.state.select(copies);
      }
      return;
    }
    if (mod) return;
    if (key === "v") {
      this.setMode("select");
      return;
    }
    if (key === "a") {
      this.setMode("direct");
      return;
    }
    if (e.key === "Delete" || e.key === "Backspace") {
      e.preventDefault();
      this.state.remove();
      return;
    }
    if (["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown"].includes(e.key)) {
      e.preventDefault();
      const amount = e.shiftKey ? 10 : 1;
      this.state.mutate("Nudge selection", (doc) => {
        for (const el of doc.elements)
          if (this.state.selection.has(el.id) && !el.locked) {
            if (e.key === "ArrowLeft") el.x -= amount;
            if (e.key === "ArrowRight") el.x += amount;
            if (e.key === "ArrowUp") el.y -= amount;
            if (e.key === "ArrowDown") el.y += amount;
          }
      });
    }
  }
  editText(el) {
    const rect = this.artboard.getBoundingClientRect(),
      form = document.createElement("form");
    form.className = "editor-inline-text";
    form.style.left = `${Math.max(8, Math.min(window.innerWidth - 280, rect.left + el.x * this.scale))}px`;
    form.style.top = `${Math.max(8, Math.min(window.innerHeight - 160, rect.top + el.y * this.scale))}px`;
    form.innerHTML = `<textarea aria-label="Edit text">${esc(el.text)}</textarea><small>Ctrl/⌘ + Enter to apply · Escape to cancel</small><button type="submit">Apply text</button>`;
    document.body.append(form);
    const textarea = form.querySelector("textarea");
    textarea.focus();
    textarea.select();
    form.addEventListener("submit", (e) => {
      e.preventDefault();
      this.state.mutate(
        "Edit text",
        (doc) =>
          (doc.elements.find((x) => x.id === el.id).text = textarea.value),
      );
      form.remove();
      this.viewport.focus();
    });
    form.addEventListener("keydown", (e) => {
      e.stopPropagation();
      if (e.key === "Escape") {
        form.remove();
        this.viewport.focus();
      }
      if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) form.requestSubmit();
    });
  }
}
