import { api, busy, escapeHTML as esc, download } from "./api.js";
const NS = "http://www.w3.org/2000/svg";

export class NativeWorkspace {
  constructor(hooks) {
    this.hooks = hooks;
    this.sessions = [];
    this.session = null;
    this.selected = null;
    this.drag = null;
    this.el = document.querySelector("#native-workspace");
    this.sequence = 0;
  }
  async discover() {
    try {
      const response = await api.get("/api/sessions");
      this.sessions = response.sessions || [];
      document.querySelector("[data-view=native]").hidden =
        !this.sessions.length;
      return this.sessions;
    } catch {
      return [];
    }
  }
  async mount() {
    this.el.innerHTML =
      '<div class="research-main"><div class="loading-placeholder" role="status">Loading attached Matplotlib figures…</div></div>';
    await this.discover();
    if (!this.sessions.length) {
      this.el.innerHTML =
        '<div class="empty-state"><h2>Attach a Matplotlib figure</h2><p>From a trusted Python process, call <code>pylustrator.web.serve(fig)</code>. The figure remains a live Matplotlib figure and can be edited here without Qt.</p></div>';
      return;
    }
    this.el.innerHTML = `<aside class="research-sidebar"><h2>Matplotlib figures</h2><label>Attached figure<select id="native-session">${this.sessions.map((s) => `<option value="${esc(s.id)}">${esc(s.name || s.id)}</option>`).join("")}</select></label><div id="native-artist-list" class="native-artist-list"></div></aside><div class="native-canvas-wrap"><div id="native-frame" class="native-frame"><img id="native-image" src="/api/sessions/${encodeURIComponent(this.session?.id || this.sessions[0].id)}/render.svg" alt="Live Matplotlib figure"><svg id="native-overlay" xmlns="http://www.w3.org/2000/svg" aria-label="Figure artist selection overlay"></svg></div></div><aside class="research-controls"><h3>Native artist properties</h3><div id="native-properties"><p class="muted">Select an artist in the list or on the figure.</p></div><hr><div class="inline-actions"><button id="native-undo" disabled>Undo</button><button id="native-redo" disabled>Redo</button><button id="native-refresh">Refresh</button></div><p id="native-status" class="inline-status" role="status"></p><label>Export attached figure<select id="native-export-format"><option value="svg">SVG vector</option><option value="png">PNG image</option><option value="pdf">PDF figure</option><option value="python">Python changes</option></select></label><button id="native-export" class="full-width">Download attached figure</button><p class="small-note">Python changes replay against the original Figure; keep the original plotting script.</p><p class="small-note">Native edits use the existing Matplotlib adapter transaction engine. Unsupported operations report their capability reason.</p></aside>`;
    this.el.querySelector("#native-session").onchange = (e) =>
      this.load(e.target.value);
    this.el.querySelector("#native-undo").onclick = () =>
      this.command({ operation: "undo" });
    this.el.querySelector("#native-redo").onclick = () =>
      this.command({ operation: "redo" });
    this.el.querySelector("#native-refresh").onclick = () =>
      this.load(this.session.id);
    this.el.querySelector("#native-export").onclick = (e) =>
      busy(
        e.currentTarget,
        () => this.export(this.el.querySelector("#native-export-format").value),
        "Exporting…",
      ).catch((error) => this.hooks.notify(error.message, "error"));
    const overlay = this.el.querySelector("#native-overlay");
    overlay.onpointerdown = (e) => this.pointerDown(e);
    overlay.onpointermove = (e) => this.pointerMove(e);
    overlay.onpointerup = (e) => this.pointerUp(e);
    overlay.onpointercancel = () => {
      this.drag = null;
      this.renderOverlay();
    };
    await this.load(this.session?.id || this.sessions[0].id);
  }
  async load(id) {
    const sequence = ++this.sequence;
    try {
      const session = await api.get(`/api/sessions/${encodeURIComponent(id)}`);
      if (sequence !== this.sequence) return;
      this.session = session;
      if (!session.artists.some((a) => a.id === this.selected))
        this.selected = null;
      this.render();
    } catch (error) {
      this.hooks.notify(error.message, "error");
    }
  }
  render() {
    if (!this.session) return;
    const s = this.session;
    this.el.querySelector("#native-image").src =
      `/api/sessions/${encodeURIComponent(s.id)}/render.svg?revision=${s.revision}`;
    this.el.querySelector("#native-frame").style.width = `${s.width}px`;
    this.el
      .querySelector("#native-overlay")
      .setAttribute("viewBox", `0 0 ${s.width} ${s.height}`);
    this.el.querySelector("#native-session").value = s.id;
    const list = this.el.querySelector("#native-artist-list");
    list.replaceChildren();
    for (const artist of s.artists) {
      const b = document.createElement("button");
      b.textContent = `${artist.name || artist.type} · ${artist.type}`;
      b.classList.toggle("selected", artist.id === this.selected);
      b.onclick = () => {
        this.selected = artist.id;
        this.renderProperties();
        this.renderOverlay();
        this.el
          .querySelectorAll("#native-artist-list button")
          .forEach((x) => x.classList.toggle("selected", x === b));
      };
      list.append(b);
    }
    this.el.querySelector("#native-undo").disabled = !s.canUndo;
    this.el.querySelector("#native-redo").disabled = !s.canRedo;
    this.el.querySelector("#native-status").textContent =
      `Revision ${s.revision} · ${s.artists.length} artists`;
    if (!document.querySelector("#view-native").hidden) {
      document.querySelector("#undo").disabled = !s.canUndo;
      document.querySelector("#redo").disabled = !s.canRedo;
    }
    this.renderOverlay();
    this.renderProperties();
  }
  artist() {
    return this.session?.artists.find((a) => a.id === this.selected);
  }
  renderOverlay() {
    const overlay = this.el.querySelector("#native-overlay");
    if (!overlay || !this.session) return;
    overlay.replaceChildren();
    for (const artist of this.session.artists) {
      if (!artist.visible || !Array.isArray(artist.bounds)) continue;
      const [x, y, w, h] = artist.bounds,
        rect = document.createElementNS(NS, "rect");
      for (const [key, value] of Object.entries({
        x,
        y,
        width: Math.max(4, w),
        height: Math.max(4, h),
        fill: "transparent",
        stroke: artist.id === this.selected ? "#1e88e5" : "transparent",
        "stroke-width": 1.5,
        "data-artist": artist.id,
      }))
        rect.setAttribute(key, value);
      overlay.append(rect);
    }
    const a = this.artist();
    if (a?.points)
      for (const point of a.points) {
        const c = document.createElementNS(NS, "circle");
        for (const [key, value] of Object.entries({
          cx: point.x,
          cy: point.y,
          r: 4,
          fill: "#fff",
          stroke: "#175ca4",
          "stroke-width": 1.5,
          "data-point": point.key,
        }))
          c.setAttribute(key, value);
        overlay.append(c);
      }
  }
  renderProperties() {
    const target = this.el.querySelector("#native-properties"),
      a = this.artist();
    if (!a) {
      target.innerHTML =
        '<p class="muted">Select an artist in the list or on the figure.</p>';
      return;
    }
    const support = (op) =>
      a.capabilities?.[op]?.supported === true && !a.locked;
    target.innerHTML = `<form id="native-property-form" class="stack"><p class="small-note">${esc(a.name || a.type)} · ${esc(a.type)}</p><div class="field-pair"><label>Move X, px<input name="dx" type="number" value="0" step="1"></label><label>Move Y, px<input name="dy" type="number" value="0" step="1"></label></div><button type="submit" ${support("translate") ? "" : "disabled"} title="${esc(a.capabilities?.translate?.reason || "")}">Move artist</button>${a.text !== undefined ? `<label>Text<textarea id="native-text" rows="3">${esc(a.text)}</textarea></label><button type="button" id="native-text-apply" ${a.locked ? "disabled" : ""}>Apply text</button>` : ""}<label>Rotation, degrees<input id="native-angle" type="number" value="15"></label><button id="native-rotate" type="button" ${support("rotate") ? "" : "disabled"} title="${esc(a.capabilities?.rotate?.reason || "")}">Rotate artist</button><div class="field-pair"><button id="native-visible" type="button">${a.visible ? "Hide" : "Show"}</button><button id="native-lock" type="button">${a.locked ? "Unlock" : "Lock"}</button></div></form>`;
    target.querySelector("form").onsubmit = (e) => {
      e.preventDefault();
      const form = new FormData(e.currentTarget);
      this.command(
        {
          operation: "translate",
          dx: Number(form.get("dx")),
          dy: Number(form.get("dy")),
        },
        e.submitter,
      );
    };
    target
      .querySelector("#native-text-apply")
      ?.addEventListener("click", (e) =>
        this.command(
          {
            operation: "text",
            text: target.querySelector("#native-text").value,
          },
          e.currentTarget,
        ),
      );
    target.querySelector("#native-rotate").onclick = (e) =>
      this.command(
        {
          operation: "rotate",
          angle: Number(target.querySelector("#native-angle").value),
        },
        e.currentTarget,
      );
    target.querySelector("#native-visible").onclick = (e) =>
      this.command(
        {
          operation: "visible",
          properties: { visible: !a.visible },
          visible: !a.visible,
        },
        e.currentTarget,
      );
    target.querySelector("#native-lock").onclick = (e) =>
      this.command(
        {
          operation: "lock",
          properties: { locked: !a.locked },
          locked: !a.locked,
        },
        e.currentTarget,
      );
    this.renderPointControls(target, a);
  }
  renderPointControls(target, artist) {
    if (
      !artist.points?.length ||
      artist.capabilities?.edit_points?.supported !== true
    )
      return;
    const section = document.createElement("section");
    section.className = "stack";
    section.innerHTML = `<h3>Direct point coordinates</h3><label>Handle<select id="native-point-key">${artist.points.map((point) => `<option value="${point.key}">Handle ${point.key}</option>`).join("")}</select></label><div class="field-pair"><label>Figure X, px<input id="native-point-x" type="number" step="0.1"></label><label>Figure Y, px<input id="native-point-y" type="number" step="0.1"></label></div><button id="native-point-apply" ${artist.locked ? "disabled" : ""}>Apply point coordinates</button>`;
    target.append(section);
    const select = section.querySelector("#native-point-key"),
      update = () => {
        const point = artist.points.find((p) => p.key === Number(select.value));
        section.querySelector("#native-point-x").value =
          Math.round(point.x * 1000) / 1000;
        section.querySelector("#native-point-y").value =
          Math.round(point.y * 1000) / 1000;
      };
    select.onchange = update;
    update();
    section.querySelector("#native-point-apply").onclick = (e) =>
      this.command(
        {
          operation: "points",
          points: [
            {
              key: Number(select.value),
              x: Number(section.querySelector("#native-point-x").value),
              y: Number(section.querySelector("#native-point-y").value),
            },
          ],
        },
        e.currentTarget,
      );
  }
  async export(format) {
    if (!this.session) throw new Error("Choose an attached figure first.");
    const session = this.session,
      route = format === "python" ? "changes.py" : `render.${format}`;
    const blob = await api.get(
      `/api/sessions/${encodeURIComponent(session.id)}/${route}`,
      { binary: true },
    );
    download(
      blob,
      `${session.name || "matplotlib-figure"}${format === "python" ? "-changes.py" : "." + format}`,
    );
    this.hooks.notify(
      format === "python"
        ? "Downloaded native changes. Replay them against the original Figure."
        : `Exported the attached Matplotlib figure as ${format.toUpperCase()}.`,
    );
  }
  async command(command, button) {
    if (!this.session) return;
    const sessionId = this.session.id;
    try {
      await busy(
        button,
        async () => {
          const result = await api.post(
            `/api/sessions/${encodeURIComponent(this.session.id)}/commands`,
            {
              revision: this.session.revision,
              targets: this.selected ? [this.selected] : [],
              ...command,
            },
          );
          if (this.session?.id !== sessionId) return;
          const updated = result.session || result;
          if (updated.revision < this.session.revision) return;
          this.session = updated;
          this.render();
          this.hooks.notify(`Matplotlib ${command.operation} applied.`);
        },
        "Applying…",
      );
    } catch (error) {
      this.hooks.notify(
        error.status === 409
          ? `${error.message} Refresh this attached figure before retrying.`
          : error.message,
        "error",
      );
    }
  }
  point(e) {
    const rect = this.el
      .querySelector("#native-overlay")
      .getBoundingClientRect();
    return {
      x: ((e.clientX - rect.left) * this.session.width) / rect.width,
      y: ((e.clientY - rect.top) * this.session.height) / rect.height,
    };
  }
  pointerDown(e) {
    if (!this.session) return;
    const id = e.target.dataset.artist,
      key = e.target.dataset.point;
    if (id) {
      this.selected = id;
      this.renderProperties();
      this.renderOverlay();
    }
    if (!this.artist() || this.artist().locked) return;
    if (id || key !== undefined) {
      e.preventDefault();
      const point = this.point(e);
      this.drag = { start: point, key, original: this.artist() };
      this.el.querySelector("#native-overlay").setPointerCapture(e.pointerId);
    }
  }
  pointerMove(e) {
    if (!this.drag) return;
    const p = this.point(e),
      dx = p.x - this.drag.start.x,
      dy = p.y - this.drag.start.y;
    if (this.drag.key !== undefined) {
      const point = this.el.querySelector(
        `[data-point="${CSS.escape(String(this.drag.key))}"]`,
      );
      point?.setAttribute("cx", p.x);
      point?.setAttribute("cy", p.y);
    } else
      this.el
        .querySelector(`[data-artist="${CSS.escape(this.selected)}"]`)
        ?.setAttribute("transform", `translate(${dx},${dy})`);
  }
  pointerUp(e) {
    if (!this.drag) return;
    const drag = this.drag;
    this.drag = null;
    const p = this.point(e);
    if (drag.key !== undefined)
      this.command({
        operation: "points",
        points: [{ key: Number(drag.key), x: p.x, y: p.y }],
      });
    else if (Math.abs(p.x - drag.start.x) + Math.abs(p.y - drag.start.y) > 0.5)
      this.command({
        operation: "translate",
        dx: p.x - drag.start.x,
        dy: p.y - drag.start.y,
      });
    else this.renderOverlay();
  }
}
