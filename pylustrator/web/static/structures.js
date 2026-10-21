import { api, busy, projectPath, escapeHTML as esc, download } from "./api.js";
let library;
function loadViewer() {
  if (window.$3Dmol) return Promise.resolve(window.$3Dmol);
  if (!library)
    library = new Promise((resolve, reject) => {
      const script = document.createElement("script");
      script.src = "/static/vendor/3Dmol-min.js";
      script.onload = () =>
        window.$3Dmol
          ? resolve(window.$3Dmol)
          : reject(new Error("The structure renderer did not initialize."));
      script.onerror = () => {
        library = null;
        reject(
          new Error(
            "The local structure renderer could not load. Restart the server and retry.",
          ),
        );
      };
      document.head.append(script);
    });
  return library;
}

export class StructuresWorkspace {
  constructor(hooks) {
    this.hooks = hooks;
    this.project = null;
    this.structure = null;
    this.viewer = null;
    this.sequence = 0;
    this.el = document.querySelector("#structures-workspace");
  }
  mount(project) {
    ++this.sequence;
    const previous =
      this.project?.id === project?.id ? this.structure?.id : null;
    this.project = project;
    this.structure = null;
    this.dispose();
    if (!project) {
      this.el.innerHTML =
        '<div class="empty-state"><h2>Structure views belong with the research</h2><p>Create a project to import a PDB or mmCIF structure, choose its representation, and add a traceable view to your figure.</p></div>';
      return;
    }
    this.el.innerHTML = `<aside class="research-sidebar"><h2>Protein structures</h2><label>Import a structure<input id="structure-file" type="file" accept=".pdb,.cif,.mmcif,.ent"></label><p class="small-note">PDB or mmCIF. Files stay with this project.</p><div id="structure-list" class="asset-list"></div><p id="structure-import-status" class="inline-status" role="status"></p></aside><div class="research-main structure-main"><div id="structure-heading" class="research-title"><div><h2>Explore a molecular structure</h2><p class="muted">Import a structure to inspect chains, set a view, and add it to the figure.</p></div></div><div id="protein-viewer" class="protein-viewer"><div id="protein-placeholder" class="protein-error">Your structure will appear here.</div></div><div id="structure-metadata" class="metadata"></div></div><aside class="research-controls structure-controls"><div id="structure-controls" hidden><h3>Representation</h3><label>Style<select id="structure-style"><option value="cartoon">Cartoon</option><option value="stick">Sticks</option><option value="sphere">Spheres</option><option value="line">Lines</option></select></label><label>Color by<select id="structure-color"><option value="chain">Chain</option><option value="spectrum">Sequence spectrum</option><option value="element">Element</option><option value="secondary">Secondary structure</option><option value="bfactor">B-factor</option></select></label><label>Show chain<select id="structure-chain"><option value="all">All chains</option></select></label><button id="structure-fit" class="full-width">Fit structure</button><p class="small-note">Drag to rotate. Scroll to zoom. Right-drag to translate. Use Fit to restore a centered view.</p><hr><h3>Keep the view reproducible</h3><div class="stack"><button id="save-structure-view">Save camera & style</button><button id="capture-structure" class="primary">Add view to figure</button><button id="download-structure-image">Download PNG</button><button id="download-structure-source">Download source</button></div><p id="structure-status" class="inline-status" role="status"></p><hr><h3>Chains</h3><div id="chain-summary" class="chain-summary"></div></div></aside>`;
    const warningBox = document.createElement("p");
    warningBox.id = "structure-warnings";
    warningBox.className = "project-warning";
    warningBox.setAttribute("role", "status");
    warningBox.hidden = true;
    this.el.querySelector("#structure-metadata").after(warningBox);
    this.el.querySelector("#structure-file").onchange = (e) =>
      this.importFile(e);
    for (const id of ["structure-style", "structure-color", "structure-chain"])
      this.el.querySelector(`#${id}`).onchange = () => this.applyStyle();
    this.el.querySelector("#structure-fit").onclick = () => {
      this.viewer?.zoomTo();
      this.viewer?.render();
    };
    this.el.querySelector("#save-structure-view").onclick = (e) =>
      this.saveView(e.currentTarget);
    this.el.querySelector("#capture-structure").onclick = (e) =>
      this.capture(e.currentTarget);
    this.el.querySelector("#download-structure-image").onclick = () =>
      this.downloadImage();
    this.el.querySelector("#download-structure-source").onclick = () =>
      download(
        new Blob([this.structure.content], { type: "text/plain" }),
        this.structure.name,
      );
    this.renderList();
    if (previous && project.structures.some((s) => s.id === previous))
      this.loadStructure(previous);
    else if (project.structures.length)
      this.loadStructure(project.structures[0].id);
  }
  current(id) {
    return this.project?.id === id && this.hooks.getProject()?.id === id;
  }
  dispose() {
    if (this.viewer) {
      this.viewer.clear();
      this.viewer = null;
    }
    this.observer?.disconnect();
  }
  renderList() {
    const list = this.el.querySelector("#structure-list");
    list.replaceChildren();
    for (const structure of this.project.structures || []) {
      const b = document.createElement("button");
      b.className = structure.id === this.structure?.id ? "selected" : "";
      b.innerHTML = `<span>${esc(structure.name)}</span><small>${esc(structure.format?.toUpperCase() || "Structure")} · ${structure.atomCount ?? structure.metadata?.atomCount ?? "—"} atoms</small>`;
      b.onclick = () => this.loadStructure(structure.id);
      list.append(b);
    }
  }
  async importFile(e) {
    const file = e.target.files[0];
    if (!file) return;
    const status = this.el.querySelector("#structure-import-status");
    if (file.size > 16 * 1024 * 1024) {
      status.textContent = "Choose a structure smaller than 16 MB.";
      e.target.value = "";
      return;
    }
    e.target.disabled = true;
    status.textContent = `Importing ${file.name}…`;
    const projectId = this.project.id;
    try {
      await this.hooks.save();
      if (!this.current(projectId)) return;
      const format = /\.(cif|mmcif)$/i.test(file.name) ? "cif" : "pdb";
      const project = await api.post(projectPath(this.project, "/structures"), {
        name: file.name,
        format,
        content: await file.text(),
        revision: this.hooks.getProject().revision,
      });
      if (!this.current(projectId)) return;
      this.hooks.adopt(project);
      this.mount(project);
      await this.loadStructure(project.structures.at(-1).id);
      this.hooks.notify(`Imported ${file.name}.`);
    } catch (error) {
      status.textContent = error.message;
      this.hooks.notify(error.message, "error");
    } finally {
      e.target.disabled = false;
      e.target.value = "";
    }
  }
  async loadStructure(id) {
    const sequence = ++this.sequence,
      projectId = this.project.id;
    this.dispose();
    this.el.querySelector("#structure-warnings").hidden = true;
    const container = this.el.querySelector("#protein-viewer");
    container.replaceChildren();
    const loading = document.createElement("div");
    loading.className = "protein-error";
    loading.textContent = "Loading structure and initializing the 3D view…";
    container.append(loading);
    try {
      const [structure, lib] = await Promise.all([
        api.get(
          projectPath(this.project, `/structures/${encodeURIComponent(id)}`),
        ),
        loadViewer(),
      ]);
      if (sequence !== this.sequence || !this.current(projectId)) return;
      this.structure = structure;
      container.replaceChildren();
      this.viewer = lib.createViewer(container, {
        backgroundColor: "#f8fafc",
        antialias: true,
      });
      this.viewer.addModel(
        structure.content,
        structure.format === "cif" ? "cif" : "pdb",
      );
      this.el.querySelector("#structure-controls").hidden = false;
      const view = structure.view || structure.metadata?.view || {},
        chains = structure.chains || structure.metadata?.chains || [];
      this.el.querySelector("#structure-style").value =
        view.representation || "cartoon";
      this.el.querySelector("#structure-color").value = view.color || "chain";
      this.el.querySelector("#structure-chain").innerHTML =
        '<option value="all">All chains</option>' +
        chains
          .map(
            (c) =>
              `<option value="${esc(typeof c === "string" ? c : c.id)}">Chain ${esc((typeof c === "string" ? c : c.id) || "(unnamed)")}</option>`,
          )
          .join("");
      this.el.querySelector("#structure-chain").value = view.chain ?? "all";
      this.el.querySelector("#chain-summary").innerHTML = chains
        .map(
          (c) =>
            `<span>${esc((typeof c === "string" ? c : c.id) || "(unnamed)")} · ${c.residueCount ?? "—"} residues</span>`,
        )
        .join("");
      this.el.querySelector("#structure-heading").innerHTML =
        `<div><h2>${esc(structure.name)}</h2><p class="muted">${esc(structure.format?.toUpperCase() || "Structure")} source · interactive molecular view</p></div>`;
      const modelCount = structure.modelCount ?? structure.metadata?.modelCount;
      const modelSummary =
        Number.isInteger(modelCount) && modelCount > 0
          ? `<span>${modelCount} ${modelCount === 1 ? "model" : "models"}</span>`
          : "";
      this.el.querySelector("#structure-metadata").innerHTML =
        `<span>${structure.atomCount ?? structure.metadata?.atomCount ?? "—"} atoms</span><span>${structure.residueCount ?? structure.metadata?.residueCount ?? "—"} residues</span><span>${chains.length} chains</span>${modelSummary}<span>Hash <code>${esc((structure.hash || "").slice(0, 16))}</code></span>`;
      const warnings = structure.warnings ?? structure.metadata?.warnings;
      const warningBox = this.el.querySelector("#structure-warnings");
      warningBox.textContent = Array.isArray(warnings)
        ? warnings.filter((warning) => typeof warning === "string").join(" ")
        : "";
      warningBox.hidden = !warningBox.textContent;
      this.applyStyle();
      this.viewer.zoomTo();
      if (Array.isArray(view.camera) && view.camera.length === 8)
        this.viewer.setView(view.camera);
      this.viewer.render();
      this.observer = new ResizeObserver(() => {
        this.viewer?.resize();
        this.viewer?.render();
      });
      this.observer.observe(container);
      this.renderList();
    } catch (error) {
      if (sequence !== this.sequence || !this.current(projectId)) return;
      container.replaceChildren();
      const box = document.createElement("div");
      box.className = "protein-error";
      const p = document.createElement("p");
      p.textContent = `The structure view could not open. ${error.message}`;
      const b = document.createElement("button");
      b.textContent = "Retry structure view";
      b.onclick = () => this.loadStructure(id);
      box.append(p, b);
      container.append(box);
      this.hooks.notify(error.message, "error");
    }
  }
  view() {
    return {
      representation: this.el.querySelector("#structure-style").value,
      color: this.el.querySelector("#structure-color").value,
      chain: this.el.querySelector("#structure-chain").value,
      camera: this.viewer?.getView() || null,
    };
  }
  applyStyle() {
    if (!this.viewer) return;
    const config = this.view(),
      selection = config.chain === "all" ? {} : { chain: config.chain },
      palette = [
        "#175ca4",
        "#cc6f3b",
        "#408569",
        "#8b62a5",
        "#bd5261",
        "#7c8131",
      ];
    const chains = (
      this.structure.chains ||
      this.structure.metadata?.chains ||
      []
    ).map((c) => (typeof c === "string" ? c : c.id));
    let style;
    if (config.color === "chain")
      style = {
        colorfunc: (atom) =>
          palette[Math.max(0, chains.indexOf(atom.chain)) % palette.length],
      };
    else if (config.color === "spectrum") style = { color: "spectrum" };
    else if (config.color === "element") style = { colorscheme: "Jmol" };
    else if (config.color === "secondary")
      style = {
        colorfunc: (atom) =>
          atom.ss === "h" ? "#d25f62" : atom.ss === "s" ? "#d3a338" : "#63948c",
      };
    else
      style = {
        colorfunc: (atom) => {
          const t = Math.max(0, Math.min(1, (Number(atom.b) || 0) / 100));
          return `rgb(${Math.round(220 * (1 - t) + 35 * t)},${Math.round(85 * (1 - t) + 115 * t)},${Math.round(55 * (1 - t) + 190 * t)})`;
        },
      };
    this.viewer.setStyle({}, {});
    const representation = config.representation;
    if (representation === "cartoon")
      this.viewer.setStyle(selection, { cartoon: { ...style, opacity: 1 } });
    else if (representation === "stick")
      this.viewer.setStyle(selection, { stick: { ...style, radius: 0.18 } });
    else if (representation === "sphere")
      this.viewer.setStyle(selection, { sphere: { ...style, scale: 0.65 } });
    else this.viewer.setStyle(selection, { line: style });
    this.viewer.render();
    this.el.querySelector("#structure-status").textContent =
      "View changed. Save the camera and style to keep this view.";
  }
  async persistView(view = this.view()) {
    if (!this.viewer || !this.structure)
      throw new Error("Load a structure first.");
    const projectId = this.project.id,
      structureId = this.structure.id;
    await this.hooks.save();
    if (!this.current(projectId) || this.structure?.id !== structureId)
      throw new Error(
        "The selected project or structure changed. Retry from the current view.",
      );
    const project = await api.put(
      projectPath(
        this.project,
        `/structures/${encodeURIComponent(structureId)}`,
      ),
      { view, revision: this.hooks.getProject().revision },
    );
    if (!this.current(projectId) || this.structure?.id !== structureId)
      throw new Error(
        "The previous structure view was saved. The selection has changed.",
      );
    this.project = project;
    this.structure.view = view;
    this.hooks.adopt(project);
    return view;
  }
  async saveView(button) {
    try {
      await busy(
        button,
        async () => {
          await this.persistView();
          this.el.querySelector("#structure-status").textContent =
            "Camera and style saved in the project.";
          this.hooks.notify("Structure view saved.");
        },
        "Saving view…",
      );
    } catch (error) {
      this.hooks.notify(error.message, "error");
    }
  }
  async capture(button) {
    try {
      await busy(
        button,
        async () => {
          if (!this.viewer || !this.structure)
            throw new Error("Load a structure first.");
          const projectId = this.project.id,
            structureId = this.structure.id;
          this.viewer.render();
          const data = this.viewer.pngURI().split(",")[1],
            capturedView = this.view();
          const view = await this.persistView(capturedView);
          const result = await api.post(projectPath(this.project, "/assets"), {
            name: `${this.structure.name} view`,
            mimeType: "image/png",
            data,
            revision: this.hooks.getProject().revision,
            provenance: {
              kind: "structure-view",
              structureId: this.structure.id,
              sourceHash: this.structure.hash,
              view,
            },
          });
          if (!this.current(projectId) || this.structure?.id !== structureId)
            return;
          this.project = result.project;
          this.hooks.adopt(result.project);
          const box = this.el
              .querySelector("#protein-viewer")
              .getBoundingClientRect(),
            width = 360,
            height = (width * box.height) / Math.max(1, box.width);
          this.hooks.state.add("image", {
            name: `${this.structure.name} view`,
            assetId: result.asset.id,
            x: 80,
            y: 80,
            width,
            height,
          });
          this.hooks.go("figure");
          this.hooks.notify(
            "Added a structure view with its source and camera provenance.",
          );
        },
        "Capturing…",
      );
    } catch (error) {
      this.hooks.notify(error.message, "error");
    }
  }
  downloadImage() {
    if (!this.viewer) return;
    const uri = this.viewer.pngURI();
    const bytes = Uint8Array.from(atob(uri.split(",")[1]), (c) =>
      c.charCodeAt(0),
    );
    download(
      new Blob([bytes], { type: "image/png" }),
      `${this.structure.name}-view.png`,
    );
  }
  resize() {
    this.viewer?.resize();
    this.viewer?.render();
  }
}
