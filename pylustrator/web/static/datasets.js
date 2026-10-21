import { api, busy, projectPath, escapeHTML as esc } from "./api.js";

export class DataWorkspace {
  constructor(hooks) {
    this.hooks = hooks;
    this.project = null;
    this.dataset = null;
    this.preview = null;
    this.plan = null;
    this.offset = 0;
    this.requestId = 0;
    this.showPreview = false;
    this.el = document.querySelector("#data-workspace");
  }
  mount(project) {
    ++this.requestId;
    this.previewRequest = (this.previewRequest || 0) + 1;
    const previous = this.project?.id === project?.id ? this.dataset?.id : null;
    this.project = project;
    this.dataset = null;
    this.preview = null;
    this.plan = null;
    this.showPreview = false;
    if (!project) {
      this.el.innerHTML =
        '<div class="empty-state"><h2>Keep data connected to your figure</h2><p>Create a project to import CSV, TSV, or JSON data, inspect the source, and prepare reproducible plots.</p></div>';
      return;
    }
    this.el.innerHTML = `<aside class="research-sidebar"><h2>Datasets</h2><label>Import data file<input id="dataset-file" type="file" accept=".csv,.tsv,.json,text/csv,application/json"></label><p class="small-note">CSV, TSV, or a JSON array of rows. Source data stays unchanged.</p><div id="dataset-list" class="asset-list"></div><p id="dataset-import-status" class="inline-status" role="status"></p></aside><div class="research-main"><div id="dataset-content"><div class="empty-state"><h2>Inspect before you plot</h2><p>Import a data file to see column types, missing values, and a bounded preview. Processing creates a derived dataset with a reproducible recipe.</p></div></div></div><aside class="research-controls"><div id="data-controls" hidden><h3>Create a plot</h3><form id="plot-form"><label>Plot type<select name="kind"><option value="line">Line</option><option value="scatter">Scatter</option><option value="bar">Bar</option><option value="histogram">Histogram</option></select></label><div class="field-pair"><label>X column<select name="x"></select></label><label>Y column<select name="y"></select></label></div><label>Plot title<input name="title" maxlength="300" placeholder="e.g. Activity by concentration"></label><button class="primary" type="submit">Add plot to figure</button></form><hr><h3>Prepare data</h3><form id="transform-form"><label>Operation<select id="transform-op"><option value="drop_missing">Drop rows with missing values</option><option value="fill_mean">Fill missing values with mean</option><option value="fill_value">Fill missing values with a value</option><option value="filter">Filter rows</option><option value="sort">Sort ascending</option><option value="sort_desc">Sort descending</option><option value="normalize">Normalize to 0–1</option><option value="zscore">Standardize (z-score)</option><option value="log10">Log10 into a new column</option></select></label><label>Column<select id="transform-column"></select></label><div id="transform-filter" class="field-pair" hidden><label>Comparison<select id="filter-operator"><option value="gt">Greater than</option><option value="gte">At least</option><option value="lt">Less than</option><option value="lte">At most</option><option value="eq">Equals</option><option value="ne">Not equal</option><option value="contains">Contains</option></select></label><label>Value<input id="filter-value" autocomplete="off"></label></div><label id="transform-value-label" hidden>Fill value<input id="transform-value" autocomplete="off"></label><button type="submit">Preview step</button></form><details><summary>Recipe JSON · advanced and reusable</summary><label class="sr-only" for="recipe-json">Transformation recipe JSON</label><textarea id="recipe-json" class="recipe-editor" spellcheck="false">[]</textarea><button id="preview-recipe" class="full-width">Preview recipe</button></details><button id="apply-recipe" class="primary full-width" disabled>Apply to a derived dataset</button><p id="transform-status" class="inline-status" role="status">Preview a recipe before applying it.</p><hr><h3>AI-assisted preparation</h3><p class="small-note">Review the proposed steps before applying. Data stays unchanged until you apply a validated recipe.</p><form id="ai-form"><label>Provider<select id="ai-provider"><option value="">Loading providers…</option></select></label><p id="provider-note" class="small-note"></p><label>Describe the preparation<textarea id="ai-prompt" rows="3" required maxlength="4000" placeholder="Drop missing rows and normalize the activity column"></textarea></label><button type="submit">Propose steps</button></form><p id="ai-status" class="inline-status" role="status"></p><div id="ai-explanation" class="recipe-summary" hidden></div></div></aside>`;
    this.el
      .querySelector("#dataset-file")
      .addEventListener("change", (e) => this.importFile(e));
    this.el.querySelector("#transform-op").addEventListener("change", () => {
      const op = this.el.querySelector("#transform-op").value;
      this.el.querySelector("#transform-filter").hidden = op !== "filter";
      this.el.querySelector("#transform-value-label").hidden =
        op !== "fill_value";
    });
    this.el
      .querySelector("#plot-form")
      .addEventListener("submit", (e) => this.addPlot(e));
    this.el
      .querySelector("#transform-form")
      .addEventListener("submit", (e) => this.manualPreview(e));
    this.el
      .querySelector("#preview-recipe")
      .addEventListener("click", (e) => this.runPreview(e.currentTarget));
    this.el.querySelector("#recipe-json").addEventListener("input", () => {
      this.preview = null;
      this.el.querySelector("#apply-recipe").disabled = true;
    });
    this.el
      .querySelector("#apply-recipe")
      .addEventListener("click", (e) => this.apply(e.currentTarget));
    this.el
      .querySelector("#ai-form")
      .addEventListener("submit", (e) => this.propose(e));
    this.renderList();
    this.loadProviders();
    if (previous && project.datasets.some((d) => d.id === previous))
      this.loadDataset(previous);
    else if (project.datasets.length) this.loadDataset(project.datasets[0].id);
  }
  current(id) {
    return this.project?.id === id && this.hooks.getProject()?.id === id;
  }
  renderList() {
    const list = this.el.querySelector("#dataset-list");
    list.replaceChildren();
    for (const dataset of this.project.datasets || []) {
      const b = document.createElement("button");
      b.className = dataset.id === this.dataset?.id ? "selected" : "";
      b.innerHTML = `<span>${esc(dataset.name)}</span><small>${dataset.rowCount ?? "—"} rows · ${dataset.sourceId ? "Derived" : "Original source"}</small>`;
      b.onclick = () => this.loadDataset(dataset.id);
      list.append(b);
    }
  }
  async importFile(e) {
    const file = e.target.files[0];
    if (!file) return;
    const status = this.el.querySelector("#dataset-import-status");
    if (file.size > 16 * 1024 * 1024) {
      status.textContent = "Choose a data file smaller than 16 MB.";
      e.target.value = "";
      return;
    }
    e.target.disabled = true;
    status.textContent = `Importing ${file.name}…`;
    const id = this.project.id;
    try {
      await this.hooks.save();
      if (!this.current(id)) return;
      const format = file.name.toLowerCase().endsWith(".tsv")
        ? "tsv"
        : file.name.toLowerCase().endsWith(".json")
          ? "json"
          : "csv";
      const project = await api.post(projectPath(this.project, "/datasets"), {
        name: file.name,
        format,
        content: await file.text(),
        revision: this.hooks.getProject().revision,
      });
      if (!this.current(id)) return;
      this.hooks.adopt(project);
      this.mount(project);
      await this.loadDataset(project.datasets.at(-1).id);
      this.hooks.notify(`Imported ${file.name}. The source data is preserved.`);
    } catch (error) {
      status.textContent = error.message;
      this.hooks.notify(error.message, "error");
    } finally {
      e.target.disabled = false;
      e.target.value = "";
    }
  }
  async loadDataset(id, offset = 0) {
    const requestId = ++this.requestId,
      projectId = this.project.id;
    this.offset = offset;
    this.preview = null;
    this.plan = null;
    this.showPreview = false;
    this.el.querySelector("#dataset-content").innerHTML =
      '<div class="loading-placeholder" role="status">Loading data preview…</div>';
    try {
      const dataset = await api.get(
        projectPath(
          this.project,
          `/datasets/${encodeURIComponent(id)}?offset=${offset}&limit=100`,
        ),
      );
      if (requestId !== this.requestId || !this.current(projectId)) return;
      this.dataset = dataset;
      this.renderList();
      this.renderTable();
      this.configureControls();
    } catch (error) {
      if (requestId !== this.requestId || !this.current(projectId)) return;
      this.el.querySelector("#dataset-content").innerHTML =
        `<div class="form-error" role="alert">${esc(error.message)}</div><button id="retry-dataset">Retry preview</button>`;
      this.el.querySelector("#retry-dataset").onclick = () =>
        this.loadDataset(id, offset);
    }
  }
  configureControls() {
    this.el.querySelector("#data-controls").hidden = false;
    const options = this.dataset.columns
      .map(
        (c) =>
          `<option value="${esc(c.name)}">${esc(c.name)} · ${esc(c.type)}</option>`,
      )
      .join("");
    for (const selector of ["#transform-column", "[name=x]", "[name=y]"])
      this.el.querySelector(selector).innerHTML = options;
    const numeric = this.dataset.columns.filter((c) =>
      ["number", "integer", "float", "numeric"].includes(c.type),
    );
    if (numeric.length)
      this.el.querySelector("[name=y]").value = (numeric[1] || numeric[0]).name;
    this.el.querySelector("#apply-recipe").disabled = true;
    this.el.querySelector("#recipe-json").value = "[]";
    this.el.querySelector("#ai-explanation").hidden = true;
    this.el.querySelector("#transform-status").textContent =
      "Preview a recipe before applying it.";
  }
  renderTable() {
    const data = this.showPreview && this.preview ? this.preview : this.dataset;
    if (!data) return;
    const rows = data.rows || [],
      columns = data.columns || [];
    const content = this.el.querySelector("#dataset-content");
    content.innerHTML = `<div class="research-title"><div><h2>${esc(this.dataset.name)}</h2><div class="metadata"><span>${data.rowCount.toLocaleString()} rows</span><span>${columns.length} columns</span><span>${this.dataset.sourceId ? "Derived dataset" : "Immutable source"}</span></div></div></div>${this.preview ? `<div class="data-tabs"><button id="raw-data-tab" class="${this.showPreview ? "" : "active"}" aria-pressed="${!this.showPreview}">Before · ${this.dataset.rowCount} rows</button><button id="processed-data-tab" class="${this.showPreview ? "active" : ""}" aria-pressed="${this.showPreview}">Preview · ${this.preview.rowCount} rows</button></div>` : ""}<div class="table-wrap"><table><caption class="sr-only">${this.showPreview ? "Transformed preview" : "Dataset preview"} of ${esc(this.dataset.name)}</caption><thead><tr><th scope="col">Row</th>${columns.map((c) => `<th scope="col">${esc(c.name)}<small>${esc(c.type)} · ${c.missing ?? 0} missing</small></th>`).join("")}</tr></thead><tbody>${rows.map((row, index) => `<tr><td>${(data.offset || 0) + index + 1}</td>${columns.map((c) => `<td class="${row[c.name] === null || row[c.name] === undefined ? "null" : ""}" title="${esc(row[c.name])}">${row[c.name] === null || row[c.name] === undefined ? "missing" : esc(typeof row[c.name] === "object" ? JSON.stringify(row[c.name]) : row[c.name])}</td>`).join("")}</tr>`).join("") || `<tr><td colspan="${columns.length + 1}">No rows match this dataset or preview.</td></tr>`}</tbody></table></div><div class="pagination"><span>Showing ${rows.length ? `${(data.offset || 0) + 1}–${(data.offset || 0) + rows.length}` : "0"} of ${data.rowCount.toLocaleString()}</span><div class="inline-actions"><button id="previous-rows" ${this.showPreview || this.offset === 0 ? "disabled" : ""}>Previous</button><button id="next-rows" ${this.showPreview || this.offset + rows.length >= data.rowCount ? "disabled" : ""}>Next 100</button></div></div><div class="metadata"><span>Data hash <code>${esc((data.hash || this.dataset.hash || "").slice(0, 20))}</code></span>${this.dataset.sourceId ? `<span>Source <code>${esc(this.dataset.sourceId)}</code></span>` : ""}</div>${this.dataset.recipe?.length ? `<details><summary>Processing provenance</summary><pre class="recipe-summary">${esc(JSON.stringify(this.dataset.recipe, null, 2))}</pre></details>` : ""}`;
    content.querySelector("#previous-rows").onclick = () =>
      this.loadDataset(this.dataset.id, Math.max(0, this.offset - 100));
    content.querySelector("#next-rows").onclick = () =>
      this.loadDataset(this.dataset.id, this.offset + 100);
    content.querySelector("#raw-data-tab")?.addEventListener("click", () => {
      this.showPreview = false;
      this.renderTable();
    });
    content
      .querySelector("#processed-data-tab")
      ?.addEventListener("click", () => {
        this.showPreview = true;
        this.renderTable();
      });
  }
  async addPlot(e) {
    e.preventDefault();
    if (!this.dataset) return;
    const form = e.currentTarget,
      values = new FormData(form),
      plot = {
        datasetId: this.dataset.id,
        kind: values.get("kind"),
        x: values.get("x"),
        y: values.get("y"),
        title: values.get("title"),
        xlabel: values.get("x"),
        ylabel: values.get("y"),
        color: "#175ca4",
      };
    if (plot.kind === "histogram") {
      plot.y = plot.x;
      plot.ylabel = "Count";
    }
    this.hooks.state.add("plot", {
      name: plot.title || `${this.dataset.name} plot`,
      x: 60,
      y: 60,
      width: 480,
      height: 320,
      plot,
    });
    this.hooks.go("figure");
    this.hooks.notify("Added a plot linked to this dataset version.");
  }
  manualPreview(e) {
    e.preventDefault();
    const op = this.el.querySelector("#transform-op").value,
      column = this.el.querySelector("#transform-column").value;
    let step;
    if (op === "drop_missing") step = { op, columns: [column] };
    if (op === "fill_mean")
      step = { op: "fill_missing", columns: [column], strategy: "mean" };
    if (op === "fill_value") {
      const value = this.el.querySelector("#transform-value").value;
      step = {
        op: "fill_missing",
        columns: [column],
        strategy: "value",
        value:
          value !== "" && Number.isFinite(Number(value))
            ? Number(value)
            : value,
      };
    }
    if (op === "filter") {
      const value = this.el.querySelector("#filter-value").value;
      step = {
        op,
        column,
        operator: this.el.querySelector("#filter-operator").value,
        value:
          value !== "" && Number.isFinite(Number(value))
            ? Number(value)
            : value,
      };
    }
    if (op === "sort" || op === "sort_desc")
      step = { op: "sort", column, descending: op === "sort_desc" };
    if (["normalize", "zscore", "log10"].includes(op))
      step = { op: "derive", column, target: `${column}_${op}`, method: op };
    this.el.querySelector("#recipe-json").value = JSON.stringify(
      [step],
      null,
      2,
    );
    this.plan = null;
    this.runPreview(e.submitter);
  }
  async runPreview(button) {
    if (!this.dataset) return;
    const id = this.dataset.id,
      projectId = this.project.id,
      previewRequest = (this.previewRequest = (this.previewRequest || 0) + 1);
    const status = this.el.querySelector("#transform-status");
    try {
      await busy(
        button,
        async () => {
          const steps = JSON.parse(this.el.querySelector("#recipe-json").value);
          if (!Array.isArray(steps) || !steps.length)
            throw new Error("Add at least one transformation step.");
          this.preview = null;
          this.el.querySelector("#apply-recipe").disabled = true;
          status.textContent =
            "Computing a preview without changing the source…";
          const result = await api.post(
            projectPath(
              this.project,
              `/datasets/${encodeURIComponent(id)}/preview`,
            ),
            { steps },
          );
          if (
            this.dataset?.id !== id ||
            !this.current(projectId) ||
            previewRequest !== this.previewRequest
          )
            return;
          this.preview = result;
          this.previewedSteps = JSON.stringify(steps);
          this.showPreview = true;
          this.renderTable();
          this.el.querySelector("#apply-recipe").disabled = false;
          status.textContent = `Preview ready: ${this.dataset.rowCount} → ${result.rowCount} rows. The original data is unchanged.`;
        },
        "Previewing…",
      );
    } catch (error) {
      status.textContent = error.message;
      this.hooks.notify(error.message, "error");
    }
  }
  async apply(button) {
    if (!this.preview || !this.dataset) return;
    const id = this.dataset.id,
      projectId = this.project.id;
    try {
      await busy(
        button,
        async () => {
          const steps = JSON.parse(this.el.querySelector("#recipe-json").value);
          if (JSON.stringify(steps) !== this.previewedSteps)
            throw new Error(
              "The recipe changed. Preview it again before applying.",
            );
          await this.hooks.save();
          if (!this.current(projectId) || this.dataset?.id !== id) return;
          const project = await api.post(
            projectPath(
              this.project,
              `/datasets/${encodeURIComponent(id)}/transform`,
            ),
            {
              steps,
              name: `${this.dataset.name} · prepared`,
              expectedHash: this.plan?.inputHash || this.dataset.hash,
              revision: this.hooks.getProject().revision,
              provenance: this.plan
                ? {
                    provider: this.plan.provider,
                    model: this.plan.model,
                    planId: this.plan.id,
                  }
                : undefined,
            },
          );
          if (!this.current(projectId)) return;
          this.hooks.adopt(project);
          this.mount(project);
          await this.loadDataset(project.datasets.at(-1).id);
          this.hooks.notify(
            "Created a derived dataset with its processing history. The source is unchanged.",
          );
        },
        "Applying…",
      );
    } catch (error) {
      this.hooks.notify(error.message, "error");
    }
  }
  async loadProviders() {
    try {
      const response = await api.get("/api/providers");
      const select = this.el.querySelector("#ai-provider");
      if (!select) return;
      select.innerHTML = (response.providers || [])
        .map(
          (provider) =>
            `<option value="${esc(provider.id)}" ${provider.configured ? "" : "disabled"}>${esc(provider.label || provider.name || provider.id)}${provider.configured ? "" : " · not configured"}</option>`,
        )
        .join("");
      const update = () => {
        const provider = response.providers.find((p) => p.id === select.value);
        this.el.querySelector("#provider-note").textContent =
          provider?.id === "local"
            ? "Local deterministic rules helper. No external AI request is made."
            : provider
              ? `${provider.model}. Requests use server-side credentials; schema metadata is shared with the provider.`
              : "No provider configured.";
      };
      select.onchange = update;
      update();
    } catch (error) {
      const select = this.el.querySelector("#ai-provider");
      if (select)
        select.innerHTML =
          '<option value="">Provider list unavailable</option>';
    }
  }
  async propose(e) {
    e.preventDefault();
    if (!this.dataset) return;
    const id = this.dataset.id,
      projectId = this.project.id,
      button = e.submitter,
      status = this.el.querySelector("#ai-status");
    try {
      await busy(
        button,
        async () => {
          status.textContent = "Requesting a validated preparation plan…";
          const plan = await api.post(
            projectPath(
              this.project,
              `/datasets/${encodeURIComponent(id)}/ai-plan`,
            ),
            {
              provider: this.el.querySelector("#ai-provider").value,
              prompt: this.el.querySelector("#ai-prompt").value,
            },
          );
          if (this.dataset?.id !== id || !this.current(projectId)) return;
          this.plan = plan;
          this.el.querySelector("#recipe-json").value = JSON.stringify(
            plan.steps,
            null,
            2,
          );
          this.el.querySelector("#ai-explanation").hidden = false;
          this.el.querySelector("#ai-explanation").textContent =
            plan.explanation ||
            "Review the proposed recipe and its preview before applying.";
          status.textContent = `Plan from ${plan.provider} · ${plan.model}.`;
          await this.runPreview(this.el.querySelector("#preview-recipe"));
        },
        "Proposing…",
      );
    } catch (error) {
      status.textContent = error.message;
      this.hooks.notify(error.message, "error");
    }
  }
}
