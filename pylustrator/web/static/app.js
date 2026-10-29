import {
  api,
  busy,
  download,
  escapeHTML as esc,
  prettyDate,
  projectPath,
} from "./api.js";
import { EditorState, clone } from "./state.js";
import { FigureEditor } from "./editor.js";
import { DataWorkspace } from "./datasets.js";
import { StructuresWorkspace } from "./structures.js";
import { NativeWorkspace } from "./native.js";
import { ContentRegionsPanel } from "./content-regions.js";

const state = new EditorState();
let project = null,
  view = "figure",
  saving = null,
  failed = false,
  loading = false,
  navigationToken = null,
  projectListRequest = 0;
const $ = (selector) => document.querySelector(selector);
function notify(message, type = "info", action) {
  $("#notice").hidden = false;
  $("#notice").className = `notice${type === "error" ? " error" : ""}`;
  $("#notice").setAttribute("role", type === "error" ? "alert" : "status");
  $("#notice-text").textContent = message;
  $("#notice-action").hidden = !action;
  if (action) {
    $("#notice-action").textContent = action.label;
    $("#notice-action").onclick = action.run;
  }
}
function status() {
  const el = $("#save-state");
  el.className = `save-state${failed ? " failed" : state.dirty ? " dirty" : ""}`;
  el.textContent =
    view === "native"
      ? "Live figure"
      : loading
        ? "Opening…"
        : saving
          ? "Saving…"
          : failed
            ? "Save failed"
            : !project
              ? "No project"
              : state.dirty
                ? "Unsaved changes"
                : "Saved";
  $("#save-project").disabled =
    view === "native" ||
    loading ||
    !project ||
    Boolean(saving) ||
    (!state.dirty && !failed);
  $("#save-project").title =
    view === "native"
      ? "Download Python changes to preserve edits to this attached figure."
      : "Save editable project (Ctrl or Command + S)";
  $("#export-button").disabled = view === "native" ? !native.session : !project;
  $("#project-menu-button").disabled = loading || !project;
}
function adopt(next) {
  if (project && next.id !== project.id) return;
  project = next;
  editor.project = next;
  state.markSaved(next.document);
  updateAssetSummary();
  status();
}
function updateAssetSummary() {
  if (!project) return;
  $("#asset-summary").textContent =
    `${project.datasets.length} datasets · ${project.structures.length} structures · ${project.assets.length} images`;
}
const hooks = {
  state,
  getProject: () => project,
  save: () => save(),
  adopt,
  notify,
  go: (next) => go(next),
};
const editor = new FigureEditor(state, hooks),
  data = new DataWorkspace(hooks),
  structures = new StructuresWorkspace(hooks),
  native = new NativeWorkspace(hooks);
new ContentRegionsPanel(state, hooks);
state.subscribe(() => status());

function navigationBusy(active) {
  if (active) {
    document.querySelector(".editor-inline-text")?.requestSubmit();
    editor.cancel();
  }
  loading = active;
  $("#main").inert = active;
  $(".workspace-bar").inert = active;
  $("#project-select").disabled = active;
  $("#new-project").disabled = active;
  status();
}
async function transition(fn) {
  if (loading)
    throw new Error("A project change is already running. Please wait.");
  const token = Symbol("project-navigation");
  navigationToken = token;
  navigationBusy(true);
  try {
    return await fn();
  } finally {
    if (navigationToken === token) {
      navigationToken = null;
      navigationBusy(false);
    }
  }
}
async function refreshProjects(selected = project?.id) {
  const requestId = ++projectListRequest;
  const response = await api.get("/api/projects");
  if (requestId !== projectListRequest) return response.projects;
  const select = $("#project-select");
  select.innerHTML =
    '<option value="">Choose a project</option>' +
    response.projects
      .map((p) => `<option value="${esc(p.id)}">${esc(p.name)}</option>`)
      .join("");
  const activeId = project?.id || selected;
  if (activeId) select.value = activeId;
  return response.projects;
}
async function openProject(id) {
  if (!id || loading) return;
  const previous = project?.id;
  try {
    await transition(async () => {
      if (state.dirty) await save();
      const next = await api.get(`/api/projects/${encodeURIComponent(id)}`);
      project = next;
      failed = false;
      editor.setProject(next);
      data.mount(next);
      structures.mount(next);
      updateAssetSummary();
      await refreshProjects(id);
      localStorage.setItem("forge.current-project", id);
      if (view === "versions") renderVersions();
      notify(`Opened ${next.name}.`);
    });
  } catch (error) {
    $("#project-select").value = previous || "";
    notify(error.message, "error");
  }
}
async function save() {
  if (saving) {
    await saving;
    if (state.dirty) return save();
    return project;
  }
  if (!project || !state.dirty) return project;
  const snapshot = clone(state.document),
    id = project.id,
    revision = project.revision;
  failed = false;
  saving = api
    .put(projectPath(project), { revision, document: snapshot })
    .then((next) => {
      if (project?.id !== id) return next;
      const unchanged =
        JSON.stringify(state.document) === JSON.stringify(snapshot);
      project = next;
      editor.project = next;
      if (unchanged) state.document = clone(next.document);
      state.markSaved(next.document);
      return next;
    })
    .catch((error) => {
      failed = true;
      notify(
        error.status === 409
          ? "This project changed in another tab. Download your current editable figure before reloading the saved project."
          : error.message,
        "error",
        error.status === 409
          ? {
              label: "Download my figure",
              run: () =>
                download(
                  new Blob([JSON.stringify(state.document, null, 2)], {
                    type: "application/json",
                  }),
                  `${state.document.name}-unsaved.json`,
                ),
            }
          : undefined,
      );
      throw error;
    })
    .finally(() => {
      saving = null;
      status();
    });
  status();
  return saving;
}
async function createProject(values) {
  return transition(async () => {
    if (state.dirty) await save();
    const next = await api.post("/api/projects", values);
    project = next;
    failed = false;
    editor.setProject(next);
    data.mount(next);
    structures.mount(next);
    await refreshProjects(next.id);
    updateAssetSummary();
    go("figure");
    localStorage.setItem("forge.current-project", next.id);
    notify(
      values.example
        ? "Opened an example project with clearly labeled example data."
        : `Created ${next.name}.`,
    );
  });
}
function go(next) {
  view = next;
  document
    .querySelectorAll(".workspace-view")
    .forEach((section) => (section.hidden = section.id !== `view-${next}`));
  document.querySelectorAll("[data-view]").forEach((button) => {
    button.classList.toggle("active", button.dataset.view === next);
    if (button.dataset.view === next)
      button.setAttribute("aria-current", "page");
    else button.removeAttribute("aria-current");
  });
  if (next === "figure") requestAnimationFrame(() => editor.position());
  if (next === "structures") {
    if (structures.project !== project) structures.mount(project);
    requestAnimationFrame(() => structures.resize());
  }
  if (next === "data" && data.project !== project) data.mount(project);
  if (next === "versions") renderVersions();
  if (next === "native") native.mount().then(status);
  status();
  $("#undo").disabled =
    next === "native" ? !native.session?.canUndo : !state.undoStack.length;
  $("#redo").disabled =
    next === "native" ? !native.session?.canRedo : !state.redoStack.length;
}

async function renderVersions() {
  const container = $("#versions-workspace");
  if (!project) {
    container.innerHTML =
      '<div class="empty-state"><h2>Preserve the research behind each figure</h2><p>Create a project to save named checkpoints with data lineage, figure layout, and structure views.</p></div>';
    return;
  }
  const id = project.id;
  container.innerHTML = `<div class="section-header"><div><h2>Project versions</h2><p class="muted">Named checkpoints preserve the figure, datasets, structures, and notes together.</p></div><button id="versions-refresh">Refresh history</button></div><form id="checkpoint-form" class="checkpoint-form"><label>Checkpoint name<input name="name" required maxlength="200" placeholder="e.g. Figure 2 — reviewed draft"></label><button type="submit" class="primary">Save checkpoint</button></form><div id="version-list" class="version-list"><div class="loading-placeholder" role="status">Loading checkpoint history…</div></div><p class="small-note">Undo/Redo changes the current editing session. Checkpoints survive server restarts. Restoring a version first preserves your current project as a recovery checkpoint.</p>`;
  container.querySelector("#versions-refresh").onclick = () => renderVersions();
  container.querySelector("#checkpoint-form").onsubmit = async (e) => {
    e.preventDefault();
    const name = new FormData(e.currentTarget).get("name");
    try {
      await busy(
        e.submitter,
        async () => {
          await save();
          if (project?.id !== id) return;
          await api.post(projectPath({ id }, "/versions"), {
            name,
            revision: project.revision,
          });
          if (project?.id !== id) return;
          await renderVersions();
          notify("Checkpoint saved with project assets and provenance.");
        },
        "Saving checkpoint…",
      );
    } catch (error) {
      notify(error.message, "error");
    }
  };
  try {
    const response = await api.get(projectPath(project, "/versions"));
    if (project?.id !== id || view !== "versions") return;
    const versions = response.versions || [];
    const list = container.querySelector("#version-list");
    list.innerHTML = versions.length
      ? ""
      : '<div class="empty-state"><h3>No checkpoints yet</h3><p>Save the first named checkpoint when the figure or data reaches a meaningful research milestone.</p></div>';
    for (const version of versions) {
      const row = document.createElement("div");
      row.className = "version-row";
      row.innerHTML = `<div><h3>${esc(version.name)}</h3><p>${esc(prettyDate(version.createdAt))} · revision ${version.revision}</p>${version.summary ? `<p>${esc(typeof version.summary === "object" ? JSON.stringify(version.summary) : version.summary)}</p>` : ""}</div><div class="version-actions"><button>Restore this version</button></div>`;
      row.querySelector("button").onclick = async (e) => {
        const button = e.currentTarget;
        const confirmed = await confirmAction(
          "Restore this checkpoint?",
          `“${version.name}” will become the working project. The current project will be preserved as a recovery checkpoint.`,
          "Restore checkpoint",
        );
        if (!confirmed || project?.id !== id) return;
        try {
          await busy(
            button,
            () =>
              transition(async () => {
                await save();
                if (project?.id !== id) return;
                const next = await api.post(
                  projectPath(
                    { id },
                    `/versions/${encodeURIComponent(version.id)}/restore`,
                  ),
                  { revision: project.revision },
                );
                if (project?.id !== id) return;
                project = next;
                editor.setProject(next);
                data.mount(next);
                structures.mount(next);
                updateAssetSummary();
                await renderVersions();
                status();
                notify(
                  "Checkpoint restored. A recovery checkpoint preserves the previous state.",
                );
              }),
            "Restoring…",
          );
        } catch (error) {
          notify(error.message, "error");
        }
      };
      list.append(row);
    }
  } catch (error) {
    container.querySelector("#version-list").innerHTML =
      `<p class="form-error" role="alert">${esc(error.message)}</p>`;
  }
}

function confirmAction(title, message, label) {
  return new Promise((resolve) => {
    const dialog = document.createElement("dialog");
    dialog.innerHTML = `<div class="dialog-heading"><h2>${esc(title)}</h2></div><p>${esc(message)}</p><div class="dialog-actions"><button data-answer="cancel">Cancel</button><button data-answer="yes" class="primary">${esc(label)}</button></div>`;
    document.body.append(dialog);
    const close = (value) => {
      dialog.close();
      dialog.remove();
      resolve(value);
    };
    dialog
      .querySelectorAll("button")
      .forEach(
        (button) =>
          (button.onclick = () => close(button.dataset.answer === "yes")),
      );
    dialog.addEventListener("cancel", (e) => {
      e.preventDefault();
      close(false);
    });
    dialog.showModal();
  });
}

document
  .querySelectorAll("[data-view]")
  .forEach((button) => (button.onclick = () => go(button.dataset.view)));
document
  .querySelectorAll("[data-go]")
  .forEach((button) => (button.onclick = () => go(button.dataset.go)));
document
  .querySelectorAll("dialog [data-close]")
  .forEach(
    (button) => (button.onclick = () => button.closest("dialog").close()),
  );
$("#notice-dismiss").onclick = () => ($("#notice").hidden = true);
$("#help-button").onclick = () => $("#help-dialog").showModal();
$("#new-project").onclick = $("#empty-new-project").onclick = () =>
  $("#new-project-dialog").showModal();
$("#open-example").onclick = (e) =>
  busy(
    e.currentTarget,
    () => createProject({ name: "Example · enzyme activity", example: true }),
    "Opening example…",
  ).catch((error) => notify(error.message, "error"));
$("#project-select").onchange = (e) => openProject(e.target.value);
$("#save-project").onclick = () => save().catch(() => {});
$("#undo").onclick = () =>
  view === "native" ? native.command({ operation: "undo" }) : state.undo();
$("#redo").onclick = () =>
  view === "native" ? native.command({ operation: "redo" }) : state.redo();
$("#new-project-form").onsubmit = async (e) => {
  e.preventDefault();
  const form = e.currentTarget,
    values = new FormData(form);
  const errorBox = form.querySelector(".form-error");
  errorBox.hidden = true;
  try {
    await busy(
      e.submitter,
      () =>
        createProject({
          name: values.get("name"),
          description: values.get("description"),
          example: values.has("example"),
        }),
      "Creating…",
    );
    $("#new-project-dialog").close();
    form.reset();
  } catch (error) {
    errorBox.textContent = error.message;
    errorBox.hidden = false;
  }
};
$("#project-menu-button").onclick = () => {
  if (!project) return;
  const form = $("#project-form");
  form.elements.name.value = project.name;
  form.elements.description.value = project.description || "";
  form.elements.tags.value = (project.tags || []).join(", ");
  form.querySelector(".form-error").hidden = true;
  $("#project-dialog").showModal();
};
$("#project-form").onsubmit = async (e) => {
  e.preventDefault();
  const form = e.currentTarget;
  const values = new FormData(form);
  const id = project?.id;
  if (!id) return;
  try {
    await busy(
      e.submitter,
      () =>
        transition(async () => {
          if (project?.id !== id) return;
          await save();
          if (project?.id !== id) return;
          const next = await api.put(projectPath({ id }), {
            revision: project.revision,
            name: values.get("name"),
            description: values.get("description"),
            tags: values
              .get("tags")
              .split(",")
              .map((s) => s.trim())
              .filter(Boolean),
          });
          if (project?.id !== id) return;
          adopt(next);
          await refreshProjects(next.id);
          if (project?.id !== id) return;
          $("#project-dialog").close();
          notify("Project details saved.");
        }),
      "Saving…",
    );
  } catch (error) {
    const box = form.querySelector(".form-error");
    box.textContent = error.message;
    box.hidden = false;
  }
};
$("#export-button").onclick = (e) =>
  busy(
    e.currentTarget,
    async () => {
      const format = $("#export-format").value;
      if (view === "native") {
        await native.export(format);
        return;
      }
      const name = state.document.name;
      const blob = await api.post(
        projectPath(project, "/export"),
        { format, document: state.document },
        { binary: true },
      );
      download(
        blob,
        `${name.replace(/[^\p{L}\p{N}._-]+/gu, "-") || "figure"}.${format === "python" ? "py" : format}`,
      );
      notify(
        `Exported ${format === "python" ? "reproducible Python source" : format.toUpperCase() + " figure"}.`,
      );
    },
    "Exporting…",
  ).catch((error) => notify(error.message, "error"));
$("#export-bundle").onclick = (e) =>
  busy(
    e.currentTarget,
    async () => {
      await save();
      const source = project;
      const blob = await api.get(projectPath(source, "/bundle"), {
        binary: true,
      });
      download(
        blob,
        `${source.name.replace(/[^\p{L}\p{N}._-]+/gu, "-")}-project.json`,
      );
      notify("Downloaded the complete project bundle.");
    },
    "Preparing bundle…",
  ).catch((error) => notify(error.message, "error"));
$("#import-bundle").onclick = () => $("#file-bundle").click();
$("#file-bundle").onchange = async (e) => {
  const file = e.target.files[0];
  if (!file) return;
  try {
    if (file.size > 64 * 1024 * 1024)
      throw new Error("Choose a project bundle smaller than 64 MB.");
    await transition(async () => {
      await save();
      const next = await api.post(
        "/api/projects/import",
        JSON.parse(await file.text()),
      );
      project = next;
      editor.setProject(next);
      data.mount(next);
      structures.mount(next);
      await refreshProjects(next.id);
      updateAssetSummary();
      $("#project-dialog").close();
      go("figure");
      notify("Imported the project bundle as a new research project.");
    });
  } catch (error) {
    notify(error.message, "error");
  } finally {
    e.target.value = "";
  }
};
$("#add-image").onclick = () => {
  if (project) $("#file-image").click();
};
$("#file-image").onchange = async (e) => {
  const file = e.target.files[0],
    id = project?.id;
  if (!file || !id) return;
  try {
    if (file.size > 12 * 1024 * 1024)
      throw new Error("Choose a PNG image smaller than 12 MB.");
    await save();
    if (project?.id !== id) return;
    const buffer = await file.arrayBuffer();
    if (project?.id !== id) return;
    let binary = "";
    for (const byte of new Uint8Array(buffer))
      binary += String.fromCharCode(byte);
    const result = await api.post(projectPath({ id }, "/assets"), {
      name: file.name,
      mimeType: "image/png",
      data: btoa(binary),
      revision: project.revision,
    });
    if (project?.id !== id) return;
    adopt(result.project);
    const image = await createImageBitmap(file);
    if (project?.id !== id) {
      image.close();
      return;
    }
    const width = Math.min(400, image.width),
      height = (width * image.height) / image.width;
    state.add("image", {
      name: file.name,
      assetId: result.asset.id,
      x: 60,
      y: 60,
      width,
      height,
    });
    image.close();
    notify("Added the image to the project and figure.");
  } catch (error) {
    notify(error.message, "error");
  } finally {
    e.target.value = "";
  }
};
window.addEventListener("beforeunload", (e) => {
  if (state.dirty || saving) {
    e.preventDefault();
    e.returnValue = "";
  }
});
document.addEventListener("keydown", (e) => {
  if (
    e.defaultPrevented ||
    !(e.ctrlKey || e.metaKey) ||
    view === "figure" ||
    loading ||
    document.querySelector("dialog[open]")
  )
    return;
  const key = e.key.toLowerCase();
  if (view === "native" && key === "z") {
    if (e.target.closest("input, textarea, select, [contenteditable]")) return;
    e.preventDefault();
    const operation = e.shiftKey ? "redo" : "undo";
    const available = e.shiftKey
      ? native.session?.canRedo
      : native.session?.canUndo;
    if (available && !e.repeat) native.command({ operation });
  } else if (key === "s") {
    e.preventDefault();
    if (e.repeat) return;
    if (view === "native")
      native.export("python").catch((error) => notify(error.message, "error"));
    else save().catch(() => {});
  }
});

async function auth() {
  const dialog = document.createElement("dialog");
  dialog.innerHTML =
    '<form id="connect-form" class="stack"><div class="dialog-heading"><h2>Connect to your workspace</h2></div><p>This server requires an access token. Enter the token configured by the server owner.</p><label>Access token<input name="token" type="password" required autocomplete="current-password"></label><p class="form-error" role="alert" hidden></p><button class="primary" type="submit">Connect</button></form>';
  document.body.append(dialog);
  dialog.showModal();
  dialog.addEventListener("cancel", (e) => e.preventDefault());
  dialog.querySelector("form").onsubmit = async (e) => {
    e.preventDefault();
    const form = e.currentTarget;
    try {
      await busy(
        e.submitter,
        () => api.post("/api/auth", { token: new FormData(form).get("token") }),
        "Connecting…",
      );
      form.reset();
      dialog.close();
      dialog.remove();
      boot();
    } catch (error) {
      const box = form.querySelector(".form-error");
      box.textContent = error.message;
      box.hidden = false;
    }
  };
}
async function boot() {
  try {
    const projects = await refreshProjects();
    const remembered = localStorage.getItem("forge.current-project");
    const id = projects.find((p) => p.id === remembered)?.id || projects[0]?.id;
    await native.discover();
    if (id) await openProject(id);
    else {
      status();
      data.mount(null);
      structures.mount(null);
      if (native.sessions.length) go("native");
    }
    $("#notice").hidden = true;
  } catch (error) {
    if (error.status === 401) {
      $("#save-state").textContent = "Connection required";
      auth();
    } else {
      status();
      notify(error.message, "error", {
        label: "Retry connection",
        run: () => boot(),
      });
    }
  }
}
boot();
