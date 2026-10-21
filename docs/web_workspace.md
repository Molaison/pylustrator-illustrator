# Pylustrator Forge Web workspace

## Run without a desktop

Requires Python 3.11 or later. In a checkout of this branch:

```bash
python -m pip install -e '.[web]'
python -m pylustrator.web --data-dir ./research-workspace --port 8765
```

Alternatively, use the locked environment:

```bash
uv sync --locked --extra web --no-default-groups
uv run --no-sync pylustrator-web --data-dir ./research-workspace
```

Open `http://127.0.0.1:8765`. No Qt packages, X server, or browser display on the
Python host are needed. The default storage directory is
`~/.pylustrator/workspace`; explicitly choose a persistent volume in containers.
Project data and immutable content blobs live in `workspace.sqlite3`.

For remote machines, use an SSH tunnel:

```bash
ssh -L 8765:127.0.0.1:8765 user@research-host
```

This release is a local or single-user workspace, not a multi-tenant hosted
service. Non-loopback binding requires a server-configured `PYL_WEB_TOKEN` of at
least 24 characters. Put it behind a TLS reverse proxy and set
`PYL_WEB_ALLOWED_HOSTS` to its actual hostname. The browser accepts the token via
the connection form and receives an HttpOnly session cookie. Never pass a token
in a URL. Do not enable multiple Uvicorn workers when using native Figure sessions.

## A complete research workflow

1. **Create a project.** Give it a name, research notes, and tags. The optional
   example uses labeled illustrative values rather than experimental claims.
2. **Import data.** Data accepts CSV, TSV, and arrays of JSON records. Inspect
   detected types, missing values, source hash, and a paginated table.
3. **Prepare data.** Add a parameterized transform or describe the preparation
   to the selected provider. Preview before applying. Applying creates a new
   dataset; the original source is unchanged.
4. **Create a plot.** Map X/Y columns and choose line, scatter, bar, or histogram.
   The plot stores its dataset ID and mapping; Python/Matplotlib produces the
   scientific rendering.
5. **Compose the figure.** Use text, rectangles, ellipses, lines/paths, plots,
   and imported PNGs. Selection, layers, alignment, grouping, guides, numerical
   properties, and gesture history work within the composition document.
6. **Add protein views.** Import a PDB or mmCIF file. Choose chain, cartoon/stick/
   sphere/line representation, color, and camera. Save the view or add a PNG
   snapshot to the figure. The snapshot retains the structure hash and view
   settings as provenance. `pylustrator/web/examples/1crn.pdb` is an offline
   real-structure example with its original source references preserved.
7. **Save and checkpoint.** Save persists the working project. A named checkpoint
   records the project and all referenced data, images, structure sources, and
   settings. Restoring makes a checkpoint of the pre-restore state first.
8. **Export.** Download SVG, PNG, PDF, a replayable Python composition, or the
   complete editable project bundle including checkpoint history.

## Familiar editing behavior

| Action | Interaction |
|---|---|
| Object selection / direct points | `V` / `A` |
| Extend selection | Shift-click |
| Cycle overlapping objects | Alt/Option-click |
| Pan | Space-drag or Hand tool |
| Zoom | Zoom buttons, Fit, Ctrl-wheel |
| Nudge | Arrow keys; Shift for 10 pixels |
| Undo / redo | Ctrl/Command+Z / Shift+Z |
| Group / ungroup | Ctrl/Command+G / Shift+G |
| Duplicate / clipboard | Ctrl/Command+D / C / V |
| Save editable project | Ctrl/Command+S |
| Cancel in-progress gesture | Escape or pointer cancellation |

Typing in a field suppresses canvas shortcuts. A drag commits one undo item,
regardless of pointer-event count. The browser previews a gesture locally and
does not send an HTTP request for every movement. Hidden/locked objects are
excluded from applicable editing operations. Save, undo, checkpoints, and export
are separate operations; a revision conflict never silently overwrites another
tab's saved state.

## Native Matplotlib figures

Existing plotting scripts can attach a trusted in-process Figure without
starting the desktop monkey patches:

```python
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pylustrator.web import serve

fig, ax = plt.subplots()
ax.plot([0, 1, 2], [2, 1, 3], marker="o")
ax.set(xlabel="Time", ylabel="Response")
serve(fig, data_dir="./research-workspace")
```

The Matplotlib view lists live figures and artist capabilities. Edits use the
pre-existing native adapters, transaction plans, semantic text handling, and
exact endpoint/anchor behavior. Mixed unsupported operations are rejected before
mutation. Selection coordinates use browser pixels; the adapter converts to
Matplotlib coordinates. SVG/PNG/PDF export is headless.

The native editor can also export a Python **changes snippet**. Run it after the
original script has constructed `fig`; it validates the original size/DPI and
replays semantic commands using stable Matplotlib ownership references. The
snippet reflects the active undo/redo position. It is distinct from a complete
composition export, which includes its data and image assets.

Native sessions live in the Python process and preserve the caller's Figure;
they are not pickled into project bundles. Keep the original plotting script as
the base source. Persistent Web compositions are a validated JSON format for
publication assembly, not a claim to losslessly serialize every arbitrary
Matplotlib object. The desktop editor remains available with `.[desktop]`.

In this view Ctrl/Command+Z and Shift+Z operate on the attached Figure, and
Ctrl/Command+S downloads its Python changes snippet. The header says **Live
figure**, because project checkpoints do not persist the attached Python object.

## AI provider configuration

The default **Local deterministic helper** is a rule-based tool, not a language
model. It works offline for missing values, normalization, z-scores, logs, and
sorting. The UI explicitly identifies it.

To use a real OpenAI-compatible model, configure the Python server environment:

```bash
export PYL_FIGURE_AI_BASE_URL=https://api.openai.com/v1
export PYL_FIGURE_AI_MODEL=your-supported-model
export PYL_FIGURE_AI_API_KEY=your-key
pylustrator-web --data-dir ./research-workspace
```

`OPENAI_BASE_URL`, `OPENAI_MODEL`, and `OPENAI_API_KEY` are accepted fallbacks.
An explicitly configured loopback HTTP provider can run locally without an API
key. External providers require HTTPS. The integration uses the standard
Chat Completions JSON response contract, so the configured model must support it.

Only the user's preparation request, column names/types/missing counts, row
count, and input hash are sent to the provider. Data rows are omitted. A proposal
is validated and executed against the complete local dataset to produce an exact
preview summary. No generated Python, shell command, expression, or arbitrary
formula is executed. Credentials never enter project documents or checkpoints.

Supported recipe operations: drop/fill missing values, filter, sort, select,
rename, derive (normalize/z-score/log/log10/scale), aggregate. Invalid columns,
non-finite results, bad logarithm domains, and unknown parameters reject the
transaction. A proposal carries its input hash; applying to another data version
fails explicitly.

## Persistence and reproducibility

SQLite transactions atomically save metadata and blobs. SHA-256 content
addressing avoids duplicating unchanged source files across checkpoints. Project
revision numbers are monotonic, including restores. Project bundles contain
every referenced blob and history entry; import revalidates hashes, datasets,
recipes, structures, PNGs, and project-scoped references before committing.

Composition Python export includes its data and image sources and uses the
headless `pylustrator.web.render` API. Install the same version of this fork to
replay it. Unrotated plot panels remain vector in SVG/PDF; rotated plot panels
are explicitly rasterized to preserve their affine geometry.

Current resource bounds are intentional: source tables 16 MiB / 100,000 rows /
256 columns, preview pages 500 rows, structures 16 MiB / 250,000 atoms, composition
2,000 elements / 32 million pixels, and project uploads 64 MiB. Molecular views
use the first model and display a notice for multi-model sources. The browser
needs WebGL for the 3D view; data, plotting, composition, and export do not.

## Verification

```bash
uv sync --locked --all-extras --dev
QT_QPA_PLATFORM=offscreen uv run --no-sync pytest
uv run --no-sync ruff check .
uv run --no-sync ty check
uv run --no-sync python -m playwright install chromium
uv run --no-sync python scripts/test_web_browser.py
node --test tests/web/*.test.mjs
```

CI separately installs only `web,test` and confirms that PyQt and qtpy are absent.
The API suite covers the complete research workflow, project isolation, conflict
handling, same-origin boundaries, provider failures, source integrity, and
native semantic commands. Browser tests cover actual page interactions and
rendering; mocked provider tests are not claims of a live-provider round trip.
