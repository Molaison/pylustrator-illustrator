# Web architecture and capability boundaries

## Product model before implementation

Researchers need to inspect an input, understand its processing, compose a figure,
retain the scientific source, and recover an earlier result. A desktop window is
one interface to those operations. It is not the document model or the scientific
renderer. This migration therefore separates three concerns:

1. **Scientific meaning:** Matplotlib artists and the existing adapter-owned
   transaction engine; datasets and validated transformations; molecular sources.
2. **Durable research state:** projects, immutable source blobs, derived-data
   recipes, composition documents, structure views, and named checkpoints.
3. **Interaction:** browser selection, gestures, contextual controls, local
   previews, asynchronous requests, and explicit pending/error/save states.

The server runs directly with Agg. It does not launch Qt, stream a desktop, or
accept uploaded scripts for execution. Desktop dependencies are an optional extra;
existing desktop entry points are loaded only when invoked.

## Two explicit editing contracts

| Contract | Persistent composition | Native Matplotlib session |
|---|---|---|
| Source of truth | Validated JSON document and project assets | A trusted Figure constructed in the caller's Python process |
| Intended work | Publication assembly and data-bound plotting | Editing an existing scientific plotting script's live figure |
| Elements | Text, rectangle, ellipse, line/path, plot, PNG | Registered artists with explicit adapter capabilities |
| Interaction state | Browser gesture history and selection | FigureSession commands and the existing semantic transaction engine |
| Persistence | SQLite revisions, named checkpoints, complete project bundles | Python changes snippet applied after the original figure construction |
| Export | SVG, PNG, PDF, Python with data/assets | SVG, PNG, PDF, semantic Python changes |
| Lifetime | Survives process restart | Attached Figure and server process lifetime |

The browser makes this difference visible: the native view says **Live figure**
and its save shortcut downloads Python changes. It never reports that a native
Figure was checkpointed merely because an unrelated composition was saved.

Existing artist restrictions remain authoritative. Unsupported geometry or
ownership combinations reject before mutation. The Web document deliberately
does not claim to serialize arbitrary Matplotlib objects, Bézier control handles,
or every Illustrator feature.

## Module responsibilities

| Module | Responsibility |
|---|---|
| `web/document.py` | Versioned schema, finite geometry, limits, groups, style and plot validation |
| `web/render.py` | Serialized Agg rendering; scientific plot mappings; vector/raster/Python export |
| `web/session.py` | Stable artist references, capability discovery, semantic commands, rollback, history and replay |
| `web/datasets.py` | Bounded source parsing, types/missing statistics, previews and a restricted recipe language |
| `web/ai.py` | Explicit local rule helper and configured OpenAI-compatible provider; validate proposals locally |
| `web/protein.py` | Gemmi PDB/mmCIF validation, first-model metadata, bounded view settings |
| `web/store.py` | Transactional SQLite projects, immutable SHA-256 blobs, optimistic revisions, checkpoints and bundle integrity |
| `web/app.py` | Same-origin HTTP boundary, project-scoped resources, error contracts and export routes |
| `web/static/` | Browser editor, data preparation, local 3Dmol view, versions and native Figure interaction |

Matplotlib rendering is guarded by a shared lock because its rendering machinery
is not a concurrent scene server. Browser pointer previews run locally; a gesture
creates one undo item, and expensive plot previews are refreshed after geometry
changes settle. Native sessions also retain their own command serialization.

## Integrity across the complete workflow

**Data preparation:** the original source is immutable. A transformation records
its input hash, validated steps, and derived output. Provider requests contain the
user's request and column metadata/statistics, never table rows. Generated code
is not an execution format. Preview and apply are separate user actions.

**Figure composition:** plots reference project-local dataset IDs and mappings;
images reference validated project assets. Source lookup cannot escape the current
project. Python composition export embeds its required data and image sources.
Unrotated plot panels remain vector in SVG/PDF; rotated panels are rasterized.

**Molecular views:** source parsing uses Gemmi, while a locally packaged 3Dmol.js
renders interactive views in the browser. Camera, representation, color and chain
are serialized. A captured PNG records the structure hash and exact view settings.
For multi-model files, counts and the viewer refer to the first model and the UI
shows that limitation. A PNG in a figure is a reproducible view, not a 3D scene.

**Saving and recovery:** saves compare the current revision and return a conflict
instead of overwriting another tab. A named checkpoint captures all referenced
state. Restore first captures the state being replaced. Bundle import revalidates
hashes, recipes, source/output consistency, molecular sources, PNGs and every
project reference before committing.

**Asynchronous interaction:** navigation locks protect project transitions and
metadata saves. Requests capture their source project and generation; stale
responses cannot populate a different project. This is part of scientific data
integrity, not just a visual loading-state detail.

## Deployment and adoption

This is a local or single-user research workspace. Loopback is the default;
remote work can use an SSH tunnel. A non-loopback CLI bind requires an explicit
token. It is not a multi-tenant collaboration service or a Git hosting interface.

The baseline and the fourteen reference projects are documented in
[web_research.md](web_research.md). Their useful principles are mapped to the
implemented modules and workflow; their desktop runtimes and proprietary SDKs
are not imported merely to imitate their interfaces. See
[web_workspace.md](web_workspace.md) for operation and
[web_verification.md](web_verification.md) for actual release evidence.
