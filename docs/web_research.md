# Migration baseline and reference review

## Baseline

The Web branch starts at PR #14, `369f5d1a714df77e8713f28560946741df1d70c8`.
Its history contains main `27b2c94a71961d6d5ae03eee476446e6fc19c826`, preserving
the direct-anchor editing, inline semantic text, exact replay, and lifecycle
improvements that had not yet been merged. These are separate from new Web work.

| Capability before migration | Status / implementation |
|---|---|
| Illustrator-style selection, groups, layers, geometry, appearance, guides | Implemented desktop interaction and adapter core |
| Direct Polygon/linear PathPatch/Line2D/Annotation point editing | Implemented in PR #14 |
| Semantic inline text and reversible native commands | Implemented in PR #14 |
| Scientific Matplotlib rendering and generated source | Implemented; rendering reusable headlessly |
| Web server, browser editor, project persistence | Absent before this branch |
| Data preview, recipes, provider integration | Absent before this branch |
| Protein source and molecular view | Absent before this branch |
| Arbitrary Bézier handles / every Illustrator operation | Not claimed; existing capability restrictions remain |

The package initializer eagerly imported GUI entrypoints and its base install
required Qt. The migration makes GUI entrypoints lazy and moves those dependencies
into `desktop`. It keeps native scientific editing in Python and gives browser
composition its own validated interchange document, with explicit integration.

## Broad comparison → focused implementation

Sources below are the projects' primary repositories and documentation. The
adoption column describes our design decisions, not a claim of complete parity.

| Reference | Borrowed principle | Adopted implementation |
|---|---|---|
| [JupyterLab](https://github.com/jupyterlab/jupyterlab) | Workspace and contextual panels | Project identity and focused Figure/Data/Structures/Versions views |
| [LabPlot](https://github.com/KDE/labplot) | Dataset/plot/worksheet relationships | Data IDs bound to plot mappings and project assets |
| [Excalidraw](https://github.com/excalidraw/excalidraw) | Canvas commands, history, open scene | Gesture history, geometry editing, editable JSON composition |
| [tldraw](https://github.com/tldraw/tldraw) | Records, shape/asset boundaries | Own schema and separate assets; no tldraw SDK dependency |
| [Vega Editor](https://github.com/vega/editor) | Declarative mapping plus preview | Stored data mapping and rendered scientific plot |
| [Dash](https://github.com/plotly/dash) | Analytical input/computation/output | HTTP computations and explicit pending/error UI states |
| [Orange](https://github.com/biolab/orange3) | Visible preprocessing and tables | Types/missing statistics, bounded table preview, recipes |
| [OpenRefine](https://github.com/OpenRefine/OpenRefine) | Reusable operations and history | Validated JSON recipes and immutable processed datasets |
| [marimo](https://github.com/marimo-team/marimo) | Dependency and provider awareness | Dataset hashes, proposal validation, provider registry |
| [Galaxy](https://github.com/galaxyproject/galaxy) | Reproducible processing provenance | Input/output hashes, recipes, project history |
| [DVC](https://github.com/treeverse/dvc) | Content-addressed data, separate metadata | SHA-256 immutable blobs referenced by revisions |
| [napari](https://github.com/napari/napari) | Typed layers and visibility | Contextual layer/property controls |
| [3Dmol.js](https://github.com/3dmol/3Dmol.js) | Browser molecular representations | Locally bundled 2.5.5, style/camera snapshots into figures |
| [Mol*](https://github.com/molstar/molstar) | Structure selection and view state | Chain/view controls with serialized source provenance |

Desktop runtimes from LabPlot/Orange/napari are not dependencies of the Web
service. The scientific renderer remains Matplotlib. Research patterns are
integrated into one project model, rather than disconnected clone applications.

## Design reviews

User-specified references:

- [claude-design-system-prompt](https://github.com/Trystan-SA/claude-design-system-prompt)
  at `3c3ddb07d7aa3fef051d83608596470c95cfd8fe` (MIT): hierarchy, rhythm,
  accessibility, interaction states, polish.
- [base-impeccable](https://github.com/tool-source/base-impeccable)
  at `d272b9bd5dcfcb52d32482d192d06045ca31c503` (Apache-2.0): Operate mode,
  audit, harden, polish.

The initial independent review specified workflow, information architecture,
keyboard semantics, save/version distinctions, constrained AI, and resource
provenance. Fresh implementation reviewers apply these instructions to real
sources and rendered workflows. Findings and their resolution are tracked in
Linear project P-MOL-14 (MOL-251 through MOL-259).
