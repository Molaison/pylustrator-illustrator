<h1 align="center">
<img alt="docs/images/logo.png" src="docs/images/logo.png" width="300">
</h1><br>


[![DOC](https://readthedocs.org/projects/pylustrator/badge/)](https://pylustrator.readthedocs.io)
[![PyTest](https://github.com/Molaison/pylustrator-illustrator/actions/workflows/pytest.yml/badge.svg)](https://github.com/Molaison/pylustrator-illustrator/actions/workflows/pytest.yml)
[![License](https://img.shields.io/badge/License-GPLv3-blue.svg)](http://www.gnu.org/licenses/gpl-3.0.html)
[![DOI](https://img.shields.io/badge/DOI-10.21105/joss.01989-blue.svg)](https://doi.org/10.21105/joss.01989)



Pylustrator is a software to prepare your figures for publication in a reproducible way. This means you receive a figure
representing your data and alongside a generated code file that can exactly reproduce the figure as you put them in the
publication, without the need to readjust things in external programs.

Pylustrator offers an interactive interface to find the best way to present your data in a figure for publication.
Added formatting and styling can be saved by automatically generated code. To compose multiple figures to panels,
pylustrator can compose different subfigures to a single figure.

## About This Fork

This repository is a downstream fork of
[`rgerum/pylustrator`](https://github.com/rgerum/pylustrator). It keeps
Pylustrator's reproducible, source-generating workflow while rebuilding its
direct-manipulation layer around a more predictable, Adobe Illustrator-style
interaction model.

The refactor was motivated by a fundamental mismatch in the original editor:
Matplotlib artists can look alike on screen while using very different
coordinate systems, ownership rules, mutation APIs, and serialization paths.
Treating every artist as a generic rectangle made selection, dragging,
alignment, resize, undo, and the generated source behave inconsistently across
objects such as text, axes labels, legends, patches, lines, and collections.

This fork makes those differences explicit at the artist boundary and keeps
the user-facing interaction rules shared.

### Major Architectural Changes

- **Artist adapter architecture.** Each supported Matplotlib artist resolves
  to a specific adapter that owns its visible bounds, hit testing,
  capabilities, display-space transforms, snapshots, mutations, undo state,
  and reproducible replay commands. Concrete registrations are exact-only by
  default; a subclass inherits mutation semantics only after its adapter opts
  into an explicitly validated inheritance contract.
- **Unified selection model.** Object Selection and Direct Selection consume
  the same ordered hit stack used by hover, click-through, candidate menus,
  and marquee selection. Containers are excluded from marquee selection by
  default and remain available through explicit selection modes.
- **Logical editor groups.** Editor grouping and selection scope are separate
  from Matplotlib's implementation ownership, so selecting a child no longer
  implies that dragging must transform its parent axes, legend, or container.
- **Semantic operation contracts.** Move, resize, native rotation,
  shared-pivot rotation, property editing, alignment, and replay are explicit
  capabilities. A mixed selection is fully preflighted before any target is
  mutated, and controls are exposed only when the complete selection supports
  the operation. Translation, resize, and rotation plans freeze one absolute
  destination; a changed source, coordinate system, clip, layout, or group
  membership rejects the stale plan before any target or history state changes.
- **Atomic transactions and replay.** One gesture produces one reversible undo
  item. Failed or cancelled gestures restore artist geometry, generated-change
  bookkeeping, selection state, and interaction scope; semantic and
  floating-point no-ops are dropped.

### Interaction and Geometry Improvements

- Selection indicators, drag previews, alignment, and committed positions use
  the same artist-aware visible geometry, including clipping, stroke width,
  markers, transformed paths, and renderer-managed collection offsets. Hit
  testing, marquee selection, overlays, and Smart Guides reuse one revisioned
  display-geometry service instead of measuring the same scene independently.
- Deferred translation can show a renderer-faithful cached content ghost while
  the semantic Artist remains untouched until commit. Captures run only from
  idle work, retain an explicit memory/source/complexity budget, and validate a
  renderer/revision/source token on activation. Unsupported, clipped,
  canvas-edge, oversized, or non-translation cases automatically keep the
  existing analytic outline rather than showing a misleading bitmap.
- The display-geometry service also exposes opt-in, revision-bound Agg paint
  envelopes with explicit exact/conservative/unavailable states. Capture draws
  only audited disposable clones; pending or open-ended visual state stays
  conservative, and pointer cache misses never invoke a renderer.
- Alignment supports selection bounds, the canvas/artboard, and an explicit
  key object without allowing stale key-object mode to intercept ordinary
  single-object drags.
- Resize and rotation use preflighted transform plans and stable pivots;
  multi-object rotation uses one shared pivot rather than unrelated local
  angle changes. When every selected object supports exact rigid rotation, the
  on-canvas pivot can be dragged anywhere on or beyond the artboard; native-only
  rotation keeps the object's real fixed pivot. Line2D circle/dot markers use
  their actual `markevery` subset at the destination, so marker-only and mixed
  selections preview, commit, and undo around that same pivot.
- The Align panel separates geometry scale from explicit appearance scale.
  `A+`/`A−` change supported font, stroke, and marker dimensions without moving
  coordinates or reflowing layout; mixed selections are preflighted and undo
  as one command.
- Legends have stable logical ownership across selection, frame changes,
  movement, undo/redo, and source replay. Their selection bounds follow visible
  handles, labels, title, and frame rather than invisible layout boxes.
- The six core Legend layout controls reflow the existing OffsetBox tree rather
  than rebuilding the Legend. Legend, frame, handle, Text, title, DrawingArea,
  and TextArea identities survive column/spacing changes and Undo/Redo; only
  verified standard HPacker/VPacker structure is replaced.
- Axis labels and formatter-owned tick labels are edited through their semantic
  axis owner, allowing content and font properties to be changed without
  accidentally moving the containing axes. Auto-positioned titles, offset text,
  constrained-layout labels, layout-only Legend geometry, and container
  backgrounds reject independent transforms that Matplotlib would overwrite on
  draw; manually positioned and out-of-layout artwork remains editable.
- Line2D keeps its original ndarray/MaskedArray data semantics through
  movement, rotation plans, replay, and Undo/Redo, including hidden masked
  values, independent masks, dtype/shape, fill value, and hard-mask state.
  Categorical and datetime data retain appearance editing and lossless replay
  while unsupported geometry operations return an explicit reason.
- Pointer hit testing uses a conservative display-space index while native
  artist containment remains authoritative. Smart Guides add deterministic
  edge, center, baseline, insertion-anchor, cross-feature, and equal-gap
  snapping. Bounded idle batches publish the hit index atomically before
  finishing Smart Guide capture; an unfinished index keeps unmeasured Artists
  in the conservative native-hit path, so mouse press never performs a blocking
  full-scene measurement or loses a selectable object. Ordinary hover and
  click stream only to the first conclusive foreground hit; Alt click-through,
  candidate menus, double-click isolation, and ambiguous Direct Selection
  group shells retain the complete hit-stack oracle.
- Bring Forward/Backward and Send to Front/Back operate on Matplotlib's actual
  stable paint order, update hit ordering immediately, and undo atomically.
- Editor windows can be closed and reopened without accumulating hidden Qt
  managers, top-level windows, timers, callbacks, or duplicate Matplotlib key
  handling. Each Figure retains isolated history UI and source key bindings are
  suspended only while that Figure is attached to the editor.
- Large Line2D drag previews use contiguous buffers rather than one allocation
  per vertex. Interaction-scoped geometry and legend discovery caches reduce
  repeated renderer work, while source-only saves avoid replaying unrelated
  figure exports.

### Explicit Capability Boundaries

Not every Matplotlib artist can safely support every operation. This fork
rejects an unsupported transform before mutation instead of applying a visual
approximation that cannot be undone or reproduced. For example, some
layout-owned legend children and formatter-owned tick labels are selectable and
property-editable but intentionally not independently movable or resizable.

The current per-type guarantees and deliberate limitations are documented in
the [artist operation support matrix](docs/artist_operation_support_matrix.md).
The longer-term design and remaining productivity work are tracked in the
[Illustrator-style interaction roadmap](docs/illustrator_interaction_roadmap.md),
and the extension API is introduced in the [API documentation](docs/api.rst).

### Validation Status

At the current fork milestone on 2026-07-18:

- the full test suite passed with **1,281 passed and 178 skipped**;
- Ruff and Ty completed successfully, with an explicit incremental type-check
  baseline for the dynamic Matplotlib/Qt interaction modules; and
- the real multi-panel Fig2 workflow was used to validate selection, movement,
  resize, rotation, alignment references, Smart Guides, legends, masked Line2D
  replay, axis-label editing, true paint-order stacking, window close/reopen,
  cold and warm hit latency, cached content previews, save/replay, and
  undo/redo behavior. On the current real-case fork, cached-ghost activation
  stayed below 0.46 ms p95 and motion below 0.03 ms p95; warm ordinary top-hit
  resolution stayed below 1.57 ms p95. The formal editable Fig2 remained
  byte-identical during fork-based validation.

### Supported Runtime

The supported Python and direct-dependency contract reflects versions
exercised by the complete automated suite, rather than the much older inherited
package metadata:

| Runtime | Dependency set | CI contract |
|---|---|---|
| Python 3.11 | Every declared direct dependency at its lower bound | Minimum-supported lane |
| Python 3.12 | Versions resolved by `uv.lock` | Locked lane |
| Python 3.13 | Versions resolved by `uv.lock` | Locked lane |

The minimum lane currently exercises natsort 4.0.0, NumPy 1.23.5, Matplotlib
3.8.4, PyQt5 5.15.2, qtawesome 0.5.0, scikit-image 0.21.0, qtpy 2.4.3,
and pytest 7.2.0 in one environment.

Python 3.9 and Matplotlib 2.x are not compatible with the current editor
architecture. Python 3.10 is outside the supported matrix because the rollback
diagnostic contract uses Python 3.11 exception notes. Environments outside the
table may work, but are not part of the tested compatibility promise.

Please refer to the upstream
[Pylustrator documentation](https://pylustrator.readthedocs.io) for the base
application and usage guide. Fork-specific architecture and behavior are
documented in this repository.

## Installation

This fork deliberately does not publish a package under the upstream
`pylustrator` distribution name. Running `pip install pylustrator` installs the
upstream project, not the interaction architecture described above.

Install this fork directly from GitHub:

```bash
python -m pip install "pylustrator @ git+https://github.com/Molaison/pylustrator-illustrator.git@main"
```

With `uv`:

```bash
uv pip install "pylustrator @ git+https://github.com/Molaison/pylustrator-illustrator.git@main"
```

For reproducible environments, replace `main` with a release tag or a full
commit SHA. The import name remains unchanged:

```python
import pylustrator
```

For development, clone the fork and install all test and documentation
dependencies:

```bash
git clone https://github.com/Molaison/pylustrator-illustrator.git
cd pylustrator-illustrator
uv sync --locked --all-extras --dev
```

## Offline Generated-Source Doctor

Historical generated blocks can fail before Pylustrator starts—for example, an
old block containing bare `nan` or `inf` cannot be repaired by a runtime
migration because Python evaluates that block first. The fork therefore ships
an offline doctor that reads Python source without importing or executing it:

```bash
# Diagnose one file or recursively scan a directory; never writes by default.
pylustrator-source path/to/figure.py
pylustrator-source path/to/figures/

# Preview the exact changes, then opt in to an atomic migration.
pylustrator-source --diff path/to/figure.py
pylustrator-source --write path/to/figure.py
```

The doctor recognizes only exact Pylustrator marker comments, leaves user code
outside those blocks byte-for-byte unchanged, and handles schema versions,
legacy indexed Legend proxy locators, and non-finite NumPy literals. It
preserves the source encoding, newline style, and file mode; refuses to replace
symbolic links, break hardlinks, or overwrite a concurrently changed file; and
never partially writes a file containing an unknown future schema or malformed
block. `--json` provides
machine-readable diagnostics. Exit status is `0` when clean (or fully
migrated), `1` when source diagnostics remain, and `2` for an operational or
usage error. See the [source doctor reference](docs/source_doctor.rst) for the
full safety contract.

## Issues, Questions, and Suggestions

Please submit your questions, suggestions, and bug reports to the
[fork issue tracker](https://github.com/Molaison/pylustrator-illustrator/issues).
Issues that also reproduce in unmodified upstream Pylustrator can be reported
to the [upstream issue tracker](https://github.com/rgerum/pylustrator/issues).


## Contributing

You want to contribute? Great!
Contributing works best if you creat a pull request with your changes.

1. Fork the project.
2. Create a branch for your feature: `git checkout -b cool-new-feature`
3. Commit your changes: `git commit -am 'My new feature'`
4. Push to the branch: `git push origin cool-new-feature`
5. Submit a pull request!

If you are unfamilar with pull requests, you find more information on pull requests in the
 [github help](https://help.github.com/en/github/collaborating-with-issues-and-pull-requests/about-pull-requests)
