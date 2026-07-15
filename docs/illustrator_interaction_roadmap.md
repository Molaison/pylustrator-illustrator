# Illustrator-style interaction roadmap

Pylustrator should adopt Illustrator's predictable direct-manipulation model
without flattening live Matplotlib objects into generic vector paths.  The
editor must preserve semantic figure structure and reproducible Python output.

## Interaction invariants

1. The object shown as selected is exactly the object transformed.
2. Matplotlib ownership, editor grouping, and the current selection scope are
   independent concepts.
3. A preview and its committed result are generated from the same operation
   plan.
4. Unsupported operations are disabled with a reason; they never silently
   promote a child to a parent or approximate another operation.
5. One gesture is one atomic, reversible transaction.  Semantically unchanged
   state is not saved.
6. Default marquee selection excludes containers.  Container/panel selection
   remains an explicit mode.

## P0: interaction correctness

Implementation order follows architectural dependencies.

Status (2026-07-15): implemented on
``refactor/artist-adapter-architecture``.  The P0 implementation is covered by
590 passing tests, 119 explicit capability-branch skips, no xfails, Ruff, the
full Fig2 interaction probe, and a read-only smoke replay of
the unmodified formal Fig2.  The formal file retained SHA-256
``b0cd72abf3962cd6cd2354467ad57aa37ecc213332645d7cb56e6f4af598ad70``.

Manual Fig2 follow-up found two Legend-specific violations and added them to
the P0 invariants.  Selection and alignment now use the union of visible
legend children when an invisible layout frame no longer encloses manually
positioned artwork.  Appearance-only frame toggles preserve Legend identity
and child geometry instead of reconstructing the object.  Failed commits also
restore snapshots with change recording suspended, preventing a serialization
failure from recursively failing during rollback.  Repeated editor
initialization is idempotent as well, so Matplotlib methods cannot accumulate
recursive wrappers across interactive runs or test cases.

An independent 20-type adapter contract matrix subsequently found and closed
two remaining P0 gaps: rotatable snapshots now include native angles, and a
failed `TransformPlan` restores generated-change bookkeeping together with
artist geometry. Its P1/P2 follow-up also closed visible stroke/collection
bounds, duplicate group serialization, and fallback error normalization.

The real Fig2 follow-up audited all 483 selectable/serializable instances and
closed the final P0 replay gap: current, figure-level, and retained
non-current Axes legends now share one authoritative inventory, so all Legend
children have exact persistent references and replayable change records.

A final independent type-by-type QA pass then closed renderer and persistence
edge cases that the ordinary Fig2 paths did not expose: fixed-aspect Axes now
share one native preview/commit constraint; Annotation includes arrow paint;
PathCollection follows renderer item counts; LineCollection preserves NaN path
breaks; and generated code preserves exact finite floats plus qualified
NaN/Inf. A deliberately attempted 13-significant-digit canonicalization was
rejected after a `1e-12`-wide axis amplified it to roughly 90 px. Source
stability is therefore enforced by restoring transaction recording state, not
by quantizing persisted geometry. Ambiguous non-translation matrices fail
capability preflight. Explicit-offset Line/PolyCollection now share a
renderer-faithful path x offset extent model and translate offset controls
without rewriting their base paths.

The final persistence audit separated logical command ownership from call
targets. Legend creation and frame commands now survive later Axes changes and
normalize to the same keys after reload. Explicit single-glyph proxy entries
freeze to self-contained replay specifications instead of referring to the
Legend being created; composite handlers without a complete specification fail
capability preflight. Collection preflight also rejects empty/all-nonfinite
paths and singular move transforms, while empty-offset Line/PolyCollection
edits its renderer-visible base paths.
Legend property reconstruction uses the same source-handle specification, so
`markerscale` is applied exactly once and semantic composite handlers are
preserved or disabled before mutation.

The final click-surface audit also made visible stroked geometry authoritative
when Matplotlib's native picker rejects a zero-area closed Path. Degenerate
PathPatch outlines now fall back to transformed centerline distance instead of
an over-broad bounding-box hit. The real Fig2 probe promotes unclickable
candidates and missing operation categories to explicit failures: all 423 click
candidates are cycle-reachable and all 19 represented categories now have an
alignment contract workflow. Eighteen execute to subpixel accuracy; the Fig2
AxesImage category has no destination that can preserve its clip-limited visible
bounds, so it demonstrates a typed, zero-mutation constraint rejection instead
of reporting a false alignment success.

The destination-specific clip audit now treats paint clipping as part of the
visible selection contract. Rectangular clips intersect the displayed envelope;
non-rectangular clip paths use their transformed polygon intersection, which
reduced the independent circle-clip raster comparison to roughly 0.5 px. Free
drag previews transform the unclipped source envelope before applying the clip,
so newly revealed geometry appears before release and the commit shares the
same clipped result. Exact commands
such as alignment, numeric position/size, match-size, and toolbar scaling first
require a rigid visible-envelope plan. A fully hidden destination or a plan that
cannot reach its requested visible bounds raises `UnsupportedArtistError`
before any Artist, change record, selection, or undo state mutates. Legend is
explicitly exempt from its container-level clip metadata because Matplotlib
draws the frame, handles, and texts as independent children.

### P0.1 Selection kernel

- Add an ordered hit stack rather than returning only one picked Artist.
- Add Object Selection (`V`) and Direct Selection (`A`) modes.
- Add deterministic click-through selection and a candidate-list API.
- Add isolation scopes with enter/exit operations and breadcrumbs.
- Keep default marquee selection container-free.
- Make hover/preselection and all selection surfaces consume the same resolver.

Acceptance:

- Object mode selects the top logical node in scope; direct mode selects the
  exact leaf Artist.
- Click-through visits every hit candidate in deterministic visual order.
- Entering and leaving isolation does not mutate figure geometry.
- Dragging can never transform a node other than the displayed selection.

### P0.2 Logical groups and layer state

- Introduce editor nodes with stable identity and separate ownership, group,
  and selection-scope relations.
- Support group/ungroup, visibility, lock state, names, and z-order commands.
- Adapt the object tree to editor nodes instead of mutating Artist instances
  with UI-only parent fields.
- Treat built-in semantic composites such as legends as logical groups while
  keeping Axes/panel selection explicit.

Acceptance:

- A locked or hidden node cannot be selected from the canvas.
- Group selection and direct child selection behave independently of
  Matplotlib ownership.
- Group and layer mutations participate in undo/redo and serialization.

### P0.3 Semantic transform operations

- Replace coarse resize/rotate booleans with operation descriptors containing
  constraints, preview strategy, and an unsupported reason.
- Distinguish translation, geometry resize, appearance scaling, layout reflow,
  rotation, and point editing.
- Preflight a complete mixed selection before mutating any target.
- Expose handles only for operations supported by the complete selection.

Acceptance:

- Text scaling, legend layout, line geometry, and collection marker scaling are
  never conflated into one ambiguous resize operation.
- Preview-to-commit geometry error remains below 0.25 display pixel.
- A failed multi-object transform leaves every object unchanged.

### P0.4 Transactional command and replay model

- Record semantic before/after state in an atomic command.
- Preserve selection and isolation scope across undo/redo.
- Drop floating-point and semantic no-op changes using adapter-aware equality.
- Introduce stable object locators and a versioned migration boundary for old
  generated commands.
- Coalesce continuous gestures and repeated compatible nudges.

Acceptance:

- Undo/redo restores geometry, selection, and scope.
- No-op gestures do not dirty the document or emit generated commands.
- Existing generated blocks can be migrated, including legacy legend proxy
  references.

## P1: Illustrator productivity

Foundation status (2026-07-15): visible/preview bounds are now explicit and
separate from transformable geometry. Patch, Line2D, and collection adapters
include common-case per-item stroke/marker envelopes. Geometry
resize reapplies fixed display-space appearance outsets, so thick-stroke
deferred previews, commits, groups, alignment, and numeric match-size agree.
EditorGroup is also the single change-record owner for a group gesture.
Selection actions share one immutable geometry snapshot, and offset collections
use Matplotlib's path iterator. On a recorded five-run Fig2 probe, whole-canvas
marquee selects 364 targets in 70.5 ms median. This is slightly slower than the
61.9 ms run before explicit-offset PolyCollection became editable, while still
well below the earlier 125.6 ms adapter result; whole-selection move and undo
remain subsecond with unchanged numerical error.

The transform bar now has a 3x3 reference locator. Numeric X/Y addresses that
point on the exact visible selection bounds, while W/H resizes about the same
point through one preflighted atomic transaction. Physical units compose in the
correct order (native coordinates to display pixels, then inches or
centimeters). A single object with an exact native rotation contract also shows
an arbitrary-angle handle connected to its real native pivot; Shift snaps the
preview to 15-degree increments, and release emits one generated change and one
undo item. Multi-object rotation handles stay hidden until adapters can express
honest common-pivot geometry instead of merely changing every object's local
angle in place. Legend-managed Text also hides the handle: Matplotlib's Legend
packer moves its anchor by about 6.4 px after rotation in the real Fig2, so the
native Text angle is not an honest stable-pivot object transform. The exhaustive
Fig2 audit therefore records 81 representative workflows rather than inflating
the count to 83; all 6,279 instance checks and all 81 workflows pass.

Remaining feature work:

- Common-pivot multi-object rotation and movable pivots for artist types with a
  complete semantic geometry plan.
- Key-object/artboard alignment and numeric distribute spacing.
- Generic smart guides for edges, centers, baselines, anchors, and equal gaps.
- Direct path/endpoint editing and inline text editing.
- Content-following cached drag previews and spatial hit/snap indexes.
- A renderer-faithful paint-envelope policy for miter/cap joins, path effects,
  compound clip holes, and non-bbox source geometry; current axial stroke
  padding and clip-envelope polygon intersection are intentionally not
  advertised as exact raster coverage for every Matplotlib renderer effect.

## P2: workflow breadth

- Duplicate/copy/paste, Select Same, style copy, and complete z-order actions.
- Rulers, guides, grids, familiar zoom/pan shortcuts, and panel templates.
- Scientific roles and protection for panels, labels, legends, annotations, and
  data marks.

## Validation fixture

Fig2 fork validation remains the primary real-world fixture.  Formal
`editable/fig2.py` and publication outputs must not be modified by interaction
experiments.  Automated probes should report results separately for object and
direct selection modes and retain the existing drag, align, resize, reference
transform, rotation-handle, aspect, undo, replay, and performance checks.
