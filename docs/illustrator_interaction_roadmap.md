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

Status (2026-07-16): implemented on
``refactor/artist-adapter-architecture``.  The P0 implementation is covered by
712 passing tests, 147 explicit capability-branch skips, no xfails, Ruff, the
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

A later manual Fig2 pass exposed the same ownership mistake in Axis text.
Panel A's true ylabel is an empty but live Text slot, while its 11 visible
method names are formatter-owned y-tick labels. Empty Text snapshots now retain
all appearance/content state, and tick content is committed through an atomic
Axis-owned tick/label plan instead of transient `Text.set_text()`. Visible tick
labels are exact click targets in Object and Direct Selection, but reject drag
with an ownership reason and stay out of rigid marquee selections. Independent
Fig2 QA passed 22/22 click paths, 11/11 zero-mutation translation rejections,
and 8/8 text/font-size draw, Undo/Redo, and replay workflows; the default
whole-canvas marquee remains 364 non-container objects.

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

The offline migration boundary is now executable through `pylustrator-source`.
It tokenizes exact generated-block comments without importing the inspected
script, reports malformed/future schemas, and produces an idempotent candidate
for legacy indexed Legend proxy locators and pre-runtime `nan`/`inf` failures.
Writes are opt-in and atomic, preserve encoding/newlines/mode, reject symlinks
and concurrent changes, and fail closed when any block cannot be migrated
safely.

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

### P1a: Explicit appearance scaling

Implemented on 2026-07-17 as a separate semantic operation rather than a
fallback from geometry resize. The Align panel keeps its existing geometry
scale controls and adds explicit `A+`/`A−` controls. The first batch supports:

- ordinary visible point Text through font size;
- Line2D through linewidth, marker size, and marker-edge width;
- LineCollection and visibly stroked PolyCollection through linewidths; and
- PathCollection through linewidths and marker areas (`size × factor²`).

Every mixed selection is capability-checked before mutation and consumes one
frozen absolute plan for preview and commit. Factor `1` is an exact no-op;
finite near-one factors remain valid edits. Generated records contain only the
owned appearance setters, and Undo/Redo restores appearance, selection, the
primary object, alignment state, and generated bookkeeping atomically.

Appearance has its own transaction state rather than piggybacking on geometry
snapshots. This both preserves the semantic boundary and removes appearance
checks from ordinary move/rotate snapshot hot paths. Marker/collection support
requires actual rendered paint: visible colors and positive dimensions are not
enough when paths have no fill area or strokable segment. Unique marker paths
are analyzed once, keeping large scatter capability checks linear in item
metadata rather than repeatedly parsing the same path.

The first batch deliberately rejects Annotation, Legend/layout-managed Text,
wrapped/bbox/TeX/effected Text, pixel markers, hatch, filters, sketch/path
effects, Legend/layout-owned children, invalid or underflowing dimensions, and
paint-free or degenerate paths. No unsupported geometry resize silently turns
into appearance scale.

The read-only Fig2 fork scanned 1,457 live Artists. It advertised appearance
scaling for 427 instances (28 Text, 352 Line2D, 7 LineCollection, and 40
PathCollection); every supported instance produced a finite, side-effect-free
preflight. Per-type commit/draw/restore workflows had zero preview error and
emitted only their appearance-owned setters. The formal editable Fig2 remained
byte-identical before and after at SHA-256
`aba67bbd663fd16da535aa30d43f607c7205d096455f44544e518607cdce2dbb`.

### P1b: Identity-preserving Legend reflow (next)

Legend layout remains a separate operation. The discarded donor/transplant
prototype preserved the outer Legend but replaced handles, Text, title, and
packer identities, violating direct-selection and Undo/Redo contracts. The
next implementation must reuse every existing Artist and DrawingArea/TextArea,
replace only standard HPacker/VPacker structure, and preserve manual child
offsets. Its v1 layout spec is limited to `ncols`, `borderpad`, `labelspacing`,
`handlelength`, `handletextpad`, and `columnspacing`; `mode="expand"`, unknown
packers, and simultaneous Legend/descendant selection reject before mutation.

The transform bar now has a 3x3 reference locator. Numeric X/Y addresses that
point on the exact visible selection bounds, while W/H resizes about the same
point through one preflighted atomic transaction. Physical units compose in the
correct order (native coordinates to display pixels, then inches or
centimeters). Rotation now distinguishes native angle editing from rigid
display-space geometry. A supported single object or mixed selection builds an
absolute plan around the active 3x3 reference point; toolbar buttons and the
off-object handle consume that same plan, and Shift snaps handle preview to
15-degree increments. Multi-object rotation therefore changes positions around
one pivot instead of silently changing every object's local angle in place.
Release emits one generated transaction and one undo item; 360 degrees is a
no-op. Escape restores the unrecorded preview and keeps the selection, so a
second Escape remains available for ordinary deselection/isolation exit.

The first implementation deliberately keeps a strict honesty boundary.
Polygon/PathPatch, marker-free default Line2D, non-offset Line/PolyCollection,
anchor-mode Text, similarity-transform Rectangle/Ellipse, and recursively
complete EditorGroups are eligible only when their appearance, clip, owner,
active-layout participation, and native/display round trip remain exact within
0.25 px. Legend descendants and layout-managed text/backgrounds reject before
mutation even when their leaf geometry is mathematically rotatable. Partial or
non-rectangular clips, Agg filters, wrapped/bbox/effected Text, hatch/effects,
rendered offsets, non-affine/singular transforms, and explicit tuple Rectangle
pivots likewise reject. A single object without a rigid plan may use native R
only when that separate contract is honest.

Independent synthetic QA exercised ten positive Artist variants at four angles
(`13`, `-37`, `90`, `360`) around center and external pivots: all 80 plans,
single/mixed/nested-group handle and toolbar routes, replay, cancellation, and
undo/redo passed; maximum geometry error was `5.68e-14 px`, selection error was
zero, and 360 degrees emitted no mutation or record. The read-only Fig2 fork
then checked 4,830 operation descriptors and 1,816 plans across 227 supported
instances. It accepted 1,702 destinations and atomically rejected 114
clip-limited destinations with zero unexpected failures; maximum accepted
round-trip error was `4.55e-13 px`. Line2D, LineCollection, and PathPatch passed
nonzero single/mixed handle, undo/redo, snapshot, and replay workflows.
FillBetweenPolyCollection remained capability-honest but every real nonzero
destination was already partially clipped, so only its 360-degree no-op was
accepted. All 115 owner-managed instances rejected without mutation, record,
or edit.

The same fork rechecked the reported panel-D Legend failures. Drag and key
alignment errors were at most `2.27e-13 px`, selection indicators matched the
rendered Legend, and `frameon` toggled in place in about `0.124 s` while
preserving Legend identity, children, selection, undo/redo, and subsequent
Line2D dragging. The formal Fig2 remained byte-identical at SHA-256
`b0cd72abf3962cd6cd2354467ad57aa37ecc213332645d7cb56e6f4af598ad70`.

The Align panel now exposes explicit Selection, Key Object, and Artboard
references. Selection alignment uses the union of visible bounds and a single
selected object is a strict no-op; moving one object to the canvas requires the
explicit Artboard reference. The key object is stored independently from the
active primary object, receives a heavier outline, can be changed by clicking
another selected object, and never enters the move or recording plan. Match
Size uses that same explicit key when key mode is active. Automatic Selection
distribution preserves the selection envelope (with two-object distribution a
no-op), while Artboard distribution uses `Figure.bbox`. Key distribution keeps
the key fixed and either reuses the current mean display-space gap or applies a
finite signed pixel gap; negative values intentionally overlap objects. Stable
spatial ordering resolves equal-edge ties.

Every alignment/distribution action measures one immutable visible-geometry
snapshot, preflights the complete delta plan, and only then commits one atomic
undo item. A clip-limited target therefore rejects the whole action before any
Artist, generated record, selection, primary object, reference mode, or key
mutates. Alignment reference/key state is also part of interaction-state and
geometry undo/redo restoration. The regression suite covers all six alignment
directions, positive/zero/negative spacing with the key first/middle/last,
mixed selection lifecycles, exact no-ops, UI control state, clip rejection, and
undo/redo; the current full suite passes 712 tests with 147 skips.

A read-only real-Fig2 fork probe covers all 19 represented categories. Eighteen
align to key/artboard references with at most `1.14e-13` px error; AxesImage
remains the expected typed clip-constraint rejection with zero mutation,
record, or edit. Twelve automatic/numeric distribution workflows have at most
`1.71e-13` px gap error, the visible-width match-size case is within `0.0442`
px, and the real `group_selection` path preserves its key and logical group
owner through alignment and undo/redo. Every successful action creates one
undo item, while reference mode, key, and primary survive undo/redo. The formal
Fig2 remains byte-identical at SHA-256
`b0cd72abf3962cd6cd2354467ad57aa37ecc213332645d7cb56e6f4af598ad70`.

Remaining feature work:

- Freely movable/custom rotation pivots and broader safe coverage such as
  rotationally symmetric Line2D markers.
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
The parent-project legacy `fig2_artist_contract_audit.py` currently enumerates
native `_rotatable_value` and then invokes the direct-manipulation toolbar. That
contract predates the separation between native angle properties,
`native_rotation_handle_support()`, and `RIGID_ROTATE`; it must be migrated
before its rotation section is reused. The rigid milestone was validated by a
separate read-only in-memory fork probe, without changing the legacy script or
formal validation JSON.
