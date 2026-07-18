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

Status (2026-07-18): implemented on
``codex/p0-correctness-performance``. The current implementation is covered by
1,299 passing tests, 178 explicit skips, no strict xfails, Ruff, the full Fig2
interaction probe, and a read-only smoke replay of the unmodified formal Fig2.
The formal file retained SHA-256
``aba67bbd663fd16da535aa30d43f607c7205d096455f44544e518607cdce2dbb``.

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

The 2026-07-18 correctness/performance closeout completed four cross-cutting
foundations. Hit filtering, marquee selection, selection overlays, and Smart
Guides now consume one revisioned display-geometry service with a shared live
Artist roster; draw/structure revisions invalidate it and removed Artists are
pruned before they can become ghost hits. Bring Forward/Backward and Send to
Front/Back now operate on the stable paint order inside the correct owner,
including equal-z-order siblings, and update hit order through one atomic,
replayable transaction. Finally, the embedded Qt lifecycle is idempotent:
close/reopen and Figure replacement tear down managers, timers, callbacks, and
per-Figure history UI without hidden top-level windows, while the source
Figure's default key handler is suspended only while the editor owns it.

Translation, resize, and native-rotation transactions now freeze immutable
absolute display/native destinations. Commit revalidates every selected source,
adapter-specific storage token, EditorGroup membership, transform, viewport,
clip, and layout before taking a rollback snapshot or touching history; a stale
member rejects the complete selection with zero mutation. Zero translation,
identity resize, and full-turn native rotation remain strict no-ops. Matplotlib
layout ownership is also an explicit capability boundary: automatic Axes and
Figure/SubFigure titles, Axis offset text, constrained-layout labels and bbox
extras, Legend layout-only geometry, and container backgrounds reject before
the draw-time owner can pull them away from the preview. Stable manual titles,
ordinary Legend children, and layout-independent labels retain their exact
editing paths.

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

Foundation status (2026-07-18): visible/preview bounds are now explicit and
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

### P1b: Identity-preserving Legend reflow

Implemented on 2026-07-17 as a separate semantic operation. The discarded
donor/transplant prototype preserved the outer Legend but replaced handles,
Text, title, and packer identities, violating direct-selection and Undo/Redo
contracts. The replacement captures a frozen `LegendLayoutSpec/Plan/State`,
reuses every existing Artist and DrawingArea/TextArea, and replaces only
verified standard HPacker/VPacker structure. Manual child edits therefore stay
relative to the newly packed entry instead of disappearing.

The v1 layout spec is deliberately limited to `ncols`, `borderpad`,
`labelspacing`, `handlelength`, `handletextpad`, and `columnspacing`. Current
Axes Legends, retained extra Axes Legends, and Figure Legends use the same live
path. Each control consumes one atomic layout-only history item; restoring the
source spec removes the generated layout command. Multi-Legend semantic plans
roll back every target and generated-change record if any commit step fails.

Generated replay remains dependency-free: a saved block embeds a self-contained
Matplotlib-only reflow primitive before its absolute layout command. It does not
require Pylustrator to remain imported. Baselines are captured before replay,
including legacy command ordering, and Matplotlib's native DraggableLegend is
rebound to the active root across reflow and Undo/Redo.

`mode="expand"`, unknown or customized packers, detached owners, invalid
dimensions, stale plans, and simultaneous Legend/descendant selection reject
before mutation. Compatibility probes cover Matplotlib 3.8.4 and 3.10.8; Qt
signal tests prove that none of the six controls call `Axes.legend()`,
`Figure.legend()`, or `Legend.remove()`.

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

### P1c: Movable custom rotation pivots

Implemented as selection-level editor state, without changing the existing
``RigidRotationPlan`` or any generated-source schema. A selection exposes the
movable on-canvas pivot only when every member already has a complete
``RIGID_ROTATE`` plan. Native-only rotation continues to use the Artist's true
fixed pivot and cannot pretend to honor an arbitrary origin.

The custom pivot is stored in root-Figure physical inches rather than raw
display pixels or normalized Figure fractions. It therefore stays at the same
physical artboard coordinate across DPI/HiDPI and Figure-size changes. Dragging
the marker mutates only QGraphics state and creates no generated command or
Undo item. Intermediate mouse frames update only the overlay; release, reset,
or Escape emits one final UI-state notification. Escape restores the pre-drag
pivot, changing selection membership clears it, changing only the primary/key
preserves it, and clicking any point in the 3x3 locator resets it even when that
point was already active. Handle and toolbar rotation both resolve the same
pivot and destination plan; a coincident handle automatically moves to retain a
nonzero lever arm.

The pivot and 3x3 reference are captured by both geometry restore closures and
``InteractionState``. A late history-insertion failure now restores Artist
geometry, generated bookkeeping, edit history, reference/pivot state,
selection, and primary object as one transaction. Synthetic Qt coverage spans
drag/release/Escape, handle/toolbar, native denial, group and interaction-state
Undo/Redo, clip rejection, DPI/Figure-size changes, and the 3x3 reset path. A
read-only Fig2 fork accepted Line2D, LineCollection, and PathPatch with maximum
commit/Undo error ``2.27e-13 px``; the clipped FillBetweenPolyCollection was an
expected zero-mutation typed rejection. The formal editable Fig2 remained
byte-identical at SHA-256
`aba67bbd663fd16da535aa30d43f607c7205d096455f44544e518607cdce2dbb`.

### P1d: Marker-aware Line2D rigid rotation

Implemented for marker-free lines, lines whose marker paint is absent, and
visible centered canonical circle paths (including ``o``, ``.``, and an exact
custom unit-circle Path). The marker transform must remain rotationally
symmetric after scaling to renderer pixels: anisotropy plus the full sweep of
any center offset may contribute at most ``0.25 px``. Partial fill paths,
non-circle glyphs, filters/sketch effects, non-finite rendered dimensions, and
unsupported ownership/clip states reject before mutation.

The destination envelope no longer combines all Line2D controls with a source
subset's asymmetric outsets. It applies Matplotlib's real ``markevery``
semantics independently to the representable destination, computes line and
marker paint there, and requires the destination marker centers to equal the
rigidly transformed source centers within ``0.25 px``. This closes the measured
``246.687 px`` subset error and also catches rare floating-distance ties where
preview and commit could agree on the wrong marker identity. NaN/masked slots
remain in the marker index stream, while an isolated point across a NaN gap no
longer inflates the visible line-stroke box.

Rigid rotation now uses a two-stage prepare/apply commit boundary. Each frozen
``RigidRotationPlan`` keeps compact tokens only for stable source semantics,
such as Line2D raw storage and a native angle, instead of byte-hashing restored
display floats or enumerating Artist-specific appearance fields. Prepare
rebuilds a live candidate and recomputes its native/display destination and
paint/clip envelope; every selected Artist must prepare successfully before any
apply mutates geometry. Mouse-move preview is the trusted fast path: it applies
the plan produced for that frame without a second ``O(N)`` revalidation,
preserving large-Line2D frame latency while every commit remains stale-safe and
atomic.

The interactive path was optimized as part of the same correctness contract.
Line2D forward/inverse transforms are vectorized, a handle gesture reuses its
source controls and visible envelope, and ``RigidRotationPlan`` stores compact
byte-backed immutable arrays instead of hundreds of thousands of Python
tuples. A 100k-point plan fell from roughly ``1.3 s`` to ``55--64 ms`` in the
local probe; 10k points fell to about ``6--8 ms``. Ninety-six Agg raster cases over
four DPIs, eight ``markevery`` forms, and three circle paint styles matched
plan/commit analytic bounds within ``1e-8 px``. Nonzero-alpha raster coverage
can extend up to ``1.77 px`` beyond that analytic paint envelope due to
antialiasing, which remains part of the renderer-faithful P1 work below rather
than a marker-specific alignment constant.

The real Fig2 fork contains 553 Line2D instances and 245 with visible marker
paint. Its 242 non-circular marker lines and three Legend-owned marker lines all
return typed Q denials with no selection/support exception. The one ordinary
400-point affine Line2D representative was converted only in memory to a circle
marker and exercised with integer, tuple, list, float, and float-tuple
``markevery`` forms; all five plan/commit errors were ``0 px`` and restore error
was at most ``1.14e-13 px``. The formal editable Fig2 remained byte-identical at
SHA-256
`aba67bbd663fd16da535aa30d43f607c7205d096455f44544e518607cdce2dbb`.

### P1e: Lossless Line2D raw-data semantics

Implemented on 2026-07-17 without converting Matplotlib's original data into
the finite display array used for geometry. Independent ``MaskedArray`` x/y
containers now retain shape, dtype, raw hidden values, independent masks,
``nomask``, fill values, and hard-mask state through snapshot, translation,
rigid-rotation planning, generated replay, Undo, and Redo. Numeric operations
change only rows that are valid on both axes. Integer and float32 sources
promote when display-space motion cannot be represented exactly instead of
silently rounding back into the original dtype.

Categorical, datetime, and custom-unit Line2D remain selectable and retain
appearance editing. Their geometry operations return an operation-specific
typed denial rather than failing after mutation. Category and datetime arrays
have lossless replay; arbitrary object values reject serialization until a
semantic unit codec exists. NumPy datetime/timedelta scalars use explicit raw
integer plus unit literals, avoiding the ``numpy.datetime64`` versus
``np.datetime64`` repr difference between NumPy 1.23 and newer releases.

Frozen Q plans include a compact content digest over data, mask, dtype, shape,
fill value, and hard-mask state, so an in-place source mutation invalidates the
plan without retaining a second 100k-point source copy. In local probes, the
digest cost about ``0.6 ms`` and added about ``1.2 ms`` to a 100k-point Q plan.
The ordinary 100k ndarray replay path fell from roughly ``323 ms`` to ``40 ms``;
lossless MaskedArray replay was about ``104 ms``. The same semantics pass on
the minimum supported Matplotlib 3.8.4 / NumPy 1.23.5 stack.

### P1f: Indexed interaction and Smart Guides

Implemented on 2026-07-17 as two fail-open acceleration layers. Pointer hit
testing now uses a revisioned display-space grid only as a conservative coarse
filter; native/adaptor containment remains authoritative. Custom hit contracts,
composite Annotation arrows, Text bbox patches, Legend frames, Axes patches,
and any object without a provably complete envelope stay in an always-tested
set. A failed build atomically returns to the original full scan. On the real
Fig2 fork, indexed and full hit stacks agreed at every oracle point; 2,106 warm
queries had ``1.373 ms`` median and ``4.019 ms`` p95 versus the former roughly
``63.9 ms`` scan path.

Deferred Line2D preview now keeps one contiguous array instead of one Python
array per vertex. A 100k-point drag frame fell from roughly ``49.7 ms`` and
``16 MB`` peak to ``3.557 ms`` median, ``3.949 ms`` p95, and ``4.803 MB`` peak.

Smart Guides use immutable display-pixel geometry and one plan for the accepted
preview and commit. Edge, center, Text baseline, insertion-anchor, cross-feature,
and equal-gap guides share deterministic distance/semantic/z-order/paint-order
ties. Equal-gap neighbours use an exact vectorized path for scenes up to 1,024
objects and retain the ``O(n log n)`` interval sweep above that threshold.
Shift limits both the plan and its overlay to the constrained axis; Alt/Option
temporarily disables snapping. Escape, Undo/Redo, V/A tool changes, isolation,
and deactivation all close the same gesture lifecycle before changing policy or
history. Formatter-owned tick labels and empty Text remain editable where
appropriate but cannot create invisible or unstable guide sources.

Scene measurement is never performed synchronously by the normal mouse-press
path. Draw completion schedules bounded Qt-idle batches; an incomplete cache
uses legacy snaps for that gesture. With the Fig2 scene warm, gesture creation
is ``0.262 ms`` median (legacy ``getSnaps``: ``0.960 ms``), the first motion's
selection-specific exact equal-gap index is about ``10.1 ms``, and later queries
are ``0.089 ms`` median. Preview application succeeds before a plan is exposed
or drawn, so a failed adapter cannot leave an overlay ahead of real geometry.

The same bounded idle turns now build the conservative hit index incrementally
and publish it only when complete. On the current Fig2 fork, the first hit after
idle warmup fell from ``49.72/52.12 ms`` median/p95 to
``0.445/0.553 ms``; warm dense hits remain below ``4 ms`` p95. Invalidation
remains about ``3.46 ms`` and performs no renderer measurement. Idle slices
measured ``4.16/5.04/7.50 ms`` median/p95/max, below one 60 Hz frame, while
unmeasured Artists remain conservative native-hit candidates.

Within those same slices, hit envelopes are now completed before the remaining
Smart Guide capture. This reduced the real-Fig2 time-to-published-hit-index from
``126.27 ms`` to ``66.89 ms`` without delaying Guide completion. Before the
atomic index is ready, the conservative dense-point scan improves from
``41.80 ms`` immediately to ``22.82 ms`` partway through capture; after publish
it is ``1.88 ms``. All three stages return the identical eight-object hit stack
without synchronous pointer-side bounds measurement.

Ordinary hover and click now consume that conservative candidate stream only
until the first selection-policy decision. Object Selection can return its
first leaf/group promotion, Direct Selection can return its first leaf, and an
unsupported foreground object remains an immediate barrier. Alt click-through,
right-click candidate lists, double-click isolation, and Direct Selection on an
ambiguous group shell still consume the complete ``HitStack``. Across 480
real-Fig2 cold/half/warm queries, streamed and full resolution agreed exactly on
target, raw leaf, and blocked state. Cold median/p95 fell from
``49.245/58.518 ms`` to ``4.496/18.648 ms``; half-built index queries fell from
``31.335/36.170 ms`` to ``3.301/15.199 ms``; warm queries fell from
``4.571/5.730 ms`` to ``0.463/1.569 ms``. The remaining cold p95 edge comes
from a fail-open unmeasured tail whose foreground hit is late in paint order;
no renderer measurement or object is dropped to avoid it.

The combined repository suite now passes 1,299 tests with 178 explicit skips.
Matplotlib 3.8.4 / NumPy 1.23.5 passes the 87 new and directly related tests.
A read-only real-Fig2 fork produced edge/center hits, one atomic history item,
``2.27e-13 px`` preview/commit error, and zero Undo/Redo error. The formal Fig2
remained byte-identical at SHA-256
``aba67bbd663fd16da535aa30d43f607c7205d096455f44544e518607cdce2dbb``.

The shared per-revision display-geometry service is complete. It now supplies
the conservative hit index, marquee bounds, selection geometry, and Smart Guide
capture from the same revision and live roster; the remaining work starts after
that common cache boundary.

### P1g: Cached content previews and renderer paint envelopes

Implemented on 2026-07-18 as two read-only renderer layers below the semantic
transform boundary. A content ghost is a disposable visual aid: the adapter and
``TransformPlan`` remain the only preview/commit/Undo truth, and a missing ghost
never changes the accepted operation or generated source. Draw completion or a
selection change schedules capture through Qt idle; pointer press only validates
an immutable renderer/revision/selection/source token, and motion only translates
one ``QGraphicsItem``.

The cache has explicit memory, bounded-source, composite-leaf, and idle-work
budgets. It renders audited shallow clones rather than live Artists, replays a
standard Legend as its frame plus OffsetBox paint leaves in real paint order,
and retains one budgeted scratch ``RendererAgg``. No-op click/cancel hides and
reuses the still-current pixmap. A real-Fig2 prototype originally deep-copied
Matplotlib's cyclic Artist/Transform/OffsetBox graph, causing a 118--139 ms
generation-2 GC pause about every sixth capture. The shallow-clone design
removed those collections: over 40 samples, Legend capture measured
``3.72/4.55/7.50 ms`` median/p95/max and Text measured
``2.11/2.37/3.35 ms``. Legend/Text hot activation remained at or below
``0.454/0.083 ms`` p95 and motion was about ``0.029 ms`` p95.

The v1 bitmap contract is deliberately translation-only. A fixed clip or
renderer edge cannot be transformed with an already clipped raster, and an
affine bitmap resize would incorrectly scale fixed display-space stroke,
marker, font, and hatch appearance. Active rectangular/non-rectangular clips,
implicit Annotation data clipping, source/destination canvas contact,
non-translation matrices, large sources/composites/canvases, custom draw
contracts, scalar-mappable collections, and AxesImage therefore fall back to
the analytic preview before mutation. On the current Fig2 fork, 194 of 580
selectable objects receive content ghosts (161 Text, 9 Legends, 9 Line2D,
13 Rectangles, and 2 Annotations); the other 386 retain the existing analytic
path. Panel-D Legend and Text ghost/commit/Undo errors were at most
``2.28e-13 px``, and pixel-oracle alpha plus premultiplied RGB matched exactly.

``DisplayGeometryCache`` now also exposes a separate opt-in Agg paint-envelope
cache. A capture result is explicitly ``exact``, ``conservative``, or
``unavailable``; lookup never draws or falls back to analytic geometry.
Audited non-stale Patch, Line2D, and fixed-color Collection primitives render
only through disposable clones, preserving clip, cap/join, marker,
antialiasing, and whitelisted path-effect pixels. Pending Artists,
scalar-mappable collections, custom callbacks/transforms/effects, unsupported
draw contracts, and over-budget rasters remain conservative. Capture success,
failure, and denial leave the Artist, clip dependency, Legend/Axes/Figure
owners, callbacks, and derived draw state unchanged.

The combined suite passes 1,299 tests with 178 explicit skips and Ruff. The
read-only Fig2 fork retains identical accepted preview/commit/Undo geometry,
and the formal file remains byte-identical at SHA-256
``aba67bbd663fd16da535aa30d43f607c7205d096455f44544e518607cdce2dbb``.

Next implementation sequence, with no interaction-latency regression allowed:

1. Build Direct Selection path/endpoint editing and inline text editing on the
   same plan/transaction boundary; do not mutate raw arrays directly from UI
   code.
2. Add semantic duplicate/copy/paste, Select Same, and style transfer using
   stable locators rather than copying Matplotlib ownership links.
3. Extend content ghosts only through fixed scene-clip layers and closed visual
   state contracts for collections/images; do not broaden the current bitmap
   affine approximation.
4. Add workflow breadth only after those edit contracts are stable: rulers,
   persistent guides/grids, zoom/pan shortcuts, templates, and scientific-role
   protection.

Performance gates for each item are: pointer press no slower than the legacy
path, first preview below one 60 Hz frame on the Fig2 fixture, warm preview/query
below ``4 ms`` p95, bounded memory for 100k-point Artists, and identical accepted
preview/commit/Undo geometry within ``0.25 px``.

## P2: workflow breadth

- True paint-order Send to Front/Back and Bring Forward/Backward are complete.
- Duplicate/copy/paste, Select Same, and style copy remain to be implemented.
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
