# Artist operation support matrix

This document is an independent QA audit of the adapter contract on
`main`, updated 2026-07-18. It treats the
registry and `ArtistCapabilities`/`OperationSupport` as executable promises,
not merely implementation metadata.

The contract test is
`tests/test_artist_operation_contract_matrix.py`.  It enumerates every
built-in registry entry, constructs the concrete Artist, and tests the same
operation through display-space controls, snapshots, generated changes, and
rendered bounds.  Tests use Python 3.13.5, Matplotlib 3.10.8, NumPy 2.4.0, the
Agg renderer and a 0.25 display-pixel preview/commit tolerance. Every confirmed
defect found by this matrix now has a passing regression contract; no strict
xfails remain.

## Notation and scope

Advertised operations are abbreviated as:

- `S`: select and expose finite selection bounds
- `M`: display-space translate
- `Z`: geometry resize
- `R`: native rotation
- `Q`: rigid display-space rotation around a shared reference pivot
- `A`: appearance scaling without geometry or layout mutation
- `L`: identity-preserving layout reflow
- `N`: snapshot/restore
- `C`: serialize to generated changes

Every row was also tested against all ten `TransformOperation` values.
Unsupported operations must return `OperationSupport.supported == False` with
a nonempty reason.  Direct translate, resize, rotate, and snapshot entrypoints
must reject without geometry or generated-change mutation. `scale_appearance`
is implemented only for the lossless first-batch Text, Line2D, PathCollection,
LineCollection, and PolyCollection contracts below. `reflow_layout` is limited
to standard Legend OffsetBox trees; `edit_points` remains denied for every
built-in adapter.

`PASS` means every advertised transform and its rejection paths satisfy the
contract. `DENIED` means the absence of edit support is deliberate and the
denial contract passes.

Concrete adapter registrations are exact-only by default. An unregistered
subclass of one of those semantic types therefore resolves to a fail-closed
adapter with a typed reason rather than silently inheriting its parent's
geometry and replay assumptions. Subclass inheritance is allowed only when a
registration explicitly declares the `VALIDATED` policy after checking the
complete mutation, snapshot, and replay contract. The generic `Artist` fallback
is validated only as a deliberately uneditable denial contract.

`R` is deliberately a native property contract, not by itself permission to
show a direct-manipulation handle. The handle/toolbar additionally requires
`native_rotation_handle_support()`: complete visible geometry must rotate
rigidly around the displayed pivot. Default-mode Text, Annotation arrows,
anisotropic or reflected Patch transforms, hatch, path effects, Agg filters,
and layout-owned targets do not expose a misleading object-rotation control.
Default-mode Text through path-effect cases may retain saveable native angle
properties for explicit property editing; Agg-filter/layout-owned cases deny
rotation entirely.
Anchor-mode Text and ordinary similarity Patches use Q preferentially.

## Per-type matrix

| Registered Artist | Adapter | Advertised | Actually tested | Result | Gap or explicit limitation |
|---|---|---:|---|---|---|
| `Artist` fallback | `ArtistAdapter` | none | Registry resolution; all operation descriptors; direct M/display-transform/Z/R/N rejection and zero mutation | **PASS** | Deliberately uneditable. Semantic entrypoints reject through `UnsupportedArtistError` before geometry arithmetic. |
| `EditorGroup` | `EditorGroupAdapter` | S M Z Q N C | Bounds; M/Z/Q preview and commit; recursive shared-pivot plan; per-leaf fixed-stroke resize; N; replay; direct and outer-plan failing-member rollback; single-emission recording | **PASS** | Native R is denied; Q is advertised only when every leaf has a complete rigid plan. |
| `Axes` | `AxesAdapter` | S M Z N C | Bounds; M/Z; parent Figure size; N; real replay; nonuniform fixed-aspect preview/commit | **PASS** | R is explicitly unsupported. Fixed-aspect Axes advertise the `fixed_aspect` constraint and normalize preview/commit through the same native-space rule. |
| `Text` | `TextAdapter` | S M R Q A N C conditionally | Axes/data/figure/display transforms; visible text bounds; M; native R; anchor-mode Q; point-font A preview/commit/Undo/Redo/replay; complete N; real replay; Axis-owned property edits | **PASS** | A is limited to visible ordinary point Text with an invertible affine transform; Annotation, Legend/layout-owned, wrapped, bbox, TeX, filtered, sketched, or path-effect Text rejects. Q retains its stricter anchor-mode contract. Geometry resize remains denied. |
| `Annotation` | `AnnotationAdapter` | S M R N C | Mixed data/axes endpoint coordinates; text and arrow-stroke bounds; two control points; M; R; complete N; real replay | **PASS** | Geometry resize is explicitly denied. |
| `Legend` | `LegendAdapter` | S M L N C conditionally | Visible handles/text/title union; `frameon=False/True`; M; logical-owner persistence; current/extra/Figure L through frozen plans; six-field layout Undo/Redo; Matplotlib-only replay; N; real replay | **PASS** | L preserves Legend/frame/handle/Text/title/DrawingArea/TextArea identity and replaces only verified standard HPacker/VPacker nodes. `mode="expand"`, custom packers/state, detached owners, and Legend/descendant co-selection reject. Explicit composite handlers can use L even when full Legend reconstruction/replay remains unavailable. Geometry resize remains denied. |
| `Line2D` | `Line2DAdapter` | S M Q A N C conditionally | Data/log transforms; visible line, marker-path, and marker-edge bounds; M; affine shared-pivot Q with marker-aware `markevery`; linewidth/markersize/markeredgewidth A; appearance-only Undo/Redo and replay; N; lossless ndarray/MaskedArray `.set_data` replay | **PASS** | Raw ndarray and independent MaskedArray x/y semantics—including hidden payloads, masks, dtype/shape, fill value, and hard-mask state—survive geometry plans, replay, and Undo/Redo. Categorical and datetime replay is lossless; arbitrary custom-unit object values deny serialization until a semantic codec exists. A and Q retain the ownership, paint, marker, clip, and transform preflight limits described below. Geometry resize remains denied. |
| `AxesImage` | `AxesImageAdapter` | S M Z N C | M/Z preview/commit; extent replay; N; x/y viewport and Axes position invariance | **PASS** | R is explicitly unsupported. Moving or resizing the image does not autoscale the camera. |
| `Rectangle` | `RectangleAdapter` | S M Z R Q N C conditionally | Visible-stroke bounds; fixed-stroke M/Z; native R; common-pivot Q under similarity/reflection transforms; rotated M; rotated-resize denial; N; replay | **PASS** | Q additionally requires `rotation_point` `xy`/`center`, no hatch/effect, and no Legend/container owner. Tuple pivots need a richer plan. Z is only advertised at angles equivalent to 0 degrees modulo 360. |
| `Ellipse` | `EllipseAdapter` | S M Z R Q N C conditionally | Visible-stroke bounds; fixed-stroke M/Z; native R; common-pivot Q under similarity/reflection transforms; rotated M; rotated-resize denial; N; replay | **PASS** | Q denies non-similarity transforms, hatch/effects, and Legend/container owners. Z is disabled for rotated ellipses. |
| `Arc` | `ArcAdapter` | S M N C conditionally | Visible-stroke bounds; affine center M; preview/commit; N; `.set_center` replay | **PASS** | Translation preserves width, height, angle, and angular span. Non-affine or owner-managed targets reject; semantic resize and rotation remain denied. |
| `Circle` | `CircleAdapter` | S M N C conditionally | Visible-stroke bounds; affine center M; preview/commit; N; `.set_center` replay | **PASS** | Translation preserves one semantic radius without stretching. Non-affine or owner-managed targets reject; resize and rotation remain denied. |
| `FancyArrowPatch` | `FancyArrowPatchAdapter` | S M N C | Visible-stroke bounds; endpoint M; rendered preview/commit; N; real `.set_positions` replay | **PASS** | Z/R are explicitly denied. |
| `ConnectionPatch` | `ConnectionPatchAdapter` | none | Specific MRO resolution; all operation descriptors; direct M/Z/R/N rejection and zero mutation | **DENIED** | Deliberately blocked because its endpoints can occupy unrelated coordinate systems. |
| `FancyBboxPatch` | `FancyBboxPatchAdapter` | S M N C when affine | Visible-stroke bounds; affine M/N/replay; non-affine denial and zero mutation | **PASS** | Geometry resize is unsupported. A non-affine data transform disables the editable contract. |
| `RegularPolygon` | `RegularPolygonAdapter` | S M N C | Visible-stroke bounds; center M; preview/commit; N; `.xy` replay | **PASS** | Z is denied until it changes semantic radius rather than stretching its center point. |
| `CirclePolygon` | `CirclePolygonAdapter` | S M N C conditionally | Visible-stroke bounds; affine center M; preview/commit; N; `.xy` replay | **PASS** | Translation preserves radius, orientation, and resolution. Non-affine or owner-managed targets reject; semantic resize and rotation remain denied. |
| `Wedge` | `WedgeAdapter` | S M N C | Visible-stroke bounds; center M; preview/commit; N; `.set_center` replay | **PASS** | Z/R are explicitly unsupported. |
| `Polygon` | `PolygonAdapter` | S M Z Q N C conditionally | Visible-stroke bounds; fixed-stroke vertex M/Z; affine vertex Q; preview/commit; N; `.set_xy` replay | **PASS** | Native R is unsupported; Q requires invertible affine geometry with no hatch/effect or Legend owner. |
| `PathPatch` | `PathPatchAdapter` | S M Z Q N C conditionally | Visible-stroke bounds; fixed-stroke path/codes M/Z; affine control-path Q; preview/commit; N; `Path` replay | **PASS** | Native R is unsupported; Q requires invertible affine geometry with no hatch/effect or Legend owner. |
| `PathCollection` | `PathCollectionAdapter` | S M A N C conditionally | Renderer item-count semantics; per-item marker-path/size/stroke envelopes; masked offsets; affine/log offset M; marker-area × factor² and linewidth × factor A; appearance-only Undo/Redo/replay; N; `.set_offsets` replay | **PASS** | A requires a visible fill with non-zero path area or a visible strokable edge; hatch/effects/Legend/layout ownership reject. A remains valid when geometry cannot be snapshotted because its transaction stores appearance only. Z remains denied. |
| `LineCollection` | `LineCollectionAdapter` | S M Q A N C conditionally | Per-segment linewidth envelopes; NaN path-break preservation; affine/log multi-segment and explicit-offset M; non-offset affine Q; linewidth A; appearance-only Undo/Redo/replay; N; path/offset replay | **PASS** | A requires a visible strokable rendered path and rejects hatch/effects/Legend/layout ownership. Q retains its non-offset affine contract. Z remains denied. |
| `PolyCollection` | `PolyCollectionAdapter` | S M Q A N C conditionally | Per-polygon visible-edge envelopes; affine/log multi-path and explicit-offset M; non-offset affine Q; visible-edge linewidth A; appearance-only Undo/Redo/replay; N; path/offset replay | **PASS** | A scales only visible stroked edges; face-only polygons have no supported appearance dimension in v1. Q retains its strict non-offset/affine/no-effect/no-owner contract. Z remains denied. |
| `FillBetweenPolyCollection` (when provided by Matplotlib) | `PolyCollectionAdapter` | S M Q A N C conditionally | Exact concrete registration; the complete PolyCollection bounds, transform, appearance, snapshot, denial, and replay contract | **PASS** | It does not inherit PolyCollection mutation semantics implicitly. Real destinations remain subject to the same clip and offset preflight. |

Patch adapters share one common-stroke envelope implementation. Resizable
patches additionally separate transformable geometry from fixed display-space
appearance outsets, so a thick stroke does not scale during geometry resize and
the committed visible box remains at the preview handle position.

## Cross-cutting contract results

The following behavior passes for every type that advertises it:

- Registry resolution selects the exact most-specific adapter, including
  `Annotation` before `Text`, `ConnectionPatch` before `FancyArrowPatch`, and
  exact `Arc`, `Circle`, and `CirclePolygon` adapters rather than their broader
  Patch ancestors. Unvalidated semantic subclasses—including Matplotlib's 3D
  Line, Text, and Collection variants—fail closed before selection or mutation.
- Display-space translation moves control points and final selection bounds by
  the preview delta within 0.25 px.  The selected object is the object mutated;
  parent Axes and unrelated figure-level sentinel text do not move. Generated
  tick-label Text explicitly denies translation because its Axis owns position.
- Translation, geometry resize, and native rotation freeze immutable absolute
  destinations. Before any snapshot, tracker write, or Artist mutation, commit
  revalidates every source geometry, adapter storage token, group membership,
  transform, viewport, clip, layout, and destination mapping. A stale member
  rejects the complete mixed plan through `StaleTransformPlanError`; zero
  translation, identity resize, and 360-degree rotation record nothing.
- Geometry resize matches preview controls and final visible bounds within
  0.25 px for EditorGroup, Axes, AxesImage, Rectangle, Ellipse, Polygon, and
  PathPatch. Thick patch strokes remain fixed in display pixels; both direct
  resize and deferred drag land on the preview box.
- Native rotation renders and records correctly for Text, Annotation,
  Rectangle, and Ellipse. Rigid rotation uses an absolute destination plan for
  every supported leaf, so a mixed Text/Line2D/Patch/Collection selection
  shares one pivot instead of rotating each object in place. A complete rigid
  selection may move that pivot freely; native-only rotation retains its true
  Artist pivot. Handle preview, toolbar commit, generated changes, and
  Undo/Redo consume the same plan.
- Snapshots restore before/after geometry and native rotation. Failed
  multi-target transforms restore both artist state and generated-change
  bookkeeping atomically.
- Appearance plans are frozen absolute destinations. Preflight never mutates
  the target; post-draw bounds match the plan within 0.25 px; factor `1` is a
  strict no-op while near-one factors remain real edits. Appearance Undo/Redo
  stores only paint/font state, so normal geometry snapshots stay fast and an
  appearance edit does not require an invertible geometry transform.
- Appearance recording is semantically isolated: Line2D emits only linewidth,
  markersize, and markeredgewidth commands; Collections emit only linewidths
  and PathCollection sizes. Geometry moves and rigid rotations do not freeze
  unrelated appearance, and appearance edits do not emit data/offset commands.
- Legend layout plans are absolute and layout-only. Current Axes, retained extra
  Axes, and Figure Legends share one live path; mixed parent/descendant
  selections reject before mutation, multi-Legend commits roll back atomically,
  native draggable state is rebound, and generated replay embeds a Matplotlib-
  only helper instead of depending on Pylustrator at runtime.
- Real `ChangeTracker` commands replay translated rendered bounds for all 18
  serializable registry types. Axis-label replay additionally covers both
  label position and `labelpad`.
- Generated numeric literals preserve exact finite-float round trips, qualify
  NaN/Inf through `np`, and import NumPy in saved blocks. Tiny log coordinates,
  masked scatter, NaN line breaks, and a line on a `1e-12`-wide axis replay
  without display-space amplification.
- AxesImage operations preserve x/y limits. Logical-group member operations do
  not move their parent Axes.
- Legend bounds follow visible handles, text, and title; the invisible layout
  frame contributes no padding, while a visible frame is included.
- Annotation bounds include the visible arrow stroke as well as text/bbox
  artwork. Nested Legend and EditorGroup measurements reuse one immutable
  geometry value per Artist and selection action.

Native R retains the `native_rotation` preview strategy. Q has a distinct
`rigid_rotation` strategy and an immutable `RigidRotationPlan` containing the
absolute display controls, visible selection envelope, native angle destination
where needed, and recursive group member plans. Resize rejects off-diagonal
rotation/shear matrices so callers cannot bypass these semantic contracts.

## P0 findings fixed after the independent audit

### Axis-owned text properties now follow their semantic owner

Matplotlib exposes axis titles and tick labels as `Text`, but they do not have
ordinary Text lifecycle semantics. An empty axis-title slot is still a live
object, while tick-label content is formatter output and is overwritten on the
next draw. The legacy property path violated both rules: empty text serialized
only `text=''`, dropping font and geometry state, and direct `set_text()` on a
tick label appeared to succeed before draw silently reverted it.

The editor now initializes property baselines for every live Text descendant.
Empty existing Text serializes its complete changed state, so font, colour,
style, position, content, Undo/Redo, and replay remain lossless. Tick-label
content uses one Axis-owned semantic transaction: it materializes the current
tick/label mapping, restores the original view limits, preserves Text identity,
and records a replayable `set_*ticks(...), set_*lim(...)` command. Mixed text
selections, no-ops, and injected recording failures are atomic.

Visible non-empty tick labels join the canvas hit inventory in both Object and
Direct Selection. Their selection bounds are exact, but dragging is disabled
with a typed Axis-ownership reason; a click therefore cannot fall through and
move the containing Axes. They are excluded from marquee selection because a
formatter-owned leaf cannot participate in a rigid mixed transform.

The fixed real-Fig2 audit covers panel A's empty ylabel and all 11 visible
y-major tick labels: 22/22 Object/Direct hits select the exact Text without
starting a drag, 11/11 translations reject with zero mutation/record/edit, and
8/8 text/font-size draw, Undo/Redo, and replay workflows pass. Default Object
whole-canvas marquee remains 364 non-container objects with zero tick labels or
Axes. The formal Fig2 SHA-256 remains
`b0cd72abf3962cd6cd2354467ad57aa37ecc213332645d7cb56e6f4af598ad70`.

### Common-pivot rotation is now a first-class semantic operation

`RIGID_ROTATE` is intentionally separate from native angle editing. Its plan
rotates complete display geometry around the active 3x3 point on the selected
bounds and stores absolute destinations. A single geometry-backed Artist uses
the same Q path as a mixed selection; only a single object without a complete Q
plan may fall back to a separately verified stable-visual native R contract.
Multi-selection never silently applies equal local angle deltas.

The movable shared pivot is non-document interaction state. It is stored in
root-Figure physical inches, so DPI/HiDPI and Figure-size changes do not turn it
into a stale raw-pixel position. Dragging it changes only the QGraphics overlay:
no Artist, generated record, or undo item is touched. Escape restores the
pre-drag pivot; selection membership changes clear it; primary/key changes keep
it; and clicking any 3x3 reference point (including the already-active point)
returns rotation to that reference. Both ordinary geometry restore closures and
``InteractionState`` preserve it across Undo/Redo. Native-only and incomplete
mixed selections never expose a movable marker.

The initial permissive prototype exposed why type checks alone are not enough.
Resize accepted off-diagonal matrices and reported success for objects whose
rendered geometry missed by 16--69 px. Partial/non-rectangular clips produced
errors as large as 148 px, wrapped Text reflowed by about 75 px, and explicit
tuple Rectangle pivots missed by up to about 92 px. Q therefore uses a strict
v1 whitelist and typed preflight rejection:

- geometry-backed Polygon/PathPatch/Line2D/LineCollection/PolyCollection must
  have writable, invertible affine controls and no unsupported appearance or
  rendered-offset semantics; arbitrary Agg filters reject because their pixel
  offsets are not guaranteed to rotate with geometry;
- a visible Line2D marker must be a canonical circle path with no partial-fill
  alternate path and a centered similarity transform within the 0.25 px
  rendered tolerance. Marker centers are reselected with Matplotlib's exact
  ``markevery`` algorithm at the representable destination and compared with
  the rigidly transformed source centers; a float-distance tie that changes
  vertex identity therefore rejects rather than jumping to a different point;
- Text must use stable anchor rotation without wrap, bbox, path effect,
  transform-relative angle, Annotation semantics, or a Matplotlib layout owner;
- Rectangle/Ellipse require a display-similarity transform; Rectangle also
  requires the native `xy` or `center` rotation point;
- rectangular clipping requires both source and destination envelopes to be
  fully contained; partial and non-rectangular clips reject before mutation;
- EditorGroup recursively requires every leaf and is its one recording owner.

Every planned display destination is converted to native coordinates and back
before mutation. Non-finite structure must be preserved and the maximum
round-trip error must be at most 0.25 px; the plan stores that representable
native destination and applies it directly. This rejects numerically singular
coordinate systems by measured error rather than a fragile global condition-
number cutoff.

Line2D plans preserve NaN and masked slots for marker selection while stroke
bounds include only vertices belonging to a real finite segment, not isolated
MOVETO points across a gap. The raw-data codec separately preserves the original
MaskedArray containers, hidden underlying values, independent masks, dtype,
shape, fill value, and hard-mask state through snapshot/replay. Coordinate
conversion is vectorized, rotation gestures reuse their immutable source
geometry, and large plan arrays are backed by immutable bytes instead of
per-point Python tuples. In the 100k-point probe this reduced one plan from
roughly 1.3 s to 55--64 ms; the 10k-point case fell to about 6--8 ms. Capability
checks remain about 2--4 ms at 100k points and do not run the potentially
expensive float-distance marker resolver.

Semantic ownership is checked independently from geometric representability.
Every recursive Legend descendant—including nested errorbar, stem, and tuple
handler glyphs—denies independent R/Q even when its raw Line2D/Rectangle/
Collection geometry could rotate exactly. Axes titles, tick labels, offset
text, axis labels, Figure/SubFigure super labels, and Figure/SubFigure/Axes
background patches likewise deny rotation because their owner can rewrite the
position or lacks an independent replay identity. Ordinary `ax.text`,
`fig.text`, `subfig.text`, and user-created patches remain supported. A
Figure-level structural owner inventory makes these checks O(1) within a
selection snapshot and invalidates on Legend replacement, retention, removal,
or packer reconstruction.

An active constrained/tight layout introduces a second ownership channel. An
otherwise ordinary Axes child that is visible, `in_layout=True`, and contributes
a finite default bbox-extra can move the Axes during draw after it rotates.
Such Text/Line/Patch targets conservatively deny R/Q in v1; setting
`in_layout=False` makes the independent-object intent explicit. Collections
whose tight bbox is Matplotlib's null box are not overblocked. This closes
observed 10--49 px feedback errors while leaving Figure/SubFigure free text and
ordinary no-layout figures available.

Both TransformPlan and direct EditorGroup apply paths roll back geometry and
recording state if a later target/member or serializer fails. A 360-degree
toolbar request is a semantic no-op. Escape during either native or rigid
handle preview restores the pre-gesture transaction and preserves selection;
a second Escape performs the ordinary deselection/isolation action, so an
unrecorded preview can never survive as a ghost mutation.

### Rotation is now part of adapter snapshots

The QA baseline found that `ArtistAdapter.snapshot()` stored only local
control-point positions. Text, Annotation, Rectangle, and Ellipse now include
their native angle in snapshots and restore it only when it changed.

Verified behavior:

- Single-object snapshot/restore is complete for all four rotatable types.
- If a later target fails during `TransformPlan.commit()`, earlier native
  angles are restored.
- Existing explicit old/new angle Undo/Redo remains unchanged.

Passing tests:

- `test_rotatable_snapshot_restore_includes_rotation_state` (four types)
- `test_failed_multi_artist_rotation_rolls_back_native_angles`
- Gesture/history checks:
  `test_native_rotation_undo_redo_restores_angle_and_bookkeeping` and existing
  `test_rotation_routes_through_artist_capabilities_and_undo`

### Failed transforms now restore generated bookkeeping atomically

`TransformPlan.commit()` now captures each unique change tracker alongside its
artist snapshots. On failure it restores geometry with recording suspended,
then restores `changes` and `saved`, so public transform plans and logical
groups have the same atomicity guarantee as drag gestures.

Test: `test_logical_group_failure_restores_generated_change_bookkeeping`.

### Preview, renderer-count, and replay edge cases are closed

The follow-up type-by-type QA closed cases that ordinary examples did not
exercise: Bezier PathPatch geometry, a 180-degree `xy`-anchored Rectangle,
nonuniform fixed-aspect Axes resize, masked PathCollection items, style arrays
longer than the rendered marker count, and NaN separators inside a
LineCollection. Preview and commit now share one constrained native plan where
needed, and serializer literals are exact rather than fixed-decimal.

This precision rule is deliberate. Quantizing finite values to 13 significant
digits looked stable on ordinary plots but produced roughly 90 px replay error
for a Line2D on `xlim=[1, 1+1e-12]`. Finite values therefore use Python's exact
round-trip `repr`; transaction undo restores the captured change-recording
state instead of relying on lossy source canonicalization. New generated blocks
also qualify non-finite values as `np.nan`/`np.inf` and import NumPy.

### Ambiguous transforms fail safely; offset collections follow the renderer

The legacy `apply_display_transform()` entrypoint accepts pure translation
matrices only. Scale/shear/rotation matrices must enter through a semantic
operation and its capability preflight. LineCollection and PolyCollection with
explicit `offsets`/`transOffset` now use the same path x offset item-count and
extent model as Matplotlib's renderer. Translation mutates and serializes the
offset controls while preserving base paths; ordinary collections continue to
edit their path vertices. This closes the former origin-centered selection box
without reducing the 483-object Fig2 editable inventory.

Finite offsets no longer advertise editing when there is no finite path to
paint. Conversely, an empty explicit offset array on LineCollection or
PolyCollection uses the renderer's zero-offset path semantics and edits the
base paths. A singular path/offset transform keeps finite selection and source
serialization available but denies translation and snapshots during capability
preflight, before any matrix inversion can fail.

### Legend creation and dependent commands have stable logical ownership

Current Axes Legend creation, frame style, and later Axes changes used to share
the Axes as a coarse change-dictionary owner. An Axes serialization could
therefore delete the Legend commands; save/load also parsed the same frame line
under a different key. Creation and dependent commands now retain a logical
Legend owner while addressing the Axes as their executable target, and the
loader normalizes the same representation after reopening.

Explicit proxy Legends no longer emit the self-dependent expression
`ax.legend(handles=ax.get_legend().legend_handles, ...)`. A registry freezes
complete single-glyph DrawingArea entries into self-contained Patch, Line2D,
PathCollection, or LineCollection specifications. Composite handlers such as
Errorbar and tuple entries are detected from all DrawingArea children and fail
capability preflight until a complete handler specification exists.
Frozen marker dimensions are normalized by `markerscale` before replay, and
semantic Axes handles are reused only when their complete handler-glyph
signature matches the current Legend; matching labels alone are insufficient.
The property panel consumes the same unscaled source-handle resolver, so layout
changes cannot double-apply `markerscale` or collapse semantic composites.
When reconstruction is not lossless, non-frame controls are disabled before
any object or widget state changes; an in-place replayable `frameon` remains
available for non-current Axes and Figure Legends.

## P1/P2 findings fixed after the independent audit

### P1: visible bounds and transformable geometry are now separate

The editor uses Illustrator-style visible/preview bounds for selection,
alignment, and handles. Adapters now also expose transformable geometry bounds
and fixed display-space appearance outsets. The split is essential: blindly
scaling a visible box would scale its stroke padding during preview even though
Matplotlib linewidth remains fixed at commit time.

- Patch bounds include visible stroke width.
- Line2D bounds include line stroke, marker path, and marker-edge stroke.
- PathCollection evaluates each marker path, size transform, offset, and edge
  stroke separately.
- LineCollection and PolyCollection broadcast linewidth/color per item before
  unioning envelopes; an invisible PolyCollection edge adds no padding.
- Patch and EditorGroup resize transform geometry to the requested visible box,
  then reapply each leaf's fixed appearance outsets. Numeric match-size uses the
  same contract.

Passing tests include the five original strict reproductions plus
`test_line_marker_selection_bounds_include_visible_edge_stroke`,
`test_poly_collection_invisible_edges_add_no_selection_padding`,
`test_thick_patch_resize_keeps_stroke_fixed_and_preview_matches_commit`,
`test_deferred_thick_patch_resize_preview_matches_committed_visible_bounds`,
and `test_match_width_uses_visible_bounds_without_scaling_strokes`.

### P1: EditorGroup has one change-record owner

Member adapters apply their native mutations with recording suspended. After
all members succeed, the outer EditorGroup serializes every leaf exactly once.
This removes duplicate computation, signals, and side effects while retaining
atomic rollback. Translate, resize, and mixed-stroke group resize are covered.

Tests: `test_editor_group_records_each_member_change_once` and
`test_editor_group_resize_reapplies_each_members_fixed_stroke_outset`.

### P2: unsupported semantic entrypoints gate before geometry arithmetic

Fallback `translate()` and the public display-transform entrypoint now consult
`OperationSupport` before touching empty controls. Both produce stable
`UnsupportedArtistError` reasons without geometry or tracker mutation.

Tests: `test_fallback_translate_rejects_with_adapter_contract_error` and
`test_fallback_display_transform_rejects_with_adapter_contract_error`.

No confirmed product defect remains hidden behind a strict xfail.

## Explicitly unsupported versus untested

Explicitly unsupported and tested:

- all operations on fallback Artist and ConnectionPatch;
- independent geometry transforms for Figure/SubFigure/Axes background patches,
  auto-positioned Axes and super titles, Axis offset text, active-layout bbox
  extras and labels, and invisible layout-only Legend geometry;
- geometry resize for Text and Annotation; appearance scaling for Annotation
  and unsupported Text variants;
- legend geometry resize; Legend layout reflow for expand/custom/detached trees;
- Line2D resize without affine preflight;
- Arc, Circle, CirclePolygon, FancyArrowPatch, FancyBboxPatch, RegularPolygon,
  and Wedge semantic resize;
- geometry resize for all three collection adapters, plus appearance variants
  that lack a lossless executor;
- native rotation for every type without a saveable angle property;
- rigid rotation for Axes, AxesImage, Legend, Annotation, ConnectionPatch,
  FancyArrowPatch/FancyBboxPatch, PathCollection, RegularPolygon, Wedge,
  owner-managed leaves, rendered-offset collections, and lossy transform/
  appearance variants of otherwise supported types;
- point editing, appearance scaling, and layout reflow wherever no executor
  exists.

Fixture/test limitations, not confirmed implementation defects:

- This is a headless adapter/renderer audit. It does not synthesize every Qt
  mouse-hit path or replace manual Fig2 testing.
- Native R and rigid Q have separate preview strategies. The audit covers the
  off-object handle, toolbar actions, Shift snapping, Escape cancellation, and
  mixed-type shared-pivot transactions, including freely draggable custom
  pivots and reset through the 3x3 reference locator.
- Arbitrary third-party Matplotlib subclasses are outside the built-in matrix;
  one temporary custom adapter is used only to inject an atomic group failure.
- Exotic custom transforms beyond data, axes, figure, display identity, mixed
  Annotation coordinates, and log non-affine data transforms were not
  exhaustively enumerated.
- Stroke envelopes are a common-case geometry approximation, not a rasterized
  paint envelope. Miter/cap joins, path effects, and clipping can extend or trim
  painted pixels beyond the current axial `linewidth / 2` padding; a 30 pt
  miter triangle raster probe missed as much as about 18.8 px. This is tracked
  as future paint-envelope policy, not claimed as exact coverage here.
- Generated blocks saved from this version replay NaN/Inf safely. A historical
  source block that already contains bare `nan`/`inf` can fail before runtime
  migration starts; the offline `pylustrator-source` doctor diagnoses and can
  atomically migrate that block without importing it. The formal Fig2 contains
  no such token.

## Reproduction commands

```bash
uv run pytest tests/test_artist_operation_contract_matrix.py -q -rs
QT_QPA_PLATFORM=offscreen uv run pytest tests -q
uv run ruff check .
```

The current full suite reports 1,187 passed and 178 skipped tests, with no strict
xfails. Within the dedicated matrix, supported-operation tests skip denied types
while rejection tests skip supported types; those skips are branch accounting,
not missing Artist coverage. Registry equality covers all 23 always-present
registrations and the exact `FillBetweenPolyCollection` registration when the
Matplotlib version exposes that concrete type.

## Real Fig2 audit appendix

A second audit on the long-term refactor worktree builds the disposable fork through
`figure_workflow/validation/fig2_pylustrator_ab/fig2_fork_common.py`. The formal
`editable/fig2.py` remained byte-identical before and after the audit at
SHA-256
`b0cd72abf3962cd6cd2354467ad57aa37ecc213332645d7cb56e6f4af598ad70`.

### Rigid-rotation milestone audit

The current read-only fork enumerates 483 selectable instances, 13 concrete
types, 12 adapters, and 19 semantic categories. It evaluates 4,830 operation
descriptors with zero capability-honesty failures. There are 227 Q-supported
instances across FillBetweenPolyCollection, Line2D, LineCollection, and
PathPatch. Four angles (`13`, `-37`, `90`, `360`) and center/external pivots
produce 1,816 plans: 1,702 accepted destinations and 114 typed destination
rejections, with no unexpected result and maximum accepted round-trip error
`4.55e-13 px`.

Line2D, LineCollection, and PathPatch pass real nonzero single-handle commit,
Undo/Redo, snapshot, generated replay, and mixed shared-pivot workflows. The
successful panel-B three-type transaction has zero commit/indicator/replay
error, at most `2.27e-13 px` Undo/Redo error, three generated commands, and one
undo item. A panel-H mixture containing a clip-limited FillBetween rejects
atomically. Both real FillBetween instances are Q-capable in principle but
their source/destinations are partially clipped: only 360-degree no-ops accept,
while every nonzero probe rejects without mutation.

All 115 Legend/layout/container-owned instances reject independently rotating
their leaves with zero mutation, record, or edit. Panel-D Legend drag and key
alignment remain within `2.27e-13 px`; its selection indicator error is zero.
The real property widget toggles `frameon` in about `0.124 s`, preserving
Legend identity, children, selection, Undo/Redo, and subsequent child dragging.
The reported freeze was not reproduced. The formal file hash above is unchanged.

The independent synthetic companion covers ten positive Artist variants at
four angles and two pivots (80 cases), including reflected similarity,
existing angles, negative sizes, NaN path breaks, mixed/nested groups,
failure injection, cancellation, and replay. Maximum geometry error is
`5.68e-14 px`, selection error is zero, and 360 degrees emits no mutation or
record.

The parent-project `fig2_artist_contract_audit.py` rotation section is now a
stale probe contract: it enumerates native `_rotatable_value` and then invokes
the direct-manipulation action, which correctly rejects visually incomplete
Annotation rotation. The Q audit above uses `OperationSupport(RIGID_ROTATE)`
and `native_rotation_handle_support()` directly; neither the legacy script nor
formal validation JSON was modified during this milestone.

The real figure contains 483 selectable and serializable instances, resolving
13 concrete Matplotlib types through 12 adapters:

| Concrete type | Instances | Adapter |
|---|---:|---|
| Annotation | 2 | AnnotationAdapter |
| Axes | 13 | AxesAdapter |
| AxesImage | 6 | AxesImageAdapter |
| FillBetweenPolyCollection | 2 | PolyCollectionAdapter |
| Legend | 8 | LegendAdapter |
| Line2D | 189 | Line2DAdapter |
| LineCollection | 7 | LineCollectionAdapter |
| PathCollection | 40 | PathCollectionAdapter |
| PathPatch | 35 | PathPatchAdapter |
| PolyCollection | 1 | PolyCollectionAdapter |
| Rectangle | 48 | RectangleAdapter |
| RegularPolygon | 10 | RegularPolygonAdapter |
| Text | 122 | TextAdapter |

EditorGroup, Ellipse, Arc, Circle, CirclePolygon, FancyArrowPatch,
ConnectionPatch, FancyBboxPatch, Wedge, Polygon, and the fallback Artist adapter
have no selectable instance in this specific figure. They remain covered by the
synthetic per-registration matrix.

The earlier native-operation audit found that all 483 Fig2 instances advertise
and pass finite selection bounds, translate,
snapshot construction, and nonempty serialization records. Of these, 102
advertise resize and 172 advertise rotation. The audit executes 6,279
instance-level checks: 4,347 `OperationSupport` checks plus 1,932 selection,
snapshot, serialization, and exact-reference checks.

Seventy representative workflow checks span all 19 semantic/ownership
categories in the figure:

- 19 exact-selection and translate/preview/Undo/Redo workflows;
- 5 advertised resize workflows;
- 8 advertised rotation workflows;
- 19 snapshot/restore round trips, including native rotation where available;
- 19 real ChangeTracker generated-command replays.

All geometry workflows pass. Maximum error is `3.41e-13 px` for translated
control points, `2.28e-13 px` for preview/final, resize, snapshot, and Undo/Redo,
and `0 degrees` for rotation. All 19 representative generated replays pass
within `0.1 px`, including ordinary and current-Legend Line2D, Rectangle, and
Text children.

### Resolved real Fig2 reference defect: non-current Axes Legend children

The first exhaustive reference check found 469 exact/evaluable references and
14 failures. Every failure belonged to a live, selectable child of a
non-current Axes Legend stored in `axes.artists`:

| Owner | Child type | Count | Children |
|---|---|---:|---|
| `panel_a.artists[0]` | Rectangle | 3 | Diffusion, Hallucination, Astrolabe |
| `panel_a.artists[0]` | Text | 4 | Diffusion, Hallucination, Astrolabe, Types |
| `panel_g.artists[0]` | Line2D | 3 | domain-wise, full-length, ipTM < 0.7 |
| `panel_g.artists[0]` | Text | 4 | domain-wise, full-length, ipTM < 0.7, method |

The null `legend_Line2D` reference in both historical probe artifacts is the
`domain-wise` handle owned by
`plt.figure(1).ax_dict["panel_g"].artists[0]`. The alignment pair's other
member is the valid current-Legend reference
`panel_h.get_legend().legend_handles[0]`. Geometry alignment and Undo still
pass; only the persistent locator is missing.

This was a product defect, not a probe limitation. `getReference()` could resolve
the non-current Legend itself as `axes.artists[0]`, but
`get_legend_reference()` searched only `figure.legends` and each Axes' current
`get_legend()`. It therefore falls through for the children: Line2D and
Rectangle raise an empty `ValueError` when they are not found in ordinary Axes
lists, while Text raises `TypeError: <class 'matplotlib.text.Text'> not found`.
All 14 instances advertise serialization; their 20 emitted change records also
contained unreplayable child targets.

The fix introduces one authoritative Legend inventory covering
`Figure.legends`, current `Axes.legend_`, and retained Legends in Figure/Axes
`artists`. Selection discovery, Legend child discovery, and persistent
reference resolution now consume that shared inventory. The post-fix audit
resolves all `483/483` instances to the exact live object, with zero
unreplayable instances and zero unreplayable change records.

`tests/test_artist_reference_contract.py` now passes for Line2D, Rectangle,
and Text children while also proving that the non-current Legend owner has an
exact evaluable reference. The real-case artifact retains the original
inventory categories and confirms that no reference failures remain.

The visible-bounds follow-up repeats the same 6,279 exhaustive checks and 70
representative workflows in
`artifacts/adapter_p1_visible_bounds_contract.json`: all 483/483 references
resolve, all operations/replays pass, and the formal Fig2 hash is unchanged.
At that visible-bounds milestone, the full headless suite reported 693 passed,
147 explicit capability-branch skips, no xfails, four pre-existing log-limit
warnings, and two expected masked-array conversion warnings. Ruff passes.

Do not reuse the parent-project legacy audit's rotation section until it is
migrated to the Q/native-handle split described above. The package-level
reproduction commands remain authoritative for this commit; the real Q probe
must continue to run against an in-memory Fig2 fork and a disposable output.
