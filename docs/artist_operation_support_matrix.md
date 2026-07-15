# Artist operation support matrix

This document is an independent QA audit of the adapter contract on
`refactor/artist-adapter-architecture`, dated 2026-07-15.  It treats the
registry and `ArtistCapabilities`/`OperationSupport` as executable promises,
not merely implementation metadata.

The contract test is
`tests/test_artist_operation_contract_matrix.py`.  It enumerates every
built-in registry entry, constructs the concrete Artist, and tests the same
operation through display-space controls, snapshots, generated changes, and
rendered bounds.  Tests use Python 3.13.5, Matplotlib 3.10.8, NumPy 2.4.0, the
Agg renderer, a 0.25 display-pixel preview/commit tolerance, and strict xfails
for confirmed product defects.

## Notation and scope

Advertised operations are abbreviated as:

- `S`: select and expose finite selection bounds
- `M`: display-space translate
- `Z`: geometry resize
- `R`: native rotation
- `N`: snapshot/restore
- `C`: serialize to generated changes

Every row was also tested against all nine `TransformOperation` values.
Unsupported operations must return `OperationSupport.supported == False` with
a nonempty reason.  Direct translate, resize, rotate, and snapshot entrypoints
must reject without geometry or generated-change mutation.  `scale_appearance`,
`reflow_layout`, and `edit_points` are currently denied for every built-in
adapter.

`PASS*` means all advertised transforms pass, but a narrower selection,
snapshot, or transaction defect remains as a strict xfail.  `DENIED` means the
absence of edit support is deliberate and the denial contract passes.

## Per-type matrix

| Registered Artist | Adapter | Advertised | Actually tested | Result | Gap or explicit limitation |
|---|---|---:|---|---|---|
| `Artist` fallback | `ArtistAdapter` | none | Registry resolution; all operation descriptors; direct M/Z/R/N rejection and zero mutation | **FAIL** | Deliberately uneditable, but direct M reaches an empty-array NumPy broadcast error instead of the capability guard. |
| `EditorGroup` | `EditorGroupAdapter` | S M Z N C | Two-member bounds; M/Z preview and commit; parent Axes immobility; N undo/redo state; real generated-command replay; failing-member rollback | **PASS*** | Geometry and generated-change rollback are atomic. A successful group transform still applies every member's change records twice. R is explicitly denied because member positions would also need pivot rotation. |
| `Axes` | `AxesAdapter` | S M Z N C | Bounds; M/Z; parent Figure size; N; real replay; fixed-aspect capability and constraint-respecting resize | **PASS** | R is explicitly unsupported. Fixed-aspect Axes advertise the `fixed_aspect` constraint. |
| `Text` | `TextAdapter` | S M R N C | Axes/data/figure/display transforms; visible text bounds; M; native R; complete N; real replay; explicit Undo/Redo bookkeeping | **PASS** | Geometry resize is correctly denied in favor of future appearance scaling. |
| `Annotation` | `AnnotationAdapter` | S M R N C | Mixed data/axes endpoint coordinates; two control points; M; R; complete N; real replay | **PASS** | Geometry resize is explicitly denied. |
| `Legend` | `LegendAdapter` | S M N C | Visible handles/text/title union; `frameon=False/True`; M; identity preservation during live movement; N; real replay | **PASS** | Geometry resize and layout reflow are explicitly denied. Generated source replay may recreate an Axes legend, while live edits retain identity. |
| `Line2D` | `Line2DAdapter` | S M N C | Data and log transforms; M preview/commit; N; real `.set_data` replay | **PASS*** | Thick line stroke width is missing from selection bounds, although markers are included by Matplotlib. Geometry resize is explicitly denied pending affine preflight. |
| `AxesImage` | `AxesImageAdapter` | S M Z N C | M/Z preview/commit; extent replay; N; x/y viewport and Axes position invariance | **PASS** | R is explicitly unsupported. Moving or resizing the image does not autoscale the camera. |
| `Rectangle` | `RectangleAdapter` | S M Z R N C | M/Z/R; rotated M; rotated-resize denial; complete N; real replay; parent Axes invariance | **PASS*** | Generic patch bounds omit visible stroke width. Z is only advertised at angles equivalent to 0 degrees modulo 180. |
| `Ellipse` | `EllipseAdapter` | S M Z R N C | M/Z/R; rotated M; rotated-resize denial; complete N; real replay | **PASS*** | Generic patch bounds omit visible stroke width. Z is disabled for rotated ellipses. |
| `FancyArrowPatch` | `FancyArrowPatchAdapter` | S M N C | Endpoint M; rendered preview/commit; N; real `.set_positions` replay | **PASS*** | Generic patch bounds do not account for all visible stroke appearance. Z/R are explicitly denied. |
| `ConnectionPatch` | `ConnectionPatchAdapter` | none | Specific MRO resolution; all operation descriptors; direct M/Z/R/N rejection and zero mutation | **DENIED** | Deliberately blocked because its endpoints can occupy unrelated coordinate systems. |
| `FancyBboxPatch` | `FancyBboxPatchAdapter` | S M N C when affine | Affine M/N/replay; log/non-affine preflight denial and zero mutation | **PASS*** | Generic patch bounds omit stroke. Geometry resize is explicitly unsupported. A non-affine data transform disables the whole editable contract. |
| `RegularPolygon` | `RegularPolygonAdapter` | S M N C | Center M; preview/commit; N; `.xy` replay | **PASS*** | Generic patch bounds omit stroke. Z is explicitly denied until it changes semantic radius rather than stretching the center control. |
| `Wedge` | `WedgeAdapter` | S M N C | Center M; preview/commit; N; `.set_center` replay | **PASS*** | Generic patch bounds omit stroke. Z/R are explicitly unsupported. |
| `Polygon` | `PolygonAdapter` | S M Z N C | Vertex M/Z; preview/commit; N; `.set_xy` replay | **PASS*** | Generic patch bounds omit visible stroke width. R is explicitly unsupported. |
| `PathPatch` | `PathPatchAdapter` | S M Z N C | Path/codes M/Z; preview/commit; N; `Path` replay | **PASS*** | Generic patch bounds omit visible stroke width. R is explicitly unsupported. |
| `PathCollection` | `PathCollectionAdapter` | S M N C | Offset M in affine and log transforms; marker padding; N; `.set_offsets` replay | **PASS*** | Selection uses the largest marker/stroke padding around every extreme offset, rather than each item's actual visible envelope. Z and appearance scaling are explicitly denied. |
| `LineCollection` | `LineCollectionAdapter` | S M N C | Multi-segment M in affine and log transforms; group shape preservation; N; `.set_segments` replay | **PASS*** | Selection applies the largest linewidth to every segment rather than a per-segment visible envelope. Z and appearance scaling are denied. |
| `PolyCollection` | `PolyCollectionAdapter` | S M N C | Multi-path M in affine and log transforms; codes/group preservation; N; `.set_verts_and_codes` replay | **PASS*** | Selection applies the largest linewidth to every polygon rather than a per-item visible envelope. Z and appearance scaling are denied. |

The shared generic patch selection defect is proven with Rectangle as the
minimal reproduction.  Rectangle, Ellipse, FancyArrowPatch, FancyBboxPatch,
RegularPolygon, Wedge, Polygon, and PathPatch all inherit the same
`ArtistAdapter.selection_points()` window-extent behavior; the table therefore
marks the shared limitation for every affected type without duplicating eight
identical strict xfails.

## Cross-cutting contract results

The following behavior passes for every type that advertises it:

- Registry resolution selects the exact most-specific adapter, including
  `Annotation` before `Text` and `ConnectionPatch` before `FancyArrowPatch`.
- Display-space translation moves control points and final selection bounds by
  the preview delta within 0.25 px.  The selected object is the object mutated;
  parent Axes and unrelated figure-level sentinel text do not move.
- Geometry resize matches preview controls and final bounds within 0.25 px for
  EditorGroup, Axes, AxesImage, Rectangle, Ellipse, Polygon, and PathPatch.
- Native rotation renders and records correctly for Text, Annotation,
  Rectangle, and Ellipse. Explicit old/new-value Undo/Redo restores the angle
  and generated-change dictionary.
- Snapshots restore before/after geometry and native rotation. Failed
  multi-target transforms restore both artist state and generated-change
  bookkeeping atomically.
- Real `ChangeTracker` commands replay translated rendered bounds for all 18
  serializable registry types. Axis-label replay additionally covers both
  label position and `labelpad`.
- AxesImage operations preserve x/y limits. Logical-group member operations do
  not move their parent Axes.
- Legend bounds follow visible handles, text, and title; the invisible layout
  frame contributes no padding, while a visible frame is included.

Rotation has a `native_rotation` preview strategy, not a detached control-point
preview. The native render, selection overlay, snapshot, rollback, and explicit
Undo/Redo paths are consistent.

## P0 findings fixed after the independent audit

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

## Confirmed remaining defects retained as strict xfails

### P1: selection bounds mix geometric, visible, and conservative bounds

The documented adapter contract says selection bounds are visible bounds, but
the implementations are inconsistent:

- generic Patch bounds omit stroke width;
- Line2D includes markers but omits line stroke width;
- collections pad all items using global maxima, which can overestimate the
  opposite edge when sizes or linewidths differ.

This does not break translation preview/commit equality—the same inaccurate
bounds move consistently—but it can make the selection box disagree with the
actual artwork before and after the gesture. That is directly relevant to
alignment and hit-testing.

Tests:

- `test_rectangle_selection_bounds_include_visible_stroke_width`
- `test_line_selection_bounds_include_visible_stroke_width`
- `test_path_collection_selection_bounds_use_each_marker_size`
- `test_line_collection_selection_bounds_use_each_segment_linewidth`
- `test_poly_collection_selection_bounds_use_each_polygon_linewidth`

The product should first choose one explicit Illustrator-style policy:
geometric bounds, visible/preview bounds, or a user preference equivalent to
Illustrator's preview-bounds setting. The current code and roadmap claim
visible bounds, so the xfails enforce that stated policy.

### P1: EditorGroup records successful member mutations twice

`EditorGroupAdapter._apply_native_control_points()` delegates to each member's
recording adapter. The inherited outer `apply_native_control_points()` then
serializes every member again through the group adapter. A dictionary-backed
ChangeTracker hides duplicate final keys, but expensive or side-effectful
record generation still runs twice.

Test: `test_editor_group_records_each_member_change_once`.

### P2: fallback translate bypasses its capability guard

The fallback has no control points. `translate()` performs NumPy addition
before `apply_native_control_points()` checks `can_translate`, producing a
broadcasting `ValueError` rather than `UnsupportedArtistError`. Geometry and
bookkeeping remain unchanged, but the rejection is neither stable nor useful
to the UI.

Test: `test_fallback_translate_rejects_with_adapter_contract_error`.

## Explicitly unsupported versus untested

Explicitly unsupported and tested:

- all operations on fallback Artist and ConnectionPatch;
- text/annotation geometry resize and appearance scaling;
- legend geometry resize and layout reflow;
- Line2D resize without affine preflight;
- FancyArrowPatch, FancyBboxPatch, RegularPolygon, and Wedge resize;
- resize and appearance scaling for all three collection adapters;
- rotation for every type without a native, saveable angle property;
- point editing, appearance scaling, and layout reflow wherever no executor
  exists.

Fixture/test limitations, not confirmed implementation defects:

- This is a headless adapter/renderer audit. It does not synthesize every Qt
  mouse-hit path or replace manual Fig2 testing.
- Rotation uses the advertised native-preview strategy; there is no separate
  off-object rotation preview surface to compare.
- Arbitrary third-party Matplotlib subclasses are outside the built-in matrix;
  one temporary custom adapter is used only to inject an atomic group failure.
- Exotic custom transforms beyond data, axes, figure, display identity, mixed
  Annotation coordinates, and log non-affine data transforms were not
  exhaustively enumerated.

## Reproduction commands

```bash
uv run pytest tests/test_artist_operation_contract_matrix.py -q -rs
QT_QPA_PLATFORM=offscreen uv run pytest tests -q
uv run ruff check .
```

After the P0 fixes, the dedicated contract file reports 362 passed,
119 explicitly skipped branches, and 7 strict xfails. The skips are branch
accounting, not missing Artist types: supported-operation tests skip denied
types, while rejection tests skip supported types. Registry equality guarantees
that all 20 built-in registrations are present in the matrix.

## Real Fig2 audit appendix

A second audit at `447f76c` builds the disposable fork through
`figure_workflow/validation/fig2_pylustrator_ab/fig2_fork_common.py`. The formal
`editable/fig2.py` remained byte-identical before and after the audit at
SHA-256
`b0cd72abf3962cd6cd2354467ad57aa37ecc213332645d7cb56e6f4af598ad70`.

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

EditorGroup, Ellipse, FancyArrowPatch, ConnectionPatch, FancyBboxPatch, Wedge,
Polygon, and the fallback Artist adapter have no selectable instance in this
specific figure. They remain covered by the synthetic 20-type matrix.

All 483 Fig2 instances advertise and pass finite selection bounds, translate,
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

With the three reference xfails closed, the full headless suite reports 502
passed, 119 explicit capability-branch skips, 7 strict P1/P2 xfails, and the
same 4 pre-existing log-limit warnings. Ruff passes for the package, reference
tests, and real-case audit script.

Run the real audit from `Figures_source`:

```bash
QT_QPA_PLATFORM=offscreen \
PYTHONPATH=figure_workflow/vendor/pylustrator-artist-adapters \
uv run python \
  figure_workflow/validation/fig2_pylustrator_ab/fig2_artist_contract_audit.py \
  --commit 447f76c \
  --output \
  figure_workflow/validation/fig2_pylustrator_ab/artifacts/adapter_p0_real_contract.json
```
