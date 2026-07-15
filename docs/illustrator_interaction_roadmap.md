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

- Reference-point transform panel and arbitrary rotation handles.
- Key-object/artboard alignment and numeric distribute spacing.
- Generic smart guides for edges, centers, baselines, anchors, and equal gaps.
- Direct path/endpoint editing and inline text editing.
- Content-following cached drag previews and spatial hit/snap indexes.

## P2: workflow breadth

- Duplicate/copy/paste, Select Same, style copy, and complete z-order actions.
- Rulers, guides, grids, familiar zoom/pan shortcuts, and panel templates.
- Scientific roles and protection for panels, labels, legends, annotations, and
  data marks.

## Validation fixture

Fig2 fork validation remains the primary real-world fixture.  Formal
`editable/fig2.py` and publication outputs must not be modified by interaction
experiments.  Automated probes should report results separately for object and
direct selection modes and retain the existing drag, align, resize, rotation,
aspect, undo, replay, and performance checks.
