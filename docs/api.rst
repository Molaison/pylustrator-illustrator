API
===

The API of pylustrator is kept quite simple. Most interaction with the pylustrator package is by using the interactive
interface.

.. autofunction:: pylustrator.start

.. autofunction:: pylustrator.load

Artist adapters
---------------

Interactive geometry is expressed in display pixels and delegated to the most
specific registered artist adapter.  An adapter declares which operations are
lossless, converts between display and native coordinates, applies mutations,
captures undo snapshots, and records reproducible changes.  This keeps a new
artist type from silently inheriting the behaviour of an incompatible parent.

Third-party Matplotlib artists can opt in without modifying Pylustrator:

.. code-block:: python

    from pylustrator import (
        ArtistAdapter,
        ArtistCapabilities,
        ChangeRecord,
        register_artist_adapter,
    )

    @register_artist_adapter(MyArtist)
    class MyArtistAdapter(ArtistAdapter):
        default_capabilities = ArtistCapabilities(
            can_select=True,
            can_translate=True,
            can_snapshot=True,
        )

        def native_control_points(self):
            return [self.target.position]

        def _apply_native_control_points(self, points):
            self.target.position = points[0]

Registrations match only the exact Artist type by default.  This prevents a
semantic subclass (for example a 3D Artist inheriting a 2D Matplotlib class)
from reusing incompatible mutation code.  When an extension owns an entire
class hierarchy and has validated that every descendant preserves the same
geometry, snapshot, and replay contracts, it can explicitly opt in:

.. code-block:: python

    from pylustrator import AdapterInheritancePolicy

    @register_artist_adapter(
        MyArtist,
        inheritance_policy=AdapterInheritancePolicy.VALIDATED,
    )
    class MyArtistHierarchyAdapter(ArtistAdapter):
        ...

Known Matplotlib semantic subclasses are registered individually after their
contracts are verified.  For example, Arc, Circle, and CirclePolygon expose
translation-only adapters that preserve their center/radius semantics instead
of inheriting every Ellipse or RegularPolygon mutation.

Adapters that support saving should also set ``can_serialize=True`` and return
``ChangeRecord`` objects from ``serialize_changes``.  Resize capability should
only be enabled when the committed native state can exactly reproduce the
display-space preview.

.. autoclass:: pylustrator.ArtistCapabilities
   :members:

.. autoclass:: pylustrator.AdapterInheritancePolicy
   :members:

.. autoclass:: pylustrator.ArtistAdapter
   :members:

.. autoclass:: pylustrator.ChangeRecord
   :members:

.. autofunction:: pylustrator.register_artist_adapter

Selection and editor groups
---------------------------

Canvas interaction first builds a front-to-back ``HitStack`` and then resolves
it through ``SelectionKernel``.  Object Selection groups semantic composites;
Direct Selection keeps the exact leaf Artist.  Isolation scopes, hover
preselection, Alt-click cycling, and the candidate menu all use this same
resolver.

Logical ``EditorGroup`` objects are independent of Matplotlib ownership.  Their
membership, names, locks, visibility, and stable identifiers are stored in the
versioned figure editor state and restored before interaction starts.

.. autoclass:: pylustrator.SelectionMode
   :members:

.. autoclass:: pylustrator.HitStack
   :members:

.. autoclass:: pylustrator.SelectionKernel
   :members:

.. autoclass:: pylustrator.EditorGroup
   :members:

Semantic transforms and replay
------------------------------

``OperationSupport`` distinguishes geometry resize from appearance scaling,
layout reflow, rotation, and point editing.  ``TransformPlan`` preflights every
target before mutation and rolls back earlier targets if an adapter fails.
Appearance scaling uses a frozen ``AppearanceScalePlan`` and an independent
appearance state/restore path; it does not enlarge geometry snapshots or emit
geometry serialization records.
Legend reflow likewise uses a frozen ``LegendLayoutPlan`` and layout-only
state.  Its v1 ``LegendLayoutSpec`` changes columns and spacing while retaining
every persistent Legend leaf; unsupported/custom packer trees reject before
mutation.
The shared rigid-rotation pivot is selection-level editor state, stored in
physical Figure coordinates rather than in an Artist snapshot or generated
source. Moving that overlay therefore never dirties the document; handle and
toolbar rotation still consume the same absolute transform plan.
``RigidRotationPlan`` stores large geometry destinations as compact immutable
arrays backed by read-only bytes. Line2D uses vectorized coordinate conversion
and a marker-aware destination envelope; a visible marker is accepted only
when its painted path is rotationally symmetric and the source/destination
``markevery`` centers preserve the requested rigid transform.

Generated blocks carry a schema version.  Legacy proxy-legend references are
accepted during replay and rewritten through the public migration helpers.

.. autoclass:: pylustrator.TransformOperation
   :members:

.. autoclass:: pylustrator.OperationSupport
   :members:

.. autoclass:: pylustrator.TransformIntent
   :members:

.. autoclass:: pylustrator.TransformPlan
   :members:

.. autoclass:: pylustrator.RigidRotationPlan
   :members:

.. autoclass:: pylustrator.AppearanceScalePlan
   :members:

.. autoclass:: pylustrator.LegendLayoutSpec
   :members:

.. autoclass:: pylustrator.LegendLayoutPlan
   :members:

.. autofunction:: pylustrator.reflow_legend_layout

.. autofunction:: pylustrator.migrate_generated_source

.. autofunction:: pylustrator.diagnose_generated_source

.. autoclass:: pylustrator.SourceDiagnostic
   :members:

.. autoclass:: pylustrator.SourceDoctorReport
   :members:
