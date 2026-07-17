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

Adapters that support saving should also set ``can_serialize=True`` and return
``ChangeRecord`` objects from ``serialize_changes``.  Resize capability should
only be enabled when the committed native state can exactly reproduce the
display-space preview.

.. autoclass:: pylustrator.ArtistCapabilities
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

.. autoclass:: pylustrator.AppearanceScalePlan
   :members:

.. autofunction:: pylustrator.migrate_generated_source

.. autofunction:: pylustrator.diagnose_generated_source

.. autoclass:: pylustrator.SourceDiagnostic
   :members:

.. autoclass:: pylustrator.SourceDoctorReport
   :members:
