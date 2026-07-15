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
