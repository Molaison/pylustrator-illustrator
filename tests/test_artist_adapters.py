from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.artist import Artist
from matplotlib.patches import (
    ConnectionPatch,
    FancyArrowPatch,
    FancyBboxPatch,
    Rectangle,
)
from matplotlib.text import Annotation, Text
from matplotlib.transforms import Bbox, IdentityTransform

from pylustrator.artist_adapters import (
    AnnotationAdapter,
    ArtistAdapter,
    ArtistAdapterRegistry,
    ArtistCapabilities,
    ConnectionPatchAdapter,
    FancyArrowPatchAdapter,
    PathCollectionAdapter,
    RectangleAdapter,
    TextAdapter,
    UnsupportedArtistError,
    artist_adapter_registry,
    get_artist_adapter,
    register_artist_adapter,
)
from pylustrator.snap import TargetWrapper


class RecordingChangeTracker:
    def __init__(self):
        self.records = []

    def addChange(self, target, command):
        self.records.append(("command", target, command))

    def addNewTextChange(self, target):
        self.records.append(("text", target, None))

    def addNewLegendChange(self, target):
        self.records.append(("legend", target, None))

    def addNewAxesChange(self, target):
        self.records.append(("axes", target, None))


def test_extension_contract_is_available_from_the_public_package() -> None:
    import pylustrator

    assert pylustrator.ArtistAdapter is ArtistAdapter
    assert pylustrator.ArtistAdapterRegistry is ArtistAdapterRegistry
    assert pylustrator.ArtistCapabilities is ArtistCapabilities
    assert pylustrator.register_artist_adapter is register_artist_adapter
    assert pylustrator.get_artist_adapter is get_artist_adapter


def test_registry_resolves_matplotlib_subclasses_by_mro_specificity() -> None:
    assert artist_adapter_registry.resolve_type(Text) is TextAdapter
    assert artist_adapter_registry.resolve_type(Annotation) is AnnotationAdapter
    assert (
        artist_adapter_registry.resolve_type(FancyArrowPatch)
        is FancyArrowPatchAdapter
    )
    assert (
        artist_adapter_registry.resolve_type(ConnectionPatch)
        is ConnectionPatchAdapter
    )


def test_registry_cache_is_invalidated_when_a_more_specific_adapter_is_added() -> None:
    class CustomArtist(Artist):
        pass

    class CustomChild(CustomArtist):
        pass

    class CustomAdapter(ArtistAdapter):
        pass

    class ChildAdapter(ArtistAdapter):
        pass

    registry = ArtistAdapterRegistry()
    registry.register(Artist, ArtistAdapter)
    registry.register(CustomArtist, CustomAdapter)

    assert registry.resolve_type(CustomChild) is CustomAdapter
    registry.register(CustomChild, ChildAdapter)
    assert registry.resolve_type(CustomChild) is ChildAdapter


def test_third_party_adapter_registration_extends_target_wrapper() -> None:
    class OffsetArtist(Artist):
        def __init__(self, position):
            super().__init__()
            self.position = np.asarray(position, dtype=float)

        def get_window_extent(self, renderer=None):
            return Bbox.from_bounds(*self.position, 1.0, 1.0)

    @register_artist_adapter(OffsetArtist)
    class OffsetAdapter(ArtistAdapter):
        default_capabilities = ArtistCapabilities(
            can_select=True,
            can_translate=True,
            can_snapshot=True,
        )

        def get_transform(self):
            return IdentityTransform()

        def native_control_points(self):
            return [self.target.position.copy()]

        def _apply_native_control_points(self, points) -> None:
            self.target.position = np.asarray(points[0], dtype=float)

    fig = plt.figure(figsize=(2, 2), dpi=100)
    artist = OffsetArtist((20, 30))
    fig.add_artist(artist)
    try:
        wrapper = TargetWrapper(artist)
        assert isinstance(wrapper.adapter, OffsetAdapter)
        assert wrapper.supported
        wrapper.translate((7, -4))
        assert np.allclose(artist.position, (27, 26))
    finally:
        artist_adapter_registry.unregister(OffsetArtist, OffsetAdapter)
        plt.close(fig)


def test_capabilities_describe_lossless_operations_instead_of_artist_labels() -> None:
    fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
    rectangle = ax.add_patch(Rectangle((0.1, 0.2), 0.3, 0.4))
    rotated = ax.add_patch(Rectangle((0.5, 0.2), 0.2, 0.3, angle=25))
    text = ax.text(0.2, 0.8, "text")
    annotation = ax.annotate("note", (0.4, 0.6), xytext=(0.7, 0.9))
    scatter = ax.scatter([0.2, 0.8], [0.3, 0.7])
    image = ax.imshow(np.arange(4).reshape(2, 2), extent=(0, 1, 0, 1))
    fig.canvas.draw()

    assert get_artist_adapter(rectangle).capabilities.can_resize
    assert get_artist_adapter(rectangle).capabilities.can_rotate
    assert not get_artist_adapter(rotated).capabilities.can_resize
    assert not get_artist_adapter(text).capabilities.can_resize
    assert get_artist_adapter(text).capabilities.can_rotate
    assert not get_artist_adapter(annotation).capabilities.can_resize
    assert not get_artist_adapter(scatter).capabilities.can_resize
    assert not get_artist_adapter(scatter).capabilities.can_rotate
    assert get_artist_adapter(image).capabilities.can_resize
    assert isinstance(get_artist_adapter(scatter), PathCollectionAdapter)

    connection = ConnectionPatch((0, 0), (1, 1), "data", "data", axesA=ax)
    assert not get_artist_adapter(connection).capabilities.editable
    assert not TargetWrapper.supports_target(connection)
    plt.close(fig)


def test_non_affine_box_is_a_blocker_but_uses_the_specific_adapter() -> None:
    fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
    ax.set_xscale("log")
    box = FancyBboxPatch((0.2, 0.3), 0.4, 0.2, boxstyle="round,pad=0.1")
    ax.add_patch(box)

    adapter = get_artist_adapter(box)

    assert not adapter.capabilities.editable
    assert not TargetWrapper(box).supported
    plt.close(fig)


def test_facade_delegates_geometry_snapshot_and_change_serialization() -> None:
    fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
    fig.change_tracker = RecordingChangeTracker()
    rectangle = ax.add_patch(Rectangle((0.2, 0.3), 0.25, 0.2))
    fig.canvas.draw()
    wrapper = TargetWrapper(rectangle)
    before = np.asarray(wrapper.get_positions(), dtype=float)
    snapshot = wrapper.get_restore_state()

    wrapper.translate((13, -7))

    assert isinstance(wrapper.adapter, RectangleAdapter)
    assert np.allclose(np.asarray(wrapper.get_positions()) - before, (13, -7))
    assert [kind for kind, _target, _command in fig.change_tracker.records] == [
        "command",
        "command",
        "command",
    ]

    wrapper.restore_state(snapshot)
    assert np.allclose(wrapper.get_positions(), before)
    assert len(fig.change_tracker.records) == 6
    plt.close(fig)


def test_annotation_and_axis_label_use_distinct_snapshot_protocols() -> None:
    fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
    annotation = ax.annotate(
        "mixed",
        xy=(0.2, 0.3),
        xycoords="data",
        xytext=(0.7, 0.8),
        textcoords="axes fraction",
    )
    label = ax.set_xlabel("x label")
    fig.canvas.draw()

    assert TargetWrapper(annotation).get_restore_state()["type"] == "positions"
    assert TargetWrapper(label).get_restore_state()["type"] == "axis_label"
    plt.close(fig)


def test_resize_api_rejects_lossy_artist_types() -> None:
    fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
    text = ax.text(0.2, 0.3, "cannot resize")
    fig.canvas.draw()
    matrix = np.eye(3)

    with pytest.raises(UnsupportedArtistError, match="cannot be resized losslessly"):
        TargetWrapper(text).resize(matrix)
    plt.close(fig)


def test_rotation_uses_adapter_capabilities_and_serialization() -> None:
    fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
    fig.change_tracker = RecordingChangeTracker()
    rectangle = ax.add_patch(Rectangle((0.2, 0.3), 0.25, 0.2))
    text = ax.text(0.5, 0.6, "rotate")
    scatter = ax.scatter([0.2, 0.8], [0.3, 0.7])

    TargetWrapper(rectangle).set_rotation(27)
    TargetWrapper(text).set_rotation(13)

    assert rectangle.get_angle() == 27
    assert text.get_rotation() == 13
    assert [record[0] for record in fig.change_tracker.records] == [
        "command",
        "text",
    ]
    with pytest.raises(UnsupportedArtistError, match="cannot be rotated losslessly"):
        TargetWrapper(scatter).set_rotation(5)
    plt.close(fig)
