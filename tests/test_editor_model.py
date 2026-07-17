from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from pylustrator.commands import ObjectLocator
from pylustrator.editor_model import EDITOR_STATE_VERSION, EditorScene


def ownership_parent(figure):
    def parent(artist):
        axes = getattr(artist, "axes", None)
        return axes if axes is not None else figure

    return parent


def test_versioned_editor_state_round_trips_group_lock_and_visibility() -> None:
    fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
    first = ax.add_patch(Rectangle((0.1, 0.2), 0.3, 0.4, label="first"))
    second = ax.add_patch(Rectangle((0.6, 0.2), 0.3, 0.4, label="second"))
    scene = EditorScene(fig, ownership_parent=ownership_parent(fig))
    for artist in (ax, first, second):
        scene.register_artist(artist)
    group = scene.create_group([first, second], name="Pair")
    scene.set_locked([group], True)
    scene.set_visible([group], False)

    state = scene.export_state()
    scene.apply_state(state)

    restored = scene.groups[group.group_id]
    assert state["version"] == EDITOR_STATE_VERSION
    assert isinstance(state["groups"][0]["members"][0], dict)
    assert restored.name == "Pair"
    assert restored.members == [first, second]
    assert scene.is_locked(restored)
    assert scene.is_explicitly_hidden(restored)
    assert not first.get_visible()
    assert not second.get_visible()
    plt.close(fig)


def test_object_locator_falls_back_to_unique_semantic_identity() -> None:
    fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
    text = ax.text(0.5, 0.5, "unique annotation")
    scene = EditorScene(fig, ownership_parent=ownership_parent(fig))
    for artist in (ax, text):
        scene.register_artist(artist)
    locator = ObjectLocator(
        "plt.figure(999).axes[99].texts[99]",
        "Text",
        semantic_name="unique annotation",
    )

    assert locator.resolve(scene) is text
    plt.close(fig)
