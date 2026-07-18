from __future__ import annotations

from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

import pylustrator.drag_helper as drag_helper
from pylustrator.drag_helper import DragManager


def _visible_tick_labels(axes):
    labels = []
    for axis in (axes.xaxis, axes.yaxis):
        for tick in (*axis.majorTicks, *axis.minorTicks):
            labels.extend(
                label
                for label in (tick.label1, tick.label2)
                if label.get_visible() and label.get_text()
            )
    return labels


def test_unchanged_tick_inventory_does_not_repeat_artist_registration() -> None:
    fig, axes = plt.subplots()
    fig.canvas.draw()
    labels = _visible_tick_labels(axes)
    assert labels
    manager = DragManager.__new__(DragManager)
    manager._interaction_artist_ids = {id(label) for label in labels}
    calls = []
    manager.make_draggable = lambda label, parent: calls.append((label, parent))

    manager.register_axis_tick_labels(axes)

    assert calls == []
    assert all(
        getattr(label, "_pylustrator_formatter_owned_tick_label", False)
        for label in labels
    )
    plt.close(fig)


def test_marquee_uses_formatter_tag_before_compatibility_scan(monkeypatch) -> None:
    fig, axes = plt.subplots()
    fig.canvas.draw()
    label = _visible_tick_labels(axes)[0]
    label._pylustrator_formatter_owned_tick_label = True
    manager = DragManager.__new__(DragManager)
    manager.marquee_select_containers_only = False
    manager._uneditable_artists = []
    manager.iter_selectable_artists = lambda: iter((label,))
    manager._artist_intersects_bbox = lambda *_args: True
    manager._ensure_selection_kernel = lambda: SimpleNamespace(
        map_artists=lambda artists: list(artists)
    )
    manager.select_elements = lambda artists, **_kwargs: list(artists)
    monkeypatch.setattr(
        drag_helper,
        "axis_tick_label_reference",
        lambda _artist: (_ for _ in ()).throw(AssertionError("slow scan used")),
    )

    selected = manager._select_elements_in_bbox(0, 0, 100, 100)

    assert selected == []
    plt.close(fig)
