from __future__ import annotations

from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colors import Colormap
from matplotlib.figure import Figure


def test_repeated_initialize_does_not_stack_matplotlib_wrappers() -> None:
    from pylustrator import QtGuiDrag

    original_show = plt.show
    original_no_save = QtGuiDrag.no_save_allowed
    try:
        QtGuiDrag.initialize(disable_save=True)
        first = (Axes.text, Figure.text, Figure.savefig, Colormap.__call__)
        QtGuiDrag.initialize(disable_save=True)
        QtGuiDrag.initialize(disable_save=True)
        second = (Axes.text, Figure.text, Figure.savefig, Colormap.__call__)

        assert all(before is after for before, after in zip(first, second))
        fig, ax = plt.subplots()
        assert ax.text(0.5, 0.5, "one wrapper").get_text() == "one wrapper"
        plt.close(fig)
    finally:
        plt.show = original_show
        QtGuiDrag.no_save_allowed = original_no_save


def test_action_save_writes_source_without_reexporting_images() -> None:
    from pylustrator.QtGuiDrag import PlotWindow

    fig = plt.figure()
    calls = {"source": 0, "exports": []}
    fig.change_tracker = SimpleNamespace(
        save=lambda: calls.__setitem__("source", calls["source"] + 1)
    )
    request = ("existing-export.png", (), {"dpi": 300})
    fig._last_saved_figure = [request]
    fig.savefig = lambda *args, **kwargs: calls["exports"].append((args, kwargs))

    try:
        PlotWindow.actionSave(SimpleNamespace(fig=fig))

        assert calls["source"] == 1
        assert calls["exports"] == []
        assert fig._last_saved_figure == [request]
    finally:
        plt.close(fig)
