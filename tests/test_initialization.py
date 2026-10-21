from __future__ import annotations

from io import BytesIO
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
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


def test_desktop_wrappers_leave_unmanaged_figures_unmodified() -> None:
    from pylustrator import QtGuiDrag

    original_show = plt.show
    original_no_save = QtGuiDrag.no_save_allowed
    try:
        QtGuiDrag.initialize(disable_save=True)
        figure = Figure(figsize=(4, 3), dpi=96)
        ax = figure.subplots()
        label = figure.text(.2, .8, "Bare Figure", color="red", fontsize=15)
        note = ax.text(.3, .6, "Bare Axes", fontdict={"weight": "bold"}, ha="right")
        assert label.get_color() == "red"
        assert label.get_fontsize() == 15
        assert note.get_weight() == "bold"
        assert note.get_ha() == "right"
        assert not hasattr(figure, "number")
        assert not hasattr(label, "_pylustrator_reference")
        assert not hasattr(note, "_pylustrator_old_values")
        output = BytesIO()
        figure.savefig(output, format="png")
        assert output.getvalue().startswith(b"\x89PNG")
        assert not hasattr(figure, "_last_saved_figure")

        managed = plt.figure()
        try:
            tracked = managed.text(.2, .8, "Desktop source remains tracked")
            assert tracked._pylustrator_reference["reference"].endswith(".texts[0]")
        finally:
            plt.close(managed)
    finally:
        plt.show = original_show
        QtGuiDrag.no_save_allowed = original_no_save


def test_headless_render_and_session_work_after_desktop_initialization() -> None:
    from pylustrator import QtGuiDrag
    from pylustrator.web.document import new_document
    from pylustrator.web.render import render_document
    from pylustrator.web.session import FigureSession

    original_show = plt.show
    original_no_save = QtGuiDrag.no_save_allowed
    try:
        QtGuiDrag.initialize(disable_save=True)
        document = new_document()
        document["elements"] = [{"id": "text", "type": "text", "x": 30, "y": 30,
                                 "width": 200, "height": 40, "text": "Headless text"}]
        output = render_document(document, lambda _: [])
        assert b"Headless text" in output
        figure = Figure(figsize=(4, 3), dpi=96)
        text = figure.text(.2, .8, "Native text")
        session = FigureSession(figure)
        try:
            session.command({"revision": 0, "operation": "translate",
                             "targets": [session.artist_id(text)], "dx": 10, "dy": 5})
            assert text.get_position() != (.2, .8)
            session.undo(1)
            np.testing.assert_allclose(text.get_position(), (.2, .8), atol=1e-14, rtol=0)
        finally:
            session.close()
    finally:
        plt.show = original_show
        QtGuiDrag.no_save_allowed = original_no_save
