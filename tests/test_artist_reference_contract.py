from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.text import Text

from pylustrator.change_tracker import getReference


def _two_legend_figure():
    fig, ax = plt.subplots(figsize=(3, 2), dpi=100)
    non_current = ax.legend(
        handles=[
            Line2D([], [], color="black", label="line proxy"),
            Rectangle((0, 0), 1, 1, label="patch proxy"),
        ],
        labels=["line proxy", "patch proxy"],
    )
    ax.add_artist(non_current)
    ax.legend(
        handles=[Line2D([], [], color="black", label="current")],
        labels=["current"],
    )
    fig.canvas.draw()
    return fig, ax, non_current


def test_non_current_axes_legend_itself_has_an_exact_reference() -> None:
    fig, _ax, legend = _two_legend_figure()

    try:
        reference = getReference(legend)
        assert eval(reference, {"plt": plt}) is legend
    finally:
        plt.close(fig)


@pytest.mark.parametrize("child_type", [Line2D, Rectangle, Text])
def test_non_current_axes_legend_children_have_exact_replay_references(
    child_type,
) -> None:
    fig, _ax, legend = _two_legend_figure()
    children = [*legend.legend_handles, *legend.get_texts()]
    child = next(item for item in children if isinstance(item, child_type))

    try:
        reference = getReference(child)
        assert reference
        assert eval(reference, {"plt": plt}) is child
    finally:
        plt.close(fig)
