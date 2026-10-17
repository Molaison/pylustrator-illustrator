"""Generate the focused Pylustrator Forge launch-demo suite.

Every product state is captured from the real PlotWindow.  PIL only places
those captures inside a fixed, non-overlapping presentation frame suitable
for GitHub and X.
"""

from __future__ import annotations

import argparse
import io
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backend_bases import MouseEvent
from matplotlib.backends.qt_compat import QtCore, QtGui, QtWidgets
from PIL import Image, ImageDraw, ImageFont

from pylustrator import QtGuiDrag
from pylustrator.change_tracker import getReference
from pylustrator.snap import TargetWrapper
from pylustrator.smart_guides import GuideKind


FRAME_SIZE = (1200, 675)
NAVY = "#09111F"
NAVY_2 = "#101C31"
CYAN = "#20C5D8"
CYAN_SOFT = "#DDF8FB"
CORAL = "#FF6B5E"
CORAL_SOFT = "#FFE8E5"
WHITE = "#FFFFFF"
PAPER = "#F5F7FA"
INK = "#172033"
MUTED = "#66748A"
BORDER = "#D9E1EB"
MAGENTA = "#D414D4"

DEMO_FILES = (
    "pylustrator-forge-01-pain-to-solution.gif",
    "pylustrator-forge-02-visible-bounds.gif",
    "pylustrator-forge-03-smart-guides.gif",
    "pylustrator-forge-04-reproducible-source.gif",
)


def _font(size: int, *, bold: bool = False, mono: bool = False):
    if mono:
        candidates = (
            "/System/Library/Fonts/Supplemental/Courier New Bold.ttf"
            if bold
            else "/System/Library/Fonts/Supplemental/Courier New.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"
            if bold
            else "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        )
    else:
        candidates = (
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
            if bold
            else "/System/Library/Fonts/Supplemental/Arial.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
            if bold
            else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        )
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def _scientific_figure() -> tuple[plt.Figure, list[plt.Axes]]:
    """Build one deliberately uneven multi-panel scientific figure."""

    rng = np.random.default_rng(7)
    fig = plt.figure(figsize=(10.4, 4.9), dpi=100, facecolor="#F7F9FC")
    positions = (
        (0.065, 0.19, 0.255, 0.58),
        (0.385, 0.10, 0.245, 0.72),
        (0.715, 0.245, 0.215, 0.48),
    )
    axes = [fig.add_axes(position) for position in positions]

    time = np.linspace(0, 8, 90)
    response = 1 - np.exp(-time / 2.2)
    axes[0].plot(time, response, color="#1D7EA8", linewidth=2.4)
    axes[0].fill_between(time, response, alpha=0.16, color=CYAN)
    axes[0].set(
        title="A · Kinetics",
        xlabel="Time (min)",
        ylabel="Normalized response",
        ylim=(0, 1.08),
    )

    x = rng.normal(size=55)
    y = 0.82 * x + rng.normal(scale=0.45, size=x.size)
    axes[1].scatter(x, y, s=24, color=CORAL, alpha=0.78, edgecolor="none")
    fit = np.polyfit(x, y, 1)
    grid = np.linspace(x.min(), x.max(), 80)
    axes[1].plot(grid, np.polyval(fit, grid), color="#B33B57", linewidth=2)
    axes[1].set(
        title="B · Correlation",
        xlabel="Predicted score",
        ylabel="Measured score",
    )

    labels = ["Control", "Design 1", "Design 2"]
    values = [0.31, 0.74, 0.92]
    bars = axes[2].bar(
        labels,
        values,
        color=["#9AA7B8", CYAN, CORAL],
        width=0.68,
    )
    axes[2].bar_label(bars, fmt="%.2f", padding=3, fontsize=8)
    axes[2].set(title="C · Activity", ylabel="Relative activity", ylim=(0, 1.08))
    axes[2].tick_params(axis="x", rotation=18)

    for index, ax in enumerate(axes):
        ax.grid(axis="y", color="#D7DEE8", linewidth=0.7, alpha=0.8)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.title.set_fontweight("bold")
        ax.text(
            0.98,
            0.03,
            f"panel {index + 1}",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            color="#748094",
            fontsize=8,
        )
    return fig, axes


def _pixmap_to_image(pixmap: QtGui.QPixmap) -> Image.Image:
    buffer = QtCore.QBuffer()
    buffer.open(QtCore.QIODevice.WriteOnly)
    pixmap.save(buffer, "PNG")
    return Image.open(io.BytesIO(bytes(buffer.data()))).convert("RGB")


def _grab_figure_view(window) -> Image.Image:
    """Capture the figure plus selection/guide overlay, without editor rulers."""

    plot_canvas = window.plot_layout.canvas_canvas
    viewport = plot_canvas.canvas_canvas
    canvas = plot_canvas.canvas
    image = _pixmap_to_image(viewport.grab())
    origin = canvas.mapTo(viewport, QtCore.QPoint(0, 0))
    scale_x = image.width / max(viewport.width(), 1)
    scale_y = image.height / max(viewport.height(), 1)
    box = (
        round(origin.x() * scale_x),
        round(origin.y() * scale_y),
        round((origin.x() + canvas.width()) * scale_x),
        round((origin.y() + canvas.height()) * scale_y),
    )
    return image.crop(box)


def _fit_inside(
    image: Image.Image,
    size: tuple[int, int],
    *,
    background: str = WHITE,
) -> Image.Image:
    fitted = image.copy()
    fitted.thumbnail(size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, background)
    offset = ((size[0] - fitted.width) // 2, (size[1] - fitted.height) // 2)
    canvas.paste(fitted, offset)
    return canvas


def _rounded_card(
    frame: Image.Image,
    box: tuple[int, int, int, int],
    *,
    fill: str = WHITE,
    outline: str = BORDER,
    radius: int = 18,
) -> None:
    ImageDraw.Draw(frame).rounded_rectangle(
        box,
        radius=radius,
        fill=fill,
        outline=outline,
        width=2,
    )


def _phase_pill(draw: ImageDraw.ImageDraw, phase: str) -> None:
    font = _font(17, bold=True)
    text_box = draw.textbbox((0, 0), phase, font=font)
    width = text_box[2] - text_box[0] + 36
    x0 = FRAME_SIZE[0] - 38 - width
    draw.rounded_rectangle(
        (x0, 28, FRAME_SIZE[0] - 38, 66),
        radius=19,
        fill=CYAN_SOFT,
    )
    draw.text((x0 + 18, 37), phase, font=font, fill="#087C8B")


def _header(
    frame: Image.Image,
    *,
    demo_index: int,
    title: str,
    phase: str,
) -> None:
    draw = ImageDraw.Draw(frame)
    draw.rounded_rectangle((38, 26, 60, 48), radius=6, fill=CYAN)
    draw.rounded_rectangle((52, 40, 74, 62), radius=6, fill=CORAL)
    draw.text(
        (90, 24),
        f"PYLUSTRATOR FORGE  ·  DEMO {demo_index}/4",
        font=_font(15, bold=True),
        fill="#8FA3BE",
    )
    draw.text((90, 47), title, font=_font(29, bold=True), fill=WHITE)
    _phase_pill(draw, phase)


def _footer(
    frame: Image.Image,
    *,
    detail: str,
    active: int,
    total: int,
) -> None:
    draw = ImageDraw.Draw(frame)
    draw.text((42, 638), detail, font=_font(18), fill="#B7C4D6")
    start_x = 1078
    for index in range(total):
        color = CORAL if index == active else "#42526A"
        draw.rounded_rectangle(
            (start_x + index * 24, 643, start_x + index * 24 + 14, 657),
            radius=7,
            fill=color,
        )


def _plot_card(
    frame: Image.Image,
    image: Image.Image,
    box: tuple[int, int, int, int],
) -> None:
    _rounded_card(frame, box)
    x0, y0, x1, y1 = box
    fitted = _fit_inside(image, (x1 - x0 - 20, y1 - y0 - 20))
    frame.paste(fitted, (x0 + 10, y0 + 10))


def _steps_card(
    frame: Image.Image,
    box: tuple[int, int, int, int],
    *,
    eyebrow: str,
    steps: Sequence[str],
    active: int,
) -> None:
    _rounded_card(frame, box, fill="#F8FAFD")
    draw = ImageDraw.Draw(frame)
    x0, y0, x1, _y1 = box
    draw.text(
        (x0 + 24, y0 + 24),
        eyebrow,
        font=_font(15, bold=True),
        fill=MUTED,
    )
    y = y0 + 68
    for index, step in enumerate(steps):
        is_active = index == active
        fill = CORAL_SOFT if is_active else WHITE
        outline = CORAL if is_active else BORDER
        draw.rounded_rectangle(
            (x0 + 18, y, x1 - 18, y + 73),
            radius=14,
            fill=fill,
            outline=outline,
            width=2,
        )
        number_fill = CORAL if is_active else "#CBD5E1"
        draw.ellipse((x0 + 34, y + 21, x0 + 64, y + 51), fill=number_fill)
        draw.text(
            (x0 + 44, y + 26),
            str(index + 1),
            anchor="mm",
            font=_font(15, bold=True),
            fill=WHITE,
        )
        draw.text(
            (x0 + 78, y + 25),
            step,
            font=_font(18, bold=is_active),
            fill=INK if is_active else MUTED,
        )
        y += 87


def _code_card(
    frame: Image.Image,
    box: tuple[int, int, int, int],
    *,
    steps: Sequence[str],
    active: int,
    code_lines: Sequence[str],
) -> None:
    _rounded_card(frame, box, fill=NAVY_2, outline="#2E405B")
    draw = ImageDraw.Draw(frame)
    x0, y0, x1, _y1 = box
    draw.text(
        (x0 + 24, y0 + 24),
        "HISTORY  →  PYTHON",
        font=_font(15, bold=True),
        fill="#8FA3BE",
    )
    y = y0 + 67
    for index, step in enumerate(steps):
        is_active = index == active
        fill = CORAL if is_active else "#26364E"
        draw.rounded_rectangle(
            (x0 + 24, y, x0 + 150, y + 36),
            radius=18,
            fill=fill,
        )
        draw.text(
            (x0 + 87, y + 18),
            step,
            anchor="mm",
            font=_font(14, bold=True),
            fill=WHITE,
        )
        y += 47
    divider_y = y0 + 276
    draw.line((x0 + 24, divider_y, x1 - 24, divider_y), fill="#344761", width=2)
    draw.text(
        (x0 + 24, divider_y + 20),
        "GENERATED EDITS",
        font=_font(14, bold=True),
        fill=CORAL,
    )
    code_y = divider_y + 52
    if code_lines:
        for line in code_lines[:5]:
            draw.text(
                (x0 + 24, code_y),
                line,
                font=_font(15, mono=True),
                fill="#DDE8F5",
            )
            code_y += 28
    else:
        draw.text(
            (x0 + 24, code_y),
            "Edits appear here after commit.",
            font=_font(16),
            fill="#7890AD",
        )


def _compose_frame(
    plot: Image.Image,
    *,
    demo_index: int,
    title: str,
    phase: str,
    detail: str,
    active: int,
    total: int,
    layout: str = "full",
    steps: Sequence[str] = (),
    eyebrow: str = "",
    code_lines: Sequence[str] = (),
) -> Image.Image:
    frame = Image.new("RGB", FRAME_SIZE, NAVY)
    _header(frame, demo_index=demo_index, title=title, phase=phase)

    if layout == "full":
        _plot_card(frame, plot, (38, 100, 1162, 622))
    elif layout == "steps":
        _steps_card(
            frame,
            (38, 100, 322, 622),
            eyebrow=eyebrow,
            steps=steps,
            active=active,
        )
        _plot_card(frame, plot, (338, 100, 1162, 622))
    elif layout == "code":
        _plot_card(frame, plot, (38, 100, 758, 622))
        _code_card(
            frame,
            (774, 100, 1162, 622),
            steps=steps,
            active=active,
            code_lines=code_lines,
        )
    else:
        raise ValueError(f"Unknown frame layout: {layout}")

    _footer(frame, detail=detail, active=active, total=total)
    return frame


class DemoRecorder:
    def __init__(
        self,
        *,
        demo_index: int,
        title: str,
        frame_dir: Path,
    ) -> None:
        self.demo_index = demo_index
        self.title = title
        self.frame_dir = frame_dir
        self.frame_dir.mkdir(parents=True, exist_ok=True)
        self.frames: list[Image.Image] = []
        self.durations: list[int] = []

    def capture(
        self,
        window: QtWidgets.QWidget,
        *,
        phase: str,
        detail: str,
        duration: int,
        active: int,
        total: int,
        layout: str = "full",
        steps: Sequence[str] = (),
        eyebrow: str = "",
        code_lines: Sequence[str] = (),
    ) -> None:
        QtWidgets.QApplication.processEvents()
        plot = _grab_figure_view(window)
        frame = _compose_frame(
            plot,
            demo_index=self.demo_index,
            title=self.title,
            phase=phase,
            detail=detail,
            active=active,
            total=total,
            layout=layout,
            steps=steps,
            eyebrow=eyebrow,
            code_lines=code_lines,
        )
        frame.save(self.frame_dir / f"frame-{len(self.frames):02d}.png")
        self.frames.append(frame)
        self.durations.append(duration)

    def save(self, path: Path) -> None:
        if not self.frames:
            raise RuntimeError(f"No frames recorded for {path.name}")
        path.parent.mkdir(parents=True, exist_ok=True)
        palette_frames = [
            frame.quantize(
                colors=128,
                method=Image.Quantize.MEDIANCUT,
                dither=Image.Dither.NONE,
            )
            for frame in self.frames
        ]
        palette_frames[0].save(
            path,
            save_all=True,
            append_images=palette_frames[1:],
            duration=self.durations,
            loop=0,
            optimize=True,
            disposal=2,
        )


def _open_editor():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    QtGuiDrag.initialize(disable_save=True)
    fig, axes = _scientific_figure()
    (window,) = QtGuiDrag.show(hide_window=True)
    window.treeView.hide()
    window.input_properties.hide()
    window.input_align.hide()
    window.tools_scroll.hide()
    window.color_scroll.hide()
    window.resize(1260, 720)
    window.layout_main.setSizes([0, 1240, 0])
    window.show()
    app.processEvents()
    window.plot_layout.canvas_canvas.fitToView(True)
    fig.canvas.draw()
    app.processEvents()
    return app, fig, axes, window


def _close_editor(app, fig, window) -> None:
    window.deactivate()
    window.deleteLater()
    plt.close(fig)
    app.processEvents()


def _select_for_alignment(fig, axes: Sequence[plt.Axes]) -> None:
    manager = fig.figure_dragger
    manager.select_elements(axes, primary=axes[1])
    fig.selection.set_alignment_reference("key_object", key=axes[1])
    fig.selection.set_alignment_key(axes[1])
    fig.canvas.draw()
    QtWidgets.QApplication.processEvents()


def _apply_forge_layout(window) -> None:
    window.input_align.execute_action("top_y")
    window.input_align.execute_action("same_height")
    window.input_align.spacing_enabled.setChecked(True)
    window.input_align.spacing_input.setValue(26)
    window.input_align.execute_action("distribute_x")
    QtWidgets.QApplication.processEvents()


def _change_commands(fig: plt.Figure) -> list[str]:
    commands: list[str] = []
    for command_obj, command in fig.change_tracker.changes.values():
        text = f"{getReference(command_obj)}{command}".replace("\n", " ").strip()
        if len(text) > 43:
            text = text[:40] + "..."
        if text and text not in commands:
            commands.append(text)
    if commands:
        return commands[-5:]
    raise RuntimeError("The real change tracker did not produce Python edits")


def _display_bounds(artist) -> np.ndarray:
    points = np.asarray(TargetWrapper(artist).get_selection_points(), dtype=float)
    return np.array(
        [
            np.min(points[:, 0]),
            np.min(points[:, 1]),
            np.max(points[:, 0]),
            np.max(points[:, 1]),
        ],
        dtype=float,
    )


def _event(
    fig,
    name: str,
    x: float,
    y: float,
    *,
    key: str | None = None,
) -> MouseEvent:
    return MouseEvent(name, fig.canvas, x, y, button=1, key=key)


def _record_pain(frame_root: Path, output_path: Path) -> DemoRecorder:
    app, fig, axes, window = _open_editor()
    recorder = DemoRecorder(
        demo_index=1,
        title="One changed panel should not mean rebuilding the figure",
        frame_dir=frame_root / "01-pain-to-solution",
    )
    recorder.capture(
        window,
        phase="PAIN",
        detail="Different labels and plot sizes destroy the visual rhythm.",
        duration=1700,
        active=0,
        total=3,
    )

    _select_for_alignment(fig, axes)
    recorder.capture(
        window,
        phase="PROPOSAL",
        detail="Edit the native Matplotlib objects together inside Pylustrator Forge.",
        duration=1350,
        active=1,
        total=3,
    )

    _apply_forge_layout(window)
    fig.selection.clear_targets()
    fig.canvas.draw()
    app.processEvents()
    recorder.capture(
        window,
        phase="RESULT",
        detail="Visible tops, heights, and gaps now form one publication-ready system.",
        duration=1900,
        active=2,
        total=3,
    )
    recorder.save(output_path)
    _close_editor(app, fig, window)
    return recorder


def _record_visible_bounds(frame_root: Path, output_path: Path) -> DemoRecorder:
    app, fig, axes, window = _open_editor()
    steps = (
        "Select native Axes",
        "Choose key object",
        "Align visible tops",
        "Set 26 px gaps",
    )
    recorder = DemoRecorder(
        demo_index=2,
        title="Align the painted figure—not an invisible import frame",
        frame_dir=frame_root / "02-visible-bounds",
    )
    fig.figure_dragger.select_elements(axes, primary=axes[1])
    fig.canvas.draw()
    app.processEvents()
    recorder.capture(
        window,
        phase="SELECT",
        detail="Selection geometry includes titles, ticks, labels, and plotted content.",
        duration=1300,
        active=0,
        total=4,
        layout="steps",
        steps=steps,
        eyebrow="VISIBLE-BOUNDS WORKFLOW",
    )

    fig.selection.set_alignment_reference("key_object", key=axes[1])
    fig.selection.set_alignment_key(axes[1])
    fig.canvas.draw()
    app.processEvents()
    recorder.capture(
        window,
        phase="KEY OBJECT",
        detail="The middle panel stays fixed while the other panels move around it.",
        duration=1250,
        active=1,
        total=4,
        layout="steps",
        steps=steps,
        eyebrow="VISIBLE-BOUNDS WORKFLOW",
    )

    window.input_align.execute_action("top_y")
    window.input_align.execute_action("same_height")
    app.processEvents()
    recorder.capture(
        window,
        phase="ALIGN",
        detail="Visible tops and heights match despite different labels and content.",
        duration=1400,
        active=2,
        total=4,
        layout="steps",
        steps=steps,
        eyebrow="VISIBLE-BOUNDS WORKFLOW",
    )

    window.input_align.spacing_enabled.setChecked(True)
    window.input_align.spacing_input.setValue(26)
    window.input_align.execute_action("distribute_x")
    app.processEvents()
    recorder.capture(
        window,
        phase="EXACT GAP",
        detail="The final visual gap is exactly 26 px around the fixed key panel.",
        duration=1850,
        active=3,
        total=4,
        layout="steps",
        steps=steps,
        eyebrow="VISIBLE-BOUNDS WORKFLOW",
    )
    recorder.save(output_path)
    _close_editor(app, fig, window)
    return recorder


def _record_smart_guides(frame_root: Path, output_path: Path) -> DemoRecorder:
    app, fig, axes, window = _open_editor()
    recorder = DemoRecorder(
        demo_index=3,
        title="Drag by eye; Smart Guides finish the geometry",
        frame_dir=frame_root / "03-smart-guides",
    )
    steps = (
        "Select the panel",
        "Drag near equal gap",
        "Guide shows gaps",
        "Release to commit",
    )

    _select_for_alignment(fig, axes)
    _apply_forge_layout(window)
    fig.selection.clear_targets()
    moving = axes[2]
    position = moving.get_position().frozen()
    moving.set_position(
        [position.x0 + 0.055, position.y0, position.width, position.height]
    )
    fig.canvas.draw()
    fig.figure_dragger.select_element(moving)
    fig.selection.smart_guides_allow_blocking_capture = True
    app.processEvents()
    recorder.capture(
        window,
        phase="SELECT",
        detail="The right panel is close, but its visual gap is still wrong.",
        duration=1150,
        active=0,
        total=4,
        layout="steps",
        steps=steps,
        eyebrow="DIRECT-MANIPULATION WORKFLOW",
    )

    moving_bounds = _display_bounds(moving)
    press_x = float((moving_bounds[0] + moving_bounds[2]) / 2)
    press_y = float((moving_bounds[1] + moving_bounds[3]) / 2)
    selection = fig.selection
    selection.button_press_event(
        _event(fig, "button_press_event", press_x, press_y)
    )
    if selection.smart_guide_session is None:
        raise RuntimeError("The real Smart Guide session did not start")
    session = selection.smart_guide_session
    source_guides = [
        next(
            (
                guide
                for guide in session.source_guides
                if guide.stable_id.endswith(f":{id(source):x}")
            ),
            None,
        )
        for source in axes[:2]
    ]
    if any(guide is None for guide in source_guides):
        raise RuntimeError("A source Axes is absent from the Smart Guide index")
    left_guide, middle_guide = sorted(
        source_guides,
        key=lambda guide: guide.bounds.x0,
    )
    source_gap = middle_guide.bounds.x0 - left_guide.bounds.x1
    desired_dx = float(
        middle_guide.bounds.x1 + source_gap - session.moving.bounds.x0
    )

    far_dx = desired_dx - np.sign(desired_dx or 1.0) * 12.0
    selection.on_motion(
        _event(
            fig,
            "motion_notify_event",
            press_x + far_dx,
            press_y,
            key="alt",
        )
    )
    app.processEvents()
    recorder.capture(
        window,
        phase="DRAG",
        detail="Move the native Axes directly—no export, replace, and re-import cycle.",
        duration=1050,
        active=1,
        total=4,
        layout="steps",
        steps=steps,
        eyebrow="DIRECT-MANIPULATION WORKFLOW",
    )

    near_dx = desired_dx - np.sign(desired_dx or 1.0) * 2.0
    selection.on_motion(
        _event(
            fig,
            "motion_notify_event",
            press_x + near_dx,
            press_y,
            key="shift",
        )
    )
    plan = selection.smart_guide_session.last_plan
    if (
        plan is None
        or not plan.overlays
        or not any(hit.kind is GuideKind.EQUAL_GAP for hit in plan.hits)
    ):
        raise RuntimeError(
            "The real Smart Guide did not produce an equal-gap snap: "
            f"desired_dx={desired_dx:.3f}, near_dx={near_dx:.3f}, plan={plan!r}"
        )
    app.processEvents()
    recorder.capture(
        window,
        phase="SNAP",
        detail="The magenta guide shows the repeated visual gap accepted by the solver.",
        duration=1800,
        active=2,
        total=4,
        layout="steps",
        steps=steps,
        eyebrow="DIRECT-MANIPULATION WORKFLOW",
    )

    selection.button_release_event(
        _event(
            fig,
            "button_release_event",
            press_x + near_dx,
            press_y,
            key="shift",
        )
    )
    selection.clear_targets()
    fig.canvas.draw()
    app.processEvents()
    recorder.capture(
        window,
        phase="COMMIT",
        detail="Preview, commit, Undo, and Redo all share the same snapped result.",
        duration=1700,
        active=3,
        total=4,
        layout="steps",
        steps=steps,
        eyebrow="DIRECT-MANIPULATION WORKFLOW",
    )
    recorder.save(output_path)
    _close_editor(app, fig, window)
    return recorder


def _record_reproducible(frame_root: Path, output_path: Path) -> DemoRecorder:
    app, fig, axes, window = _open_editor()
    steps = ("EDIT", "UNDO", "REDO", "SOURCE")
    recorder = DemoRecorder(
        demo_index=4,
        title="Keep visual editing transactional and reproducible",
        frame_dir=frame_root / "04-reproducible-source",
    )
    _select_for_alignment(fig, axes)
    _apply_forge_layout(window)
    fig.selection.clear_targets()
    fig.canvas.draw()
    app.processEvents()
    recorder.capture(
        window,
        phase="EDIT",
        detail="Three visual operations produce one polished native Matplotlib figure.",
        duration=1300,
        active=0,
        total=4,
        layout="code",
        steps=steps,
    )

    for _ in range(3):
        window.undo()
    fig.selection.clear_targets()
    fig.canvas.draw()
    app.processEvents()
    recorder.capture(
        window,
        phase="UNDO",
        detail="Undo restores the original layout instead of leaving partial geometry.",
        duration=1250,
        active=1,
        total=4,
        layout="code",
        steps=steps,
    )

    for _ in range(3):
        window.redo()
    fig.selection.clear_targets()
    fig.canvas.draw()
    app.processEvents()
    recorder.capture(
        window,
        phase="REDO",
        detail="Redo returns to the exact accepted result.",
        duration=1250,
        active=2,
        total=4,
        layout="code",
        steps=steps,
    )

    recorder.capture(
        window,
        phase="SOURCE",
        detail="The same visual edits are emitted as reviewable Python commands.",
        duration=2100,
        active=3,
        total=4,
        layout="code",
        steps=steps,
        code_lines=_change_commands(fig),
    )
    recorder.save(output_path)
    _close_editor(app, fig, window)
    return recorder


def _contact_sheet(
    recorders: Sequence[DemoRecorder],
    path: Path,
) -> None:
    sheet = Image.new("RGB", (1600, 1040), PAPER)
    draw = ImageDraw.Draw(sheet)
    draw.text(
        (54, 35),
        "Pylustrator Forge · focused launch-demo suite",
        font=_font(34, bold=True),
        fill=INK,
    )
    draw.text(
        (54, 80),
        "Each card shows the first and final state; the GIF contains the real steps between them.",
        font=_font(19),
        fill=MUTED,
    )
    labels = (
        "01 · Pain → proposal → result",
        "02 · Visible bounds + key object",
        "03 · Smart Guides",
        "04 · Undo / Redo / Python",
    )
    for index, (recorder, label) in enumerate(zip(recorders, labels)):
        column = index % 2
        row = index // 2
        x0 = 44 + column * 780
        y0 = 130 + row * 445
        card = (x0, y0, x0 + 744, y0 + 410)
        _rounded_card(sheet, card, fill=WHITE)
        draw.text(
            (x0 + 24, y0 + 22),
            label,
            font=_font(22, bold=True),
            fill=INK,
        )
        first = _fit_inside(recorder.frames[0], (338, 190), background=NAVY)
        last = _fit_inside(recorder.frames[-1], (338, 190), background=NAVY)
        sheet.paste(first, (x0 + 24, y0 + 72))
        sheet.paste(last, (x0 + 382, y0 + 72))
        draw.text(
            (x0 + 24, y0 + 278),
            "FIRST STATE",
            font=_font(14, bold=True),
            fill=MUTED,
        )
        draw.text(
            (x0 + 382, y0 + 278),
            "FINAL STATE",
            font=_font(14, bold=True),
            fill=MUTED,
        )
        draw.line(
            (x0 + 24, y0 + 318, x0 + 720, y0 + 318),
            fill=BORDER,
            width=2,
        )
        draw.text(
            (x0 + 24, y0 + 340),
            f"{len(recorder.frames)} real product states · no content overlays",
            font=_font(17),
            fill="#50627A",
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)


def record_suite(frame_root: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    recorders = (
        _record_pain(frame_root, output_dir / DEMO_FILES[0]),
        _record_visible_bounds(frame_root, output_dir / DEMO_FILES[1]),
        _record_smart_guides(frame_root, output_dir / DEMO_FILES[2]),
        _record_reproducible(frame_root, output_dir / DEMO_FILES[3]),
    )
    _contact_sheet(
        recorders,
        output_dir / "pylustrator-forge-demo-suite-contact-sheet.png",
    )
    for filename in DEMO_FILES:
        print(output_dir / filename)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--frames-dir",
        type=Path,
        default=Path("build/demo-forge-suite"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("build/demo-forge-suite-output"),
    )
    args = parser.parse_args()
    record_suite(args.frames_dir, args.output_dir)


if __name__ == "__main__":
    main()
