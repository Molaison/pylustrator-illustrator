"""Headless native Matplotlib sessions using the existing semantic adapters.

Only trusted in-process Figures enter this API. Uploaded Python and pickles are
never accepted. Each operation is revision checked and is one undo transaction.
"""

from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from io import BytesIO
from threading import RLock
from uuid import uuid4

import matplotlib as mpl
from matplotlib.artist import Artist
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.colors import is_color_like
from matplotlib.text import Text
import numpy as np

from ..artist_adapters import get_artist_adapter, suspend_change_recording
from ..commands import semantic_equal
from ..operations import TransformIntent, TransformOperation
from ..property_adapters import TextContentPlan, text_content_support
from ..property_transactions import PropertyPlan
from ..transform_engine import PointEditPlan, PointEditSource, TransformPlan
from .document import DocumentError, number
from .render import RENDER_LOCK


class SessionConflict(ValueError):
    """The caller edited an older native figure revision."""


class _Tracker:
    """The semantic core's history/recording protocol, with no GUI or source I/O."""

    def __init__(self):
        self.edits = []
        self.last_edit = -1
        self.changes = {}
        self.replay_prefix = []

    def addEdit(self, edit):
        self.edits = self.edits[:self.last_edit + 1]
        self.edits.append(edit)
        if len(self.edits) > 200:
            self.replay_prefix.extend(item[3] for item in self.edits[:-200] if len(item) > 3)
        self.edits = self.edits[-200:]
        self.last_edit = len(self.edits) - 1

    def addChange(self, target, command, reference_obj=None, reference_command=None):
        self.changes[(reference_obj or target, reference_command or command.split("(", 1)[0])] = (target, command)

    addOwnedChange = addChange

    def addNewTextChange(self, target):
        self.addChange(target, ".set_text(" + repr(target.get_text()) + ")")

    def addNewAxesChange(self, target):
        self.addChange(target, ".set_position(" + repr(list(target.get_position().bounds)) + ")")

    def addNewLegendChange(self, target):
        self.addChange(target, ".legend_layout()")

    def capture_recording_state(self):
        return dict(self.changes)

    def restore_recording_state(self, state):
        self.changes = dict(state)


class FigureSession:
    """Attach a native Figure to the headless browser editing protocol.

    Mutating the Figure outside this session while a browser edit is in progress
    is unsupported; callers should serialize external updates with ``session.lock``.
    ``close()`` restores the prior canvas and drops references held by history.
    """

    def __init__(self, figure: Figure, name="Figure"):
        if not isinstance(figure, Figure):
            raise TypeError("FigureSession requires a trusted matplotlib.figure.Figure.")
        self.figure = figure
        self.id = "session-" + uuid4().hex
        self.name = str(name)[:500]
        self.revision = 0
        self._source_size = tuple(float(value) for value in figure.get_size_inches())
        self._source_dpi = float(figure.dpi)
        self.lock = RLock()
        self._tracker = _Tracker()
        self._old_canvas = figure.canvas
        self._canvas = FigureCanvasAgg(figure)
        self._artists = {}
        self._ids = {}
        self._locked = set()
        self._closed = False
        with RENDER_LOCK:
            self._canvas.draw()
            self._refresh()

    @contextmanager
    def _tracking(self):
        absent = object()
        old = getattr(self.figure, "change_tracker", absent)
        old_name = getattr(self.figure, "_variable_name", absent)
        self.figure.change_tracker = self._tracker
        if old_name is absent:
            self.figure._variable_name = "fig"
        try:
            yield
        finally:
            if old is absent:
                del self.figure.change_tracker
            else:
                self.figure.change_tracker = old
            if old_name is absent:
                del self.figure._variable_name

    def _check(self, revision=None):
        if self._closed:
            raise DocumentError("This native figure session is closed.")
        if revision is not None and (isinstance(revision, bool) or revision != self.revision):
            raise SessionConflict(f"Figure changed: expected revision {revision}, current revision {self.revision}.")

    def _refresh(self):
        current = {}
        for artist in self.figure.findobj():
            if artist is self.figure or not isinstance(artist, Artist):
                continue
            adapter = get_artist_adapter(artist)
            if not adapter.supported:
                continue
            key = self._ids.setdefault(id(artist), "artist-" + uuid4().hex)
            current[key] = artist
        self._artists = current

    def _targets(self, ids):
        if not isinstance(ids, list) or not ids or len(ids) > 1000:
            raise DocumentError("Select between 1 and 1000 native artists.")
        if any(not isinstance(key, str) for key in ids):
            raise DocumentError("Native artist IDs must be strings.")
        if len(set(ids)) != len(ids):
            raise DocumentError("Duplicate native artist IDs.")
        self._refresh()
        try:
            return [self._artists[key] for key in ids]
        except (KeyError, TypeError) as exc:
            raise DocumentError("A selected artist no longer exists in the figure.") from exc

    def artist_id(self, artist):
        """Resolve a trusted live Artist to its stable ID in this session."""
        with self.lock, RENDER_LOCK:
            self._check()
            self._refresh()
            key = self._ids.get(id(artist))
            if key is None or self._artists.get(key) is not artist:
                raise DocumentError("This artist is not editable in the attached figure.")
            return key

    def _source_reference(self, artist):
        from ..change_tracker import getReference

        absent = object()
        original = getattr(self.figure, "_variable_name", absent)
        self.figure._variable_name = "fig"
        try:
            reference = getReference(artist)
            # Canonical numeric ownership avoids requiring ax_dict setup and
            # remains independent of an axes' presentation label.
            axes_refs = [(getReference(ax), f"{getReference(ax.figure)}.axes[{ax.figure.axes.index(ax)}]")
                         for ax in self.figure.axes]
            for old, canonical in sorted(axes_refs, key=lambda pair: len(pair[0]), reverse=True):
                if reference == old or reference.startswith(old + "."):
                    reference = canonical + reference[len(old):]
            if not reference.startswith("fig."):
                raise DocumentError("Artist has no stable figure-relative source address.")
            return reference
        finally:
            if original is absent:
                del self.figure._variable_name
            else:
                self.figure._variable_name = original

    def describe(self):
        with self.lock, RENDER_LOCK, self._tracking():
            self._check()
            self._canvas.draw()
            self._refresh()
            W, H = self._canvas.get_width_height()
            artists = []
            for key, artist in self._artists.items():
                adapter = get_artist_adapter(artist)
                try:
                    p = adapter.finite_points(adapter.clip_selection_points(adapter.selection_points()))
                    if not len(p):
                        p = adapter.finite_points(adapter.control_points())
                    if not len(p):
                        continue
                    low, high = np.min(p, axis=0), np.max(p, axis=0)
                    bounds = [float(low[0]), float(H - high[1]), float(high[0] - low[0]), float(high[1] - low[1])]
                except (ValueError, TypeError, AttributeError, RuntimeError):
                    continue
                capabilities = {}
                for op in TransformOperation:
                    support = adapter.operation_support(op)
                    capabilities[op.value] = {"supported": support.supported, "reason": support.reason,
                                              "constraints": list(support.constraints)}
                entry = {"id": key, "type": type(artist).__name__,
                         "name": str(artist.get_label() or type(artist).__name__), "bounds": bounds,
                         "visible": bool(artist.get_visible()), "locked": key in self._locked,
                         "capabilities": capabilities, "points": []}
                if capabilities["edit_points"]["supported"]:
                    try:
                        handles = adapter.point_handle_model()
                        entry["points"] = [{"key": int(k), "x": float(p[0]), "y": float(H - p[1])}
                                           for k, p in zip(handles.keys, handles.positions_array())]
                    except (ValueError, TypeError, AttributeError, RuntimeError) as exc:
                        entry["capabilities"]["edit_points"] = {"supported": False, "reason": str(exc)}
                if isinstance(artist, Text):
                    entry["text"] = artist.get_text()
                    entry["name"] = artist.get_text() or entry["name"]
                    support = text_content_support(artist)
                    entry["capabilities"]["text"] = {"supported": support.supported, "reason": support.reason or ""}
                artists.append(entry)
            return {"id": self.id, "name": self.name, "width": W, "height": H,
                    "revision": self.revision, "artists": artists,
                    "canUndo": self._tracker.last_edit >= 0,
                    "canRedo": self._tracker.last_edit < len(self._tracker.edits) - 1,
                    "undoLabel": self._tracker.edits[self._tracker.last_edit][2] if self._tracker.last_edit >= 0 else "",
                    "redoLabel": self._tracker.edits[self._tracker.last_edit + 1][2] if self._tracker.last_edit < len(self._tracker.edits) - 1 else ""}

    def _history(self, adapters, before, after, label, *, appearance=False, points=None):
        if all(semantic_equal(a, b) for a, b in zip(before, after)):
            return False

        def restore(states):
            with suspend_change_recording():
                for adapter, state in zip(adapters, states):
                    if points is not None:
                        adapter.restore_point_edit_history_state(state)
                    elif appearance:
                        adapter.restore_appearance_state(state, record_changes=False)
                    else:
                        adapter.restore(state)

        self._tracker.addEdit([lambda: restore(before), lambda: restore(after), label])
        return True

    def command(self, payload):
        if not isinstance(payload, dict):
            raise DocumentError("A native command must be an object.")
        if "revision" not in payload or isinstance(payload["revision"], bool) or not isinstance(payload["revision"], int):
            raise DocumentError("A native command requires its integer revision.")
        with self.lock, RENDER_LOCK, self._tracking():
            self._check(payload.get("revision"))
            op = payload.get("operation")
            if not isinstance(op, str):
                raise DocumentError("A native operation must be a string.")
            if op == "undo":
                return self.undo(payload.get("revision"))
            if op == "redo":
                return self.redo(payload.get("revision"))
            ids = payload.get("targets")
            targets = self._targets(ids)
            try:
                source_refs = [self._source_reference(target) for target in targets]
                source_error = None
            except (AttributeError, TypeError, ValueError) as exc:
                source_refs, source_error = [], str(exc)
            if op != "lock" and any(key in self._locked for key in ids):
                raise DocumentError("Unlock the selected artist before editing it.")
            adapters = [get_artist_adapter(target) for target in targets]
            changed = False
            if op in {"translate", "resize", "rotate", "rigid_rotate", "scale_appearance"}:
                if any(target.axes in targets and target.axes is not target for target in targets):
                    raise DocumentError("Select an axes container or its children, not both in the same transform.")
                if op == "translate":
                    intent = TransformIntent.translate((number(payload.get("dx", 0), "dx"), -number(payload.get("dy", 0), "dy")))
                elif op == "resize":
                    raw = payload.get("matrix")
                    if not isinstance(raw, list) or len(raw) != 3 or any(not isinstance(row, list) or len(row) != 3 for row in raw):
                        raise DocumentError("Resize matrix must be a 3 by 3 affine matrix in screen coordinates.")
                    matrix = np.array([[number(v, "matrix") for v in row] for row in raw])
                    if not np.allclose(matrix[2], [0, 0, 1]):
                        raise DocumentError("Perspective transforms are unsupported.")
                    H = self._canvas.get_width_height()[1]
                    flip = np.array([[1, 0, 0], [0, -1, H], [0, 0, 1]])
                    intent = TransformIntent.resize(flip @ matrix @ flip)
                elif op == "rotate":
                    intent = TransformIntent.rotate(-number(payload.get("angle", 0), "angle"))
                elif op == "rigid_rotate":
                    pivot = payload.get("pivot")
                    if not isinstance(pivot, list) or len(pivot) != 2:
                        raise DocumentError("Rigid rotation needs a [x, y] screen-space pivot.")
                    intent = TransformIntent.rigid_rotate(-number(payload.get("angle", 0), "angle"),
                        (number(pivot[0], "pivot.x"), self._canvas.get_width_height()[1] - number(pivot[1], "pivot.y")))
                else:
                    intent = TransformIntent.scale_appearance(number(payload.get("factor"), "factor", 0.001, 1000))
                plan = TransformPlan.preflight(targets, intent)
                appearance = op == "scale_appearance"
                # Adapters own snapshot copying. A generic deepcopy would clone
                # live transform/Artist identities in Legend and group snapshots.
                before = [a.appearance_state() if appearance else a.snapshot() for a in adapters]
                with suspend_change_recording():
                    plan.commit()
                after = [a.appearance_state() if appearance else a.snapshot() for a in adapters]
                changed = self._history(adapters, before, after, intent.label, appearance=appearance)
            elif op == "points":
                if len(targets) != 1:
                    raise DocumentError("Direct selection edits one artist's points per gesture.")
                points = payload.get("points")
                if not isinstance(points, list) or not 1 <= len(points) <= 1024:
                    raise DocumentError("A point edit needs between 1 and 1024 handles.")
                keys, positions = [], []
                H = self._canvas.get_width_height()[1]
                for point in points:
                    if not isinstance(point, dict) or isinstance(point.get("key"), bool) or not isinstance(point.get("key"), int):
                        raise DocumentError("Each point needs an integer key and numeric x/y.")
                    keys.append(point["key"])
                    positions.append([number(point.get("x"), "point.x"), H - number(point.get("y"), "point.y")])
                source = PointEditSource.capture(targets[0])
                plan = PointEditPlan.preview(source, keys, positions)
                before = [adapters[0].point_edit_history_snapshot(keys)]
                with suspend_change_recording():
                    plan.commit()
                after = [adapters[0].point_edit_history_snapshot(keys)]
                changed = self._history(adapters, before, after, "Edit points", points=keys)
            elif op == "text":
                value = payload.get("text")
                if len(targets) != 1 or not isinstance(value, str) or len(value) > 10000:
                    raise DocumentError("Text editing needs one text target and at most 10000 characters.")
                changed = TextContentPlan.preflight(targets[0]).commit(value)
            elif op in {"style", "visible"}:
                properties = {"visible": payload.get("visible")} if op == "visible" else payload.get("properties")
                allowed = {"color", "facecolor", "edgecolor", "linewidth", "linestyle", "alpha", "fontsize", "fontweight", "fontfamily", "markersize", "visible", "zorder"}
                if not isinstance(properties, dict) or not properties or not set(properties).issubset(allowed):
                    raise DocumentError("Unknown or empty native style properties.")
                for key, value in properties.items():
                    if key == "visible" and not isinstance(value, bool):
                        raise DocumentError("visible must be a boolean.")
                    if key in {"linewidth", "fontsize", "markersize"}:
                        number(value, key, 0, 1000)
                    elif key == "alpha":
                        number(value, key, 0, 1)
                    elif key == "zorder":
                        number(value, key)
                    elif key in {"color", "facecolor", "edgecolor"}:
                        if not isinstance(value, str) or len(value) > 64 or not is_color_like(value):
                            raise DocumentError(f"{key} must be a supported color.")
                    elif key in {"linestyle", "fontweight", "fontfamily"}:
                        if not isinstance(value, str) or len(value) > 100:
                            raise DocumentError(f"{key} must be a short string.")
                changed = PropertyPlan.for_targets(targets, properties).commit("Change " + op)
            elif op == "lock":
                locked = payload.get("locked")
                if not isinstance(locked, bool):
                    raise DocumentError("locked must be a boolean.")
                before = set(self._locked)
                after = before | set(ids) if locked else before - set(ids)
                if after != before:
                    self._locked = after
                    self._tracker.addEdit([lambda: setattr(self, "_locked", set(before)),
                                           lambda: setattr(self, "_locked", set(after)), "Lock" if locked else "Unlock"])
                    changed = True
            else:
                raise DocumentError(f"Unsupported native operation: {op!r}.")
            if changed:
                record = {"payload": deepcopy({key: value for key, value in payload.items()
                                                if key not in {"targets", "revision"}}),
                          "references": source_refs, "error": source_error}
                self._tracker.edits[self._tracker.last_edit].append(record)
                self.revision += 1
                self._canvas.draw()
            return self.describe()

    def undo(self, revision=None):
        with self.lock, RENDER_LOCK, self._tracking():
            self._check(revision)
            if self._tracker.last_edit >= 0:
                self._tracker.edits[self._tracker.last_edit][0]()
                self._tracker.last_edit -= 1
                self.revision += 1
                self._canvas.draw()
            return self.describe()

    def redo(self, revision=None):
        with self.lock, RENDER_LOCK, self._tracking():
            self._check(revision)
            if self._tracker.last_edit < len(self._tracker.edits) - 1:
                self._tracker.edits[self._tracker.last_edit + 1][1]()
                self._tracker.last_edit += 1
                self.revision += 1
                self._canvas.draw()
            return self.describe()

    def render(self, format="svg"):
        if not isinstance(format, str) or format not in {"svg", "png", "pdf"}:
            raise DocumentError("Native sessions export svg, png, or pdf.")
        with self.lock, RENDER_LOCK, mpl.rc_context({"text.usetex": False, "svg.fonttype": "none"}):
            self._check()
            output = BytesIO()
            self.figure.savefig(output, format=format, dpi=self.figure.dpi)
            return output.getvalue()

    def replay_source(self) -> str:
        """Export active semantic edits for a freshly rebuilt original ``fig``.

        This is an edit snippet, not a serializer for arbitrary Python Figures.
        Run it after the original plotting code at the same size/DPI. No eval or
        uploaded Python is involved in the server's command execution path.
        """
        with self.lock, RENDER_LOCK:
            self._check()
            records = [*self._tracker.replay_prefix,
                       *(item[3] for item in self._tracker.edits[:self._tracker.last_edit + 1] if len(item) > 3)]
            if any(record.get("error") for record in records):
                raise DocumentError("An active edit has no stable source address; undo that edit before exporting native replay.")
            source = [
                "# Pylustrator Forge native edit snippet",
                "# Run after rebuilding the ORIGINAL figure as `fig`, at the original size/DPI.",
                "# This file contains current active edits; it does not contain your original plot/data.",
                "from pylustrator.web.session import FigureSession as _PylustratorFigureSession",
                f"if tuple(float(v) for v in fig.get_size_inches()) != {self._source_size!r} or float(fig.dpi) != {self._source_dpi!r}:",
                "    raise ValueError('Replay requires the original figure size and DPI; resize after applying edits.')",
                "_pyl_session = _PylustratorFigureSession(fig)",
                "try:",
            ]
            for record in records:
                source.append(f"    _pyl_command = {record['payload']!r}")
                targets = ", ".join(f"_pyl_session.artist_id({reference})" for reference in record["references"])
                source.append(f"    _pyl_command['targets'] = [{targets}]")
                source.append("    _pyl_command['revision'] = _pyl_session.revision")
                source.append("    _pyl_session.command(_pyl_command)")
            if not records:
                source.append("    pass  # No active edits")
            source.extend(["finally:", "    _pyl_session.close()", ""])
            return "\n".join(source)

    def close(self):
        with self.lock, RENDER_LOCK:
            if not self._closed:
                if self.figure.canvas is self._canvas:
                    self.figure.set_canvas(self._old_canvas)
                self._tracker.edits.clear()
                self._tracker.changes.clear()
                self._tracker.replay_prefix.clear()
                self._artists.clear()
                self._ids.clear()
                self._closed = True
