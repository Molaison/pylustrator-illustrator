"""Observe → act → observe for the existing native FigureSession.

Install explicitly with ``install_arena(app)``. This adapter owns no renderer,
scientific data, geometry engine, or undo snapshots. See docs/agent_arena.md.
"""

from __future__ import annotations

import base64
import json
from copy import deepcopy
from typing import ClassVar
from uuid import uuid4
from weakref import WeakKeyDictionary

import matplotlib as mpl
from matplotlib.axes import Axes

from .document import identifier, number
from .render import RENDER_LOCK
from .session import FigureSession


class ArenaError(ValueError):
    def __init__(self, code, message, status=400, *, receipt=None):
        super().__init__(message)
        self.code, self.status, self.receipt = code, status, receipt


def _revision(value):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ArenaError("validation", "revision must be a nonnegative integer.")
    return value


def _number(value, field, minimum=-1_000_000, maximum=1_000_000):
    try:
        return number(value, field, minimum, maximum)
    except (ValueError, OverflowError) as exc:
        raise ArenaError("validation", f"Invalid {field}: use a finite number in [{minimum}, {maximum}].") from exc


class FigureArena:
    """A process-local, serialized adapter around one live FigureSession.

    IDs are tied to Artist identity, never labels, array positions or Python id().
    Requests and IDs live as long as this adapter. Keep one adapter per session.
    All figure mutations must use FigureSession; direct Python mutations have no
    revision contract. Detached targets are nevertheless rejected on every action.
    """

    MAX_REQUESTS = 1000
    MAX_PREVIEW_PIXELS = 8_000_000
    _FIELDS: ClassVar[dict[str, set[str]]] = {
        "move": {"target", "dx", "dy"},
        "scale": {"target", "factorX", "factorY", "basis", "anchor"},
        "align": {"target", "reference", "edge", "basis"},
        "undo": set(),
        "redo": set(),
    }
    _REQUIRED: ClassVar[dict[str, set[str]]] = {
        "move": {"target", "dx", "dy"},
        "scale": {"target", "factorX", "factorY", "basis"},
        "align": {"target", "reference", "edge", "basis"},
        "undo": set(),
        "redo": set(),
    }

    def __init__(self, session: FigureSession):
        self.session = session
        self._ids = WeakKeyDictionary()
        self._requests = {}
        # Ownership only: the existing session retains all undo/redo snapshots.
        # Stop at an intervening native UI edit instead of undoing scientific edits.
        self._undo, self._redo = [], []
        self._history_revision = session.revision

    def _sync_history(self):
        if self._history_revision != self.session.revision:
            self._undo.clear()
            self._redo.clear()
            self._history_revision = self.session.revision

    def _check_canvas(self):
        figure = self.session.figure
        width, height = figure.bbox.size
        if not (0 < width <= 8192 and 0 < height <= 8192
                and width * height <= self.MAX_PREVIEW_PIXELS):
            raise ArenaError("preview_limit", "Arena previews require a canvas of at most 8 megapixels and 8192 pixels per side.")

    def _restriction(self, artist):
        figure = self.session.figure
        if figure.get_layout_engine() is not None:
            return "Disable automatic figure layout before editing geometry."
        if isinstance(artist, Axes):
            if (artist.figure is not figure or artist.name != "rectilinear"
                    or artist.get_aspect() != "auto" or artist.get_axes_locator() is not None):
                return "Arena v1 edits independent Cartesian panels with automatic aspect and no axes locator."
            if len(artist._twinned_axes.get_siblings(artist)) > 1:
                return "Twinned axes require a coupled-panel operation; Arena v1 does not expose one."
        elif artist.get_transform() is not figure.transFigure:
            return "Arena v1 moves only text in figure coordinates."
        return ""

    def _describe(self):
        self._check_canvas()
        state = self.session.describe()
        self._sync_history()
        native = {entry["id"]: entry for entry in state["artists"]}
        objects, targets = [], {}
        for artist in [*self.session.figure.axes, *self.session.figure.texts]:
            native_id = self.session.artist_id(artist)
            entry = native.get(native_id)
            if entry is None:
                continue
            if artist not in self._ids:
                self._ids[artist] = "object-" + uuid4().hex
            key = self._ids[artist]
            reason = self._restriction(artist)
            if entry["locked"]:
                reason = "Unlock this object in the native editor first."
            elif not entry["visible"]:
                reason = "Show this object in the native editor first."
            translate = entry["capabilities"]["translate"]["supported"]
            scale = isinstance(artist, Axes) and entry["capabilities"]["resize_geometry"]["supported"]
            record = {
                "id": key, "nativeId": native_id, "name": entry["name"],
                "type": entry["type"], "bounds": entry["bounds"],
                "basis": "plot_area" if isinstance(artist, Axes) else "text",
                "visible": entry["visible"], "locked": entry["locked"],
                "capabilities": {"move": bool(translate and not reason),
                                 "scale": bool(scale and not reason),
                                 "align": bool(translate and not reason)},
                "restriction": reason,
            }
            objects.append(record)
            targets[key] = record
        return {
            "protocolVersion": 1, "sessionId": self.session.id,
            "name": state["name"], "revision": state["revision"],
            "canvas": {"width": state["width"], "height": state["height"],
                       "dpi": float(self.session.figure.dpi), "origin": "top-left", "units": "pixels"},
            "objects": objects,
            "history": {"canUndo": bool(self._undo), "canRedo": bool(self._redo),
                        "scope": "Arena actions since the latest external session edit"},
        }, targets

    def _observe(self):
        # One lock covers state and pixels; a separate render URL could race a writer.
        state, _ = self._describe()
        with mpl.rc_context({"savefig.bbox": None, "savefig.pad_inches": 0}):
            preview = self.session.render("png")
        state["preview"] = {"mimeType": "image/png", "encoding": "base64",
                            "width": state["canvas"]["width"], "height": state["canvas"]["height"],
                            "data": base64.b64encode(preview).decode("ascii")}
        return state

    def observe(self):
        with self.session.lock, RENDER_LOCK:
            return self._observe()

    def receipt(self, request_id):
        with self.session.lock:
            if request_id not in self._requests:
                raise ArenaError("request_not_found", "No completed action has this requestId.", 404)
            return deepcopy(self._requests[request_id][1])

    def _validate(self, payload):
        if not isinstance(payload, dict):
            raise ArenaError("validation", "An action must be a JSON object.")
        op = payload.get("operation")
        if not isinstance(op, str) or op not in self._FIELDS:
            raise ArenaError("validation", "Use move, scale, align, undo or redo.")
        common = {"requestId", "revision", "operation"}
        if set(payload) - common - self._FIELDS[op] or (common | self._REQUIRED[op]) - set(payload):
            raise ArenaError("validation", "Unknown or missing action fields; see the Arena API contract.")
        try:
            identifier(payload["requestId"], "requestId")
            fingerprint = json.dumps(payload, sort_keys=True, allow_nan=False)
        except (TypeError, ValueError, RecursionError) as exc:
            raise ArenaError("validation", "Use a short requestId and finite JSON values.") from exc
        if len(fingerprint) > 4096:
            raise ArenaError("validation", "An Arena action must be at most 4096 characters.")
        _revision(payload["revision"])
        return op, fingerprint

    @staticmethod
    def _target(targets, key):
        if not isinstance(key, str):
            raise ArenaError("validation", "An object ID must be a string from observe().")
        if key not in targets:
            raise ArenaError("stale_object", "This object is absent from this Arena. Observe again; IDs are session-local.", 409)
        return targets[key]

    @staticmethod
    def _inside(bounds, canvas):
        x, y, width, height = bounds
        for value in bounds:
            _number(value, "resulting bounds")
        if (width < 1 or height < 1 or x < -1e-6 or y < -1e-6
                or x + width > canvas["width"] + 1e-6 or y + height > canvas["height"] + 1e-6):
            raise ArenaError("geometry_out_of_bounds", "The resulting object bounds must fit the canvas and measure at least one pixel.")

    def _command(self, payload, state, targets):
        op = payload["operation"]
        command = {"revision": payload["revision"]}
        if op in {"undo", "redo"}:
            if not (self._undo if op == "undo" else self._redo):
                raise ArenaError("history_boundary", f"No Arena action is available to {op}; native editor history is unchanged.", 409)
            return {**command, "operation": op}
        target = self._target(targets, payload["target"])
        if not target["capabilities"][op]:
            raise ArenaError("unsupported_operation", target["restriction"] or "This object does not support the requested Arena operation.")
        x, y, width, height = target["bounds"]
        command["targets"] = [target["nativeId"]]
        if op == "scale":
            if payload["basis"] != "plot_area":
                raise ArenaError("validation", "Scaling requires basis='plot_area'; it does not scale fonts or data values.")
            sx = _number(payload["factorX"], "factorX", 0.01, 100)
            sy = _number(payload["factorY"], "factorY", 0.01, 100)
            anchor = payload.get("anchor", "top-left")
            if anchor not in ("top-left", "center"):
                raise ArenaError("validation", "anchor must be top-left or center.")
            px, py = (x, y) if anchor == "top-left" else (x + width / 2, y + height / 2)
            self._inside([px + (x - px) * sx, py + (y - py) * sy, width * sx, height * sy], state["canvas"])
            return {**command, "operation": "resize",
                    "matrix": [[sx, 0, px * (1 - sx)], [0, sy, py * (1 - sy)], [0, 0, 1]]}
        if op == "move":
            dx, dy = _number(payload["dx"], "dx"), _number(payload["dy"], "dy")
        else:
            if payload["basis"] != target["basis"]:
                raise ArenaError("validation", "Align basis must match the observed target basis.")
            if payload["reference"] == "artboard":
                box = [0, 0, state["canvas"]["width"], state["canvas"]["height"]]
            else:
                reference = self._target(targets, payload["reference"])
                if reference["basis"] != target["basis"] or not reference["visible"]:
                    raise ArenaError("validation", "Align to a visible object with the same bounds basis.")
                box = reference["bounds"]
            rx, ry, rw, rh = box
            deltas = {"left": (rx - x, 0), "center": (rx + rw / 2 - x - width / 2, 0),
                      "right": (rx + rw - x - width, 0), "top": (0, ry - y),
                      "middle": (0, ry + rh / 2 - y - height / 2), "bottom": (0, ry + rh - y - height)}
            edge = payload["edge"]
            if not isinstance(edge, str) or edge not in deltas:
                raise ArenaError("validation", "edge must be left, center, right, top, middle or bottom.")
            dx, dy = deltas[edge]
        self._inside([x + dx, y + dy, width, height], state["canvas"])
        return {**command, "operation": "translate", "dx": dx, "dy": dy}

    def _record(self, payload, fingerprint, before):
        op = payload["operation"]
        changed = self.session.revision != before
        if changed:
            if op == "undo":
                self._redo.append(self._undo.pop())
            elif op == "redo":
                self._undo.append(self._redo.pop())
            else:
                self._undo.append(payload["requestId"])
                # FigureSession retains 200 undo items.
                self._undo = self._undo[-200:]
                self._redo.clear()
        self._history_revision = self.session.revision
        receipt = {"requestId": payload["requestId"], "operation": op,
                   "sessionId": self.session.id, "beforeRevision": before,
                   "appliedRevision": self.session.revision, "changed": changed}
        self._requests[payload["requestId"]] = (fingerprint, receipt)
        return receipt

    def act(self, payload):
        payload = deepcopy(payload)
        _, fingerprint = self._validate(payload)
        with self.session.lock, RENDER_LOCK:
            key = payload["requestId"]
            if key in self._requests:
                saved, receipt = self._requests[key]
                if fingerprint != saved:
                    raise ArenaError("request_conflict", "requestId already belongs to a different payload.", 409)
                return self._result(receipt, replayed=True)
            if len(self._requests) >= self.MAX_REQUESTS:
                raise ArenaError("request_limit", "This Arena has reached its 1000-action budget; existing receipts remain readable.", 409)
            state, targets = self._describe()
            before = state["revision"]
            if payload["revision"] != before:
                raise ArenaError("revision_conflict", f"Expected revision {payload['revision']}; current revision is {before}. Observe again.", 409)
            command = self._command(payload, state, targets)
            try:
                self.session.command(command)
            except Exception as exc:
                # FigureSession can commit and then fail while rendering. Reserve
                # the request before returning an error so retry never moves twice.
                if self.session.revision != before:
                    receipt = self._record(payload, fingerprint, before)
                    raise ArenaError("result_unavailable", "The action committed, but its result could not be rendered. Read the receipt or retry the identical request.", 503, receipt=receipt) from exc
                raise
            receipt = self._record(payload, fingerprint, before)
            return self._result(receipt, replayed=False)

    def _result(self, receipt, *, replayed):
        try:
            observation = self._observe()
        except Exception as exc:
            raise ArenaError("result_unavailable", "The action receipt is recorded, but observation failed. Read the receipt or retry the identical request.", 503, receipt=deepcopy(receipt)) from exc
        return {"receipt": deepcopy(receipt), "replayed": replayed, "observation": observation}


def install_arena(app):
    """Attach routes to an existing web.app.create_app() instance, once.

    Reuses its sessions, same-origin/auth middleware, error contracts and locks.
    The ordinary editor and its routes remain unchanged.
    """
    from fastapi import Body
    from fastapi.responses import JSONResponse

    if hasattr(app.state, "arenas"):
        return app
    app.state.arenas = {}

    @app.exception_handler(ArenaError)
    async def arena_error(request, exc):
        error = {"code": exc.code, "message": str(exc)}
        if exc.receipt is not None:
            error["receipt"] = exc.receipt
        return JSONResponse({"error": error}, status_code=exc.status)

    def arena_for(session_id):
        session = app.state.sessions.get(session_id)
        if session is None:
            raise ArenaError("session_not_found", "This native figure session is not open.", 404)
        arena = app.state.arenas.get(session_id)
        if arena is None or arena.session is not session:
            arena = app.state.arenas[session_id] = FigureArena(session)
        return arena

    @app.get("/api/arena/sessions/{session_id}/observe")
    def observe(session_id: str):
        with app.state.session_lock:
            return arena_for(session_id).observe()

    action_body = Body(...)

    @app.post("/api/arena/sessions/{session_id}/actions")
    def act(session_id: str, body: dict = action_body):
        with app.state.session_lock:
            return arena_for(session_id).act(body)

    @app.get("/api/arena/sessions/{session_id}/requests/{request_id}")
    def receipt(session_id: str, request_id: str):
        with app.state.session_lock:
            return arena_for(session_id).receipt(request_id)

    return app
