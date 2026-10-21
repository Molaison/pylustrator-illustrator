"""Same-origin HTTP adapter for the headless scientific workspace."""

from __future__ import annotations

import base64
import binascii
import hmac
import io
import os
import threading
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from fastapi import Body, FastAPI, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError
from starlette.middleware.trustedhost import TrustedHostMiddleware

from . import ai
from .document import new_document, validate_document
from .protein import parse_structure, validate_view
from .render import render_document, render_plot
from .session import SessionConflict
from .store import ConflictError, NotFoundError, ProjectStore, ValidationError

MAX_REQUEST_BYTES = 64 * 1024 * 1024
STATIC_DIR = Path(__file__).parent / "static"
MIMETYPES = {
    "svg": "image/svg+xml", "png": "image/png", "pdf": "application/pdf",
    "python": "text/x-python; charset=utf-8",
}


def error_response(status: int, code: str, message: str) -> JSONResponse:
    return JSONResponse({"error": {"code": code, "message": message}}, status_code=status)


class RequestBoundary:
    """Bound actual streamed bytes, not only a client-supplied Content-Length."""

    def __init__(self, app, token: str | None = None):
        self.app = app
        self.token = token

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        request = Request(scope)
        is_api = scope["path"].startswith("/api/")
        method = scope["method"]
        if is_api:
            origin = request.headers.get("origin")
            site = request.headers.get("sec-fetch-site", "")
            if site == "cross-site" or (origin and (
                urlsplit(origin).scheme not in {"http", "https"}
                or urlsplit(origin).netloc != request.headers.get("host")
            )):
                return await error_response(403, "origin", "Use this workspace from its own browser address.")(scope, receive, send)
            if self.token and scope["path"] not in {"/api/auth", "/api/health"}:
                bearer = request.headers.get("authorization", "").removeprefix("Bearer ")
                credential = bearer or request.cookies.get("pylustrator_access", "")
                if not hmac.compare_digest(credential.encode(), self.token.encode()):
                    return await error_response(401, "authentication", "Enter this workspace's access token to connect.")(scope, receive, send)
            if method in {"POST", "PUT", "PATCH"}:
                content_type = request.headers.get("content-type", "").split(";", 1)[0]
                if content_type != "application/json":
                    return await error_response(415, "content_type", "This endpoint accepts JSON only.")(scope, receive, send)
        if method in {"POST", "PUT", "PATCH"}:
            size = 0
            chunks = []
            while True:
                message = await receive()
                if message["type"] == "http.disconnect":
                    return
                size += len(message.get("body", b""))
                if size > MAX_REQUEST_BYTES:
                    return await error_response(413, "too_large", "This request exceeds the 64 MiB project upload limit.")(scope, receive, send)
                chunks.append(message.get("body", b""))
                if not message.get("more_body", False):
                    break
            body = b"".join(chunks)
            delivered = False

            async def bounded_receive():
                nonlocal delivered
                if not delivered:
                    delivered = True
                    return {"type": "http.request", "body": body, "more_body": False}
                return await receive()

            await self.app(scope, bounded_receive, send)
        else:
            await self.app(scope, receive, send)


def _string(body: dict, key: str, default=None, limit: int = 512) -> str:
    value = body.get(key, default)
    if not isinstance(value, str) or not value.strip() or len(value) > limit:
        raise ValidationError(f"{key} must be a non-empty string of at most {limit} characters.")
    return value


def _revision(body: dict, *, required: bool = True):
    revision = body.get("revision")
    if revision is None and not required:
        return None
    if isinstance(revision, bool) or not isinstance(revision, int) or revision < 0:
        raise ValidationError("Include the current project revision before saving.")
    return revision


def _png(data: Any) -> bytes:
    if not isinstance(data, str):
        raise ValidationError("Image data must be a base64 PNG string.")
    if data.startswith("data:image/png;base64,"):
        data = data.split(",", 1)[1]
    try:
        decoded = base64.b64decode(data, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise ValidationError("The image data is not valid base64.") from exc
    if len(decoded) > 12 * 1024 * 1024:
        raise ValidationError("PNG assets must be at most 12 MiB.")
    try:
        with Image.open(io.BytesIO(decoded)) as image:
            if image.format != "PNG" or image.width * image.height > 24_000_000:
                raise ValidationError("Use a PNG image with at most 24 million pixels.")
            image.verify()
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise ValidationError("The image could not be verified as a PNG.") from exc
    return decoded


def create_app(data_dir: str | Path | None = None, *, token: str | None = None,
               allowed_hosts: list[str] | None = None, sessions=None) -> FastAPI:
    """Build a single-process local workspace with per-project durable revisions."""
    root = Path(data_dir or os.environ.get("PYL_WORKSPACE_DIR", "~/.pylustrator/workspace")).expanduser()
    store = ProjectStore(root)
    app = FastAPI(title="Pylustrator Forge Web", version="1.0.0", docs_url=None, redoc_url=None)
    app.state.store = store
    app.state.sessions = {session.id: session for session in (sessions or [])}
    app.state.session_lock = threading.RLock()
    hosts = allowed_hosts or ["127.0.0.1", "localhost", "[::1]", "testserver"]
    hosts += [host.strip() for host in os.environ.get("PYL_WEB_ALLOWED_HOSTS", "").split(",") if host.strip()]
    app.add_middleware(RequestBoundary, token=token)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=hosts)

    @app.middleware("http")
    async def security_headers(request: Request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "same-origin"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data: blob:; connect-src 'self'; worker-src 'self' blob:; "
            "object-src 'none'; base-uri 'self'; frame-ancestors 'none'"
        )
        if request.url.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"
        return response

    @app.exception_handler(ConflictError)
    async def conflict_error(request, exc):
        return error_response(409, "revision_conflict", str(exc))

    @app.exception_handler(SessionConflict)
    async def session_conflict(request, exc):
        return error_response(409, "revision_conflict", str(exc))

    @app.exception_handler(ai.ProviderError)
    async def provider_error(request, exc):
        return error_response(502, "provider_unavailable", str(exc))

    @app.exception_handler(NotFoundError)
    async def not_found(request, exc):
        return error_response(404, "not_found", str(exc))

    @app.exception_handler(ValueError)
    async def value_error(request, exc):
        return error_response(400, "validation", str(exc))

    @app.exception_handler(RequestValidationError)
    async def request_error(request, exc):
        return error_response(422, "invalid_request", "The request body or query parameters are invalid.")

    @app.get("/api/health")
    def health():
        return {"status": "ok", "version": "1.0.0", "authentication": bool(token)}

    @app.post("/api/auth")
    def authenticate(request: Request, body: dict = Body(...)):
        supplied = body.get("token", "")
        if not isinstance(supplied, str) or not token or not hmac.compare_digest(supplied.encode(), token.encode()):
            return error_response(401, "authentication", "The access token was not accepted.")
        response = JSONResponse({"authenticated": True})
        response.set_cookie("pylustrator_access", token, httponly=True, samesite="strict",
                            secure=request.url.scheme == "https", max_age=8 * 60 * 60)
        return response

    @app.get("/api/projects")
    def list_projects():
        return {"projects": store.list_projects()}

    @app.post("/api/projects")
    def create_project(body: dict = Body(...)):
        name = _string(body, "name", "Untitled project", 160)
        project = store.create_project(name, description=body.get("description", ""), document=new_document())
        if body.get("example") is True:
            project = store.add_dataset(project["id"], "Example: enzyme response.csv",
                                        "time,response\n0,0.1\n1,0.28\n2,0.52\n3,0.67\n4,0.78\n5,0.84\n6,0.89\n", "csv")
            dataset_id = project["datasets"][0]["id"]
            document = new_document("Example scientific figure")
            document["elements"] = [
                {"id": "example-title", "type": "text", "name": "Figure title", "x": 65, "y": 30,
                 "width": 800, "height": 42, "text": "Enzyme response · example data",
                 "style": {"fill": "#213547", "fontSize": 24}},
                {"id": "example-panel", "type": "text", "name": "Panel label", "x": 35, "y": 92,
                 "width": 35, "height": 35, "text": "A", "style": {"fill": "#213547", "fontSize": 26}},
                {"id": "example-plot", "type": "plot", "name": "Response over time", "x": 80, "y": 95,
                 "width": 790, "height": 440, "plot": {"datasetId": dataset_id, "kind": "line", "x": "time",
                 "y": "response", "xlabel": "Time (min)", "ylabel": "Response (a.u.)", "color": "#2874a6"}},
                {"id": "example-caption", "type": "text", "name": "Caption", "x": 80, "y": 565,
                 "width": 800, "height": 35, "text": "Illustrative values. Import your data in the Data workspace.",
                 "style": {"fill": "#536475", "fontSize": 14}},
            ]
            project = store.update_project(project["id"], project["revision"], document=validate_document(document))
        return project

    @app.post("/api/projects/import")
    def import_project(body: dict = Body(...)):
        return store.import_bundle(body)

    @app.get("/api/projects/{project_id}")
    def get_project(project_id: str):
        return store.get_project(project_id)

    @app.put("/api/projects/{project_id}")
    def update_project(project_id: str, body: dict = Body(...)):
        changes = {key: body[key] for key in ("document", "name", "description", "tags") if key in body}
        if "document" in changes:
            changes["document"] = validate_document(changes["document"])
        return store.update_project(project_id, _revision(body), **changes)

    @app.get("/api/projects/{project_id}/bundle")
    def export_bundle(project_id: str):
        return JSONResponse(store.export_bundle(project_id), headers={
            "Content-Disposition": 'attachment; filename="pylustrator-project.json"'})

    @app.get("/api/projects/{project_id}/versions")
    def list_versions(project_id: str):
        return {"versions": store.list_versions(project_id)}

    @app.post("/api/projects/{project_id}/versions")
    def save_version(project_id: str, body: dict = Body(...)):
        return store.save_version(project_id, _string(body, "name", "Checkpoint", 160), _revision(body))

    @app.post("/api/projects/{project_id}/versions/{version_id}/restore")
    def restore_version(project_id: str, version_id: str, body: dict = Body(...)):
        return store.restore_version(project_id, version_id, _revision(body))

    @app.post("/api/projects/{project_id}/datasets")
    def import_dataset(project_id: str, body: dict = Body(...)):
        return store.add_dataset(project_id, _string(body, "name", "Data", 240),
                                 _string(body, "content", limit=16 * 1024 * 1024),
                                 format=body.get("format", "csv"), expected_revision=_revision(body, required=False))

    @app.get("/api/projects/{project_id}/datasets/{dataset_id}")
    def get_dataset(project_id: str, dataset_id: str, offset: int = Query(0, ge=0), limit: int = Query(100, ge=1, le=500)):
        return store.get_dataset(project_id, dataset_id, offset=offset, limit=limit)

    @app.post("/api/projects/{project_id}/datasets/{dataset_id}/preview")
    def preview_dataset(project_id: str, dataset_id: str, body: dict = Body(...)):
        return store.preview_transform(project_id, dataset_id, body.get("steps", []))

    @app.post("/api/projects/{project_id}/datasets/{dataset_id}/transform")
    def transform_dataset(project_id: str, dataset_id: str, body: dict = Body(...)):
        return store.transform_dataset(project_id, dataset_id, body.get("steps", []), name=body.get("name"),
                                       expected_hash=body.get("expectedHash"), expected_revision=_revision(body, required=False),
                                       provenance=body.get("provenance"))

    @app.get("/api/providers")
    def providers():
        return {"providers": ai.list_providers(), "operations": ai.operations()}

    @app.post("/api/projects/{project_id}/datasets/{dataset_id}/ai-plan")
    def ai_plan(project_id: str, dataset_id: str, body: dict = Body(...)):
        dataset = store.get_dataset_data(project_id, dataset_id)
        return ai.propose_plan(dataset, _string(body, "prompt", limit=4000), body.get("provider", "local"))

    @app.post("/api/projects/{project_id}/structures")
    def import_structure(project_id: str, body: dict = Body(...)):
        content = _string(body, "content", limit=16 * 1024 * 1024)
        metadata = parse_structure(content, body.get("format", "pdb"))
        return store.add_structure(project_id, _string(body, "name", "Structure", 240), content,
                                   format=metadata["format"], metadata=metadata,
                                   view=validate_view(None), expected_revision=_revision(body, required=False))

    @app.get("/api/projects/{project_id}/structures/{structure_id}")
    def get_structure(project_id: str, structure_id: str):
        return store.get_structure(project_id, structure_id)

    @app.put("/api/projects/{project_id}/structures/{structure_id}")
    def update_structure(project_id: str, structure_id: str, body: dict = Body(...)):
        structure = store.get_structure(project_id, structure_id)
        ids = structure.get("chainIds") or structure.get("metadata", {}).get("chainIds")
        return store.update_structure(project_id, structure_id, validate_view(body.get("view"), ids), _revision(body))

    @app.post("/api/projects/{project_id}/assets")
    def import_asset(project_id: str, body: dict = Body(...)):
        if body.get("mimeType", "image/png") != "image/png":
            raise ValidationError("Only PNG image assets are accepted.")
        return store.add_asset(project_id, _string(body, "name", "Image", 240), _png(body.get("data")),
                               mime_type="image/png", provenance=body.get("provenance"),
                               expected_revision=_revision(body, required=False))

    @app.get("/api/projects/{project_id}/assets/{asset_id}")
    def get_asset(project_id: str, asset_id: str):
        data, metadata = store.get_asset(project_id, asset_id)
        return Response(data, media_type=metadata.get("mimeType", "image/png"))

    @app.post("/api/render/plot")
    def plot_preview(body: dict = Body(...)):
        plot = body.get("plot", {})
        if not isinstance(plot, dict) or not isinstance(plot.get("datasetId"), str):
            raise ValidationError("Choose a dataset for the plot.")
        rows = store.dataset_rows(_string(body, "projectId"), plot["datasetId"])
        return Response(render_plot(plot, rows, width=body.get("width", 480), height=body.get("height", 320)),
                        media_type="image/svg+xml")

    @app.post("/api/projects/{project_id}/export")
    def export_figure(project_id: str, body: dict = Body(...)):
        project = store.get_project(project_id)
        document = validate_document(body.get("document", project["document"]))
        format = body.get("format", "svg")
        if format not in MIMETYPES:
            raise ValidationError("Choose SVG, PNG, PDF, or Python export.")
        data = render_document(document, lambda did: store.dataset_rows(project_id, did),
                               lambda aid: store.get_asset(project_id, aid)[0], format=format)
        extension = "py" if format == "python" else format
        return Response(data, media_type=MIMETYPES[format], headers={
            "Content-Disposition": f'attachment; filename="figure.{extension}"'})

    def session_by_id(session_id: str):
        try:
            return app.state.sessions[session_id]
        except KeyError as exc:
            raise NotFoundError("This native figure session is not open.") from exc

    @app.get("/api/sessions")
    def list_sessions():
        with app.state.session_lock:
            return {"sessions": [session.describe() for session in app.state.sessions.values()]}

    @app.get("/api/sessions/{session_id}")
    def describe_session(session_id: str):
        return session_by_id(session_id).describe()

    @app.get("/api/sessions/{session_id}/render.{format}")
    def render_session(session_id: str, format: str):
        if format not in {"svg", "png", "pdf"}:
            raise ValidationError("Choose SVG, PNG, or PDF for a native figure.")
        return Response(session_by_id(session_id).render(format=format), media_type=MIMETYPES[format])

    @app.get("/api/sessions/{session_id}/changes.py")
    def replay_session(session_id: str):
        return Response(session_by_id(session_id).replay_source(), media_type=MIMETYPES["python"],
                        headers={"Content-Disposition": 'attachment; filename="figure-native-changes.py"'})

    @app.post("/api/sessions/{session_id}/commands")
    def command_session(session_id: str, body: dict = Body(...)):
        with app.state.session_lock:
            return session_by_id(session_id).command(body)

    @app.get("/")
    def index():
        return FileResponse(STATIC_DIR / "index.html", headers={"Cache-Control": "no-cache"})

    app.mount("/static", StaticFiles(directory=STATIC_DIR, check_dir=False), name="static")
    return app
