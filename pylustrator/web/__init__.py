"""Headless scientific workspaces; importing this module never starts a GUI."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from matplotlib.figure import Figure


def create_app(data_dir: str | Path | None = None, **kwargs):
    """Create the optional FastAPI application without starting a server."""
    from .app import create_app as factory

    return factory(data_dir=data_dir, **kwargs)


def serve(
    figure: Figure | None = None,
    *,
    data_dir: str | Path | None = None,
    host: str = "127.0.0.1",
    port: int = 8765,
    token: str | None = None,
) -> None:
    """Serve a workspace, optionally attaching a trusted in-process Figure.

    For remote headless machines, use an SSH tunnel to the default loopback address.
    Non-loopback binding requires an explicit access token and a TLS reverse proxy.
    Uploaded Python and pickle files are deliberately not execution inputs.
    """
    import uvicorn

    from .__main__ import validate_bind
    from .session import FigureSession

    validate_bind(host, token)
    sessions = [FigureSession(figure)] if figure is not None else []
    app = create_app(data_dir, token=token, sessions=sessions)
    try:
        uvicorn.run(app, host=host, port=port, proxy_headers=False)
    finally:
        for session in sessions:
            session.close()


__all__ = ["create_app", "serve"]
