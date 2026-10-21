"""Run the local-first Web workspace: python -m pylustrator.web."""

from __future__ import annotations

import argparse
import ipaddress
import os
from pathlib import Path


def validate_bind(host: str, token: str | None) -> None:
    loopback = host.lower() == "localhost"
    try:
        loopback = loopback or ipaddress.ip_address(host).is_loopback
    except ValueError:
        pass
    if not loopback and (not token or len(token) < 24):
        raise ValueError(
            "Non-loopback binding requires PYL_WEB_TOKEN with at least 24 characters. "
            "For remote work, prefer an SSH tunnel to 127.0.0.1:8765."
        )


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Pylustrator Forge headless Web workspace")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--data-dir", type=Path, default=None,
                        help="Persistent project directory (default: ~/.pylustrator/workspace)")
    args = parser.parse_args(argv)
    token = os.environ.get("PYL_WEB_TOKEN")
    try:
        validate_bind(args.host, token)
    except ValueError as exc:
        parser.error(str(exc))
    try:
        import uvicorn

        from .app import create_app
    except ImportError as exc:
        parser.error(f"Install the Web dependencies with pip install 'pylustrator[web]': {exc}")
    uvicorn.run(
        create_app(args.data_dir, token=token), host=args.host, port=args.port,
        proxy_headers=False,
    )


if __name__ == "__main__":
    main()
