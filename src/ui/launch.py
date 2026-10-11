"""CLI and programmatic launcher for GCG Discrete Coordinate Lab Dashboard.

Provides clean command-line and programmatic server execution with configurable
host, port, auto-reload, and automatic browser launch options.
"""

from __future__ import annotations

import argparse
import logging
import sys
import threading
import time
import webbrowser

import uvicorn

logger = logging.getLogger("ui_launcher")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def launch(
    host: str = "127.0.0.1",
    port: int = 8000,
    reload: bool = False,
    open_browser: bool = False,
) -> None:
    """Launch the FastAPI dashboard server programmatically.

    Args:
        host: Host network interface to bind (e.g. '127.0.0.1' or '0.0.0.0').
        port: Listening port.
        reload: Whether to enable uvicorn hot-reloading.
        open_browser: Whether to open the dashboard URL in default web browser.
    """
    target_url = f"http://{host}:{port}"

    if open_browser:
        def _open():
            time.sleep(1.2)
            logger.info("Opening dashboard in default browser at %s", target_url)
            webbrowser.open(target_url)

        thread = threading.Thread(target=_open, daemon=True)
        thread.start()

    logger.info("Starting GCG Research Dashboard server on %s (reload=%s)", target_url, reload)

    uvicorn.run(
        "src.ui.server:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info",
    )


def build_parser() -> argparse.ArgumentParser:
    """Construct command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Launch GCG Discrete Coordinate Lab interactive research dashboard.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host interface to bind server to.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="TCP port to bind server to.",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable uvicorn hot reloading for rapid local development.",
    )
    parser.add_argument(
        "--open-browser",
        action="store_true",
        help="Automatically launch system web browser once server starts.",
    )
    return parser


def main() -> None:
    """CLI entrypoint."""
    parser = build_parser()
    args = parser.parse_args()

    launch(
        host=args.host,
        port=args.port,
        reload=args.reload,
        open_browser=args.open_browser,
    )


if __name__ == "__main__":
    main()
