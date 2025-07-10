"""
Convenience launcher for the SCR Financial Networks dashboard.

Usage
-----
Start **both** the Dash UI and the FastAPI backend::

    python dashboard/run.py

Start only the Dash UI (no separate API server)::

    python dashboard/run.py --ui-only

Start only the FastAPI server::

    python dashboard/run.py --api-only

Environment variables
---------------------
CEREBRAS_API_KEY   Your Cerebras API key (required for AI Analysis tab)
DASHBOARD_PORT     Dash UI port (default 8050)
API_PORT           FastAPI port (default 8000)
DASHBOARD_DEBUG    Set to "false" to disable Dash hot-reload (default "true")
"""

from __future__ import annotations

import argparse
import os
import sys
import threading

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def _run_api(port: int) -> None:
    import uvicorn
    from dashboard.api import app
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")


def _run_dash(port: int, debug: bool) -> None:
    from dashboard.app import app
    app.run(debug=debug, port=port, use_reloader=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="SCR Financial Networks dashboard launcher")
    parser.add_argument("--ui-only", action="store_true", help="Start only the Dash UI")
    parser.add_argument("--api-only", action="store_true", help="Start only the FastAPI server")
    args = parser.parse_args()

    dash_port = int(os.environ.get("DASHBOARD_PORT", 8050))
    api_port = int(os.environ.get("API_PORT", 8000))
    debug = os.environ.get("DASHBOARD_DEBUG", "true").lower() == "true"

    if args.api_only:
        print(f"Starting FastAPI on http://localhost:{api_port}")
        _run_api(api_port)
        return

    if args.ui_only:
        print(f"Starting Dash UI on http://localhost:{dash_port}")
        _run_dash(dash_port, debug)
        return

    # Start API in background thread, Dash in foreground
    api_thread = threading.Thread(target=_run_api, args=(api_port,), daemon=True)
    api_thread.start()
    print(f"FastAPI running on http://localhost:{api_port}/docs")
    print(f"Dash UI running on http://localhost:{dash_port}")
    _run_dash(dash_port, debug)


if __name__ == "__main__":
    main()
