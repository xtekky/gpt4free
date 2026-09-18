"""
Bootstrap the g4f Flask GUI server inside the Android app.

MainActivity extracts the bundled g4f package and GUI assets from the APK
into filesDir/app, then calls main(app_root, port) here. This module sets
up the environment and runs the server on 127.0.0.1:<port>.
"""
from __future__ import annotations

import os
import sys


def _log(*args):
    print("[g4f-bootstrap]", *args, flush=True)


def main(app_root: str, port: int) -> None:
    _log("starting, app_root=", app_root, "port=", port)

    # HOME must be writable for g4f config (~/.g4f). Chaquopy sets it to the
    # app's internal storage dir already, but be defensive.
    if "HOME" not in os.environ or not os.path.isdir(os.environ.get("HOME", "")):
        os.environ["HOME"] = app_root

    os.environ["G4F_VERSION"] = "0.1.0-android"
    os.environ["G4F_API_HOST"] = "127.0.0.1"
    os.environ["G4F_API_PORT"] = str(port)
    os.environ["G4F_NO_VERSION_CHECK"] = "1"
    # g4f resolves DIST_DIR relative to CWD: ./g4f.dev/dist
    os.chdir(app_root)
    if app_root not in sys.path:
        sys.path.insert(0, app_root)

    _log("importing g4f.gui ...")
    import g4f.gui  # noqa: E402

    _log("creating GUI app ...")
    app = g4f.gui.get_gui_app()
    app.debug = False
    app.timeout = 600
    app.stream_timeout = 300

    _log(f"serving on 127.0.0.1:{port}")
    # Werkzeug dev server is fine for a local, single-user app.
    app.run(host="127.0.0.1", port=port, debug=False, use_reloader=False, threaded=True)
