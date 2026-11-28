#!/bin/sh
set -eu
# g4f-go owns the CPython runtime and the installed g4f package, so no venv
# or PYTHONPATH is exported here. The launcher is rewritten during install
# with --bind host:port (the CLI expects a single "host:port" value).
export G4F_VNC_PORT=5901
export G4F_NOVNC_DIR=/usr/share/novnc
# An upgraded install can still carry the old profile's G4F_PYTHON venv path.
unset PYTHONHOME G4F_PYTHON VIRTUAL_ENV PYTHONPATH
exec /usr/local/bin/g4f-api
