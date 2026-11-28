#!/bin/sh
set -eu
# The prebuilt overlay's editable metadata predates the g4f/ wrapper directory.
# PYTHONPATH makes "import g4f" resolve against the bundled source tree.
export PYTHONPATH=/g4f-src${PYTHONPATH:+:$PYTHONPATH}
export VIRTUAL_ENV=/opt/g4f-venv
export G4F_PYTHON=/opt/g4f-venv/bin/python
export PATH="$VIRTUAL_ENV/bin:$PATH"
# Register the g4f API's /desktop/ VNC relay for Xvnc :1 (port 5901).
export G4F_VNC_PORT=5901
export G4F_NOVNC_DIR=/usr/share/novnc
unset PYTHONHOME
# The launcher is rewritten during install with --bind host:port (the CLI
# expects a single "host:port" value, not separate --bind/--port arguments).
exec /usr/local/bin/g4f-api
