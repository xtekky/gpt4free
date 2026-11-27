#!/bin/sh
set -eu
# g4f-go and the repo bin/ commands must use the bundled interpreter in every
# shell, not download a platform CPython.
export G4F_PYTHON=/opt/g4f-venv/bin/python
unset PYTHONHOME
if [ ! -x /usr/local/bin/g4f-desktop ]; then
    echo "Desktop is not installed. Enable 'Install desktop' during setup." >&2
    exit 1
fi
exec /usr/local/bin/g4f-desktop
