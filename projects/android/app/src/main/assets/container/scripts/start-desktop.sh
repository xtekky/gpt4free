#!/bin/sh
set -eu
# g4f-go manages the container's Python runtime; never inherit a stale root.
unset PYTHONHOME
if [ ! -x /usr/local/bin/g4f-desktop ]; then
    echo "Desktop is not installed. Enable 'Install desktop' during setup." >&2
    exit 1
fi
exec /usr/local/bin/g4f-desktop
