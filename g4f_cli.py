#!/usr/bin/env python3
"""
Entry point for g4f CLI executable builds
This file is used as the main entry point for building executables with Nuitka
"""

import g4f.debug
g4f.debug.version_check = False

if __name__ == "__main__":
    from g4f.cli import main
    main()