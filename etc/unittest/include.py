import sys
import pathlib

sys.path.append(str(pathlib.Path(__file__).parent.parent.parent))

import g4f

g4f.debug.logging = False
g4f.debug.version_check = False

DEFAULT_MESSAGES = [{'role': 'user', 'content': 'Hello'}]