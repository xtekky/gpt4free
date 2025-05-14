import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from g4f.gui import run_gui
run_gui()
