import ssl
import certifi
from functools import partial

ssl.default_ca_certs = certifi.where()
ssl.create_default_context = partial(
    ssl.create_default_context,
    cafile=certifi.where()
)

from g4f.gui.run import run_gui_args, gui_parser
import g4f.debug
g4f.debug.version_check = False
g4f.debug.version = "0.3.1.7"

if __name__ == "__main__":
    parser = gui_parser()
    args = parser.parse_args()
    run_gui_args(args)