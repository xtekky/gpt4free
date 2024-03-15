import webview
from functools import partial
try:
    from platformdirs import user_config_dir
    has_platformdirs = True
except ImportError:
    has_platformdirs = False

from g4f.gui import run_gui
from g4f.gui.run import gui_parser
import g4f.version
import g4f.debug

def run_webview(
    host: str = "0.0.0.0",
    port: int = 8080,
    debug: bool = False,
    storage_path: str = None
):
    webview.create_window(
        f"g4f - {g4f.version.utils.current_version}",
        f"http://{host}:{port}/",
        text_select=True
    )
    if has_platformdirs and storage_path is None:
        storage_path = user_config_dir("g4f-webview")
    webview.start(
        partial(run_gui, host, port),
        private_mode=False,
        storage_path=storage_path,
        debug=debug
    )

if __name__ == "__main__":
    parser = gui_parser()
    args = parser.parse_args()
    if args.debug:
        g4f.debug.logging = True
    run_webview(args.host, args.port, args.debug)