from __future__ import annotations

import sys
import os.path
import webview
try:
    from platformdirs import user_config_dir
    has_platformdirs = True
except ImportError:
    has_platformdirs = False

from g4f.gui.gui_parser import gui_parser
from g4f.gui.server.js_api import JsApi
import g4f.version
import g4f.debug

def run_webview(
    debug: bool = False,
    http_port: int = None,
    ssl: bool = True,
    storage_path: str = None,
    gui: str = None
):
    if getattr(sys, 'frozen', False):
        dirname = sys._MEIPASS
    else:
        dirname = os.path.dirname(__file__)
    webview.settings['OPEN_EXTERNAL_LINKS_IN_BROWSER'] = True
    webview.settings['ALLOW_DOWNLOADS'] = True
    webview.create_window(
        f"g4f - {g4f.version.utils.current_version}",
        os.path.join(dirname, "client/index.html"),
        text_select=True,
        js_api=JsApi(),
    )
    if has_platformdirs and storage_path is None:
        storage_path = user_config_dir("g4f-webview")
    webview.start(
        private_mode=False,
        storage_path=storage_path,
        debug=debug,
        http_port=http_port,
        ssl=ssl,
        gui=gui
    )

if __name__ == "__main__":
    parser = gui_parser()
    args = parser.parse_args()
    if args.debug:
        g4f.debug.logging = True
    run_webview(args.debug, args.port)