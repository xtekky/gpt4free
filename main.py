import ssl
import certifi
from functools import partial

ssl.default_ca_certs = certifi.where()
ssl.create_default_context = partial(
    ssl.create_default_context,
    cafile=certifi.where()
)

from g4f.gui.webview import run_webview
import g4f.debug
g4f.debug.version_check = False

run_webview(True);