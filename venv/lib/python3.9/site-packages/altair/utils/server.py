"""
A Simple server used to show altair graphics from a prompt or script.

This is adapted from the mpld3 package; see
https://github.com/mpld3/mpld3/blob/master/mpld3/_server.py
"""
import sys
import threading
import webbrowser
import socket
from http import server
from io import BytesIO as IO
import itertools
import random

JUPYTER_WARNING = """
Note: if you're in the Jupyter notebook, Chart.serve() is not the best
      way to view plots. Consider using Chart.display().
You must interrupt the kernel to cancel this command.
"""


# Mock server used for testing


class MockRequest(object):
    def makefile(self, *args, **kwargs):
        return IO(b"GET /")

    def sendall(self, response):
        pass


class MockServer(object):
    def __init__(self, ip_port, Handler):
        Handler(MockRequest(), ip_port[0], self)

    def serve_forever(self):
        pass

    def server_close(self):
        pass


def generate_handler(html, files=None):
    if files is None:
        files = {}

    class MyHandler(server.BaseHTTPRequestHandler):
        def do_GET(self):
            """Respond to a GET request."""
            if self.path == "/":
                self.send_response(200)
                self.send_header("Content-type", "text/html")
                self.end_headers()
                self.wfile.write(html.encode())
            elif self.path in files:
                content_type, content = files[self.path]
                self.send_response(200)
                self.send_header("Content-type", content_type)
                self.end_headers()
                self.wfile.write(content.encode())
            else:
                self.send_error(404)

    return MyHandler


def find_open_port(ip, port, n=50):
    """Find an open port near the specified port"""
    ports = itertools.chain(
        (port + i for i in range(n)), (port + random.randint(-2 * n, 2 * n))
    )

    for port in ports:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        result = s.connect_ex((ip, port))
        s.close()
        if result != 0:
            return port
    raise ValueError("no open ports found")


def serve(
    html,
    ip="127.0.0.1",
    port=8888,
    n_retries=50,
    files=None,
    jupyter_warning=True,
    open_browser=True,
    http_server=None,
):
    """Start a server serving the given HTML, and (optionally) open a browser

    Parameters
    ----------
    html : string
        HTML to serve
    ip : string (default = '127.0.0.1')
        ip address at which the HTML will be served.
    port : int (default = 8888)
        the port at which to serve the HTML
    n_retries : int (default = 50)
        the number of nearby ports to search if the specified port is in use.
    files : dictionary (optional)
        dictionary of extra content to serve
    jupyter_warning : bool (optional)
        if True (default), then print a warning if this is used within Jupyter
    open_browser : bool (optional)
        if True (default), then open a web browser to the given HTML
    http_server : class (optional)
        optionally specify an HTTPServer class to use for showing the
        figure. The default is Python's basic HTTPServer.
    """
    port = find_open_port(ip, port, n_retries)
    Handler = generate_handler(html, files)

    if http_server is None:
        srvr = server.HTTPServer((ip, port), Handler)
    else:
        srvr = http_server((ip, port), Handler)

    if jupyter_warning:
        try:
            __IPYTHON__  # noqa
        except NameError:
            pass
        else:
            print(JUPYTER_WARNING)

    # Start the server
    print("Serving to http://{}:{}/    [Ctrl-C to exit]".format(ip, port))
    sys.stdout.flush()

    if open_browser:
        # Use a thread to open a web browser pointing to the server
        def b():
            return webbrowser.open("http://{}:{}".format(ip, port))

        threading.Thread(target=b).start()

    try:
        srvr.serve_forever()
    except (KeyboardInterrupt, SystemExit):
        print("\nstopping Server...")

    srvr.server_close()
