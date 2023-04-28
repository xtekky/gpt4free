import html
import os
from os.path import realpath, join, dirname
import sys
import time
import warnings
import webbrowser

import jinja2

from ..frontend_semver import DECKGL_SEMVER


def in_jupyter():
    try:
        ip = get_ipython()  # noqa

        return ip.has_trait("kernel")
    except NameError:
        return False


def convert_js_bool(py_bool):
    if type(py_bool) != bool:
        return py_bool
    return "true" if py_bool else "false"


in_google_colab = "google.colab" in sys.modules


TEMPLATES_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "./templates/")
j2_loader = jinja2.FileSystemLoader(TEMPLATES_PATH)
j2_env = jinja2.Environment(loader=j2_loader, trim_blocks=True)
CDN_URL = "https://cdn.jsdelivr.net/npm/@deck.gl/jupyter-widget@{}/dist/index.js".format(DECKGL_SEMVER)


def cdn_picker(offline=False):
    # Support hot-reloading
    dev_port = os.getenv("PYDECK_DEV_PORT")
    if dev_port:
        print("pydeck running in development mode, expecting @deck.gl/jupyter-widget served at {}".format(dev_port))
        return (
            "<script type='text/javascript' src='http://localhost:{dev_port}/dist/index.js'></script>\n"
            "<script type='text/javascript' src='http://localhost:{dev_port}/dist/index.js.map'></script>\n"
        ).format(dev_port=dev_port)
    if offline:
        RELPATH_TO_BUNDLE = "../nbextension/static/index.js"
        with open(join(dirname(__file__), RELPATH_TO_BUNDLE), "r", encoding="utf-8") as file:
            js = file.read()
        return "<script type='text/javascript'>{}</script>".format(js)

    return "<script src='{}'></script>".format(CDN_URL)


def render_json_to_html(
    json_input,
    mapbox_key=None,
    google_maps_key=None,
    tooltip=True,
    css_background_color=None,
    custom_libraries=None,
    configuration=None,
    offline=False,
):
    js = j2_env.get_template("index.j2")
    css = j2_env.get_template("style.j2")
    css_text = css.render(css_background_color=css_background_color)
    html_str = js.render(
        mapbox_key=mapbox_key,
        google_maps_key=google_maps_key,
        json_input=json_input,
        deckgl_jupyter_widget_bundle=cdn_picker(offline=offline),
        tooltip=convert_js_bool(tooltip),
        css_text=css_text,
        custom_libraries=custom_libraries,
        configuration=configuration,
    )
    return html_str


def display_html(filename):
    """Converts HTML into a temporary file and opens it in the system browser."""
    url = "file://{}".format(filename)
    # Hack to prevent blank page
    time.sleep(0.5)
    webbrowser.open(url)


def iframe_with_srcdoc(html_str, width="100%", height=500):
    if isinstance(width, str):
        width = f'"{width}"'
    srcdoc = html.escape(html_str)

    iframe = f"""
        <iframe
            width={width}
            height={height}
            frameborder="0"
            srcdoc="{srcdoc}"
        ></iframe>
    """

    from IPython.display import HTML  # noqa

    with warnings.catch_warnings():
        msg = "Consider using IPython.display.IFrame instead"
        warnings.filterwarnings("ignore", message=msg)
        return HTML(iframe)


def render_for_colab(html_str, iframe_height):
    from IPython.display import HTML, Javascript  # noqa

    js_height_snippet = f"google.colab.output.setIframeHeight({iframe_height}, true, {{minHeight: {iframe_height}}})"
    display(Javascript(js_height_snippet))  # noqa
    display(HTML(html_str))  # noqa


def deck_to_html(
    deck_json,
    mapbox_key=None,
    google_maps_key=None,
    filename=None,
    open_browser=False,
    notebook_display=None,
    css_background_color=None,
    iframe_height=500,
    iframe_width="100%",
    tooltip=True,
    custom_libraries=None,
    configuration=None,
    as_string=False,
    offline=False,
):
    """Converts deck.gl format JSON to an HTML page"""
    html_str = render_json_to_html(
        deck_json,
        mapbox_key=mapbox_key,
        google_maps_key=google_maps_key,
        tooltip=tooltip,
        css_background_color=css_background_color,
        custom_libraries=custom_libraries,
        configuration=configuration,
        offline=offline,
    )

    if filename:
        with open(filename, "w+", encoding="utf-8") as f:
            f.write(html_str)

        if open_browser:
            display_html(realpath(f.name))

    if notebook_display is None:
        notebook_display = in_jupyter()

    if in_google_colab:
        render_for_colab(html_str, iframe_height)
        return None
    elif not filename and as_string:
        return html_str
    elif notebook_display:
        return iframe_with_srcdoc(html_str, iframe_width, iframe_height)
