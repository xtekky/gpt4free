"""Tools for enabling and registering chart themes"""

from ...utils.theme import ThemeRegistry

VEGA_THEMES = ["ggplot2", "quartz", "vox", "fivethirtyeight", "dark", "latimes"]


class VegaTheme(object):
    """Implementation of a builtin vega theme."""

    def __init__(self, theme):
        self.theme = theme

    def __call__(self):
        return {
            "usermeta": {"embedOptions": {"theme": self.theme}},
            "config": {
                "view": {"width": 400, "height": 300},
                "mark": {"tooltip": None},
            },
        }

    def __repr__(self):
        return "VegaTheme({!r})".format(self.theme)


# The entry point group that can be used by other packages to declare other
# renderers that will be auto-detected. Explicit registration is also
# allowed by the PluginRegistery API.
ENTRY_POINT_GROUP = "altair.vegalite.v3.theme"  # type: str
themes = ThemeRegistry(entry_point_group=ENTRY_POINT_GROUP)

themes.register(
    "default",
    lambda: {
        "config": {"view": {"width": 400, "height": 300}, "mark": {"tooltip": None}}
    },
)
themes.register(
    "opaque",
    lambda: {
        "config": {
            "background": "white",
            "view": {"width": 400, "height": 300},
            "mark": {"tooltip": None},
        }
    },
)
themes.register("none", lambda: {})

for theme in VEGA_THEMES:
    themes.register(theme, VegaTheme(theme))

themes.enable("default")
