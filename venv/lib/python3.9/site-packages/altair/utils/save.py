import json
import pathlib

from .mimebundle import spec_to_mimebundle


def write_file_or_filename(fp, content, mode="w"):
    """Write content to fp, whether fp is a string, a pathlib Path or a
    file-like object"""
    if isinstance(fp, str) or isinstance(fp, pathlib.PurePath):
        with open(fp, mode) as f:
            f.write(content)
    else:
        fp.write(content)


def save(
    chart,
    fp,
    vega_version,
    vegaembed_version,
    format=None,
    mode=None,
    vegalite_version=None,
    embed_options=None,
    json_kwds=None,
    webdriver="chrome",
    scale_factor=1,
    **kwargs,
):
    """Save a chart to file in a variety of formats

    Supported formats are [json, html, png, svg]

    Parameters
    ----------
    chart : alt.Chart
        the chart instance to save
    fp : string filename, pathlib.Path or file-like object
        file to which to write the chart.
    format : string (optional)
        the format to write: one of ['json', 'html', 'png', 'svg'].
        If not specified, the format will be determined from the filename.
    mode : string (optional)
        Either 'vega' or 'vegalite'. If not specified, then infer the mode from
        the '$schema' property of the spec, or the ``opt`` dictionary.
        If it's not specified in either of those places, then use 'vegalite'.
    vega_version : string
        For html output, the version of vega.js to use
    vegalite_version : string
        For html output, the version of vegalite.js to use
    vegaembed_version : string
        For html output, the version of vegaembed.js to use
    embed_options : dict
        The vegaEmbed options dictionary. Default is {}
        (See https://github.com/vega/vega-embed for details)
    json_kwds : dict
        Additional keyword arguments are passed to the output method
        associated with the specified format.
    webdriver : string {'chrome' | 'firefox'}
        Webdriver to use for png or svg output
    scale_factor : float
        scale_factor to use to change size/resolution of png or svg output
    **kwargs :
        additional kwargs passed to spec_to_mimebundle.
    """
    if json_kwds is None:
        json_kwds = {}

    if embed_options is None:
        embed_options = {}

    if format is None:
        if isinstance(fp, str):
            format = fp.split(".")[-1]
        elif isinstance(fp, pathlib.PurePath):
            format = fp.suffix.lstrip(".")
        else:
            raise ValueError(
                "must specify file format: " "['png', 'svg', 'pdf', 'html', 'json']"
            )

    spec = chart.to_dict()

    if mode is None:
        if "mode" in embed_options:
            mode = embed_options["mode"]
        elif "$schema" in spec:
            mode = spec["$schema"].split("/")[-2]
        else:
            mode = "vega-lite"

    if mode not in ["vega", "vega-lite"]:
        raise ValueError("mode must be 'vega' or 'vega-lite', " "not '{}'".format(mode))

    if mode == "vega-lite" and vegalite_version is None:
        raise ValueError("must specify vega-lite version")

    if format == "json":
        json_spec = json.dumps(spec, **json_kwds)
        write_file_or_filename(fp, json_spec, mode="w")
    elif format == "html":
        mimebundle = spec_to_mimebundle(
            spec=spec,
            format=format,
            mode=mode,
            vega_version=vega_version,
            vegalite_version=vegalite_version,
            vegaembed_version=vegaembed_version,
            embed_options=embed_options,
            json_kwds=json_kwds,
            **kwargs,
        )
        write_file_or_filename(fp, mimebundle["text/html"], mode="w")
    elif format in ["png", "svg", "pdf"]:
        mimebundle = spec_to_mimebundle(
            spec=spec,
            format=format,
            mode=mode,
            vega_version=vega_version,
            vegalite_version=vegalite_version,
            vegaembed_version=vegaembed_version,
            webdriver=webdriver,
            scale_factor=scale_factor,
            **kwargs,
        )
        if format == "png":
            write_file_or_filename(fp, mimebundle["image/png"], mode="wb")
        elif format == "pdf":
            write_file_or_filename(fp, mimebundle["application/pdf"], mode="wb")
        else:
            write_file_or_filename(fp, mimebundle["image/svg+xml"], mode="w")
    else:
        raise ValueError("unrecognized format: '{}'".format(format))
