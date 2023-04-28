"""
Altair Plot Sphinx Extension
============================

This extension provides a means of inserting live-rendered Altair plots within
sphinx documentation. There are two directives defined: ``altair-setup`` and
``altair-plot``. ``altair-setup`` code is used to set-up various options
prior to running the plot code. For example::

    .. altair-plot::
        :output: none

        from altair import *
        import pandas as pd
        data = pd.DataFrame({'a': list('CCCDDDEEE'),
                             'b': [2, 7, 4, 1, 2, 6, 8, 4, 7]})

    .. altair-plot::

        Chart(data).mark_point().encode(
            x='a',
            y='b'
        )

In the case of the ``altair-plot`` code, the *last statement* of the code-block
should contain the chart object you wish to be rendered.

Options
-------
The directives have the following options::

    .. altair-plot::
        :namespace:  # specify a plotting namespace that is persistent within the doc
        :hide-code:  # if set, then hide the code and only show the plot
        :code-below:  # if set, then code is below rather than above the figure
        :output:  [plot|repr|stdout|none]
        :alt: text  # Alternate text when plot cannot be rendered
        :links: editor source export  # specify one or more of these options
        :chart-var-name: chart  # name of variable in namespace containing output


Additionally, this extension introduces a global configuration
``altairplot_links``, set in your ``conf.py`` which is a dictionary
of links that will appear below plots, unless the ``:links:`` option
again overrides it. It should look something like this::

    # conf.py
    # ...
    altairplot_links = {'editor': True, 'source': True, 'export': True}
    # ...

If this configuration is not specified, all are set to True.
"""

import contextlib
import io
import os
import json
import warnings

import jinja2

from docutils import nodes
from docutils.parsers.rst import Directive
from docutils.parsers.rst.directives import flag, unchanged

from sphinx.locale import _

import altair as alt
from altair.utils.execeval import eval_block

# These default URLs can be changed in conf.py; see setup() below.
VEGA_JS_URL_DEFAULT = "https://cdn.jsdelivr.net/npm/vega@{}".format(alt.VEGA_VERSION)
VEGALITE_JS_URL_DEFAULT = "https://cdn.jsdelivr.net/npm/vega-lite@{}".format(
    alt.VEGALITE_VERSION
)
VEGAEMBED_JS_URL_DEFAULT = "https://cdn.jsdelivr.net/npm/vega-embed@{}".format(
    alt.VEGAEMBED_VERSION
)


VGL_TEMPLATE = jinja2.Template(
    """
<div id="{{ div_id }}">
<script>
  // embed when document is loaded, to ensure vega library is available
  // this works on all modern browsers, except IE8 and older
  document.addEventListener("DOMContentLoaded", function(event) {
      var spec = {{ spec }};
      var opt = {
        "mode": "{{ mode }}",
        "renderer": "{{ renderer }}",
        "actions": {{ actions}}
      };
      vegaEmbed('#{{ div_id }}', spec, opt).catch(console.err);
  });
</script>
</div>
"""
)


class altair_plot(nodes.General, nodes.Element):
    pass


def purge_altair_namespaces(app, env, docname):
    if not hasattr(env, "_altair_namespaces"):
        return
    env._altair_namespaces.pop(docname, {})


DEFAULT_ALTAIRPLOT_LINKS = {"editor": True, "source": True, "export": True}


def validate_links(links):
    if links.strip().lower() == "none":
        return False

    links = links.strip().split()
    diff = set(links) - set(DEFAULT_ALTAIRPLOT_LINKS.keys())
    if diff:
        raise ValueError("Following links are invalid: {}".format(list(diff)))
    return {link: link in links for link in DEFAULT_ALTAIRPLOT_LINKS}


def validate_output(output):
    output = output.strip().lower()
    if output not in ["plot", "repr", "stdout", "none"]:
        raise ValueError(":output: flag must be one of [plot|repr|stdout|none]")
    return output


class AltairPlotDirective(Directive):
    has_content = True

    option_spec = {
        "hide-code": flag,
        "code-below": flag,
        "namespace": unchanged,
        "output": validate_output,
        "alt": unchanged,
        "links": validate_links,
        "chart-var-name": unchanged,
    }

    def run(self):
        env = self.state.document.settings.env
        app = env.app

        show_code = "hide-code" not in self.options
        code_below = "code-below" in self.options

        if not hasattr(env, "_altair_namespaces"):
            env._altair_namespaces = {}
        namespace_id = self.options.get("namespace", "default")
        namespace = env._altair_namespaces.setdefault(env.docname, {}).setdefault(
            namespace_id, {}
        )

        code = "\n".join(self.content)

        if show_code:
            source_literal = nodes.literal_block(code, code)
            source_literal["language"] = "python"

        # get the name of the source file we are currently processing
        rst_source = self.state_machine.document["source"]
        rst_dir = os.path.dirname(rst_source)
        rst_filename = os.path.basename(rst_source)

        # use the source file name to construct a friendly target_id
        serialno = env.new_serialno("altair-plot")
        rst_base = rst_filename.replace(".", "-")
        div_id = "{}-altair-plot-{}".format(rst_base, serialno)
        target_id = "{}-altair-source-{}".format(rst_base, serialno)
        target_node = nodes.target("", "", ids=[target_id])

        # create the node in which the plot will appear;
        # this will be processed by html_visit_altair_plot
        plot_node = altair_plot()
        plot_node["target_id"] = target_id
        plot_node["div_id"] = div_id
        plot_node["code"] = code
        plot_node["namespace"] = namespace
        plot_node["relpath"] = os.path.relpath(rst_dir, env.srcdir)
        plot_node["rst_source"] = rst_source
        plot_node["rst_lineno"] = self.lineno
        plot_node["links"] = self.options.get(
            "links", app.builder.config.altairplot_links
        )
        plot_node["output"] = self.options.get("output", "plot")
        plot_node["chart-var-name"] = self.options.get("chart-var-name", None)

        if "alt" in self.options:
            plot_node["alt"] = self.options["alt"]

        result = [target_node]

        if code_below:
            result += [plot_node]
        if show_code:
            result += [source_literal]
        if not code_below:
            result += [plot_node]

        return result


def html_visit_altair_plot(self, node):
    # Execute the code, saving output and namespace
    namespace = node["namespace"]
    try:
        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            chart = eval_block(node["code"], namespace)
        stdout = f.getvalue()
    except Exception as e:
        warnings.warn(
            "altair-plot: {}:{} Code Execution failed:"
            "{}: {}".format(
                node["rst_source"], node["rst_lineno"], e.__class__.__name__, str(e)
            )
        )
        raise nodes.SkipNode

    chart_name = node["chart-var-name"]
    if chart_name is not None:
        if chart_name not in namespace:
            raise ValueError(
                "chart-var-name='{}' not present in namespace" "".format(chart_name)
            )
        chart = namespace[chart_name]

    output = node["output"]

    if output == "none":
        raise nodes.SkipNode
    elif output == "stdout":
        if not stdout:
            raise nodes.SkipNode
        else:
            output_literal = nodes.literal_block(stdout, stdout)
            output_literal["language"] = "none"
            node.extend([output_literal])
    elif output == "repr":
        if chart is None:
            raise nodes.SkipNode
        else:
            rep = "    " + repr(chart).replace("\n", "\n    ")
            repr_literal = nodes.literal_block(rep, rep)
            repr_literal["language"] = "none"
            node.extend([repr_literal])
    elif output == "plot":
        if isinstance(chart, alt.TopLevelMixin):
            # Last line should be a chart; convert to spec dict
            try:
                spec = chart.to_dict()
            except alt.utils.schemapi.SchemaValidationError:
                raise ValueError("Invalid chart: {0}".format(node["code"]))
            actions = node["links"]

            # TODO: add an option to save spects to file & load from there.
            # TODO: add renderer option

            # Write spec to a *.vl.json file
            # dest_dir = os.path.join(self.builder.outdir, node['relpath'])
            # if not os.path.exists(dest_dir):
            #     os.makedirs(dest_dir)
            # filename = "{0}.vl.json".format(node['target_id'])
            # dest_path = os.path.join(dest_dir, filename)
            # with open(dest_path, 'w') as f:
            #     json.dump(spec, f)

            # Pass relevant info into the template and append to the output
            html = VGL_TEMPLATE.render(
                div_id=node["div_id"],
                spec=json.dumps(spec),
                mode="vega-lite",
                renderer="canvas",
                actions=json.dumps(actions),
            )
            self.body.append(html)
        else:
            warnings.warn(
                "altair-plot: {}:{} Malformed block. Last line of "
                "code block should define a valid altair Chart object."
                "".format(node["rst_source"], node["rst_lineno"])
            )
        raise nodes.SkipNode


def generic_visit_altair_plot(self, node):
    # TODO: generate PNGs and insert them here
    if "alt" in node.attributes:
        self.body.append(_("[ graph: %s ]") % node["alt"])
    else:
        self.body.append(_("[ graph ]"))
    raise nodes.SkipNode


def depart_altair_plot(self, node):
    return


def builder_inited(app):
    app.add_js_file(app.config.altairplot_vega_js_url)
    app.add_js_file(app.config.altairplot_vegalite_js_url)
    app.add_js_file(app.config.altairplot_vegaembed_js_url)


def setup(app):
    setup.app = app
    setup.config = app.config
    setup.confdir = app.confdir

    app.add_config_value("altairplot_links", DEFAULT_ALTAIRPLOT_LINKS, "env")

    app.add_config_value("altairplot_vega_js_url", VEGA_JS_URL_DEFAULT, "html")
    app.add_config_value("altairplot_vegalite_js_url", VEGALITE_JS_URL_DEFAULT, "html")
    app.add_config_value(
        "altairplot_vegaembed_js_url", VEGAEMBED_JS_URL_DEFAULT, "html"
    )

    app.add_directive("altair-plot", AltairPlotDirective)

    app.add_css_file("altair-plot.css")

    app.add_node(
        altair_plot,
        html=(html_visit_altair_plot, depart_altair_plot),
        latex=(generic_visit_altair_plot, depart_altair_plot),
        texinfo=(generic_visit_altair_plot, depart_altair_plot),
        text=(generic_visit_altair_plot, depart_altair_plot),
        man=(generic_visit_altair_plot, depart_altair_plot),
    )

    app.connect("env-purge-doc", purge_altair_namespaces)
    app.connect("builder-inited", builder_inited)

    return {"version": "0.1"}
