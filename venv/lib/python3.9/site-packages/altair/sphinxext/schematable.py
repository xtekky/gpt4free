import importlib
import warnings
import re

from docutils.parsers.rst import Directive
from docutils import nodes, utils
from sphinx import addnodes
from recommonmark.parser import CommonMarkParser


def type_description(schema):
    """Return a concise type description for the given schema"""
    if not schema or not isinstance(schema, dict) or schema.keys() == {"description"}:
        return "any"
    elif "$ref" in schema:
        return ":class:`{}`".format(schema["$ref"].split("/")[-1])
    elif "enum" in schema:
        return "[{}]".format(", ".join(repr(s) for s in schema["enum"]))
    elif "type" in schema:
        if isinstance(schema["type"], list):
            return "[{}]".format(", ".join(schema["type"]))
        elif schema["type"] == "array":
            return "array({})".format(type_description(schema.get("items", {})))
        elif schema["type"] == "object":
            return "dict"
        else:
            return "`{}`".format(schema["type"])
    elif "anyOf" in schema:
        return "anyOf({})".format(
            ", ".join(type_description(s) for s in schema["anyOf"])
        )
    else:
        warnings.warn(
            "cannot infer type for schema with keys {}" "".format(schema.keys())
        )
        return "--"


def prepare_table_header(titles, widths):
    """Build docutil empty table"""
    ncols = len(titles)
    assert len(widths) == ncols

    tgroup = nodes.tgroup(cols=ncols)
    for width in widths:
        tgroup += nodes.colspec(colwidth=width)
    header = nodes.row()
    for title in titles:
        header += nodes.entry("", nodes.paragraph(text=title))
    tgroup += nodes.thead("", header)

    tbody = nodes.tbody()
    tgroup += tbody

    return nodes.table("", tgroup), tbody


reClassDef = re.compile(r":class:`([^`]+)`")
reCode = re.compile(r"`([^`]+)`")


def add_class_def(node, classDef):
    """Add reference on classDef to node"""

    ref = addnodes.pending_xref(
        reftarget=classDef,
        reftype="class",
        refdomain="py",  # py:class="None" py:module="altair" refdoc="user_guide/marks"
        refexplicit=False,
        # refdoc="",
        refwarn=False,
    )
    ref["py:class"] = "None"
    ref["py:module"] = "altair"

    ref += nodes.literal(text=classDef, classes=["xref", "py", "py-class"])
    node += ref
    return node


def add_text(node, text):
    """Add text with inline code to node"""
    is_text = True
    for part in reCode.split(text):
        if part:
            if is_text:
                node += nodes.Text(part, part)
            else:
                node += nodes.literal(part, part)

        is_text = not is_text

    return node


def build_row(item):
    """Return nodes.row with property description"""

    prop, propschema, required = item
    row = nodes.row()

    # Property

    row += nodes.entry("", nodes.paragraph(text=prop), classes=["vl-prop"])

    # Type
    str_type = type_description(propschema)
    par_type = nodes.paragraph()

    is_text = True
    for part in reClassDef.split(str_type):
        if part:
            if is_text:
                add_text(par_type, part)
            else:
                add_class_def(par_type, part)
        is_text = not is_text

    # row += nodes.entry('')
    row += nodes.entry("", par_type)  # , classes=["vl-type-def"]

    # Description
    md_parser = CommonMarkParser()
    # str_descr = "***Required.*** " if required else ""
    str_descr = ""
    str_descr += propschema.get("description", " ")
    doc_descr = utils.new_document("schema_description")
    md_parser.parse(str_descr, doc_descr)

    # row += nodes.entry('', *doc_descr.children, classes="vl-decsr")
    row += nodes.entry("", *doc_descr.children, classes=["vl-decsr"])

    return row


def build_schema_tabel(items):
    """Return schema table of items (iterator of prop, schema.item, requred)"""
    table, tbody = prepare_table_header(
        ["Property", "Type", "Description"], [10, 20, 50]
    )
    for item in items:
        tbody += build_row(item)

    return table


def select_items_from_schema(schema, props=None):
    """Return iterator  (prop, schema.item, requred) on prop, return all in None"""
    properties = schema.get("properties", {})
    required = schema.get("required", [])
    if not props:
        for prop, item in properties.items():
            yield prop, item, prop in required
    else:
        for prop in props:
            try:
                yield prop, properties[prop], prop in required
            except KeyError:
                warnings.warn("Can't find property:", prop)


def prepare_schema_tabel(schema, props=None):

    items = select_items_from_schema(schema, props)
    return build_schema_tabel(items)


class AltairObjectTableDirective(Directive):
    """
    Directive for building a table of attribute descriptions.

    Usage:

    .. altair-object-table:: altair.MarkConfig

    """

    has_content = False
    required_arguments = 1

    def run(self):

        objectname = self.arguments[0]
        modname, classname = objectname.rsplit(".", 1)
        module = importlib.import_module(modname)
        cls = getattr(module, classname)
        schema = cls.resolve_references(cls._schema)

        # create the table from the object
        table = prepare_schema_tabel(schema)
        return [table]


def setup(app):
    app.add_directive("altair-object-table", AltairObjectTableDirective)
