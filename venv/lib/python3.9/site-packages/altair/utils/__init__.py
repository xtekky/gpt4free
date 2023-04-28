from .core import (
    infer_vegalite_type,
    infer_encoding_types,
    sanitize_dataframe,
    parse_shorthand,
    use_signature,
    update_subtraits,
    update_nested,
    display_traceback,
    SchemaBase,
    Undefined,
)
from .html import spec_to_html
from .plugin_registry import PluginRegistry
from .deprecation import AltairDeprecationWarning


__all__ = (
    "infer_vegalite_type",
    "infer_encoding_types",
    "sanitize_dataframe",
    "spec_to_html",
    "parse_shorthand",
    "use_signature",
    "update_subtraits",
    "update_nested",
    "display_traceback",
    "AltairDeprecationWarning",
    "SchemaBase",
    "Undefined",
    "PluginRegistry",
)
