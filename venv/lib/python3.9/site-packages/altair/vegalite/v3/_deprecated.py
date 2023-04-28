from ...utils.deprecation import _deprecate
from . import channels

# Deprecated classes (see https://github.com/altair-viz/altair/issues/1474).
# TODO: Remove these in Altair 3.2.
Fillopacity = _deprecate(channels.FillOpacity, "Fillopacity")
FillopacityValue = _deprecate(channels.FillOpacityValue, "FillopacityValue")
Strokeopacity = _deprecate(channels.StrokeOpacity, "Strokeopacity")
StrokeopacityValue = _deprecate(channels.StrokeOpacityValue, "StrokeopacityValue")
Strokewidth = _deprecate(channels.StrokeWidth, "Strokewidth")
StrokewidthValue = _deprecate(channels.StrokeWidthValue, "StrokewidthValue")
Xerror = _deprecate(channels.XError, "Xerror")
XerrorValue = _deprecate(channels.XErrorValue, "XerrorValue")
Xerror2 = _deprecate(channels.XError2, "Xerror2")
Xerror2Value = _deprecate(channels.XError2Value, "Xerror2Value")
Yerror = _deprecate(channels.YError, "Yerror")
YerrorValue = _deprecate(channels.YErrorValue, "YerrorValue")
Yerror2 = _deprecate(channels.YError2, "Yerror2")
Yerror2Value = _deprecate(channels.YError2Value, "Yerror2Value")
