# The contents of this file are automatically written by
# tools/generate_schema_wrapper.py. Do not modify directly.

from altair.utils.schemapi import SchemaBase, Undefined, _subclasses

import pkgutil
import json

def load_schema():
    """Load the json schema associated with this module's functions"""
    return json.loads(pkgutil.get_data(__name__, 'vega-lite-schema.json').decode('utf-8'))


class VegaLiteSchema(SchemaBase):
    _rootschema = load_schema()
    @classmethod
    def _default_wrapper_classes(cls):
        return _subclasses(VegaLiteSchema)


class Root(VegaLiteSchema):
    """Root schema wrapper

    anyOf(:class:`TopLevelUnitSpec`, :class:`TopLevelFacetSpec`, :class:`TopLevelLayerSpec`,
    :class:`TopLevelRepeatSpec`, :class:`TopLevelNormalizedConcatSpecGenericSpec`,
    :class:`TopLevelNormalizedVConcatSpecGenericSpec`,
    :class:`TopLevelNormalizedHConcatSpecGenericSpec`)
    A Vega-Lite top-level specification. This is the root class for all Vega-Lite
    specifications. (The json schema is generated from this type.)
    """
    _schema = VegaLiteSchema._rootschema

    def __init__(self, *args, **kwds):
        super(Root, self).__init__(*args, **kwds)


class Aggregate(VegaLiteSchema):
    """Aggregate schema wrapper

    anyOf(:class:`NonArgAggregateOp`, :class:`ArgmaxDef`, :class:`ArgminDef`)
    """
    _schema = {'$ref': '#/definitions/Aggregate'}

    def __init__(self, *args, **kwds):
        super(Aggregate, self).__init__(*args, **kwds)


class AggregateOp(VegaLiteSchema):
    """AggregateOp schema wrapper

    enum('argmax', 'argmin', 'average', 'count', 'distinct', 'max', 'mean', 'median', 'min',
    'missing', 'product', 'q1', 'q3', 'ci0', 'ci1', 'stderr', 'stdev', 'stdevp', 'sum', 'valid',
    'values', 'variance', 'variancep')
    """
    _schema = {'$ref': '#/definitions/AggregateOp'}

    def __init__(self, *args):
        super(AggregateOp, self).__init__(*args)


class AggregatedFieldDef(VegaLiteSchema):
    """AggregatedFieldDef schema wrapper

    Mapping(required=[op, as])

    Attributes
    ----------

    op : :class:`AggregateOp`
        The aggregation operation to apply to the fields (e.g., ``"sum"``, ``"average"``, or
        ``"count"`` ). See the `full list of supported aggregation operations
        <https://vega.github.io/vega-lite/docs/aggregate.html#ops>`__ for more information.
    field : :class:`FieldName`
        The data field for which to compute aggregate function. This is required for all
        aggregation operations except ``"count"``.
    as : :class:`FieldName`
        The output field names to use for each aggregated field.
    """
    _schema = {'$ref': '#/definitions/AggregatedFieldDef'}

    def __init__(self, op=Undefined, field=Undefined, **kwds):
        super(AggregatedFieldDef, self).__init__(op=op, field=field, **kwds)


class Align(VegaLiteSchema):
    """Align schema wrapper

    enum('left', 'center', 'right')
    """
    _schema = {'$ref': '#/definitions/Align'}

    def __init__(self, *args):
        super(Align, self).__init__(*args)


class AnyMark(VegaLiteSchema):
    """AnyMark schema wrapper

    anyOf(:class:`CompositeMark`, :class:`CompositeMarkDef`, :class:`Mark`, :class:`MarkDef`)
    """
    _schema = {'$ref': '#/definitions/AnyMark'}

    def __init__(self, *args, **kwds):
        super(AnyMark, self).__init__(*args, **kwds)


class AnyMarkConfig(VegaLiteSchema):
    """AnyMarkConfig schema wrapper

    anyOf(:class:`MarkConfig`, :class:`AreaConfig`, :class:`BarConfig`, :class:`RectConfig`,
    :class:`LineConfig`, :class:`TickConfig`)
    """
    _schema = {'$ref': '#/definitions/AnyMarkConfig'}

    def __init__(self, *args, **kwds):
        super(AnyMarkConfig, self).__init__(*args, **kwds)


class AreaConfig(AnyMarkConfig):
    """AreaConfig schema wrapper

    Mapping(required=[])

    Attributes
    ----------

    align : anyOf(:class:`Align`, :class:`ExprRef`)
        The horizontal alignment of the text or ranged marks (area, bar, image, rect, rule).
        One of ``"left"``, ``"right"``, ``"center"``.

        **Note:** Expression reference is *not* supported for range marks.
    angle : anyOf(float, :class:`ExprRef`)

    aria : anyOf(boolean, :class:`ExprRef`)

    ariaRole : anyOf(string, :class:`ExprRef`)

    ariaRoleDescription : anyOf(string, :class:`ExprRef`)

    aspect : anyOf(boolean, :class:`ExprRef`)

    baseline : anyOf(:class:`TextBaseline`, :class:`ExprRef`)
        For text marks, the vertical text baseline. One of ``"alphabetic"`` (default),
        ``"top"``, ``"middle"``, ``"bottom"``, ``"line-top"``, ``"line-bottom"``, or an
        expression reference that provides one of the valid values. The ``"line-top"`` and
        ``"line-bottom"`` values operate similarly to ``"top"`` and ``"bottom"``, but are
        calculated relative to the ``lineHeight`` rather than ``fontSize`` alone.

        For range marks, the vertical alignment of the marks. One of ``"top"``,
        ``"middle"``, ``"bottom"``.

        **Note:** Expression reference is *not* supported for range marks.
    blend : anyOf(:class:`Blend`, :class:`ExprRef`)

    color : anyOf(:class:`Color`, :class:`Gradient`, :class:`ExprRef`)
        Default color.

        **Default value:** :raw-html:`<span style="color: #4682b4;">&#9632;</span>`
        ``"#4682b4"``

        **Note:** - This property cannot be used in a `style config
        <https://vega.github.io/vega-lite/docs/mark.html#style-config>`__. - The ``fill``
        and ``stroke`` properties have higher precedence than ``color`` and will override
        ``color``.
    cornerRadius : anyOf(float, :class:`ExprRef`)

    cornerRadiusBottomLeft : anyOf(float, :class:`ExprRef`)

    cornerRadiusBottomRight : anyOf(float, :class:`ExprRef`)

    cornerRadiusTopLeft : anyOf(float, :class:`ExprRef`)

    cornerRadiusTopRight : anyOf(float, :class:`ExprRef`)

    cursor : anyOf(:class:`Cursor`, :class:`ExprRef`)

    description : anyOf(string, :class:`ExprRef`)

    dir : anyOf(:class:`TextDirection`, :class:`ExprRef`)

    dx : anyOf(float, :class:`ExprRef`)

    dy : anyOf(float, :class:`ExprRef`)

    ellipsis : anyOf(string, :class:`ExprRef`)

    endAngle : anyOf(float, :class:`ExprRef`)

    fill : anyOf(:class:`Color`, :class:`Gradient`, None, :class:`ExprRef`)
        Default fill color. This property has higher precedence than ``config.color``. Set
        to ``null`` to remove fill.

        **Default value:** (None)
    fillOpacity : anyOf(float, :class:`ExprRef`)

    filled : boolean
        Whether the mark's color should be used as fill color instead of stroke color.

        **Default value:** ``false`` for all ``point``, ``line``, and ``rule`` marks as well
        as ``geoshape`` marks for `graticule
        <https://vega.github.io/vega-lite/docs/data.html#graticule>`__ data sources;
        otherwise, ``true``.

        **Note:** This property cannot be used in a `style config
        <https://vega.github.io/vega-lite/docs/mark.html#style-config>`__.
    font : anyOf(string, :class:`ExprRef`)

    fontSize : anyOf(float, :class:`ExprRef`)

    fontStyle : anyOf(:class:`FontStyle`, :class:`ExprRef`)

    fontWeight : anyOf(:class:`FontWeight`, :class:`ExprRef`)

    height : anyOf(float, :class:`ExprRef`)

    href : anyOf(:class:`URI`, :class:`ExprRef`)

    innerRadius : anyOf(float, :class:`ExprRef`)
        The inner radius in pixels of arc marks. ``innerRadius`` is an alias for
        ``radius2``.
    interpolate : anyOf(:class:`Interpolate`, :class:`ExprRef`)

    invalid : enum('filter', None)
        Defines how Vega-Lite should handle marks for invalid values ( ``null`` and ``NaN``
        ). - If set to ``"filter"`` (default), all data items with null values will be
        skipped (for line, trail, and area marks) or filtered (for other marks). - If
        ``null``, all data items are included. In this case, invalid values will be
        interpreted as zeroes.
    limit : anyOf(float, :class:`ExprRef`)

    line : anyOf(boolean, :class:`OverlayMarkDef`)
        A flag for overlaying line on top of area marks, or an object defining the
        properties of the overlayed lines.


        If this value is an empty object ( ``{}`` ) or ``true``, lines with default
        properties will be used.

        If this value is ``false``, no lines would be automatically added to area marks.

        **Default value:** ``false``.
    lineBreak : anyOf(string, :class:`ExprRef`)

    lineHeight : anyOf(float, :class:`ExprRef`)

    opacity : anyOf(float, :class:`ExprRef`)
        The overall opacity (value between [0,1]).

        **Default value:** ``0.7`` for non-aggregate plots with ``point``, ``tick``,
        ``circle``, or ``square`` marks or layered ``bar`` charts and ``1`` otherwise.
    order : anyOf(None, boolean)
        For line and trail marks, this ``order`` property can be set to ``null`` or
        ``false`` to make the lines use the original order in the data sources.
    orient : :class:`Orientation`
        The orientation of a non-stacked bar, tick, area, and line charts. The value is
        either horizontal (default) or vertical. - For bar, rule and tick, this determines
        whether the size of the bar and tick should be applied to x or y dimension. - For
        area, this property determines the orient property of the Vega output. - For line
        and trail marks, this property determines the sort order of the points in the line
        if ``config.sortLineBy`` is not specified. For stacked charts, this is always
        determined by the orientation of the stack; therefore explicitly specified value
        will be ignored.
    outerRadius : anyOf(float, :class:`ExprRef`)
        The outer radius in pixels of arc marks. ``outerRadius`` is an alias for ``radius``.
    padAngle : anyOf(float, :class:`ExprRef`)

    point : anyOf(boolean, :class:`OverlayMarkDef`, string)
        A flag for overlaying points on top of line or area marks, or an object defining the
        properties of the overlayed points.


        If this property is ``"transparent"``, transparent points will be used (for
        enhancing tooltips and selections).

        If this property is an empty object ( ``{}`` ) or ``true``, filled points with
        default properties will be used.

        If this property is ``false``, no points would be automatically added to line or
        area marks.

        **Default value:** ``false``.
    radius : anyOf(float, :class:`ExprRef`)
        For arc mark, the primary (outer) radius in pixels.

        For text marks, polar coordinate radial offset, in pixels, of the text from the
        origin determined by the ``x`` and ``y`` properties.
    radius2 : anyOf(float, :class:`ExprRef`)
        The secondary (inner) radius in pixels of arc marks.
    shape : anyOf(anyOf(:class:`SymbolShape`, string), :class:`ExprRef`)

    size : anyOf(float, :class:`ExprRef`)
        Default size for marks. - For ``point`` / ``circle`` / ``square``, this represents
        the pixel area of the marks. Note that this value sets the area of the symbol; the
        side lengths will increase with the square root of this value. - For ``bar``, this
        represents the band size of the bar, in pixels. - For ``text``, this represents the
        font size, in pixels.

        **Default value:** - ``30`` for point, circle, square marks; width/height's ``step``
        - ``2`` for bar marks with discrete dimensions; - ``5`` for bar marks with
        continuous dimensions; - ``11`` for text marks.
    smooth : anyOf(boolean, :class:`ExprRef`)

    startAngle : anyOf(float, :class:`ExprRef`)

    stroke : anyOf(:class:`Color`, :class:`Gradient`, None, :class:`ExprRef`)
        Default stroke color. This property has higher precedence than ``config.color``. Set
        to ``null`` to remove stroke.

        **Default value:** (None)
    strokeCap : anyOf(:class:`StrokeCap`, :class:`ExprRef`)

    strokeDash : anyOf(List(float), :class:`ExprRef`)

    strokeDashOffset : anyOf(float, :class:`ExprRef`)

    strokeJoin : anyOf(:class:`StrokeJoin`, :class:`ExprRef`)

    strokeMiterLimit : anyOf(float, :class:`ExprRef`)

    strokeOffset : anyOf(float, :class:`ExprRef`)

    strokeOpacity : anyOf(float, :class:`ExprRef`)

    strokeWidth : anyOf(float, :class:`ExprRef`)

    tension : anyOf(float, :class:`ExprRef`)

    text : anyOf(:class:`Text`, :class:`ExprRef`)

    theta : anyOf(float, :class:`ExprRef`)
        For arc marks, the arc length in radians if theta2 is not specified, otherwise the
        start arc angle. (A value of 0 indicates up or “north”, increasing values proceed
        clockwise.)

        For text marks, polar coordinate angle in radians.
    theta2 : anyOf(float, :class:`ExprRef`)
        The end angle of arc marks in radians. A value of 0 indicates up or “north”,
        increasing values proceed clockwise.
    timeUnitBand : float
        Default relative band size for a time unit. If set to ``1``, the bandwidth of the
        marks will be equal to the time unit band step. If set to ``0.5``, bandwidth of the
        marks will be half of the time unit band step.
    timeUnitBandPosition : float
        Default relative band position for a time unit. If set to ``0``, the marks will be
        positioned at the beginning of the time unit band step. If set to ``0.5``, the marks
        will be positioned in the middle of the time unit band step.
    tooltip : anyOf(float, string, boolean, :class:`TooltipContent`, :class:`ExprRef`, None)
        The tooltip text string to show upon mouse hover or an object defining which fields
        should the tooltip be derived from.


        * If ``tooltip`` is ``true`` or ``{"content": "encoding"}``, then all fields from
          ``encoding`` will be used. - If ``tooltip`` is ``{"content": "data"}``, then all
          fields that appear in the highlighted data point will be used. - If set to
          ``null`` or ``false``, then no tooltip will be used.

        See the `tooltip <https://vega.github.io/vega-lite/docs/tooltip.html>`__
        documentation for a detailed discussion about tooltip  in Vega-Lite.

        **Default value:** ``null``
    url : anyOf(:class:`URI`, :class:`ExprRef`)

    width : anyOf(float, :class:`ExprRef`)

    x : anyOf(float, string, :class:`ExprRef`)
        X coordinates of the marks, or width of horizontal ``"bar"`` and ``"area"`` without
        specified ``x2`` or ``width``.

        The ``value`` of this channel can be a number or a string ``"width"`` for the width
        of the plot.
    x2 : anyOf(float, string, :class:`ExprRef`)
        X2 coordinates for ranged ``"area"``, ``"bar"``, ``"rect"``, and  ``"rule"``.

        The ``value`` of this channel can be a number or a string ``"width"`` for the width
        of the plot.
    y : anyOf(float, string, :class:`ExprRef`)
        Y coordinates of the marks, or height of vertical ``"bar"`` and ``"area"`` without
        specified ``y2`` or ``height``.

        The ``value`` of this channel can be a number or a string ``"height"`` for the
        height of the plot.
    y2 : anyOf(float, string, :class:`ExprRef`)
        Y2 coordinates for ranged ``"area"``, ``"bar"``, ``"rect"``, and  ``"rule"``.

        The ``value`` of this channel can be a number or a string ``"height"`` for the
        height of the plot.
    """
    _schema = {'$ref': '#/definitions/AreaConfig'}

    def __init__(self, align=Undefined, angle=Undefined, aria=Undefined, ariaRole=Undefined,
                 ariaRoleDescription=Undefined, aspect=Undefined, baseline=Undefined, blend=Undefined,
                 color=Undefined, cornerRadius=Undefined, cornerRadiusBottomLeft=Undefined,
                 cornerRadiusBottomRight=Undefined, cornerRadiusTopLeft=Undefined,
                 cornerRadiusTopRight=Undefined, cursor=Undefined, description=Undefined, dir=Undefined,
                 dx=Undefined, dy=Undefined, ellipsis=Undefined, endAngle=Undefined, fill=Undefined,
                 fillOpacity=Undefined, filled=Undefined, font=Undefined, fontSize=Undefined,
                 fontStyle=Undefined, fontWeight=Undefined, height=Undefined, href=Undefined,
                 innerRadius=Undefined, interpolate=Undefined, invalid=Undefined, limit=Undefined,
                 line=Undefined, lineBreak=Undefined, lineHeight=Undefined, opacity=Undefined,
                 order=Undefined, orient=Undefined, outerRadius=Undefined, padAngle=Undefined,
                 point=Undefined, radius=Undefined, radius2=Undefined, shape=Undefined, size=Undefined,
                 smooth=Undefined, startAngle=Undefined, stroke=Undefined, strokeCap=Undefined,
                 strokeDash=Undefined, strokeDashOffset=Undefined, strokeJoin=Undefined,
                 strokeMiterLimit=Undefined, strokeOffset=Undefined, strokeOpacity=Undefined,
                 strokeWidth=Undefined, tension=Undefined, text=Undefined, theta=Undefined,
                 theta2=Undefined, timeUnitBand=Undefined, timeUnitBandPosition=Undefined,
                 tooltip=Undefined, url=Undefined, width=Undefined, x=Undefined, x2=Undefined,
                 y=Undefined, y2=Undefined, **kwds):
        super(AreaConfig, self).__init__(align=align, angle=angle, aria=aria, ariaRole=ariaRole,
                                         ariaRoleDescription=ariaRoleDescription, aspect=aspect,
                                         baseline=baseline, blend=blend, color=color,
                                         cornerRadius=cornerRadius,
                                         cornerRadiusBottomLeft=cornerRadiusBottomLeft,
                                         cornerRadiusBottomRight=cornerRadiusBottomRight,
                                         cornerRadiusTopLeft=cornerRadiusTopLeft,
                                         cornerRadiusTopRight=cornerRadiusTopRight, cursor=cursor,
                                         description=description, dir=dir, dx=dx, dy=dy,
                                         ellipsis=ellipsis, endAngle=endAngle, fill=fill,
                                         fillOpacity=fillOpacity, filled=filled, font=font,
                                         fontSize=fontSize, fontStyle=fontStyle, fontWeight=fontWeight,
                                         height=height, href=href, innerRadius=innerRadius,
                                         interpolate=interpolate, invalid=invalid, limit=limit,
                                         line=line, lineBreak=lineBreak, lineHeight=lineHeight,
                                         opacity=opacity, order=order, orient=orient,
                                         outerRadius=outerRadius, padAngle=padAngle, point=point,
                                         radius=radius, radius2=radius2, shape=shape, size=size,
                                         smooth=smooth, startAngle=startAngle, stroke=stroke,
                                         strokeCap=strokeCap, strokeDash=strokeDash,
                                         strokeDashOffset=strokeDashOffset, strokeJoin=strokeJoin,
                                         strokeMiterLimit=strokeMiterLimit, strokeOffset=strokeOffset,
                                         strokeOpacity=strokeOpacity, strokeWidth=strokeWidth,
                                         tension=tension, text=text, theta=theta, theta2=theta2,
                                         timeUnitBand=timeUnitBand,
                                         timeUnitBandPosition=timeUnitBandPosition, tooltip=tooltip,
                                         url=url, width=width, x=x, x2=x2, y=y, y2=y2, **kwds)


class ArgmaxDef(Aggregate):
    """ArgmaxDef schema wrapper

    Mapping(required=[argmax])

    Attributes
    ----------

    argmax : string

    """
    _schema = {'$ref': '#/definitions/ArgmaxDef'}

    def __init__(self, argmax=Undefined, **kwds):
        super(ArgmaxDef, self).__init__(argmax=argmax, **kwds)


class ArgminDef(Aggregate):
    """ArgminDef schema wrapper

    Mapping(required=[argmin])

    Attributes
    ----------

    argmin : string

    """
    _schema = {'$ref': '#/definitions/ArgminDef'}

    def __init__(self, argmin=Undefined, **kwds):
        super(ArgminDef, self).__init__(argmin=argmin, **kwds)


class AutoSizeParams(VegaLiteSchema):
    """AutoSizeParams schema wrapper

    Mapping(required=[])

    Attributes
    ----------

    contains : enum('content', 'padding')
        Determines how size calculation should be performed, one of ``"content"`` or
        ``"padding"``. The default setting ( ``"content"`` ) interprets the width and height
        settings as the data rectangle (plotting) dimensions, to which padding is then
        added. In contrast, the ``"padding"`` setting includes the padding within the view
        size calculations, such that the width and height settings indicate the **total**
        intended size of the view.

        **Default value** : ``"content"``
    resize : boolean
        A boolean flag indicating if autosize layout should be re-calculated on every view
        update.

        **Default value** : ``false``
    type : :class:`AutosizeType`
        The sizing format type. One of ``"pad"``, ``"fit"``, ``"fit-x"``, ``"fit-y"``,  or
        ``"none"``. See the `autosize type
        <https://vega.github.io/vega-lite/docs/size.html#autosize>`__ documentation for
        descriptions of each.

        **Default value** : ``"pad"``
    """
    _schema = {'$ref': '#/definitions/AutoSizeParams'}

    def __init__(self, contains=Undefined, resize=Undefined, type=Undefined, **kwds):
        super(AutoSizeParams, self).__init__(contains=contains, resize=resize, type=type, **kwds)


class AutosizeType(VegaLiteSchema):
    """AutosizeType schema wrapper

    enum('pad', 'none', 'fit', 'fit-x', 'fit-y')
    """
    _schema = {'$ref': '#/definitions/AutosizeType'}

    def __init__(self, *args):
        super(AutosizeType, self).__init__(*args)


class Axis(VegaLiteSchema):
    """Axis schema wrapper

    Mapping(required=[])

    Attributes
    ----------

    aria : anyOf(boolean, :class:`ExprRef`)

    bandPosition : anyOf(float, :class:`ExprRef`)

    description : anyOf(string, :class:`ExprRef`)

    domain : anyOf(boolean, :class:`ExprRef`)

    domainCap : anyOf(:class:`StrokeCap`, :class:`ExprRef`)

    domainColor : anyOf(anyOf(None, :class:`Color`), :class:`ExprRef`)

    domainDash : anyOf(List(float), :class:`ExprRef`)

    domainDashOffset : anyOf(float, :class:`ExprRef`)

    domainOpacity : anyOf(float, :class:`ExprRef`)

    domainWidth : anyOf(float, :class:`ExprRef`)

    format : anyOf(string, :class:`Dictunknown`)
        When used with the default ``"number"`` and ``"time"`` format type, the text
        formatting pattern for labels of guides (axes, legends, headers) and text marks.


        * If the format type is ``"number"`` (e.g., for quantitative fields), this is D3's
          `number format pattern <https://github.com/d3/d3-format#locale_format>`__. - If
          the format type is ``"time"`` (e.g., for temporal fields), this is D3's `time
          format pattern <https://github.com/d3/d3-time-format#locale_format>`__.

        See the `format documentation <https://vega.github.io/vega-lite/docs/format.html>`__
        for more examples.

        When used with a `custom formatType
        <https://vega.github.io/vega-lite/docs/config.html#custom-format-type>`__, this
        value will be passed as ``format`` alongside ``datum.value`` to the registered
        function.

        **Default value:**  Derived from `numberFormat
        <https://vega.github.io/vega-lite/docs/config.html#format>`__ config for number
        format and from `timeFormat
        <https://vega.github.io/vega-lite/docs/config.html#format>`__ config for time
        format.
    formatType : string
        The format type for labels. One of ``"number"``, ``"time"``, or a `registered custom
        format type
        <https://vega.github.io/vega-lite/docs/config.html#custom-format-type>`__.

        **Default value:** - ``"time"`` for temporal fields and ordinal and nominal fields
        with ``timeUnit``. - ``"number"`` for quantitative fields as well as ordinal and
        nominal fields without ``timeUnit``.
    grid : boolean
        A boolean flag indicating if grid lines should be included as part of the axis

        **Default value:** ``true`` for `continuous scales
        <https://vega.github.io/vega-lite/docs/scale.html#continuous>`__ that are not
        binned; otherwise, ``false``.
    gridCap : anyOf(:class:`StrokeCap`, :class:`ExprRef`)

    gridColor : anyOf(anyOf(None, :class:`Color`), :class:`ExprRef`,
    :class:`ConditionalAxisColor`)

    gridDash : anyOf(List(float), :class:`ExprRef`, :class:`ConditionalAxisNumberArray`)

    gridDashOffset : anyOf(float, :class:`ExprRef`, :class:`ConditionalAxisNumber`)

    gridOpacity : anyOf(float, :class:`ExprRef`, :class:`ConditionalAxisNumber`)

    gridWidth : anyOf(float, :class:`ExprRef`, :class:`ConditionalAxisNumber`)

    labelAlign : anyOf(:class:`Align`, :class:`ExprRef`, :class:`ConditionalAxisLabelAlign`)

    labelAngle : anyOf(float, :class:`ExprRef`)

    labelBaseline : anyOf(:class:`TextBaseline`, :class:`ExprRef`,
    :class:`ConditionalAxisLabelBaseline`)

    labelBound : anyOf(anyOf(float, boolean), :class:`ExprRef`)

    labelColor : anyOf(anyOf(None, :class:`Color`), :class:`ExprRef`,
    :class:`ConditionalAxisColor`)

    labelExpr : string
        `Vega expression <https://vega.github.io/vega/docs/expressions/>`__ for customizing
        labels.

        **Note:** The label text and value can be assessed via the ``label`` and ``value``
        properties of the axis's backing ``datum`` object.
    labelFlush : anyOf(boolean, float)
        Indicates if the first and last axis labels should be aligned flush with the scale
        range. Flush alignment for a horizontal axis will left-align the first label and
        right-align the last label. For vertical axes, bottom and top text baselines are
        applied instead. If this property is a number, it also indicates the number of
        pixels by which to offset the first and last labels; for example, a value of 2 will
        flush-align the first and last labels and also push them 2 pixels outward from the
        center of the axis. The additional adjustment can sometimes help the labels better
        visually group with corresponding axis ticks.

        **Default value:** ``true`` for axis of a continuous x-scale. Otherwise, ``false``.
    labelFlushOffset : anyOf(float, :class:`ExprRef`)

    labelFont : anyOf(string, :class:`ExprRef`, :class:`ConditionalAxisString`)

    labelFontSize : anyOf(float, :class:`ExprRef`, :class:`ConditionalAxisNumber`)

    labelFontStyle : anyOf(:class:`FontStyle`, :class:`ExprRef`,
    :class:`ConditionalAxisLabelFontStyle`)

    labelFontWeight : anyOf(:class:`FontWeight`, :class:`ExprRef`,
    :class:`ConditionalAxisLabelFontWeight`)

    labelLimit : anyOf(float, :class:`ExprRef`)

    labelLineHeight : anyOf(float, :class:`ExprRef`)

    labelOffset : anyOf(float, :class:`ExprRef`, :class:`ConditionalAxisNumber`)

    labelOpacity : anyOf(float, :class:`ExprRef`, :class:`ConditionalAxisNumber`)

    labelOverlap : anyOf(:class:`LabelOverlap`, :class:`ExprRef`)
        The strategy to use for resolving overlap of axis labels. If ``false`` (the
        default), no overlap reduction is attempted. If set to ``true`` or ``"parity"``, a
        strategy of removing every other label is used (this works well for standard linear
        axes). If set to ``"greedy"``, a linear scan of the labels is performed, removing
        any labels that overlaps with the last visible label (this often works better for
        log-scaled axes).

        **Default value:** ``true`` for non-nominal fields with non-log scales; ``"greedy"``
        for log scales; otherwise ``false``.
    labelPadding : anyOf(float, :class:`ExprRef`, :class:`ConditionalAxisNumber`)

    labelSeparation : anyOf(float, :class:`ExprRef`)

    labels : anyOf(boolean, :class:`ExprRef`)

    maxExtent : anyOf(float, :class:`ExprRef`)

    minExtent : anyOf(float, :class:`ExprRef`)

    offset : float
        The offset, in pixels, by which to displace the axis from the edge of the enclosing
        group or data rectangle.

        **Default value:** derived from the `axis config
        <https://vega.github.io/vega-lite/docs/config.html#facet-scale-config>`__ 's
        ``offset`` ( ``0`` by default)
    orient : anyOf(:class:`AxisOrient`, :class:`ExprRef`)
        The orientation of the axis. One of ``"top"``, ``"bottom"``, ``"left"`` or
        ``"right"``. The orientation can be used to further specialize the axis type (e.g.,
        a y-axis oriented towards the right edge of the chart).

        **Default value:** ``"bottom"`` for x-axes and ``"left"`` for y-axes.
    position : anyOf(float, :class:`ExprRef`)
        The anchor position of the axis in pixels. For x-axes with top or bottom
        orientation, this sets the axis group x coordinate. For y-axes with left or right
        orientation, this sets the axis group y coordinate.

        **Default value** : ``0``
    style : anyOf(string, List(string))
        A string or array of strings indicating the name of custom styles to apply to the
        axis. A style is a named collection of axis property defined within the `style
        configuration <https://vega.github.io/vega-lite/docs/mark.html#style-config>`__. If
        style is an array, later styles will override earlier styles.

        **Default value:** (none) **Note:** Any specified style will augment the default
        style. For example, an x-axis mark with ``"style": "foo"`` will use ``config.axisX``
        and ``config.style.foo`` (the specified style ``"foo"`` has higher precedence).
    tickBand : anyOf(enum('center', 'extent'), :class:`ExprRef`)

    tickCap : anyOf(:class:`StrokeCap`, :class:`ExprRef`)

    tickColor : anyOf(anyOf(None, :class:`Color`), :class:`ExprRef`,
    :class:`ConditionalAxisColor`)

    tickCount : anyOf(float, :class:`TimeInterval`, :class:`TimeIntervalStep`, :class:`ExprRef`)
        A desired number of ticks, for axes visualizing quantitative scales. The resulting
        number may be different so that values are "nice" (multiples of 2, 5, 10) and lie
        within the underlying scale's range.

        For scales of type ``"time"`` or ``"utc"``, the tick count can instead be a time
        interval specifier. Legal string values are ``"millisecond"``, ``"second"``,
        ``"minute"``, ``"hour"``, ``"day"``, ``"week"``, ``"month"``, and ``"year"``.
        Alternatively, an object-valued interval specifier of the form ``{"interval":
        "month", "step": 3}`` includes a desired number of interval steps. Here, ticks are
        generated for each quarter (Jan, Apr, Jul, Oct) boundary.

        **Default value** : Determine using a formula ``ceil(width/40)`` for x and
        ``ceil(height/40)`` for y.
    tickDash : anyOf(List(float), :class:`ExprRef`, :class:`ConditionalAxisNumberArray`)

    tickDashOffset : anyOf(float, :class:`ExprRef`, :class:`ConditionalAxisNumber`)

    tickExtra : anyOf(boolean, :class:`ExprRef`)

    tickMinStep : anyOf(float, :class:`ExprRef`)
        The minimum desired step between axis ticks, in terms of scale domain values. For
        example, a value of ``1`` indicates that ticks should not be less than 1 unit apart.
        If ``tickMinStep`` is specified, the ``tickCount`` value will be adjusted, if
        necessary, to enforce the minimum step value.
    tickOffset : anyOf(float, :class:`ExprRef`)

    tickOpacity : anyOf(float, :class:`ExprRef`, :class:`ConditionalAxisNumber`)

    tickRound : anyOf(boolean, :class:`ExprRef`)

    tickSize : anyOf(float, :class:`ExprRef`, :class:`ConditionalAxisNumber`)

    tickWidth : anyOf(float, :class:`ExprRef`, :class:`ConditionalAxisNumber`)

    ticks : anyOf(boolean, :class:`ExprRef`)

    title : anyOf(:class:`Text`, None)
        A title for the field. If ``null``, the title will be removed.

        **Default value:**  derived from the field's name and transformation function (
        ``aggregate``, ``bin`` and ``timeUnit`` ). If the field has an aggregate function,
        the function is displayed as part of the title (e.g., ``"Sum of Profit"`` ). If the
        field is binned or has a time unit applied, the applied function is shown in
        parentheses (e.g., ``"Profit (binned)"``, ``"Transaction Date (year-month)"`` ).
        Otherwise, the title is simply the field name.

        **Notes** :

        1) You can customize the default field title format by providing the `fieldTitle
        <https://vega.github.io/vega-lite/docs/config.html#top-level-config>`__ property in
        the `config <https://vega.github.io/vega-lite/docs/config.html>`__ or `fieldTitle
        function via the compile function's options
        <https://vega.github.io/vega-lite/docs/compile.html#field-title>`__.

        2) If both field definition's ``title`` and axis, header, or legend ``title`` are
        defined, axis/header/legend title will be used.
    titleAlign : anyOf(:class:`Align`, :class:`ExprRef`)

    titleAnchor : anyOf(:class:`TitleAnchor`, :class:`ExprRef`)

    titleAngle : anyOf(float, :class:`ExprRef`)

    titleBaseline : anyOf(:class:`TextBaseline`, :class:`ExprRef`)

    titleColor : anyOf(anyOf(None, :class:`Color`), :class:`ExprRef`)

    titleFont : anyOf(string, :class:`ExprRef`)

    titleFontSize : anyOf(float, :class:`ExprRef`)

    titleFontStyle : anyOf(:class:`FontStyle`, :class:`ExprRef`)

    titleFontWeight : anyOf(:class:`FontWeight`, :class:`ExprRef`)

    titleLimit : anyOf(float, :class:`ExprRef`)

    titleLineHeight : anyOf(float, :class:`ExprRef`)

    titleOpacity : anyOf(float, :class:`ExprRef`)

    titlePadding : anyOf(float, :class:`ExprRef`)

    titleX : anyOf(float, :class:`ExprRef`)

    titleY : anyOf(float, :class:`ExprRef`)

    translate : anyOf(float, :class:`ExprRef`)

    values : anyOf(List(float), List(string), List(boolean), List(:class:`DateTime`),
    :class:`ExprRef`)
        Explicitly set the visible axis tick values.
    zindex : float
        A non-negative integer indicating the z-index of the axis. If zindex is 0, axes
        should be drawn behind all chart elements. To put them in front, set ``zindex`` to
        ``1`` or more.

        **Default value:** ``0`` (behind the marks).
    """
    _schema = {'$ref': '#/definitions/Axis'}

    def __init__(self, aria=Undefined, bandPosition=Undefined, description=Undefined, domain=Undefined,
                 domainCap=Undefined, domainColor=Undefined, domainDash=Undefined,
                 domainDashOffset=Undefined, domainOpacity=Undefined, domainWidth=Undefined,
                 format=Undefined, formatType=Undefined, grid=Undefined, gridCap=Undefined,
                 gridColor=Undefined, gridDash=Undefined, gridDashOffset=Undefined,
                 gridOpacity=Undefined, gridWidth=Undefined, labelAlign=Undefined, labelAngle=Undefined,
                 labelBaseline=Undefined, labelBound=Undefined, labelColor=Undefined,
                 labelExpr=Undefined, labelFlush=Undefined, labelFlushOffset=Undefined,
                 labelFont=Undefined, labelFontSize=Undefined, labelFontStyle=Undefined,
                 labelFontWeight=Undefined, labelLimit=Undefined, labelLineHeight=Undefined,
                 labelOffset=Undefined, labelOpacity=Undefined, labelOverlap=Undefined,
                 labelPadding=Undefined, labelSeparation=Undefined, labels=Undefined,
                 maxExtent=Undefined, minExtent=Undefined, offset=Undefined, orient=Undefined,
                 position=Undefined, style=Undefined, tickBand=Undefined, tickCap=Undefined,
                 tickColor=Undefined, tickCount=Undefined, tickDash=Undefined, tickDashOffset=Undefined,
                 tickExtra=Undefined, tickMinStep=Undefined, tickOffset=Undefined,
                 tickOpacity=Undefined, tickRound=Undefined, tickSize=Undefined, tickWidth=Undefined,
                 ticks=Undefined, title=Undefined, titleAlign=Undefined, titleAnchor=Undefined,
                 titleAngle=Undefined, titleBaseline=Undefined, titleColor=Undefined,
                 titleFont=Undefined, titleFontSize=Undefined, titleFontStyle=Undefined,
                 titleFontWeight=Undefined, titleLimit=Undefined, titleLineHeight=Undefined,
                 titleOpacity=Undefined, titlePadding=Undefined, titleX=Undefined, titleY=Undefined,
                 translate=Undefined, values=Undefined, zindex=Undefined, **kwds):
        super(Axis, self).__init__(aria=aria, bandPosition=bandPosition, description=description,
                                   domain=domain, domainCap=domainCap, domainColor=domainColor,
                                   domainDash=domainDash, domainDashOffset=domainDashOffset,
                                   domainOpacity=domainOpacity, domainWidth=domainWidth, format=format,
                                   formatType=formatType, grid=grid, gridCap=gridCap,
                                   gridColor=gridColor, gridDash=gridDash,
                                   gridDashOffset=gridDashOffset, gridOpacity=gridOpacity,
                                   gridWidth=gridWidth, labelAlign=labelAlign, labelAngle=labelAngle,
                                   labelBaseline=labelBaseline, labelBound=labelBound,
                                   labelColor=labelColor, labelExpr=labelExpr, labelFlush=labelFlush,
                                   labelFlushOffset=labelFlushOffset, labelFont=labelFont,
                                   labelFontSize=labelFontSize, labelFontStyle=labelFontStyle,
                                   labelFontWeight=labelFontWeight, labelLimit=labelLimit,
                                   labelLineHeight=labelLineHeight, labelOffset=labelOffset,
                                   labelOpacity=labelOpacity, labelOverlap=labelOverlap,
                                   labelPadding=labelPadding, labelSeparation=labelSeparation,
                                   labels=labels, maxExtent=maxExtent, minExtent=minExtent,
                                   offset=offset, orient=orient, position=position, style=style,
                                   tickBand=tickBand, tickCap=tickCap, tickColor=tickColor,
                                   tickCount=tickCount, tickDash=tickDash,
                                   tickDashOffset=tickDashOffset, tickExtra=tickExtra,
                                   tickMinStep=tickMinStep, tickOffset=tickOffset,
                                   tickOpacity=tickOpacity, tickRound=tickRound, tickSize=tickSize,
                                   tickWidth=tickWidth, ticks=ticks, title=title, titleAlign=titleAlign,
                                   titleAnchor=titleAnchor, titleAngle=titleAngle,
                                   titleBaseline=titleBaseline, titleColor=titleColor,
                                   titleFont=titleFont, titleFontSize=titleFontSize,
                                   titleFontStyle=titleFontStyle, titleFontWeight=titleFontWeight,
                                   titleLimit=titleLimit, titleLineHeight=titleLineHeight,
                                   titleOpacity=titleOpacity, titlePadding=titlePadding, titleX=titleX,
                                   titleY=titleY, translate=translate, values=values, zindex=zindex,
                                   **kwds)


class AxisConfig(VegaLiteSchema):
    """AxisConfig schema wrapper

    Mapping(required=[])

    Attributes
    ----------

    aria : anyOf(boolean, :class:`ExprRef`)

    bandPosition : anyOf(float, :class:`ExprRef`)

    description : anyOf(string, :class:`ExprRef`)

    disable : boolean
        Disable axis by default.
    domain : anyOf(boolean, :class:`ExprRef`)

    domainCap : anyOf(:class:`StrokeCap`, :class:`ExprRef`)

    domainColor : anyOf(anyOf(None, :class:`Color`), :class:`ExprRef`)

    domainDash : anyOf(List(float), :class:`ExprRef`)

    domainDashOffset : anyOf(float, :class:`ExprRef`)

    domainOpacity : anyOf(float, :class:`ExprRef`)

    domainWidth : anyOf(float, :class:`ExprRef`)

    format : anyOf(string, :class:`Dictunknown`)
        When used with the default ``"number"`` and ``"time"`` format type, the text
        formatting pattern for labels of guides (axes, legends, headers) and text marks.


        * If the format type is ``"number"`` (e.g., for quantitative fields), this is D3's
          `number format pattern <https://github.com/d3/d3-format#locale_format>`__. - If
          the format type is ``"time"`` (e.g., for temporal fields), this is D3's `time
          format pattern <https://github.com/d3/d3-time-format#locale_format>`__.

        See the `format documentation <https://vega.github.io/vega-lite/docs/format.html>`__
        for more examples.

        When used with a `custom formatType
        <https://vega.github.io/vega-lite/docs/config.html#custom-format-type>`__, this
        value will be passed as ``format`` alongside ``datum.value`` to the registered
        function.

        **Default value:**  Derived from `numberFormat
        <https://vega.github.io/vega-lite/docs/config.html#format>`__ config for number
        format and from `timeFormat
        <https://vega.github.io/vega-lite/docs/config.html#format>`__ config for time
        format.
    formatType : string
        The format type for labels. One of ``"number"``, ``"time"``, or a `registered custom
        format type
        <https://vega.github.io/vega-lite/docs/config.html#custom-format-type>`__.

        **Default value:** - ``"time"`` for temporal fields and ordinal and nominal fields
        with ``timeUnit``. - ``"number"`` for quantitative fields as well as ordinal and
        nominal fields without ``timeUnit``.
    grid : boolean
        A boolean flag indicating if grid lines should be included as part of the axis

        **Default value:** ``true`` for `continuous scales
        <https://vega.github.io/vega-lite/docs/scale.html#continuous>`__ that are not
        binned; otherwise, ``false``.
    gridCap : anyOf(:class:`StrokeCap`, :class:`ExprRef`)

    gridColor : anyOf(anyOf(None, :class:`Color`), :class:`ExprRef`,
    :class:`ConditionalAxisColor`)

    gridDash : anyOf(List(float), :class:`ExprRef`, :class:`ConditionalAxisNumberArray`)

    gridDashOffset : anyOf(float, :class:`ExprRef`, :class:`ConditionalAxisNumber`)

    gridOpacity : anyOf(float, :class:`ExprRef`, :class:`ConditionalAxisNumber`)

    gridWidth : anyOf(float, :class:`ExprRef`, :class:`ConditionalAxisNumber`)

    labelAlign : anyOf(:class:`Align`, :class:`ExprRef`, :class:`ConditionalAxisLabelAlign`)

    labelAngle : anyOf(float, :class:`ExprRef`)

    labelBaseline : anyOf(:class:`TextBaseline`, :class:`ExprRef`,
    :class:`ConditionalAxisLabelBaseline`)

    labelBound : anyOf(anyOf(float, boolean), :class:`ExprRef`)

    labelColor : anyOf(anyOf(None, :class:`Color`), :class:`ExprRef`,
    :class:`ConditionalAxisColor`)

    labelExpr : string
        `Vega expression <https://vega.github.io/vega/docs/expressions/>`__ for customizing
        labels text.

        **Note:** The label text and value can be assessed via the ``label`` and ``value``
        properties of the axis's backing ``datum`` object.
    labelFlush : anyOf(boolean, float)
        Indicates if the first and last axis labels should be aligned flush with the scale
        range. Flush alignment for a horizontal axis will left-align the first label and
        right-align the last label. For vertical axes, bottom and top text baselines are
        applied instead. If this property is a number, it also indicates the number of
        pixels by which to offset the first and last labels; for example, a value of 2 will
        flush-align the first and last labels and also push them 2 pixels outward from the
        center of the axis. The additional adjustment can sometimes help the labels better
        visually group with corresponding axis ticks.

        **Default value:** ``true`` for axis of a continuous x-scale. Otherwise, ``false``.
    labelFlushOffset : anyOf(float, :class:`ExprRef`)

    labelFont : anyOf(string, :class:`ExprRef`, :class:`ConditionalAxisString`)

    labelFontSize : anyOf(float, :class:`ExprRef`, :class:`ConditionalAxisNumber`)

    labelFontStyle : anyOf(:class:`FontStyle`, :class:`ExprRef`,
    :class:`ConditionalAxisLabelFontStyle`)

    labelFontWeight : anyOf(:class:`FontWeight`, :class:`ExprRef`,
    :class:`ConditionalAxisLabelFontWeight`)

    labelLimit : anyOf(float, :class:`ExprRef`)

    labelLineHeight : anyOf(float, :class:`ExprRef`)

    labelOffset : anyOf(float, :class:`ExprRef`, :class:`ConditionalAxisNumber`)

    labelOpacity : anyOf(float, :class:`ExprRef`, :class:`ConditionalAxisNumber`)

    labelOverlap : anyOf(:class:`LabelOverlap`, :class:`ExprRef`)
        The strategy to use for resolving overlap of axis labels. If ``false`` (the
        default), no overlap reduction is attempted. If set to ``true`` or ``"parity"``, a
        strategy of removing every other label is used (this works well for standard linear
        axes). If set to ``"greedy"``, a linear scan of the labels is performed, removing
        any labels that overlaps with the last visible label (this often works better for
        log-scaled axes).

        **Default value:** ``true`` for non-nominal fields with non-log scales; ``"greedy"``
        for log scales; otherwise ``false``.
    labelPadding : anyOf(float, :class:`ExprRef`, :class:`ConditionalAxisNumber`)

    labelSeparation : anyOf(float, :class:`ExprRef`)

    labels : anyOf(boolean, :class:`ExprRef`)

    maxExtent : anyOf(float, :class:`ExprRef`)

    minExtent : anyOf(float, :class:`ExprRef`)

    offset : float
        The offset, in pixels, by which to displace the axis from the edge of the enclosing
        group or data rectangle.

        **Default value:** derived from the `axis config
        <https://vega.github.io/vega-lite/docs/config.html#facet-scale-config>`__ 's
        ``offset`` ( ``0`` by default)
    orient : anyOf(:class:`AxisOrient`, :class:`ExprRef`)
        The orientation of the axis. One of ``"top"``, ``"bottom"``, ``"left"`` or
        ``"right"``. The orientation can be used to further specialize the axis type (e.g.,
        a y-axis oriented towards the right edge of the chart).

        **Default value:** ``"bottom"`` for x-axes and ``"left"`` for y-axes.
    position : anyOf(float, :class:`ExprRef`)
        The anchor position of the axis in pixels. For x-axes with top or bottom
        orientation, this sets the axis group x coordinate. For y-axes with left or right
        orientation, this sets the axis group y coordinate.

        **Default value** : ``0``
    style : anyOf(string, List(string))
        A string or array of strings indicating the name of custom styles to apply to the
        axis. A style is a named collection of axis property defined within the `style
        configuration <https://vega.github.io/vega-lite/docs/mark.html#style-config>`__. If
        style is an array, later styles will override earlier styles.

        **Default value:** (none) **Note:** Any specified style will augment the default
        style. For example, an x-axis mark with ``"style": "foo"`` will use ``config.axisX``
        and ``config.style.foo`` (the specified style ``"foo"`` has higher precedence).
    tickBand : anyOf(enum('center', 'extent'), :class:`ExprRef`)

    tickCap : anyOf(:class:`StrokeCap`, :class:`ExprRef`)

    tickColor : anyOf(anyOf(None, :class:`Color`), :class:`ExprRef`,
    :class:`ConditionalAxisColor`)

    tickCount : anyOf(float, :class:`TimeInterval`, :class:`TimeIntervalStep`, :class:`ExprRef`)
        A desired number of ticks, for axes visualizing quantitative scales. The resulting
        number may be different so that values are "nice" (multiples of 2, 5, 10) and lie
        within the underlying scale's range.

        For scales of type ``"time"`` or ``"utc"``, the tick count can instead be a time
        interval specifier. Legal string values are ``"millisecond"``, ``"second"``,
        ``"minute"``, ``"hour"``, ``"day"``, ``"week"``, ``"month"``, and ``"year"``.
        Alternatively, an object-valued interval specifier of the form ``{"interval":
        "month", "step": 3}`` includes a desired number of interval steps. Here, ticks are
        generated for each quarter (Jan, Apr, Jul, Oct) boundary.

        **Default value** : Determine using a formula ``ceil(width/40)`` for x and
        ``ceil(height/40)`` for y.
    tickDash : anyOf(List(float), :class:`ExprRef`, :class:`ConditionalAxisNumberArray`)

    tickDashOffset : anyOf(float, :class:`ExprRef`, :class:`ConditionalAxisNumber`)

    tickExtra : anyOf(boolean, :class:`ExprRef`)

    tickMinStep : anyOf(float, :class:`ExprRef`)
        The minimum desired step between axis ticks, in terms of scale domain values. For
        example, a value of ``1`` indicates that ticks should not be less than 1 unit apart.
        If ``tickMinStep`` is specified, the ``tickCount`` value will be adjusted, if
        necessary, to enforce the minimum step value.
    tickOffset : anyOf(float, :class:`ExprRef`)

    tickOpacity : anyOf(float, :class:`ExprRef`, :class:`ConditionalAxisNumber`)

    tickRound : anyOf(boolean, :class:`ExprRef`)

    tickSize : anyOf(float, :class:`ExprRef`, :class:`ConditionalAxisNumber`)

    tickWidth : anyOf(float, :class:`ExprRef`, :class:`ConditionalAxisNumber`)

    ticks : anyOf(boolean, :class:`ExprRef`)

    title : anyOf(:class:`Text`, None)
        A title for the field. If ``null``, the title will be removed.

        **Default value:**  derived from the field's name and transformation function (
        ``aggregate``, ``bin`` and ``timeUnit`` ). If the field has an aggregate function,
        the function is displayed as part of the title (e.g., ``"Sum of Profit"`` ). If the
        field is binned or has a time unit applied, the applied function is shown in
        parentheses (e.g., ``"Profit (binned)"``, ``"Transaction Date (year-month)"`` ).
        Otherwise, the title is simply the field name.

        **Notes** :

        1) You can customize the default field title format by providing the `fieldTitle
        <https://vega.github.io/vega-lite/docs/config.html#top-level-config>`__ property in
        the `config <https://vega.github.io/vega-lite/docs/config.html>`__ or `fieldTitle
        function via the compile function's options
        <https://vega.github.io/vega-lite/docs/compile.html#field-title>`__.

        2) If both field definition's ``title`` and axis, header, or legend ``title`` are
        defined, axis/header/legend title will be used.
    titleAlign : anyOf(:class:`Align`, :class:`ExprRef`)

    titleAnchor : anyOf(:class:`TitleAnchor`, :class:`ExprRef`)

    titleAngle : anyOf(float, :class:`ExprRef`)

    titleBaseline : anyOf(:class:`TextBaseline`, :class:`ExprRef`)

    titleColor : anyOf(anyOf(None, :class:`Color`), :class:`ExprRef`)

    titleFont : anyOf(string, :class:`ExprRef`)

    titleFontSize : anyOf(float, :class:`ExprRef`)

    titleFontStyle : anyOf(:class:`FontStyle`, :class:`ExprRef`)

    titleFontWeight : anyOf(:class:`FontWeight`, :class:`ExprRef`)

    titleLimit : anyOf(float, :class:`ExprRef`)

    titleLineHeight : anyOf(float, :class:`ExprRef`)

    titleOpacity : anyOf(float, :class:`ExprRef`)

    titlePadding : anyOf(float, :class:`ExprRef`)

    titleX : anyOf(float, :class:`ExprRef`)

    titleY : anyOf(float, :class:`ExprRef`)

    translate : anyOf(float, :class:`ExprRef`)

    values : anyOf(List(float), List(string), List(boolean), List(:class:`DateTime`),
    :class:`ExprRef`)
        Explicitly set the visible axis tick values.
    zindex : float
        A non-negative integer indicating the z-index of the axis. If zindex is 0, axes
        should be drawn behind all chart elements. To put them in front, set ``zindex`` to
        ``1`` or more.

        **Default value:** ``0`` (behind the marks).
    """
    _schema = {'$ref': '#/definitions/AxisConfig'}

    def __init__(self, aria=Undefined, bandPosition=Undefined, description=Undefined, disable=Undefined,
                 domain=Undefined, domainCap=Undefined, domainColor=Undefined, domainDash=Undefined,
                 domainDashOffset=Undefined, domainOpacity=Undefined, domainWidth=Undefined,
                 format=Undefined, formatType=Undefined, grid=Undefined, gridCap=Undefined,
                 gridColor=Undefined, gridDash=Undefined, gridDashOffset=Undefined,
                 gridOpacity=Undefined, gridWidth=Undefined, labelAlign=Undefined, labelAngle=Undefined,
                 labelBaseline=Undefined, labelBound=Undefined, labelColor=Undefined,
                 labelExpr=Undefined, labelFlush=Undefined, labelFlushOffset=Undefined,
                 labelFont=Undefined, labelFontSize=Undefined, labelFontStyle=Undefined,
                 labelFontWeight=Undefined, labelLimit=Undefined, labelLineHeight=Undefined,
                 labelOffset=Undefined, labelOpacity=Undefined, labelOverlap=Undefined,
                 labelPadding=Undefined, labelSeparation=Undefined, labels=Undefined,
                 maxExtent=Undefined, minExtent=Undefined, offset=Undefined, orient=Undefined,
                 position=Undefined, style=Undefined, tickBand=Undefined, tickCap=Undefined,
                 tickColor=Undefined, tickCount=Undefined, tickDash=Undefined, tickDashOffset=Undefined,
                 tickExtra=Undefined, tickMinStep=Undefined, tickOffset=Undefined,
                 tickOpacity=Undefined, tickRound=Undefined, tickSize=Undefined, tickWidth=Undefined,
                 ticks=Undefined, title=Undefined, titleAlign=Undefined, titleAnchor=Undefined,
                 titleAngle=Undefined, titleBaseline=Undefined, titleColor=Undefined,
                 titleFont=Undefined, titleFontSize=Undefined, titleFontStyle=Undefined,
                 titleFontWeight=Undefined, titleLimit=Undefined, titleLineHeight=Undefined,
                 titleOpacity=Undefined, titlePadding=Undefined, titleX=Undefined, titleY=Undefined,
                 translate=Undefined, values=Undefined, zindex=Undefined, **kwds):
        super(AxisConfig, self).__init__(aria=aria, bandPosition=bandPosition, description=description,
                                         disable=disable, domain=domain, domainCap=domainCap,
                                         domainColor=domainColor, domainDash=domainDash,
                                         domainDashOffset=domainDashOffset, domainOpacity=domainOpacity,
                                         domainWidth=domainWidth, format=format, formatType=formatType,
                                         grid=grid, gridCap=gridCap, gridColor=gridColor,
                                         gridDash=gridDash, gridDashOffset=gridDashOffset,
                                         gridOpacity=gridOpacity, gridWidth=gridWidth,
                                         labelAlign=labelAlign, labelAngle=labelAngle,
                                         labelBaseline=labelBaseline, labelBound=labelBound,
                                         labelColor=labelColor, labelExpr=labelExpr,
                                         labelFlush=labelFlush, labelFlushOffset=labelFlushOffset,
                                         labelFont=labelFont, labelFontSize=labelFontSize,
                                         labelFontStyle=labelFontStyle, labelFontWeight=labelFontWeight,
                                         labelLimit=labelLimit, labelLineHeight=labelLineHeight,
                                         labelOffset=labelOffset, labelOpacity=labelOpacity,
                                         labelOverlap=labelOverlap, labelPadding=labelPadding,
                                         labelSeparation=labelSeparation, labels=labels,
                                         maxExtent=maxExtent, minExtent=minExtent, offset=offset,
                                         orient=orient, position=position, style=style,
                                         tickBand=tickBand, tickCap=tickCap, tickColor=tickColor,
                                         tickCount=tickCount, tickDash=tickDash,
                                         tickDashOffset=tickDashOffset, tickExtra=tickExtra,
                                         tickMinStep=tickMinStep, tickOffset=tickOffset,
                                         tickOpacity=tickOpacity, tickRound=tickRound,
                                         tickSize=tickSize, tickWidth=tickWidth, ticks=ticks,
                                         title=title, titleAlign=titleAlign, titleAnchor=titleAnchor,
                                         titleAngle=titleAngle, titleBaseline=titleBaseline,
                                         titleColor=titleColor, titleFont=titleFont,
                                         titleFontSize=titleFontSize, titleFontStyle=titleFontStyle,
                                         titleFontWeight=titleFontWeight, titleLimit=titleLimit,
                                         titleLineHeight=titleLineHeight, titleOpacity=titleOpacity,
                                         titlePadding=titlePadding, titleX=titleX, titleY=titleY,
                                         translate=translate, values=values, zindex=zindex, **kwds)


class AxisOrient(VegaLiteSchema):
    """AxisOrient schema wrapper

    enum('top', 'bottom', 'left', 'right')
    """
    _schema = {'$ref': '#/definitions/AxisOrient'}

    def __init__(self, *args):
        super(AxisOrient, self).__init__(*args)


class AxisResolveMap(VegaLiteSchema):
    """AxisResolveMap schema wrapper

    Mapping(required=[])

    Attributes
    ----------

    x : :class:`ResolveMode`

    y : :class:`ResolveMode`

    """
    _schema = {'$ref': '#/definitions/AxisResolveMap'}

    def __init__(self, x=Undefined, y=Undefined, **kwds):
        super(AxisResolveMap, self).__init__(x=x, y=y, **kwds)


class BarConfig(AnyMarkConfig):
    """BarConfig schema wrapper

    Mapping(required=[])

    Attributes
    ----------

    align : anyOf(:class:`Align`, :class:`ExprRef`)
        The horizontal alignment of the text or ranged marks (area, bar, image, rect, rule).
        One of ``"left"``, ``"right"``, ``"center"``.

        **Note:** Expression reference is *not* supported for range marks.
    angle : anyOf(float, :class:`ExprRef`)

    aria : anyOf(boolean, :class:`ExprRef`)

    ariaRole : anyOf(string, :class:`ExprRef`)

    ariaRoleDescription : anyOf(string, :class:`ExprRef`)

    aspect : anyOf(boolean, :class:`ExprRef`)

    baseline : anyOf(:class:`TextBaseline`, :class:`ExprRef`)
        For text marks, the vertical text baseline. One of ``"alphabetic"`` (default),
        ``"top"``, ``"middle"``, ``"bottom"``, ``"line-top"``, ``"line-bottom"``, or an
        expression reference that provides one of the valid values. The ``"line-top"`` and
        ``"line-bottom"`` values operate similarly to ``"top"`` and ``"bottom"``, but are
        calculated relative to the ``lineHeight`` rather than ``fontSize`` alone.

        For range marks, the vertical alignment of the marks. One of ``"top"``,
        ``"middle"``, ``"bottom"``.

        **Note:** Expression reference is *not* supported for range marks.
    binSpacing : float
        Offset between bars for binned field. The ideal value for this is either 0
        (preferred by statisticians) or 1 (Vega-Lite default, D3 example style).

        **Default value:** ``1``
    blend : anyOf(:class:`Blend`, :class:`ExprRef`)

    color : anyOf(:class:`Color`, :class:`Gradient`, :class:`ExprRef`)
        Default color.

        **Default value:** :raw-html:`<span style="color: #4682b4;">&#9632;</span>`
        ``"#4682b4"``

        **Note:** - This property cannot be used in a `style config
        <https://vega.github.io/vega-lite/docs/mark.html#style-config>`__. - The ``fill``
        and ``stroke`` properties have higher precedence than ``color`` and will override
        ``color``.
    continuousBandSize : float
        The default size of the bars on continuous scales.

        **Default value:** ``5``
    cornerRadius : anyOf(float, :class:`ExprRef`)

    cornerRadiusBottomLeft : anyOf(float, :class:`ExprRef`)

    cornerRadiusBottomRight : anyOf(float, :class:`ExprRef`)

    cornerRadiusEnd : anyOf(float, :class:`ExprRef`)
        * For vertical bars, top-left and top-right corner radius. - For horizontal bars,
          top-right and bottom-right corner radius.
    cornerRadiusTopLeft : anyOf(float, :class:`ExprRef`)

    cornerRadiusTopRight : anyOf(float, :class:`ExprRef`)

    cursor : anyOf(:class:`Cursor`, :class:`ExprRef`)

    description : anyOf(string, :class:`ExprRef`)

    dir : anyOf(:class:`TextDirection`, :class:`ExprRef`)

    discreteBandSize : float
        The default size of the bars with discrete dimensions. If unspecified, the default
        size is  ``step-2``, which provides 2 pixel offset between bars.
    dx : anyOf(float, :class:`ExprRef`)

    dy : anyOf(float, :class:`ExprRef`)

    ellipsis : anyOf(string, :class:`ExprRef`)

    endAngle : anyOf(float, :class:`ExprRef`)

    fill : anyOf(:class:`Color`, :class:`Gradient`, None, :class:`ExprRef`)
        Default fill color. This property has higher precedence than ``config.color``. Set
        to ``null`` to remove fill.

        **Default value:** (None)
    fillOpacity : anyOf(float, :class:`ExprRef`)

    filled : boolean
        Whether the mark's color should be used as fill color instead of stroke color.

        **Default value:** ``false`` for all ``point``, ``line``, and ``rule`` marks as well
        as ``geoshape`` marks for `graticule
        <https://vega.github.io/vega-lite/docs/data.html#graticule>`__ data sources;
        otherwise, ``true``.

        **Note:** This property cannot be used in a `style config
        <https://vega.github.io/vega-lite/docs/mark.html#style-config>`__.
    font : anyOf(string, :class:`ExprRef`)

    fontSize : anyOf(float, :class:`ExprRef`)

    fontStyle : anyOf(:class:`FontStyle`, :class:`ExprRef`)

    fontWeight : anyOf(:class:`FontWeight`, :class:`ExprRef`)

    height : anyOf(float, :class:`ExprRef`)

    href : anyOf(:class:`URI`, :class:`ExprRef`)

    innerRadius : anyOf(float, :class:`ExprRef`)
        The inner radius in pixels of arc marks. ``innerRadius`` is an alias for
        ``radius2``.
    interpolate : anyOf(:class:`Interpolate`, :class:`ExprRef`)

    invalid : enum('filter', None)
        Defines how Vega-Lite should handle marks for invalid values ( ``null`` and ``NaN``
        ). - If set to ``"filter"`` (default), all data items with null values will be
        skipped (for line, trail, and area marks) or filtered (for other marks). - If
        ``null``, all data items are included. In this case, invalid values will be
        interpreted as zeroes.
    limit : anyOf(float, :class:`ExprRef`)

    lineBreak : anyOf(string, :class:`ExprRef`)

    lineHeight : anyOf(float, :class:`ExprRef`)

    opacity : anyOf(float, :class:`ExprRef`)
        The overall opacity (value between [0,1]).

        **Default value:** ``0.7`` for non-aggregate plots with ``point``, ``tick``,
        ``circle``, or ``square`` marks or layered ``bar`` charts and ``1`` otherwise.
    order : anyOf(None, boolean)
        For line and trail marks, this ``order`` property can be set to ``null`` or
        ``false`` to make the lines use the original order in the data sources.
    orient : :class:`Orientation`
        The orientation of a non-stacked bar, tick, area, and line charts. The value is
        either horizontal (default) or vertical. - For bar, rule and tick, this determines
        whether the size of the bar and tick should be applied to x or y dimension. - For
        area, this property determines the orient property of the Vega output. - For line
        and trail marks, this property determines the sort order of the points in the line
        if ``config.sortLineBy`` is not specified. For stacked charts, this is always
        determined by the orientation of the stack; therefore explicitly specified value
        will be ignored.
    outerRadius : anyOf(float, :class:`ExprRef`)
        The outer radius in pixels of arc marks. ``outerRadius`` is an alias for ``radius``.
    padAngle : anyOf(float, :class:`ExprRef`)

    radius : anyOf(float, :class:`ExprRef`)
        For arc mark, the primary (outer) radius in pixels.

        For text marks, polar coordinate radial offset, in pixels, of the text from the
        origin determined by the ``x`` and ``y`` properties.
    radius2 : anyOf(float, :class:`ExprRef`)
        The secondary (inner) radius in pixels of arc marks.
    shape : anyOf(anyOf(:class:`SymbolShape`, string), :class:`ExprRef`)

    size : anyOf(float, :class:`ExprRef`)
        Default size for marks. - For ``point`` / ``circle`` / ``square``, this represents
        the pixel area of the marks. Note that this value sets the area of the symbol; the
        side lengths will increase with the square root of this value. - For ``bar``, this
        represents the band size of the bar, in pixels. - For ``text``, this represents the
        font size, in pixels.

        **Default value:** - ``30`` for point, circle, square marks; width/height's ``step``
        - ``2`` for bar marks with discrete dimensions; - ``5`` for bar marks with
        continuous dimensions; - ``11`` for text marks.
    smooth : anyOf(boolean, :class:`ExprRef`)

    startAngle : anyOf(float, :class:`ExprRef`)

    stroke : anyOf(:class:`Color`, :class:`Gradient`, None, :class:`ExprRef`)
        Default stroke color. This property has higher precedence than ``config.color``. Set
        to ``null`` to remove stroke.

        **Default value:** (None)
    strokeCap : anyOf(:class:`StrokeCap`, :class:`ExprRef`)

    strokeDash : anyOf(List(float), :class:`ExprRef`)

    strokeDashOffset : anyOf(float, :class:`ExprRef`)

    strokeJoin : anyOf(:class:`StrokeJoin`, :class:`ExprRef`)

    strokeMiterLimit : anyOf(float, :class:`ExprRef`)

    strokeOffset : anyOf(float, :class:`ExprRef`)

    strokeOpacity : anyOf(float, :class:`ExprRef`)

    strokeWidth : anyOf(float, :class:`ExprRef`)

    tension : anyOf(float, :class:`ExprRef`)

    text : anyOf(:class:`Text`, :class:`ExprRef`)

    theta : anyOf(float, :class:`ExprRef`)
        For arc marks, the arc length in radians if theta2 is not specified, otherwise the
        start arc angle. (A value of 0 indicates up or “north”, increasing values proceed
        clockwise.)

        For text marks, polar coordinate angle in radians.
    theta2 : anyOf(float, :class:`ExprRef`)
        The end angle of arc marks in radians. A value of 0 indicates up or “north”,
        increasing values proceed clockwise.
    timeUnitBand : float
        Default relative band size for a time unit. If set to ``1``, the bandwidth of the
        marks will be equal to the time unit band step. If set to ``0.5``, bandwidth of the
        marks will be half of the time unit band step.
    timeUnitBandPosition : float
        Default relative band position for a time unit. If set to ``0``, the marks will be
        positioned at the beginning of the time unit band step. If set to ``0.5``, the marks
        will be positioned in the middle of the time unit band step.
    tooltip : anyOf(float, string, boolean, :class:`TooltipContent`, :class:`ExprRef`, None)
        The tooltip text string to show upon mouse hover or an object defining which fields
        should the tooltip be derived from.


        * If ``tooltip`` is ``true`` or ``{"content": "encoding"}``, then all fields from
          ``encoding`` will be used. - If ``tooltip`` is ``{"content": "data"}``, then all
          fields that appear in the highlighted data point will be used. - If set to
          ``null`` or ``false``, then no tooltip will be used.

        See the `tooltip <https://vega.github.io/vega-lite/docs/tooltip.html>`__
        documentation for a detailed discussion about tooltip  in Vega-Lite.

        **Default value:** ``null``
    url : anyOf(:class:`URI`, :class:`ExprRef`)

    width : anyOf(float, :class:`ExprRef`)

    x : anyOf(float, string, :class:`ExprRef`)
        X coordinates of the marks, or width of horizontal ``"bar"`` and ``"area"`` without
        specified ``x2`` or ``width``.

        The ``value`` of this channel can be a number or a string ``"width"`` for the width
        of the plot.
    x2 : anyOf(float, string, :class:`ExprRef`)
        X2 coordinates for ranged ``"area"``, ``"bar"``, ``"rect"``, and  ``"rule"``.

        The ``value`` of this channel can be a number or a string ``"width"`` for the width
        of the plot.
    y : anyOf(float, string, :class:`ExprRef`)
        Y coordinates of the marks, or height of vertical ``"bar"`` and ``"area"`` without
        specified ``y2`` or ``height``.

        The ``value`` of this channel can be a number or a string ``"height"`` for the
        height of the plot.
    y2 : anyOf(float, string, :class:`ExprRef`)
        Y2 coordinates for ranged ``"area"``, ``"bar"``, ``"rect"``, and  ``"rule"``.

        The ``value`` of this channel can be a number or a string ``"height"`` for the
        height of the plot.
    """
    _schema = {'$ref': '#/definitions/BarConfig'}

    def __init__(self, align=Undefined, angle=Undefined, aria=Undefined, ariaRole=Undefined,
                 ariaRoleDescription=Undefined, aspect=Undefined, baseline=Undefined,
                 binSpacing=Undefined, blend=Undefined, color=Undefined, continuousBandSize=Undefined,
                 cornerRadius=Undefined, cornerRadiusBottomLeft=Undefined,
                 cornerRadiusBottomRight=Undefined, cornerRadiusEnd=Undefined,
                 cornerRadiusTopLeft=Undefined, cornerRadiusTopRight=Undefined, cursor=Undefined,
                 description=Undefined, dir=Undefined, discreteBandSize=Undefined, dx=Undefined,
                 dy=Undefined, ellipsis=Undefined, endAngle=Undefined, fill=Undefined,
                 fillOpacity=Undefined, filled=Undefined, font=Undefined, fontSize=Undefined,
                 fontStyle=Undefined, fontWeight=Undefined, height=Undefined, href=Undefined,
                 innerRadius=Undefined, interpolate=Undefined, invalid=Undefined, limit=Undefined,
                 lineBreak=Undefined, lineHeight=Undefined, opacity=Undefined, order=Undefined,
                 orient=Undefined, outerRadius=Undefined, padAngle=Undefined, radius=Undefined,
                 radius2=Undefined, shape=Undefined, size=Undefined, smooth=Undefined,
                 startAngle=Undefined, stroke=Undefined, strokeCap=Undefined, strokeDash=Undefined,
                 strokeDashOffset=Undefined, strokeJoin=Undefined, strokeMiterLimit=Undefined,
                 strokeOffset=Undefined, strokeOpacity=Undefined, strokeWidth=Undefined,
                 tension=Undefined, text=Undefined, theta=Undefined, theta2=Undefined,
                 timeUnitBand=Undefined, timeUnitBandPosition=Undefined, tooltip=Undefined,
                 url=Undefined, width=Undefined, x=Undefined, x2=Undefined, y=Undefined, y2=Undefined,
                 **kwds):
        super(BarConfig, self).__init__(align=align, angle=angle, aria=aria, ariaRole=ariaRole,
                                        ariaRoleDescription=ariaRoleDescription, aspect=aspect,
                                        baseline=baseline, binSpacing=binSpacing, blend=blend,
                                        color=color, continuousBandSize=continuousBandSize,
                                        cornerRadius=cornerRadius,
                                        cornerRadiusBottomLeft=cornerRadiusBottomLeft,
                                        cornerRadiusBottomRight=cornerRadiusBottomRight,
                                        cornerRadiusEnd=cornerRadiusEnd,
                                        cornerRadiusTopLeft=cornerRadiusTopLeft,
                                        cornerRadiusTopRight=cornerRadiusTopRight, cursor=cursor,
                                        description=description, dir=dir,
                                        discreteBandSize=discreteBandSize, dx=dx, dy=dy,
                                        ellipsis=ellipsis, endAngle=endAngle, fill=fill,
                                        fillOpacity=fillOpacity, filled=filled, font=font,
                                        fontSize=fontSize, fontStyle=fontStyle, fontWeight=fontWeight,
                                        height=height, href=href, innerRadius=innerRadius,
                                        interpolate=interpolate, invalid=invalid, limit=limit,
                                        lineBreak=lineBreak, lineHeight=lineHeight, opacity=opacity,
                                        order=order, orient=orient, outerRadius=outerRadius,
                                        padAngle=padAngle, radius=radius, radius2=radius2, shape=shape,
                                        size=size, smooth=smooth, startAngle=startAngle, stroke=stroke,
                                        strokeCap=strokeCap, strokeDash=strokeDash,
                                        strokeDashOffset=strokeDashOffset, strokeJoin=strokeJoin,
                                        strokeMiterLimit=strokeMiterLimit, strokeOffset=strokeOffset,
                                        strokeOpacity=strokeOpacity, strokeWidth=strokeWidth,
                                        tension=tension, text=text, theta=theta, theta2=theta2,
                                        timeUnitBand=timeUnitBand,
                                        timeUnitBandPosition=timeUnitBandPosition, tooltip=tooltip,
                                        url=url, width=width, x=x, x2=x2, y=y, y2=y2, **kwds)


class BaseTitleNoValueRefs(VegaLiteSchema):
    """BaseTitleNoValueRefs schema wrapper

    Mapping(required=[])

    Attributes
    ----------

    align : :class:`Align`
        Horizontal text alignment for title text. One of ``"left"``, ``"center"``, or
        ``"right"``.
    anchor : anyOf(:class:`TitleAnchor`, :class:`ExprRef`)

    angle : anyOf(float, :class:`ExprRef`)

    aria : anyOf(boolean, :class:`ExprRef`)

    baseline : :class:`TextBaseline`
        Vertical text baseline for title and subtitle text. One of ``"alphabetic"``
        (default), ``"top"``, ``"middle"``, ``"bottom"``, ``"line-top"``, or
        ``"line-bottom"``. The ``"line-top"`` and ``"line-bottom"`` values operate similarly
        to ``"top"`` and ``"bottom"``, but are calculated relative to the *lineHeight*
        rather than *fontSize* alone.
    color : anyOf(anyOf(None, :class:`Color`), :class:`ExprRef`)

    dx : anyOf(float, :class:`ExprRef`)

    dy : anyOf(float, :class:`ExprRef`)

    font : anyOf(string, :class:`ExprRef`)

    fontSize : anyOf(float, :class:`ExprRef`)

    fontStyle : anyOf(:class:`FontStyle`, :class:`ExprRef`)

    fontWeight : anyOf(:class:`FontWeight`, :class:`ExprRef`)

    frame : anyOf(anyOf(:class:`TitleFrame`, string), :class:`ExprRef`)

    limit : anyOf(float, :class:`ExprRef`)

    lineHeight : anyOf(float, :class:`ExprRef`)

    offset : anyOf(float, :class:`ExprRef`)

    orient : anyOf(:class:`TitleOrient`, :class:`ExprRef`)

    subtitleColor : anyOf(anyOf(None, :class:`Color`), :class:`ExprRef`)

    subtitleFont : anyOf(string, :class:`ExprRef`)

    subtitleFontSize : anyOf(float, :class:`ExprRef`)

    subtitleFontStyle : anyOf(:class:`FontStyle`, :class:`ExprRef`)

    subtitleFontWeight : anyOf(:class:`FontWeight`, :class:`ExprRef`)

    subtitleLineHeight : anyOf(float, :class:`ExprRef`)

    subtitlePadding : anyOf(float, :class:`ExprRef`)

    zindex : anyOf(float, :class:`ExprRef`)

    """
    _schema = {'$ref': '#/definitions/BaseTitleNoValueRefs'}

    def __init__(self, align=Undefined, anchor=Undefined, angle=Undefined, aria=Undefined,
                 baseline=Undefined, color=Undefined, dx=Undefined, dy=Undefined, font=Undefined,
                 fontSize=Undefined, fontStyle=Undefined, fontWeight=Undefined, frame=Undefined,
                 limit=Undefined, lineHeight=Undefined, offset=Undefined, orient=Undefined,
                 subtitleColor=Undefined, subtitleFont=Undefined, subtitleFontSize=Undefined,
                 subtitleFontStyle=Undefined, subtitleFontWeight=Undefined,
                 subtitleLineHeight=Undefined, subtitlePadding=Undefined, zindex=Undefined, **kwds):
        super(BaseTitleNoValueRefs, self).__init__(align=align, anchor=anchor, angle=angle, aria=aria,
                                                   baseline=baseline, color=color, dx=dx, dy=dy,
                                                   font=font, fontSize=fontSize, fontStyle=fontStyle,
                                                   fontWeight=fontWeight, frame=frame, limit=limit,
                                                   lineHeight=lineHeight, offset=offset, orient=orient,
                                                   subtitleColor=subtitleColor,
                                                   subtitleFont=subtitleFont,
                                                   subtitleFontSize=subtitleFontSize,
                                                   subtitleFontStyle=subtitleFontStyle,
                                                   subtitleFontWeight=subtitleFontWeight,
                                                   subtitleLineHeight=subtitleLineHeight,
                                                   subtitlePadding=subtitlePadding, zindex=zindex,
                                                   **kwds)


class BinExtent(VegaLiteSchema):
    """BinExtent schema wrapper

    anyOf(List([float, float]), :class:`SelectionExtent`)
    """
    _schema = {'$ref': '#/definitions/BinExtent'}

    def __init__(self, *args, **kwds):
        super(BinExtent, self).__init__(*args, **kwds)


class BinParams(VegaLiteSchema):
    """BinParams schema wrapper

    Mapping(required=[])
    Binning properties or boolean flag for determining whether to bin data or not.

    Attributes
    ----------

    anchor : float
        A value in the binned domain at which to anchor the bins, shifting the bin
        boundaries if necessary to ensure that a boundary aligns with the anchor value.

        **Default value:** the minimum bin extent value
    base : float
        The number base to use for automatic bin determination (default is base 10).

        **Default value:** ``10``
    binned : boolean
        When set to ``true``, Vega-Lite treats the input data as already binned.
    divide : List([float, float])
        Scale factors indicating allowable subdivisions. The default value is [5, 2], which
        indicates that for base 10 numbers (the default base), the method may consider
        dividing bin sizes by 5 and/or 2. For example, for an initial step size of 10, the
        method can check if bin sizes of 2 (= 10/5), 5 (= 10/2), or 1 (= 10/(5*2)) might
        also satisfy the given constraints.

        **Default value:** ``[5, 2]``
    extent : :class:`BinExtent`
        A two-element ( ``[min, max]`` ) array indicating the range of desired bin values.
    maxbins : float
        Maximum number of bins.

        **Default value:** ``6`` for ``row``, ``column`` and ``shape`` channels; ``10`` for
        other channels
    minstep : float
        A minimum allowable step size (particularly useful for integer values).
    nice : boolean
        If true, attempts to make the bin boundaries use human-friendly boundaries, such as
        multiples of ten.

        **Default value:** ``true``
    step : float
        An exact step size to use between bins.

        **Note:** If provided, options such as maxbins will be ignored.
    steps : List(float)
        An array of allowable step sizes to choose from.
    """
    _schema = {'$ref': '#/definitions/BinParams'}

    def __init__(self, anchor=Undefined, base=Undefined, binned=Undefined, divide=Undefined,
                 extent=Undefined, maxbins=Undefined, minstep=Undefined, nice=Undefined, step=Undefined,
                 steps=Undefined, **kwds):
        super(BinParams, self).__init__(anchor=anchor, base=base, binned=binned, divide=divide,
                                        extent=extent, maxbins=maxbins, minstep=minstep, nice=nice,
                                        step=step, steps=steps, **kwds)


class Binding(VegaLiteSchema):
    """Binding schema wrapper

    anyOf(:class:`BindCheckbox`, :class:`BindRadioSelect`, :class:`BindRange`,
    :class:`InputBinding`)
    """
    _schema = {'$ref': '#/definitions/Binding'}

    def __init__(self, *args, **kwds):
        super(Binding, self).__init__(*args, **kwds)


class BindCheckbox(Binding):
    """BindCheckbox schema wrapper

    Mapping(required=[input])

    Attributes
    ----------

    input : string

    debounce : float

    element : :class:`Element`

    name : string

    type : string

    """
    _schema = {'$ref': '#/definitions/BindCheckbox'}

    def __init__(self, input=Undefined, debounce=Undefined, element=Undefined, name=Undefined,
                 type=Undefined, **kwds):
        super(BindCheckbox, self).__init__(input=input, debounce=debounce, element=element, name=name,
                                           type=type, **kwds)


class BindRadioSelect(Binding):
    """BindRadioSelect schema wrapper

    Mapping(required=[input, options])

    Attributes
    ----------

    input : enum('radio', 'select')

    options : List(Any)

    debounce : float

    element : :class:`Element`

    labels : List(string)

    name : string

    type : string

    """
    _schema = {'$ref': '#/definitions/BindRadioSelect'}

    def __init__(self, input=Undefined, options=Undefined, debounce=Undefined, element=Undefined,
                 labels=Undefined, name=Undefined, type=Undefined, **kwds):
        super(BindRadioSelect, self).__init__(input=input, options=options, debounce=debounce,
                                              element=element, labels=labels, name=name, type=type,
                                              **kwds)


class BindRange(Binding):
    """BindRange schema wrapper

    Mapping(required=[input])

    Attributes
    ----------

    input : string

    debounce : float

    element : :class:`Element`

    max : float

    min : float

    name : string

    step : float

    type : string

    """
    _schema = {'$ref': '#/definitions/BindRange'}

    def __init__(self, input=Undefined, debounce=Undefined, element=Undefined, max=Undefined,
                 min=Undefined, name=Undefined, step=Undefined, type=Undefined, **kwds):
        super(BindRange, self).__init__(input=input, debounce=debounce, element=element, max=max,
                                        min=min, name=name, step=step, type=type, **kwds)


class Blend(VegaLiteSchema):
    """Blend schema wrapper

    enum(None, 'multiply', 'screen', 'overlay', 'darken', 'lighten', 'color-dodge',
    'color-burn', 'hard-light', 'soft-light', 'difference', 'exclusion', 'hue', 'saturation',
    'color', 'luminosity')
    """
    _schema = {'$ref': '#/definitions/Blend'}

    def __init__(self, *args):
        super(Blend, self).__init__(*args)


class BoxPlotConfig(VegaLiteSchema):
    """BoxPlotConfig schema wrapper

    Mapping(required=[])

    Attributes
    ----------

    box : anyOf(boolean, :class:`MarkConfigExprOrSignalRef`)

    extent : anyOf(string, float)
        The extent of the whiskers. Available options include: - ``"min-max"`` : min and max
        are the lower and upper whiskers respectively. - A number representing multiple of
        the interquartile range. This number will be multiplied by the IQR to determine
        whisker boundary, which spans from the smallest data to the largest data within the
        range *[Q1 - k * IQR, Q3 + k * IQR]* where *Q1* and *Q3* are the first and third
        quartiles while *IQR* is the interquartile range ( *Q3-Q1* ).

        **Default value:** ``1.5``.
    median : anyOf(boolean, :class:`MarkConfigExprOrSignalRef`)

    outliers : anyOf(boolean, :class:`MarkConfigExprOrSignalRef`)

    rule : anyOf(boolean, :class:`MarkConfigExprOrSignalRef`)

    size : float
        Size of the box and median tick of a box plot
    ticks : anyOf(boolean, :class:`MarkConfigExprOrSignalRef`)

    """
    _schema = {'$ref': '#/definitions/BoxPlotConfig'}

    def __init__(self, box=Undefined, extent=Undefined, median=Undefined, outliers=Undefined,
                 rule=Undefined, size=Undefined, ticks=Undefined, **kwds):
        super(BoxPlotConfig, self).__init__(box=box, extent=extent, median=median, outliers=outliers,
                                            rule=rule, size=size, ticks=ticks, **kwds)


class BrushConfig(VegaLiteSchema):
    """BrushConfig schema wrapper

    Mapping(required=[])

    Attributes
    ----------

    cursor : :class:`Cursor`
        The mouse cursor used over the interval mark. Any valid `CSS cursor type
        <https://developer.mozilla.org/en-US/docs/Web/CSS/cursor#Values>`__ can be used.
    fill : :class:`Color`
        The fill color of the interval mark.

        **Default value:** ``"#333333"``
    fillOpacity : float
        The fill opacity of the interval mark (a value between ``0`` and ``1`` ).

        **Default value:** ``0.125``
    stroke : :class:`Color`
        The stroke color of the interval mark.

        **Default value:** ``"#ffffff"``
    strokeDash : List(float)
        An array of alternating stroke and space lengths, for creating dashed or dotted
        lines.
    strokeDashOffset : float
        The offset (in pixels) with which to begin drawing the stroke dash array.
    strokeOpacity : float
        The stroke opacity of the interval mark (a value between ``0`` and ``1`` ).
    strokeWidth : float
        The stroke width of the interval mark.
    """
    _schema = {'$ref': '#/definitions/BrushConfig'}

    def __init__(self, cursor=Undefined, fill=Undefined, fillOpacity=Undefined, stroke=Undefined,
                 strokeDash=Undefined, strokeDashOffset=Undefined, strokeOpacity=Undefined,
                 strokeWidth=Undefined, **kwds):
        super(BrushConfig, self).__init__(cursor=cursor, fill=fill, fillOpacity=fillOpacity,
                                          stroke=stroke, strokeDash=strokeDash,
                                          strokeDashOffset=strokeDashOffset,
                                          strokeOpacity=strokeOpacity, strokeWidth=strokeWidth, **kwds)


class Color(VegaLiteSchema):
    """Color schema wrapper

    anyOf(:class:`ColorName`, :class:`HexColor`, string)
    """
    _schema = {'$ref': '#/definitions/Color'}

    def __init__(self, *args, **kwds):
        super(Color, self).__init__(*args, **kwds)


class ColorDef(VegaLiteSchema):
    """ColorDef schema wrapper

    anyOf(:class:`FieldOrDatumDefWithConditionMarkPropFieldDefGradientstringnull`,
    :class:`FieldOrDatumDefWithConditionDatumDefGradientstringnull`,
    :class:`ValueDefWithConditionMarkPropFieldOrDatumDefGradientstringnull`)
    """
    _schema = {'$ref': '#/definitions/ColorDef'}

    def __init__(self, *args, **kwds):
        super(ColorDef, self).__init__(*args, **kwds)


class ColorName(Color):
    """ColorName schema wrapper

    enum('black', 'silver', 'gray', 'white', 'maroon', 'red', 'purple', 'fuchsia', 'green',
    'lime', 'olive', 'yellow', 'navy', 'blue', 'teal', 'aqua', 'orange', 'aliceblue',
    'antiquewhite', 'aquamarine', 'azure', 'beige', 'bisque', 'blanchedalmond', 'blueviolet',
    'brown', 'burlywood', 'cadetblue', 'chartreuse', 'chocolate', 'coral', 'cornflowerblue',
    'cornsilk', 'crimson', 'cyan', 'darkblue', 'darkcyan', 'darkgoldenrod', 'darkgray',
    'darkgreen', 'darkgrey', 'darkkhaki', 'darkmagenta', 'darkolivegreen', 'darkorange',
    'darkorchid', 'darkred', 'darksalmon', 'darkseagreen', 'darkslateblue', 'darkslategray',
    'darkslategrey', 'darkturquoise', 'darkviolet', 'deeppink', 'deepskyblue', 'dimgray',
    'dimgrey', 'dodgerblue', 'firebrick', 'floralwhite', 'forestgreen', 'gainsboro',
    'ghostwhite', 'gold', 'goldenrod', 'greenyellow', 'grey', 'honeydew', 'hotpink',
    'indianred', 'indigo', 'ivory', 'khaki', 'lavender', 'lavenderblush', 'lawngreen',
    'lemonchiffon', 'lightblue', 'lightcoral', 'lightcyan', 'lightgoldenrodyellow', 'lightgray',
    'lightgreen', 'lightgrey', 'lightpink', 'lightsalmon', 'lightseagreen', 'lightskyblue',
    'lightslategray', 'lightslategrey', 'lightsteelblue', 'lightyellow', 'limegreen', 'linen',
    'magenta', 'mediumaquamarine', 'mediumblue', 'mediumorchid', 'mediumpurple',
    'mediumseagreen', 'mediumslateblue', 'mediumspringgreen', 'mediumturquoise',
    'mediumvioletred', 'midnightblue', 'mintcream', 'mistyrose', 'moccasin', 'navajowhite',
    'oldlace', 'olivedrab', 'orangered', 'orchid', 'palegoldenrod', 'palegreen',
    'paleturquoise', 'palevioletred', 'papayawhip', 'peachpuff', 'peru', 'pink', 'plum',
    'powderblue', 'rosybrown', 'royalblue', 'saddlebrown', 'salmon', 'sandybrown', 'seagreen',
    'seashell', 'sienna', 'skyblue', 'slateblue', 'slategray', 'slategrey', 'snow',
    'springgreen', 'steelblue', 'tan', 'thistle', 'tomato', 'turquoise', 'violet', 'wheat',
    'whitesmoke', 'yellowgreen', 'rebeccapurple')
    """
    _schema = {'$ref': '#/definitions/ColorName'}

    def __init__(self, *args):
        super(ColorName, self).__init__(*args)


class ColorScheme(VegaLiteSchema):
    """ColorScheme schema wrapper

    anyOf(:class:`Categorical`, :class:`SequentialSingleHue`, :class:`SequentialMultiHue`,
    :class:`Diverging`, :class:`Cyclical`)
    """
    _schema = {'$ref': '#/definitions/ColorScheme'}

    def __init__(self, *args, **kwds):
        super(ColorScheme, self).__init__(*args, **kwds)


class Categorical(ColorScheme):
    """Categorical schema wrapper

    enum('accent', 'category10', 'category20', 'category20b', 'category20c', 'dark2', 'paired',
    'pastel1', 'pastel2', 'set1', 'set2', 'set3', 'tableau10', 'tableau20')
    """
    _schema = {'$ref': '#/definitions/Categorical'}

    def __init__(self, *args):
        super(Categorical, self).__init__(*args)


class CompositeMark(AnyMark):
    """CompositeMark schema wrapper

    anyOf(:class:`BoxPlot`, :class:`ErrorBar`, :class:`ErrorBand`)
    """
    _schema = {'$ref': '#/definitions/CompositeMark'}

    def __init__(self, *args, **kwds):
        super(CompositeMark, self).__init__(*args, **kwds)


class BoxPlot(CompositeMark):
    """BoxPlot schema wrapper

    string
    """
    _schema = {'$ref': '#/definitions/BoxPlot'}

    def __init__(self, *args):
        super(BoxPlot, self).__init__(*args)


class CompositeMarkDef(AnyMark):
    """CompositeMarkDef schema wrapper

    anyOf(:class:`BoxPlotDef`, :class:`ErrorBarDef`, :class:`ErrorBandDef`)
    """
    _schema = {'$ref': '#/definitions/CompositeMarkDef'}

    def __init__(self, *args, **kwds):
        super(CompositeMarkDef, self).__init__(*args, **kwds)


class BoxPlotDef(CompositeMarkDef):
    """BoxPlotDef schema wrapper

    Mapping(required=[type])

    Attributes
    ----------

    type : :class:`BoxPlot`
        The mark type. This could a primitive mark type (one of ``"bar"``, ``"circle"``,
        ``"square"``, ``"tick"``, ``"line"``, ``"area"``, ``"point"``, ``"geoshape"``,
        ``"rule"``, and ``"text"`` ) or a composite mark type ( ``"boxplot"``,
        ``"errorband"``, ``"errorbar"`` ).
    box : anyOf(boolean, :class:`MarkConfigExprOrSignalRef`)

    clip : boolean
        Whether a composite mark be clipped to the enclosing group’s width and height.
    color : anyOf(:class:`Color`, :class:`Gradient`, :class:`ExprRef`)
        Default color.

        **Default value:** :raw-html:`<span style="color: #4682b4;">&#9632;</span>`
        ``"#4682b4"``

        **Note:** - This property cannot be used in a `style config
        <https://vega.github.io/vega-lite/docs/mark.html#style-config>`__. - The ``fill``
        and ``stroke`` properties have higher precedence than ``color`` and will override
        ``color``.
    extent : anyOf(string, float)
        The extent of the whiskers. Available options include: - ``"min-max"`` : min and max
        are the lower and upper whiskers respectively. - A number representing multiple of
        the interquartile range. This number will be multiplied by the IQR to determine
        whisker boundary, which spans from the smallest data to the largest data within the
        range *[Q1 - k * IQR, Q3 + k * IQR]* where *Q1* and *Q3* are the first and third
        quartiles while *IQR* is the interquartile range ( *Q3-Q1* ).

        **Default value:** ``1.5``.
    median : anyOf(boolean, :class:`MarkConfigExprOrSignalRef`)

    opacity : float
        The opacity (value between [0,1]) of the mark.
    orient : :class:`Orientation`
        Orientation of the box plot. This is normally automatically determined based on
        types of fields on x and y channels. However, an explicit ``orient`` be specified
        when the orientation is ambiguous.

        **Default value:** ``"vertical"``.
    outliers : anyOf(boolean, :class:`MarkConfigExprOrSignalRef`)

    rule : anyOf(boolean, :class:`MarkConfigExprOrSignalRef`)

    size : float
        Size of the box and median tick of a box plot
    ticks : anyOf(boolean, :class:`MarkConfigExprOrSignalRef`)

    """
    _schema = {'$ref': '#/definitions/BoxPlotDef'}

    def __init__(self, type=Undefined, box=Undefined, clip=Undefined, color=Undefined, extent=Undefined,
                 median=Undefined, opacity=Undefined, orient=Undefined, outliers=Undefined,
                 rule=Undefined, size=Undefined, ticks=Undefined, **kwds):
        super(BoxPlotDef, self).__init__(type=type, box=box, clip=clip, color=color, extent=extent,
                                         median=median, opacity=opacity, orient=orient,
                                         outliers=outliers, rule=rule, size=size, ticks=ticks, **kwds)


class CompositionConfig(VegaLiteSchema):
    """CompositionConfig schema wrapper

    Mapping(required=[])

    Attributes
    ----------

    columns : float
        The number of columns to include in the view composition layout.

        **Default value** : ``undefined`` -- An infinite number of columns (a single row)
        will be assumed. This is equivalent to ``hconcat`` (for ``concat`` ) and to using
        the ``column`` channel (for ``facet`` and ``repeat`` ).

        **Note** :

        1) This property is only for: - the general (wrappable) ``concat`` operator (not
        ``hconcat`` / ``vconcat`` ) - the ``facet`` and ``repeat`` operator with one
        field/repetition definition (without row/column nesting)

        2) Setting the ``columns`` to ``1`` is equivalent to ``vconcat`` (for ``concat`` )
        and to using the ``row`` channel (for ``facet`` and ``repeat`` ).
    spacing : float
        The default spacing in pixels between composed sub-views.

        **Default value** : ``20``
    """
    _schema = {'$ref': '#/definitions/CompositionConfig'}

    def __init__(self, columns=Undefined, spacing=Undefined, **kwds):
        super(CompositionConfig, self).__init__(columns=columns, spacing=spacing, **kwds)


class ConditionalAxisColor(VegaLiteSchema):
    """ConditionalAxisColor schema wrapper

    anyOf(Mapping(required=[condition, value]), Mapping(required=[condition, expr]))
    """
    _schema = {'$ref': '#/definitions/ConditionalAxisColor'}

    def __init__(self, *args, **kwds):
        super(ConditionalAxisColor, self).__init__(*args, **kwds)


class ConditionalAxisLabelAlign(VegaLiteSchema):
    """ConditionalAxisLabelAlign schema wrapper

    anyOf(Mapping(required=[condition, value]), Mapping(required=[condition, expr]))
    """
    _schema = {'$ref': '#/definitions/ConditionalAxisLabelAlign'}

    def __init__(self, *args, **kwds):
        super(ConditionalAxisLabelAlign, self).__init__(*args, **kwds)


class ConditionalAxisLabelBaseline(VegaLiteSchema):
    """ConditionalAxisLabelBaseline schema wrapper

    anyOf(Mapping(required=[condition, value]), Mapping(required=[condition, expr]))
    """
    _schema = {'$ref': '#/definitions/ConditionalAxisLabelBaseline'}

    def __init__(self, *args, **kwds):
        super(ConditionalAxisLabelBaseline, self).__init__(*args, **kwds)


class ConditionalAxisLabelFontStyle(VegaLiteSchema):
    """ConditionalAxisLabelFontStyle schema wrapper

    anyOf(Mapping(required=[condition, value]), Mapping(required=[condition, expr]))
    """
    _schema = {'$ref': '#/definitions/ConditionalAxisLabelFontStyle'}

    def __init__(self, *args, **kwds):
        super(ConditionalAxisLabelFontStyle, self).__init__(*args, **kwds)


class ConditionalAxisLabelFontWeight(VegaLiteSchema):
    """ConditionalAxisLabelFontWeight schema wrapper

    anyOf(Mapping(required=[condition, value]), Mapping(required=[condition, expr]))
    """
    _schema = {'$ref': '#/definitions/ConditionalAxisLabelFontWeight'}

    def __init__(self, *args, **kwds):
        super(ConditionalAxisLabelFontWeight, self).__init__(*args, **kwds)


class ConditionalAxisNumber(VegaLiteSchema):
    """ConditionalAxisNumber schema wrapper

    anyOf(Mapping(required=[condition, value]), Mapping(required=[condition, expr]))
    """
    _schema = {'$ref': '#/definitions/ConditionalAxisNumber'}

    def __init__(self, *args, **kwds):
        super(ConditionalAxisNumber, self).__init__(*args, **kwds)


class ConditionalAxisNumberArray(VegaLiteSchema):
    """ConditionalAxisNumberArray schema wrapper

    anyOf(Mapping(required=[condition, value]), Mapping(required=[condition, expr]))
    """
    _schema = {'$ref': '#/definitions/ConditionalAxisNumberArray'}

    def __init__(self, *args, **kwds):
        super(ConditionalAxisNumberArray, self).__init__(*args, **kwds)


class ConditionalAxisPropertyAlignnull(VegaLiteSchema):
    """ConditionalAxisPropertyAlignnull schema wrapper

    anyOf(Mapping(required=[condition, value]), Mapping(required=[condition, expr]))
    """
    _schema = {'$ref': '#/definitions/ConditionalAxisProperty<(Align|null)>'}

    def __init__(self, *args, **kwds):
        super(ConditionalAxisPropertyAlignnull, self).__init__(*args, **kwds)


class ConditionalAxisPropertyColornull(VegaLiteSchema):
    """ConditionalAxisPropertyColornull schema wrapper

    anyOf(Mapping(required=[condition, value]), Mapping(required=[condition, expr]))
    """
    _schema = {'$ref': '#/definitions/ConditionalAxisProperty<(Color|null)>'}

    def __init__(self, *args, **kwds):
        super(ConditionalAxisPropertyColornull, self).__init__(*args, **kwds)


class ConditionalAxisPropertyFontStylenull(VegaLiteSchema):
    """ConditionalAxisPropertyFontStylenull schema wrapper

    anyOf(Mapping(required=[condition, value]), Mapping(required=[condition, expr]))
    """
    _schema = {'$ref': '#/definitions/ConditionalAxisProperty<(FontStyle|null)>'}

    def __init__(self, *args, **kwds):
        super(ConditionalAxisPropertyFontStylenull, self).__init__(*args, **kwds)


class ConditionalAxisPropertyFontWeightnull(VegaLiteSchema):
    """ConditionalAxisPropertyFontWeightnull schema wrapper

    anyOf(Mapping(required=[condition, value]), Mapping(required=[condition, expr]))
    """
    _schema = {'$ref': '#/definitions/ConditionalAxisProperty<(FontWeight|null)>'}

    def __init__(self, *args, **kwds):
        super(ConditionalAxisPropertyFontWeightnull, self).__init__(*args, **kwds)


class ConditionalAxisPropertyTextBaselinenull(VegaLiteSchema):
    """ConditionalAxisPropertyTextBaselinenull schema wrapper

    anyOf(Mapping(required=[condition, value]), Mapping(required=[condition, expr]))
    """
    _schema = {'$ref': '#/definitions/ConditionalAxisProperty<(TextBaseline|null)>'}

    def __init__(self, *args, **kwds):
        super(ConditionalAxisPropertyTextBaselinenull, self).__init__(*args, **kwds)


class ConditionalAxisPropertynumberArraynull(VegaLiteSchema):
    """ConditionalAxisPropertynumberArraynull schema wrapper

    anyOf(Mapping(required=[condition, value]), Mapping(required=[condition, expr]))
    """
    _schema = {'$ref': '#/definitions/ConditionalAxisProperty<(number[]|null)>'}

    def __init__(self, *args, **kwds):
        super(ConditionalAxisPropertynumberArraynull, self).__init__(*args, **kwds)


class ConditionalAxisPropertynumbernull(VegaLiteSchema):
    """ConditionalAxisPropertynumbernull schema wrapper

    anyOf(Mapping(required=[condition, value]), Mapping(required=[condition, expr]))
    """
    _schema = {'$ref': '#/definitions/ConditionalAxisProperty<(number|null)>'}

    def __init__(self, *args, **kwds):
        super(ConditionalAxisPropertynumbernull, self).__init__(*args, **kwds)


class ConditionalAxisPropertystringnull(VegaLiteSchema):
    """ConditionalAxisPropertystringnull schema wrapper

    anyOf(Mapping(required=[condition, value]), Mapping(required=[condition, expr]))
    """
    _schema = {'$ref': '#/definitions/ConditionalAxisProperty<(string|null)>'}

    def __init__(self, *args, **kwds):
        super(ConditionalAxisPropertystringnull, self).__init__(*args, **kwds)


class ConditionalAxisString(VegaLiteSchema):
    """ConditionalAxisString schema wrapper

    anyOf(Mapping(required=[condition, value]), Mapping(required=[condition, expr]))
    """
    _schema = {'$ref': '#/definitions/ConditionalAxisString'}

    def __init__(self, *args, **kwds):
        super(ConditionalAxisString, self).__init__(*args, **kwds)


class ConditionalMarkPropFieldOrDatumDef(VegaLiteSchema):
    """ConditionalMarkPropFieldOrDatumDef schema wrapper

    anyOf(:class:`ConditionalPredicateMarkPropFieldOrDatumDef`,
    :class:`ConditionalSelectionMarkPropFieldOrDatumDef`)
    """
    _schema = {'$ref': '#/definitions/ConditionalMarkPropFieldOrDatumDef'}

    def __init__(self, *args, **kwds):
        super(ConditionalMarkPropFieldOrDatumDef, self).__init__(*args, **kwds)


class ConditionalMarkPropFieldOrDatumDefTypeForShape(VegaLiteSchema):
    """ConditionalMarkPropFieldOrDatumDefTypeForShape schema wrapper

    anyOf(:class:`ConditionalPredicateMarkPropFieldOrDatumDefTypeForShape`,
    :class:`ConditionalSelectionMarkPropFieldOrDatumDefTypeForShape`)
    """
    _schema = {'$ref': '#/definitions/ConditionalMarkPropFieldOrDatumDef<TypeForShape>'}

    def __init__(self, *args, **kwds):
        super(ConditionalMarkPropFieldOrDatumDefTypeForShape, self).__init__(*args, **kwds)


class ConditionalPredicateMarkPropFieldOrDatumDef(ConditionalMarkPropFieldOrDatumDef):
    """ConditionalPredicateMarkPropFieldOrDatumDef schema wrapper

    anyOf(Mapping(required=[test]), Mapping(required=[test]))
    """
    _schema = {'$ref': '#/definitions/ConditionalPredicate<MarkPropFieldOrDatumDef>'}

    def __init__(self, *args, **kwds):
        super(ConditionalPredicateMarkPropFieldOrDatumDef, self).__init__(*args, **kwds)


class ConditionalPredicateMarkPropFieldOrDatumDefTypeForShape(ConditionalMarkPropFieldOrDatumDefTypeForShape):
    """ConditionalPredicateMarkPropFieldOrDatumDefTypeForShape schema wrapper

    anyOf(Mapping(required=[test]), Mapping(required=[test]))
    """
    _schema = {'$ref': '#/definitions/ConditionalPredicate<MarkPropFieldOrDatumDef<TypeForShape>>'}

    def __init__(self, *args, **kwds):
        super(ConditionalPredicateMarkPropFieldOrDatumDefTypeForShape, self).__init__(*args, **kwds)


class ConditionalPredicateValueDefAlignnullExprRef(VegaLiteSchema):
    """ConditionalPredicateValueDefAlignnullExprRef schema wrapper

    anyOf(Mapping(required=[test, value]), Mapping(required=[expr, test]))
    """
    _schema = {'$ref': '#/definitions/ConditionalPredicate<(ValueDef<(Align|null)>|ExprRef)>'}

    def __init__(self, *args, **kwds):
        super(ConditionalPredicateValueDefAlignnullExprRef, self).__init__(*args, **kwds)


class ConditionalPredicateValueDefColornullExprRef(VegaLiteSchema):
    """ConditionalPredicateValueDefColornullExprRef schema wrapper

    anyOf(Mapping(required=[test, value]), Mapping(required=[expr, test]))
    """
    _schema = {'$ref': '#/definitions/ConditionalPredicate<(ValueDef<(Color|null)>|ExprRef)>'}

    def __init__(self, *args, **kwds):
        super(ConditionalPredicateValueDefColornullExprRef, self).__init__(*args, **kwds)


class ConditionalPredicateValueDefFontStylenullExprRef(VegaLiteSchema):
    """ConditionalPredicateValueDefFontStylenullExprRef schema wrapper

    anyOf(Mapping(required=[test, value]), Mapping(required=[expr, test]))
    """
    _schema = {'$ref': '#/definitions/ConditionalPredicate<(ValueDef<(FontStyle|null)>|ExprRef)>'}

    def __init__(self, *args, **kwds):
        super(ConditionalPredicateValueDefFontStylenullExprRef, self).__init__(*args, **kwds)


class ConditionalPredicateValueDefFontWeightnullExprRef(VegaLiteSchema):
    """ConditionalPredicateValueDefFontWeightnullExprRef schema wrapper

    anyOf(Mapping(required=[test, value]), Mapping(required=[expr, test]))
    """
    _schema = {'$ref': '#/definitions/ConditionalPredicate<(ValueDef<(FontWeight|null)>|ExprRef)>'}

    def __init__(self, *args, **kwds):
        super(ConditionalPredicateValueDefFontWeightnullExprRef, self).__init__(*args, **kwds)


class ConditionalPredicateValueDefTextBaselinenullExprRef(VegaLiteSchema):
    """ConditionalPredicateValueDefTextBaselinenullExprRef schema wrapper

    anyOf(Mapping(required=[test, value]), Mapping(required=[expr, test]))
    """
    _schema = {'$ref': '#/definitions/ConditionalPredicate<(ValueDef<(TextBaseline|null)>|ExprRef)>'}

    def __init__(self, *args, **kwds):
        super(ConditionalPredicateValueDefTextBaselinenullExprRef, self).__init__(*args, **kwds)


class ConditionalPredicateValueDefnumberArraynullExprRef(VegaLiteSchema):
    """ConditionalPredicateValueDefnumberArraynullExprRef schema wrapper

    anyOf(Mapping(required=[test, value]), Mapping(required=[expr, test]))
    """
    _schema = {'$ref': '#/definitions/ConditionalPredicate<(ValueDef<(number[]|null)>|ExprRef)>'}

    def __init__(self, *args, **kwds):
        super(ConditionalPredicateValueDefnumberArraynullExprRef, self).__init__(*args, **kwds)


class ConditionalPredicateValueDefnumbernullExprRef(VegaLiteSchema):
    """ConditionalPredicateValueDefnumbernullExprRef schema wrapper

    anyOf(Mapping(required=[test, value]), Mapping(required=[expr, test]))
    """
    _schema = {'$ref': '#/definitions/ConditionalPredicate<(ValueDef<(number|null)>|ExprRef)>'}

    def __init__(self, *args, **kwds):
        super(ConditionalPredicateValueDefnumbernullExprRef, self).__init__(*args, **kwds)


class ConditionalSelectionMarkPropFieldOrDatumDef(ConditionalMarkPropFieldOrDatumDef):
    """ConditionalSelectionMarkPropFieldOrDatumDef schema wrapper

    anyOf(Mapping(required=[selection]), Mapping(required=[selection]))
    """
    _schema = {'$ref': '#/definitions/ConditionalSelection<MarkPropFieldOrDatumDef>'}

    def __init__(self, *args, **kwds):
        super(ConditionalSelectionMarkPropFieldOrDatumDef, self).__init__(*args, **kwds)


class ConditionalSelectionMarkPropFieldOrDatumDefTypeForShape(ConditionalMarkPropFieldOrDatumDefTypeForShape):
    """ConditionalSelectionMarkPropFieldOrDatumDefTypeForShape schema wrapper

    anyOf(Mapping(required=[selection]), Mapping(required=[selection]))
    """
    _schema = {'$ref': '#/definitions/ConditionalSelection<MarkPropFieldOrDatumDef<TypeForShape>>'}

    def __init__(self, *args, **kwds):
        super(ConditionalSelectionMarkPropFieldOrDatumDefTypeForShape, self).__init__(*args, **kwds)


class ConditionalStringFieldDef(VegaLiteSchema):
    """ConditionalStringFieldDef schema wrapper

    anyOf(:class:`ConditionalPredicateStringFieldDef`,
    :class:`ConditionalSelectionStringFieldDef`)
    """
    _schema = {'$ref': '#/definitions/ConditionalStringFieldDef'}

    def __init__(self, *args, **kwds):
        super(ConditionalStringFieldDef, self).__init__(*args, **kwds)


class ConditionalPredicateStringFieldDef(ConditionalStringFieldDef):
    """ConditionalPredicateStringFieldDef schema wrapper

    Mapping(required=[test])

    Attributes
    ----------

    test : :class:`PredicateComposition`
        Predicate for triggering the condition
    aggregate : :class:`Aggregate`
        Aggregation function for the field (e.g., ``"mean"``, ``"sum"``, ``"median"``,
        ``"min"``, ``"max"``, ``"count"`` ).

        **Default value:** ``undefined`` (None)

        **See also:** `aggregate <https://vega.github.io/vega-lite/docs/aggregate.html>`__
        documentation.
    band : float
        For rect-based marks ( ``rect``, ``bar``, and ``image`` ), mark size relative to
        bandwidth of `band scales
        <https://vega.github.io/vega-lite/docs/scale.html#band>`__, bins or time units. If
        set to ``1``, the mark size is set to the bandwidth, the bin interval, or the time
        unit interval. If set to ``0.5``, the mark size is half of the bandwidth or the time
        unit interval.

        For other marks, relative position on a band of a stacked, binned, time unit or band
        scale. If set to ``0``, the marks will be positioned at the beginning of the band.
        If set to ``0.5``, the marks will be positioned in the middle of the band.
    bin : anyOf(boolean, :class:`BinParams`, string, None)
        A flag for binning a ``quantitative`` field, `an object defining binning parameters
        <https://vega.github.io/vega-lite/docs/bin.html#params>`__, or indicating that the
        data for ``x`` or ``y`` channel are binned before they are imported into Vega-Lite (
        ``"binned"`` ).


        If ``true``, default `binning parameters
        <https://vega.github.io/vega-lite/docs/bin.html>`__ will be applied.

        If ``"binned"``, this indicates that the data for the ``x`` (or ``y`` ) channel are
        already binned. You can map the bin-start field to ``x`` (or ``y`` ) and the bin-end
        field to ``x2`` (or ``y2`` ). The scale and axis will be formatted similar to
        binning in Vega-Lite.  To adjust the axis ticks based on the bin step, you can also
        set the axis's `tickMinStep
        <https://vega.github.io/vega-lite/docs/axis.html#ticks>`__ property.

        **Default value:** ``false``

        **See also:** `bin <https://vega.github.io/vega-lite/docs/bin.html>`__
        documentation.
    field : :class:`Field`
        **Required.** A string defining the name of the field from which to pull a data
        value or an object defining iterated values from the `repeat
        <https://vega.github.io/vega-lite/docs/repeat.html>`__ operator.

        **See also:** `field <https://vega.github.io/vega-lite/docs/field.html>`__
        documentation.

        **Notes:** 1)  Dots ( ``.`` ) and brackets ( ``[`` and ``]`` ) can be used to access
        nested objects (e.g., ``"field": "foo.bar"`` and ``"field": "foo['bar']"`` ). If
        field names contain dots or brackets but are not nested, you can use ``\\`` to
        escape dots and brackets (e.g., ``"a\\.b"`` and ``"a\\[0\\]"`` ). See more details
        about escaping in the `field documentation
        <https://vega.github.io/vega-lite/docs/field.html>`__. 2) ``field`` is not required
        if ``aggregate`` is ``count``.
    format : anyOf(string, :class:`Dictunknown`)
        When used with the default ``"number"`` and ``"time"`` format type, the text
        formatting pattern for labels of guides (axes, legends, headers) and text marks.


        * If the format type is ``"number"`` (e.g., for quantitative fields), this is D3's
          `number format pattern <https://github.com/d3/d3-format#locale_format>`__. - If
          the format type is ``"time"`` (e.g., for temporal fields), this is D3's `time
          format pattern <https://github.com/d3/d3-time-format#locale_format>`__.

        See the `format documentation <https://vega.github.io/vega-lite/docs/format.html>`__
        for more examples.

        When used with a `custom formatType
        <https://vega.github.io/vega-lite/docs/config.html#custom-format-type>`__, this
        value will be passed as ``format`` alongside ``datum.value`` to the registered
        function.

        **Default value:**  Derived from `numberFormat
        <https://vega.github.io/vega-lite/docs/config.html#format>`__ config for number
        format and from `timeFormat
        <https://vega.github.io/vega-lite/docs/config.html#format>`__ config for time
        format.
    formatType : string
        The format type for labels. One of ``"number"``, ``"time"``, or a `registered custom
        format type
        <https://vega.github.io/vega-lite/docs/config.html#custom-format-type>`__.

        **Default value:** - ``"time"`` for temporal fields and ordinal and nominal fields
        with ``timeUnit``. - ``"number"`` for quantitative fields as well as ordinal and
        nominal fields without ``timeUnit``.
    labelExpr : string
        `Vega expression <https://vega.github.io/vega/docs/expressions/>`__ for customizing
        labels text.

        **Note:** The label text and value can be assessed via the ``label`` and ``value``
        properties of the axis's backing ``datum`` object.
    timeUnit : anyOf(:class:`TimeUnit`, :class:`TimeUnitParams`)
        Time unit (e.g., ``year``, ``yearmonth``, ``month``, ``hours`` ) for a temporal
        field. or `a temporal field that gets casted as ordinal
        <https://vega.github.io/vega-lite/docs/type.html#cast>`__.

        **Default value:** ``undefined`` (None)

        **See also:** `timeUnit <https://vega.github.io/vega-lite/docs/timeunit.html>`__
        documentation.
    title : anyOf(:class:`Text`, None)
        A title for the field. If ``null``, the title will be removed.

        **Default value:**  derived from the field's name and transformation function (
        ``aggregate``, ``bin`` and ``timeUnit`` ). If the field has an aggregate function,
        the function is displayed as part of the title (e.g., ``"Sum of Profit"`` ). If the
        field is binned or has a time unit applied, the applied function is shown in
        parentheses (e.g., ``"Profit (binned)"``, ``"Transaction Date (year-month)"`` ).
        Otherwise, the title is simply the field name.

        **Notes** :

        1) You can customize the default field title format by providing the `fieldTitle
        <https://vega.github.io/vega-lite/docs/config.html#top-level-config>`__ property in
        the `config <https://vega.github.io/vega-lite/docs/config.html>`__ or `fieldTitle
        function via the compile function's options
        <https://vega.github.io/vega-lite/docs/compile.html#field-title>`__.

        2) If both field definition's ``title`` and axis, header, or legend ``title`` are
        defined, axis/header/legend title will be used.
    type : :class:`StandardType`
        The type of measurement ( ``"quantitative"``, ``"temporal"``, ``"ordinal"``, or
        ``"nominal"`` ) for the encoded field or constant value ( ``datum`` ). It can also
        be a ``"geojson"`` type for encoding `'geoshape'
        <https://vega.github.io/vega-lite/docs/geoshape.html>`__.

        Vega-Lite automatically infers data types in many cases as discussed below. However,
        type is required for a field if: (1) the field is not nominal and the field encoding
        has no specified ``aggregate`` (except ``argmin`` and ``argmax`` ), ``bin``, scale
        type, custom ``sort`` order, nor ``timeUnit`` or (2) if you wish to use an ordinal
        scale for a field with ``bin`` or ``timeUnit``.

        **Default value:**

        1) For a data ``field``, ``"nominal"`` is the default data type unless the field
        encoding has ``aggregate``, ``channel``, ``bin``, scale type, ``sort``, or
        ``timeUnit`` that satisfies the following criteria: - ``"quantitative"`` is the
        default type if (1) the encoded field contains ``bin`` or ``aggregate`` except
        ``"argmin"`` and ``"argmax"``, (2) the encoding channel is ``latitude`` or
        ``longitude`` channel or (3) if the specified scale type is `a quantitative scale
        <https://vega.github.io/vega-lite/docs/scale.html#type>`__. - ``"temporal"`` is the
        default type if (1) the encoded field contains ``timeUnit`` or (2) the specified
        scale type is a time or utc scale - ``ordinal""`` is the default type if (1) the
        encoded field contains a `custom sort order
        <https://vega.github.io/vega-lite/docs/sort.html#specifying-custom-sort-order>`__,
        (2) the specified scale type is an ordinal/point/band scale, or (3) the encoding
        channel is ``order``.

        2) For a constant value in data domain ( ``datum`` ): - ``"quantitative"`` if the
        datum is a number - ``"nominal"`` if the datum is a string - ``"temporal"`` if the
        datum is `a date time object
        <https://vega.github.io/vega-lite/docs/datetime.html>`__

        **Note:** - Data ``type`` describes the semantics of the data rather than the
        primitive data types (number, string, etc.). The same primitive data type can have
        different types of measurement. For example, numeric data can represent
        quantitative, ordinal, or nominal data. - Data values for a temporal field can be
        either a date-time string (e.g., ``"2015-03-07 12:32:17"``, ``"17:01"``,
        ``"2015-03-16"``. ``"2015"`` ) or a timestamp number (e.g., ``1552199579097`` ). -
        When using with `bin <https://vega.github.io/vega-lite/docs/bin.html>`__, the
        ``type`` property can be either ``"quantitative"`` (for using a linear bin scale) or
        `"ordinal" (for using an ordinal bin scale)
        <https://vega.github.io/vega-lite/docs/type.html#cast-bin>`__. - When using with
        `timeUnit <https://vega.github.io/vega-lite/docs/timeunit.html>`__, the ``type``
        property can be either ``"temporal"`` (default, for using a temporal scale) or
        `"ordinal" (for using an ordinal scale)
        <https://vega.github.io/vega-lite/docs/type.html#cast-bin>`__. - When using with
        `aggregate <https://vega.github.io/vega-lite/docs/aggregate.html>`__, the ``type``
        property refers to the post-aggregation data type. For example, we can calculate
        count ``distinct`` of a categorical field ``"cat"`` using ``{"aggregate":
        "distinct", "field": "cat"}``. The ``"type"`` of the aggregate output is
        ``"quantitative"``. - Secondary channels (e.g., ``x2``, ``y2``, ``xError``,
        ``yError`` ) do not have ``type`` as they must have exactly the same type as their
        primary channels (e.g., ``x``, ``y`` ).

        **See also:** `type <https://vega.github.io/vega-lite/docs/type.html>`__
        documentation.
    """
    _schema = {'$ref': '#/definitions/ConditionalPredicate<StringFieldDef>'}

    def __init__(self, test=Undefined, aggregate=Undefined, band=Undefined, bin=Undefined,
                 field=Undefined, format=Undefined, formatType=Undefined, labelExpr=Undefined,
                 timeUnit=Undefined, title=Undefined, type=Undefined, **kwds):
        super(ConditionalPredicateStringFieldDef, self).__init__(test=test, aggregate=aggregate,
                                                                 band=band, bin=bin, field=field,
                                                                 format=format, formatType=formatType,
                                                                 labelExpr=labelExpr, timeUnit=timeUnit,
                                                                 title=title, type=type, **kwds)


class ConditionalSelectionStringFieldDef(ConditionalStringFieldDef):
    """ConditionalSelectionStringFieldDef schema wrapper

    Mapping(required=[selection])

    Attributes
    ----------

    selection : :class:`SelectionComposition`
        A `selection name <https://vega.github.io/vega-lite/docs/selection.html>`__, or a
        series of `composed selections
        <https://vega.github.io/vega-lite/docs/selection.html#compose>`__.
    aggregate : :class:`Aggregate`
        Aggregation function for the field (e.g., ``"mean"``, ``"sum"``, ``"median"``,
        ``"min"``, ``"max"``, ``"count"`` ).

        **Default value:** ``undefined`` (None)

        **See also:** `aggregate <https://vega.github.io/vega-lite/docs/aggregate.html>`__
        documentation.
    band : float
        For rect-based marks ( ``rect``, ``bar``, and ``image`` ), mark size relative to
        bandwidth of `band scales
        <https://vega.github.io/vega-lite/docs/scale.html#band>`__, bins or time units. If
        set to ``1``, the mark size is set to the bandwidth, the bin interval, or the time
        unit interval. If set to ``0.5``, the mark size is half of the bandwidth or the time
        unit interval.

        For other marks, relative position on a band of a stacked, binned, time unit or band
        scale. If set to ``0``, the marks will be positioned at the beginning of the band.
        If set to ``0.5``, the marks will be positioned in the middle of the band.
    bin : anyOf(boolean, :class:`BinParams`, string, None)
        A flag for binning a ``quantitative`` field, `an object defining binning parameters
        <https://vega.github.io/vega-lite/docs/bin.html#params>`__, or indicating that the
        data for ``x`` or ``y`` channel are binned before they are imported into Vega-Lite (
        ``"binned"`` ).


        If ``true``, default `binning parameters
        <https://vega.github.io/vega-lite/docs/bin.html>`__ will be applied.

        If ``"binned"``, this indicates that the data for the ``x`` (or ``y`` ) channel are
        already binned. You can map the bin-start field to ``x`` (or ``y`` ) and the bin-end
        field to ``x2`` (or ``y2`` ). The scale and axis will be formatted similar to
        binning in Vega-Lite.  To adjust the axis ticks based on the bin step, you can also
        set the axis's `tickMinStep
        <https://vega.github.io/vega-lite/docs/axis.html#ticks>`__ property.

        **Default value:** ``false``

        **See also:** `bin <https://vega.github.io/vega-lite/docs/bin.html>`__
        documentation.
    field : :class:`Field`
        **Required.** A string defining the name of the field from which to pull a data
        value or an object defining iterated values from the `repeat
        <https://vega.github.io/vega-lite/docs/repeat.html>`__ operator.

        **See also:** `field <https://vega.github.io/vega-lite/docs/field.html>`__
        documentation.

        **Notes:** 1)  Dots ( ``.`` ) and brackets ( ``[`` and ``]`` ) can be used to access
        nested objects (e.g., ``"field": "foo.bar"`` and ``"field": "foo['bar']"`` ). If
        field names contain dots or brackets but are not nested, you can use ``\\`` to
        escape dots and brackets (e.g., ``"a\\.b"`` and ``"a\\[0\\]"`` ). See more details
        about escaping in the `field documentation
        <https://vega.github.io/vega-lite/docs/field.html>`__. 2) ``field`` is not required
        if ``aggregate`` is ``count``.
    format : anyOf(string, :class:`Dictunknown`)
        When used with the default ``"number"`` and ``"time"`` format type, the text
        formatting pattern for labels of guides (axes, legends, headers) and text marks.


        * If the format type is ``"number"`` (e.g., for quantitative fields), this is D3's
          `number format pattern <https://github.com/d3/d3-format#locale_format>`__. - If
          the format type is ``"time"`` (e.g., for temporal fields), this is D3's `time
          format pattern <https://github.com/d3/d3-time-format#locale_format>`__.

        See the `format documentation <https://vega.github.io/vega-lite/docs/format.html>`__
        for more examples.

        When used with a `custom formatType
        <https://vega.github.io/vega-lite/docs/config.html#custom-format-type>`__, this
        value will be passed as ``format`` alongside ``datum.value`` to the registered
        function.

        **Default value:**  Derived from `numberFormat
        <https://vega.github.io/vega-lite/docs/config.html#format>`__ config for number
        format and from `timeFormat
        <https://vega.github.io/vega-lite/docs/config.html#format>`__ config for time
        format.
    formatType : string
        The format type for labels. One of ``"number"``, ``"time"``, or a `registered custom
        format type
        <https://vega.github.io/vega-lite/docs/config.html#custom-format-type>`__.

        **Default value:** - ``"time"`` for temporal fields and ordinal and nominal fields
        with ``timeUnit``. - ``"number"`` for quantitative fields as well as ordinal and
        nominal fields without ``timeUnit``.
    labelExpr : string
        `Vega expression <https://vega.github.io/vega/docs/expressions/>`__ for customizing
        labels text.

        **Note:** The label text and value can be assessed via the ``label`` and ``value``
        properties of the axis's backing ``datum`` object.
    timeUnit : anyOf(:class:`TimeUnit`, :class:`TimeUnitParams`)
        Time unit (e.g., ``year``, ``yearmonth``, ``month``, ``hours`` ) for a temporal
        field. or `a temporal field that gets casted as ordinal
        <https://vega.github.io/vega-lite/docs/type.html#cast>`__.

        **Default value:** ``undefined`` (None)

        **See also:** `timeUnit <https://vega.github.io/vega-lite/docs/timeunit.html>`__
        documentation.
    title : anyOf(:class:`Text`, None)
        A title for the field. If ``null``, the title will be removed.

        **Default value:**  derived from the field's name and transformation function (
        ``aggregate``, ``bin`` and ``timeUnit`` ). If the field has an aggregate function,
        the function is displayed as part of the title (e.g., ``"Sum of Profit"`` ). If the
        field is binned or has a time unit applied, the applied function is shown in
        parentheses (e.g., ``"Profit (binned)"``, ``"Transaction Date (year-month)"`` ).
        Otherwise, the title is simply the field name.

        **Notes** :

        1) You can customize the default field title format by providing the `fieldTitle
        <https://vega.github.io/vega-lite/docs/config.html#top-level-config>`__ property in
        the `config <https://vega.github.io/vega-lite/docs/config.html>`__ or `fieldTitle
        function via the compile function's options
        <https://vega.github.io/vega-lite/docs/compile.html#field-title>`__.

        2) If both field definition's ``title`` and axis, header, or legend ``title`` are
        defined, axis/header/legend title will be used.
    type : :class:`StandardType`
        The type of measurement ( ``"quantitative"``, ``"temporal"``, ``"ordinal"``, or
        ``"nominal"`` ) for the encoded field or constant value ( ``datum`` ). It can also
        be a ``"geojson"`` type for encoding `'geoshape'
        <https://vega.github.io/vega-lite/docs/geoshape.html>`__.

        Vega-Lite automatically infers data types in many cases as discussed below. However,
        type is required for a field if: (1) the field is not nominal and the field encoding
        has no specified ``aggregate`` (except ``argmin`` and ``argmax`` ), ``bin``, scale
        type, custom ``sort`` order, nor ``timeUnit`` or (2) if you wish to use an ordinal
        scale for a field with ``bin`` or ``timeUnit``.

        **Default value:**

        1) For a data ``field``, ``"nominal"`` is the default data type unless the field
        encoding has ``aggregate``, ``channel``, ``bin``, scale type, ``sort``, or
        ``timeUnit`` that satisfies the following criteria: - ``"quantitative"`` is the
        default type if (1) the encoded field contains ``bin`` or ``aggregate`` except
        ``"argmin"`` and ``"argmax"``, (2) the encoding channel is ``latitude`` or
        ``longitude`` channel or (3) if the specified scale type is `a quantitative scale
        <https://vega.github.io/vega-lite/docs/scale.html#type>`__. - ``"temporal"`` is the
        default type if (1) the encoded field contains ``timeUnit`` or (2) the specified
        scale type is a time or utc scale - ``ordinal""`` is the default type if (1) the
        encoded field contains a `custom sort order
        <https://vega.github.io/vega-lite/docs/sort.html#specifying-custom-sort-order>`__,
        (2) the specified scale type is an ordinal/point/band scale, or (3) the encoding
        channel is ``order``.

        2) For a constant value in data domain ( ``datum`` ): - ``"quantitative"`` if the
        datum is a number - ``"nominal"`` if the datum is a string - ``"temporal"`` if the
        datum is `a date time object
        <https://vega.github.io/vega-lite/docs/datetime.html>`__

        **Note:** - Data ``type`` describes the semantics of the data rather than the
        primitive data types (number, string, etc.). The same primitive data type can have
        different types of measurement. For example, numeric data can represent
        quantitative, ordinal, or nominal data. - Data values for a temporal field can be
        either a date-time string (e.g., ``"2015-03-07 12:32:17"``, ``"17:01"``,
        ``"2015-03-16"``. ``"2015"`` ) or a timestamp number (e.g., ``1552199579097`` ). -
        When using with `bin <https://vega.github.io/vega-lite/docs/bin.html>`__, the
        ``type`` property can be either ``"quantitative"`` (for using a linear bin scale) or
        `"ordinal" (for using an ordinal bin scale)
        <https://vega.github.io/vega-lite/docs/type.html#cast-bin>`__. - When using with
        `timeUnit <https://vega.github.io/vega-lite/docs/timeunit.html>`__, the ``type``
        property can be either ``"temporal"`` (default, for using a temporal scale) or
        `"ordinal" (for using an ordinal scale)
        <https://vega.github.io/vega-lite/docs/type.html#cast-bin>`__. - When using with
        `aggregate <https://vega.github.io/vega-lite/docs/aggregate.html>`__, the ``type``
        property refers to the post-aggregation data type. For example, we can calculate
        count ``distinct`` of a categorical field ``"cat"`` using ``{"aggregate":
        "distinct", "field": "cat"}``. The ``"type"`` of the aggregate output is
        ``"quantitative"``. - Secondary channels (e.g., ``x2``, ``y2``, ``xError``,
        ``yError`` ) do not have ``type`` as they must have exactly the same type as their
        primary channels (e.g., ``x``, ``y`` ).

        **See also:** `type <https://vega.github.io/vega-lite/docs/type.html>`__
        documentation.
    """
    _schema = {'$ref': '#/definitions/ConditionalSelection<StringFieldDef>'}

    def __init__(self, selection=Undefined, aggregate=Undefined, band=Undefined, bin=Undefined,
                 field=Undefined, format=Undefined, formatType=Undefined, labelExpr=Undefined,
                 timeUnit=Undefined, title=Undefined, type=Undefined, **kwds):
        super(ConditionalSelectionStringFieldDef, self).__init__(selection=selection,
                                                                 aggregate=aggregate, band=band,
                                                                 bin=bin, field=field, format=format,
                                                                 formatType=formatType,
                                                                 labelExpr=labelExpr, timeUnit=timeUnit,
                                                                 title=title, type=type, **kwds)


class ConditionalValueDefGradientstringnullExprRef(VegaLiteSchema):
    """ConditionalValueDefGradientstringnullExprRef schema wrapper

    anyOf(:class:`ConditionalPredicateValueDefGradientstringnullExprRef`,
    :class:`ConditionalSelectionValueDefGradientstringnullExprRef`)
    """
    _schema = {'$ref': '#/definitions/ConditionalValueDef<(Gradient|string|null|ExprRef)>'}

    def __init__(self, *args, **kwds):
        super(ConditionalValueDefGradientstringnullExprRef, self).__init__(*args, **kwds)


class ConditionalPredicateValueDefGradientstringnullExprRef(ConditionalValueDefGradientstringnullExprRef):
    """ConditionalPredicateValueDefGradientstringnullExprRef schema wrapper

    Mapping(required=[test, value])

    Attributes
    ----------

    test : :class:`PredicateComposition`
        Predicate for triggering the condition
    value : anyOf(:class:`Gradient`, string, None, :class:`ExprRef`)
        A constant value in visual domain (e.g., ``"red"`` / ``"#0099ff"`` / `gradient
        definition <https://vega.github.io/vega-lite/docs/types.html#gradient>`__ for color,
        values between ``0`` to ``1`` for opacity).
    """
    _schema = {'$ref': '#/definitions/ConditionalPredicate<ValueDef<(Gradient|string|null|ExprRef)>>'}

    def __init__(self, test=Undefined, value=Undefined, **kwds):
        super(ConditionalPredicateValueDefGradientstringnullExprRef, self).__init__(test=test,
                                                                                    value=value, **kwds)


class ConditionalSelectionValueDefGradientstringnullExprRef(ConditionalValueDefGradientstringnullExprRef):
    """ConditionalSelectionValueDefGradientstringnullExprRef schema wrapper

    Mapping(required=[selection, value])

    Attributes
    ----------

    selection : :class:`SelectionComposition`
        A `selection name <https://vega.github.io/vega-lite/docs/selection.html>`__, or a
        series of `composed selections
        <https://vega.github.io/vega-lite/docs/selection.html#compose>`__.
    value : anyOf(:class:`Gradient`, string, None, :class:`ExprRef`)
        A constant value in visual domain (e.g., ``"red"`` / ``"#0099ff"`` / `gradient
        definition <https://vega.github.io/vega-lite/docs/types.html#gradient>`__ for color,
        values between ``0`` to ``1`` for opacity).
    """
    _schema = {'$ref': '#/definitions/ConditionalSelection<ValueDef<(Gradient|string|null|ExprRef)>>'}

    def __init__(self, selection=Undefined, value=Undefined, **kwds):
        super(ConditionalSelectionValueDefGradientstringnullExprRef, self).__init__(selection=selection,
                                                                                    value=value, **kwds)


class ConditionalValueDefTextExprRef(VegaLiteSchema):
    """ConditionalValueDefTextExprRef schema wrapper

    anyOf(:class:`ConditionalPredicateValueDefTextExprRef`,
    :class:`ConditionalSelectionValueDefTextExprRef`)
    """
    _schema = {'$ref': '#/definitions/ConditionalValueDef<(Text|ExprRef)>'}

    def __init__(self, *args, **kwds):
        super(ConditionalValueDefTextExprRef, self).__init__(*args, **kwds)


class ConditionalPredicateValueDefTextExprRef(ConditionalValueDefTextExprRef):
    """ConditionalPredicateValueDefTextExprRef schema wrapper

    Mapping(required=[test, value])

    Attributes
    ----------

    test : :class:`PredicateComposition`
        Predicate for triggering the condition
    value : anyOf(:class:`Text`, :class:`ExprRef`)
        A constant value in visual domain (e.g., ``"red"`` / ``"#0099ff"`` / `gradient
        definition <https://vega.github.io/vega-lite/docs/types.html#gradient>`__ for color,
        values between ``0`` to ``1`` for opacity).
    """
    _schema = {'$ref': '#/definitions/ConditionalPredicate<ValueDef<(Text|ExprRef)>>'}

    def __init__(self, test=Undefined, value=Undefined, **kwds):
        super(ConditionalPredicateValueDefTextExprRef, self).__init__(test=test, value=value, **kwds)


class ConditionalSelectionValueDefTextExprRef(ConditionalValueDefTextExprRef):
    """ConditionalSelectionValueDefTextExprRef schema wrapper

    Mapping(required=[selection, value])

    Attributes
    ----------

    selection : :class:`SelectionComposition`
        A `selection name <https://vega.github.io/vega-lite/docs/selection.html>`__, or a
        series of `composed selections
        <https://vega.github.io/vega-lite/docs/selection.html#compose>`__.
    value : anyOf(:class:`Text`, :class:`ExprRef`)
        A constant value in visual domain (e.g., ``"red"`` / ``"#0099ff"`` / `gradient
        definition <https://vega.github.io/vega-lite/docs/types.html#gradient>`__ for color,
        values between ``0`` to ``1`` for opacity).
    """
    _schema = {'$ref': '#/definitions/ConditionalSelection<ValueDef<(Text|ExprRef)>>'}

    def __init__(self, selection=Undefined, value=Undefined, **kwds):
        super(ConditionalSelectionValueDefTextExprRef, self).__init__(selection=selection, value=value,
                                                                      **kwds)


class ConditionalValueDefnumber(VegaLiteSchema):
    """ConditionalValueDefnumber schema wrapper

    anyOf(:class:`ConditionalPredicateValueDefnumber`,
    :class:`ConditionalSelectionValueDefnumber`)
    """
    _schema = {'$ref': '#/definitions/ConditionalValueDef<number>'}

    def __init__(self, *args, **kwds):
        super(ConditionalValueDefnumber, self).__init__(*args, **kwds)


class ConditionalPredicateValueDefnumber(ConditionalValueDefnumber):
    """ConditionalPredicateValueDefnumber schema wrapper

    Mapping(required=[test, value])

    Attributes
    ----------

    test : :class:`PredicateComposition`
        Predicate for triggering the condition
    value : float
        A constant value in visual domain (e.g., ``"red"`` / ``"#0099ff"`` / `gradient
        definition <https://vega.github.io/vega-lite/docs/types.html#gradient>`__ for color,
        values between ``0`` to ``1`` for opacity).
    """
    _schema = {'$ref': '#/definitions/ConditionalPredicate<ValueDef<number>>'}

    def __init__(self, test=Undefined, value=Undefined, **kwds):
        super(ConditionalPredicateValueDefnumber, self).__init__(test=test, value=value, **kwds)


class ConditionalSelectionValueDefnumber(ConditionalValueDefnumber):
    """ConditionalSelectionValueDefnumber schema wrapper

    Mapping(required=[selection, value])

    Attributes
    ----------

    selection : :class:`SelectionComposition`
        A `selection name <https://vega.github.io/vega-lite/docs/selection.html>`__, or a
        series of `composed selections
        <https://vega.github.io/vega-lite/docs/selection.html#compose>`__.
    value : float
        A constant value in visual domain (e.g., ``"red"`` / ``"#0099ff"`` / `gradient
        definition <https://vega.github.io/vega-lite/docs/types.html#gradient>`__ for color,
        values between ``0`` to ``1`` for opacity).
    """
    _schema = {'$ref': '#/definitions/ConditionalSelection<ValueDef<number>>'}

    def __init__(self, selection=Undefined, value=Undefined, **kwds):
        super(ConditionalSelectionValueDefnumber, self).__init__(selection=selection, value=value,
                                                                 **kwds)


class ConditionalValueDefnumberArrayExprRef(VegaLiteSchema):
    """ConditionalValueDefnumberArrayExprRef schema wrapper

    anyOf(:class:`ConditionalPredicateValueDefnumberArrayExprRef`,
    :class:`ConditionalSelectionValueDefnumberArrayExprRef`)
    """
    _schema = {'$ref': '#/definitions/ConditionalValueDef<(number[]|ExprRef)>'}

    def __init__(self, *args, **kwds):
        super(ConditionalValueDefnumberArrayExprRef, self).__init__(*args, **kwds)


class ConditionalPredicateValueDefnumberArrayExprRef(ConditionalValueDefnumberArrayExprRef):
    """ConditionalPredicateValueDefnumberArrayExprRef schema wrapper

    Mapping(required=[test, value])

    Attributes
    ----------

    test : :class:`PredicateComposition`
        Predicate for triggering the condition
    value : anyOf(List(float), :class:`ExprRef`)
        A constant value in visual domain (e.g., ``"red"`` / ``"#0099ff"`` / `gradient
        definition <https://vega.github.io/vega-lite/docs/types.html#gradient>`__ for color,
        values between ``0`` to ``1`` for opacity).
    """
    _schema = {'$ref': '#/definitions/ConditionalPredicate<ValueDef<(number[]|ExprRef)>>'}

    def __init__(self, test=Undefined, value=Undefined, **kwds):
        super(ConditionalPredicateValueDefnumberArrayExprRef, self).__init__(test=test, value=value,
                                                                             **kwds)


class ConditionalSelectionValueDefnumberArrayExprRef(ConditionalValueDefnumberArrayExprRef):
    """ConditionalSelectionValueDefnumberArrayExprRef schema wrapper

    Mapping(required=[selection, value])

    Attributes
    ----------

    selection : :class:`SelectionComposition`
        A `selection name <https://vega.github.io/vega-lite/docs/selection.html>`__, or a
        series of `composed selections
        <https://vega.github.io/vega-lite/docs/selection.html#compose>`__.
    value : anyOf(List(float), :class:`ExprRef`)
        A constant value in visual domain (e.g., ``"red"`` / ``"#0099ff"`` / `gradient
        definition <https://vega.github.io/vega-lite/docs/types.html#gradient>`__ for color,
        values between ``0`` to ``1`` for opacity).
    """
    _schema = {'$ref': '#/definitions/ConditionalSelection<ValueDef<(number[]|ExprRef)>>'}

    def __init__(self, selection=Undefined, value=Undefined, **kwds):
        super(ConditionalSelectionValueDefnumberArrayExprRef, self).__init__(selection=selection,
                                                                             value=value, **kwds)


class ConditionalValueDefnumberExprRef(VegaLiteSchema):
    """ConditionalValueDefnumberExprRef schema wrapper

    anyOf(:class:`ConditionalPredicateValueDefnumberExprRef`,
    :class:`ConditionalSelectionValueDefnumberExprRef`)
    """
    _schema = {'$ref': '#/definitions/ConditionalValueDef<(number|ExprRef)>'}

    def __init__(self, *args, **kwds):
        super(ConditionalValueDefnumberExprRef, self).__init__(*args, **kwds)


class ConditionalPredicateValueDefnumberExprRef(ConditionalValueDefnumberExprRef):
    """ConditionalPredicateValueDefnumberExprRef schema wrapper

    Mapping(required=[test, value])

    Attributes
    ----------

    test : :class:`PredicateComposition`
        Predicate for triggering the condition
    value : anyOf(float, :class:`ExprRef`)
        A constant value in visual domain (e.g., ``"red"`` / ``"#0099ff"`` / `gradient
        definition <https://vega.github.io/vega-lite/docs/types.html#gradient>`__ for color,
        values between ``0`` to ``1`` for opacity).
    """
    _schema = {'$ref': '#/definitions/ConditionalPredicate<ValueDef<(number|ExprRef)>>'}

    def __init__(self, test=Undefined, value=Undefined, **kwds):
        super(ConditionalPredicateValueDefnumberExprRef, self).__init__(test=test, value=value, **kwds)


class ConditionalSelectionValueDefnumberExprRef(ConditionalValueDefnumberExprRef):
    """ConditionalSelectionValueDefnumberExprRef schema wrapper

    Mapping(required=[selection, value])

    Attributes
    ----------

    selection : :class:`SelectionComposition`
        A `selection name <https://vega.github.io/vega-lite/docs/selection.html>`__, or a
        series of `composed selections
        <https://vega.github.io/vega-lite/docs/selection.html#compose>`__.
    value : anyOf(float, :class:`ExprRef`)
        A constant value in visual domain (e.g., ``"red"`` / ``"#0099ff"`` / `gradient
        definition <https://vega.github.io/vega-lite/docs/types.html#gradient>`__ for color,
        values between ``0`` to ``1`` for opacity).
    """
    _schema = {'$ref': '#/definitions/ConditionalSelection<ValueDef<(number|ExprRef)>>'}

    def __init__(self, selection=Undefined, value=Undefined, **kwds):
        super(ConditionalSelectionValueDefnumberExprRef, self).__init__(selection=selection,
                                                                        value=value, **kwds)


class ConditionalValueDefstringExprRef(VegaLiteSchema):
    """ConditionalValueDefstringExprRef schema wrapper

    anyOf(:class:`ConditionalPredicateValueDefstringExprRef`,
    :class:`ConditionalSelectionValueDefstringExprRef`)
    """
    _schema = {'$ref': '#/definitions/ConditionalValueDef<(string|ExprRef)>'}

    def __init__(self, *args, **kwds):
        super(ConditionalValueDefstringExprRef, self).__init__(*args, **kwds)


class ConditionalPredicateValueDefstringExprRef(ConditionalValueDefstringExprRef):
    """ConditionalPredicateValueDefstringExprRef schema wrapper

    Mapping(required=[test, value])

    Attributes
    ----------

    test : :class:`PredicateComposition`
        Predicate for triggering the condition
    value : anyOf(string, :class:`ExprRef`)
        A constant value in visual domain (e.g., ``"red"`` / ``"#0099ff"`` / `gradient
        definition <https://vega.github.io/vega-lite/docs/types.html#gradient>`__ for color,
        values between ``0`` to ``1`` for opacity).
    """
    _schema = {'$ref': '#/definitions/ConditionalPredicate<ValueDef<(string|ExprRef)>>'}

    def __init__(self, test=Undefined, value=Undefined, **kwds):
        super(ConditionalPredicateValueDefstringExprRef, self).__init__(test=test, value=value, **kwds)


class ConditionalSelectionValueDefstringExprRef(ConditionalValueDefstringExprRef):
    """ConditionalSelectionValueDefstringExprRef schema wrapper

    Mapping(required=[selection, value])

    Attributes
    ----------

    selection : :class:`SelectionComposition`
        A `selection name <https://vega.github.io/vega-lite/docs/selection.html>`__, or a
        series of `composed selections
        <https://vega.github.io/vega-lite/docs/selection.html#compose>`__.
    value : anyOf(string, :class:`ExprRef`)
        A constant value in visual domain (e.g., ``"red"`` / ``"#0099ff"`` / `gradient
        definition <https://vega.github.io/vega-lite/docs/types.html#gradient>`__ for color,
        values between ``0`` to ``1`` for opacity).
    """
    _schema = {'$ref': '#/definitions/ConditionalSelection<ValueDef<(string|ExprRef)>>'}

    def __init__(self, selection=Undefined, value=Undefined, **kwds):
        super(ConditionalSelectionValueDefstringExprRef, self).__init__(selection=selection,
                                                                        value=value, **kwds)


class ConditionalValueDefstringnullExprRef(VegaLiteSchema):
    """ConditionalValueDefstringnullExprRef schema wrapper

    anyOf(:class:`ConditionalPredicateValueDefstringnullExprRef`,
    :class:`ConditionalSelectionValueDefstringnullExprRef`)
    """
    _schema = {'$ref': '#/definitions/ConditionalValueDef<(string|null|ExprRef)>'}

    def __init__(self, *args, **kwds):
        super(ConditionalValueDefstringnullExprRef, self).__init__(*args, **kwds)


class ConditionalPredicateValueDefstringnullExprRef(ConditionalValueDefstringnullExprRef):
    """ConditionalPredicateValueDefstringnullExprRef schema wrapper

    Mapping(required=[test, value])

    Attributes
    ----------

    test : :class:`PredicateComposition`
        Predicate for triggering the condition
    value : anyOf(string, None, :class:`ExprRef`)
        A constant value in visual domain (e.g., ``"red"`` / ``"#0099ff"`` / `gradient
        definition <https://vega.github.io/vega-lite/docs/types.html#gradient>`__ for color,
        values between ``0`` to ``1`` for opacity).
    """
    _schema = {'$ref': '#/definitions/ConditionalPredicate<ValueDef<(string|null|ExprRef)>>'}

    def __init__(self, test=Undefined, value=Undefined, **kwds):
        super(ConditionalPredicateValueDefstringnullExprRef, self).__init__(test=test, value=value,
                                                                            **kwds)


class ConditionalSelectionValueDefstringnullExprRef(ConditionalValueDefstringnullExprRef):
    """ConditionalSelectionValueDefstringnullExprRef schema wrapper

    Mapping(required=[selection, value])

    Attributes
    ----------

    selection : :class:`SelectionComposition`
        A `selection name <https://vega.github.io/vega-lite/docs/selection.html>`__, or a
        series of `composed selections
        <https://vega.github.io/vega-lite/docs/selection.html#compose>`__.
    value : anyOf(string, None, :class:`ExprRef`)
        A constant value in visual domain (e.g., ``"red"`` / ``"#0099ff"`` / `gradient
        definition <https://vega.github.io/vega-lite/docs/types.html#gradient>`__ for color,
        values between ``0`` to ``1`` for opacity).
    """
    _schema = {'$ref': '#/definitions/ConditionalSelection<ValueDef<(string|null|ExprRef)>>'}

    def __init__(self, selection=Undefined, value=Undefined, **kwds):
        super(ConditionalSelectionValueDefstringnullExprRef, self).__init__(selection=selection,
                                                                            value=value, **kwds)


class Config(VegaLiteSchema):
    """Config schema wrapper

    Mapping(required=[])

    Attributes
    ----------

    arc : :class:`RectConfig`
        Arc-specific Config
    area : :class:`AreaConfig`
        Area-Specific Config
    aria : boolean
        A boolean flag indicating if ARIA default attributes should be included for marks
        and guides (SVG output only). If false, the ``"aria-hidden"`` attribute will be set
        for all guides, removing them from the ARIA accessibility tree and Vega-Lite will
        not generate default descriptions for marks.

        **Default value:** ``true``.
    autosize : anyOf(:class:`AutosizeType`, :class:`AutoSizeParams`)
        How the visualization size should be determined. If a string, should be one of
        ``"pad"``, ``"fit"`` or ``"none"``. Object values can additionally specify
        parameters for content sizing and automatic resizing.

        **Default value** : ``pad``
    axis : :class:`AxisConfig`
        Axis configuration, which determines default properties for all ``x`` and ``y``
        `axes <https://vega.github.io/vega-lite/docs/axis.html>`__. For a full list of axis
        configuration options, please see the `corresponding section of the axis
        documentation <https://vega.github.io/vega-lite/docs/axis.html#config>`__.
    axisBand : :class:`AxisConfig`
        Config for axes with "band" scales.
    axisBottom : :class:`AxisConfig`
        Config for x-axis along the bottom edge of the chart.
    axisDiscrete : :class:`AxisConfig`
        Config for axes with "point" or "band" scales.
    axisLeft : :class:`AxisConfig`
        Config for y-axis along the left edge of the chart.
    axisPoint : :class:`AxisConfig`
        Config for axes with "point" scales.
    axisQuantitative : :class:`AxisConfig`
        Config for quantitative axes.
    axisRight : :class:`AxisConfig`
        Config for y-axis along the right edge of the chart.
    axisTemporal : :class:`AxisConfig`
        Config for temporal axes.
    axisTop : :class:`AxisConfig`
        Config for x-axis along the top edge of the chart.
    axisX : :class:`AxisConfig`
        X-axis specific config.
    axisXBand : :class:`AxisConfig`
        Config for x-axes with "band" scales.
    axisXDiscrete : :class:`AxisConfig`
        Config for x-axes with "point" or "band" scales.
    axisXPoint : :class:`AxisConfig`
        Config for x-axes with "point" scales.
    axisXQuantitative : :class:`AxisConfig`
        Config for x-quantitative axes.
    axisXTemporal : :class:`AxisConfig`
        Config for x-temporal axes.
    axisY : :class:`AxisConfig`
        Y-axis specific config.
    axisYBand : :class:`AxisConfig`
        Config for y-axes with "band" scales.
    axisYDiscrete : :class:`AxisConfig`
        Config for y-axes with "point" or "band" scales.
    axisYPoint : :class:`AxisConfig`
        Config for y-axes with "point" scales.
    axisYQuantitative : :class:`AxisConfig`
        Config for y-quantitative axes.
    axisYTemporal : :class:`AxisConfig`
        Config for y-temporal axes.
    background : anyOf(:class:`Color`, :class:`ExprRef`)
        CSS color property to use as the background of the entire view.

        **Default value:** ``"white"``
    bar : :class:`BarConfig`
        Bar-Specific Config
    boxplot : :class:`BoxPlotConfig`
        Box Config
    circle : :class:`MarkConfig`
        Circle-Specific Config
    concat : :class:`CompositionConfig`
        Default configuration for all concatenation and repeat view composition operators (
        ``concat``, ``hconcat``, ``vconcat``, and ``repeat`` )
    countTitle : string
        Default axis and legend title for count fields.

        **Default value:** ``'Count of Records``.
    customFormatTypes : boolean
        Allow the ``formatType`` property for text marks and guides to accept a custom
        formatter function `registered as a Vega expression
        <https://vega.github.io/vega-lite/usage/compile.html#format-type>`__.
    errorband : :class:`ErrorBandConfig`
        ErrorBand Config
    errorbar : :class:`ErrorBarConfig`
        ErrorBar Config
    facet : :class:`CompositionConfig`
        Default configuration for the ``facet`` view composition operator
    fieldTitle : enum('verbal', 'functional', 'plain')
        Defines how Vega-Lite generates title for fields. There are three possible styles: -
        ``"verbal"`` (Default) - displays function in a verbal style (e.g., "Sum of field",
        "Year-month of date", "field (binned)"). - ``"function"`` - displays function using
        parentheses and capitalized texts (e.g., "SUM(field)", "YEARMONTH(date)",
        "BIN(field)"). - ``"plain"`` - displays only the field name without functions (e.g.,
        "field", "date", "field").
    font : string
        Default font for all text marks, titles, and labels.
    geoshape : :class:`MarkConfig`
        Geoshape-Specific Config
    header : :class:`HeaderConfig`
        Header configuration, which determines default properties for all `headers
        <https://vega.github.io/vega-lite/docs/header.html>`__.

        For a full list of header configuration options, please see the `corresponding
        section of in the header documentation
        <https://vega.github.io/vega-lite/docs/header.html#config>`__.
    headerColumn : :class:`HeaderConfig`
        Header configuration, which determines default properties for column `headers
        <https://vega.github.io/vega-lite/docs/header.html>`__.

        For a full list of header configuration options, please see the `corresponding
        section of in the header documentation
        <https://vega.github.io/vega-lite/docs/header.html#config>`__.
    headerFacet : :class:`HeaderConfig`
        Header configuration, which determines default properties for non-row/column facet
        `headers <https://vega.github.io/vega-lite/docs/header.html>`__.

        For a full list of header configuration options, please see the `corresponding
        section of in the header documentation
        <https://vega.github.io/vega-lite/docs/header.html#config>`__.
    headerRow : :class:`HeaderConfig`
        Header configuration, which determines default properties for row `headers
        <https://vega.github.io/vega-lite/docs/header.html>`__.

        For a full list of header configuration options, please see the `corresponding
        section of in the header documentation
        <https://vega.github.io/vega-lite/docs/header.html#config>`__.
    image : :class:`RectConfig`
        Image-specific Config
    legend : :class:`LegendConfig`
        Legend configuration, which determines default properties for all `legends
        <https://vega.github.io/vega-lite/docs/legend.html>`__. For a full list of legend
        configuration options, please see the `corresponding section of in the legend
        documentation <https://vega.github.io/vega-lite/docs/legend.html#config>`__.
    line : :class:`LineConfig`
        Line-Specific Config
    lineBreak : anyOf(string, :class:`ExprRef`)
        A delimiter, such as a newline character, upon which to break text strings into
        multiple lines. This property provides a global default for text marks, which is
        overridden by mark or style config settings, and by the lineBreak mark encoding
        channel. If signal-valued, either string or regular expression (regexp) values are
        valid.
    mark : :class:`MarkConfig`
        Mark Config
    numberFormat : string
        D3 Number format for guide labels and text marks. For example ``"s"`` for SI units.
        Use `D3's number format pattern <https://github.com/d3/d3-format#locale_format>`__.
    padding : anyOf(:class:`Padding`, :class:`ExprRef`)
        The default visualization padding, in pixels, from the edge of the visualization
        canvas to the data rectangle. If a number, specifies padding for all sides. If an
        object, the value should have the format ``{"left": 5, "top": 5, "right": 5,
        "bottom": 5}`` to specify padding for each side of the visualization.

        **Default value** : ``5``
    params : List(:class:`Parameter`)
        Dynamic variables that parameterize a visualization.
    point : :class:`MarkConfig`
        Point-Specific Config
    projection : :class:`ProjectionConfig`
        Projection configuration, which determines default properties for all `projections
        <https://vega.github.io/vega-lite/docs/projection.html>`__. For a full list of
        projection configuration options, please see the `corresponding section of the
        projection documentation
        <https://vega.github.io/vega-lite/docs/projection.html#config>`__.
    range : :class:`RangeConfig`
        An object hash that defines default range arrays or schemes for using with scales.
        For a full list of scale range configuration options, please see the `corresponding
        section of the scale documentation
        <https://vega.github.io/vega-lite/docs/scale.html#config>`__.
    rect : :class:`RectConfig`
        Rect-Specific Config
    rule : :class:`MarkConfig`
        Rule-Specific Config
    scale : :class:`ScaleConfig`
        Scale configuration determines default properties for all `scales
        <https://vega.github.io/vega-lite/docs/scale.html>`__. For a full list of scale
        configuration options, please see the `corresponding section of the scale
        documentation <https://vega.github.io/vega-lite/docs/scale.html#config>`__.
    selection : :class:`SelectionConfig`
        An object hash for defining default properties for each type of selections.
    square : :class:`MarkConfig`
        Square-Specific Config
    style : :class:`StyleConfigIndex`
        An object hash that defines key-value mappings to determine default properties for
        marks with a given `style
        <https://vega.github.io/vega-lite/docs/mark.html#mark-def>`__. The keys represent
        styles names; the values have to be valid `mark configuration objects
        <https://vega.github.io/vega-lite/docs/mark.html#config>`__.
    text : :class:`MarkConfig`
        Text-Specific Config
    tick : :class:`TickConfig`
        Tick-Specific Config
    timeFormat : string
        Default time format for raw time values (without time units) in text marks, legend
        labels and header labels.

        **Default value:** ``"%b %d, %Y"`` **Note:** Axes automatically determine the format
        for each label automatically so this config does not affect axes.
    title : :class:`TitleConfig`
        Title configuration, which determines default properties for all `titles
        <https://vega.github.io/vega-lite/docs/title.html>`__. For a full list of title
        configuration options, please see the `corresponding section of the title
        documentation <https://vega.github.io/vega-lite/docs/title.html#config>`__.
    trail : :class:`LineConfig`
        Trail-Specific Config
    view : :class:`ViewConfig`
        Default properties for `single view plots
        <https://vega.github.io/vega-lite/docs/spec.html#single>`__.
    """
    _schema = {'$ref': '#/definitions/Config'}

    def __init__(self, arc=Undefined, area=Undefined, aria=Undefined, autosize=Undefined,
                 axis=Undefined, axisBand=Undefined, axisBottom=Undefined, axisDiscrete=Undefined,
                 axisLeft=Undefined, axisPoint=Undefined, axisQuantitative=Undefined,
                 axisRight=Undefined, axisTemporal=Undefined, axisTop=Undefined, axisX=Undefined,
                 axisXBand=Undefined, axisXDiscrete=Undefined, axisXPoint=Undefined,
                 axisXQuantitative=Undefined, axisXTemporal=Undefined, axisY=Undefined,
                 axisYBand=Undefined, axisYDiscrete=Undefined, axisYPoint=Undefined,
                 axisYQuantitative=Undefined, axisYTemporal=Undefined, background=Undefined,
                 bar=Undefined, boxplot=Undefined, circle=Undefined, concat=Undefined,
                 countTitle=Undefined, customFormatTypes=Undefined, errorband=Undefined,
                 errorbar=Undefined, facet=Undefined, fieldTitle=Undefined, font=Undefined,
                 geoshape=Undefined, header=Undefined, headerColumn=Undefined, headerFacet=Undefined,
                 headerRow=Undefined, image=Undefined, legend=Undefined, line=Undefined,
                 lineBreak=Undefined, mark=Undefined, numberFormat=Undefined, padding=Undefined,
                 params=Undefined, point=Undefined, projection=Undefined, range=Undefined,
                 rect=Undefined, rule=Undefined, scale=Undefined, selection=Undefined, square=Undefined,
                 style=Undefined, text=Undefined, tick=Undefined, timeFormat=Undefined, title=Undefined,
                 trail=Undefined, view=Undefined, **kwds):
        super(Config, self).__init__(arc=arc, area=area, aria=aria, autosize=autosize, axis=axis,
                                     axisBand=axisBand, axisBottom=axisBottom,
                                     axisDiscrete=axisDiscrete, axisLeft=axisLeft, axisPoint=axisPoint,
                                     axisQuantitative=axisQuantitative, axisRight=axisRight,
                                     axisTemporal=axisTemporal, axisTop=axisTop, axisX=axisX,
                                     axisXBand=axisXBand, axisXDiscrete=axisXDiscrete,
                                     axisXPoint=axisXPoint, axisXQuantitative=axisXQuantitative,
                                     axisXTemporal=axisXTemporal, axisY=axisY, axisYBand=axisYBand,
                                     axisYDiscrete=axisYDiscrete, axisYPoint=axisYPoint,
                                     axisYQuantitative=axisYQuantitative, axisYTemporal=axisYTemporal,
                                     background=background, bar=bar, boxplot=boxplot, circle=circle,
                                     concat=concat, countTitle=countTitle,
                                     customFormatTypes=customFormatTypes, errorband=errorband,
                                     errorbar=errorbar, facet=facet, fieldTitle=fieldTitle, font=font,
                                     geoshape=geoshape, header=header, headerColumn=headerColumn,
                                     headerFacet=headerFacet, headerRow=headerRow, image=image,
                                     legend=legend, line=line, lineBreak=lineBreak, mark=mark,
                                     numberFormat=numberFormat, padding=padding, params=params,
                                     point=point, projection=projection, range=range, rect=rect,
                                     rule=rule, scale=scale, selection=selection, square=square,
                                     style=style, text=text, tick=tick, timeFormat=timeFormat,
                                     title=title, trail=trail, view=view, **kwds)


class Cursor(VegaLiteSchema):
    """Cursor schema wrapper

    enum('auto', 'default', 'none', 'context-menu', 'help', 'pointer', 'progress', 'wait',
    'cell', 'crosshair', 'text', 'vertical-text', 'alias', 'copy', 'move', 'no-drop',
    'not-allowed', 'e-resize', 'n-resize', 'ne-resize', 'nw-resize', 's-resize', 'se-resize',
    'sw-resize', 'w-resize', 'ew-resize', 'ns-resize', 'nesw-resize', 'nwse-resize',
    'col-resize', 'row-resize', 'all-scroll', 'zoom-in', 'zoom-out', 'grab', 'grabbing')
    """
    _schema = {'$ref': '#/definitions/Cursor'}

    def __init__(self, *args):
        super(Cursor, self).__init__(*args)


class Cyclical(ColorScheme):
    """Cyclical schema wrapper

    enum('rainbow', 'sinebow')
    """
    _schema = {'$ref': '#/definitions/Cyclical'}

    def __init__(self, *args):
        super(Cyclical, self).__init__(*args)


class Data(VegaLiteSchema):
    """Data schema wrapper

    anyOf(:class:`DataSource`, :class:`Generator`)
    """
    _schema = {'$ref': '#/definitions/Data'}

    def __init__(self, *args, **kwds):
        super(Data, self).__init__(*args, **kwds)


class DataFormat(VegaLiteSchema):
    """DataFormat schema wrapper

    anyOf(:class:`CsvDataFormat`, :class:`DsvDataFormat`, :class:`JsonDataFormat`,
    :class:`TopoDataFormat`)
    """
    _schema = {'$ref': '#/definitions/DataFormat'}

    def __init__(self, *args, **kwds):
        super(DataFormat, self).__init__(*args, **kwds)


class CsvDataFormat(DataFormat):
    """CsvDataFormat schema wrapper

    Mapping(required=[])

    Attributes
    ----------

    parse : anyOf(:class:`Parse`, None)
        If set to ``null``, disable type inference based on the spec and only use type
        inference based on the data. Alternatively, a parsing directive object can be
        provided for explicit data types. Each property of the object corresponds to a field
        name, and the value to the desired data type (one of ``"number"``, ``"boolean"``,
        ``"date"``, or null (do not parse the field)). For example, ``"parse":
        {"modified_on": "date"}`` parses the ``modified_on`` field in each input record a
        Date value.

        For ``"date"``, we parse data based using Javascript's `Date.parse()
        <https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Date/parse>`__.
        For Specific date formats can be provided (e.g., ``{foo: "date:'%m%d%Y'"}`` ), using
        the `d3-time-format syntax <https://github.com/d3/d3-time-format#locale_format>`__.
        UTC date format parsing is supported similarly (e.g., ``{foo: "utc:'%m%d%Y'"}`` ).
        See more about `UTC time
        <https://vega.github.io/vega-lite/docs/timeunit.html#utc>`__
    type : enum('csv', 'tsv')
        Type of input data: ``"json"``, ``"csv"``, ``"tsv"``, ``"dsv"``.

        **Default value:**  The default format type is determined by the extension of the
        file URL. If no extension is detected, ``"json"`` will be used by default.
    """
    _schema = {'$ref': '#/definitions/CsvDataFormat'}

    def __init__(self, parse=Undefined, type=Undefined, **kwds):
        super(CsvDataFormat, self).__init__(parse=parse, type=type, **kwds)


class DataSource(Data):
    """DataSource schema wrapper

    anyOf(:class:`UrlData`, :class:`InlineData`, :class:`NamedData`)
    """
    _schema = {'$ref': '#/definitions/DataSource'}

    def __init__(self, *args, **kwds):
        super(DataSource, self).__init__(*args, **kwds)


class Datasets(VegaLiteSchema):
    """Datasets schema wrapper

    Mapping(required=[])
    """
    _schema = {'$ref': '#/definitions/Datasets'}

    def __init__(self, **kwds):
        super(Datasets, self).__init__(**kwds)


class Day(VegaLiteSchema):
    """Day schema wrapper

    float
    """
    _schema = {'$ref': '#/definitions/Day'}

    def __init__(self, *args):
        super(Day, self).__init__(*args)


class DictInlineDataset(VegaLiteSchema):
    """DictInlineDataset schema wrapper

    Mapping(required=[])
    """
    _schema = {'$ref': '#/definitions/Dict<InlineDataset>'}

    def __init__(self, **kwds):
        super(DictInlineDataset, self).__init__(**kwds)


class DictSelectionInit(VegaLiteSchema):
    """DictSelectionInit schema wrapper

    Mapping(required=[])
    """
    _schema = {'$ref': '#/definitions/Dict<SelectionInit>'}

    def __init__(self, **kwds):
        super(DictSelectionInit, self).__init__(**kwds)


class DictSelectionInitInterval(VegaLiteSchema):
    """DictSelectionInitInterval schema wrapper

    Mapping(required=[])
    """
    _schema = {'$ref': '#/definitions/Dict<SelectionInitInterval>'}

    def __init__(self, **kwds):
        super(DictSelectionInitInterval, self).__init__(**kwds)


class Dictunknown(VegaLiteSchema):
    """Dictunknown schema wrapper

    Mapping(required=[])
    """
    _schema = {'$ref': '#/definitions/Dict<unknown>'}

    def __init__(self, **kwds):
        super(Dictunknown, self).__init__(**kwds)


class Diverging(ColorScheme):
    """Diverging schema wrapper

    enum('blueorange', 'blueorange-3', 'blueorange-4', 'blueorange-5', 'blueorange-6',
    'blueorange-7', 'blueorange-8', 'blueorange-9', 'blueorange-10', 'blueorange-11',
    'brownbluegreen', 'brownbluegreen-3', 'brownbluegreen-4', 'brownbluegreen-5',
    'brownbluegreen-6', 'brownbluegreen-7', 'brownbluegreen-8', 'brownbluegreen-9',
    'brownbluegreen-10', 'brownbluegreen-11', 'purplegreen', 'purplegreen-3', 'purplegreen-4',
    'purplegreen-5', 'purplegreen-6', 'purplegreen-7', 'purplegreen-8', 'purplegreen-9',
    'purplegreen-10', 'purplegreen-11', 'pinkyellowgreen', 'pinkyellowgreen-3',
    'pinkyellowgreen-4', 'pinkyellowgreen-5', 'pinkyellowgreen-6', 'pinkyellowgreen-7',
    'pinkyellowgreen-8', 'pinkyellowgreen-9', 'pinkyellowgreen-10', 'pinkyellowgreen-11',
    'purpleorange', 'purpleorange-3', 'purpleorange-4', 'purpleorange-5', 'purpleorange-6',
    'purpleorange-7', 'purpleorange-8', 'purpleorange-9', 'purpleorange-10', 'purpleorange-11',
    'redblue', 'redblue-3', 'redblue-4', 'redblue-5', 'redblue-6', 'redblue-7', 'redblue-8',
    'redblue-9', 'redblue-10', 'redblue-11', 'redgrey', 'redgrey-3', 'redgrey-4', 'redgrey-5',
    'redgrey-6', 'redgrey-7', 'redgrey-8', 'redgrey-9', 'redgrey-10', 'redgrey-11',
    'redyellowblue', 'redyellowblue-3', 'redyellowblue-4', 'redyellowblue-5', 'redyellowblue-6',
    'redyellowblue-7', 'redyellowblue-8', 'redyellowblue-9', 'redyellowblue-10',
    'redyellowblue-11', 'redyellowgreen', 'redyellowgreen-3', 'redyellowgreen-4',
    'redyellowgreen-5', 'redyellowgreen-6', 'redyellowgreen-7', 'redyellowgreen-8',
    'redyellowgreen-9', 'redyellowgreen-10', 'redyellowgreen-11', 'spectral', 'spectral-3',
    'spectral-4', 'spectral-5', 'spectral-6', 'spectral-7', 'spectral-8', 'spectral-9',
    'spectral-10', 'spectral-11')
    """
    _schema = {'$ref': '#/definitions/Diverging'}

    def __init__(self, *args):
        super(Diverging, self).__init__(*args)


class DomainUnionWith(VegaLiteSchema):
    """DomainUnionWith schema wrapper

    Mapping(required=[unionWith])

    Attributes
    ----------

    unionWith : anyOf(List(float), List(string), List(boolean), List(:class:`DateTime`))
        Customized domain values to be union with the field's values.

        1) ``domain`` for *quantitative* fields can take one of the following forms:


        * a two-element array with minimum and maximum values. - an array with more than two
          entries, for `Piecewise  quantitative scales
          <https://vega.github.io/vega-lite/docs/scale.html#piecewise>`__. (Alternatively,
          the ``domainMid`` property can be set for a diverging scale.) - a string value
          ``"unaggregated"``, if the input field is aggregated, to indicate that the domain
          should include the raw data values prior to the aggregation.

        2) ``domain`` for *temporal* fields can be a two-element array minimum and maximum
        values, in the form of either timestamps or the `DateTime definition objects
        <https://vega.github.io/vega-lite/docs/types.html#datetime>`__.

        3) ``domain`` for *ordinal* and *nominal* fields can be an array that lists valid
        input values.
    """
    _schema = {'$ref': '#/definitions/DomainUnionWith'}

    def __init__(self, unionWith=Undefined, **kwds):
        super(DomainUnionWith, self).__init__(unionWith=unionWith, **kwds)


class DsvDataFormat(DataFormat):
    """DsvDataFormat schema wrapper

    Mapping(required=[delimiter])

    Attributes
    ----------

    delimiter : string
        The delimiter between records. The delimiter must be a single character (i.e., a
        single 16-bit code unit); so, ASCII delimiters are fine, but emoji delimiters are
        not.
    parse : anyOf(:class:`Parse`, None)
        If set to ``null``, disable type inference based on the spec and only use type
        inference based on the data. Alternatively, a parsing directive object can be
        provided for explicit data types. Each property of the object corresponds to a field
        name, and the value to the desired data type (one of ``"number"``, ``"boolean"``,
        ``"date"``, or null (do not parse the field)). For example, ``"parse":
        {"modified_on": "date"}`` parses the ``modified_on`` field in each input record a
        Date value.

        For ``"date"``, we parse data based using Javascript's `Date.parse()
        <https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Date/parse>`__.
        For Specific date formats can be provided (e.g., ``{foo: "date:'%m%d%Y'"}`` ), using
        the `d3-time-format syntax <https://github.com/d3/d3-time-format#locale_format>`__.
        UTC date format parsing is supported similarly (e.g., ``{foo: "utc:'%m%d%Y'"}`` ).
        See more about `UTC time
        <https://vega.github.io/vega-lite/docs/timeunit.html#utc>`__
    type : string
        Type of input data: ``"json"``, ``"csv"``, ``"tsv"``, ``"dsv"``.

        **Default value:**  The default format type is determined by the extension of the
        file URL. If no extension is detected, ``"json"`` will be used by default.
    """
    _schema = {'$ref': '#/definitions/DsvDataFormat'}

    def __init__(self, delimiter=Undefined, parse=Undefined, type=Undefined, **kwds):
        super(DsvDataFormat, self).__init__(delimiter=delimiter, parse=parse, type=type, **kwds)


class Element(VegaLiteSchema):
    """Element schema wrapper

    string
    """
    _schema = {'$ref': '#/definitions/Element'}

    def __init__(self, *args):
        super(Element, self).__init__(*args)


class Encoding(VegaLiteSchema):
    """Encoding schema wrapper

    Mapping(required=[])

    Attributes
    ----------

    angle : :class:`NumericMarkPropDef`
        Rotation angle of point and text marks.
    color : :class:`ColorDef`
        Color of the marks – either fill or stroke color based on  the ``filled`` property
        of mark definition. By default, ``color`` represents fill color for ``"area"``,
        ``"bar"``, ``"tick"``, ``"text"``, ``"trail"``, ``"circle"``, and ``"square"`` /
        stroke color for ``"line"`` and ``"point"``.

        **Default value:** If undefined, the default color depends on `mark config
        <https://vega.github.io/vega-lite/docs/config.html#mark-config>`__ 's ``color``
        property.

        *Note:* 1) For fine-grained control over both fill and stroke colors of the marks,
        please use the ``fill`` and ``stroke`` channels. The ``fill`` or ``stroke``
        encodings have higher precedence than ``color``, thus may override the ``color``
        encoding if conflicting encodings are specified. 2) See the scale documentation for
        more information about customizing `color scheme
        <https://vega.github.io/vega-lite/docs/scale.html#scheme>`__.
    description : anyOf(:class:`StringFieldDefWithCondition`,
    :class:`StringValueDefWithCondition`)
        A text description of this mark for ARIA accessibility (SVG output only). For SVG
        output the ``"aria-label"`` attribute will be set to this description.
    detail : anyOf(:class:`FieldDefWithoutScale`, List(:class:`FieldDefWithoutScale`))
        Additional levels of detail for grouping data in aggregate views and in line, trail,
        and area marks without mapping data to a specific visual channel.
    fill : :class:`ColorDef`
        Fill color of the marks. **Default value:** If undefined, the default color depends
        on `mark config <https://vega.github.io/vega-lite/docs/config.html#mark-config>`__
        's ``color`` property.

        *Note:* The ``fill`` encoding has higher precedence than ``color``, thus may
        override the ``color`` encoding if conflicting encodings are specified.
    fillOpacity : :class:`NumericMarkPropDef`
        Fill opacity of the marks.

        **Default value:** If undefined, the default opacity depends on `mark config
        <https://vega.github.io/vega-lite/docs/config.html#mark-config>`__ 's
        ``fillOpacity`` property.
    href : anyOf(:class:`StringFieldDefWithCondition`, :class:`StringValueDefWithCondition`)
        A URL to load upon mouse click.
    key : :class:`FieldDefWithoutScale`
        A data field to use as a unique key for data binding. When a visualization’s data is
        updated, the key value will be used to match data elements to existing mark
        instances. Use a key channel to enable object constancy for transitions over dynamic
        data.
    latitude : :class:`LatLongDef`
        Latitude position of geographically projected marks.
    latitude2 : :class:`Position2Def`
        Latitude-2 position for geographically projected ranged ``"area"``, ``"bar"``,
        ``"rect"``, and  ``"rule"``.
    longitude : :class:`LatLongDef`
        Longitude position of geographically projected marks.
    longitude2 : :class:`Position2Def`
        Longitude-2 position for geographically projected ranged ``"area"``, ``"bar"``,
        ``"rect"``, and  ``"rule"``.
    opacity : :class:`NumericMarkPropDef`
        Opacity of the marks.

        **Default value:** If undefined, the default opacity depends on `mark config
        <https://vega.github.io/vega-lite/docs/config.html#mark-config>`__ 's ``opacity``
        property.
    order : anyOf(:class:`OrderFieldDef`, List(:class:`OrderFieldDef`), :class:`OrderValueDef`)
        Order of the marks. - For stacked marks, this ``order`` channel encodes `stack order
        <https://vega.github.io/vega-lite/docs/stack.html#order>`__. - For line and trail
        marks, this ``order`` channel encodes order of data points in the lines. This can be
        useful for creating `a connected scatterplot
        <https://vega.github.io/vega-lite/examples/connected_scatterplot.html>`__. Setting
        ``order`` to ``{"value": null}`` makes the line marks use the original order in the
        data sources. - Otherwise, this ``order`` channel encodes layer order of the marks.

        **Note** : In aggregate plots, ``order`` field should be ``aggregate`` d to avoid
        creating additional aggregation grouping.
    radius : :class:`PolarDef`
        The outer radius in pixels of arc marks.
    radius2 : :class:`Position2Def`
        The inner radius in pixels of arc marks.
    shape : :class:`ShapeDef`
        Shape of the mark.


        #.
        For ``point`` marks the supported values include:    - plotting shapes:
        ``"circle"``, ``"square"``, ``"cross"``, ``"diamond"``, ``"triangle-up"``,
        ``"triangle-down"``, ``"triangle-right"``, or ``"triangle-left"``.    - the line
        symbol ``"stroke"``    - centered directional shapes ``"arrow"``, ``"wedge"``, or
        ``"triangle"``    - a custom `SVG path string
        <https://developer.mozilla.org/en-US/docs/Web/SVG/Tutorial/Paths>`__ (For correct
        sizing, custom shape paths should be defined within a square bounding box with
        coordinates ranging from -1 to 1 along both the x and y dimensions.)

        #.
        For ``geoshape`` marks it should be a field definition of the geojson data

        **Default value:** If undefined, the default shape depends on `mark config
        <https://vega.github.io/vega-lite/docs/config.html#point-config>`__ 's ``shape``
        property. ( ``"circle"`` if unset.)
    size : :class:`NumericMarkPropDef`
        Size of the mark. - For ``"point"``, ``"square"`` and ``"circle"``, – the symbol
        size, or pixel area of the mark. - For ``"bar"`` and ``"tick"`` – the bar and tick's
        size. - For ``"text"`` – the text's font size. - Size is unsupported for ``"line"``,
        ``"area"``, and ``"rect"``. (Use ``"trail"`` instead of line with varying size)
    stroke : :class:`ColorDef`
        Stroke color of the marks. **Default value:** If undefined, the default color
        depends on `mark config
        <https://vega.github.io/vega-lite/docs/config.html#mark-config>`__ 's ``color``
        property.

        *Note:* The ``stroke`` encoding has higher precedence than ``color``, thus may
        override the ``color`` encoding if conflicting encodings are specified.
    strokeDash : :class:`NumericArrayMarkPropDef`
        Stroke dash of the marks.

        **Default value:** ``[1,0]`` (No dash).
    strokeOpacity : :class:`NumericMarkPropDef`
        Stroke opacity of the marks.

        **Default value:** If undefined, the default opacity depends on `mark config
        <https://vega.github.io/vega-lite/docs/config.html#mark-config>`__ 's
        ``strokeOpacity`` property.
    strokeWidth : :class:`NumericMarkPropDef`
        Stroke width of the marks.

        **Default value:** If undefined, the default stroke width depends on `mark config
        <https://vega.github.io/vega-lite/docs/config.html#mark-config>`__ 's
        ``strokeWidth`` property.
    text : :class:`TextDef`
        Text of the ``text`` mark.
    theta : :class:`PolarDef`
        For arc marks, the arc length in radians if theta2 is not specified, otherwise the
        start arc angle. (A value of 0 indicates up or “north”, increasing values proceed
        clockwise.)

        For text marks, polar coordinate angle in radians.
    theta2 : :class:`Position2Def`
        The end angle of arc marks in radians. A value of 0 indicates up or “north”,
        increasing values proceed clockwise.
    tooltip : anyOf(:class:`StringFieldDefWithCondition`, :class:`StringValueDefWithCondition`,
    List(:class:`StringFieldDef`), None)
        The tooltip text to show upon mouse hover. Specifying ``tooltip`` encoding overrides
        `the tooltip property in the mark definition
        <https://vega.github.io/vega-lite/docs/mark.html#mark-def>`__.

        See the `tooltip <https://vega.github.io/vega-lite/docs/tooltip.html>`__
        documentation for a detailed discussion about tooltip in Vega-Lite.
    url : anyOf(:class:`StringFieldDefWithCondition`, :class:`StringValueDefWithCondition`)
        The URL of an image mark.
    x : :class:`PositionDef`
        X coordinates of the marks, or width of horizontal ``"bar"`` and ``"area"`` without
        specified ``x2`` or ``width``.

        The ``value`` of this channel can be a number or a string ``"width"`` for the width
        of the plot.
    x2 : :class:`Position2Def`
        X2 coordinates for ranged ``"area"``, ``"bar"``, ``"rect"``, and  ``"rule"``.

        The ``value`` of this channel can be a number or a string ``"width"`` for the width
        of the plot.
    xError : anyOf(:class:`SecondaryFieldDef`, :class:`ValueDefnumber`)
        Error value of x coordinates for error specified ``"errorbar"`` and ``"errorband"``.
    xError2 : anyOf(:class:`SecondaryFieldDef`, :class:`ValueDefnumber`)
        Secondary error value of x coordinates for error specified ``"errorbar"`` and
        ``"errorband"``.
    y : :class:`PositionDef`
        Y coordinates of the marks, or height of vertical ``"bar"`` and ``"area"`` without
        specified ``y2`` or ``height``.

        The ``value`` of this channel can be a number or a string ``"height"`` for the
        height of the plot.
    y2 : :class:`Position2Def`
        Y2 coordinates for ranged ``"area"``, ``"bar"``, ``"rect"``, and  ``"rule"``.

        The ``value`` of this channel can be a number or a string ``"height"`` for the
        height of the plot.
    yError : anyOf(:class:`SecondaryFieldDef`, :class:`ValueDefnumber`)
        Error value of y coordinates for error specified ``"errorbar"`` and ``"errorband"``.
    yError2 : anyOf(:class:`SecondaryFieldDef`, :class:`ValueDefnumber`)
        Secondary error value of y coordinates for error specified ``"errorbar"`` and
        ``"errorband"``.
    """
    _schema = {'$ref': '#/definitions/Encoding'}

    def __init__(self, angle=Undefined, color=Undefined, description=Undefined, detail=Undefined,
                 fill=Undefined, fillOpacity=Undefined, href=Undefined, key=Undefined,
                 latitude=Undefined, latitude2=Undefined, longitude=Undefined, longitude2=Undefined,
                 opacity=Undefined, order=Undefined, radius=Undefined, radius2=Undefined,
                 shape=Undefined, size=Undefined, stroke=Undefined, strokeDash=Undefined,
                 strokeOpacity=Undefined, strokeWidth=Undefined, text=Undefined, theta=Undefined,
                 theta2=Undefined, tooltip=Undefined, url=Undefined, x=Undefined, x2=Undefined,
                 xError=Undefined, xError2=Undefined, y=Undefined, y2=Undefined, yError=Undefined,
                 yError2=Undefined, **kwds):
        super(Encoding, self).__init__(angle=angle, color=color, description=description, detail=detail,
                                       fill=fill, fillOpacity=fillOpacity, href=href, key=key,
                                       latitude=latitude, latitude2=latitude2, longitude=longitude,
                                       longitude2=longitude2, opacity=opacity, order=order,
                                       radius=radius, radius2=radius2, shape=shape, size=size,
                                       stroke=stroke, strokeDash=strokeDash,
                                       strokeOpacity=strokeOpacity, strokeWidth=strokeWidth, text=text,
                                       theta=theta, theta2=theta2, tooltip=tooltip, url=url, x=x, x2=x2,
                                       xError=xError, xError2=xError2, y=y, y2=y2, yError=yError,
                                       yError2=yError2, **kwds)


class EncodingSortFieldFieldName(VegaLiteSchema):
    """EncodingSortFieldFieldName schema wrapper

    Mapping(required=[])
    A sort definition for sorting a discrete scale in an encoding field definition.

    Attributes
    ----------

    field : :class:`FieldName`
        The data `field <https://vega.github.io/vega-lite/docs/field.html>`__ to sort by.

        **Default value:** If unspecified, defaults to the field specified in the outer data
        reference.
    op : :class:`NonArgAggregateOp`
        An `aggregate operation
        <https://vega.github.io/vega-lite/docs/aggregate.html#ops>`__ to perform on the
        field prior to sorting (e.g., ``"count"``, ``"mean"`` and ``"median"`` ). An
        aggregation is required when there are multiple values of the sort field for each
        encoded data field. The input data objects will be aggregated, grouped by the
        encoded data field.

        For a full list of operations, please see the documentation for `aggregate
        <https://vega.github.io/vega-lite/docs/aggregate.html#ops>`__.

        **Default value:** ``"sum"`` for stacked plots. Otherwise, ``"min"``.
    order : anyOf(:class:`SortOrder`, None)
        The sort order. One of ``"ascending"`` (default), ``"descending"``, or ``null`` (no
        not sort).
    """
    _schema = {'$ref': '#/definitions/EncodingSortField<FieldName>'}

    def __init__(self, field=Undefined, op=Undefined, order=Undefined, **kwds):
        super(EncodingSortFieldFieldName, self).__init__(field=field, op=op, order=order, **kwds)


class ErrorBand(CompositeMark):
    """ErrorBand schema wrapper

    string
    """
    _schema = {'$ref': '#/definitions/ErrorBand'}

    def __init__(self, *args):
        super(ErrorBand, self).__init__(*args)


class ErrorBandConfig(VegaLiteSchema):
    """ErrorBandConfig schema wrapper

    Mapping(required=[])

    Attributes
    ----------

    band : anyOf(boolean, :class:`MarkConfigExprOrSignalRef`)

    borders : anyOf(boolean, :class:`MarkConfigExprOrSignalRef`)

    extent : :class:`ErrorBarExtent`
        The extent of the band. Available options include: - `"ci"`: Extend the band to the
        confidence interval of the mean. - `"stderr"`: The size of band are set to the value
        of standard error, extending from the mean. - `"stdev"`: The size of band are set to
        the value of standard deviation, extending from the mean. - `"iqr"`: Extend the band
        to the q1 and q3.

        **Default value:** ``"stderr"``.
    interpolate : :class:`Interpolate`
        The line interpolation method for the error band. One of the following: -
        `"linear"`: piecewise linear segments, as in a polyline. - `"linear-closed"`: close
        the linear segments to form a polygon. - `"step"`: a piecewise constant function (a
        step function) consisting of alternating horizontal and vertical lines. The y-value
        changes at the midpoint of each pair of adjacent x-values. - `"step-before"`: a
        piecewise constant function (a step function) consisting of alternating horizontal
        and vertical lines. The y-value changes before the x-value. - `"step-after"`: a
        piecewise constant function (a step function) consisting of alternating horizontal
        and vertical lines. The y-value changes after the x-value. - `"basis"`: a B-spline,
        with control point duplication on the ends. - `"basis-open"`: an open B-spline; may
        not intersect the start or end. - `"basis-closed"`: a closed B-spline, as in a loop.
        - `"cardinal"`: a Cardinal spline, with control point duplication on the ends. -
        `"cardinal-open"`: an open Cardinal spline; may not intersect the start or end, but
        will intersect other control points. - `"cardinal-closed"`: a closed Cardinal
        spline, as in a loop. - `"bundle"`: equivalent to basis, except the tension
        parameter is used to straighten the spline. - ``"monotone"`` : cubic interpolation
        that preserves monotonicity in y.
    tension : float
        The tension parameter for the interpolation type of the error band.
    """
    _schema = {'$ref': '#/definitions/ErrorBandConfig'}

    def __init__(self, band=Undefined, borders=Undefined, extent=Undefined, interpolate=Undefined,
                 tension=Undefined, **kwds):
        super(ErrorBandConfig, self).__init__(band=band, borders=borders, extent=extent,
                                              interpolate=interpolate, tension=tension, **kwds)


class ErrorBandDef(CompositeMarkDef):
    """ErrorBandDef schema wrapper

    Mapping(required=[type])

    Attributes
    ----------

    type : :class:`ErrorBand`
        The mark type. This could a primitive mark type (one of ``"bar"``, ``"circle"``,
        ``"square"``, ``"tick"``, ``"line"``, ``"area"``, ``"point"``, ``"geoshape"``,
        ``"rule"``, and ``"text"`` ) or a composite mark type ( ``"boxplot"``,
        ``"errorband"``, ``"errorbar"`` ).
    band : anyOf(boolean, :class:`MarkConfigExprOrSignalRef`)

    borders : anyOf(boolean, :class:`MarkConfigExprOrSignalRef`)

    clip : boolean
        Whether a composite mark be clipped to the enclosing group’s width and height.
    color : anyOf(:class:`Color`, :class:`Gradient`, :class:`ExprRef`)
        Default color.

        **Default value:** :raw-html:`<span style="color: #4682b4;">&#9632;</span>`
        ``"#4682b4"``

        **Note:** - This property cannot be used in a `style config
        <https://vega.github.io/vega-lite/docs/mark.html#style-config>`__. - The ``fill``
        and ``stroke`` properties have higher precedence than ``color`` and will override
        ``color``.
    extent : :class:`ErrorBarExtent`
        The extent of the band. Available options include: - `"ci"`: Extend the band to the
        confidence interval of the mean. - `"stderr"`: The size of band are set to the value
        of standard error, extending from the mean. - `"stdev"`: The size of band are set to
        the value of standard deviation, extending from the mean. - `"iqr"`: Extend the band
        to the q1 and q3.

        **Default value:** ``"stderr"``.
    interpolate : :class:`Interpolate`
        The line interpolation method for the error band. One of the following: -
        `"linear"`: piecewise linear segments, as in a polyline. - `"linear-closed"`: close
        the linear segments to form a polygon. - `"step"`: a piecewise constant function (a
        step function) consisting of alternating horizontal and vertical lines. The y-value
        changes at the midpoint of each pair of adjacent x-values. - `"step-before"`: a
        piecewise constant function (a step function) consisting of alternating horizontal
        and vertical lines. The y-value changes before the x-value. - `"step-after"`: a
        piecewise constant function (a step function) consisting of alternating horizontal
        and vertical lines. The y-value changes after the x-value. - `"basis"`: a B-spline,
        with control point duplication on the ends. - `"basis-open"`: an open B-spline; may
        not intersect the start or end. - `"basis-closed"`: a closed B-spline, as in a loop.
        - `"cardinal"`: a Cardinal spline, with control point duplication on the ends. -
        `"cardinal-open"`: an open Cardinal spline; may not intersect the start or end, but
        will intersect other control points. - `"cardinal-closed"`: a closed Cardinal
        spline, as in a loop. - `"bundle"`: equivalent to basis, except the tension
        parameter is used to straighten the spline. - ``"monotone"`` : cubic interpolation
        that preserves monotonicity in y.
    opacity : float
        The opacity (value between [0,1]) of the mark.
    orient : :class:`Orientation`
        Orientation of the error band. This is normally automatically determined, but can be
        specified when the orientation is ambiguous and cannot be automatically determined.
    tension : float
        The tension parameter for the interpolation type of the error band.
    """
    _schema = {'$ref': '#/definitions/ErrorBandDef'}

    def __init__(self, type=Undefined, band=Undefined, borders=Undefined, clip=Undefined,
                 color=Undefined, extent=Undefined, interpolate=Undefined, opacity=Undefined,
                 orient=Undefined, tension=Undefined, **kwds):
        super(ErrorBandDef, self).__init__(type=type, band=band, borders=borders, clip=clip,
                                           color=color, extent=extent, interpolate=interpolate,
                                           opacity=opacity, orient=orient, tension=tension, **kwds)


class ErrorBar(CompositeMark):
    """ErrorBar schema wrapper

    string
    """
    _schema = {'$ref': '#/definitions/ErrorBar'}

    def __init__(self, *args):
        super(ErrorBar, self).__init__(*args)


class ErrorBarConfig(VegaLiteSchema):
    """ErrorBarConfig schema wrapper

    Mapping(required=[])

    Attributes
    ----------

    extent : :class:`ErrorBarExtent`
        The extent of the rule. Available options include: - `"ci"`: Extend the rule to the
        confidence interval of the mean. - `"stderr"`: The size of rule are set to the value
        of standard error, extending from the mean. - `"stdev"`: The size of rule are set to
        the value of standard deviation, extending from the mean. - `"iqr"`: Extend the rule
        to the q1 and q3.

        **Default value:** ``"stderr"``.
    rule : anyOf(boolean, :class:`MarkConfigExprOrSignalRef`)

    size : float
        Size of the ticks of an error bar
    thickness : float
        Thickness of the ticks and the bar of an error bar
    ticks : anyOf(boolean, :class:`MarkConfigExprOrSignalRef`)

    """
    _schema = {'$ref': '#/definitions/ErrorBarConfig'}

    def __init__(self, extent=Undefined, rule=Undefined, size=Undefined, thickness=Undefined,
                 ticks=Undefined, **kwds):
        super(ErrorBarConfig, self).__init__(extent=extent, rule=rule, size=size, thickness=thickness,
                                             ticks=ticks, **kwds)


class ErrorBarDef(CompositeMarkDef):
    """ErrorBarDef schema wrapper

    Mapping(required=[type])

    Attributes
    ----------

    type : :class:`ErrorBar`
        The mark type. This could a primitive mark type (one of ``"bar"``, ``"circle"``,
        ``"square"``, ``"tick"``, ``"line"``, ``"area"``, ``"point"``, ``"geoshape"``,
        ``"rule"``, and ``"text"`` ) or a composite mark type ( ``"boxplot"``,
        ``"errorband"``, ``"errorbar"`` ).
    clip : boolean
        Whether a composite mark be clipped to the enclosing group’s width and height.
    color : anyOf(:class:`Color`, :class:`Gradient`, :class:`ExprRef`)
        Default color.

        **Default value:** :raw-html:`<span style="color: #4682b4;">&#9632;</span>`
        ``"#4682b4"``

        **Note:** - This property cannot be used in a `style config
        <https://vega.github.io/vega-lite/docs/mark.html#style-config>`__. - The ``fill``
        and ``stroke`` properties have higher precedence than ``color`` and will override
        ``color``.
    extent : :class:`ErrorBarExtent`
        The extent of the rule. Available options include: - `"ci"`: Extend the rule to the
        confidence interval of the mean. - `"stderr"`: The size of rule are set to the value
        of standard error, extending from the mean. - `"stdev"`: The size of rule are set to
        the value of standard deviation, extending from the mean. - `"iqr"`: Extend the rule
        to the q1 and q3.

        **Default value:** ``"stderr"``.
    opacity : float
        The opacity (value between [0,1]) of the mark.
    orient : :class:`Orientation`
        Orientation of the error bar. This is normally automatically determined, but can be
        specified when the orientation is ambiguous and cannot be automatically determined.
    rule : anyOf(boolean, :class:`MarkConfigExprOrSignalRef`)

    size : float
        Size of the ticks of an error bar
    thickness : float
        Thickness of the ticks and the bar of an error bar
    ticks : anyOf(boolean, :class:`MarkConfigExprOrSignalRef`)

    """
    _schema = {'$ref': '#/definitions/ErrorBarDef'}

    def __init__(self, type=Undefined, clip=Undefined, color=Undefined, extent=Undefined,
                 opacity=Undefined, orient=Undefined, rule=Undefined, size=Undefined,
                 thickness=Undefined, ticks=Undefined, **kwds):
        super(ErrorBarDef, self).__init__(type=type, clip=clip, color=color, extent=extent,
                                          opacity=opacity, orient=orient, rule=rule, size=size,
                                          thickness=thickness, ticks=ticks, **kwds)


class ErrorBarExtent(VegaLiteSchema):
    """ErrorBarExtent schema wrapper

    enum('ci', 'iqr', 'stderr', 'stdev')
    """
    _schema = {'$ref': '#/definitions/ErrorBarExtent'}

    def __init__(self, *args):
        super(ErrorBarExtent, self).__init__(*args)


class Expr(VegaLiteSchema):
    """Expr schema wrapper

    string
    """
    _schema = {'$ref': '#/definitions/Expr'}

    def __init__(self, *args):
        super(Expr, self).__init__(*args)


class ExprOrSignalRef(VegaLiteSchema):
    """ExprOrSignalRef schema wrapper

    Mapping(required=[expr])

    Attributes
    ----------

    expr : string
        Vega expression (which can refer to Vega-Lite parameters).
    """
    _schema = {'$ref': '#/definitions/ExprOrSignalRef'}

    def __init__(self, expr=Undefined, **kwds):
        super(ExprOrSignalRef, self).__init__(expr=expr, **kwds)


class ExprRef(VegaLiteSchema):
    """ExprRef schema wrapper

    Mapping(required=[expr])

    Attributes
    ----------

    expr : string
        Vega expression (which can refer to Vega-Lite parameters).
    """
    _schema = {'$ref': '#/definitions/ExprRef'}

    def __init__(self, expr=Undefined, **kwds):
        super(ExprRef, self).__init__(expr=expr, **kwds)


class FacetEncodingFieldDef(VegaLiteSchema):
    """FacetEncodingFieldDef schema wrapper

    Mapping(required=[])

    Attributes
    ----------

    aggregate : :class:`Aggregate`
        Aggregation function for the field (e.g., ``"mean"``, ``"sum"``, ``"median"``,
        ``"min"``, ``"max"``, ``"count"`` ).

        **Default value:** ``undefined`` (None)

        **See also:** `aggregate <https://vega.github.io/vega-lite/docs/aggregate.html>`__
        documentation.
    align : anyOf(:class:`LayoutAlign`, :class:`RowColLayoutAlign`)
        The alignment to apply to grid rows and columns. The supported string values are
        ``"all"``, ``"each"``, and ``"none"``.


        * For ``"none"``, a flow layout will be used, in which adjacent subviews are simply
          placed one after the other. - For ``"each"``, subviews will be aligned into a
          clean grid structure, but each row or column may be of variable size. - For
          ``"all"``, subviews will be aligned and each row or column will be sized
          identically based on the maximum observed size. String values for this property
          will be applied to both grid rows and columns.

        Alternatively, an object value of the form ``{"row": string, "column": string}`` can
        be used to supply different alignments for rows and columns.

        **Default value:** ``"all"``.
    band : float
        For rect-based marks ( ``rect``, ``bar``, and ``image`` ), mark size relative to
        bandwidth of `band scales
        <https://vega.github.io/vega-lite/docs/scale.html#band>`__, bins or time units. If
        set to ``1``, the mark size is set to the bandwidth, the bin interval, or the time
        unit interval. If set to ``0.5``, the mark size is half of the bandwidth or the time
        unit interval.

        For other marks, relative position on a band of a stacked, binned, time unit or band
        scale. If set to ``0``, the marks will be positioned at the beginning of the band.
        If set to ``0.5``, the marks will be positioned in the middle of the band.
    bin : anyOf(boolean, :class:`BinParams`, None)
        A flag for binning a ``quantitative`` field, `an object defining binning parameters
        <https://vega.github.io/vega-lite/docs/bin.html#params>`__, or indicating that the
        data for ``x`` or ``y`` channel are binned before they are imported into Vega-Lite (
        ``"binned"`` ).


        If ``true``, default `binning parameters
        <https://vega.github.io/vega-lite/docs/bin.html>`__ will be applied.

        If ``"binned"``, this indicates that the data for the ``x`` (or ``y`` ) channel are
        already binned. You can map the bin-start field to ``x`` (or ``y`` ) and the bin-end
        field to ``x2`` (or ``y2`` ). The scale and axis will be formatted similar to
        binning in Vega-Lite.  To adjust the axis ticks based on the bin step, you can also
        set the axis's `tickMinStep
        <https://vega.github.io/vega-lite/docs/axis.html#ticks>`__ property.

        **Default value:** ``false``

        **See also:** `bin <https://vega.github.io/vega-lite/docs/bin.html>`__
        documentation.
    bounds : enum('full', 'flush')
        The bounds calculation method to use for determining the extent of a sub-plot. One
        of ``full`` (the default) or ``flush``.


        * If set to ``full``, the entire calculated bounds (including axes, title, and
          legend) will be used. - If set to ``flush``, only the specified width and height
          values for the sub-view will be used. The ``flush`` setting can be useful when
          attempting to place sub-plots without axes or legends into a uniform grid
          structure.

        **Default value:** ``"full"``
    center : anyOf(boolean, :class:`RowColboolean`)
        Boolean flag indicating if subviews should be centered relative to their respective
        rows or columns.

        An object value of the form ``{"row": boolean, "column": boolean}`` can be used to
        supply different centering values for rows and columns.

        **Default value:** ``false``
    columns : float
        The number of columns to include in the view composition layout.

        **Default value** : ``undefined`` -- An infinite number of columns (a single row)
        will be assumed. This is equivalent to ``hconcat`` (for ``concat`` ) and to using
        the ``column`` channel (for ``facet`` and ``repeat`` ).

        **Note** :

        1) This property is only for: - the general (wrappable) ``concat`` operator (not
        ``hconcat`` / ``vconcat`` ) - the ``facet`` and ``repeat`` operator with one
        field/repetition definition (without row/column nesting)

        2) Setting the ``columns`` to ``1`` is equivalent to ``vconcat`` (for ``concat`` )
        and to using the ``row`` channel (for ``facet`` and ``repeat`` ).
    field : :class:`Field`
        **Required.** A string defining the name of the field from which to pull a data
        value or an object defining iterated values from the `repeat
        <https://vega.github.io/vega-lite/docs/repeat.html>`__ operator.

        **See also:** `field <https://vega.github.io/vega-lite/docs/field.html>`__
        documentation.

        **Notes:** 1)  Dots ( ``.`` ) and brackets ( ``[`` and ``]`` ) can be used to access
        nested objects (e.g., ``"field": "foo.bar"`` and ``"field": "foo['bar']"`` ). If
        field names contain dots or brackets but are not nested, you can use ``\\`` to
        escape dots and brackets (e.g., ``"a\\.b"`` and ``"a\\[0\\]"`` ). See more details
        about escaping in the `field documentation
        <https://vega.github.io/vega-lite/docs/field.html>`__. 2) ``field`` is not required
        if ``aggregate`` is ``count``.
    header : :class:`Header`
        An object defining properties of a facet's header.
    sort : anyOf(:class:`SortArray`, :class:`SortOrder`, :class:`EncodingSortField`, None)
        Sort order for the encoded field.

        For continuous fields (quantitative or temporal), ``sort`` can be either
        ``"ascending"`` or ``"descending"``.

        For discrete fields, ``sort`` can be one of the following: - ``"ascending"`` or
        ``"descending"`` -- for sorting by the values' natural order in JavaScript. - `A
        sort field definition
        <https://vega.github.io/vega-lite/docs/sort.html#sort-field>`__ for sorting by
        another field. - `An array specifying the field values in preferred order
        <https://vega.github.io/vega-lite/docs/sort.html#sort-array>`__. In this case, the
        sort order will obey the values in the array, followed by any unspecified values in
        their original order. For discrete time field, values in the sort array can be
        `date-time definition objects <types#datetime>`__. In addition, for time units
        ``"month"`` and ``"day"``, the values can be the month or day names (case
        insensitive) or their 3-letter initials (e.g., ``"Mon"``, ``"Tue"`` ). - ``null``
        indicating no sort.

        **Default value:** ``"ascending"``

        **Note:** ``null`` is not supported for ``row`` and ``column``.
    spacing : anyOf(float, :class:`RowColnumber`)
        The spacing in pixels between sub-views of the composition operator. An object of
        the form ``{"row": number, "column": number}`` can be used to set different spacing
        values for rows and columns.

        **Default value** : Depends on ``"spacing"`` property of `the view composition
        configuration <https://vega.github.io/vega-lite/docs/config.html#view-config>`__ (
        ``20`` by default)
    timeUnit : anyOf(:class:`TimeUnit`, :class:`TimeUnitParams`)
        Time unit (e.g., ``year``, ``yearmonth``, ``month``, ``hours`` ) for a temporal
        field. or `a temporal field that gets casted as ordinal
        <https://vega.github.io/vega-lite/docs/type.html#cast>`__.

        **Default value:** ``undefined`` (None)

        **See also:** `timeUnit <https://vega.github.io/vega-lite/docs/timeunit.html>`__
        documentation.
    title : anyOf(:class:`Text`, None)
        A title for the field. If ``null``, the title will be removed.

        **Default value:**  derived from the field's name and transformation function (
        ``aggregate``, ``bin`` and ``timeUnit`` ). If the field has an aggregate function,
        the function is displayed as part of the title (e.g., ``"Sum of Profit"`` ). If the
        field is binned or has a time unit applied, the applied function is shown in
        parentheses (e.g., ``"Profit (binned)"``, ``"Transaction Date (year-month)"`` ).
        Otherwise, the title is simply the field name.

        **Notes** :

        1) You can customize the default field title format by providing the `fieldTitle
        <https://vega.github.io/vega-lite/docs/config.html#top-level-config>`__ property in
        the `config <https://vega.github.io/vega-lite/docs/config.html>`__ or `fieldTitle
        function via the compile function's options
        <https://vega.github.io/vega-lite/docs/compile.html#field-title>`__.

        2) If both field definition's ``title`` and axis, header, or legend ``title`` are
        defined, axis/header/legend title will be used.
    type : :class:`StandardType`
        The type of measurement ( ``"quantitative"``, ``"temporal"``, ``"ordinal"``, or
        ``"nominal"`` ) for the encoded field or constant value ( ``datum`` ). It can also
        be a ``"geojson"`` type for encoding `'geoshape'
        <https://vega.github.io/vega-lite/docs/geoshape.html>`__.

        Vega-Lite automatically infers data types in many cases as discussed below. However,
        type is required for a field if: (1) the field is not nominal and the field encoding
        has no specified ``aggregate`` (except ``argmin`` and ``argmax`` ), ``bin``, scale
        type, custom ``sort`` order, nor ``timeUnit`` or (2) if you wish to use an ordinal
        scale for a field with ``bin`` or ``timeUnit``.

        **Default value:**

        1) For a data ``field``, ``"nominal"`` is the default data type unless the field
        encoding has ``aggregate``, ``channel``, ``bin``, scale type, ``sort``, or
        ``timeUnit`` that satisfies the following criteria: - ``"quantitative"`` is the
        default type if (1) the encoded field contains ``bin`` or ``aggregate`` except
        ``"argmin"`` and ``"argmax"``, (2) the encoding channel is ``latitude`` or
        ``longitude`` channel or (3) if the specified scale type is `a quantitative scale
        <https://vega.github.io/vega-lite/docs/scale.html#type>`__. - ``"temporal"`` is the
        default type if (1) the encoded field contains ``timeUnit`` or (2) the specified
        scale type is a time or utc scale - ``ordinal""`` is the default type if (1) the
        encoded field contains a `custom sort order
        <https://vega.github.io/vega-lite/docs/sort.html#specifying-custom-sort-order>`__,
        (2) the specified scale type is an ordinal/point/band scale, or (3) the encoding
        channel is ``order``.

        2) For a constant value in data domain ( ``datum`` ): - ``"quantitative"`` if the
        datum is a number - ``"nominal"`` if the datum is a string - ``"temporal"`` if the
        datum is `a date time object
        <https://vega.github.io/vega-lite/docs/datetime.html>`__

        **Note:** - Data ``type`` describes the semantics of the data rather than the
        primitive data types (number, string, etc.). The same primitive data type can have
        different types of measurement. For example, numeric data can represent
        quantitative, ordinal, or nominal data. - Data values for a temporal field can be
        either a date-time string (e.g., ``"2015-03-07 12:32:17"``, ``"17:01"``,
        ``"2015-03-16"``. ``"2015"`` ) or a timestamp number (e.g., ``1552199579097`` ). -
        When using with `bin <https://vega.github.io/vega-lite/docs/bin.html>`__, the
        ``type`` property can be either ``"quantitative"`` (for using a linear bin scale) or
        `"ordinal" (for using an ordinal bin scale)
        <https://vega.github.io/vega-lite/docs/type.html#cast-bin>`__. - When using with
        `timeUnit <https://vega.github.io/vega-lite/docs/timeunit.html>`__, the ``type``
        property can be either ``"temporal"`` (default, for using a temporal scale) or
        `"ordinal" (for using an ordinal scale)
        <https://vega.github.io/vega-lite/docs/type.html#cast-bin>`__. - When using with
        `aggregate <https://vega.github.io/vega-lite/docs/aggregate.html>`__, the ``type``
        property refers to the post-aggregation data type. For example, we can calculate
        count ``distinct`` of a categorical field ``"cat"`` using ``{"aggregate":
        "distinct", "field": "cat"}``. The ``"type"`` of the aggregate output is
        ``"quantitative"``. - Secondary channels (e.g., ``x2``, ``y2``, ``xError``,
        ``yError`` ) do not have ``type`` as they must have exactly the same type as their
        primary channels (e.g., ``x``, ``y`` ).

        **See also:** `type <https://vega.github.io/vega-lite/docs/type.html>`__
        documentation.
    """
    _schema = {'$ref': '#/definitions/FacetEncodingFieldDef'}

    def __init__(self, aggregate=Undefined, align=Undefined, band=Undefined, bin=Undefined,
                 bounds=Undefined, center=Undefined, columns=Undefined, field=Undefined,
                 header=Undefined, sort=Undefined, spacing=Undefined, timeUnit=Undefined,
                 title=Undefined, type=Undefined, **kwds):
        super(FacetEncodingFieldDef, self).__init__(aggregate=aggregate, align=align, band=band,
                                                    bin=bin, bounds=bounds, center=center,
                                                    columns=columns, field=field, header=header,
                                                    sort=sort, spacing=spacing, timeUnit=timeUnit,
                                                    title=title, type=type, **kwds)


class FacetFieldDef(VegaLiteSchema):
    """FacetFieldDef schema wrapper

    Mapping(required=[])

    Attributes
    ----------

    aggregate : :class:`Aggregate`
        Aggregation function for the field (e.g., ``"mean"``, ``"sum"``, ``"median"``,
        ``"min"``, ``"max"``, ``"count"`` ).

        **Default value:** ``undefined`` (None)

        **See also:** `aggregate <https://vega.github.io/vega-lite/docs/aggregate.html>`__
        documentation.
    band : float
        For rect-based marks ( ``rect``, ``bar``, and ``image`` ), mark size relative to
        bandwidth of `band scales
        <https://vega.github.io/vega-lite/docs/scale.html#band>`__, bins or time units. If
        set to ``1``, the mark size is set to the bandwidth, the bin interval, or the time
        unit interval. If set to ``0.5``, the mark size is half of the bandwidth or the time
        unit interval.

        For other marks, relative position on a band of a stacked, binned, time unit or band
        scale. If set to ``0``, the marks will be positioned at the beginning of the band.
        If set to ``0.5``, the marks will be positioned in the middle of the band.
    bin : anyOf(boolean, :class:`BinParams`, None)
        A flag for binning a ``quantitative`` field, `an object defining binning parameters
        <https://vega.github.io/vega-lite/docs/bin.html#params>`__, or indicating that the
        data for ``x`` or ``y`` channel are binned before they are imported into Vega-Lite (
        ``"binned"`` ).


        If ``true``, default `binning parameters
        <https://vega.github.io/vega-lite/docs/bin.html>`__ will be applied.

        If ``"binned"``, this indicates that the data for the ``x`` (or ``y`` ) channel are
        already binned. You can map the bin-start field to ``x`` (or ``y`` ) and the bin-end
        field to ``x2`` (or ``y2`` ). The scale and axis will be formatted similar to
        binning in Vega-Lite.  To adjust the axis ticks based on the bin step, you can also
        set the axis's `tickMinStep
        <https://vega.github.io/vega-lite/docs/axis.html#ticks>`__ property.

        **Default value:** ``false``

        **See also:** `bin <https://vega.github.io/vega-lite/docs/bin.html>`__
        documentation.
    field : :class:`Field`
        **Required.** A string defining the name of the field from which to pull a data
        value or an object defining iterated values from the `repeat
        <https://vega.github.io/vega-lite/docs/repeat.html>`__ operator.

        **See also:** `field <https://vega.github.io/vega-lite/docs/field.html>`__
        documentation.

        **Notes:** 1)  Dots ( ``.`` ) and brackets ( ``[`` and ``]`` ) can be used to access
        nested objects (e.g., ``"field": "foo.bar"`` and ``"field": "foo['bar']"`` ). If
        field names contain dots or brackets but are not nested, you can use ``\\`` to
        escape dots and brackets (e.g., ``"a\\.b"`` and ``"a\\[0\\]"`` ). See more details
        about escaping in the `field documentation
        <https://vega.github.io/vega-lite/docs/field.html>`__. 2) ``field`` is not required
        if ``aggregate`` is ``count``.
    header : :class:`Header`
        An object defining properties of a facet's header.
    sort : anyOf(:class:`SortArray`, :class:`SortOrder`, :class:`EncodingSortField`, None)
        Sort order for the encoded field.

        For continuous fields (quantitative or temporal), ``sort`` can be either
        ``"ascending"`` or ``"descending"``.

        For discrete fields, ``sort`` can be one of the following: - ``"ascending"`` or
        ``"descending"`` -- for sorting by the values' natural order in JavaScript. - `A
        sort field definition
        <https://vega.github.io/vega-lite/docs/sort.html#sort-field>`__ for sorting by
        another field. - `An array specifying the field values in preferred order
        <https://vega.github.io/vega-lite/docs/sort.html#sort-array>`__. In this case, the
        sort order will obey the values in the array, followed by any unspecified values in
        their original order. For discrete time field, values in the sort array can be
        `date-time definition objects <types#datetime>`__. In addition, for time units
        ``"month"`` and ``"day"``, the values can be the month or day names (case
        insensitive) or their 3-letter initials (e.g., ``"Mon"``, ``"Tue"`` ). - ``null``
        indicating no sort.

        **Default value:** ``"ascending"``

        **Note:** ``null`` is not supported for ``row`` and ``column``.
    timeUnit : anyOf(:class:`TimeUnit`, :class:`TimeUnitParams`)
        Time unit (e.g., ``year``, ``yearmonth``, ``month``, ``hours`` ) for a temporal
        field. or `a temporal field that gets casted as ordinal
        <https://vega.github.io/vega-lite/docs/type.html#cast>`__.

        **Default value:** ``undefined`` (None)

        **See also:** `timeUnit <https://vega.github.io/vega-lite/docs/timeunit.html>`__
        documentation.
    title : anyOf(:class:`Text`, None)
        A title for the field. If ``null``, the title will be removed.

        **Default value:**  derived from the field's name and transformation function (
        ``aggregate``, ``bin`` and ``timeUnit`` ). If the field has an aggregate function,
        the function is displayed as part of the title (e.g., ``"Sum of Profit"`` ). If the
        field is binned or has a time unit applied, the applied function is shown in
        parentheses (e.g., ``"Profit (binned)"``, ``"Transaction Date (year-month)"`` ).
        Otherwise, the title is simply the field name.

        **Notes** :

        1) You can customize the default field title format by providing the `fieldTitle
        <https://vega.github.io/vega-lite/docs/config.html#top-level-config>`__ property in
        the `config <https://vega.github.io/vega-lite/docs/config.html>`__ or `fieldTitle
        function via the compile function's options
        <https://vega.github.io/vega-lite/docs/compile.html#field-title>`__.

        2) If both field definition's ``title`` and axis, header, or legend ``title`` are
        defined, axis/header/legend title will be used.
    type : :class:`StandardType`
        The type of measurement ( ``"quantitative"``, ``"temporal"``, ``"ordinal"``, or
        ``"nominal"`` ) for the encoded field or constant value ( ``datum`` ). It can also
        be a ``"geojson"`` type for encoding `'geoshape'
        <https://vega.github.io/vega-lite/docs/geoshape.html>`__.

        Vega-Lite automatically infers data types in many cases as discussed below. However,
        type is required for a field if: (1) the field is not nominal and the field encoding
        has no specified ``aggregate`` (except ``argmin`` and ``argmax`` ), ``bin``, scale
        type, custom ``sort`` order, nor ``timeUnit`` or (2) if you wish to use an ordinal
        scale for a field with ``bin`` or ``timeUnit``.

        **Default value:**

        1) For a data ``field``, ``"nominal"`` is the default data type unless the field
        encoding has ``aggregate``, ``channel``, ``bin``, scale type, ``sort``, or
        ``timeUnit`` that satisfies the following criteria: - ``"quantitative"`` is the
        default type if (1) the encoded field contains ``bin`` or ``aggregate`` except
        ``"argmin"`` and ``"argmax"``, (2) the encoding channel is ``latitude`` or
        ``longitude`` channel or (3) if the specified scale type is `a quantitative scale
        <https://vega.github.io/vega-lite/docs/scale.html#type>`__. - ``"temporal"`` is the
        default type if (1) the encoded field contains ``timeUnit`` or (2) the specified
        scale type is a time or utc scale - ``ordinal""`` is the default type if (1) the
        encoded field contains a `custom sort order
        <https://vega.github.io/vega-lite/docs/sort.html#specifying-custom-sort-order>`__,
        (2) the specified scale type is an ordinal/point/band scale, or (3) the encoding
        channel is ``order``.

        2) For a constant value in data domain ( ``datum`` ): - ``"quantitative"`` if the
        datum is a number - ``"nominal"`` if the datum is a string - ``"temporal"`` if the
        datum is `a date time object
        <https://vega.github.io/vega-lite/docs/datetime.html>`__

        **Note:** - Data ``type`` describes the semantics of the data rather than the
        primitive data types (number, string, etc.). The same primitive data type can have
        different types of measurement. For example, numeric data can represent
        quantitative, ordinal, or nominal data. - Data values for a temporal field can be
        either a date-time string (e.g., ``"2015-03-07 12:32:17"``, ``"17:01"``,
        ``"2015-03-16"``. ``"2015"`` ) or a timestamp number (e.g., ``1552199579097`` ). -
        When using with `bin <https://vega.github.io/vega-lite/docs/bin.html>`__, the
        ``type`` property can be either ``"quantitative"`` (for using a linear bin scale) or
        `"ordinal" (for using an ordinal bin scale)
        <https://vega.github.io/vega-lite/docs/type.html#cast-bin>`__. - When using with
        `timeUnit <https://vega.github.io/vega-lite/docs/timeunit.html>`__, the ``type``
        property can be either ``"temporal"`` (default, for using a temporal scale) or
        `"ordinal" (for using an ordinal scale)
        <https://vega.github.io/vega-lite/docs/type.html#cast-bin>`__. - When using with
        `aggregate <https://vega.github.io/vega-lite/docs/aggregate.html>`__, the ``type``
        property refers to the post-aggregation data type. For example, we can calculate
        count ``distinct`` of a categorical field ``"cat"`` using ``{"aggregate":
        "distinct", "field": "cat"}``. The ``"type"`` of the aggregate output is
        ``"quantitative"``. - Secondary channels (e.g., ``x2``, ``y2``, ``xError``,
        ``yError`` ) do not have ``type`` as they must have exactly the same type as their
        primary channels (e.g., ``x``, ``y`` ).

        **See also:** `type <https://vega.github.io/vega-lite/docs/type.html>`__
        documentation.
    """
    _schema = {'$ref': '#/definitions/FacetFieldDef'}

    def __init__(self, aggregate=Undefined, band=Undefined, bin=Undefined, field=Undefined,
                 header=Undefined, sort=Undefined, timeUnit=Undefined, title=Undefined, type=Undefined,
                 **kwds):
        super(FacetFieldDef, self).__init__(aggregate=aggregate, band=band, bin=bin, field=field,
                                            header=header, sort=sort, timeUnit=timeUnit, title=title,
                                            type=type, **kwds)


class FacetFieldDefFieldName(VegaLiteSchema):
    """FacetFieldDefFieldName schema wrapper

    Mapping(required=[])

    Attributes
    ----------

    aggregate : :class:`Aggregate`
        Aggregation function for the field (e.g., ``"mean"``, ``"sum"``, ``"median"``,
        ``"min"``, ``"max"``, ``"count"`` ).

        **Default value:** ``undefined`` (None)

        **See also:** `aggregate <https://vega.github.io/vega-lite/docs/aggregate.html>`__
        documentation.
    band : float
        For rect-based marks ( ``rect``, ``bar``, and ``image`` ), mark size relative to
        bandwidth of `band scales
        <https://vega.github.io/vega-lite/docs/scale.html#band>`__, bins or time units. If
        set to ``1``, the mark size is set to the bandwidth, the bin interval, or the time
        unit interval. If set to ``0.5``, the mark size is half of the bandwidth or the time
        unit interval.

        For other marks, relative position on a band of a stacked, binned, time unit or band
        scale. If set to ``0``, the marks will be positioned at the beginning of the band.
        If set to ``0.5``, the marks will be positioned in the middle of the band.
    bin : anyOf(boolean, :class:`BinParams`, None)
        A flag for binning a ``quantitative`` field, `an object defining binning parameters
        <https://vega.github.io/vega-lite/docs/bin.html#params>`__, or indicating that the
        data for ``x`` or ``y`` channel are binned before they are imported into Vega-Lite (
        ``"binned"`` ).


        If ``true``, default `binning parameters
        <https://vega.github.io/vega-lite/docs/bin.html>`__ will be applied.

        If ``"binned"``, this indicates that the data for the ``x`` (or ``y`` ) channel are
        already binned. You can map the bin-start field to ``x`` (or ``y`` ) and the bin-end
        field to ``x2`` (or ``y2`` ). The scale and axis will be formatted similar to
        binning in Vega-Lite.  To adjust the axis ticks based on the bin step, you can also
        set the axis's `tickMinStep
        <https://vega.github.io/vega-lite/docs/axis.html#ticks>`__ property.

        **Default value:** ``false``

        **See also:** `bin <https://vega.github.io/vega-lite/docs/bin.html>`__
        documentation.
    field : :class:`FieldName`
        **Required.** A string defining the name of the field from which to pull a data
        value or an object defining iterated values from the `repeat
        <https://vega.github.io/vega-lite/docs/repeat.html>`__ operator.

        **See also:** `field <https://vega.github.io/vega-lite/docs/field.html>`__
        documentation.

        **Notes:** 1)  Dots ( ``.`` ) and brackets ( ``[`` and ``]`` ) can be used to access
        nested objects (e.g., ``"field": "foo.bar"`` and ``"field": "foo['bar']"`` ). If
        field names contain dots or brackets but are not nested, you can use ``\\`` to
        escape dots and brackets (e.g., ``"a\\.b"`` and ``"a\\[0\\]"`` ). See more details
        about escaping in the `field documentation
        <https://vega.github.io/vega-lite/docs/field.html>`__. 2) ``field`` is not required
        if ``aggregate`` is ``count``.
    header : :class:`Header`
        An object defining properties of a facet's header.
    sort : anyOf(:class:`SortArray`, :class:`SortOrder`, :class:`EncodingSortFieldFieldName`,
    None)
        Sort order for the encoded field.

        For continuous fields (quantitative or temporal), ``sort`` can be either
        ``"ascending"`` or ``"descending"``.

        For discrete fields, ``sort`` can be one of the following: - ``"ascending"`` or
        ``"descending"`` -- for sorting by the values' natural order in JavaScript. - `A
        sort field definition
        <https://vega.github.io/vega-lite/docs/sort.html#sort-field>`__ for sorting by
        another field. - `An array specifying the field values in preferred order
        <https://vega.github.io/vega-lite/docs/sort.html#sort-array>`__. In this case, the
        sort order will obey the values in the array, followed by any unspecified values in
        their original order. For discrete time field, values in the sort array can be
        `date-time definition objects <types#datetime>`__. In addition, for time units
        ``"month"`` and ``"day"``, the values can be the month or day names (case
        insensitive) or their 3-letter initials (e.g., ``"Mon"``, ``"Tue"`` ). - ``null``
        indicating no sort.

        **Default value:** ``"ascending"``

        **Note:** ``null`` is not supported for ``row`` and ``column``.
    timeUnit : anyOf(:class:`TimeUnit`, :class:`TimeUnitParams`)
        Time unit (e.g., ``year``, ``yearmonth``, ``month``, ``hours`` ) for a temporal
        field. or `a temporal field that gets casted as ordinal
        <https://vega.github.io/vega-lite/docs/type.html#cast>`__.

        **Default value:** ``undefined`` (None)

        **See also:** `timeUnit <https://vega.github.io/vega-lite/docs/timeunit.html>`__
        documentation.
    title : anyOf(:class:`Text`, None)
        A title for the field. If ``null``, the title will be removed.

        **Default value:**  derived from the field's name and transformation function (
        ``aggregate``, ``bin`` and ``timeUnit`` ). If the field has an aggregate function,
        the function is displayed as part of the title (e.g., ``"Sum of Profit"`` ). If the
        field is binned or has a time unit applied, the applied function is shown in
        parentheses (e.g., ``"Profit (binned)"``, ``"Transaction Date (year-month)"`` ).
        Otherwise, the title is simply the field name.

        **Notes** :

        1) You can customize the default field title format by providing the `fieldTitle
        <https://vega.github.io/vega-lite/docs/config.html#top-level-config>`__ property in
        the `config <https://vega.github.io/vega-lite/docs/config.html>`__ or `fieldTitle
        function via the compile function's options
        <https://vega.github.io/vega-lite/docs/compile.html#field-title>`__.

        2) If both field definition's ``title`` and axis, header, or legend ``title`` are
        defined, axis/header/legend title will be used.
    type : :class:`StandardType`
        The type of measurement ( ``"quantitative"``, ``"temporal"``, ``"ordinal"``, or
        ``"nominal"`` ) for the encoded field or constant value ( ``datum`` ). It can also
        be a ``"geojson"`` type for encoding `'geoshape'
        <https://vega.github.io/vega-lite/docs/geoshape.html>`__.

        Vega-Lite automatically infers data types in many cases as discussed below. However,
        type is required for a field if: (1) the field is not nominal and the field encoding
        has no specified ``aggregate`` (except ``argmin`` and ``argmax`` ), ``bin``, scale
        type, custom ``sort`` order, nor ``timeUnit`` or (2) if you wish to use an ordinal
        scale for a field with ``bin`` or ``timeUnit``.

        **Default value:**

        1) For a data ``field``, ``"nominal"`` is the default data type unless the field
        encoding has ``aggregate``, ``channel``, ``bin``, scale type, ``sort``, or
        ``timeUnit`` that satisfies the following criteria: - ``"quantitative"`` is the
        default type if (1) the encoded field contains ``bin`` or ``aggregate`` except
        ``"argmin"`` and ``"argmax"``, (2) the encoding channel is ``latitude`` or
        ``longitude`` channel or (3) if the specified scale type is `a quantitative scale
        <https://vega.github.io/vega-lite/docs/scale.html#type>`__. - ``"temporal"`` is the
        default type if (1) the encoded field contains ``timeUnit`` or (2) the specified
        scale type is a time or utc scale - ``ordinal""`` is the default type if (1) the
        encoded field contains a `custom sort order
        <https://vega.github.io/vega-lite/docs/sort.html#specifying-custom-sort-order>`__,
        (2) the specified scale type is an ordinal/point/band scale, or (3) the encoding
        channel is ``order``.

        2) For a constant value in data domain ( ``datum`` ): - ``"quantitative"`` if the
        datum is a number - ``"nominal"`` if the datum is a string - ``"temporal"`` if the
        datum is `a date time object
        <https://vega.github.io/vega-lite/docs/datetime.html>`__

        **Note:** - Data ``type`` describes the semantics of the data rather than the
        primitive data types (number, string, etc.). The same primitive data type can have
        different types of measurement. For example, numeric data can represent
        quantitative, ordinal, or nominal data. - Data values for a temporal field can be
        either a date-time string (e.g., ``"2015-03-07 12:32:17"``, ``"17:01"``,
        ``"2015-03-16"``. ``"2015"`` ) or a timestamp number (e.g., ``1552199579097`` ). -
        When using with `bin <https://vega.github.io/vega-lite/docs/bin.html>`__, the
        ``type`` property can be either ``"quantitative"`` (for using a linear bin scale) or
        `"ordinal" (for using an ordinal bin scale)
        <https://vega.github.io/vega-lite/docs/type.html#cast-bin>`__. - When using with
        `timeUnit <https://vega.github.io/vega-lite/docs/timeunit.html>`__, the ``type``
        property can be either ``"temporal"`` (default, for using a temporal scale) or
        `"ordinal" (for using an ordinal scale)
        <https://vega.github.io/vega-lite/docs/type.html#cast-bin>`__. - When using with
        `aggregate <https://vega.github.io/vega-lite/docs/aggregate.html>`__, the ``type``
        property refers to the post-aggregation data type. For example, we can calculate
        count ``distinct`` of a categorical field ``"cat"`` using ``{"aggregate":
        "distinct", "field": "cat"}``. The ``"type"`` of the aggregate output is
        ``"quantitative"``. - Secondary channels (e.g., ``x2``, ``y2``, ``xError``,
        ``yError`` ) do not have ``type`` as they must have exactly the same type as their
        primary channels (e.g., ``x``, ``y`` ).

        **See also:** `type <https://vega.github.io/vega-lite/docs/type.html>`__
        documentation.
    """
    _schema = {'$ref': '#/definitions/FacetFieldDef<FieldName>'}

    def __init__(self, aggregate=Undefined, band=Undefined, bin=Undefined, field=Undefined,
                 header=Undefined, sort=Undefined, timeUnit=Undefined, title=Undefined, type=Undefined,
                 **kwds):
        super(FacetFieldDefFieldName, self).__init__(aggregate=aggregate, band=band, bin=bin,
                                                     field=field, header=header, sort=sort,
                                                     timeUnit=timeUnit, title=title, type=type, **kwds)


class FacetMapping(VegaLiteSchema):
    """FacetMapping schema wrapper

    Mapping(required=[])

    Attributes
    ----------

    column : :class:`FacetFieldDef`
        A field definition for the horizontal facet of trellis plots.
    row : :class:`FacetFieldDef`
        A field definition for the vertical facet of trellis plots.
    """
    _schema = {'$ref': '#/definitions/FacetMapping'}

    def __init__(self, column=Undefined, row=Undefined, **kwds):
        super(FacetMapping, self).__init__(column=column, row=row, **kwds)


class FacetMappingFieldName(VegaLiteSchema):
    """FacetMappingFieldName schema wrapper

    Mapping(required=[])

    Attributes
    ----------

    column : :class:`FacetFieldDefFieldName`
        A field definition for the horizontal facet of trellis plots.
    row : :class:`FacetFieldDefFieldName`
        A field definition for the vertical facet of trellis plots.
    """
    _schema = {'$ref': '#/definitions/FacetMapping<FieldName>'}

    def __init__(self, column=Undefined, row=Undefined, **kwds):
        super(FacetMappingFieldName, self).__init__(column=column, row=row, **kwds)


class FacetedEncoding(VegaLiteSchema):
    """FacetedEncoding schema wrapper

    Mapping(required=[])

    Attributes
    ----------

    angle : :class:`NumericMarkPropDef`
        Rotation angle of point and text marks.
    color : :class:`ColorDef`
        Color of the marks – either fill or stroke color based on  the ``filled`` property
        of mark definition. By default, ``color`` represents fill color for ``"area"``,
        ``"bar"``, ``"tick"``, ``"text"``, ``"trail"``, ``"circle"``, and ``"square"`` /
        stroke color for ``"line"`` and ``"point"``.

        **Default value:** If undefined, the default color depends on `mark config
        <https://vega.github.io/vega-lite/docs/config.html#mark-config>`__ 's ``color``
        property.

        *Note:* 1) For fine-grained control over both fill and stroke colors of the marks,
        please use the ``fill`` and ``stroke`` channels. The ``fill`` or ``stroke``
        encodings have higher precedence than ``color``, thus may override the ``color``
        encoding if conflicting encodings are specified. 2) See the scale documentation for
        more information about customizing `color scheme
        <https://vega.github.io/vega-lite/docs/scale.html#scheme>`__.
    column : :class:`RowColumnEncodingFieldDef`
        A field definition for the horizontal facet of trellis plots.
    description : anyOf(:class:`StringFieldDefWithCondition`,
    :class:`StringValueDefWithCondition`)
        A text description of this mark for ARIA accessibility (SVG output only). For SVG
        output the ``"aria-label"`` attribute will be set to this description.
    detail : anyOf(:class:`FieldDefWithoutScale`, List(:class:`FieldDefWithoutScale`))
        Additional levels of detail for grouping data in aggregate views and in line, trail,
        and area marks without mapping data to a specific visual channel.
    facet : :class:`FacetEncodingFieldDef`
        A field definition for the (flexible) facet of trellis plots.

        If either ``row`` or ``column`` is specified, this channel will be ignored.
    fill : :class:`ColorDef`
        Fill color of the marks. **Default value:** If undefined, the default color depends
        on `mark config <https://vega.github.io/vega-lite/docs/config.html#mark-config>`__
        's ``color`` property.

        *Note:* The ``fill`` encoding has higher precedence than ``color``, thus may
        override the ``color`` encoding if conflicting encodings are specified.
    fillOpacity : :class:`NumericMarkPropDef`
        Fill opacity of the marks.

        **Default value:** If undefined, the default opacity depends on `mark config
        <https://vega.github.io/vega-lite/docs/config.html#mark-config>`__ 's
        ``fillOpacity`` property.
    href : anyOf(:class:`StringFieldDefWithCondition`, :class:`StringValueDefWithCondition`)
        A URL to load upon mouse click.
    key : :class:`FieldDefWithoutScale`
        A data field to use as a unique key for data binding. When a visualization’s data is
        updated, the key value will be used to match data elements to existing mark
        instances. Use a key channel to enable object constancy for transitions over dynamic
        data.
    latitude : :class:`LatLongDef`
        Latitude position of geographically projected marks.
    latitude2 : :class:`Position2Def`
        Latitude-2 position for geographically projected ranged ``"area"``, ``"bar"``,
        ``"rect"``, and  ``"rule"``.
    longitude : :class:`LatLongDef`
        Longitude position of geographically projected marks.
    longitude2 : :class:`Position2Def`
        Longitude-2 position for geographically projected ranged ``"area"``, ``"bar"``,
        ``"rect"``, and  ``"rule"``.
    opacity : :class:`NumericMarkPropDef`
        Opacity of the marks.

        **Default value:** If undefined, the default opacity depends on `mark config
        <https://vega.github.io/vega-lite/docs/config.html#mark-config>`__ 's ``opacity``
        property.
    order : anyOf(:class:`OrderFieldDef`, List(:class:`OrderFieldDef`), :class:`OrderValueDef`)
        Order of the marks. - For stacked marks, this ``order`` channel encodes `stack order
        <https://vega.github.io/vega-lite/docs/stack.html#order>`__. - For line and trail
        marks, this ``order`` channel encodes order of data points in the lines. This can be
        useful for creating `a connected scatterplot
        <https://vega.github.io/vega-lite/examples/connected_scatterplot.html>`__. Setting
        ``order`` to ``{"value": null}`` makes the line marks use the original order in the
        data sources. - Otherwise, this ``order`` channel encodes layer order of the marks.

        **Note** : In aggregate plots, ``order`` field should be ``aggregate`` d to avoid
        creating additional aggregation grouping.
    radius : :class:`PolarDef`
        The outer radius in pixels of arc marks.
    radius2 : :class:`Position2Def`
        The inner radius in pixels of arc marks.
    row : :class:`RowColumnEncodingFieldDef`
        A field definition for the vertical facet of trellis plots.
    shape : :class:`ShapeDef`
        Shape of the mark.


        #.
        For ``point`` marks the supported values include:    - plotting shapes:
        ``"circle"``, ``"square"``, ``"cross"``, ``"diamond"``, ``"triangle-up"``,
        ``"triangle-down"``, ``"triangle-right"``, or ``"triangle-left"``.    - the line
        symbol ``"stroke"``    - centered directional shapes ``"arrow"``, ``"wedge"``, or
        ``"triangle"``    - a custom `SVG path string
        <https://developer.mozilla.org/en-US/docs/Web/SVG/Tutorial/Paths>`__ (For correct
        sizing, custom shape paths should be defined within a square bounding box with
        coordinates ranging from -1 to 1 along both the x and y dimensions.)

        #.
        For ``geoshape`` marks it should be a field definition of the geojson data

        **Default value:** If undefined, the default shape depends on `mark config
        <https://vega.github.io/vega-lite/docs/config.html#point-config>`__ 's ``shape``
        property. ( ``"circle"`` if unset.)
    size : :class:`NumericMarkPropDef`
        Size of the mark. - For ``"point"``, ``"square"`` and ``"circle"``, – the symbol
        size, or pixel area of the mark. - For ``"bar"`` and ``"tick"`` – the bar and tick's
        size. - For ``"text"`` – the text's font size. - Size is unsupported for ``"line"``,
        ``"area"``, and ``"rect"``. (Use ``"trail"`` instead of line with varying size)
    stroke : :class:`ColorDef`
        Stroke color of the marks. **Default value:** If undefined, the default color
        depends on `mark config
        <https://vega.github.io/vega-lite/docs/config.html#mark-config>`__ 's ``color``
        property.

        *Note:* The ``stroke`` encoding has higher precedence than ``color``, thus may
        override the ``color`` encoding if conflicting encodings are specified.
    strokeDash : :class:`NumericArrayMarkPropDef`
        Stroke dash of the marks.

        **Default value:** ``[1,0]`` (No dash).
    strokeOpacity : :class:`NumericMarkPropDef`
        Stroke opacity of the marks.

        **Default value:** If undefined, the default opacity depends on `mark config
        <https://vega.github.io/vega-lite/docs/config.html#mark-config>`__ 's
        ``strokeOpacity`` property.
    strokeWidth : :class:`NumericMarkPropDef`
        Stroke width of the marks.

        **Default value:** If undefined, the default stroke width depends on `mark config
        <https://vega.github.io/vega-lite/docs/config.html#mark-config>`__ 's
        ``strokeWidth`` property.
    text : :class:`TextDef`
        Text of the ``text`` mark.
    theta : :class:`PolarDef`
        For arc marks, the arc length in radians if theta2 is not specified, otherwise the
        start arc angle. (A value of 0 indicates up or “north”, increasing values proceed
        clockwise.)

        For text marks, polar coordinate angle in radians.
    theta2 : :class:`Position2Def`
        The end angle of arc marks in radians. A value of 0 indicates up or “north”,
        increasing values proceed clockwise.
    tooltip : anyOf(:class:`StringFieldDefWithCondition`, :class:`StringValueDefWithCondition`,
    List(:class:`StringFieldDef`), None)
        The tooltip text to show upon mouse hover. Specifying ``tooltip`` encoding overrides
        `the tooltip property in the mark definition
        <https://vega.github.io/vega-lite/docs/mark.html#mark-def>`__.

        See the `tooltip <https://vega.github.io/vega-lite/docs/tooltip.html>`__
        documentation for a detailed discussion about tooltip in Vega-Lite.
    url : anyOf(:class:`StringFieldDefWithCondition`, :class:`StringValueDefWithCondition`)
        The URL of an image mark.
    x : :class:`PositionDef`
        X coordinates of the marks, or width of horizontal ``"bar"`` and ``"area"`` without
        specified ``x2`` or ``width``.

        The ``value`` of this channel can be a number or a string ``"width"`` for the width
        of the plot.
    x2 : :class:`Position2Def`
        X2 coordinates for ranged ``"area"``, ``"bar"``, ``"rect"``, and  ``"rule"``.

        The ``value`` of this channel can be a number or a string ``"width"`` for the width
        of the plot.
    xError : anyOf(:class:`SecondaryFieldDef`, :class:`ValueDefnumber`)
        Error value of x coordinates for error specified ``"errorbar"`` and ``"errorband"``.
    xError2 : anyOf(:class:`SecondaryFieldDef`, :class:`ValueDefnumber`)
        Secondary error value of x coordinates for error specified ``"errorbar"`` and
        ``"errorband"``.
    y : :class:`PositionDef`
        Y coordinates of the marks, or height of vertical ``"bar"`` and ``"area"`` without
        specified ``y2`` or ``height``.

        The ``value`` of this channel can be a number or a string ``"height"`` for the
        height of the plot.
    y2 : :class:`Position2Def`
        Y2 coordinates for ranged ``"area"``, ``"bar"``, ``"rect"``, and  ``"rule"``.

        The ``value`` of this channel can be a number or a string ``"height"`` for the
        height of the plot.
    yError : anyOf(:class:`SecondaryFieldDef`, :class:`ValueDefnumber`)
        Error value of y coordinates for error specified ``"errorbar"`` and ``"errorband"``.
    yError2 : anyOf(:class:`SecondaryFieldDef`, :class:`ValueDefnumber`)
        Secondary error value of y coordinates for error specified ``"errorbar"`` and
        ``"errorband"``.
    """
    _schema = {'$ref': '#/definitions/FacetedEncoding'}

    def __init__(self, angle=Undefined, color=Undefined, column=Undefined, description=Undefined,
                 detail=Undefined, facet=Undefined, fill=Undefined, fillOpacity=Undefined,
                 href=Undefined, key=Undefined, latitude=Undefined, latitude2=Undefined,
                 longitude=Undefined, longitude2=Undefined, opacity=Undefined, order=Undefined,
                 radius=Undefined, radius2=Undefined, row=Undefined, shape=Undefined, size=Undefined,
                 stroke=Undefined, strokeDash=Undefined, strokeOpacity=Undefined, strokeWidth=Undefined,
                 text=Undefined, theta=Undefined, theta2=Undefined, tooltip=Undefined, url=Undefined,
                 x=Undefined, x2=Undefined, xError=Undefined, xError2=Undefined, y=Undefined,
                 y2=Undefined, yError=Undefined, yError2=Undefined, **kwds):
        super(FacetedEncoding, self).__init__(angle=angle, color=color, column=column,
                                              description=description, detail=detail, facet=facet,
                                              fill=fill, fillOpacity=fillOpacity, href=href, key=key,
                                              latitude=latitude, latitude2=latitude2,
                                              longitude=longitude, longitude2=longitude2,
                                              opacity=opacity, order=order, radius=radius,
                                              radius2=radius2, row=row, shape=shape, size=size,
                                              stroke=stroke, strokeDash=strokeDash,
                                              strokeOpacity=strokeOpacity, strokeWidth=strokeWidth,
                                              text=text, theta=theta, theta2=theta2, tooltip=tooltip,
                                              url=url, x=x, x2=x2, xError=xError, xError2=xError2, y=y,
                                              y2=y2, yError=yError, yError2=yError2, **kwds)


class Field(VegaLiteSchema):
    """Field schema wrapper

    anyOf(:class:`FieldName`, :class:`RepeatRef`)
    """
    _schema = {'$ref': '#/definitions/Field'}

    def __init__(self, *args, **kwds):
        super(Field, self).__init__(*args, **kwds)


class FieldDefWithoutScale(VegaLiteSchema):
    """FieldDefWithoutScale schema wrapper

    Mapping(required=[])
    Definition object for a data field, its type and transformation of an encoding channel.

    Attributes
    ----------

    aggregate : :class:`Aggregate`
        Aggregation function for the field (e.g., ``"mean"``, ``"sum"``, ``"median"``,
        ``"min"``, ``"max"``, ``"count"`` ).

        **Default value:** ``undefined`` (None)

        **See also:** `aggregate <https://vega.github.io/vega-lite/docs/aggregate.html>`__
        documentation.
    band : float
        For rect-based marks ( ``rect``, ``bar``, and ``image`` ), mark size relative to
        bandwidth of `band scales
        <https://vega.github.io/vega-lite/docs/scale.html#band>`__, bins or time units. If
        set to ``1``, the mark size is set to the bandwidth, the bin interval, or the time
        unit interval. If set to ``0.5``, the mark size is half of the bandwidth or the time
        unit interval.

        For other marks, relative position on a band of a stacked, binned, time unit or band
        scale. If set to ``0``, the marks will be positioned at the beginning of the band.
        If set to ``0.5``, the marks will be positioned in the middle of the band.
    bin : anyOf(boolean, :class:`BinParams`, string, None)
        A flag for binning a ``quantitative`` field, `an object defining binning parameters
        <https://vega.github.io/vega-lite/docs/bin.html#params>`__, or indicating that the
        data for ``x`` or ``y`` channel are binned before they are imported into Vega-Lite (
        ``"binned"`` ).


        If ``true``, default `binning parameters
        <https://vega.github.io/vega-lite/docs/bin.html>`__ will be applied.

        If ``"binned"``, this indicates that the data for the ``x`` (or ``y`` ) channel are
        already binned. You can map the bin-start field to ``x`` (or ``y`` ) and the bin-end
        field to ``x2`` (or ``y2`` ). The scale and axis will be formatted similar to
        binning in Vega-Lite.  To adjust the axis ticks based on the bin step, you can also
        set the axis's `tickMinStep
        <https://vega.github.io/vega-lite/docs/axis.html#ticks>`__ property.

        **Default value:** ``false``

        **See also:** `bin <https://vega.github.io/vega-lite/docs/bin.html>`__
        documentation.
    field : :class:`Field`
        **Required.** A string defining the name of the field from which to pull a data
        value or an object defining iterated values from the `repeat
        <https://vega.github.io/vega-lite/docs/repeat.html>`__ operator.

        **See also:** `field <https://vega.github.io/vega-lite/docs/field.html>`__
        documentation.

        **Notes:** 1)  Dots ( ``.`` ) and brackets ( ``[`` and ``]`` ) can be used to access
        nested objects (e.g., ``"field": "foo.bar"`` and ``"field": "foo['bar']"`` ). If
        field names contain dots or brackets but are not nested, you can use ``\\`` to
        escape dots and brackets (e.g., ``"a\\.b"`` and ``"a\\[0\\]"`` ). See more details
        about escaping in the `field documentation
        <https://vega.github.io/vega-lite/docs/field.html>`__. 2) ``field`` is not required
        if ``aggregate`` is ``count``.
    timeUnit : anyOf(:class:`TimeUnit`, :class:`TimeUnitParams`)
        Time unit (e.g., ``year``, ``yearmonth``, ``month``, ``hours`` ) for a temporal
        field. or `a temporal field that gets casted as ordinal
        <https://vega.github.io/vega-lite/docs/type.html#cast>`__.

        **Default value:** ``undefined`` (None)

        **See also:** `timeUnit <https://vega.github.io/vega-lite/docs/timeunit.html>`__
        documentation.
    title : anyOf(:class:`Text`, None)
        A title for the field. If ``null``, the title will be removed.

        **Default value:**  derived from the field's name and transformation function (
        ``aggregate``, ``bin`` and ``timeUnit`` ). If the field has an aggregate function,
        the function is displayed as part of the title (e.g., ``"Sum of Profit"`` ). If the
        field is binned or has a time unit applied, the applied function is shown in
        parentheses (e.g., ``"Profit (binned)"``, ``"Transaction Date (year-month)"`` ).
        Otherwise, the title is simply the field name.

        **Notes** :

        1) You can customize the default field title format by providing the `fieldTitle
        <https://vega.github.io/vega-lite/docs/config.html#top-level-config>`__ property in
        the `config <https://vega.github.io/vega-lite/docs/config.html>`__ or `fieldTitle
        function via the compile function's options
        <https://vega.github.io/vega-lite/docs/compile.html#field-title>`__.

        2) If both field definition's ``title`` and axis, header, or legend ``title`` are
        defined, axis/header/legend title will be used.
    type : :class:`StandardType`
        The type of measurement ( ``"quantitative"``, ``"temporal"``, ``"ordinal"``, or
        ``"nominal"`` ) for the encoded field or constant value ( ``datum`` ). It can also
        be a ``"geojson"`` type for encoding `'geoshape'
        <https://vega.github.io/vega-lite/docs/geoshape.html>`__.

        Vega-Lite automatically infers data types in many cases as discussed below. However,
        type is required for a field if: (1) the field is not nominal and the field encoding
        has no specified ``aggregate`` (except ``argmin`` and ``argmax`` ), ``bin``, scale
        type, custom ``sort`` order, nor ``timeUnit`` or (2) if you wish to use an ordinal
        scale for a field with ``bin`` or ``timeUnit``.

        **Default value:**

        1) For a data ``field``, ``"nominal"`` is the default data type unless the field
        encoding has ``aggregate``, ``channel``, ``bin``, scale type, ``sort``, or
        ``timeUnit`` that satisfies the following criteria: - ``"quantitative"`` is the
        default type if (1) the encoded field contains ``bin`` or ``aggregate`` except
        ``"argmin"`` and ``"argmax"``, (2) the encoding channel is ``latitude`` or
        ``longitude`` channel or (3) if the specified scale type is `a quantitative scale
        <https://vega.github.io/vega-lite/docs/scale.html#type>`__. - ``"temporal"`` is the
        default type if (1) the encoded field contains ``timeUnit`` or (2) the specified
        scale type is a time or utc scale - ``ordinal""`` is the default type if (1) the
        encoded field contains a `custom sort order
        <https://vega.github.io/vega-lite/docs/sort.html#specifying-custom-sort-order>`__,
        (2) the specified scale type is an ordinal/point/band scale, or (3) the encoding
        channel is ``order``.

        2) For a constant value in data domain ( ``datum`` ): - ``"quantitative"`` if the
        datum is a number - ``"nominal"`` if the datum is a string - ``"temporal"`` if the
        datum is `a date time object
        <https://vega.github.io/vega-lite/docs/datetime.html>`__

        **Note:** - Data ``type`` describes the semantics of the data rather than the
        primitive data types (number, string, etc.). The same primitive data type can have
        different types of measurement. For example, numeric data can represent
        quantitative, ordinal, or nominal data. - Data values for a temporal field can be
        either a date-time string (e.g., ``"2015-03-07 12:32:17"``, ``"17:01"``,
        ``"2015-03-16"``. ``"2015"`` ) or a timestamp number (e.g., ``1552199579097`` ). -
        When using with `bin <https://vega.github.io/vega-lite/docs/bin.html>`__, the
        ``type`` property can be either ``"quantitative"`` (for using a linear bin scale) or
        `"ordinal" (for using an ordinal bin scale)
        <https://vega.github.io/vega-lite/docs/type.html#cast-bin>`__. - When using with
        `timeUnit <https://vega.github.io/vega-lite/docs/timeunit.html>`__, the ``type``
        property can be either ``"temporal"`` (default, for using a temporal scale) or
        `"ordinal" (for using an ordinal scale)
        <https://vega.github.io/vega-lite/docs/type.html#cast-bin>`__. - When using with
        `aggregate <https://vega.github.io/vega-lite/docs/aggregate.html>`__, the ``type``
        property refers to the post-aggregation data type. For example, we can calculate
        count ``distinct`` of a categorical field ``"cat"`` using ``{"aggregate":
        "distinct", "field": "cat"}``. The ``"type"`` of the aggregate output is
        ``"quantitative"``. - Secondary channels (e.g., ``x2``, ``y2``, ``xError``,
        ``yError`` ) do not have ``type`` as they must have exactly the same type as their
        primary channels (e.g., ``x``, ``y`` ).

        **See also:** `type <https://vega.github.io/vega-lite/docs/type.html>`__
        documentation.
    """
    _schema = {'$ref': '#/definitions/FieldDefWithoutScale'}

    def __init__(self, aggregate=Undefined, band=Undefined, bin=Undefined, field=Undefined,
                 timeUnit=Undefined, title=Undefined, type=Undefined, **kwds):
        super(FieldDefWithoutScale, self).__init__(aggregate=aggregate, band=band, bin=bin, field=field,
                                                   timeUnit=timeUnit, title=title, type=type, **kwds)


class FieldName(Field):
    """FieldName schema wrapper

    string
    """
    _schema = {'$ref': '#/definitions/FieldName'}

    def __init__(self, *args):
        super(FieldName, self).__init__(*args)


class FieldOrDatumDefWithConditionStringFieldDefstring(VegaLiteSchema):
    """FieldOrDatumDefWithConditionStringFieldDefstring schema wrapper

    Mapping(required=[])
    A FieldDef with Condition :raw-html:`<ValueDef>` {    condition: {value: ...},    field:
    ...,    ... }

    Attributes
    ----------

    aggregate : :class:`Aggregate`
        Aggregation function for the field (e.g., ``"mean"``, ``"sum"``, ``"median"``,
        ``"min"``, ``"max"``, ``"count"`` ).

        **Default value:** ``undefined`` (None)

        **See also:** `aggregate <https://vega.github.io/vega-lite/docs/aggregate.html>`__
        documentation.
    band : float
        For rect-based marks ( ``rect``, ``bar``, and ``image`` ), mark size relative to
        bandwidth of `band scales
        <https://vega.github.io/vega-lite/docs/scale.html#band>`__, bins or time units. If
        set to ``1``, the mark size is set to the bandwidth, the bin interval, or the time
        unit interval. If set to ``0.5``, the mark size is half of the bandwidth or the time
        unit interval.

        For other marks, relative position on a band of a stacked, binned, time unit or band
        scale. If set to ``0``, the marks will be positioned at the beginning of the band.
        If set to ``0.5``, the marks will be positioned in the middle of the band.
    bin : anyOf(boolean, :class:`BinParams`, string, None)
        A flag for binning a ``quantitative`` field, `an object defining binning parameters
        <https://vega.github.io/vega-lite/docs/bin.html#params>`__, or indicating that the
        data for ``x`` or ``y`` channel are binned before they are imported into Vega-Lite (
        ``"binned"`` ).


        If ``true``, default `binning parameters
        <https://vega.github.io/vega-lite/docs/bin.html>`__ will be applied.

        If ``"binned"``, this indicates that the data for the ``x`` (or ``y`` ) channel are
        already binned. You can map the bin-start field to ``x`` (or ``y`` ) and the bin-end
        field to ``x2`` (or ``y2`` ). The scale and axis will be formatted similar to
        binning in Vega-Lite.  To adjust the axis ticks based on the bin step, you can also
        set the axis's `tickMinStep
        <https://vega.github.io/vega-lite/docs/axis.html#ticks>`__ property.

        **Default value:** ``false``

        **See also:** `bin <https://vega.github.io/vega-lite/docs/bin.html>`__
        documentation.
    condition : anyOf(:class:`ConditionalValueDefstringExprRef`,
    List(:class:`ConditionalValueDefstringExprRef`))
        One or more value definition(s) with `a selection or a test predicate
        <https://vega.github.io/vega-lite/docs/condition.html>`__.

        **Note:** A field definition's ``condition`` property can only contain `conditional
        value definitions <https://vega.github.io/vega-lite/docs/condition.html#value>`__
        since Vega-Lite only allows at most one encoded field per encoding channel.
    field : :class:`Field`
        **Required.** A string defining the name of the field from which to pull a data
        value or an object defining iterated values from the `repeat
        <https://vega.github.io/vega-lite/docs/repeat.html>`__ operator.

        **See also:** `field <https://vega.github.io/vega-lite/docs/field.html>`__
        documentation.

        **Notes:** 1)  Dots ( ``.`` ) and brackets ( ``[`` and ``]`` ) can be used to access
        nested objects (e.g., ``"field": "foo.bar"`` and ``"field": "foo['bar']"`` ). If
        field names contain dots or brackets but are not nested, you can use ``\\`` to
        escape dots and brackets (e.g., ``"a\\.b"`` and ``"a\\[0\\]"`` ). See more details
        about escaping in the `field documentation
        <https://vega.github.io/vega-lite/docs/field.html>`__. 2) ``field`` is not required
        if ``aggregate`` is ``count``.
    format : anyOf(string, :class:`Dictunknown`)
        When used with the default ``"number"`` and ``"time"`` format type, the text
        formatting pattern for labels of guides (axes, legends, headers) and text marks.


        * If the format type is ``"number"`` (e.g., for quantitative fields), this is D3's
          `number format pattern <https://github.com/d3/d3-format#locale_format>`__. - If
          the format type is ``"time"`` (e.g., for temporal fields), this is D3's `time
          format pattern <https://github.com/d3/d3-time-format#locale_format>`__.

        See the `format documentation <https://vega.github.io/vega-lite/docs/format.html>`__
        for more examples.

        When used with a `custom formatType
        <https://vega.github.io/vega-lite/docs/config.html#custom-format-type>`__, this
        value will be passed as ``format`` alongside ``datum.value`` to the registered
        function.

        **Default value:**  Derived from `numberFormat
        <https://vega.github.io/vega-lite/docs/config.html#format>`__ config for number
        format and from `timeFormat
        <https://vega.github.io/vega-lite/docs/config.html#format>`__ config for time
        format.
    formatType : string
        The format type for labels. One of ``"number"``, ``"time"``, or a `registered custom
        format type
        <https://vega.github.io/vega-lite/docs/config.html#custom-format-type>`__.

        **Default value:** - ``"time"`` for temporal fields and ordinal and nominal fields
        with ``timeUnit``. - ``"number"`` for quantitative fields as well as ordinal and
        nominal fields without ``timeUnit``.
    labelExpr : string
        `Vega expression <https://vega.github.io/vega/docs/expressions/>`__ for customizing
        labels text.

        **Note:** The label text and value can be assessed via the ``label`` and ``value``
        properties of the axis's backing ``datum`` object.
    timeUnit : anyOf(:class:`TimeUnit`, :class:`TimeUnitParams`)
        Time unit (e.g., ``year``, ``yearmonth``, ``month``, ``hours`` ) for a temporal
        field. or `a temporal field that gets casted as ordinal
        <https://vega.github.io/vega-lite/docs/type.html#cast>`__.

        **Default value:** ``undefined`` (None)

        **See also:** `timeUnit <https://vega.github.io/vega-lite/docs/timeunit.html>`__
        documentation.
    title : anyOf(:class:`Text`, None)
        A title for the field. If ``null``, the title will be removed.

        **Default value:**  derived from the field's name and transformation function (
        ``aggregate``, ``bin`` and ``timeUnit`` ). If the field has an aggregate function,
        the function is displayed as part of the title (e.g., ``"Sum of Profit"`` ). If the
        field is binned or has a time unit applied, the applied function is shown in
        parentheses (e.g., ``"Profit (binned)"``, ``"Transaction Date (year-month)"`` ).
        Otherwise, the title is simply the field name.

        **Notes** :

        1) You can customize the default field title format by providing the `fieldTitle
        <https://vega.github.io/vega-lite/docs/config.html#top-level-config>`__ property in
        the `config <https://vega.github.io/vega-lite/docs/config.html>`__ or `fieldTitle
        function via the compile function's options
        <https://vega.github.io/vega-lite/docs/compile.html#field-title>`__.

        2) If both field definition's ``title`` and axis, header, or legend ``title`` are
        defined, axis/header/legend title will be used.
    type : :class:`StandardType`
        The type of measurement ( ``"quantitative"``, ``"temporal"``, ``"ordinal"``, or
        ``"nominal"`` ) for the encoded field or constant value ( ``datum`` ). It can also
        be a ``"geojson"`` type for encoding `'geoshape'
        <https://vega.github.io/vega-lite/docs/geoshape.html>`__.

        Vega-Lite automatically infers data types in many cases as discussed below. However,
        type is required for a field if: (1) the field is not nominal and the field encoding
        has no specified ``aggregate`` (except ``argmin`` and ``argmax`` ), ``bin``, scale
        type, custom ``sort`` order, nor ``timeUnit`` or (2) if you wish to use an ordinal
        scale for a field with ``bin`` or ``timeUnit``.

        **Default value:**

        1) For a data ``field``, ``"nominal"`` is the default data type unless the field
        encoding has ``aggregate``, ``channel``, ``bin``, scale type, ``sort``, or
        ``timeUnit`` that satisfies the following criteria: - ``"quantitative"`` is the
        default type if (1) the encoded field contains ``bin`` or ``aggregate`` except
        ``"argmin"`` and ``"argmax"``, (2) the encoding channel is ``latitude`` or
        ``longitude`` channel or (3) if the specified scale type is `a quantitative scale
        <https://vega.github.io/vega-lite/docs/scale.html#type>`__. - ``"temporal"`` is the
        default type if (1) the encoded field contains ``timeUnit`` or (2) the specified
        scale type is a time or utc scale - ``ordinal""`` is the default type if (1) the
        encoded field contains a `custom sort order
        <https://vega.github.io/vega-lite/docs/sort.html#specifying-custom-sort-order>`__,
        (2) the specified scale type is an ordinal/point/band scale, or (3) the encoding
        channel is ``order``.

        2) For a constant value in data domain ( ``datum`` ): - ``"quantitative"`` if the
        datum is a number - ``"nominal"`` if the datum is a string - ``"temporal"`` if the
        datum is `a date time object
        <https://vega.github.io/vega-lite/docs/datetime.html>`__

        **Note:** - Data ``type`` describes the semantics of the data rather than the
        primitive data types (number, string, etc.). The same primitive data type can have
        different types of measurement. For example, numeric data can represent
        quantitative, ordinal, or nominal data. - Data values for a temporal field can be
        either a date-time string (e.g., ``"2015-03-07 12:32:17"``, ``"17:01"``,
        ``"2015-03-16"``. ``"2015"`` ) or a timestamp number (e.g., ``1552199579097`` ). -
        When using with `bin <https://vega.github.io/vega-lite/docs/bin.html>`__, the
        ``type`` property can be either ``"quantitative"`` (for using a linear bin scale) or
        `"ordinal" (for using an ordinal bin scale)
        <https://vega.github.io/vega-lite/docs/type.html#cast-bin>`__. - When using with
        `timeUnit <https://vega.github.io/vega-lite/docs/timeunit.html>`__, the ``type``
        property can be either ``"temporal"`` (default, for using a temporal scale) or
        `"ordinal" (for using an ordinal scale)
        <https://vega.github.io/vega-lite/docs/type.html#cast-bin>`__. - When using with
        `aggregate <https://vega.github.io/vega-lite/docs/aggregate.html>`__, the ``type``
        property refers to the post-aggregation data type. For example, we can calculate
        count ``distinct`` of a categorical field ``"cat"`` using ``{"aggregate":
        "distinct", "field": "cat"}``. The ``"type"`` of the aggregate output is
        ``"quantitative"``. - Secondary channels (e.g., ``x2``, ``y2``, ``xError``,
        ``yError`` ) do not have ``type`` as they must have exactly the same type as their
        primary channels (e.g., ``x``, ``y`` ).

        **See also:** `type <https://vega.github.io/vega-lite/docs/type.html>`__
        documentation.
    """
    _schema = {'$ref': '#/definitions/FieldOrDatumDefWithCondition<StringFieldDef,string>'}

    def __init__(self, aggregate=Undefined, band=Undefined, bin=Undefined, condition=Undefined,
                 field=Undefined, format=Undefined, formatType=Undefined, labelExpr=Undefined,
                 timeUnit=Undefined, title=Undefined, type=Undefined, **kwds):
        super(FieldOrDatumDefWithConditionStringFieldDefstring, self).__init__(aggregate=aggregate,
                                                                               band=band, bin=bin,
                                                                               condition=condition,
                                                                               field=field,
                                                                               format=format,
                                                                               formatType=formatType,
                                                                               labelExpr=labelExpr,
                                                                               timeUnit=timeUnit,
                                                                               title=title, type=type,
                                                                               **kwds)


class Fit(VegaLiteSchema):
    """Fit schema wrapper

    anyOf(:class:`GeoJsonFeature`, :class:`GeoJsonFeatureCollection`,
    List(:class:`GeoJsonFeature`))
    """
    _schema = {'$ref': '#/definitions/Fit'}

    def __init__(self, *args, **kwds):
        super(Fit, self).__init__(*args, **kwds)


class FontStyle(VegaLiteSchema):
    """FontStyle schema wrapper

    string
    """
    _schema = {'$ref': '#/definitions/FontStyle'}

    def __init__(self, *args):
        super(FontStyle, self).__init__(*args)


class FontWeight(VegaLiteSchema):
    """FontWeight schema wrapper

    enum('normal', 'bold', 'lighter', 'bolder', 100, 200, 300, 400, 500, 600, 700, 800, 900)
    """
    _schema = {'$ref': '#/definitions/FontWeight'}

    def __init__(self, *args):
        super(FontWeight, self).__init__(*args)


class Generator(Data):
    """Generator schema wrapper

    anyOf(:class:`SequenceGenerator`, :class:`SphereGenerator`, :class:`GraticuleGenerator`)
    """
    _schema = {'$ref': '#/definitions/Generator'}

    def __init__(self, *args, **kwds):
        super(Generator, self).__init__(*args, **kwds)


class GeoJsonFeature(Fit):
    """GeoJsonFeature schema wrapper

    Any
    """
    _schema = {'$ref': '#/definitions/GeoJsonFeature'}

    def __init__(self, *args, **kwds):
        super(GeoJsonFeature, self).__init__(*args, **kwds)


class GeoJsonFeatureCollection(Fit):
    """GeoJsonFeatureCollection schema wrapper

    Any
    """
    _schema = {'$ref': '#/definitions/GeoJsonFeatureCollection'}

    def __init__(self, *args, **kwds):
        super(GeoJsonFeatureCollection, self).__init__(*args, **kwds)


class Gradient(VegaLiteSchema):
    """Gradient schema wrapper

    anyOf(:class:`LinearGradient`, :class:`RadialGradient`)
    """
    _schema = {'$ref': '#/definitions/Gradient'}

    def __init__(self, *args, **kwds):
        super(Gradient, self).__init__(*args, **kwds)


class GradientStop(VegaLiteSchema):
    """GradientStop schema wrapper

    Mapping(required=[offset, color])

    Attributes
    ----------

    color : :class:`Color`
        The color value at this point in the gradient.
    offset : float
        The offset fraction for the color stop, indicating its position within the gradient.
    """
    _schema = {'$ref': '#/definitions/GradientStop'}

    def __init__(self, color=Undefined, offset=Undefined, **kwds):
        super(GradientStop, self).__init__(color=color, offset=offset, **kwds)


class GraticuleGenerator(Generator):
    """GraticuleGenerator schema wrapper

    Mapping(required=[graticule])

    Attributes
    ----------

    graticule : anyOf(boolean, :class:`GraticuleParams`)
        Generate graticule GeoJSON data for geographic reference lines.
    name : string
        Provide a placeholder name and bind data at runtime.
    """
    _schema = {'$ref': '#/definitions/GraticuleGenerator'}

    def __init__(self, graticule=Undefined, name=Undefined, **kwds):
        super(GraticuleGenerator, self).__init__(graticule=graticule, name=name, **kwds)


class GraticuleParams(VegaLiteSchema):
    """GraticuleParams schema wrapper

    Mapping(required=[])

    Attributes
    ----------

    extent : :class:`Vector2Vector2number`
        Sets both the major and minor extents to the same values.
    extentMajor : :class:`Vector2Vector2number`
        The major extent of the graticule as a two-element array of coordinates.
    extentMinor : :class:`Vector2Vector2number`
        The minor extent of the graticule as a two-element array of coordinates.
    precision : float
        The precision of the graticule in degrees.

        **Default value:** ``2.5``
    step : :class:`Vector2number`
        Sets both the major and minor step angles to the same values.
    stepMajor : :class:`Vector2number`
        The major step angles of the graticule.

        **Default value:** ``[90, 360]``
    stepMinor : :class:`Vector2number`
        The minor step angles of the graticule.

        **Default value:** ``[10, 10]``
    """
    _schema = {'$ref': '#/definitions/GraticuleParams'}

    def __init__(self, extent=Undefined, extentMajor=Undefined, extentMinor=Undefined,
                 precision=Undefined, step=Undefined, stepMajor=Undefined, stepMinor=Undefined, **kwds):
        super(GraticuleParams, self).__init__(extent=extent, extentMajor=extentMajor,
                                              extentMinor=extentMinor, precision=precision, step=step,
                                              stepMajor=stepMajor, stepMinor=stepMinor, **kwds)


class Header(VegaLiteSchema):
    """Header schema wrapper

    Mapping(required=[])
    Headers of row / column channels for faceted plots.

    Attributes
    ----------

    format : anyOf(string, :class:`Dictunknown`)
        When used with the default ``"number"`` and ``"time"`` format type, the text
        formatting pattern for labels of guides (axes, legends, headers) and text marks.


        * If the format type is ``"number"`` (e.g., for quantitative fields), this is D3's
          `number format pattern <https://github.com/d3/d3-format#locale_format>`__. - If
          the format type is ``"time"`` (e.g., for temporal fields), this is D3's `time
          format pattern <https://github.com/d3/d3-time-format#locale_format>`__.

        See the `format documentation <https://vega.github.io/vega-lite/docs/format.html>`__
        for more examples.

        When used with a `custom formatType
        <https://vega.github.io/vega-lite/docs/config.html#custom-format-type>`__, this
        value will be passed as ``format`` alongside ``datum.value`` to the registered
        function.

        **Default value:**  Derived from `numberFormat
        <https://vega.github.io/vega-lite/docs/config.html#format>`__ config for number
        format and from `timeFormat
        <https://vega.github.io/vega-lite/docs/config.html#format>`__ config for time
        format.
    formatType : string
        The format type for labels. One of ``"number"``, ``"time"``, or a `registered custom
        format type
        <https://vega.github.io/vega-lite/docs/config.html#custom-format-type>`__.

        **Default value:** - ``"time"`` for temporal fields and ordinal and nominal fields
        with ``timeUnit``. - ``"number"`` for quantitative fields as well as ordinal and
        nominal fields without ``timeUnit``.
    labelAlign : anyOf(:class:`Align`, :class:`ExprRef`)
        Horizontal text alignment of header labels. One of ``"left"``, ``"center"``, or
        ``"right"``.
    labelAnchor : :class:`TitleAnchor`
        The anchor position for placing the labels. One of ``"start"``, ``"middle"``, or
        ``"end"``. For example, with a label orientation of top these anchor positions map
        to a left-, center-, or right-aligned label.
    labelAngle : float
        The rotation angle of the header labels.

        **Default value:** ``0`` for column header, ``-90`` for row header.
    labelBaseline : anyOf(:class:`TextBaseline`, :class:`ExprRef`)
        The vertical text baseline for the header labels. One of ``"alphabetic"`` (default),
        ``"top"``, ``"middle"``, ``"bottom"``, ``"line-top"``, or ``"line-bottom"``. The
        ``"line-top"`` and ``"line-bottom"`` values operate similarly to ``"top"`` and
        ``"bottom"``, but are calculated relative to the ``titleLineHeight`` rather than
        ``titleFontSize`` alone.
    labelColor : anyOf(:class:`Color`, :class:`ExprRef`)
        The color of the header label, can be in hex color code or regular color name.
    labelExpr : string
        `Vega expression <https://vega.github.io/vega/docs/expressions/>`__ for customizing
        labels.

        **Note:** The label text and value can be assessed via the ``label`` and ``value``
        properties of the header's backing ``datum`` object.
    labelFont : anyOf(string, :class:`ExprRef`)
        The font of the header label.
    labelFontSize : anyOf(float, :class:`ExprRef`)
        The font size of the header label, in pixels.
    labelFontStyle : anyOf(:class:`FontStyle`, :class:`ExprRef`)
        The font style of the header label.
    labelFontWeight : anyOf(:class:`FontWeight`, :class:`ExprRef`)
        The font weight of the header label.
    labelLimit : anyOf(float, :class:`ExprRef`)
        The maximum length of the header label in pixels. The text value will be
        automatically truncated if the rendered size exceeds the limit.

        **Default value:** ``0``, indicating no limit
    labelLineHeight : anyOf(float, :class:`ExprRef`)
        Line height in pixels for multi-line header labels or title text with ``"line-top"``
        or ``"line-bottom"`` baseline.
    labelOrient : :class:`Orient`
        The orientation of the header label. One of ``"top"``, ``"bottom"``, ``"left"`` or
        ``"right"``.
    labelPadding : anyOf(float, :class:`ExprRef`)
        The padding, in pixel, between facet header's label and the plot.

        **Default value:** ``10``
    labels : boolean
        A boolean flag indicating if labels should be included as part of the header.

        **Default value:** ``true``.
    orient : :class:`Orient`
        Shortcut for setting both labelOrient and titleOrient.
    title : anyOf(:class:`Text`, None)
        A title for the field. If ``null``, the title will be removed.

        **Default value:**  derived from the field's name and transformation function (
        ``aggregate``, ``bin`` and ``timeUnit`` ). If the field has an aggregate function,
        the function is displayed as part of the title (e.g., ``"Sum of Profit"`` ). If the
        field is binned or has a time unit applied, the applied function is shown in
        parentheses (e.g., ``"Profit (binned)"``, ``"Transaction Date (year-month)"`` ).
        Otherwise, the title is simply the field name.

        **Notes** :

        1) You can customize the default field title format by providing the `fieldTitle
        <https://vega.github.io/vega-lite/docs/config.html#top-level-config>`__ property in
        the `config <https://vega.github.io/vega-lite/docs/config.html>`__ or `fieldTitle
        function via the compile function's options
        <https://vega.github.io/vega-lite/docs/compile.html#field-title>`__.

        2) If both field definition's ``title`` and axis, header, or legend ``title`` are
        defined, axis/header/legend title will be used.
    titleAlign : anyOf(:class:`Align`, :class:`ExprRef`)
        Horizontal text alignment (to the anchor) of header titles.
    titleAnchor : :class:`TitleAnchor`
        The anchor position for placing the title. One of ``"start"``, ``"middle"``, or
        ``"end"``. For example, with an orientation of top these anchor positions map to a
        left-, center-, or right-aligned title.
    titleAngle : float
        The rotation angle of the header title.

        **Default value:** ``0``.
    titleBaseline : anyOf(:class:`TextBaseline`, :class:`ExprRef`)
        The vertical text baseline for the header title. One of ``"alphabetic"`` (default),
        ``"top"``, ``"middle"``, ``"bottom"``, ``"line-top"``, or ``"line-bottom"``. The
        ``"line-top"`` and ``"line-bottom"`` values operate similarly to ``"top"`` and
        ``"bottom"``, but are calculated relative to the ``titleLineHeight`` rather than
        ``titleFontSize`` alone.

        **Default value:** ``"middle"``
    titleColor : anyOf(:class:`Color`, :class:`ExprRef`)
        Color of the header title, can be in hex color code or regular color name.
    titleFont : anyOf(string, :class:`ExprRef`)
        Font of the header title. (e.g., ``"Helvetica Neue"`` ).
    titleFontSize : anyOf(float, :class:`ExprRef`)
        Font size of the header title.
    titleFontStyle : anyOf(:class:`FontStyle`, :class:`ExprRef`)
        The font style of the header title.
    titleFontWeight : anyOf(:class:`FontWeight`, :class:`ExprRef`)
        Font weight of the header title. This can be either a string (e.g ``"bold"``,
        ``"normal"`` ) or a number ( ``100``, ``200``, ``300``, ..., ``900`` where
        ``"normal"`` = ``400`` and ``"bold"`` = ``700`` ).
    titleLimit : anyOf(float, :class:`ExprRef`)
        The maximum length of the header title in pixels. The text value will be
        automatically truncated if the rendered size exceeds the limit.

        **Default value:** ``0``, indicating no limit
    titleLineHeight : anyOf(float, :class:`ExprRef`)
        Line height in pixels for multi-line header title text or title text with
        ``"line-top"`` or ``"line-bottom"`` baseline.
    titleOrient : :class:`Orient`
        The orientation of the header title. One of ``"top"``, ``"bottom"``, ``"left"`` or
        ``"right"``.
    titlePadding : anyOf(float, :class:`ExprRef`)
        The padding, in pixel, between facet header's title and the label.

        **Default value:** ``10``
    """
    _schema = {'$ref': '#/definitions/Header'}

    def __init__(self, format=Undefined, formatType=Undefined, labelAlign=Undefined,
                 labelAnchor=Undefined, labelAngle=Undefined, labelBaseline=Undefined,
                 labelColor=Undefined, labelExpr=Undefined, labelFont=Undefined,
                 labelFontSize=Undefined, labelFontStyle=Undefined, labelFontWeight=Undefined,
                 labelLimit=Undefined, labelLineHeight=Undefined, labelOrient=Undefined,
                 labelPadding=Undefined, labels=Undefined, orient=Undefined, title=Undefined,
                 titleAlign=Undefined, titleAnchor=Undefined, titleAngle=Undefined,
                 titleBaseline=Undefined, titleColor=Undefined, titleFont=Undefined,
                 titleFontSize=Undefined, titleFontStyle=Undefined, titleFontWeight=Undefined,
                 titleLimit=Undefined, titleLineHeight=Undefined, titleOrient=Undefined,
                 titlePadding=Undefined, **kwds):
        super(Header, self).__init__(format=format, formatType=formatType, labelAlign=labelAlign,
                                     labelAnchor=labelAnchor, labelAngle=labelAngle,
                                     labelBaseline=labelBaseline, labelColor=labelColor,
                                     labelExpr=labelExpr, labelFont=labelFont,
                                     labelFontSize=labelFontSize, labelFontStyle=labelFontStyle,
                                     labelFontWeight=labelFontWeight, labelLimit=labelLimit,
                                     labelLineHeight=labelLineHeight, labelOrient=labelOrient,
                                     labelPadding=labelPadding, labels=labels, orient=orient,
                                     title=title, titleAlign=titleAlign, titleAnchor=titleAnchor,
                                     titleAngle=titleAngle, titleBaseline=titleBaseline,
                                     titleColor=titleColor, titleFont=titleFont,
                                     titleFontSize=titleFontSize, titleFontStyle=titleFontStyle,
                                     titleFontWeight=titleFontWeight, titleLimit=titleLimit,
                                     titleLineHeight=titleLineHeight, titleOrient=titleOrient,
                                     titlePadding=titlePadding, **kwds)


class HeaderConfig(VegaLiteSchema):
    """HeaderConfig schema wrapper

    Mapping(required=[])

    Attributes
    ----------

    format : anyOf(string, :class:`Dictunknown`)
        When used with the default ``"number"`` and ``"time"`` format type, the text
        formatting pattern for labels of guides (axes, legends, headers) and text marks.


        * If the format type is ``"number"`` (e.g., for quantitative fields), this is D3's
          `number format pattern <https://github.com/d3/d3-format#locale_format>`__. - If
          the format type is ``"time"`` (e.g., for temporal fields), this is D3's `time
          format pattern <https://github.com/d3/d3-time-format#locale_format>`__.

        See the `format documentation <https://vega.github.io/vega-lite/docs/format.html>`__
        for more examples.

        When used with a `custom formatType
        <https://vega.github.io/vega-lite/docs/config.html#custom-format-type>`__, this
        value will be passed as ``format`` alongside ``datum.value`` to the registered
        function.

        **Default value:**  Derived from `numberFormat
        <https://vega.github.io/vega-lite/docs/config.html#format>`__ config for number
        format and from `timeFormat
        <https://vega.github.io/vega-lite/docs/config.html#format>`__ config for time
        format.
    formatType : string
        The format type for labels. One of ``"number"``, ``"time"``, or a `registered custom
        format type
        <https://vega.github.io/vega-lite/docs/config.html#custom-format-type>`__.

        **Default value:** - ``"time"`` for temporal fields and ordinal and nominal fields
        with ``timeUnit``. - ``"number"`` for quantitative fields as well as ordinal and
        nominal fields without ``timeUnit``.
    labelAlign : anyOf(:class:`Align`, :class:`ExprRef`)
        Horizontal text alignment of header labels. One of ``"left"``, ``"center"``, or
        ``"right"``.
    labelAnchor : :class:`TitleAnchor`
        The anchor position for placing the labels. One of ``"start"``, ``"middle"``, or
        ``"end"``. For example, with a label orientation of top these anchor positions map
        to a left-, center-, or right-aligned label.
    labelAngle : float
        The rotation angle of the header labels.

        **Default value:** ``0`` for column header, ``-90`` for row header.
    labelBaseline : anyOf(:class:`TextBaseline`, :class:`ExprRef`)
        The vertical text baseline for the header labels. One of ``"alphabetic"`` (default),
        ``"top"``, ``"middle"``, ``"bottom"``, ``"line-top"``, or ``"line-bottom"``. The
        ``"line-top"`` and ``"line-bottom"`` values operate similarly to ``"top"`` and
        ``"bottom"``, but are calculated relative to the ``titleLineHeight`` rather than
        ``titleFontSize`` alone.
    labelColor : anyOf(:class:`Color`, :class:`ExprRef`)
        The color of the header label, can be in hex color code or regular color name.
    labelExpr : string
        `Vega expression <https://vega.github.io/vega/docs/expressions/>`__ for customizing
        labels.

        **Note:** The label text and value can be assessed via the ``label`` and ``value``
        properties of the header's backing ``datum`` object.
    labelFont : anyOf(string, :class:`ExprRef`)
        The font of the header label.
    labelFontSize : anyOf(float, :class:`ExprRef`)
        The font size of the header label, in pixels.
    labelFontStyle : anyOf(:class:`FontStyle`, :class:`ExprRef`)
        The font style of the header label.
    labelFontWeight : anyOf(:class:`FontWeight`, :class:`ExprRef`)
        The font weight of the header label.
    labelLimit : anyOf(float, :class:`ExprRef`)
        The maximum length of the header label in pixels. The text value will be
        automatically truncated if the rendered size exceeds the limit.

        **Default value:** ``0``, indicating no limit
    labelLineHeight : anyOf(float, :class:`ExprRef`)
        Line height in pixels for multi-line header labels or title text with ``"line-top"``
        or ``"line-bottom"`` baseline.
    labelOrient : :class:`Orient`
        The orientation of the header label. One of ``"top"``, ``"bottom"``, ``"left"`` or
        ``"right"``.
    labelPadding : anyOf(float, :class:`ExprRef`)
        The padding, in pixel, between facet header's label and the plot.

        **Default value:** ``10``
    labels : boolean
        A boolean flag indicating if labels should be included as part of the header.

        **Default value:** ``true``.
    orient : :class:`Orient`
        Shortcut for setting both labelOrient and titleOrient.
    title : None
        Set to null to disable title for the axis, legend, or header.
    titleAlign : anyOf(:class:`Align`, :class:`ExprRef`)
        Horizontal text alignment (to the anchor) of header titles.
    titleAnchor : :class:`TitleAnchor`
        The anchor position for placing the title. One of ``"start"``, ``"middle"``, or
        ``"end"``. For example, with an orientation of top these anchor positions map to a
        left-, center-, or right-aligned title.
    titleAngle : float
        The rotation angle of the header title.

        **Default value:** ``0``.
    titleBaseline : anyOf(:class:`TextBaseline`, :class:`ExprRef`)
        The vertical text baseline for the header title. One of ``"alphabetic"`` (default),
        ``"top"``, ``"middle"``, ``"bottom"``, ``"line-top"``, or ``"line-bottom"``. The
        ``"line-top"`` and ``"line-bottom"`` values operate similarly to ``"top"`` and
        ``"bottom"``, but are calculated relative to the ``titleLineHeight`` rather than
        ``titleFontSize`` alone.

        **Default value:** ``"middle"``
    titleColor : anyOf(:class:`Color`, :class:`ExprRef`)
        Color of the header title, can be in hex color code or regular color name.
    titleFont : anyOf(string, :class:`ExprRef`)
        Font of the header title. (e.g., ``"Helvetica Neue"`` ).
    titleFontSize : anyOf(float, :class:`ExprRef`)
        Font size of the header title.
    titleFontStyle : anyOf(:class:`FontStyle`, :class:`ExprRef`)
        The font style of the header title.
    titleFontWeight : anyOf(:class:`FontWeight`, :class:`ExprRef`)
        Font weight of the header title. This can be either a string (e.g ``"bold"``,
        ``"normal"`` ) or a number ( ``100``, ``200``, ``300``, ..., ``900`` where
        ``"normal"`` = ``400`` and ``"bold"`` = ``700`` ).
    titleLimit : anyOf(float, :class:`ExprRef`)
        The maximum length of the header title in pixels. The text value will be
        automatically truncated if the rendered size exceeds the limit.

        **Default value:** ``0``, indicating no limit
    titleLineHeight : anyOf(float, :class:`ExprRef`)
        Line height in pixels for multi-line header title text or title text with
        ``"line-top"`` or ``"line-bottom"`` baseline.
    titleOrient : :class:`Orient`
        The orientation of the header title. One of ``"top"``, ``"bottom"``, ``"left"`` or
        ``"right"``.
    titlePadding : anyOf(float, :class:`ExprRef`)
        The padding, in pixel, between facet header's title and the label.

        **Default value:** ``10``
    """
    _schema = {'$ref': '#/definitions/HeaderConfig'}

    def __init__(self, format=Undefined, formatType=Undefined, labelAlign=Undefined,
                 labelAnchor=Undefined, labelAngle=Undefined, labelBaseline=Undefined,
                 labelColor=Undefined, labelExpr=Undefined, labelFont=Undefined,
                 labelFontSize=Undefined, labelFontStyle=Undefined, labelFontWeight=Undefined,
                 labelLimit=Undefined, labelLineHeight=Undefined, labelOrient=Undefined,
                 labelPadding=Undefined, labels=Undefined, orient=Undefined, title=Undefined,
                 titleAlign=Undefined, titleAnchor=Undefined, titleAngle=Undefined,
                 titleBaseline=Undefined, titleColor=Undefined, titleFont=Undefined,
                 titleFontSize=Undefined, titleFontStyle=Undefined, titleFontWeight=Undefined,
                 titleLimit=Undefined, titleLineHeight=Undefined, titleOrient=Undefined,
                 titlePadding=Undefined, **kwds):
        super(HeaderConfig, self).__init__(format=format, formatType=formatType, labelAlign=labelAlign,
                                           labelAnchor=labelAnchor, labelAngle=labelAngle,
                                           labelBaseline=labelBaseline, labelColor=labelColor,
                                           labelExpr=labelExpr, labelFont=labelFont,
                                           labelFontSize=labelFontSize, labelFontStyle=labelFontStyle,
                                           labelFontWeight=labelFontWeight, labelLimit=labelLimit,
                                           labelLineHeight=labelLineHeight, labelOrient=labelOrient,
                                           labelPadding=labelPadding, labels=labels, orient=orient,
                                           title=title, titleAlign=titleAlign, titleAnchor=titleAnchor,
                                           titleAngle=titleAngle, titleBaseline=titleBaseline,
                                           titleColor=titleColor, titleFont=titleFont,
                                           titleFontSize=titleFontSize, titleFontStyle=titleFontStyle,
                                           titleFontWeight=titleFontWeight, titleLimit=titleLimit,
                                           titleLineHeight=titleLineHeight, titleOrient=titleOrient,
                                           titlePadding=titlePadding, **kwds)


class HexColor(Color):
    """HexColor schema wrapper

    string
    """
    _schema = {'$ref': '#/definitions/HexColor'}

    def __init__(self, *args):
        super(HexColor, self).__init__(*args)


class ImputeMethod(VegaLiteSchema):
    """ImputeMethod schema wrapper

    enum('value', 'median', 'max', 'min', 'mean')
    """
    _schema = {'$ref': '#/definitions/ImputeMethod'}

    def __init__(self, *args):
        super(ImputeMethod, self).__init__(*args)


class ImputeParams(VegaLiteSchema):
    """ImputeParams schema wrapper

    Mapping(required=[])

    Attributes
    ----------

    frame : List([anyOf(None, float), anyOf(None, float)])
        A frame specification as a two-element array used to control the window over which
        the specified method is applied. The array entries should either be a number
        indicating the offset from the current data object, or null to indicate unbounded
        rows preceding or following the current data object. For example, the value ``[-5,
        5]`` indicates that the window should include five objects preceding and five
        objects following the current object.

        **Default value:** :  ``[null, null]`` indicating that the window includes all
        objects.
    keyvals : anyOf(List(Any), :class:`ImputeSequence`)
        Defines the key values that should be considered for imputation. An array of key
        values or an object defining a `number sequence
        <https://vega.github.io/vega-lite/docs/impute.html#sequence-def>`__.

        If provided, this will be used in addition to the key values observed within the
        input data. If not provided, the values will be derived from all unique values of
        the ``key`` field. For ``impute`` in ``encoding``, the key field is the x-field if
        the y-field is imputed, or vice versa.

        If there is no impute grouping, this property *must* be specified.
    method : :class:`ImputeMethod`
        The imputation method to use for the field value of imputed data objects. One of
        ``"value"``, ``"mean"``, ``"median"``, ``"max"`` or ``"min"``.

        **Default value:**  ``"value"``
    value : Any
        The field value to use when the imputation ``method`` is ``"value"``.
    """
    _schema = {'$ref': '#/definitions/ImputeParams'}

    def __init__(self, frame=Undefined, keyvals=Undefined, method=Undefined, value=Undefined, **kwds):
        super(ImputeParams, self).__init__(frame=frame, keyvals=keyvals, method=method, value=value,
                                           **kwds)


class ImputeSequence(VegaLiteSchema):
    """ImputeSequence schema wrapper

    Mapping(required=[stop])

    Attributes
    ----------

    stop : float
        The ending value(exclusive) of the sequence.
    start : float
        The starting value of the sequence. **Default value:** ``0``
    step : float
        The step value between sequence entries. **Default value:** ``1`` or ``-1`` if
        ``stop < start``
    """
    _schema = {'$ref': '#/definitions/ImputeSequence'}

    def __init__(self, stop=Undefined, start=Undefined, step=Undefined, **kwds):
        super(ImputeSequence, self).__init__(stop=stop, start=start, step=step, **kwds)


class InlineData(DataSource):
    """InlineData schema wrapper

    Mapping(required=[values])

    Attributes
    ----------

    values : :class:`InlineDataset`
        The full data set, included inline. This can be an array of objects or primitive
        values, an object, or a string. Arrays of primitive values are ingested as objects
        with a ``data`` property. Strings are parsed according to the specified format type.
    format : :class:`DataFormat`
        An object that specifies the format for parsing the data.
    name : string
        Provide a placeholder name and bind data at runtime.
    """
    _schema = {'$ref': '#/definitions/InlineData'}

    def __init__(self, values=Undefined, format=Undefined, name=Undefined, **kwds):
        super(InlineData, self).__init__(values=values, format=format, name=name, **kwds)


class InlineDataset(VegaLiteSchema):
    """InlineDataset schema wrapper

    anyOf(List(float), List(string), List(boolean), List(Mapping(required=[])), string,
    Mapping(required=[]))
    """
    _schema = {'$ref': '#/definitions/InlineDataset'}

    def __init__(self, *args, **kwds):
        super(InlineDataset, self).__init__(*args, **kwds)


class InputBinding(Binding):
    """InputBinding schema wrapper

    Mapping(required=[])

    Attributes
    ----------

    autocomplete : string

    debounce : float

    element : :class:`Element`

    input : string

    name : string

    placeholder : string

    type : string

    """
    _schema = {'$ref': '#/definitions/InputBinding'}

    def __init__(self, autocomplete=Undefined, debounce=Undefined, element=Undefined, input=Undefined,
                 name=Undefined, placeholder=Undefined, type=Undefined, **kwds):
        super(InputBinding, self).__init__(autocomplete=autocomplete, debounce=debounce,
                                           element=element, input=input, name=name,
                                           placeholder=placeholder, type=type, **kwds)


class Interpolate(VegaLiteSchema):
    """Interpolate schema wrapper

    enum('basis', 'basis-open', 'basis-closed', 'bundle', 'cardinal', 'cardinal-open',
    'cardinal-closed', 'catmull-rom', 'linear', 'linear-closed', 'monotone', 'natural', 'step',
    'step-before', 'step-after')
    """
    _schema = {'$ref': '#/definitions/Interpolate'}

    def __init__(self, *args):
        super(Interpolate, self).__init__(*args)


class IntervalSelectionConfig(VegaLiteSchema):
    """IntervalSelectionConfig schema wrapper

    Mapping(required=[])

    Attributes
    ----------

    bind : string
        Establishes a two-way binding between the interval selection and the scales used
        within the same view. This allows a user to interactively pan and zoom the view.

        **See also:** `bind <https://vega.github.io/vega-lite/docs/bind.html>`__
        documentation.
    clear : anyOf(:class:`Stream`, string, boolean)
        Clears the selection, emptying it of all values. Can be a `Event Stream
        <https://vega.github.io/vega/docs/event-streams/>`__ or ``false`` to disable.

        **Default value:** ``dblclick``.

        **See also:** `clear <https://vega.github.io/vega-lite/docs/clear.html>`__
        documentation.
    empty : enum('all', 'none')
        By default, ``all`` data values are considered to lie within an empty selection.
        When set to ``none``, empty selections contain no data values.
    encodings : List(:class:`SingleDefUnitChannel`)
        An array of encoding channels. The corresponding data field values must match for a
        data tuple to fall within the selection.

        **See also:** `encodings <https://vega.github.io/vega-lite/docs/project.html>`__
        documentation.
    fields : List(:class:`FieldName`)
        An array of field names whose values must match for a data tuple to fall within the
        selection.

        **See also:** `fields <https://vega.github.io/vega-lite/docs/project.html>`__
        documentation.
    init : :class:`SelectionInitIntervalMapping`
        Initialize the selection with a mapping between `projected channels or field names
        <https://vega.github.io/vega-lite/docs/project.html>`__ and arrays of initial
        values.

        **See also:** `init <https://vega.github.io/vega-lite/docs/init.html>`__
        documentation.
    mark : :class:`BrushConfig`
        An interval selection also adds a rectangle mark to depict the extents of the
        interval. The ``mark`` property can be used to customize the appearance of the mark.

        **See also:** `mark <https://vega.github.io/vega-lite/docs/selection-mark.html>`__
        documentation.
    on : anyOf(:class:`Stream`, string)
        A `Vega event stream <https://vega.github.io/vega/docs/event-streams/>`__ (object or
        selector) that triggers the selection. For interval selections, the event stream
        must specify a `start and end
        <https://vega.github.io/vega/docs/event-streams/#between-filters>`__.
    resolve : :class:`SelectionResolution`
        With layered and multi-view displays, a strategy that determines how selections'
        data queries are resolved when applied in a filter transform, conditional encoding
        rule, or scale domain.

        **See also:** `resolve
        <https://vega.github.io/vega-lite/docs/selection-resolve.html>`__ documentation.
    translate : anyOf(string, boolean)
        When truthy, allows a user to interactively move an interval selection
        back-and-forth. Can be ``true``, ``false`` (to disable panning), or a `Vega event
        stream definition <https://vega.github.io/vega/docs/event-streams/>`__ which must
        include a start and end event to trigger continuous panning.

        **Default value:** ``true``, which corresponds to ``[mousedown, window:mouseup] >
        window:mousemove!`` which corresponds to clicks and dragging within an interval
        selection to reposition it.

        **See also:** `translate <https://vega.github.io/vega-lite/docs/translate.html>`__
        documentation.
    zoom : anyOf(string, boolean)
        When truthy, allows a user to interactively resize an interval selection. Can be
        ``true``, ``false`` (to disable zooming), or a `Vega event stream definition
        <https://vega.github.io/vega/docs/event-streams/>`__. Currently, only ``wheel``
        events are supported.

        **Default value:** ``true``, which corresponds to ``wheel!``.

        **See also:** `zoom <https://vega.github.io/vega-lite/docs/zoom.html>`__
        documentation.
    """
    _schema = {'$ref': '#/definitions/IntervalSelectionConfig'}

    def __init__(self, bind=Undefined, clear=Undefined, empty=Undefined, encodings=Undefined,
                 fields=Undefined, init=Undefined, mark=Undefined, on=Undefined, resolve=Undefined,
                 translate=Undefined, zoom=Undefined, **kwds):
        super(IntervalSelectionConfig, self).__init__(bind=bind, clear=clear, empty=empty,
                                                      encodings=encodings, fields=fields, init=init,
                                                      mark=mark, on=on, resolve=resolve,
                                                      translate=translate, zoom=zoom, **kwds)


class JoinAggregateFieldDef(VegaLiteSchema):
    """JoinAggregateFieldDef schema wrapper

    Mapping(required=[op, as])

    Attributes
    ----------

    op : :class:`AggregateOp`
        The aggregation operation to apply (e.g., ``"sum"``, ``"average"`` or ``"count"`` ).
        See the list of all supported operations `here
        <https://vega.github.io/vega-lite/docs/aggregate.html#ops>`__.
    field : :class:`FieldName`
        The data field for which to compute the aggregate function. This can be omitted for
        functions that do not operate over a field such as ``"count"``.
    as : :class:`FieldName`
        The output name for the join aggregate operation.
    """
    _schema = {'$ref': '#/definitions/JoinAggregateFieldDef'}

    def __init__(self, op=Undefined, field=Undefined, **kwds):
        super(JoinAggregateFieldDef, self).__init__(op=op, field=field, **kwds)


class JsonDataFormat(DataFormat):
    """JsonDataFormat schema wrapper

    Mapping(required=[])

    Attributes
    ----------

    parse : anyOf(:class:`Parse`, None)
        If set to ``null``, disable type inference based on the spec and only use type
        inference based on the data. Alternatively, a parsing directive object can be
        provided for explicit data types. Each property of the object corresponds to a field
        name, and the value to the desired data type (one of ``"number"``, ``"boolean"``,
        ``"date"``, or null (do not parse the field)). For example, ``"parse":
        {"modified_on": "date"}`` parses the ``modified_on`` field in each input record a
        Date value.

        For ``"date"``, we parse data based using Javascript's `Date.parse()
        <https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Date/parse>`__.
        For Specific date formats can be provided (e.g., ``{foo: "date:'%m%d%Y'"}`` ), using
        the `d3-time-format syntax <https://github.com/d3/d3-time-format#locale_format>`__.
        UTC date format parsing is supported similarly (e.g., ``{foo: "utc:'%m%d%Y'"}`` ).
        See more about `UTC time
        <https://vega.github.io/vega-lite/docs/timeunit.html#utc>`__
    property : string
        The JSON property containing the desired data. This parameter can be used when the
        loaded JSON file may have surrounding structure or meta-data. For example
        ``"property": "values.features"`` is equivalent to retrieving
        ``json.values.features`` from the loaded JSON object.
    type : string
        Type of input data: ``"json"``, ``"csv"``, ``"tsv"``, ``"dsv"``.

        **Default value:**  The default format type is determined by the extension of the
        file URL. If no extension is detected, ``"json"`` will be used by default.
    """
    _schema = {'$ref': '#/definitions/JsonDataFormat'}

    def __init__(self, parse=Undefined, property=Undefined, type=Undefined, **kwds):
        super(JsonDataFormat, self).__init__(parse=parse, property=property, type=type, **kwds)


class LabelOverlap(VegaLiteSchema):
    """LabelOverlap schema wrapper

    anyOf(boolean, string, string)
    """
    _schema = {'$ref': '#/definitions/LabelOverlap'}

    def __init__(self, *args, **kwds):
        super(LabelOverlap, self).__init__(*args, **kwds)


class LatLongDef(VegaLiteSchema):
    """LatLongDef schema wrapper

    anyOf(:class:`LatLongFieldDef`, :class:`DatumDef`, :class:`NumericValueDef`)
    """
    _schema = {'$ref': '#/definitions/LatLongDef'}

    def __init__(self, *args, **kwds):
        super(LatLongDef, self).__init__(*args, **kwds)


class LatLongFieldDef(LatLongDef):
    """LatLongFieldDef schema wrapper

    Mapping(required=[])

    Attributes
    ----------

    aggregate : :class:`Aggregate`
        Aggregation function for the field (e.g., ``"mean"``, ``"sum"``, ``"median"``,
        ``"min"``, ``"max"``, ``"count"`` ).

        **Default value:** ``undefined`` (None)

        **See also:** `aggregate <https://vega.github.io/vega-lite/docs/aggregate.html>`__
        documentation.
    band : float
        For rect-based marks ( ``rect``, ``bar``, and ``image`` ), mark size relative to
        bandwidth of `band scales
        <https://vega.github.io/vega-lite/docs/scale.html#band>`__, bins or time units. If
        set to ``1``, the mark size is set to the bandwidth, the bin interval, or the time
        unit interval. If set to ``0.5``, the mark size is half of the bandwidth or the time
        unit interval.

        For other marks, relative position on a band of a stacked, binned, time unit or band
        scale. If set to ``0``, the marks will be positioned at the beginning of the band.
        If set to ``0.5``, the marks will be positioned in the middle of the band.
    bin : None
        A flag for binning a ``quantitative`` field, `an object defining binning parameters
        <https://vega.github.io/vega-lite/docs/bin.html#params>`__, or indicating that the
        data for ``x`` or ``y`` channel are binned before they are imported into Vega-Lite (
        ``"binned"`` ).


        If ``true``, default `binning parameters
        <https://vega.github.io/vega-lite/docs/bin.html>`__ will be applied.

        If ``"binned"``, this indicates that the data for the ``x`` (or ``y`` ) channel are
        already binned. You can map the bin-start field to ``x`` (or ``y`` ) and the bin-end
        field to ``x2`` (or ``y2`` ). The scale and axis will be formatted similar to
        binning in Vega-Lite.  To adjust the axis ticks based on the bin step, you can also
        set the axis's `tickMinStep
        <https://vega.github.io/vega-lite/docs/axis.html#ticks>`__ property.

        **Default value:** ``false``

        **See also:** `bin <https://vega.github.io/vega-lite/docs/bin.html>`__
        documentation.
    field : :class:`Field`
        **Required.** A string defining the name of the field from which to pull a data
        value or an object defining iterated values from the `repeat
        <https://vega.github.io/vega-lite/docs/repeat.html>`__ operator.

        **See also:** `field <https://vega.github.io/vega-lite/docs/field.html>`__
        documentation.

        **Notes:** 1)  Dots ( ``.`` ) and brackets ( ``[`` and ``]`` ) can be used to access
        nested objects (e.g., ``"field": "foo.bar"`` and ``"field": "foo['bar']"`` ). If
        field names contain dots or brackets but are not nested, you can use ``\\`` to
        escape dots and brackets (e.g., ``"a\\.b"`` and ``"a\\[0\\]"`` ). See more details
        about escaping in the `field documentation
        <https://vega.github.io/vega-lite/docs/field.html>`__. 2) ``field`` is not required
        if ``aggregate`` is ``count``.
    timeUnit : anyOf(:class:`TimeUnit`, :class:`TimeUnitParams`)
        Time unit (e.g., ``year``, ``yearmonth``, ``month``, ``hours`` ) for a temporal
        field. or `a temporal field that gets casted as ordinal
        <https://vega.github.io/vega-lite/docs/type.html#cast>`__.

        **Default value:** ``undefined`` (None)

        **See also:** `timeUnit <https://vega.github.io/vega-lite/docs/timeunit.html>`__
        documentation.
    title : anyOf(:class:`Text`, None)
        A title for the field. If ``null``, the title will be removed.

        **Default value:**  derived from the field's name and transformation function (
        ``aggregate``, ``bin`` and ``timeUnit`` ). If the field has an aggregate function,
        the function is displayed as part of the title (e.g., ``"Sum of Profit"`` ). If the
        field is binned or has a time unit applied, the applied function is shown in
        parentheses (e.g., ``"Profit (binned)"``, ``"Transaction Date (year-month)"`` ).
        Otherwise, the title is simply the field name.

        **Notes** :

        1) You can customize the default field title format by providing the `fieldTitle
        <https://vega.github.io/vega-lite/docs/config.html#top-level-config>`__ property in
        the `config <https://vega.github.io/vega-lite/docs/config.html>`__ or `fieldTitle
        function via the compile function's options
        <https://vega.github.io/vega-lite/docs/compile.html#field-title>`__.

        2) If both field definition's ``title`` and axis, header, or legend ``title`` are
        defined, axis/header/legend title will be used.
    type : string
        The type of measurement ( ``"quantitative"``, ``"temporal"``, ``"ordinal"``, or
        ``"nominal"`` ) for the encoded field or constant value ( ``datum`` ). It can also
        be a ``"geojson"`` type for encoding `'geoshape'
        <https://vega.github.io/vega-lite/docs/geoshape.html>`__.

        Vega-Lite automatically infers data types in many cases as discussed below. However,
        type is required for a field if: (1) the field is not nominal and the field encoding
        has no specified ``aggregate`` (except ``argmin`` and ``argmax`` ), ``bin``, scale
        type, custom ``sort`` order, nor ``timeUnit`` or (2) if you wish to use an ordinal
        scale for a field with ``bin`` or ``timeUnit``.

        **Default value:**

        1) For a data ``field``, ``"nominal"`` is the default data type unless the field
        encoding has ``aggregate``, ``channel``, ``bin``, scale type, ``sort``, or
        ``timeUnit`` that satisfies the following criteria: - ``"quantitative"`` is the
        default type if (1) the encoded field contains ``bin`` or ``aggregate`` except
        ``"argmin"`` and ``"argmax"``, (2) the encoding channel is ``latitude`` or
        ``longitude`` channel or (3) if the specified scale type is `a quantitative scale
        <https://vega.github.io/vega-lite/docs/scale.html#type>`__. - ``"temporal"`` is the
        default type if (1) the encoded field contains ``timeUnit`` or (2) the specified
        scale type is a time or utc scale - ``ordinal""`` is the default type if (1) the
        encoded field contains a `custom sort order
        <https://vega.github.io/vega-lite/docs/sort.html#specifying-custom-sort-order>`__,
        (2) the specified scale type is an ordinal/point/band scale, or (3) the encoding
        channel is ``order``.

        2) For a constant value in data domain ( ``datum`` ): - ``"quantitative"`` if the
        datum is a number - ``"nominal"`` if the datum is a string - ``"temporal"`` if the
        datum is `a date time object
        <https://vega.github.io/vega-lite/docs/datetime.html>`__

        **Note:** - Data ``type`` describes the semantics of the data rather than the
        primitive data types (number, string, etc.). The same primitive data type can have
        different types of measurement. For example, numeric data can represent
        quantitative, ordinal, or nominal data. - Data values for a temporal field can be
        either a date-time string (e.g., ``"2015-03-07 12:32:17"``, ``"17:01"``,
        ``"2015-03-16"``. ``"2015"`` ) or a timestamp number (e.g., ``1552199579097`` ). -
        When using with `bin <https://vega.github.io/vega-lite/docs/bin.html>`__, the
        ``type`` property can be either ``"quantitative"`` (for using a linear bin scale) or
        `"ordinal" (for using an ordinal bin scale)
        <https://vega.github.io/vega-lite/docs/type.html#cast-bin>`__. - When using with
        `timeUnit <https://vega.github.io/vega-lite/docs/timeunit.html>`__, the ``type``
        property can be either ``"temporal"`` (default, for using a temporal scale) or
        `"ordinal" (for using an ordinal scale)
        <https://vega.github.io/vega-lite/docs/type.html#cast-bin>`__. - When using with
        `aggregate <https://vega.github.io/vega-lite/docs/aggregate.html>`__, the ``type``
        property refers to the post-aggregation data type. For example, we can calculate
        count ``distinct`` of a categorical field ``"cat"`` using ``{"aggregate":
        "distinct", "field": "cat"}``. The ``"type"`` of the aggregate output is
        ``"quantitative"``. - Secondary channels (e.g., ``x2``, ``y2``, ``xError``,
        ``yError`` ) do not have ``type`` as they must have exactly the same type as their
        primary channels (e.g., ``x``, ``y`` ).

        **See also:** `type <https://vega.github.io/vega-lite/docs/type.html>`__
        documentation.
    """
    _schema = {'$ref': '#/definitions/LatLongFieldDef'}

    def __init__(self, aggregate=Undefined, band=Undefined, bin=Undefined, field=Undefined,
                 timeUnit=Undefined, title=Undefined, type=Undefined, **kwds):
        super(LatLongFieldDef, self).__init__(aggregate=aggregate, band=band, bin=bin, field=field,
                                              timeUnit=timeUnit, title=title, type=type, **kwds)


class LayerRepeatMapping(VegaLiteSchema):
    """LayerRepeatMapping schema wrapper

    Mapping(required=[layer])

    Attributes
    ----------

    layer : List(string)
        An array of fields to be repeated as layers.
    column : List(string)
        An array of fields to be repeated horizontally.
    row : List(string)
        An array of fields to be repeated vertically.
    """
    _schema = {'$ref': '#/definitions/LayerRepeatMapping'}

    def __init__(self, layer=Undefined, column=Undefined, row=Undefined, **kwds):
        super(LayerRepeatMapping, self).__init__(layer=layer, column=column, row=row, **kwds)


class LayoutAlign(VegaLiteSchema):
    """LayoutAlign schema wrapper

    enum('all', 'each', 'none')
    """
    _schema = {'$ref': '#/definitions/LayoutAlign'}

    def __init__(self, *args):
        super(LayoutAlign, self).__init__(*args)


class Legend(VegaLiteSchema):
    """Legend schema wrapper

    Mapping(required=[])
    Properties of a legend or boolean flag for determining whether to show it.

    Attributes
    ----------

    aria : anyOf(boolean, :class:`ExprRef`)

    clipHeight : anyOf(float, :class:`ExprRef`)

    columnPadding : anyOf(float, :class:`ExprRef`)

    columns : anyOf(float, :class:`ExprRef`)

    cornerRadius : anyOf(float, :class:`ExprRef`)

    description : anyOf(string, :class:`ExprRef`)

    direction : :class:`Orientation`
        The direction of the legend, one of ``"vertical"`` or ``"horizontal"``.

        **Default value:** - For top-/bottom- ``orient`` ed legends, ``"horizontal"`` - For
        left-/right- ``orient`` ed legends, ``"vertical"`` - For top/bottom-left/right-
        ``orient`` ed legends, ``"horizontal"`` for gradient legends and ``"vertical"`` for
        symbol legends.
    fillColor : anyOf(anyOf(None, :class:`Color`), :class:`ExprRef`)

    format : anyOf(string, :class:`Dictunknown`)
        When used with the default ``"number"`` and ``"time"`` format type, the text
        formatting pattern for labels of guides (axes, legends, headers) and text marks.


        * If the format type is ``"number"`` (e.g., for quantitative fields), this is D3's
          `number format pattern <https://github.com/d3/d3-format#locale_format>`__. - If
          the format type is ``"time"`` (e.g., for temporal fields), this is D3's `time
          format pattern <https://github.com/d3/d3-time-format#locale_format>`__.

        See the `format documentation <https://vega.github.io/vega-lite/docs/format.html>`__
        for more examples.

        When used with a `custom formatType
        <https://vega.github.io/vega-lite/docs/config.html#custom-format-type>`__, this
        value will be passed as ``format`` alongside ``datum.value`` to the registered
        function.

        **Default value:**  Derived from `numberFormat
        <https://vega.github.io/vega-lite/docs/config.html#format>`__ config for number
        format and from `timeFormat
        <https://vega.github.io/vega-lite/docs/config.html#format>`__ config for time
        format.
    formatType : string
        The format type for labels. One of ``"number"``, ``"time"``, or a `registered custom
        format type
        <https://vega.github.io/vega-lite/docs/config.html#custom-format-type>`__.

        **Default value:** - ``"time"`` for temporal fields and ordinal and nominal fields
        with ``timeUnit``. - ``"number"`` for quantitative fields as well as ordinal and
        nominal fields without ``timeUnit``.
    gradientLength : anyOf(float, :class:`ExprRef`)

    gradientOpacity : anyOf(float, :class:`ExprRef`)

    gradientStrokeColor : anyOf(anyOf(None, :class:`Color`), :class:`ExprRef`)

    gradientStrokeWidth : anyOf(float, :class:`ExprRef`)

    gradientThickness : anyOf(float, :class:`ExprRef`)

    gridAlign : anyOf(:class:`LayoutAlign`, :class:`ExprRef`)

    labelAlign : anyOf(:class:`Align`, :class:`ExprRef`)

    labelBaseline : anyOf(:class:`TextBaseline`, :class:`ExprRef`)

    labelColor : anyOf(anyOf(None, :class:`Color`), :class:`ExprRef`)

    labelExpr : string
        `Vega expression <https://vega.github.io/vega/docs/expressions/>`__ for customizing
        labels.

        **Note:** The label text and value can be assessed via the ``label`` and ``value``
        properties of the legend's backing ``datum`` object.
    labelFont : anyOf(string, :class:`ExprRef`)

    labelFontSize : anyOf(float, :class:`ExprRef`)

    labelFontStyle : anyOf(:class:`FontStyle`, :class:`ExprRef`)

    labelFontWeight : anyOf(:class:`FontWeight`, :class:`ExprRef`)

    labelLimit : anyOf(float, :class:`ExprRef`)

    labelOffset : anyOf(float, :class:`ExprRef`)

    labelOpacity : anyOf(float, :class:`ExprRef`)

    labelOverlap : anyOf(:class:`LabelOverlap`, :class:`ExprRef`)

    labelPadding : anyOf(float, :class:`ExprRef`)

    labelSeparation : anyOf(float, :class:`ExprRef`)

    legendX : anyOf(float, :class:`ExprRef`)

    legendY : anyOf(float, :class:`ExprRef`)

    offset : anyOf(float, :class:`ExprRef`)

    orient : :class:`LegendOrient`
        The orientation of the legend, which determines how the legend is positioned within
        the scene. One of ``"left"``, ``"right"``, ``"top"``, ``"bottom"``, ``"top-left"``,
        ``"top-right"``, ``"bottom-left"``, ``"bottom-right"``, ``"none"``.

        **Default value:** ``"right"``
    padding : anyOf(float, :class:`ExprRef`)

    rowPadding : anyOf(float, :class:`ExprRef`)

    strokeColor : anyOf(anyOf(None, :class:`Color`), :class:`ExprRef`)

    symbolDash : anyOf(List(float), :class:`ExprRef`)

    symbolDashOffset : anyOf(float, :class:`ExprRef`)

    symbolFillColor : anyOf(anyOf(None, :class:`Color`), :class:`ExprRef`)

    symbolLimit : anyOf(float, :class:`ExprRef`)

    symbolOffset : anyOf(float, :class:`ExprRef`)

    symbolOpacity : anyOf(float, :class:`ExprRef`)

    symbolSize : anyOf(float, :class:`ExprRef`)

    symbolStrokeColor : anyOf(anyOf(None, :class:`Color`), :class:`ExprRef`)

    symbolStrokeWidth : anyOf(float, :class:`ExprRef`)

    symbolType : anyOf(:class:`SymbolShape`, :class:`ExprRef`)

    tickCount : anyOf(:class:`TickCount`, :class:`ExprRef`)

    tickMinStep : anyOf(float, :class:`ExprRef`)
        The minimum desired step between legend ticks, in terms of scale domain values. For
        example, a value of ``1`` indicates that ticks should not be less than 1 unit apart.
        If ``tickMinStep`` is specified, the ``tickCount`` value will be adjusted, if
        necessary, to enforce the minimum step value.

        **Default value** : ``undefined``
    title : anyOf(:class:`Text`, None)
        A title for the field. If ``null``, the title will be removed.

        **Default value:**  derived from the field's name and transformation function (
        ``aggregate``, ``bin`` and ``timeUnit`` ). If the field has an aggregate function,
        the function is displayed as part of the title (e.g., ``"Sum of Profit"`` ). If the
        field is binned or has a time unit applied, the applied function is shown in
        parentheses (e.g., ``"Profit (binned)"``, ``"Transaction Date (year-month)"`` ).
        Otherwise, the title is simply the field name.

        **Notes** :

        1) You can customize the default field title format by providing the `fieldTitle
        <https://vega.github.io/vega-lite/docs/config.html#top-level-config>`__ property in
        the `config <https://vega.github.io/vega-lite/docs/config.html>`__ or `fieldTitle
        function via the compile function's options
        <https://vega.github.io/vega-lite/docs/compile.html#field-title>`__.

        2) If both field definition's ``title`` and axis, header, or legend ``title`` are
        defined, axis/header/legend title will be used.
    titleAlign : anyOf(:class:`Align`, :class:`ExprRef`)

    titleAnchor : anyOf(:class:`TitleAnchor`, :class:`ExprRef`)

    titleBaseline : anyOf(:class:`TextBaseline`, :class:`ExprRef`)

    titleColor : anyOf(anyOf(None, :class:`Color`), :class:`ExprRef`)

    titleFont : anyOf(string, :class:`ExprRef`)

    titleFontSize : anyOf(float, :class:`ExprRef`)

    titleFontStyle : anyOf(:class:`FontStyle`, :class:`ExprRef`)

    titleFontWeight : anyOf(:class:`FontWeight`, :class:`ExprRef`)

    titleLimit : anyOf(float, :class:`ExprRef`)

    titleLineHeight : anyOf(float, :class:`ExprRef`)

    titleOpacity : anyOf(float, :class:`ExprRef`)

    titleOrient : anyOf(:class:`Orient`, :class:`ExprRef`)

    titlePadding : anyOf(float, :class:`ExprRef`)

    type : enum('symbol', 'gradient')
        The type of the legend. Use ``"symbol"`` to create a discrete legend and
        ``"gradient"`` for a continuous color gradient.

        **Default value:** ``"gradient"`` for non-binned quantitative fields and temporal
        fields; ``"symbol"`` otherwise.
    values : anyOf(List(float), List(string), List(boolean), List(:class:`DateTime`),
    :class:`ExprRef`)
        Explicitly set the visible legend values.
    zindex : float
        A non-negative integer indicating the z-index of the legend. If zindex is 0, legend
        should be drawn behind all chart elements. To put them in front, use zindex = 1.
    """
    _schema = {'$ref': '#/definitions/Legend'}

    def __init__(self, aria=Undefined, clipHeight=Undefined, columnPadding=Undefined, columns=Undefined,
                 cornerRadius=Undefined, description=Undefined, direction=Undefined,
                 fillColor=Undefined, format=Undefined, formatType=Undefined, gradientLength=Undefined,
                 gradientOpacity=Undefined, gradientStrokeColor=Undefined,
                 gradientStrokeWidth=Undefined, gradientThickness=Undefined, gridAlign=Undefined,
                 labelAlign=Undefined, labelBaseline=Undefined, labelColor=Undefined,
                 labelExpr=Undefined, labelFont=Undefined, labelFontSize=Undefined,
                 labelFontStyle=Undefined, labelFontWeight=Undefined, labelLimit=Undefined,
                 labelOffset=Undefined, labelOpacity=Undefined, labelOverlap=Undefined,
                 labelPadding=Undefined, labelSeparation=Undefined, legendX=Undefined,
                 legendY=Undefined, offset=Undefined, orient=Undefined, padding=Undefined,
                 rowPadding=Undefined, strokeColor=Undefined, symbolDash=Undefined,
                 symbolDashOffset=Undefined, symbolFillColor=Undefined, symbolLimit=Undefined,
                 symbolOffset=Undefined, symbolOpacity=Undefined, symbolSize=Undefined,
                 symbolStrokeColor=Undefined, symbolStrokeWidth=Undefined, symbolType=Undefined,
                 tickCount=Undefined, tickMinStep=Undefined, title=Undefined, titleAlign=Undefined,
                 titleAnchor=Undefined, titleBaseline=Undefined, titleColor=Undefined,
                 titleFont=Undefined, titleFontSize=Undefined, titleFontStyle=Undefined,
                 titleFontWeight=Undefined, titleLimit=Undefined, titleLineHeight=Undefined,
                 titleOpacity=Undefined, titleOrient=Undefined, titlePadding=Undefined, type=Undefined,
                 values=Undefined, zindex=Undefined, **kwds):
        super(Legend, self).__init__(aria=aria, clipHeight=clipHeight, columnPadding=columnPadding,
                                     columns=columns, cornerRadius=cornerRadius,
                                     description=description, direction=direction, fillColor=fillColor,
                                     format=format, formatType=formatType,
                                     gradientLength=gradientLength, gradientOpacity=gradientOpacity,
                                     gradientStrokeColor=gradientStrokeColor,
                                     gradientStrokeWidth=gradientStrokeWidth,
                                     gradientThickness=gradientThickness, gridAlign=gridAlign,
                                     labelAlign=labelAlign, labelBaseline=labelBaseline,
                                     labelColor=labelColor, labelExpr=labelExpr, labelFont=labelFont,
                                     labelFontSize=labelFontSize, labelFontStyle=labelFontStyle,
                                     labelFontWeight=labelFontWeight, labelLimit=labelLimit,
                                     labelOffset=labelOffset, labelOpacity=labelOpacity,
                                     labelOverlap=labelOverlap, labelPadding=labelPadding,
                                     labelSeparation=labelSeparation, legendX=legendX, legendY=legendY,
                                     offset=offset, orient=orient, padding=padding,
                                     rowPadding=rowPadding, strokeColor=strokeColor,
                                     symbolDash=symbolDash, symbolDashOffset=symbolDashOffset,
                                     symbolFillColor=symbolFillColor, symbolLimit=symbolLimit,
                                     symbolOffset=symbolOffset, symbolOpacity=symbolOpacity,
                                     symbolSize=symbolSize, symbolStrokeColor=symbolStrokeColor,
                                     symbolStrokeWidth=symbolStrokeWidth, symbolType=symbolType,
                                     tickCount=tickCount, tickMinStep=tickMinStep, title=title,
                                     titleAlign=titleAlign, titleAnchor=titleAnchor,
                                     titleBaseline=titleBaseline, titleColor=titleColor,
                                     titleFont=titleFont, titleFontSize=titleFontSize,
                                     titleFontStyle=titleFontStyle, titleFontWeight=titleFontWeight,
                                     titleLimit=titleLimit, titleLineHeight=titleLineHeight,
                                     titleOpacity=titleOpacity, titleOrient=titleOrient,
                                     titlePadding=titlePadding, type=type, values=values, zindex=zindex,
                                     **kwds)


class LegendBinding(VegaLiteSchema):
    """LegendBinding schema wrapper

    anyOf(string, :class:`LegendStreamBinding`)
    """
    _schema = {'$ref': '#/definitions/LegendBinding'}

    def __init__(self, *args, **kwds):
        super(LegendBinding, self).__init__(*args, **kwds)


class LegendConfig(VegaLiteSchema):
    """LegendConfig schema wrapper

    Mapping(required=[])

    Attributes
    ----------

    aria : anyOf(boolean, :class:`ExprRef`)

    clipHeight : anyOf(float, :class:`ExprRef`)

    columnPadding : anyOf(float, :class:`ExprRef`)

    columns : anyOf(float, :class:`ExprRef`)

    cornerRadius : anyOf(float, :class:`ExprRef`)

    description : anyOf(string, :class:`ExprRef`)

    direction : :class:`Orientation`
        The direction of the legend, one of ``"vertical"`` or ``"horizontal"``.

        **Default value:** - For top-/bottom- ``orient`` ed legends, ``"horizontal"`` - For
        left-/right- ``orient`` ed legends, ``"vertical"`` - For top/bottom-left/right-
        ``orient`` ed legends, ``"horizontal"`` for gradient legends and ``"vertical"`` for
        symbol legends.
    disable : boolean
        Disable legend by default
    fillColor : anyOf(anyOf(None, :class:`Color`), :class:`ExprRef`)

    gradientDirection : anyOf(:class:`Orientation`, :class:`ExprRef`)

    gradientHorizontalMaxLength : float
        Max legend length for a horizontal gradient when ``config.legend.gradientLength`` is
        undefined.

        **Default value:** ``200``
    gradientHorizontalMinLength : float
        Min legend length for a horizontal gradient when ``config.legend.gradientLength`` is
        undefined.

        **Default value:** ``100``
    gradientLabelLimit : anyOf(float, :class:`ExprRef`)

    gradientLabelOffset : anyOf(float, :class:`ExprRef`)

    gradientLength : anyOf(float, :class:`ExprRef`)

    gradientOpacity : anyOf(float, :class:`ExprRef`)

    gradientStrokeColor : anyOf(anyOf(None, :class:`Color`), :class:`ExprRef`)

    gradientStrokeWidth : anyOf(float, :class:`ExprRef`)

    gradientThickness : anyOf(float, :class:`ExprRef`)

    gradientVerticalMaxLength : float
        Max legend length for a vertical gradient when ``config.legend.gradientLength`` is
        undefined.

        **Default value:** ``200``
    gradientVerticalMinLength : float
        Min legend length for a vertical gradient when ``config.legend.gradientLength`` is
        undefined.

        **Default value:** ``100``
    gridAlign : anyOf(:class:`LayoutAlign`, :class:`ExprRef`)

    labelAlign : anyOf(:class:`Align`, :class:`ExprRef`)

    labelBaseline : anyOf(:class:`TextBaseline`, :class:`ExprRef`)

    labelColor : anyOf(anyOf(None, :class:`Color`), :class:`ExprRef`)

    labelFont : anyOf(string, :class:`ExprRef`)

    labelFontSize : anyOf(float, :class:`ExprRef`)

    labelFontStyle : anyOf(:class:`FontStyle`, :class:`ExprRef`)

    labelFontWeight : anyOf(:class:`FontWeight`, :class:`ExprRef`)

    labelLimit : anyOf(float, :class:`ExprRef`)

    labelOffset : anyOf(float, :class:`ExprRef`)

    labelOpacity : anyOf(float, :class:`ExprRef`)

    labelOverlap : anyOf(:class:`LabelOverlap`, :class:`ExprRef`)
        The strategy to use for resolving overlap of labels in gradient legends. If
        ``false``, no overlap reduction is attempted. If set to ``true`` or ``"parity"``, a
        strategy of removing every other label is used. If set to ``"greedy"``, a linear
        scan of the labels is performed, removing any label that overlaps with the last
        visible label (this often works better for log-scaled axes).

        **Default value:** ``"greedy"`` for ``log scales otherwise`` true`.
    labelPadding : anyOf(float, :class:`ExprRef`)

    labelSeparation : anyOf(float, :class:`ExprRef`)

    layout : :class:`ExprRef`

    legendX : anyOf(float, :class:`ExprRef`)

    legendY : anyOf(float, :class:`ExprRef`)

    offset : anyOf(float, :class:`ExprRef`)

    orient : :class:`LegendOrient`
        The orientation of the legend, which determines how the legend is positioned within
        the scene. One of ``"left"``, ``"right"``, ``"top"``, ``"bottom"``, ``"top-left"``,
        ``"top-right"``, ``"bottom-left"``, ``"bottom-right"``, ``"none"``.

        **Default value:** ``"right"``
    padding : anyOf(float, :class:`ExprRef`)

    rowPadding : anyOf(float, :class:`ExprRef`)

    strokeColor : anyOf(anyOf(None, :class:`Color`), :class:`ExprRef`)

    strokeDash : anyOf(List(float), :class:`ExprRef`)

    strokeWidth : anyOf(float, :class:`ExprRef`)

    symbolBaseFillColor : anyOf(anyOf(None, :class:`Color`), :class:`ExprRef`)

    symbolBaseStrokeColor : anyOf(anyOf(None, :class:`Color`), :class:`ExprRef`)

    symbolDash : anyOf(List(float), :class:`ExprRef`)

    symbolDashOffset : anyOf(float, :class:`ExprRef`)

    symbolDirection : anyOf(:class:`Orientation`, :class:`ExprRef`)

    symbolFillColor : anyOf(anyOf(None, :class:`Color`), :class:`ExprRef`)

    symbolLimit : anyOf(float, :class:`ExprRef`)

    symbolOffset : anyOf(float, :class:`ExprRef`)

    symbolOpacity : anyOf(float, :class:`ExprRef`)

    symbolSize : anyOf(float, :class:`ExprRef`)

    symbolStrokeColor : anyOf(anyOf(None, :class:`Color`), :class:`ExprRef`)

    symbolStrokeWidth : anyOf(float, :class:`ExprRef`)

    symbolType : anyOf(:class:`SymbolShape`, :class:`ExprRef`)

    tickCount : anyOf(:class:`TickCount`, :class:`ExprRef`)

    title : None
        Set to null to disable title for the axis, legend, or header.
    titleAlign : anyOf(:class:`Align`, :class:`ExprRef`)

    titleAnchor : anyOf(:class:`TitleAnchor`, :class:`ExprRef`)

    titleBaseline : anyOf(:class:`TextBaseline`, :class:`ExprRef`)

    titleColor : anyOf(anyOf(None, :class:`Color`), :class:`ExprRef`)

    titleFont : anyOf(string, :class:`ExprRef`)

    titleFontSize : anyOf(float, :class:`ExprRef`)

    titleFontStyle : anyOf(:class:`FontStyle`, :class:`ExprRef`)

    titleFontWeight : anyOf(:class:`FontWeight`, :class:`ExprRef`)

    titleLimit : anyOf(float, :class:`ExprRef`)

    titleLineHeight : anyOf(float, :class:`ExprRef`)

    titleOpacity : anyOf(float, :class:`ExprRef`)

    titleOrient : anyOf(:class:`Orient`, :class:`ExprRef`)

    titlePadding : anyOf(float, :class:`ExprRef`)

    unselectedOpacity : float
        The opacity of unselected legend entries.

        **Default value:** 0.35.
    zindex : anyOf(float, :class:`ExprRef`)

    """
    _schema = {'$ref': '#/definitions/LegendConfig'}

    def __init__(self, aria=Undefined, clipHeight=Undefined, columnPadding=Undefined, columns=Undefined,
                 cornerRadius=Undefined, description=Undefined, direction=Undefined, disable=Undefined,
                 fillColor=Undefined, gradientDirection=Undefined,
                 gradientHorizontalMaxLength=Undefined, gradientHorizontalMinLength=Undefined,
                 gradientLabelLimit=Undefined, gradientLabelOffset=Undefined, gradientLength=Undefined,
                 gradientOpacity=Undefined, gradientStrokeColor=Undefined,
                 gradientStrokeWidth=Undefined, gradientThickness=Undefined,
                 gradientVerticalMaxLength=Undefined, gradientVerticalMinLength=Undefined,
                 gridAlign=Undefined, labelAlign=Undefined, labelBaseline=Undefined,
                 labelColor=Undefined, labelFont=Undefined, labelFontSize=Undefined,
                 labelFontStyle=Undefined, labelFontWeight=Undefined, labelLimit=Undefined,
                 labelOffset=Undefined, labelOpacity=Undefined, labelOverlap=Undefined,
                 labelPadding=Undefined, labelSeparation=Undefined, layout=Undefined, legendX=Undefined,
                 legendY=Undefined, offset=Undefined, orient=Undefined, padding=Undefined,
                 rowPadding=Undefined, strokeColor=Undefined, strokeDash=Undefined,
                 strokeWidth=Undefined, symbolBaseFillColor=Undefined, symbolBaseStrokeColor=Undefined,
                 symbolDash=Undefined, symbolDashOffset=Undefined, symbolDirection=Undefined,
                 symbolFillColor=Undefined, symbolLimit=Undefined, symbolOffset=Undefined,
                 symbolOpacity=Undefined, symbolSize=Undefined, symbolStrokeColor=Undefined,
                 symbolStrokeWidth=Undefined, symbolType=Undefined, tickCount=Undefined,
                 title=Undefined, titleAlign=Undefined, titleAnchor=Undefined, titleBaseline=Undefined,
                 titleColor=Undefined, titleFont=Undefined, titleFontSize=Undefined,
                 titleFontStyle=Undefined, titleFontWeight=Undefined, titleLimit=Undefined,
                 titleLineHeight=Undefined, titleOpacity=Undefined, titleOrient=Undefined,
                 titlePadding=Undefined, unselectedOpacity=Undefined, zindex=Undefined, **kwds):
        super(LegendConfig, self).__init__(aria=aria, clipHeight=clipHeight,
                                           columnPadding=columnPadding, columns=columns,
                                           cornerRadius=cornerRadius, description=description,
                                           direction=direction, disable=disable, fillColor=fillColor,
                                           gradientDirection=gradientDirection,
                                           gradientHorizontalMaxLength=gradientHorizontalMaxLength,
                                           gradientHorizontalMinLength=gradientHorizontalMinLength,
                                           gradientLabelLimit=gradientLabelLimit,
                                           gradientLabelOffset=gradientLabelOffset,
                                           gradientLength=gradientLength,
                                           gradientOpacity=gradientOpacity,
                                           gradientStrokeColor=gradientStrokeColor,
                                           gradientStrokeWidth=gradientStrokeWidth,
                                           gradientThickness=gradientThickness,
                                           gradientVerticalMaxLength=gradientVerticalMaxLength,
                                           gradientVerticalMinLength=gradientVerticalMinLength,
                                           gridAlign=gridAlign, labelAlign=labelAlign,
                                           labelBaseline=labelBaseline, labelColor=labelColor,
                                           labelFont=labelFont, labelFontSize=labelFontSize,
                                           labelFontStyle=labelFontStyle,
                                           labelFontWeight=labelFontWeight, labelLimit=labelLimit,
                                           labelOffset=labelOffset, labelOpacity=labelOpacity,
                                           labelOverlap=labelOverlap, labelPadding=labelPadding,
                                           labelSeparation=labelSeparation, layout=layout,
                                           legendX=legendX, legendY=legendY, offset=offset,
                                           orient=orient, padding=padding, rowPadding=rowPadding,
                                           strokeColor=strokeColor, strokeDash=strokeDash,
                                           strokeWidth=strokeWidth,
                                           symbolBaseFillColor=symbolBaseFillColor,
                                           symbolBaseStrokeColor=symbolBaseStrokeColor,
                                           symbolDash=symbolDash, symbolDashOffset=symbolDashOffset,
                                           symbolDirection=symbolDirection,
                                           symbolFillColor=symbolFillColor, symbolLimit=symbolLimit,
                                           symbolOffset=symbolOffset, symbolOpacity=symbolOpacity,
                                           symbolSize=symbolSize, symbolStrokeColor=symbolStrokeColor,
                                           symbolStrokeWidth=symbolStrokeWidth, symbolType=symbolType,
                                           tickCount=tickCount, title=title, titleAlign=titleAlign,
                                           titleAnchor=titleAnchor, titleBaseline=titleBaseline,
                                           titleColor=titleColor, titleFont=titleFont,
                                           titleFontSize=titleFontSize, titleFontStyle=titleFontStyle,
                                           titleFontWeight=titleFontWeight, titleLimit=titleLimit,
                                           titleLineHeight=titleLineHeight, titleOpacity=titleOpacity,
                                           titleOrient=titleOrient, titlePadding=titlePadding,
                                           unselectedOpacity=unselectedOpacity, zindex=zindex, **kwds)


class LegendOrient(VegaLiteSchema):
    """LegendOrient schema wrapper

    enum('none', 'left', 'right', 'top', 'bottom', 'top-left', 'top-right', 'bottom-left',
    'bottom-right')
    """
    _schema = {'$ref': '#/definitions/LegendOrient'}

    def __init__(self, *args):
        super(LegendOrient, self).__init__(*args)


class LegendResolveMap(VegaLiteSchema):
    """LegendResolveMap schema wrapper

    Mapping(required=[])

    Attributes
    ----------

    angle : :class:`ResolveMode`

    color : :class:`ResolveMode`

    fill : :class:`ResolveMode`

    fillOpacity : :class:`ResolveMode`

    opacity : :class:`ResolveMode`

    shape : :class:`ResolveMode`

    size : :class:`ResolveMode`

    stroke : :class:`ResolveMode`

    strokeDash : :class:`ResolveMode`

    strokeOpacity : :class:`ResolveMode`

    strokeWidth : :class:`ResolveMode`

    """
    _schema = {'$ref': '#/definitions/LegendResolveMap'}

    def __init__(self, angle=Undefined, color=Undefined, fill=Undefined, fillOpacity=Undefined,
                 opacity=Undefined, shape=Undefined, size=Undefined, stroke=Undefined,
                 strokeDash=Undefined, strokeOpacity=Undefined, strokeWidth=Undefined, **kwds):
        super(LegendResolveMap, self).__init__(angle=angle, color=color, fill=fill,
                                               fillOpacity=fillOpacity, opacity=opacity, shape=shape,
                                               size=size, stroke=stroke, strokeDash=strokeDash,
                                               strokeOpacity=strokeOpacity, strokeWidth=strokeWidth,
                                               **kwds)


class LegendStreamBinding(LegendBinding):
    """LegendStreamBinding schema wrapper

    Mapping(required=[legend])

    Attributes
    ----------

    legend : anyOf(string, :class:`Stream`)

    """
    _schema = {'$ref': '#/definitions/LegendStreamBinding'}

    def __init__(self, legend=Undefined, **kwds):
        super(LegendStreamBinding, self).__init__(legend=legend, **kwds)


class LineConfig(AnyMarkConfig):
    """LineConfig schema wrapper

    Mapping(required=[])

    Attributes
    ----------

    align : anyOf(:class:`Align`, :class:`ExprRef`)
        The horizontal alignment of the text or ranged marks (area, bar, image, rect, rule).
        One of ``"left"``, ``"right"``, ``"center"``.

        **Note:** Expression reference is *not* supported for range marks.
    angle : anyOf(float, :class:`ExprRef`)

    aria : anyOf(boolean, :class:`ExprRef`)

    ariaRole : anyOf(string, :class:`ExprRef`)

    ariaRoleDescription : anyOf(string, :class:`ExprRef`)

    aspect : anyOf(boolean, :class:`ExprRef`)

    baseline : anyOf(:class:`TextBaseline`, :class:`ExprRef`)
        For text marks, the vertical text baseline. One of ``"alphabetic"`` (default),
        ``"top"``, ``"middle"``, ``"bottom"``, ``"line-top"``, ``"line-bottom"``, or an
        expression reference that provides one of the valid values. The ``"line-top"`` and
        ``"line-bottom"`` values operate similarly to ``"top"`` and ``"bottom"``, but are
        calculated relative to the ``lineHeight`` rather than ``fontSize`` alone.

        For range marks, the vertical alignment of the marks. One of ``"top"``,
        ``"middle"``, ``"bottom"``.

        **Note:** Expression reference is *not* supported for range marks.
    blend : anyOf(:class:`Blend`, :class:`ExprRef`)

    color : anyOf(:class:`Color`, :class:`Gradient`, :class:`ExprRef`)
        Default color.

        **Default value:** :raw-html:`<span style="color: #4682b4;">&#9632;</span>`
        ``"#4682b4"``

        **Note:** - This property cannot be used in a `style config
        <https://vega.github.io/vega-lite/docs/mark.html#style-config>`__. - The ``fill``
        and ``stroke`` properties have higher precedence than ``color`` and will override
        ``color``.
    cornerRadius : anyOf(float, :class:`ExprRef`)

    cornerRadiusBottomLeft : anyOf(float, :class:`ExprRef`)

    cornerRadiusBottomRight : anyOf(float, :class:`ExprRef`)

    cornerRadiusTopLeft : anyOf(float, :class:`ExprRef`)

    cornerRadiusTopRight : anyOf(float, :class:`ExprRef`)

    cursor : anyOf(:class:`Cursor`, :class:`ExprRef`)

    description : anyOf(string, :class:`ExprRef`)

    dir : anyOf(:class:`TextDirection`, :class:`ExprRef`)

    dx : anyOf(float, :class:`ExprRef`)

    dy : anyOf(float, :class:`ExprRef`)

    ellipsis : anyOf(string, :class:`ExprRef`)

    endAngle : anyOf(float, :class:`ExprRef`)

    fill : anyOf(:class:`Color`, :class:`Gradient`, None, :class:`ExprRef`)
        Default fill color. This property has higher precedence than ``config.color``. Set
        to ``null`` to remove fill.

        **Default value:** (None)
    fillOpacity : anyOf(float, :class:`ExprRef`)

    filled : boolean
        Whether the mark's color should be used as fill color instead of stroke color.

        **Default value:** ``false`` for all ``point``, ``line``, and ``rule`` marks as well
        as ``geoshape`` marks for `graticule
        <https://vega.github.io/vega-lite/docs/data.html#graticule>`__ data sources;
        otherwise, ``true``.

        **Note:** This property cannot be used in a `style config
        <https://vega.github.io/vega-lite/docs/mark.html#style-config>`__.
    font : anyOf(string, :class:`ExprRef`)

    fontSize : anyOf(float, :class:`ExprRef`)

    fontStyle : anyOf(:class:`FontStyle`, :class:`ExprRef`)

    fontWeight : anyOf(:class:`FontWeight`, :class:`ExprRef`)

    height : anyOf(float, :class:`ExprRef`)

    href : anyOf(:class:`URI`, :class:`ExprRef`)

    innerRadius : anyOf(float, :class:`ExprRef`)
        The inner radius in pixels of arc marks. ``innerRadius`` is an alias for
        ``radius2``.
    interpolate : anyOf(:class:`Interpolate`, :class:`ExprRef`)

    invalid : enum('filter', None)
        Defines how Vega-Lite should handle marks for invalid values ( ``null`` and ``NaN``
        ). - If set to ``"filter"`` (default), all data items with null values will be
        skipped (for line, trail, and area marks) or filtered (for other marks). - If
        ``null``, all data items are included. In this case, invalid values will be
        interpreted as zeroes.
    limit : anyOf(float, :class:`ExprRef`)

    lineBreak : anyOf(string, :class:`ExprRef`)

    lineHeight : anyOf(float, :class:`ExprRef`)

    opacity : anyOf(float, :class:`ExprRef`)
        The overall opacity (value between [0,1]).

        **Default value:** ``0.7`` for non-aggregate plots with ``point``, ``tick``,
        ``circle``, or ``square`` marks or layered ``bar`` charts and ``1`` otherwise.
    order : anyOf(None, boolean)
        For line and trail marks, this ``order`` property can be set to ``null`` or
        ``false`` to make the lines use the original order in the data sources.
    orient : :class:`Orientation`
        The orientation of a non-stacked bar, tick, area, and line charts. The value is
        either horizontal (default) or vertical. - For bar, rule and tick, this determines
        whether the size of the bar and tick should be applied to x or y dimension. - For
        area, this property determines the orient property of the Vega output. - For line
        and trail marks, this property determines the sort order of the points in the line
        if ``config.sortLineBy`` is not specified. For stacked charts, this is always
        determined by the orientation of the stack; therefore explicitly specified value
        will be ignored.
    outerRadius : anyOf(float, :class:`ExprRef`)
        The outer radius in pixels of arc marks. ``outerRadius`` is an alias for ``radius``.
    padAngle : anyOf(float, :class:`ExprRef`)

    point : anyOf(boolean, :class:`OverlayMarkDef`, string)
        A flag for overlaying points on top of line or area marks, or an object defining the
        properties of the overlayed points.


        If this property is ``"transparent"``, transparent points will be used (for
        enhancing tooltips and selections).

        If this property is an empty object ( ``{}`` ) or ``true``, filled points with
        default properties will be used.

        If this property is ``false``, no points would be automatically added to line or
        area marks.

        **Default value:** ``false``.
    radius : anyOf(float, :class:`ExprRef`)
        For arc mark, the primary (outer) radius in pixels.

        For text marks, polar coordinate radial offset, in pixels, of the text from the
        origin determined by the ``x`` and ``y`` properties.
    radius2 : anyOf(float, :class:`ExprRef`)
        The secondary (inner) radius in pixels of arc marks.
    shape : anyOf(anyOf(:class:`SymbolShape`, string), :class:`ExprRef`)

    size : anyOf(float, :class:`ExprRef`)
        Default size for marks. - For ``point`` / ``circle`` / ``square``, this represents
        the pixel area of the marks. Note that this value sets the area of the symbol; the
        side lengths will increase with the square root of this value. - For ``bar``, this
        represents the band size of the bar, in pixels. - For ``text``, this represents the
        font size, in pixels.

        **Default value:** - ``30`` for point, circle, square marks; width/height's ``step``
        - ``2`` for bar marks with discrete dimensions; - ``5`` for bar marks with
        continuous dimensions; - ``11`` for text marks.
    smooth : anyOf(boolean, :class:`ExprRef`)

    startAngle : anyOf(float, :class:`ExprRef`)

    stroke : anyOf(:class:`Color`, :class:`Gradient`, None, :class:`ExprRef`)
        Default stroke color. This property has higher precedence than ``config.color``. Set
        to ``null`` to remove stroke.

        **Default value:** (None)
    strokeCap : anyOf(:class:`StrokeCap`, :class:`ExprRef`)

    strokeDash : anyOf(List(float), :class:`ExprRef`)

    strokeDashOffset : anyOf(float, :class:`ExprRef`)

    strokeJoin : anyOf(:class:`StrokeJoin`, :class:`ExprRef`)

    strokeMiterLimit : anyOf(float, :class:`ExprRef`)

    strokeOffset : anyOf(float, :class:`ExprRef`)

    strokeOpacity : anyOf(float, :class:`ExprRef`)

    strokeWidth : anyOf(float, :class:`ExprRef`)

    tension : anyOf(float, :class:`ExprRef`)

    text : anyOf(:class:`Text`, :class:`ExprRef`)

    theta : anyOf(float, :class:`ExprRef`)
        For arc marks, the arc length in radians if theta2 is not specified, otherwise the
        start arc angle. (A value of 0 indicates up or “north”, increasing values proceed
        clockwise.)

        For text marks, polar coordinate angle in radians.
    theta2 : anyOf(float, :class:`ExprRef`)
        The end angle of arc marks in radians. A value of 0 indicates up or “north”,
        increasing values proceed clockwise.
    timeUnitBand : float
        Default relative band size for a time unit. If set to ``1``, the bandwidth of the
        marks will be equal to the time unit band step. If set to ``0.5``, bandwidth of the
        marks will be half of the time unit band step.
    timeUnitBandPosition : float
        Default relative band position for a time unit. If set to ``0``, the marks will be
        positioned at the beginning of the time unit band step. If set to ``0.5``, the marks
        will be positioned in the middle of the time unit band step.
    tooltip : anyOf(float, string, boolean, :class:`TooltipContent`, :class:`ExprRef`, None)
        The tooltip text string to show upon mouse hover or an object defining which fields
        should the tooltip be derived from.


        * If ``tooltip`` is ``true`` or ``{"content": "encoding"}``, then all fields from
          ``encoding`` will be used. - If ``tooltip`` is ``{"content": "data"}``, then all
          fields that appear in the highlighted data point will be used. - If set to
          ``null`` or ``false``, then no tooltip will be used.

        See the `tooltip <https://vega.github.io/vega-lite/docs/tooltip.html>`__
        documentation for a detailed discussion about tooltip  in Vega-Lite.

        **Default value:** ``null``
    url : anyOf(:class:`URI`, :class:`ExprRef`)

    width : anyOf(float, :class:`ExprRef`)

    x : anyOf(float, string, :class:`ExprRef`)
        X coordinates of the marks, or width of horizontal ``"bar"`` and ``"area"`` without
        specified ``x2`` or ``width``.

        The ``value`` of this channel can be a number or a string ``"width"`` for the width
        of the plot.
    x2 : anyOf(float, string, :class:`ExprRef`)
        X2 coordinates for ranged ``"area"``, ``"bar"``, ``"rect"``, and  ``"rule"``.

        The ``value`` of this channel can be a number or a string ``"width"`` for the width
        of the plot.
    y : anyOf(float, string, :class:`ExprRef`)
        Y coordinates of the marks, or height of vertical ``"bar"`` and ``"area"`` without
        specified ``y2`` or ``height``.

        The ``value`` of this channel can be a number or a string ``"height"`` for the
        height of the plot.
    y2 : anyOf(float, string, :class:`ExprRef`)
        Y2 coordinates for ranged ``"area"``, ``"bar"``, ``"rect"``, and  ``"rule"``.

        The ``value`` of this channel can be a number or a string ``"height"`` for the
        height of the plot.
    """
    _schema = {'$ref': '#/definitions/LineConfig'}

    def __init__(self, align=Undefined, angle=Undefined, aria=Undefined, ariaRole=Undefined,
                 ariaRoleDescription=Undefined, aspect=Undefined, baseline=Undefined, blend=Undefined,
                 color=Undefined, cornerRadius=Undefined, cornerRadiusBottomLeft=Undefined,
                 cornerRadiusBottomRight=Undefined, cornerRadiusTopLeft=Undefined,
                 cornerRadiusTopRight=Undefined, cursor=Undefined, description=Undefined, dir=Undefined,
                 dx=Undefined, dy=Undefined, ellipsis=Undefined, endAngle=Undefined, fill=Undefined,
                 fillOpacity=Undefined, filled=Undefined, font=Undefined, fontSize=Undefined,
                 fontStyle=Undefined, fontWeight=Undefined, height=Undefined, href=Undefined,
                 innerRadius=Undefined, interpolate=Undefined, invalid=Undefined, limit=Undefined,
                 lineBreak=Undefined, lineHeight=Undefined, opacity=Undefined, order=Undefined,
                 orient=Undefined, outerRadius=Undefined, padAngle=Undefined, point=Undefined,
                 radius=Undefined, radius2=Undefined, shape=Undefined, size=Undefined, smooth=Undefined,
                 startAngle=Undefined, stroke=Undefined, strokeCap=Undefined, strokeDash=Undefined,
                 strokeDashOffset=Undefined, strokeJoin=Undefined, strokeMiterLimit=Undefined,
                 strokeOffset=Undefined, strokeOpacity=Undefined, strokeWidth=Undefined,
                 tension=Undefined, text=Undefined, theta=Undefined, theta2=Undefined,
                 timeUnitBand=Undefined, timeUnitBandPosition=Undefined, tooltip=Undefined,
                 url=Undefined, width=Undefined, x=Undefined, x2=Undefined, y=Undefined, y2=Undefined,
                 **kwds):
        super(LineConfig, self).__init__(align=align, angle=angle, aria=aria, ariaRole=ariaRole,
                                         ariaRoleDescription=ariaRoleDescription, aspect=aspect,
                                         baseline=baseline, blend=blend, color=color,
                                         cornerRadius=cornerRadius,
                                         cornerRadiusBottomLeft=cornerRadiusBottomLeft,
                                         cornerRadiusBottomRight=cornerRadiusBottomRight,
                                         cornerRadiusTopLeft=cornerRadiusTopLeft,
                                         cornerRadiusTopRight=cornerRadiusTopRight, cursor=cursor,
                                         description=description, dir=dir, dx=dx, dy=dy,
                                         ellipsis=ellipsis, endAngle=endAngle, fill=fill,
                                         fillOpacity=fillOpacity, filled=filled, font=font,
                                         fontSize=fontSize, fontStyle=fontStyle, fontWeight=fontWeight,
                                         height=height, href=href, innerRadius=innerRadius,
                                         interpolate=interpolate, invalid=invalid, limit=limit,
                                         lineBreak=lineBreak, lineHeight=lineHeight, opacity=opacity,
                                         order=order, orient=orient, outerRadius=outerRadius,
                                         padAngle=padAngle, point=point, radius=radius, radius2=radius2,
                                         shape=shape, size=size, smooth=smooth, startAngle=startAngle,
                                         stroke=stroke, strokeCap=strokeCap, strokeDash=strokeDash,
                                         strokeDashOffset=strokeDashOffset, strokeJoin=strokeJoin,
                                         strokeMiterLimit=strokeMiterLimit, strokeOffset=strokeOffset,
                                         strokeOpacity=strokeOpacity, strokeWidth=strokeWidth,
                                         tension=tension, text=text, theta=theta, theta2=theta2,
                                         timeUnitBand=timeUnitBand,
                                         timeUnitBandPosition=timeUnitBandPosition, tooltip=tooltip,
                                         url=url, width=width, x=x, x2=x2, y=y, y2=y2, **kwds)


class LinearGradient(Gradient):
    """LinearGradient schema wrapper

    Mapping(required=[gradient, stops])

    Attributes
    ----------

    gradient : string
        The type of gradient. Use ``"linear"`` for a linear gradient.
    stops : List(:class:`GradientStop`)
        An array of gradient stops defining the gradient color sequence.
    id : string

    x1 : float
        The starting x-coordinate, in normalized [0, 1] coordinates, of the linear gradient.

        **Default value:** ``0``
    x2 : float
        The ending x-coordinate, in normalized [0, 1] coordinates, of the linear gradient.

        **Default value:** ``1``
    y1 : float
        The starting y-coordinate, in normalized [0, 1] coordinates, of the linear gradient.

        **Default value:** ``0``
    y2 : float
        The ending y-coordinate, in normalized [0, 1] coordinates, of the linear gradient.

        **Default value:** ``0``
    """
    _schema = {'$ref': '#/definitions/LinearGradient'}

    def __init__(self, gradient=Undefined, stops=Undefined, id=Undefined, x1=Undefined, x2=Undefined,
                 y1=Undefined, y2=Undefined, **kwds):
        super(LinearGradient, self).__init__(gradient=gradient, stops=stops, id=id, x1=x1, x2=x2, y1=y1,
                                             y2=y2, **kwds)


class LookupData(VegaLiteSchema):
    """LookupData schema wrapper

    Mapping(required=[data, key])

    Attributes
    ----------

    data : :class:`Data`
        Secondary data source to lookup in.
    key : :class:`FieldName`
        Key in data to lookup.
    fields : List(:class:`FieldName`)
        Fields in foreign data or selection to lookup. If not specified, the entire object
        is queried.
    """
    _schema = {'$ref': '#/definitions/LookupData'}

    def __init__(self, data=Undefined, key=Undefined, fields=Undefined, **kwds):
        super(LookupData, self).__init__(data=data, key=key, fields=fields, **kwds)


class LookupSelection(VegaLiteSchema):
    """LookupSelection schema wrapper

    Mapping(required=[key, selection])

    Attributes
    ----------

    key : :class:`FieldName`
        Key in data to lookup.
    selection : string
        Selection name to look up.
    fields : List(:class:`FieldName`)
        Fields in foreign data or selection to lookup. If not specified, the entire object
        is queried.
    """
    _schema = {'$ref': '#/definitions/LookupSelection'}

    def __init__(self, key=Undefined, selection=Undefined, fields=Undefined, **kwds):
        super(LookupSelection, self).__init__(key=key, selection=selection, fields=fields, **kwds)


class Mark(AnyMark):
    """Mark schema wrapper

    enum('arc', 'area', 'bar', 'image', 'line', 'point', 'rect', 'rule', 'text', 'tick',
    'trail', 'circle', 'square', 'geoshape')
    All types of primitive marks.
    """
    _schema = {'$ref': '#/definitions/Mark'}

    def __init__(self, *args):
        super(Mark, self).__init__(*args)


class MarkConfig(AnyMarkConfig):
    """MarkConfig schema wrapper

    Mapping(required=[])

    Attributes
    ----------

    align : anyOf(:class:`Align`, :class:`ExprRef`)
        The horizontal alignment of the text or ranged marks (area, bar, image, rect, rule).
        One of ``"left"``, ``"right"``, ``"center"``.

        **Note:** Expression reference is *not* supported for range marks.
    angle : anyOf(float, :class:`ExprRef`)

    aria : anyOf(boolean, :class:`ExprRef`)

    ariaRole : anyOf(string, :class:`ExprRef`)

    ariaRoleDescription : anyOf(string, :class:`ExprRef`)

    aspect : anyOf(boolean, :class:`ExprRef`)

    baseline : anyOf(:class:`TextBaseline`, :class:`ExprRef`)
        For text marks, the vertical text baseline. One of ``"alphabetic"`` (default),
        ``"top"``, ``"middle"``, ``"bottom"``, ``"line-top"``, ``"line-bottom"``, or an
        expression reference that provides one of the valid values. The ``"line-top"`` and
        ``"line-bottom"`` values operate similarly to ``"top"`` and ``"bottom"``, but are
        calculated relative to the ``lineHeight`` rather than ``fontSize`` alone.

        For range marks, the vertical alignment of the marks. One of ``"top"``,
        ``"middle"``, ``"bottom"``.

        **Note:** Expression reference is *not* supported for range marks.
    blend : anyOf(:class:`Blend`, :class:`ExprRef`)

    color : anyOf(:class:`Color`, :class:`Gradient`, :class:`ExprRef`)
        Default color.

        **Default value:** :raw-html:`<span style="color: #4682b4;">&#9632;</span>`
        ``"#4682b4"``

        **Note:** - This property cannot be used in a `style config
        <https://vega.github.io/vega-lite/docs/mark.html#style-config>`__. - The ``fill``
        and ``stroke`` properties have higher precedence than ``color`` and will override
        ``color``.
    cornerRadius : anyOf(float, :class:`ExprRef`)

    cornerRadiusBottomLeft : anyOf(float, :class:`ExprRef`)

    cornerRadiusBottomRight : anyOf(float, :class:`ExprRef`)

    cornerRadiusTopLeft : anyOf(float, :class:`ExprRef`)

    cornerRadiusTopRight : anyOf(float, :class:`ExprRef`)

    cursor : anyOf(:class:`Cursor`, :class:`ExprRef`)

    description : anyOf(string, :class:`ExprRef`)

    dir : anyOf(:class:`TextDirection`, :class:`ExprRef`)

    dx : anyOf(float, :class:`ExprRef`)

    dy : anyOf(float, :class:`ExprRef`)

    ellipsis : anyOf(string, :class:`ExprRef`)

    endAngle : anyOf(float, :class:`ExprRef`)

    fill : anyOf(:class:`Color`, :class:`Gradient`, None, :class:`ExprRef`)
        Default fill color. This property has higher precedence than ``config.color``. Set
        to ``null`` to remove fill.

        **Default value:** (None)
    fillOpacity : anyOf(float, :class:`ExprRef`)

    filled : boolean
        Whether the mark's color should be used as fill color instead of stroke color.

        **Default value:** ``false`` for all ``point``, ``line``, and ``rule`` marks as well
        as ``geoshape`` marks for `graticule
        <https://vega.github.io/vega-lite/docs/data.html#graticule>`__ data sources;
        otherwise, ``true``.

        **Note:** This property cannot be used in a `style config
        <https://vega.github.io/vega-lite/docs/mark.html#style-config>`__.
    font : anyOf(string, :class:`ExprRef`)

    fontSize : anyOf(float, :class:`ExprRef`)

    fontStyle : anyOf(:class:`FontStyle`, :class:`ExprRef`)

    fontWeight : anyOf(:class:`FontWeight`, :class:`ExprRef`)

    height : anyOf(float, :class:`ExprRef`)

    href : anyOf(:class:`URI`, :class:`ExprRef`)

    innerRadius : anyOf(float, :class:`ExprRef`)
        The inner radius in pixels of arc marks. ``innerRadius`` is an alias for
        ``radius2``.
    interpolate : anyOf(:class:`Interpolate`, :class:`ExprRef`)

    invalid : enum('filter', None)
        Defines how Vega-Lite should handle marks for invalid values ( ``null`` and ``NaN``
        ). - If set to ``"filter"`` (default), all data items with null values will be
        skipped (for line, trail, and area marks) or filtered (for other marks). - If
        ``null``, all data items are included. In this case, invalid values will be
        interpreted as zeroes.
    limit : anyOf(float, :class:`ExprRef`)

    lineBreak : anyOf(string, :class:`ExprRef`)

    lineHeight : anyOf(float, :class:`ExprRef`)

    opacity : anyOf(float, :class:`ExprRef`)
        The overall opacity (value between [0,1]).

        **Default value:** ``0.7`` for non-aggregate plots with ``point``, ``tick``,
        ``circle``, or ``square`` marks or layered ``bar`` charts and ``1`` otherwise.
    order : anyOf(None, boolean)
        For line and trail marks, this ``order`` property can be set to ``null`` or
        ``false`` to make the lines use the original order in the data sources.
    orient : :class:`Orientation`
        The orientation of a non-stacked bar, tick, area, and line charts. The value is
        either horizontal (default) or vertical. - For bar, rule and tick, this determines
        whether the size of the bar and tick should be applied to x or y dimension. - For
        area, this property determines the orient property of the Vega output. - For line
        and trail marks, this property determines the sort order of the points in the line
        if ``config.sortLineBy`` is not specified. For stacked charts, this is always
        determined by the orientation of the stack; therefore explicitly specified value
        will be ignored.
    outerRadius : anyOf(float, :class:`ExprRef`)
        The outer radius in pixels of arc marks. ``outerRadius`` is an alias for ``radius``.
    padAngle : anyOf(float, :class:`ExprRef`)

    radius : anyOf(float, :class:`ExprRef`)
        For arc mark, the primary (outer) radius in pixels.

        For text marks, polar coordinate radial offset, in pixels, of the text from the
        origin determined by the ``x`` and ``y`` properties.
    radius2 : anyOf(float, :class:`ExprRef`)
        The secondary (inner) radius in pixels of arc marks.
    shape : anyOf(anyOf(:class:`SymbolShape`, string), :class:`ExprRef`)

    size : anyOf(float, :class:`ExprRef`)
        Default size for marks. - For ``point`` / ``circle`` / ``square``, this represents
        the pixel area of the marks. Note that this value sets the area of the symbol; the
        side lengths will increase with the square root of this value. - For ``bar``, this
        represents the band size of the bar, in pixels. - For ``text``, this represents the
        font size, in pixels.

        **Default value:** - ``30`` for point, circle, square marks; width/height's ``step``
        - ``2`` for bar marks with discrete dimensions; - ``5`` for bar marks with
        continuous dimensions; - ``11`` for text marks.
    smooth : anyOf(boolean, :class:`ExprRef`)

    startAngle : anyOf(float, :class:`ExprRef`)

    stroke : anyOf(:class:`Color`, :class:`Gradient`, None, :class:`ExprRef`)
        Default stroke color. This property has higher precedence than ``config.color``. Set
        to ``null`` to remove stroke.

        **Default value:** (None)
    strokeCap : anyOf(:class:`StrokeCap`, :class:`ExprRef`)

    strokeDash : anyOf(List(float), :class:`ExprRef`)

    strokeDashOffset : anyOf(float, :class:`ExprRef`)

    strokeJoin : anyOf(:class:`StrokeJoin`, :class:`ExprRef`)

    strokeMiterLimit : anyOf(float, :class:`ExprRef`)

    strokeOffset : anyOf(float, :class:`ExprRef`)

    strokeOpacity : anyOf(float, :class:`ExprRef`)

    strokeWidth : anyOf(float, :class:`ExprRef`)

    tension : anyOf(float, :class:`ExprRef`)

    text : anyOf(:class:`Text`, :class:`ExprRef`)

    theta : anyOf(float, :class:`ExprRef`)
        For arc marks, the arc length in radians if theta2 is not specified, otherwise the
        start arc angle. (A value of 0 indicates up or “north”, increasing values proceed
        clockwise.)

        For text marks, polar coordinate angle in radians.
    theta2 : anyOf(float, :class:`ExprRef`)
        The end angle of arc marks in radians. A value of 0 indicates up or “north”,
        increasing values proceed clockwise.
    timeUnitBand : float
        Default relative band size for a time unit. If set to ``1``, the bandwidth of the
        marks will be equal to the time unit band step. If set to ``0.5``, bandwidth of the
        marks will be half of the time unit band step.
    timeUnitBandPosition : float
        Default relative band position for a time unit. If set to ``0``, the marks will be
        positioned at the beginning of the time unit band step. If set to ``0.5``, the marks
        will be positioned in the middle of the time unit band step.
    tooltip : anyOf(float, string, boolean, :class:`TooltipContent`, :class:`ExprRef`, None)
        The tooltip text string to show upon mouse hover or an object defining which fields
        should the tooltip be derived from.


        * If ``tooltip`` is ``true`` or ``{"content": "encoding"}``, then all fields from
          ``encoding`` will be used. - If ``tooltip`` is ``{"content": "data"}``, then all
          fields that appear in the highlighted data point will be used. - If set to
          ``null`` or ``false``, then no tooltip will be used.

        See the `tooltip <https://vega.github.io/vega-lite/docs/tooltip.html>`__
        documentation for a detailed discussion about tooltip  in Vega-Lite.

        **Default value:** ``null``
    url : anyOf(:class:`URI`, :class:`ExprRef`)

    width : anyOf(float, :class:`ExprRef`)

    x : anyOf(float, string, :class:`ExprRef`)
        X coordinates of the marks, or width of horizontal ``"bar"`` and ``"area"`` without
        specified ``x2`` or ``width``.

        The ``value`` of this channel can be a number or a string ``"width"`` for the width
        of the plot.
    x2 : anyOf(float, string, :class:`ExprRef`)
        X2 coordinates for ranged ``"area"``, ``"bar"``, ``"rect"``, and  ``"rule"``.

        The ``value`` of this channel can be a number or a string ``"width"`` for the width
        of the plot.
    y : anyOf(float, string, :class:`ExprRef`)
        Y coordinates of the marks, or height of vertical ``"bar"`` and ``"area"`` without
        specified ``y2`` or ``height``.

        The ``value`` of this channel can be a number or a string ``"height"`` for the
        height of the plot.
    y2 : anyOf(float, string, :class:`ExprRef`)
        Y2 coordinates for ranged ``"area"``, ``"bar"``, ``"rect"``, and  ``"rule"``.

        The ``value`` of this channel can be a number or a string ``"height"`` for the
        height of the plot.
    """
    _schema = {'$ref': '#/definitions/MarkConfig'}

    def __init__(self, align=Undefined, angle=Undefined, aria=Undefined, ariaRole=Undefined,
                 ariaRoleDescription=Undefined, aspect=Undefined, baseline=Undefined, blend=Undefined,
                 color=Undefined, cornerRadius=Undefined, cornerRadiusBottomLeft=Undefined,
                 cornerRadiusBottomRight=Undefined, cornerRadiusTopLeft=Undefined,
                 cornerRadiusTopRight=Undefined, cursor=Undefined, description=Undefined, dir=Undefined,
                 dx=Undefined, dy=Undefined, ellipsis=Undefined, endAngle=Undefined, fill=Undefined,
                 fillOpacity=Undefined, filled=Undefined, font=Undefined, fontSize=Undefined,
                 fontStyle=Undefined, fontWeight=Undefined, height=Undefined, href=Undefined,
                 innerRadius=Undefined, interpolate=Undefined, invalid=Undefined, limit=Undefined,
                 lineBreak=Undefined, lineHeight=Undefined, opacity=Undefined, order=Undefined,
                 orient=Undefined, outerRadius=Undefined, padAngle=Undefined, radius=Undefined,
                 radius2=Undefined, shape=Undefined, size=Undefined, smooth=Undefined,
                 startAngle=Undefined, stroke=Undefined, strokeCap=Undefined, strokeDash=Undefined,
                 strokeDashOffset=Undefined, strokeJoin=Undefined, strokeMiterLimit=Undefined,
                 strokeOffset=Undefined, strokeOpacity=Undefined, strokeWidth=Undefined,
                 tension=Undefined, text=Undefined, theta=Undefined, theta2=Undefined,
                 timeUnitBand=Undefined, timeUnitBandPosition=Undefined, tooltip=Undefined,
                 url=Undefined, width=Undefined, x=Undefined, x2=Undefined, y=Undefined, y2=Undefined,
                 **kwds):
        super(MarkConfig, self).__init__(align=align, angle=angle, aria=aria, ariaRole=ariaRole,
                                         ariaRoleDescription=ariaRoleDescription, aspect=aspect,
                                         baseline=baseline, blend=blend, color=color,
                                         cornerRadius=cornerRadius,
                                         cornerRadiusBottomLeft=cornerRadiusBottomLeft,
                                         cornerRadiusBottomRight=cornerRadiusBottomRight,
                                         cornerRadiusTopLeft=cornerRadiusTopLeft,
                                         cornerRadiusTopRight=cornerRadiusTopRight, cursor=cursor,
                                         description=description, dir=dir, dx=dx, dy=dy,
                                         ellipsis=ellipsis, endAngle=endAngle, fill=fill,
                                         fillOpacity=fillOpacity, filled=filled, font=font,
                                         fontSize=fontSize, fontStyle=fontStyle, fontWeight=fontWeight,
                                         height=height, href=href, innerRadius=innerRadius,
                                         interpolate=interpolate, invalid=invalid, limit=limit,
                                         lineBreak=lineBreak, lineHeight=lineHeight, opacity=opacity,
                                         order=order, orient=orient, outerRadius=outerRadius,
                                         padAngle=padAngle, radius=radius, radius2=radius2, shape=shape,
                                         size=size, smooth=smooth, startAngle=startAngle, stroke=stroke,
                                         strokeCap=strokeCap, strokeDash=strokeDash,
                                         strokeDashOffset=strokeDashOffset, strokeJoin=strokeJoin,
                                         strokeMiterLimit=strokeMiterLimit, strokeOffset=strokeOffset,
                                         strokeOpacity=strokeOpacity, strokeWidth=strokeWidth,
                                         tension=tension, text=text, theta=theta, theta2=theta2,
                                         timeUnitBand=timeUnitBand,
                                         timeUnitBandPosition=timeUnitBandPosition, tooltip=tooltip,
                                         url=url, width=width, x=x, x2=x2, y=y, y2=y2, **kwds)


class MarkConfigExprOrSignalRef(VegaLiteSchema):
    """MarkConfigExprOrSignalRef schema wrapper

    Mapping(required=[])

    Attributes
    ----------

    align : anyOf(:class:`Align`, :class:`ExprOrSignalRef`)
        The horizontal alignment of the text or ranged marks (area, bar, image, rect, rule).
        One of ``"left"``, ``"right"``, ``"center"``.

        **Note:** Expression reference is *not* supported for range marks.
    angle : anyOf(float, :class:`ExprOrSignalRef`)

    aria : anyOf(boolean, :class:`ExprOrSignalRef`)

    ariaRole : anyOf(string, :class:`ExprOrSignalRef`)

    ariaRoleDescription : anyOf(string, :class:`ExprOrSignalRef`)

    aspect : anyOf(boolean, :class:`ExprOrSignalRef`)

    baseline : anyOf(:class:`TextBaseline`, :class:`ExprOrSignalRef`)
        For text marks, the vertical text baseline. One of ``"alphabetic"`` (default),
        ``"top"``, ``"middle"``, ``"bottom"``, ``"line-top"``, ``"line-bottom"``, or an
        expression reference that provides one of the valid values. The ``"line-top"`` and
        ``"line-bottom"`` values operate similarly to ``"top"`` and ``"bottom"``, but are
        calculated relative to the ``lineHeight`` rather than ``fontSize`` alone.

        For range marks, the vertical alignment of the marks. One of ``"top"``,
        ``"middle"``, ``"bottom"``.

        **Note:** Expression reference is *not* supported for range marks.
    blend : anyOf(:class:`Blend`, :class:`ExprOrSignalRef`)

    color : anyOf(:class:`Color`, :class:`Gradient`, :class:`ExprOrSignalRef`)
        Default color.

        **Default value:** :raw-html:`<span style="color: #4682b4;">&#9632;</span>`
        ``"#4682b4"``

        **Note:** - This property cannot be used in a `style config
        <https://vega.github.io/vega-lite/docs/mark.html#style-config>`__. - The ``fill``
        and ``stroke`` properties have higher precedence than ``color`` and will override
        ``color``.
    cornerRadius : anyOf(float, :class:`ExprOrSignalRef`)

    cornerRadiusBottomLeft : anyOf(float, :class:`ExprOrSignalRef`)

    cornerRadiusBottomRight : anyOf(float, :class:`ExprOrSignalRef`)

    cornerRadiusTopLeft : anyOf(float, :class:`ExprOrSignalRef`)

    cornerRadiusTopRight : anyOf(float, :class:`ExprOrSignalRef`)

    cursor : anyOf(:class:`Cursor`, :class:`ExprOrSignalRef`)

    description : anyOf(string, :class:`ExprOrSignalRef`)

    dir : anyOf(:class:`TextDirection`, :class:`ExprOrSignalRef`)

    dx : anyOf(float, :class:`ExprOrSignalRef`)

    dy : anyOf(float, :class:`ExprOrSignalRef`)

    ellipsis : anyOf(string, :class:`ExprOrSignalRef`)

    endAngle : anyOf(float, :class:`ExprOrSignalRef`)

    fill : anyOf(:class:`Color`, :class:`Gradient`, None, :class:`ExprOrSignalRef`)
        Default fill color. This property has higher precedence than ``config.color``. Set
        to ``null`` to remove fill.

        **Default value:** (None)
    fillOpacity : anyOf(float, :class:`ExprOrSignalRef`)

    filled : boolean
        Whether the mark's color should be used as fill color instead of stroke color.

        **Default value:** ``false`` for all ``point``, ``line``, and ``rule`` marks as well
        as ``geoshape`` marks for `graticule
        <https://vega.github.io/vega-lite/docs/data.html#graticule>`__ data sources;
        otherwise, ``true``.

        **Note:** This property cannot be used in a `style config
        <https://vega.github.io/vega-lite/docs/mark.html#style-config>`__.
    font : anyOf(string, :class:`ExprOrSignalRef`)

    fontSize : anyOf(float, :class:`ExprOrSignalRef`)

    fontStyle : anyOf(:class:`FontStyle`, :class:`ExprOrSignalRef`)

    fontWeight : anyOf(:class:`FontWeight`, :class:`ExprOrSignalRef`)

    height : anyOf(float, :class:`ExprOrSignalRef`)

    href : anyOf(:class:`URI`, :class:`ExprOrSignalRef`)

    innerRadius : anyOf(float, :class:`ExprOrSignalRef`)
        The inner radius in pixels of arc marks. ``innerRadius`` is an alias for
        ``radius2``.
    interpolate : anyOf(:class:`Interpolate`, :class:`ExprOrSignalRef`)

    invalid : enum('filter', None)
        Defines how Vega-Lite should handle marks for invalid values ( ``null`` and ``NaN``
        ). - If set to ``"filter"`` (default), all data items with null values will be
        skipped (for line, trail, and area marks) or filtered (for other marks). - If
        ``null``, all data items are included. In this case, invalid values will be
        interpreted as zeroes.
    limit : anyOf(float, :class:`ExprOrSignalRef`)

    lineBreak : anyOf(string, :class:`ExprOrSignalRef`)

    lineHeight : anyOf(float, :class:`ExprOrSignalRef`)

    opacity : anyOf(float, :class:`ExprOrSignalRef`)
        The overall opacity (value between [0,1]).

        **Default value:** ``0.7`` for non-aggregate plots with ``point``, ``tick``,
        ``circle``, or ``square`` marks or layered ``bar`` charts and ``1`` otherwise.
    order : anyOf(None, boolean)
        For line and trail marks, this ``order`` property can be set to ``null`` or
        ``false`` to make the lines use the original order in the data sources.
    orient : :class:`Orientation`
        The orientation of a non-stacked bar, tick, area, and line charts. The value is
        either horizontal (default) or vertical. - For bar, rule and tick, this determines
        whether the size of the bar and tick should be applied to x or y dimension. - For
        area, this property determines the orient property of the Vega output. - For line
        and trail marks, this property determines the sort order of the points in the line
        if ``config.sortLineBy`` is not specified. For stacked charts, this is always
        determined by the orientation of the stack; therefore explicitly specified value
        will be ignored.
    outerRadius : anyOf(float, :class:`ExprOrSignalRef`)
        The outer radius in pixels of arc marks. ``outerRadius`` is an alias for ``radius``.
    padAngle : anyOf(float, :class:`ExprOrSignalRef`)

    radius : anyOf(float, :class:`ExprOrSignalRef`)
        For arc mark, the primary (outer) radius in pixels.

        For text marks, polar coordinate radial offset, in pixels, of the text from the
        origin determined by the ``x`` and ``y`` properties.
    radius2 : anyOf(float, :class:`ExprOrSignalRef`)
        The secondary (inner) radius in pixels of arc marks.
    shape : anyOf(anyOf(:class:`SymbolShape`, string), :class:`ExprOrSignalRef`)

    size : anyOf(float, :class:`ExprOrSignalRef`)
        Default size for marks. - For ``point`` / ``circle`` / ``square``, this represents
        the pixel area of the marks. Note that this value sets the area of the symbol; the
        side lengths will increase with the square root of this value. - For ``bar``, this
        represents the band size of the bar, in pixels. - For ``text``, this represents the
        font size, in pixels.

        **Default value:** - ``30`` for point, circle, square marks; width/height's ``step``
        - ``2`` for bar marks with discrete dimensions; - ``5`` for bar marks with
        continuous dimensions; - ``11`` for text marks.
    smooth : anyOf(boolean, :class:`ExprOrSignalRef`)

    startAngle : anyOf(float, :class:`ExprOrSignalRef`)

    stroke : anyOf(:class:`Color`, :class:`Gradient`, None, :class:`ExprOrSignalRef`)
        Default stroke color. This property has higher precedence than ``config.color``. Set
        to ``null`` to remove stroke.

        **Default value:** (None)
    strokeCap : anyOf(:class:`StrokeCap`, :class:`ExprOrSignalRef`)

    strokeDash : anyOf(List(float), :class:`ExprOrSignalRef`)

    strokeDashOffset : anyOf(float, :class:`ExprOrSignalRef`)

    strokeJoin : anyOf(:class:`StrokeJoin`, :class:`ExprOrSignalRef`)

    strokeMiterLimit : anyOf(float, :class:`ExprOrSignalRef`)

    strokeOffset : anyOf(float, :class:`ExprOrSignalRef`)

    strokeOpacity : anyOf(float, :class:`ExprOrSignalRef`)

    strokeWidth : anyOf(float, :class:`ExprOrSignalRef`)

    tension : anyOf(float, :class:`ExprOrSignalRef`)

    text : anyOf(:class:`Text`, :class:`ExprOrSignalRef`)

    theta : anyOf(float, :class:`ExprOrSignalRef`)
        For arc marks, the arc length in radians if theta2 is not specified, otherwise the
        start arc angle. (A value of 0 indicates up or “north”, increasing values proceed
        clockwise.)

        For text marks, polar coordinate angle in radians.
    theta2 : anyOf(float, :class:`ExprOrSignalRef`)
        The end angle of arc marks in radians. A value of 0 indicates up or “north”,
        increasing values proceed clockwise.
    timeUnitBand : float
        Default relative band size for a time unit. If set to ``1``, the bandwidth of the
        marks will be equal to the time unit band step. If set to ``0.5``, bandwidth of the
        marks will be half of the time unit band step.
    timeUnitBandPosition : float
        Default relative band position for a time unit. If set to ``0``, the marks will be
        positioned at the beginning of the time unit band step. If set to ``0.5``, the marks
        will be positioned in the middle of the time unit band step.
    tooltip : anyOf(float, string, boolean, :class:`TooltipContent`, :class:`ExprOrSignalRef`,
    None)
        The tooltip text string to show upon mouse hover or an object defining which fields
        should the tooltip be derived from.


        * If ``tooltip`` is ``true`` or ``{"content": "encoding"}``, then all fields from
          ``encoding`` will be used. - If ``tooltip`` is ``{"content": "data"}``, then all
          fields that appear in the highlighted data point will be used. - If set to
          ``null`` or ``false``, then no tooltip will be used.

        See the `tooltip <https://vega.github.io/vega-lite/docs/tooltip.html>`__
        documentation for a detailed discussion about tooltip  in Vega-Lite.

        **Default value:** ``null``
    url : anyOf(:class:`URI`, :class:`ExprOrSignalRef`)

    width : anyOf(float, :class:`ExprOrSignalRef`)

    x : anyOf(float, string, :class:`ExprOrSignalRef`)
        X coordinates of the marks, or width of horizontal ``"bar"`` and ``"area"`` without
        specified ``x2`` or ``width``.

        The ``value`` of this channel can be a number or a string ``"width"`` for the width
        of the plot.
    x2 : anyOf(float, string, :class:`ExprOrSignalRef`)
        X2 coordinates for ranged ``"area"``, ``"bar"``, ``"rect"``, and  ``"rule"``.

        The ``value`` of this channel can be a number or a string ``"width"`` for the width
        of the plot.
    y : anyOf(float, string, :class:`ExprOrSignalRef`)
        Y coordinates of the marks, or height of vertical ``"bar"`` and ``"area"`` without
        specified ``y2`` or ``height``.

        The ``value`` of this channel can be a number or a string ``"height"`` for the
        height of the plot.
    y2 : anyOf(float, string, :class:`ExprOrSignalRef`)
        Y2 coordinates for ranged ``"area"``, ``"bar"``, ``"rect"``, and  ``"rule"``.

        The ``value`` of this channel can be a number or a string ``"height"`` for the
        height of the plot.
    """
    _schema = {'$ref': '#/definitions/MarkConfig<ExprOrSignalRef>'}

    def __init__(self, align=Undefined, angle=Undefined, aria=Undefined, ariaRole=Undefined,
                 ariaRoleDescription=Undefined, aspect=Undefined, baseline=Undefined, blend=Undefined,
                 color=Undefined, cornerRadius=Undefined, cornerRadiusBottomLeft=Undefined,
                 cornerRadiusBottomRight=Undefined, cornerRadiusTopLeft=Undefined,
                 cornerRadiusTopRight=Undefined, cursor=Undefined, description=Undefined, dir=Undefined,
                 dx=Undefined, dy=Undefined, ellipsis=Undefined, endAngle=Undefined, fill=Undefined,
                 fillOpacity=Undefined, filled=Undefined, font=Undefined, fontSize=Undefined,
                 fontStyle=Undefined, fontWeight=Undefined, height=Undefined, href=Undefined,
                 innerRadius=Undefined, interpolate=Undefined, invalid=Undefined, limit=Undefined,
                 lineBreak=Undefined, lineHeight=Undefined, opacity=Undefined, order=Undefined,
                 orient=Undefined, outerRadius=Undefined, padAngle=Undefined, radius=Undefined,
                 radius2=Undefined, shape=Undefined, size=Undefined, smooth=Undefined,
                 startAngle=Undefined, stroke=Undefined, strokeCap=Undefined, strokeDash=Undefined,
                 strokeDashOffset=Undefined, strokeJoin=Undefined, strokeMiterLimit=Undefined,
                 strokeOffset=Undefined, strokeOpacity=Undefined, strokeWidth=Undefined,
                 tension=Undefined, text=Undefined, theta=Undefined, theta2=Undefined,
                 timeUnitBand=Undefined, timeUnitBandPosition=Undefined, tooltip=Undefined,
                 url=Undefined, width=Undefined, x=Undefined, x2=Undefined, y=Undefined, y2=Undefined,
                 **kwds):
        super(MarkConfigExprOrSignalRef, self).__init__(align=align, angle=angle, aria=aria,
                                                        ariaRole=ariaRole,
                                                        ariaRoleDescription=ariaRoleDescription,
                                                        aspect=aspect, baseline=baseline, blend=blend,
                                                        color=color, cornerRadius=cornerRadius,
                                                        cornerRadiusBottomLeft=cornerRadiusBottomLeft,
                                                        cornerRadiusBottomRight=cornerRadiusBottomRight,
                                                        cornerRadiusTopLeft=cornerRadiusTopLeft,
                                                        cornerRadiusTopRight=cornerRadiusTopRight,
                                                        cursor=cursor, description=description, dir=dir,
                                                        dx=dx, dy=dy, ellipsis=ellipsis,
                                                        endAngle=endAngle, fill=fill,
                                                        fillOpacity=fillOpacity, filled=filled,
                                                        font=font, fontSize=fontSize,
                                                        fontStyle=fontStyle, fontWeight=fontWeight,
                                                        height=height, href=href,
                                                        innerRadius=innerRadius,
                                                        interpolate=interpolate, invalid=invalid,
                                                        limit=limit, lineBreak=lineBreak,
                                                        lineHeight=lineHeight, opacity=opacity,
                                                        order=order, orient=orient,
                                                        outerRadius=outerRadius, padAngle=padAngle,
                                                        radius=radius, radius2=radius2, shape=shape,
                                                        size=size, smooth=smooth, startAngle=startAngle,
                                                        stroke=stroke, strokeCap=strokeCap,
                                                        strokeDash=strokeDash,
                                                        strokeDashOffset=strokeDashOffset,
                                                        strokeJoin=strokeJoin,
                                                        strokeMiterLimit=strokeMiterLimit,
                                                        strokeOffset=strokeOffset,
                                                        strokeOpacity=strokeOpacity,
                                                        strokeWidth=strokeWidth, tension=tension,
                                                        text=text, theta=theta, theta2=theta2,
                                                        timeUnitBand=timeUnitBand,
                                                        timeUnitBandPosition=timeUnitBandPosition,
                                                        tooltip=tooltip, url=url, width=width, x=x,
                                                        x2=x2, y=y, y2=y2, **kwds)


class MarkDef(AnyMark):
    """MarkDef schema wrapper

    Mapping(required=[type])

    Attributes
    ----------

    type : :class:`Mark`
        The mark type. This could a primitive mark type (one of ``"bar"``, ``"circle"``,
        ``"square"``, ``"tick"``, ``"line"``, ``"area"``, ``"point"``, ``"geoshape"``,
        ``"rule"``, and ``"text"`` ) or a composite mark type ( ``"boxplot"``,
        ``"errorband"``, ``"errorbar"`` ).
    align : anyOf(:class:`Align`, :class:`ExprRef`)
        The horizontal alignment of the text or ranged marks (area, bar, image, rect, rule).
        One of ``"left"``, ``"right"``, ``"center"``.

        **Note:** Expression reference is *not* supported for range marks.
    angle : anyOf(float, :class:`ExprRef`)

    aria : anyOf(boolean, :class:`ExprRef`)

    ariaRole : anyOf(string, :class:`ExprRef`)

    ariaRoleDescription : anyOf(string, :class:`ExprRef`)

    aspect : anyOf(boolean, :class:`ExprRef`)

    bandSize : float
        The width of the ticks.

        **Default value:**  3/4 of step (width step for horizontal ticks and height step for
        vertical ticks).
    baseline : anyOf(:class:`TextBaseline`, :class:`ExprRef`)
        For text marks, the vertical text baseline. One of ``"alphabetic"`` (default),
        ``"top"``, ``"middle"``, ``"bottom"``, ``"line-top"``, ``"line-bottom"``, or an
        expression reference that provides one of the valid values. The ``"line-top"`` and
        ``"line-bottom"`` values operate similarly to ``"top"`` and ``"bottom"``, but are
        calculated relative to the ``lineHeight`` rather than ``fontSize`` alone.

        For range marks, the vertical alignment of the marks. One of ``"top"``,
        ``"middle"``, ``"bottom"``.

        **Note:** Expression reference is *not* supported for range marks.
    binSpacing : float
        Offset between bars for binned field. The ideal value for this is either 0
        (preferred by statisticians) or 1 (Vega-Lite default, D3 example style).

        **Default value:** ``1``
    blend : anyOf(:class:`Blend`, :class:`ExprRef`)

    clip : boolean
        Whether a mark be clipped to the enclosing group’s width and height.
    color : anyOf(:class:`Color`, :class:`Gradient`, :class:`ExprRef`)
        Default color.

        **Default value:** :raw-html:`<span style="color: #4682b4;">&#9632;</span>`
        ``"#4682b4"``

        **Note:** - This property cannot be used in a `style config
        <https://vega.github.io/vega-lite/docs/mark.html#style-config>`__. - The ``fill``
        and ``stroke`` properties have higher precedence than ``color`` and will override
        ``color``.
    continuousBandSize : float
        The default size of the bars on continuous scales.

        **Default value:** ``5``
    cornerRadius : anyOf(float, :class:`ExprRef`)

    cornerRadiusBottomLeft : anyOf(float, :class:`ExprRef`)

    cornerRadiusBottomRight : anyOf(float, :class:`ExprRef`)

    cornerRadiusEnd : anyOf(float, :class:`ExprRef`)
        * For vertical bars, top-left and top-right corner radius. - For horizontal bars,
          top-right and bottom-right corner radius.
    cornerRadiusTopLeft : anyOf(float, :class:`ExprRef`)

    cornerRadiusTopRight : anyOf(float, :class:`ExprRef`)

    cursor : anyOf(:class:`Cursor`, :class:`ExprRef`)

    description : anyOf(string, :class:`ExprRef`)

    dir : anyOf(:class:`TextDirection`, :class:`ExprRef`)

    discreteBandSize : float
        The default size of the bars with discrete dimensions. If unspecified, the default
        size is  ``step-2``, which provides 2 pixel offset between bars.
    dx : anyOf(float, :class:`ExprRef`)

    dy : anyOf(float, :class:`ExprRef`)

    ellipsis : anyOf(string, :class:`ExprRef`)

    fill : anyOf(:class:`Color`, :class:`Gradient`, None, :class:`ExprRef`)
        Default fill color. This property has higher precedence than ``config.color``. Set
        to ``null`` to remove fill.

        **Default value:** (None)
    fillOpacity : anyOf(float, :class:`ExprRef`)

    filled : boolean
        Whether the mark's color should be used as fill color instead of stroke color.

        **Default value:** ``false`` for all ``point``, ``line``, and ``rule`` marks as well
        as ``geoshape`` marks for `graticule
        <https://vega.github.io/vega-lite/docs/data.html#graticule>`__ data sources;
        otherwise, ``true``.

        **Note:** This property cannot be used in a `style config
        <https://vega.github.io/vega-lite/docs/mark.html#style-config>`__.
    font : anyOf(string, :class:`ExprRef`)

    fontSize : anyOf(float, :class:`ExprRef`)

    fontStyle : anyOf(:class:`FontStyle`, :class:`ExprRef`)

    fontWeight : anyOf(:class:`FontWeight`, :class:`ExprRef`)

    height : anyOf(float, :class:`ExprRef`)

    href : anyOf(:class:`URI`, :class:`ExprRef`)

    innerRadius : anyOf(float, :class:`ExprRef`)
        The inner radius in pixels of arc marks. ``innerRadius`` is an alias for
        ``radius2``.
    interpolate : anyOf(:class:`Interpolate`, :class:`ExprRef`)

    invalid : enum('filter', None)
        Defines how Vega-Lite should handle marks for invalid values ( ``null`` and ``NaN``
        ). - If set to ``"filter"`` (default), all data items with null values will be
        skipped (for line, trail, and area marks) or filtered (for other marks). - If
        ``null``, all data items are included. In this case, invalid values will be
        interpreted as zeroes.
    limit : anyOf(float, :class:`ExprRef`)

    line : anyOf(boolean, :class:`OverlayMarkDef`)
        A flag for overlaying line on top of area marks, or an object defining the
        properties of the overlayed lines.


        If this value is an empty object ( ``{}`` ) or ``true``, lines with default
        properties will be used.

        If this value is ``false``, no lines would be automatically added to area marks.

        **Default value:** ``false``.
    lineBreak : anyOf(string, :class:`ExprRef`)

    lineHeight : anyOf(float, :class:`ExprRef`)

    opacity : anyOf(float, :class:`ExprRef`)
        The overall opacity (value between [0,1]).

        **Default value:** ``0.7`` for non-aggregate plots with ``point``, ``tick``,
        ``circle``, or ``square`` marks or layered ``bar`` charts and ``1`` otherwise.
    order : anyOf(None, boolean)
        For line and trail marks, this ``order`` property can be set to ``null`` or
        ``false`` to make the lines use the original order in the data sources.
    orient : :class:`Orientation`
        The orientation of a non-stacked bar, tick, area, and line charts. The value is
        either horizontal (default) or vertical. - For bar, rule and tick, this determines
        whether the size of the bar and tick should be applied to x or y dimension. - For
        area, this property determines the orient property of the Vega output. - For line
        and trail marks, this property determines the sort order of the points in the line
        if ``config.sortLineBy`` is not specified. For stacked charts, this is always
        determined by the orientation of the stack; therefore explicitly specified value
        will be ignored.
    outerRadius : anyOf(float, :class:`ExprRef`)
        The outer radius in pixels of arc marks. ``outerRadius`` is an alias for ``radius``.
    padAngle : anyOf(float, :class:`ExprRef`)

    point : anyOf(boolean, :class:`OverlayMarkDef`, string)
        A flag for overlaying points on top of line or area marks, or an object defining the
        properties of the overlayed points.


        If this property is ``"transparent"``, transparent points will be used (for
        enhancing tooltips and selections).

        If this property is an empty object ( ``{}`` ) or ``true``, filled points with
        default properties will be used.

        If this property is ``false``, no points would be automatically added to line or
        area marks.

        **Default value:** ``false``.
    radius : anyOf(float, :class:`ExprRef`)
        For arc mark, the primary (outer) radius in pixels.

        For text marks, polar coordinate radial offset, in pixels, of the text from the
        origin determined by the ``x`` and ``y`` properties.
    radius2 : anyOf(float, :class:`ExprRef`)
        The secondary (inner) radius in pixels of arc marks.
    radius2Offset : anyOf(float, :class:`ExprRef`)
        Offset for radius2.
    radiusOffset : anyOf(float, :class:`ExprRef`)
        Offset for radius.
    shape : anyOf(anyOf(:class:`SymbolShape`, string), :class:`ExprRef`)

    size : anyOf(float, :class:`ExprRef`)
        Default size for marks. - For ``point`` / ``circle`` / ``square``, this represents
        the pixel area of the marks. Note that this value sets the area of the symbol; the
        side lengths will increase with the square root of this value. - For ``bar``, this
        represents the band size of the bar, in pixels. - For ``text``, this represents the
        font size, in pixels.

        **Default value:** - ``30`` for point, circle, square marks; width/height's ``step``
        - ``2`` for bar marks with discrete dimensions; - ``5`` for bar marks with
        continuous dimensions; - ``11`` for text marks.
    smooth : anyOf(boolean, :class:`ExprRef`)

    stroke : anyOf(:class:`Color`, :class:`Gradient`, None, :class:`ExprRef`)
        Default stroke color. This property has higher precedence than ``config.color``. Set
        to ``null`` to remove stroke.

        **Default value:** (None)
    strokeCap : anyOf(:class:`StrokeCap`, :class:`ExprRef`)

    strokeDash : anyOf(List(float), :class:`ExprRef`)

    strokeDashOffset : anyOf(float, :class:`ExprRef`)

    strokeJoin : anyOf(:class:`StrokeJoin`, :class:`ExprRef`)

    strokeMiterLimit : anyOf(float, :class:`ExprRef`)

    strokeOffset : anyOf(float, :class:`ExprRef`)

    strokeOpacity : anyOf(float, :class:`ExprRef`)

    strokeWidth : anyOf(float, :class:`ExprRef`)

    style : anyOf(string, List(string))
        A string or array of strings indicating the name of custom styles to apply to the
        mark. A style is a named collection of mark property defaults defined within the
        `style configuration
        <https://vega.github.io/vega-lite/docs/mark.html#style-config>`__. If style is an
        array, later styles will override earlier styles. Any `mark properties
        <https://vega.github.io/vega-lite/docs/encoding.html#mark-prop>`__ explicitly
        defined within the ``encoding`` will override a style default.

        **Default value:** The mark's name. For example, a bar mark will have style
        ``"bar"`` by default. **Note:** Any specified style will augment the default style.
        For example, a bar mark with ``"style": "foo"`` will receive from
        ``config.style.bar`` and ``config.style.foo`` (the specified style ``"foo"`` has
        higher precedence).
    tension : anyOf(float, :class:`ExprRef`)

    text : anyOf(:class:`Text`, :class:`ExprRef`)

    theta : anyOf(float, :class:`ExprRef`)
        For arc marks, the arc length in radians if theta2 is not specified, otherwise the
        start arc angle. (A value of 0 indicates up or “north”, increasing values proceed
        clockwise.)

        For text marks, polar coordinate angle in radians.
    theta2 : anyOf(float, :class:`ExprRef`)
        The end angle of arc marks in radians. A value of 0 indicates up or “north”,
        increasing values proceed clockwise.
    theta2Offset : anyOf(float, :class:`ExprRef`)
        Offset for theta2.
    thetaOffset : anyOf(float, :class:`ExprRef`)
        Offset for theta.
    thickness : float
        Thickness of the tick mark.

        **Default value:**  ``1``
    timeUnitBand : float
        Default relative band size for a time unit. If set to ``1``, the bandwidth of the
        marks will be equal to the time unit band step. If set to ``0.5``, bandwidth of the
        marks will be half of the time unit band step.
    timeUnitBandPosition : float
        Default relative band position for a time unit. If set to ``0``, the marks will be
        positioned at the beginning of the time unit band step. If set to ``0.5``, the marks
        will be positioned in the middle of the time unit band step.
    tooltip : anyOf(float, string, boolean, :class:`TooltipContent`, :class:`ExprRef`, None)
        The tooltip text string to show upon mouse hover or an object defining which fields
        should the tooltip be derived from.


        * If ``tooltip`` is ``true`` or ``{"content": "encoding"}``, then all fields from
          ``encoding`` will be used. - If ``tooltip`` is ``{"content": "data"}``, then all
          fields that appear in the highlighted data point will be used. - If set to
          ``null`` or ``false``, then no tooltip will be used.

        See the `tooltip <https://vega.github.io/vega-lite/docs/tooltip.html>`__
        documentation for a detailed discussion about tooltip  in Vega-Lite.

        **Default value:** ``null``
    url : anyOf(:class:`URI`, :class:`ExprRef`)

    width : anyOf(float, :class:`ExprRef`)

    x : anyOf(float, string, :class:`ExprRef`)
        X coordinates of the marks, or width of horizontal ``"bar"`` and ``"area"`` without
        specified ``x2`` or ``width``.

        The ``value`` of this channel can be a number or a string ``"width"`` for the width
        of the plot.
    x2 : anyOf(float, string, :class:`ExprRef`)
        X2 coordinates for ranged ``"area"``, ``"bar"``, ``"rect"``, and  ``"rule"``.

        The ``value`` of this channel can be a number or a string ``"width"`` for the width
        of the plot.
    x2Offset : anyOf(float, :class:`ExprRef`)
        Offset for x2-position.
    xOffset : anyOf(float, :class:`ExprRef`)
        Offset for x-position.
    y : anyOf(float, string, :class:`ExprRef`)
        Y coordinates of the marks, or height of vertical ``"bar"`` and ``"area"`` without
        specified ``y2`` or ``height``.

        The ``value`` of this channel can be a number or a string ``"height"`` for the
        height of the plot.
    y2 : anyOf(float, string, :class:`ExprRef`)
        Y2 coordinates for ranged ``"area"``, ``"bar"``, ``"rect"``, and  ``"rule"``.

        The ``value`` of this channel can be a number or a string ``"height"`` for the
        height of the plot.
    y2Offset : anyOf(float, :class:`ExprRef`)
        Offset for y2-position.
    yOffset : anyOf(float, :class:`ExprRef`)
        Offset for y-position.
    """
    _schema = {'$ref': '#/definitions/MarkDef'}

    def __init__(self, type=Undefined, align=Undefined, angle=Undefined, aria=Undefined,
                 ariaRole=Undefined, ariaRoleDescription=Undefined, aspect=Undefined,
                 bandSize=Undefined, baseline=Undefined, binSpacing=Undefined, blend=Undefined,
                 clip=Undefined, color=Undefined, continuousBandSize=Undefined, cornerRadius=Undefined,
                 cornerRadiusBottomLeft=Undefined, cornerRadiusBottomRight=Undefined,
                 cornerRadiusEnd=Undefined, cornerRadiusTopLeft=Undefined,
                 cornerRadiusTopRight=Undefined, cursor=Undefined, description=Undefined, dir=Undefined,
                 discreteBandSize=Undefined, dx=Undefined, dy=Undefined, ellipsis=Undefined,
                 fill=Undefined, fillOpacity=Undefined, filled=Undefined, font=Undefined,
                 fontSize=Undefined, fontStyle=Undefined, fontWeight=Undefined, height=Undefined,
                 href=Undefined, innerRadius=Undefined, interpolate=Undefined, invalid=Undefined,
                 limit=Undefined, line=Undefined, lineBreak=Undefined, lineHeight=Undefined,
                 opacity=Undefined, order=Undefined, orient=Undefined, outerRadius=Undefined,
                 padAngle=Undefined, point=Undefined, radius=Undefined, radius2=Undefined,
                 radius2Offset=Undefined, radiusOffset=Undefined, shape=Undefined, size=Undefined,
                 smooth=Undefined, stroke=Undefined, strokeCap=Undefined, strokeDash=Undefined,
                 strokeDashOffset=Undefined, strokeJoin=Undefined, strokeMiterLimit=Undefined,
                 strokeOffset=Undefined, strokeOpacity=Undefined, strokeWidth=Undefined,
                 style=Undefined, tension=Undefined, text=Undefined, theta=Undefined, theta2=Undefined,
                 theta2Offset=Undefined, thetaOffset=Undefined, thickness=Undefined,
                 timeUnitBand=Undefined, timeUnitBandPosition=Undefined, tooltip=Undefined,
                 url=Undefined, width=Undefined, x=Undefined, x2=Undefined, x2Offset=Undefined,
                 xOffset=Undefined, y=Undefined, y2=Undefined, y2Offset=Undefined, yOffset=Undefined,
                 **kwds):
        super(MarkDef, self).__init__(type=type, align=align, angle=angle, aria=aria, ariaRole=ariaRole,
                                      ariaRoleDescription=ariaRoleDescription, aspect=aspect,
                                      bandSize=bandSize, baseline=baseline, binSpacing=binSpacing,
                                      blend=blend, clip=clip, color=color,
                                      continuousBandSize=continuousBandSize, cornerRadius=cornerRadius,
                                      cornerRadiusBottomLeft=cornerRadiusBottomLeft,
                                      cornerRadiusBottomRight=cornerRadiusBottomRight,
                                      cornerRadiusEnd=cornerRadiusEnd,
                                      cornerRadiusTopLeft=cornerRadiusTopLeft,
                                      cornerRadiusTopRight=cornerRadiusTopRight, cursor=cursor,
                                      description=description, dir=dir,
                                      discreteBandSize=discreteBandSize, dx=dx, dy=dy,
                                      ellipsis=ellipsis, fill=fill, fillOpacity=fillOpacity,
                                      filled=filled, font=font, fontSize=fontSize, fontStyle=fontStyle,
                                      fontWeight=fontWeight, height=height, href=href,
                                      innerRadius=innerRadius, interpolate=interpolate, invalid=invalid,
                                      limit=limit, line=line, lineBreak=lineBreak,
                                      lineHeight=lineHeight, opacity=opacity, order=order,
                                      orient=orient, outerRadius=outerRadius, padAngle=padAngle,
                                      point=point, radius=radius, radius2=radius2,
                                      radius2Offset=radius2Offset, radiusOffset=radiusOffset,
                                      shape=shape, size=size, smooth=smooth, stroke=stroke,
                                      strokeCap=strokeCap, strokeDash=strokeDash,
                                      strokeDashOffset=strokeDashOffset, strokeJoin=strokeJoin,
                                      strokeMiterLimit=strokeMiterLimit, strokeOffset=strokeOffset,
                                      strokeOpacity=strokeOpacity, strokeWidth=strokeWidth, style=style,
                                      tension=tension, text=text, theta=theta, theta2=theta2,
                                      theta2Offset=theta2Offset, thetaOffset=thetaOffset,
                                      thickness=thickness, timeUnitBand=timeUnitBand,
                                      timeUnitBandPosition=timeUnitBandPosition, tooltip=tooltip,
                                      url=url, width=width, x=x, x2=x2, x2Offset=x2Offset,
                                      xOffset=xOffset, y=y, y2=y2, y2Offset=y2Offset, yOffset=yOffset,
                                      **kwds)


class MarkPropDefGradientstringnull(VegaLiteSchema):
    """MarkPropDefGradientstringnull schema wrapper

    anyOf(:class:`FieldOrDatumDefWithConditionMarkPropFieldDefGradientstringnull`,
    :class:`FieldOrDatumDefWithConditionDatumDefGradientstringnull`,
    :class:`ValueDefWithConditionMarkPropFieldOrDatumDefGradientstringnull`)
    """
    _schema = {'$ref': '#/definitions/MarkPropDef<(Gradient|string|null)>'}

    def __init__(self, *args, **kwds):
        super(MarkPropDefGradientstringnull, self).__init__(*args, **kwds)


class FieldOrDatumDefWithConditionDatumDefGradientstringnull(ColorDef, MarkPropDefGradientstringnull):
    """FieldOrDatumDefWithConditionDatumDefGradientstringnull schema wrapper

    Mapping(required=[])
    A FieldDef with Condition :raw-html:`<ValueDef>` {    condition: {value: ...},    field:
    ...,    ... }

    Attributes
    ----------

    band : float
        For rect-based marks ( ``rect``, ``bar``, and ``image`` ), mark size relative to
        bandwidth of `band scales
        <https://vega.github.io/vega-lite/docs/scale.html#band>`__, bins or time units. If
        set to ``1``, the mark size is set to the bandwidth, the bin interval, or the time
        unit interval. If set to ``0.5``, the mark size is half of the bandwidth or the time
        unit interval.

        For other marks, relative position on a band of a stacked, binned, time unit or band
        scale. If set to ``0``, the marks will be positioned at the beginning of the band.
        If set to ``0.5``, the marks will be positioned in the middle of the band.
    condition : anyOf(:class:`ConditionalValueDefGradientstringnullExprRef`,
    List(:class:`ConditionalValueDefGradientstringnullExprRef`))
        One or more value definition(s) with `a selection or a test predicate
        <https://vega.github.io/vega-lite/docs/condition.html>`__.

        **Note:** A field definition's ``condition`` property can only contain `conditional
        value definitions <https://vega.github.io/vega-lite/docs/condition.html#value>`__
        since Vega-Lite only allows at most one encoded field per encoding channel.
    datum : anyOf(:class:`PrimitiveValue`, :class:`DateTime`, :class:`ExprRef`,
    :class:`RepeatRef`)
        A constant value in data domain.
    type : :class:`Type`
        The type of measurement ( ``"quantitative"``, ``"temporal"``, ``"ordinal"``, or
        ``"nominal"`` ) for the encoded field or constant value ( ``datum`` ). It can also
        be a ``"geojson"`` type for encoding `'geoshape'
        <https://vega.github.io/vega-lite/docs/geoshape.html>`__.

        Vega-Lite automatically infers data types in many cases as discussed below. However,
        type is required for a field if: (1) the field is not nominal and the field encoding
        has no specified ``aggregate`` (except ``argmin`` and ``argmax`` ), ``bin``, scale
        type, custom ``sort`` order, nor ``timeUnit`` or (2) if you wish to use an ordinal
        scale for a field with ``bin`` or ``timeUnit``.

        **Default value:**

        1) For a data ``field``, ``"nominal"`` is the default data type unless the field
        encoding has ``aggregate``, ``channel``, ``bin``, scale type, ``sort``, or
        ``timeUnit`` that satisfies the following criteria: - ``"quantitative"`` is the
        default type if (1) the encoded field contains ``bin`` or ``aggregate`` except
        ``"argmin"`` and ``"argmax"``, (2) the encoding channel is ``latitude`` or
        ``longitude`` channel or (3) if the specified scale type is `a quantitative scale
        <https://vega.github.io/vega-lite/docs/scale.html#type>`__. - ``"temporal"`` is the
        default type if (1) the encoded field contains ``timeUnit`` or (2) the specified
        scale type is a time or utc scale - ``ordinal""`` is the default type if (1) the
        encoded field contains a `custom sort order
        <https://vega.github.io/vega-lite/docs/sort.html#specifying-custom-sort-order>`__,
        (2) the specified scale type is an ordinal/point/band scale, or (3) the encoding
        channel is ``order``.

        2) For a constant value in data domain ( ``datum`` ): - ``"quantitative"`` if the
        datum is a number - ``"nominal"`` if the datum is a string - ``"temporal"`` if the
        datum is `a date time object
        <https://vega.github.io/vega-lite/docs/datetime.html>`__

        **Note:** - Data ``type`` describes the semantics of the data rather than the
        primitive data types (number, string, etc.). The same primitive data type can have
        different types of measurement. For example, numeric data can represent
        quantitative, ordinal, or nominal data. - Data values for a temporal field can be
        either a date-time string (e.g., ``"2015-03-07 12:32:17"``, ``"17:01"``,
        ``"2015-03-16"``. ``"2015"`` ) or a timestamp number (e.g., ``1552199579097`` ). -
        When using with `bin <https://vega.github.io/vega-lite/docs/bin.html>`__, the
        ``type`` property can be either ``"quantitative"`` (for using a linear bin scale) or
        `"ordinal" (for using an ordinal bin scale)
        <https://vega.github.io/vega-lite/docs/type.html#cast-bin>`__. - When using with
        `timeUnit <https://vega.github.io/vega-lite/docs/timeunit.html>`__, the ``type``
        property can be either ``"temporal"`` (default, for using a temporal scale) or
        `"ordinal" (for using an ordinal scale)
        <https://vega.github.io/vega-lite/docs/type.html#cast-bin>`__. - When using with
        `aggregate <https://vega.github.io/vega-lite/docs/aggregate.html>`__, the ``type``
        property refers to the post-aggregation data type. For example, we can calculate
        count ``distinct`` of a categorical field ``"cat"`` using ``{"aggregate":
        "distinct", "field": "cat"}``. The ``"type"`` of the aggregate output is
        ``"quantitative"``. - Secondary channels (e.g., ``x2``, ``y2``, ``xError``,
        ``yError`` ) do not have ``type`` as they must have exactly the same type as their
        primary channels (e.g., ``x``, ``y`` ).

        **See also:** `type <https://vega.github.io/vega-lite/docs/type.html>`__
        documentation.
    """
    _schema = {'$ref': '#/definitions/FieldOrDatumDefWithCondition<DatumDef,(Gradient|string|null)>'}

    def __init__(self, band=Undefined, condition=Undefined, datum=Undefined, type=Undefined, **kwds):
        super(FieldOrDatumDefWithConditionDatumDefGradientstringnull, self).__init__(band=band,
                                                                                     condition=condition,
                                                                                     datum=datum,
                                                                                     type=type, **kwds)


class FieldOrDatumDefWithConditionMarkPropFieldDefGradientstringnull(ColorDef, MarkPropDefGradientstringnull):
    """FieldOrDatumDefWithConditionMarkPropFieldDefGradientstringnull schema wrapper

    Mapping(required=[])
    A FieldDef with Condition :raw-html:`<ValueDef>` {    condition: {value: ...},    field:
    ...,    ... }

    Attributes
    ----------

    aggregate : :class:`Aggregate`
        Aggregation function for the field (e.g., ``"mean"``, ``"sum"``, ``"median"``,
        ``"min"``, ``"max"``, ``"count"`` ).

        **Default value:** ``undefined`` (None)

        **See also:** `aggregate <https://vega.github.io/vega-lite/docs/aggregate.html>`__
        documentation.
    band : float
        For rect-based marks ( ``rect``, ``bar``, and ``image`` ), mark size relative to
        bandwidth of `band scales
        <https://vega.github.io/vega-lite/docs/scale.html#band>`__, bins or time units. If
        set to ``1``, the mark size is set to the bandwidth, the bin interval, or the time
        unit interval. If set to ``0.5``, the mark size is half of the bandwidth or the time
        unit interval.

        For other marks, relative position on a band of a stacked, binned, time unit or band
        scale. If set to ``0``, the marks will be positioned at the beginning of the band.
        If set to ``0.5``, the marks will be positioned in the middle of the band.
    bin : anyOf(boolean, :class:`BinParams`, None)
        A flag for binning a ``quantitative`` field, `an object defining binning parameters
        <https://vega.github.io/vega-lite/docs/bin.html#params>`__, or indicating that the
        data for ``x`` or ``y`` channel are binned before they are imported into Vega-Lite (
        ``"binned"`` ).


        If ``true``, default `binning parameters
        <https://vega.github.io/vega-lite/docs/bin.html>`__ will be applied.

        If ``"binned"``, this indicates that the data for the ``x`` (or ``y`` ) channel are
        already binned. You can map the bin-start field to ``x`` (or ``y`` ) and the bin-end
        field to ``x2`` (or ``y2`` ). The scale and axis will be formatted similar to
        binning in Vega-Lite.  To adjust the axis ticks based on the bin step, you can also
        set the axis's `tickMinStep
        <https://vega.github.io/vega-lite/docs/axis.html#ticks>`__ property.

        **Default value:** ``false``

        **See also:** `bin <https://vega.github.io/vega-lite/docs/bin.html>`__
        documentation.
    condition : anyOf(:class:`ConditionalValueDefGradientstringnullExprRef`,
    List(:class:`ConditionalValueDefGradientstringnullExprRef`))
        One or more value definition(s) with `a selection or a test predicate
        <https://vega.github.io/vega-lite/docs/condition.html>`__.

        **Note:** A field definition's ``condition`` property can only contain `conditional
        value definitions <https://vega.github.io/vega-lite/docs/condition.html#value>`__
        since Vega-Lite only allows at most one encoded field per encoding channel.
    field : :class:`Field`
        **Required.** A string defining the name of the field from which to pull a data
        value or an object defining iterated values from the `repeat
        <https://vega.github.io/vega-lite/docs/repeat.html>`__ operator.

        **See also:** `field <https://vega.github.io/vega-lite/docs/field.html>`__
        documentation.

        **Notes:** 1)  Dots ( ``.`` ) and brackets ( ``[`` and ``]`` ) can be used to access
        nested objects (e.g., ``"field": "foo.bar"`` and ``"field": "foo['bar']"`` ). If
        field names contain dots or brackets but are not nested, you can use ``\\`` to
        escape dots and brackets (e.g., ``"a\\.b"`` and ``"a\\[0\\]"`` ). See more details
        about escaping in the `field documentation
        <https://vega.github.io/vega-lite/docs/field.html>`__. 2) ``field`` is not required
        if ``aggregate`` is ``count``.
    legend : anyOf(:class:`Legend`, None)
        An object defining properties of the legend. If ``null``, the legend for the
        encoding channel will be removed.

        **Default value:** If undefined, default `legend properties
        <https://vega.github.io/vega-lite/docs/legend.html>`__ are applied.

        **See also:** `legend <https://vega.github.io/vega-lite/docs/legend.html>`__
        documentation.
    scale : anyOf(:class:`Scale`, None)
        An object defining properties of the channel's scale, which is the function that
        transforms values in the data domain (numbers, dates, strings, etc) to visual values
        (pixels, colors, sizes) of the encoding channels.

        If ``null``, the scale will be `disabled and the data value will be directly encoded
        <https://vega.github.io/vega-lite/docs/scale.html#disable>`__.

        **Default value:** If undefined, default `scale properties
        <https://vega.github.io/vega-lite/docs/scale.html>`__ are applied.

        **See also:** `scale <https://vega.github.io/vega-lite/docs/scale.html>`__
        documentation.
    sort : :class:`Sort`
        Sort order for the encoded field.

        For continuous fields (quantitative or temporal), ``sort`` can be either
        ``"ascending"`` or ``"descending"``.

        For discrete fields, ``sort`` can be one of the following: - ``"ascending"`` or
        ``"descending"`` -- for sorting by the values' natural order in JavaScript. - `A
        string indicating an encoding channel name to sort by
        <https://vega.github.io/vega-lite/docs/sort.html#sort-by-encoding>`__ (e.g., ``"x"``
        or ``"y"`` ) with an optional minus prefix for descending sort (e.g., ``"-x"`` to
        sort by x-field, descending). This channel string is short-form of `a
        sort-by-encoding definition
        <https://vega.github.io/vega-lite/docs/sort.html#sort-by-encoding>`__. For example,
        ``"sort": "-x"`` is equivalent to ``"sort": {"encoding": "x", "order":
        "descending"}``. - `A sort field definition
        <https://vega.github.io/vega-lite/docs/sort.html#sort-field>`__ for sorting by
        another field. - `An array specifying the field values in preferred order
        <https://vega.github.io/vega-lite/docs/sort.html#sort-array>`__. In this case, the
        sort order will obey the values in the array, followed by any unspecified values in
        their original order. For discrete time field, values in the sort array can be
        `date-time definition objects <types#datetime>`__. In addition, for time units
        ``"month"`` and ``"day"``, the values can be the month or day names (case
        insensitive) or their 3-letter initials (e.g., ``"Mon"``, ``"Tue"`` ). - ``null``
        indicating no sort.

        **Default value:** ``"ascending"``

        **Note:** ``null`` and sorting by another channel is not supported for ``row`` and
        ``column``.

        **See also:** `sort <https://vega.github.io/vega-lite/docs/sort.html>`__
        documentation.
    timeUnit : anyOf(:class:`TimeUnit`, :class:`TimeUnitParams`)
        Time unit (e.g., ``year``, ``yearmonth``, ``month``, ``hours`` ) for a temporal
        field. or `a temporal field that gets casted as ordinal
        <https://vega.github.io/vega-lite/docs/type.html#cast>`__.

        **Default value:** ``undefined`` (None)

        **See also:** `timeUnit <https://vega.github.io/vega-lite/docs/timeunit.html>`__
        documentation.
    title : anyOf(:class:`Text`, None)
        A title for the field. If ``null``, the title will be removed.

        **Default value:**  derived from the field's name and transformation function (
        ``aggregate``, ``bin`` and ``timeUnit`` ). If the field has an aggregate function,
        the function is displayed as part of the title (e.g., ``"Sum of Profit"`` ). If the
        field is binned or has a time unit applied, the applied function is shown in
        parentheses (e.g., ``"Profit (binned)"``, ``"Transaction Date (year-month)"`` ).
        Otherwise, the title is simply the field name.

        **Notes** :

        1) You can customize the default field title format by providing the `fieldTitle
        <https://vega.github.io/vega-lite/docs/config.html#top-level-config>`__ property in
        the `config <https://vega.github.io/vega-lite/docs/config.html>`__ or `fieldTitle
        function via the compile function's options
        <https://vega.github.io/vega-lite/docs/compile.html#field-title>`__.

        2) If both field definition's ``title`` and axis, header, or legend ``title`` are
        defined, axis/header/legend title will be used.
    type : :class:`StandardType`
        The type of measurement ( ``"quantitative"``, ``"temporal"``, ``"ordinal"``, or
        ``"nominal"`` ) for the encoded field or constant value ( ``datum`` ). It can also
        be a ``"geojson"`` type for encoding `'geoshape'
        <https://vega.github.io/vega-lite/docs/geoshape.html>`__.

        Vega-Lite automatically infers data types in many cases as discussed below. However,
        type is required for a field if: (1) the field is not nominal and the field encoding
        has no specified ``aggregate`` (except ``argmin`` and ``argmax`` ), ``bin``, scale
        type, custom ``sort`` order, nor ``timeUnit`` or (2) if you wish to use an ordinal
        scale for a field with ``bin`` or ``timeUnit``.

        **Default value:**

        1) For a data ``field``, ``"nominal"`` is the default data type unless the field
        encoding has ``aggregate``, ``channel``, ``bin``, scale type, ``sort``, or
        ``timeUnit`` that satisfies the following criteria: - ``"quantitative"`` is the
        default type if (1) the encoded field contains ``bin`` or ``aggregate`` except
        ``"argmin"`` and ``"argmax"``, (2) the encoding channel is ``latitude`` or
        ``longitude`` channel or (3) if the specified scale type is `a quantitative scale
        <https://vega.github.io/vega-lite/docs/scale.html#type>`__. - ``"temporal"`` is the
        default type if (1) the encoded field contains ``timeUnit`` or (2) the specified
        scale type is a time or utc scale - ``ordinal""`` is the default type if (1) the
        encoded field contains a `custom sort order
        <https://vega.github.io/vega-lite/docs/sort.html#specifying-custom-sort-order>`__,
        (2) the specified scale type is an ordinal/point/band scale, or (3) the encoding
        channel is ``order``.

        2) For a constant value in data domain ( ``datum`` ): - ``"quantitative"`` if the
        datum is a number - ``"nominal"`` if the datum is a string - ``"temporal"`` if the
        datum is `a date time object
        <https://vega.github.io/vega-lite/docs/datetime.html>`__

        **Note:** - Data ``type`` describes the semantics of the data rather than the
        primitive data types (number, string, etc.). The same primitive data type can have
        different types of measurement. For example, numeric data can represent
        quantitative, ordinal, or nominal data. - Data values for a temporal field can be
        either a date-time string (e.g., ``"2015-03-07 12:32:17"``, ``"17:01"``,
        ``"2015-03-16"``. ``"2015"`` ) or a timestamp number (e.g., ``1552199579097`` ). -
        When using with `bin <https://vega.github.io/vega-lite/docs/bin.html>`__, the
        ``type`` property can be either ``"quantitative"`` (for using a linear bin scale) or
        `"ordinal" (for using an ordinal bin scale)
        <https://vega.github.io/vega-lite/docs/type.html#cast-bin>`__. - When using with
        `timeUnit <https://vega.github.io/vega-lite/docs/timeunit.html>`__, the ``type``
        property can be either ``"temporal"`` (default, for using a temporal scale) or
        `"ordinal" (for using an ordinal scale)
        <https://vega.github.io/vega-lite/docs/type.html#cast-bin>`__. - When using with
        `aggregate <https://vega.github.io/vega-lite/docs/aggregate.html>`__, the ``type``
        property refers to the post-aggregation data type. For example, we can calculate
        count ``distinct`` of a categorical field ``"cat"`` using ``{"aggregate":
        "distinct", "field": "cat"}``. The ``"type"`` of the aggregate output is
        ``"quantitative"``. - Secondary channels (e.g., ``x2``, ``y2``, ``xError``,
        ``yError`` ) do not have ``type`` as they must have exactly the same type as their
        primary channels (e.g., ``x``, ``y`` ).

        **See also:** `type <https://vega.github.io/vega-lite/docs/type.html>`__
        documentation.
    """
    _schema = {'$ref': '#/definitions/FieldOrDatumDefWithCondition<MarkPropFieldDef,(Gradient|string|null)>'}

    def __init__(self, aggregate=Undefined, band=Undefined, bin=Undefined, condition=Undefined,
                 field=Undefined, legend=Undefined, scale=Undefined, sort=Undefined, timeUnit=Undefined,
                 title=Undefined, type=Undefined, **kwds):
        super(FieldOrDatumDefWithConditionMarkPropFieldDefGradientstringnull, self).__init__(aggregate=aggregate,
                                                                                             band=band,
                                                                                             bin=bin,
                                                                                             condition=condition,
                                                                                             field=field,
                                                                                             legend=legend,
                                                                                             scale=scale,
                                                                                             sort=sort,
                                                                                             timeUnit=timeUnit,
                                                                                             title=title,
                                                                                             type=type,
                                                                                             **kwds)


class MarkPropDefnumber(VegaLiteSchema):
    """MarkPropDefnumber schema wrapper

    anyOf(:class:`FieldOrDatumDefWithConditionMarkPropFieldDefnumber`,
    :class:`FieldOrDatumDefWithConditionDatumDefnumber`,
    :class:`ValueDefWithConditionMarkPropFieldOrDatumDefnumber`)
    """
    _schema = {'$ref': '#/definitions/MarkPropDef<number>'}

    def __init__(self, *args, **kwds):
        super(MarkPropDefnumber, self).__init__(*args, **kwds)


class MarkPropDefnumberArray(VegaLiteSchema):
    """MarkPropDefnumberArray schema wrapper

    anyOf(:class:`FieldOrDatumDefWithConditionMarkPropFieldDefnumberArray`,
    :class:`FieldOrDatumDefWithConditionDatumDefnumberArray`,
    :class:`ValueDefWithConditionMarkPropFieldOrDatumDefnumberArray`)
    """
    _schema = {'$ref': '#/definitions/MarkPropDef<number[]>'}

    def __init__(self, *args, **kwds):
        super(MarkPropDefnumberArray, self).__init__(*args, **kwds)


class MarkPropDefstringnullTypeForShape(VegaLiteSchema):
    """MarkPropDefstringnullTypeForShape schema wrapper

    anyOf(:class:`FieldOrDatumDefWithConditionMarkPropFieldDefTypeForShapestringnull`,
    :class:`FieldOrDatumDefWithConditionDatumDefstringnull`,
    :class:`ValueDefWithConditionMarkPropFieldOrDatumDefTypeForShapestringnull`)
    """
    _schema = {'$ref': '#/definitions/MarkPropDef<(string|null),TypeForShape>'}

    def __init__(self, *args, **kwds):
        super(MarkPropDefstringnullTypeForShape, self).__init__(*args, **kwds)


class MarkType(VegaLiteSchema):
    """MarkType schema wrapper

    enum('arc', 'area', 'image', 'group', 'line', 'path', 'rect', 'rule', 'shape', 'symbol',
    'text', 'trail')
    """
    _schema = {'$ref': '#/definitions/MarkType'}

    def __init__(self, *args):
        super(MarkType, self).__init__(*args)


class Month(VegaLiteSchema):
    """Month schema wrapper

    float
    """
    _schema = {'$ref': '#/definitions/Month'}

    def __init__(self, *args):
        super(Month, self).__init__(*args)


class MultiSelectionConfig(VegaLiteSchema):
    """MultiSelectionConfig schema wrapper

    Mapping(required=[])

    Attributes
    ----------

    bind : :class:`LegendBinding`
        When set, a selection is populated by interacting with the corresponding legend.
        Direct manipulation interaction is disabled by default; to re-enable it, set the
        selection's `on
        <https://vega.github.io/vega-lite/docs/selection.html#common-selection-properties>`__
        property.

        Legend bindings are restricted to selections that only specify a single field or
        encoding.
    clear : anyOf(:class:`Stream`, string, boolean)
        Clears the selection, emptying it of all values. Can be a `Event Stream
        <https://vega.github.io/vega/docs/event-streams/>`__ or ``false`` to disable.

        **Default value:** ``dblclick``.

        **See also:** `clear <https://vega.github.io/vega-lite/docs/clear.html>`__
        documentation.
    empty : enum('all', 'none')
        By default, ``all`` data values are considered to lie within an empty selection.
        When set to ``none``, empty selections contain no data values.
    encodings : List(:class:`SingleDefUnitChannel`)
        An array of encoding channels. The corresponding data field values must match for a
        data tuple to fall within the selection.

        **See also:** `encodings <https://vega.github.io/vega-lite/docs/project.html>`__
        documentation.
    fields : List(:class:`FieldName`)
        An array of field names whose values must match for a data tuple to fall within the
        selection.

        **See also:** `fields <https://vega.github.io/vega-lite/docs/project.html>`__
        documentation.
    init : List(:class:`SelectionInitMapping`)
        Initialize the selection with a mapping between `projected channels or field names
        <https://vega.github.io/vega-lite/docs/project.html>`__ and an initial value (or
        array of values).

        **See also:** `init <https://vega.github.io/vega-lite/docs/init.html>`__
        documentation.
    nearest : boolean
        When true, an invisible voronoi diagram is computed to accelerate discrete
        selection. The data value *nearest* the mouse cursor is added to the selection.

        **See also:** `nearest <https://vega.github.io/vega-lite/docs/nearest.html>`__
        documentation.
    on : anyOf(:class:`Stream`, string)
        A `Vega event stream <https://vega.github.io/vega/docs/event-streams/>`__ (object or
        selector) that triggers the selection. For interval selections, the event stream
        must specify a `start and end
        <https://vega.github.io/vega/docs/event-streams/#between-filters>`__.
    resolve : :class:`SelectionResolution`
        With layered and multi-view displays, a strategy that determines how selections'
        data queries are resolved when applied in a filter transform, conditional encoding
        rule, or scale domain.

        **See also:** `resolve
        <https://vega.github.io/vega-lite/docs/selection-resolve.html>`__ documentation.
    toggle : anyOf(string, boolean)
        Controls whether data values should be toggled or only ever inserted into multi
        selections. Can be ``true``, ``false`` (for insertion only), or a `Vega expression
        <https://vega.github.io/vega/docs/expressions/>`__.

        **Default value:** ``true``, which corresponds to ``event.shiftKey`` (i.e., data
        values are toggled when a user interacts with the shift-key pressed).

        Setting the value to the Vega expression ``"true"`` will toggle data values without
        the user pressing the shift-key.

        **See also:** `toggle <https://vega.github.io/vega-lite/docs/toggle.html>`__
        documentation.
    """
    _schema = {'$ref': '#/definitions/MultiSelectionConfig'}

    def __init__(self, bind=Undefined, clear=Undefined, empty=Undefined, encodings=Undefined,
                 fields=Undefined, init=Undefined, nearest=Undefined, on=Undefined, resolve=Undefined,
                 toggle=Undefined, **kwds):
        super(MultiSelectionConfig, self).__init__(bind=bind, clear=clear, empty=empty,
                                                   encodings=encodings, fields=fields, init=init,
                                                   nearest=nearest, on=on, resolve=resolve,
                                                   toggle=toggle, **kwds)


class NamedData(DataSource):
    """NamedData schema wrapper

    Mapping(required=[name])

    Attributes
    ----------

    name : string
        Provide a placeholder name and bind data at runtime.
    format : :class:`DataFormat`
        An object that specifies the format for parsing the data.
    """
    _schema = {'$ref': '#/definitions/NamedData'}

    def __init__(self, name=Undefined, format=Undefined, **kwds):
        super(NamedData, self).__init__(name=name, format=format, **kwds)


class NonArgAggregateOp(Aggregate):
    """NonArgAggregateOp schema wrapper

    enum('average', 'count', 'distinct', 'max', 'mean', 'median', 'min', 'missing', 'product',
    'q1', 'q3', 'ci0', 'ci1', 'stderr', 'stdev', 'stdevp', 'sum', 'valid', 'values', 'variance',
    'variancep')
    """
    _schema = {'$ref': '#/definitions/NonArgAggregateOp'}

    def __init__(self, *args):
        super(NonArgAggregateOp, self).__init__(*args)


class NormalizedSpec(VegaLiteSchema):
    """NormalizedSpec schema wrapper

    anyOf(:class:`FacetedUnitSpec`, :class:`LayerSpec`, :class:`RepeatSpec`,
    :class:`NormalizedFacetSpec`, :class:`NormalizedConcatSpecGenericSpec`,
    :class:`NormalizedVConcatSpecGenericSpec`, :class:`NormalizedHConcatSpecGenericSpec`)
    Any specification in Vega-Lite.
    """
    _schema = {'$ref': '#/definitions/NormalizedSpec'}

    def __init__(self, *args, **kwds):
        super(NormalizedSpec, self).__init__(*args, **kwds)


class NormalizedConcatSpecGenericSpec(NormalizedSpec):
    """NormalizedConcatSpecGenericSpec schema wrapper

    Mapping(required=[concat])
    Base interface for a generalized concatenation specification.

    Attributes
    ----------

    concat : List(:class:`NormalizedSpec`)
        A list of views to be concatenated.
    align : anyOf(:class:`LayoutAlign`, :class:`RowColLayoutAlign`)
        The alignment to apply to grid rows and columns. The supported string values are
        ``"all"``, ``"each"``, and ``"none"``.


        * For ``"none"``, a flow layout will be used, in which adjacent subviews are simply
          placed one after the other. - For ``"each"``, subviews will be aligned into a
          clean grid structure, but each row or column may be of variable size. - For
          ``"all"``, subviews will be aligned and each row or column will be sized
          identically based on the maximum observed size. String values for this property
          will be applied to both grid rows and columns.

        Alternatively, an object value of the form ``{"row": string, "column": string}`` can
        be used to supply different alignments for rows and columns.

        **Default value:** ``"all"``.
    bounds : enum('full', 'flush')
        The bounds calculation method to use for determining the extent of a sub-plot. One
        of ``full`` (the default) or ``flush``.


        * If set to ``full``, the entire calculated bounds (including axes, title, and
          legend) will be used. - If set to ``flush``, only the specified width and height
          values for the sub-view will be used. The ``flush`` setting can be useful when
          attempting to place sub-plots without axes or legends into a uniform grid
          structure.

        **Default value:** ``"full"``
    center : anyOf(boolean, :class:`RowColboolean`)
        Boolean flag indicating if subviews should be centered relative to their respective
        rows or columns.

        An object value of the form ``{"row": boolean, "column": boolean}`` can be used to
        supply different centering values for rows and columns.

        **Default value:** ``false``
    columns : float
        The number of columns to include in the view composition layout.

        **Default value** : ``undefined`` -- An infinite number of columns (a single row)
        will be assumed. This is equivalent to ``hconcat`` (for ``concat`` ) and to using
        the ``column`` channel (for ``facet`` and ``repeat`` ).

        **Note** :

        1) This property is only for: - the general (wrappable) ``concat`` operator (not
        ``hconcat`` / ``vconcat`` ) - the ``facet`` and ``repeat`` operator with one
        field/repetition definition (without row/column nesting)

        2) Setting the ``columns`` to ``1`` is equivalent to ``vconcat`` (for ``concat`` )
        and to using the ``row`` channel (for ``facet`` and ``repeat`` ).
    data : anyOf(:class:`Data`, None)
        An object describing the data source. Set to ``null`` to ignore the parent's data
        source. If no data is set, it is derived from the parent.
    description : string
        Description of this mark for commenting purpose.
    name : string
        Name of the visualization for later reference.
    resolve : :class:`Resolve`
        Scale, axis, and legend resolutions for view composition specifications.
    spacing : anyOf(float, :class:`RowColnumber`)
        The spacing in pixels between sub-views of the composition operator. An object of
        the form ``{"row": number, "column": number}`` can be used to set different spacing
        values for rows and columns.

        **Default value** : Depends on ``"spacing"`` property of `the view composition
        configuration <https://vega.github.io/vega-lite/docs/config.html#view-config>`__ (
        ``20`` by default)
    title : anyOf(:class:`Text`, :class:`TitleParams`)
        Title for the plot.
    transform : List(:class:`Transform`)
        An array of data transformations such as filter and new field calculation.
    """
    _schema = {'$ref': '#/definitions/NormalizedConcatSpec<GenericSpec>'}

    def __init__(self, concat=Undefined, align=Undefined, bounds=Undefined, center=Undefined,
                 columns=Undefined, data=Undefined, description=Undefined, name=Undefined,
                 resolve=Undefined, spacing=Undefined, title=Undefined, transform=Undefined, **kwds):
        super(NormalizedConcatSpecGenericSpec, self).__init__(concat=concat, align=align, bounds=bounds,
                                                              center=center, columns=columns, data=data,
                                                              description=description, name=name,
                                                              resolve=resolve, spacing=spacing,
                                                              title=title, transform=transform, **kwds)


class NormalizedFacetSpec(NormalizedSpec):
    """NormalizedFacetSpec schema wrapper

    Mapping(required=[facet, spec])
    Base interface for a facet specification.

    Attributes
    ----------

    facet : anyOf(:class:`FacetFieldDef`, :class:`FacetMapping`)
        Definition for how to facet the data. One of: 1) `a field definition for faceting
        the plot by one field
        <https://vega.github.io/vega-lite/docs/facet.html#field-def>`__ 2) `An object that
        maps row and column channels to their field definitions
        <https://vega.github.io/vega-lite/docs/facet.html#mapping>`__
    spec : anyOf(:class:`LayerSpec`, :class:`FacetedUnitSpec`)
        A specification of the view that gets faceted.
    align : anyOf(:class:`LayoutAlign`, :class:`RowColLayoutAlign`)
        The alignment to apply to grid rows and columns. The supported string values are
        ``"all"``, ``"each"``, and ``"none"``.


        * For ``"none"``, a flow layout will be used, in which adjacent subviews are simply
          placed one after the other. - For ``"each"``, subviews will be aligned into a
          clean grid structure, but each row or column may be of variable size. - For
          ``"all"``, subviews will be aligned and each row or column will be sized
          identically based on the maximum observed size. String values for this property
          will be applied to both grid rows and columns.

        Alternatively, an object value of the form ``{"row": string, "column": string}`` can
        be used to supply different alignments for rows and columns.

        **Default value:** ``"all"``.
    bounds : enum('full', 'flush')
        The bounds calculation method to use for determining the extent of a sub-plot. One
        of ``full`` (the default) or ``flush``.


        * If set to ``full``, the entire calculated bounds (including axes, title, and
          legend) will be used. - If set to ``flush``, only the specified width and height
          values for the sub-view will be used. The ``flush`` setting can be useful when
          attempting to place sub-plots without axes or legends into a uniform grid
          structure.

        **Default value:** ``"full"``
    center : anyOf(boolean, :class:`RowColboolean`)
        Boolean flag indicating if subviews should be centered relative to their respective
        rows or columns.

        An object value of the form ``{"row": boolean, "column": boolean}`` can be used to
        supply different centering values for rows and columns.

        **Default value:** ``false``
    columns : float
        The number of columns to include in the view composition layout.

        **Default value** : ``undefined`` -- An infinite number of columns (a single row)
        will be assumed. This is equivalent to ``hconcat`` (for ``concat`` ) and to using
        the ``column`` channel (for ``facet`` and ``repeat`` ).

        **Note** :

        1) This property is only for: - the general (wrappable) ``concat`` operator (not
        ``hconcat`` / ``vconcat`` ) - the ``facet`` and ``repeat`` operator with one
        field/repetition definition (without row/column nesting)

        2) Setting the ``columns`` to ``1`` is equivalent to ``vconcat`` (for ``concat`` )
        and to using the ``row`` channel (for ``facet`` and ``repeat`` ).
    data : anyOf(:class:`Data`, None)
        An object describing the data source. Set to ``null`` to ignore the parent's data
        source. If no data is set, it is derived from the parent.
    description : string
        Description of this mark for commenting purpose.
    name : string
        Name of the visualization for later reference.
    resolve : :class:`Resolve`
        Scale, axis, and legend resolutions for view composition specifications.
    spacing : anyOf(float, :class:`RowColnumber`)
        The spacing in pixels between sub-views of the composition operator. An object of
        the form ``{"row": number, "column": number}`` can be used to set different spacing
        values for rows and columns.

        **Default value** : Depends on ``"spacing"`` property of `the view composition
        configuration <https://vega.github.io/vega-lite/docs/config.html#view-config>`__ (
        ``20`` by default)
    title : anyOf(:class:`Text`, :class:`TitleParams`)
        Title for the plot.
    transform : List(:class:`Transform`)
        An array of data transformations such as filter and new field calculation.
    """
    _schema = {'$ref': '#/definitions/NormalizedFacetSpec'}

    def __init__(self, facet=Undefined, spec=Undefined, align=Undefined, bounds=Undefined,
                 center=Undefined, columns=Undefined, data=Undefined, description=Undefined,
                 name=Undefined, resolve=Undefined, spacing=Undefined, title=Undefined,
                 transform=Undefined, **kwds):
        super(NormalizedFacetSpec, self).__init__(facet=facet, spec=spec, align=align, bounds=bounds,
                                                  center=center, columns=columns, data=data,
                                                  description=description, name=name, resolve=resolve,
                                                  spacing=spacing, title=title, transform=transform,
                                                  **kwds)


class NormalizedHConcatSpecGenericSpec(NormalizedSpec):
    """NormalizedHConcatSpecGenericSpec schema wrapper

    Mapping(required=[hconcat])
    Base interface for a horizontal concatenation specification.

    Attributes
    ----------

    hconcat : List(:class:`NormalizedSpec`)
        A list of views to be concatenated and put into a row.
    bounds : enum('full', 'flush')
        The bounds calculation method to use for determining the extent of a sub-plot. One
        of ``full`` (the default) or ``flush``.


        * If set to ``full``, the entire calculated bounds (including axes, title, and
          legend) will be used. - If set to ``flush``, only the specified width and height
          values for the sub-view will be used. The ``flush`` setting can be useful when
          attempting to place sub-plots without axes or legends into a uniform grid
          structure.

        **Default value:** ``"full"``
    center : boolean
        Boolean flag indicating if subviews should be centered relative to their respective
        rows or columns.

        **Default value:** ``false``
    data : anyOf(:class:`Data`, None)
        An object describing the data source. Set to ``null`` to ignore the parent's data
        source. If no data is set, it is derived from the parent.
    description : string
        Description of this mark for commenting purpose.
    name : string
        Name of the visualization for later reference.
    resolve : :class:`Resolve`
        Scale, axis, and legend resolutions for view composition specifications.
    spacing : float
        The spacing in pixels between sub-views of the concat operator.

        **Default value** : ``10``
    title : anyOf(:class:`Text`, :class:`TitleParams`)
        Title for the plot.
    transform : List(:class:`Transform`)
        An array of data transformations such as filter and new field calculation.
    """
    _schema = {'$ref': '#/definitions/NormalizedHConcatSpec<GenericSpec>'}

    def __init__(self, hconcat=Undefined, bounds=Undefined, center=Undefined, data=Undefined,
                 description=Undefined, name=Undefined, resolve=Undefined, spacing=Undefined,
                 title=Undefined, transform=Undefined, **kwds):
        super(NormalizedHConcatSpecGenericSpec, self).__init__(hconcat=hconcat, bounds=bounds,
                                                               center=center, data=data,
                                                               description=description, name=name,
                                                               resolve=resolve, spacing=spacing,
                                                               title=title, transform=transform, **kwds)


class NormalizedVConcatSpecGenericSpec(NormalizedSpec):
    """NormalizedVConcatSpecGenericSpec schema wrapper

    Mapping(required=[vconcat])
    Base interface for a vertical concatenation specification.

    Attributes
    ----------

    vconcat : List(:class:`NormalizedSpec`)
        A list of views to be concatenated and put into a column.
    bounds : enum('full', 'flush')
        The bounds calculation method to use for determining the extent of a sub-plot. One
        of ``full`` (the default) or ``flush``.


        * If set to ``full``, the entire calculated bounds (including axes, title, and
          legend) will be used. - If set to ``flush``, only the specified width and height
          values for the sub-view will be used. The ``flush`` setting can be useful when
          attempting to place sub-plots without axes or legends into a uniform grid
          structure.

        **Default value:** ``"full"``
    center : boolean
        Boolean flag indicating if subviews should be centered relative to their respective
        rows or columns.

        **Default value:** ``false``
    data : anyOf(:class:`Data`, None)
        An object describing the data source. Set to ``null`` to ignore the parent's data
        source. If no data is set, it is derived from the parent.
    description : string
        Description of this mark for commenting purpose.
    name : string
        Name of the visualization for later reference.
    resolve : :class:`Resolve`
        Scale, axis, and legend resolutions for view composition specifications.
    spacing : float
        The spacing in pixels between sub-views of the concat operator.

        **Default value** : ``10``
    title : anyOf(:class:`Text`, :class:`TitleParams`)
        Title for the plot.
    transform : List(:class:`Transform`)
        An array of data transformations such as filter and new field calculation.
    """
    _schema = {'$ref': '#/definitions/NormalizedVConcatSpec<GenericSpec>'}

    def __init__(self, vconcat=Undefined, bounds=Undefined, center=Undefined, data=Undefined,
                 description=Undefined, name=Undefined, resolve=Undefined, spacing=Undefined,
                 title=Undefined, transform=Undefined, **kwds):
        super(NormalizedVConcatSpecGenericSpec, self).__init__(vconcat=vconcat, bounds=bounds,
                                                               center=center, data=data,
                                                               description=description, name=name,
                                                               resolve=resolve, spacing=spacing,
                                                               title=title, transform=transform, **kwds)


class NumericArrayMarkPropDef(VegaLiteSchema):
    """NumericArrayMarkPropDef schema wrapper

    anyOf(:class:`FieldOrDatumDefWithConditionMarkPropFieldDefnumberArray`,
    :class:`FieldOrDatumDefWithConditionDatumDefnumberArray`,
    :class:`ValueDefWithConditionMarkPropFieldOrDatumDefnumberArray`)
    """
    _schema = {'$ref': '#/definitions/NumericArrayMarkPropDef'}

    def __init__(self, *args, **kwds):
        super(NumericArrayMarkPropDef, self).__init__(*args, **kwds)


class FieldOrDatumDefWithConditionDatumDefnumberArray(MarkPropDefnumberArray, NumericArrayMarkPropDef):
    """FieldOrDatumDefWithConditionDatumDefnumberArray schema wrapper

    Mapping(required=[])
    A FieldDef with Condition :raw-html:`<ValueDef>` {    condition: {value: ...},    field:
    ...,    ... }

    Attributes
    ----------

    band : float
        For rect-based marks ( ``rect``, ``bar``, and ``image`` ), mark size relative to
        bandwidth of `band scales
        <https://vega.github.io/vega-lite/docs/scale.html#band>`__, bins or time units. If
        set to ``1``, the mark size is set to the bandwidth, the bin interval, or the time
        unit interval. If set to ``0.5``, the mark size is half of the bandwidth or the time
        unit interval.

        For other marks, relative position on a band of a stacked, binned, time unit or band
        scale. If set to ``0``, the marks will be positioned at the beginning of the band.
        If set to ``0.5``, the marks will be positioned in the middle of the band.
    condition : anyOf(:class:`ConditionalValueDefnumberArrayExprRef`,
    List(:class:`ConditionalValueDefnumberArrayExprRef`))
        One or more value definition(s) with `a selection or a test predicate
        <https://vega.github.io/vega-lite/docs/condition.html>`__.

        **Note:** A field definition's ``condition`` property can only contain `conditional
        value definitions <https://vega.github.io/vega-lite/docs/condition.html#value>`__
        since Vega-Lite only allows at most one encoded field per encoding channel.
    datum : anyOf(:class:`PrimitiveValue`, :class:`DateTime`, :class:`ExprRef`,
    :class:`RepeatRef`)
        A constant value in data domain.
    type : :class:`Type`
        The type of measurement ( ``"quantitative"``, ``"temporal"``, ``"ordinal"``, or
        ``"nominal"`` ) for the encoded field or constant value ( ``datum`` ). It can also
        be a ``"geojson"`` type for encoding `'geoshape'
        <https://vega.github.io/vega-lite/docs/geoshape.html>`__.

        Vega-Lite automatically infers data types in many cases as discussed below. However,
        type is required for a field if: (1) the field is not nominal and the field encoding
        has no specified ``aggregate`` (except ``argmin`` and ``argmax`` ), ``bin``, scale
        type, custom ``sort`` order, nor ``timeUnit`` or (2) if you wish to use an ordinal
        scale for a field with ``bin`` or ``timeUnit``.

        **Default value:**

        1) For a data ``field``, ``"nominal"`` is the default data type unless the field
        encoding has ``aggregate``, ``channel``, ``bin``, scale type, ``sort``, or
        ``timeUnit`` that satisfies the following criteria: - ``"quantitative"`` is the
        default type if (1) the encoded field contains ``bin`` or ``aggregate`` except
        ``"argmin"`` and ``"argmax"``, (2) the encoding channel is ``latitude`` or
        ``longitude`` channel or (3) if the specified scale type is `a quantitative scale
        <https://vega.github.io/vega-lite/docs/scale.html#type>`__. - ``"temporal"`` is the
        default type if (1) the encoded field contains ``timeUnit`` or (2) the specified
        scale type is a time or utc scale - ``ordinal""`` is the default type if (1) the
        encoded field contains a `custom sort order
        <https://vega.github.io/vega-lite/docs/sort.html#specifying-custom-sort-order>`__,
        (2) the specified scale type is an ordinal/point/band scale, or (3) the encoding
        channel is ``order``.

        2) For a constant value in data domain ( ``datum`` ): - ``"quantitative"`` if the
        datum is a number - ``"nominal"`` if the datum is a string - ``"temporal"`` if the
        datum is `a date time object
        <https://vega.github.io/vega-lite/docs/datetime.html>`__

        **Note:** - Data ``type`` describes the semantics of the data rather than the
        primitive data types (number, string, etc.). The same primitive data type can have
        different types of measurement. For example, numeric data can represent
        quantitative, ordinal, or nominal data. - Data values for a temporal field can be
        either a date-time string (e.g., ``"2015-03-07 12:32:17"``, ``"17:01"``,
        ``"2015-03-16"``. ``"2015"`` ) or a timestamp number (e.g., ``1552199579097`` ). -
        When using with `bin <https://vega.github.io/vega-lite/docs/bin.html>`__, the
        ``type`` property can be either ``"quantitative"`` (for using a linear bin scale) or
        `"ordinal" (for using an ordinal bin scale)
        <https://vega.github.io/vega-lite/docs/type.html#cast-bin>`__. - When using with
        `timeUnit <https://vega.github.io/vega-lite/docs/timeunit.html>`__, the ``type``
        property can be either ``"temporal"`` (default, for using a temporal scale) or
        `"ordinal" (for using an ordinal scale)
        <https://vega.github.io/vega-lite/docs/type.html#cast-bin>`__. - When using with
        `aggregate <https://vega.github.io/vega-lite/docs/aggregate.html>`__, the ``type``
        property refers to the post-aggregation data type. For example, we can calculate
        count ``distinct`` of a categorical field ``"cat"`` using ``{"aggregate":
        "distinct", "field": "cat"}``. The ``"type"`` of the aggregate output is
        ``"quantitative"``. - Secondary channels (e.g., ``x2``, ``y2``, ``xError``,
        ``yError`` ) do not have ``type`` as they must have exactly the same type as their
        primary channels (e.g., ``x``, ``y`` ).

        **See also:** `type <https://vega.github.io/vega-lite/docs/type.html>`__
        documentation.
    """
    _schema = {'$ref': '#/definitions/FieldOrDatumDefWithCondition<DatumDef,number[]>'}

    def __init__(self, band=Undefined, condition=Undefined, datum=Undefined, type=Undefined, **kwds):
        super(FieldOrDatumDefWithConditionDatumDefnumberArray, self).__init__(band=band,
                                                                              condition=condition,
                                                                              datum=datum, type=type,
                                                                              **kwds)


class FieldOrDatumDefWithConditionMarkPropFieldDefnumberArray(MarkPropDefnumberArray, NumericArrayMarkPropDef):
    """FieldOrDatumDefWithConditionMarkPropFieldDefnumberArray schema wrapper

    Mapping(required=[])
    A FieldDef with Condition :raw-html:`<ValueDef>` {    condition: {value: ...},    field:
    ...,    ... }

    Attributes
    ----------

    aggregate : :class:`Aggregate`
        Aggregation function for the field (e.g., ``"mean"``, ``"sum"``, ``"median"``,
        ``"min"``, ``"max"``, ``"count"`` ).

        **Default value:** ``undefined`` (None)

        **See also:** `aggregate <https://vega.github.io/vega-lite/docs/aggregate.html>`__
        documentation.
    band : float
        For rect-based marks ( ``rect``, ``bar``, and ``image`` ), mark size relative to
        bandwidth of `band scales
        <https://vega.github.io/vega-lite/docs/scale.html#band>`__, bins or time units. If
        set to ``1``, the mark size is set to the bandwidth, the bin interval, or the time
        unit interval. If set to ``0.5``, the mark size is half of the bandwidth or the time
        unit interval.

        For other marks, relative position on a band of a stacked, binned, time unit or band
        scale. If set to ``0``, the marks will be positioned at the beginning of the band.
        If set to ``0.5``, the marks will be positioned in the middle of the band.
    bin : anyOf(boolean, :class:`BinParams`, None)
        A flag for binning a ``quantitative`` field, `an object defining binning parameters
        <https://vega.github.io/vega-lite/docs/bin.html#params>`__, or indicating that the
        data for ``x`` or ``y`` channel are binned before they are imported into Vega-Lite (
        ``"binned"`` ).


        If ``true``, default `binning parameters
        <https://vega.github.io/vega-lite/docs/bin.html>`__ will be applied.

        If ``"binned"``, this indicates that the data for the ``x`` (or ``y`` ) channel are
        already binned. You can map the bin-start field to ``x`` (or ``y`` ) and the bin-end
        field to ``x2`` (or ``y2`` ). The scale and axis will be formatted similar to
        binning in Vega-Lite.  To adjust the axis ticks based on the bin step, you can also
        set the axis's `tickMinStep
        <https://vega.github.io/vega-lite/docs/axis.html#ticks>`__ property.

        **Default value:** ``false``

        **See also:** `bin <https://vega.github.io/vega-lite/docs/bin.html>`__
        documentation.
    condition : anyOf(:class:`ConditionalValueDefnumberArrayExprRef`,
    List(:class:`ConditionalValueDefnumberArrayExprRef`))
        One or more value definition(s) with `a selection or a test predicate
        <https://vega.github.io/vega-lite/docs/condition.html>`__.

        **Note:** A field definition's ``condition`` property can only contain `conditional
        value definitions <https://vega.github.io/vega-lite/docs/condition.html#value>`__
        since Vega-Lite only allows at most one encoded field per encoding channel.
    field : :class:`Field`
        **Required.** A string defining the name of the field from which to pull a data
        value or an object defining iterated values from the `repeat
        <https://vega.github.io/vega-lite/docs/repeat.html>`__ operator.

        **See also:** `field <https://vega.github.io/vega-lite/docs/field.html>`__
        documentation.

        **Notes:** 1)  Dots ( ``.`` ) and brackets ( ``[`` and ``]`` ) can be used to access
        nested objects (e.g., ``"field": "foo.bar"`` and ``"field": "foo['bar']"`` ). If
        field names contain dots or brackets but are not nested, you can use ``\\`` to
        escape dots and brackets (e.g., ``"a\\.b"`` and ``"a\\[0\\]"`` ). See more details
        about escaping in the `field documentation
        <https://vega.github.io/vega-lite/docs/field.html>`__. 2) ``field`` is not required
        if ``aggregate`` is ``count``.
    legend : anyOf(:class:`Legend`, None)
        An object defining properties of the legend. If ``null``, the legend for the
        encoding channel will be removed.

        **Default value:** If undefined, default `legend properties
        <https://vega.github.io/vega-lite/docs/legend.html>`__ are applied.

        **See also:** `legend <https://vega.github.io/vega-lite/docs/legend.html>`__
        documentation.
    scale : anyOf(:class:`Scale`, None)
        An object defining properties of the channel's scale, which is the function that
        transforms values in the data domain (numbers, dates, strings, etc) to visual values
        (pixels, colors, sizes) of the encoding channels.

        If ``null``, the scale will be `disabled and the data value will be directly encoded
        <https://vega.github.io/vega-lite/docs/scale.html#disable>`__.

        **Default value:** If undefined, default `scale properties
        <https://vega.github.io/vega-lite/docs/scale.html>`__ are applied.

        **See also:** `scale <https://vega.github.io/vega-lite/docs/scale.html>`__
        documentation.
    sort : :class:`Sort`
        Sort order for the encoded field.

        For continuous fields (quantitative or temporal), ``sort`` can be either
        ``"ascending"`` or ``"descending"``.

        For discrete fields, ``sort`` can be one of the following: - ``"ascending"`` or
        ``"descending"`` -- for sorting by the values' natural order in JavaScript. - `A
        string indicating an encoding channel name to sort by
        <https://vega.github.io/vega-lite/docs/sort.html#sort-by-encoding>`__ (e.g., ``"x"``
        or ``"y"`` ) with an optional minus prefix for descending sort (e.g., ``"-x"`` to
        sort by x-field, descending). This channel string is short-form of `a
        sort-by-encoding definition
        <https://vega.github.io/vega-lite/docs/sort.html#sort-by-encoding>`__. For example,
        ``"sort": "-x"`` is equivalent to ``"sort": {"encoding": "x", "order":
        "descending"}``. - `A sort field definition
        <https://vega.github.io/vega-lite/docs/sort.html#sort-field>`__ for sorting by
        another field. - `An array specifying the field values in preferred order
        <https://vega.github.io/vega-lite/docs/sort.html#sort-array>`__. In this case, the
        sort order will obey the values in the array, followed by any unspecified values in
        their original order. For discrete time field, values in the sort array can be
        `date-time definition objects <types#datetime>`__. In addition, for time units
        ``"month"`` and ``"day"``, the values can be the month or day names (case
        insensitive) or their 3-letter initials (e.g., ``"Mon"``, ``"Tue"`` ). - ``null``
        indicating no sort.

        **Default value:** ``"ascending"``

        **Note:** ``null`` and sorting by another channel is not supported for ``row`` and
        ``column``.

        **See also:** `sort <https://vega.github.io/vega-lite/docs/sort.html>`__
        documentation.
    timeUnit : anyOf(:class:`TimeUnit`, :class:`TimeUnitParams`)
        Time unit (e.g., ``year``, ``yearmonth``, ``month``, ``hours`` ) for a temporal
        field. or `a temporal field that gets casted as ordinal
        <https://vega.github.io/vega-lite/docs/type.html#cast>`__.

        **Default value:** ``undefined`` (None)

        **See also:** `timeUnit <https://vega.github.io/vega-lite/docs/timeunit.html>`__
        documentation.
    title : anyOf(:class:`Text`, None)
        A title for the field. If ``null``, the title will be removed.

        **Default value:**  derived from the field's name and transformation function (
        ``aggregate``, ``bin`` and ``timeUnit`` ). If the field has an aggregate function,
        the function is displayed as part of the title (e.g., ``"Sum of Profit"`` ). If the
        field is binned or has a time unit applied, the applied function is shown in
        parentheses (e.g., ``"Profit (binned)"``, ``"Transaction Date (year-month)"`` ).
        Otherwise, the title is simply the field name.

        **Notes** :

        1) You can customize the default field title format by providing the `fieldTitle
        <https://vega.github.io/vega-lite/docs/config.html#top-level-config>`__ property in
        the `config <https://vega.github.io/vega-lite/docs/config.html>`__ or `fieldTitle
        function via the compile function's options
        <https://vega.github.io/vega-lite/docs/compile.html#field-title>`__.

        2) If both field definition's ``title`` and axis, header, or legend ``title`` are
        defined, axis/header/legend title will be used.
    type : :class:`StandardType`
        The type of measurement ( ``"quantitative"``, ``"temporal"``, ``"ordinal"``, or
        ``"nominal"`` ) for the encoded field or constant value ( ``datum`` ). It can also
        be a ``"geojson"`` type for encoding `'geoshape'
        <https://vega.github.io/vega-lite/docs/geoshape.html>`__.

        Vega-Lite automatically infers data types in many cases as discussed below. However,
        type is required for a field if: (1) the field is not nominal and the field encoding
        has no specified ``aggregate`` (except ``argmin`` and ``argmax`` ), ``bin``, scale
        type, custom ``sort`` order, nor ``timeUnit`` or (2) if you wish to use an ordinal
        scale for a field with ``bin`` or ``timeUnit``.

        **Default value:**

        1) For a data ``field``, ``"nominal"`` is the default data type unless the field
        encoding has ``aggregate``, ``channel``, ``bin``, scale type, ``sort``, or
        ``timeUnit`` that satisfies the following criteria: - ``"quantitative"`` is the
        default type if (1) the encoded field contains ``bin`` or ``aggregate`` except
        ``"argmin"`` and ``"argmax"``, (2) the encoding channel is ``latitude`` or
        ``longitude`` channel or (3) if the specified scale type is `a quantitative scale
        <https://vega.github.io/vega-lite/docs/scale.html#type>`__. - ``"temporal"`` is the
        default type if (1) the encoded field contains ``timeUnit`` or (2) the specified
        scale type is a time or utc scale - ``ordinal""`` is the default type if (1) the
        encoded field contains a `custom sort order
        <https://vega.github.io/vega-lite/docs/sort.html#specifying-custom-sort-order>`__,
        (2) the specified scale type is an ordinal/point/band scale, or (3) the encoding
        channel is ``order``.

        2) For a constant value in data domain ( ``datum`` ): - ``"quantitative"`` if the
        datum is a number - ``"nominal"`` if the datum is a string - ``"temporal"`` if the
        datum is `a date time object
        <https://vega.github.io/vega-lite/docs/datetime.html>`__

        **Note:** - Data ``type`` describes the semantics of the data rather than the
        primitive data types (number, string, etc.). The same primitive data type can have
        different types of measurement. For example, numeric data can represent
        quantitative, ordinal, or nominal data. - Data values for a temporal field can be
        either a date-time string (e.g., ``"2015-03-07 12:32:17"``, ``"17:01"``,
        ``"2015-03-16"``. ``"2015"`` ) or a timestamp number (e.g., ``1552199579097`` ). -
        When using with `bin <https://vega.github.io/vega-lite/docs/bin.html>`__, the
        ``type`` property can be either ``"quantitative"`` (for using a linear bin scale) or
        `"ordinal" (for using an ordinal bin scale)
        <https://vega.github.io/vega-lite/docs/type.html#cast-bin>`__. - When using with
        `timeUnit <https://vega.github.io/vega-lite/docs/timeunit.html>`__, the ``type``
        property can be either ``"temporal"`` (default, for using a temporal scale) or
        `"ordinal" (for using an ordinal scale)
        <https://vega.github.io/vega-lite/docs/type.html#cast-bin>`__. - When using with
        `aggregate <https://vega.github.io/vega-lite/docs/aggregate.html>`__, the ``type``
        property refers to the post-aggregation data type. For example, we can calculate
        count ``distinct`` of a categorical field ``"cat"`` using ``{"aggregate":
        "distinct", "field": "cat"}``. The ``"type"`` of the aggregate output is
        ``"quantitative"``. - Secondary channels (e.g., ``x2``, ``y2``, ``xError``,
        ``yError`` ) do not have ``type`` as they must have exactly the same type as their
        primary channels (e.g., ``x``, ``y`` ).

        **See also:** `type <https://vega.github.io/vega-lite/docs/type.html>`__
        documentation.
    """
    _schema = {'$ref': '#/definitions/FieldOrDatumDefWithCondition<MarkPropFieldDef,number[]>'}

    def __init__(self, aggregate=Undefined, band=Undefined, bin=Undefined, condition=Undefined,
                 field=Undefined, legend=Undefined, scale=Undefined, sort=Undefined, timeUnit=Undefined,
                 title=Undefined, type=Undefined, **kwds):
        super(FieldOrDatumDefWithConditionMarkPropFieldDefnumberArray, self).__init__(aggregate=aggregate,
                                                                                      band=band,
                                                                                      bin=bin,
                                                                                      condition=condition,
                                                                                      field=field,
                                                                                      legend=legend,
                                                                                      scale=scale,
                                                                                      sort=sort,
                                                                                      timeUnit=timeUnit,
                                                                                      title=title,
                                                                                      type=type, **kwds)


class NumericMarkPropDef(VegaLiteSchema):
    """NumericMarkPropDef schema wrapper

    anyOf(:class:`FieldOrDatumDefWithConditionMarkPropFieldDefnumber`,
    :class:`FieldOrDatumDefWithConditionDatumDefnumber`,
    :class:`ValueDefWithConditionMarkPropFieldOrDatumDefnumber`)
    """
    _schema = {'$ref': '#/definitions/NumericMarkPropDef'}

    def __init__(self, *args, **kwds):
        super(NumericMarkPropDef, self).__init__(*args, **kwds)


class FieldOrDatumDefWithConditionDatumDefnumber(MarkPropDefnumber, NumericMarkPropDef):
    """FieldOrDatumDefWithConditionDatumDefnumber schema wrapper

    Mapping(required=[])
    A FieldDef with Condition :raw-html:`<ValueDef>` {    condition: {value: ...},    field:
    ...,    ... }

    Attributes
    ----------

    band : float
        For rect-based marks ( ``rect``, ``bar``, and ``image`` ), mark size relative to
        bandwidth of `band scales
        <https://vega.github.io/vega-lite/docs/scale.html#band>`__, bins or time units. If
        set to ``1``, the mark size is set to the bandwidth, the bin interval, or the time
        unit interval. If set to ``0.5``, the mark size is half of the bandwidth or the time
        unit interval.

        For other marks, relative position on a band of a stacked, binned, time unit or band
        scale. If set to ``0``, the marks will be positioned at the beginning of the band.
        If set to ``0.5``, the marks will be positioned in the middle of the band.
    condition : anyOf(:class:`ConditionalValueDefnumberExprRef`,
    List(:class:`ConditionalValueDefnumberExprRef`))
        One or more value definition(s) with `a selection or a test predicate
        <https://vega.github.io/vega-lite/docs/condition.html>`__.

        **Note:** A field definition's ``condition`` property can only contain `conditional
        value definitions <https://vega.github.io/vega-lite/docs/condition.html#value>`__
        since Vega-Lite only allows at most one encoded field per encoding channel.
    datum : anyOf(:class:`PrimitiveValue`, :class:`DateTime`, :class:`ExprRef`,
    :class:`RepeatRef`)
        A constant value in data domain.
    type : :class:`Type`
        The type of measurement ( ``"quantitative"``, ``"temporal"``, ``"ordinal"``, or
        ``"nominal"`` ) for the encoded field or constant value ( ``datum`` ). It can also
        be a ``"geojson"`` type for encoding `'geoshape'
        <https://vega.github.io/vega-lite/docs/geoshape.html>`__.

        Vega-Lite automatically infers data types in many cases as discussed below. However,
        type is required for a field if: (1) the field is not nominal and the field encoding
        has no specified ``aggregate`` (except ``argmin`` and ``argmax`` ), ``bin``, scale
        type, custom ``sort`` order, nor ``timeUnit`` or (2) if you wish to use an ordinal
        scale for a field with ``bin`` or ``timeUnit``.

        **Default value:**

        1) For a data ``field``, ``"nominal"`` is the default data type unless the field
        encoding has ``aggregate``, ``channel``, ``bin``, scale type, ``sort``, or
        ``timeUnit`` that satisfies the following criteria: - ``"quantitative"`` is the
        default type if (1) the encoded field contains ``bin`` or ``aggregate`` except
        ``"argmin"`` and ``"argmax"``, (2) the encoding channel is ``latitude`` or
        ``longitude`` channel or (3) if the specified scale type is `a quantitative scale
        <https://vega.github.io/vega-lite/docs/scale.html#type>`__. - ``"temporal"`` is the
        default type if (1) the encoded field contains ``timeUnit`` or (2) the specified
        scale type is a time or utc scale - ``ordinal""`` is the default type if (1) the
        encoded field contains a `custom sort order
        <https://vega.github.io/vega-lite/docs/sort.html#specifying-custom-sort-order>`__,
        (2) the specified scale type is an ordinal/point/band scale, or (3) the encoding
        channel is ``order``.

        2) For a constant value in data domain ( ``datum`` ): - ``"quantitative"`` if the
        datum is a number - ``"nominal"`` if the datum is a string - ``"temporal"`` if the
        datum is `a date time object
        <https://vega.github.io/vega-lite/docs/datetime.html>`__

        **Note:** - Data ``type`` describes the semantics of the data rather than the
        primitive data types (number, string, etc.). The same primitive data type can have
        different types of measurement. For example, numeric data can represent
        quantitative, ordinal, or nominal data. - Data values for a temporal field can be
        either a date-time string (e.g., ``"2015-03-07 12:32:17"``, ``"17:01"``,
        ``"2015-03-16"``. ``"2015"`` ) or a timestamp number (e.g., ``1552199579097`` ). -
        When using with `bin <https://vega.github.io/vega-lite/docs/bin.html>`__, the
        ``type`` property can be either ``"quantitative"`` (for using a linear bin scale) or
        `"ordinal" (for using an ordinal bin scale)
        <https://vega.github.io/vega-lite/docs/type.html#cast-bin>`__. - When using with
        `timeUnit <https://vega.github.io/vega-lite/docs/timeunit.html>`__, the ``type``
        property can be either ``"temporal"`` (default, for using a temporal scale) or
        `"ordinal" (for using an ordinal scale)
        <https://vega.github.io/vega-lite/docs/type.html#cast-bin>`__. - When using with
        `aggregate <https://vega.github.io/vega-lite/docs/aggregate.html>`__, the ``type``
        property refers to the post-aggregation data type. For example, we can calculate
        count ``distinct`` of a categorical field ``"cat"`` using ``{"aggregate":
        "distinct", "field": "cat"}``. The ``"type"`` of the aggregate output is
        ``"quantitative"``. - Secondary channels (e.g., ``x2``, ``y2``, ``xError``,
        ``yError`` ) do not have ``type`` as they must have exactly the same type as their
        primary channels (e.g., ``x``, ``y`` ).

        **See also:** `type <https://vega.github.io/vega-lite/docs/type.html>`__
        documentation.
    """
    _schema = {'$ref': '#/definitions/FieldOrDatumDefWithCondition<DatumDef,number>'}

    def __init__(self, band=Undefined, condition=Undefined, datum=Undefined, type=Undefined, **kwds):
        super(FieldOrDatumDefWithConditionDatumDefnumber, self).__init__(band=band, condition=condition,
                                                                         datum=datum, type=type, **kwds)


class FieldOrDatumDefWithConditionMarkPropFieldDefnumber(MarkPropDefnumber, NumericMarkPropDef):
    """FieldOrDatumDefWithConditionMarkPropFieldDefnumber schema wrapper

    Mapping(required=[])
    A FieldDef with Condition :raw-html:`<ValueDef>` {    condition: {value: ...},    field:
    ...,    ... }

    Attributes
    ----------

    aggregate : :class:`Aggregate`
        Aggregation function for the field (e.g., ``"mean"``, ``"sum"``, ``"median"``,
        ``"min"``, ``"max"``, ``"count"`` ).

        **Default value:** ``undefined`` (None)

        **See also:** `aggregate <https://vega.github.io/vega-lite/docs/aggregate.html>`__
        documentation.
    band : float
        For rect-based marks ( ``rect``, ``bar``, and ``image`` ), mark size relative to
        bandwidth of `band scales
        <https://vega.github.io/vega-lite/docs/scale.html#band>`__, bins or time units. If
        set to ``1``, the mark size is set to the bandwidth, the bin interval, or the time
        unit interval. If set to ``0.5``, the mark size is half of the bandwidth or the time
        unit interval.

        For other marks, relative position on a band of a stacked, binned, time unit or band
        scale. If set to ``0``, the marks will be positioned at the beginning of the band.
        If set to ``0.5``, the marks will be positioned in the middle of the band.
    bin : anyOf(boolean, :class:`BinParams`, None)
        A flag for binning a ``quantitative`` field, `an object defining binning parameters
        <https://vega.github.io/vega-lite/docs/bin.html#params>`__, or indicating that the
        data for ``x`` or ``y`` channel are binned before they are imported into Vega-Lite (
        ``"binned"`` ).


        If ``true``, default `binning parameters
        <https://vega.github.io/vega-lite/docs/bin.html>`__ will be applied.

        If ``"binned"``, this indicates that the data for the ``x`` (or ``y`` ) channel are
        already binned. You can map the bin-start field to ``x`` (or ``y`` ) and the bin-end
        field to ``x2`` (or ``y2`` ). The scale and axis will be formatted similar to
        binning in Vega-Lite.  To adjust the axis ticks based on the bin step, you can also
        set the axis's `tickMinStep
        <https://vega.github.io/vega-lite/docs/axis.html#ticks>`__ property.

        **Default value:** ``false``

        **See also:** `bin <https://vega.github.io/vega-lite/docs/bin.html>`__
        documentation.
    condition : anyOf(:class:`ConditionalValueDefnumberExprRef`,
    List(:class:`ConditionalValueDefnumberExprRef`))
        One or more value definition(s) with `a selection or a test predicate
        <https://vega.github.io/vega-lite/docs/condition.html>`__.

        **Note:** A field definition's ``condition`` property can only contain `conditional
        value definitions <https://vega.github.io/vega-lite/docs/condition.html#value>`__
        since Vega-Lite only allows at most one encoded field per encoding channel.
    field : :class:`Field`
        **Required.** A string defining the name of the field from which to pull a data
        value or an object defining iterated values from the `repeat
        <https://vega.github.io/vega-lite/docs/repeat.html>`__ operator.

        **See also:** `field <https://vega.github.io/vega-lite/docs/field.html>`__
        documentation.

        **Notes:** 1)  Dots ( ``.`` ) and brackets ( ``[`` and ``]`` ) can be used to access
        nested objects (e.g., ``"field": "foo.bar"`` and ``"field": "foo['bar']"`` ). If
        field names contain dots or brackets but are not nested, you can use ``\\`` to
        escape dots and brackets (e.g., ``"a\\.b"`` and ``"a\\[0\\]"`` ). See more details
        about escaping in the `field documentation
        <https://vega.github.io/vega-lite/docs/field.html>`__. 2) ``field`` is not required
        if ``aggregate`` is ``count``.
    legend : anyOf(:class:`Legend`, None)
        An object defining properties of the legend. If ``null``, the legend for the
        encoding channel will be removed.

        **Default value:** If undefined, default `legend properties
        <https://vega.github.io/vega-lite/docs/legend.html>`__ are applied.

        **See also:** `legend <https://vega.github.io/vega-lite/docs/legend.html>`__
        documentation.
    scale : anyOf(:class:`Scale`, None)
        An object defining properties of the channel's scale, which is the function that
        transforms values in the data domain (numbers, dates, strings, etc) to visual values
        (pixels, colors, sizes) of the encoding channels.

        If ``null``, the scale will be `disabled and the data value will be directly encoded
        <https://vega.github.io/vega-lite/docs/scale.html#disable>`__.

        **Default value:** If undefined, default `scale properties
        <https://vega.github.io/vega-lite/docs/scale.html>`__ are applied.

        **See also:** `scale <https://vega.github.io/vega-lite/docs/scale.html>`__
        documentation.
    sort : :class:`Sort`
        Sort order for the encoded field.

        For continuous fields (quantitative or temporal), ``sort`` can be either
        ``"ascending"`` or ``"descending"``.

        For discrete fields, ``sort`` can be one of the following: - ``"ascending"`` or
        ``"descending"`` -- for sorting by the values' natural order in JavaScript. - `A
        string indicating an encoding channel name to sort by
        <https://vega.github.io/vega-lite/docs/sort.html#sort-by-encoding>`__ (e.g., ``"x"``
        or ``"y"`` ) with an optional minus prefix for descending sort (e.g., ``"-x"`` to
        sort by x-field, descending). This channel string is short-form of `a
        sort-by-encoding definition
        <https://vega.github.io/vega-lite/docs/sort.html#sort-by-encoding>`__. For example,
        ``"sort": "-x"`` is equivalent to ``"sort": {"encoding": "x", "order":
        "descending"}``. - `A sort field definition
        <https://vega.github.io/vega-lite/docs/sort.html#sort-field>`__ for sorting by
        another field. - `An array specifying the field values in preferred order
        <https://vega.github.io/vega-lite/docs/sort.html#sort-array>`__. In this case, the
        sort order will obey the values in the array, followed by any unspecified values in
        their original order. For discrete time field, values in the sort array can be
        `date-time definition objects <types#datetime>`__. In addition, for time units
        ``"month"`` and ``"day"``, the values can be the month or day names (case
        insensitive) or their 3-letter initials (e.g., ``"Mon"``, ``"Tue"`` ). - ``null``
        indicating no sort.

        **Default value:** ``"ascending"``

        **Note:** ``null`` and sorting by another channel is not supported for ``row`` and
        ``column``.

        **See also:** `sort <https://vega.github.io/vega-lite/docs/sort.html>`__
        documentation.
    timeUnit : anyOf(:class:`TimeUnit`, :class:`TimeUnitParams`)
        Time unit (e.g., ``year``, ``yearmonth``, ``month``, ``hours`` ) for a temporal
        field. or `a temporal field that gets casted as ordinal
        <https://vega.github.io/vega-lite/docs/type.html#cast>`__.

        **Default value:** ``undefined`` (None)

        **See also:** `timeUnit <https://vega.github.io/vega-lite/docs/timeunit.html>`__
        documentation.
    title : anyOf(:class:`Text`, None)
        A title for the field. If ``null``, the title will be removed.

        **Default value:**  derived from the field's name and transformation function (
        ``aggregate``, ``bin`` and ``timeUnit`` ). If the field has an aggregate function,
        the function is displayed as part of the title (e.g., ``"Sum of Profit"`` ). If the
        field is binned or has a time unit applied, the applied function is shown in
        parentheses (e.g., ``"Profit (binned)"``, ``"Transaction Date (year-month)"`` ).
        Otherwise, the title is simply the field name.

        **Notes** :

        1) You can customize the default field title format by providing the `fieldTitle
        <https://vega.github.io/vega-lite/docs/config.html#top-level-config>`__ property in
        the `config <https://vega.github.io/vega-lite/docs/config.html>`__ or `fieldTitle
        function via the compile function's options
        <https://vega.github.io/vega-lite/docs/compile.html#field-title>`__.

        2) If both field definition's ``title`` and axis, header, or legend ``title`` are
        defined, axis/header/legend title will be used.
    type : :class:`StandardType`
        The type of measurement ( ``"quantitative"``, ``"temporal"``, ``"ordinal"``, or
        ``"nominal"`` ) for the encoded field or constant value ( ``datum`` ). It can also
        be a ``"geojson"`` type for encoding `'geoshape'
        <https://vega.github.io/vega-lite/docs/geoshape.html>`__.

        Vega-Lite automatically infers data types in many cases as discussed below. However,
        type is required for a field if: (1) the field is not nominal and the field encoding
        has no specified ``aggregate`` (except ``argmin`` and ``argmax`` ), ``bin``, scale
        type, custom ``sort`` order, nor ``timeUnit`` or (2) if you wish to use an ordinal
        scale for a field with ``bin`` or ``timeUnit``.

        **Default value:**

        1) For a data ``field``, ``"nominal"`` is the default data type unless the field
        encoding has ``aggregate``, ``channel``, ``bin``, scale type, ``sort``, or
        ``timeUnit`` that satisfies the following criteria: - ``"quantitative"`` is the
        default type if (1) the encoded field contains ``bin`` or ``aggregate`` except
        ``"argmin"`` and ``"argmax"``, (2) the encoding channel is ``latitude`` or
        ``longitude`` channel or (3) if the specified scale type is `a quantitative scale
        <https://vega.github.io/vega-lite/docs/scale.html#type>`__. - ``"temporal"`` is the
        default type if (1) the encoded field contains ``timeUnit`` or (2) the specified
        scale type is a time or utc scale - ``ordinal""`` is the default type if (1) the
        encoded field contains a `custom sort order
        <https://vega.github.io/vega-lite/docs/sort.html#specifying-custom-sort-order>`__,
        (2) the specified scale type is an ordinal/point/band scale, or (3) the encoding
        channel is ``order``.

        2) For a constant value in data domain ( ``datum`` ): - ``"quantitative"`` if the
        datum is a number - ``"nominal"`` if the datum is a string - ``"temporal"`` if the
        datum is `a date time object
        <https://vega.github.io/vega-lite/docs/datetime.html>`__

        **Note:** - Data ``type`` describes the semantics of the data rather than the
        primitive data types (number, string, etc.). The same primitive data type can have
        different types of measurement. For example, numeric data can represent
        quantitative, ordinal, or nominal data. - Data values for a temporal field can be
        either a date-time string (e.g., ``"2015-03-07 12:32:17"``, ``"17:01"``,
        ``"2015-03-16"``. ``"2015"`` ) or a timestamp number (e.g., ``1552199579097`` ). -
        When using with `bin <https://vega.github.io/vega-lite/docs/bin.html>`__, the
        ``type`` property can be either ``"quantitative"`` (for using a linear bin scale) or
        `"ordinal" (for using an ordinal bin scale)
        <https://vega.github.io/vega-lite/docs/type.html#cast-bin>`__. - When using with
        `timeUnit <https://vega.github.io/vega-lite/docs/timeunit.html>`__, the ``type``
        property can be either ``"temporal"`` (default, for using a temporal scale) or
        `"ordinal" (for using an ordinal scale)
        <https://vega.github.io/vega-lite/docs/type.html#cast-bin>`__. - When using with
        `aggregate <https://vega.github.io/vega-lite/docs/aggregate.html>`__, the ``type``
        property refers to the post-aggregation data type. For example, we can calculate
        count ``distinct`` of a categorical field ``"cat"`` using ``{"aggregate":
        "distinct", "field": "cat"}``. The ``"type"`` of the aggregate output is
        ``"quantitative"``. - Secondary channels (e.g., ``x2``, ``y2``, ``xError``,
        ``yError`` ) do not have ``type`` as they must have exactly the same type as their
        primary channels (e.g., ``x``, ``y`` ).

        **See also:** `type <https://vega.github.io/vega-lite/docs/type.html>`__
        documentation.
    """
    _schema = {'$ref': '#/definitions/FieldOrDatumDefWithCondition<MarkPropFieldDef,number>'}

    def __init__(self, aggregate=Undefined, band=Undefined, bin=Undefined, condition=Undefined,
                 field=Undefined, legend=Undefined, scale=Undefined, sort=Undefined, timeUnit=Undefined,
                 title=Undefined, type=Undefined, **kwds):
        super(FieldOrDatumDefWithConditionMarkPropFieldDefnumber, self).__init__(aggregate=aggregate,
                                                                                 band=band, bin=bin,
                                                                                 condition=condition,
                                                                                 field=field,
                                                                                 legend=legend,
                                                                                 scale=scale, sort=sort,
                                                                                 timeUnit=timeUnit,
                                                                                 title=title, type=type,
                                                                                 **kwds)


class NumericValueDef(LatLongDef):
    """NumericValueDef schema wrapper

    Mapping(required=[value])
    Definition object for a constant value (primitive value or gradient definition) of an
    encoding channel.

    Attributes
    ----------

    value : anyOf(float, :class:`ExprRef`)
        A constant value in visual domain (e.g., ``"red"`` / ``"#0099ff"`` / `gradient
        definition <https://vega.github.io/vega-lite/docs/types.html#gradient>`__ for color,
        values between ``0`` to ``1`` for opacity).
    """
    _schema = {'$ref': '#/definitions/NumericValueDef'}

    def __init__(self, value=Undefined, **kwds):
        super(NumericValueDef, self).__init__(value=value, **kwds)


class OrderFieldDef(VegaLiteSchema):
    """OrderFieldDef schema wrapper

    Mapping(required=[])

    Attributes
    ----------

    aggregate : :class:`Aggregate`
        Aggregation function for the field (e.g., ``"mean"``, ``"sum"``, ``"median"``,
        ``"min"``, ``"max"``, ``"count"`` ).

        **Default value:** ``undefined`` (None)

        **See also:** `aggregate <https://vega.github.io/vega-lite/docs/aggregate.html>`__
        documentation.
    band : float
        For rect-based marks ( ``rect``, ``bar``, and ``image`` ), mark size relative to
        bandwidth of `band scales
        <https://vega.github.io/vega-lite/docs/scale.html#band>`__, bins or time units. If
        set to ``1``, the mark size is set to the bandwidth, the bin interval, or the time
        unit interval. If set to ``0.5``, the mark size is half of the bandwidth or the time
        unit interval.

        For other marks, relative position on a band of a stacked, binned, time unit or band
        scale. If set to ``0``, the marks will be positioned at the beginning of the band.
        If set to ``0.5``, the marks will be positioned in the middle of the band.
    bin : anyOf(boolean, :class:`BinParams`, string, None)
        A flag for binning a ``quantitative`` field, `an object defining binning parameters
        <https://vega.github.io/vega-lite/docs/bin.html#params>`__, or indicating that the
        data for ``x`` or ``y`` channel are binned before they are imported into Vega-Lite (
        ``"binned"`` ).


        If ``true``, default `binning parameters
        <https://vega.github.io/vega-lite/docs/bin.html>`__ will be applied.

        If ``"binned"``, this indicates that the data for the ``x`` (or ``y`` ) channel are
        already binned. You can map the bin-start field to ``x`` (or ``y`` ) and the bin-end
        field to ``x2`` (or ``y2`` ). The scale and axis will be formatted similar to
        binning in Vega-Lite.  To adjust the axis ticks based on the bin step, you can also
        set the axis's `tickMinStep
        <https://vega.github.io/vega-lite/docs/axis.html#ticks>`__ property.

        **Default value:** ``false``

        **See also:** `bin <https://vega.github.io/vega-lite/docs/bin.html>`__
        documentation.
    field : :class:`Field`
        **Required.** A string defining the name of the field from which to pull a data
        value or an object defining iterated values from the `repeat
        <https://vega.github.io/vega-lite/docs/repeat.html>`__ operator.

        **See also:** `field <https://vega.github.io/vega-lite/docs/field.html>`__
        documentation.

        **Notes:** 1)  Dots ( ``.`` ) and brackets ( ``[`` and ``]`` ) can be used to access
        nested objects (e.g., ``"field": "foo.bar"`` and ``"field": "foo['bar']"`` ). If
        field names contain dots or brackets but are not nested, you can use ``\\`` to
        escape dots and brackets (e.g., ``"a\\.b"`` and ``"a\\[0\\]"`` ). See more details
        about escaping in the `field documentation
        <https://vega.github.io/vega-lite/docs/field.html>`__. 2) ``field`` is not required
        if ``aggregate`` is ``count``.
    sort : :class:`SortOrder`
        The sort order. One of ``"ascending"`` (default) or ``"descending"``.
    timeUnit : anyOf(:class:`TimeUnit`, :class:`TimeUnitParams`)
        Time unit (e.g., ``year``, ``yearmonth``, ``month``, ``hours`` ) for a temporal
        field. or `a temporal field that gets casted as ordinal
        <https://vega.github.io/vega-lite/docs/type.html#cast>`__.

        **Default value:** ``undefined`` (None)

        **See also:** `timeUnit <https://vega.github.io/vega-lite/docs/timeunit.html>`__
        documentation.
    title : anyOf(:class:`Text`, None)
        A title for the field. If ``null``, the title will be removed.

        **Default value:**  derived from the field's name and transformation function (
        ``aggregate``, ``bin`` and ``timeUnit`` ). If the field has an aggregate function,
        the function is displayed as part of the title (e.g., ``"Sum of Profit"`` ). If the
        field is binned or has a time unit applied, the applied function is shown in
        parentheses (e.g., ``"Profit (binned)"``, ``"Transaction Date (year-month)"`` ).
        Otherwise, the title is simply the field name.

        **Notes** :

        1) You can customize the default field title format by providing the `fieldTitle
        <https://vega.github.io/vega-lite/docs/config.html#top-level-config>`__ property in
        the `config <https://vega.github.io/vega-lite/docs/config.html>`__ or `fieldTitle
        function via the compile function's options
        <https://vega.github.io/vega-lite/docs/compile.html#field-title>`__.

        2) If both field definition's ``title`` and axis, header, or legend ``title`` are
        defined, axis/header/legend title will be used.
    type : :class:`StandardType`
        The type of measurement ( ``"quantitative"``, ``"temporal"``, ``"ordinal"``, or
        ``"nominal"`` ) for the encoded field or constant value ( ``datum`` ). It can also
        be a ``"geojson"`` type for encoding `'geoshape'
        <https://vega.github.io/vega-lite/docs/geoshape.html>`__.

        Vega-Lite automatically infers data types in many cases as discussed below. However,
        type is required for a field if: (1) the field is not nominal and the field encoding
        has no specified ``aggregate`` (except ``argmin`` and ``argmax`` ), ``bin``, scale
        type, custom ``sort`` order, nor ``timeUnit`` or (2) if you wish to use an ordinal
        scale for a field with ``bin`` or ``timeUnit``.

        **Default value:**

        1) For a data ``field``, ``"nominal"`` is the default data type unless the field
        encoding has ``aggregate``, ``channel``, ``bin``, scale type, ``sort``, or
        ``timeUnit`` that satisfies the following criteria: - ``"quantitative"`` is the
        default type if (1) the encoded field contains ``bin`` or ``aggregate`` except
        ``"argmin"`` and ``"argmax"``, (2) the encoding channel is ``latitude`` or
        ``longitude`` channel or (3) if the specified scale type is `a quantitative scale
        <https://vega.github.io/vega-lite/docs/scale.html#type>`__. - ``"temporal"`` is the
        default type if (1) the encoded field contains ``timeUnit`` or (2) the specified
        scale type is a time or utc scale - ``ordinal""`` is the default type if (1) the
        encoded field contains a `custom sort order
        <https://vega.github.io/vega-lite/docs/sort.html#specifying-custom-sort-order>`__,
        (2) the specified scale type is an ordinal/point/band scale, or (3) the encoding
        channel is ``order``.

        2) For a constant value in data domain ( ``datum`` ): - ``"quantitative"`` if the
        datum is a number - ``"nominal"`` if the datum is a string - ``"temporal"`` if the
        datum is `a date time object
        <https://vega.github.io/vega-lite/docs/datetime.html>`__

        **Note:** - Data ``type`` describes the semantics of the data rather than the
        primitive data types (number, string, etc.). The same primitive data type can have
        different types of measurement. For example, numeric data can represent
        quantitative, ordinal, or nominal data. - Data values for a temporal field can be
        either a date-time string (e.g., ``"2015-03-07 12:32:17"``, ``"17:01"``,
        ``"2015-03-16"``. ``"2015"`` ) or a timestamp number (e.g., ``1552199579097`` ). -
        When using with `bin <https://vega.github.io/vega-lite/docs/bin.html>`__, the
        ``type`` property can be either ``"quantitative"`` (for using a linear bin scale) or
        `"ordinal" (for using an ordinal bin scale)
        <https://vega.github.io/vega-lite/docs/type.html#cast-bin>`__. - When using with
        `timeUnit <https://vega.github.io/vega-lite/docs/timeunit.html>`__, the ``type``
        property can be either ``"temporal"`` (default, for using a temporal scale) or
        `"ordinal" (for using an ordinal scale)
        <https://vega.github.io/vega-lite/docs/type.html#cast-bin>`__. - When using with
        `aggregate <https://vega.github.io/vega-lite/docs/aggregate.html>`__, the ``type``
        property refers to the post-aggregation data type. For example, we can calculate
        count ``distinct`` of a categorical field ``"cat"`` using ``{"aggregate":
        "distinct", "field": "cat"}``. The ``"type"`` of the aggregate output is
        ``"quantitative"``. - Secondary channels (e.g., ``x2``, ``y2``, ``xError``,
        ``yError`` ) do not have ``type`` as they must have exactly the same type as their
        primary channels (e.g., ``x``, ``y`` ).

        **See also:** `type <https://vega.github.io/vega-lite/docs/type.html>`__
        documentation.
    """
    _schema = {'$ref': '#/definitions/OrderFieldDef'}

    def __init__(self, aggregate=Undefined, band=Undefined, bin=Undefined, field=Undefined,
                 sort=Undefined, timeUnit=Undefined, title=Undefined, type=Undefined, **kwds):
        super(OrderFieldDef, self).__init__(aggregate=aggregate, band=band, bin=bin, field=field,
                                            sort=sort, timeUnit=timeUnit, title=title, type=type, **kwds)


class OrderValueDef(VegaLiteSchema):
    """OrderValueDef schema wrapper

    Mapping(required=[value])

    Attributes
    ----------

    value : anyOf(float, :class:`ExprRef`)
        A constant value in visual domain (e.g., ``"red"`` / ``"#0099ff"`` / `gradient
        definition <https://vega.github.io/vega-lite/docs/types.html#gradient>`__ for color,
        values between ``0`` to ``1`` for opacity).
    condition : anyOf(:class:`ConditionalValueDefnumber`,
    List(:class:`ConditionalValueDefnumber`))
        One or more value definition(s) with `a selection or a test predicate
        <https://vega.github.io/vega-lite/docs/condition.html>`__.

        **Note:** A field definition's ``condition`` property can only contain `conditional
        value definitions <https://vega.github.io/vega-lite/docs/condition.html#value>`__
        since Vega-Lite only allows at most one encoded field per encoding channel.
    """
    _schema = {'$ref': '#/definitions/OrderValueDef'}

    def __init__(self, value=Undefined, condition=Undefined, **kwds):
        super(OrderValueDef, self).__init__(value=value, condition=condition, **kwds)


class Orient(VegaLiteSchema):
    """Orient schema wrapper

    enum('left', 'right', 'top', 'bottom')
    """
    _schema = {'$ref': '#/definitions/Orient'}

    def __init__(self, *args):
        super(Orient, self).__init__(*args)


class Orientation(VegaLiteSchema):
    """Orientation schema wrapper

    enum('horizontal', 'vertical')
    """
    _schema = {'$ref': '#/definitions/Orientation'}

    def __init__(self, *args):
        super(Orientation, self).__init__(*args)


class OverlayMarkDef(VegaLiteSchema):
    """OverlayMarkDef schema wrapper

    Mapping(required=[])

    Attributes
    ----------

    align : anyOf(:class:`Align`, :class:`ExprRef`)
        The horizontal alignment of the text or ranged marks (area, bar, image, rect, rule).
        One of ``"left"``, ``"right"``, ``"center"``.

        **Note:** Expression reference is *not* supported for range marks.
    angle : anyOf(float, :class:`ExprRef`)

    aria : anyOf(boolean, :class:`ExprRef`)

    ariaRole : anyOf(string, :class:`ExprRef`)

    ariaRoleDescription : anyOf(string, :class:`ExprRef`)

    aspect : anyOf(boolean, :class:`ExprRef`)

    baseline : anyOf(:class:`TextBaseline`, :class:`ExprRef`)
        For text marks, the vertical text baseline. One of ``"alphabetic"`` (default),
        ``"top"``, ``"middle"``, ``"bottom"``, ``"line-top"``, ``"line-bottom"``, or an
        expression reference that provides one of the valid values. The ``"line-top"`` and
        ``"line-bottom"`` values operate similarly to ``"top"`` and ``"bottom"``, but are
        calculated relative to the ``lineHeight`` rather than ``fontSize`` alone.

        For range marks, the vertical alignment of the marks. One of ``"top"``,
        ``"middle"``, ``"bottom"``.

        **Note:** Expression reference is *not* supported for range marks.
    blend : anyOf(:class:`Blend`, :class:`ExprRef`)

    clip : boolean
        Whether a mark be clipped to the enclosing group’s width and height.
    color : anyOf(:class:`Color`, :class:`Gradient`, :class:`ExprRef`)
        Default color.

        **Default value:** :raw-html:`<span style="color: #4682b4;">&#9632;</span>`
        ``"#4682b4"``

        **Note:** - This property cannot be used in a `style config
        <https://vega.github.io/vega-lite/docs/mark.html#style-config>`__. - The ``fill``
        and ``stroke`` properties have higher precedence than ``color`` and will override
        ``color``.
    cornerRadius : anyOf(float, :class:`ExprRef`)

    cornerRadiusBottomLeft : anyOf(float, :class:`ExprRef`)

    cornerRadiusBottomRight : anyOf(float, :class:`ExprRef`)

    cornerRadiusTopLeft : anyOf(float, :class:`ExprRef`)

    cornerRadiusTopRight : anyOf(float, :class:`ExprRef`)

    cursor : anyOf(:class:`Cursor`, :class:`ExprRef`)

    description : anyOf(string, :class:`ExprRef`)

    dir : anyOf(:class:`TextDirection`, :class:`ExprRef`)

    dx : anyOf(float, :class:`ExprRef`)

    dy : anyOf(float, :class:`ExprRef`)

    ellipsis : anyOf(string, :class:`ExprRef`)

    endAngle : anyOf(float, :class:`ExprRef`)

    fill : anyOf(:class:`Color`, :class:`Gradient`, None, :class:`ExprRef`)
        Default fill color. This property has higher precedence than ``config.color``. Set
        to ``null`` to remove fill.

        **Default value:** (None)
    fillOpacity : anyOf(float, :class:`ExprRef`)

    filled : boolean
        Whether the mark's color should be used as fill color instead of stroke color.

        **Default value:** ``false`` for all ``point``, ``line``, and ``rule`` marks as well
        as ``geoshape`` marks for `graticule
        <https://vega.github.io/vega-lite/docs/data.html#graticule>`__ data sources;
        otherwise, ``true``.

        **Note:** This property cannot be used in a `style config
        <https://vega.github.io/vega-lite/docs/mark.html#style-config>`__.
    font : anyOf(string, :class:`ExprRef`)

    fontSize : anyOf(float, :class:`ExprRef`)

    fontStyle : anyOf(:class:`FontStyle`, :class:`ExprRef`)

    fontWeight : anyOf(:class:`FontWeight`, :class:`ExprRef`)

    height : anyOf(float, :class:`ExprRef`)

    href : anyOf(:class:`URI`, :class:`ExprRef`)

    innerRadius : anyOf(float, :class:`ExprRef`)
        The inner radius in pixels of arc marks. ``innerRadius`` is an alias for
        ``radius2``.
    interpolate : anyOf(:class:`Interpolate`, :class:`ExprRef`)

    invalid : enum('filter', None)
        Defines how Vega-Lite should handle marks for invalid values ( ``null`` and ``NaN``
        ). - If set to ``"filter"`` (default), all data items with null values will be
        skipped (for line, trail, and area marks) or filtered (for other marks). - If
        ``null``, all data items are included. In this case, invalid values will be
        interpreted as zeroes.
    limit : anyOf(float, :class:`ExprRef`)

    lineBreak : anyOf(string, :class:`ExprRef`)

    lineHeight : anyOf(float, :class:`ExprRef`)

    opacity : anyOf(float, :class:`ExprRef`)
        The overall opacity (value between [0,1]).

        **Default value:** ``0.7`` for non-aggregate plots with ``point``, ``tick``,
        ``circle``, or ``square`` marks or layered ``bar`` charts and ``1`` otherwise.
    order : anyOf(None, boolean)
        For line and trail marks, this ``order`` property can be set to ``null`` or
        ``false`` to make the lines use the original order in the data sources.
    orient : :class:`Orientation`
        The orientation of a non-stacked bar, tick, area, and line charts. The value is
        either horizontal (default) or vertical. - For bar, rule and tick, this determines
        whether the size of the bar and tick should be applied to x or y dimension. - For
        area, this property determines the orient property of the Vega output. - For line
        and trail marks, this property determines the sort order of the points in the line
        if ``config.sortLineBy`` is not specified. For stacked charts, this is always
        determined by the orientation of the stack; therefore explicitly specified value
        will be ignored.
    outerRadius : anyOf(float, :class:`ExprRef`)
        The outer radius in pixels of arc marks. ``outerRadius`` is an alias for ``radius``.
    padAngle : anyOf(float, :class:`ExprRef`)

    radius : anyOf(float, :class:`ExprRef`)
        For arc mark, the primary (outer) radius in pixels.

        For text marks, polar coordinate radial offset, in pixels, of the text from the
        origin determined by the ``x`` and ``y`` properties.
    radius2 : anyOf(float, :class:`ExprRef`)
        The secondary (inner) radius in pixels of arc marks.
    radius2Offset : anyOf(float, :class:`ExprRef`)
        Offset for radius2.
    radiusOffset : anyOf(float, :class:`ExprRef`)
        Offset for radius.
    shape : anyOf(anyOf(:class:`SymbolShape`, string), :class:`ExprRef`)

    size : anyOf(float, :class:`ExprRef`)
        Default size for marks. - For ``point`` / ``circle`` / ``square``, this represents
        the pixel area of the marks. Note that this value sets the area of the symbol; the
        side lengths will increase with the square root of this value. - For ``bar``, this
        represents the band size of the bar, in pixels. - For ``text``, this represents the
        font size, in pixels.

        **Default value:** - ``30`` for point, circle, square marks; width/height's ``step``
        - ``2`` for bar marks with discrete dimensions; - ``5`` for bar marks with
        continuous dimensions; - ``11`` for text marks.
    smooth : anyOf(boolean, :class:`ExprRef`)

    startAngle : anyOf(float, :class:`ExprRef`)

    stroke : anyOf(:class:`Color`, :class:`Gradient`, None, :class:`ExprRef`)
        Default stroke color. This property has higher precedence than ``config.color``. Set
        to ``null`` to remove stroke.

        **Default value:** (None)
    strokeCap : anyOf(:class:`StrokeCap`, :class:`ExprRef`)

    strokeDash : anyOf(List(float), :class:`ExprRef`)

    strokeDashOffset : anyOf(float, :class:`ExprRef`)

    strokeJoin : anyOf(:class:`StrokeJoin`, :class:`ExprRef`)

    strokeMiterLimit : anyOf(float, :class:`ExprRef`)

    strokeOffset : anyOf(float, :class:`ExprRef`)

    strokeOpacity : anyOf(float, :class:`ExprRef`)

    strokeWidth : anyOf(float, :class:`ExprRef`)

    style : anyOf(string, List(string))
        A string or array of strings indicating the name of custom styles to apply to the
        mark. A style is a named collection of mark property defaults defined within the
        `style configuration
        <https://vega.github.io/vega-lite/docs/mark.html#style-config>`__. If style is an
        array, later styles will override earlier styles. Any `mark properties
        <https://vega.github.io/vega-lite/docs/encoding.html#mark-prop>`__ explicitly
        defined within the ``encoding`` will override a style default.

        **Default value:** The mark's name. For example, a bar mark will have style
        ``"bar"`` by default. **Note:** Any specified style will augment the default style.
        For example, a bar mark with ``"style": "foo"`` will receive from
        ``config.style.bar`` and ``config.style.foo`` (the specified style ``"foo"`` has
        higher precedence).
    tension : anyOf(float, :class:`ExprRef`)

    text : anyOf(:class:`Text`, :class:`ExprRef`)

    theta : anyOf(float, :class:`ExprRef`)
        For arc marks, the arc length in radians if theta2 is not specified, otherwise the
        start arc angle. (A value of 0 indicates up or “north”, increasing values proceed
        clockwise.)

        For text marks, polar coordinate angle in radians.
    theta2 : anyOf(float, :class:`ExprRef`)
        The end angle of arc marks in radians. A value of 0 indicates up or “north”,
        increasing values proceed clockwise.
    theta2Offset : anyOf(float, :class:`ExprRef`)
        Offset for theta2.
    thetaOffset : anyOf(float, :class:`ExprRef`)
        Offset for theta.
    timeUnitBand : float
        Default relative band size for a time unit. If set to ``1``, the bandwidth of the
        marks will be equal to the time unit band step. If set to ``0.5``, bandwidth of the
        marks will be half of the time unit band step.
    timeUnitBandPosition : float
        Default relative band position for a time unit. If set to ``0``, the marks will be
        positioned at the beginning of the time unit band step. If set to ``0.5``, the marks
        will be positioned in the middle of the time unit band step.
    tooltip : anyOf(float, string, boolean, :class:`TooltipContent`, :class:`ExprRef`, None)
        The tooltip text string to show upon mouse hover or an object defining which fields
        should the tooltip be derived from.


        * If ``tooltip`` is ``true`` or ``{"content": "encoding"}``, then all fields from
          ``encoding`` will be used. - If ``tooltip`` is ``{"content": "data"}``, then all
          fields that appear in the highlighted data point will be used. - If set to
          ``null`` or ``false``, then no tooltip will be used.

        See the `tooltip <https://vega.github.io/vega-lite/docs/tooltip.html>`__
        documentation for a detailed discussion about tooltip  in Vega-Lite.

        **Default value:** ``null``
    url : anyOf(:class:`URI`, :class:`ExprRef`)

    width : anyOf(float, :class:`ExprRef`)

    x : anyOf(float, string, :class:`ExprRef`)
        X coordinates of the marks, or width of horizontal ``"bar"`` and ``"area"`` without
        specified ``x2`` or ``width``.

        The ``value`` of this channel can be a number or a string ``"width"`` for the width
        of the plot.
    x2 : anyOf(float, string, :class:`ExprRef`)
        X2 coordinates for ranged ``"area"``, ``"bar"``, ``"rect"``, and  ``"rule"``.

        The ``value`` of this channel can be a number or a string ``"width"`` for the width
        of the plot.
    x2Offset : anyOf(float, :class:`ExprRef`)
        Offset for x2-position.
    xOffset : anyOf(float, :class:`ExprRef`)
        Offset for x-position.
    y : anyOf(float, string, :class:`ExprRef`)
        Y coordinates of the marks, or height of vertical ``"bar"`` and ``"area"`` without
        specified ``y2`` or ``height``.

        The ``value`` of this channel can be a number or a string ``"height"`` for the
        height of the plot.
    y2 : anyOf(float, string, :class:`ExprRef`)
        Y2 coordinates for ranged ``"area"``, ``"bar"``, ``"rect"``, and  ``"rule"``.

        The ``value`` of this channel can be a number or a string ``"height"`` for the
        height of the plot.
    y2Offset : anyOf(float, :class:`ExprRef`)
        Offset for y2-position.
    yOffset : anyOf(float, :class:`ExprRef`)
        Offset for y-position.
    """
    _schema = {'$ref': '#/definitions/OverlayMarkDef'}

    def __init__(self, align=Undefined, angle=Undefined, aria=Undefined, ariaRole=Undefined,
                 ariaRoleDescription=Undefined, aspect=Undefined, baseline=Undefined, blend=Undefined,
                 clip=Undefined, color=Undefined, cornerRadius=Undefined,
                 cornerRadiusBottomLeft=Undefined, cornerRadiusBottomRight=Undefined,
                 cornerRadiusTopLeft=Undefined, cornerRadiusTopRight=Undefined, cursor=Undefined,
                 description=Undefined, dir=Undefined, dx=Undefined, dy=Undefined, ellipsis=Undefined,
                 endAngle=Undefined, fill=Undefined, fillOpacity=Undefined, filled=Undefined,
                 font=Undefined, fontSize=Undefined, fontStyle=Undefined, fontWeight=Undefined,
                 height=Undefined, href=Undefined, innerRadius=Undefined, interpolate=Undefined,
                 invalid=Undefined, limit=Undefined, lineBreak=Undefined, lineHeight=Undefined,
                 opacity=Undefined, order=Undefined, orient=Undefined, outerRadius=Undefined,
                 padAngle=Undefined, radius=Undefined, radius2=Undefined, radius2Offset=Undefined,
                 radiusOffset=Undefined, shape=Undefined, size=Undefined, smooth=Undefined,
                 startAngle=Undefined, stroke=Undefined, strokeCap=Undefined, strokeDash=Undefined,
                 strokeDashOffset=Undefined, strokeJoin=Undefined, strokeMiterLimit=Undefined,
                 strokeOffset=Undefined, strokeOpacity=Undefined, strokeWidth=Undefined,
                 style=Undefined, tension=Undefined, text=Undefined, theta=Undefined, theta2=Undefined,
                 theta2Offset=Undefined, thetaOffset=Undefined, timeUnitBand=Undefined,
                 timeUnitBandPosition=Undefined, tooltip=Undefined, url=Undefined, width=Undefined,
                 x=Undefined, x2=Undefined, x2Offset=Undefined, xOffset=Undefined, y=Undefined,
                 y2=Undefined, y2Offset=Undefined, yOffset=Undefined, **kwds):
        super(OverlayMarkDef, self).__init__(align=align, angle=angle, aria=aria, ariaRole=ariaRole,
                                             ariaRoleDescription=ariaRoleDescription, aspect=aspect,
                                             baseline=baseline, blend=blend, clip=clip, color=color,
                                             cornerRadius=cornerRadius,
                                             cornerRadiusBottomLeft=cornerRadiusBottomLeft,
                                             cornerRadiusBottomRight=cornerRadiusBottomRight,
                                             cornerRadiusTopLeft=cornerRadiusTopLeft,
                                             cornerRadiusTopRight=cornerRadiusTopRight, cursor=cursor,
                                             description=description, dir=dir, dx=dx, dy=dy,
                                             ellipsis=ellipsis, endAngle=endAngle, fill=fill,
                                             fillOpacity=fillOpacity, filled=filled, font=font,
                                             fontSize=fontSize, fontStyle=fontStyle,
                                             fontWeight=fontWeight, height=height, href=href,
                                             innerRadius=innerRadius, interpolate=interpolate,
                                             invalid=invalid, limit=limit, lineBreak=lineBreak,
                                             lineHeight=lineHeight, opacity=opacity, order=order,
                                             orient=orient, outerRadius=outerRadius, padAngle=padAngle,
                                             radius=radius, radius2=radius2,
                                             radius2Offset=radius2Offset, radiusOffset=radiusOffset,
                                             shape=shape, size=size, smooth=smooth,
                                             startAngle=startAngle, stroke=stroke, strokeCap=strokeCap,
                                             strokeDash=strokeDash, strokeDashOffset=strokeDashOffset,
                                             strokeJoin=strokeJoin, strokeMiterLimit=strokeMiterLimit,
                                             strokeOffset=strokeOffset, strokeOpacity=strokeOpacity,
                                             strokeWidth=strokeWidth, style=style, tension=tension,
                                             text=text, theta=theta, theta2=theta2,
                                             theta2Offset=theta2Offset, thetaOffset=thetaOffset,
                                             timeUnitBand=timeUnitBand,
                                             timeUnitBandPosition=timeUnitBandPosition, tooltip=tooltip,
                                             url=url, width=width, x=x, x2=x2, x2Offset=x2Offset,
                                             xOffset=xOffset, y=y, y2=y2, y2Offset=y2Offset,
                                             yOffset=yOffset, **kwds)


class Padding(VegaLiteSchema):
    """Padding schema wrapper

    anyOf(float, Mapping(required=[]))
    """
    _schema = {'$ref': '#/definitions/Padding'}

    def __init__(self, *args, **kwds):
        super(Padding, self).__init__(*args, **kwds)


class Parameter(VegaLiteSchema):
    """Parameter schema wrapper

    Mapping(required=[name])

    Attributes
    ----------

    name : string
        Required. A unique name for the parameter. Parameter names should be valid
        JavaScript identifiers: they should contain only alphanumeric characters (or “$”, or
        “_”) and may not start with a digit. Reserved keywords that may not be used as
        parameter names are "datum", "event", "item", and "parent".
    bind : :class:`Binding`
        Binds the parameter to an external input element such as a slider, selection list or
        radio button group.
    description : string
        A text description of the parameter, useful for inline documentation.
    expr : :class:`Expr`
        An expression for the value of the parameter. This expression may include other
        parameters, in which case the parameter will automatically update in response to
        upstream parameter changes.
    value : Any
        The initial value of the parameter.

        **Default value:** ``undefined``
    """
    _schema = {'$ref': '#/definitions/Parameter'}

    def __init__(self, name=Undefined, bind=Undefined, description=Undefined, expr=Undefined,
                 value=Undefined, **kwds):
        super(Parameter, self).__init__(name=name, bind=bind, description=description, expr=expr,
                                        value=value, **kwds)


class Parse(VegaLiteSchema):
    """Parse schema wrapper

    Mapping(required=[])
    """
    _schema = {'$ref': '#/definitions/Parse'}

    def __init__(self, **kwds):
        super(Parse, self).__init__(**kwds)


class ParseValue(VegaLiteSchema):
    """ParseValue schema wrapper

    anyOf(None, string, string, string, string, string)
    """
    _schema = {'$ref': '#/definitions/ParseValue'}

    def __init__(self, *args, **kwds):
        super(ParseValue, self).__init__(*args, **kwds)


class PolarDef(VegaLiteSchema):
    """PolarDef schema wrapper

    anyOf(:class:`PositionFieldDefBase`, :class:`PositionDatumDefBase`,
    :class:`PositionValueDef`)
    """
    _schema = {'$ref': '#/definitions/PolarDef'}

    def __init__(self, *args, **kwds):
        super(PolarDef, self).__init__(*args, **kwds)


class Position2Def(VegaLiteSchema):
    """Position2Def schema wrapper

    anyOf(:class:`SecondaryFieldDef`, :class:`DatumDef`, :class:`PositionValueDef`)
    """
    _schema = {'$ref': '#/definitions/Position2Def'}

    def __init__(self, *args, **kwds):
        super(Position2Def, self).__init__(*args, **kwds)


class DatumDef(LatLongDef, Position2Def):
    """DatumDef schema wrapper

    Mapping(required=[])

    Attributes
    ----------

    band : float
        For rect-based marks ( ``rect``, ``bar``, and ``image`` ), mark size relative to
        bandwidth of `band scales
        <https://vega.github.io/vega-lite/docs/scale.html#band>`__, bins or time units. If
        set to ``1``, the mark size is set to the bandwidth, the bin interval, or the time
        unit interval. If set to ``0.5``, the mark size is half of the bandwidth or the time
        unit interval.

        For other marks, relative position on a band of a stacked, binned, time unit or band
        scale. If set to ``0``, the marks will be positioned at the beginning of the band.
        If set to ``0.5``, the marks will be positioned in the middle of the band.
    datum : anyOf(:class:`PrimitiveValue`, :class:`DateTime`, :class:`ExprRef`,
    :class:`RepeatRef`)
        A constant value in data domain.
    type : :class:`Type`
        The type of measurement ( ``"quantitative"``, ``"temporal"``, ``"ordinal"``, or
        ``"nominal"`` ) for the encoded field or constant value ( ``datum`` ). It can also
        be a ``"geojson"`` type for encoding `'geoshape'
        <https://vega.github.io/vega-lite/docs/geoshape.html>`__.

        Vega-Lite automatically infers data types in many cases as discussed below. However,
        type is required for a field if: (1) the field is not nominal and the field encoding
        has no specified ``aggregate`` (except ``argmin`` and ``argmax`` ), ``bin``, scale
        type, custom ``sort`` order, nor ``timeUnit`` or (2) if you wish to use an ordinal
        scale for a field with ``bin`` or ``timeUnit``.

        **Default value:**

        1) For a data ``field``, ``"nominal"`` is the default data type unless the field
        encoding has ``aggregate``, ``channel``, ``bin``, scale type, ``sort``, or
        ``timeUnit`` that satisfies the following criteria: - ``"quantitative"`` is the
        default type if (1) the encoded field contains ``bin`` or ``aggregate`` except
        ``"argmin"`` and ``"argmax"``, (2) the encoding channel is ``latitude`` or
        ``longitude`` channel or (3) if the specified scale type is `a quantitative scale
        <https://vega.github.io/vega-lite/docs/scale.html#type>`__. - ``"temporal"`` is the
        default type if (1) the encoded field contains ``timeUnit`` or (2) the specified
        scale type is a time or utc scale - ``ordinal""`` is the default type if (1) the
        encoded field contains a `custom sort order
        <https://vega.github.io/vega-lite/docs/sort.html#specifying-custom-sort-order>`__,
        (2) the specified scale type is an ordinal/point/band scale, or (3) the encoding
        channel is ``order``.

        2) For a constant value in data domain ( ``datum`` ): - ``"quantitative"`` if the
        datum is a number - ``"nominal"`` if the datum is a string - ``"temporal"`` if the
        datum is `a date time object
        <https://vega.github.io/vega-lite/docs/datetime.html>`__

        **Note:** - Data ``type`` describes the semantics of the data rather than the
        primitive data types (number, string, etc.). The same primitive data type can have
        different types of measurement. For example, numeric data can represent
        quantitative, ordinal, or nominal data. - Data values for a temporal field can be
        either a date-time string (e.g., ``"2015-03-07 12:32:17"``, ``"17:01"``,
        ``"2015-03-16"``. ``"2015"`` ) or a timestamp number (e.g., ``1552199579097`` ). -
        When using with `bin <https://vega.github.io/vega-lite/docs/bin.html>`__, the
        ``type`` property can be either ``"quantitative"`` (for using a linear bin scale) or
        `"ordinal" (for using an ordinal bin scale)
        <https://vega.github.io/vega-lite/docs/type.html#cast-bin>`__. - When using with
        `timeUnit <https://vega.github.io/vega-lite/docs/timeunit.html>`__, the ``type``
        property can be either ``"temporal"`` (default, for using a temporal scale) or
        `"ordinal" (for using an ordinal scale)
        <https://vega.github.io/vega-lite/docs/type.html#cast-bin>`__. - When using with
        `aggregate <https://vega.github.io/vega-lite/docs/aggregate.html>`__, the ``type``
        property refers to the post-aggregation data type. For example, we can calculate
        count ``distinct`` of a categorical field ``"cat"`` using ``{"aggregate":
        "distinct", "field": "cat"}``. The ``"type"`` of the aggregate output is
        ``"quantitative"``. - Secondary channels (e.g., ``x2``, ``y2``, ``xError``,
        ``yError`` ) do not have ``type`` as they must have exactly the same type as their
        primary channels (e.g., ``x``, ``y`` ).

        **See also:** `type <https://vega.github.io/vega-lite/docs/type.html>`__
        documentation.
    """
    _schema = {'$ref': '#/definitions/DatumDef'}

    def __init__(self, band=Undefined, datum=Undefined, type=Undefined, **kwds):
        super(DatumDef, self).__init__(band=band, datum=datum, type=type, **kwds)


class PositionDatumDefBase(PolarDef):
    """PositionDatumDefBase schema wrapper

    Mapping(required=[])

    Attributes
    ----------

    band : float
        For rect-based marks ( ``rect``, ``bar``, and ``image`` ), mark size relative to
        bandwidth of `band scales
        <https://vega.github.io/vega-lite/docs/scale.html#band>`__, bins or time units. If
        set to ``1``, the mark size is set to the bandwidth, the bin interval, or the time
        unit interval. If set to ``0.5``, the mark size is half of the bandwidth or the time
        unit interval.

        For other marks, relative position on a band of a stacked, binned, time unit or band
        scale. If set to ``0``, the marks will be positioned at the beginning of the band.
        If set to ``0.5``, the marks will be positioned in the middle of the band.
    datum : anyOf(:class:`PrimitiveValue`, :class:`DateTime`, :class:`ExprRef`,
    :class:`RepeatRef`)
        A constant value in data domain.
    scale : anyOf(:class:`Scale`, None)
        An object defining properties of the channel's scale, which is the function that
        transforms values in the data domain (numbers, dates, strings, etc) to visual values
        (pixels, colors, sizes) of the encoding channels.

        If ``null``, the scale will be `disabled and the data value will be directly encoded
        <https://vega.github.io/vega-lite/docs/scale.html#disable>`__.

        **Default value:** If undefined, default `scale properties
        <https://vega.github.io/vega-lite/docs/scale.html>`__ are applied.

        **See also:** `scale <https://vega.github.io/vega-lite/docs/scale.html>`__
        documentation.
    stack : anyOf(:class:`StackOffset`, None, boolean)
        Type of stacking offset if the field should be stacked. ``stack`` is only applicable
        for ``x``, ``y``, ``theta``, and ``radius`` channels with continuous domains. For
        example, ``stack`` of ``y`` can be used to customize stacking for a vertical bar
        chart.

        ``stack`` can be one of the following values: - ``"zero"`` or `true`: stacking with
        baseline offset at zero value of the scale (for creating typical stacked
        [bar](https://vega.github.io/vega-lite/docs/stack.html#bar) and `area
        <https://vega.github.io/vega-lite/docs/stack.html#area>`__ chart). - ``"normalize"``
        - stacking with normalized domain (for creating `normalized stacked bar and area
        charts <https://vega.github.io/vega-lite/docs/stack.html#normalized>`__.
        :raw-html:`<br/>` - ``"center"`` - stacking with center baseline (for `streamgraph
        <https://vega.github.io/vega-lite/docs/stack.html#streamgraph>`__ ). - ``null`` or
        ``false`` - No-stacking. This will produce layered `bar
        <https://vega.github.io/vega-lite/docs/stack.html#layered-bar-chart>`__ and area
        chart.

        **Default value:** ``zero`` for plots with all of the following conditions are true:
        (1) the mark is ``bar``, ``area``, or ``arc`` ; (2) the stacked measure channel (x
        or y) has a linear scale; (3) At least one of non-position channels mapped to an
        unaggregated field that is different from x and y. Otherwise, ``null`` by default.

        **See also:** `stack <https://vega.github.io/vega-lite/docs/stack.html>`__
        documentation.
    type : :class:`Type`
        The type of measurement ( ``"quantitative"``, ``"temporal"``, ``"ordinal"``, or
        ``"nominal"`` ) for the encoded field or constant value ( ``datum`` ). It can also
        be a ``"geojson"`` type for encoding `'geoshape'
        <https://vega.github.io/vega-lite/docs/geoshape.html>`__.

        Vega-Lite automatically infers data types in many cases as discussed below. However,
        type is required for a field if: (1) the field is not nominal and the field encoding
        has no specified ``aggregate`` (except ``argmin`` and ``argmax`` ), ``bin``, scale
        type, custom ``sort`` order, nor ``timeUnit`` or (2) if you wish to use an ordinal
        scale for a field with ``bin`` or ``timeUnit``.

        **Default value:**

        1) For a data ``field``, ``"nominal"`` is the default data type unless the field
        encoding has ``aggregate``, ``channel``, ``bin``, scale type, ``sort``, or
        ``timeUnit`` that satisfies the following criteria: - ``"quantitative"`` is the
        default type if (1) the encoded field contains ``bin`` or ``aggregate`` except
        ``"argmin"`` and ``"argmax"``, (2) the encoding channel is ``latitude`` or
        ``longitude`` channel or (3) if the specified scale type is `a quantitative scale
        <https://vega.github.io/vega-lite/docs/scale.html#type>`__. - ``"temporal"`` is the
        default type if (1) the encoded field contains ``timeUnit`` or (2) the specified
        scale type is a time or utc scale - ``ordinal""`` is the default type if (1) the
        encoded field contains a `custom sort order
        <https://vega.github.io/vega-lite/docs/sort.html#specifying-custom-sort-order>`__,
        (2) the specified scale type is an ordinal/point/band scale, or (3) the encoding
        channel is ``order``.

        2) For a constant value in data domain ( ``datum`` ): - ``"quantitative"`` if the
        datum is a number - ``"nominal"`` if the datum is a string - ``"temporal"`` if the
        datum is `a date time object
        <https://vega.github.io/vega-lite/docs/datetime.html>`__

        **Note:** - Data ``type`` describes the semantics of the data rather than the
        primitive data types (number, string, etc.). The same primitive data type can have
        different types of measurement. For example, numeric data can represent
        quantitative, ordinal, or nominal data. - Data values for a temporal field can be
        either a date-time string (e.g., ``"2015-03-07 12:32:17"``, ``"17:01"``,
        ``"2015-03-16"``. ``"2015"`` ) or a timestamp number (e.g., ``1552199579097`` ). -
        When using with `bin <https://vega.github.io/vega-lite/docs/bin.html>`__, the
        ``type`` property can be either ``"quantitative"`` (for using a linear bin scale) or
        `"ordinal" (for using an ordinal bin scale)
        <https://vega.github.io/vega-lite/docs/type.html#cast-bin>`__. - When using with
        `timeUnit <https://vega.github.io/vega-lite/docs/timeunit.html>`__, the ``type``
        property can be either ``"temporal"`` (default, for using a temporal scale) or
        `"ordinal" (for using an ordinal scale)
        <https://vega.github.io/vega-lite/docs/type.html#cast-bin>`__. - When using with
        `aggregate <https://vega.github.io/vega-lite/docs/aggregate.html>`__, the ``type``
        property refers to the post-aggregation data type. For example, we can calculate
        count ``distinct`` of a categorical field ``"cat"`` using ``{"aggregate":
        "distinct", "field": "cat"}``. The ``"type"`` of the aggregate output is
        ``"quantitative"``. - Secondary channels (e.g., ``x2``, ``y2``, ``xError``,
        ``yError`` ) do not have ``type`` as they must have exactly the same type as their
        primary channels (e.g., ``x``, ``y`` ).

        **See also:** `type <https://vega.github.io/vega-lite/docs/type.html>`__
        documentation.
    """
    _schema = {'$ref': '#/definitions/PositionDatumDefBase'}

    def __init__(self, band=Undefined, datum=Undefined, scale=Undefined, stack=Undefined,
                 type=Undefined, **kwds):
        super(PositionDatumDefBase, self).__init__(band=band, datum=datum, scale=scale, stack=stack,
                                                   type=type, **kwds)


class PositionDef(VegaLiteSchema):
    """PositionDef schema wrapper

    anyOf(:class:`PositionFieldDef`, :class:`PositionDatumDef`, :class:`PositionValueDef`)
    """
    _schema = {'$ref': '#/definitions/PositionDef'}

    def __init__(self, *args, **kwds):
        super(PositionDef, self).__init__(*args, **kwds)


class PositionDatumDef(PositionDef):
    """PositionDatumDef schema wrapper

    Mapping(required=[])

    Attributes
    ----------

    axis : anyOf(:class:`Axis`, None)
        An object defining properties of axis's gridlines, ticks and labels. If ``null``,
        the axis for the encoding channel will be removed.

        **Default value:** If undefined, default `axis properties
        <https://vega.github.io/vega-lite/docs/axis.html>`__ are applied.

        **See also:** `axis <https://vega.github.io/vega-lite/docs/axis.html>`__
        documentation.
    band : float
        For rect-based marks ( ``rect``, ``bar``, and ``image`` ), mark size relative to
        bandwidth of `band scales
        <https://vega.github.io/vega-lite/docs/scale.html#band>`__, bins or time units. If
        set to ``1``, the mark size is set to the bandwidth, the bin interval, or the time
        unit interval. If set to ``0.5``, the mark size is half of the bandwidth or the time
        unit interval.

        For other marks, relative position on a band of a stacked, binned, time unit or band
        scale. If set to ``0``, the marks will be positioned at the beginning of the band.
        If set to ``0.5``, the marks will be positioned in the middle of the band.
    datum : anyOf(:class:`PrimitiveValue`, :class:`DateTime`, :class:`ExprRef`,
    :class:`RepeatRef`)
        A constant value in data domain.
    impute : anyOf(:class:`ImputeParams`, None)
        An object defining the properties of the Impute Operation to be applied. The field
        value of the other positional channel is taken as ``key`` of the ``Impute``
        Operation. The field of the ``color`` channel if specified is used as ``groupby`` of
        the ``Impute`` Operation.

        **See also:** `impute <https://vega.github.io/vega-lite/docs/impute.html>`__
        documentation.
    scale : anyOf(:class:`Scale`, None)
        An object defining properties of the channel's scale, which is the function that
        transforms values in the data domain (numbers, dates, strings, etc) to visual values
        (pixels, colors, sizes) of the encoding channels.

        If ``null``, the scale will be `disabled and the data value will be directly encoded
        <https://vega.github.io/vega-lite/docs/scale.html#disable>`__.

        **Default value:** If undefined, default `scale properties
        <https://vega.github.io/vega-lite/docs/scale.html>`__ are applied.

        **See also:** `scale <https://vega.github.io/vega-lite/docs/scale.html>`__
        documentation.
    stack : anyOf(:class:`StackOffset`, None, boolean)
        Type of stacking offset if the field should be stacked. ``stack`` is only applicable
        for ``x``, ``y``, ``theta``, and ``radius`` channels with continuous domains. For
        example, ``stack`` of ``y`` can be used to customize stacking for a vertical bar
        chart.

        ``stack`` can be one of the following values: - ``"zero"`` or `true`: stacking with
        baseline offset at zero value of the scale (for creating typical stacked
        [bar](https://vega.github.io/vega-lite/docs/stack.html#bar) and `area
        <https://vega.github.io/vega-lite/docs/stack.html#area>`__ chart). - ``"normalize"``
        - stacking with normalized domain (for creating `normalized stacked bar and area
        charts <https://vega.github.io/vega-lite/docs/stack.html#normalized>`__.
        :raw-html:`<br/>` - ``"center"`` - stacking with center baseline (for `streamgraph
        <https://vega.github.io/vega-lite/docs/stack.html#streamgraph>`__ ). - ``null`` or
        ``false`` - No-stacking. This will produce layered `bar
        <https://vega.github.io/vega-lite/docs/stack.html#layered-bar-chart>`__ and area
        chart.

        **Default value:** ``zero`` for plots with all of the following conditions are true:
        (1) the mark is ``bar``, ``area``, or ``arc`` ; (2) the stacked measure channel (x
        or y) has a linear scale; (3) At least one of non-position channels mapped to an
        unaggregated field that is different from x and y. Otherwise, ``null`` by default.

        **See also:** `stack <https://vega.github.io/vega-lite/docs/stack.html>`__
        documentation.
    type : :class:`Type`
        The type of measurement ( ``"quantitative"``, ``"temporal"``, ``"ordinal"``, or
        ``"nominal"`` ) for the encoded field or constant value ( ``datum`` ). It can also
        be a ``"geojson"`` type for encoding `'geoshape'
        <https://vega.github.io/vega-lite/docs/geoshape.html>`__.

        Vega-Lite automatically infers data types in many cases as discussed below. However,
        type is required for a field if: (1) the field is not nominal and the field encoding
        has no specified ``aggregate`` (except ``argmin`` and ``argmax`` ), ``bin``, scale
        type, custom ``sort`` order, nor ``timeUnit`` or (2) if you wish to use an ordinal
        scale for a field with ``bin`` or ``timeUnit``.

        **Default value:**

        1) For a data ``field``, ``"nominal"`` is the default data type unless the field
        encoding has ``aggregate``, ``channel``, ``bin``, scale type, ``sort``, or
        ``timeUnit`` that satisfies the following criteria: - ``"quantitative"`` is the
        default type if (1) the encoded field contains ``bin`` or ``aggregate`` except
        ``"argmin"`` and ``"argmax"``, (2) the encoding channel is ``latitude`` or
        ``longitude`` channel or (3) if the specified scale type is `a quantitative scale
        <https://vega.github.io/vega-lite/docs/scale.html#type>`__. - ``"temporal"`` is the
        default type if (1) the encoded field contains ``timeUnit`` or (2) the specified
        scale type is a time or utc scale - ``ordinal""`` is the default type if (1) the
        encoded field contains a `custom sort order
        <https://vega.github.io/vega-lite/docs/sort.html#specifying-custom-sort-order>`__,
        (2) the specified scale type is an ordinal/point/band scale, or (3) the encoding
        channel is ``order``.

        2) For a constant value in data domain ( ``datum`` ): - ``"quantitative"`` if the
        datum is a number - ``"nominal"`` if the datum is a string - ``"temporal"`` if the
        datum is `a date time object
        <https://vega.github.io/vega-lite/docs/datetime.html>`__

        **Note:** - Data ``type`` describes the semantics of the data rather than the
        primitive data types (number, string, etc.). The same primitive data type can have
        different types of measurement. For example, numeric data can represent
        quantitative, ordinal, or nominal data. - Data values for a temporal field can be
        either a date-time string (e.g., ``"2015-03-07 12:32:17"``, ``"17:01"``,
        ``"2015-03-16"``. ``"2015"`` ) or a timestamp number (e.g., ``1552199579097`` ). -
        When using with `bin <https://vega.github.io/vega-lite/docs/bin.html>`__, the
        ``type`` property can be either ``"quantitative"`` (for using a linear bin scale) or
        `"ordinal" (for using an ordinal bin scale)
        <https://vega.github.io/vega-lite/docs/type.html#cast-bin>`__. - When using with
        `timeUnit <https://vega.github.io/vega-lite/docs/timeunit.html>`__, the ``type``
        property can be either ``"temporal"`` (default, for using a temporal scale) or
        `"ordinal" (for using an ordinal scale)
        <https://vega.github.io/vega-lite/docs/type.html#cast-bin>`__. - When using with
        `aggregate <https://vega.github.io/vega-lite/docs/aggregate.html>`__, the ``type``
        property refers to the post-aggregation data type. For example, we can calculate
        count ``distinct`` of a categorical field ``"cat"`` using ``{"aggregate":
        "distinct", "field": "cat"}``. The ``"type"`` of the aggregate output is
        ``"quantitative"``. - Secondary channels (e.g., ``x2``, ``y2``, ``xError``,
        ``yError`` ) do not have ``type`` as they must have exactly the same type as their
        primary channels (e.g., ``x``, ``y`` ).

        **See also:** `type <https://vega.github.io/vega-lite/docs/type.html>`__
        documentation.
    """
    _schema = {'$ref': '#/definitions/PositionDatumDef'}

    def __init__(self, axis=Undefined, band=Undefined, datum=Undefined, impute=Undefined,
                 scale=Undefined, stack=Undefined, type=Undefined, **kwds):
        super(PositionDatumDef, self).__init__(axis=axis, band=band, datum=datum, impute=impute,
                                               scale=scale, stack=stack, type=type, **kwds)


class PositionFieldDef(PositionDef):
    """PositionFieldDef schema wrapper

    Mapping(required=[])

    Attributes
    ----------

    aggregate : :class:`Aggregate`
        Aggregation function for the field (e.g., ``"mean"``, ``"sum"``, ``"median"``,
        ``"min"``, ``"max"``, ``"count"`` ).

        **Default value:** ``undefined`` (None)

        **See also:** `aggregate <https://vega.github.io/vega-lite/docs/aggregate.html>`__
        documentation.
    axis : anyOf(:class:`Axis`, None)
        An object defining properties of axis's gridlines, ticks and labels. If ``null``,
        the axis for the encoding channel will be removed.

        **Default value:** If undefined, default `axis properties
        <https://vega.github.io/vega-lite/docs/axis.html>`__ are applied.

        **See also:** `axis <https://vega.github.io/vega-lite/docs/axis.html>`__
        documentation.
    band : float
        For rect-based marks ( ``rect``, ``bar``, and ``image`` ), mark size relative to
        bandwidth of `band scales
        <https://vega.github.io/vega-lite/docs/scale.html#band>`__, bins or time units. If
        set to ``1``, the mark size is set to the bandwidth, the bin interval, or the time
        unit interval. If set to ``0.5``, the mark size is half of the bandwidth or the time
        unit interval.

        For other marks, relative position on a band of a stacked, binned, time unit or band
        scale. If set to ``0``, the marks will be positioned at the beginning of the band.
        If set to ``0.5``, the marks will be positioned in the middle of the band.
    bin : anyOf(boolean, :class:`BinParams`, string, None)
        A flag for binning a ``quantitative`` field, `an object defining binning parameters
        <https://vega.github.io/vega-lite/docs/bin.html#params>`__, or indicating that the
        data for ``x`` or ``y`` channel are binned before they are imported into Vega-Lite (
        ``"binned"`` ).


        If ``true``, default `binning parameters
        <https://vega.github.io/vega-lite/docs/bin.html>`__ will be applied.

        If ``"binned"``, this indicates that the data for the ``x`` (or ``y`` ) channel are
        already binned. You can map the bin-start field to ``x`` (or ``y`` ) and the bin-end
        field to ``x2`` (or ``y2`` ). The scale and axis will be formatted similar to
        binning in Vega-Lite.  To adjust the axis ticks based on the bin step, you can also
        set the axis's `tickMinStep
        <https://vega.github.io/vega-lite/docs/axis.html#ticks>`__ property.

        **Default value:** ``false``

        **See also:** `bin <https://vega.github.io/vega-lite/docs/bin.html>`__
        documentation.
    field : :class:`Field`
        **Required.** A string defining the name of the field from which to pull a data
        value or an object defining iterated values from the `repeat
        <https://vega.github.io/vega-lite/docs/repeat.html>`__ operator.

        **See also:** `field <https://vega.github.io/vega-lite/docs/field.html>`__
        documentation.

        **Notes:** 1)  Dots ( ``.`` ) and brackets ( ``[`` and ``]`` ) can be used to access
        nested objects (e.g., ``"field": "foo.bar"`` and ``"field": "foo['bar']"`` ). If
        field names contain dots or brackets but are not nested, you can use ``\\`` to
        escape dots and brackets (e.g., ``"a\\.b"`` and ``"a\\[0\\]"`` ). See more details
        about escaping in the `field documentation
        <https://vega.github.io/vega-lite/docs/field.html>`__. 2) ``field`` is not required
        if ``aggregate`` is ``count``.
    impute : anyOf(:class:`ImputeParams`, None)
        An object defining the properties of the Impute Operation to be applied. The field
        value of the other positional channel is taken as ``key`` of the ``Impute``
        Operation. The field of the ``color`` channel if specified is used as ``groupby`` of
        the ``Impute`` Operation.

        **See also:** `impute <https://vega.github.io/vega-lite/docs/impute.html>`__
        documentation.
    scale : anyOf(:class:`Scale`, None)
        An object defining properties of the channel's scale, which is the function that
        transforms values in the data domain (numbers, dates, strings, etc) to visual values
        (pixels, colors, sizes) of the encoding channels.

        If ``null``, the scale will be `disabled and the data value will be directly encoded
        <https://vega.github.io/vega-lite/docs/scale.html#disable>`__.

        **Default value:** If undefined, default `scale properties
        <https://vega.github.io/vega-lite/docs/scale.html>`__ are applied.

        **See also:** `scale <https://vega.github.io/vega-lite/docs/scale.html>`__
        documentation.
    sort : :class:`Sort`
        Sort order for the encoded field.

        For continuous fields (quantitative or temporal), ``sort`` can be either
        ``"ascending"`` or ``"descending"``.

        For discrete fields, ``sort`` can be one of the following: - ``"ascending"`` or
        ``"descending"`` -- for sorting by the values' natural order in JavaScript. - `A
        string indicating an encoding channel name to sort by
        <https://vega.github.io/vega-lite/docs/sort.html#sort-by-encoding>`__ (e.g., ``"x"``
        or ``"y"`` ) with an optional minus prefix for descending sort (e.g., ``"-x"`` to
        sort by x-field, descending). This channel string is short-form of `a
        sort-by-encoding definition
        <https://vega.github.io/vega-lite/docs/sort.html#sort-by-encoding>`__. For example,
        ``"sort": "-x"`` is equivalent to ``"sort": {"encoding": "x", "order":
        "descending"}``. - `A sort field definition
        <https://vega.github.io/vega-lite/docs/sort.html#sort-field>`__ for sorting by
        another field. - `An array specifying the field values in preferred order
        <https://vega.github.io/vega-lite/docs/sort.html#sort-array>`__. In this case, the
        sort order will obey the values in the array, followed by any unspecified values in
        their original order. For discrete time field, values in the sort array can be
        `date-time definition objects <types#datetime>`__. In addition, for time units
        ``"month"`` and ``"day"``, the values can be the month or day names (case
        insensitive) or their 3-letter initials (e.g., ``"Mon"``, ``"Tue"`` ). - ``null``
        indicating no sort.

        **Default value:** ``"ascending"``

        **Note:** ``null`` and sorting by another channel is not supported for ``row`` and
        ``column``.

        **See also:** `sort <https://vega.github.io/vega-lite/docs/sort.html>`__
        documentation.
    stack : anyOf(:class:`StackOffset`, None, boolean)
        Type of stacking offset if the field should be stacked. ``stack`` is only applicable
        for ``x``, ``y``, ``theta``, and ``radius`` channels with continuous domains. For
        example, ``stack`` of ``y`` can be used to customize stacking for a vertical bar
        chart.

        ``stack`` can be one of the following values: - ``"zero"`` or `true`: stacking with
        baseline offset at zero value of the scale (for creating typical stacked
        [bar](https://vega.github.io/vega-lite/docs/stack.html#bar) and `area
        <https://vega.github.io/vega-lite/docs/stack.html#area>`__ chart). - ``"normalize"``
        - stacking with normalized domain (for creating `normalized stacked bar and area
        charts <https://vega.github.io/vega-lite/docs/stack.html#normalized>`__.
        :raw-html:`<br/>` - ``"center"`` - stacking with center baseline (for `streamgraph
        <https://vega.github.io/vega-lite/docs/stack.html#streamgraph>`__ ). - ``null`` or
        ``false`` - No-stacking. This will produce layered `bar
        <https://vega.github.io/vega-lite/docs/stack.html#layered-bar-chart>`__ and area
        chart.

        **Default value:** ``zero`` for plots with all of the following conditions are true:
        (1) the mark is ``bar``, ``area``, or ``arc`` ; (2) the stacked measure channel (x
        or y) has a linear scale; (3) At least one of non-position channels mapped to an
        unaggregated field that is different from x and y. Otherwise, ``null`` by default.

        **See also:** `stack <https://vega.github.io/vega-lite/docs/stack.html>`__
        documentation.
    timeUnit : anyOf(:class:`TimeUnit`, :class:`TimeUnitParams`)
        Time unit (e.g., ``year``, ``yearmonth``, ``month``, ``hours`` ) for a temporal
        field. or `a temporal field that gets casted as ordinal
        <https://vega.github.io/vega-lite/docs/type.html#cast>`__.

        **Default value:** ``undefined`` (None)

        **See also:** `timeUnit <https://vega.github.io/vega-lite/docs/timeunit.html>`__
        documentation.
    title : anyOf(:class:`Text`, None)
        A title for the field. If ``null``, the title will be removed.

        **Default value:**  derived from the field's name and transformation function (
        ``aggregate``, ``bin`` and ``timeUnit`` ). If the field has an aggregate function,
        the function is displayed as part of the title (e.g., ``"Sum of Profit"`` ). If the
        field is binned or has a time unit applied, the applied function is shown in
        parentheses (e.g., ``"Profit (binned)"``, ``"Transaction Date (year-month)"`` ).
        Otherwise, the title is simply the field name.

        **Notes** :

        1) You can customize the default field title format by providing the `fieldTitle
        <https://vega.github.io/vega-lite/docs/config.html#top-level-config>`__ property in
        the `config <https://vega.github.io/vega-lite/docs/config.html>`__ or `fieldTitle
        function via the compile function's options
        <https://vega.github.io/vega-lite/docs/compile.html#field-title>`__.

        2) If both field definition's ``title`` and axis, header, or legend ``title`` are
        defined, axis/header/legend title will be used.
    type : :class:`StandardType`
        The type of measurement ( ``"quantitative"``, ``"temporal"``, ``"ordinal"``, or
        ``"nominal"`` ) for the encoded field or constant value ( ``datum`` ). It can also
        be a ``"geojson"`` type for encoding `'geoshape'
        <https://vega.github.io/vega-lite/docs/geoshape.html>`__.

        Vega-Lite automatically infers data types in many cases as discussed below. However,
        type is required for a field if: (1) the field is not nominal and the field encoding
        has no specified ``aggregate`` (except ``argmin`` and ``argmax`` ), ``bin``, scale
        type, custom ``sort`` order, nor ``timeUnit`` or (2) if you wish to use an ordinal
        scale for a field with ``bin`` or ``timeUnit``.

        **Default value:**

        1) For a data ``field``, ``"nominal"`` is the default data type unless the field
        encoding has ``aggregate``, ``channel``, ``bin``, scale type, ``sort``, or
        ``timeUnit`` that satisfies the following criteria: - ``"quantitative"`` is the
        default type if (1) the encoded field contains ``bin`` or ``aggregate`` except
        ``"argmin"`` and ``"argmax"``, (2) the encoding channel is ``latitude`` or
        ``longitude`` channel or (3) if the specified scale type is `a quantitative scale
        <https://vega.github.io/vega-lite/docs/scale.html#type>`__. - ``"temporal"`` is the
        default type if (1) the encoded field contains ``timeUnit`` or (2) the specified
        scale type is a time or utc scale - ``ordinal""`` is the default type if (1) the
        encoded field contains a `custom sort order
        <https://vega.github.io/vega-lite/docs/sort.html#specifying-custom-sort-order>`__,
        (2) the specified scale type is an ordinal/point/band scale, or (3) the encoding
        channel is ``order``.

        2) For a constant value in data domain ( ``datum`` ): - ``"quantitative"`` if the
        datum is a number - ``"nominal"`` if the datum is a string - ``"temporal"`` if the
        datum is `a date time object
        <https://vega.github.io/vega-lite/docs/datetime.html>`__

        **Note:** - Data ``type`` describes the semantics of the data rather than the
        primitive data types (number, string, etc.). The same primitive data type can have
        different types of measurement. For example, numeric data can represent
        quantitative, ordinal, or nominal data. - Data values for a temporal field can be
        either a date-time string (e.g., ``"2015-03-07 12:32:17"``, ``"17:01"``,
        ``"2015-03-16"``. ``"2015"`` ) or a timestamp number (e.g., ``1552199579097`` ). -
        When using with `bin <https://vega.github.io/vega-lite/docs/bin.html>`__, the
        ``type`` property can be either ``"quantitative"`` (for using a linear bin scale) or
        `"ordinal" (for using an ordinal bin scale)
        <https://vega.github.io/vega-lite/docs/type.html#cast-bin>`__. - When using with
        `timeUnit <https://vega.github.io/vega-lite/docs/timeunit.html>`__, the ``type``
        property can be either ``"temporal"`` (default, for using a temporal scale) or
        `"ordinal" (for using an ordinal scale)
        <https://vega.github.io/vega-lite/docs/type.html#cast-bin>`__. - When using with
        `aggregate <https://vega.github.io/vega-lite/docs/aggregate.html>`__, the ``type``
        property refers to the post-aggregation data type. For example, we can calculate
        count ``distinct`` of a categorical field ``"cat"`` using ``{"aggregate":
        "distinct", "field": "cat"}``. The ``"type"`` of the aggregate output is
        ``"quantitative"``. - Secondary channels (e.g., ``x2``, ``y2``, ``xError``,
        ``yError`` ) do not have ``type`` as they must have exactly the same type as their
        primary channels (e.g., ``x``, ``y`` ).

        **See also:** `type <https://vega.github.io/vega-lite/docs/type.html>`__
        documentation.
    """
    _schema = {'$ref': '#/definitions/PositionFieldDef'}

    def __init__(self, aggregate=Undefined, axis=Undefined, band=Undefined, bin=Undefined,
                 field=Undefined, impute=Undefined, scale=Undefined, sort=Undefined, stack=Undefined,
                 timeUnit=Undefined, title=Undefined, type=Undefined, **kwds):
        super(PositionFieldDef, self).__init__(aggregate=aggregate, axis=axis, band=band, bin=bin,
                                               field=field, impute=impute, scale=scale, sort=sort,
                                               stack=stack, timeUnit=timeUnit, title=title, type=type,
                                               **kwds)


class PositionFieldDefBase(PolarDef):
    """PositionFieldDefBase schema wrapper

    Mapping(required=[])

    Attributes
    ----------

    aggregate : :class:`Aggregate`
        Aggregation function for the field (e.g., ``"mean"``, ``"sum"``, ``"median"``,
        ``"min"``, ``"max"``, ``"count"`` ).

        **Default value:** ``undefined`` (None)

        **See also:** `aggregate <https://vega.github.io/vega-lite/docs/aggregate.html>`__
        documentation.
    band : float
        For rect-based marks ( ``rect``, ``bar``, and ``image`` ), mark size relative to
        bandwidth of `band scales
        <https://vega.github.io/vega-lite/docs/scale.html#band>`__, bins or time units. If
        set to ``1``, the mark size is set to the bandwidth, the bin interval, or the time
        unit interval. If set to ``0.5``, the mark size is half of the bandwidth or the time
        unit interval.

        For other marks, relative position on a band of a stacked, binned, time unit or band
        scale. If set to ``0``, the marks will be positioned at the beginning of the band.
        If set to ``0.5``, the marks will be positioned in the middle of the band.
    bin : anyOf(boolean, :class:`BinParams`, string, None)
        A flag for binning a ``quantitative`` field, `an object defining binning parameters
        <https://vega.github.io/vega-lite/docs/bin.html#params>`__, or indicating that the
        data for ``x`` or ``y`` channel are binned before they are imported into Vega-Lite (
        ``"binned"`` ).


        If ``true``, default `binning parameters
        <https://vega.github.io/vega-lite/docs/bin.html>`__ will be applied.

        If ``"binned"``, this indicates that the data for the ``x`` (or ``y`` ) channel are
        already binned. You can map the bin-start field to ``x`` (or ``y`` ) and the bin-end
        field to ``x2`` (or ``y2`` ). The scale and axis will be formatted similar to
        binning in Vega-Lite.  To adjust the axis ticks based on the bin step, you can also
        set the axis's `tickMinStep
        <https://vega.github.io/vega-lite/docs/axis.html#ticks>`__ property.

        **Default value:** ``false``

        **See also:** `bin <https://vega.github.io/vega-lite/docs/bin.html>`__
        documentation.
    field : :class:`Field`
        **Required.** A string defining the name of the field from which to pull a data
        value or an object defining iterated values from the `repeat
        <https://vega.github.io/vega-lite/docs/repeat.html>`__ operator.

        **See also:** `field <https://vega.github.io/vega-lite/docs/field.html>`__
        documentation.

        **Notes:** 1)  Dots ( ``.`` ) and brackets ( ``[`` and ``]`` ) can be used to access
        nested objects (e.g., ``"field": "foo.bar"`` and ``"field": "foo['bar']"`` ). If
        field names contain dots or brackets but are not nested, you can use ``\\`` to
        escape dots and brackets (e.g., ``"a\\.b"`` and ``"a\\[0\\]"`` ). See more details
        about escaping in the `field documentation
        <https://vega.github.io/vega-lite/docs/field.html>`__. 2) ``field`` is not required
        if ``aggregate`` is ``count``.
    scale : anyOf(:class:`Scale`, None)
        An object defining properties of the channel's scale, which is the function that
        transforms values in the data domain (numbers, dates, strings, etc) to visual values
        (pixels, colors, sizes) of the encoding channels.

        If ``null``, the scale will be `disabled and the data value will be directly encoded
        <https://vega.github.io/vega-lite/docs/scale.html#disable>`__.

        **Default value:** If undefined, default `scale properties
        <https://vega.github.io/vega-lite/docs/scale.html>`__ are applied.

        **See also:** `scale <https://vega.github.io/vega-lite/docs/scale.html>`__
        documentation.
    sort : :class:`Sort`
        Sort order for the encoded field.

        For continuous fields (quantitative or temporal), ``sort`` can be either
        ``"ascending"`` or ``"descending"``.

        For discrete fields, ``sort`` can be one of the following: - ``"ascending"`` or
        ``"descending"`` -- for sorting by the values' natural order in JavaScript. - `A
        string indicating an encoding channel name to sort by
        <https://vega.github.io/vega-lite/docs/sort.html#sort-by-encoding>`__ (e.g., ``"x"``
        or ``"y"`` ) with an optional minus prefix for descending sort (e.g., ``"-x"`` to
        sort by x-field, descending). This channel string is short-form of `a
        sort-by-encoding definition
        <https://vega.github.io/vega-lite/docs/sort.html#sort-by-encoding>`__. For example,
        ``"sort": "-x"`` is equivalent to ``"sort": {"encoding": "x", "order":
        "descending"}``. - `A sort field definition
        <https://vega.github.io/vega-lite/docs/sort.html#sort-field>`__ for sorting by
        another field. - `An array specifying the field values in preferred order
        <https://vega.github.io/vega-lite/docs/sort.html#sort-array>`__. In this case, the
        sort order will obey the values in the array, followed by any unspecified values in
        their original order. For discrete time field, values in the sort array can be
        `date-time definition objects <types#datetime>`__. In addition, for time units
        ``"month"`` and ``"day"``, the values can be the month or day names (case
        insensitive) or their 3-letter initials (e.g., ``"Mon"``, ``"Tue"`` ). - ``null``
        indicating no sort.

        **Default value:** ``"ascending"``

        **Note:** ``null`` and sorting by another channel is not supported for ``row`` and
        ``column``.

        **See also:** `sort <https://vega.github.io/vega-lite/docs/sort.html>`__
        documentation.
    stack : anyOf(:class:`StackOffset`, None, boolean)
        Type of stacking offset if the field should be stacked. ``stack`` is only applicable
        for ``x``, ``y``, ``theta``, and ``radius`` channels with continuous domains. For
        example, ``stack`` of ``y`` can be used to customize stacking for a vertical bar
        chart.

        ``stack`` can be one of the following values: - ``"zero"`` or `true`: stacking with
        baseline offset at zero value of the scale (for creating typical stacked
        [bar](https://vega.github.io/vega-lite/docs/stack.html#bar) and `area
        <https://vega.github.io/vega-lite/docs/stack.html#area>`__ chart). - ``"normalize"``
        - stacking with normalized domain (for creating `normalized stacked bar and area
        charts <https://vega.github.io/vega-lite/docs/stack.html#normalized>`__.
        :raw-html:`<br/>` - ``"center"`` - stacking with center baseline (for `streamgraph
        <https://vega.github.io/vega-lite/docs/stack.html#streamgraph>`__ ). - ``null`` or
        ``false`` - No-stacking. This will produce layered `bar
        <https://vega.github.io/vega-lite/docs/stack.html#layered-bar-chart>`__ and area
        chart.

        **Default value:** ``zero`` for plots with all of the following conditions are true:
        (1) the mark is ``bar``, ``area``, or ``arc`` ; (2) the stacked measure channel (x
        or y) has a linear scale; (3) At least one of non-position channels mapped to an
        unaggregated field that is different from x and y. Otherwise, ``null`` by default.

        **See also:** `stack <https://vega.github.io/vega-lite/docs/stack.html>`__
        documentation.
    timeUnit : anyOf(:class:`TimeUnit`, :class:`TimeUnitParams`)
        Time unit (e.g., ``year``, ``yearmonth``, ``month``, ``hours`` ) for a temporal
        field. or `a temporal field that gets casted as ordinal
        <https://vega.github.io/vega-lite/docs/type.html#cast>`__.

        **Default value:** ``undefined`` (None)

        **See also:** `timeUnit <https://vega.github.io/vega-lite/docs/timeunit.html>`__
        documentation.
    title : anyOf(:class:`Text`, None)
        A title for the field. If ``null``, the title will be removed.

        **Default value:**  derived from the field's name and transformation function (
        ``aggregate``, ``bin`` and ``timeUnit`` ). If the field has an aggregate function,
        the function is displayed as part of the title (e.g., ``"Sum of Profit"`` ). If the
        field is binned or has a time unit applied, the applied function is shown in
        parentheses (e.g., ``"Profit (binned)"``, ``"Transaction Date (year-month)"`` ).
        Otherwise, the title is simply the field name.

        **Notes** :

        1) You can customize the default field title format by providing the `fieldTitle
        <https://vega.github.io/vega-lite/docs/config.html#top-level-config>`__ property in
        the `config <https://vega.github.io/vega-lite/docs/config.html>`__ or `fieldTitle
        function via the compile function's options
        <https://vega.github.io/vega-lite/docs/compile.html#field-title>`__.

        2) If both field definition's ``title`` and axis, header, or legend ``title`` are
        defined, axis/header/legend title will be used.
    type : :class:`StandardType`
        The type of measurement ( ``"quantitative"``, ``"temporal"``, ``"ordinal"``, or
        ``"nominal"`` ) for the encoded field or constant value ( ``datum`` ). It can also
        be a ``"geojson"`` type for encoding `'geoshape'
        <https://vega.github.io/vega-lite/docs/geoshape.html>`__.

        Vega-Lite automatically infers data types in many cases as discussed below. However,
        type is required for a field if: (1) the field is not nominal and the field encoding
        has no specified ``aggregate`` (except ``argmin`` and ``argmax`` ), ``bin``, scale
        type, custom ``sort`` order, nor ``timeUnit`` or (2) if you wish to use an ordinal
        scale for a field with ``bin`` or ``timeUnit``.

        **Default value:**

        1) For a data ``field``, ``"nominal"`` is the default data type unless the field
        encoding has ``aggregate``, ``channel``, ``bin``, scale type, ``sort``, or
        ``timeUnit`` that satisfies the following criteria: - ``"quantitative"`` is the
        default type if (1) the encoded field contains ``bin`` or ``aggregate`` except
        ``"argmin"`` and ``"argmax"``, (2) the encoding channel is ``latitude`` or
        ``longitude`` channel or (3) if the specified scale type is `a quantitative scale
        <https://vega.github.io/vega-lite/docs/scale.html#type>`__. - ``"temporal"`` is the
        default type if (1) the encoded field contains ``timeUnit`` or (2) the specified
        scale type is a time or utc scale - ``ordinal""`` is the default type if (1) the
        encoded field contains a `custom sort order
        <https://vega.github.io/vega-lite/docs/sort.html#specifying-custom-sort-order>`__,
        (2) the specified scale type is an ordinal/point/band scale, or (3) the encoding
        channel is ``order``.

        2) For a constant value in data domain ( ``datum`` ): - ``"quantitative"`` if the
        datum is a number - ``"nominal"`` if the datum is a string - ``"temporal"`` if the
        datum is `a date time object
        <https://vega.github.io/vega-lite/docs/datetime.html>`__

        **Note:** - Data ``type`` describes the semantics of the data rather than the
        primitive data types (number, string, etc.). The same primitive data type can have
        different types of measurement. For example, numeric data can represent
        quantitative, ordinal, or nominal data. - Data values for a temporal field can be
        either a date-time string (e.g., ``"2015-03-07 12:32:17"``, ``"17:01"``,
        ``"2015-03-16"``. ``"2015"`` ) or a timestamp number (e.g., ``1552199579097`` ). -
        When using with `bin <https://vega.github.io/vega-lite/docs/bin.html>`__, the
        ``type`` property can be either ``"quantitative"`` (for using a linear bin scale) or
        `"ordinal" (for using an ordinal bin scale)
        <https://vega.github.io/vega-lite/docs/type.html#cast-bin>`__. - When using with
        `timeUnit <https://vega.github.io/vega-lite/docs/timeunit.html>`__, the ``type``
        property can be either ``"temporal"`` (default, for using a temporal scale) or
        `"ordinal" (for using an ordinal scale)
        <https://vega.github.io/vega-lite/docs/type.html#cast-bin>`__. - When using with
        `aggregate <https://vega.github.io/vega-lite/docs/aggregate.html>`__, the ``type``
        property refers to the post-aggregation data type. For example, we can calculate
        count ``distinct`` of a categorical field ``"cat"`` using ``{"aggregate":
        "distinct", "field": "cat"}``. The ``"type"`` of the aggregate output is
        ``"quantitative"``. - Secondary channels (e.g., ``x2``, ``y2``, ``xError``,
        ``yError`` ) do not have ``type`` as they must have exactly the same type as their
        primary channels (e.g., ``x``, ``y`` ).

        **See also:** `type <https://vega.github.io/vega-lite/docs/type.html>`__
        documentation.
    """
    _schema = {'$ref': '#/definitions/PositionFieldDefBase'}

    def __init__(self, aggregate=Undefined, band=Undefined, bin=Undefined, field=Undefined,
                 scale=Undefined, sort=Undefined, stack=Undefined, timeUnit=Undefined, title=Undefined,
                 type=Undefined, **kwds):
        super(PositionFieldDefBase, self).__init__(aggregate=aggregate, band=band, bin=bin, field=field,
                                                   scale=scale, sort=sort, stack=stack,
                                                   timeUnit=timeUnit, title=title, type=type, **kwds)


class PositionValueDef(PolarDef, Position2Def, PositionDef):
    """PositionValueDef schema wrapper

    Mapping(required=[value])
    Definition object for a constant value (primitive value or gradient definition) of an
    encoding channel.

    Attributes
    ----------

    value : anyOf(float, string, string, :class:`ExprRef`)
        A constant value in visual domain (e.g., ``"red"`` / ``"#0099ff"`` / `gradient
        definition <https://vega.github.io/vega-lite/docs/types.html#gradient>`__ for color,
        values between ``0`` to ``1`` for opacity).
    """
    _schema = {'$ref': '#/definitions/PositionValueDef'}

    def __init__(self, value=Undefined, **kwds):
        super(PositionValueDef, self).__init__(value=value, **kwds)


class PredicateComposition(VegaLiteSchema):
    """PredicateComposition schema wrapper

    anyOf(:class:`LogicalNotPredicate`, :class:`LogicalAndPredicate`,
    :class:`LogicalOrPredicate`, :class:`Predicate`)
    """
    _schema = {'$ref': '#/definitions/PredicateComposition'}

    def __init__(self, *args, **kwds):
        super(PredicateComposition, self).__init__(*args, **kwds)


class LogicalAndPredicate(PredicateComposition):
    """LogicalAndPredicate schema wrapper

    Mapping(required=[and])

    Attributes
    ----------

    and : List(:class:`PredicateComposition`)

    """
    _schema = {'$ref': '#/definitions/LogicalAnd<Predicate>'}

    def __init__(self, **kwds):
        super(LogicalAndPredicate, self).__init__(**kwds)


class LogicalNotPredicate(PredicateComposition):
    """LogicalNotPredicate schema wrapper

    Mapping(required=[not])

    Attributes
    ----------

    not : :class:`PredicateComposition`

    """
    _schema = {'$ref': '#/definitions/LogicalNot<Predicate>'}

    def __init__(self, **kwds):
        super(LogicalNotPredicate, self).__init__(**kwds)


class LogicalOrPredicate(PredicateComposition):
    """LogicalOrPredicate schema wrapper

    Mapping(required=[or])

    Attributes
    ----------

    or : List(:class:`PredicateComposition`)

    """
    _schema = {'$ref': '#/definitions/LogicalOr<Predicate>'}

    def __init__(self, **kwds):
        super(LogicalOrPredicate, self).__init__(**kwds)


class Predicate(PredicateComposition):
    """Predicate schema wrapper

    anyOf(:class:`FieldEqualPredicate`, :class:`FieldRangePredicate`,
    :class:`FieldOneOfPredicate`, :class:`FieldLTPredicate`, :class:`FieldGTPredicate`,
    :class:`FieldLTEPredicate`, :class:`FieldGTEPredicate`, :class:`FieldValidPredicate`,
    :class:`SelectionPredicate`, string)
    """
    _schema = {'$ref': '#/definitions/Predicate'}

    def __init__(self, *args, **kwds):
        super(Predicate, self).__init__(*args, **kwds)


class FieldEqualPredicate(Predicate):
    """FieldEqualPredicate schema wrapper

    Mapping(required=[equal, field])

    Attributes
    ----------

    equal : anyOf(string, float, boolean, :class:`DateTime`, :class:`ExprRef`)
        The value that the field should be equal to.
    field : :class:`FieldName`
        Field to be tested.
    timeUnit : anyOf(:class:`TimeUnit`, :class:`TimeUnitParams`)
        Time unit for the field to be tested.
    """
    _schema = {'$ref': '#/definitions/FieldEqualPredicate'}

    def __init__(self, equal=Undefined, field=Undefined, timeUnit=Undefined, **kwds):
        super(FieldEqualPredicate, self).__init__(equal=equal, field=field, timeUnit=timeUnit, **kwds)


class FieldGTEPredicate(Predicate):
    """FieldGTEPredicate schema wrapper

    Mapping(required=[field, gte])

    Attributes
    ----------

    field : :class:`FieldName`
        Field to be tested.
    gte : anyOf(string, float, :class:`DateTime`, :class:`ExprRef`)
        The value that the field should be greater than or equals to.
    timeUnit : anyOf(:class:`TimeUnit`, :class:`TimeUnitParams`)
        Time unit for the field to be tested.
    """
    _schema = {'$ref': '#/definitions/FieldGTEPredicate'}

    def __init__(self, field=Undefined, gte=Undefined, timeUnit=Undefined, **kwds):
        super(FieldGTEPredicate, self).__init__(field=field, gte=gte, timeUnit=timeUnit, **kwds)


class FieldGTPredicate(Predicate):
    """FieldGTPredicate schema wrapper

    Mapping(required=[field, gt])

    Attributes
    ----------

    field : :class:`FieldName`
        Field to be tested.
    gt : anyOf(string, float, :class:`DateTime`, :class:`ExprRef`)
        The value that the field should be greater than.
    timeUnit : anyOf(:class:`TimeUnit`, :class:`TimeUnitParams`)
        Time unit for the field to be tested.
    """
    _schema = {'$ref': '#/definitions/FieldGTPredicate'}

    def __init__(self, field=Undefined, gt=Undefined, timeUnit=Undefined, **kwds):
        super(FieldGTPredicate, self).__init__(field=field, gt=gt, timeUnit=timeUnit, **kwds)


class FieldLTEPredicate(Predicate):
    """FieldLTEPredicate schema wrapper

    Mapping(required=[field, lte])

    Attributes
    ----------

    field : :class:`FieldName`
        Field to be tested.
    lte : anyOf(string, float, :class:`DateTime`, :class:`ExprRef`)
        The value that the field should be less than or equals to.
    timeUnit : anyOf(:class:`TimeUnit`, :class:`TimeUnitParams`)
        Time unit for the field to be tested.
    """
    _schema = {'$ref': '#/definitions/FieldLTEPredicate'}

    def __init__(self, field=Undefined, lte=Undefined, timeUnit=Undefined, **kwds):
        super(FieldLTEPredicate, self).__init__(field=field, lte=lte, timeUnit=timeUnit, **kwds)


class FieldLTPredicate(Predicate):
    """FieldLTPredicate schema wrapper

    Mapping(required=[field, lt])

    Attributes
    ----------

    field : :class:`FieldName`
        Field to be tested.
    lt : anyOf(string, float, :class:`DateTime`, :class:`ExprRef`)
        The value that the field should be less than.
    timeUnit : anyOf(:class:`TimeUnit`, :class:`TimeUnitParams`)
        Time unit for the field to be tested.
    """
    _schema = {'$ref': '#/definitions/FieldLTPredicate'}

    def __init__(self, field=Undefined, lt=Undefined, timeUnit=Undefined, **kwds):
        super(FieldLTPredicate, self).__init__(field=field, lt=lt, timeUnit=timeUnit, **kwds)


class FieldOneOfPredicate(Predicate):
    """FieldOneOfPredicate schema wrapper

    Mapping(required=[field, oneOf])

    Attributes
    ----------

    field : :class:`FieldName`
        Field to be tested.
    oneOf : anyOf(List(string), List(float), List(boolean), List(:class:`DateTime`))
        A set of values that the ``field`` 's value should be a member of, for a data item
        included in the filtered data.
    timeUnit : anyOf(:class:`TimeUnit`, :class:`TimeUnitParams`)
        Time unit for the field to be tested.
    """
    _schema = {'$ref': '#/definitions/FieldOneOfPredicate'}

    def __init__(self, field=Undefined, oneOf=Undefined, timeUnit=Undefined, **kwds):
        super(FieldOneOfPredicate, self).__init__(field=field, oneOf=oneOf, timeUnit=timeUnit, **kwds)


class FieldRangePredicate(Predicate):
    """FieldRangePredicate schema wrapper

    Mapping(required=[field, range])

    Attributes
    ----------

    field : :class:`FieldName`
        Field to be tested.
    range : anyOf(List(anyOf(float, :class:`DateTime`, None, :class:`ExprRef`)),
    :class:`ExprRef`)
        An array of inclusive minimum and maximum values for a field value of a data item to
        be included in the filtered data.
    timeUnit : anyOf(:class:`TimeUnit`, :class:`TimeUnitParams`)
        Time unit for the field to be tested.
    """
    _schema = {'$ref': '#/definitions/FieldRangePredicate'}

    def __init__(self, field=Undefined, range=Undefined, timeUnit=Undefined, **kwds):
        super(FieldRangePredicate, self).__init__(field=field, range=range, timeUnit=timeUnit, **kwds)


class FieldValidPredicate(Predicate):
    """FieldValidPredicate schema wrapper

    Mapping(required=[field, valid])

    Attributes
    ----------

    field : :class:`FieldName`
        Field to be tested.
    valid : boolean
        If set to true the field's value has to be valid, meaning both not ``null`` and not
        `NaN
        <https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/NaN>`__.
    timeUnit : anyOf(:class:`TimeUnit`, :class:`TimeUnitParams`)
        Time unit for the field to be tested.
    """
    _schema = {'$ref': '#/definitions/FieldValidPredicate'}

    def __init__(self, field=Undefined, valid=Undefined, timeUnit=Undefined, **kwds):
        super(FieldValidPredicate, self).__init__(field=field, valid=valid, timeUnit=timeUnit, **kwds)


class Projection(VegaLiteSchema):
    """Projection schema wrapper

    Mapping(required=[])

    Attributes
    ----------

    center : :class:`Vector2number`
        The projection's center, a two-element array of longitude and latitude in degrees.

        **Default value:** ``[0, 0]``
    clipAngle : float
        The projection's clipping circle radius to the specified angle in degrees. If
        ``null``, switches to `antimeridian <http://bl.ocks.org/mbostock/3788999>`__ cutting
        rather than small-circle clipping.
    clipExtent : :class:`Vector2Vector2number`
        The projection's viewport clip extent to the specified bounds in pixels. The extent
        bounds are specified as an array ``[[x0, y0], [x1, y1]]``, where ``x0`` is the
        left-side of the viewport, ``y0`` is the top, ``x1`` is the right and ``y1`` is the
        bottom. If ``null``, no viewport clipping is performed.
    coefficient : float

    distance : float

    extent : :class:`Vector2Vector2number`

    fit : anyOf(:class:`Fit`, List(:class:`Fit`))

    fraction : float

    lobes : float

    parallel : float

    parallels : List(float)
        For conic projections, the `two standard parallels
        <https://en.wikipedia.org/wiki/Map_projection#Conic>`__ that define the map layout.
        The default depends on the specific conic projection used.
    pointRadius : float
        The default radius (in pixels) to use when drawing GeoJSON ``Point`` and
        ``MultiPoint`` geometries. This parameter sets a constant default value. To modify
        the point radius in response to data, see the corresponding parameter of the GeoPath
        and GeoShape transforms.

        **Default value:** ``4.5``
    precision : float
        The threshold for the projection's `adaptive resampling
        <http://bl.ocks.org/mbostock/3795544>`__ to the specified value in pixels. This
        value corresponds to the `Douglas–Peucker distance
        <http://en.wikipedia.org/wiki/Ramer%E2%80%93Douglas%E2%80%93Peucker_algorithm>`__.
        If precision is not specified, returns the projection's current resampling precision
        which defaults to ``√0.5 ≅ 0.70710…``.
    radius : float

    ratio : float

    reflectX : boolean

    reflectY : boolean

    rotate : anyOf(:class:`Vector2number`, :class:`Vector3number`)
        The projection's three-axis rotation to the specified angles, which must be a two-
        or three-element array of numbers [ ``lambda``, ``phi``, ``gamma`` ] specifying the
        rotation angles in degrees about each spherical axis. (These correspond to yaw,
        pitch and roll.)

        **Default value:** ``[0, 0, 0]``
    scale : float
        The projection’s scale (zoom) factor, overriding automatic fitting. The default
        scale is projection-specific. The scale factor corresponds linearly to the distance
        between projected points; however, scale factor values are not equivalent across
        projections.
    size : :class:`Vector2number`

    spacing : float

    tilt : float

    translate : :class:`Vector2number`
        The projection’s translation offset as a two-element array ``[tx, ty]``.
    type : :class:`ProjectionType`
        The cartographic projection to use. This value is case-insensitive, for example
        ``"albers"`` and ``"Albers"`` indicate the same projection type. You can find all
        valid projection types `in the documentation
        <https://vega.github.io/vega-lite/docs/projection.html#projection-types>`__.

        **Default value:** ``mercator``
    """
    _schema = {'$ref': '#/definitions/Projection'}

    def __init__(self, center=Undefined, clipAngle=Undefined, clipExtent=Undefined,
                 coefficient=Undefined, distance=Undefined, extent=Undefined, fit=Undefined,
                 fraction=Undefined, lobes=Undefined, parallel=Undefined, parallels=Undefined,
                 pointRadius=Undefined, precision=Undefined, radius=Undefined, ratio=Undefined,
                 reflectX=Undefined, reflectY=Undefined, rotate=Undefined, scale=Undefined,
                 size=Undefined, spacing=Undefined, tilt=Undefined, translate=Undefined, type=Undefined,
                 **kwds):
        super(Projection, self).__init__(center=center, clipAngle=clipAngle, clipExtent=clipExtent,
                                         coefficient=coefficient, distance=distance, extent=extent,
                                         fit=fit, fraction=fraction, lobes=lobes, parallel=parallel,
                                         parallels=parallels, pointRadius=pointRadius,
                                         precision=precision, radius=radius, ratio=ratio,
                                         reflectX=reflectX, reflectY=reflectY, rotate=rotate,
                                         scale=scale, size=size, spacing=spacing, tilt=tilt,
                                         translate=translate, type=type, **kwds)


class ProjectionConfig(VegaLiteSchema):
    """ProjectionConfig schema wrapper

    Mapping(required=[])

    Attributes
    ----------

    center : :class:`Vector2number`
        The projection's center, a two-element array of longitude and latitude in degrees.

        **Default value:** ``[0, 0]``
    clipAngle : float
        The projection's clipping circle radius to the specified angle in degrees. If
        ``null``, switches to `antimeridian <http://bl.ocks.org/mbostock/3788999>`__ cutting
        rather than small-circle clipping.
    clipExtent : :class:`Vector2Vector2number`
        The projection's viewport clip extent to the specified bounds in pixels. The extent
        bounds are specified as an array ``[[x0, y0], [x1, y1]]``, where ``x0`` is the
        left-side of the viewport, ``y0`` is the top, ``x1`` is the right and ``y1`` is the
        bottom. If ``null``, no viewport clipping is performed.
    coefficient : float

    distance : float

    extent : :class:`Vector2Vector2number`

    fit : anyOf(:class:`Fit`, List(:class:`Fit`))

    fraction : float

    lobes : float

    parallel : float

    parallels : List(float)
        For conic projections, the `two standard parallels
        <https://en.wikipedia.org/wiki/Map_projection#Conic>`__ that define the map layout.
        The default depends on the specific conic projection used.
    pointRadius : float
        The default radius (in pixels) to use when drawing GeoJSON ``Point`` and
        ``MultiPoint`` geometries. This parameter sets a constant default value. To modify
        the point radius in response to data, see the corresponding parameter of the GeoPath
        and GeoShape transforms.

        **Default value:** ``4.5``
    precision : float
        The threshold for the projection's `adaptive resampling
        <http://bl.ocks.org/mbostock/3795544>`__ to the specified value in pixels. This
        value corresponds to the `Douglas–Peucker distance
        <http://en.wikipedia.org/wiki/Ramer%E2%80%93Douglas%E2%80%93Peucker_algorithm>`__.
        If precision is not specified, returns the projection's current resampling precision
        which defaults to ``√0.5 ≅ 0.70710…``.
    radius : float

    ratio : float

    reflectX : boolean

    reflectY : boolean

    rotate : anyOf(:class:`Vector2number`, :class:`Vector3number`)
        The projection's three-axis rotation to the specified angles, which must be a two-
        or three-element array of numbers [ ``lambda``, ``phi``, ``gamma`` ] specifying the
        rotation angles in degrees about each spherical axis. (These correspond to yaw,
        pitch and roll.)

        **Default value:** ``[0, 0, 0]``
    scale : float
        The projection’s scale (zoom) factor, overriding automatic fitting. The default
        scale is projection-specific. The scale factor corresponds linearly to the distance
        between projected points; however, scale factor values are not equivalent across
        projections.
    size : :class:`Vector2number`

    spacing : float

    tilt : float

    translate : :class:`Vector2number`
        The projection’s translation offset as a two-element array ``[tx, ty]``.
    type : :class:`ProjectionType`
        The cartographic projection to use. This value is case-insensitive, for example
        ``"albers"`` and ``"Albers"`` indicate the same projection type. You can find all
        valid projection types `in the documentation
        <https://vega.github.io/vega-lite/docs/projection.html#projection-types>`__.

        **Default value:** ``mercator``
    """
    _schema = {'$ref': '#/definitions/ProjectionConfig'}

    def __init__(self, center=Undefined, clipAngle=Undefined, clipExtent=Undefined,
                 coefficient=Undefined, distance=Undefined, extent=Undefined, fit=Undefined,
                 fraction=Undefined, lobes=Undefined, parallel=Undefined, parallels=Undefined,
                 pointRadius=Undefined, precision=Undefined, radius=Undefined, ratio=Undefined,
                 reflectX=Undefined, reflectY=Undefined, rotate=Undefined, scale=Undefined,
                 size=Undefined, spacing=Undefined, tilt=Undefined, translate=Undefined, type=Undefined,
                 **kwds):
        super(ProjectionConfig, self).__init__(center=center, clipAngle=clipAngle,
                                               clipExtent=clipExtent, coefficient=coefficient,
                                               distance=distance, extent=extent, fit=fit,
                                               fraction=fraction, lobes=lobes, parallel=parallel,
                                               parallels=parallels, pointRadius=pointRadius,
                                               precision=precision, radius=radius, ratio=ratio,
                                               reflectX=reflectX, reflectY=reflectY, rotate=rotate,
                                               scale=scale, size=size, spacing=spacing, tilt=tilt,
                                               translate=translate, type=type, **kwds)


class ProjectionType(VegaLiteSchema):
    """ProjectionType schema wrapper

    enum('albers', 'albersUsa', 'azimuthalEqualArea', 'azimuthalEquidistant', 'conicConformal',
    'conicEqualArea', 'conicEquidistant', 'equalEarth', 'equirectangular', 'gnomonic',
    'identity', 'mercator', 'naturalEarth1', 'orthographic', 'stereographic',
    'transverseMercator')
    """
    _schema = {'$ref': '#/definitions/ProjectionType'}

    def __init__(self, *args):
        super(ProjectionType, self).__init__(*args)


class RadialGradient(Gradient):
    """RadialGradient schema wrapper

    Mapping(required=[gradient, stops])

    Attributes
    ----------

    gradient : string
        The type of gradient. Use ``"radial"`` for a radial gradient.
    stops : List(:class:`GradientStop`)
        An array of gradient stops defining the gradient color sequence.
    id : string

    r1 : float
        The radius length, in normalized [0, 1] coordinates, of the inner circle for the
        gradient.

        **Default value:** ``0``
    r2 : float
        The radius length, in normalized [0, 1] coordinates, of the outer circle for the
        gradient.

        **Default value:** ``0.5``
    x1 : float
        The x-coordinate, in normalized [0, 1] coordinates, for the center of the inner
        circle for the gradient.

        **Default value:** ``0.5``
    x2 : float
        The x-coordinate, in normalized [0, 1] coordinates, for the center of the outer
        circle for the gradient.

        **Default value:** ``0.5``
    y1 : float
        The y-coordinate, in normalized [0, 1] coordinates, for the center of the inner
        circle for the gradient.

        **Default value:** ``0.5``
    y2 : float
        The y-coordinate, in normalized [0, 1] coordinates, for the center of the outer
        circle for the gradient.

        **Default value:** ``0.5``
    """
    _schema = {'$ref': '#/definitions/RadialGradient'}

    def __init__(self, gradient=Undefined, stops=Undefined, id=Undefined, r1=Undefined, r2=Undefined,
                 x1=Undefined, x2=Undefined, y1=Undefined, y2=Undefined, **kwds):
        super(RadialGradient, self).__init__(gradient=gradient, stops=stops, id=id, r1=r1, r2=r2, x1=x1,
                                             x2=x2, y1=y1, y2=y2, **kwds)


class RangeConfig(VegaLiteSchema):
    """RangeConfig schema wrapper

    Mapping(required=[])

    Attributes
    ----------

    category : anyOf(:class:`RangeScheme`, List(:class:`Color`))
        Default `color scheme <https://vega.github.io/vega/docs/schemes/>`__ for categorical
        data.
    diverging : anyOf(:class:`RangeScheme`, List(:class:`Color`))
        Default `color scheme <https://vega.github.io/vega/docs/schemes/>`__ for diverging
        quantitative ramps.
    heatmap : anyOf(:class:`RangeScheme`, List(:class:`Color`))
        Default `color scheme <https://vega.github.io/vega/docs/schemes/>`__ for
        quantitative heatmaps.
    ordinal : anyOf(:class:`RangeScheme`, List(:class:`Color`))
        Default `color scheme <https://vega.github.io/vega/docs/schemes/>`__ for
        rank-ordered data.
    ramp : anyOf(:class:`RangeScheme`, List(:class:`Color`))
        Default `color scheme <https://vega.github.io/vega/docs/schemes/>`__ for sequential
        quantitative ramps.
    symbol : List(:class:`SymbolShape`)
        Array of `symbol <https://vega.github.io/vega/docs/marks/symbol/>`__ names or paths
        for the default shape palette.
    """
    _schema = {'$ref': '#/definitions/RangeConfig'}

    def __init__(self, category=Undefined, diverging=Undefined, heatmap=Undefined, ordinal=Undefined,
                 ramp=Undefined, symbol=Undefined, **kwds):
        super(RangeConfig, self).__init__(category=category, diverging=diverging, heatmap=heatmap,
                                          ordinal=ordinal, ramp=ramp, symbol=symbol, **kwds)


class RangeRawArray(VegaLiteSchema):
    """RangeRawArray schema wrapper

    List(float)
    """
    _schema = {'$ref': '#/definitions/RangeRawArray'}

    def __init__(self, *args):
        super(RangeRawArray, self).__init__(*args)


class RangeScheme(VegaLiteSchema):
    """RangeScheme schema wrapper

    anyOf(:class:`RangeEnum`, :class:`RangeRaw`, Mapping(required=[scheme]))
    """
    _schema = {'$ref': '#/definitions/RangeScheme'}

    def __init__(self, *args, **kwds):
        super(RangeScheme, self).__init__(*args, **kwds)


class RangeEnum(RangeScheme):
    """RangeEnum schema wrapper

    enum('width', 'height', 'symbol', 'category', 'ordinal', 'ramp', 'diverging', 'heatmap')
    """
    _schema = {'$ref': '#/definitions/RangeEnum'}

    def __init__(self, *args):
        super(RangeEnum, self).__init__(*args)


class RangeRaw(RangeScheme):
    """RangeRaw schema wrapper

    List(anyOf(None, boolean, string, float, :class:`RangeRawArray`))
    """
    _schema = {'$ref': '#/definitions/RangeRaw'}

    def __init__(self, *args):
        super(RangeRaw, self).__init__(*args)


class RectConfig(AnyMarkConfig):
    """RectConfig schema wrapper

    Mapping(required=[])

    Attributes
    ----------

    align : anyOf(:class:`Align`, :class:`ExprRef`)
        The horizontal alignment of the text or ranged marks (area, bar, image, rect, rule).
        One of ``"left"``, ``"right"``, ``"center"``.

        **Note:** Expression reference is *not* supported for range marks.
    angle : anyOf(float, :class:`ExprRef`)

    aria : anyOf(boolean, :class:`ExprRef`)

    ariaRole : anyOf(string, :class:`ExprRef`)

    ariaRoleDescription : anyOf(string, :class:`ExprRef`)

    aspect : anyOf(boolean, :class:`ExprRef`)

    baseline : anyOf(:class:`TextBaseline`, :class:`ExprRef`)
        For text marks, the vertical text baseline. One of ``"alphabetic"`` (default),
        ``"top"``, ``"middle"``, ``"bottom"``, ``"line-top"``, ``"line-bottom"``, or an
        expression reference that provides one of the valid values. The ``"line-top"`` and
        ``"line-bottom"`` values operate similarly to ``"top"`` and ``"bottom"``, but are
        calculated relative to the ``lineHeight`` rather than ``fontSize`` alone.

        For range marks, the vertical alignment of the marks. One of ``"top"``,
        ``"middle"``, ``"bottom"``.

        **Note:** Expression reference is *not* supported for range marks.
    binSpacing : float
        Offset between bars for binned field. The ideal value for this is either 0
        (preferred by statisticians) or 1 (Vega-Lite default, D3 example style).

        **Default value:** ``1``
    blend : anyOf(:class:`Blend`, :class:`ExprRef`)

    color : anyOf(:class:`Color`, :class:`Gradient`, :class:`ExprRef`)
        Default color.

        **Default value:** :raw-html:`<span style="color: #4682b4;">&#9632;</span>`
        ``"#4682b4"``

        **Note:** - This property cannot be used in a `style config
        <https://vega.github.io/vega-lite/docs/mark.html#style-config>`__. - The ``fill``
        and ``stroke`` properties have higher precedence than ``color`` and will override
        ``color``.
    continuousBandSize : float
        The default size of the bars on continuous scales.

        **Default value:** ``5``
    cornerRadius : anyOf(float, :class:`ExprRef`)

    cornerRadiusBottomLeft : anyOf(float, :class:`ExprRef`)

    cornerRadiusBottomRight : anyOf(float, :class:`ExprRef`)

    cornerRadiusTopLeft : anyOf(float, :class:`ExprRef`)

    cornerRadiusTopRight : anyOf(float, :class:`ExprRef`)

    cursor : anyOf(:class:`Cursor`, :class:`ExprRef`)

    description : anyOf(string, :class:`ExprRef`)

    dir : anyOf(:class:`TextDirection`, :class:`ExprRef`)

    discreteBandSize : float
        The default size of the bars with discrete dimensions. If unspecified, the default
        size is  ``step-2``, which provides 2 pixel offset between bars.
    dx : anyOf(float, :class:`ExprRef`)

    dy : anyOf(float, :class:`ExprRef`)

    ellipsis : anyOf(string, :class:`ExprRef`)

    endAngle : anyOf(float, :class:`ExprRef`)

    fill : anyOf(:class:`Color`, :class:`Gradient`, None, :class:`ExprRef`)
        Default fill color. This property has higher precedence than ``config.color``. Set
        to ``null`` to remove fill.

        **Default value:** (None)
    fillOpacity : anyOf(float, :class:`ExprRef`)

    filled : boolean
        Whether the mark's color should be used as fill color instead of stroke color.

        **Default value:** ``false`` for all ``point``, ``line``, and ``rule`` marks as well
        as ``geoshape`` marks for `graticule
        <https://vega.github.io/vega-lite/docs/data.html#graticule>`__ data sources;
        otherwise, ``true``.

        **Note:** This property cannot be used in a `style config
        <https://vega.github.io/vega-lite/docs/mark.html#style-config>`__.
    font : anyOf(string, :class:`ExprRef`)

    fontSize : anyOf(float, :class:`ExprRef`)

    fontStyle : anyOf(:class:`FontStyle`, :class:`ExprRef`)

    fontWeight : anyOf(:class:`FontWeight`, :class:`ExprRef`)

    height : anyOf(float, :class:`ExprRef`)

    href : anyOf(:class:`URI`, :class:`ExprRef`)

    innerRadius : anyOf(float, :class:`ExprRef`)
        The inner radius in pixels of arc marks. ``innerRadius`` is an alias for
        ``radius2``.
    interpolate : anyOf(:class:`Interpolate`, :class:`ExprRef`)

    invalid : enum('filter', None)
        Defines how Vega-Lite should handle marks for invalid values ( ``null`` and ``NaN``
        ). - If set to ``"filter"`` (default), all data items with null values will be
        skipped (for line, trail, and area marks) or filtered (for other marks). - If
        ``null``, all data items are included. In this case, invalid values will be
        interpreted as zeroes.
    limit : anyOf(float, :class:`ExprRef`)

    lineBreak : anyOf(string, :class:`ExprRef`)

    lineHeight : anyOf(float, :class:`ExprRef`)

    opacity : anyOf(float, :class:`ExprRef`)
        The overall opacity (value between [0,1]).

        **Default value:** ``0.7`` for non-aggregate plots with ``point``, ``tick``,
        ``circle``, or ``square`` marks or layered ``bar`` charts and ``1`` otherwise.
    order : anyOf(None, boolean)
        For line and trail marks, this ``order`` property can be set to ``null`` or
        ``false`` to make the lines use the original order in the data sources.
    orient : :class:`Orientation`
        The orientation of a non-stacked bar, tick, area, and line charts. The value is
        either horizontal (default) or vertical. - For bar, rule and tick, this determines
        whether the size of the bar and tick should be applied to x or y dimension. - For
        area, this property determines the orient property of the Vega output. - For line
        and trail marks, this property determines the sort order of the points in the line
        if ``config.sortLineBy`` is not specified. For stacked charts, this is always
        determined by the orientation of the stack; therefore explicitly specified value
        will be ignored.
    outerRadius : anyOf(float, :class:`ExprRef`)
        The outer radius in pixels of arc marks. ``outerRadius`` is an alias for ``radius``.
    padAngle : anyOf(float, :class:`ExprRef`)

    radius : anyOf(float, :class:`ExprRef`)
        For arc mark, the primary (outer) radius in pixels.

        For text marks, polar coordinate radial offset, in pixels, of the text from the
        origin determined by the ``x`` and ``y`` properties.
    radius2 : anyOf(float, :class:`ExprRef`)
        The secondary (inner) radius in pixels of arc marks.
    shape : anyOf(anyOf(:class:`SymbolShape`, string), :class:`ExprRef`)

    size : anyOf(float, :class:`ExprRef`)
        Default size for marks. - For ``point`` / ``circle`` / ``square``, this represents
        the pixel area of the marks. Note that this value sets the area of the symbol; the
        side lengths will increase with the square root of this value. - For ``bar``, this
        represents the band size of the bar, in pixels. - For ``text``, this represents the
        font size, in pixels.

        **Default value:** - ``30`` for point, circle, square marks; width/height's ``step``
        - ``2`` for bar marks with discrete dimensions; - ``5`` for bar marks with
        continuous dimensions; - ``11`` for text marks.
    smooth : anyOf(boolean, :class:`ExprRef`)

    startAngle : anyOf(float, :class:`ExprRef`)

    stroke : anyOf(:class:`Color`, :class:`Gradient`, None, :class:`ExprRef`)
        Default stroke color. This property has higher precedence than ``config.color``. Set
        to ``null`` to remove stroke.

        **Default value:** (None)
    strokeCap : anyOf(:class:`StrokeCap`, :class:`ExprRef`)

    strokeDash : anyOf(List(float), :class:`ExprRef`)

    strokeDashOffset : anyOf(float, :class:`ExprRef`)

    strokeJoin : anyOf(:class:`StrokeJoin`, :class:`ExprRef`)

    strokeMiterLimit : anyOf(float, :class:`ExprRef`)

    strokeOffset : anyOf(float, :class:`ExprRef`)

    strokeOpacity : anyOf(float, :class:`ExprRef`)

    strokeWidth : anyOf(float, :class:`ExprRef`)

    tension : anyOf(float, :class:`ExprRef`)

    text : anyOf(:class:`Text`, :class:`ExprRef`)

    theta : anyOf(float, :class:`ExprRef`)
        For arc marks, the arc length in radians if theta2 is not specified, otherwise the
        start arc angle. (A value of 0 indicates up or “north”, increasing values proceed
        clockwise.)

        For text marks, polar coordinate angle in radians.
    theta2 : anyOf(float, :class:`ExprRef`)
        The end angle of arc marks in radians. A value of 0 indicates up or “north”,
        increasing values proceed clockwise.
    timeUnitBand : float
        Default relative band size for a time unit. If set to ``1``, the bandwidth of the
        marks will be equal to the time unit band step. If set to ``0.5``, bandwidth of the
        marks will be half of the time unit band step.
    timeUnitBandPosition : float
        Default relative band position for a time unit. If set to ``0``, the marks will be
        positioned at the beginning of the time unit band step. If set to ``0.5``, the marks
        will be positioned in the middle of the time unit band step.
    tooltip : anyOf(float, string, boolean, :class:`TooltipContent`, :class:`ExprRef`, None)
        The tooltip text string to show upon mouse hover or an object defining which fields
        should the tooltip be derived from.


        * If ``tooltip`` is ``true`` or ``{"content": "encoding"}``, then all fields from
          ``encoding`` will be used. - If ``tooltip`` is ``{"content": "data"}``, then all
          fields that appear in the highlighted data point will be used. - If set to
          ``null`` or ``false``, then no tooltip will be used.

        See the `tooltip <https://vega.github.io/vega-lite/docs/tooltip.html>`__
        documentation for a detailed discussion about tooltip  in Vega-Lite.

        **Default value:** ``null``
    url : anyOf(:class:`URI`, :class:`ExprRef`)

    width : anyOf(float, :class:`ExprRef`)

    x : anyOf(float, string, :class:`ExprRef`)
        X coordinates of the marks, or width of horizontal ``"bar"`` and ``"area"`` without
        specified ``x2`` or ``width``.

        The ``value`` of this channel can be a number or a string ``"width"`` for the width
        of the plot.
    x2 : anyOf(float, string, :class:`ExprRef`)
        X2 coordinates for ranged ``"area"``, ``"bar"``, ``"rect"``, and  ``"rule"``.

        The ``value`` of this channel can be a number or a string ``"width"`` for the width
        of the plot.
    y : anyOf(float, string, :class:`ExprRef`)
        Y coordinates of the marks, or height of vertical ``"bar"`` and ``"area"`` without
        specified ``y2`` or ``height``.

        The ``value`` of this channel can be a number or a string ``"height"`` for the
        height of the plot.
    y2 : anyOf(float, string, :class:`ExprRef`)
        Y2 coordinates for ranged ``"area"``, ``"bar"``, ``"rect"``, and  ``"rule"``.

        The ``value`` of this channel can be a number or a string ``"height"`` for the
        height of the plot.
    """
    _schema = {'$ref': '#/definitions/RectConfig'}

    def __init__(self, align=Undefined, angle=Undefined, aria=Undefined, ariaRole=Undefined,
                 ariaRoleDescription=Undefined, aspect=Undefined, baseline=Undefined,
                 binSpacing=Undefined, blend=Undefined, color=Undefined, continuousBandSize=Undefined,
                 cornerRadius=Undefined, cornerRadiusBottomLeft=Undefined,
                 cornerRadiusBottomRight=Undefined, cornerRadiusTopLeft=Undefined,
                 cornerRadiusTopRight=Undefined, cursor=Undefined, description=Undefined, dir=Undefined,
                 discreteBandSize=Undefined, dx=Undefined, dy=Undefined, ellipsis=Undefined,
                 endAngle=Undefined, fill=Undefined, fillOpacity=Undefined, filled=Undefined,
                 font=Undefined, fontSize=Undefined, fontStyle=Undefined, fontWeight=Undefined,
                 height=Undefined, href=Undefined, innerRadius=Undefined, interpolate=Undefined,
                 invalid=Undefined, limit=Undefined, lineBreak=Undefined, lineHeight=Undefined,
                 opacity=Undefined, order=Undefined, orient=Undefined, outerRadius=Undefined,
                 padAngle=Undefined, radius=Undefined, radius2=Undefined, shape=Undefined,
                 size=Undefined, smooth=Undefined, startAngle=Undefined, stroke=Undefined,
                 strokeCap=Undefined, strokeDash=Undefined, strokeDashOffset=Undefined,
                 strokeJoin=Undefined, strokeMiterLimit=Undefined, strokeOffset=Undefined,
                 strokeOpacity=Undefined, strokeWidth=Undefined, tension=Undefined, text=Undefined,
                 theta=Undefined, theta2=Undefined, timeUnitBand=Undefined,
                 timeUnitBandPosition=Undefined, tooltip=Undefined, url=Undefined, width=Undefined,
                 x=Undefined, x2=Undefined, y=Undefined, y2=Undefined, **kwds):
        super(RectConfig, self).__init__(align=align, angle=angle, aria=aria, ariaRole=ariaRole,
                                         ariaRoleDescription=ariaRoleDescription, aspect=aspect,
                                         baseline=baseline, binSpacing=binSpacing, blend=blend,
                                         color=color, continuousBandSize=continuousBandSize,
                                         cornerRadius=cornerRadius,
                                         cornerRadiusBottomLeft=cornerRadiusBottomLeft,
                                         cornerRadiusBottomRight=cornerRadiusBottomRight,
                                         cornerRadiusTopLeft=cornerRadiusTopLeft,
                                         cornerRadiusTopRight=cornerRadiusTopRight, cursor=cursor,
                                         description=description, dir=dir,
                                         discreteBandSize=discreteBandSize, dx=dx, dy=dy,
                                         ellipsis=ellipsis, endAngle=endAngle, fill=fill,
                                         fillOpacity=fillOpacity, filled=filled, font=font,
                                         fontSize=fontSize, fontStyle=fontStyle, fontWeight=fontWeight,
                                         height=height, href=href, innerRadius=innerRadius,
                                         interpolate=interpolate, invalid=invalid, limit=limit,
                                         lineBreak=lineBreak, lineHeight=lineHeight, opacity=opacity,
                                         order=order, orient=orient, outerRadius=outerRadius,
                                         padAngle=padAngle, radius=radius, radius2=radius2, shape=shape,
                                         size=size, smooth=smooth, startAngle=startAngle, stroke=stroke,
                                         strokeCap=strokeCap, strokeDash=strokeDash,
                                         strokeDashOffset=strokeDashOffset, strokeJoin=strokeJoin,
                                         strokeMiterLimit=strokeMiterLimit, strokeOffset=strokeOffset,
                                         strokeOpacity=strokeOpacity, strokeWidth=strokeWidth,
                                         tension=tension, text=text, theta=theta, theta2=theta2,
                                         timeUnitBand=timeUnitBand,
                                         timeUnitBandPosition=timeUnitBandPosition, tooltip=tooltip,
                                         url=url, width=width, x=x, x2=x2, y=y, y2=y2, **kwds)


class RepeatMapping(VegaLiteSchema):
    """RepeatMapping schema wrapper

    Mapping(required=[])

    Attributes
    ----------

    column : List(string)
        An array of fields to be repeated horizontally.
    row : List(string)
        An array of fields to be repeated vertically.
    """
    _schema = {'$ref': '#/definitions/RepeatMapping'}

    def __init__(self, column=Undefined, row=Undefined, **kwds):
        super(RepeatMapping, self).__init__(column=column, row=row, **kwds)


class RepeatRef(Field):
    """RepeatRef schema wrapper

    Mapping(required=[repeat])
    Reference to a repeated value.

    Attributes
    ----------

    repeat : enum('row', 'column', 'repeat', 'layer')

    """
    _schema = {'$ref': '#/definitions/RepeatRef'}

    def __init__(self, repeat=Undefined, **kwds):
        super(RepeatRef, self).__init__(repeat=repeat, **kwds)


class Resolve(VegaLiteSchema):
    """Resolve schema wrapper

    Mapping(required=[])
    Defines how scales, axes, and legends from different specs should be combined. Resolve is a
    mapping from ``scale``, ``axis``, and ``legend`` to a mapping from channels to resolutions.
    Scales and guides can be resolved to be ``"independent"`` or ``"shared"``.

    Attributes
    ----------

    axis : :class:`AxisResolveMap`

    legend : :class:`LegendResolveMap`

    scale : :class:`ScaleResolveMap`

    """
    _schema = {'$ref': '#/definitions/Resolve'}

    def __init__(self, axis=Undefined, legend=Undefined, scale=Undefined, **kwds):
        super(Resolve, self).__init__(axis=axis, legend=legend, scale=scale, **kwds)


class ResolveMode(VegaLiteSchema):
    """ResolveMode schema wrapper

    enum('independent', 'shared')
    """
    _schema = {'$ref': '#/definitions/ResolveMode'}

    def __init__(self, *args):
        super(ResolveMode, self).__init__(*args)


class RowColLayoutAlign(VegaLiteSchema):
    """RowColLayoutAlign schema wrapper

    Mapping(required=[])

    Attributes
    ----------

    column : :class:`LayoutAlign`

    row : :class:`LayoutAlign`

    """
    _schema = {'$ref': '#/definitions/RowCol<LayoutAlign>'}

    def __init__(self, column=Undefined, row=Undefined, **kwds):
        super(RowColLayoutAlign, self).__init__(column=column, row=row, **kwds)


class RowColboolean(VegaLiteSchema):
    """RowColboolean schema wrapper

    Mapping(required=[])

    Attributes
    ----------

    column : boolean

    row : boolean

    """
    _schema = {'$ref': '#/definitions/RowCol<boolean>'}

    def __init__(self, column=Undefined, row=Undefined, **kwds):
        super(RowColboolean, self).__init__(column=column, row=row, **kwds)


class RowColnumber(VegaLiteSchema):
    """RowColnumber schema wrapper

    Mapping(required=[])

    Attributes
    ----------

    column : float

    row : float

    """
    _schema = {'$ref': '#/definitions/RowCol<number>'}

    def __init__(self, column=Undefined, row=Undefined, **kwds):
        super(RowColnumber, self).__init__(column=column, row=row, **kwds)


class RowColumnEncodingFieldDef(VegaLiteSchema):
    """RowColumnEncodingFieldDef schema wrapper

    Mapping(required=[])

    Attributes
    ----------

    aggregate : :class:`Aggregate`
        Aggregation function for the field (e.g., ``"mean"``, ``"sum"``, ``"median"``,
        ``"min"``, ``"max"``, ``"count"`` ).

        **Default value:** ``undefined`` (None)

        **See also:** `aggregate <https://vega.github.io/vega-lite/docs/aggregate.html>`__
        documentation.
    align : :class:`LayoutAlign`
        The alignment to apply to row/column facet's subplot. The supported string values
        are ``"all"``, ``"each"``, and ``"none"``.


        * For ``"none"``, a flow layout will be used, in which adjacent subviews are simply
          placed one after the other. - For ``"each"``, subviews will be aligned into a
          clean grid structure, but each row or column may be of variable size. - For
          ``"all"``, subviews will be aligned and each row or column will be sized
          identically based on the maximum observed size. String values for this property
          will be applied to both grid rows and columns.

        **Default value:** ``"all"``.
    band : float
        For rect-based marks ( ``rect``, ``bar``, and ``image`` ), mark size relative to
        bandwidth of `band scales
        <https://vega.github.io/vega-lite/docs/scale.html#band>`__, bins or time units. If
        set to ``1``, the mark size is set to the bandwidth, the bin interval, or the time
        unit interval. If set to ``0.5``, the mark size is half of the bandwidth or the time
        unit interval.

        For other marks, relative position on a band of a stacked, binned, time unit or band
        scale. If set to ``0``, the marks will be positioned at the beginning of the band.
        If set to ``0.5``, the marks will be positioned in the middle of the band.
    bin : anyOf(boolean, :class:`BinParams`, None)
        A flag for binning a ``quantitative`` field, `an object defining binning parameters
        <https://vega.github.io/vega-lite/docs/bin.html#params>`__, or indicating that the
        data for ``x`` or ``y`` channel are binned before they are imported into Vega-Lite (
        ``"binned"`` ).


        If ``true``, default `binning parameters
        <https://vega.github.io/vega-lite/docs/bin.html>`__ will be applied.

        If ``"binned"``, this indicates that the data for the ``x`` (or ``y`` ) channel are
        already binned. You can map the bin-start field to ``x`` (or ``y`` ) and the bin-end
        field to ``x2`` (or ``y2`` ). The scale and axis will be formatted similar to
        binning in Vega-Lite.  To adjust the axis ticks based on the bin step, you can also
        set the axis's `tickMinStep
        <https://vega.github.io/vega-lite/docs/axis.html#ticks>`__ property.

        **Default value:** ``false``

        **See also:** `bin <https://vega.github.io/vega-lite/docs/bin.html>`__
        documentation.
    center : boolean
        Boolean flag indicating if facet's subviews should be centered relative to their
        respective rows or columns.

        **Default value:** ``false``
    field : :class:`Field`
        **Required.** A string defining the name of the field from which to pull a data
        value or an object defining iterated values from the `repeat
        <https://vega.github.io/vega-lite/docs/repeat.html>`__ operator.

        **See also:** `field <https://vega.github.io/vega-lite/docs/field.html>`__
        documentation.

        **Notes:** 1)  Dots ( ``.`` ) and brackets ( ``[`` and ``]`` ) can be used to access
        nested objects (e.g., ``"field": "foo.bar"`` and ``"field": "foo['bar']"`` ). If
        field names contain dots or brackets but are not nested, you can use ``\\`` to
        escape dots and brackets (e.g., ``"a\\.b"`` and ``"a\\[0\\]"`` ). See more details
        about escaping in the `field documentation
        <https://vega.github.io/vega-lite/docs/field.html>`__. 2) ``field`` is not required
        if ``aggregate`` is ``count``.
    header : :class:`Header`
        An object defining properties of a facet's header.
    sort : anyOf(:class:`SortArray`, :class:`SortOrder`, :class:`EncodingSortField`, None)
        Sort order for the encoded field.

        For continuous fields (quantitative or temporal), ``sort`` can be either
        ``"ascending"`` or ``"descending"``.

        For discrete fields, ``sort`` can be one of the following: - ``"ascending"`` or
        ``"descending"`` -- for sorting by the values' natural order in JavaScript. - `A
        sort field definition
        <https://vega.github.io/vega-lite/docs/sort.html#sort-field>`__ for sorting by
        another field. - `An array specifying the field values in preferred order
        <https://vega.github.io/vega-lite/docs/sort.html#sort-array>`__. In this case, the
        sort order will obey the values in the array, followed by any unspecified values in
        their original order. For discrete time field, values in the sort array can be
        `date-time definition objects <types#datetime>`__. In addition, for time units
        ``"month"`` and ``"day"``, the values can be the month or day names (case
        insensitive) or their 3-letter initials (e.g., ``"Mon"``, ``"Tue"`` ). - ``null``
        indicating no sort.

        **Default value:** ``"ascending"``

        **Note:** ``null`` is not supported for ``row`` and ``column``.
    spacing : float
        The spacing in pixels between facet's sub-views.

        **Default value** : Depends on ``"spacing"`` property of `the view composition
        configuration <https://vega.github.io/vega-lite/docs/config.html#view-config>`__ (
        ``20`` by default)
    timeUnit : anyOf(:class:`TimeUnit`, :class:`TimeUnitParams`)
        Time unit (e.g., ``year``, ``yearmonth``, ``month``, ``hours`` ) for a temporal
        field. or `a temporal field that gets casted as ordinal
        <https://vega.github.io/vega-lite/docs/type.html#cast>`__.

        **Default value:** ``undefined`` (None)

        **See also:** `timeUnit <https://vega.github.io/vega-lite/docs/timeunit.html>`__
        documentation.
    title : anyOf(:class:`Text`, None)
        A title for the field. If ``null``, the title will be removed.

        **Default value:**  derived from the field's name and transformation function (
        ``aggregate``, ``bin`` and ``timeUnit`` ). If the field has an aggregate function,
        the function is displayed as part of the title (e.g., ``"Sum of Profit"`` ). If the
        field is binned or has a time unit applied, the applied function is shown in
        parentheses (e.g., ``"Profit (binned)"``, ``"Transaction Date (year-month)"`` ).
        Otherwise, the title is simply the field name.

        **Notes** :

        1) You can customize the default field title format by providing the `fieldTitle
        <https://vega.github.io/vega-lite/docs/config.html#top-level-config>`__ property in
        the `config <https://vega.github.io/vega-lite/docs/config.html>`__ or `fieldTitle
        function via the compile function's options
        <https://vega.github.io/vega-lite/docs/compile.html#field-title>`__.

        2) If both field definition's ``title`` and axis, header, or legend ``title`` are
        defined, axis/header/legend title will be used.
    type : :class:`StandardType`
        The type of measurement ( ``"quantitative"``, ``"temporal"``, ``"ordinal"``, or
        ``"nominal"`` ) for the encoded field or constant value ( ``datum`` ). It can also
        be a ``"geojson"`` type for encoding `'geoshape'
        <https://vega.github.io/vega-lite/docs/geoshape.html>`__.

        Vega-Lite automatically infers data types in many cases as discussed below. However,
        type is required for a field if: (1) the field is not nominal and the field encoding
        has no specified ``aggregate`` (except ``argmin`` and ``argmax`` ), ``bin``, scale
        type, custom ``sort`` order, nor ``timeUnit`` or (2) if you wish to use an ordinal
        scale for a field with ``bin`` or ``timeUnit``.

        **Default value:**

        1) For a data ``field``, ``"nominal"`` is the default data type unless the field
        encoding has ``aggregate``, ``channel``, ``bin``, scale type, ``sort``, or
        ``timeUnit`` that satisfies the following criteria: - ``"quantitative"`` is the
        default type if (1) the encoded field contains ``bin`` or ``aggregate`` except
        ``"argmin"`` and ``"argmax"``, (2) the encoding channel is ``latitude`` or
        ``longitude`` channel or (3) if the specified scale type is `a quantitative scale
        <https://vega.github.io/vega-lite/docs/scale.html#type>`__. - ``"temporal"`` is the
        default type if (1) the encoded field contains ``timeUnit`` or (2) the specified
        scale type is a time or utc scale - ``ordinal""`` is the default type if (1) the
        encoded field contains a `custom sort order
        <https://vega.github.io/vega-lite/docs/sort.html#specifying-custom-sort-order>`__,
        (2) the specified scale type is an ordinal/point/band scale, or (3) the encoding
        channel is ``order``.

        2) For a constant value in data domain ( ``datum`` ): - ``"quantitative"`` if the
        datum is a number - ``"nominal"`` if the datum is a string - ``"temporal"`` if the
        datum is `a date time object
        <https://vega.github.io/vega-lite/docs/datetime.html>`__

        **Note:** - Data ``type`` describes the semantics of the data rather than the
        primitive data types (number, string, etc.). The same primitive data type can have
        different types of measurement. For example, numeric data can represent
        quantitative, ordinal, or nominal data. - Data values for a temporal field can be
        either a date-time string (e.g., ``"2015-03-07 12:32:17"``, ``"17:01"``,
        ``"2015-03-16"``. ``"2015"`` ) or a timestamp number (e.g., ``1552199579097`` ). -
        When using with `bin <https://vega.github.io/vega-lite/docs/bin.html>`__, the
        ``type`` property can be either ``"quantitative"`` (for using a linear bin scale) or
        `"ordinal" (for using an ordinal bin scale)
        <https://vega.github.io/vega-lite/docs/type.html#cast-bin>`__. - When using with
        `timeUnit <https://vega.github.io/vega-lite/docs/timeunit.html>`__, the ``type``
        property can be either ``"temporal"`` (default, for using a temporal scale) or
        `"ordinal" (for using an ordinal scale)
        <https://vega.github.io/vega-lite/docs/type.html#cast-bin>`__. - When using with
        `aggregate <https://vega.github.io/vega-lite/docs/aggregate.html>`__, the ``type``
        property refers to the post-aggregation data type. For example, we can calculate
        count ``distinct`` of a categorical field ``"cat"`` using ``{"aggregate":
        "distinct", "field": "cat"}``. The ``"type"`` of the aggregate output is
        ``"quantitative"``. - Secondary channels (e.g., ``x2``, ``y2``, ``xError``,
        ``yError`` ) do not have ``type`` as they must have exactly the same type as their
        primary channels (e.g., ``x``, ``y`` ).

        **See also:** `type <https://vega.github.io/vega-lite/docs/type.html>`__
        documentation.
    """
    _schema = {'$ref': '#/definitions/RowColumnEncodingFieldDef'}

    def __init__(self, aggregate=Undefined, align=Undefined, band=Undefined, bin=Undefined,
                 center=Undefined, field=Undefined, header=Undefined, sort=Undefined, spacing=Undefined,
                 timeUnit=Undefined, title=Undefined, type=Undefined, **kwds):
        super(RowColumnEncodingFieldDef, self).__init__(aggregate=aggregate, align=align, band=band,
                                                        bin=bin, center=center, field=field,
                                                        header=header, sort=sort, spacing=spacing,
                                                        timeUnit=timeUnit, title=title, type=type,
                                                        **kwds)


class Scale(VegaLiteSchema):
    """Scale schema wrapper

    Mapping(required=[])

    Attributes
    ----------

    align : anyOf(float, :class:`ExprRef`)
        The alignment of the steps within the scale range.

        This value must lie in the range ``[0,1]``. A value of ``0.5`` indicates that the
        steps should be centered within the range. A value of ``0`` or ``1`` may be used to
        shift the bands to one side, say to position them adjacent to an axis.

        **Default value:** ``0.5``
    base : anyOf(float, :class:`ExprRef`)
        The logarithm base of the ``log`` scale (default ``10`` ).
    bins : :class:`ScaleBins`
        Bin boundaries can be provided to scales as either an explicit array of bin
        boundaries or as a bin specification object. The legal values are: - An `array
        <../types/#Array>`__ literal of bin boundary values. For example, ``[0, 5, 10, 15,
        20]``. The array must include both starting and ending boundaries. The previous
        example uses five values to indicate a total of four bin intervals: [0-5), [5-10),
        [10-15), [15-20]. Array literals may include signal references as elements. - A `bin
        specification object <https://vega.github.io/vega-lite/docs/scale.html#bins>`__ that
        indicates the bin *step* size, and optionally the *start* and *stop* boundaries. -
        An array of bin boundaries over the scale domain. If provided, axes and legends will
        use the bin boundaries to inform the choice of tick marks and text labels.
    clamp : anyOf(boolean, :class:`ExprRef`)
        If ``true``, values that exceed the data domain are clamped to either the minimum or
        maximum range value

        **Default value:** derived from the `scale config
        <https://vega.github.io/vega-lite/docs/config.html#scale-config>`__ 's ``clamp`` (
        ``true`` by default).
    constant : anyOf(float, :class:`ExprRef`)
        A constant determining the slope of the symlog function around zero. Only used for
        ``symlog`` scales.

        **Default value:** ``1``
    domain : anyOf(List(anyOf(None, string, float, boolean, :class:`DateTime`,
    :class:`ExprRef`)), string, :class:`SelectionExtent`, :class:`DomainUnionWith`,
    :class:`ExprRef`)
        Customized domain values in the form of constant values or dynamic values driven by
        a selection.

        1) Constant ``domain`` for *quantitative* fields can take one of the following
        forms:


        * A two-element array with minimum and maximum values. To create a diverging scale,
          this two-element array can be combined with the ``domainMid`` property. - An array
          with more than two entries, for `Piecewise quantitative scales
          <https://vega.github.io/vega-lite/docs/scale.html#piecewise>`__. - A string value
          ``"unaggregated"``, if the input field is aggregated, to indicate that the domain
          should include the raw data values prior to the aggregation.

        2) Constant ``domain`` for *temporal* fields can be a two-element array with minimum
        and maximum values, in the form of either timestamps or the `DateTime definition
        objects <https://vega.github.io/vega-lite/docs/types.html#datetime>`__.

        3) Constant ``domain`` for *ordinal* and *nominal* fields can be an array that lists
        valid input values.

        4) To combine (union) specified constant domain with the field's values, ``domain``
        can be an object with a ``unionWith`` property that specify constant domain to be
        combined. For example, ``domain: {unionWith: [0, 100]}`` for a quantitative scale
        means that the scale domain always includes ``[0, 100]``, but will include other
        values in the fields beyond ``[0, 100]``.

        5) Domain can also takes an object defining a field or encoding of a selection that
        `interactively determines
        <https://vega.github.io/vega-lite/docs/selection.html#scale-domains>`__ the scale
        domain.
    domainMax : anyOf(float, :class:`DateTime`, :class:`ExprRef`)
        Sets the maximum value in the scale domain, overriding the ``domain`` property. This
        property is only intended for use with scales having continuous domains.
    domainMid : anyOf(float, :class:`ExprRef`)
        Inserts a single mid-point value into a two-element domain. The mid-point value must
        lie between the domain minimum and maximum values. This property can be useful for
        setting a midpoint for `diverging color scales
        <https://vega.github.io/vega-lite/docs/scale.html#piecewise>`__. The domainMid
        property is only intended for use with scales supporting continuous, piecewise
        domains.
    domainMin : anyOf(float, :class:`DateTime`, :class:`ExprRef`)
        Sets the minimum value in the scale domain, overriding the domain property. This
        property is only intended for use with scales having continuous domains.
    exponent : anyOf(float, :class:`ExprRef`)
        The exponent of the ``pow`` scale.
    interpolate : anyOf(:class:`ScaleInterpolateEnum`, :class:`ExprRef`,
    :class:`ScaleInterpolateParams`)
        The interpolation method for range values. By default, a general interpolator for
        numbers, dates, strings and colors (in HCL space) is used. For color ranges, this
        property allows interpolation in alternative color spaces. Legal values include
        ``rgb``, ``hsl``, ``hsl-long``, ``lab``, ``hcl``, ``hcl-long``, ``cubehelix`` and
        ``cubehelix-long`` ('-long' variants use longer paths in polar coordinate spaces).
        If object-valued, this property accepts an object with a string-valued *type*
        property and an optional numeric *gamma* property applicable to rgb and cubehelix
        interpolators. For more, see the `d3-interpolate documentation
        <https://github.com/d3/d3-interpolate>`__.


        * **Default value:** ``hcl``
    nice : anyOf(boolean, float, :class:`TimeInterval`, :class:`TimeIntervalStep`,
    :class:`ExprRef`)
        Extending the domain so that it starts and ends on nice round values. This method
        typically modifies the scale’s domain, and may only extend the bounds to the nearest
        round value. Nicing is useful if the domain is computed from data and may be
        irregular. For example, for a domain of *[0.201479…, 0.996679…]*, a nice domain
        might be *[0.2, 1.0]*.

        For quantitative scales such as linear, ``nice`` can be either a boolean flag or a
        number. If ``nice`` is a number, it will represent a desired tick count. This allows
        greater control over the step size used to extend the bounds, guaranteeing that the
        returned ticks will exactly cover the domain.

        For temporal fields with time and utc scales, the ``nice`` value can be a string
        indicating the desired time interval. Legal values are ``"millisecond"``,
        ``"second"``, ``"minute"``, ``"hour"``, ``"day"``, ``"week"``, ``"month"``, and
        ``"year"``. Alternatively, ``time`` and ``utc`` scales can accept an object-valued
        interval specifier of the form ``{"interval": "month", "step": 3}``, which includes
        a desired number of interval steps. Here, the domain would snap to quarter (Jan,
        Apr, Jul, Oct) boundaries.

        **Default value:** ``true`` for unbinned *quantitative* fields; ``false`` otherwise.
    padding : anyOf(float, :class:`ExprRef`)
        For * `continuous <https://vega.github.io/vega-lite/docs/scale.html#continuous>`__ *
        scales, expands the scale domain to accommodate the specified number of pixels on
        each of the scale range. The scale range must represent pixels for this parameter to
        function as intended. Padding adjustment is performed prior to all other
        adjustments, including the effects of the  ``zero``,  ``nice``,  ``domainMin``, and
        ``domainMax``  properties.

        For * `band <https://vega.github.io/vega-lite/docs/scale.html#band>`__ * scales,
        shortcut for setting ``paddingInner`` and ``paddingOuter`` to the same value.

        For * `point <https://vega.github.io/vega-lite/docs/scale.html#point>`__ * scales,
        alias for ``paddingOuter``.

        **Default value:** For *continuous* scales, derived from the `scale config
        <https://vega.github.io/vega-lite/docs/scale.html#config>`__ 's
        ``continuousPadding``. For *band and point* scales, see ``paddingInner`` and
        ``paddingOuter``. By default, Vega-Lite sets padding such that *width/height =
        number of unique values * step*.
    paddingInner : anyOf(float, :class:`ExprRef`)
        The inner padding (spacing) within each band step of band scales, as a fraction of
        the step size. This value must lie in the range [0,1].

        For point scale, this property is invalid as point scales do not have internal band
        widths (only step sizes between bands).

        **Default value:** derived from the `scale config
        <https://vega.github.io/vega-lite/docs/scale.html#config>`__ 's
        ``bandPaddingInner``.
    paddingOuter : anyOf(float, :class:`ExprRef`)
        The outer padding (spacing) at the ends of the range of band and point scales, as a
        fraction of the step size. This value must lie in the range [0,1].

        **Default value:** derived from the `scale config
        <https://vega.github.io/vega-lite/docs/scale.html#config>`__ 's ``bandPaddingOuter``
        for band scales and ``pointPadding`` for point scales. By default, Vega-Lite sets
        outer padding such that *width/height = number of unique values * step*.
    range : anyOf(:class:`RangeEnum`, List(anyOf(float, string, List(float), :class:`ExprRef`)),
    Mapping(required=[field]))
        The range of the scale. One of:


        A string indicating a `pre-defined named scale range
        <https://vega.github.io/vega-lite/docs/scale.html#range-config>`__ (e.g., example,
        ``"symbol"``, or ``"diverging"`` ).

        For `continuous scales
        <https://vega.github.io/vega-lite/docs/scale.html#continuous>`__, two-element array
        indicating  minimum and maximum values, or an array with more than two entries for
        specifying a `piecewise scale
        <https://vega.github.io/vega-lite/docs/scale.html#piecewise>`__.

        For `discrete <https://vega.github.io/vega-lite/docs/scale.html#discrete>`__ and
        `discretizing <https://vega.github.io/vega-lite/docs/scale.html#discretizing>`__
        scales, an array of desired output values or an object with a ``field`` property
        representing the range values.  For example, if a field ``color`` contains CSS color
        names, we can set ``range`` to ``{field: "color"}``.

        **Notes:**

        1) For color scales you can also specify a color `scheme
        <https://vega.github.io/vega-lite/docs/scale.html#scheme>`__ instead of ``range``.

        2) Any directly specified ``range`` for ``x`` and ``y`` channels will be ignored.
        Range can be customized via the view's corresponding `size
        <https://vega.github.io/vega-lite/docs/size.html>`__ ( ``width`` and ``height`` ).
    rangeMax : anyOf(float, string, :class:`ExprRef`)
        Sets the maximum value in the scale range, overriding the ``range`` property or the
        default range. This property is only intended for use with scales having continuous
        ranges.
    rangeMin : anyOf(float, string, :class:`ExprRef`)
        Sets the minimum value in the scale range, overriding the ``range`` property or the
        default range. This property is only intended for use with scales having continuous
        ranges.
    reverse : anyOf(boolean, :class:`ExprRef`)
        If true, reverses the order of the scale range. **Default value:** ``false``.
    round : anyOf(boolean, :class:`ExprRef`)
        If ``true``, rounds numeric output values to integers. This can be helpful for
        snapping to the pixel grid.

        **Default value:** ``false``.
    scheme : anyOf(string, :class:`SchemeParams`, :class:`ExprRef`)
        A string indicating a color `scheme
        <https://vega.github.io/vega-lite/docs/scale.html#scheme>`__ name (e.g.,
        ``"category10"`` or ``"blues"`` ) or a `scheme parameter object
        <https://vega.github.io/vega-lite/docs/scale.html#scheme-params>`__.

        Discrete color schemes may be used with `discrete
        <https://vega.github.io/vega-lite/docs/scale.html#discrete>`__ or `discretizing
        <https://vega.github.io/vega-lite/docs/scale.html#discretizing>`__ scales.
        Continuous color schemes are intended for use with color scales.

        For the full list of supported schemes, please refer to the `Vega Scheme
        <https://vega.github.io/vega/docs/schemes/#reference>`__ reference.
    type : :class:`ScaleType`
        The type of scale. Vega-Lite supports the following categories of scale types:

        1) `Continuous Scales
        <https://vega.github.io/vega-lite/docs/scale.html#continuous>`__ -- mapping
        continuous domains to continuous output ranges ( `"linear"
        <https://vega.github.io/vega-lite/docs/scale.html#linear>`__, `"pow"
        <https://vega.github.io/vega-lite/docs/scale.html#pow>`__, `"sqrt"
        <https://vega.github.io/vega-lite/docs/scale.html#sqrt>`__, `"symlog"
        <https://vega.github.io/vega-lite/docs/scale.html#symlog>`__, `"log"
        <https://vega.github.io/vega-lite/docs/scale.html#log>`__, `"time"
        <https://vega.github.io/vega-lite/docs/scale.html#time>`__, `"utc"
        <https://vega.github.io/vega-lite/docs/scale.html#utc>`__.

        2) `Discrete Scales <https://vega.github.io/vega-lite/docs/scale.html#discrete>`__
        -- mapping discrete domains to discrete ( `"ordinal"
        <https://vega.github.io/vega-lite/docs/scale.html#ordinal>`__ ) or continuous (
        `"band" <https://vega.github.io/vega-lite/docs/scale.html#band>`__ and `"point"
        <https://vega.github.io/vega-lite/docs/scale.html#point>`__ ) output ranges.

        3) `Discretizing Scales
        <https://vega.github.io/vega-lite/docs/scale.html#discretizing>`__ -- mapping
        continuous domains to discrete output ranges `"bin-ordinal"
        <https://vega.github.io/vega-lite/docs/scale.html#bin-ordinal>`__, `"quantile"
        <https://vega.github.io/vega-lite/docs/scale.html#quantile>`__, `"quantize"
        <https://vega.github.io/vega-lite/docs/scale.html#quantize>`__ and `"threshold"
        <https://vega.github.io/vega-lite/docs/scale.html#threshold>`__.

        **Default value:** please see the `scale type table
        <https://vega.github.io/vega-lite/docs/scale.html#type>`__.
    zero : anyOf(boolean, :class:`ExprRef`)
        If ``true``, ensures that a zero baseline value is included in the scale domain.

        **Default value:** ``true`` for x and y channels if the quantitative field is not
        binned and no custom ``domain`` is provided; ``false`` otherwise.

        **Note:** Log, time, and utc scales do not support ``zero``.
    """
    _schema = {'$ref': '#/definitions/Scale'}

    def __init__(self, align=Undefined, base=Undefined, bins=Undefined, clamp=Undefined,
                 constant=Undefined, domain=Undefined, domainMax=Undefined, domainMid=Undefined,
                 domainMin=Undefined, exponent=Undefined, interpolate=Undefined, nice=Undefined,
                 padding=Undefined, paddingInner=Undefined, paddingOuter=Undefined, range=Undefined,
                 rangeMax=Undefined, rangeMin=Undefined, reverse=Undefined, round=Undefined,
                 scheme=Undefined, type=Undefined, zero=Undefined, **kwds):
        super(Scale, self).__init__(align=align, base=base, bins=bins, clamp=clamp, constant=constant,
                                    domain=domain, domainMax=domainMax, domainMid=domainMid,
                                    domainMin=domainMin, exponent=exponent, interpolate=interpolate,
                                    nice=nice, padding=padding, paddingInner=paddingInner,
                                    paddingOuter=paddingOuter, range=range, rangeMax=rangeMax,
                                    rangeMin=rangeMin, reverse=reverse, round=round, scheme=scheme,
                                    type=type, zero=zero, **kwds)


class ScaleBins(VegaLiteSchema):
    """ScaleBins schema wrapper

    anyOf(List(float), :class:`ScaleBinParams`)
    """
    _schema = {'$ref': '#/definitions/ScaleBins'}

    def __init__(self, *args, **kwds):
        super(ScaleBins, self).__init__(*args, **kwds)


class ScaleBinParams(ScaleBins):
    """ScaleBinParams schema wrapper

    Mapping(required=[step])

    Attributes
    ----------

    step : float
        The step size defining the bin interval width.
    start : float
        The starting (lowest-valued) bin boundary.

        **Default value:** The lowest value of the scale domain will be used.
    stop : float
        The stopping (highest-valued) bin boundary.

        **Default value:** The highest value of the scale domain will be used.
    """
    _schema = {'$ref': '#/definitions/ScaleBinParams'}

    def __init__(self, step=Undefined, start=Undefined, stop=Undefined, **kwds):
        super(ScaleBinParams, self).__init__(step=step, start=start, stop=stop, **kwds)


class ScaleConfig(VegaLiteSchema):
    """ScaleConfig schema wrapper

    Mapping(required=[])

    Attributes
    ----------

    bandPaddingInner : anyOf(float, :class:`ExprRef`)
        Default inner padding for ``x`` and ``y`` band-ordinal scales.

        **Default value:** - ``barBandPaddingInner`` for bar marks ( ``0.1`` by default) -
        ``rectBandPaddingInner`` for rect and other marks ( ``0`` by default)
    bandPaddingOuter : anyOf(float, :class:`ExprRef`)
        Default outer padding for ``x`` and ``y`` band-ordinal scales.

        **Default value:** ``paddingInner/2`` (which makes *width/height = number of unique
        values * step* )
    barBandPaddingInner : anyOf(float, :class:`ExprRef`)
        Default inner padding for ``x`` and ``y`` band-ordinal scales of ``"bar"`` marks.

        **Default value:** ``0.1``
    clamp : anyOf(boolean, :class:`ExprRef`)
        If true, values that exceed the data domain are clamped to either the minimum or
        maximum range value
    continuousPadding : anyOf(float, :class:`ExprRef`)
        Default padding for continuous scales.

        **Default:** ``5`` for continuous x-scale of a vertical bar and continuous y-scale
        of a horizontal bar.; ``0`` otherwise.
    maxBandSize : float
        The default max value for mapping quantitative fields to bar's size/bandSize.

        If undefined (default), we will use the axis's size (width or height) - 1.
    maxFontSize : float
        The default max value for mapping quantitative fields to text's size/fontSize.

        **Default value:** ``40``
    maxOpacity : float
        Default max opacity for mapping a field to opacity.

        **Default value:** ``0.8``
    maxSize : float
        Default max value for point size scale.
    maxStrokeWidth : float
        Default max strokeWidth for the scale of strokeWidth for rule and line marks and of
        size for trail marks.

        **Default value:** ``4``
    minBandSize : float
        The default min value for mapping quantitative fields to bar and tick's
        size/bandSize scale with zero=false.

        **Default value:** ``2``
    minFontSize : float
        The default min value for mapping quantitative fields to tick's size/fontSize scale
        with zero=false

        **Default value:** ``8``
    minOpacity : float
        Default minimum opacity for mapping a field to opacity.

        **Default value:** ``0.3``
    minSize : float
        Default minimum value for point size scale with zero=false.

        **Default value:** ``9``
    minStrokeWidth : float
        Default minimum strokeWidth for the scale of strokeWidth for rule and line marks and
        of size for trail marks with zero=false.

        **Default value:** ``1``
    pointPadding : anyOf(float, :class:`ExprRef`)
        Default outer padding for ``x`` and ``y`` point-ordinal scales.

        **Default value:** ``0.5`` (which makes *width/height = number of unique values *
        step* )
    quantileCount : float
        Default range cardinality for `quantile
        <https://vega.github.io/vega-lite/docs/scale.html#quantile>`__ scale.

        **Default value:** ``4``
    quantizeCount : float
        Default range cardinality for `quantize
        <https://vega.github.io/vega-lite/docs/scale.html#quantize>`__ scale.

        **Default value:** ``4``
    rectBandPaddingInner : anyOf(float, :class:`ExprRef`)
        Default inner padding for ``x`` and ``y`` band-ordinal scales of ``"rect"`` marks.

        **Default value:** ``0``
    round : anyOf(boolean, :class:`ExprRef`)
        If true, rounds numeric output values to integers. This can be helpful for snapping
        to the pixel grid. (Only available for ``x``, ``y``, and ``size`` scales.)
    useUnaggregatedDomain : boolean
        Use the source data range before aggregation as scale domain instead of aggregated
        data for aggregate axis.

        This is equivalent to setting ``domain`` to ``"unaggregate"`` for aggregated
        *quantitative* fields by default.

        This property only works with aggregate functions that produce values within the raw
        data domain ( ``"mean"``, ``"average"``, ``"median"``, ``"q1"``, ``"q3"``,
        ``"min"``, ``"max"`` ). For other aggregations that produce values outside of the
        raw data domain (e.g. ``"count"``, ``"sum"`` ), this property is ignored.

        **Default value:** ``false``
    xReverse : anyOf(boolean, :class:`ExprRef`)
        Reverse x-scale by default (useful for right-to-left charts).
    """
    _schema = {'$ref': '#/definitions/ScaleConfig'}

    def __init__(self, bandPaddingInner=Undefined, bandPaddingOuter=Undefined,
                 barBandPaddingInner=Undefined, clamp=Undefined, continuousPadding=Undefined,
                 maxBandSize=Undefined, maxFontSize=Undefined, maxOpacity=Undefined, maxSize=Undefined,
                 maxStrokeWidth=Undefined, minBandSize=Undefined, minFontSize=Undefined,
                 minOpacity=Undefined, minSize=Undefined, minStrokeWidth=Undefined,
                 pointPadding=Undefined, quantileCount=Undefined, quantizeCount=Undefined,
                 rectBandPaddingInner=Undefined, round=Undefined, useUnaggregatedDomain=Undefined,
                 xReverse=Undefined, **kwds):
        super(ScaleConfig, self).__init__(bandPaddingInner=bandPaddingInner,
                                          bandPaddingOuter=bandPaddingOuter,
                                          barBandPaddingInner=barBandPaddingInner, clamp=clamp,
                                          continuousPadding=continuousPadding, maxBandSize=maxBandSize,
                                          maxFontSize=maxFontSize, maxOpacity=maxOpacity,
                                          maxSize=maxSize, maxStrokeWidth=maxStrokeWidth,
                                          minBandSize=minBandSize, minFontSize=minFontSize,
                                          minOpacity=minOpacity, minSize=minSize,
                                          minStrokeWidth=minStrokeWidth, pointPadding=pointPadding,
                                          quantileCount=quantileCount, quantizeCount=quantizeCount,
                                          rectBandPaddingInner=rectBandPaddingInner, round=round,
                                          useUnaggregatedDomain=useUnaggregatedDomain,
                                          xReverse=xReverse, **kwds)


class ScaleInterpolateEnum(VegaLiteSchema):
    """ScaleInterpolateEnum schema wrapper

    enum('rgb', 'lab', 'hcl', 'hsl', 'hsl-long', 'hcl-long', 'cubehelix', 'cubehelix-long')
    """
    _schema = {'$ref': '#/definitions/ScaleInterpolateEnum'}

    def __init__(self, *args):
        super(ScaleInterpolateEnum, self).__init__(*args)


class ScaleInterpolateParams(VegaLiteSchema):
    """ScaleInterpolateParams schema wrapper

    Mapping(required=[type])

    Attributes
    ----------

    type : enum('rgb', 'cubehelix', 'cubehelix-long')

    gamma : float

    """
    _schema = {'$ref': '#/definitions/ScaleInterpolateParams'}

    def __init__(self, type=Undefined, gamma=Undefined, **kwds):
        super(ScaleInterpolateParams, self).__init__(type=type, gamma=gamma, **kwds)


class ScaleResolveMap(VegaLiteSchema):
    """ScaleResolveMap schema wrapper

    Mapping(required=[])

    Attributes
    ----------

    angle : :class:`ResolveMode`

    color : :class:`ResolveMode`

    fill : :class:`ResolveMode`

    fillOpacity : :class:`ResolveMode`

    opacity : :class:`ResolveMode`

    radius : :class:`ResolveMode`

    shape : :class:`ResolveMode`

    size : :class:`ResolveMode`

    stroke : :class:`ResolveMode`

    strokeDash : :class:`ResolveMode`

    strokeOpacity : :class:`ResolveMode`

    strokeWidth : :class:`ResolveMode`

    theta : :class:`ResolveMode`

    x : :class:`ResolveMode`

    y : :class:`ResolveMode`

    """
    _schema = {'$ref': '#/definitions/ScaleResolveMap'}

    def __init__(self, angle=Undefined, color=Undefined, fill=Undefined, fillOpacity=Undefined,
                 opacity=Undefined, radius=Undefined, shape=Undefined, size=Undefined, stroke=Undefined,
                 strokeDash=Undefined, strokeOpacity=Undefined, strokeWidth=Undefined, theta=Undefined,
                 x=Undefined, y=Undefined, **kwds):
        super(ScaleResolveMap, self).__init__(angle=angle, color=color, fill=fill,
                                              fillOpacity=fillOpacity, opacity=opacity, radius=radius,
                                              shape=shape, size=size, stroke=stroke,
                                              strokeDash=strokeDash, strokeOpacity=strokeOpacity,
                                              strokeWidth=strokeWidth, theta=theta, x=x, y=y, **kwds)


class ScaleType(VegaLiteSchema):
    """ScaleType schema wrapper

    enum('linear', 'log', 'pow', 'sqrt', 'symlog', 'identity', 'sequential', 'time', 'utc',
    'quantile', 'quantize', 'threshold', 'bin-ordinal', 'ordinal', 'point', 'band')
    """
    _schema = {'$ref': '#/definitions/ScaleType'}

    def __init__(self, *args):
        super(ScaleType, self).__init__(*args)


class SchemeParams(VegaLiteSchema):
    """SchemeParams schema wrapper

    Mapping(required=[name])

    Attributes
    ----------

    name : string
        A color scheme name for ordinal scales (e.g., ``"category10"`` or ``"blues"`` ).

        For the full list of supported schemes, please refer to the `Vega Scheme
        <https://vega.github.io/vega/docs/schemes/#reference>`__ reference.
    count : float
        The number of colors to use in the scheme. This can be useful for scale types such
        as ``"quantize"``, which use the length of the scale range to determine the number
        of discrete bins for the scale domain.
    extent : List(float)
        The extent of the color range to use. For example ``[0.2, 1]`` will rescale the
        color scheme such that color values in the range *[0, 0.2)* are excluded from the
        scheme.
    """
    _schema = {'$ref': '#/definitions/SchemeParams'}

    def __init__(self, name=Undefined, count=Undefined, extent=Undefined, **kwds):
        super(SchemeParams, self).__init__(name=name, count=count, extent=extent, **kwds)


class SecondaryFieldDef(Position2Def):
    """SecondaryFieldDef schema wrapper

    Mapping(required=[])
    A field definition of a secondary channel that shares a scale with another primary channel.
    For example, ``x2``, ``xError`` and ``xError2`` share the same scale with ``x``.

    Attributes
    ----------

    aggregate : :class:`Aggregate`
        Aggregation function for the field (e.g., ``"mean"``, ``"sum"``, ``"median"``,
        ``"min"``, ``"max"``, ``"count"`` ).

        **Default value:** ``undefined`` (None)

        **See also:** `aggregate <https://vega.github.io/vega-lite/docs/aggregate.html>`__
        documentation.
    band : float
        For rect-based marks ( ``rect``, ``bar``, and ``image`` ), mark size relative to
        bandwidth of `band scales
        <https://vega.github.io/vega-lite/docs/scale.html#band>`__, bins or time units. If
        set to ``1``, the mark size is set to the bandwidth, the bin interval, or the time
        unit interval. If set to ``0.5``, the mark size is half of the bandwidth or the time
        unit interval.

        For other marks, relative position on a band of a stacked, binned, time unit or band
        scale. If set to ``0``, the marks will be positioned at the beginning of the band.
        If set to ``0.5``, the marks will be positioned in the middle of the band.
    bin : None
        A flag for binning a ``quantitative`` field, `an object defining binning parameters
        <https://vega.github.io/vega-lite/docs/bin.html#params>`__, or indicating that the
        data for ``x`` or ``y`` channel are binned before they are imported into Vega-Lite (
        ``"binned"`` ).


        If ``true``, default `binning parameters
        <https://vega.github.io/vega-lite/docs/bin.html>`__ will be applied.

        If ``"binned"``, this indicates that the data for the ``x`` (or ``y`` ) channel are
        already binned. You can map the bin-start field to ``x`` (or ``y`` ) and the bin-end
        field to ``x2`` (or ``y2`` ). The scale and axis will be formatted similar to
        binning in Vega-Lite.  To adjust the axis ticks based on the bin step, you can also
        set the axis's `tickMinStep
        <https://vega.github.io/vega-lite/docs/axis.html#ticks>`__ property.

        **Default value:** ``false``

        **See also:** `bin <https://vega.github.io/vega-lite/docs/bin.html>`__
        documentation.
    field : :class:`Field`
        **Required.** A string defining the name of the field from which to pull a data
        value or an object defining iterated values from the `repeat
        <https://vega.github.io/vega-lite/docs/repeat.html>`__ operator.

        **See also:** `field <https://vega.github.io/vega-lite/docs/field.html>`__
        documentation.

        **Notes:** 1)  Dots ( ``.`` ) and brackets ( ``[`` and ``]`` ) can be used to access
        nested objects (e.g., ``"field": "foo.bar"`` and ``"field": "foo['bar']"`` ). If
        field names contain dots or brackets but are not nested, you can use ``\\`` to
        escape dots and brackets (e.g., ``"a\\.b"`` and ``"a\\[0\\]"`` ). See more details
        about escaping in the `field documentation
        <https://vega.github.io/vega-lite/docs/field.html>`__. 2) ``field`` is not required
        if ``aggregate`` is ``count``.
    timeUnit : anyOf(:class:`TimeUnit`, :class:`TimeUnitParams`)
        Time unit (e.g., ``year``, ``yearmonth``, ``month``, ``hours`` ) for a temporal
        field. or `a temporal field that gets casted as ordinal
        <https://vega.github.io/vega-lite/docs/type.html#cast>`__.

        **Default value:** ``undefined`` (None)

        **See also:** `timeUnit <https://vega.github.io/vega-lite/docs/timeunit.html>`__
        documentation.
    title : anyOf(:class:`Text`, None)
        A title for the field. If ``null``, the title will be removed.

        **Default value:**  derived from the field's name and transformation function (
        ``aggregate``, ``bin`` and ``timeUnit`` ). If the field has an aggregate function,
        the function is displayed as part of the title (e.g., ``"Sum of Profit"`` ). If the
        field is binned or has a time unit applied, the applied function is shown in
        parentheses (e.g., ``"Profit (binned)"``, ``"Transaction Date (year-month)"`` ).
        Otherwise, the title is simply the field name.

        **Notes** :

        1) You can customize the default field title format by providing the `fieldTitle
        <https://vega.github.io/vega-lite/docs/config.html#top-level-config>`__ property in
        the `config <https://vega.github.io/vega-lite/docs/config.html>`__ or `fieldTitle
        function via the compile function's options
        <https://vega.github.io/vega-lite/docs/compile.html#field-title>`__.

        2) If both field definition's ``title`` and axis, header, or legend ``title`` are
        defined, axis/header/legend title will be used.
    """
    _schema = {'$ref': '#/definitions/SecondaryFieldDef'}

    def __init__(self, aggregate=Undefined, band=Undefined, bin=Undefined, field=Undefined,
                 timeUnit=Undefined, title=Undefined, **kwds):
        super(SecondaryFieldDef, self).__init__(aggregate=aggregate, band=band, bin=bin, field=field,
                                                timeUnit=timeUnit, title=title, **kwds)


class SelectionComposition(VegaLiteSchema):
    """SelectionComposition schema wrapper

    anyOf(:class:`SelectionNot`, :class:`SelectionAnd`, :class:`SelectionOr`, string)
    """
    _schema = {'$ref': '#/definitions/SelectionComposition'}

    def __init__(self, *args, **kwds):
        super(SelectionComposition, self).__init__(*args, **kwds)


class SelectionAnd(SelectionComposition):
    """SelectionAnd schema wrapper

    Mapping(required=[and])

    Attributes
    ----------

    and : List(:class:`SelectionComposition`)

    """
    _schema = {'$ref': '#/definitions/SelectionAnd'}

    def __init__(self, **kwds):
        super(SelectionAnd, self).__init__(**kwds)


class SelectionConfig(VegaLiteSchema):
    """SelectionConfig schema wrapper

    Mapping(required=[])

    Attributes
    ----------

    interval : :class:`IntervalSelectionConfig`
        The default definition for an `interval
        <https://vega.github.io/vega-lite/docs/selection.html#type>`__ selection. All
        properties and transformations for an interval selection definition (except ``type``
        ) may be specified here.

        For instance, setting ``interval`` to ``{"translate": false}`` disables the ability
        to move interval selections by default.
    multi : :class:`MultiSelectionConfig`
        The default definition for a `multi
        <https://vega.github.io/vega-lite/docs/selection.html#type>`__ selection. All
        properties and transformations for a multi selection definition (except ``type`` )
        may be specified here.

        For instance, setting ``multi`` to ``{"toggle": "event.altKey"}`` adds additional
        values to multi selections when clicking with the alt-key pressed by default.
    single : :class:`SingleSelectionConfig`
        The default definition for a `single
        <https://vega.github.io/vega-lite/docs/selection.html#type>`__ selection. All
        properties and transformations   for a single selection definition (except ``type``
        ) may be specified here.

        For instance, setting ``single`` to ``{"on": "dblclick"}`` populates single
        selections on double-click by default.
    """
    _schema = {'$ref': '#/definitions/SelectionConfig'}

    def __init__(self, interval=Undefined, multi=Undefined, single=Undefined, **kwds):
        super(SelectionConfig, self).__init__(interval=interval, multi=multi, single=single, **kwds)


class SelectionDef(VegaLiteSchema):
    """SelectionDef schema wrapper

    anyOf(:class:`SingleSelection`, :class:`MultiSelection`, :class:`IntervalSelection`)
    """
    _schema = {'$ref': '#/definitions/SelectionDef'}

    def __init__(self, *args, **kwds):
        super(SelectionDef, self).__init__(*args, **kwds)


class IntervalSelection(SelectionDef):
    """IntervalSelection schema wrapper

    Mapping(required=[type])

    Attributes
    ----------

    type : string
        Determines the default event processing and data query for the selection. Vega-Lite
        currently supports three selection types:


        * ``"single"`` -- to select a single discrete data value on ``click``. - ``"multi"``
          -- to select multiple discrete data value; the first value is selected on
          ``click`` and additional values toggled on shift- ``click``. - ``"interval"`` --
          to select a continuous range of data values on ``drag``.
    bind : string
        Establishes a two-way binding between the interval selection and the scales used
        within the same view. This allows a user to interactively pan and zoom the view.

        **See also:** `bind <https://vega.github.io/vega-lite/docs/bind.html>`__
        documentation.
    clear : anyOf(:class:`Stream`, string, boolean)
        Clears the selection, emptying it of all values. Can be a `Event Stream
        <https://vega.github.io/vega/docs/event-streams/>`__ or ``false`` to disable.

        **Default value:** ``dblclick``.

        **See also:** `clear <https://vega.github.io/vega-lite/docs/clear.html>`__
        documentation.
    empty : enum('all', 'none')
        By default, ``all`` data values are considered to lie within an empty selection.
        When set to ``none``, empty selections contain no data values.
    encodings : List(:class:`SingleDefUnitChannel`)
        An array of encoding channels. The corresponding data field values must match for a
        data tuple to fall within the selection.

        **See also:** `encodings <https://vega.github.io/vega-lite/docs/project.html>`__
        documentation.
    fields : List(:class:`FieldName`)
        An array of field names whose values must match for a data tuple to fall within the
        selection.

        **See also:** `fields <https://vega.github.io/vega-lite/docs/project.html>`__
        documentation.
    init : :class:`SelectionInitIntervalMapping`
        Initialize the selection with a mapping between `projected channels or field names
        <https://vega.github.io/vega-lite/docs/project.html>`__ and arrays of initial
        values.

        **See also:** `init <https://vega.github.io/vega-lite/docs/init.html>`__
        documentation.
    mark : :class:`BrushConfig`
        An interval selection also adds a rectangle mark to depict the extents of the
        interval. The ``mark`` property can be used to customize the appearance of the mark.

        **See also:** `mark <https://vega.github.io/vega-lite/docs/selection-mark.html>`__
        documentation.
    on : anyOf(:class:`Stream`, string)
        A `Vega event stream <https://vega.github.io/vega/docs/event-streams/>`__ (object or
        selector) that triggers the selection. For interval selections, the event stream
        must specify a `start and end
        <https://vega.github.io/vega/docs/event-streams/#between-filters>`__.
    resolve : :class:`SelectionResolution`
        With layered and multi-view displays, a strategy that determines how selections'
        data queries are resolved when applied in a filter transform, conditional encoding
        rule, or scale domain.

        **See also:** `resolve
        <https://vega.github.io/vega-lite/docs/selection-resolve.html>`__ documentation.
    translate : anyOf(string, boolean)
        When truthy, allows a user to interactively move an interval selection
        back-and-forth. Can be ``true``, ``false`` (to disable panning), or a `Vega event
        stream definition <https://vega.github.io/vega/docs/event-streams/>`__ which must
        include a start and end event to trigger continuous panning.

        **Default value:** ``true``, which corresponds to ``[mousedown, window:mouseup] >
        window:mousemove!`` which corresponds to clicks and dragging within an interval
        selection to reposition it.

        **See also:** `translate <https://vega.github.io/vega-lite/docs/translate.html>`__
        documentation.
    zoom : anyOf(string, boolean)
        When truthy, allows a user to interactively resize an interval selection. Can be
        ``true``, ``false`` (to disable zooming), or a `Vega event stream definition
        <https://vega.github.io/vega/docs/event-streams/>`__. Currently, only ``wheel``
        events are supported.

        **Default value:** ``true``, which corresponds to ``wheel!``.

        **See also:** `zoom <https://vega.github.io/vega-lite/docs/zoom.html>`__
        documentation.
    """
    _schema = {'$ref': '#/definitions/IntervalSelection'}

    def __init__(self, type=Undefined, bind=Undefined, clear=Undefined, empty=Undefined,
                 encodings=Undefined, fields=Undefined, init=Undefined, mark=Undefined, on=Undefined,
                 resolve=Undefined, translate=Undefined, zoom=Undefined, **kwds):
        super(IntervalSelection, self).__init__(type=type, bind=bind, clear=clear, empty=empty,
                                                encodings=encodings, fields=fields, init=init,
                                                mark=mark, on=on, resolve=resolve, translate=translate,
                                                zoom=zoom, **kwds)


class MultiSelection(SelectionDef):
    """MultiSelection schema wrapper

    Mapping(required=[type])

    Attributes
    ----------

    type : string
        Determines the default event processing and data query for the selection. Vega-Lite
        currently supports three selection types:


        * ``"single"`` -- to select a single discrete data value on ``click``. - ``"multi"``
          -- to select multiple discrete data value; the first value is selected on
          ``click`` and additional values toggled on shift- ``click``. - ``"interval"`` --
          to select a continuous range of data values on ``drag``.
    bind : :class:`LegendBinding`
        When set, a selection is populated by interacting with the corresponding legend.
        Direct manipulation interaction is disabled by default; to re-enable it, set the
        selection's `on
        <https://vega.github.io/vega-lite/docs/selection.html#common-selection-properties>`__
        property.

        Legend bindings are restricted to selections that only specify a single field or
        encoding.
    clear : anyOf(:class:`Stream`, string, boolean)
        Clears the selection, emptying it of all values. Can be a `Event Stream
        <https://vega.github.io/vega/docs/event-streams/>`__ or ``false`` to disable.

        **Default value:** ``dblclick``.

        **See also:** `clear <https://vega.github.io/vega-lite/docs/clear.html>`__
        documentation.
    empty : enum('all', 'none')
        By default, ``all`` data values are considered to lie within an empty selection.
        When set to ``none``, empty selections contain no data values.
    encodings : List(:class:`SingleDefUnitChannel`)
        An array of encoding channels. The corresponding data field values must match for a
        data tuple to fall within the selection.

        **See also:** `encodings <https://vega.github.io/vega-lite/docs/project.html>`__
        documentation.
    fields : List(:class:`FieldName`)
        An array of field names whose values must match for a data tuple to fall within the
        selection.

        **See also:** `fields <https://vega.github.io/vega-lite/docs/project.html>`__
        documentation.
    init : List(:class:`SelectionInitMapping`)
        Initialize the selection with a mapping between `projected channels or field names
        <https://vega.github.io/vega-lite/docs/project.html>`__ and an initial value (or
        array of values).

        **See also:** `init <https://vega.github.io/vega-lite/docs/init.html>`__
        documentation.
    nearest : boolean
        When true, an invisible voronoi diagram is computed to accelerate discrete
        selection. The data value *nearest* the mouse cursor is added to the selection.

        **See also:** `nearest <https://vega.github.io/vega-lite/docs/nearest.html>`__
        documentation.
    on : anyOf(:class:`Stream`, string)
        A `Vega event stream <https://vega.github.io/vega/docs/event-streams/>`__ (object or
        selector) that triggers the selection. For interval selections, the event stream
        must specify a `start and end
        <https://vega.github.io/vega/docs/event-streams/#between-filters>`__.
    resolve : :class:`SelectionResolution`
        With layered and multi-view displays, a strategy that determines how selections'
        data queries are resolved when applied in a filter transform, conditional encoding
        rule, or scale domain.

        **See also:** `resolve
        <https://vega.github.io/vega-lite/docs/selection-resolve.html>`__ documentation.
    toggle : anyOf(string, boolean)
        Controls whether data values should be toggled or only ever inserted into multi
        selections. Can be ``true``, ``false`` (for insertion only), or a `Vega expression
        <https://vega.github.io/vega/docs/expressions/>`__.

        **Default value:** ``true``, which corresponds to ``event.shiftKey`` (i.e., data
        values are toggled when a user interacts with the shift-key pressed).

        Setting the value to the Vega expression ``"true"`` will toggle data values without
        the user pressing the shift-key.

        **See also:** `toggle <https://vega.github.io/vega-lite/docs/toggle.html>`__
        documentation.
    """
    _schema = {'$ref': '#/definitions/MultiSelection'}

    def __init__(self, type=Undefined, bind=Undefined, clear=Undefined, empty=Undefined,
                 encodings=Undefined, fields=Undefined, init=Undefined, nearest=Undefined, on=Undefined,
                 resolve=Undefined, toggle=Undefined, **kwds):
        super(MultiSelection, self).__init__(type=type, bind=bind, clear=clear, empty=empty,
                                             encodings=encodings, fields=fields, init=init,
                                             nearest=nearest, on=on, resolve=resolve, toggle=toggle,
                                             **kwds)


class SelectionExtent(BinExtent):
    """SelectionExtent schema wrapper

    anyOf(Mapping(required=[selection]), Mapping(required=[selection]))
    """
    _schema = {'$ref': '#/definitions/SelectionExtent'}

    def __init__(self, *args, **kwds):
        super(SelectionExtent, self).__init__(*args, **kwds)


class SelectionInit(VegaLiteSchema):
    """SelectionInit schema wrapper

    anyOf(:class:`PrimitiveValue`, :class:`DateTime`)
    """
    _schema = {'$ref': '#/definitions/SelectionInit'}

    def __init__(self, *args, **kwds):
        super(SelectionInit, self).__init__(*args, **kwds)


class DateTime(SelectionInit):
    """DateTime schema wrapper

    Mapping(required=[])
    Object for defining datetime in Vega-Lite Filter. If both month and quarter are provided,
    month has higher precedence. ``day`` cannot be combined with other date. We accept string
    for month and day names.

    Attributes
    ----------

    date : float
        Integer value representing the date (day of the month) from 1-31.
    day : anyOf(:class:`Day`, string)
        Value representing the day of a week. This can be one of: (1) integer value -- ``1``
        represents Monday; (2) case-insensitive day name (e.g., ``"Monday"`` ); (3)
        case-insensitive, 3-character short day name (e.g., ``"Mon"`` ).

        **Warning:** A DateTime definition object with ``day`` ** should not be combined
        with ``year``, ``quarter``, ``month``, or ``date``.
    hours : float
        Integer value representing the hour of a day from 0-23.
    milliseconds : float
        Integer value representing the millisecond segment of time.
    minutes : float
        Integer value representing the minute segment of time from 0-59.
    month : anyOf(:class:`Month`, string)
        One of: (1) integer value representing the month from ``1`` - ``12``. ``1``
        represents January; (2) case-insensitive month name (e.g., ``"January"`` ); (3)
        case-insensitive, 3-character short month name (e.g., ``"Jan"`` ).
    quarter : float
        Integer value representing the quarter of the year (from 1-4).
    seconds : float
        Integer value representing the second segment (0-59) of a time value
    utc : boolean
        A boolean flag indicating if date time is in utc time. If false, the date time is in
        local time
    year : float
        Integer value representing the year.
    """
    _schema = {'$ref': '#/definitions/DateTime'}

    def __init__(self, date=Undefined, day=Undefined, hours=Undefined, milliseconds=Undefined,
                 minutes=Undefined, month=Undefined, quarter=Undefined, seconds=Undefined,
                 utc=Undefined, year=Undefined, **kwds):
        super(DateTime, self).__init__(date=date, day=day, hours=hours, milliseconds=milliseconds,
                                       minutes=minutes, month=month, quarter=quarter, seconds=seconds,
                                       utc=utc, year=year, **kwds)


class PrimitiveValue(SelectionInit):
    """PrimitiveValue schema wrapper

    anyOf(float, string, boolean, None)
    """
    _schema = {'$ref': '#/definitions/PrimitiveValue'}

    def __init__(self, *args):
        super(PrimitiveValue, self).__init__(*args)


class SelectionInitInterval(VegaLiteSchema):
    """SelectionInitInterval schema wrapper

    anyOf(:class:`Vector2boolean`, :class:`Vector2number`, :class:`Vector2string`,
    :class:`Vector2DateTime`)
    """
    _schema = {'$ref': '#/definitions/SelectionInitInterval'}

    def __init__(self, *args, **kwds):
        super(SelectionInitInterval, self).__init__(*args, **kwds)


class SelectionInitIntervalMapping(VegaLiteSchema):
    """SelectionInitIntervalMapping schema wrapper

    Mapping(required=[])
    """
    _schema = {'$ref': '#/definitions/SelectionInitIntervalMapping'}

    def __init__(self, **kwds):
        super(SelectionInitIntervalMapping, self).__init__(**kwds)


class SelectionInitMapping(VegaLiteSchema):
    """SelectionInitMapping schema wrapper

    Mapping(required=[])
    """
    _schema = {'$ref': '#/definitions/SelectionInitMapping'}

    def __init__(self, **kwds):
        super(SelectionInitMapping, self).__init__(**kwds)


class SelectionNot(SelectionComposition):
    """SelectionNot schema wrapper

    Mapping(required=[not])

    Attributes
    ----------

    not : :class:`SelectionComposition`

    """
    _schema = {'$ref': '#/definitions/SelectionNot'}

    def __init__(self, **kwds):
        super(SelectionNot, self).__init__(**kwds)


class SelectionOr(SelectionComposition):
    """SelectionOr schema wrapper

    Mapping(required=[or])

    Attributes
    ----------

    or : List(:class:`SelectionComposition`)

    """
    _schema = {'$ref': '#/definitions/SelectionOr'}

    def __init__(self, **kwds):
        super(SelectionOr, self).__init__(**kwds)


class SelectionPredicate(Predicate):
    """SelectionPredicate schema wrapper

    Mapping(required=[selection])

    Attributes
    ----------

    selection : :class:`SelectionComposition`
        Filter using a selection name or a logical composition of selection names.
    """
    _schema = {'$ref': '#/definitions/SelectionPredicate'}

    def __init__(self, selection=Undefined, **kwds):
        super(SelectionPredicate, self).__init__(selection=selection, **kwds)


class SelectionResolution(VegaLiteSchema):
    """SelectionResolution schema wrapper

    enum('global', 'union', 'intersect')
    """
    _schema = {'$ref': '#/definitions/SelectionResolution'}

    def __init__(self, *args):
        super(SelectionResolution, self).__init__(*args)


class SequenceGenerator(Generator):
    """SequenceGenerator schema wrapper

    Mapping(required=[sequence])

    Attributes
    ----------

    sequence : :class:`SequenceParams`
        Generate a sequence of numbers.
    name : string
        Provide a placeholder name and bind data at runtime.
    """
    _schema = {'$ref': '#/definitions/SequenceGenerator'}

    def __init__(self, sequence=Undefined, name=Undefined, **kwds):
        super(SequenceGenerator, self).__init__(sequence=sequence, name=name, **kwds)


class SequenceParams(VegaLiteSchema):
    """SequenceParams schema wrapper

    Mapping(required=[start, stop])

    Attributes
    ----------

    start : float
        The starting value of the sequence (inclusive).
    stop : float
        The ending value of the sequence (exclusive).
    step : float
        The step value between sequence entries.

        **Default value:** ``1``
    as : :class:`FieldName`
        The name of the generated sequence field.

        **Default value:** ``"data"``
    """
    _schema = {'$ref': '#/definitions/SequenceParams'}

    def __init__(self, start=Undefined, stop=Undefined, step=Undefined, **kwds):
        super(SequenceParams, self).__init__(start=start, stop=stop, step=step, **kwds)


class SequentialMultiHue(ColorScheme):
    """SequentialMultiHue schema wrapper

    enum('turbo', 'viridis', 'inferno', 'magma', 'plasma', 'cividis', 'bluegreen',
    'bluegreen-3', 'bluegreen-4', 'bluegreen-5', 'bluegreen-6', 'bluegreen-7', 'bluegreen-8',
    'bluegreen-9', 'bluepurple', 'bluepurple-3', 'bluepurple-4', 'bluepurple-5', 'bluepurple-6',
    'bluepurple-7', 'bluepurple-8', 'bluepurple-9', 'goldgreen', 'goldgreen-3', 'goldgreen-4',
    'goldgreen-5', 'goldgreen-6', 'goldgreen-7', 'goldgreen-8', 'goldgreen-9', 'goldorange',
    'goldorange-3', 'goldorange-4', 'goldorange-5', 'goldorange-6', 'goldorange-7',
    'goldorange-8', 'goldorange-9', 'goldred', 'goldred-3', 'goldred-4', 'goldred-5',
    'goldred-6', 'goldred-7', 'goldred-8', 'goldred-9', 'greenblue', 'greenblue-3',
    'greenblue-4', 'greenblue-5', 'greenblue-6', 'greenblue-7', 'greenblue-8', 'greenblue-9',
    'orangered', 'orangered-3', 'orangered-4', 'orangered-5', 'orangered-6', 'orangered-7',
    'orangered-8', 'orangered-9', 'purplebluegreen', 'purplebluegreen-3', 'purplebluegreen-4',
    'purplebluegreen-5', 'purplebluegreen-6', 'purplebluegreen-7', 'purplebluegreen-8',
    'purplebluegreen-9', 'purpleblue', 'purpleblue-3', 'purpleblue-4', 'purpleblue-5',
    'purpleblue-6', 'purpleblue-7', 'purpleblue-8', 'purpleblue-9', 'purplered', 'purplered-3',
    'purplered-4', 'purplered-5', 'purplered-6', 'purplered-7', 'purplered-8', 'purplered-9',
    'redpurple', 'redpurple-3', 'redpurple-4', 'redpurple-5', 'redpurple-6', 'redpurple-7',
    'redpurple-8', 'redpurple-9', 'yellowgreenblue', 'yellowgreenblue-3', 'yellowgreenblue-4',
    'yellowgreenblue-5', 'yellowgreenblue-6', 'yellowgreenblue-7', 'yellowgreenblue-8',
    'yellowgreenblue-9', 'yellowgreen', 'yellowgreen-3', 'yellowgreen-4', 'yellowgreen-5',
    'yellowgreen-6', 'yellowgreen-7', 'yellowgreen-8', 'yellowgreen-9', 'yelloworangebrown',
    'yelloworangebrown-3', 'yelloworangebrown-4', 'yelloworangebrown-5', 'yelloworangebrown-6',
    'yelloworangebrown-7', 'yelloworangebrown-8', 'yelloworangebrown-9', 'yelloworangered',
    'yelloworangered-3', 'yelloworangered-4', 'yelloworangered-5', 'yelloworangered-6',
    'yelloworangered-7', 'yelloworangered-8', 'yelloworangered-9', 'darkblue', 'darkblue-3',
    'darkblue-4', 'darkblue-5', 'darkblue-6', 'darkblue-7', 'darkblue-8', 'darkblue-9',
    'darkgold', 'darkgold-3', 'darkgold-4', 'darkgold-5', 'darkgold-6', 'darkgold-7',
    'darkgold-8', 'darkgold-9', 'darkgreen', 'darkgreen-3', 'darkgreen-4', 'darkgreen-5',
    'darkgreen-6', 'darkgreen-7', 'darkgreen-8', 'darkgreen-9', 'darkmulti', 'darkmulti-3',
    'darkmulti-4', 'darkmulti-5', 'darkmulti-6', 'darkmulti-7', 'darkmulti-8', 'darkmulti-9',
    'darkred', 'darkred-3', 'darkred-4', 'darkred-5', 'darkred-6', 'darkred-7', 'darkred-8',
    'darkred-9', 'lightgreyred', 'lightgreyred-3', 'lightgreyred-4', 'lightgreyred-5',
    'lightgreyred-6', 'lightgreyred-7', 'lightgreyred-8', 'lightgreyred-9', 'lightgreyteal',
    'lightgreyteal-3', 'lightgreyteal-4', 'lightgreyteal-5', 'lightgreyteal-6',
    'lightgreyteal-7', 'lightgreyteal-8', 'lightgreyteal-9', 'lightmulti', 'lightmulti-3',
    'lightmulti-4', 'lightmulti-5', 'lightmulti-6', 'lightmulti-7', 'lightmulti-8',
    'lightmulti-9', 'lightorange', 'lightorange-3', 'lightorange-4', 'lightorange-5',
    'lightorange-6', 'lightorange-7', 'lightorange-8', 'lightorange-9', 'lighttealblue',
    'lighttealblue-3', 'lighttealblue-4', 'lighttealblue-5', 'lighttealblue-6',
    'lighttealblue-7', 'lighttealblue-8', 'lighttealblue-9')
    """
    _schema = {'$ref': '#/definitions/SequentialMultiHue'}

    def __init__(self, *args):
        super(SequentialMultiHue, self).__init__(*args)


class SequentialSingleHue(ColorScheme):
    """SequentialSingleHue schema wrapper

    enum('blues', 'tealblues', 'teals', 'greens', 'browns', 'greys', 'purples', 'warmgreys',
    'reds', 'oranges')
    """
    _schema = {'$ref': '#/definitions/SequentialSingleHue'}

    def __init__(self, *args):
        super(SequentialSingleHue, self).__init__(*args)


class ShapeDef(VegaLiteSchema):
    """ShapeDef schema wrapper

    anyOf(:class:`FieldOrDatumDefWithConditionMarkPropFieldDefTypeForShapestringnull`,
    :class:`FieldOrDatumDefWithConditionDatumDefstringnull`,
    :class:`ValueDefWithConditionMarkPropFieldOrDatumDefTypeForShapestringnull`)
    """
    _schema = {'$ref': '#/definitions/ShapeDef'}

    def __init__(self, *args, **kwds):
        super(ShapeDef, self).__init__(*args, **kwds)


class FieldOrDatumDefWithConditionDatumDefstringnull(MarkPropDefstringnullTypeForShape, ShapeDef):
    """FieldOrDatumDefWithConditionDatumDefstringnull schema wrapper

    Mapping(required=[])
    A FieldDef with Condition :raw-html:`<ValueDef>` {    condition: {value: ...},    field:
    ...,    ... }

    Attributes
    ----------

    band : float
        For rect-based marks ( ``rect``, ``bar``, and ``image`` ), mark size relative to
        bandwidth of `band scales
        <https://vega.github.io/vega-lite/docs/scale.html#band>`__, bins or time units. If
        set to ``1``, the mark size is set to the bandwidth, the bin interval, or the time
        unit interval. If set to ``0.5``, the mark size is half of the bandwidth or the time
        unit interval.

        For other marks, relative position on a band of a stacked, binned, time unit or band
        scale. If set to ``0``, the marks will be positioned at the beginning of the band.
        If set to ``0.5``, the marks will be positioned in the middle of the band.
    condition : anyOf(:class:`ConditionalValueDefstringnullExprRef`,
    List(:class:`ConditionalValueDefstringnullExprRef`))
        One or more value definition(s) with `a selection or a test predicate
        <https://vega.github.io/vega-lite/docs/condition.html>`__.

        **Note:** A field definition's ``condition`` property can only contain `conditional
        value definitions <https://vega.github.io/vega-lite/docs/condition.html#value>`__
        since Vega-Lite only allows at most one encoded field per encoding channel.
    datum : anyOf(:class:`PrimitiveValue`, :class:`DateTime`, :class:`ExprRef`,
    :class:`RepeatRef`)
        A constant value in data domain.
    type : :class:`Type`
        The type of measurement ( ``"quantitative"``, ``"temporal"``, ``"ordinal"``, or
        ``"nominal"`` ) for the encoded field or constant value ( ``datum`` ). It can also
        be a ``"geojson"`` type for encoding `'geoshape'
        <https://vega.github.io/vega-lite/docs/geoshape.html>`__.

        Vega-Lite automatically infers data types in many cases as discussed below. However,
        type is required for a field if: (1) the field is not nominal and the field encoding
        has no specified ``aggregate`` (except ``argmin`` and ``argmax`` ), ``bin``, scale
        type, custom ``sort`` order, nor ``timeUnit`` or (2) if you wish to use an ordinal
        scale for a field with ``bin`` or ``timeUnit``.

        **Default value:**

        1) For a data ``field``, ``"nominal"`` is the default data type unless the field
        encoding has ``aggregate``, ``channel``, ``bin``, scale type, ``sort``, or
        ``timeUnit`` that satisfies the following criteria: - ``"quantitative"`` is the
        default type if (1) the encoded field contains ``bin`` or ``aggregate`` except
        ``"argmin"`` and ``"argmax"``, (2) the encoding channel is ``latitude`` or
        ``longitude`` channel or (3) if the specified scale type is `a quantitative scale
        <https://vega.github.io/vega-lite/docs/scale.html#type>`__. - ``"temporal"`` is the
        default type if (1) the encoded field contains ``timeUnit`` or (2) the specified
        scale type is a time or utc scale - ``ordinal""`` is the default type if (1) the
        encoded field contains a `custom sort order
        <https://vega.github.io/vega-lite/docs/sort.html#specifying-custom-sort-order>`__,
        (2) the specified scale type is an ordinal/point/band scale, or (3) the encoding
        channel is ``order``.

        2) For a constant value in data domain ( ``datum`` ): - ``"quantitative"`` if the
        datum is a number - ``"nominal"`` if the datum is a string - ``"temporal"`` if the
        datum is `a date time object
        <https://vega.github.io/vega-lite/docs/datetime.html>`__

        **Note:** - Data ``type`` describes the semantics of the data rather than the
        primitive data types (number, string, etc.). The same primitive data type can have
        different types of measurement. For example, numeric data can represent
        quantitative, ordinal, or nominal data. - Data values for a temporal field can be
        either a date-time string (e.g., ``"2015-03-07 12:32:17"``, ``"17:01"``,
        ``"2015-03-16"``. ``"2015"`` ) or a timestamp number (e.g., ``1552199579097`` ). -
        When using with `bin <https://vega.github.io/vega-lite/docs/bin.html>`__, the
        ``type`` property can be either ``"quantitative"`` (for using a linear bin scale) or
        `"ordinal" (for using an ordinal bin scale)
        <https://vega.github.io/vega-lite/docs/type.html#cast-bin>`__. - When using with
        `timeUnit <https://vega.github.io/vega-lite/docs/timeunit.html>`__, the ``type``
        property can be either ``"temporal"`` (default, for using a temporal scale) or
        `"ordinal" (for using an ordinal scale)
        <https://vega.github.io/vega-lite/docs/type.html#cast-bin>`__. - When using with
        `aggregate <https://vega.github.io/vega-lite/docs/aggregate.html>`__, the ``type``
        property refers to the post-aggregation data type. For example, we can calculate
        count ``distinct`` of a categorical field ``"cat"`` using ``{"aggregate":
        "distinct", "field": "cat"}``. The ``"type"`` of the aggregate output is
        ``"quantitative"``. - Secondary channels (e.g., ``x2``, ``y2``, ``xError``,
        ``yError`` ) do not have ``type`` as they must have exactly the same type as their
        primary channels (e.g., ``x``, ``y`` ).

        **See also:** `type <https://vega.github.io/vega-lite/docs/type.html>`__
        documentation.
    """
    _schema = {'$ref': '#/definitions/FieldOrDatumDefWithCondition<DatumDef,(string|null)>'}

    def __init__(self, band=Undefined, condition=Undefined, datum=Undefined, type=Undefined, **kwds):
        super(FieldOrDatumDefWithConditionDatumDefstringnull, self).__init__(band=band,
                                                                             condition=condition,
                                                                             datum=datum, type=type,
                                                                             **kwds)


class FieldOrDatumDefWithConditionMarkPropFieldDefTypeForShapestringnull(MarkPropDefstringnullTypeForShape, ShapeDef):
    """FieldOrDatumDefWithConditionMarkPropFieldDefTypeForShapestringnull schema wrapper

    Mapping(required=[])
    A FieldDef with Condition :raw-html:`<ValueDef>` {    condition: {value: ...},    field:
    ...,    ... }

    Attributes
    ----------

    aggregate : :class:`Aggregate`
        Aggregation function for the field (e.g., ``"mean"``, ``"sum"``, ``"median"``,
        ``"min"``, ``"max"``, ``"count"`` ).

        **Default value:** ``undefined`` (None)

        **See also:** `aggregate <https://vega.github.io/vega-lite/docs/aggregate.html>`__
        documentation.
    band : float
        For rect-based marks ( ``rect``, ``bar``, and ``image`` ), mark size relative to
        bandwidth of `band scales
        <https://vega.github.io/vega-lite/docs/scale.html#band>`__, bins or time units. If
        set to ``1``, the mark size is set to the bandwidth, the bin interval, or the time
        unit interval. If set to ``0.5``, the mark size is half of the bandwidth or the time
        unit interval.

        For other marks, relative position on a band of a stacked, binned, time unit or band
        scale. If set to ``0``, the marks will be positioned at the beginning of the band.
        If set to ``0.5``, the marks will be positioned in the middle of the band.
    bin : anyOf(boolean, :class:`BinParams`, None)
        A flag for binning a ``quantitative`` field, `an object defining binning parameters
        <https://vega.github.io/vega-lite/docs/bin.html#params>`__, or indicating that the
        data for ``x`` or ``y`` channel are binned before they are imported into Vega-Lite (
        ``"binned"`` ).


        If ``true``, default `binning parameters
        <https://vega.github.io/vega-lite/docs/bin.html>`__ will be applied.

        If ``"binned"``, this indicates that the data for the ``x`` (or ``y`` ) channel are
        already binned. You can map the bin-start field to ``x`` (or ``y`` ) and the bin-end
        field to ``x2`` (or ``y2`` ). The scale and axis will be formatted similar to
        binning in Vega-Lite.  To adjust the axis ticks based on the bin step, you can also
        set the axis's `tickMinStep
        <https://vega.github.io/vega-lite/docs/axis.html#ticks>`__ property.

        **Default value:** ``false``

        **See also:** `bin <https://vega.github.io/vega-lite/docs/bin.html>`__
        documentation.
    condition : anyOf(:class:`ConditionalValueDefstringnullExprRef`,
    List(:class:`ConditionalValueDefstringnullExprRef`))
        One or more value definition(s) with `a selection or a test predicate
        <https://vega.github.io/vega-lite/docs/condition.html>`__.

        **Note:** A field definition's ``condition`` property can only contain `conditional
        value definitions <https://vega.github.io/vega-lite/docs/condition.html#value>`__
        since Vega-Lite only allows at most one encoded field per encoding channel.
    field : :class:`Field`
        **Required.** A string defining the name of the field from which to pull a data
        value or an object defining iterated values from the `repeat
        <https://vega.github.io/vega-lite/docs/repeat.html>`__ operator.

        **See also:** `field <https://vega.github.io/vega-lite/docs/field.html>`__
        documentation.

        **Notes:** 1)  Dots ( ``.`` ) and brackets ( ``[`` and ``]`` ) can be used to access
        nested objects (e.g., ``"field": "foo.bar"`` and ``"field": "foo['bar']"`` ). If
        field names contain dots or brackets but are not nested, you can use ``\\`` to
        escape dots and brackets (e.g., ``"a\\.b"`` and ``"a\\[0\\]"`` ). See more details
        about escaping in the `field documentation
        <https://vega.github.io/vega-lite/docs/field.html>`__. 2) ``field`` is not required
        if ``aggregate`` is ``count``.
    legend : anyOf(:class:`Legend`, None)
        An object defining properties of the legend. If ``null``, the legend for the
        encoding channel will be removed.

        **Default value:** If undefined, default `legend properties
        <https://vega.github.io/vega-lite/docs/legend.html>`__ are applied.

        **See also:** `legend <https://vega.github.io/vega-lite/docs/legend.html>`__
        documentation.
    scale : anyOf(:class:`Scale`, None)
        An object defining properties of the channel's scale, which is the function that
        transforms values in the data domain (numbers, dates, strings, etc) to visual values
        (pixels, colors, sizes) of the encoding channels.

        If ``null``, the scale will be `disabled and the data value will be directly encoded
        <https://vega.github.io/vega-lite/docs/scale.html#disable>`__.

        **Default value:** If undefined, default `scale properties
        <https://vega.github.io/vega-lite/docs/scale.html>`__ are applied.

        **See also:** `scale <https://vega.github.io/vega-lite/docs/scale.html>`__
        documentation.
    sort : :class:`Sort`
        Sort order for the encoded field.

        For continuous fields (quantitative or temporal), ``sort`` can be either
        ``"ascending"`` or ``"descending"``.

        For discrete fields, ``sort`` can be one of the following: - ``"ascending"`` or
        ``"descending"`` -- for sorting by the values' natural order in JavaScript. - `A
        string indicating an encoding channel name to sort by
        <https://vega.github.io/vega-lite/docs/sort.html#sort-by-encoding>`__ (e.g., ``"x"``
        or ``"y"`` ) with an optional minus prefix for descending sort (e.g., ``"-x"`` to
        sort by x-field, descending). This channel string is short-form of `a
        sort-by-encoding definition
        <https://vega.github.io/vega-lite/docs/sort.html#sort-by-encoding>`__. For example,
        ``"sort": "-x"`` is equivalent to ``"sort": {"encoding": "x", "order":
        "descending"}``. - `A sort field definition
        <https://vega.github.io/vega-lite/docs/sort.html#sort-field>`__ for sorting by
        another field. - `An array specifying the field values in preferred order
        <https://vega.github.io/vega-lite/docs/sort.html#sort-array>`__. In this case, the
        sort order will obey the values in the array, followed by any unspecified values in
        their original order. For discrete time field, values in the sort array can be
        `date-time definition objects <types#datetime>`__. In addition, for time units
        ``"month"`` and ``"day"``, the values can be the month or day names (case
        insensitive) or their 3-letter initials (e.g., ``"Mon"``, ``"Tue"`` ). - ``null``
        indicating no sort.

        **Default value:** ``"ascending"``

        **Note:** ``null`` and sorting by another channel is not supported for ``row`` and
        ``column``.

        **See also:** `sort <https://vega.github.io/vega-lite/docs/sort.html>`__
        documentation.
    timeUnit : anyOf(:class:`TimeUnit`, :class:`TimeUnitParams`)
        Time unit (e.g., ``year``, ``yearmonth``, ``month``, ``hours`` ) for a temporal
        field. or `a temporal field that gets casted as ordinal
        <https://vega.github.io/vega-lite/docs/type.html#cast>`__.

        **Default value:** ``undefined`` (None)

        **See also:** `timeUnit <https://vega.github.io/vega-lite/docs/timeunit.html>`__
        documentation.
    title : anyOf(:class:`Text`, None)
        A title for the field. If ``null``, the title will be removed.

        **Default value:**  derived from the field's name and transformation function (
        ``aggregate``, ``bin`` and ``timeUnit`` ). If the field has an aggregate function,
        the function is displayed as part of the title (e.g., ``"Sum of Profit"`` ). If the
        field is binned or has a time unit applied, the applied function is shown in
        parentheses (e.g., ``"Profit (binned)"``, ``"Transaction Date (year-month)"`` ).
        Otherwise, the title is simply the field name.

        **Notes** :

        1) You can customize the default field title format by providing the `fieldTitle
        <https://vega.github.io/vega-lite/docs/config.html#top-level-config>`__ property in
        the `config <https://vega.github.io/vega-lite/docs/config.html>`__ or `fieldTitle
        function via the compile function's options
        <https://vega.github.io/vega-lite/docs/compile.html#field-title>`__.

        2) If both field definition's ``title`` and axis, header, or legend ``title`` are
        defined, axis/header/legend title will be used.
    type : :class:`TypeForShape`
        The type of measurement ( ``"quantitative"``, ``"temporal"``, ``"ordinal"``, or
        ``"nominal"`` ) for the encoded field or constant value ( ``datum`` ). It can also
        be a ``"geojson"`` type for encoding `'geoshape'
        <https://vega.github.io/vega-lite/docs/geoshape.html>`__.

        Vega-Lite automatically infers data types in many cases as discussed below. However,
        type is required for a field if: (1) the field is not nominal and the field encoding
        has no specified ``aggregate`` (except ``argmin`` and ``argmax`` ), ``bin``, scale
        type, custom ``sort`` order, nor ``timeUnit`` or (2) if you wish to use an ordinal
        scale for a field with ``bin`` or ``timeUnit``.

        **Default value:**

        1) For a data ``field``, ``"nominal"`` is the default data type unless the field
        encoding has ``aggregate``, ``channel``, ``bin``, scale type, ``sort``, or
        ``timeUnit`` that satisfies the following criteria: - ``"quantitative"`` is the
        default type if (1) the encoded field contains ``bin`` or ``aggregate`` except
        ``"argmin"`` and ``"argmax"``, (2) the encoding channel is ``latitude`` or
        ``longitude`` channel or (3) if the specified scale type is `a quantitative scale
        <https://vega.github.io/vega-lite/docs/scale.html#type>`__. - ``"temporal"`` is the
        default type if (1) the encoded field contains ``timeUnit`` or (2) the specified
        scale type is a time or utc scale - ``ordinal""`` is the default type if (1) the
        encoded field contains a `custom sort order
        <https://vega.github.io/vega-lite/docs/sort.html#specifying-custom-sort-order>`__,
        (2) the specified scale type is an ordinal/point/band scale, or (3) the encoding
        channel is ``order``.

        2) For a constant value in data domain ( ``datum`` ): - ``"quantitative"`` if the
        datum is a number - ``"nominal"`` if the datum is a string - ``"temporal"`` if the
        datum is `a date time object
        <https://vega.github.io/vega-lite/docs/datetime.html>`__

        **Note:** - Data ``type`` describes the semantics of the data rather than the
        primitive data types (number, string, etc.). The same primitive data type can have
        different types of measurement. For example, numeric data can represent
        quantitative, ordinal, or nominal data. - Data values for a temporal field can be
        either a date-time string (e.g., ``"2015-03-07 12:32:17"``, ``"17:01"``,
        ``"2015-03-16"``. ``"2015"`` ) or a timestamp number (e.g., ``1552199579097`` ). -
        When using with `bin <https://vega.github.io/vega-lite/docs/bin.html>`__, the
        ``type`` property can be either ``"quantitative"`` (for using a linear bin scale) or
        `"ordinal" (for using an ordinal bin scale)
        <https://vega.github.io/vega-lite/docs/type.html#cast-bin>`__. - When using with
        `timeUnit <https://vega.github.io/vega-lite/docs/timeunit.html>`__, the ``type``
        property can be either ``"temporal"`` (default, for using a temporal scale) or
        `"ordinal" (for using an ordinal scale)
        <https://vega.github.io/vega-lite/docs/type.html#cast-bin>`__. - When using with
        `aggregate <https://vega.github.io/vega-lite/docs/aggregate.html>`__, the ``type``
        property refers to the post-aggregation data type. For example, we can calculate
        count ``distinct`` of a categorical field ``"cat"`` using ``{"aggregate":
        "distinct", "field": "cat"}``. The ``"type"`` of the aggregate output is
        ``"quantitative"``. - Secondary channels (e.g., ``x2``, ``y2``, ``xError``,
        ``yError`` ) do not have ``type`` as they must have exactly the same type as their
        primary channels (e.g., ``x``, ``y`` ).

        **See also:** `type <https://vega.github.io/vega-lite/docs/type.html>`__
        documentation.
    """
    _schema = {'$ref': '#/definitions/FieldOrDatumDefWithCondition<MarkPropFieldDef<TypeForShape>,(string|null)>'}

    def __init__(self, aggregate=Undefined, band=Undefined, bin=Undefined, condition=Undefined,
                 field=Undefined, legend=Undefined, scale=Undefined, sort=Undefined, timeUnit=Undefined,
                 title=Undefined, type=Undefined, **kwds):
        super(FieldOrDatumDefWithConditionMarkPropFieldDefTypeForShapestringnull, self).__init__(aggregate=aggregate,
                                                                                                 band=band,
                                                                                                 bin=bin,
                                                                                                 condition=condition,
                                                                                                 field=field,
                                                                                                 legend=legend,
                                                                                                 scale=scale,
                                                                                                 sort=sort,
                                                                                                 timeUnit=timeUnit,
                                                                                                 title=title,
                                                                                                 type=type,
                                                                                                 **kwds)


class SharedEncoding(VegaLiteSchema):
    """SharedEncoding schema wrapper

    Mapping(required=[])

    Attributes
    ----------

    angle : Mapping(required=[])

    color : Mapping(required=[])

    description : Mapping(required=[])

    detail : anyOf(:class:`FieldDefWithoutScale`, List(:class:`FieldDefWithoutScale`))
        Additional levels of detail for grouping data in aggregate views and in line, trail,
        and area marks without mapping data to a specific visual channel.
    fill : Mapping(required=[])

    fillOpacity : Mapping(required=[])

    href : Mapping(required=[])

    key : Mapping(required=[])

    latitude : Mapping(required=[])

    latitude2 : Mapping(required=[])

    longitude : Mapping(required=[])

    longitude2 : Mapping(required=[])

    opacity : Mapping(required=[])

    order : anyOf(:class:`OrderFieldDef`, List(:class:`OrderFieldDef`), :class:`OrderValueDef`)
        Order of the marks. - For stacked marks, this ``order`` channel encodes `stack order
        <https://vega.github.io/vega-lite/docs/stack.html#order>`__. - For line and trail
        marks, this ``order`` channel encodes order of data points in the lines. This can be
        useful for creating `a connected scatterplot
        <https://vega.github.io/vega-lite/examples/connected_scatterplot.html>`__. Setting
        ``order`` to ``{"value": null}`` makes the line marks use the original order in the
        data sources. - Otherwise, this ``order`` channel encodes layer order of the marks.

        **Note** : In aggregate plots, ``order`` field should be ``aggregate`` d to avoid
        creating additional aggregation grouping.
    radius : Mapping(required=[])

    radius2 : Mapping(required=[])

    shape : Mapping(required=[])

    size : Mapping(required=[])

    stroke : Mapping(required=[])

    strokeDash : Mapping(required=[])

    strokeOpacity : Mapping(required=[])

    strokeWidth : Mapping(required=[])

    text : Mapping(required=[])

    theta : Mapping(required=[])

    theta2 : Mapping(required=[])

    tooltip : anyOf(:class:`StringFieldDefWithCondition`, :class:`StringValueDefWithCondition`,
    List(:class:`StringFieldDef`), None)
        The tooltip text to show upon mouse hover. Specifying ``tooltip`` encoding overrides
        `the tooltip property in the mark definition
        <https://vega.github.io/vega-lite/docs/mark.html#mark-def>`__.

        See the `tooltip <https://vega.github.io/vega-lite/docs/tooltip.html>`__
        documentation for a detailed discussion about tooltip in Vega-Lite.
    url : Mapping(required=[])

    x : Mapping(required=[])

    x2 : Mapping(required=[])

    xError : Mapping(required=[])

    xError2 : Mapping(required=[])

    y : Mapping(required=[])

    y2 : Mapping(required=[])

    yError : Mapping(required=[])

    yError2 : Mapping(required=[])

    """
    _schema = {'$ref': '#/definitions/SharedEncoding'}

    def __init__(self, angle=Undefined, color=Undefined, description=Undefined, detail=Undefined,
                 fill=Undefined, fillOpacity=Undefined, href=Undefined, key=Undefined,
                 latitude=Undefined, latitude2=Undefined, longitude=Undefined, longitude2=Undefined,
                 opacity=Undefined, order=Undefined, radius=Undefined, radius2=Undefined,
                 shape=Undefined, size=Undefined, stroke=Undefined, strokeDash=Undefined,
                 strokeOpacity=Undefined, strokeWidth=Undefined, text=Undefined, theta=Undefined,
                 theta2=Undefined, tooltip=Undefined, url=Undefined, x=Undefined, x2=Undefined,
                 xError=Undefined, xError2=Undefined, y=Undefined, y2=Undefined, yError=Undefined,
                 yError2=Undefined, **kwds):
        super(SharedEncoding, self).__init__(angle=angle, color=color, description=description,
                                             detail=detail, fill=fill, fillOpacity=fillOpacity,
                                             href=href, key=key, latitude=latitude, latitude2=latitude2,
                                             longitude=longitude, longitude2=longitude2,
                                             opacity=opacity, order=order, radius=radius,
                                             radius2=radius2, shape=shape, size=size, stroke=stroke,
                                             strokeDash=strokeDash, strokeOpacity=strokeOpacity,
                                             strokeWidth=strokeWidth, text=text, theta=theta,
                                             theta2=theta2, tooltip=tooltip, url=url, x=x, x2=x2,
                                             xError=xError, xError2=xError2, y=y, y2=y2, yError=yError,
                                             yError2=yError2, **kwds)


class SingleDefUnitChannel(VegaLiteSchema):
    """SingleDefUnitChannel schema wrapper

    enum('x', 'y', 'x2', 'y2', 'longitude', 'latitude', 'longitude2', 'latitude2', 'theta',
    'theta2', 'radius', 'radius2', 'color', 'fill', 'stroke', 'opacity', 'fillOpacity',
    'strokeOpacity', 'strokeWidth', 'strokeDash', 'size', 'angle', 'shape', 'key', 'text',
    'href', 'url', 'description')
    """
    _schema = {'$ref': '#/definitions/SingleDefUnitChannel'}

    def __init__(self, *args):
        super(SingleDefUnitChannel, self).__init__(*args)


class SingleSelection(SelectionDef):
    """SingleSelection schema wrapper

    Mapping(required=[type])

    Attributes
    ----------

    type : string
        Determines the default event processing and data query for the selection. Vega-Lite
        currently supports three selection types:


        * ``"single"`` -- to select a single discrete data value on ``click``. - ``"multi"``
          -- to select multiple discrete data value; the first value is selected on
          ``click`` and additional values toggled on shift- ``click``. - ``"interval"`` --
          to select a continuous range of data values on ``drag``.
    bind : anyOf(:class:`Binding`, Mapping(required=[]), :class:`LegendBinding`)
        When set, a selection is populated by input elements (also known as dynamic query
        widgets) or by interacting with the corresponding legend. Direct manipulation
        interaction is disabled by default; to re-enable it, set the selection's `on
        <https://vega.github.io/vega-lite/docs/selection.html#common-selection-properties>`__
        property.

        Legend bindings are restricted to selections that only specify a single field or
        encoding.

        Query widget binding takes the form of Vega's `input element binding definition
        <https://vega.github.io/vega/docs/signals/#bind>`__ or can be a mapping between
        projected field/encodings and binding definitions.

        **See also:** `bind <https://vega.github.io/vega-lite/docs/bind.html>`__
        documentation.
    clear : anyOf(:class:`Stream`, string, boolean)
        Clears the selection, emptying it of all values. Can be a `Event Stream
        <https://vega.github.io/vega/docs/event-streams/>`__ or ``false`` to disable.

        **Default value:** ``dblclick``.

        **See also:** `clear <https://vega.github.io/vega-lite/docs/clear.html>`__
        documentation.
    empty : enum('all', 'none')
        By default, ``all`` data values are considered to lie within an empty selection.
        When set to ``none``, empty selections contain no data values.
    encodings : List(:class:`SingleDefUnitChannel`)
        An array of encoding channels. The corresponding data field values must match for a
        data tuple to fall within the selection.

        **See also:** `encodings <https://vega.github.io/vega-lite/docs/project.html>`__
        documentation.
    fields : List(:class:`FieldName`)
        An array of field names whose values must match for a data tuple to fall within the
        selection.

        **See also:** `fields <https://vega.github.io/vega-lite/docs/project.html>`__
        documentation.
    init : :class:`SelectionInitMapping`
        Initialize the selection with a mapping between `projected channels or field names
        <https://vega.github.io/vega-lite/docs/project.html>`__ and initial values.

        **See also:** `init <https://vega.github.io/vega-lite/docs/init.html>`__
        documentation.
    nearest : boolean
        When true, an invisible voronoi diagram is computed to accelerate discrete
        selection. The data value *nearest* the mouse cursor is added to the selection.

        **See also:** `nearest <https://vega.github.io/vega-lite/docs/nearest.html>`__
        documentation.
    on : anyOf(:class:`Stream`, string)
        A `Vega event stream <https://vega.github.io/vega/docs/event-streams/>`__ (object or
        selector) that triggers the selection. For interval selections, the event stream
        must specify a `start and end
        <https://vega.github.io/vega/docs/event-streams/#between-filters>`__.
    resolve : :class:`SelectionResolution`
        With layered and multi-view displays, a strategy that determines how selections'
        data queries are resolved when applied in a filter transform, conditional encoding
        rule, or scale domain.

        **See also:** `resolve
        <https://vega.github.io/vega-lite/docs/selection-resolve.html>`__ documentation.
    """
    _schema = {'$ref': '#/definitions/SingleSelection'}

    def __init__(self, type=Undefined, bind=Undefined, clear=Undefined, empty=Undefined,
                 encodings=Undefined, fields=Undefined, init=Undefined, nearest=Undefined, on=Undefined,
                 resolve=Undefined, **kwds):
        super(SingleSelection, self).__init__(type=type, bind=bind, clear=clear, empty=empty,
                                              encodings=encodings, fields=fields, init=init,
                                              nearest=nearest, on=on, resolve=resolve, **kwds)


class SingleSelectionConfig(VegaLiteSchema):
    """SingleSelectionConfig schema wrapper

    Mapping(required=[])

    Attributes
    ----------

    bind : anyOf(:class:`Binding`, Mapping(required=[]), :class:`LegendBinding`)
        When set, a selection is populated by input elements (also known as dynamic query
        widgets) or by interacting with the corresponding legend. Direct manipulation
        interaction is disabled by default; to re-enable it, set the selection's `on
        <https://vega.github.io/vega-lite/docs/selection.html#common-selection-properties>`__
        property.

        Legend bindings are restricted to selections that only specify a single field or
        encoding.

        Query widget binding takes the form of Vega's `input element binding definition
        <https://vega.github.io/vega/docs/signals/#bind>`__ or can be a mapping between
        projected field/encodings and binding definitions.

        **See also:** `bind <https://vega.github.io/vega-lite/docs/bind.html>`__
        documentation.
    clear : anyOf(:class:`Stream`, string, boolean)
        Clears the selection, emptying it of all values. Can be a `Event Stream
        <https://vega.github.io/vega/docs/event-streams/>`__ or ``false`` to disable.

        **Default value:** ``dblclick``.

        **See also:** `clear <https://vega.github.io/vega-lite/docs/clear.html>`__
        documentation.
    empty : enum('all', 'none')
        By default, ``all`` data values are considered to lie within an empty selection.
        When set to ``none``, empty selections contain no data values.
    encodings : List(:class:`SingleDefUnitChannel`)
        An array of encoding channels. The corresponding data field values must match for a
        data tuple to fall within the selection.

        **See also:** `encodings <https://vega.github.io/vega-lite/docs/project.html>`__
        documentation.
    fields : List(:class:`FieldName`)
        An array of field names whose values must match for a data tuple to fall within the
        selection.

        **See also:** `fields <https://vega.github.io/vega-lite/docs/project.html>`__
        documentation.
    init : :class:`SelectionInitMapping`
        Initialize the selection with a mapping between `projected channels or field names
        <https://vega.github.io/vega-lite/docs/project.html>`__ and initial values.

        **See also:** `init <https://vega.github.io/vega-lite/docs/init.html>`__
        documentation.
    nearest : boolean
        When true, an invisible voronoi diagram is computed to accelerate discrete
        selection. The data value *nearest* the mouse cursor is added to the selection.

        **See also:** `nearest <https://vega.github.io/vega-lite/docs/nearest.html>`__
        documentation.
    on : anyOf(:class:`Stream`, string)
        A `Vega event stream <https://vega.github.io/vega/docs/event-streams/>`__ (object or
        selector) that triggers the selection. For interval selections, the event stream
        must specify a `start and end
        <https://vega.github.io/vega/docs/event-streams/#between-filters>`__.
    resolve : :class:`SelectionResolution`
        With layered and multi-view displays, a strategy that determines how selections'
        data queries are resolved when applied in a filter transform, conditional encoding
        rule, or scale domain.

        **See also:** `resolve
        <https://vega.github.io/vega-lite/docs/selection-resolve.html>`__ documentation.
    """
    _schema = {'$ref': '#/definitions/SingleSelectionConfig'}

    def __init__(self, bind=Undefined, clear=Undefined, empty=Undefined, encodings=Undefined,
                 fields=Undefined, init=Undefined, nearest=Undefined, on=Undefined, resolve=Undefined,
                 **kwds):
        super(SingleSelectionConfig, self).__init__(bind=bind, clear=clear, empty=empty,
                                                    encodings=encodings, fields=fields, init=init,
                                                    nearest=nearest, on=on, resolve=resolve, **kwds)


class Sort(VegaLiteSchema):
    """Sort schema wrapper

    anyOf(:class:`SortArray`, :class:`AllSortString`, :class:`EncodingSortField`,
    :class:`SortByEncoding`, None)
    """
    _schema = {'$ref': '#/definitions/Sort'}

    def __init__(self, *args, **kwds):
        super(Sort, self).__init__(*args, **kwds)


class AllSortString(Sort):
    """AllSortString schema wrapper

    anyOf(:class:`SortOrder`, :class:`SortByChannel`, :class:`SortByChannelDesc`)
    """
    _schema = {'$ref': '#/definitions/AllSortString'}

    def __init__(self, *args, **kwds):
        super(AllSortString, self).__init__(*args, **kwds)


class EncodingSortField(Sort):
    """EncodingSortField schema wrapper

    Mapping(required=[])
    A sort definition for sorting a discrete scale in an encoding field definition.

    Attributes
    ----------

    field : :class:`Field`
        The data `field <https://vega.github.io/vega-lite/docs/field.html>`__ to sort by.

        **Default value:** If unspecified, defaults to the field specified in the outer data
        reference.
    op : :class:`NonArgAggregateOp`
        An `aggregate operation
        <https://vega.github.io/vega-lite/docs/aggregate.html#ops>`__ to perform on the
        field prior to sorting (e.g., ``"count"``, ``"mean"`` and ``"median"`` ). An
        aggregation is required when there are multiple values of the sort field for each
        encoded data field. The input data objects will be aggregated, grouped by the
        encoded data field.

        For a full list of operations, please see the documentation for `aggregate
        <https://vega.github.io/vega-lite/docs/aggregate.html#ops>`__.

        **Default value:** ``"sum"`` for stacked plots. Otherwise, ``"min"``.
    order : anyOf(:class:`SortOrder`, None)
        The sort order. One of ``"ascending"`` (default), ``"descending"``, or ``null`` (no
        not sort).
    """
    _schema = {'$ref': '#/definitions/EncodingSortField'}

    def __init__(self, field=Undefined, op=Undefined, order=Undefined, **kwds):
        super(EncodingSortField, self).__init__(field=field, op=op, order=order, **kwds)


class SortArray(Sort):
    """SortArray schema wrapper

    anyOf(List(float), List(string), List(boolean), List(:class:`DateTime`))
    """
    _schema = {'$ref': '#/definitions/SortArray'}

    def __init__(self, *args, **kwds):
        super(SortArray, self).__init__(*args, **kwds)


class SortByChannel(AllSortString):
    """SortByChannel schema wrapper

    enum('x', 'y', 'color', 'fill', 'stroke', 'strokeWidth', 'size', 'shape', 'fillOpacity',
    'strokeOpacity', 'opacity', 'text')
    """
    _schema = {'$ref': '#/definitions/SortByChannel'}

    def __init__(self, *args):
        super(SortByChannel, self).__init__(*args)


class SortByChannelDesc(AllSortString):
    """SortByChannelDesc schema wrapper

    enum('-x', '-y', '-color', '-fill', '-stroke', '-strokeWidth', '-size', '-shape',
    '-fillOpacity', '-strokeOpacity', '-opacity', '-text')
    """
    _schema = {'$ref': '#/definitions/SortByChannelDesc'}

    def __init__(self, *args):
        super(SortByChannelDesc, self).__init__(*args)


class SortByEncoding(Sort):
    """SortByEncoding schema wrapper

    Mapping(required=[encoding])

    Attributes
    ----------

    encoding : :class:`SortByChannel`
        The `encoding channel
        <https://vega.github.io/vega-lite/docs/encoding.html#channels>`__ to sort by (e.g.,
        ``"x"``, ``"y"`` )
    order : anyOf(:class:`SortOrder`, None)
        The sort order. One of ``"ascending"`` (default), ``"descending"``, or ``null`` (no
        not sort).
    """
    _schema = {'$ref': '#/definitions/SortByEncoding'}

    def __init__(self, encoding=Undefined, order=Undefined, **kwds):
        super(SortByEncoding, self).__init__(encoding=encoding, order=order, **kwds)


class SortField(VegaLiteSchema):
    """SortField schema wrapper

    Mapping(required=[field])
    A sort definition for transform

    Attributes
    ----------

    field : :class:`FieldName`
        The name of the field to sort.
    order : anyOf(:class:`SortOrder`, None)
        Whether to sort the field in ascending or descending order. One of ``"ascending"``
        (default), ``"descending"``, or ``null`` (no not sort).
    """
    _schema = {'$ref': '#/definitions/SortField'}

    def __init__(self, field=Undefined, order=Undefined, **kwds):
        super(SortField, self).__init__(field=field, order=order, **kwds)


class SortOrder(AllSortString):
    """SortOrder schema wrapper

    enum('ascending', 'descending')
    """
    _schema = {'$ref': '#/definitions/SortOrder'}

    def __init__(self, *args):
        super(SortOrder, self).__init__(*args)


class Spec(VegaLiteSchema):
    """Spec schema wrapper

    anyOf(:class:`FacetedUnitSpec`, :class:`LayerSpec`, :class:`RepeatSpec`, :class:`FacetSpec`,
    :class:`ConcatSpecGenericSpec`, :class:`VConcatSpecGenericSpec`,
    :class:`HConcatSpecGenericSpec`)
    Any specification in Vega-Lite.
    """
    _schema = {'$ref': '#/definitions/Spec'}

    def __init__(self, *args, **kwds):
        super(Spec, self).__init__(*args, **kwds)


class ConcatSpecGenericSpec(Spec):
    """ConcatSpecGenericSpec schema wrapper

    Mapping(required=[concat])
    Base interface for a generalized concatenation specification.

    Attributes
    ----------

    concat : List(:class:`Spec`)
        A list of views to be concatenated.
    align : anyOf(:class:`LayoutAlign`, :class:`RowColLayoutAlign`)
        The alignment to apply to grid rows and columns. The supported string values are
        ``"all"``, ``"each"``, and ``"none"``.


        * For ``"none"``, a flow layout will be used, in which adjacent subviews are simply
          placed one after the other. - For ``"each"``, subviews will be aligned into a
          clean grid structure, but each row or column may be of variable size. - For
          ``"all"``, subviews will be aligned and each row or column will be sized
          identically based on the maximum observed size. String values for this property
          will be applied to both grid rows and columns.

        Alternatively, an object value of the form ``{"row": string, "column": string}`` can
        be used to supply different alignments for rows and columns.

        **Default value:** ``"all"``.
    bounds : enum('full', 'flush')
        The bounds calculation method to use for determining the extent of a sub-plot. One
        of ``full`` (the default) or ``flush``.


        * If set to ``full``, the entire calculated bounds (including axes, title, and
          legend) will be used. - If set to ``flush``, only the specified width and height
          values for the sub-view will be used. The ``flush`` setting can be useful when
          attempting to place sub-plots without axes or legends into a uniform grid
          structure.

        **Default value:** ``"full"``
    center : anyOf(boolean, :class:`RowColboolean`)
        Boolean flag indicating if subviews should be centered relative to their respective
        rows or columns.

        An object value of the form ``{"row": boolean, "column": boolean}`` can be used to
        supply different centering values for rows and columns.

        **Default value:** ``false``
    columns : float
        The number of columns to include in the view composition layout.

        **Default value** : ``undefined`` -- An infinite number of columns (a single row)
        will be assumed. This is equivalent to ``hconcat`` (for ``concat`` ) and to using
        the ``column`` channel (for ``facet`` and ``repeat`` ).

        **Note** :

        1) This property is only for: - the general (wrappable) ``concat`` operator (not
        ``hconcat`` / ``vconcat`` ) - the ``facet`` and ``repeat`` operator with one
        field/repetition definition (without row/column nesting)

        2) Setting the ``columns`` to ``1`` is equivalent to ``vconcat`` (for ``concat`` )
        and to using the ``row`` channel (for ``facet`` and ``repeat`` ).
    data : anyOf(:class:`Data`, None)
        An object describing the data source. Set to ``null`` to ignore the parent's data
        source. If no data is set, it is derived from the parent.
    description : string
        Description of this mark for commenting purpose.
    name : string
        Name of the visualization for later reference.
    resolve : :class:`Resolve`
        Scale, axis, and legend resolutions for view composition specifications.
    spacing : anyOf(float, :class:`RowColnumber`)
        The spacing in pixels between sub-views of the composition operator. An object of
        the form ``{"row": number, "column": number}`` can be used to set different spacing
        values for rows and columns.

        **Default value** : Depends on ``"spacing"`` property of `the view composition
        configuration <https://vega.github.io/vega-lite/docs/config.html#view-config>`__ (
        ``20`` by default)
    title : anyOf(:class:`Text`, :class:`TitleParams`)
        Title for the plot.
    transform : List(:class:`Transform`)
        An array of data transformations such as filter and new field calculation.
    """
    _schema = {'$ref': '#/definitions/ConcatSpec<GenericSpec>'}

    def __init__(self, concat=Undefined, align=Undefined, bounds=Undefined, center=Undefined,
                 columns=Undefined, data=Undefined, description=Undefined, name=Undefined,
                 resolve=Undefined, spacing=Undefined, title=Undefined, transform=Undefined, **kwds):
        super(ConcatSpecGenericSpec, self).__init__(concat=concat, align=align, bounds=bounds,
                                                    center=center, columns=columns, data=data,
                                                    description=description, name=name, resolve=resolve,
                                                    spacing=spacing, title=title, transform=transform,
                                                    **kwds)


class FacetSpec(Spec):
    """FacetSpec schema wrapper

    Mapping(required=[facet, spec])
    Base interface for a facet specification.

    Attributes
    ----------

    facet : anyOf(:class:`FacetFieldDefFieldName`, :class:`FacetMappingFieldName`)
        Definition for how to facet the data. One of: 1) `a field definition for faceting
        the plot by one field
        <https://vega.github.io/vega-lite/docs/facet.html#field-def>`__ 2) `An object that
        maps row and column channels to their field definitions
        <https://vega.github.io/vega-lite/docs/facet.html#mapping>`__
    spec : anyOf(:class:`LayerSpec`, :class:`FacetedUnitSpec`)
        A specification of the view that gets faceted.
    align : anyOf(:class:`LayoutAlign`, :class:`RowColLayoutAlign`)
        The alignment to apply to grid rows and columns. The supported string values are
        ``"all"``, ``"each"``, and ``"none"``.


        * For ``"none"``, a flow layout will be used, in which adjacent subviews are simply
          placed one after the other. - For ``"each"``, subviews will be aligned into a
          clean grid structure, but each row or column may be of variable size. - For
          ``"all"``, subviews will be aligned and each row or column will be sized
          identically based on the maximum observed size. String values for this property
          will be applied to both grid rows and columns.

        Alternatively, an object value of the form ``{"row": string, "column": string}`` can
        be used to supply different alignments for rows and columns.

        **Default value:** ``"all"``.
    bounds : enum('full', 'flush')
        The bounds calculation method to use for determining the extent of a sub-plot. One
        of ``full`` (the default) or ``flush``.


        * If set to ``full``, the entire calculated bounds (including axes, title, and
          legend) will be used. - If set to ``flush``, only the specified width and height
          values for the sub-view will be used. The ``flush`` setting can be useful when
          attempting to place sub-plots without axes or legends into a uniform grid
          structure.

        **Default value:** ``"full"``
    center : anyOf(boolean, :class:`RowColboolean`)
        Boolean flag indicating if subviews should be centered relative to their respective
        rows or columns.

        An object value of the form ``{"row": boolean, "column": boolean}`` can be used to
        supply different centering values for rows and columns.

        **Default value:** ``false``
    columns : float
        The number of columns to include in the view composition layout.

        **Default value** : ``undefined`` -- An infinite number of columns (a single row)
        will be assumed. This is equivalent to ``hconcat`` (for ``concat`` ) and to using
        the ``column`` channel (for ``facet`` and ``repeat`` ).

        **Note** :

        1) This property is only for: - the general (wrappable) ``concat`` operator (not
        ``hconcat`` / ``vconcat`` ) - the ``facet`` and ``repeat`` operator with one
        field/repetition definition (without row/column nesting)

        2) Setting the ``columns`` to ``1`` is equivalent to ``vconcat`` (for ``concat`` )
        and to using the ``row`` channel (for ``facet`` and ``repeat`` ).
    data : anyOf(:class:`Data`, None)
        An object describing the data source. Set to ``null`` to ignore the parent's data
        source. If no data is set, it is derived from the parent.
    description : string
        Description of this mark for commenting purpose.
    name : string
        Name of the visualization for later reference.
    resolve : :class:`Resolve`
        Scale, axis, and legend resolutions for view composition specifications.
    spacing : anyOf(float, :class:`RowColnumber`)
        The spacing in pixels between sub-views of the composition operator. An object of
        the form ``{"row": number, "column": number}`` can be used to set different spacing
        values for rows and columns.

        **Default value** : Depends on ``"spacing"`` property of `the view composition
        configuration <https://vega.github.io/vega-lite/docs/config.html#view-config>`__ (
        ``20`` by default)
    title : anyOf(:class:`Text`, :class:`TitleParams`)
        Title for the plot.
    transform : List(:class:`Transform`)
        An array of data transformations such as filter and new field calculation.
    """
    _schema = {'$ref': '#/definitions/FacetSpec'}

    def __init__(self, facet=Undefined, spec=Undefined, align=Undefined, bounds=Undefined,
                 center=Undefined, columns=Undefined, data=Undefined, description=Undefined,
                 name=Undefined, resolve=Undefined, spacing=Undefined, title=Undefined,
                 transform=Undefined, **kwds):
        super(FacetSpec, self).__init__(facet=facet, spec=spec, align=align, bounds=bounds,
                                        center=center, columns=columns, data=data,
                                        description=description, name=name, resolve=resolve,
                                        spacing=spacing, title=title, transform=transform, **kwds)


class FacetedUnitSpec(NormalizedSpec, Spec):
    """FacetedUnitSpec schema wrapper

    Mapping(required=[mark])
    Unit spec that can have a composite mark and row or column channels (shorthand for a facet
    spec).

    Attributes
    ----------

    mark : :class:`AnyMark`
        A string describing the mark type (one of ``"bar"``, ``"circle"``, ``"square"``,
        ``"tick"``, ``"line"``, ``"area"``, ``"point"``, ``"rule"``, ``"geoshape"``, and
        ``"text"`` ) or a `mark definition object
        <https://vega.github.io/vega-lite/docs/mark.html#mark-def>`__.
    align : anyOf(:class:`LayoutAlign`, :class:`RowColLayoutAlign`)
        The alignment to apply to grid rows and columns. The supported string values are
        ``"all"``, ``"each"``, and ``"none"``.


        * For ``"none"``, a flow layout will be used, in which adjacent subviews are simply
          placed one after the other. - For ``"each"``, subviews will be aligned into a
          clean grid structure, but each row or column may be of variable size. - For
          ``"all"``, subviews will be aligned and each row or column will be sized
          identically based on the maximum observed size. String values for this property
          will be applied to both grid rows and columns.

        Alternatively, an object value of the form ``{"row": string, "column": string}`` can
        be used to supply different alignments for rows and columns.

        **Default value:** ``"all"``.
    bounds : enum('full', 'flush')
        The bounds calculation method to use for determining the extent of a sub-plot. One
        of ``full`` (the default) or ``flush``.


        * If set to ``full``, the entire calculated bounds (including axes, title, and
          legend) will be used. - If set to ``flush``, only the specified width and height
          values for the sub-view will be used. The ``flush`` setting can be useful when
          attempting to place sub-plots without axes or legends into a uniform grid
          structure.

        **Default value:** ``"full"``
    center : anyOf(boolean, :class:`RowColboolean`)
        Boolean flag indicating if subviews should be centered relative to their respective
        rows or columns.

        An object value of the form ``{"row": boolean, "column": boolean}`` can be used to
        supply different centering values for rows and columns.

        **Default value:** ``false``
    data : anyOf(:class:`Data`, None)
        An object describing the data source. Set to ``null`` to ignore the parent's data
        source. If no data is set, it is derived from the parent.
    description : string
        Description of this mark for commenting purpose.
    encoding : :class:`FacetedEncoding`
        A key-value mapping between encoding channels and definition of fields.
    height : anyOf(float, string, :class:`Step`)
        The height of a visualization.


        * For a plot with a continuous y-field, height should be a number. - For a plot with
          either a discrete y-field or no y-field, height can be either a number indicating
          a fixed height or an object in the form of ``{step: number}`` defining the height
          per discrete step. (No y-field is equivalent to having one discrete step.) - To
          enable responsive sizing on height, it should be set to ``"container"``.

        **Default value:** Based on ``config.view.continuousHeight`` for a plot with a
        continuous y-field and ``config.view.discreteHeight`` otherwise.

        **Note:** For plots with `row and column channels
        <https://vega.github.io/vega-lite/docs/encoding.html#facet>`__, this represents the
        height of a single view and the ``"container"`` option cannot be used.

        **See also:** `height <https://vega.github.io/vega-lite/docs/size.html>`__
        documentation.
    name : string
        Name of the visualization for later reference.
    projection : :class:`Projection`
        An object defining properties of geographic projection, which will be applied to
        ``shape`` path for ``"geoshape"`` marks and to ``latitude`` and ``"longitude"``
        channels for other marks.
    resolve : :class:`Resolve`
        Scale, axis, and legend resolutions for view composition specifications.
    selection : Mapping(required=[])
        A key-value mapping between selection names and definitions.
    spacing : anyOf(float, :class:`RowColnumber`)
        The spacing in pixels between sub-views of the composition operator. An object of
        the form ``{"row": number, "column": number}`` can be used to set different spacing
        values for rows and columns.

        **Default value** : Depends on ``"spacing"`` property of `the view composition
        configuration <https://vega.github.io/vega-lite/docs/config.html#view-config>`__ (
        ``20`` by default)
    title : anyOf(:class:`Text`, :class:`TitleParams`)
        Title for the plot.
    transform : List(:class:`Transform`)
        An array of data transformations such as filter and new field calculation.
    view : :class:`ViewBackground`
        An object defining the view background's fill and stroke.

        **Default value:** none (transparent)
    width : anyOf(float, string, :class:`Step`)
        The width of a visualization.


        * For a plot with a continuous x-field, width should be a number. - For a plot with
          either a discrete x-field or no x-field, width can be either a number indicating a
          fixed width or an object in the form of ``{step: number}`` defining the width per
          discrete step. (No x-field is equivalent to having one discrete step.) - To enable
          responsive sizing on width, it should be set to ``"container"``.

        **Default value:** Based on ``config.view.continuousWidth`` for a plot with a
        continuous x-field and ``config.view.discreteWidth`` otherwise.

        **Note:** For plots with `row and column channels
        <https://vega.github.io/vega-lite/docs/encoding.html#facet>`__, this represents the
        width of a single view and the ``"container"`` option cannot be used.

        **See also:** `width <https://vega.github.io/vega-lite/docs/size.html>`__
        documentation.
    """
    _schema = {'$ref': '#/definitions/FacetedUnitSpec'}

    def __init__(self, mark=Undefined, align=Undefined, bounds=Undefined, center=Undefined,
                 data=Undefined, description=Undefined, encoding=Undefined, height=Undefined,
                 name=Undefined, projection=Undefined, resolve=Undefined, selection=Undefined,
                 spacing=Undefined, title=Undefined, transform=Undefined, view=Undefined,
                 width=Undefined, **kwds):
        super(FacetedUnitSpec, self).__init__(mark=mark, align=align, bounds=bounds, center=center,
                                              data=data, description=description, encoding=encoding,
                                              height=height, name=name, projection=projection,
                                              resolve=resolve, selection=selection, spacing=spacing,
                                              title=title, transform=transform, view=view, width=width,
                                              **kwds)


class HConcatSpecGenericSpec(Spec):
    """HConcatSpecGenericSpec schema wrapper

    Mapping(required=[hconcat])
    Base interface for a horizontal concatenation specification.

    Attributes
    ----------

    hconcat : List(:class:`Spec`)
        A list of views to be concatenated and put into a row.
    bounds : enum('full', 'flush')
        The bounds calculation method to use for determining the extent of a sub-plot. One
        of ``full`` (the default) or ``flush``.


        * If set to ``full``, the entire calculated bounds (including axes, title, and
          legend) will be used. - If set to ``flush``, only the specified width and height
          values for the sub-view will be used. The ``flush`` setting can be useful when
          attempting to place sub-plots without axes or legends into a uniform grid
          structure.

        **Default value:** ``"full"``
    center : boolean
        Boolean flag indicating if subviews should be centered relative to their respective
        rows or columns.

        **Default value:** ``false``
    data : anyOf(:class:`Data`, None)
        An object describing the data source. Set to ``null`` to ignore the parent's data
        source. If no data is set, it is derived from the parent.
    description : string
        Description of this mark for commenting purpose.
    name : string
        Name of the visualization for later reference.
    resolve : :class:`Resolve`
        Scale, axis, and legend resolutions for view composition specifications.
    spacing : float
        The spacing in pixels between sub-views of the concat operator.

        **Default value** : ``10``
    title : anyOf(:class:`Text`, :class:`TitleParams`)
        Title for the plot.
    transform : List(:class:`Transform`)
        An array of data transformations such as filter and new field calculation.
    """
    _schema = {'$ref': '#/definitions/HConcatSpec<GenericSpec>'}

    def __init__(self, hconcat=Undefined, bounds=Undefined, center=Undefined, data=Undefined,
                 description=Undefined, name=Undefined, resolve=Undefined, spacing=Undefined,
                 title=Undefined, transform=Undefined, **kwds):
        super(HConcatSpecGenericSpec, self).__init__(hconcat=hconcat, bounds=bounds, center=center,
                                                     data=data, description=description, name=name,
                                                     resolve=resolve, spacing=spacing, title=title,
                                                     transform=transform, **kwds)


class LayerSpec(NormalizedSpec, Spec):
    """LayerSpec schema wrapper

    Mapping(required=[layer])
    A full layered plot specification, which may contains ``encoding`` and ``projection``
    properties that will be applied to underlying unit (single-view) specifications.

    Attributes
    ----------

    layer : List(anyOf(:class:`LayerSpec`, :class:`UnitSpec`))
        Layer or single view specifications to be layered.

        **Note** : Specifications inside ``layer`` cannot use ``row`` and ``column``
        channels as layering facet specifications is not allowed. Instead, use the `facet
        operator <https://vega.github.io/vega-lite/docs/facet.html>`__ and place a layer
        inside a facet.
    data : anyOf(:class:`Data`, None)
        An object describing the data source. Set to ``null`` to ignore the parent's data
        source. If no data is set, it is derived from the parent.
    description : string
        Description of this mark for commenting purpose.
    encoding : :class:`SharedEncoding`
        A shared key-value mapping between encoding channels and definition of fields in the
        underlying layers.
    height : anyOf(float, string, :class:`Step`)
        The height of a visualization.


        * For a plot with a continuous y-field, height should be a number. - For a plot with
          either a discrete y-field or no y-field, height can be either a number indicating
          a fixed height or an object in the form of ``{step: number}`` defining the height
          per discrete step. (No y-field is equivalent to having one discrete step.) - To
          enable responsive sizing on height, it should be set to ``"container"``.

        **Default value:** Based on ``config.view.continuousHeight`` for a plot with a
        continuous y-field and ``config.view.discreteHeight`` otherwise.

        **Note:** For plots with `row and column channels
        <https://vega.github.io/vega-lite/docs/encoding.html#facet>`__, this represents the
        height of a single view and the ``"container"`` option cannot be used.

        **See also:** `height <https://vega.github.io/vega-lite/docs/size.html>`__
        documentation.
    name : string
        Name of the visualization for later reference.
    projection : :class:`Projection`
        An object defining properties of the geographic projection shared by underlying
        layers.
    resolve : :class:`Resolve`
        Scale, axis, and legend resolutions for view composition specifications.
    title : anyOf(:class:`Text`, :class:`TitleParams`)
        Title for the plot.
    transform : List(:class:`Transform`)
        An array of data transformations such as filter and new field calculation.
    view : :class:`ViewBackground`
        An object defining the view background's fill and stroke.

        **Default value:** none (transparent)
    width : anyOf(float, string, :class:`Step`)
        The width of a visualization.


        * For a plot with a continuous x-field, width should be a number. - For a plot with
          either a discrete x-field or no x-field, width can be either a number indicating a
          fixed width or an object in the form of ``{step: number}`` defining the width per
          discrete step. (No x-field is equivalent to having one discrete step.) - To enable
          responsive sizing on width, it should be set to ``"container"``.

        **Default value:** Based on ``config.view.continuousWidth`` for a plot with a
        continuous x-field and ``config.view.discreteWidth`` otherwise.

        **Note:** For plots with `row and column channels
        <https://vega.github.io/vega-lite/docs/encoding.html#facet>`__, this represents the
        width of a single view and the ``"container"`` option cannot be used.

        **See also:** `width <https://vega.github.io/vega-lite/docs/size.html>`__
        documentation.
    """
    _schema = {'$ref': '#/definitions/LayerSpec'}

    def __init__(self, layer=Undefined, data=Undefined, description=Undefined, encoding=Undefined,
                 height=Undefined, name=Undefined, projection=Undefined, resolve=Undefined,
                 title=Undefined, transform=Undefined, view=Undefined, width=Undefined, **kwds):
        super(LayerSpec, self).__init__(layer=layer, data=data, description=description,
                                        encoding=encoding, height=height, name=name,
                                        projection=projection, resolve=resolve, title=title,
                                        transform=transform, view=view, width=width, **kwds)


class RepeatSpec(NormalizedSpec, Spec):
    """RepeatSpec schema wrapper

    anyOf(:class:`NonLayerRepeatSpec`, :class:`LayerRepeatSpec`)
    """
    _schema = {'$ref': '#/definitions/RepeatSpec'}

    def __init__(self, *args, **kwds):
        super(RepeatSpec, self).__init__(*args, **kwds)


class LayerRepeatSpec(RepeatSpec):
    """LayerRepeatSpec schema wrapper

    Mapping(required=[repeat, spec])

    Attributes
    ----------

    repeat : :class:`LayerRepeatMapping`
        Definition for fields to be repeated. One of: 1) An array of fields to be repeated.
        If ``"repeat"`` is an array, the field can be referred to as ``{"repeat":
        "repeat"}``. The repeated views are laid out in a wrapped row. You can set the
        number of columns to control the wrapping. 2) An object that maps ``"row"`` and/or
        ``"column"`` to the listed fields to be repeated along the particular orientations.
        The objects ``{"repeat": "row"}`` and ``{"repeat": "column"}`` can be used to refer
        to the repeated field respectively.
    spec : anyOf(:class:`LayerSpec`, :class:`UnitSpec`)
        A specification of the view that gets repeated.
    align : anyOf(:class:`LayoutAlign`, :class:`RowColLayoutAlign`)
        The alignment to apply to grid rows and columns. The supported string values are
        ``"all"``, ``"each"``, and ``"none"``.


        * For ``"none"``, a flow layout will be used, in which adjacent subviews are simply
          placed one after the other. - For ``"each"``, subviews will be aligned into a
          clean grid structure, but each row or column may be of variable size. - For
          ``"all"``, subviews will be aligned and each row or column will be sized
          identically based on the maximum observed size. String values for this property
          will be applied to both grid rows and columns.

        Alternatively, an object value of the form ``{"row": string, "column": string}`` can
        be used to supply different alignments for rows and columns.

        **Default value:** ``"all"``.
    bounds : enum('full', 'flush')
        The bounds calculation method to use for determining the extent of a sub-plot. One
        of ``full`` (the default) or ``flush``.


        * If set to ``full``, the entire calculated bounds (including axes, title, and
          legend) will be used. - If set to ``flush``, only the specified width and height
          values for the sub-view will be used. The ``flush`` setting can be useful when
          attempting to place sub-plots without axes or legends into a uniform grid
          structure.

        **Default value:** ``"full"``
    center : anyOf(boolean, :class:`RowColboolean`)
        Boolean flag indicating if subviews should be centered relative to their respective
        rows or columns.

        An object value of the form ``{"row": boolean, "column": boolean}`` can be used to
        supply different centering values for rows and columns.

        **Default value:** ``false``
    columns : float
        The number of columns to include in the view composition layout.

        **Default value** : ``undefined`` -- An infinite number of columns (a single row)
        will be assumed. This is equivalent to ``hconcat`` (for ``concat`` ) and to using
        the ``column`` channel (for ``facet`` and ``repeat`` ).

        **Note** :

        1) This property is only for: - the general (wrappable) ``concat`` operator (not
        ``hconcat`` / ``vconcat`` ) - the ``facet`` and ``repeat`` operator with one
        field/repetition definition (without row/column nesting)

        2) Setting the ``columns`` to ``1`` is equivalent to ``vconcat`` (for ``concat`` )
        and to using the ``row`` channel (for ``facet`` and ``repeat`` ).
    data : anyOf(:class:`Data`, None)
        An object describing the data source. Set to ``null`` to ignore the parent's data
        source. If no data is set, it is derived from the parent.
    description : string
        Description of this mark for commenting purpose.
    name : string
        Name of the visualization for later reference.
    resolve : :class:`Resolve`
        Scale, axis, and legend resolutions for view composition specifications.
    spacing : anyOf(float, :class:`RowColnumber`)
        The spacing in pixels between sub-views of the composition operator. An object of
        the form ``{"row": number, "column": number}`` can be used to set different spacing
        values for rows and columns.

        **Default value** : Depends on ``"spacing"`` property of `the view composition
        configuration <https://vega.github.io/vega-lite/docs/config.html#view-config>`__ (
        ``20`` by default)
    title : anyOf(:class:`Text`, :class:`TitleParams`)
        Title for the plot.
    transform : List(:class:`Transform`)
        An array of data transformations such as filter and new field calculation.
    """
    _schema = {'$ref': '#/definitions/LayerRepeatSpec'}

    def __init__(self, repeat=Undefined, spec=Undefined, align=Undefined, bounds=Undefined,
                 center=Undefined, columns=Undefined, data=Undefined, description=Undefined,
                 name=Undefined, resolve=Undefined, spacing=Undefined, title=Undefined,
                 transform=Undefined, **kwds):
        super(LayerRepeatSpec, self).__init__(repeat=repeat, spec=spec, align=align, bounds=bounds,
                                              center=center, columns=columns, data=data,
                                              description=description, name=name, resolve=resolve,
                                              spacing=spacing, title=title, transform=transform, **kwds)


class NonLayerRepeatSpec(RepeatSpec):
    """NonLayerRepeatSpec schema wrapper

    Mapping(required=[repeat, spec])
    Base interface for a repeat specification.

    Attributes
    ----------

    repeat : anyOf(List(string), :class:`RepeatMapping`)
        Definition for fields to be repeated. One of: 1) An array of fields to be repeated.
        If ``"repeat"`` is an array, the field can be referred to as ``{"repeat":
        "repeat"}``. The repeated views are laid out in a wrapped row. You can set the
        number of columns to control the wrapping. 2) An object that maps ``"row"`` and/or
        ``"column"`` to the listed fields to be repeated along the particular orientations.
        The objects ``{"repeat": "row"}`` and ``{"repeat": "column"}`` can be used to refer
        to the repeated field respectively.
    spec : :class:`Spec`
        A specification of the view that gets repeated.
    align : anyOf(:class:`LayoutAlign`, :class:`RowColLayoutAlign`)
        The alignment to apply to grid rows and columns. The supported string values are
        ``"all"``, ``"each"``, and ``"none"``.


        * For ``"none"``, a flow layout will be used, in which adjacent subviews are simply
          placed one after the other. - For ``"each"``, subviews will be aligned into a
          clean grid structure, but each row or column may be of variable size. - For
          ``"all"``, subviews will be aligned and each row or column will be sized
          identically based on the maximum observed size. String values for this property
          will be applied to both grid rows and columns.

        Alternatively, an object value of the form ``{"row": string, "column": string}`` can
        be used to supply different alignments for rows and columns.

        **Default value:** ``"all"``.
    bounds : enum('full', 'flush')
        The bounds calculation method to use for determining the extent of a sub-plot. One
        of ``full`` (the default) or ``flush``.


        * If set to ``full``, the entire calculated bounds (including axes, title, and
          legend) will be used. - If set to ``flush``, only the specified width and height
          values for the sub-view will be used. The ``flush`` setting can be useful when
          attempting to place sub-plots without axes or legends into a uniform grid
          structure.

        **Default value:** ``"full"``
    center : anyOf(boolean, :class:`RowColboolean`)
        Boolean flag indicating if subviews should be centered relative to their respective
        rows or columns.

        An object value of the form ``{"row": boolean, "column": boolean}`` can be used to
        supply different centering values for rows and columns.

        **Default value:** ``false``
    columns : float
        The number of columns to include in the view composition layout.

        **Default value** : ``undefined`` -- An infinite number of columns (a single row)
        will be assumed. This is equivalent to ``hconcat`` (for ``concat`` ) and to using
        the ``column`` channel (for ``facet`` and ``repeat`` ).

        **Note** :

        1) This property is only for: - the general (wrappable) ``concat`` operator (not
        ``hconcat`` / ``vconcat`` ) - the ``facet`` and ``repeat`` operator with one
        field/repetition definition (without row/column nesting)

        2) Setting the ``columns`` to ``1`` is equivalent to ``vconcat`` (for ``concat`` )
        and to using the ``row`` channel (for ``facet`` and ``repeat`` ).
    data : anyOf(:class:`Data`, None)
        An object describing the data source. Set to ``null`` to ignore the parent's data
        source. If no data is set, it is derived from the parent.
    description : string
        Description of this mark for commenting purpose.
    name : string
        Name of the visualization for later reference.
    resolve : :class:`Resolve`
        Scale, axis, and legend resolutions for view composition specifications.
    spacing : anyOf(float, :class:`RowColnumber`)
        The spacing in pixels between sub-views of the composition operator. An object of
        the form ``{"row": number, "column": number}`` can be used to set different spacing
        values for rows and columns.

        **Default value** : Depends on ``"spacing"`` property of `the view composition
        configuration <https://vega.github.io/vega-lite/docs/config.html#view-config>`__ (
        ``20`` by default)
    title : anyOf(:class:`Text`, :class:`TitleParams`)
        Title for the plot.
    transform : List(:class:`Transform`)
        An array of data transformations such as filter and new field calculation.
    """
    _schema = {'$ref': '#/definitions/NonLayerRepeatSpec'}

    def __init__(self, repeat=Undefined, spec=Undefined, align=Undefined, bounds=Undefined,
                 center=Undefined, columns=Undefined, data=Undefined, description=Undefined,
                 name=Undefined, resolve=Undefined, spacing=Undefined, title=Undefined,
                 transform=Undefined, **kwds):
        super(NonLayerRepeatSpec, self).__init__(repeat=repeat, spec=spec, align=align, bounds=bounds,
                                                 center=center, columns=columns, data=data,
                                                 description=description, name=name, resolve=resolve,
                                                 spacing=spacing, title=title, transform=transform,
                                                 **kwds)


class SphereGenerator(Generator):
    """SphereGenerator schema wrapper

    Mapping(required=[sphere])

    Attributes
    ----------

    sphere : anyOf(boolean, Mapping(required=[]))
        Generate sphere GeoJSON data for the full globe.
    name : string
        Provide a placeholder name and bind data at runtime.
    """
    _schema = {'$ref': '#/definitions/SphereGenerator'}

    def __init__(self, sphere=Undefined, name=Undefined, **kwds):
        super(SphereGenerator, self).__init__(sphere=sphere, name=name, **kwds)


class StackOffset(VegaLiteSchema):
    """StackOffset schema wrapper

    enum('zero', 'center', 'normalize')
    """
    _schema = {'$ref': '#/definitions/StackOffset'}

    def __init__(self, *args):
        super(StackOffset, self).__init__(*args)


class StandardType(VegaLiteSchema):
    """StandardType schema wrapper

    enum('quantitative', 'ordinal', 'temporal', 'nominal')
    """
    _schema = {'$ref': '#/definitions/StandardType'}

    def __init__(self, *args):
        super(StandardType, self).__init__(*args)


class Step(VegaLiteSchema):
    """Step schema wrapper

    Mapping(required=[step])

    Attributes
    ----------

    step : float
        The size (width/height) per discrete step.
    """
    _schema = {'$ref': '#/definitions/Step'}

    def __init__(self, step=Undefined, **kwds):
        super(Step, self).__init__(step=step, **kwds)


class Stream(VegaLiteSchema):
    """Stream schema wrapper

    anyOf(:class:`EventStream`, :class:`DerivedStream`, :class:`MergedStream`)
    """
    _schema = {'$ref': '#/definitions/Stream'}

    def __init__(self, *args, **kwds):
        super(Stream, self).__init__(*args, **kwds)


class DerivedStream(Stream):
    """DerivedStream schema wrapper

    Mapping(required=[stream])

    Attributes
    ----------

    stream : :class:`Stream`

    between : List(:class:`Stream`)

    consume : boolean

    debounce : float

    filter : anyOf(:class:`Expr`, List(:class:`Expr`))

    markname : string

    marktype : :class:`MarkType`

    throttle : float

    """
    _schema = {'$ref': '#/definitions/DerivedStream'}

    def __init__(self, stream=Undefined, between=Undefined, consume=Undefined, debounce=Undefined,
                 filter=Undefined, markname=Undefined, marktype=Undefined, throttle=Undefined, **kwds):
        super(DerivedStream, self).__init__(stream=stream, between=between, consume=consume,
                                            debounce=debounce, filter=filter, markname=markname,
                                            marktype=marktype, throttle=throttle, **kwds)


class EventStream(Stream):
    """EventStream schema wrapper

    anyOf(Mapping(required=[type]), Mapping(required=[source, type]))
    """
    _schema = {'$ref': '#/definitions/EventStream'}

    def __init__(self, *args, **kwds):
        super(EventStream, self).__init__(*args, **kwds)


class MergedStream(Stream):
    """MergedStream schema wrapper

    Mapping(required=[merge])

    Attributes
    ----------

    merge : List(:class:`Stream`)

    between : List(:class:`Stream`)

    consume : boolean

    debounce : float

    filter : anyOf(:class:`Expr`, List(:class:`Expr`))

    markname : string

    marktype : :class:`MarkType`

    throttle : float

    """
    _schema = {'$ref': '#/definitions/MergedStream'}

    def __init__(self, merge=Undefined, between=Undefined, consume=Undefined, debounce=Undefined,
                 filter=Undefined, markname=Undefined, marktype=Undefined, throttle=Undefined, **kwds):
        super(MergedStream, self).__init__(merge=merge, between=between, consume=consume,
                                           debounce=debounce, filter=filter, markname=markname,
                                           marktype=marktype, throttle=throttle, **kwds)


class StringFieldDef(VegaLiteSchema):
    """StringFieldDef schema wrapper

    Mapping(required=[])

    Attributes
    ----------

    aggregate : :class:`Aggregate`
        Aggregation function for the field (e.g., ``"mean"``, ``"sum"``, ``"median"``,
        ``"min"``, ``"max"``, ``"count"`` ).

        **Default value:** ``undefined`` (None)

        **See also:** `aggregate <https://vega.github.io/vega-lite/docs/aggregate.html>`__
        documentation.
    band : float
        For rect-based marks ( ``rect``, ``bar``, and ``image`` ), mark size relative to
        bandwidth of `band scales
        <https://vega.github.io/vega-lite/docs/scale.html#band>`__, bins or time units. If
        set to ``1``, the mark size is set to the bandwidth, the bin interval, or the time
        unit interval. If set to ``0.5``, the mark size is half of the bandwidth or the time
        unit interval.

        For other marks, relative position on a band of a stacked, binned, time unit or band
        scale. If set to ``0``, the marks will be positioned at the beginning of the band.
        If set to ``0.5``, the marks will be positioned in the middle of the band.
    bin : anyOf(boolean, :class:`BinParams`, string, None)
        A flag for binning a ``quantitative`` field, `an object defining binning parameters
        <https://vega.github.io/vega-lite/docs/bin.html#params>`__, or indicating that the
        data for ``x`` or ``y`` channel are binned before they are imported into Vega-Lite (
        ``"binned"`` ).


        If ``true``, default `binning parameters
        <https://vega.github.io/vega-lite/docs/bin.html>`__ will be applied.

        If ``"binned"``, this indicates that the data for the ``x`` (or ``y`` ) channel are
        already binned. You can map the bin-start field to ``x`` (or ``y`` ) and the bin-end
        field to ``x2`` (or ``y2`` ). The scale and axis will be formatted similar to
        binning in Vega-Lite.  To adjust the axis ticks based on the bin step, you can also
        set the axis's `tickMinStep
        <https://vega.github.io/vega-lite/docs/axis.html#ticks>`__ property.

        **Default value:** ``false``

        **See also:** `bin <https://vega.github.io/vega-lite/docs/bin.html>`__
        documentation.
    field : :class:`Field`
        **Required.** A string defining the name of the field from which to pull a data
        value or an object defining iterated values from the `repeat
        <https://vega.github.io/vega-lite/docs/repeat.html>`__ operator.

        **See also:** `field <https://vega.github.io/vega-lite/docs/field.html>`__
        documentation.

        **Notes:** 1)  Dots ( ``.`` ) and brackets ( ``[`` and ``]`` ) can be used to access
        nested objects (e.g., ``"field": "foo.bar"`` and ``"field": "foo['bar']"`` ). If
        field names contain dots or brackets but are not nested, you can use ``\\`` to
        escape dots and brackets (e.g., ``"a\\.b"`` and ``"a\\[0\\]"`` ). See more details
        about escaping in the `field documentation
        <https://vega.github.io/vega-lite/docs/field.html>`__. 2) ``field`` is not required
        if ``aggregate`` is ``count``.
    format : anyOf(string, :class:`Dictunknown`)
        When used with the default ``"number"`` and ``"time"`` format type, the text
        formatting pattern for labels of guides (axes, legends, headers) and text marks.


        * If the format type is ``"number"`` (e.g., for quantitative fields), this is D3's
          `number format pattern <https://github.com/d3/d3-format#locale_format>`__. - If
          the format type is ``"time"`` (e.g., for temporal fields), this is D3's `time
          format pattern <https://github.com/d3/d3-time-format#locale_format>`__.

        See the `format documentation <https://vega.github.io/vega-lite/docs/format.html>`__
        for more examples.

        When used with a `custom formatType
        <https://vega.github.io/vega-lite/docs/config.html#custom-format-type>`__, this
        value will be passed as ``format`` alongside ``datum.value`` to the registered
        function.

        **Default value:**  Derived from `numberFormat
        <https://vega.github.io/vega-lite/docs/config.html#format>`__ config for number
        format and from `timeFormat
        <https://vega.github.io/vega-lite/docs/config.html#format>`__ config for time
        format.
    formatType : string
        The format type for labels. One of ``"number"``, ``"time"``, or a `registered custom
        format type
        <https://vega.github.io/vega-lite/docs/config.html#custom-format-type>`__.

        **Default value:** - ``"time"`` for temporal fields and ordinal and nominal fields
        with ``timeUnit``. - ``"number"`` for quantitative fields as well as ordinal and
        nominal fields without ``timeUnit``.
    labelExpr : string
        `Vega expression <https://vega.github.io/vega/docs/expressions/>`__ for customizing
        labels text.

        **Note:** The label text and value can be assessed via the ``label`` and ``value``
        properties of the axis's backing ``datum`` object.
    timeUnit : anyOf(:class:`TimeUnit`, :class:`TimeUnitParams`)
        Time unit (e.g., ``year``, ``yearmonth``, ``month``, ``hours`` ) for a temporal
        field. or `a temporal field that gets casted as ordinal
        <https://vega.github.io/vega-lite/docs/type.html#cast>`__.

        **Default value:** ``undefined`` (None)

        **See also:** `timeUnit <https://vega.github.io/vega-lite/docs/timeunit.html>`__
        documentation.
    title : anyOf(:class:`Text`, None)
        A title for the field. If ``null``, the title will be removed.

        **Default value:**  derived from the field's name and transformation function (
        ``aggregate``, ``bin`` and ``timeUnit`` ). If the field has an aggregate function,
        the function is displayed as part of the title (e.g., ``"Sum of Profit"`` ). If the
        field is binned or has a time unit applied, the applied function is shown in
        parentheses (e.g., ``"Profit (binned)"``, ``"Transaction Date (year-month)"`` ).
        Otherwise, the title is simply the field name.

        **Notes** :

        1) You can customize the default field title format by providing the `fieldTitle
        <https://vega.github.io/vega-lite/docs/config.html#top-level-config>`__ property in
        the `config <https://vega.github.io/vega-lite/docs/config.html>`__ or `fieldTitle
        function via the compile function's options
        <https://vega.github.io/vega-lite/docs/compile.html#field-title>`__.

        2) If both field definition's ``title`` and axis, header, or legend ``title`` are
        defined, axis/header/legend title will be used.
    type : :class:`StandardType`
        The type of measurement ( ``"quantitative"``, ``"temporal"``, ``"ordinal"``, or
        ``"nominal"`` ) for the encoded field or constant value ( ``datum`` ). It can also
        be a ``"geojson"`` type for encoding `'geoshape'
        <https://vega.github.io/vega-lite/docs/geoshape.html>`__.

        Vega-Lite automatically infers data types in many cases as discussed below. However,
        type is required for a field if: (1) the field is not nominal and the field encoding
        has no specified ``aggregate`` (except ``argmin`` and ``argmax`` ), ``bin``, scale
        type, custom ``sort`` order, nor ``timeUnit`` or (2) if you wish to use an ordinal
        scale for a field with ``bin`` or ``timeUnit``.

        **Default value:**

        1) For a data ``field``, ``"nominal"`` is the default data type unless the field
        encoding has ``aggregate``, ``channel``, ``bin``, scale type, ``sort``, or
        ``timeUnit`` that satisfies the following criteria: - ``"quantitative"`` is the
        default type if (1) the encoded field contains ``bin`` or ``aggregate`` except
        ``"argmin"`` and ``"argmax"``, (2) the encoding channel is ``latitude`` or
        ``longitude`` channel or (3) if the specified scale type is `a quantitative scale
        <https://vega.github.io/vega-lite/docs/scale.html#type>`__. - ``"temporal"`` is the
        default type if (1) the encoded field contains ``timeUnit`` or (2) the specified
        scale type is a time or utc scale - ``ordinal""`` is the default type if (1) the
        encoded field contains a `custom sort order
        <https://vega.github.io/vega-lite/docs/sort.html#specifying-custom-sort-order>`__,
        (2) the specified scale type is an ordinal/point/band scale, or (3) the encoding
        channel is ``order``.

        2) For a constant value in data domain ( ``datum`` ): - ``"quantitative"`` if the
        datum is a number - ``"nominal"`` if the datum is a string - ``"temporal"`` if the
        datum is `a date time object
        <https://vega.github.io/vega-lite/docs/datetime.html>`__

        **Note:** - Data ``type`` describes the semantics of the data rather than the
        primitive data types (number, string, etc.). The same primitive data type can have
        different types of measurement. For example, numeric data can represent
        quantitative, ordinal, or nominal data. - Data values for a temporal field can be
        either a date-time string (e.g., ``"2015-03-07 12:32:17"``, ``"17:01"``,
        ``"2015-03-16"``. ``"2015"`` ) or a timestamp number (e.g., ``1552199579097`` ). -
        When using with `bin <https://vega.github.io/vega-lite/docs/bin.html>`__, the
        ``type`` property can be either ``"quantitative"`` (for using a linear bin scale) or
        `"ordinal" (for using an ordinal bin scale)
        <https://vega.github.io/vega-lite/docs/type.html#cast-bin>`__. - When using with
        `timeUnit <https://vega.github.io/vega-lite/docs/timeunit.html>`__, the ``type``
        property can be either ``"temporal"`` (default, for using a temporal scale) or
        `"ordinal" (for using an ordinal scale)
        <https://vega.github.io/vega-lite/docs/type.html#cast-bin>`__. - When using with
        `aggregate <https://vega.github.io/vega-lite/docs/aggregate.html>`__, the ``type``
        property refers to the post-aggregation data type. For example, we can calculate
        count ``distinct`` of a categorical field ``"cat"`` using ``{"aggregate":
        "distinct", "field": "cat"}``. The ``"type"`` of the aggregate output is
        ``"quantitative"``. - Secondary channels (e.g., ``x2``, ``y2``, ``xError``,
        ``yError`` ) do not have ``type`` as they must have exactly the same type as their
        primary channels (e.g., ``x``, ``y`` ).

        **See also:** `type <https://vega.github.io/vega-lite/docs/type.html>`__
        documentation.
    """
    _schema = {'$ref': '#/definitions/StringFieldDef'}

    def __init__(self, aggregate=Undefined, band=Undefined, bin=Undefined, field=Undefined,
                 format=Undefined, formatType=Undefined, labelExpr=Undefined, timeUnit=Undefined,
                 title=Undefined, type=Undefined, **kwds):
        super(StringFieldDef, self).__init__(aggregate=aggregate, band=band, bin=bin, field=field,
                                             format=format, formatType=formatType, labelExpr=labelExpr,
                                             timeUnit=timeUnit, title=title, type=type, **kwds)


class StringFieldDefWithCondition(VegaLiteSchema):
    """StringFieldDefWithCondition schema wrapper

    Mapping(required=[])
    A FieldDef with Condition :raw-html:`<ValueDef>` {    condition: {value: ...},    field:
    ...,    ... }

    Attributes
    ----------

    aggregate : :class:`Aggregate`
        Aggregation function for the field (e.g., ``"mean"``, ``"sum"``, ``"median"``,
        ``"min"``, ``"max"``, ``"count"`` ).

        **Default value:** ``undefined`` (None)

        **See also:** `aggregate <https://vega.github.io/vega-lite/docs/aggregate.html>`__
        documentation.
    band : float
        For rect-based marks ( ``rect``, ``bar``, and ``image`` ), mark size relative to
        bandwidth of `band scales
        <https://vega.github.io/vega-lite/docs/scale.html#band>`__, bins or time units. If
        set to ``1``, the mark size is set to the bandwidth, the bin interval, or the time
        unit interval. If set to ``0.5``, the mark size is half of the bandwidth or the time
        unit interval.

        For other marks, relative position on a band of a stacked, binned, time unit or band
        scale. If set to ``0``, the marks will be positioned at the beginning of the band.
        If set to ``0.5``, the marks will be positioned in the middle of the band.
    bin : anyOf(boolean, :class:`BinParams`, string, None)
        A flag for binning a ``quantitative`` field, `an object defining binning parameters
        <https://vega.github.io/vega-lite/docs/bin.html#params>`__, or indicating that the
        data for ``x`` or ``y`` channel are binned before they are imported into Vega-Lite (
        ``"binned"`` ).


        If ``true``, default `binning parameters
        <https://vega.github.io/vega-lite/docs/bin.html>`__ will be applied.

        If ``"binned"``, this indicates that the data for the ``x`` (or ``y`` ) channel are
        already binned. You can map the bin-start field to ``x`` (or ``y`` ) and the bin-end
        field to ``x2`` (or ``y2`` ). The scale and axis will be formatted similar to
        binning in Vega-Lite.  To adjust the axis ticks based on the bin step, you can also
        set the axis's `tickMinStep
        <https://vega.github.io/vega-lite/docs/axis.html#ticks>`__ property.

        **Default value:** ``false``

        **See also:** `bin <https://vega.github.io/vega-lite/docs/bin.html>`__
        documentation.
    condition : anyOf(:class:`ConditionalValueDefstringExprRef`,
    List(:class:`ConditionalValueDefstringExprRef`))
        One or more value definition(s) with `a selection or a test predicate
        <https://vega.github.io/vega-lite/docs/condition.html>`__.

        **Note:** A field definition's ``condition`` property can only contain `conditional
        value definitions <https://vega.github.io/vega-lite/docs/condition.html#value>`__
        since Vega-Lite only allows at most one encoded field per encoding channel.
    field : :class:`Field`
        **Required.** A string defining the name of the field from which to pull a data
        value or an object defining iterated values from the `repeat
        <https://vega.github.io/vega-lite/docs/repeat.html>`__ operator.

        **See also:** `field <https://vega.github.io/vega-lite/docs/field.html>`__
        documentation.

        **Notes:** 1)  Dots ( ``.`` ) and brackets ( ``[`` and ``]`` ) can be used to access
        nested objects (e.g., ``"field": "foo.bar"`` and ``"field": "foo['bar']"`` ). If
        field names contain dots or brackets but are not nested, you can use ``\\`` to
        escape dots and brackets (e.g., ``"a\\.b"`` and ``"a\\[0\\]"`` ). See more details
        about escaping in the `field documentation
        <https://vega.github.io/vega-lite/docs/field.html>`__. 2) ``field`` is not required
        if ``aggregate`` is ``count``.
    format : anyOf(string, :class:`Dictunknown`)
        When used with the default ``"number"`` and ``"time"`` format type, the text
        formatting pattern for labels of guides (axes, legends, headers) and text marks.


        * If the format type is ``"number"`` (e.g., for quantitative fields), this is D3's
          `number format pattern <https://github.com/d3/d3-format#locale_format>`__. - If
          the format type is ``"time"`` (e.g., for temporal fields), this is D3's `time
          format pattern <https://github.com/d3/d3-time-format#locale_format>`__.

        See the `format documentation <https://vega.github.io/vega-lite/docs/format.html>`__
        for more examples.

        When used with a `custom formatType
        <https://vega.github.io/vega-lite/docs/config.html#custom-format-type>`__, this
        value will be passed as ``format`` alongside ``datum.value`` to the registered
        function.

        **Default value:**  Derived from `numberFormat
        <https://vega.github.io/vega-lite/docs/config.html#format>`__ config for number
        format and from `timeFormat
        <https://vega.github.io/vega-lite/docs/config.html#format>`__ config for time
        format.
    formatType : string
        The format type for labels. One of ``"number"``, ``"time"``, or a `registered custom
        format type
        <https://vega.github.io/vega-lite/docs/config.html#custom-format-type>`__.

        **Default value:** - ``"time"`` for temporal fields and ordinal and nominal fields
        with ``timeUnit``. - ``"number"`` for quantitative fields as well as ordinal and
        nominal fields without ``timeUnit``.
    labelExpr : string
        `Vega expression <https://vega.github.io/vega/docs/expressions/>`__ for customizing
        labels text.

        **Note:** The label text and value can be assessed via the ``label`` and ``value``
        properties of the axis's backing ``datum`` object.
    timeUnit : anyOf(:class:`TimeUnit`, :class:`TimeUnitParams`)
        Time unit (e.g., ``year``, ``yearmonth``, ``month``, ``hours`` ) for a temporal
        field. or `a temporal field that gets casted as ordinal
        <https://vega.github.io/vega-lite/docs/type.html#cast>`__.

        **Default value:** ``undefined`` (None)

        **See also:** `timeUnit <https://vega.github.io/vega-lite/docs/timeunit.html>`__
        documentation.
    title : anyOf(:class:`Text`, None)
        A title for the field. If ``null``, the title will be removed.

        **Default value:**  derived from the field's name and transformation function (
        ``aggregate``, ``bin`` and ``timeUnit`` ). If the field has an aggregate function,
        the function is displayed as part of the title (e.g., ``"Sum of Profit"`` ). If the
        field is binned or has a time unit applied, the applied function is shown in
        parentheses (e.g., ``"Profit (binned)"``, ``"Transaction Date (year-month)"`` ).
        Otherwise, the title is simply the field name.

        **Notes** :

        1) You can customize the default field title format by providing the `fieldTitle
        <https://vega.github.io/vega-lite/docs/config.html#top-level-config>`__ property in
        the `config <https://vega.github.io/vega-lite/docs/config.html>`__ or `fieldTitle
        function via the compile function's options
        <https://vega.github.io/vega-lite/docs/compile.html#field-title>`__.

        2) If both field definition's ``title`` and axis, header, or legend ``title`` are
        defined, axis/header/legend title will be used.
    type : :class:`StandardType`
        The type of measurement ( ``"quantitative"``, ``"temporal"``, ``"ordinal"``, or
        ``"nominal"`` ) for the encoded field or constant value ( ``datum`` ). It can also
        be a ``"geojson"`` type for encoding `'geoshape'
        <https://vega.github.io/vega-lite/docs/geoshape.html>`__.

        Vega-Lite automatically infers data types in many cases as discussed below. However,
        type is required for a field if: (1) the field is not nominal and the field encoding
        has no specified ``aggregate`` (except ``argmin`` and ``argmax`` ), ``bin``, scale
        type, custom ``sort`` order, nor ``timeUnit`` or (2) if you wish to use an ordinal
        scale for a field with ``bin`` or ``timeUnit``.

        **Default value:**

        1) For a data ``field``, ``"nominal"`` is the default data type unless the field
        encoding has ``aggregate``, ``channel``, ``bin``, scale type, ``sort``, or
        ``timeUnit`` that satisfies the following criteria: - ``"quantitative"`` is the
        default type if (1) the encoded field contains ``bin`` or ``aggregate`` except
        ``"argmin"`` and ``"argmax"``, (2) the encoding channel is ``latitude`` or
        ``longitude`` channel or (3) if the specified scale type is `a quantitative scale
        <https://vega.github.io/vega-lite/docs/scale.html#type>`__. - ``"temporal"`` is the
        default type if (1) the encoded field contains ``timeUnit`` or (2) the specified
        scale type is a time or utc scale - ``ordinal""`` is the default type if (1) the
        encoded field contains a `custom sort order
        <https://vega.github.io/vega-lite/docs/sort.html#specifying-custom-sort-order>`__,
        (2) the specified scale type is an ordinal/point/band scale, or (3) the encoding
        channel is ``order``.

        2) For a constant value in data domain ( ``datum`` ): - ``"quantitative"`` if the
        datum is a number - ``"nominal"`` if the datum is a string - ``"temporal"`` if the
        datum is `a date time object
        <https://vega.github.io/vega-lite/docs/datetime.html>`__

        **Note:** - Data ``type`` describes the semantics of the data rather than the
        primitive data types (number, string, etc.). The same primitive data type can have
        different types of measurement. For example, numeric data can represent
        quantitative, ordinal, or nominal data. - Data values for a temporal field can be
        either a date-time string (e.g., ``"2015-03-07 12:32:17"``, ``"17:01"``,
        ``"2015-03-16"``. ``"2015"`` ) or a timestamp number (e.g., ``1552199579097`` ). -
        When using with `bin <https://vega.github.io/vega-lite/docs/bin.html>`__, the
        ``type`` property can be either ``"quantitative"`` (for using a linear bin scale) or
        `"ordinal" (for using an ordinal bin scale)
        <https://vega.github.io/vega-lite/docs/type.html#cast-bin>`__. - When using with
        `timeUnit <https://vega.github.io/vega-lite/docs/timeunit.html>`__, the ``type``
        property can be either ``"temporal"`` (default, for using a temporal scale) or
        `"ordinal" (for using an ordinal scale)
        <https://vega.github.io/vega-lite/docs/type.html#cast-bin>`__. - When using with
        `aggregate <https://vega.github.io/vega-lite/docs/aggregate.html>`__, the ``type``
        property refers to the post-aggregation data type. For example, we can calculate
        count ``distinct`` of a categorical field ``"cat"`` using ``{"aggregate":
        "distinct", "field": "cat"}``. The ``"type"`` of the aggregate output is
        ``"quantitative"``. - Secondary channels (e.g., ``x2``, ``y2``, ``xError``,
        ``yError`` ) do not have ``type`` as they must have exactly the same type as their
        primary channels (e.g., ``x``, ``y`` ).

        **See also:** `type <https://vega.github.io/vega-lite/docs/type.html>`__
        documentation.
    """
    _schema = {'$ref': '#/definitions/StringFieldDefWithCondition'}

    def __init__(self, aggregate=Undefined, band=Undefined, bin=Undefined, condition=Undefined,
                 field=Undefined, format=Undefined, formatType=Undefined, labelExpr=Undefined,
                 timeUnit=Undefined, title=Undefined, type=Undefined, **kwds):
        super(StringFieldDefWithCondition, self).__init__(aggregate=aggregate, band=band, bin=bin,
                                                          condition=condition, field=field,
                                                          format=format, formatType=formatType,
                                                          labelExpr=labelExpr, timeUnit=timeUnit,
                                                          title=title, type=type, **kwds)


class StringValueDefWithCondition(VegaLiteSchema):
    """StringValueDefWithCondition schema wrapper

    Mapping(required=[])

    Attributes
    ----------

    condition : anyOf(:class:`ConditionalMarkPropFieldOrDatumDef`,
    :class:`ConditionalValueDefstringnullExprRef`,
    List(:class:`ConditionalValueDefstringnullExprRef`))
        A field definition or one or more value definition(s) with a selection predicate.
    value : anyOf(string, None, :class:`ExprRef`)
        A constant value in visual domain (e.g., ``"red"`` / ``"#0099ff"`` / `gradient
        definition <https://vega.github.io/vega-lite/docs/types.html#gradient>`__ for color,
        values between ``0`` to ``1`` for opacity).
    """
    _schema = {'$ref': '#/definitions/StringValueDefWithCondition'}

    def __init__(self, condition=Undefined, value=Undefined, **kwds):
        super(StringValueDefWithCondition, self).__init__(condition=condition, value=value, **kwds)


class StrokeCap(VegaLiteSchema):
    """StrokeCap schema wrapper

    enum('butt', 'round', 'square')
    """
    _schema = {'$ref': '#/definitions/StrokeCap'}

    def __init__(self, *args):
        super(StrokeCap, self).__init__(*args)


class StrokeJoin(VegaLiteSchema):
    """StrokeJoin schema wrapper

    enum('miter', 'round', 'bevel')
    """
    _schema = {'$ref': '#/definitions/StrokeJoin'}

    def __init__(self, *args):
        super(StrokeJoin, self).__init__(*args)


class StyleConfigIndex(VegaLiteSchema):
    """StyleConfigIndex schema wrapper

    Mapping(required=[])

    Attributes
    ----------

    arc : :class:`RectConfig`
        Arc-specific Config
    area : :class:`AreaConfig`
        Area-Specific Config
    bar : :class:`BarConfig`
        Bar-Specific Config
    circle : :class:`MarkConfig`
        Circle-Specific Config
    geoshape : :class:`MarkConfig`
        Geoshape-Specific Config
    image : :class:`RectConfig`
        Image-specific Config
    line : :class:`LineConfig`
        Line-Specific Config
    mark : :class:`MarkConfig`
        Mark Config
    point : :class:`MarkConfig`
        Point-Specific Config
    rect : :class:`RectConfig`
        Rect-Specific Config
    rule : :class:`MarkConfig`
        Rule-Specific Config
    square : :class:`MarkConfig`
        Square-Specific Config
    text : :class:`MarkConfig`
        Text-Specific Config
    tick : :class:`TickConfig`
        Tick-Specific Config
    trail : :class:`LineConfig`
        Trail-Specific Config
    group-subtitle : :class:`MarkConfig`
        Default style for chart subtitles
    group-title : :class:`MarkConfig`
        Default style for chart titles
    guide-label : :class:`MarkConfig`
        Default style for axis, legend, and header labels.
    guide-title : :class:`MarkConfig`
        Default style for axis, legend, and header titles.
    """
    _schema = {'$ref': '#/definitions/StyleConfigIndex'}

    def __init__(self, arc=Undefined, area=Undefined, bar=Undefined, circle=Undefined,
                 geoshape=Undefined, image=Undefined, line=Undefined, mark=Undefined, point=Undefined,
                 rect=Undefined, rule=Undefined, square=Undefined, text=Undefined, tick=Undefined,
                 trail=Undefined, **kwds):
        super(StyleConfigIndex, self).__init__(arc=arc, area=area, bar=bar, circle=circle,
                                               geoshape=geoshape, image=image, line=line, mark=mark,
                                               point=point, rect=rect, rule=rule, square=square,
                                               text=text, tick=tick, trail=trail, **kwds)


class SymbolShape(VegaLiteSchema):
    """SymbolShape schema wrapper

    string
    """
    _schema = {'$ref': '#/definitions/SymbolShape'}

    def __init__(self, *args):
        super(SymbolShape, self).__init__(*args)


class Text(VegaLiteSchema):
    """Text schema wrapper

    anyOf(string, List(string))
    """
    _schema = {'$ref': '#/definitions/Text'}

    def __init__(self, *args, **kwds):
        super(Text, self).__init__(*args, **kwds)


class TextBaseline(VegaLiteSchema):
    """TextBaseline schema wrapper

    anyOf(string, :class:`Baseline`, string, string)
    """
    _schema = {'$ref': '#/definitions/TextBaseline'}

    def __init__(self, *args, **kwds):
        super(TextBaseline, self).__init__(*args, **kwds)


class Baseline(TextBaseline):
    """Baseline schema wrapper

    enum('top', 'middle', 'bottom')
    """
    _schema = {'$ref': '#/definitions/Baseline'}

    def __init__(self, *args):
        super(Baseline, self).__init__(*args)


class TextDef(VegaLiteSchema):
    """TextDef schema wrapper

    anyOf(:class:`FieldOrDatumDefWithConditionStringFieldDefText`,
    :class:`FieldOrDatumDefWithConditionStringDatumDefText`,
    :class:`ValueDefWithConditionStringFieldDefText`)
    """
    _schema = {'$ref': '#/definitions/TextDef'}

    def __init__(self, *args, **kwds):
        super(TextDef, self).__init__(*args, **kwds)


class FieldOrDatumDefWithConditionStringDatumDefText(TextDef):
    """FieldOrDatumDefWithConditionStringDatumDefText schema wrapper

    Mapping(required=[])
    A FieldDef with Condition :raw-html:`<ValueDef>` {    condition: {value: ...},    field:
    ...,    ... }

    Attributes
    ----------

    band : float
        For rect-based marks ( ``rect``, ``bar``, and ``image`` ), mark size relative to
        bandwidth of `band scales
        <https://vega.github.io/vega-lite/docs/scale.html#band>`__, bins or time units. If
        set to ``1``, the mark size is set to the bandwidth, the bin interval, or the time
        unit interval. If set to ``0.5``, the mark size is half of the bandwidth or the time
        unit interval.

        For other marks, relative position on a band of a stacked, binned, time unit or band
        scale. If set to ``0``, the marks will be positioned at the beginning of the band.
        If set to ``0.5``, the marks will be positioned in the middle of the band.
    condition : anyOf(:class:`ConditionalValueDefTextExprRef`,
    List(:class:`ConditionalValueDefTextExprRef`))
        One or more value definition(s) with `a selection or a test predicate
        <https://vega.github.io/vega-lite/docs/condition.html>`__.

        **Note:** A field definition's ``condition`` property can only contain `conditional
        value definitions <https://vega.github.io/vega-lite/docs/condition.html#value>`__
        since Vega-Lite only allows at most one encoded field per encoding channel.
    datum : anyOf(:class:`PrimitiveValue`, :class:`DateTime`, :class:`ExprRef`,
    :class:`RepeatRef`)
        A constant value in data domain.
    format : anyOf(string, :class:`Dictunknown`)
        When used with the default ``"number"`` and ``"time"`` format type, the text
        formatting pattern for labels of guides (axes, legends, headers) and text marks.


        * If the format type is ``"number"`` (e.g., for quantitative fields), this is D3's
          `number format pattern <https://github.com/d3/d3-format#locale_format>`__. - If
          the format type is ``"time"`` (e.g., for temporal fields), this is D3's `time
          format pattern <https://github.com/d3/d3-time-format#locale_format>`__.

        See the `format documentation <https://vega.github.io/vega-lite/docs/format.html>`__
        for more examples.

        When used with a `custom formatType
        <https://vega.github.io/vega-lite/docs/config.html#custom-format-type>`__, this
        value will be passed as ``format`` alongside ``datum.value`` to the registered
        function.

        **Default value:**  Derived from `numberFormat
        <https://vega.github.io/vega-lite/docs/config.html#format>`__ config for number
        format and from `timeFormat
        <https://vega.github.io/vega-lite/docs/config.html#format>`__ config for time
        format.
    formatType : string
        The format type for labels. One of ``"number"``, ``"time"``, or a `registered custom
        format type
        <https://vega.github.io/vega-lite/docs/config.html#custom-format-type>`__.

        **Default value:** - ``"time"`` for temporal fields and ordinal and nominal fields
        with ``timeUnit``. - ``"number"`` for quantitative fields as well as ordinal and
        nominal fields without ``timeUnit``.
    labelExpr : string
        `Vega expression <https://vega.github.io/vega/docs/expressions/>`__ for customizing
        labels text.

        **Note:** The label text and value can be assessed via the ``label`` and ``value``
        properties of the axis's backing ``datum`` object.
    type : :class:`Type`
        The type of measurement ( ``"quantitative"``, ``"temporal"``, ``"ordinal"``, or
        ``"nominal"`` ) for the encoded field or constant value ( ``datum`` ). It can also
        be a ``"geojson"`` type for encoding `'geoshape'
        <https://vega.github.io/vega-lite/docs/geoshape.html>`__.

        Vega-Lite automatically infers data types in many cases as discussed below. However,
        type is required for a field if: (1) the field is not nominal and the field encoding
        has no specified ``aggregate`` (except ``argmin`` and ``argmax`` ), ``bin``, scale
        type, custom ``sort`` order, nor ``timeUnit`` or (2) if you wish to use an ordinal
        scale for a field with ``bin`` or ``timeUnit``.

        **Default value:**

        1) For a data ``field``, ``"nominal"`` is the default data type unless the field
        encoding has ``aggregate``, ``channel``, ``bin``, scale type, ``sort``, or
        ``timeUnit`` that satisfies the following criteria: - ``"quantitative"`` is the
        default type if (1) the encoded field contains ``bin`` or ``aggregate`` except
        ``"argmin"`` and ``"argmax"``, (2) the encoding channel is ``latitude`` or
        ``longitude`` channel or (3) if the specified scale type is `a quantitative scale
        <https://vega.github.io/vega-lite/docs/scale.html#type>`__. - ``"temporal"`` is the
        default type if (1) the encoded field contains ``timeUnit`` or (2) the specified
        scale type is a time or utc scale - ``ordinal""`` is the default type if (1) the
        encoded field contains a `custom sort order
        <https://vega.github.io/vega-lite/docs/sort.html#specifying-custom-sort-order>`__,
        (2) the specified scale type is an ordinal/point/band scale, or (3) the encoding
        channel is ``order``.

        2) For a constant value in data domain ( ``datum`` ): - ``"quantitative"`` if the
        datum is a number - ``"nominal"`` if the datum is a string - ``"temporal"`` if the
        datum is `a date time object
        <https://vega.github.io/vega-lite/docs/datetime.html>`__

        **Note:** - Data ``type`` describes the semantics of the data rather than the
        primitive data types (number, string, etc.). The same primitive data type can have
        different types of measurement. For example, numeric data can represent
        quantitative, ordinal, or nominal data. - Data values for a temporal field can be
        either a date-time string (e.g., ``"2015-03-07 12:32:17"``, ``"17:01"``,
        ``"2015-03-16"``. ``"2015"`` ) or a timestamp number (e.g., ``1552199579097`` ). -
        When using with `bin <https://vega.github.io/vega-lite/docs/bin.html>`__, the
        ``type`` property can be either ``"quantitative"`` (for using a linear bin scale) or
        `"ordinal" (for using an ordinal bin scale)
        <https://vega.github.io/vega-lite/docs/type.html#cast-bin>`__. - When using with
        `timeUnit <https://vega.github.io/vega-lite/docs/timeunit.html>`__, the ``type``
        property can be either ``"temporal"`` (default, for using a temporal scale) or
        `"ordinal" (for using an ordinal scale)
        <https://vega.github.io/vega-lite/docs/type.html#cast-bin>`__. - When using with
        `aggregate <https://vega.github.io/vega-lite/docs/aggregate.html>`__, the ``type``
        property refers to the post-aggregation data type. For example, we can calculate
        count ``distinct`` of a categorical field ``"cat"`` using ``{"aggregate":
        "distinct", "field": "cat"}``. The ``"type"`` of the aggregate output is
        ``"quantitative"``. - Secondary channels (e.g., ``x2``, ``y2``, ``xError``,
        ``yError`` ) do not have ``type`` as they must have exactly the same type as their
        primary channels (e.g., ``x``, ``y`` ).

        **See also:** `type <https://vega.github.io/vega-lite/docs/type.html>`__
        documentation.
    """
    _schema = {'$ref': '#/definitions/FieldOrDatumDefWithCondition<StringDatumDef,Text>'}

    def __init__(self, band=Undefined, condition=Undefined, datum=Undefined, format=Undefined,
                 formatType=Undefined, labelExpr=Undefined, type=Undefined, **kwds):
        super(FieldOrDatumDefWithConditionStringDatumDefText, self).__init__(band=band,
                                                                             condition=condition,
                                                                             datum=datum, format=format,
                                                                             formatType=formatType,
                                                                             labelExpr=labelExpr,
                                                                             type=type, **kwds)


class FieldOrDatumDefWithConditionStringFieldDefText(TextDef):
    """FieldOrDatumDefWithConditionStringFieldDefText schema wrapper

    Mapping(required=[])
    A FieldDef with Condition :raw-html:`<ValueDef>` {    condition: {value: ...},    field:
    ...,    ... }

    Attributes
    ----------

    aggregate : :class:`Aggregate`
        Aggregation function for the field (e.g., ``"mean"``, ``"sum"``, ``"median"``,
        ``"min"``, ``"max"``, ``"count"`` ).

        **Default value:** ``undefined`` (None)

        **See also:** `aggregate <https://vega.github.io/vega-lite/docs/aggregate.html>`__
        documentation.
    band : float
        For rect-based marks ( ``rect``, ``bar``, and ``image`` ), mark size relative to
        bandwidth of `band scales
        <https://vega.github.io/vega-lite/docs/scale.html#band>`__, bins or time units. If
        set to ``1``, the mark size is set to the bandwidth, the bin interval, or the time
        unit interval. If set to ``0.5``, the mark size is half of the bandwidth or the time
        unit interval.

        For other marks, relative position on a band of a stacked, binned, time unit or band
        scale. If set to ``0``, the marks will be positioned at the beginning of the band.
        If set to ``0.5``, the marks will be positioned in the middle of the band.
    bin : anyOf(boolean, :class:`BinParams`, string, None)
        A flag for binning a ``quantitative`` field, `an object defining binning parameters
        <https://vega.github.io/vega-lite/docs/bin.html#params>`__, or indicating that the
        data for ``x`` or ``y`` channel are binned before they are imported into Vega-Lite (
        ``"binned"`` ).


        If ``true``, default `binning parameters
        <https://vega.github.io/vega-lite/docs/bin.html>`__ will be applied.

        If ``"binned"``, this indicates that the data for the ``x`` (or ``y`` ) channel are
        already binned. You can map the bin-start field to ``x`` (or ``y`` ) and the bin-end
        field to ``x2`` (or ``y2`` ). The scale and axis will be formatted similar to
        binning in Vega-Lite.  To adjust the axis ticks based on the bin step, you can also
        set the axis's `tickMinStep
        <https://vega.github.io/vega-lite/docs/axis.html#ticks>`__ property.

        **Default value:** ``false``

        **See also:** `bin <https://vega.github.io/vega-lite/docs/bin.html>`__
        documentation.
    condition : anyOf(:class:`ConditionalValueDefTextExprRef`,
    List(:class:`ConditionalValueDefTextExprRef`))
        One or more value definition(s) with `a selection or a test predicate
        <https://vega.github.io/vega-lite/docs/condition.html>`__.

        **Note:** A field definition's ``condition`` property can only contain `conditional
        value definitions <https://vega.github.io/vega-lite/docs/condition.html#value>`__
        since Vega-Lite only allows at most one encoded field per encoding channel.
    field : :class:`Field`
        **Required.** A string defining the name of the field from which to pull a data
        value or an object defining iterated values from the `repeat
        <https://vega.github.io/vega-lite/docs/repeat.html>`__ operator.

        **See also:** `field <https://vega.github.io/vega-lite/docs/field.html>`__
        documentation.

        **Notes:** 1)  Dots ( ``.`` ) and brackets ( ``[`` and ``]`` ) can be used to access
        nested objects (e.g., ``"field": "foo.bar"`` and ``"field": "foo['bar']"`` ). If
        field names contain dots or brackets but are not nested, you can use ``\\`` to
        escape dots and brackets (e.g., ``"a\\.b"`` and ``"a\\[0\\]"`` ). See more details
        about escaping in the `field documentation
        <https://vega.github.io/vega-lite/docs/field.html>`__. 2) ``field`` is not required
        if ``aggregate`` is ``count``.
    format : anyOf(string, :class:`Dictunknown`)
        When used with the default ``"number"`` and ``"time"`` format type, the text
        formatting pattern for labels of guides (axes, legends, headers) and text marks.


        * If the format type is ``"number"`` (e.g., for quantitative fields), this is D3's
          `number format pattern <https://github.com/d3/d3-format#locale_format>`__. - If
          the format type is ``"time"`` (e.g., for temporal fields), this is D3's `time
          format pattern <https://github.com/d3/d3-time-format#locale_format>`__.

        See the `format documentation <https://vega.github.io/vega-lite/docs/format.html>`__
        for more examples.

        When used with a `custom formatType
        <https://vega.github.io/vega-lite/docs/config.html#custom-format-type>`__, this
        value will be passed as ``format`` alongside ``datum.value`` to the registered
        function.

        **Default value:**  Derived from `numberFormat
        <https://vega.github.io/vega-lite/docs/config.html#format>`__ config for number
        format and from `timeFormat
        <https://vega.github.io/vega-lite/docs/config.html#format>`__ config for time
        format.
    formatType : string
        The format type for labels. One of ``"number"``, ``"time"``, or a `registered custom
        format type
        <https://vega.github.io/vega-lite/docs/config.html#custom-format-type>`__.

        **Default value:** - ``"time"`` for temporal fields and ordinal and nominal fields
        with ``timeUnit``. - ``"number"`` for quantitative fields as well as ordinal and
        nominal fields without ``timeUnit``.
    labelExpr : string
        `Vega expression <https://vega.github.io/vega/docs/expressions/>`__ for customizing
        labels text.

        **Note:** The label text and value can be assessed via the ``label`` and ``value``
        properties of the axis's backing ``datum`` object.
    timeUnit : anyOf(:class:`TimeUnit`, :class:`TimeUnitParams`)
        Time unit (e.g., ``year``, ``yearmonth``, ``month``, ``hours`` ) for a temporal
        field. or `a temporal field that gets casted as ordinal
        <https://vega.github.io/vega-lite/docs/type.html#cast>`__.

        **Default value:** ``undefined`` (None)

        **See also:** `timeUnit <https://vega.github.io/vega-lite/docs/timeunit.html>`__
        documentation.
    title : anyOf(:class:`Text`, None)
        A title for the field. If ``null``, the title will be removed.

        **Default value:**  derived from the field's name and transformation function (
        ``aggregate``, ``bin`` and ``timeUnit`` ). If the field has an aggregate function,
        the function is displayed as part of the title (e.g., ``"Sum of Profit"`` ). If the
        field is binned or has a time unit applied, the applied function is shown in
        parentheses (e.g., ``"Profit (binned)"``, ``"Transaction Date (year-month)"`` ).
        Otherwise, the title is simply the field name.

        **Notes** :

        1) You can customize the default field title format by providing the `fieldTitle
        <https://vega.github.io/vega-lite/docs/config.html#top-level-config>`__ property in
        the `config <https://vega.github.io/vega-lite/docs/config.html>`__ or `fieldTitle
        function via the compile function's options
        <https://vega.github.io/vega-lite/docs/compile.html#field-title>`__.

        2) If both field definition's ``title`` and axis, header, or legend ``title`` are
        defined, axis/header/legend title will be used.
    type : :class:`StandardType`
        The type of measurement ( ``"quantitative"``, ``"temporal"``, ``"ordinal"``, or
        ``"nominal"`` ) for the encoded field or constant value ( ``datum`` ). It can also
        be a ``"geojson"`` type for encoding `'geoshape'
        <https://vega.github.io/vega-lite/docs/geoshape.html>`__.

        Vega-Lite automatically infers data types in many cases as discussed below. However,
        type is required for a field if: (1) the field is not nominal and the field encoding
        has no specified ``aggregate`` (except ``argmin`` and ``argmax`` ), ``bin``, scale
        type, custom ``sort`` order, nor ``timeUnit`` or (2) if you wish to use an ordinal
        scale for a field with ``bin`` or ``timeUnit``.

        **Default value:**

        1) For a data ``field``, ``"nominal"`` is the default data type unless the field
        encoding has ``aggregate``, ``channel``, ``bin``, scale type, ``sort``, or
        ``timeUnit`` that satisfies the following criteria: - ``"quantitative"`` is the
        default type if (1) the encoded field contains ``bin`` or ``aggregate`` except
        ``"argmin"`` and ``"argmax"``, (2) the encoding channel is ``latitude`` or
        ``longitude`` channel or (3) if the specified scale type is `a quantitative scale
        <https://vega.github.io/vega-lite/docs/scale.html#type>`__. - ``"temporal"`` is the
        default type if (1) the encoded field contains ``timeUnit`` or (2) the specified
        scale type is a time or utc scale - ``ordinal""`` is the default type if (1) the
        encoded field contains a `custom sort order
        <https://vega.github.io/vega-lite/docs/sort.html#specifying-custom-sort-order>`__,
        (2) the specified scale type is an ordinal/point/band scale, or (3) the encoding
        channel is ``order``.

        2) For a constant value in data domain ( ``datum`` ): - ``"quantitative"`` if the
        datum is a number - ``"nominal"`` if the datum is a string - ``"temporal"`` if the
        datum is `a date time object
        <https://vega.github.io/vega-lite/docs/datetime.html>`__

        **Note:** - Data ``type`` describes the semantics of the data rather than the
        primitive data types (number, string, etc.). The same primitive data type can have
        different types of measurement. For example, numeric data can represent
        quantitative, ordinal, or nominal data. - Data values for a temporal field can be
        either a date-time string (e.g., ``"2015-03-07 12:32:17"``, ``"17:01"``,
        ``"2015-03-16"``. ``"2015"`` ) or a timestamp number (e.g., ``1552199579097`` ). -
        When using with `bin <https://vega.github.io/vega-lite/docs/bin.html>`__, the
        ``type`` property can be either ``"quantitative"`` (for using a linear bin scale) or
        `"ordinal" (for using an ordinal bin scale)
        <https://vega.github.io/vega-lite/docs/type.html#cast-bin>`__. - When using with
        `timeUnit <https://vega.github.io/vega-lite/docs/timeunit.html>`__, the ``type``
        property can be either ``"temporal"`` (default, for using a temporal scale) or
        `"ordinal" (for using an ordinal scale)
        <https://vega.github.io/vega-lite/docs/type.html#cast-bin>`__. - When using with
        `aggregate <https://vega.github.io/vega-lite/docs/aggregate.html>`__, the ``type``
        property refers to the post-aggregation data type. For example, we can calculate
        count ``distinct`` of a categorical field ``"cat"`` using ``{"aggregate":
        "distinct", "field": "cat"}``. The ``"type"`` of the aggregate output is
        ``"quantitative"``. - Secondary channels (e.g., ``x2``, ``y2``, ``xError``,
        ``yError`` ) do not have ``type`` as they must have exactly the same type as their
        primary channels (e.g., ``x``, ``y`` ).

        **See also:** `type <https://vega.github.io/vega-lite/docs/type.html>`__
        documentation.
    """
    _schema = {'$ref': '#/definitions/FieldOrDatumDefWithCondition<StringFieldDef,Text>'}

    def __init__(self, aggregate=Undefined, band=Undefined, bin=Undefined, condition=Undefined,
                 field=Undefined, format=Undefined, formatType=Undefined, labelExpr=Undefined,
                 timeUnit=Undefined, title=Undefined, type=Undefined, **kwds):
        super(FieldOrDatumDefWithConditionStringFieldDefText, self).__init__(aggregate=aggregate,
                                                                             band=band, bin=bin,
                                                                             condition=condition,
                                                                             field=field, format=format,
                                                                             formatType=formatType,
                                                                             labelExpr=labelExpr,
                                                                             timeUnit=timeUnit,
                                                                             title=title, type=type,
                                                                             **kwds)


class TextDirection(VegaLiteSchema):
    """TextDirection schema wrapper

    enum('ltr', 'rtl')
    """
    _schema = {'$ref': '#/definitions/TextDirection'}

    def __init__(self, *args):
        super(TextDirection, self).__init__(*args)


class TickConfig(AnyMarkConfig):
    """TickConfig schema wrapper

    Mapping(required=[])

    Attributes
    ----------

    align : anyOf(:class:`Align`, :class:`ExprRef`)
        The horizontal alignment of the text or ranged marks (area, bar, image, rect, rule).
        One of ``"left"``, ``"right"``, ``"center"``.

        **Note:** Expression reference is *not* supported for range marks.
    angle : anyOf(float, :class:`ExprRef`)

    aria : anyOf(boolean, :class:`ExprRef`)

    ariaRole : anyOf(string, :class:`ExprRef`)

    ariaRoleDescription : anyOf(string, :class:`ExprRef`)

    aspect : anyOf(boolean, :class:`ExprRef`)

    bandSize : float
        The width of the ticks.

        **Default value:**  3/4 of step (width step for horizontal ticks and height step for
        vertical ticks).
    baseline : anyOf(:class:`TextBaseline`, :class:`ExprRef`)
        For text marks, the vertical text baseline. One of ``"alphabetic"`` (default),
        ``"top"``, ``"middle"``, ``"bottom"``, ``"line-top"``, ``"line-bottom"``, or an
        expression reference that provides one of the valid values. The ``"line-top"`` and
        ``"line-bottom"`` values operate similarly to ``"top"`` and ``"bottom"``, but are
        calculated relative to the ``lineHeight`` rather than ``fontSize`` alone.

        For range marks, the vertical alignment of the marks. One of ``"top"``,
        ``"middle"``, ``"bottom"``.

        **Note:** Expression reference is *not* supported for range marks.
    blend : anyOf(:class:`Blend`, :class:`ExprRef`)

    color : anyOf(:class:`Color`, :class:`Gradient`, :class:`ExprRef`)
        Default color.

        **Default value:** :raw-html:`<span style="color: #4682b4;">&#9632;</span>`
        ``"#4682b4"``

        **Note:** - This property cannot be used in a `style config
        <https://vega.github.io/vega-lite/docs/mark.html#style-config>`__. - The ``fill``
        and ``stroke`` properties have higher precedence than ``color`` and will override
        ``color``.
    cornerRadius : anyOf(float, :class:`ExprRef`)

    cornerRadiusBottomLeft : anyOf(float, :class:`ExprRef`)

    cornerRadiusBottomRight : anyOf(float, :class:`ExprRef`)

    cornerRadiusTopLeft : anyOf(float, :class:`ExprRef`)

    cornerRadiusTopRight : anyOf(float, :class:`ExprRef`)

    cursor : anyOf(:class:`Cursor`, :class:`ExprRef`)

    description : anyOf(string, :class:`ExprRef`)

    dir : anyOf(:class:`TextDirection`, :class:`ExprRef`)

    dx : anyOf(float, :class:`ExprRef`)

    dy : anyOf(float, :class:`ExprRef`)

    ellipsis : anyOf(string, :class:`ExprRef`)

    endAngle : anyOf(float, :class:`ExprRef`)

    fill : anyOf(:class:`Color`, :class:`Gradient`, None, :class:`ExprRef`)
        Default fill color. This property has higher precedence than ``config.color``. Set
        to ``null`` to remove fill.

        **Default value:** (None)
    fillOpacity : anyOf(float, :class:`ExprRef`)

    filled : boolean
        Whether the mark's color should be used as fill color instead of stroke color.

        **Default value:** ``false`` for all ``point``, ``line``, and ``rule`` marks as well
        as ``geoshape`` marks for `graticule
        <https://vega.github.io/vega-lite/docs/data.html#graticule>`__ data sources;
        otherwise, ``true``.

        **Note:** This property cannot be used in a `style config
        <https://vega.github.io/vega-lite/docs/mark.html#style-config>`__.
    font : anyOf(string, :class:`ExprRef`)

    fontSize : anyOf(float, :class:`ExprRef`)

    fontStyle : anyOf(:class:`FontStyle`, :class:`ExprRef`)

    fontWeight : anyOf(:class:`FontWeight`, :class:`ExprRef`)

    height : anyOf(float, :class:`ExprRef`)

    href : anyOf(:class:`URI`, :class:`ExprRef`)

    innerRadius : anyOf(float, :class:`ExprRef`)
        The inner radius in pixels of arc marks. ``innerRadius`` is an alias for
        ``radius2``.
    interpolate : anyOf(:class:`Interpolate`, :class:`ExprRef`)

    invalid : enum('filter', None)
        Defines how Vega-Lite should handle marks for invalid values ( ``null`` and ``NaN``
        ). - If set to ``"filter"`` (default), all data items with null values will be
        skipped (for line, trail, and area marks) or filtered (for other marks). - If
        ``null``, all data items are included. In this case, invalid values will be
        interpreted as zeroes.
    limit : anyOf(float, :class:`ExprRef`)

    lineBreak : anyOf(string, :class:`ExprRef`)

    lineHeight : anyOf(float, :class:`ExprRef`)

    opacity : anyOf(float, :class:`ExprRef`)
        The overall opacity (value between [0,1]).

        **Default value:** ``0.7`` for non-aggregate plots with ``point``, ``tick``,
        ``circle``, or ``square`` marks or layered ``bar`` charts and ``1`` otherwise.
    order : anyOf(None, boolean)
        For line and trail marks, this ``order`` property can be set to ``null`` or
        ``false`` to make the lines use the original order in the data sources.
    orient : :class:`Orientation`
        The orientation of a non-stacked bar, tick, area, and line charts. The value is
        either horizontal (default) or vertical. - For bar, rule and tick, this determines
        whether the size of the bar and tick should be applied to x or y dimension. - For
        area, this property determines the orient property of the Vega output. - For line
        and trail marks, this property determines the sort order of the points in the line
        if ``config.sortLineBy`` is not specified. For stacked charts, this is always
        determined by the orientation of the stack; therefore explicitly specified value
        will be ignored.
    outerRadius : anyOf(float, :class:`ExprRef`)
        The outer radius in pixels of arc marks. ``outerRadius`` is an alias for ``radius``.
    padAngle : anyOf(float, :class:`ExprRef`)

    radius : anyOf(float, :class:`ExprRef`)
        For arc mark, the primary (outer) radius in pixels.

        For text marks, polar coordinate radial offset, in pixels, of the text from the
        origin determined by the ``x`` and ``y`` properties.
    radius2 : anyOf(float, :class:`ExprRef`)
        The secondary (inner) radius in pixels of arc marks.
    shape : anyOf(anyOf(:class:`SymbolShape`, string), :class:`ExprRef`)

    size : anyOf(float, :class:`ExprRef`)
        Default size for marks. - For ``point`` / ``circle`` / ``square``, this represents
        the pixel area of the marks. Note that this value sets the area of the symbol; the
        side lengths will increase with the square root of this value. - For ``bar``, this
        represents the band size of the bar, in pixels. - For ``text``, this represents the
        font size, in pixels.

        **Default value:** - ``30`` for point, circle, square marks; width/height's ``step``
        - ``2`` for bar marks with discrete dimensions; - ``5`` for bar marks with
        continuous dimensions; - ``11`` for text marks.
    smooth : anyOf(boolean, :class:`ExprRef`)

    startAngle : anyOf(float, :class:`ExprRef`)

    stroke : anyOf(:class:`Color`, :class:`Gradient`, None, :class:`ExprRef`)
        Default stroke color. This property has higher precedence than ``config.color``. Set
        to ``null`` to remove stroke.

        **Default value:** (None)
    strokeCap : anyOf(:class:`StrokeCap`, :class:`ExprRef`)

    strokeDash : anyOf(List(float), :class:`ExprRef`)

    strokeDashOffset : anyOf(float, :class:`ExprRef`)

    strokeJoin : anyOf(:class:`StrokeJoin`, :class:`ExprRef`)

    strokeMiterLimit : anyOf(float, :class:`ExprRef`)

    strokeOffset : anyOf(float, :class:`ExprRef`)

    strokeOpacity : anyOf(float, :class:`ExprRef`)

    strokeWidth : anyOf(float, :class:`ExprRef`)

    tension : anyOf(float, :class:`ExprRef`)

    text : anyOf(:class:`Text`, :class:`ExprRef`)

    theta : anyOf(float, :class:`ExprRef`)
        For arc marks, the arc length in radians if theta2 is not specified, otherwise the
        start arc angle. (A value of 0 indicates up or “north”, increasing values proceed
        clockwise.)

        For text marks, polar coordinate angle in radians.
    theta2 : anyOf(float, :class:`ExprRef`)
        The end angle of arc marks in radians. A value of 0 indicates up or “north”,
        increasing values proceed clockwise.
    thickness : float
        Thickness of the tick mark.

        **Default value:**  ``1``
    timeUnitBand : float
        Default relative band size for a time unit. If set to ``1``, the bandwidth of the
        marks will be equal to the time unit band step. If set to ``0.5``, bandwidth of the
        marks will be half of the time unit band step.
    timeUnitBandPosition : float
        Default relative band position for a time unit. If set to ``0``, the marks will be
        positioned at the beginning of the time unit band step. If set to ``0.5``, the marks
        will be positioned in the middle of the time unit band step.
    tooltip : anyOf(float, string, boolean, :class:`TooltipContent`, :class:`ExprRef`, None)
        The tooltip text string to show upon mouse hover or an object defining which fields
        should the tooltip be derived from.


        * If ``tooltip`` is ``true`` or ``{"content": "encoding"}``, then all fields from
          ``encoding`` will be used. - If ``tooltip`` is ``{"content": "data"}``, then all
          fields that appear in the highlighted data point will be used. - If set to
          ``null`` or ``false``, then no tooltip will be used.

        See the `tooltip <https://vega.github.io/vega-lite/docs/tooltip.html>`__
        documentation for a detailed discussion about tooltip  in Vega-Lite.

        **Default value:** ``null``
    url : anyOf(:class:`URI`, :class:`ExprRef`)

    width : anyOf(float, :class:`ExprRef`)

    x : anyOf(float, string, :class:`ExprRef`)
        X coordinates of the marks, or width of horizontal ``"bar"`` and ``"area"`` without
        specified ``x2`` or ``width``.

        The ``value`` of this channel can be a number or a string ``"width"`` for the width
        of the plot.
    x2 : anyOf(float, string, :class:`ExprRef`)
        X2 coordinates for ranged ``"area"``, ``"bar"``, ``"rect"``, and  ``"rule"``.

        The ``value`` of this channel can be a number or a string ``"width"`` for the width
        of the plot.
    y : anyOf(float, string, :class:`ExprRef`)
        Y coordinates of the marks, or height of vertical ``"bar"`` and ``"area"`` without
        specified ``y2`` or ``height``.

        The ``value`` of this channel can be a number or a string ``"height"`` for the
        height of the plot.
    y2 : anyOf(float, string, :class:`ExprRef`)
        Y2 coordinates for ranged ``"area"``, ``"bar"``, ``"rect"``, and  ``"rule"``.

        The ``value`` of this channel can be a number or a string ``"height"`` for the
        height of the plot.
    """
    _schema = {'$ref': '#/definitions/TickConfig'}

    def __init__(self, align=Undefined, angle=Undefined, aria=Undefined, ariaRole=Undefined,
                 ariaRoleDescription=Undefined, aspect=Undefined, bandSize=Undefined,
                 baseline=Undefined, blend=Undefined, color=Undefined, cornerRadius=Undefined,
                 cornerRadiusBottomLeft=Undefined, cornerRadiusBottomRight=Undefined,
                 cornerRadiusTopLeft=Undefined, cornerRadiusTopRight=Undefined, cursor=Undefined,
                 description=Undefined, dir=Undefined, dx=Undefined, dy=Undefined, ellipsis=Undefined,
                 endAngle=Undefined, fill=Undefined, fillOpacity=Undefined, filled=Undefined,
                 font=Undefined, fontSize=Undefined, fontStyle=Undefined, fontWeight=Undefined,
                 height=Undefined, href=Undefined, innerRadius=Undefined, interpolate=Undefined,
                 invalid=Undefined, limit=Undefined, lineBreak=Undefined, lineHeight=Undefined,
                 opacity=Undefined, order=Undefined, orient=Undefined, outerRadius=Undefined,
                 padAngle=Undefined, radius=Undefined, radius2=Undefined, shape=Undefined,
                 size=Undefined, smooth=Undefined, startAngle=Undefined, stroke=Undefined,
                 strokeCap=Undefined, strokeDash=Undefined, strokeDashOffset=Undefined,
                 strokeJoin=Undefined, strokeMiterLimit=Undefined, strokeOffset=Undefined,
                 strokeOpacity=Undefined, strokeWidth=Undefined, tension=Undefined, text=Undefined,
                 theta=Undefined, theta2=Undefined, thickness=Undefined, timeUnitBand=Undefined,
                 timeUnitBandPosition=Undefined, tooltip=Undefined, url=Undefined, width=Undefined,
                 x=Undefined, x2=Undefined, y=Undefined, y2=Undefined, **kwds):
        super(TickConfig, self).__init__(align=align, angle=angle, aria=aria, ariaRole=ariaRole,
                                         ariaRoleDescription=ariaRoleDescription, aspect=aspect,
                                         bandSize=bandSize, baseline=baseline, blend=blend, color=color,
                                         cornerRadius=cornerRadius,
                                         cornerRadiusBottomLeft=cornerRadiusBottomLeft,
                                         cornerRadiusBottomRight=cornerRadiusBottomRight,
                                         cornerRadiusTopLeft=cornerRadiusTopLeft,
                                         cornerRadiusTopRight=cornerRadiusTopRight, cursor=cursor,
                                         description=description, dir=dir, dx=dx, dy=dy,
                                         ellipsis=ellipsis, endAngle=endAngle, fill=fill,
                                         fillOpacity=fillOpacity, filled=filled, font=font,
                                         fontSize=fontSize, fontStyle=fontStyle, fontWeight=fontWeight,
                                         height=height, href=href, innerRadius=innerRadius,
                                         interpolate=interpolate, invalid=invalid, limit=limit,
                                         lineBreak=lineBreak, lineHeight=lineHeight, opacity=opacity,
                                         order=order, orient=orient, outerRadius=outerRadius,
                                         padAngle=padAngle, radius=radius, radius2=radius2, shape=shape,
                                         size=size, smooth=smooth, startAngle=startAngle, stroke=stroke,
                                         strokeCap=strokeCap, strokeDash=strokeDash,
                                         strokeDashOffset=strokeDashOffset, strokeJoin=strokeJoin,
                                         strokeMiterLimit=strokeMiterLimit, strokeOffset=strokeOffset,
                                         strokeOpacity=strokeOpacity, strokeWidth=strokeWidth,
                                         tension=tension, text=text, theta=theta, theta2=theta2,
                                         thickness=thickness, timeUnitBand=timeUnitBand,
                                         timeUnitBandPosition=timeUnitBandPosition, tooltip=tooltip,
                                         url=url, width=width, x=x, x2=x2, y=y, y2=y2, **kwds)


class TickCount(VegaLiteSchema):
    """TickCount schema wrapper

    anyOf(float, :class:`TimeInterval`, :class:`TimeIntervalStep`)
    """
    _schema = {'$ref': '#/definitions/TickCount'}

    def __init__(self, *args, **kwds):
        super(TickCount, self).__init__(*args, **kwds)


class TimeInterval(TickCount):
    """TimeInterval schema wrapper

    enum('millisecond', 'second', 'minute', 'hour', 'day', 'week', 'month', 'year')
    """
    _schema = {'$ref': '#/definitions/TimeInterval'}

    def __init__(self, *args):
        super(TimeInterval, self).__init__(*args)


class TimeIntervalStep(TickCount):
    """TimeIntervalStep schema wrapper

    Mapping(required=[interval, step])

    Attributes
    ----------

    interval : :class:`TimeInterval`

    step : float

    """
    _schema = {'$ref': '#/definitions/TimeIntervalStep'}

    def __init__(self, interval=Undefined, step=Undefined, **kwds):
        super(TimeIntervalStep, self).__init__(interval=interval, step=step, **kwds)


class TimeUnit(VegaLiteSchema):
    """TimeUnit schema wrapper

    anyOf(:class:`SingleTimeUnit`, :class:`MultiTimeUnit`)
    """
    _schema = {'$ref': '#/definitions/TimeUnit'}

    def __init__(self, *args, **kwds):
        super(TimeUnit, self).__init__(*args, **kwds)


class MultiTimeUnit(TimeUnit):
    """MultiTimeUnit schema wrapper

    anyOf(:class:`LocalMultiTimeUnit`, :class:`UtcMultiTimeUnit`)
    """
    _schema = {'$ref': '#/definitions/MultiTimeUnit'}

    def __init__(self, *args, **kwds):
        super(MultiTimeUnit, self).__init__(*args, **kwds)


class LocalMultiTimeUnit(MultiTimeUnit):
    """LocalMultiTimeUnit schema wrapper

    enum('yearquarter', 'yearquartermonth', 'yearmonth', 'yearmonthdate', 'yearmonthdatehours',
    'yearmonthdatehoursminutes', 'yearmonthdatehoursminutesseconds', 'yearweek', 'yearweekday',
    'yearweekdayhours', 'yearweekdayhoursminutes', 'yearweekdayhoursminutesseconds',
    'yeardayofyear', 'quartermonth', 'monthdate', 'monthdatehours', 'monthdatehoursminutes',
    'monthdatehoursminutesseconds', 'weekday', 'weeksdayhours', 'weekdayhoursminutes',
    'weekdayhoursminutesseconds', 'dayhours', 'dayhoursminutes', 'dayhoursminutesseconds',
    'hoursminutes', 'hoursminutesseconds', 'minutesseconds', 'secondsmilliseconds')
    """
    _schema = {'$ref': '#/definitions/LocalMultiTimeUnit'}

    def __init__(self, *args):
        super(LocalMultiTimeUnit, self).__init__(*args)


class SingleTimeUnit(TimeUnit):
    """SingleTimeUnit schema wrapper

    anyOf(:class:`LocalSingleTimeUnit`, :class:`UtcSingleTimeUnit`)
    """
    _schema = {'$ref': '#/definitions/SingleTimeUnit'}

    def __init__(self, *args, **kwds):
        super(SingleTimeUnit, self).__init__(*args, **kwds)


class LocalSingleTimeUnit(SingleTimeUnit):
    """LocalSingleTimeUnit schema wrapper

    enum('year', 'quarter', 'month', 'week', 'day', 'dayofyear', 'date', 'hours', 'minutes',
    'seconds', 'milliseconds')
    """
    _schema = {'$ref': '#/definitions/LocalSingleTimeUnit'}

    def __init__(self, *args):
        super(LocalSingleTimeUnit, self).__init__(*args)


class TimeUnitParams(VegaLiteSchema):
    """TimeUnitParams schema wrapper

    Mapping(required=[])

    Attributes
    ----------

    maxbins : float
        If no ``unit`` is specified, maxbins is used to infer time units.
    step : float
        The number of steps between bins, in terms of the least significant unit provided.
    unit : :class:`TimeUnit`
        Defines how date-time values should be binned.
    utc : boolean
        True to use UTC timezone. Equivalent to using a ``utc`` prefixed ``TimeUnit``.
    """
    _schema = {'$ref': '#/definitions/TimeUnitParams'}

    def __init__(self, maxbins=Undefined, step=Undefined, unit=Undefined, utc=Undefined, **kwds):
        super(TimeUnitParams, self).__init__(maxbins=maxbins, step=step, unit=unit, utc=utc, **kwds)


class TitleAnchor(VegaLiteSchema):
    """TitleAnchor schema wrapper

    enum(None, 'start', 'middle', 'end')
    """
    _schema = {'$ref': '#/definitions/TitleAnchor'}

    def __init__(self, *args):
        super(TitleAnchor, self).__init__(*args)


class TitleConfig(VegaLiteSchema):
    """TitleConfig schema wrapper

    Mapping(required=[])

    Attributes
    ----------

    align : :class:`Align`
        Horizontal text alignment for title text. One of ``"left"``, ``"center"``, or
        ``"right"``.
    anchor : anyOf(:class:`TitleAnchor`, :class:`ExprRef`)

    angle : anyOf(float, :class:`ExprRef`)

    aria : anyOf(boolean, :class:`ExprRef`)

    baseline : :class:`TextBaseline`
        Vertical text baseline for title and subtitle text. One of ``"alphabetic"``
        (default), ``"top"``, ``"middle"``, ``"bottom"``, ``"line-top"``, or
        ``"line-bottom"``. The ``"line-top"`` and ``"line-bottom"`` values operate similarly
        to ``"top"`` and ``"bottom"``, but are calculated relative to the *lineHeight*
        rather than *fontSize* alone.
    color : anyOf(anyOf(None, :class:`Color`), :class:`ExprRef`)

    dx : anyOf(float, :class:`ExprRef`)

    dy : anyOf(float, :class:`ExprRef`)

    font : anyOf(string, :class:`ExprRef`)

    fontSize : anyOf(float, :class:`ExprRef`)

    fontStyle : anyOf(:class:`FontStyle`, :class:`ExprRef`)

    fontWeight : anyOf(:class:`FontWeight`, :class:`ExprRef`)

    frame : anyOf(anyOf(:class:`TitleFrame`, string), :class:`ExprRef`)

    limit : anyOf(float, :class:`ExprRef`)

    lineHeight : anyOf(float, :class:`ExprRef`)

    offset : anyOf(float, :class:`ExprRef`)

    orient : anyOf(:class:`TitleOrient`, :class:`ExprRef`)

    subtitleColor : anyOf(anyOf(None, :class:`Color`), :class:`ExprRef`)

    subtitleFont : anyOf(string, :class:`ExprRef`)

    subtitleFontSize : anyOf(float, :class:`ExprRef`)

    subtitleFontStyle : anyOf(:class:`FontStyle`, :class:`ExprRef`)

    subtitleFontWeight : anyOf(:class:`FontWeight`, :class:`ExprRef`)

    subtitleLineHeight : anyOf(float, :class:`ExprRef`)

    subtitlePadding : anyOf(float, :class:`ExprRef`)

    zindex : anyOf(float, :class:`ExprRef`)

    """
    _schema = {'$ref': '#/definitions/TitleConfig'}

    def __init__(self, align=Undefined, anchor=Undefined, angle=Undefined, aria=Undefined,
                 baseline=Undefined, color=Undefined, dx=Undefined, dy=Undefined, font=Undefined,
                 fontSize=Undefined, fontStyle=Undefined, fontWeight=Undefined, frame=Undefined,
                 limit=Undefined, lineHeight=Undefined, offset=Undefined, orient=Undefined,
                 subtitleColor=Undefined, subtitleFont=Undefined, subtitleFontSize=Undefined,
                 subtitleFontStyle=Undefined, subtitleFontWeight=Undefined,
                 subtitleLineHeight=Undefined, subtitlePadding=Undefined, zindex=Undefined, **kwds):
        super(TitleConfig, self).__init__(align=align, anchor=anchor, angle=angle, aria=aria,
                                          baseline=baseline, color=color, dx=dx, dy=dy, font=font,
                                          fontSize=fontSize, fontStyle=fontStyle, fontWeight=fontWeight,
                                          frame=frame, limit=limit, lineHeight=lineHeight,
                                          offset=offset, orient=orient, subtitleColor=subtitleColor,
                                          subtitleFont=subtitleFont, subtitleFontSize=subtitleFontSize,
                                          subtitleFontStyle=subtitleFontStyle,
                                          subtitleFontWeight=subtitleFontWeight,
                                          subtitleLineHeight=subtitleLineHeight,
                                          subtitlePadding=subtitlePadding, zindex=zindex, **kwds)


class TitleFrame(VegaLiteSchema):
    """TitleFrame schema wrapper

    enum('bounds', 'group')
    """
    _schema = {'$ref': '#/definitions/TitleFrame'}

    def __init__(self, *args):
        super(TitleFrame, self).__init__(*args)


class TitleOrient(VegaLiteSchema):
    """TitleOrient schema wrapper

    enum('none', 'left', 'right', 'top', 'bottom')
    """
    _schema = {'$ref': '#/definitions/TitleOrient'}

    def __init__(self, *args):
        super(TitleOrient, self).__init__(*args)


class TitleParams(VegaLiteSchema):
    """TitleParams schema wrapper

    Mapping(required=[text])

    Attributes
    ----------

    text : anyOf(:class:`Text`, :class:`ExprRef`)
        The title text.
    align : :class:`Align`
        Horizontal text alignment for title text. One of ``"left"``, ``"center"``, or
        ``"right"``.
    anchor : :class:`TitleAnchor`
        The anchor position for placing the title. One of ``"start"``, ``"middle"``, or
        ``"end"``. For example, with an orientation of top these anchor positions map to a
        left-, center-, or right-aligned title.

        **Default value:** ``"middle"`` for `single
        <https://vega.github.io/vega-lite/docs/spec.html>`__ and `layered
        <https://vega.github.io/vega-lite/docs/layer.html>`__ views. ``"start"`` for other
        composite views.

        **Note:** `For now <https://github.com/vega/vega-lite/issues/2875>`__, ``anchor`` is
        only customizable only for `single
        <https://vega.github.io/vega-lite/docs/spec.html>`__ and `layered
        <https://vega.github.io/vega-lite/docs/layer.html>`__ views. For other composite
        views, ``anchor`` is always ``"start"``.
    angle : anyOf(float, :class:`ExprRef`)

    aria : anyOf(boolean, :class:`ExprRef`)

    baseline : :class:`TextBaseline`
        Vertical text baseline for title and subtitle text. One of ``"alphabetic"``
        (default), ``"top"``, ``"middle"``, ``"bottom"``, ``"line-top"``, or
        ``"line-bottom"``. The ``"line-top"`` and ``"line-bottom"`` values operate similarly
        to ``"top"`` and ``"bottom"``, but are calculated relative to the *lineHeight*
        rather than *fontSize* alone.
    color : anyOf(anyOf(None, :class:`Color`), :class:`ExprRef`)

    dx : anyOf(float, :class:`ExprRef`)

    dy : anyOf(float, :class:`ExprRef`)

    font : anyOf(string, :class:`ExprRef`)

    fontSize : anyOf(float, :class:`ExprRef`)

    fontStyle : anyOf(:class:`FontStyle`, :class:`ExprRef`)

    fontWeight : anyOf(:class:`FontWeight`, :class:`ExprRef`)

    frame : anyOf(anyOf(:class:`TitleFrame`, string), :class:`ExprRef`)

    limit : anyOf(float, :class:`ExprRef`)

    lineHeight : anyOf(float, :class:`ExprRef`)

    offset : anyOf(float, :class:`ExprRef`)

    orient : anyOf(:class:`TitleOrient`, :class:`ExprRef`)

    style : anyOf(string, List(string))
        A `mark style property <https://vega.github.io/vega-lite/docs/config.html#style>`__
        to apply to the title text mark.

        **Default value:** ``"group-title"``.
    subtitle : :class:`Text`
        The subtitle Text.
    subtitleColor : anyOf(anyOf(None, :class:`Color`), :class:`ExprRef`)

    subtitleFont : anyOf(string, :class:`ExprRef`)

    subtitleFontSize : anyOf(float, :class:`ExprRef`)

    subtitleFontStyle : anyOf(:class:`FontStyle`, :class:`ExprRef`)

    subtitleFontWeight : anyOf(:class:`FontWeight`, :class:`ExprRef`)

    subtitleLineHeight : anyOf(float, :class:`ExprRef`)

    subtitlePadding : anyOf(float, :class:`ExprRef`)

    zindex : float
        The integer z-index indicating the layering of the title group relative to other
        axis, mark and legend groups.

        **Default value:** ``0``.
    """
    _schema = {'$ref': '#/definitions/TitleParams'}

    def __init__(self, text=Undefined, align=Undefined, anchor=Undefined, angle=Undefined,
                 aria=Undefined, baseline=Undefined, color=Undefined, dx=Undefined, dy=Undefined,
                 font=Undefined, fontSize=Undefined, fontStyle=Undefined, fontWeight=Undefined,
                 frame=Undefined, limit=Undefined, lineHeight=Undefined, offset=Undefined,
                 orient=Undefined, style=Undefined, subtitle=Undefined, subtitleColor=Undefined,
                 subtitleFont=Undefined, subtitleFontSize=Undefined, subtitleFontStyle=Undefined,
                 subtitleFontWeight=Undefined, subtitleLineHeight=Undefined, subtitlePadding=Undefined,
                 zindex=Undefined, **kwds):
        super(TitleParams, self).__init__(text=text, align=align, anchor=anchor, angle=angle, aria=aria,
                                          baseline=baseline, color=color, dx=dx, dy=dy, font=font,
                                          fontSize=fontSize, fontStyle=fontStyle, fontWeight=fontWeight,
                                          frame=frame, limit=limit, lineHeight=lineHeight,
                                          offset=offset, orient=orient, style=style, subtitle=subtitle,
                                          subtitleColor=subtitleColor, subtitleFont=subtitleFont,
                                          subtitleFontSize=subtitleFontSize,
                                          subtitleFontStyle=subtitleFontStyle,
                                          subtitleFontWeight=subtitleFontWeight,
                                          subtitleLineHeight=subtitleLineHeight,
                                          subtitlePadding=subtitlePadding, zindex=zindex, **kwds)


class TooltipContent(VegaLiteSchema):
    """TooltipContent schema wrapper

    Mapping(required=[content])

    Attributes
    ----------

    content : enum('encoding', 'data')

    """
    _schema = {'$ref': '#/definitions/TooltipContent'}

    def __init__(self, content=Undefined, **kwds):
        super(TooltipContent, self).__init__(content=content, **kwds)


class TopLevelSpec(VegaLiteSchema):
    """TopLevelSpec schema wrapper

    anyOf(:class:`TopLevelUnitSpec`, :class:`TopLevelFacetSpec`, :class:`TopLevelLayerSpec`,
    :class:`TopLevelRepeatSpec`, :class:`TopLevelNormalizedConcatSpecGenericSpec`,
    :class:`TopLevelNormalizedVConcatSpecGenericSpec`,
    :class:`TopLevelNormalizedHConcatSpecGenericSpec`)
    A Vega-Lite top-level specification. This is the root class for all Vega-Lite
    specifications. (The json schema is generated from this type.)
    """
    _schema = {'$ref': '#/definitions/TopLevelSpec'}

    def __init__(self, *args, **kwds):
        super(TopLevelSpec, self).__init__(*args, **kwds)


class TopLevelFacetSpec(TopLevelSpec):
    """TopLevelFacetSpec schema wrapper

    Mapping(required=[data, facet, spec])

    Attributes
    ----------

    data : anyOf(:class:`Data`, None)
        An object describing the data source. Set to ``null`` to ignore the parent's data
        source. If no data is set, it is derived from the parent.
    facet : anyOf(:class:`FacetFieldDef`, :class:`FacetMapping`)
        Definition for how to facet the data. One of: 1) `a field definition for faceting
        the plot by one field
        <https://vega.github.io/vega-lite/docs/facet.html#field-def>`__ 2) `An object that
        maps row and column channels to their field definitions
        <https://vega.github.io/vega-lite/docs/facet.html#mapping>`__
    spec : anyOf(:class:`LayerSpec`, :class:`UnitSpecWithFrame`)
        A specification of the view that gets faceted.
    align : anyOf(:class:`LayoutAlign`, :class:`RowColLayoutAlign`)
        The alignment to apply to grid rows and columns. The supported string values are
        ``"all"``, ``"each"``, and ``"none"``.


        * For ``"none"``, a flow layout will be used, in which adjacent subviews are simply
          placed one after the other. - For ``"each"``, subviews will be aligned into a
          clean grid structure, but each row or column may be of variable size. - For
          ``"all"``, subviews will be aligned and each row or column will be sized
          identically based on the maximum observed size. String values for this property
          will be applied to both grid rows and columns.

        Alternatively, an object value of the form ``{"row": string, "column": string}`` can
        be used to supply different alignments for rows and columns.

        **Default value:** ``"all"``.
    autosize : anyOf(:class:`AutosizeType`, :class:`AutoSizeParams`)
        How the visualization size should be determined. If a string, should be one of
        ``"pad"``, ``"fit"`` or ``"none"``. Object values can additionally specify
        parameters for content sizing and automatic resizing.

        **Default value** : ``pad``
    background : anyOf(:class:`Color`, :class:`ExprRef`)
        CSS color property to use as the background of the entire view.

        **Default value:** ``"white"``
    bounds : enum('full', 'flush')
        The bounds calculation method to use for determining the extent of a sub-plot. One
        of ``full`` (the default) or ``flush``.


        * If set to ``full``, the entire calculated bounds (including axes, title, and
          legend) will be used. - If set to ``flush``, only the specified width and height
          values for the sub-view will be used. The ``flush`` setting can be useful when
          attempting to place sub-plots without axes or legends into a uniform grid
          structure.

        **Default value:** ``"full"``
    center : anyOf(boolean, :class:`RowColboolean`)
        Boolean flag indicating if subviews should be centered relative to their respective
        rows or columns.

        An object value of the form ``{"row": boolean, "column": boolean}`` can be used to
        supply different centering values for rows and columns.

        **Default value:** ``false``
    columns : float
        The number of columns to include in the view composition layout.

        **Default value** : ``undefined`` -- An infinite number of columns (a single row)
        will be assumed. This is equivalent to ``hconcat`` (for ``concat`` ) and to using
        the ``column`` channel (for ``facet`` and ``repeat`` ).

        **Note** :

        1) This property is only for: - the general (wrappable) ``concat`` operator (not
        ``hconcat`` / ``vconcat`` ) - the ``facet`` and ``repeat`` operator with one
        field/repetition definition (without row/column nesting)

        2) Setting the ``columns`` to ``1`` is equivalent to ``vconcat`` (for ``concat`` )
        and to using the ``row`` channel (for ``facet`` and ``repeat`` ).
    config : :class:`Config`
        Vega-Lite configuration object. This property can only be defined at the top-level
        of a specification.
    datasets : :class:`Datasets`
        A global data store for named datasets. This is a mapping from names to inline
        datasets. This can be an array of objects or primitive values or a string. Arrays of
        primitive values are ingested as objects with a ``data`` property.
    description : string
        Description of this mark for commenting purpose.
    name : string
        Name of the visualization for later reference.
    padding : anyOf(:class:`Padding`, :class:`ExprRef`)
        The default visualization padding, in pixels, from the edge of the visualization
        canvas to the data rectangle. If a number, specifies padding for all sides. If an
        object, the value should have the format ``{"left": 5, "top": 5, "right": 5,
        "bottom": 5}`` to specify padding for each side of the visualization.

        **Default value** : ``5``
    params : List(:class:`Parameter`)
        Dynamic variables that parameterize a visualization.
    resolve : :class:`Resolve`
        Scale, axis, and legend resolutions for view composition specifications.
    spacing : anyOf(float, :class:`RowColnumber`)
        The spacing in pixels between sub-views of the composition operator. An object of
        the form ``{"row": number, "column": number}`` can be used to set different spacing
        values for rows and columns.

        **Default value** : Depends on ``"spacing"`` property of `the view composition
        configuration <https://vega.github.io/vega-lite/docs/config.html#view-config>`__ (
        ``20`` by default)
    title : anyOf(:class:`Text`, :class:`TitleParams`)
        Title for the plot.
    transform : List(:class:`Transform`)
        An array of data transformations such as filter and new field calculation.
    usermeta : :class:`Dictunknown`
        Optional metadata that will be passed to Vega. This object is completely ignored by
        Vega and Vega-Lite and can be used for custom metadata.
    $schema : string
        URL to `JSON schema <http://json-schema.org/>`__ for a Vega-Lite specification.
        Unless you have a reason to change this, use
        ``https://vega.github.io/schema/vega-lite/v4.json``. Setting the ``$schema``
        property allows automatic validation and autocomplete in editors that support JSON
        schema.
    """
    _schema = {'$ref': '#/definitions/TopLevelFacetSpec'}

    def __init__(self, data=Undefined, facet=Undefined, spec=Undefined, align=Undefined,
                 autosize=Undefined, background=Undefined, bounds=Undefined, center=Undefined,
                 columns=Undefined, config=Undefined, datasets=Undefined, description=Undefined,
                 name=Undefined, padding=Undefined, params=Undefined, resolve=Undefined,
                 spacing=Undefined, title=Undefined, transform=Undefined, usermeta=Undefined, **kwds):
        super(TopLevelFacetSpec, self).__init__(data=data, facet=facet, spec=spec, align=align,
                                                autosize=autosize, background=background, bounds=bounds,
                                                center=center, columns=columns, config=config,
                                                datasets=datasets, description=description, name=name,
                                                padding=padding, params=params, resolve=resolve,
                                                spacing=spacing, title=title, transform=transform,
                                                usermeta=usermeta, **kwds)


class TopLevelLayerSpec(TopLevelSpec):
    """TopLevelLayerSpec schema wrapper

    Mapping(required=[layer])

    Attributes
    ----------

    layer : List(anyOf(:class:`LayerSpec`, :class:`UnitSpec`))
        Layer or single view specifications to be layered.

        **Note** : Specifications inside ``layer`` cannot use ``row`` and ``column``
        channels as layering facet specifications is not allowed. Instead, use the `facet
        operator <https://vega.github.io/vega-lite/docs/facet.html>`__ and place a layer
        inside a facet.
    autosize : anyOf(:class:`AutosizeType`, :class:`AutoSizeParams`)
        How the visualization size should be determined. If a string, should be one of
        ``"pad"``, ``"fit"`` or ``"none"``. Object values can additionally specify
        parameters for content sizing and automatic resizing.

        **Default value** : ``pad``
    background : anyOf(:class:`Color`, :class:`ExprRef`)
        CSS color property to use as the background of the entire view.

        **Default value:** ``"white"``
    config : :class:`Config`
        Vega-Lite configuration object. This property can only be defined at the top-level
        of a specification.
    data : anyOf(:class:`Data`, None)
        An object describing the data source. Set to ``null`` to ignore the parent's data
        source. If no data is set, it is derived from the parent.
    datasets : :class:`Datasets`
        A global data store for named datasets. This is a mapping from names to inline
        datasets. This can be an array of objects or primitive values or a string. Arrays of
        primitive values are ingested as objects with a ``data`` property.
    description : string
        Description of this mark for commenting purpose.
    encoding : :class:`SharedEncoding`
        A shared key-value mapping between encoding channels and definition of fields in the
        underlying layers.
    height : anyOf(float, string, :class:`Step`)
        The height of a visualization.


        * For a plot with a continuous y-field, height should be a number. - For a plot with
          either a discrete y-field or no y-field, height can be either a number indicating
          a fixed height or an object in the form of ``{step: number}`` defining the height
          per discrete step. (No y-field is equivalent to having one discrete step.) - To
          enable responsive sizing on height, it should be set to ``"container"``.

        **Default value:** Based on ``config.view.continuousHeight`` for a plot with a
        continuous y-field and ``config.view.discreteHeight`` otherwise.

        **Note:** For plots with `row and column channels
        <https://vega.github.io/vega-lite/docs/encoding.html#facet>`__, this represents the
        height of a single view and the ``"container"`` option cannot be used.

        **See also:** `height <https://vega.github.io/vega-lite/docs/size.html>`__
        documentation.
    name : string
        Name of the visualization for later reference.
    padding : anyOf(:class:`Padding`, :class:`ExprRef`)
        The default visualization padding, in pixels, from the edge of the visualization
        canvas to the data rectangle. If a number, specifies padding for all sides. If an
        object, the value should have the format ``{"left": 5, "top": 5, "right": 5,
        "bottom": 5}`` to specify padding for each side of the visualization.

        **Default value** : ``5``
    params : List(:class:`Parameter`)
        Dynamic variables that parameterize a visualization.
    projection : :class:`Projection`
        An object defining properties of the geographic projection shared by underlying
        layers.
    resolve : :class:`Resolve`
        Scale, axis, and legend resolutions for view composition specifications.
    title : anyOf(:class:`Text`, :class:`TitleParams`)
        Title for the plot.
    transform : List(:class:`Transform`)
        An array of data transformations such as filter and new field calculation.
    usermeta : :class:`Dictunknown`
        Optional metadata that will be passed to Vega. This object is completely ignored by
        Vega and Vega-Lite and can be used for custom metadata.
    view : :class:`ViewBackground`
        An object defining the view background's fill and stroke.

        **Default value:** none (transparent)
    width : anyOf(float, string, :class:`Step`)
        The width of a visualization.


        * For a plot with a continuous x-field, width should be a number. - For a plot with
          either a discrete x-field or no x-field, width can be either a number indicating a
          fixed width or an object in the form of ``{step: number}`` defining the width per
          discrete step. (No x-field is equivalent to having one discrete step.) - To enable
          responsive sizing on width, it should be set to ``"container"``.

        **Default value:** Based on ``config.view.continuousWidth`` for a plot with a
        continuous x-field and ``config.view.discreteWidth`` otherwise.

        **Note:** For plots with `row and column channels
        <https://vega.github.io/vega-lite/docs/encoding.html#facet>`__, this represents the
        width of a single view and the ``"container"`` option cannot be used.

        **See also:** `width <https://vega.github.io/vega-lite/docs/size.html>`__
        documentation.
    $schema : string
        URL to `JSON schema <http://json-schema.org/>`__ for a Vega-Lite specification.
        Unless you have a reason to change this, use
        ``https://vega.github.io/schema/vega-lite/v4.json``. Setting the ``$schema``
        property allows automatic validation and autocomplete in editors that support JSON
        schema.
    """
    _schema = {'$ref': '#/definitions/TopLevelLayerSpec'}

    def __init__(self, layer=Undefined, autosize=Undefined, background=Undefined, config=Undefined,
                 data=Undefined, datasets=Undefined, description=Undefined, encoding=Undefined,
                 height=Undefined, name=Undefined, padding=Undefined, params=Undefined,
                 projection=Undefined, resolve=Undefined, title=Undefined, transform=Undefined,
                 usermeta=Undefined, view=Undefined, width=Undefined, **kwds):
        super(TopLevelLayerSpec, self).__init__(layer=layer, autosize=autosize, background=background,
                                                config=config, data=data, datasets=datasets,
                                                description=description, encoding=encoding,
                                                height=height, name=name, padding=padding,
                                                params=params, projection=projection, resolve=resolve,
                                                title=title, transform=transform, usermeta=usermeta,
                                                view=view, width=width, **kwds)


class TopLevelNormalizedConcatSpecGenericSpec(TopLevelSpec):
    """TopLevelNormalizedConcatSpecGenericSpec schema wrapper

    Mapping(required=[concat])

    Attributes
    ----------

    concat : List(:class:`NormalizedSpec`)
        A list of views to be concatenated.
    align : anyOf(:class:`LayoutAlign`, :class:`RowColLayoutAlign`)
        The alignment to apply to grid rows and columns. The supported string values are
        ``"all"``, ``"each"``, and ``"none"``.


        * For ``"none"``, a flow layout will be used, in which adjacent subviews are simply
          placed one after the other. - For ``"each"``, subviews will be aligned into a
          clean grid structure, but each row or column may be of variable size. - For
          ``"all"``, subviews will be aligned and each row or column will be sized
          identically based on the maximum observed size. String values for this property
          will be applied to both grid rows and columns.

        Alternatively, an object value of the form ``{"row": string, "column": string}`` can
        be used to supply different alignments for rows and columns.

        **Default value:** ``"all"``.
    autosize : anyOf(:class:`AutosizeType`, :class:`AutoSizeParams`)
        How the visualization size should be determined. If a string, should be one of
        ``"pad"``, ``"fit"`` or ``"none"``. Object values can additionally specify
        parameters for content sizing and automatic resizing.

        **Default value** : ``pad``
    background : anyOf(:class:`Color`, :class:`ExprRef`)
        CSS color property to use as the background of the entire view.

        **Default value:** ``"white"``
    bounds : enum('full', 'flush')
        The bounds calculation method to use for determining the extent of a sub-plot. One
        of ``full`` (the default) or ``flush``.


        * If set to ``full``, the entire calculated bounds (including axes, title, and
          legend) will be used. - If set to ``flush``, only the specified width and height
          values for the sub-view will be used. The ``flush`` setting can be useful when
          attempting to place sub-plots without axes or legends into a uniform grid
          structure.

        **Default value:** ``"full"``
    center : anyOf(boolean, :class:`RowColboolean`)
        Boolean flag indicating if subviews should be centered relative to their respective
        rows or columns.

        An object value of the form ``{"row": boolean, "column": boolean}`` can be used to
        supply different centering values for rows and columns.

        **Default value:** ``false``
    columns : float
        The number of columns to include in the view composition layout.

        **Default value** : ``undefined`` -- An infinite number of columns (a single row)
        will be assumed. This is equivalent to ``hconcat`` (for ``concat`` ) and to using
        the ``column`` channel (for ``facet`` and ``repeat`` ).

        **Note** :

        1) This property is only for: - the general (wrappable) ``concat`` operator (not
        ``hconcat`` / ``vconcat`` ) - the ``facet`` and ``repeat`` operator with one
        field/repetition definition (without row/column nesting)

        2) Setting the ``columns`` to ``1`` is equivalent to ``vconcat`` (for ``concat`` )
        and to using the ``row`` channel (for ``facet`` and ``repeat`` ).
    config : :class:`Config`
        Vega-Lite configuration object. This property can only be defined at the top-level
        of a specification.
    data : anyOf(:class:`Data`, None)
        An object describing the data source. Set to ``null`` to ignore the parent's data
        source. If no data is set, it is derived from the parent.
    datasets : :class:`Datasets`
        A global data store for named datasets. This is a mapping from names to inline
        datasets. This can be an array of objects or primitive values or a string. Arrays of
        primitive values are ingested as objects with a ``data`` property.
    description : string
        Description of this mark for commenting purpose.
    name : string
        Name of the visualization for later reference.
    padding : anyOf(:class:`Padding`, :class:`ExprRef`)
        The default visualization padding, in pixels, from the edge of the visualization
        canvas to the data rectangle. If a number, specifies padding for all sides. If an
        object, the value should have the format ``{"left": 5, "top": 5, "right": 5,
        "bottom": 5}`` to specify padding for each side of the visualization.

        **Default value** : ``5``
    params : List(:class:`Parameter`)
        Dynamic variables that parameterize a visualization.
    resolve : :class:`Resolve`
        Scale, axis, and legend resolutions for view composition specifications.
    spacing : anyOf(float, :class:`RowColnumber`)
        The spacing in pixels between sub-views of the composition operator. An object of
        the form ``{"row": number, "column": number}`` can be used to set different spacing
        values for rows and columns.

        **Default value** : Depends on ``"spacing"`` property of `the view composition
        configuration <https://vega.github.io/vega-lite/docs/config.html#view-config>`__ (
        ``20`` by default)
    title : anyOf(:class:`Text`, :class:`TitleParams`)
        Title for the plot.
    transform : List(:class:`Transform`)
        An array of data transformations such as filter and new field calculation.
    usermeta : :class:`Dictunknown`
        Optional metadata that will be passed to Vega. This object is completely ignored by
        Vega and Vega-Lite and can be used for custom metadata.
    $schema : string
        URL to `JSON schema <http://json-schema.org/>`__ for a Vega-Lite specification.
        Unless you have a reason to change this, use
        ``https://vega.github.io/schema/vega-lite/v4.json``. Setting the ``$schema``
        property allows automatic validation and autocomplete in editors that support JSON
        schema.
    """
    _schema = {'$ref': '#/definitions/TopLevelNormalizedConcatSpec<GenericSpec>'}

    def __init__(self, concat=Undefined, align=Undefined, autosize=Undefined, background=Undefined,
                 bounds=Undefined, center=Undefined, columns=Undefined, config=Undefined,
                 data=Undefined, datasets=Undefined, description=Undefined, name=Undefined,
                 padding=Undefined, params=Undefined, resolve=Undefined, spacing=Undefined,
                 title=Undefined, transform=Undefined, usermeta=Undefined, **kwds):
        super(TopLevelNormalizedConcatSpecGenericSpec, self).__init__(concat=concat, align=align,
                                                                      autosize=autosize,
                                                                      background=background,
                                                                      bounds=bounds, center=center,
                                                                      columns=columns, config=config,
                                                                      data=data, datasets=datasets,
                                                                      description=description,
                                                                      name=name, padding=padding,
                                                                      params=params, resolve=resolve,
                                                                      spacing=spacing, title=title,
                                                                      transform=transform,
                                                                      usermeta=usermeta, **kwds)


class TopLevelNormalizedHConcatSpecGenericSpec(TopLevelSpec):
    """TopLevelNormalizedHConcatSpecGenericSpec schema wrapper

    Mapping(required=[hconcat])

    Attributes
    ----------

    hconcat : List(:class:`NormalizedSpec`)
        A list of views to be concatenated and put into a row.
    autosize : anyOf(:class:`AutosizeType`, :class:`AutoSizeParams`)
        How the visualization size should be determined. If a string, should be one of
        ``"pad"``, ``"fit"`` or ``"none"``. Object values can additionally specify
        parameters for content sizing and automatic resizing.

        **Default value** : ``pad``
    background : anyOf(:class:`Color`, :class:`ExprRef`)
        CSS color property to use as the background of the entire view.

        **Default value:** ``"white"``
    bounds : enum('full', 'flush')
        The bounds calculation method to use for determining the extent of a sub-plot. One
        of ``full`` (the default) or ``flush``.


        * If set to ``full``, the entire calculated bounds (including axes, title, and
          legend) will be used. - If set to ``flush``, only the specified width and height
          values for the sub-view will be used. The ``flush`` setting can be useful when
          attempting to place sub-plots without axes or legends into a uniform grid
          structure.

        **Default value:** ``"full"``
    center : boolean
        Boolean flag indicating if subviews should be centered relative to their respective
        rows or columns.

        **Default value:** ``false``
    config : :class:`Config`
        Vega-Lite configuration object. This property can only be defined at the top-level
        of a specification.
    data : anyOf(:class:`Data`, None)
        An object describing the data source. Set to ``null`` to ignore the parent's data
        source. If no data is set, it is derived from the parent.
    datasets : :class:`Datasets`
        A global data store for named datasets. This is a mapping from names to inline
        datasets. This can be an array of objects or primitive values or a string. Arrays of
        primitive values are ingested as objects with a ``data`` property.
    description : string
        Description of this mark for commenting purpose.
    name : string
        Name of the visualization for later reference.
    padding : anyOf(:class:`Padding`, :class:`ExprRef`)
        The default visualization padding, in pixels, from the edge of the visualization
        canvas to the data rectangle. If a number, specifies padding for all sides. If an
        object, the value should have the format ``{"left": 5, "top": 5, "right": 5,
        "bottom": 5}`` to specify padding for each side of the visualization.

        **Default value** : ``5``
    params : List(:class:`Parameter`)
        Dynamic variables that parameterize a visualization.
    resolve : :class:`Resolve`
        Scale, axis, and legend resolutions for view composition specifications.
    spacing : float
        The spacing in pixels between sub-views of the concat operator.

        **Default value** : ``10``
    title : anyOf(:class:`Text`, :class:`TitleParams`)
        Title for the plot.
    transform : List(:class:`Transform`)
        An array of data transformations such as filter and new field calculation.
    usermeta : :class:`Dictunknown`
        Optional metadata that will be passed to Vega. This object is completely ignored by
        Vega and Vega-Lite and can be used for custom metadata.
    $schema : string
        URL to `JSON schema <http://json-schema.org/>`__ for a Vega-Lite specification.
        Unless you have a reason to change this, use
        ``https://vega.github.io/schema/vega-lite/v4.json``. Setting the ``$schema``
        property allows automatic validation and autocomplete in editors that support JSON
        schema.
    """
    _schema = {'$ref': '#/definitions/TopLevelNormalizedHConcatSpec<GenericSpec>'}

    def __init__(self, hconcat=Undefined, autosize=Undefined, background=Undefined, bounds=Undefined,
                 center=Undefined, config=Undefined, data=Undefined, datasets=Undefined,
                 description=Undefined, name=Undefined, padding=Undefined, params=Undefined,
                 resolve=Undefined, spacing=Undefined, title=Undefined, transform=Undefined,
                 usermeta=Undefined, **kwds):
        super(TopLevelNormalizedHConcatSpecGenericSpec, self).__init__(hconcat=hconcat,
                                                                       autosize=autosize,
                                                                       background=background,
                                                                       bounds=bounds, center=center,
                                                                       config=config, data=data,
                                                                       datasets=datasets,
                                                                       description=description,
                                                                       name=name, padding=padding,
                                                                       params=params, resolve=resolve,
                                                                       spacing=spacing, title=title,
                                                                       transform=transform,
                                                                       usermeta=usermeta, **kwds)


class TopLevelNormalizedVConcatSpecGenericSpec(TopLevelSpec):
    """TopLevelNormalizedVConcatSpecGenericSpec schema wrapper

    Mapping(required=[vconcat])

    Attributes
    ----------

    vconcat : List(:class:`NormalizedSpec`)
        A list of views to be concatenated and put into a column.
    autosize : anyOf(:class:`AutosizeType`, :class:`AutoSizeParams`)
        How the visualization size should be determined. If a string, should be one of
        ``"pad"``, ``"fit"`` or ``"none"``. Object values can additionally specify
        parameters for content sizing and automatic resizing.

        **Default value** : ``pad``
    background : anyOf(:class:`Color`, :class:`ExprRef`)
        CSS color property to use as the background of the entire view.

        **Default value:** ``"white"``
    bounds : enum('full', 'flush')
        The bounds calculation method to use for determining the extent of a sub-plot. One
        of ``full`` (the default) or ``flush``.


        * If set to ``full``, the entire calculated bounds (including axes, title, and
          legend) will be used. - If set to ``flush``, only the specified width and height
          values for the sub-view will be used. The ``flush`` setting can be useful when
          attempting to place sub-plots without axes or legends into a uniform grid
          structure.

        **Default value:** ``"full"``
    center : boolean
        Boolean flag indicating if subviews should be centered relative to their respective
        rows or columns.

        **Default value:** ``false``
    config : :class:`Config`
        Vega-Lite configuration object. This property can only be defined at the top-level
        of a specification.
    data : anyOf(:class:`Data`, None)
        An object describing the data source. Set to ``null`` to ignore the parent's data
        source. If no data is set, it is derived from the parent.
    datasets : :class:`Datasets`
        A global data store for named datasets. This is a mapping from names to inline
        datasets. This can be an array of objects or primitive values or a string. Arrays of
        primitive values are ingested as objects with a ``data`` property.
    description : string
        Description of this mark for commenting purpose.
    name : string
        Name of the visualization for later reference.
    padding : anyOf(:class:`Padding`, :class:`ExprRef`)
        The default visualization padding, in pixels, from the edge of the visualization
        canvas to the data rectangle. If a number, specifies padding for all sides. If an
        object, the value should have the format ``{"left": 5, "top": 5, "right": 5,
        "bottom": 5}`` to specify padding for each side of the visualization.

        **Default value** : ``5``
    params : List(:class:`Parameter`)
        Dynamic variables that parameterize a visualization.
    resolve : :class:`Resolve`
        Scale, axis, and legend resolutions for view composition specifications.
    spacing : float
        The spacing in pixels between sub-views of the concat operator.

        **Default value** : ``10``
    title : anyOf(:class:`Text`, :class:`TitleParams`)
        Title for the plot.
    transform : List(:class:`Transform`)
        An array of data transformations such as filter and new field calculation.
    usermeta : :class:`Dictunknown`
        Optional metadata that will be passed to Vega. This object is completely ignored by
        Vega and Vega-Lite and can be used for custom metadata.
    $schema : string
        URL to `JSON schema <http://json-schema.org/>`__ for a Vega-Lite specification.
        Unless you have a reason to change this, use
        ``https://vega.github.io/schema/vega-lite/v4.json``. Setting the ``$schema``
        property allows automatic validation and autocomplete in editors that support JSON
        schema.
    """
    _schema = {'$ref': '#/definitions/TopLevelNormalizedVConcatSpec<GenericSpec>'}

    def __init__(self, vconcat=Undefined, autosize=Undefined, background=Undefined, bounds=Undefined,
                 center=Undefined, config=Undefined, data=Undefined, datasets=Undefined,
                 description=Undefined, name=Undefined, padding=Undefined, params=Undefined,
                 resolve=Undefined, spacing=Undefined, title=Undefined, transform=Undefined,
                 usermeta=Undefined, **kwds):
        super(TopLevelNormalizedVConcatSpecGenericSpec, self).__init__(vconcat=vconcat,
                                                                       autosize=autosize,
                                                                       background=background,
                                                                       bounds=bounds, center=center,
                                                                       config=config, data=data,
                                                                       datasets=datasets,
                                                                       description=description,
                                                                       name=name, padding=padding,
                                                                       params=params, resolve=resolve,
                                                                       spacing=spacing, title=title,
                                                                       transform=transform,
                                                                       usermeta=usermeta, **kwds)


class TopLevelRepeatSpec(TopLevelSpec):
    """TopLevelRepeatSpec schema wrapper

    anyOf(Mapping(required=[repeat, spec]), Mapping(required=[repeat, spec]))
    """
    _schema = {'$ref': '#/definitions/TopLevelRepeatSpec'}

    def __init__(self, *args, **kwds):
        super(TopLevelRepeatSpec, self).__init__(*args, **kwds)


class TopLevelUnitSpec(TopLevelSpec):
    """TopLevelUnitSpec schema wrapper

    Mapping(required=[data, mark])

    Attributes
    ----------

    data : anyOf(:class:`Data`, None)
        An object describing the data source. Set to ``null`` to ignore the parent's data
        source. If no data is set, it is derived from the parent.
    mark : :class:`AnyMark`
        A string describing the mark type (one of ``"bar"``, ``"circle"``, ``"square"``,
        ``"tick"``, ``"line"``, ``"area"``, ``"point"``, ``"rule"``, ``"geoshape"``, and
        ``"text"`` ) or a `mark definition object
        <https://vega.github.io/vega-lite/docs/mark.html#mark-def>`__.
    align : anyOf(:class:`LayoutAlign`, :class:`RowColLayoutAlign`)
        The alignment to apply to grid rows and columns. The supported string values are
        ``"all"``, ``"each"``, and ``"none"``.


        * For ``"none"``, a flow layout will be used, in which adjacent subviews are simply
          placed one after the other. - For ``"each"``, subviews will be aligned into a
          clean grid structure, but each row or column may be of variable size. - For
          ``"all"``, subviews will be aligned and each row or column will be sized
          identically based on the maximum observed size. String values for this property
          will be applied to both grid rows and columns.

        Alternatively, an object value of the form ``{"row": string, "column": string}`` can
        be used to supply different alignments for rows and columns.

        **Default value:** ``"all"``.
    autosize : anyOf(:class:`AutosizeType`, :class:`AutoSizeParams`)
        How the visualization size should be determined. If a string, should be one of
        ``"pad"``, ``"fit"`` or ``"none"``. Object values can additionally specify
        parameters for content sizing and automatic resizing.

        **Default value** : ``pad``
    background : anyOf(:class:`Color`, :class:`ExprRef`)
        CSS color property to use as the background of the entire view.

        **Default value:** ``"white"``
    bounds : enum('full', 'flush')
        The bounds calculation method to use for determining the extent of a sub-plot. One
        of ``full`` (the default) or ``flush``.


        * If set to ``full``, the entire calculated bounds (including axes, title, and
          legend) will be used. - If set to ``flush``, only the specified width and height
          values for the sub-view will be used. The ``flush`` setting can be useful when
          attempting to place sub-plots without axes or legends into a uniform grid
          structure.

        **Default value:** ``"full"``
    center : anyOf(boolean, :class:`RowColboolean`)
        Boolean flag indicating if subviews should be centered relative to their respective
        rows or columns.

        An object value of the form ``{"row": boolean, "column": boolean}`` can be used to
        supply different centering values for rows and columns.

        **Default value:** ``false``
    config : :class:`Config`
        Vega-Lite configuration object. This property can only be defined at the top-level
        of a specification.
    datasets : :class:`Datasets`
        A global data store for named datasets. This is a mapping from names to inline
        datasets. This can be an array of objects or primitive values or a string. Arrays of
        primitive values are ingested as objects with a ``data`` property.
    description : string
        Description of this mark for commenting purpose.
    encoding : :class:`FacetedEncoding`
        A key-value mapping between encoding channels and definition of fields.
    height : anyOf(float, string, :class:`Step`)
        The height of a visualization.


        * For a plot with a continuous y-field, height should be a number. - For a plot with
          either a discrete y-field or no y-field, height can be either a number indicating
          a fixed height or an object in the form of ``{step: number}`` defining the height
          per discrete step. (No y-field is equivalent to having one discrete step.) - To
          enable responsive sizing on height, it should be set to ``"container"``.

        **Default value:** Based on ``config.view.continuousHeight`` for a plot with a
        continuous y-field and ``config.view.discreteHeight`` otherwise.

        **Note:** For plots with `row and column channels
        <https://vega.github.io/vega-lite/docs/encoding.html#facet>`__, this represents the
        height of a single view and the ``"container"`` option cannot be used.

        **See also:** `height <https://vega.github.io/vega-lite/docs/size.html>`__
        documentation.
    name : string
        Name of the visualization for later reference.
    padding : anyOf(:class:`Padding`, :class:`ExprRef`)
        The default visualization padding, in pixels, from the edge of the visualization
        canvas to the data rectangle. If a number, specifies padding for all sides. If an
        object, the value should have the format ``{"left": 5, "top": 5, "right": 5,
        "bottom": 5}`` to specify padding for each side of the visualization.

        **Default value** : ``5``
    params : List(:class:`Parameter`)
        Dynamic variables that parameterize a visualization.
    projection : :class:`Projection`
        An object defining properties of geographic projection, which will be applied to
        ``shape`` path for ``"geoshape"`` marks and to ``latitude`` and ``"longitude"``
        channels for other marks.
    resolve : :class:`Resolve`
        Scale, axis, and legend resolutions for view composition specifications.
    selection : Mapping(required=[])
        A key-value mapping between selection names and definitions.
    spacing : anyOf(float, :class:`RowColnumber`)
        The spacing in pixels between sub-views of the composition operator. An object of
        the form ``{"row": number, "column": number}`` can be used to set different spacing
        values for rows and columns.

        **Default value** : Depends on ``"spacing"`` property of `the view composition
        configuration <https://vega.github.io/vega-lite/docs/config.html#view-config>`__ (
        ``20`` by default)
    title : anyOf(:class:`Text`, :class:`TitleParams`)
        Title for the plot.
    transform : List(:class:`Transform`)
        An array of data transformations such as filter and new field calculation.
    usermeta : :class:`Dictunknown`
        Optional metadata that will be passed to Vega. This object is completely ignored by
        Vega and Vega-Lite and can be used for custom metadata.
    view : :class:`ViewBackground`
        An object defining the view background's fill and stroke.

        **Default value:** none (transparent)
    width : anyOf(float, string, :class:`Step`)
        The width of a visualization.


        * For a plot with a continuous x-field, width should be a number. - For a plot with
          either a discrete x-field or no x-field, width can be either a number indicating a
          fixed width or an object in the form of ``{step: number}`` defining the width per
          discrete step. (No x-field is equivalent to having one discrete step.) - To enable
          responsive sizing on width, it should be set to ``"container"``.

        **Default value:** Based on ``config.view.continuousWidth`` for a plot with a
        continuous x-field and ``config.view.discreteWidth`` otherwise.

        **Note:** For plots with `row and column channels
        <https://vega.github.io/vega-lite/docs/encoding.html#facet>`__, this represents the
        width of a single view and the ``"container"`` option cannot be used.

        **See also:** `width <https://vega.github.io/vega-lite/docs/size.html>`__
        documentation.
    $schema : string
        URL to `JSON schema <http://json-schema.org/>`__ for a Vega-Lite specification.
        Unless you have a reason to change this, use
        ``https://vega.github.io/schema/vega-lite/v4.json``. Setting the ``$schema``
        property allows automatic validation and autocomplete in editors that support JSON
        schema.
    """
    _schema = {'$ref': '#/definitions/TopLevelUnitSpec'}

    def __init__(self, data=Undefined, mark=Undefined, align=Undefined, autosize=Undefined,
                 background=Undefined, bounds=Undefined, center=Undefined, config=Undefined,
                 datasets=Undefined, description=Undefined, encoding=Undefined, height=Undefined,
                 name=Undefined, padding=Undefined, params=Undefined, projection=Undefined,
                 resolve=Undefined, selection=Undefined, spacing=Undefined, title=Undefined,
                 transform=Undefined, usermeta=Undefined, view=Undefined, width=Undefined, **kwds):
        super(TopLevelUnitSpec, self).__init__(data=data, mark=mark, align=align, autosize=autosize,
                                               background=background, bounds=bounds, center=center,
                                               config=config, datasets=datasets,
                                               description=description, encoding=encoding,
                                               height=height, name=name, padding=padding, params=params,
                                               projection=projection, resolve=resolve,
                                               selection=selection, spacing=spacing, title=title,
                                               transform=transform, usermeta=usermeta, view=view,
                                               width=width, **kwds)


class TopoDataFormat(DataFormat):
    """TopoDataFormat schema wrapper

    Mapping(required=[])

    Attributes
    ----------

    feature : string
        The name of the TopoJSON object set to convert to a GeoJSON feature collection. For
        example, in a map of the world, there may be an object set named ``"countries"``.
        Using the feature property, we can extract this set and generate a GeoJSON feature
        object for each country.
    mesh : string
        The name of the TopoJSON object set to convert to mesh. Similar to the ``feature``
        option, ``mesh`` extracts a named TopoJSON object set.   Unlike the ``feature``
        option, the corresponding geo data is returned as a single, unified mesh instance,
        not as individual GeoJSON features. Extracting a mesh is useful for more efficiently
        drawing borders or other geographic elements that you do not need to associate with
        specific regions such as individual countries, states or counties.
    parse : anyOf(:class:`Parse`, None)
        If set to ``null``, disable type inference based on the spec and only use type
        inference based on the data. Alternatively, a parsing directive object can be
        provided for explicit data types. Each property of the object corresponds to a field
        name, and the value to the desired data type (one of ``"number"``, ``"boolean"``,
        ``"date"``, or null (do not parse the field)). For example, ``"parse":
        {"modified_on": "date"}`` parses the ``modified_on`` field in each input record a
        Date value.

        For ``"date"``, we parse data based using Javascript's `Date.parse()
        <https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Date/parse>`__.
        For Specific date formats can be provided (e.g., ``{foo: "date:'%m%d%Y'"}`` ), using
        the `d3-time-format syntax <https://github.com/d3/d3-time-format#locale_format>`__.
        UTC date format parsing is supported similarly (e.g., ``{foo: "utc:'%m%d%Y'"}`` ).
        See more about `UTC time
        <https://vega.github.io/vega-lite/docs/timeunit.html#utc>`__
    type : string
        Type of input data: ``"json"``, ``"csv"``, ``"tsv"``, ``"dsv"``.

        **Default value:**  The default format type is determined by the extension of the
        file URL. If no extension is detected, ``"json"`` will be used by default.
    """
    _schema = {'$ref': '#/definitions/TopoDataFormat'}

    def __init__(self, feature=Undefined, mesh=Undefined, parse=Undefined, type=Undefined, **kwds):
        super(TopoDataFormat, self).__init__(feature=feature, mesh=mesh, parse=parse, type=type, **kwds)


class Transform(VegaLiteSchema):
    """Transform schema wrapper

    anyOf(:class:`AggregateTransform`, :class:`BinTransform`, :class:`CalculateTransform`,
    :class:`DensityTransform`, :class:`FilterTransform`, :class:`FlattenTransform`,
    :class:`FoldTransform`, :class:`ImputeTransform`, :class:`JoinAggregateTransform`,
    :class:`LoessTransform`, :class:`LookupTransform`, :class:`QuantileTransform`,
    :class:`RegressionTransform`, :class:`TimeUnitTransform`, :class:`SampleTransform`,
    :class:`StackTransform`, :class:`WindowTransform`, :class:`PivotTransform`)
    """
    _schema = {'$ref': '#/definitions/Transform'}

    def __init__(self, *args, **kwds):
        super(Transform, self).__init__(*args, **kwds)


class AggregateTransform(Transform):
    """AggregateTransform schema wrapper

    Mapping(required=[aggregate])

    Attributes
    ----------

    aggregate : List(:class:`AggregatedFieldDef`)
        Array of objects that define fields to aggregate.
    groupby : List(:class:`FieldName`)
        The data fields to group by. If not specified, a single group containing all data
        objects will be used.
    """
    _schema = {'$ref': '#/definitions/AggregateTransform'}

    def __init__(self, aggregate=Undefined, groupby=Undefined, **kwds):
        super(AggregateTransform, self).__init__(aggregate=aggregate, groupby=groupby, **kwds)


class BinTransform(Transform):
    """BinTransform schema wrapper

    Mapping(required=[bin, field, as])

    Attributes
    ----------

    bin : anyOf(boolean, :class:`BinParams`)
        An object indicating bin properties, or simply ``true`` for using default bin
        parameters.
    field : :class:`FieldName`
        The data field to bin.
    as : anyOf(:class:`FieldName`, List(:class:`FieldName`))
        The output fields at which to write the start and end bin values. This can be either
        a string or an array of strings with two elements denoting the name for the fields
        for bin start and bin end respectively. If a single string (e.g., ``"val"`` ) is
        provided, the end field will be ``"val_end"``.
    """
    _schema = {'$ref': '#/definitions/BinTransform'}

    def __init__(self, bin=Undefined, field=Undefined, **kwds):
        super(BinTransform, self).__init__(bin=bin, field=field, **kwds)


class CalculateTransform(Transform):
    """CalculateTransform schema wrapper

    Mapping(required=[calculate, as])

    Attributes
    ----------

    calculate : string
        A `expression <https://vega.github.io/vega-lite/docs/types.html#expression>`__
        string. Use the variable ``datum`` to refer to the current data object.
    as : :class:`FieldName`
        The field for storing the computed formula value.
    """
    _schema = {'$ref': '#/definitions/CalculateTransform'}

    def __init__(self, calculate=Undefined, **kwds):
        super(CalculateTransform, self).__init__(calculate=calculate, **kwds)


class DensityTransform(Transform):
    """DensityTransform schema wrapper

    Mapping(required=[density])

    Attributes
    ----------

    density : :class:`FieldName`
        The data field for which to perform density estimation.
    bandwidth : float
        The bandwidth (standard deviation) of the Gaussian kernel. If unspecified or set to
        zero, the bandwidth value is automatically estimated from the input data using
        Scott’s rule.
    counts : boolean
        A boolean flag indicating if the output values should be probability estimates
        (false) or smoothed counts (true).

        **Default value:** ``false``
    cumulative : boolean
        A boolean flag indicating whether to produce density estimates (false) or cumulative
        density estimates (true).

        **Default value:** ``false``
    extent : List([float, float])
        A [min, max] domain from which to sample the distribution. If unspecified, the
        extent will be determined by the observed minimum and maximum values of the density
        value field.
    groupby : List(:class:`FieldName`)
        The data fields to group by. If not specified, a single group containing all data
        objects will be used.
    maxsteps : float
        The maximum number of samples to take along the extent domain for plotting the
        density.

        **Default value:** ``200``
    minsteps : float
        The minimum number of samples to take along the extent domain for plotting the
        density.

        **Default value:** ``25``
    steps : float
        The exact number of samples to take along the extent domain for plotting the
        density. If specified, overrides both minsteps and maxsteps to set an exact number
        of uniform samples. Potentially useful in conjunction with a fixed extent to ensure
        consistent sample points for stacked densities.
    as : List([:class:`FieldName`, :class:`FieldName`])
        The output fields for the sample value and corresponding density estimate.

        **Default value:** ``["value", "density"]``
    """
    _schema = {'$ref': '#/definitions/DensityTransform'}

    def __init__(self, density=Undefined, bandwidth=Undefined, counts=Undefined, cumulative=Undefined,
                 extent=Undefined, groupby=Undefined, maxsteps=Undefined, minsteps=Undefined,
                 steps=Undefined, **kwds):
        super(DensityTransform, self).__init__(density=density, bandwidth=bandwidth, counts=counts,
                                               cumulative=cumulative, extent=extent, groupby=groupby,
                                               maxsteps=maxsteps, minsteps=minsteps, steps=steps, **kwds)


class FilterTransform(Transform):
    """FilterTransform schema wrapper

    Mapping(required=[filter])

    Attributes
    ----------

    filter : :class:`PredicateComposition`
        The ``filter`` property must be a predication definition, which can take one of the
        following forms:

        1) an `expression <https://vega.github.io/vega-lite/docs/types.html#expression>`__
        string, where ``datum`` can be used to refer to the current data object. For
        example, ``{filter: "datum.b2 > 60"}`` would make the output data includes only
        items that have values in the field ``b2`` over 60.

        2) one of the `field predicates
        <https://vega.github.io/vega-lite/docs/predicate.html#field-predicate>`__ :  `equal
        <https://vega.github.io/vega-lite/docs/predicate.html#field-equal-predicate>`__, `lt
        <https://vega.github.io/vega-lite/docs/predicate.html#lt-predicate>`__, `lte
        <https://vega.github.io/vega-lite/docs/predicate.html#lte-predicate>`__, `gt
        <https://vega.github.io/vega-lite/docs/predicate.html#gt-predicate>`__, `gte
        <https://vega.github.io/vega-lite/docs/predicate.html#gte-predicate>`__, `range
        <https://vega.github.io/vega-lite/docs/predicate.html#range-predicate>`__, `oneOf
        <https://vega.github.io/vega-lite/docs/predicate.html#one-of-predicate>`__, or
        `valid <https://vega.github.io/vega-lite/docs/predicate.html#valid-predicate>`__,

        3) a `selection predicate
        <https://vega.github.io/vega-lite/docs/predicate.html#selection-predicate>`__, which
        define the names of a selection that the data point should belong to (or a logical
        composition of selections).

        4) a `logical composition
        <https://vega.github.io/vega-lite/docs/predicate.html#composition>`__ of (1), (2),
        or (3).
    """
    _schema = {'$ref': '#/definitions/FilterTransform'}

    def __init__(self, filter=Undefined, **kwds):
        super(FilterTransform, self).__init__(filter=filter, **kwds)


class FlattenTransform(Transform):
    """FlattenTransform schema wrapper

    Mapping(required=[flatten])

    Attributes
    ----------

    flatten : List(:class:`FieldName`)
        An array of one or more data fields containing arrays to flatten. If multiple fields
        are specified, their array values should have a parallel structure, ideally with the
        same length. If the lengths of parallel arrays do not match, the longest array will
        be used with ``null`` values added for missing entries.
    as : List(:class:`FieldName`)
        The output field names for extracted array values.

        **Default value:** The field name of the corresponding array field
    """
    _schema = {'$ref': '#/definitions/FlattenTransform'}

    def __init__(self, flatten=Undefined, **kwds):
        super(FlattenTransform, self).__init__(flatten=flatten, **kwds)


class FoldTransform(Transform):
    """FoldTransform schema wrapper

    Mapping(required=[fold])

    Attributes
    ----------

    fold : List(:class:`FieldName`)
        An array of data fields indicating the properties to fold.
    as : List([:class:`FieldName`, :class:`FieldName`])
        The output field names for the key and value properties produced by the fold
        transform. **Default value:** ``["key", "value"]``
    """
    _schema = {'$ref': '#/definitions/FoldTransform'}

    def __init__(self, fold=Undefined, **kwds):
        super(FoldTransform, self).__init__(fold=fold, **kwds)


class ImputeTransform(Transform):
    """ImputeTransform schema wrapper

    Mapping(required=[impute, key])

    Attributes
    ----------

    impute : :class:`FieldName`
        The data field for which the missing values should be imputed.
    key : :class:`FieldName`
        A key field that uniquely identifies data objects within a group. Missing key values
        (those occurring in the data but not in the current group) will be imputed.
    frame : List([anyOf(None, float), anyOf(None, float)])
        A frame specification as a two-element array used to control the window over which
        the specified method is applied. The array entries should either be a number
        indicating the offset from the current data object, or null to indicate unbounded
        rows preceding or following the current data object. For example, the value ``[-5,
        5]`` indicates that the window should include five objects preceding and five
        objects following the current object.

        **Default value:** :  ``[null, null]`` indicating that the window includes all
        objects.
    groupby : List(:class:`FieldName`)
        An optional array of fields by which to group the values. Imputation will then be
        performed on a per-group basis.
    keyvals : anyOf(List(Any), :class:`ImputeSequence`)
        Defines the key values that should be considered for imputation. An array of key
        values or an object defining a `number sequence
        <https://vega.github.io/vega-lite/docs/impute.html#sequence-def>`__.

        If provided, this will be used in addition to the key values observed within the
        input data. If not provided, the values will be derived from all unique values of
        the ``key`` field. For ``impute`` in ``encoding``, the key field is the x-field if
        the y-field is imputed, or vice versa.

        If there is no impute grouping, this property *must* be specified.
    method : :class:`ImputeMethod`
        The imputation method to use for the field value of imputed data objects. One of
        ``"value"``, ``"mean"``, ``"median"``, ``"max"`` or ``"min"``.

        **Default value:**  ``"value"``
    value : Any
        The field value to use when the imputation ``method`` is ``"value"``.
    """
    _schema = {'$ref': '#/definitions/ImputeTransform'}

    def __init__(self, impute=Undefined, key=Undefined, frame=Undefined, groupby=Undefined,
                 keyvals=Undefined, method=Undefined, value=Undefined, **kwds):
        super(ImputeTransform, self).__init__(impute=impute, key=key, frame=frame, groupby=groupby,
                                              keyvals=keyvals, method=method, value=value, **kwds)


class JoinAggregateTransform(Transform):
    """JoinAggregateTransform schema wrapper

    Mapping(required=[joinaggregate])

    Attributes
    ----------

    joinaggregate : List(:class:`JoinAggregateFieldDef`)
        The definition of the fields in the join aggregate, and what calculations to use.
    groupby : List(:class:`FieldName`)
        The data fields for partitioning the data objects into separate groups. If
        unspecified, all data points will be in a single group.
    """
    _schema = {'$ref': '#/definitions/JoinAggregateTransform'}

    def __init__(self, joinaggregate=Undefined, groupby=Undefined, **kwds):
        super(JoinAggregateTransform, self).__init__(joinaggregate=joinaggregate, groupby=groupby,
                                                     **kwds)


class LoessTransform(Transform):
    """LoessTransform schema wrapper

    Mapping(required=[loess, on])

    Attributes
    ----------

    loess : :class:`FieldName`
        The data field of the dependent variable to smooth.
    on : :class:`FieldName`
        The data field of the independent variable to use a predictor.
    bandwidth : float
        A bandwidth parameter in the range ``[0, 1]`` that determines the amount of
        smoothing.

        **Default value:** ``0.3``
    groupby : List(:class:`FieldName`)
        The data fields to group by. If not specified, a single group containing all data
        objects will be used.
    as : List([:class:`FieldName`, :class:`FieldName`])
        The output field names for the smoothed points generated by the loess transform.

        **Default value:** The field names of the input x and y values.
    """
    _schema = {'$ref': '#/definitions/LoessTransform'}

    def __init__(self, loess=Undefined, on=Undefined, bandwidth=Undefined, groupby=Undefined, **kwds):
        super(LoessTransform, self).__init__(loess=loess, on=on, bandwidth=bandwidth, groupby=groupby,
                                             **kwds)


class LookupTransform(Transform):
    """LookupTransform schema wrapper

    Mapping(required=[lookup, from])

    Attributes
    ----------

    lookup : string
        Key in primary data source.
    default : string
        The default value to use if lookup fails.

        **Default value:** ``null``
    as : anyOf(:class:`FieldName`, List(:class:`FieldName`))
        The output fields on which to store the looked up data values.

        For data lookups, this property may be left blank if ``from.fields`` has been
        specified (those field names will be used); if ``from.fields`` has not been
        specified, ``as`` must be a string.

        For selection lookups, this property is optional: if unspecified, looked up values
        will be stored under a property named for the selection; and if specified, it must
        correspond to ``from.fields``.
    from : anyOf(:class:`LookupData`, :class:`LookupSelection`)
        Data source or selection for secondary data reference.
    """
    _schema = {'$ref': '#/definitions/LookupTransform'}

    def __init__(self, lookup=Undefined, default=Undefined, **kwds):
        super(LookupTransform, self).__init__(lookup=lookup, default=default, **kwds)


class PivotTransform(Transform):
    """PivotTransform schema wrapper

    Mapping(required=[pivot, value])

    Attributes
    ----------

    pivot : :class:`FieldName`
        The data field to pivot on. The unique values of this field become new field names
        in the output stream.
    value : :class:`FieldName`
        The data field to populate pivoted fields. The aggregate values of this field become
        the values of the new pivoted fields.
    groupby : List(:class:`FieldName`)
        The optional data fields to group by. If not specified, a single group containing
        all data objects will be used.
    limit : float
        An optional parameter indicating the maximum number of pivoted fields to generate.
        The default ( ``0`` ) applies no limit. The pivoted ``pivot`` names are sorted in
        ascending order prior to enforcing the limit. **Default value:** ``0``
    op : string
        The aggregation operation to apply to grouped ``value`` field values. **Default
        value:** ``sum``
    """
    _schema = {'$ref': '#/definitions/PivotTransform'}

    def __init__(self, pivot=Undefined, value=Undefined, groupby=Undefined, limit=Undefined,
                 op=Undefined, **kwds):
        super(PivotTransform, self).__init__(pivot=pivot, value=value, groupby=groupby, limit=limit,
                                             op=op, **kwds)


class QuantileTransform(Transform):
    """QuantileTransform schema wrapper

    Mapping(required=[quantile])

    Attributes
    ----------

    quantile : :class:`FieldName`
        The data field for which to perform quantile estimation.
    groupby : List(:class:`FieldName`)
        The data fields to group by. If not specified, a single group containing all data
        objects will be used.
    probs : List(float)
        An array of probabilities in the range (0, 1) for which to compute quantile values.
        If not specified, the *step* parameter will be used.
    step : float
        A probability step size (default 0.01) for sampling quantile values. All values from
        one-half the step size up to 1 (exclusive) will be sampled. This parameter is only
        used if the *probs* parameter is not provided.
    as : List([:class:`FieldName`, :class:`FieldName`])
        The output field names for the probability and quantile values.

        **Default value:** ``["prob", "value"]``
    """
    _schema = {'$ref': '#/definitions/QuantileTransform'}

    def __init__(self, quantile=Undefined, groupby=Undefined, probs=Undefined, step=Undefined, **kwds):
        super(QuantileTransform, self).__init__(quantile=quantile, groupby=groupby, probs=probs,
                                                step=step, **kwds)


class RegressionTransform(Transform):
    """RegressionTransform schema wrapper

    Mapping(required=[regression, on])

    Attributes
    ----------

    on : :class:`FieldName`
        The data field of the independent variable to use a predictor.
    regression : :class:`FieldName`
        The data field of the dependent variable to predict.
    extent : List([float, float])
        A [min, max] domain over the independent (x) field for the starting and ending
        points of the generated trend line.
    groupby : List(:class:`FieldName`)
        The data fields to group by. If not specified, a single group containing all data
        objects will be used.
    method : enum('linear', 'log', 'exp', 'pow', 'quad', 'poly')
        The functional form of the regression model. One of ``"linear"``, ``"log"``,
        ``"exp"``, ``"pow"``, ``"quad"``, or ``"poly"``.

        **Default value:** ``"linear"``
    order : float
        The polynomial order (number of coefficients) for the 'poly' method.

        **Default value:** ``3``
    params : boolean
        A boolean flag indicating if the transform should return the regression model
        parameters (one object per group), rather than trend line points. The resulting
        objects include a ``coef`` array of fitted coefficient values (starting with the
        intercept term and then including terms of increasing order) and an ``rSquared``
        value (indicating the total variance explained by the model).

        **Default value:** ``false``
    as : List([:class:`FieldName`, :class:`FieldName`])
        The output field names for the smoothed points generated by the regression
        transform.

        **Default value:** The field names of the input x and y values.
    """
    _schema = {'$ref': '#/definitions/RegressionTransform'}

    def __init__(self, on=Undefined, regression=Undefined, extent=Undefined, groupby=Undefined,
                 method=Undefined, order=Undefined, params=Undefined, **kwds):
        super(RegressionTransform, self).__init__(on=on, regression=regression, extent=extent,
                                                  groupby=groupby, method=method, order=order,
                                                  params=params, **kwds)


class SampleTransform(Transform):
    """SampleTransform schema wrapper

    Mapping(required=[sample])

    Attributes
    ----------

    sample : float
        The maximum number of data objects to include in the sample.

        **Default value:** ``1000``
    """
    _schema = {'$ref': '#/definitions/SampleTransform'}

    def __init__(self, sample=Undefined, **kwds):
        super(SampleTransform, self).__init__(sample=sample, **kwds)


class StackTransform(Transform):
    """StackTransform schema wrapper

    Mapping(required=[stack, groupby, as])

    Attributes
    ----------

    groupby : List(:class:`FieldName`)
        The data fields to group by.
    stack : :class:`FieldName`
        The field which is stacked.
    offset : enum('zero', 'center', 'normalize')
        Mode for stacking marks. One of ``"zero"`` (default), ``"center"``, or
        ``"normalize"``. The ``"zero"`` offset will stack starting at ``0``. The
        ``"center"`` offset will center the stacks. The ``"normalize"`` offset will compute
        percentage values for each stack point, with output values in the range ``[0,1]``.

        **Default value:** ``"zero"``
    sort : List(:class:`SortField`)
        Field that determines the order of leaves in the stacked charts.
    as : anyOf(:class:`FieldName`, List([:class:`FieldName`, :class:`FieldName`]))
        Output field names. This can be either a string or an array of strings with two
        elements denoting the name for the fields for stack start and stack end
        respectively. If a single string(e.g., ``"val"`` ) is provided, the end field will
        be ``"val_end"``.
    """
    _schema = {'$ref': '#/definitions/StackTransform'}

    def __init__(self, groupby=Undefined, stack=Undefined, offset=Undefined, sort=Undefined, **kwds):
        super(StackTransform, self).__init__(groupby=groupby, stack=stack, offset=offset, sort=sort,
                                             **kwds)


class TimeUnitTransform(Transform):
    """TimeUnitTransform schema wrapper

    Mapping(required=[timeUnit, field, as])

    Attributes
    ----------

    field : :class:`FieldName`
        The data field to apply time unit.
    timeUnit : anyOf(:class:`TimeUnit`, :class:`TimeUnitParams`)
        The timeUnit.
    as : :class:`FieldName`
        The output field to write the timeUnit value.
    """
    _schema = {'$ref': '#/definitions/TimeUnitTransform'}

    def __init__(self, field=Undefined, timeUnit=Undefined, **kwds):
        super(TimeUnitTransform, self).__init__(field=field, timeUnit=timeUnit, **kwds)


class Type(VegaLiteSchema):
    """Type schema wrapper

    enum('quantitative', 'ordinal', 'temporal', 'nominal', 'geojson')
    Data type based on level of measurement
    """
    _schema = {'$ref': '#/definitions/Type'}

    def __init__(self, *args):
        super(Type, self).__init__(*args)


class TypeForShape(VegaLiteSchema):
    """TypeForShape schema wrapper

    enum('nominal', 'ordinal', 'geojson')
    """
    _schema = {'$ref': '#/definitions/TypeForShape'}

    def __init__(self, *args):
        super(TypeForShape, self).__init__(*args)


class TypedFieldDef(VegaLiteSchema):
    """TypedFieldDef schema wrapper

    Mapping(required=[])
    Definition object for a data field, its type and transformation of an encoding channel.

    Attributes
    ----------

    aggregate : :class:`Aggregate`
        Aggregation function for the field (e.g., ``"mean"``, ``"sum"``, ``"median"``,
        ``"min"``, ``"max"``, ``"count"`` ).

        **Default value:** ``undefined`` (None)

        **See also:** `aggregate <https://vega.github.io/vega-lite/docs/aggregate.html>`__
        documentation.
    band : float
        For rect-based marks ( ``rect``, ``bar``, and ``image`` ), mark size relative to
        bandwidth of `band scales
        <https://vega.github.io/vega-lite/docs/scale.html#band>`__, bins or time units. If
        set to ``1``, the mark size is set to the bandwidth, the bin interval, or the time
        unit interval. If set to ``0.5``, the mark size is half of the bandwidth or the time
        unit interval.

        For other marks, relative position on a band of a stacked, binned, time unit or band
        scale. If set to ``0``, the marks will be positioned at the beginning of the band.
        If set to ``0.5``, the marks will be positioned in the middle of the band.
    bin : anyOf(boolean, :class:`BinParams`, string, None)
        A flag for binning a ``quantitative`` field, `an object defining binning parameters
        <https://vega.github.io/vega-lite/docs/bin.html#params>`__, or indicating that the
        data for ``x`` or ``y`` channel are binned before they are imported into Vega-Lite (
        ``"binned"`` ).


        If ``true``, default `binning parameters
        <https://vega.github.io/vega-lite/docs/bin.html>`__ will be applied.

        If ``"binned"``, this indicates that the data for the ``x`` (or ``y`` ) channel are
        already binned. You can map the bin-start field to ``x`` (or ``y`` ) and the bin-end
        field to ``x2`` (or ``y2`` ). The scale and axis will be formatted similar to
        binning in Vega-Lite.  To adjust the axis ticks based on the bin step, you can also
        set the axis's `tickMinStep
        <https://vega.github.io/vega-lite/docs/axis.html#ticks>`__ property.

        **Default value:** ``false``

        **See also:** `bin <https://vega.github.io/vega-lite/docs/bin.html>`__
        documentation.
    field : :class:`Field`
        **Required.** A string defining the name of the field from which to pull a data
        value or an object defining iterated values from the `repeat
        <https://vega.github.io/vega-lite/docs/repeat.html>`__ operator.

        **See also:** `field <https://vega.github.io/vega-lite/docs/field.html>`__
        documentation.

        **Notes:** 1)  Dots ( ``.`` ) and brackets ( ``[`` and ``]`` ) can be used to access
        nested objects (e.g., ``"field": "foo.bar"`` and ``"field": "foo['bar']"`` ). If
        field names contain dots or brackets but are not nested, you can use ``\\`` to
        escape dots and brackets (e.g., ``"a\\.b"`` and ``"a\\[0\\]"`` ). See more details
        about escaping in the `field documentation
        <https://vega.github.io/vega-lite/docs/field.html>`__. 2) ``field`` is not required
        if ``aggregate`` is ``count``.
    timeUnit : anyOf(:class:`TimeUnit`, :class:`TimeUnitParams`)
        Time unit (e.g., ``year``, ``yearmonth``, ``month``, ``hours`` ) for a temporal
        field. or `a temporal field that gets casted as ordinal
        <https://vega.github.io/vega-lite/docs/type.html#cast>`__.

        **Default value:** ``undefined`` (None)

        **See also:** `timeUnit <https://vega.github.io/vega-lite/docs/timeunit.html>`__
        documentation.
    title : anyOf(:class:`Text`, None)
        A title for the field. If ``null``, the title will be removed.

        **Default value:**  derived from the field's name and transformation function (
        ``aggregate``, ``bin`` and ``timeUnit`` ). If the field has an aggregate function,
        the function is displayed as part of the title (e.g., ``"Sum of Profit"`` ). If the
        field is binned or has a time unit applied, the applied function is shown in
        parentheses (e.g., ``"Profit (binned)"``, ``"Transaction Date (year-month)"`` ).
        Otherwise, the title is simply the field name.

        **Notes** :

        1) You can customize the default field title format by providing the `fieldTitle
        <https://vega.github.io/vega-lite/docs/config.html#top-level-config>`__ property in
        the `config <https://vega.github.io/vega-lite/docs/config.html>`__ or `fieldTitle
        function via the compile function's options
        <https://vega.github.io/vega-lite/docs/compile.html#field-title>`__.

        2) If both field definition's ``title`` and axis, header, or legend ``title`` are
        defined, axis/header/legend title will be used.
    type : :class:`StandardType`
        The type of measurement ( ``"quantitative"``, ``"temporal"``, ``"ordinal"``, or
        ``"nominal"`` ) for the encoded field or constant value ( ``datum`` ). It can also
        be a ``"geojson"`` type for encoding `'geoshape'
        <https://vega.github.io/vega-lite/docs/geoshape.html>`__.

        Vega-Lite automatically infers data types in many cases as discussed below. However,
        type is required for a field if: (1) the field is not nominal and the field encoding
        has no specified ``aggregate`` (except ``argmin`` and ``argmax`` ), ``bin``, scale
        type, custom ``sort`` order, nor ``timeUnit`` or (2) if you wish to use an ordinal
        scale for a field with ``bin`` or ``timeUnit``.

        **Default value:**

        1) For a data ``field``, ``"nominal"`` is the default data type unless the field
        encoding has ``aggregate``, ``channel``, ``bin``, scale type, ``sort``, or
        ``timeUnit`` that satisfies the following criteria: - ``"quantitative"`` is the
        default type if (1) the encoded field contains ``bin`` or ``aggregate`` except
        ``"argmin"`` and ``"argmax"``, (2) the encoding channel is ``latitude`` or
        ``longitude`` channel or (3) if the specified scale type is `a quantitative scale
        <https://vega.github.io/vega-lite/docs/scale.html#type>`__. - ``"temporal"`` is the
        default type if (1) the encoded field contains ``timeUnit`` or (2) the specified
        scale type is a time or utc scale - ``ordinal""`` is the default type if (1) the
        encoded field contains a `custom sort order
        <https://vega.github.io/vega-lite/docs/sort.html#specifying-custom-sort-order>`__,
        (2) the specified scale type is an ordinal/point/band scale, or (3) the encoding
        channel is ``order``.

        2) For a constant value in data domain ( ``datum`` ): - ``"quantitative"`` if the
        datum is a number - ``"nominal"`` if the datum is a string - ``"temporal"`` if the
        datum is `a date time object
        <https://vega.github.io/vega-lite/docs/datetime.html>`__

        **Note:** - Data ``type`` describes the semantics of the data rather than the
        primitive data types (number, string, etc.). The same primitive data type can have
        different types of measurement. For example, numeric data can represent
        quantitative, ordinal, or nominal data. - Data values for a temporal field can be
        either a date-time string (e.g., ``"2015-03-07 12:32:17"``, ``"17:01"``,
        ``"2015-03-16"``. ``"2015"`` ) or a timestamp number (e.g., ``1552199579097`` ). -
        When using with `bin <https://vega.github.io/vega-lite/docs/bin.html>`__, the
        ``type`` property can be either ``"quantitative"`` (for using a linear bin scale) or
        `"ordinal" (for using an ordinal bin scale)
        <https://vega.github.io/vega-lite/docs/type.html#cast-bin>`__. - When using with
        `timeUnit <https://vega.github.io/vega-lite/docs/timeunit.html>`__, the ``type``
        property can be either ``"temporal"`` (default, for using a temporal scale) or
        `"ordinal" (for using an ordinal scale)
        <https://vega.github.io/vega-lite/docs/type.html#cast-bin>`__. - When using with
        `aggregate <https://vega.github.io/vega-lite/docs/aggregate.html>`__, the ``type``
        property refers to the post-aggregation data type. For example, we can calculate
        count ``distinct`` of a categorical field ``"cat"`` using ``{"aggregate":
        "distinct", "field": "cat"}``. The ``"type"`` of the aggregate output is
        ``"quantitative"``. - Secondary channels (e.g., ``x2``, ``y2``, ``xError``,
        ``yError`` ) do not have ``type`` as they must have exactly the same type as their
        primary channels (e.g., ``x``, ``y`` ).

        **See also:** `type <https://vega.github.io/vega-lite/docs/type.html>`__
        documentation.
    """
    _schema = {'$ref': '#/definitions/TypedFieldDef'}

    def __init__(self, aggregate=Undefined, band=Undefined, bin=Undefined, field=Undefined,
                 timeUnit=Undefined, title=Undefined, type=Undefined, **kwds):
        super(TypedFieldDef, self).__init__(aggregate=aggregate, band=band, bin=bin, field=field,
                                            timeUnit=timeUnit, title=title, type=type, **kwds)


class URI(VegaLiteSchema):
    """URI schema wrapper

    string
    """
    _schema = {'$ref': '#/definitions/URI'}

    def __init__(self, *args):
        super(URI, self).__init__(*args)


class UnitSpec(VegaLiteSchema):
    """UnitSpec schema wrapper

    Mapping(required=[mark])
    A unit specification, which can contain either `primitive marks or composite marks
    <https://vega.github.io/vega-lite/docs/mark.html#types>`__.

    Attributes
    ----------

    mark : :class:`AnyMark`
        A string describing the mark type (one of ``"bar"``, ``"circle"``, ``"square"``,
        ``"tick"``, ``"line"``, ``"area"``, ``"point"``, ``"rule"``, ``"geoshape"``, and
        ``"text"`` ) or a `mark definition object
        <https://vega.github.io/vega-lite/docs/mark.html#mark-def>`__.
    data : anyOf(:class:`Data`, None)
        An object describing the data source. Set to ``null`` to ignore the parent's data
        source. If no data is set, it is derived from the parent.
    description : string
        Description of this mark for commenting purpose.
    encoding : :class:`Encoding`
        A key-value mapping between encoding channels and definition of fields.
    height : anyOf(float, string, :class:`Step`)
        **Deprecated:** Please avoid using width in a unit spec that's a part of a layer
        spec.
    name : string
        Name of the visualization for later reference.
    projection : :class:`Projection`
        An object defining properties of geographic projection, which will be applied to
        ``shape`` path for ``"geoshape"`` marks and to ``latitude`` and ``"longitude"``
        channels for other marks.
    selection : Mapping(required=[])
        A key-value mapping between selection names and definitions.
    title : anyOf(:class:`Text`, :class:`TitleParams`)
        Title for the plot.
    transform : List(:class:`Transform`)
        An array of data transformations such as filter and new field calculation.
    view : :class:`ViewBackground`
        **Deprecated:** Please avoid using width in a unit spec that's a part of a layer
        spec.
    width : anyOf(float, string, :class:`Step`)
        **Deprecated:** Please avoid using width in a unit spec that's a part of a layer
        spec.
    """
    _schema = {'$ref': '#/definitions/UnitSpec'}

    def __init__(self, mark=Undefined, data=Undefined, description=Undefined, encoding=Undefined,
                 height=Undefined, name=Undefined, projection=Undefined, selection=Undefined,
                 title=Undefined, transform=Undefined, view=Undefined, width=Undefined, **kwds):
        super(UnitSpec, self).__init__(mark=mark, data=data, description=description, encoding=encoding,
                                       height=height, name=name, projection=projection,
                                       selection=selection, title=title, transform=transform, view=view,
                                       width=width, **kwds)


class UnitSpecWithFrame(VegaLiteSchema):
    """UnitSpecWithFrame schema wrapper

    Mapping(required=[mark])

    Attributes
    ----------

    mark : :class:`AnyMark`
        A string describing the mark type (one of ``"bar"``, ``"circle"``, ``"square"``,
        ``"tick"``, ``"line"``, ``"area"``, ``"point"``, ``"rule"``, ``"geoshape"``, and
        ``"text"`` ) or a `mark definition object
        <https://vega.github.io/vega-lite/docs/mark.html#mark-def>`__.
    data : anyOf(:class:`Data`, None)
        An object describing the data source. Set to ``null`` to ignore the parent's data
        source. If no data is set, it is derived from the parent.
    description : string
        Description of this mark for commenting purpose.
    encoding : :class:`Encoding`
        A key-value mapping between encoding channels and definition of fields.
    height : anyOf(float, string, :class:`Step`)
        The height of a visualization.


        * For a plot with a continuous y-field, height should be a number. - For a plot with
          either a discrete y-field or no y-field, height can be either a number indicating
          a fixed height or an object in the form of ``{step: number}`` defining the height
          per discrete step. (No y-field is equivalent to having one discrete step.) - To
          enable responsive sizing on height, it should be set to ``"container"``.

        **Default value:** Based on ``config.view.continuousHeight`` for a plot with a
        continuous y-field and ``config.view.discreteHeight`` otherwise.

        **Note:** For plots with `row and column channels
        <https://vega.github.io/vega-lite/docs/encoding.html#facet>`__, this represents the
        height of a single view and the ``"container"`` option cannot be used.

        **See also:** `height <https://vega.github.io/vega-lite/docs/size.html>`__
        documentation.
    name : string
        Name of the visualization for later reference.
    projection : :class:`Projection`
        An object defining properties of geographic projection, which will be applied to
        ``shape`` path for ``"geoshape"`` marks and to ``latitude`` and ``"longitude"``
        channels for other marks.
    selection : Mapping(required=[])
        A key-value mapping between selection names and definitions.
    title : anyOf(:class:`Text`, :class:`TitleParams`)
        Title for the plot.
    transform : List(:class:`Transform`)
        An array of data transformations such as filter and new field calculation.
    view : :class:`ViewBackground`
        An object defining the view background's fill and stroke.

        **Default value:** none (transparent)
    width : anyOf(float, string, :class:`Step`)
        The width of a visualization.


        * For a plot with a continuous x-field, width should be a number. - For a plot with
          either a discrete x-field or no x-field, width can be either a number indicating a
          fixed width or an object in the form of ``{step: number}`` defining the width per
          discrete step. (No x-field is equivalent to having one discrete step.) - To enable
          responsive sizing on width, it should be set to ``"container"``.

        **Default value:** Based on ``config.view.continuousWidth`` for a plot with a
        continuous x-field and ``config.view.discreteWidth`` otherwise.

        **Note:** For plots with `row and column channels
        <https://vega.github.io/vega-lite/docs/encoding.html#facet>`__, this represents the
        width of a single view and the ``"container"`` option cannot be used.

        **See also:** `width <https://vega.github.io/vega-lite/docs/size.html>`__
        documentation.
    """
    _schema = {'$ref': '#/definitions/UnitSpecWithFrame'}

    def __init__(self, mark=Undefined, data=Undefined, description=Undefined, encoding=Undefined,
                 height=Undefined, name=Undefined, projection=Undefined, selection=Undefined,
                 title=Undefined, transform=Undefined, view=Undefined, width=Undefined, **kwds):
        super(UnitSpecWithFrame, self).__init__(mark=mark, data=data, description=description,
                                                encoding=encoding, height=height, name=name,
                                                projection=projection, selection=selection, title=title,
                                                transform=transform, view=view, width=width, **kwds)


class UrlData(DataSource):
    """UrlData schema wrapper

    Mapping(required=[url])

    Attributes
    ----------

    url : string
        An URL from which to load the data set. Use the ``format.type`` property to ensure
        the loaded data is correctly parsed.
    format : :class:`DataFormat`
        An object that specifies the format for parsing the data.
    name : string
        Provide a placeholder name and bind data at runtime.
    """
    _schema = {'$ref': '#/definitions/UrlData'}

    def __init__(self, url=Undefined, format=Undefined, name=Undefined, **kwds):
        super(UrlData, self).__init__(url=url, format=format, name=name, **kwds)


class UtcMultiTimeUnit(MultiTimeUnit):
    """UtcMultiTimeUnit schema wrapper

    enum('utcyearquarter', 'utcyearquartermonth', 'utcyearmonth', 'utcyearmonthdate',
    'utcyearmonthdatehours', 'utcyearmonthdatehoursminutes',
    'utcyearmonthdatehoursminutesseconds', 'utcyearweek', 'utcyearweekday',
    'utcyearweekdayhours', 'utcyearweekdayhoursminutes', 'utcyearweekdayhoursminutesseconds',
    'utcyeardayofyear', 'utcquartermonth', 'utcmonthdate', 'utcmonthdatehours',
    'utcmonthdatehoursminutes', 'utcmonthdatehoursminutesseconds', 'utcweekday',
    'utcweeksdayhours', 'utcweekdayhoursminutes', 'utcweekdayhoursminutesseconds',
    'utcdayhours', 'utcdayhoursminutes', 'utcdayhoursminutesseconds', 'utchoursminutes',
    'utchoursminutesseconds', 'utcminutesseconds', 'utcsecondsmilliseconds')
    """
    _schema = {'$ref': '#/definitions/UtcMultiTimeUnit'}

    def __init__(self, *args):
        super(UtcMultiTimeUnit, self).__init__(*args)


class UtcSingleTimeUnit(SingleTimeUnit):
    """UtcSingleTimeUnit schema wrapper

    enum('utcyear', 'utcquarter', 'utcmonth', 'utcweek', 'utcday', 'utcdayofyear', 'utcdate',
    'utchours', 'utcminutes', 'utcseconds', 'utcmilliseconds')
    """
    _schema = {'$ref': '#/definitions/UtcSingleTimeUnit'}

    def __init__(self, *args):
        super(UtcSingleTimeUnit, self).__init__(*args)


class VConcatSpecGenericSpec(Spec):
    """VConcatSpecGenericSpec schema wrapper

    Mapping(required=[vconcat])
    Base interface for a vertical concatenation specification.

    Attributes
    ----------

    vconcat : List(:class:`Spec`)
        A list of views to be concatenated and put into a column.
    bounds : enum('full', 'flush')
        The bounds calculation method to use for determining the extent of a sub-plot. One
        of ``full`` (the default) or ``flush``.


        * If set to ``full``, the entire calculated bounds (including axes, title, and
          legend) will be used. - If set to ``flush``, only the specified width and height
          values for the sub-view will be used. The ``flush`` setting can be useful when
          attempting to place sub-plots without axes or legends into a uniform grid
          structure.

        **Default value:** ``"full"``
    center : boolean
        Boolean flag indicating if subviews should be centered relative to their respective
        rows or columns.

        **Default value:** ``false``
    data : anyOf(:class:`Data`, None)
        An object describing the data source. Set to ``null`` to ignore the parent's data
        source. If no data is set, it is derived from the parent.
    description : string
        Description of this mark for commenting purpose.
    name : string
        Name of the visualization for later reference.
    resolve : :class:`Resolve`
        Scale, axis, and legend resolutions for view composition specifications.
    spacing : float
        The spacing in pixels between sub-views of the concat operator.

        **Default value** : ``10``
    title : anyOf(:class:`Text`, :class:`TitleParams`)
        Title for the plot.
    transform : List(:class:`Transform`)
        An array of data transformations such as filter and new field calculation.
    """
    _schema = {'$ref': '#/definitions/VConcatSpec<GenericSpec>'}

    def __init__(self, vconcat=Undefined, bounds=Undefined, center=Undefined, data=Undefined,
                 description=Undefined, name=Undefined, resolve=Undefined, spacing=Undefined,
                 title=Undefined, transform=Undefined, **kwds):
        super(VConcatSpecGenericSpec, self).__init__(vconcat=vconcat, bounds=bounds, center=center,
                                                     data=data, description=description, name=name,
                                                     resolve=resolve, spacing=spacing, title=title,
                                                     transform=transform, **kwds)


class ValueDefWithConditionMarkPropFieldOrDatumDefGradientstringnull(ColorDef, MarkPropDefGradientstringnull):
    """ValueDefWithConditionMarkPropFieldOrDatumDefGradientstringnull schema wrapper

    Mapping(required=[])

    Attributes
    ----------

    condition : anyOf(:class:`ConditionalMarkPropFieldOrDatumDef`,
    :class:`ConditionalValueDefGradientstringnullExprRef`,
    List(:class:`ConditionalValueDefGradientstringnullExprRef`))
        A field definition or one or more value definition(s) with a selection predicate.
    value : anyOf(:class:`Gradient`, string, None, :class:`ExprRef`)
        A constant value in visual domain (e.g., ``"red"`` / ``"#0099ff"`` / `gradient
        definition <https://vega.github.io/vega-lite/docs/types.html#gradient>`__ for color,
        values between ``0`` to ``1`` for opacity).
    """
    _schema = {'$ref': '#/definitions/ValueDefWithCondition<MarkPropFieldOrDatumDef,(Gradient|string|null)>'}

    def __init__(self, condition=Undefined, value=Undefined, **kwds):
        super(ValueDefWithConditionMarkPropFieldOrDatumDefGradientstringnull, self).__init__(condition=condition,
                                                                                             value=value,
                                                                                             **kwds)


class ValueDefWithConditionMarkPropFieldOrDatumDefTypeForShapestringnull(MarkPropDefstringnullTypeForShape, ShapeDef):
    """ValueDefWithConditionMarkPropFieldOrDatumDefTypeForShapestringnull schema wrapper

    Mapping(required=[])

    Attributes
    ----------

    condition : anyOf(:class:`ConditionalMarkPropFieldOrDatumDefTypeForShape`,
    :class:`ConditionalValueDefstringnullExprRef`,
    List(:class:`ConditionalValueDefstringnullExprRef`))
        A field definition or one or more value definition(s) with a selection predicate.
    value : anyOf(string, None, :class:`ExprRef`)
        A constant value in visual domain (e.g., ``"red"`` / ``"#0099ff"`` / `gradient
        definition <https://vega.github.io/vega-lite/docs/types.html#gradient>`__ for color,
        values between ``0`` to ``1`` for opacity).
    """
    _schema = {'$ref': '#/definitions/ValueDefWithCondition<MarkPropFieldOrDatumDef<TypeForShape>,(string|null)>'}

    def __init__(self, condition=Undefined, value=Undefined, **kwds):
        super(ValueDefWithConditionMarkPropFieldOrDatumDefTypeForShapestringnull, self).__init__(condition=condition,
                                                                                                 value=value,
                                                                                                 **kwds)


class ValueDefWithConditionMarkPropFieldOrDatumDefnumber(MarkPropDefnumber, NumericMarkPropDef):
    """ValueDefWithConditionMarkPropFieldOrDatumDefnumber schema wrapper

    Mapping(required=[])

    Attributes
    ----------

    condition : anyOf(:class:`ConditionalMarkPropFieldOrDatumDef`,
    :class:`ConditionalValueDefnumberExprRef`, List(:class:`ConditionalValueDefnumberExprRef`))
        A field definition or one or more value definition(s) with a selection predicate.
    value : anyOf(float, :class:`ExprRef`)
        A constant value in visual domain (e.g., ``"red"`` / ``"#0099ff"`` / `gradient
        definition <https://vega.github.io/vega-lite/docs/types.html#gradient>`__ for color,
        values between ``0`` to ``1`` for opacity).
    """
    _schema = {'$ref': '#/definitions/ValueDefWithCondition<MarkPropFieldOrDatumDef,number>'}

    def __init__(self, condition=Undefined, value=Undefined, **kwds):
        super(ValueDefWithConditionMarkPropFieldOrDatumDefnumber, self).__init__(condition=condition,
                                                                                 value=value, **kwds)


class ValueDefWithConditionMarkPropFieldOrDatumDefnumberArray(MarkPropDefnumberArray, NumericArrayMarkPropDef):
    """ValueDefWithConditionMarkPropFieldOrDatumDefnumberArray schema wrapper

    Mapping(required=[])

    Attributes
    ----------

    condition : anyOf(:class:`ConditionalMarkPropFieldOrDatumDef`,
    :class:`ConditionalValueDefnumberArrayExprRef`,
    List(:class:`ConditionalValueDefnumberArrayExprRef`))
        A field definition or one or more value definition(s) with a selection predicate.
    value : anyOf(List(float), :class:`ExprRef`)
        A constant value in visual domain (e.g., ``"red"`` / ``"#0099ff"`` / `gradient
        definition <https://vega.github.io/vega-lite/docs/types.html#gradient>`__ for color,
        values between ``0`` to ``1`` for opacity).
    """
    _schema = {'$ref': '#/definitions/ValueDefWithCondition<MarkPropFieldOrDatumDef,number[]>'}

    def __init__(self, condition=Undefined, value=Undefined, **kwds):
        super(ValueDefWithConditionMarkPropFieldOrDatumDefnumberArray, self).__init__(condition=condition,
                                                                                      value=value,
                                                                                      **kwds)


class ValueDefWithConditionMarkPropFieldOrDatumDefstringnull(VegaLiteSchema):
    """ValueDefWithConditionMarkPropFieldOrDatumDefstringnull schema wrapper

    Mapping(required=[])

    Attributes
    ----------

    condition : anyOf(:class:`ConditionalMarkPropFieldOrDatumDef`,
    :class:`ConditionalValueDefstringnullExprRef`,
    List(:class:`ConditionalValueDefstringnullExprRef`))
        A field definition or one or more value definition(s) with a selection predicate.
    value : anyOf(string, None, :class:`ExprRef`)
        A constant value in visual domain (e.g., ``"red"`` / ``"#0099ff"`` / `gradient
        definition <https://vega.github.io/vega-lite/docs/types.html#gradient>`__ for color,
        values between ``0`` to ``1`` for opacity).
    """
    _schema = {'$ref': '#/definitions/ValueDefWithCondition<MarkPropFieldOrDatumDef,(string|null)>'}

    def __init__(self, condition=Undefined, value=Undefined, **kwds):
        super(ValueDefWithConditionMarkPropFieldOrDatumDefstringnull, self).__init__(condition=condition,
                                                                                     value=value, **kwds)


class ValueDefWithConditionStringFieldDefText(TextDef):
    """ValueDefWithConditionStringFieldDefText schema wrapper

    Mapping(required=[])

    Attributes
    ----------

    condition : anyOf(:class:`ConditionalStringFieldDef`,
    :class:`ConditionalValueDefTextExprRef`, List(:class:`ConditionalValueDefTextExprRef`))
        A field definition or one or more value definition(s) with a selection predicate.
    value : anyOf(:class:`Text`, :class:`ExprRef`)
        A constant value in visual domain (e.g., ``"red"`` / ``"#0099ff"`` / `gradient
        definition <https://vega.github.io/vega-lite/docs/types.html#gradient>`__ for color,
        values between ``0`` to ``1`` for opacity).
    """
    _schema = {'$ref': '#/definitions/ValueDefWithCondition<StringFieldDef,Text>'}

    def __init__(self, condition=Undefined, value=Undefined, **kwds):
        super(ValueDefWithConditionStringFieldDefText, self).__init__(condition=condition, value=value,
                                                                      **kwds)


class ValueDefnumber(VegaLiteSchema):
    """ValueDefnumber schema wrapper

    Mapping(required=[value])
    Definition object for a constant value (primitive value or gradient definition) of an
    encoding channel.

    Attributes
    ----------

    value : float
        A constant value in visual domain (e.g., ``"red"`` / ``"#0099ff"`` / `gradient
        definition <https://vega.github.io/vega-lite/docs/types.html#gradient>`__ for color,
        values between ``0`` to ``1`` for opacity).
    """
    _schema = {'$ref': '#/definitions/ValueDef<number>'}

    def __init__(self, value=Undefined, **kwds):
        super(ValueDefnumber, self).__init__(value=value, **kwds)


class ValueDefnumberExprRef(VegaLiteSchema):
    """ValueDefnumberExprRef schema wrapper

    Mapping(required=[value])
    Definition object for a constant value (primitive value or gradient definition) of an
    encoding channel.

    Attributes
    ----------

    value : anyOf(float, :class:`ExprRef`)
        A constant value in visual domain (e.g., ``"red"`` / ``"#0099ff"`` / `gradient
        definition <https://vega.github.io/vega-lite/docs/types.html#gradient>`__ for color,
        values between ``0`` to ``1`` for opacity).
    """
    _schema = {'$ref': '#/definitions/ValueDef<(number|ExprRef)>'}

    def __init__(self, value=Undefined, **kwds):
        super(ValueDefnumberExprRef, self).__init__(value=value, **kwds)


class ValueDefnumberwidthheightExprRef(VegaLiteSchema):
    """ValueDefnumberwidthheightExprRef schema wrapper

    Mapping(required=[value])
    Definition object for a constant value (primitive value or gradient definition) of an
    encoding channel.

    Attributes
    ----------

    value : anyOf(float, string, string, :class:`ExprRef`)
        A constant value in visual domain (e.g., ``"red"`` / ``"#0099ff"`` / `gradient
        definition <https://vega.github.io/vega-lite/docs/types.html#gradient>`__ for color,
        values between ``0`` to ``1`` for opacity).
    """
    _schema = {'$ref': '#/definitions/ValueDef<(number|"width"|"height"|ExprRef)>'}

    def __init__(self, value=Undefined, **kwds):
        super(ValueDefnumberwidthheightExprRef, self).__init__(value=value, **kwds)


class Vector2DateTime(SelectionInitInterval):
    """Vector2DateTime schema wrapper

    List([:class:`DateTime`, :class:`DateTime`])
    """
    _schema = {'$ref': '#/definitions/Vector2<DateTime>'}

    def __init__(self, *args):
        super(Vector2DateTime, self).__init__(*args)


class Vector2Vector2number(VegaLiteSchema):
    """Vector2Vector2number schema wrapper

    List([:class:`Vector2number`, :class:`Vector2number`])
    """
    _schema = {'$ref': '#/definitions/Vector2<Vector2<number>>'}

    def __init__(self, *args):
        super(Vector2Vector2number, self).__init__(*args)


class Vector2boolean(SelectionInitInterval):
    """Vector2boolean schema wrapper

    List([boolean, boolean])
    """
    _schema = {'$ref': '#/definitions/Vector2<boolean>'}

    def __init__(self, *args):
        super(Vector2boolean, self).__init__(*args)


class Vector2number(SelectionInitInterval):
    """Vector2number schema wrapper

    List([float, float])
    """
    _schema = {'$ref': '#/definitions/Vector2<number>'}

    def __init__(self, *args):
        super(Vector2number, self).__init__(*args)


class Vector2string(SelectionInitInterval):
    """Vector2string schema wrapper

    List([string, string])
    """
    _schema = {'$ref': '#/definitions/Vector2<string>'}

    def __init__(self, *args):
        super(Vector2string, self).__init__(*args)


class Vector3number(VegaLiteSchema):
    """Vector3number schema wrapper

    List([float, float, float])
    """
    _schema = {'$ref': '#/definitions/Vector3<number>'}

    def __init__(self, *args):
        super(Vector3number, self).__init__(*args)


class ViewBackground(VegaLiteSchema):
    """ViewBackground schema wrapper

    Mapping(required=[])

    Attributes
    ----------

    cornerRadius : anyOf(float, :class:`ExprRef`)

    cursor : :class:`Cursor`
        The mouse cursor used over the view. Any valid `CSS cursor type
        <https://developer.mozilla.org/en-US/docs/Web/CSS/cursor#Values>`__ can be used.
    fill : anyOf(:class:`Color`, None, :class:`ExprRef`)
        The fill color.

        **Default value:** ``undefined``
    fillOpacity : anyOf(float, :class:`ExprRef`)

    opacity : anyOf(float, :class:`ExprRef`)
        The overall opacity (value between [0,1]).

        **Default value:** ``0.7`` for non-aggregate plots with ``point``, ``tick``,
        ``circle``, or ``square`` marks or layered ``bar`` charts and ``1`` otherwise.
    stroke : anyOf(:class:`Color`, None, :class:`ExprRef`)
        The stroke color.

        **Default value:** ``"#ddd"``
    strokeCap : anyOf(:class:`StrokeCap`, :class:`ExprRef`)

    strokeDash : anyOf(List(float), :class:`ExprRef`)

    strokeDashOffset : anyOf(float, :class:`ExprRef`)

    strokeJoin : anyOf(:class:`StrokeJoin`, :class:`ExprRef`)

    strokeMiterLimit : anyOf(float, :class:`ExprRef`)

    strokeOpacity : anyOf(float, :class:`ExprRef`)

    strokeWidth : anyOf(float, :class:`ExprRef`)

    style : anyOf(string, List(string))
        A string or array of strings indicating the name of custom styles to apply to the
        view background. A style is a named collection of mark property defaults defined
        within the `style configuration
        <https://vega.github.io/vega-lite/docs/mark.html#style-config>`__. If style is an
        array, later styles will override earlier styles.

        **Default value:** ``"cell"`` **Note:** Any specified view background properties
        will augment the default style.
    """
    _schema = {'$ref': '#/definitions/ViewBackground'}

    def __init__(self, cornerRadius=Undefined, cursor=Undefined, fill=Undefined, fillOpacity=Undefined,
                 opacity=Undefined, stroke=Undefined, strokeCap=Undefined, strokeDash=Undefined,
                 strokeDashOffset=Undefined, strokeJoin=Undefined, strokeMiterLimit=Undefined,
                 strokeOpacity=Undefined, strokeWidth=Undefined, style=Undefined, **kwds):
        super(ViewBackground, self).__init__(cornerRadius=cornerRadius, cursor=cursor, fill=fill,
                                             fillOpacity=fillOpacity, opacity=opacity, stroke=stroke,
                                             strokeCap=strokeCap, strokeDash=strokeDash,
                                             strokeDashOffset=strokeDashOffset, strokeJoin=strokeJoin,
                                             strokeMiterLimit=strokeMiterLimit,
                                             strokeOpacity=strokeOpacity, strokeWidth=strokeWidth,
                                             style=style, **kwds)


class ViewConfig(VegaLiteSchema):
    """ViewConfig schema wrapper

    Mapping(required=[])

    Attributes
    ----------

    clip : boolean
        Whether the view should be clipped.
    continuousHeight : float
        The default height when the plot has a continuous y-field for x or latitude, or has
        arc marks.

        **Default value:** ``200``
    continuousWidth : float
        The default width when the plot has a continuous field for x or longitude, or has
        arc marks.

        **Default value:** ``200``
    cornerRadius : anyOf(float, :class:`ExprRef`)

    cursor : :class:`Cursor`
        The mouse cursor used over the view. Any valid `CSS cursor type
        <https://developer.mozilla.org/en-US/docs/Web/CSS/cursor#Values>`__ can be used.
    discreteHeight : anyOf(float, Mapping(required=[step]))
        The default height when the plot has non arc marks and either a discrete y-field or
        no y-field. The height can be either a number indicating a fixed height or an object
        in the form of ``{step: number}`` defining the height per discrete step.

        **Default value:** a step size based on ``config.view.step``.
    discreteWidth : anyOf(float, Mapping(required=[step]))
        The default width when the plot has non-arc marks and either a discrete x-field or
        no x-field. The width can be either a number indicating a fixed width or an object
        in the form of ``{step: number}`` defining the width per discrete step.

        **Default value:** a step size based on ``config.view.step``.
    fill : anyOf(:class:`Color`, None, :class:`ExprRef`)
        The fill color.

        **Default value:** ``undefined``
    fillOpacity : anyOf(float, :class:`ExprRef`)

    height : float
        Default height

        **Deprecated:** Since Vega-Lite 4.0. Please use continuousHeight and discreteHeight
        instead.
    opacity : anyOf(float, :class:`ExprRef`)
        The overall opacity (value between [0,1]).

        **Default value:** ``0.7`` for non-aggregate plots with ``point``, ``tick``,
        ``circle``, or ``square`` marks or layered ``bar`` charts and ``1`` otherwise.
    step : float
        Default step size for x-/y- discrete fields.
    stroke : anyOf(:class:`Color`, None, :class:`ExprRef`)
        The stroke color.

        **Default value:** ``"#ddd"``
    strokeCap : anyOf(:class:`StrokeCap`, :class:`ExprRef`)

    strokeDash : anyOf(List(float), :class:`ExprRef`)

    strokeDashOffset : anyOf(float, :class:`ExprRef`)

    strokeJoin : anyOf(:class:`StrokeJoin`, :class:`ExprRef`)

    strokeMiterLimit : anyOf(float, :class:`ExprRef`)

    strokeOpacity : anyOf(float, :class:`ExprRef`)

    strokeWidth : anyOf(float, :class:`ExprRef`)

    width : float
        Default width

        **Deprecated:** Since Vega-Lite 4.0. Please use continuousWidth and discreteWidth
        instead.
    """
    _schema = {'$ref': '#/definitions/ViewConfig'}

    def __init__(self, clip=Undefined, continuousHeight=Undefined, continuousWidth=Undefined,
                 cornerRadius=Undefined, cursor=Undefined, discreteHeight=Undefined,
                 discreteWidth=Undefined, fill=Undefined, fillOpacity=Undefined, height=Undefined,
                 opacity=Undefined, step=Undefined, stroke=Undefined, strokeCap=Undefined,
                 strokeDash=Undefined, strokeDashOffset=Undefined, strokeJoin=Undefined,
                 strokeMiterLimit=Undefined, strokeOpacity=Undefined, strokeWidth=Undefined,
                 width=Undefined, **kwds):
        super(ViewConfig, self).__init__(clip=clip, continuousHeight=continuousHeight,
                                         continuousWidth=continuousWidth, cornerRadius=cornerRadius,
                                         cursor=cursor, discreteHeight=discreteHeight,
                                         discreteWidth=discreteWidth, fill=fill,
                                         fillOpacity=fillOpacity, height=height, opacity=opacity,
                                         step=step, stroke=stroke, strokeCap=strokeCap,
                                         strokeDash=strokeDash, strokeDashOffset=strokeDashOffset,
                                         strokeJoin=strokeJoin, strokeMiterLimit=strokeMiterLimit,
                                         strokeOpacity=strokeOpacity, strokeWidth=strokeWidth,
                                         width=width, **kwds)


class WindowEventType(VegaLiteSchema):
    """WindowEventType schema wrapper

    anyOf(:class:`EventType`, string)
    """
    _schema = {'$ref': '#/definitions/WindowEventType'}

    def __init__(self, *args, **kwds):
        super(WindowEventType, self).__init__(*args, **kwds)


class EventType(WindowEventType):
    """EventType schema wrapper

    enum('click', 'dblclick', 'dragenter', 'dragleave', 'dragover', 'keydown', 'keypress',
    'keyup', 'mousedown', 'mousemove', 'mouseout', 'mouseover', 'mouseup', 'mousewheel',
    'timer', 'touchend', 'touchmove', 'touchstart', 'wheel')
    """
    _schema = {'$ref': '#/definitions/EventType'}

    def __init__(self, *args):
        super(EventType, self).__init__(*args)


class WindowFieldDef(VegaLiteSchema):
    """WindowFieldDef schema wrapper

    Mapping(required=[op, as])

    Attributes
    ----------

    op : anyOf(:class:`AggregateOp`, :class:`WindowOnlyOp`)
        The window or aggregation operation to apply within a window (e.g., ``"rank"``,
        ``"lead"``, ``"sum"``, ``"average"`` or ``"count"`` ). See the list of all supported
        operations `here <https://vega.github.io/vega-lite/docs/window.html#ops>`__.
    field : :class:`FieldName`
        The data field for which to compute the aggregate or window function. This can be
        omitted for window functions that do not operate over a field such as ``"count"``,
        ``"rank"``, ``"dense_rank"``.
    param : float
        Parameter values for the window functions. Parameter values can be omitted for
        operations that do not accept a parameter.

        See the list of all supported operations and their parameters `here
        <https://vega.github.io/vega-lite/docs/transforms/window.html>`__.
    as : :class:`FieldName`
        The output name for the window operation.
    """
    _schema = {'$ref': '#/definitions/WindowFieldDef'}

    def __init__(self, op=Undefined, field=Undefined, param=Undefined, **kwds):
        super(WindowFieldDef, self).__init__(op=op, field=field, param=param, **kwds)


class WindowOnlyOp(VegaLiteSchema):
    """WindowOnlyOp schema wrapper

    enum('row_number', 'rank', 'dense_rank', 'percent_rank', 'cume_dist', 'ntile', 'lag',
    'lead', 'first_value', 'last_value', 'nth_value')
    """
    _schema = {'$ref': '#/definitions/WindowOnlyOp'}

    def __init__(self, *args):
        super(WindowOnlyOp, self).__init__(*args)


class WindowTransform(Transform):
    """WindowTransform schema wrapper

    Mapping(required=[window])

    Attributes
    ----------

    window : List(:class:`WindowFieldDef`)
        The definition of the fields in the window, and what calculations to use.
    frame : List(anyOf(None, float))
        A frame specification as a two-element array indicating how the sliding window
        should proceed. The array entries should either be a number indicating the offset
        from the current data object, or null to indicate unbounded rows preceding or
        following the current data object. The default value is ``[null, 0]``, indicating
        that the sliding window includes the current object and all preceding objects. The
        value ``[-5, 5]`` indicates that the window should include five objects preceding
        and five objects following the current object. Finally, ``[null, null]`` indicates
        that the window frame should always include all data objects. If you this frame and
        want to assign the same value to add objects, you can use the simpler `join
        aggregate transform <https://vega.github.io/vega-lite/docs/joinaggregate.html>`__.
        The only operators affected are the aggregation operations and the ``first_value``,
        ``last_value``, and ``nth_value`` window operations. The other window operations are
        not affected by this.

        **Default value:** :  ``[null, 0]`` (includes the current object and all preceding
        objects)
    groupby : List(:class:`FieldName`)
        The data fields for partitioning the data objects into separate windows. If
        unspecified, all data points will be in a single window.
    ignorePeers : boolean
        Indicates if the sliding window frame should ignore peer values (data that are
        considered identical by the sort criteria). The default is false, causing the window
        frame to expand to include all peer values. If set to true, the window frame will be
        defined by offset values only. This setting only affects those operations that
        depend on the window frame, namely aggregation operations and the first_value,
        last_value, and nth_value window operations.

        **Default value:** ``false``
    sort : List(:class:`SortField`)
        A sort field definition for sorting data objects within a window. If two data
        objects are considered equal by the comparator, they are considered "peer" values of
        equal rank. If sort is not specified, the order is undefined: data objects are
        processed in the order they are observed and none are considered peers (the
        ignorePeers parameter is ignored and treated as if set to ``true`` ).
    """
    _schema = {'$ref': '#/definitions/WindowTransform'}

    def __init__(self, window=Undefined, frame=Undefined, groupby=Undefined, ignorePeers=Undefined,
                 sort=Undefined, **kwds):
        super(WindowTransform, self).__init__(window=window, frame=frame, groupby=groupby,
                                              ignorePeers=ignorePeers, sort=sort, **kwds)

