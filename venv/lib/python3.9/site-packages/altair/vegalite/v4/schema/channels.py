# The contents of this file are automatically written by
# tools/generate_schema_wrapper.py. Do not modify directly.

from . import core
import pandas as pd
from altair.utils.schemapi import Undefined
from altair.utils import parse_shorthand


class FieldChannelMixin(object):
    def to_dict(self, validate=True, ignore=(), context=None):
        context = context or {}
        shorthand = self._get('shorthand')
        field = self._get('field')

        if shorthand is not Undefined and field is not Undefined:
            raise ValueError("{} specifies both shorthand={} and field={}. "
                             "".format(self.__class__.__name__, shorthand, field))

        if isinstance(shorthand, (tuple, list)):
            # If given a list of shorthands, then transform it to a list of classes
            kwds = self._kwds.copy()
            kwds.pop('shorthand')
            return [self.__class__(sh, **kwds).to_dict(validate=validate, ignore=ignore, context=context)
                    for sh in shorthand]

        if shorthand is Undefined:
            parsed = {}
        elif isinstance(shorthand, str):
            parsed = parse_shorthand(shorthand, data=context.get('data', None))
            type_required = 'type' in self._kwds
            type_in_shorthand = 'type' in parsed
            type_defined_explicitly = self._get('type') is not Undefined
            if not type_required:
                # Secondary field names don't require a type argument in VegaLite 3+.
                # We still parse it out of the shorthand, but drop it here.
                parsed.pop('type', None)
            elif not (type_in_shorthand or type_defined_explicitly):
                if isinstance(context.get('data', None), pd.DataFrame):
                    raise ValueError("{} encoding field is specified without a type; "
                                     "the type cannot be inferred because it does not "
                                     "match any column in the data.".format(shorthand))
                else:
                    raise ValueError("{} encoding field is specified without a type; "
                                     "the type cannot be automatically inferred because "
                                     "the data is not specified as a pandas.DataFrame."
                                     "".format(shorthand))
        else:
            # Shorthand is not a string; we pass the definition to field,
            # and do not do any parsing.
            parsed = {'field': shorthand}

        # Set shorthand to Undefined, because it's not part of the base schema.
        self.shorthand = Undefined
        self._kwds.update({k: v for k, v in parsed.items()
                           if self._get(k) is Undefined})
        return super(FieldChannelMixin, self).to_dict(
            validate=validate,
            ignore=ignore,
            context=context
        )


class ValueChannelMixin(object):
    def to_dict(self, validate=True, ignore=(), context=None):
        context = context or {}
        condition = getattr(self, 'condition', Undefined)
        copy = self  # don't copy unless we need to
        if condition is not Undefined:
            if isinstance(condition, core.SchemaBase):
                pass
            elif 'field' in condition and 'type' not in condition:
                kwds = parse_shorthand(condition['field'], context.get('data', None))
                copy = self.copy(deep=['condition'])
                copy.condition.update(kwds)
        return super(ValueChannelMixin, copy).to_dict(validate=validate,
                                                      ignore=ignore,
                                                      context=context)


class DatumChannelMixin(object):
    def to_dict(self, validate=True, ignore=(), context=None):
        context = context or {}
        datum = getattr(self, 'datum', Undefined)
        copy = self  # don't copy unless we need to
        if datum is not Undefined:
            if isinstance(datum, core.SchemaBase):
                pass
        return super(DatumChannelMixin, copy).to_dict(validate=validate,
                                                      ignore=ignore,
                                                      context=context)


class Angle(FieldChannelMixin, core.FieldOrDatumDefWithConditionMarkPropFieldDefnumber):
    """Angle schema wrapper

    Mapping(required=[shorthand])
    A FieldDef with Condition :raw-html:`<ValueDef>` {    condition: {value: ...},    field:
    ...,    ... }

    Attributes
    ----------

    shorthand : string
        shorthand for field, aggregate, and type
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
    _class_is_valid_at_instantiation = False
    _encoding_name = "angle"

    def __init__(self, shorthand=Undefined, aggregate=Undefined, band=Undefined, bin=Undefined,
                 condition=Undefined, field=Undefined, legend=Undefined, scale=Undefined,
                 sort=Undefined, timeUnit=Undefined, title=Undefined, type=Undefined, **kwds):
        super(Angle, self).__init__(shorthand=shorthand, aggregate=aggregate, band=band, bin=bin,
                                    condition=condition, field=field, legend=legend, scale=scale,
                                    sort=sort, timeUnit=timeUnit, title=title, type=type, **kwds)


class AngleDatum(DatumChannelMixin, core.FieldOrDatumDefWithConditionDatumDefnumber):
    """AngleDatum schema wrapper

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
    _class_is_valid_at_instantiation = False
    _encoding_name = "angle"
    def __init__(self, datum, band=Undefined, condition=Undefined, type=Undefined, **kwds):
        super(AngleDatum, self).__init__(datum=datum, band=band, condition=condition, type=type, **kwds)


class AngleValue(ValueChannelMixin, core.ValueDefWithConditionMarkPropFieldOrDatumDefnumber):
    """AngleValue schema wrapper

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
    _class_is_valid_at_instantiation = False
    _encoding_name = "angle"

    def __init__(self, value, condition=Undefined, **kwds):
        super(AngleValue, self).__init__(value=value, condition=condition, **kwds)


class Color(FieldChannelMixin, core.FieldOrDatumDefWithConditionMarkPropFieldDefGradientstringnull):
    """Color schema wrapper

    Mapping(required=[shorthand])
    A FieldDef with Condition :raw-html:`<ValueDef>` {    condition: {value: ...},    field:
    ...,    ... }

    Attributes
    ----------

    shorthand : string
        shorthand for field, aggregate, and type
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
    _class_is_valid_at_instantiation = False
    _encoding_name = "color"

    def __init__(self, shorthand=Undefined, aggregate=Undefined, band=Undefined, bin=Undefined,
                 condition=Undefined, field=Undefined, legend=Undefined, scale=Undefined,
                 sort=Undefined, timeUnit=Undefined, title=Undefined, type=Undefined, **kwds):
        super(Color, self).__init__(shorthand=shorthand, aggregate=aggregate, band=band, bin=bin,
                                    condition=condition, field=field, legend=legend, scale=scale,
                                    sort=sort, timeUnit=timeUnit, title=title, type=type, **kwds)


class ColorDatum(DatumChannelMixin, core.FieldOrDatumDefWithConditionDatumDefGradientstringnull):
    """ColorDatum schema wrapper

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
    _class_is_valid_at_instantiation = False
    _encoding_name = "color"
    def __init__(self, datum, band=Undefined, condition=Undefined, type=Undefined, **kwds):
        super(ColorDatum, self).__init__(datum=datum, band=band, condition=condition, type=type, **kwds)


class ColorValue(ValueChannelMixin, core.ValueDefWithConditionMarkPropFieldOrDatumDefGradientstringnull):
    """ColorValue schema wrapper

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
    _class_is_valid_at_instantiation = False
    _encoding_name = "color"

    def __init__(self, value, condition=Undefined, **kwds):
        super(ColorValue, self).__init__(value=value, condition=condition, **kwds)


class Column(FieldChannelMixin, core.RowColumnEncodingFieldDef):
    """Column schema wrapper

    Mapping(required=[shorthand])

    Attributes
    ----------

    shorthand : string
        shorthand for field, aggregate, and type
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
    _class_is_valid_at_instantiation = False
    _encoding_name = "column"

    def __init__(self, shorthand=Undefined, aggregate=Undefined, align=Undefined, band=Undefined,
                 bin=Undefined, center=Undefined, field=Undefined, header=Undefined, sort=Undefined,
                 spacing=Undefined, timeUnit=Undefined, title=Undefined, type=Undefined, **kwds):
        super(Column, self).__init__(shorthand=shorthand, aggregate=aggregate, align=align, band=band,
                                     bin=bin, center=center, field=field, header=header, sort=sort,
                                     spacing=spacing, timeUnit=timeUnit, title=title, type=type, **kwds)


class Description(FieldChannelMixin, core.StringFieldDefWithCondition):
    """Description schema wrapper

    Mapping(required=[shorthand])
    A FieldDef with Condition :raw-html:`<ValueDef>` {    condition: {value: ...},    field:
    ...,    ... }

    Attributes
    ----------

    shorthand : string
        shorthand for field, aggregate, and type
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
    _class_is_valid_at_instantiation = False
    _encoding_name = "description"

    def __init__(self, shorthand=Undefined, aggregate=Undefined, band=Undefined, bin=Undefined,
                 condition=Undefined, field=Undefined, format=Undefined, formatType=Undefined,
                 labelExpr=Undefined, timeUnit=Undefined, title=Undefined, type=Undefined, **kwds):
        super(Description, self).__init__(shorthand=shorthand, aggregate=aggregate, band=band, bin=bin,
                                          condition=condition, field=field, format=format,
                                          formatType=formatType, labelExpr=labelExpr, timeUnit=timeUnit,
                                          title=title, type=type, **kwds)


class DescriptionValue(ValueChannelMixin, core.StringValueDefWithCondition):
    """DescriptionValue schema wrapper

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
    _class_is_valid_at_instantiation = False
    _encoding_name = "description"

    def __init__(self, value, condition=Undefined, **kwds):
        super(DescriptionValue, self).__init__(value=value, condition=condition, **kwds)


class Detail(FieldChannelMixin, core.FieldDefWithoutScale):
    """Detail schema wrapper

    Mapping(required=[shorthand])
    Definition object for a data field, its type and transformation of an encoding channel.

    Attributes
    ----------

    shorthand : string
        shorthand for field, aggregate, and type
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
    _class_is_valid_at_instantiation = False
    _encoding_name = "detail"

    def __init__(self, shorthand=Undefined, aggregate=Undefined, band=Undefined, bin=Undefined,
                 field=Undefined, timeUnit=Undefined, title=Undefined, type=Undefined, **kwds):
        super(Detail, self).__init__(shorthand=shorthand, aggregate=aggregate, band=band, bin=bin,
                                     field=field, timeUnit=timeUnit, title=title, type=type, **kwds)


class Facet(FieldChannelMixin, core.FacetEncodingFieldDef):
    """Facet schema wrapper

    Mapping(required=[shorthand])

    Attributes
    ----------

    shorthand : string
        shorthand for field, aggregate, and type
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
    _class_is_valid_at_instantiation = False
    _encoding_name = "facet"

    def __init__(self, shorthand=Undefined, aggregate=Undefined, align=Undefined, band=Undefined,
                 bin=Undefined, bounds=Undefined, center=Undefined, columns=Undefined, field=Undefined,
                 header=Undefined, sort=Undefined, spacing=Undefined, timeUnit=Undefined,
                 title=Undefined, type=Undefined, **kwds):
        super(Facet, self).__init__(shorthand=shorthand, aggregate=aggregate, align=align, band=band,
                                    bin=bin, bounds=bounds, center=center, columns=columns, field=field,
                                    header=header, sort=sort, spacing=spacing, timeUnit=timeUnit,
                                    title=title, type=type, **kwds)


class Fill(FieldChannelMixin, core.FieldOrDatumDefWithConditionMarkPropFieldDefGradientstringnull):
    """Fill schema wrapper

    Mapping(required=[shorthand])
    A FieldDef with Condition :raw-html:`<ValueDef>` {    condition: {value: ...},    field:
    ...,    ... }

    Attributes
    ----------

    shorthand : string
        shorthand for field, aggregate, and type
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
    _class_is_valid_at_instantiation = False
    _encoding_name = "fill"

    def __init__(self, shorthand=Undefined, aggregate=Undefined, band=Undefined, bin=Undefined,
                 condition=Undefined, field=Undefined, legend=Undefined, scale=Undefined,
                 sort=Undefined, timeUnit=Undefined, title=Undefined, type=Undefined, **kwds):
        super(Fill, self).__init__(shorthand=shorthand, aggregate=aggregate, band=band, bin=bin,
                                   condition=condition, field=field, legend=legend, scale=scale,
                                   sort=sort, timeUnit=timeUnit, title=title, type=type, **kwds)


class FillDatum(DatumChannelMixin, core.FieldOrDatumDefWithConditionDatumDefGradientstringnull):
    """FillDatum schema wrapper

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
    _class_is_valid_at_instantiation = False
    _encoding_name = "fill"
    def __init__(self, datum, band=Undefined, condition=Undefined, type=Undefined, **kwds):
        super(FillDatum, self).__init__(datum=datum, band=band, condition=condition, type=type, **kwds)


class FillValue(ValueChannelMixin, core.ValueDefWithConditionMarkPropFieldOrDatumDefGradientstringnull):
    """FillValue schema wrapper

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
    _class_is_valid_at_instantiation = False
    _encoding_name = "fill"

    def __init__(self, value, condition=Undefined, **kwds):
        super(FillValue, self).__init__(value=value, condition=condition, **kwds)


class FillOpacity(FieldChannelMixin, core.FieldOrDatumDefWithConditionMarkPropFieldDefnumber):
    """FillOpacity schema wrapper

    Mapping(required=[shorthand])
    A FieldDef with Condition :raw-html:`<ValueDef>` {    condition: {value: ...},    field:
    ...,    ... }

    Attributes
    ----------

    shorthand : string
        shorthand for field, aggregate, and type
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
    _class_is_valid_at_instantiation = False
    _encoding_name = "fillOpacity"

    def __init__(self, shorthand=Undefined, aggregate=Undefined, band=Undefined, bin=Undefined,
                 condition=Undefined, field=Undefined, legend=Undefined, scale=Undefined,
                 sort=Undefined, timeUnit=Undefined, title=Undefined, type=Undefined, **kwds):
        super(FillOpacity, self).__init__(shorthand=shorthand, aggregate=aggregate, band=band, bin=bin,
                                          condition=condition, field=field, legend=legend, scale=scale,
                                          sort=sort, timeUnit=timeUnit, title=title, type=type, **kwds)


class FillOpacityDatum(DatumChannelMixin, core.FieldOrDatumDefWithConditionDatumDefnumber):
    """FillOpacityDatum schema wrapper

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
    _class_is_valid_at_instantiation = False
    _encoding_name = "fillOpacity"
    def __init__(self, datum, band=Undefined, condition=Undefined, type=Undefined, **kwds):
        super(FillOpacityDatum, self).__init__(datum=datum, band=band, condition=condition, type=type,
                                               **kwds)


class FillOpacityValue(ValueChannelMixin, core.ValueDefWithConditionMarkPropFieldOrDatumDefnumber):
    """FillOpacityValue schema wrapper

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
    _class_is_valid_at_instantiation = False
    _encoding_name = "fillOpacity"

    def __init__(self, value, condition=Undefined, **kwds):
        super(FillOpacityValue, self).__init__(value=value, condition=condition, **kwds)


class Href(FieldChannelMixin, core.StringFieldDefWithCondition):
    """Href schema wrapper

    Mapping(required=[shorthand])
    A FieldDef with Condition :raw-html:`<ValueDef>` {    condition: {value: ...},    field:
    ...,    ... }

    Attributes
    ----------

    shorthand : string
        shorthand for field, aggregate, and type
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
    _class_is_valid_at_instantiation = False
    _encoding_name = "href"

    def __init__(self, shorthand=Undefined, aggregate=Undefined, band=Undefined, bin=Undefined,
                 condition=Undefined, field=Undefined, format=Undefined, formatType=Undefined,
                 labelExpr=Undefined, timeUnit=Undefined, title=Undefined, type=Undefined, **kwds):
        super(Href, self).__init__(shorthand=shorthand, aggregate=aggregate, band=band, bin=bin,
                                   condition=condition, field=field, format=format,
                                   formatType=formatType, labelExpr=labelExpr, timeUnit=timeUnit,
                                   title=title, type=type, **kwds)


class HrefValue(ValueChannelMixin, core.StringValueDefWithCondition):
    """HrefValue schema wrapper

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
    _class_is_valid_at_instantiation = False
    _encoding_name = "href"

    def __init__(self, value, condition=Undefined, **kwds):
        super(HrefValue, self).__init__(value=value, condition=condition, **kwds)


class Key(FieldChannelMixin, core.FieldDefWithoutScale):
    """Key schema wrapper

    Mapping(required=[shorthand])
    Definition object for a data field, its type and transformation of an encoding channel.

    Attributes
    ----------

    shorthand : string
        shorthand for field, aggregate, and type
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
    _class_is_valid_at_instantiation = False
    _encoding_name = "key"

    def __init__(self, shorthand=Undefined, aggregate=Undefined, band=Undefined, bin=Undefined,
                 field=Undefined, timeUnit=Undefined, title=Undefined, type=Undefined, **kwds):
        super(Key, self).__init__(shorthand=shorthand, aggregate=aggregate, band=band, bin=bin,
                                  field=field, timeUnit=timeUnit, title=title, type=type, **kwds)


class Latitude(FieldChannelMixin, core.LatLongFieldDef):
    """Latitude schema wrapper

    Mapping(required=[shorthand])

    Attributes
    ----------

    shorthand : string
        shorthand for field, aggregate, and type
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
    _class_is_valid_at_instantiation = False
    _encoding_name = "latitude"

    def __init__(self, shorthand=Undefined, aggregate=Undefined, band=Undefined, bin=Undefined,
                 field=Undefined, timeUnit=Undefined, title=Undefined, type=Undefined, **kwds):
        super(Latitude, self).__init__(shorthand=shorthand, aggregate=aggregate, band=band, bin=bin,
                                       field=field, timeUnit=timeUnit, title=title, type=type, **kwds)


class LatitudeDatum(DatumChannelMixin, core.DatumDef):
    """LatitudeDatum schema wrapper

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
    _class_is_valid_at_instantiation = False
    _encoding_name = "latitude"
    def __init__(self, datum, band=Undefined, type=Undefined, **kwds):
        super(LatitudeDatum, self).__init__(datum=datum, band=band, type=type, **kwds)


class LatitudeValue(ValueChannelMixin, core.NumericValueDef):
    """LatitudeValue schema wrapper

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
    _class_is_valid_at_instantiation = False
    _encoding_name = "latitude"

    def __init__(self, value, **kwds):
        super(LatitudeValue, self).__init__(value=value, **kwds)


class Latitude2(FieldChannelMixin, core.SecondaryFieldDef):
    """Latitude2 schema wrapper

    Mapping(required=[shorthand])
    A field definition of a secondary channel that shares a scale with another primary channel.
    For example, ``x2``, ``xError`` and ``xError2`` share the same scale with ``x``.

    Attributes
    ----------

    shorthand : string
        shorthand for field, aggregate, and type
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
    _class_is_valid_at_instantiation = False
    _encoding_name = "latitude2"

    def __init__(self, shorthand=Undefined, aggregate=Undefined, band=Undefined, bin=Undefined,
                 field=Undefined, timeUnit=Undefined, title=Undefined, **kwds):
        super(Latitude2, self).__init__(shorthand=shorthand, aggregate=aggregate, band=band, bin=bin,
                                        field=field, timeUnit=timeUnit, title=title, **kwds)


class Latitude2Datum(DatumChannelMixin, core.DatumDef):
    """Latitude2Datum schema wrapper

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
    _class_is_valid_at_instantiation = False
    _encoding_name = "latitude2"
    def __init__(self, datum, band=Undefined, type=Undefined, **kwds):
        super(Latitude2Datum, self).__init__(datum=datum, band=band, type=type, **kwds)


class Latitude2Value(ValueChannelMixin, core.PositionValueDef):
    """Latitude2Value schema wrapper

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
    _class_is_valid_at_instantiation = False
    _encoding_name = "latitude2"

    def __init__(self, value, **kwds):
        super(Latitude2Value, self).__init__(value=value, **kwds)


class Longitude(FieldChannelMixin, core.LatLongFieldDef):
    """Longitude schema wrapper

    Mapping(required=[shorthand])

    Attributes
    ----------

    shorthand : string
        shorthand for field, aggregate, and type
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
    _class_is_valid_at_instantiation = False
    _encoding_name = "longitude"

    def __init__(self, shorthand=Undefined, aggregate=Undefined, band=Undefined, bin=Undefined,
                 field=Undefined, timeUnit=Undefined, title=Undefined, type=Undefined, **kwds):
        super(Longitude, self).__init__(shorthand=shorthand, aggregate=aggregate, band=band, bin=bin,
                                        field=field, timeUnit=timeUnit, title=title, type=type, **kwds)


class LongitudeDatum(DatumChannelMixin, core.DatumDef):
    """LongitudeDatum schema wrapper

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
    _class_is_valid_at_instantiation = False
    _encoding_name = "longitude"
    def __init__(self, datum, band=Undefined, type=Undefined, **kwds):
        super(LongitudeDatum, self).__init__(datum=datum, band=band, type=type, **kwds)


class LongitudeValue(ValueChannelMixin, core.NumericValueDef):
    """LongitudeValue schema wrapper

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
    _class_is_valid_at_instantiation = False
    _encoding_name = "longitude"

    def __init__(self, value, **kwds):
        super(LongitudeValue, self).__init__(value=value, **kwds)


class Longitude2(FieldChannelMixin, core.SecondaryFieldDef):
    """Longitude2 schema wrapper

    Mapping(required=[shorthand])
    A field definition of a secondary channel that shares a scale with another primary channel.
    For example, ``x2``, ``xError`` and ``xError2`` share the same scale with ``x``.

    Attributes
    ----------

    shorthand : string
        shorthand for field, aggregate, and type
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
    _class_is_valid_at_instantiation = False
    _encoding_name = "longitude2"

    def __init__(self, shorthand=Undefined, aggregate=Undefined, band=Undefined, bin=Undefined,
                 field=Undefined, timeUnit=Undefined, title=Undefined, **kwds):
        super(Longitude2, self).__init__(shorthand=shorthand, aggregate=aggregate, band=band, bin=bin,
                                         field=field, timeUnit=timeUnit, title=title, **kwds)


class Longitude2Datum(DatumChannelMixin, core.DatumDef):
    """Longitude2Datum schema wrapper

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
    _class_is_valid_at_instantiation = False
    _encoding_name = "longitude2"
    def __init__(self, datum, band=Undefined, type=Undefined, **kwds):
        super(Longitude2Datum, self).__init__(datum=datum, band=band, type=type, **kwds)


class Longitude2Value(ValueChannelMixin, core.PositionValueDef):
    """Longitude2Value schema wrapper

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
    _class_is_valid_at_instantiation = False
    _encoding_name = "longitude2"

    def __init__(self, value, **kwds):
        super(Longitude2Value, self).__init__(value=value, **kwds)


class Opacity(FieldChannelMixin, core.FieldOrDatumDefWithConditionMarkPropFieldDefnumber):
    """Opacity schema wrapper

    Mapping(required=[shorthand])
    A FieldDef with Condition :raw-html:`<ValueDef>` {    condition: {value: ...},    field:
    ...,    ... }

    Attributes
    ----------

    shorthand : string
        shorthand for field, aggregate, and type
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
    _class_is_valid_at_instantiation = False
    _encoding_name = "opacity"

    def __init__(self, shorthand=Undefined, aggregate=Undefined, band=Undefined, bin=Undefined,
                 condition=Undefined, field=Undefined, legend=Undefined, scale=Undefined,
                 sort=Undefined, timeUnit=Undefined, title=Undefined, type=Undefined, **kwds):
        super(Opacity, self).__init__(shorthand=shorthand, aggregate=aggregate, band=band, bin=bin,
                                      condition=condition, field=field, legend=legend, scale=scale,
                                      sort=sort, timeUnit=timeUnit, title=title, type=type, **kwds)


class OpacityDatum(DatumChannelMixin, core.FieldOrDatumDefWithConditionDatumDefnumber):
    """OpacityDatum schema wrapper

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
    _class_is_valid_at_instantiation = False
    _encoding_name = "opacity"
    def __init__(self, datum, band=Undefined, condition=Undefined, type=Undefined, **kwds):
        super(OpacityDatum, self).__init__(datum=datum, band=band, condition=condition, type=type,
                                           **kwds)


class OpacityValue(ValueChannelMixin, core.ValueDefWithConditionMarkPropFieldOrDatumDefnumber):
    """OpacityValue schema wrapper

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
    _class_is_valid_at_instantiation = False
    _encoding_name = "opacity"

    def __init__(self, value, condition=Undefined, **kwds):
        super(OpacityValue, self).__init__(value=value, condition=condition, **kwds)


class Order(FieldChannelMixin, core.OrderFieldDef):
    """Order schema wrapper

    Mapping(required=[shorthand])

    Attributes
    ----------

    shorthand : string
        shorthand for field, aggregate, and type
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
    _class_is_valid_at_instantiation = False
    _encoding_name = "order"

    def __init__(self, shorthand=Undefined, aggregate=Undefined, band=Undefined, bin=Undefined,
                 field=Undefined, sort=Undefined, timeUnit=Undefined, title=Undefined, type=Undefined,
                 **kwds):
        super(Order, self).__init__(shorthand=shorthand, aggregate=aggregate, band=band, bin=bin,
                                    field=field, sort=sort, timeUnit=timeUnit, title=title, type=type,
                                    **kwds)


class OrderValue(ValueChannelMixin, core.OrderValueDef):
    """OrderValue schema wrapper

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
    _class_is_valid_at_instantiation = False
    _encoding_name = "order"

    def __init__(self, value, condition=Undefined, **kwds):
        super(OrderValue, self).__init__(value=value, condition=condition, **kwds)


class Radius(FieldChannelMixin, core.PositionFieldDefBase):
    """Radius schema wrapper

    Mapping(required=[shorthand])

    Attributes
    ----------

    shorthand : string
        shorthand for field, aggregate, and type
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
    _class_is_valid_at_instantiation = False
    _encoding_name = "radius"

    def __init__(self, shorthand=Undefined, aggregate=Undefined, band=Undefined, bin=Undefined,
                 field=Undefined, scale=Undefined, sort=Undefined, stack=Undefined, timeUnit=Undefined,
                 title=Undefined, type=Undefined, **kwds):
        super(Radius, self).__init__(shorthand=shorthand, aggregate=aggregate, band=band, bin=bin,
                                     field=field, scale=scale, sort=sort, stack=stack,
                                     timeUnit=timeUnit, title=title, type=type, **kwds)


class RadiusDatum(DatumChannelMixin, core.PositionDatumDefBase):
    """RadiusDatum schema wrapper

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
    _class_is_valid_at_instantiation = False
    _encoding_name = "radius"
    def __init__(self, datum, band=Undefined, scale=Undefined, stack=Undefined, type=Undefined, **kwds):
        super(RadiusDatum, self).__init__(datum=datum, band=band, scale=scale, stack=stack, type=type,
                                          **kwds)


class RadiusValue(ValueChannelMixin, core.PositionValueDef):
    """RadiusValue schema wrapper

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
    _class_is_valid_at_instantiation = False
    _encoding_name = "radius"

    def __init__(self, value, **kwds):
        super(RadiusValue, self).__init__(value=value, **kwds)


class Radius2(FieldChannelMixin, core.SecondaryFieldDef):
    """Radius2 schema wrapper

    Mapping(required=[shorthand])
    A field definition of a secondary channel that shares a scale with another primary channel.
    For example, ``x2``, ``xError`` and ``xError2`` share the same scale with ``x``.

    Attributes
    ----------

    shorthand : string
        shorthand for field, aggregate, and type
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
    _class_is_valid_at_instantiation = False
    _encoding_name = "radius2"

    def __init__(self, shorthand=Undefined, aggregate=Undefined, band=Undefined, bin=Undefined,
                 field=Undefined, timeUnit=Undefined, title=Undefined, **kwds):
        super(Radius2, self).__init__(shorthand=shorthand, aggregate=aggregate, band=band, bin=bin,
                                      field=field, timeUnit=timeUnit, title=title, **kwds)


class Radius2Datum(DatumChannelMixin, core.DatumDef):
    """Radius2Datum schema wrapper

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
    _class_is_valid_at_instantiation = False
    _encoding_name = "radius2"
    def __init__(self, datum, band=Undefined, type=Undefined, **kwds):
        super(Radius2Datum, self).__init__(datum=datum, band=band, type=type, **kwds)


class Radius2Value(ValueChannelMixin, core.PositionValueDef):
    """Radius2Value schema wrapper

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
    _class_is_valid_at_instantiation = False
    _encoding_name = "radius2"

    def __init__(self, value, **kwds):
        super(Radius2Value, self).__init__(value=value, **kwds)


class Row(FieldChannelMixin, core.RowColumnEncodingFieldDef):
    """Row schema wrapper

    Mapping(required=[shorthand])

    Attributes
    ----------

    shorthand : string
        shorthand for field, aggregate, and type
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
    _class_is_valid_at_instantiation = False
    _encoding_name = "row"

    def __init__(self, shorthand=Undefined, aggregate=Undefined, align=Undefined, band=Undefined,
                 bin=Undefined, center=Undefined, field=Undefined, header=Undefined, sort=Undefined,
                 spacing=Undefined, timeUnit=Undefined, title=Undefined, type=Undefined, **kwds):
        super(Row, self).__init__(shorthand=shorthand, aggregate=aggregate, align=align, band=band,
                                  bin=bin, center=center, field=field, header=header, sort=sort,
                                  spacing=spacing, timeUnit=timeUnit, title=title, type=type, **kwds)


class Shape(FieldChannelMixin, core.FieldOrDatumDefWithConditionMarkPropFieldDefTypeForShapestringnull):
    """Shape schema wrapper

    Mapping(required=[shorthand])
    A FieldDef with Condition :raw-html:`<ValueDef>` {    condition: {value: ...},    field:
    ...,    ... }

    Attributes
    ----------

    shorthand : string
        shorthand for field, aggregate, and type
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
    _class_is_valid_at_instantiation = False
    _encoding_name = "shape"

    def __init__(self, shorthand=Undefined, aggregate=Undefined, band=Undefined, bin=Undefined,
                 condition=Undefined, field=Undefined, legend=Undefined, scale=Undefined,
                 sort=Undefined, timeUnit=Undefined, title=Undefined, type=Undefined, **kwds):
        super(Shape, self).__init__(shorthand=shorthand, aggregate=aggregate, band=band, bin=bin,
                                    condition=condition, field=field, legend=legend, scale=scale,
                                    sort=sort, timeUnit=timeUnit, title=title, type=type, **kwds)


class ShapeDatum(DatumChannelMixin, core.FieldOrDatumDefWithConditionDatumDefstringnull):
    """ShapeDatum schema wrapper

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
    _class_is_valid_at_instantiation = False
    _encoding_name = "shape"
    def __init__(self, datum, band=Undefined, condition=Undefined, type=Undefined, **kwds):
        super(ShapeDatum, self).__init__(datum=datum, band=band, condition=condition, type=type, **kwds)


class ShapeValue(ValueChannelMixin, core.ValueDefWithConditionMarkPropFieldOrDatumDefTypeForShapestringnull):
    """ShapeValue schema wrapper

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
    _class_is_valid_at_instantiation = False
    _encoding_name = "shape"

    def __init__(self, value, condition=Undefined, **kwds):
        super(ShapeValue, self).__init__(value=value, condition=condition, **kwds)


class Size(FieldChannelMixin, core.FieldOrDatumDefWithConditionMarkPropFieldDefnumber):
    """Size schema wrapper

    Mapping(required=[shorthand])
    A FieldDef with Condition :raw-html:`<ValueDef>` {    condition: {value: ...},    field:
    ...,    ... }

    Attributes
    ----------

    shorthand : string
        shorthand for field, aggregate, and type
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
    _class_is_valid_at_instantiation = False
    _encoding_name = "size"

    def __init__(self, shorthand=Undefined, aggregate=Undefined, band=Undefined, bin=Undefined,
                 condition=Undefined, field=Undefined, legend=Undefined, scale=Undefined,
                 sort=Undefined, timeUnit=Undefined, title=Undefined, type=Undefined, **kwds):
        super(Size, self).__init__(shorthand=shorthand, aggregate=aggregate, band=band, bin=bin,
                                   condition=condition, field=field, legend=legend, scale=scale,
                                   sort=sort, timeUnit=timeUnit, title=title, type=type, **kwds)


class SizeDatum(DatumChannelMixin, core.FieldOrDatumDefWithConditionDatumDefnumber):
    """SizeDatum schema wrapper

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
    _class_is_valid_at_instantiation = False
    _encoding_name = "size"
    def __init__(self, datum, band=Undefined, condition=Undefined, type=Undefined, **kwds):
        super(SizeDatum, self).__init__(datum=datum, band=band, condition=condition, type=type, **kwds)


class SizeValue(ValueChannelMixin, core.ValueDefWithConditionMarkPropFieldOrDatumDefnumber):
    """SizeValue schema wrapper

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
    _class_is_valid_at_instantiation = False
    _encoding_name = "size"

    def __init__(self, value, condition=Undefined, **kwds):
        super(SizeValue, self).__init__(value=value, condition=condition, **kwds)


class Stroke(FieldChannelMixin, core.FieldOrDatumDefWithConditionMarkPropFieldDefGradientstringnull):
    """Stroke schema wrapper

    Mapping(required=[shorthand])
    A FieldDef with Condition :raw-html:`<ValueDef>` {    condition: {value: ...},    field:
    ...,    ... }

    Attributes
    ----------

    shorthand : string
        shorthand for field, aggregate, and type
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
    _class_is_valid_at_instantiation = False
    _encoding_name = "stroke"

    def __init__(self, shorthand=Undefined, aggregate=Undefined, band=Undefined, bin=Undefined,
                 condition=Undefined, field=Undefined, legend=Undefined, scale=Undefined,
                 sort=Undefined, timeUnit=Undefined, title=Undefined, type=Undefined, **kwds):
        super(Stroke, self).__init__(shorthand=shorthand, aggregate=aggregate, band=band, bin=bin,
                                     condition=condition, field=field, legend=legend, scale=scale,
                                     sort=sort, timeUnit=timeUnit, title=title, type=type, **kwds)


class StrokeDatum(DatumChannelMixin, core.FieldOrDatumDefWithConditionDatumDefGradientstringnull):
    """StrokeDatum schema wrapper

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
    _class_is_valid_at_instantiation = False
    _encoding_name = "stroke"
    def __init__(self, datum, band=Undefined, condition=Undefined, type=Undefined, **kwds):
        super(StrokeDatum, self).__init__(datum=datum, band=band, condition=condition, type=type, **kwds)


class StrokeValue(ValueChannelMixin, core.ValueDefWithConditionMarkPropFieldOrDatumDefGradientstringnull):
    """StrokeValue schema wrapper

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
    _class_is_valid_at_instantiation = False
    _encoding_name = "stroke"

    def __init__(self, value, condition=Undefined, **kwds):
        super(StrokeValue, self).__init__(value=value, condition=condition, **kwds)


class StrokeDash(FieldChannelMixin, core.FieldOrDatumDefWithConditionMarkPropFieldDefnumberArray):
    """StrokeDash schema wrapper

    Mapping(required=[shorthand])
    A FieldDef with Condition :raw-html:`<ValueDef>` {    condition: {value: ...},    field:
    ...,    ... }

    Attributes
    ----------

    shorthand : string
        shorthand for field, aggregate, and type
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
    _class_is_valid_at_instantiation = False
    _encoding_name = "strokeDash"

    def __init__(self, shorthand=Undefined, aggregate=Undefined, band=Undefined, bin=Undefined,
                 condition=Undefined, field=Undefined, legend=Undefined, scale=Undefined,
                 sort=Undefined, timeUnit=Undefined, title=Undefined, type=Undefined, **kwds):
        super(StrokeDash, self).__init__(shorthand=shorthand, aggregate=aggregate, band=band, bin=bin,
                                         condition=condition, field=field, legend=legend, scale=scale,
                                         sort=sort, timeUnit=timeUnit, title=title, type=type, **kwds)


class StrokeDashDatum(DatumChannelMixin, core.FieldOrDatumDefWithConditionDatumDefnumberArray):
    """StrokeDashDatum schema wrapper

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
    _class_is_valid_at_instantiation = False
    _encoding_name = "strokeDash"
    def __init__(self, datum, band=Undefined, condition=Undefined, type=Undefined, **kwds):
        super(StrokeDashDatum, self).__init__(datum=datum, band=band, condition=condition, type=type,
                                              **kwds)


class StrokeDashValue(ValueChannelMixin, core.ValueDefWithConditionMarkPropFieldOrDatumDefnumberArray):
    """StrokeDashValue schema wrapper

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
    _class_is_valid_at_instantiation = False
    _encoding_name = "strokeDash"

    def __init__(self, value, condition=Undefined, **kwds):
        super(StrokeDashValue, self).__init__(value=value, condition=condition, **kwds)


class StrokeOpacity(FieldChannelMixin, core.FieldOrDatumDefWithConditionMarkPropFieldDefnumber):
    """StrokeOpacity schema wrapper

    Mapping(required=[shorthand])
    A FieldDef with Condition :raw-html:`<ValueDef>` {    condition: {value: ...},    field:
    ...,    ... }

    Attributes
    ----------

    shorthand : string
        shorthand for field, aggregate, and type
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
    _class_is_valid_at_instantiation = False
    _encoding_name = "strokeOpacity"

    def __init__(self, shorthand=Undefined, aggregate=Undefined, band=Undefined, bin=Undefined,
                 condition=Undefined, field=Undefined, legend=Undefined, scale=Undefined,
                 sort=Undefined, timeUnit=Undefined, title=Undefined, type=Undefined, **kwds):
        super(StrokeOpacity, self).__init__(shorthand=shorthand, aggregate=aggregate, band=band,
                                            bin=bin, condition=condition, field=field, legend=legend,
                                            scale=scale, sort=sort, timeUnit=timeUnit, title=title,
                                            type=type, **kwds)


class StrokeOpacityDatum(DatumChannelMixin, core.FieldOrDatumDefWithConditionDatumDefnumber):
    """StrokeOpacityDatum schema wrapper

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
    _class_is_valid_at_instantiation = False
    _encoding_name = "strokeOpacity"
    def __init__(self, datum, band=Undefined, condition=Undefined, type=Undefined, **kwds):
        super(StrokeOpacityDatum, self).__init__(datum=datum, band=band, condition=condition, type=type,
                                                 **kwds)


class StrokeOpacityValue(ValueChannelMixin, core.ValueDefWithConditionMarkPropFieldOrDatumDefnumber):
    """StrokeOpacityValue schema wrapper

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
    _class_is_valid_at_instantiation = False
    _encoding_name = "strokeOpacity"

    def __init__(self, value, condition=Undefined, **kwds):
        super(StrokeOpacityValue, self).__init__(value=value, condition=condition, **kwds)


class StrokeWidth(FieldChannelMixin, core.FieldOrDatumDefWithConditionMarkPropFieldDefnumber):
    """StrokeWidth schema wrapper

    Mapping(required=[shorthand])
    A FieldDef with Condition :raw-html:`<ValueDef>` {    condition: {value: ...},    field:
    ...,    ... }

    Attributes
    ----------

    shorthand : string
        shorthand for field, aggregate, and type
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
    _class_is_valid_at_instantiation = False
    _encoding_name = "strokeWidth"

    def __init__(self, shorthand=Undefined, aggregate=Undefined, band=Undefined, bin=Undefined,
                 condition=Undefined, field=Undefined, legend=Undefined, scale=Undefined,
                 sort=Undefined, timeUnit=Undefined, title=Undefined, type=Undefined, **kwds):
        super(StrokeWidth, self).__init__(shorthand=shorthand, aggregate=aggregate, band=band, bin=bin,
                                          condition=condition, field=field, legend=legend, scale=scale,
                                          sort=sort, timeUnit=timeUnit, title=title, type=type, **kwds)


class StrokeWidthDatum(DatumChannelMixin, core.FieldOrDatumDefWithConditionDatumDefnumber):
    """StrokeWidthDatum schema wrapper

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
    _class_is_valid_at_instantiation = False
    _encoding_name = "strokeWidth"
    def __init__(self, datum, band=Undefined, condition=Undefined, type=Undefined, **kwds):
        super(StrokeWidthDatum, self).__init__(datum=datum, band=band, condition=condition, type=type,
                                               **kwds)


class StrokeWidthValue(ValueChannelMixin, core.ValueDefWithConditionMarkPropFieldOrDatumDefnumber):
    """StrokeWidthValue schema wrapper

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
    _class_is_valid_at_instantiation = False
    _encoding_name = "strokeWidth"

    def __init__(self, value, condition=Undefined, **kwds):
        super(StrokeWidthValue, self).__init__(value=value, condition=condition, **kwds)


class Text(FieldChannelMixin, core.FieldOrDatumDefWithConditionStringFieldDefText):
    """Text schema wrapper

    Mapping(required=[shorthand])
    A FieldDef with Condition :raw-html:`<ValueDef>` {    condition: {value: ...},    field:
    ...,    ... }

    Attributes
    ----------

    shorthand : string
        shorthand for field, aggregate, and type
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
    _class_is_valid_at_instantiation = False
    _encoding_name = "text"

    def __init__(self, shorthand=Undefined, aggregate=Undefined, band=Undefined, bin=Undefined,
                 condition=Undefined, field=Undefined, format=Undefined, formatType=Undefined,
                 labelExpr=Undefined, timeUnit=Undefined, title=Undefined, type=Undefined, **kwds):
        super(Text, self).__init__(shorthand=shorthand, aggregate=aggregate, band=band, bin=bin,
                                   condition=condition, field=field, format=format,
                                   formatType=formatType, labelExpr=labelExpr, timeUnit=timeUnit,
                                   title=title, type=type, **kwds)


class TextDatum(DatumChannelMixin, core.FieldOrDatumDefWithConditionStringDatumDefText):
    """TextDatum schema wrapper

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
    _class_is_valid_at_instantiation = False
    _encoding_name = "text"
    def __init__(self, datum, band=Undefined, condition=Undefined, format=Undefined,
                 formatType=Undefined, labelExpr=Undefined, type=Undefined, **kwds):
        super(TextDatum, self).__init__(datum=datum, band=band, condition=condition, format=format,
                                        formatType=formatType, labelExpr=labelExpr, type=type, **kwds)


class TextValue(ValueChannelMixin, core.ValueDefWithConditionStringFieldDefText):
    """TextValue schema wrapper

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
    _class_is_valid_at_instantiation = False
    _encoding_name = "text"

    def __init__(self, value, condition=Undefined, **kwds):
        super(TextValue, self).__init__(value=value, condition=condition, **kwds)


class Theta(FieldChannelMixin, core.PositionFieldDefBase):
    """Theta schema wrapper

    Mapping(required=[shorthand])

    Attributes
    ----------

    shorthand : string
        shorthand for field, aggregate, and type
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
    _class_is_valid_at_instantiation = False
    _encoding_name = "theta"

    def __init__(self, shorthand=Undefined, aggregate=Undefined, band=Undefined, bin=Undefined,
                 field=Undefined, scale=Undefined, sort=Undefined, stack=Undefined, timeUnit=Undefined,
                 title=Undefined, type=Undefined, **kwds):
        super(Theta, self).__init__(shorthand=shorthand, aggregate=aggregate, band=band, bin=bin,
                                    field=field, scale=scale, sort=sort, stack=stack, timeUnit=timeUnit,
                                    title=title, type=type, **kwds)


class ThetaDatum(DatumChannelMixin, core.PositionDatumDefBase):
    """ThetaDatum schema wrapper

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
    _class_is_valid_at_instantiation = False
    _encoding_name = "theta"
    def __init__(self, datum, band=Undefined, scale=Undefined, stack=Undefined, type=Undefined, **kwds):
        super(ThetaDatum, self).__init__(datum=datum, band=band, scale=scale, stack=stack, type=type,
                                         **kwds)


class ThetaValue(ValueChannelMixin, core.PositionValueDef):
    """ThetaValue schema wrapper

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
    _class_is_valid_at_instantiation = False
    _encoding_name = "theta"

    def __init__(self, value, **kwds):
        super(ThetaValue, self).__init__(value=value, **kwds)


class Theta2(FieldChannelMixin, core.SecondaryFieldDef):
    """Theta2 schema wrapper

    Mapping(required=[shorthand])
    A field definition of a secondary channel that shares a scale with another primary channel.
    For example, ``x2``, ``xError`` and ``xError2`` share the same scale with ``x``.

    Attributes
    ----------

    shorthand : string
        shorthand for field, aggregate, and type
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
    _class_is_valid_at_instantiation = False
    _encoding_name = "theta2"

    def __init__(self, shorthand=Undefined, aggregate=Undefined, band=Undefined, bin=Undefined,
                 field=Undefined, timeUnit=Undefined, title=Undefined, **kwds):
        super(Theta2, self).__init__(shorthand=shorthand, aggregate=aggregate, band=band, bin=bin,
                                     field=field, timeUnit=timeUnit, title=title, **kwds)


class Theta2Datum(DatumChannelMixin, core.DatumDef):
    """Theta2Datum schema wrapper

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
    _class_is_valid_at_instantiation = False
    _encoding_name = "theta2"
    def __init__(self, datum, band=Undefined, type=Undefined, **kwds):
        super(Theta2Datum, self).__init__(datum=datum, band=band, type=type, **kwds)


class Theta2Value(ValueChannelMixin, core.PositionValueDef):
    """Theta2Value schema wrapper

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
    _class_is_valid_at_instantiation = False
    _encoding_name = "theta2"

    def __init__(self, value, **kwds):
        super(Theta2Value, self).__init__(value=value, **kwds)


class Tooltip(FieldChannelMixin, core.StringFieldDefWithCondition):
    """Tooltip schema wrapper

    Mapping(required=[shorthand])
    A FieldDef with Condition :raw-html:`<ValueDef>` {    condition: {value: ...},    field:
    ...,    ... }

    Attributes
    ----------

    shorthand : string
        shorthand for field, aggregate, and type
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
    _class_is_valid_at_instantiation = False
    _encoding_name = "tooltip"

    def __init__(self, shorthand=Undefined, aggregate=Undefined, band=Undefined, bin=Undefined,
                 condition=Undefined, field=Undefined, format=Undefined, formatType=Undefined,
                 labelExpr=Undefined, timeUnit=Undefined, title=Undefined, type=Undefined, **kwds):
        super(Tooltip, self).__init__(shorthand=shorthand, aggregate=aggregate, band=band, bin=bin,
                                      condition=condition, field=field, format=format,
                                      formatType=formatType, labelExpr=labelExpr, timeUnit=timeUnit,
                                      title=title, type=type, **kwds)


class TooltipValue(ValueChannelMixin, core.StringValueDefWithCondition):
    """TooltipValue schema wrapper

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
    _class_is_valid_at_instantiation = False
    _encoding_name = "tooltip"

    def __init__(self, value, condition=Undefined, **kwds):
        super(TooltipValue, self).__init__(value=value, condition=condition, **kwds)


class Url(FieldChannelMixin, core.StringFieldDefWithCondition):
    """Url schema wrapper

    Mapping(required=[shorthand])
    A FieldDef with Condition :raw-html:`<ValueDef>` {    condition: {value: ...},    field:
    ...,    ... }

    Attributes
    ----------

    shorthand : string
        shorthand for field, aggregate, and type
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
    _class_is_valid_at_instantiation = False
    _encoding_name = "url"

    def __init__(self, shorthand=Undefined, aggregate=Undefined, band=Undefined, bin=Undefined,
                 condition=Undefined, field=Undefined, format=Undefined, formatType=Undefined,
                 labelExpr=Undefined, timeUnit=Undefined, title=Undefined, type=Undefined, **kwds):
        super(Url, self).__init__(shorthand=shorthand, aggregate=aggregate, band=band, bin=bin,
                                  condition=condition, field=field, format=format,
                                  formatType=formatType, labelExpr=labelExpr, timeUnit=timeUnit,
                                  title=title, type=type, **kwds)


class UrlValue(ValueChannelMixin, core.StringValueDefWithCondition):
    """UrlValue schema wrapper

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
    _class_is_valid_at_instantiation = False
    _encoding_name = "url"

    def __init__(self, value, condition=Undefined, **kwds):
        super(UrlValue, self).__init__(value=value, condition=condition, **kwds)


class X(FieldChannelMixin, core.PositionFieldDef):
    """X schema wrapper

    Mapping(required=[shorthand])

    Attributes
    ----------

    shorthand : string
        shorthand for field, aggregate, and type
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
    _class_is_valid_at_instantiation = False
    _encoding_name = "x"

    def __init__(self, shorthand=Undefined, aggregate=Undefined, axis=Undefined, band=Undefined,
                 bin=Undefined, field=Undefined, impute=Undefined, scale=Undefined, sort=Undefined,
                 stack=Undefined, timeUnit=Undefined, title=Undefined, type=Undefined, **kwds):
        super(X, self).__init__(shorthand=shorthand, aggregate=aggregate, axis=axis, band=band, bin=bin,
                                field=field, impute=impute, scale=scale, sort=sort, stack=stack,
                                timeUnit=timeUnit, title=title, type=type, **kwds)


class XDatum(DatumChannelMixin, core.PositionDatumDef):
    """XDatum schema wrapper

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
    _class_is_valid_at_instantiation = False
    _encoding_name = "x"
    def __init__(self, datum, axis=Undefined, band=Undefined, impute=Undefined, scale=Undefined,
                 stack=Undefined, type=Undefined, **kwds):
        super(XDatum, self).__init__(datum=datum, axis=axis, band=band, impute=impute, scale=scale,
                                     stack=stack, type=type, **kwds)


class XValue(ValueChannelMixin, core.PositionValueDef):
    """XValue schema wrapper

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
    _class_is_valid_at_instantiation = False
    _encoding_name = "x"

    def __init__(self, value, **kwds):
        super(XValue, self).__init__(value=value, **kwds)


class X2(FieldChannelMixin, core.SecondaryFieldDef):
    """X2 schema wrapper

    Mapping(required=[shorthand])
    A field definition of a secondary channel that shares a scale with another primary channel.
    For example, ``x2``, ``xError`` and ``xError2`` share the same scale with ``x``.

    Attributes
    ----------

    shorthand : string
        shorthand for field, aggregate, and type
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
    _class_is_valid_at_instantiation = False
    _encoding_name = "x2"

    def __init__(self, shorthand=Undefined, aggregate=Undefined, band=Undefined, bin=Undefined,
                 field=Undefined, timeUnit=Undefined, title=Undefined, **kwds):
        super(X2, self).__init__(shorthand=shorthand, aggregate=aggregate, band=band, bin=bin,
                                 field=field, timeUnit=timeUnit, title=title, **kwds)


class X2Datum(DatumChannelMixin, core.DatumDef):
    """X2Datum schema wrapper

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
    _class_is_valid_at_instantiation = False
    _encoding_name = "x2"
    def __init__(self, datum, band=Undefined, type=Undefined, **kwds):
        super(X2Datum, self).__init__(datum=datum, band=band, type=type, **kwds)


class X2Value(ValueChannelMixin, core.PositionValueDef):
    """X2Value schema wrapper

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
    _class_is_valid_at_instantiation = False
    _encoding_name = "x2"

    def __init__(self, value, **kwds):
        super(X2Value, self).__init__(value=value, **kwds)


class XError(FieldChannelMixin, core.SecondaryFieldDef):
    """XError schema wrapper

    Mapping(required=[shorthand])
    A field definition of a secondary channel that shares a scale with another primary channel.
    For example, ``x2``, ``xError`` and ``xError2`` share the same scale with ``x``.

    Attributes
    ----------

    shorthand : string
        shorthand for field, aggregate, and type
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
    _class_is_valid_at_instantiation = False
    _encoding_name = "xError"

    def __init__(self, shorthand=Undefined, aggregate=Undefined, band=Undefined, bin=Undefined,
                 field=Undefined, timeUnit=Undefined, title=Undefined, **kwds):
        super(XError, self).__init__(shorthand=shorthand, aggregate=aggregate, band=band, bin=bin,
                                     field=field, timeUnit=timeUnit, title=title, **kwds)


class XErrorValue(ValueChannelMixin, core.ValueDefnumber):
    """XErrorValue schema wrapper

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
    _class_is_valid_at_instantiation = False
    _encoding_name = "xError"

    def __init__(self, value, **kwds):
        super(XErrorValue, self).__init__(value=value, **kwds)


class XError2(FieldChannelMixin, core.SecondaryFieldDef):
    """XError2 schema wrapper

    Mapping(required=[shorthand])
    A field definition of a secondary channel that shares a scale with another primary channel.
    For example, ``x2``, ``xError`` and ``xError2`` share the same scale with ``x``.

    Attributes
    ----------

    shorthand : string
        shorthand for field, aggregate, and type
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
    _class_is_valid_at_instantiation = False
    _encoding_name = "xError2"

    def __init__(self, shorthand=Undefined, aggregate=Undefined, band=Undefined, bin=Undefined,
                 field=Undefined, timeUnit=Undefined, title=Undefined, **kwds):
        super(XError2, self).__init__(shorthand=shorthand, aggregate=aggregate, band=band, bin=bin,
                                      field=field, timeUnit=timeUnit, title=title, **kwds)


class XError2Value(ValueChannelMixin, core.ValueDefnumber):
    """XError2Value schema wrapper

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
    _class_is_valid_at_instantiation = False
    _encoding_name = "xError2"

    def __init__(self, value, **kwds):
        super(XError2Value, self).__init__(value=value, **kwds)


class Y(FieldChannelMixin, core.PositionFieldDef):
    """Y schema wrapper

    Mapping(required=[shorthand])

    Attributes
    ----------

    shorthand : string
        shorthand for field, aggregate, and type
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
    _class_is_valid_at_instantiation = False
    _encoding_name = "y"

    def __init__(self, shorthand=Undefined, aggregate=Undefined, axis=Undefined, band=Undefined,
                 bin=Undefined, field=Undefined, impute=Undefined, scale=Undefined, sort=Undefined,
                 stack=Undefined, timeUnit=Undefined, title=Undefined, type=Undefined, **kwds):
        super(Y, self).__init__(shorthand=shorthand, aggregate=aggregate, axis=axis, band=band, bin=bin,
                                field=field, impute=impute, scale=scale, sort=sort, stack=stack,
                                timeUnit=timeUnit, title=title, type=type, **kwds)


class YDatum(DatumChannelMixin, core.PositionDatumDef):
    """YDatum schema wrapper

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
    _class_is_valid_at_instantiation = False
    _encoding_name = "y"
    def __init__(self, datum, axis=Undefined, band=Undefined, impute=Undefined, scale=Undefined,
                 stack=Undefined, type=Undefined, **kwds):
        super(YDatum, self).__init__(datum=datum, axis=axis, band=band, impute=impute, scale=scale,
                                     stack=stack, type=type, **kwds)


class YValue(ValueChannelMixin, core.PositionValueDef):
    """YValue schema wrapper

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
    _class_is_valid_at_instantiation = False
    _encoding_name = "y"

    def __init__(self, value, **kwds):
        super(YValue, self).__init__(value=value, **kwds)


class Y2(FieldChannelMixin, core.SecondaryFieldDef):
    """Y2 schema wrapper

    Mapping(required=[shorthand])
    A field definition of a secondary channel that shares a scale with another primary channel.
    For example, ``x2``, ``xError`` and ``xError2`` share the same scale with ``x``.

    Attributes
    ----------

    shorthand : string
        shorthand for field, aggregate, and type
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
    _class_is_valid_at_instantiation = False
    _encoding_name = "y2"

    def __init__(self, shorthand=Undefined, aggregate=Undefined, band=Undefined, bin=Undefined,
                 field=Undefined, timeUnit=Undefined, title=Undefined, **kwds):
        super(Y2, self).__init__(shorthand=shorthand, aggregate=aggregate, band=band, bin=bin,
                                 field=field, timeUnit=timeUnit, title=title, **kwds)


class Y2Datum(DatumChannelMixin, core.DatumDef):
    """Y2Datum schema wrapper

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
    _class_is_valid_at_instantiation = False
    _encoding_name = "y2"
    def __init__(self, datum, band=Undefined, type=Undefined, **kwds):
        super(Y2Datum, self).__init__(datum=datum, band=band, type=type, **kwds)


class Y2Value(ValueChannelMixin, core.PositionValueDef):
    """Y2Value schema wrapper

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
    _class_is_valid_at_instantiation = False
    _encoding_name = "y2"

    def __init__(self, value, **kwds):
        super(Y2Value, self).__init__(value=value, **kwds)


class YError(FieldChannelMixin, core.SecondaryFieldDef):
    """YError schema wrapper

    Mapping(required=[shorthand])
    A field definition of a secondary channel that shares a scale with another primary channel.
    For example, ``x2``, ``xError`` and ``xError2`` share the same scale with ``x``.

    Attributes
    ----------

    shorthand : string
        shorthand for field, aggregate, and type
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
    _class_is_valid_at_instantiation = False
    _encoding_name = "yError"

    def __init__(self, shorthand=Undefined, aggregate=Undefined, band=Undefined, bin=Undefined,
                 field=Undefined, timeUnit=Undefined, title=Undefined, **kwds):
        super(YError, self).__init__(shorthand=shorthand, aggregate=aggregate, band=band, bin=bin,
                                     field=field, timeUnit=timeUnit, title=title, **kwds)


class YErrorValue(ValueChannelMixin, core.ValueDefnumber):
    """YErrorValue schema wrapper

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
    _class_is_valid_at_instantiation = False
    _encoding_name = "yError"

    def __init__(self, value, **kwds):
        super(YErrorValue, self).__init__(value=value, **kwds)


class YError2(FieldChannelMixin, core.SecondaryFieldDef):
    """YError2 schema wrapper

    Mapping(required=[shorthand])
    A field definition of a secondary channel that shares a scale with another primary channel.
    For example, ``x2``, ``xError`` and ``xError2`` share the same scale with ``x``.

    Attributes
    ----------

    shorthand : string
        shorthand for field, aggregate, and type
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
    _class_is_valid_at_instantiation = False
    _encoding_name = "yError2"

    def __init__(self, shorthand=Undefined, aggregate=Undefined, band=Undefined, bin=Undefined,
                 field=Undefined, timeUnit=Undefined, title=Undefined, **kwds):
        super(YError2, self).__init__(shorthand=shorthand, aggregate=aggregate, band=band, bin=bin,
                                      field=field, timeUnit=timeUnit, title=title, **kwds)


class YError2Value(ValueChannelMixin, core.ValueDefnumber):
    """YError2Value schema wrapper

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
    _class_is_valid_at_instantiation = False
    _encoding_name = "yError2"

    def __init__(self, value, **kwds):
        super(YError2Value, self).__init__(value=value, **kwds)
