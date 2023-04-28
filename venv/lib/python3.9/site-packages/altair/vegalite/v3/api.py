import warnings

import hashlib
import io
import json
import jsonschema
import pandas as pd
from toolz.curried import pipe as _pipe

from .schema import core, channels, mixins, Undefined, SCHEMA_URL

from .data import data_transformers
from ... import utils, expr
from .display import renderers, VEGALITE_VERSION, VEGAEMBED_VERSION, VEGA_VERSION
from .theme import themes


# ------------------------------------------------------------------------
# Data Utilities
def _dataset_name(values):
    """Generate a unique hash of the data

    Parameters
    ----------
    values : list or dict
        A list/dict representation of data values.

    Returns
    -------
    name : string
        A unique name generated from the hash of the values.
    """
    if isinstance(values, core.InlineDataset):
        values = values.to_dict()
    values_json = json.dumps(values, sort_keys=True)
    hsh = hashlib.md5(values_json.encode()).hexdigest()
    return "data-" + hsh


def _consolidate_data(data, context):
    """If data is specified inline, then move it to context['datasets']

    This function will modify context in-place, and return a new version of data
    """
    values = Undefined
    kwds = {}

    if isinstance(data, core.InlineData):
        if data.name is Undefined and data.values is not Undefined:
            values = data.values
            kwds = {"format": data.format}

    elif isinstance(data, dict):
        if "name" not in data and "values" in data:
            values = data["values"]
            kwds = {k: v for k, v in data.items() if k != "values"}

    if values is not Undefined:
        name = _dataset_name(values)
        data = core.NamedData(name=name, **kwds)
        context.setdefault("datasets", {})[name] = values

    return data


def _prepare_data(data, context=None):
    """Convert input data to data for use within schema

    Parameters
    ----------
    data :
        The input dataset in the form of a DataFrame, dictionary, altair data
        object, or other type that is recognized by the data transformers.
    context : dict (optional)
        The to_dict context in which the data is being prepared. This is used
        to keep track of information that needs to be passed up and down the
        recursive serialization routine, such as global named datasets.
    """
    if data is Undefined:
        return data

    # convert dataframes  or objects with __geo_interface__ to dict
    if isinstance(data, pd.DataFrame) or hasattr(data, "__geo_interface__"):
        data = _pipe(data, data_transformers.get())

    # convert string input to a URLData
    if isinstance(data, str):
        data = core.UrlData(data)

    # consolidate inline data to top-level datasets
    if context is not None and data_transformers.consolidate_datasets:
        data = _consolidate_data(data, context)

    # if data is still not a recognized type, then return
    if not isinstance(data, (dict, core.Data)):
        warnings.warn("data of type {} not recognized".format(type(data)))

    return data


# ------------------------------------------------------------------------
# Aliases & specializations
Bin = core.BinParams


@utils.use_signature(core.LookupData)
class LookupData(core.LookupData):
    def to_dict(self, *args, **kwargs):
        """Convert the chart to a dictionary suitable for JSON export"""
        copy = self.copy(deep=False)
        copy.data = _prepare_data(copy.data, kwargs.get("context"))
        return super(LookupData, copy).to_dict(*args, **kwargs)


@utils.use_signature(core.FacetMapping)
class FacetMapping(core.FacetMapping):
    _class_is_valid_at_instantiation = False

    def to_dict(self, *args, **kwargs):
        copy = self.copy(deep=False)
        context = kwargs.get("context", {})
        data = context.get("data", None)
        if isinstance(self.row, str):
            copy.row = core.FacetFieldDef(**utils.parse_shorthand(self.row, data))
        if isinstance(self.column, str):
            copy.column = core.FacetFieldDef(**utils.parse_shorthand(self.column, data))
        return super(FacetMapping, copy).to_dict(*args, **kwargs)


# ------------------------------------------------------------------------
# Encoding will contain channel objects that aren't valid at instantiation
core.FacetedEncoding._class_is_valid_at_instantiation = False

# ------------------------------------------------------------------------
# These are parameters that are valid at the top level, but are not valid
# for specs that are within a composite chart
# (layer, hconcat, vconcat, facet, repeat)
TOPLEVEL_ONLY_KEYS = {"background", "config", "autosize", "padding", "$schema"}


def _get_channels_mapping():
    mapping = {}
    for attr in dir(channels):
        cls = getattr(channels, attr)
        if isinstance(cls, type) and issubclass(cls, core.SchemaBase):
            mapping[cls] = attr.replace("Value", "").lower()
    return mapping


# -------------------------------------------------------------------------
# Tools for working with selections
class Selection(object):
    """A Selection object"""

    _counter = 0

    @classmethod
    def _get_name(cls):
        cls._counter += 1
        return "selector{:03d}".format(cls._counter)

    def __init__(self, name, selection):
        if name is None:
            name = self._get_name()
        self.name = name
        self.selection = selection

    def __repr__(self):
        return "Selection({0!r}, {1})".format(self.name, self.selection)

    def ref(self):
        return self.to_dict()

    def to_dict(self):
        return {
            "selection": self.name.to_dict()
            if hasattr(self.name, "to_dict")
            else self.name
        }

    def __invert__(self):
        return Selection(core.SelectionNot(**{"not": self.name}), self.selection)

    def __and__(self, other):
        if isinstance(other, Selection):
            other = other.name
        return Selection(
            core.SelectionAnd(**{"and": [self.name, other]}), self.selection
        )

    def __or__(self, other):
        if isinstance(other, Selection):
            other = other.name
        return Selection(core.SelectionOr(**{"or": [self.name, other]}), self.selection)

    def __getattr__(self, field_name):
        if field_name.startswith("__") and field_name.endswith("__"):
            raise AttributeError(field_name)
        return expr.core.GetAttrExpression(self.name, field_name)

    def __getitem__(self, field_name):
        return expr.core.GetItemExpression(self.name, field_name)


# ------------------------------------------------------------------------
# Top-Level Functions


def value(value, **kwargs):
    """Specify a value for use in an encoding"""
    return dict(value=value, **kwargs)


def selection(name=None, type=Undefined, **kwds):
    """Create a named selection.

    Parameters
    ----------
    name : string (optional)
        The name of the selection. If not specified, a unique name will be
        created.
    type : string
        The type of the selection: one of ["interval", "single", or "multi"]
    **kwds :
        additional keywords will be used to construct a SelectionDef instance
        that controls the selection.

    Returns
    -------
    selection: Selection
        The selection object that can be used in chart creation.
    """
    return Selection(name, core.SelectionDef(type=type, **kwds))


@utils.use_signature(core.IntervalSelection)
def selection_interval(**kwargs):
    """Create a selection with type='interval'"""
    return selection(type="interval", **kwargs)


@utils.use_signature(core.MultiSelection)
def selection_multi(**kwargs):
    """Create a selection with type='multi'"""
    return selection(type="multi", **kwargs)


@utils.use_signature(core.SingleSelection)
def selection_single(**kwargs):
    """Create a selection with type='single'"""
    return selection(type="single", **kwargs)


@utils.use_signature(core.Binding)
def binding(input, **kwargs):
    """A generic binding"""
    return core.Binding(input=input, **kwargs)


@utils.use_signature(core.BindCheckbox)
def binding_checkbox(**kwargs):
    """A checkbox binding"""
    return core.BindCheckbox(input="checkbox", **kwargs)


@utils.use_signature(core.BindRadioSelect)
def binding_radio(**kwargs):
    """A radio button binding"""
    return core.BindRadioSelect(input="radio", **kwargs)


@utils.use_signature(core.BindRadioSelect)
def binding_select(**kwargs):
    """A select binding"""
    return core.BindRadioSelect(input="select", **kwargs)


@utils.use_signature(core.BindRange)
def binding_range(**kwargs):
    """A range binding"""
    return core.BindRange(input="range", **kwargs)


def condition(predicate, if_true, if_false, **kwargs):
    """A conditional attribute or encoding

    Parameters
    ----------
    predicate: Selection, LogicalOperandPredicate, expr.Expression, dict, or string
        the selection predicate or test predicate for the condition.
        if a string is passed, it will be treated as a test operand.
    if_true:
        the spec or object to use if the selection predicate is true
    if_false:
        the spec or object to use if the selection predicate is false
    **kwargs:
        additional keyword args are added to the resulting dict

    Returns
    -------
    spec: dict or VegaLiteSchema
        the spec that describes the condition
    """
    test_predicates = (str, expr.Expression, core.LogicalOperandPredicate)

    if isinstance(predicate, Selection):
        condition = {"selection": predicate.name}
    elif isinstance(predicate, core.SelectionOperand):
        condition = {"selection": predicate}
    elif isinstance(predicate, test_predicates):
        condition = {"test": predicate}
    elif isinstance(predicate, dict):
        condition = predicate
    else:
        raise NotImplementedError(
            "condition predicate of type {}" "".format(type(predicate))
        )

    if isinstance(if_true, core.SchemaBase):
        # convert to dict for now; the from_dict call below will wrap this
        # dict in the appropriate schema
        if_true = if_true.to_dict()
    elif isinstance(if_true, str):
        if_true = {"shorthand": if_true}
        if_true.update(kwargs)
    condition.update(if_true)

    if isinstance(if_false, core.SchemaBase):
        # For the selection, the channel definitions all allow selections
        # already. So use this SchemaBase wrapper if possible.
        selection = if_false.copy()
        selection.condition = condition
    elif isinstance(if_false, str):
        selection = {"condition": condition, "shorthand": if_false}
        selection.update(kwargs)
    else:
        selection = dict(condition=condition, **if_false)

    return selection


# --------------------------------------------------------------------
# Top-level objects


class TopLevelMixin(mixins.ConfigMethodMixin):
    """Mixin for top-level chart objects such as Chart, LayeredChart, etc."""

    _class_is_valid_at_instantiation = False

    def to_dict(self, *args, **kwargs):
        """Convert the chart to a dictionary suitable for JSON export"""
        # We make use of three context markers:
        # - 'data' points to the data that should be referenced for column type
        #   inference.
        # - 'top_level' is a boolean flag that is assumed to be true; if it's
        #   true then a "$schema" arg is added to the dict.
        # - 'datasets' is a dict of named datasets that should be inserted
        #   in the top-level object

        # note: not a deep copy because we want datasets and data arguments to
        # be passed by reference
        context = kwargs.get("context", {}).copy()
        context.setdefault("datasets", {})
        is_top_level = context.get("top_level", True)

        copy = self.copy(deep=False)
        original_data = getattr(copy, "data", Undefined)
        copy.data = _prepare_data(original_data, context)

        if original_data is not Undefined:
            context["data"] = original_data

        # remaining to_dict calls are not at top level
        context["top_level"] = False
        kwargs["context"] = context

        try:
            dct = super(TopLevelMixin, copy).to_dict(*args, **kwargs)
        except jsonschema.ValidationError:
            dct = None

        # If we hit an error, then re-convert with validate='deep' to get
        # a more useful traceback. We don't do this by default because it's
        # much slower in the case that there are no errors.
        if dct is None:
            kwargs["validate"] = "deep"
            dct = super(TopLevelMixin, copy).to_dict(*args, **kwargs)

        # TODO: following entries are added after validation. Should they be validated?
        if is_top_level:
            # since this is top-level we add $schema if it's missing
            if "$schema" not in dct:
                dct["$schema"] = SCHEMA_URL

            # apply theme from theme registry
            the_theme = themes.get()
            dct = utils.update_nested(the_theme(), dct, copy=True)

            # update datasets
            if context["datasets"]:
                dct.setdefault("datasets", {}).update(context["datasets"])

        return dct

    def to_html(
        self,
        base_url="https://cdn.jsdelivr.net/npm/",
        output_div="vis",
        embed_options=None,
        json_kwds=None,
        fullhtml=True,
        requirejs=False,
    ):
        return utils.spec_to_html(
            self.to_dict(),
            mode="vega-lite",
            vegalite_version=VEGALITE_VERSION,
            vegaembed_version=VEGAEMBED_VERSION,
            vega_version=VEGA_VERSION,
            base_url=base_url,
            output_div=output_div,
            embed_options=embed_options,
            json_kwds=json_kwds,
            fullhtml=fullhtml,
            requirejs=requirejs,
        )

    @utils.deprecation.deprecated(
        "Chart.savechart is deprecated in favor of Chart.save"
    )
    def savechart(self, fp, format=None, **kwargs):
        """Save a chart to file in a variety of formats

        Supported formats are json, html, png, svg

        Parameters
        ----------
        fp : string filename or file-like object
            file in which to write the chart.
        format : string (optional)
            the format to write: one of ['json', 'html', 'png', 'svg'].
            If not specified, the format will be determined from the filename.
        **kwargs :
            Additional keyword arguments are passed to the output method
            associated with the specified format.

        """
        return self.save(fp, format=None, **kwargs)

    def save(
        self,
        fp,
        format=None,
        override_data_transformer=True,
        scale_factor=1.0,
        vegalite_version=VEGALITE_VERSION,
        vega_version=VEGA_VERSION,
        vegaembed_version=VEGAEMBED_VERSION,
        **kwargs,
    ):
        """Save a chart to file in a variety of formats

        Supported formats are json, html, png, svg, pdf; the last three require
        the altair_saver package to be installed.

        Parameters
        ----------
        fp : string filename or file-like object
            file in which to write the chart.
        format : string (optional)
            the format to write: one of ['json', 'html', 'png', 'svg'].
            If not specified, the format will be determined from the filename.
        override_data_transformer : boolean (optional)
            If True (default), then the save action will be done with
            the MaxRowsError disabled. If False, then do not change the data
            transformer.
        scale_factor : float
            For svg or png formats, scale the image by this factor when saving.
            This can be used to control the size or resolution of the output.
            Default is 1.0
        **kwargs :
            Additional keyword arguments are passed to the output method
            associated with the specified format.

        """
        from ...utils.save import save

        kwds = dict(
            chart=self,
            fp=fp,
            format=format,
            scale_factor=scale_factor,
            vegalite_version=vegalite_version,
            vega_version=vega_version,
            vegaembed_version=vegaembed_version,
            **kwargs,
        )

        # By default we override the data transformer. This makes it so
        # that save() will succeed even for large datasets that would
        # normally trigger a MaxRowsError
        if override_data_transformer:
            with data_transformers.disable_max_rows():
                result = save(**kwds)
        else:
            result = save(**kwds)
        return result

    # Fallback for when rendering fails; the full repr is too long to be
    # useful in nearly all cases.
    def __repr__(self):
        return "alt.{}(...)".format(self.__class__.__name__)

    # Layering and stacking
    def __add__(self, other):
        if not isinstance(other, TopLevelMixin):
            raise ValueError("Only Chart objects can be layered.")
        return layer(self, other)

    def __and__(self, other):
        if not isinstance(other, TopLevelMixin):
            raise ValueError("Only Chart objects can be concatenated.")
        return vconcat(self, other)

    def __or__(self, other):
        if not isinstance(other, TopLevelMixin):
            raise ValueError("Only Chart objects can be concatenated.")
        return hconcat(self, other)

    def repeat(
        self,
        repeat=Undefined,
        row=Undefined,
        column=Undefined,
        columns=Undefined,
        **kwargs,
    ):
        """Return a RepeatChart built from the chart

        Fields within the chart can be set to correspond to the row or
        column using `alt.repeat('row')` and `alt.repeat('column')`.

        Parameters
        ----------
        repeat : list
            a list of data column names to be repeated. This cannot be
            used along with the ``row`` or ``column`` argument.
        row : list
            a list of data column names to be mapped to the row facet
        column : list
            a list of data column names to be mapped to the column facet
        columns : int
            the maximum number of columns before wrapping. Only referenced
            if ``repeat`` is specified.
        **kwargs :
            additional keywords passed to RepeatChart.

        Returns
        -------
        chart : RepeatChart
            a repeated chart.
        """
        repeat_specified = repeat is not Undefined
        rowcol_specified = row is not Undefined or column is not Undefined

        if repeat_specified and rowcol_specified:
            raise ValueError(
                "repeat argument cannot be combined with row/column argument."
            )

        if repeat_specified:
            repeat = repeat
        else:
            repeat = core.RepeatMapping(row=row, column=column)

        return RepeatChart(spec=self, repeat=repeat, columns=columns, **kwargs)

    def properties(self, **kwargs):
        """Set top-level properties of the Chart.

        Argument names and types are the same as class initialization.
        """
        copy = self.copy(deep=False)
        for key, val in kwargs.items():
            if key == "selection" and isinstance(val, Selection):
                # For backward compatibility with old selection interface.
                setattr(copy, key, {val.name: val.selection})
            else:
                # Don't validate data, because it hasn't been processed.
                if key != "data":
                    self.validate_property(key, val)
                setattr(copy, key, val)
        return copy

    def project(
        self,
        type="mercator",
        center=Undefined,
        clipAngle=Undefined,
        clipExtent=Undefined,
        coefficient=Undefined,
        distance=Undefined,
        fraction=Undefined,
        lobes=Undefined,
        parallel=Undefined,
        precision=Undefined,
        radius=Undefined,
        ratio=Undefined,
        reflectX=Undefined,
        reflectY=Undefined,
        rotate=Undefined,
        scale=Undefined,
        spacing=Undefined,
        tilt=Undefined,
        translate=Undefined,
        **kwds,
    ):
        """Add a geographic projection to the chart.

        This is generally used either with ``mark_geoshape`` or with the
        ``latitude``/``longitude`` encodings.

        Available projection types are
        ['albers', 'albersUsa', 'azimuthalEqualArea', 'azimuthalEquidistant',
        'conicConformal', 'conicEqualArea', 'conicEquidistant', 'equalEarth', 'equirectangular',
        'gnomonic', 'identity', 'mercator', 'orthographic', 'stereographic', 'transverseMercator']

        Attributes
        ----------
        type : ProjectionType
            The cartographic projection to use. This value is case-insensitive, for example
            `"albers"` and `"Albers"` indicate the same projection type. You can find all valid
            projection types [in the
            documentation](https://vega.github.io/vega-lite/docs/projection.html#projection-types).

            **Default value:** `mercator`
        center : List(float)
            Sets the projection’s center to the specified center, a two-element array of
            longitude and latitude in degrees.

            **Default value:** `[0, 0]`
        clipAngle : float
            Sets the projection’s clipping circle radius to the specified angle in degrees. If
            `null`, switches to [antimeridian](http://bl.ocks.org/mbostock/3788999) cutting
            rather than small-circle clipping.
        clipExtent : List(List(float))
            Sets the projection’s viewport clip extent to the specified bounds in pixels. The
            extent bounds are specified as an array `[[x0, y0], [x1, y1]]`, where `x0` is the
            left-side of the viewport, `y0` is the top, `x1` is the right and `y1` is the
            bottom. If `null`, no viewport clipping is performed.
        coefficient : float

        distance : float

        fraction : float

        lobes : float

        parallel : float

        precision : Mapping(required=[length])
            Sets the threshold for the projection’s [adaptive
            resampling](http://bl.ocks.org/mbostock/3795544) to the specified value in pixels.
            This value corresponds to the [Douglas–Peucker
            distance](http://en.wikipedia.org/wiki/Ramer%E2%80%93Douglas%E2%80%93Peucker_algorithm).
             If precision is not specified, returns the projection’s current resampling
            precision which defaults to `√0.5 ≅ 0.70710…`.
        radius : float

        ratio : float

        reflectX : boolean

        reflectY : boolean

        rotate : List(float)
            Sets the projection’s three-axis rotation to the specified angles, which must be a
            two- or three-element array of numbers [`lambda`, `phi`, `gamma`] specifying the
            rotation angles in degrees about each spherical axis. (These correspond to yaw,
            pitch and roll.)

            **Default value:** `[0, 0, 0]`
        scale : float
            Sets the projection's scale (zoom) value, overriding automatic fitting.

        spacing : float

        tilt : float

        translate : List(float)
            Sets the projection's translation (pan) value, overriding automatic fitting.

        """
        projection = core.Projection(
            center=center,
            clipAngle=clipAngle,
            clipExtent=clipExtent,
            coefficient=coefficient,
            distance=distance,
            fraction=fraction,
            lobes=lobes,
            parallel=parallel,
            precision=precision,
            radius=radius,
            ratio=ratio,
            reflectX=reflectX,
            reflectY=reflectY,
            rotate=rotate,
            scale=scale,
            spacing=spacing,
            tilt=tilt,
            translate=translate,
            type=type,
            **kwds,
        )
        return self.properties(projection=projection)

    def _add_transform(self, *transforms):
        """Copy the chart and add specified transforms to chart.transform"""
        copy = self.copy(deep=["transform"])
        if copy.transform is Undefined:
            copy.transform = []
        copy.transform.extend(transforms)
        return copy

    def transform_aggregate(self, aggregate=Undefined, groupby=Undefined, **kwds):
        """
        Add an AggregateTransform to the schema.

        Parameters
        ----------
        aggregate : List(:class:`AggregatedFieldDef`)
            Array of objects that define fields to aggregate.
        groupby : List(string)
            The data fields to group by. If not specified, a single group containing all data
            objects will be used.
        **kwds :
            additional keywords are converted to aggregates using standard
            shorthand parsing.

        Returns
        -------
        self : Chart object
            returns chart to allow for chaining

        Examples
        --------
        The aggregate transform allows you to specify transforms directly using
        the same shorthand syntax as used in encodings:

        >>> import altair as alt
        >>> chart1 = alt.Chart().transform_aggregate(
        ...     mean_acc='mean(Acceleration)',
        ...     groupby=['Origin']
        ... )
        >>> print(chart1.transform[0].to_json())  # doctest: +NORMALIZE_WHITESPACE
        {
          "aggregate": [
            {
              "as": "mean_acc",
              "field": "Acceleration",
              "op": "mean"
            }
          ],
          "groupby": [
            "Origin"
          ]
        }

        It also supports including AggregatedFieldDef instances or dicts directly,
        so you can create the above transform like this:

        >>> chart2 = alt.Chart().transform_aggregate(
        ...     [alt.AggregatedFieldDef(field='Acceleration', op='mean',
        ...                             **{'as': 'mean_acc'})],
        ...     groupby=['Origin']
        ... )
        >>> chart2.transform == chart1.transform
        True

        See Also
        --------
        alt.AggregateTransform : underlying transform object

        """
        if aggregate is Undefined:
            aggregate = []
        for key, val in kwds.items():
            parsed = utils.parse_shorthand(val)
            dct = {
                "as": key,
                "field": parsed.get("field", Undefined),
                "op": parsed.get("aggregate", Undefined),
            }
            aggregate.append(core.AggregatedFieldDef(**dct))
        return self._add_transform(
            core.AggregateTransform(aggregate=aggregate, groupby=groupby)
        )

    def transform_bin(self, as_=Undefined, field=Undefined, bin=True, **kwargs):
        """
        Add a BinTransform to the schema.

        Parameters
        ----------
        as_ : anyOf(string, List(string))
            The output fields at which to write the start and end bin values.
        bin : anyOf(boolean, :class:`BinParams`)
            An object indicating bin properties, or simply ``true`` for using default bin
            parameters.
        field : string
            The data field to bin.

        Returns
        -------
        self : Chart object
            returns chart to allow for chaining

        Examples
        --------
        >>> import altair as alt
        >>> chart = alt.Chart().transform_bin("x_binned", "x")
        >>> chart.transform[0]
        BinTransform({
          as: 'x_binned',
          bin: True,
          field: 'x'
        })

        >>> chart = alt.Chart().transform_bin("x_binned", "x",
        ...                                   bin=alt.Bin(maxbins=10))
        >>> chart.transform[0]
        BinTransform({
          as: 'x_binned',
          bin: BinParams({
            maxbins: 10
          }),
          field: 'x'
        })

        See Also
        --------
        alt.BinTransform : underlying transform object

        """
        if as_ is not Undefined:
            if "as" in kwargs:
                raise ValueError(
                    "transform_bin: both 'as_' and 'as' passed as arguments."
                )
            kwargs["as"] = as_
        kwargs["bin"] = bin
        kwargs["field"] = field
        return self._add_transform(core.BinTransform(**kwargs))

    def transform_calculate(self, as_=Undefined, calculate=Undefined, **kwargs):
        """
        Add a CalculateTransform to the schema.

        Parameters
        ----------
        as_ : string
            The field for storing the computed formula value.
        calculate : string or alt.expr expression
            A `expression <https://vega.github.io/vega-lite/docs/types.html#expression>`__
            string. Use the variable ``datum`` to refer to the current data object.
        **kwargs
            transforms can also be passed by keyword argument; see Examples

        Returns
        -------
        self : Chart object
            returns chart to allow for chaining

        Examples
        --------
        >>> import altair as alt
        >>> from altair import datum, expr

        >>> chart = alt.Chart().transform_calculate(y = 2 * expr.sin(datum.x))
        >>> chart.transform[0]
        CalculateTransform({
          as: 'y',
          calculate: (2 * sin(datum.x))
        })

        It's also possible to pass the ``CalculateTransform`` arguments directly:

        >>> kwds = {'as': 'y', 'calculate': '2 * sin(datum.x)'}
        >>> chart = alt.Chart().transform_calculate(**kwds)
        >>> chart.transform[0]
        CalculateTransform({
          as: 'y',
          calculate: '2 * sin(datum.x)'
        })

        As the first form is easier to write and understand, that is the
        recommended method.

        See Also
        --------
        alt.CalculateTransform : underlying transform object
        """
        if as_ is Undefined:
            as_ = kwargs.pop("as", Undefined)
        else:
            if "as" in kwargs:
                raise ValueError(
                    "transform_calculate: both 'as_' and 'as' passed as arguments."
                )
        if as_ is not Undefined or calculate is not Undefined:
            dct = {"as": as_, "calculate": calculate}
            self = self._add_transform(core.CalculateTransform(**dct))
        for as_, calculate in kwargs.items():
            dct = {"as": as_, "calculate": calculate}
            self = self._add_transform(core.CalculateTransform(**dct))
        return self

    def transform_impute(
        self,
        impute,
        key,
        frame=Undefined,
        groupby=Undefined,
        keyvals=Undefined,
        method=Undefined,
        value=Undefined,
    ):
        """
        Add an ImputeTransform to the schema.

        Parameters
        ----------
        impute : string
            The data field for which the missing values should be imputed.
        key : string
            A key field that uniquely identifies data objects within a group.
            Missing key values (those occurring in the data but not in the current group) will
            be imputed.
        frame : List(anyOf(None, float))
            A frame specification as a two-element array used to control the window over which
            the specified method is applied. The array entries should either be a number
            indicating the offset from the current data object, or null to indicate unbounded
            rows preceding or following the current data object.  For example, the value ``[-5,
            5]`` indicates that the window should include five objects preceding and five
            objects following the current object.
            **Default value:** :  ``[null, null]`` indicating that the window includes all
            objects.
        groupby : List(string)
            An optional array of fields by which to group the values.
            Imputation will then be performed on a per-group basis.
        keyvals : anyOf(List(Mapping(required=[])), :class:`ImputeSequence`)
            Defines the key values that should be considered for imputation.
            An array of key values or an object defining a `number sequence
            <https://vega.github.io/vega-lite/docs/impute.html#sequence-def>`__.
            If provided, this will be used in addition to the key values observed within the
            input data.  If not provided, the values will be derived from all unique values of
            the ``key`` field. For ``impute`` in ``encoding``, the key field is the x-field if
            the y-field is imputed, or vice versa.
            If there is no impute grouping, this property *must* be specified.
        method : :class:`ImputeMethod`
            The imputation method to use for the field value of imputed data objects.
            One of ``value``, ``mean``, ``median``, ``max`` or ``min``.
            **Default value:**  ``"value"``
        value : Mapping(required=[])
            The field value to use when the imputation ``method`` is ``"value"``.

        Returns
        -------
        self : Chart object
            returns chart to allow for chaining

        See Also
        --------
        alt.ImputeTransform : underlying transform object
        """
        return self._add_transform(
            core.ImputeTransform(
                impute=impute,
                key=key,
                frame=frame,
                groupby=groupby,
                keyvals=keyvals,
                method=method,
                value=value,
            )
        )

    def transform_joinaggregate(
        self, joinaggregate=Undefined, groupby=Undefined, **kwargs
    ):
        """
        Add a JoinAggregateTransform to the schema.

        Parameters
        ----------
        joinaggregate : List(:class:`JoinAggregateFieldDef`)
            The definition of the fields in the join aggregate, and what calculations to use.
        groupby : List(string)
            The data fields for partitioning the data objects into separate groups. If
            unspecified, all data points will be in a single group.
        **kwargs
            joinaggregates can also be passed by keyword argument; see Examples.

        Returns
        -------
        self : Chart object
            returns chart to allow for chaining

        Examples
        --------
        >>> import altair as alt
        >>> chart = alt.Chart().transform_joinaggregate(x='sum(y)')
        >>> chart.transform[0]
        JoinAggregateTransform({
          joinaggregate: [JoinAggregateFieldDef({
            as: 'x',
            field: 'y',
            op: 'sum'
          })]
        })

        See Also
        --------
        alt.JoinAggregateTransform : underlying transform object
        """
        if joinaggregate is Undefined:
            joinaggregate = []
        for key, val in kwargs.items():
            parsed = utils.parse_shorthand(val)
            dct = {
                "as": key,
                "field": parsed.get("field", Undefined),
                "op": parsed.get("aggregate", Undefined),
            }
            joinaggregate.append(core.JoinAggregateFieldDef(**dct))
        return self._add_transform(
            core.JoinAggregateTransform(joinaggregate=joinaggregate, groupby=groupby)
        )

    def transform_filter(self, filter, **kwargs):
        """
        Add a FilterTransform to the schema.

        Parameters
        ----------
        filter : a filter expression or :class:`LogicalOperandPredicate`
            The `filter` property must be one of the predicate definitions:
            (1) a string or alt.expr expression
            (2) a range predicate
            (3) a selection predicate
            (4) a logical operand combining (1)-(3)
            (5) a Selection object

        Returns
        -------
        self : Chart object
            returns chart to allow for chaining

        See Also
        --------
        alt.FilterTransform : underlying transform object

        """
        if isinstance(filter, Selection):
            filter = {"selection": filter.name}
        elif isinstance(filter, core.SelectionOperand):
            filter = {"selection": filter}
        return self._add_transform(core.FilterTransform(filter=filter, **kwargs))

    def transform_flatten(self, flatten, as_=Undefined):
        """Add a FlattenTransform to the schema.

        Parameters
        ----------
        flatten : List(string)
            An array of one or more data fields containing arrays to flatten.
            If multiple fields are specified, their array values should have a parallel
            structure, ideally with the same length.
            If the lengths of parallel arrays do not match,
            the longest array will be used with ``null`` values added for missing entries.
        as : List(string)
            The output field names for extracted array values.
            **Default value:** The field name of the corresponding array field

        Returns
        -------
        self : Chart object
            returns chart to allow for chaining

        See Also
        --------
        alt.FlattenTransform : underlying transform object
        """
        return self._add_transform(
            core.FlattenTransform(flatten=flatten, **{"as": as_})
        )

    def transform_fold(self, fold, as_=Undefined):
        """Add a FoldTransform to the schema.

        Parameters
        ----------
        fold : List(string)
            An array of data fields indicating the properties to fold.
        as : [string, string]
            The output field names for the key and value properties produced by the fold
            transform. Default: ``["key", "value"]``

        Returns
        -------
        self : Chart object
            returns chart to allow for chaining

        See Also
        --------
        alt.FoldTransform : underlying transform object
        """
        return self._add_transform(core.FoldTransform(fold=fold, **{"as": as_}))

    def transform_lookup(
        self,
        as_=Undefined,
        from_=Undefined,
        lookup=Undefined,
        default=Undefined,
        **kwargs,
    ):
        """Add a LookupTransform to the schema

        Attributes
        ----------
        as_ : anyOf(string, List(string))
            The field or fields for storing the computed formula value.
            If ``from.fields`` is specified, the transform will use the same names for ``as``.
            If ``from.fields`` is not specified, ``as`` has to be a string and we put the whole
            object into the data under the specified name.
        from_ : :class:`LookupData`
            Secondary data reference.
        lookup : string
            Key in primary data source.
        default : string
            The default value to use if lookup fails. **Default value:** ``null``

        Returns
        -------
        self : Chart object
            returns chart to allow for chaining

        See Also
        --------
        alt.LookupTransform : underlying transform object
        """
        if as_ is not Undefined:
            if "as" in kwargs:
                raise ValueError(
                    "transform_lookup: both 'as_' and 'as' passed as arguments."
                )
            kwargs["as"] = as_
        if from_ is not Undefined:
            if "from" in kwargs:
                raise ValueError(
                    "transform_lookup: both 'from_' and 'from' passed as arguments."
                )
            kwargs["from"] = from_
        kwargs["lookup"] = lookup
        kwargs["default"] = default
        return self._add_transform(core.LookupTransform(**kwargs))

    def transform_sample(self, sample=1000):
        """
        Add a SampleTransform to the schema.

        Parameters
        ----------
        sample : float
            The maximum number of data objects to include in the sample. Default: 1000.

        Returns
        -------
        self : Chart object
            returns chart to allow for chaining

        See Also
        --------
        alt.SampleTransform : underlying transform object
        """
        return self._add_transform(core.SampleTransform(sample))

    def transform_stack(self, as_, stack, groupby, offset=Undefined, sort=Undefined):
        """
        Add a StackTransform to the schema.

        Parameters
        ----------
        as_ : anyOf(string, List(string))
            Output field names. This can be either a string or an array of strings with
            two elements denoting the name for the fields for stack start and stack end
            respectively.
            If a single string(eg."val") is provided, the end field will be "val_end".
        stack : string
            The field which is stacked.
        groupby : List(string)
            The data fields to group by.
        offset : enum('zero', 'center', 'normalize')
            Mode for stacking marks. Default: 'zero'.
        sort : List(:class:`SortField`)
            Field that determines the order of leaves in the stacked charts.

        Returns
        -------
        self : Chart object
            returns chart to allow for chaining

        See Also
        --------
        alt.StackTransform : underlying transform object
        """
        return self._add_transform(
            core.StackTransform(
                stack=stack, groupby=groupby, offset=offset, sort=sort, **{"as": as_}
            )
        )

    def transform_timeunit(
        self, as_=Undefined, field=Undefined, timeUnit=Undefined, **kwargs
    ):
        """
        Add a TimeUnitTransform to the schema.

        Parameters
        ----------
        as_ : string
            The output field to write the timeUnit value.
        field : string
            The data field to apply time unit.
        timeUnit : :class:`TimeUnit`
            The timeUnit.
        **kwargs
            transforms can also be passed by keyword argument; see Examples

        Returns
        -------
        self : Chart object
            returns chart to allow for chaining

        Examples
        --------
        >>> import altair as alt
        >>> from altair import datum, expr

        >>> chart = alt.Chart().transform_timeunit(month='month(date)')
        >>> chart.transform[0]
        TimeUnitTransform({
          as: 'month',
          field: 'date',
          timeUnit: 'month'
        })

        It's also possible to pass the ``TimeUnitTransform`` arguments directly;
        this is most useful in cases where the desired field name is not a
        valid python identifier:

        >>> kwds = {'as': 'month', 'timeUnit': 'month', 'field': 'The Month'}
        >>> chart = alt.Chart().transform_timeunit(**kwds)
        >>> chart.transform[0]
        TimeUnitTransform({
          as: 'month',
          field: 'The Month',
          timeUnit: 'month'
        })

        As the first form is easier to write and understand, that is the
        recommended method.

        See Also
        --------
        alt.TimeUnitTransform : underlying transform object

        """
        if as_ is Undefined:
            as_ = kwargs.pop("as", Undefined)
        else:
            if "as" in kwargs:
                raise ValueError(
                    "transform_timeunit: both 'as_' and 'as' passed as arguments."
                )
        if as_ is not Undefined:
            dct = {"as": as_, "timeUnit": timeUnit, "field": field}
            self = self._add_transform(core.TimeUnitTransform(**dct))
        for as_, shorthand in kwargs.items():
            dct = utils.parse_shorthand(
                shorthand,
                parse_timeunits=True,
                parse_aggregates=False,
                parse_types=False,
            )
            dct.pop("type", None)
            dct["as"] = as_
            if "timeUnit" not in dct:
                raise ValueError("'{}' must include a valid timeUnit".format(shorthand))
            self = self._add_transform(core.TimeUnitTransform(**dct))
        return self

    def transform_window(
        self,
        window=Undefined,
        frame=Undefined,
        groupby=Undefined,
        ignorePeers=Undefined,
        sort=Undefined,
        **kwargs,
    ):
        """Add a WindowTransform to the schema

        Parameters
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
            that the window frame should always include all data objects. The only operators
            affected are the aggregation operations and the ``first_value``, ``last_value``, and
            ``nth_value`` window operations. The other window operations are not affected by
            this.

            **Default value:** :  ``[null, 0]`` (includes the current object and all preceding
            objects)
        groupby : List(string)
            The data fields for partitioning the data objects into separate windows. If
            unspecified, all data points will be in a single group.
        ignorePeers : boolean
            Indicates if the sliding window frame should ignore peer values. (Peer values are
            those considered identical by the sort criteria). The default is false, causing the
            window frame to expand to include all peer values. If set to true, the window frame
            will be defined by offset values only. This setting only affects those operations
            that depend on the window frame, namely aggregation operations and the first_value,
            last_value, and nth_value window operations.

            **Default value:** ``false``
        sort : List(:class:`SortField`)
            A sort field definition for sorting data objects within a window. If two data
            objects are considered equal by the comparator, they are considered “peer” values of
            equal rank. If sort is not specified, the order is undefined: data objects are
            processed in the order they are observed and none are considered peers (the
            ignorePeers parameter is ignored and treated as if set to ``true`` ).
        **kwargs
            transforms can also be passed by keyword argument; see Examples

        Examples
        --------
        A cumulative line chart

        >>> import altair as alt
        >>> import numpy as np
        >>> import pandas as pd
        >>> data = pd.DataFrame({'x': np.arange(100),
        ...                      'y': np.random.randn(100)})
        >>> chart = alt.Chart(data).mark_line().encode(
        ...     x='x:Q',
        ...     y='ycuml:Q'
        ... ).transform_window(
        ...     ycuml='sum(y)'
        ... )
        >>> chart.transform[0]
        WindowTransform({
          window: [WindowFieldDef({
            as: 'ycuml',
            field: 'y',
            op: 'sum'
          })]
        })

        """
        if kwargs:
            if window is Undefined:
                window = []
            for as_, shorthand in kwargs.items():
                kwds = {"as": as_}
                kwds.update(
                    utils.parse_shorthand(
                        shorthand,
                        parse_aggregates=False,
                        parse_window_ops=True,
                        parse_timeunits=False,
                        parse_types=False,
                    )
                )
                window.append(core.WindowFieldDef(**kwds))

        return self._add_transform(
            core.WindowTransform(
                window=window,
                frame=frame,
                groupby=groupby,
                ignorePeers=ignorePeers,
                sort=sort,
            )
        )

    # Display-related methods

    def _repr_mimebundle_(self, include=None, exclude=None):
        """Return a MIME bundle for display in Jupyter frontends."""
        # Catch errors explicitly to get around issues in Jupyter frontend
        # see https://github.com/ipython/ipython/issues/11038
        try:
            dct = self.to_dict()
        except Exception:
            utils.display_traceback(in_ipython=True)
            return {}
        else:
            return renderers.get()(dct)

    def display(self, renderer=Undefined, theme=Undefined, actions=Undefined, **kwargs):
        """Display chart in Jupyter notebook or JupyterLab

        Parameters are passed as options to vega-embed within supported frontends.
        See https://github.com/vega/vega-embed#options for details.

        Parameters
        ----------
        renderer : string ('canvas' or 'svg')
            The renderer to use
        theme : string
            The Vega theme name to use; see https://github.com/vega/vega-themes
        actions : bool or dict
            Specify whether action links ("Open In Vega Editor", etc.) are
            included in the view.
        **kwargs :
            Additional parameters are also passed to vega-embed as options.

        """
        from IPython.display import display

        if renderer is not Undefined:
            kwargs["renderer"] = renderer
        if theme is not Undefined:
            kwargs["theme"] = theme
        if actions is not Undefined:
            kwargs["actions"] = actions

        if kwargs:
            options = renderers.options.copy()
            options["embed_options"] = options.get("embed_options", {}).copy()
            options["embed_options"].update(kwargs)
            with renderers.enable(**options):
                display(self)
        else:
            display(self)

    def serve(
        self,
        ip="127.0.0.1",
        port=8888,
        n_retries=50,
        files=None,
        jupyter_warning=True,
        open_browser=True,
        http_server=None,
        **kwargs,
    ):
        """Open a browser window and display a rendering of the chart

        Parameters
        ----------
        html : string
            HTML to serve
        ip : string (default = '127.0.0.1')
            ip address at which the HTML will be served.
        port : int (default = 8888)
            the port at which to serve the HTML
        n_retries : int (default = 50)
            the number of nearby ports to search if the specified port
            is already in use.
        files : dictionary (optional)
            dictionary of extra content to serve
        jupyter_warning : bool (optional)
            if True (default), then print a warning if this is used
            within the Jupyter notebook
        open_browser : bool (optional)
            if True (default), then open a web browser to the given HTML
        http_server : class (optional)
            optionally specify an HTTPServer class to use for showing the
            figure. The default is Python's basic HTTPServer.
        **kwargs :
            additional keyword arguments passed to the save() method

        """
        from ...utils.server import serve

        html = io.StringIO()
        self.save(html, format="html", **kwargs)
        html.seek(0)

        serve(
            html.read(),
            ip=ip,
            port=port,
            n_retries=n_retries,
            files=files,
            jupyter_warning=jupyter_warning,
            open_browser=open_browser,
            http_server=http_server,
        )

    @utils.use_signature(core.Resolve)
    def _set_resolve(self, **kwargs):
        """Copy the chart and update the resolve property with kwargs"""
        if not hasattr(self, "resolve"):
            raise ValueError(
                "{} object has no attribute " "'resolve'".format(self.__class__)
            )
        copy = self.copy(deep=["resolve"])
        if copy.resolve is Undefined:
            copy.resolve = core.Resolve()
        for key, val in kwargs.items():
            copy.resolve[key] = val
        return copy

    @utils.use_signature(core.AxisResolveMap)
    def resolve_axis(self, *args, **kwargs):
        return self._set_resolve(axis=core.AxisResolveMap(*args, **kwargs))

    @utils.use_signature(core.LegendResolveMap)
    def resolve_legend(self, *args, **kwargs):
        return self._set_resolve(legend=core.LegendResolveMap(*args, **kwargs))

    @utils.use_signature(core.ScaleResolveMap)
    def resolve_scale(self, *args, **kwargs):
        return self._set_resolve(scale=core.ScaleResolveMap(*args, **kwargs))


class _EncodingMixin(object):
    @utils.use_signature(core.FacetedEncoding)
    def encode(self, *args, **kwargs):
        # Convert args to kwargs based on their types.
        kwargs = utils.infer_encoding_types(args, kwargs, channels)

        # get a copy of the dict representation of the previous encoding
        copy = self.copy(deep=["encoding"])
        encoding = copy._get("encoding", {})
        if isinstance(encoding, core.VegaLiteSchema):
            encoding = {k: v for k, v in encoding._kwds.items() if v is not Undefined}

        # update with the new encodings, and apply them to the copy
        encoding.update(kwargs)
        copy.encoding = core.FacetedEncoding(**encoding)
        return copy

    def facet(
        self,
        facet=Undefined,
        row=Undefined,
        column=Undefined,
        data=Undefined,
        columns=Undefined,
        **kwargs,
    ):
        """Create a facet chart from the current chart.

        Faceted charts require data to be specified at the top level; if data
        is not specified, the data from the current chart will be used at the
        top level.

        Parameters
        ----------
        facet : string or alt.Facet (optional)
            The data column to use as an encoding for a wrapped facet.
            If specified, then neither row nor column may be specified.
        column : string or alt.Column (optional)
            The data column to use as an encoding for a column facet.
            May be combined with row argument, but not with facet argument.
        row : string or alt.Column (optional)
            The data column to use as an encoding for a row facet.
            May be combined with column argument, but not with facet argument.
        data : string or dataframe (optional)
            The dataset to use for faceting. If not supplied, then data must
            be specified in the top-level chart that calls this method.
        columns : integer
            the maximum number of columns for a wrapped facet.

        Returns
        -------
        self :
            for chaining
        """
        facet_specified = facet is not Undefined
        rowcol_specified = row is not Undefined or column is not Undefined

        if facet_specified and rowcol_specified:
            raise ValueError(
                "facet argument cannot be combined with row/column argument."
            )

        if data is Undefined:
            if self.data is Undefined:
                raise ValueError(
                    "Facet charts require data to be specified at the top level."
                )
            self = self.copy(deep=False)
            data, self.data = self.data, Undefined

        if facet_specified:
            if isinstance(facet, str):
                facet = channels.Facet(facet)
        else:
            facet = FacetMapping(row=row, column=column)

        return FacetChart(spec=self, facet=facet, data=data, columns=columns, **kwargs)


class Chart(
    TopLevelMixin, _EncodingMixin, mixins.MarkMethodMixin, core.TopLevelUnitSpec
):
    """Create a basic Altair/Vega-Lite chart.

    Although it is possible to set all Chart properties as constructor attributes,
    it is more idiomatic to use methods such as ``mark_point()``, ``encode()``,
    ``transform_filter()``, ``properties()``, etc. See Altair's documentation
    for details and examples: http://altair-viz.github.io/.

    Attributes
    ----------
    data : Data
        An object describing the data source
    mark : AnyMark
        A string describing the mark type (one of `"bar"`, `"circle"`, `"square"`, `"tick"`,
         `"line"`, * `"area"`, `"point"`, `"rule"`, `"geoshape"`, and `"text"`) or a
         MarkDef object.
    encoding : FacetedEncoding
        A key-value mapping between encoding channels and definition of fields.
    autosize : anyOf(AutosizeType, AutoSizeParams)
        Sets how the visualization size should be determined. If a string, should be one of
        `"pad"`, `"fit"` or `"none"`. Object values can additionally specify parameters for
        content sizing and automatic resizing. `"fit"` is only supported for single and
        layered views that don't use `rangeStep`.  __Default value__: `pad`
    background : string
        CSS color property to use as the background of visualization.

        **Default value:** none (transparent)
    config : Config
        Vega-Lite configuration object.  This property can only be defined at the top-level
        of a specification.
    description : string
        Description of this mark for commenting purpose.
    height : float
        The height of a visualization.
    name : string
        Name of the visualization for later reference.
    padding : Padding
        The default visualization padding, in pixels, from the edge of the visualization
        canvas to the data rectangle.  If a number, specifies padding for all sides. If an
        object, the value should have the format `{"left": 5, "top": 5, "right": 5,
        "bottom": 5}` to specify padding for each side of the visualization.  __Default
        value__: `5`
    projection : Projection
        An object defining properties of geographic projection.  Works with `"geoshape"`
        marks and `"point"` or `"line"` marks that have a channel (one or more of `"X"`,
        `"X2"`, `"Y"`, `"Y2"`) with type `"latitude"`, or `"longitude"`.
    selection : Mapping(required=[])
        A key-value mapping between selection names and definitions.
    title : anyOf(string, TitleParams)
        Title for the plot.
    transform : List(Transform)
        An array of data transformations such as filter and new field calculation.
    width : float
        The width of a visualization.
    """

    def __init__(
        self,
        data=Undefined,
        encoding=Undefined,
        mark=Undefined,
        width=Undefined,
        height=Undefined,
        **kwargs,
    ):
        super(Chart, self).__init__(
            data=data,
            encoding=encoding,
            mark=mark,
            width=width,
            height=height,
            **kwargs,
        )

    @classmethod
    def from_dict(cls, dct, validate=True):
        """Construct class from a dictionary representation

        Parameters
        ----------
        dct : dictionary
            The dict from which to construct the class
        validate : boolean
            If True (default), then validate the input against the schema.

        Returns
        -------
        obj : Chart object
            The wrapped schema

        Raises
        ------
        jsonschema.ValidationError :
            if validate=True and dct does not conform to the schema
        """
        for class_ in TopLevelMixin.__subclasses__():
            if class_ is Chart:
                class_ = super(Chart, cls)
            try:
                return class_.from_dict(dct, validate=validate)
            except jsonschema.ValidationError:
                pass

        # As a last resort, try using the Root vegalite object
        return core.Root.from_dict(dct, validate)

    def add_selection(self, *selections):
        """Add one or more selections to the chart."""
        if not selections:
            return self
        copy = self.copy(deep=["selection"])
        if copy.selection is Undefined:
            copy.selection = {}

        for s in selections:
            copy.selection[s.name] = s.selection
        return copy

    def interactive(self, name=None, bind_x=True, bind_y=True):
        """Make chart axes scales interactive

        Parameters
        ----------
        name : string
            The selection name to use for the axes scales. This name should be
            unique among all selections within the chart.
        bind_x : boolean, default True
            If true, then bind the interactive scales to the x-axis
        bind_y : boolean, default True
            If true, then bind the interactive scales to the y-axis

        Returns
        -------
        chart :
            copy of self, with interactive axes added

        """
        encodings = []
        if bind_x:
            encodings.append("x")
        if bind_y:
            encodings.append("y")
        return self.add_selection(
            selection_interval(bind="scales", encodings=encodings)
        )


def _check_if_valid_subspec(spec, classname):
    """Check if the spec is a valid sub-spec.

    If it is not, then raise a ValueError
    """
    err = (
        'Objects with "{0}" attribute cannot be used within {1}. '
        "Consider defining the {0} attribute in the {1} object instead."
    )

    if not isinstance(spec, (core.SchemaBase, dict)):
        raise ValueError("Only chart objects can be used in {0}.".format(classname))
    for attr in TOPLEVEL_ONLY_KEYS:
        if isinstance(spec, core.SchemaBase):
            val = getattr(spec, attr, Undefined)
        else:
            val = spec.get(attr, Undefined)
        if val is not Undefined:
            raise ValueError(err.format(attr, classname))


def _check_if_can_be_layered(spec):
    """Check if the spec can be layered."""

    def _get(spec, attr):
        if isinstance(spec, core.SchemaBase):
            return spec._get(attr)
        else:
            return spec.get(attr, Undefined)

    encoding = _get(spec, "encoding")
    if encoding is not Undefined:
        for channel in ["row", "column", "facet"]:
            if _get(encoding, channel) is not Undefined:
                raise ValueError("Faceted charts cannot be layered.")
    if isinstance(spec, (Chart, LayerChart)):
        return

    if not isinstance(spec, (core.SchemaBase, dict)):
        raise ValueError("Only chart objects can be layered.")
    if _get(spec, "facet") is not Undefined:
        raise ValueError("Faceted charts cannot be layered.")
    if isinstance(spec, FacetChart) or _get(spec, "facet") is not Undefined:
        raise ValueError("Faceted charts cannot be layered.")
    if isinstance(spec, RepeatChart) or _get(spec, "repeat") is not Undefined:
        raise ValueError("Repeat charts cannot be layered.")
    if isinstance(spec, ConcatChart) or _get(spec, "concat") is not Undefined:
        raise ValueError("Concatenated charts cannot be layered.")
    if isinstance(spec, HConcatChart) or _get(spec, "hconcat") is not Undefined:
        raise ValueError("Concatenated charts cannot be layered.")
    if isinstance(spec, VConcatChart) or _get(spec, "vconcat") is not Undefined:
        raise ValueError("Concatenated charts cannot be layered.")


@utils.use_signature(core.TopLevelRepeatSpec)
class RepeatChart(TopLevelMixin, core.TopLevelRepeatSpec):
    """A chart repeated across rows and columns with small changes"""

    def __init__(self, data=Undefined, spec=Undefined, repeat=Undefined, **kwargs):
        _check_if_valid_subspec(spec, "RepeatChart")
        super(RepeatChart, self).__init__(data=data, spec=spec, repeat=repeat, **kwargs)

    def interactive(self, name=None, bind_x=True, bind_y=True):
        """Make chart axes scales interactive

        Parameters
        ----------
        name : string
            The selection name to use for the axes scales. This name should be
            unique among all selections within the chart.
        bind_x : boolean, default True
            If true, then bind the interactive scales to the x-axis
        bind_y : boolean, default True
            If true, then bind the interactive scales to the y-axis

        Returns
        -------
        chart :
            copy of self, with interactive axes added

        """
        copy = self.copy(deep=False)
        copy.spec = copy.spec.interactive(name=name, bind_x=bind_x, bind_y=bind_y)
        return copy

    def add_selection(self, *selections):
        """Add one or more selections to the chart."""
        if not selections or self.spec is Undefined:
            return self
        copy = self.copy()
        copy.spec = copy.spec.add_selection(*selections)
        return copy


def repeat(repeater="repeat"):
    """Tie a channel to the row or column within a repeated chart

    The output of this should be passed to the ``field`` attribute of
    a channel.

    Parameters
    ----------
    repeater : {'row'|'column'|'repeat'}
        The repeater to tie the field to. Default is 'repeat'.

    Returns
    -------
    repeat : RepeatRef object
    """
    if repeater not in ["row", "column", "repeat"]:
        raise ValueError("repeater must be one of ['row', 'column', 'repeat']")
    return core.RepeatRef(repeat=repeater)


@utils.use_signature(core.TopLevelConcatSpec)
class ConcatChart(TopLevelMixin, core.TopLevelConcatSpec):
    """A chart with horizontally-concatenated facets"""

    def __init__(self, data=Undefined, concat=(), columns=Undefined, **kwargs):
        # TODO: move common data to top level?
        for spec in concat:
            _check_if_valid_subspec(spec, "ConcatChart")
        super(ConcatChart, self).__init__(
            data=data, concat=list(concat), columns=columns, **kwargs
        )
        self.data, self.concat = _combine_subchart_data(self.data, self.concat)

    def __ior__(self, other):
        _check_if_valid_subspec(other, "ConcatChart")
        self.concat.append(other)
        self.data, self.concat = _combine_subchart_data(self.data, self.concat)
        return self

    def __or__(self, other):
        copy = self.copy(deep=["concat"])
        copy |= other
        return copy

    def add_selection(self, *selections):
        """Add one or more selections to all subcharts."""
        if not selections or not self.concat:
            return self
        copy = self.copy()
        copy.concat = [chart.add_selection(*selections) for chart in copy.concat]
        return copy


def concat(*charts, **kwargs):
    """Concatenate charts horizontally"""
    return ConcatChart(concat=charts, **kwargs)


@utils.use_signature(core.TopLevelHConcatSpec)
class HConcatChart(TopLevelMixin, core.TopLevelHConcatSpec):
    """A chart with horizontally-concatenated facets"""

    def __init__(self, data=Undefined, hconcat=(), **kwargs):
        # TODO: move common data to top level?
        for spec in hconcat:
            _check_if_valid_subspec(spec, "HConcatChart")
        super(HConcatChart, self).__init__(data=data, hconcat=list(hconcat), **kwargs)
        self.data, self.hconcat = _combine_subchart_data(self.data, self.hconcat)

    def __ior__(self, other):
        _check_if_valid_subspec(other, "HConcatChart")
        self.hconcat.append(other)
        self.data, self.hconcat = _combine_subchart_data(self.data, self.hconcat)
        return self

    def __or__(self, other):
        copy = self.copy(deep=["hconcat"])
        copy |= other
        return copy

    def add_selection(self, *selections):
        """Add one or more selections to all subcharts."""
        if not selections or not self.hconcat:
            return self
        copy = self.copy()
        copy.hconcat = [chart.add_selection(*selections) for chart in copy.hconcat]
        return copy


def hconcat(*charts, **kwargs):
    """Concatenate charts horizontally"""
    return HConcatChart(hconcat=charts, **kwargs)


@utils.use_signature(core.TopLevelVConcatSpec)
class VConcatChart(TopLevelMixin, core.TopLevelVConcatSpec):
    """A chart with vertically-concatenated facets"""

    def __init__(self, data=Undefined, vconcat=(), **kwargs):
        # TODO: move common data to top level?
        for spec in vconcat:
            _check_if_valid_subspec(spec, "VConcatChart")
        super(VConcatChart, self).__init__(data=data, vconcat=list(vconcat), **kwargs)
        self.data, self.vconcat = _combine_subchart_data(self.data, self.vconcat)

    def __iand__(self, other):
        _check_if_valid_subspec(other, "VConcatChart")
        self.vconcat.append(other)
        self.data, self.vconcat = _combine_subchart_data(self.data, self.vconcat)
        return self

    def __and__(self, other):
        copy = self.copy(deep=["vconcat"])
        copy &= other
        return copy

    def add_selection(self, *selections):
        """Add one or more selections to all subcharts."""
        if not selections or not self.vconcat:
            return self
        copy = self.copy()
        copy.vconcat = [chart.add_selection(*selections) for chart in copy.vconcat]
        return copy


def vconcat(*charts, **kwargs):
    """Concatenate charts vertically"""
    return VConcatChart(vconcat=charts, **kwargs)


@utils.use_signature(core.TopLevelLayerSpec)
class LayerChart(TopLevelMixin, _EncodingMixin, core.TopLevelLayerSpec):
    """A Chart with layers within a single panel"""

    def __init__(self, data=Undefined, layer=(), **kwargs):
        # TODO: move common data to top level?
        # TODO: check for conflicting interaction
        for spec in layer:
            _check_if_valid_subspec(spec, "LayerChart")
            _check_if_can_be_layered(spec)
        super(LayerChart, self).__init__(data=data, layer=list(layer), **kwargs)
        self.data, self.layer = _combine_subchart_data(self.data, self.layer)

    def __iadd__(self, other):
        _check_if_valid_subspec(other, "LayerChart")
        _check_if_can_be_layered(other)
        self.layer.append(other)
        self.data, self.layer = _combine_subchart_data(self.data, self.layer)
        return self

    def __add__(self, other):
        copy = self.copy(deep=["layer"])
        copy += other
        return copy

    def add_layers(self, *layers):
        copy = self.copy(deep=["layer"])
        for layer in layers:
            copy += layer
        return copy

    def interactive(self, name=None, bind_x=True, bind_y=True):
        """Make chart axes scales interactive

        Parameters
        ----------
        name : string
            The selection name to use for the axes scales. This name should be
            unique among all selections within the chart.
        bind_x : boolean, default True
            If true, then bind the interactive scales to the x-axis
        bind_y : boolean, default True
            If true, then bind the interactive scales to the y-axis

        Returns
        -------
        chart :
            copy of self, with interactive axes added

        """
        if not self.layer:
            raise ValueError(
                "LayerChart: cannot call interactive() until a " "layer is defined"
            )
        copy = self.copy(deep=["layer"])
        copy.layer[0] = copy.layer[0].interactive(
            name=name, bind_x=bind_x, bind_y=bind_y
        )
        return copy

    def add_selection(self, *selections):
        """Add one or more selections to all subcharts."""
        if not selections or not self.layer:
            return self
        copy = self.copy()
        copy.layer[0] = copy.layer[0].add_selection(*selections)
        return copy


def layer(*charts, **kwargs):
    """layer multiple charts"""
    return LayerChart(layer=charts, **kwargs)


@utils.use_signature(core.TopLevelFacetSpec)
class FacetChart(TopLevelMixin, core.TopLevelFacetSpec):
    """A Chart with layers within a single panel"""

    def __init__(self, data=Undefined, spec=Undefined, facet=Undefined, **kwargs):
        _check_if_valid_subspec(spec, "FacetChart")
        super(FacetChart, self).__init__(data=data, spec=spec, facet=facet, **kwargs)

    def interactive(self, name=None, bind_x=True, bind_y=True):
        """Make chart axes scales interactive

        Parameters
        ----------
        name : string
            The selection name to use for the axes scales. This name should be
            unique among all selections within the chart.
        bind_x : boolean, default True
            If true, then bind the interactive scales to the x-axis
        bind_y : boolean, default True
            If true, then bind the interactive scales to the y-axis

        Returns
        -------
        chart :
            copy of self, with interactive axes added

        """
        copy = self.copy(deep=False)
        copy.spec = copy.spec.interactive(name=name, bind_x=bind_x, bind_y=bind_y)
        return copy

    def add_selection(self, *selections):
        """Add one or more selections to the chart."""
        if not selections or self.spec is Undefined:
            return self
        copy = self.copy()
        copy.spec = copy.spec.add_selection(*selections)
        return copy


def topo_feature(url, feature, **kwargs):
    """A convenience function for extracting features from a topojson url

    Parameters
    ----------
    url : string
        An URL from which to load the data set.

    feature : string
        The name of the TopoJSON object set to convert to a GeoJSON feature collection. For
        example, in a map of the world, there may be an object set named `"countries"`.
        Using the feature property, we can extract this set and generate a GeoJSON feature
        object for each country.

    **kwargs :
        additional keywords passed to TopoDataFormat
    """
    return core.UrlData(
        url=url, format=core.TopoDataFormat(type="topojson", feature=feature, **kwargs)
    )


def _combine_subchart_data(data, subcharts):
    def remove_data(subchart):
        if subchart.data is not Undefined:
            subchart = subchart.copy()
            subchart.data = Undefined
        return subchart

    if not subcharts:
        # No subcharts = nothing to do.
        pass
    elif data is Undefined:
        # Top level has no data; all subchart data must
        # be identical to proceed.
        subdata = subcharts[0].data
        if subdata is not Undefined and all(c.data is subdata for c in subcharts):
            data = subdata
            subcharts = [remove_data(c) for c in subcharts]
    else:
        # Top level has data; subchart data must be either
        # undefined or identical to proceed.
        if all(c.data is Undefined or c.data is data for c in subcharts):
            subcharts = [remove_data(c) for c in subcharts]

    return data, subcharts


@utils.use_signature(core.SequenceParams)
def sequence(start, stop=None, step=Undefined, as_=Undefined, **kwds):
    """Sequence generator."""
    if stop is None:
        start, stop = 0, start
    params = core.SequenceParams(start=start, stop=stop, step=step, **{"as": as_})
    return core.SequenceGenerator(sequence=params, **kwds)


@utils.use_signature(core.GraticuleParams)
def graticule(**kwds):
    """Graticule generator."""
    if not kwds:
        # graticule: True indicates default parameters
        graticule = True
    else:
        graticule = core.GraticuleParams(**kwds)
    return core.GraticuleGenerator(graticule=graticule)


def sphere():
    """Sphere generator."""
    return core.SphereGenerator(sphere=True)
