from .core import FunctionExpression


FUNCTION_LISTING = {
    "isArray": r"Returns true if _value_ is an array, false otherwise.",
    "isBoolean": r"Returns true if _value_ is a boolean (`true` or `false`), false otherwise.",
    "isDate": r"Returns true if _value_ is a Date object, false otherwise. This method will return false for timestamp numbers or date-formatted strings; it recognizes Date objects only.",
    "isNumber": r"Returns true if _value_ is a number, false otherwise. `NaN` and `Infinity` are considered numbers.",
    "isObject": r"Returns true if _value_ is an object (including arrays and Dates), false otherwise.",
    "isRegExp": r"Returns true if _value_ is a RegExp (regular expression) object, false otherwise.",
    "isString": r"Returns true if _value_ is a string, false otherwise.",
    "toBoolean": r"Coerces the input _value_ to a string. Null values and empty strings are mapped to `null`.",
    "toDate": r"Coerces the input _value_ to a Date instance. Null values and empty strings are mapped to `null`. If an optional _parser_ function is provided, it is used to perform date parsing, otherwise `Date.parse` is used. Be aware that `Date.parse` has different implementations across browsers!",
    "toNumber": r"Coerces the input _value_ to a number. Null values and empty strings are mapped to `null`.",
    "toString": r"Coerces the input _value_ to a string. Null values and empty strings are mapped to `null`.",
    "if": r"If _test_ is truthy, returns _thenValue_. Otherwise, returns _elseValue_. The _if_ function is equivalent to the ternary operator `a ? b : c`.",
    "isNaN": r"Returns true if _value_ is not a number. Same as JavaScript's `isNaN`.",
    "isFinite": r"Returns true if _value_ is a finite number. Same as JavaScript's `isFinite`.",
    "abs": r"Returns the absolute value of _value_. Same as JavaScript's `Math.abs`.",
    "acos": r"Trigonometric arccosine. Same as JavaScript's `Math.acos`.",
    "asin": r"Trigonometric arcsine. Same as JavaScript's `Math.asin`.",
    "atan": r"Trigonometric arctangent. Same as JavaScript's `Math.atan`.",
    "atan2": r"Returns the arctangent of _dy / dx_. Same as JavaScript's `Math.atan2`.",
    "ceil": r"Rounds _value_ to the nearest integer of equal or greater value. Same as JavaScript's `Math.ceil`.",
    "clamp": r"Restricts _value_ to be between the specified _min_ and _max_.",
    "cos": r"Trigonometric cosine. Same as JavaScript's `Math.cos`.",
    "exp": r"Returns the value of _e_ raised to the provided _exponent_. Same as JavaScript's `Math.exp`.",
    "floor": r"Rounds _value_ to the nearest integer of equal or lower value. Same as JavaScript's `Math.floor`.",
    "log": r"Returns the natural logarithm of _value_. Same as JavaScript's `Math.log`.",
    "max": r"Returns the maximum argument value. Same as JavaScript's `Math.max`.",
    "min": r"Returns the minimum argument value. Same as JavaScript's `Math.min`.",
    "pow": r"Returns _value_ raised to the given _exponent_. Same as JavaScript's `Math.pow`.",
    "random": r"Returns a pseudo-random number in the range [0,1). Same as JavaScript's `Math.random`.",
    "round": r"Rounds _value_ to the nearest integer. Same as JavaScript's `Math.round`.",
    "sin": r"Trigonometric sine. Same as JavaScript's `Math.sin`.",
    "sqrt": r"Square root function. Same as JavaScript's `Math.sqrt`.",
    "tan": r"Trigonometric tangent. Same as JavaScript's `Math.tan`.",
    "now": r"Returns the timestamp for the current time.",
    "datetime": r"Returns a new `Date` instance. The _month_ is 0-based, such that `1` represents February.",
    "date": r"Returns the day of the month for the given _datetime_ value, in local time.",
    "day": r"Returns the day of the week for the given _datetime_ value, in local time.",
    "year": r"Returns the year for the given _datetime_ value, in local time.",
    "quarter": r"Returns the quarter of the year (0-3) for the given _datetime_ value, in local time.",
    "month": r"Returns the (zero-based) month for the given _datetime_ value, in local time.",
    "hours": r"Returns the hours component for the given _datetime_ value, in local time.",
    "minutes": r"Returns the minutes component for the given _datetime_ value, in local time.",
    "seconds": r"Returns the seconds component for the given _datetime_ value, in local time.",
    "milliseconds": r"Returns the milliseconds component for the given _datetime_ value, in local time.",
    "time": r"Returns the epoch-based timestamp for the given _datetime_ value.",
    "timezoneoffset": r"Returns the timezone offset from the local timezone to UTC for the given _datetime_ value.",
    "utc": r"Returns a timestamp for the given UTC date. The _month_ is 0-based, such that `1` represents February.",
    "utcdate": r"Returns the day of the month for the given _datetime_ value, in UTC time.",
    "utcday": r"Returns the day of the week for the given _datetime_ value, in UTC time.",
    "utcyear": r"Returns the year for the given _datetime_ value, in UTC time.",
    "utcquarter": r"Returns the quarter of the year (0-3) for the given _datetime_ value, in UTC time.",
    "utcmonth": r"Returns the (zero-based) month for the given _datetime_ value, in UTC time.",
    "utchours": r"Returns the hours component for the given _datetime_ value, in UTC time.",
    "utcminutes": r"Returns the minutes component for the given _datetime_ value, in UTC time.",
    "utcseconds": r"Returns the seconds component for the given _datetime_ value, in UTC time.",
    "utcmilliseconds": r"Returns the milliseconds component for the given _datetime_ value, in UTC time.",
    "extent": r"Returns a new _[min, max]_ array with the minimum and maximum values of the input array, ignoring `null`, `undefined`, and `NaN` values.",
    "clampRange": r"Clamps a two-element _range_ array in a span-preserving manner. If the span of the input _range_ is less than _(max - min)_ and an endpoint exceeds either the _min_ or _max_ value, the range is translated such that the span is preserved and one endpoint touches the boundary of the _[min, max]_ range. If the span exceeds _(max - min)_, the range _[min, max]_ is returned.",
    "indexof": r"Returns the first index of _value_ in the input _array_, or the first index of _substring_ in the input _string_..",
    "inrange": r"Tests whether _value_ lies within (or is equal to either) the first and last values of the _range_ array.",
    "join": r"Returns a new string by concatenating all of the elements of the input _array_, separated by commas or a specified _separator_ string.",
    "lastindexof": r"Returns the last index of _value_ in the input _array_, or the last index of _substring_ in the input _string_..",
    "length": r"Returns the length of the input _array_, or the length of the input _string_.",
    "lerp": r"Returns the linearly interpolated value between the first and last entries in the _array_ for the provided interpolation _fraction_ (typically between 0 and 1). For example, `lerp([0, 50], 0.5)` returns 25.",
    "peek": r"Returns the last element in the input _array_. Similar to the built-in `Array.pop` method, except that it does not remove the last element. This method is a convenient shorthand for `array[array.length - 1]`.",
    "reverse": r"Returns a new array with elements in a reverse order of the input _array_. The first array element becomes the last, and the last array element becomes the first.",
    "sequence": r"Returns an array containing an arithmetic sequence of numbers. If _step_ is omitted, it defaults to 1. If _start_ is omitted, it defaults to 0. The _stop_ value is exclusive; it is not included in the result. If _step_ is positive, the last element is the largest _start + i * step_ less than _stop_; if _step_ is negative, the last element is the smallest _start + i * step_ greater than _stop_. If the returned array would contain an infinite number of values, an empty range is returned. The arguments are not required to be integers.",
    "slice": r"Returns a section of _array_ between the _start_ and _end_ indices. If the _end_ argument is negative, it is treated as an offset from the end of the array (_length(array) + end_).",
    "span": r"Returns the span of _array_: the difference between the last and first elements, or _array[array.length-1] - array[0]_. Or if input is a string: a section of _string_ between the _start_ and _end_ indices. If the _end_ argument is negative, it is treated as an offset from the end of the string (_length(string) + end_)..",
    "lower": r"Transforms _string_ to lower-case letters.",
    "pad": r"Pads a _string_ value with repeated instances of a _character_ up to a specified _length_. If _character_ is not specified, a space (' ') is used. By default, padding is added to the end of a string. An optional _align_ parameter specifies if padding should be added to the `'left'` (beginning), `'center'`, or `'right'` (end) of the input string.",
    "parseFloat": r"Parses the input _string_ to a floating-point value. Same as JavaScript's `parseFloat`.",
    "parseInt": r"Parses the input _string_ to an integer value. Same as JavaScript's `parseInt`.",
    "replace": r"Returns a new string with some or all matches of _pattern_ replaced by a _replacement_ string. The _pattern_ can be a string or a regular expression. If _pattern_ is a string, only the first instance will be replaced. Same as [JavaScript's String.replace](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/String/replace).",
    "split": r"Returns an array of tokens created by splitting the input _string_ according to a provided _separator_ pattern. The result can optionally be constrained to return at most _limit_ tokens.",
    "substring": r"Returns a section of _string_ between the _start_ and _end_ indices.",
    "trim": r"Returns a trimmed string with preceding and trailing whitespace removed.",
    "truncate": r"Truncates an input _string_ to a target _length_. The optional _align_ argument indicates what part of the string should be truncated: `'left'` (the beginning), `'center'`, or `'right'` (the end). By default, the `'right'` end of the string is truncated. The optional _ellipsis_ argument indicates the string to use to indicate truncated content; by default the ellipsis character `...` (`\\u2026`) is used.",
    "upper": r"Transforms _string_ to upper-case letters.",
    "merge": r"Merges the input objects _object1_, _object2_, etc into a new output object. Inputs are visited in sequential order, such that key values from later arguments can overwrite those from earlier arguments. Example: `merge({a:1, b:2}, {a:3}) -> {a:3, b:2}`.",
    "dayFormat": r"Formats a (0-6) _weekday_ number as a full week day name, according to the current locale. For example: `dayFormat(0) -> \"Sunday\"`.",
    "dayAbbrevFormat": r"Formats a (0-6) _weekday_ number as an abbreviated week day name, according to the current locale. For example: `dayAbbrevFormat(0) -> \"Sun\"`.",
    "format": r"Formats a numeric _value_ as a string. The _specifier_ must be a valid [d3-format specifier](https://github.com/d3/d3-format/) (e.g., `format(value, ',.2f')`.",
    "monthFormat": r"Formats a (zero-based) _month_ number as a full month name, according to the current locale. For example: `monthFormat(0) -> \"January\"`.",
    "monthAbbrevFormat": r"Formats a (zero-based) _month_ number as an abbreviated month name, according to the current locale. For example: `monthAbbrevFormat(0) -> \"Jan\"`.",
    "timeFormat": r"Formats a datetime _value_ (either a `Date` object or timestamp) as a string, according to the local time. The _specifier_ must be a valid [d3-time-format specifier](https://github.com/d3/d3-time-format/). For example: `timeFormat(timestamp, '%A')`.",
    "timeParse": r"Parses a _string_ value to a Date object, according to the local time. The _specifier_ must be a valid [d3-time-format specifier](https://github.com/d3/d3-time-format/). For example: `timeParse('June 30, 2015', '%B %d, %Y')`.",
    "utcFormat": r"Formats a datetime _value_ (either a `Date` object or timestamp) as a string, according to [UTC](https://en.wikipedia.org/wiki/Coordinated_Universal_Time) time. The _specifier_ must be a valid [d3-time-format specifier](https://github.com/d3/d3-time-format/). For example: `utcFormat(timestamp, '%A')`.",
    "utcParse": r"Parses a _string_ value to a Date object, according to [UTC](https://en.wikipedia.org/wiki/Coordinated_Universal_Time) time. The _specifier_ must be a valid [d3-time-format specifier](https://github.com/d3/d3-time-format/). For example: `utcParse('June 30, 2015', '%B %d, %Y')`.",
    "regexp": r"Creates a regular expression instance from an input _pattern_ string and optional _flags_. Same as [JavaScript's `RegExp`](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/RegExp).",
    "test": r"Evaluates a regular expression _regexp_ against the input _string_, returning `true` if the string matches the pattern, `false` otherwise. For example: `test(/\\d{3}/, \"32-21-9483\") -> true`.",
    "rgb": r"Constructs a new [RGB](https://en.wikipedia.org/wiki/RGB_color_model) color. If _r_, _g_ and _b_ are specified, these represent the channel values of the returned color; an _opacity_ may also be specified. If a CSS Color Module Level 3 _specifier_ string is specified, it is parsed and then converted to the RGB color space. Uses [d3-color's rgb function](https://github.com/d3/d3-color#rgb).",
    "hsl": r"Constructs a new [HSL](https://en.wikipedia.org/wiki/HSL_and_HSV) color. If _h_, _s_ and _l_ are specified, these represent the channel values of the returned color; an _opacity_ may also be specified. If a CSS Color Module Level 3 _specifier_ string is specified, it is parsed and then converted to the HSL color space. Uses [d3-color's hsl function](https://github.com/d3/d3-color#hsl).",
    "lab": r"Constructs a new [CIE LAB](https://en.wikipedia.org/wiki/Lab_color_space#CIELAB) color. If _l_, _a_ and _b_ are specified, these represent the channel values of the returned color; an _opacity_ may also be specified. If a CSS Color Module Level 3 _specifier_ string is specified, it is parsed and then converted to the LAB color space. Uses [d3-color's lab function](https://github.com/d3/d3-color#lab).",
    "hcl": r"Constructs a new [HCL](https://en.wikipedia.org/wiki/Lab_color_space#CIELAB) (hue, chroma, luminance) color. If _h_, _c_ and _l_ are specified, these represent the channel values of the returned color; an _opacity_ may also be specified. If a CSS Color Module Level 3 _specifier_ string is specified, it is parsed and then converted to the HCL color space. Uses [d3-color's hcl function](https://github.com/d3/d3-color#hcl).",
    "item": r"Returns the current scenegraph item that is the target of the event.",
    "group": r"Returns the scenegraph group mark item in which the current event has occurred. If no arguments are provided, the immediate parent group is returned. If a group name is provided, the matching ancestor group item is returned.",
    "xy": r"Returns the x- and y-coordinates for the current event as a two-element array. If no arguments are provided, the top-level coordinate space of the view is used. If a scenegraph _item_ (or string group name) is provided, the coordinate space of the group item is used.",
    "x": r"Returns the x coordinate for the current event. If no arguments are provided, the top-level coordinate space of the view is used. If a scenegraph _item_ (or string group name) is provided, the coordinate space of the group item is used.",
    "y": r"Returns the y coordinate for the current event. If no arguments are provided, the top-level coordinate space of the view is used. If a scenegraph _item_ (or string group name) is provided, the coordinate space of the group item is used.",
    "pinchDistance": r"Returns the pixel distance between the first two touch points of a multi-touch event.",
    "pinchAngle": r"Returns the angle of the line connecting the first two touch points of a multi-touch event.",
    "inScope": r"Returns true if the given scenegraph _item_ is a descendant of the group mark in which the event handler was defined, false otherwise.",
    "data": r"Returns the array of data objects for the Vega data set with the given _name_. If the data set is not found, returns an empty array.",
    "indata": r"Tests if the data set with a given _name_ contains a datum with a _field_ value that matches the input _value_. For example: `indata('table', 'category', value)`.",
    "scale": r"Applies the named scale transform (or projection) to the specified _value_. The optional _group_ argument takes a scenegraph group mark item to indicate the specific scope in which to look up the scale or projection.",
    "invert": r"Inverts the named scale transform (or projection) for the specified _value_. The optional _group_ argument takes a scenegraph group mark item to indicate the specific scope in which to look up the scale or projection.",
    "copy": r"Returns a copy (a new cloned instance) of the named scale transform of projection, or `undefined` if no scale or projection is found. The optional _group_ argument takes a scenegraph group mark item to indicate the specific scope in which to look up the scale or projection.",
    "domain": r"Returns the scale domain array for the named scale transform, or an empty array if the scale is not found. The optional _group_ argument takes a scenegraph group mark item to indicate the specific scope in which to look up the scale.",
    "range": r"Returns the scale range array for the named scale transform, or an empty array if the scale is not found. The optional _group_ argument takes a scenegraph group mark item to indicate the specific scope in which to look up the scale.",
    "bandwidth": r"Returns the current band width for the named band scale transform, or zero if the scale is not found or is not a band scale. The optional _group_ argument takes a scenegraph group mark item to indicate the specific scope in which to look up the scale.",
    "bandspace": r"Returns the number of steps needed within a band scale, based on the _count_ of domain elements and the inner and outer padding values. While normally calculated within the scale itself, this function can be helpful for determining the size of a chart's layout.",
    "gradient": r"Returns a linear color gradient for the _scale_ (whose range must be a [continuous color scheme](../schemes)) and starting and ending points _p0_ and _p1_, each an _[x, y]_ array. The points _p0_ and _p1_ should be expressed in normalized coordinates in the domain [0, 1], relative to the bounds of the item being colored. If unspecified, _p0_ defaults to `[0, 0]` and _p1_ defaults to `[1, 0]`, for a horizontal gradient that spans the full bounds of an item. The optional _count_ argument indicates a desired target number of sample points to take from the color scale.",
    "panLinear": r"Given a linear scale _domain_ array with numeric or datetime values, returns a new two-element domain array that is the result of panning the domain by a fractional _delta_. The _delta_ value represents fractional units of the scale range; for example, `0.5` indicates panning the scale domain to the right by half the scale range.",
    "panLog": r"Given a log scale _domain_ array with numeric or datetime values, returns a new two-element domain array that is the result of panning the domain by a fractional _delta_. The _delta_ value represents fractional units of the scale range; for example, `0.5` indicates panning the scale domain to the right by half the scale range.",
    "panPow": r"Given a power scale _domain_ array with numeric or datetime values and the given _exponent_, returns a new two-element domain array that is the result of panning the domain by a fractional _delta_. The _delta_ value represents fractional units of the scale range; for example, `0.5` indicates panning the scale domain to the right by half the scale range.",
    "panSymlog": r"Given a symmetric log scale _domain_ array with numeric or datetime values parameterized by the given _constant_, returns a new two-element domain array that is the result of panning the domain by a fractional _delta_. The _delta_ value represents fractional units of the scale range; for example, `0.5` indicates panning the scale domain to the right by half the scale range.",
    "zoomLinear": r"Given a linear scale _domain_ array with numeric or datetime values, returns a new two-element domain array that is the result of zooming the domain by a _scaleFactor_, centered at the provided fractional _anchor_. The _anchor_ value represents the zoom position in terms of fractional units of the scale range; for example, `0.5` indicates a zoom centered on the mid-point of the scale range.",
    "zoomLog": r"Given a log scale _domain_ array with numeric or datetime values, returns a new two-element domain array that is the result of zooming the domain by a _scaleFactor_, centered at the provided fractional _anchor_. The _anchor_ value represents the zoom position in terms of fractional units of the scale range; for example, `0.5` indicates a zoom centered on the mid-point of the scale range.",
    "zoomPow": r"Given a power scale _domain_ array with numeric or datetime values and the given _exponent_, returns a new two-element domain array that is the result of zooming the domain by a _scaleFactor_, centered at the provided fractional _anchor_. The _anchor_ value represents the zoom position in terms of fractional units of the scale range; for example, `0.5` indicates a zoom centered on the mid-point of the scale range.",
    "zoomSymlog": r"Given a symmetric log scale _domain_ array with numeric or datetime values parameterized by the given _constant_, returns a new two-element domain array that is the result of zooming the domain by a _scaleFactor_, centered at the provided fractional _anchor_. The _anchor_ value represents the zoom position in terms of fractional units of the scale range; for example, `0.5` indicates a zoom centered on the mid-point of the scale range.",
    "geoArea": r"Returns the projected planar area (typically in square pixels) of a GeoJSON _feature_ according to the named _projection_. If the _projection_ argument is `null`, computes the spherical area in steradians using unprojected longitude, latitude coordinates. The optional _group_ argument takes a scenegraph group mark item to indicate the specific scope in which to look up the projection. Uses d3-geo's [geoArea](https://github.com/d3/d3-geo#geoArea) and [path.area](https://github.com/d3/d3-geo#path_area) methods.",
    "geoBounds": r"Returns the projected planar bounding box (typically in pixels) for the specified GeoJSON _feature_, according to the named _projection_. The bounding box is represented by a two-dimensional array: [[_x0_, _y0_], [_x1_, _y1_]], where _x0_ is the minimum x-coordinate, _y0_ is the minimum y-coordinate, _x1_ is the maximum x-coordinate, and _y1_ is the maximum y-coordinate. If the _projection_ argument is `null`, computes the spherical bounding box using unprojected longitude, latitude coordinates. The optional _group_ argument takes a scenegraph group mark item to indicate the specific scope in which to look up the projection. Uses d3-geo's [geoBounds](https://github.com/d3/d3-geo#geoBounds) and [path.bounds](https://github.com/d3/d3-geo#path_bounds) methods.",
    "geoCentroid": r"Returns the projected planar centroid (typically in pixels) for the specified GeoJSON _feature_, according to the named _projection_. If the _projection_ argument is `null`, computes the spherical centroid using unprojected longitude, latitude coordinates. The optional _group_ argument takes a scenegraph group mark item to indicate the specific scope in which to look up the projection. Uses d3-geo's [geoCentroid](https://github.com/d3/d3-geo#geoCentroid) and [path.centroid](https://github.com/d3/d3-geo#path_centroid) methods.",
    "treePath": r"For the hierarchy data set with the given _name_, returns the shortest path through from the _source_ node id to the _target_ node id. The path starts at the _source_ node, ascends to the least common ancestor of the _source_ node and the _target_ node, and then descends to the _target_ node.",
    "treeAncestors": r"For the hierarchy data set with the given _name_, returns the array of ancestors nodes, starting with the input _node_, then followed by each parent up to the root.",
    "warn": r"Logs a warning message and returns the last argument. For the message to appear in the console, the visualization view must have the appropriate logging level set.",
    "info": r"Logs an informative message and returns the last argument. For the message to appear in the console, the visualization view must have the appropriate logging level set.",
    "debug": r"Logs a debugging message and returns the last argument. For the message to appear in the console, the visualization view must have the appropriate logging level set.",
    "eventGroup": "returns the scenegraph group mark item within which the current event has occurred. If no arguments are provided, the immediate parent group is returned. If a group name is provided, the matching ancestor group item is returned.",
    "eventItem": "a zero-argument function that returns the current scenegraph item that is the subject of the event.",
    "eventX": "returns the x-coordinate for the current event. If no arguments are provided, the top-level coordinate space of the visualization is used. If a group name is provided, the coordinate-space of the matching ancestor group item is used.",
    "eventY": "returns the y-coordinate for the current event. If no arguments are provided, the top-level coordinate space of the visualization is used. If a group name is provided, the coordinate-space of the matching ancestor group item is used.",
    "iscale": 'applies an inverse scale transform to a specified value; by default, looks for the scale at the top-level of the specification, but an optional signal can also be supplied corresponding to the group which contains the scale (i.e., `iscale("x", val, group)`). *Note:* This function is only legal within signal stream handlers and mark [production rules](https://github.com/vega/vega/wiki/Marks#production-rules). Invoking this function elsewhere (e.g., with filter or formula transforms) will result in an error.',
    "open": "opens a hyperlink (alias to `window.open`). This function is only valid when running in the browser. It should not be invoked within a server-side (e.g., node.js) environment.",
}


# This maps vega expression function names to the Python name
NAME_MAP = {"if": "if_"}


class ExprFunc(object):
    def __init__(self, name, doc):
        self.name = name
        self.doc = doc
        self.__doc__ = """{}(*args)\n    {}""".format(name, doc)

    def __call__(self, *args):
        return FunctionExpression(self.name, args)

    def __repr__(self):
        return "<function expr.{}(*args)>".format(self.name)


def _populate_namespace():
    globals_ = globals()
    for name, doc in FUNCTION_LISTING.items():
        py_name = NAME_MAP.get(name, name)
        globals_[py_name] = ExprFunc(name, doc)
        yield py_name


__all__ = list(_populate_namespace())
