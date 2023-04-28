import uuid

import numpy as np

from ..data_utils import is_pandas_df, has_geo_interface, records_from_geo_interface
from .json_tools import JSONMixin, camel_and_lower
from ..settings import settings as pydeck_settings

from pydeck.types import Image, Function
from pydeck.exceptions import BinaryTransportException


TYPE_IDENTIFIER = "@@type"
FUNCTION_IDENTIFIER = "@@="
QUOTE_CHARS = {"'", '"', "`"}


class Layer(JSONMixin):
    def __init__(self, type, data=None, id=None, use_binary_transport=None, **kwargs):
        """Configures a deck.gl layer for rendering on a map. Parameters passed
        here will be specific to the particular deck.gl layer that you are choosing to use.

        Please see the deck.gl
        `Layer catalog <https://deck.gl/docs/api-reference/layers>`_
        to determine the particular parameters of your layer. You are highly encouraged to look
        at the examples in the pydeck documentation.

        Parameters
        ==========

        type : str
            Type of layer to render, e.g., `HexagonLayer`
        id : str, default None
            Unique name for layer
        data : str or list of dict of {str: Any} or pandas.DataFrame, default None
            Either a URL of data to load in or an array of data
        use_binary_transport : bool, default None
            Boolean indicating binary data
        **kwargs
            Any of the parameters passable to a deck.gl layer.

        Examples
        ========

        For example, here is a HexagonLayer which reads data from a URL.

          >>> import pydeck
          >>> # 2014 location of car accidents in the UK
          >>> UK_ACCIDENTS_DATA = ('https://raw.githubusercontent.com/uber-common/'
          >>>                     'deck.gl-data/master/examples/3d-heatmap/heatmap-data.csv')
          >>> # Define a layer to display on a map
          >>> layer = pydeck.Layer(
          >>>     'HexagonLayer',
          >>>     UK_ACCIDENTS_DATA,
          >>>     get_position=['lng', 'lat'],
          >>>     auto_highlight=True,
          >>>     elevation_scale=50,
          >>>     pickable=True,
          >>>     elevation_range=[0, 3000],
          >>>     extruded=True,
          >>>     coverage=1)

        Alternately, input can be a pandas.DataFrame:

          >>> import pydeck
          >>> df = pd.read_csv(UK_ACCIDENTS_DATA)
          >>> layer = pydeck.Layer(
          >>>     'HexagonLayer',
          >>>     df,
          >>>     get_position=['lng', 'lat'],
          >>>     auto_highlight=True,
          >>>     elevation_scale=50,
          >>>     pickable=True,
          >>>     elevation_range=[0, 3000],
          >>>     extruded=True,
          >>>     coverage=1)
        """
        self.type = type
        self.id = id or str(uuid.uuid4())

        kwargs = self._add_default_layer_attributes(kwargs)

        # Add any other kwargs to the JSON output
        self._kwargs = kwargs.copy()

        if kwargs:
            for k, v in kwargs.items():
                # We assume strings and arrays of strings are identifiers
                # ["lng", "lat"] would be converted to '[lng, lat]'
                # TODO given that data here is usually a list of records,
                # we could probably check that the identifier is in the row
                # Errors on case like get_position='-', however

                if isinstance(v, str) and v[0] in QUOTE_CHARS and v[0] == v[-1]:
                    # Skip quoted strings
                    kwargs[k] = v.replace(v[0], "")
                elif isinstance(v, str) and Image.validate(v):
                    # Have pydeck convert local images to strings and/or apply extra quotes
                    kwargs[k] = Image(v)
                elif isinstance(v, str):
                    # Have @deck.gl/json treat strings values as functions
                    kwargs[k] = FUNCTION_IDENTIFIER + v
                elif isinstance(v, list) and v != [] and isinstance(v[0], str):
                    # Allows the user to pass lists e.g. to specify coordinates
                    array_as_str = ""
                    for i, identifier in enumerate(v):
                        if i == len(v) - 1:
                            array_as_str += "{}".format(identifier)
                        else:
                            array_as_str += "{}, ".format(identifier)
                    kwargs[k] = "{}[{}]".format(FUNCTION_IDENTIFIER, array_as_str)
                elif isinstance(v, Function):
                    kwargs[k] = v.serialize()

            self.__dict__.update(kwargs)

        self._data = None
        self.use_binary_transport = use_binary_transport
        self._binary_data = None
        self.data = data

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, data_set):
        """Make the data attribute a list no matter the input type, unless
        use_binary_transport is specified, which case we circumvent
        serializing the data to JSON
        """
        if self.use_binary_transport:
            self._binary_data = self._prepare_binary_data(data_set)
        elif is_pandas_df(data_set):
            self._data = data_set.to_dict(orient="records")
        elif has_geo_interface(data_set):
            self._data = records_from_geo_interface(data_set)
        else:
            self._data = data_set

    def get_binary_data(self):
        if not self.use_binary_transport:
            raise BinaryTransportException("Layer must be flagged with `use_binary_transport=True`")
        return self._binary_data

    def _prepare_binary_data(self, data_set):
        # Binary format conversion gives a sizable speedup but requires
        # slightly stricter standards for data input
        if not is_pandas_df(data_set):
            raise BinaryTransportException("Layer data must be a `pandas.DataFrame` type")

        layer_accessors = self._kwargs
        inverted_accessor_map = {v: k for k, v in layer_accessors.items() if type(v) not in [list, dict, set]}

        binary_transmission = []
        # Loop through data columns and convert them to numpy arrays
        for column in data_set.columns:
            # np.stack will take data arrays and conveniently extract the shape
            np_data = np.stack(data_set[column].to_numpy())
            # Get rid of the accessor so it doesn't appear in the JSON output
            del self.__dict__[inverted_accessor_map[column]]
            binary_transmission.append(
                {
                    "layer_id": self.id,
                    "column_name": column,
                    "accessor": camel_and_lower(inverted_accessor_map[column]),
                    "np_data": np_data,
                }
            )
        return binary_transmission

    @property
    def type(self):
        return getattr(self, TYPE_IDENTIFIER)

    @type.setter
    def type(self, type_name):
        self.__setattr__(TYPE_IDENTIFIER, type_name)

    def _add_default_layer_attributes(self, kwargs):
        attributes = pydeck_settings.default_layer_attributes

        if isinstance(attributes, dict) and self.type in attributes and isinstance(attributes[self.type], dict):
            kwargs = {**attributes[self.type], **kwargs}

        return kwargs
