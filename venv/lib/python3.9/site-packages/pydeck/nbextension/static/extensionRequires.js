/* eslint-disable */
define(function() {
  'use strict';
  requirejs.config({
    map: {
      '*': {
        '@deck.gl/jupyter-widget': 'nbextensions/pydeck/index'
      }
    }
  });
  // Export the required load_ipython_extension function
  return {
    load_ipython_extension: function() {}
  };
});
