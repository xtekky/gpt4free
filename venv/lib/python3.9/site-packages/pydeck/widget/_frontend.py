#!/usr/bin/env python
# coding: utf-8

from ..frontend_semver import DECKGL_SEMVER

"""
Information about the frontend package of the widget.
"""

# module_name is the name of the NPM package for the widget
module_name = "@deck.gl/jupyter-widget"
# module_version is the current version of the module of the JS portion of the widget
# It appears to be important only for JupyterLab and ignored for Jupyter Notebooks
module_version = DECKGL_SEMVER
