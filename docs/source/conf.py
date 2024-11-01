# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.abspath("../../"))


# -- Project information -----------------------------------------------------

project = "PathML"
copyright = (
    f"{datetime.now().year}, Dana-Farber Cancer Institute and Weill Cornell Medicine"
)
author = "Jacob Rosenthal et al."

about = {}
with open("../../pathml/_version.py") as f:
    exec(f.read(), about)
version = about["__version__"]

# The full version, including alpha/beta/rc tags
release = version


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "nbsphinx",
    "nbsphinx_link",
    "sphinx.ext.napoleon",
    "sphinx.ext.imgmath",
    "IPython.sphinxext.ipython_console_highlighting",
    "autoapi.extension",
    "sphinx_copybutton",
]

autodoc_default_options = {"members": True, "undoc-members": True}

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["../tests/*", "build", "../*.ipynb_checkpoints"]

# using autoapi to generate docs which should use less resources and improve readthedocs builds:
# https://docs.readthedocs.io/en/stable/guides/build-using-too-many-resources.html#document-python-modules-api-statically
autoapi_dirs = ["../../pathml"]
# still use autodoc directives:
# https://sphinx-autoapi.readthedocs.io/en/latest/reference/directives.html#autodoc-style-directives
autoapi_generate_api_docs = False

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"

# from https://sphinx-rtd-theme.readthedocs.io/en/stable/configuring.html
html_theme_options = {
    "logo_only": True,
    "display_version": True,
    "style_nav_header_background": "grey",
    "collapse_navigation": False,
    "prev_next_buttons_location": "both",
    "style_external_links": True,
}

# link to logo
html_logo = "_static/images/logo.png"

html_show_sphinx = False

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]


def setup(app):
    app.add_css_file("css/pathml.css")
