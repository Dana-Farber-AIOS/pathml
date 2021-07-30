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
sys.path.insert(0, os.path.abspath('../../'))


# -- Project information -----------------------------------------------------

project = 'PathML'
copyright = '2021, Dana-Farber Cancer Institute and Weill Cornell Medicine'
author = 'Jacob Rosenthal et al.'

version = '1.0.0'
# The full version, including alpha/beta/rc tags
release = version


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'nbsphinx',
    'nbsphinx_link',
    'sphinx.ext.napoleon',
    'sphinx.ext.imgmath',
    'IPython.sphinxext.ipython_console_highlighting'
]

autodoc_default_options = {
    'members': True,
    'undoc-members': True
}

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['../tests/*', 'build', '../*.ipynb_checkpoints']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# from https://sphinx-rtd-theme.readthedocs.io/en/stable/configuring.html
html_theme_options = {
    'logo_only': True,
    'display_version': True,
    'style_nav_header_background': 'grey',
    'collapse_navigation': False,
    'prev_next_buttons_location': 'both',
    'style_external_links': True
}

# link to logo
html_logo = '_static/images/logo.png'


# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# mocking imports which require C extension for readthedocs builds
# see: https://docs.readthedocs.io/en/stable/faq.html#i-get-import-errors-on-libraries-that-depend-on-c-modules
autodoc_mock_imports = ["openslide", "spams"]


def setup(app):
    app.add_css_file('css/pathml.css')
