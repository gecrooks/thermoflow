# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

import sys
import os
import guzzle_sphinx_theme
import thermoflow

sys.path.insert(0, os.path.abspath(".."))

__version__ = thermoflow.__version__
print("__version__", __version__)


# -- Project information -----------------------------------------------------

project = "thermoflow"
copyright = "2022-2024 Gavin Crooks"
author = "Gavin Crooks"
release = __version__

html_title = "ThermoFlow Documentation"
html_short_title = "ThermoFlow"

# Insert version in side bar
with open("./_templates/version.html", "w") as f:
    f.write('<div align="center">v%s</div>' % __version__)

# Custom sidebar templates, maps document names to template names.
html_sidebars = {
    "**": ["logo-text.html", "version.html", "searchbox.html", "globaltoc.html"]
}


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.githubpages",
]

extensions.append("guzzle_sphinx_theme")


# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme_path = guzzle_sphinx_theme.html_theme_path()
html_theme = "guzzle_sphinx_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]


autodoc_typehints = ["description"]
autodoc_member_order = ["bysource"]
