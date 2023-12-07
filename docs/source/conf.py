import os
import sys

# Add the project root directory to the sys.path
sys.path.insert(0, os.path.abspath('../..'))

import sphinx_nested_apidoc

# -- Project information -----------------------------------------------------

project = 'IslandOfMisfitToys'
author = 'Tyler'

# -- General configuration ---------------------------------------------------

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
]

# Configure sphinx-nested-apidoc
# nested_apidoc_package_dir = 'misfit_toys'
# nested_apidoc_package_name = 'misfit_toys'
# nested_apidoc_module = True
# nested_apidoc_exclude_modules = ['tests']

# -- Options for HTML output -------------------------------------------------

html_theme = 'sphinx_rtd_theme'
autodoc_member_order = 'bysource'
autodoc_typehints = 'signature'
napoleon_google_docstring = True

# html_static_path = ['_static']
