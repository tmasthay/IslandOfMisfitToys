# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Island of Misfit Toys'
copyright = '2024, Tyler Masthay'
author = 'Tyler Masthay'
release = '0.1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.napoleon', 'sphinx.ext.viewcode', 'autoapi.extension']
autoapi_dirs = ['../misfit_toys']
autoapi_generate_api_docs = True
autoapi_options = [
    'members',
    'undoc-members',
    'show-inheritance',
    'show-module-summary',
    'special-members',
    'imported-members',
    'no-signatures',
]
autoapi_keep_files = True
autoapi_member_order = 'bysource'
autoapi_own_page_level = 'module'

templates_path = ['_templates']
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'furo'
# html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

viewcode_follow_imported_members = True
# napoleon_use_param = False
viewcode_enable_epub = True
