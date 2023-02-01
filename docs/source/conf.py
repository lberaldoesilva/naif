# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys

sys.path.insert(0, os.path.abspath('../..'))

project = 'naif'
copyright = '2023, Leandro Beraldo e Silva'
author = 'Leandro Beraldo e Silva'
release = '0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ["sphinx.ext.autodoc", "sphinx.ext.napoleon", "sphinx.ext.viewcode", "nbsphinx"]
#nbsphinx_prompt_width = 0 # no prompts in nbsphinx

templates_path = ['_templates']
exclude_patterns = ['.ipynb_checkpoints/*']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

#html_theme = 'alabaster'
html_theme = 'sphinxdoc'
#html_theme = 'agogo'
html_static_path = ['_static']

#body_max_width = 1200
master_doc = 'index'
