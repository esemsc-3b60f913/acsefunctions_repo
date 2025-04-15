# docs/source/conf.py
extensions = [
    "sphinx.ext.autodoc",  # For pulling docstrings (even if minimal)
    "sphinx.ext.napoleon",  # For Google/NumPy docstring support
    "sphinx.ext.mathjax",  # For math rendering
]

# LaTeX settings
latex_elements = {
    "papersize": "a4paper",
    "pointsize": "10pt",
}

# Project info
project = "acsefunctions"
author = "Your Name"
release = "0.1"
