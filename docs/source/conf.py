import jaxbind

needs_sphinx = "3.2.0"

extensions = [
    "sphinx.ext.napoleon",  # Support for NumPy and Google style docstrings
    "sphinx.ext.imgmath",  # Render math as images
    "sphinx.ext.viewcode",  # Add links to highlighted source code
    "sphinx.ext.intersphinx",  # Links to other sphinx docs (mostly numpy)
    "myst_parser",  # Parse markdown
]
master_doc = "index"

myst_enable_extensions = [
    "amsmath",
    "dollarmath",
    "strikethrough",
    "tasklist",
]

intersphinx_mapping = {
    "numpy": ("https://numpy.org/doc/stable/", None),
    # "matplotlib": ('https://matplotlib.org/stable/', None),
    "ducc0": ("https://mtr.pages.mpcdf.de/ducc/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/reference/", None),
}

autodoc_default_options = {"special-members": "__init__"}

napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_ivar = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_references = True
napoleon_include_special_with_doc = True

imgmath_embed = True

project = "JAXbind"
copyright = "2024, Max-Planck-Society"
author = "Jakob Roth, Martin Reinecke, Gordian Edenhofer"

release = jaxbind.__version__
version = release[:-2]

language = "en"
exclude_patterns = []
add_module_names = False

html_theme = "pydata_sphinx_theme"
html_context = {"default_mode": "dark"}

html_theme_options = {
    "icon_links": [
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/jaxbind/",
            "icon": "fas fa-box",
        },
        {
            "name": "Github",
            "url": "https://github.com/NIFTy-PPL/JAXbind",
            "icon": "fab fa-github",
        },
    ],
    "navbar_persistent": ["search-field"],
    "navbar_end": ["navbar-icon-links"],
}


html_last_updated_fmt = "%b %d, %Y"

exclude_patterns = ["mod/modules.rst"]
