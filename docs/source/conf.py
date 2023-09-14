# Get version
from importlib.metadata import version
release = version('obswinlib')
# for example take major/minor
version = '.'.join(release.split('.')[:2])

# -- Project information -----------------------------------------------------

project = 'obswinlib'
copyright = '2023, Lucas Sawade'
author = 'Lucas Sawade'
release = release

# -- General configuration ---------------------------------------------------

extensions = [
    'sphinx.ext.autodoc',
    "sphinx.ext.autosummary",
    'sphinx.ext.viewcode',
    'sphinx.ext.githubpages',
    'sphinx.ext.mathjax',
    'sphinx.ext.intersphinx',
    'numpydoc',
    'sphinx_design',
    "sphinx_togglebutton",
]

templates_path = ['_templates']
exclude_patterns = ["_build", "Thumbs.db",
                    ".DS_Store", "**.ipynb_checkpoints", "build"]


# -- Numpy Doc ---------------------------------------------------------------
add_module_names = False
autoclass_content = 'both'
autosummary_generate = True
numpydoc_class_members_toctree = False


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'furo'
html_static_path = ['_static']

html_theme_options = {

    # "light_logo": "logo.svg",
    # "dark_logo": "logo.svg",
    # "navbar_end": ["navbar-icon-links"],
    # "icon_links": [
    #     {
    #         # Label for this link
    #         "name": "GitHub",
    #         # URL where the link will redirect
    #         "url": "https://github.com/lsawade/GF3D",  # required
    #         # Icon class (if "type": "fontawesome"), or path to local image (if "type": "local")
    #         "icon": "fa-brands fa-square-github",
    #         # The type of image to be used (see below for details)
    #         "type": "fontawesome",
    #     },
    # ]

}

html_favicon = '_static/favicon.ico'

html_context = {
    "default_mode": "auto",
    # "github_url": "https://github.com", # or your GitHub Enterprise site
    "github_user": "lsawade",
    "github_repo": "obswinlib",
    "github_version": "main",
    "doc_path": "docs/source",
}

