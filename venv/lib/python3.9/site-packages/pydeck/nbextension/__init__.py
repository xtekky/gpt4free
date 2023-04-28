def _jupyter_nbextension_paths():
    """Integrates Widget with a Jupyter notebook.
    Required for building a widget. `See the Jupyter Notebook docs.`_

    Users should not explicitly call this function.

    _ https://testnb.readthedocs.io/en/latest/examples/Notebook/Distributing%20Jupyter%20Extensions%20as%20Python%20Packages.html#Defining-the-server-extension-and-nbextension
    """
    return [
        {"section": "notebook", "src": "nbextension/static", "dest": "pydeck", "require": "pydeck/extensionRequires"}
    ]
