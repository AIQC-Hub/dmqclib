Step 1: Installation
========================

``dmqclib`` can be installed using several popular Python package managers. We recommend an approach that combines `mamba` and `uv` for the best performance and dependency management, but standard methods are also fully supported.

Recommended Approach: Mamba + uv
---------------------------------

This method is recommended for developers and users who want a fast, robust, and isolated environment. `Mamba <https://mamba.readthedocs.io/>`_ is a fast, drop-in replacement for ``conda``, and `uv <https://docs.astral.sh/uv/>`_ is an extremely fast Python package installer and resolver.

**Step 1: Create and activate a new environment with Mamba**

This command creates a new environment named ``dmqclib-env`` and pre-installs Python and ``uv`` from the `conda-forge` channel.

.. code-block:: bash

   # Create a new environment
   mamba create -n dmqclib-env -c conda-forge python=3.12 uv

   # Activate the environment
   mamba activate dmqclib-env

**Step 2: Install dmqclib using uv**

Once inside the activated environment, use ``uv`` to install the package from PyPI.

.. code-block:: bash

   uv pip install dmqclib


Alternative Approaches
----------------------

If you prefer a simpler setup or are already using a different workflow, you can use one of the following standard methods.

Using pip
~~~~~~~~~

You can install ``dmqclib`` directly from PyPI using ``pip``.

.. note::
   It is highly recommended to install the package inside a virtual environment (like ``venv`` or ``virtualenv``) to avoid conflicts with other projects or system packages.

.. code-block:: bash

   pip install dmqclib

Using conda or mamba
~~~~~~~~~~~~~~~~~~~~

The package is available on the ``conda-forge`` channel, which is the recommended community channel for Conda packages. You can use either ``conda`` or ``mamba``.

.. code-block:: bash

   # Using conda
   conda install -c conda-forge dmqclib

   # Or using mamba (for a faster installation)
   mamba install -c conda-forge dmqclib

.. tip::
   While the package is also indexed on the `takayasaito` Anaconda channel, we recommend using `conda-forge` for consistency and access to a wider range of community-maintained packages.

Using uv (Standalone)
~~~~~~~~~~~~~~~~~~~~~

If you prefer to use ``uv`` for both environment management and installation, you can follow these steps:

**Step 1: Create and activate a virtual environment with uv**

.. code-block:: bash

   # Create a virtual environment in a .venv directory
   uv venv

   # Activate it (on Linux/macOS)
   source .venv/bin/activate
   # On Windows, use: .venv\Scripts\activate

**Step 2: Install dmqclib**

.. code-block:: bash

   uv pip install dmqclib

Next Steps
----------

You have now successfully installed the ``dmqclib`` library! The next step is to prepare training data sets.

Proceed to the next tutorial: :doc:`./preparation`.
