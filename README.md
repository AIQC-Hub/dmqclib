# DMQCLib

The *DMQCLib* package offers helper functions and classes that simplify model building and evaluation for the *AIQC* project.

## Package manager
You can create a new environment using any package management system, such as *conda* and *mamba*. 

Using *uv* is recommended when contributing modifications to the package.

 - uv (https://docs.astral.sh/uv/)

After the installation of *uv*, running `uv sync` inside the project folder will create the environment.

## Unit test
You may need to install the library in editable mode at least once before running unit tests.

```
 uv pip install -e .
```

After the library installation, you can run unit tests with *pytest*.

```
 uv run pytest -v
```

