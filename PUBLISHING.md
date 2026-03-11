# How to Publish Vyntri to PyPI

This guide will help you upload your library to the Python Package Index (PyPI) so anyone can install it via `pip install vyntri`.

## Prerequisites
1.  **Create a PyPI Account**: Go to [pypi.org](https://pypi.org/) and register.
2.  **Install Build Tools**:
    ```bash
    python -m pip install --upgrade build twine
    ```

## Step 1: Prepare the Package
Ensure your `pyproject.toml` has the correct version.
```toml
[project]
name = "vyntri"
version = "0.1.0"
```

## Step 2: Build the Distribution
Run this command in the root directory (where `pyproject.toml` is):
```bash
python -m build
```
This will create a `dist/` folder containing `.tar.gz` and `.whl` files.

## Step 3: Upload to PyPI
Use `twine` to upload the built files.
```bash
python -m twine upload dist/*
```
You will be prompted for your username (`__token__`) and password (your PyPI API token).

## Step 4: Install!
Now anyone in the world can run:
```bash
pip install vyntri
```

## Updating the Package
1.  Bump `version` in `pyproject.toml` (e.g., to `0.1.1`).
2.  Remove old `dist/` folder.
3.  Run `python -m build`.
4.  Run `python -m twine upload dist/*`.
