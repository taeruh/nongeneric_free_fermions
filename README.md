Until I have a container, do the following:

### Creating the Environment

- Once sagemath is installed, do `python -m venv .venv --system-site-packages` to get the
system packages into the environment.
- Then activate it and do `pip install -r requirements.txt` to install the rest. (note
that the requirements file does only list the top-level packages directly installed using
pip; i.e., it's not useful for reproducibility).
