## What is needed to run the code?

- The code is written in Python and Rust and uses the Sagemath Python library (and some
other libraries that come with sagemath).
- Below are two options to run the code

### Creating a Container

- The `apptainer.def` together `dockerfile` provide a container definition that should
work to execute the code `src/`.
- To build the container, compare the `apptainer.sif target` in the `makefile`.

### Creating an Environment

- Install sagemath on the system (don't know how to do it easily in an python virtual
environment
- Once sagemath is installed, do `python -m venv .venv --system-site-packages` to get the
system packages into the environment.
- Then activate it, install `rust_backend/py_lib` (this will require a Rust compiler) and
finally do `pip install -r requirements.txt` to install the rest. (note that the
requirements file does only list the top-level packages directly installed using pip;
i.e., it's not useful for reproducibility). 

## License

This project is distributed under the terms of both the MIT license and the
Apache License (Version 2.0).

See [LICENSE-APACHE](LICENSE-APACHE) and [LICENSE-MIT](LICENSE-MIT).
