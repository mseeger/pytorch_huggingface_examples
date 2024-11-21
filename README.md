# Example Code using PyTorch and Hugging Face

* My personal playground

## Installation

First, you need to have `Python 3.10` or later installed, as well as `pkg-config`.
If you use `homebrew`:
```bash
brew install cmake
brew install pkg-config
```
Make sure that `/opt/homebrew/bin` appears in `PATH` before `/usr/bin` or `/usr/local/bin`.

If your Python version is 3.13, please read [here](#pytorch-250-does-not-support-python-313).

For a full installation, run:
```bash
python3 -m venv deeplearn_venv
. scripts_venv/bin/activate
pip install --upgrade pip
pip install -e '.[all]'
```

For a minimal installation, replace the last command by:
```bash
pip install -e '.'
```

Dependency tags (groups):
* `dl`: Deep learning (PyTorch, Hugging Face)
* `datasci`: Data science extras (e.g., Jupyter)
* `aws`: Working with AWS
* `dev`: Testing, linting
* `all`: All groups are installed

Do not forget to active the venv `deeplearn_venv` before running any of the scripts
or notebooks.

This setup is for local development and debugging. On Mac laptops with Apple silicon,
the built-in GPU can be used. The device is called `mps`. Scripts use this automatically
if detected.

**TODO**: Documentation for running on GPUs (e.g., AWS EC2).

### Troubleshooting

#### PyTorch 2.5.0 does not support Python 3.13

See https://github.com/pytorch/pytorch/issues/130249. The next release 2.5.1 is
projected to support Python 3.13.

For now, make sure that Python 3.12 is installed via Homebrew as well, and create
the virtual environment with this version:

```bash
brew install python@3.12
$(brew --prefix python@3.12)/bin/python3.12 --version
$(brew --prefix python@3.12)/bin/python3.12 -m venv deeplearn_venv
```

The remainder is as above. Once PyTorch 2.5.1 is out, recreate the virtual
environment with Python 3.13 as stated above.
