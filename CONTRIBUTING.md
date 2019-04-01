# TensorLayer Contributor Guideline

* Welcome to contribute!
* Continuous integration
* Build from sources and Unittest

## Welcome to contribute!
You are more than welcome to contribute to TensorLayer! If you have any improvement, please send us your [pull requests](https://help.github.com/en/articles/about-pull-requests). You may implement your improvement on your [folk](https://help.github.com/en/articles/working-with-forks).

## Continuous integration

We appreciate contributions
either by adding / improving examples or extending / fixing the core library.
To make your contributions, you would need to follow the [pep8](https://www.python.org/dev/peps/pep-0008/) coding style and [numpydoc](https://numpydoc.readthedocs.io/en/latest/) document style.
We rely on Continuous Integration (CI) for checking push commits.
The following tools are used to ensure that your commits can pass through the CI test:

* [yapf](https://github.com/google/yapf) (format code), compulsory
* [isort](https://github.com/timothycrosley/isort) (sort imports), optional
* [autoflake](https://github.com/myint/autoflake) (remove unused imports), optional

You can simply run

```bash
make format
```

to apply those tools before submitting your PR.

## Build from sources and Unittest

```bash
# First clone the repository and change the current directory to the newly cloned repository
git clone https://github.com/tensorlayer/tensorlayer.git
cd tensorlayer

# Install virtualenv if necessary
pip install virtualenv

# Then create a virtualenv called `venv`
virtualenv venv

# Activate the virtualenv

## Linux:
source venv/bin/activate

## Windows:
venv\Scripts\activate.bat

# ============= IF TENSORFLOW IS NOT ALREADY INSTALLED ============= #

# for a machine **without** an NVIDIA GPU
pip install -e .[all_cpu_dev] --upgrade

# for a machine **with** an NVIDIA GPU
pip install -e .[all_gpu_dev] --upgrade
```

Launching the unittest:

```bash
pytest
```
