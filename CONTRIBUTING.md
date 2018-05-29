# TensorLayer Contributor Guideline

## Continuous integration

We appreciate contributions
either by adding / improving examples or extending / fixing the core library. 
To make your contributions, you would need to follow the [pep8](https://www.python.org/dev/peps/pep-0008/) coding style and [numpydoc](https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt) document style.
We rely on Continuous Integration (CI) for checking push commits.
The following tools are used to ensure that your commits can pass through the CI test:

* [yapf](https://github.com/google/yapf) (format code), compulsory
* [isort](https://github.com/timothycrosley/isort) (sort imports), optional
* [autoflake](https://github.com/myint/autoflake) (remove unused imports), optional

You can simply run

```
make format
```

to apply those tools before submitting your PR.

## Build from sources

```bash
# First clone the repository
git clone https://github.com/tensorlayer/tensorlayer.git
cd tensorlayer

# Install virtualenv if necessary
pip install virtualenv

# Then create a virtualenv called venv inside
virtualenv venv

# Activate the virtualenv  

# Linux:
source venv/bin/activate

# Windows:
venv\Scripts\activate.bat

# for a machine **without** an NVIDIA GPU
pip install -e .[tf_cpu,db,dev,doc,extra,test]

# for a machine **with** an NVIDIA GPU
pip install -e .[tf_gpu,db,dev,doc,extra,test]
```

Launching the unittest:

```bash
$ pytest
```
