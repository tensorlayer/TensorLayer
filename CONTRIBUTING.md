# Contributing to Tensorlayer

We highly appreciate any contribution towards TensorLayer 
either by adding / improving new examples or extending / fixing the TensorLayer core library. 

To make your contributions, you would need to follow the following guideline:

We are using [pep8](https://www.python.org/dev/peps/pep-0008/) coding style and [numpy doc style](https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt).
The following tools are used to ensure that we comply the convention:

* yapf (format code)
* isort (sort imports)
* autoflake (remove unused imports)

Please run them for the changed files before you make a commit.

yapf is now enforced in our travis-ci test, and we are moving to enforce pydocstyle in the future.
