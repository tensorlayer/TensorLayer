# Contributing to TensorLayer

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
