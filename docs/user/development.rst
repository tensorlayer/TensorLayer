Development
===========

The TuneLayer project was started by Hao Dong, Imperial College London in Jun
2016. It is developed by a core team (in alphabetical order:
`Akara Supratak <https://akaraspt.github.io>`_,
`Hao Dong <https://zsdonghao.github.io>`_,
`Simiao Yu <https://github.com/zsdonghao>`_,)
and numerous additional contributors on `GitHub`_,

As an open-source project by Researchers for Researchers and Engineers,
we highly welcome contributions!
Every bit helps and will be credited.


What to contribute
------------------

Your method
~~~~~~~~~~~~~

If you publish a new method in term of Deep learning and Reinforcement learning,
you are welcome to contribute your method to TuneLayer.

* Explain how it would work, and link to a scientific paper if applicable.
* Keep the scope as narrow as possible, to make it easier to implement.


Report bugs
~~~~~~~~~~~

Report bugs at the `GitHub`_,
If you are reporting a bug, please include:

* your TuneLayer and TensorFlow version.
* steps to reproduce the bug, ideally reduced to a few Python commands.
* the results you obtain, and the results you expected instead.

If you are unsure whether the behavior you experience is a bug, or if you are
unsure whether it is related to TuneLayer or TensorFlow, please just ask on `our
mailing list`_ first.


Fix bugs
~~~~~~~~

Look through the GitHub issues for bug reports. Anything tagged with "bug" is
open to whoever wants to implement it. If you discover a bug in TuneLayer you can
fix yourself, by all means feel free to just implement a fix and not report it
first.


Write documentation
~~~~~~~~~~~~~~~~~~~

Whenever you find something not explained well, misleading, glossed over or
just wrong, please update it! The *Edit on GitHub* link on the top right of
every documentation page and the *[source]* link for every documented entity
in the API reference will help you to quickly locate the origin of any text.



How to contribute
-----------------

Edit on GitHub
~~~~~~~~~~~~~~

As a very easy way of just fixing issues in the documentation, use the *Edit
on GitHub* link on the top right of a documentation page or the *[source]* link
of an entity in the API reference to open the corresponding source file in
GitHub, then click the *Edit this file* link to edit the file in your browser
and send us a Pull Request. All you need for this is a free GitHub account.

For any more substantial changes, please follow the steps below to setup
TuneLayer for development.


Documentation
~~~~~~~~~~~~~

The documentation is generated with `Sphinx
<http://sphinx-doc.org/latest/index.html>`_. To build it locally, run the
following commands:

.. code:: bash

    pip install Sphinx
    sphinx-quickstart

    cd docs
    make html

If you want to re-generate the whole docs, run the following commands:

.. code :: bash

    cd docs
    make clean
    make html


To write the docs, we recommend you to install `Local RTD VM <http://docs.readthedocs.io/en/latest/custom_installs/local_rtd_vm.html>`_.




Afterwards, open ``docs/_build/html/index.html`` to view the documentation as
it would appear on `readthedocs <http://tunelayer.readthedocs.org/>`_. If you
changed a lot and seem to get misleading error messages or warnings, run
``make clean html`` to force Sphinx to recreate all files from scratch.

When writing docstrings, follow existing documentation as much as possible to
ensure consistency throughout the library. For additional information on the
syntax and conventions used, please refer to the following documents:

* `reStructuredText Primer <http://sphinx-doc.org/rest.html>`_
* `Sphinx reST markup constructs <http://sphinx-doc.org/markup/index.html>`_
* `A Guide to NumPy/SciPy Documentation <https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt>`_


Testing
~~~~~~~

TuneLayer has a code coverage of 100%, which has proven very helpful in the past,
but also creates some duties:

* Whenever you change any code, you should test whether it breaks existing
  features by just running the test scripts.
* Every bug you fix indicates a missing test case, so a proposed bug fix should
  come with a new test that fails without your fix.


Sending Pull Requests
~~~~~~~~~~~~~~~~~~~~~

When you're satisfied with your addition, the tests pass and the documentation
looks good without any markup errors, commit your changes to a new branch, push
that branch to your fork and send us a Pull Request via GitHub's web interface.

All these steps are nicely explained on GitHub:
https://guides.github.com/introduction/flow/

When filing your Pull Request, please include a description of what it does, to
help us reviewing it. If it is fixing an open issue, say, issue #123, add
*Fixes #123*, *Resolves #123* or *Closes #123* to the description text, so
GitHub will close it when your request is merged.



.. _GitHub: https://github.com/zsdonghao/tunelayer
.. _our mailing list: hao.dong11@imperial.ac.uk
