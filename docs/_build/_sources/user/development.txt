Development
===========

The Lasagne project was started by Sander Dieleman in September 2014. It is
developed by a core team of eight people (in alphabetical order:
`Eric Battenberg <http://ericbattenberg.com/>`_,
`Sander Dieleman <http://benanne.github.io>`_,
`Daniel Nouri <http://danielnouri.org>`_,
`Eben Olson <https://github.com/ebenolson>`_,
`Aäron van den Oord <https://twitter.com/avdnoord>`_,
`Colin Raffel <http://colinraffel.com/>`_,
`Jan Schlüter <http://www.ofai.at/~jan.schlueter/>`_,
`Søren Kaae Sønderby <http://www1.bio.ku.dk/english/staff/?pure=en/persons/418078>`_)
and `numerous additional contributors
<https://github.com/Lasagne/Lasagne/graphs/contributors>`_ on GitHub:
https://github.com/Lasagne/Lasagne

As an open-source project by researchers for researchers, we highly welcome
contributions! Every bit helps and will be credited.



.. _lasagne-philosopy:

Philosophy
----------

Lasagne grew out of a need to combine the flexibility of Theano with the availability of the right building blocks for training neural networks. Its development is guided by a number of design goals:

* **Simplicity**: Be easy to use, easy to understand and easy to extend, to
  facilitate use in research. Interfaces should be kept small, with as few
  classes and methods as possible. Every added abstraction and feature should
  be carefully scrutinized, to determine whether the added complexity is
  justified.

* **Transparency**: Do not hide Theano behind abstractions, directly process
  and return Theano expressions or Python / numpy data types. Try to rely on
  Theano's functionality where possible, and follow Theano's conventions.

* **Modularity**: Allow all parts (layers, regularizers, optimizers, ...) to be
  used independently of Lasagne. Make it easy to use components in isolation or
  in conjunction with other frameworks.

* **Pragmatism**: Make common use cases easy, do not overrate uncommon cases.
  Ideally, everything should be possible, but common use cases shouldn't be
  made more difficult just to cater for exotic ones.

* **Restraint**: Do not obstruct users with features they decide not to use.
  Both in using and in extending components, it should be possible for users to
  be fully oblivious to features they do not need.

* **Focus**: "Do one thing and do it well". Do not try to provide a library for
  everything to do with deep learning.



What to contribute
------------------

Give feedback
~~~~~~~~~~~~~

To send us general feedback, questions or ideas for improvement, please post on
`our mailing list`_.

If you have a very concrete feature proposal, add it to the `issue tracker on
GitHub`_:

* Explain how it would work, and link to a scientific paper if applicable.
* Keep the scope as narrow as possible, to make it easier to implement.


Report bugs
~~~~~~~~~~~

Report bugs at the `issue tracker on GitHub`_.
If you are reporting a bug, please include:

* your Lasagne and Theano version.
* steps to reproduce the bug, ideally reduced to a few Python commands.
* the results you obtain, and the results you expected instead.

If you are unsure whether the behavior you experience is a bug, or if you are
unsure whether it is related to Lasagne or Theano, please just ask on `our
mailing list`_ first.


Fix bugs
~~~~~~~~

Look through the GitHub issues for bug reports. Anything tagged with "bug" is
open to whoever wants to implement it. If you discover a bug in Lasagne you can
fix yourself, by all means feel free to just implement a fix and not report it
first.


Implement features
~~~~~~~~~~~~~~~~~~

Look through the GitHub issues for feature proposals. Anything tagged with
"feature" or "enhancement" is open to whoever wants to implement it. If you
have a feature in mind you want to implement yourself, please note that Lasagne
has a fairly narrow focus and we strictly follow a set of :ref:`design
principles <lasagne-philosopy>`, so we cannot guarantee upfront that your code
will be included. Please do not hesitate to just propose your idea in a GitHub
issue or on the mailing list first, so we can discuss it and/or guide you
through the implementation.


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
Lasagne for development.


Development setup
~~~~~~~~~~~~~~~~~

First, follow the instructions for performing a development installation of
Lasagne (including forking on GitHub): :ref:`lasagne-development-install`

To be able to run the tests and build the documentation locally, install
additional requirements with: ``pip install -r requirements-dev.txt`` (adding
``--user`` if you want to install to your home directory instead).

If you use the bleeding-edge version of Theano, then instead of running that
command, just use ``pip install`` to manually install all dependencies listed
in ``requirements-dev.txt`` with their correct versions; otherwise it will
attempt to downgrade Theano to the known good version in ``requirements.txt``.


Documentation
~~~~~~~~~~~~~

The documentation is generated with `Sphinx
<http://sphinx-doc.org/latest/index.html>`_. To build it locally, run the
following commands:

.. code:: bash

    cd docs
    make html

Afterwards, open ``docs/_build/html/index.html`` to view the documentation as
it would appear on `readthedocs <http://lasagne.readthedocs.org/>`_. If you
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

Lasagne has a code coverage of 100%, which has proven very helpful in the past,
but also creates some duties:

* Whenever you change any code, you should test whether it breaks existing
  features by just running the test suite. The test suite will also be run by
  `Travis <https://travis-ci.org/>`_ for any Pull Request to Lasagne.
* Any code you add needs to be accompanied by tests ensuring that nobody else
  breaks it in future. `Coveralls <https://coveralls.io/>`_ will check whether
  the code coverage stays at 100% for any Pull Request to Lasagne.
* Every bug you fix indicates a missing test case, so a proposed bug fix should
  come with a new test that fails without your fix.

To run the full test suite, just do

.. code:: bash

    py.test

Testing will take over 5 minutes for the first run, but less than a minute for
subsequent runs when Theano can reuse compiled code. It will end with a code
coverage report specifying which code lines are not covered by tests, if any.
Furthermore, it will list any failed tests, and failed `PEP8
<https://www.python.org/dev/peps/pep-0008/>`_ checks.

To only run tests matching a certain name pattern, use the ``-k`` command line
switch, e.g., ``-k pool`` will run the pooling layer tests only.

To land in a ``pdb`` debug prompt on a failure to inspect it more closely, use
the ``--pdb`` switch.

Finally, for a loop-on-failing mode, do ``pip install pytest-xdist`` and run
``py.test -f``. This will pause after the run, wait for any source file to
change and run all previously failing tests again.


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



.. _issue tracker on GitHub: https://github.com/Lasagne/Lasagne/issues
.. _our mailing list: https://groups.google.com/forum/#!forum/lasagne-users
