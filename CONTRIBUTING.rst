Contributing
************

``PathML`` is an open source project. Consider contributing to benefit the entire community!

There are many ways to contribute to PathML, including:

* Submitting bug reports
* Submitting feature requests
* Writing documentation
* Fixing bugs
* Writing code for new features
* Sharing trained model parameters [coming soon]
* Sharing ``PathML`` with colleagues, students, etc.


Submitting a bug report
=======================
Report bugs or errors by filing an issue on GitHub. Make sure to include the following information:

* Short description of the bug
* Minimum working example to reproduce the bug
* Expected result vs. actual result
* Any other useful information

If a bug cannot be reproduced by someone else on a different machine, it will usually be hard to identify
what is causing it.

Requesting a new feature
=========================
Request a new feature by filing an issue on GitHub. Make sure to include the following information:

* Description of the feature
* Pseudocode of how the feature might work (if applicable)
* Any other useful information

For developers
==============

Coordinate system conventions
-----------------------------

With multiple tools for interacting with matrices/images, conflicting coordinate systems has been a common source of
bugs. This is typically caused when mixing up (X, Y) coordinate systems and (i, j) coordinate systems. **To avoid these
issues, we have adopted the (i, j) coordinate convention throughout PathML.** This follows the convention used by
NumPy and many others, where ``A[i, j]`` refers to the element of matrix A in the ith row, jth column.
Developers should be careful about coordinate systems and make the necessary adjustments when using third-party tools
so that users of PathML can rely on a consistent coordinate system when using our tools.

Setting up a local development environment
-------------------------------------------

1. Create a new fork of the ``PathML`` repository
2. Clone your fork to your local machine
3. Set up the PathML environment: ``conda env create -f environment.yml; conda activate pathml``
4. Install PathML: ``pip install -e .``
5. Install pre-commit hooks: ``pre-commit install``

Running tests
-------------

To run the full testing suite:

.. code-block::

    python -m pytest

Some tests are known to be very slow. To skip them, run instead:

.. code-block::

    python -m pytest -m "not slow"


Building documentation locally
------------------------------

.. code-block::

    cd docs                                     # enter docs directory
    pip install -r readthedocs-requirements.txt     # install packages to build docs
    make html                                   # build docs in html format

Then use your favorite web browser to open ``pathml/docs/build/html/index.html``

Checking code coverage
----------------------

.. code-block::

    conda install coverage  # install coverage package for code coverage
    coverage run            # run tests and calculate code coverage
    coverage report         # view coverage report
    coverage html           # optionally generate HTML coverage report

How to contribute code, documentation, etc.
-------------------------------------------

1. Create a new GitHub issue for what you will be working on, if one does not already exist
2. Create a local development environment (see above)
3. Create a new branch from the dev branch and implement your changes
4. Write new tests as needed to maintain code coverage
5. Ensure that all tests pass
6. Push your changes and open a pull request on GitHub referencing the corresponding issue
7. Respond to discussion/feedback about the pull request, make changes as necessary

Versioning and Distributing
---------------------------

We use `semantic versioning`_. The version is tracked in ``pathml/_version.py`` and should be updated there as required.
When new code is merged to the master branch on GitHub, the version should be incremented and a new release should be
pushed. Releases can be created using the GitHub website interface, and should be tagged in version format
(e.g., "v1.0.0" for version 1.0.0) and include release notes indicating what has changed.
Once a new release is created, GitHub Actions workflows will automatically build and publish the updated package on
PyPI and TestPyPI, as well as build and publish the Docker image to Docker Hub.

Code Quality
------------

We want PathML to be built on high-quality code. However, the idea of "code quality" is somewhat subjective.
If the code works perfectly but cannot be read and understood by someone else, then it can't be maintained,
and this accumulated tech debt is something we want to avoid.
Writing code that "works", i.e. does what you want it to do, is therefore necessary but not sufficient.
Good code also demands efficiency, consistency, good design, clarity, and many other factors.

Here are some general tips and ideas:

- Strive to make code concise, but not at the expense of clarity.
- Seek efficient and general designs, but avoid premature optimization.
- Prefer informative variable names.
- Encapsulate code in functions or objects.
- Comment, comment, comment your code.

All code should be reviewed by someone else before merging.

We use `Black`_ to enforce consistency of code style.

Documentation Standards
-----------------------

All code should be documented, including docstrings for users AND inline comments for
other developers whenever possible! Both are crucial for ensuring long-term usability and maintainability.
Documentation is automatically generated using the Sphinx `autodoc`_ and `napoleon`_ extensions from
properly formatted Google-style docstrings.
All documentation (including docstrings) is written in `reStructuredText`_ format.
See this `docstring example`_ to get started.

Testing Standards
-----------------

All code should be accompanied by tests, whenever possible, to ensure that everything is working as intended.

The type of testing required may vary depending on the type of contribution:

- New features should use tests to ensure that the code is working as intended, e.g. comparing output of
  a function with the expected output.
- Bug fixes should first add a failing test, then make it pass by fixing the bug

No pull request can be merged unless all tests pass.
We aim to maintain good code coverage for the testing suite (target >90%).
We use the `pytest`_ testing framework.
To run the test suite and check code coverage:

.. code-block::

    conda install pytest    # first install pytest package
    conda install coverage  # install coverage package for code coverage
    coverage run            # run tests and calculate code coverage
    coverage report         # view coverage report
    coverage html           # optionally generate HTML coverage report

We suggest using test-driven development when applicable. I.e., if you're fixing a bug or adding new features,
write the tests first! (they should all fail). Then, write the actual code. When all tests pass, you know
that your implementation is working. This helps ensure that all code is tested and that the tests are testing
what we want them to.

Thank You!
==========

Thank you for helping make ``PathML`` better!


.. _pytest: https://docs.pytest.org/en/stable/
.. _autodoc: https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html
.. _reStructuredText: https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html
.. _docstring example: https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html
.. _napoleon: https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html
.. _Black: https://black.readthedocs.io/en/stable
.. _semantic versioning: https://semver.org/
