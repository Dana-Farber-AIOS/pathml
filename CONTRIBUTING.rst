Contributing
************

``PathML`` is an open source project. Consider contributing to benefit the entire community!

Ways to Contribute
==================
There are many ways to contribute to PathML, including:

* Submitting bug reports
* Submitting feature requests
* Writing documentation
* Fixing bugs
* Writing code for new features
* Sharing workflows [coming soon]
* Sharing trained model parameters [coming soon]
* Sharing ``PathML`` with colleagues, students, etc.


Submitting a bug report
=======================
Report bugs or errors by filing an issue on GitHub. Make sure to include the following information:

* Short description of the bug
* Minimum working example to reproduce the bug
* Expected result
* Actual result
* Any other useful information

Requesting a new feature
=========================
Request a new feature by filing an issue on GitHub. Make sure to include the following information:

* Description of the feature
* Pseudocode of how the feature might work (if applicable)
* Any other useful information

Writing code and/or documentation
==============================================
Here's how to contribute code, documentation, etc.

1. Create a new GitHub issue for what you will be working on, if one does not already exist
2. Create a new fork of the ``PathML`` repository
3. Clone your fork to your local machine
4. Ensure that your environment is properly configured and that all tests pass
5. Implement your changes
6. Write new tests as needed to maintain code coverage
7. Ensure that all tests still pass
8. Commit your changes and submit a pull request referencing the corresponding issue
9. Respond to discussion/feedback about the pull request. Make changes as necessary.

Code Standards
==============


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


Documentation Standards
-----------------------

All code should be documented, including docstrings for users AND inline comments for
other developers whenever possible! Both are crucial for ensuring long-term usability and maintainability.
Documentation is automatically generated using the Sphinx `autodoc`_ and `napoleon`_ extensions from
properly formatted Google-style docstrings.
All documentation (including docstrings) is written in `reStructuredText`_ format.
See this `docstring example`_ to get started.

To build documentation:

.. code-block::

    # first install packages for generating docs
    pip install ipython sphinx nbsphinx nbsphinx-link sphinx-rtd-theme
    cd docs                 # enter docs directory
    make html               # build docs in html format

Then use your favorite web browser to open ``pathml/docs/build/html/index.html``

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


Thank You!
==========
Thank you for helping make ``PathML`` better!


.. _pytest: https://docs.pytest.org/en/stable/
.. _autodoc: https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html
.. _reStructuredText: https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html
.. _docstring example: https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html
.. _napoleon: https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html