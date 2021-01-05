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

1. Create an issue for what you will be working on, if one does not already exist 
2. Create a new fork of the ``PathML`` repository
3. Clone your fork to your local machine
4. Ensure that your environment is properly configured and that all tests pass
5. Implement your changes
6. Write new tests as needed to maintain code coverage
7. Ensure that all tests still pass
8. Commit your changes and submit a pull request reference the corresponding issue

Documentation Standards
=======================

All code should be documented, including docstrings for users AND inline comments for
other developers whenever possible! Both are crucial for ensuring long-term usability and maintainability.
Documentation is automatically generated using the Sphinx `autodoc`_ extension from properly formatted docstrings.
All documentation (including docstrings) are written in `reStructuredText`_ format.
See this `docstring example`_ to get started.

To build documentation:

.. code-block::

    conda install sphinx    # install sphinx package for generating docs
    cd docs                 # enter docs directory
    make html               # build docs in html format

Testing Standards
=================

All new code should be accompanied by tests, whenever possible, to maintain good code coverage.
We use the `pytest`_ testing framework.
All tests should pass for new code, and new tests should be added as necessary when fixing bugs.

To run tests and check code coverage:

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
.. _docstring example: https://sphinx-rtd-tutorial.readthedocs.io/en/latest/docstrings.html
