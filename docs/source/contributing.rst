Contributing
============

``PathML`` is an open source project. Consider contributing to benefit the entire community!

How to Contribute
-----------------
There are many ways to contribute to PathML, including:

* Submitting bug reports
* Submitting feature requests
* Writing documentation
* Fixing bugs
* Writing code for new features
* Sharing workflows
* Sharing trained model parameters
* Sharing PathML with colleagues, students, etc.


Documentation
-------------

All code should be documented, including docstrings for users AND inline comments for
other developers whenever possible! Both are crucial for ensuring long-term usability and maintainability.

Documentation is automatically generated using the Sphinx `autodoc`_ extension from properly formatted docstrings.
See this `docstring example`_ to get started.

To do: transition to `Google style docstrings`_.

All documentation (including docstrings) are written in `reStructuredText`_ format.

.. _autodoc: https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html
.. _reStructuredText: https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html
.. _docstring example: https://sphinx-rtd-tutorial.readthedocs.io/en/latest/docstrings.html
.. _Google style docstrings: https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html

Testing
-------

All new code should be accompanied by tests, whenever possible. We use the `pytest`_ testing framework.
All tests should pass for new code, and new tests should be added as necessary when fixing bugs.

.. _pytest: https://docs.pytest.org/en/stable/