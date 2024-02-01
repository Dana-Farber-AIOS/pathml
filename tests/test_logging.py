"""
Copyright 2021, Dana-Farber Cancer Institute and Weill Cornell Medicine
License: GNU GPL 2.0
"""

import logging

import pytest
from _pytest.logging import caplog as _caplog  # noqa: F401
from loguru import logger

from pathml._logging import PathMLLogger as pml
from pathml._logging import logger_wraps
from pathml.utils import _test_log


@pytest.fixture
def caplog(_caplog):  # noqa: F811
    """
    loguru is not built on the standard library logging module, but pytest's logging functionality is.
    In this fixture, we make sure that all logs to loguru are also propagated to the standard library logger so that
    we can test them with pytest.
    This overrides the `caplog` fixture that comes builtin to pytest

    Based on: https://youtu.be/eFdVlyAGeZU
    """

    class PropagateHandler(logging.Handler):
        def emit(self, record):
            logging.getLogger(record.name).handle(record)

    handler_id = logger.add(PropagateHandler(), format="{message}")
    yield _caplog
    logger.remove(handler_id)


def test_toggle_logging(caplog):
    _test_log("this should not be logged because we haven't enabled logging yet")
    assert (
        "this should not be logged because we haven't enabled logging yet"
        not in caplog.text
    )
    pml.enable(sink=caplog.handler)
    _test_log("this should be logged now")
    assert "this should be logged now" in caplog.text
    pml.disable()
    _test_log("this should definitely not be logged")
    assert "this should definitely not be logged" not in caplog.text


@logger_wraps(entry=True, exit=True, level="DEBUG")
def dummy_function(arg1, arg2, kwarg1=None):
    return arg1 + arg2


def test_logger_wraps(caplog):
    # Call the dummy function
    result = dummy_function(5, 10, kwarg1="test")

    # Expected log messages
    expected_entry_log = (
        "Entering 'dummy_function' (args=(5, 10), kwargs={'kwarg1': 'test'})"
    )
    expected_exit_log = "Exiting 'dummy_function' (result=15)"

    # Check if the expected logs are in the caplog
    assert expected_entry_log in caplog.text, "Entry log message not found"
    assert expected_exit_log in caplog.text, "Exit log message not found"

    # Check the result of the function
    assert result == 15, "Function result is incorrect"
