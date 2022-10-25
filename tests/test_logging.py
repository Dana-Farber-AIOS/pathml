"""
Copyright 2021, Dana-Farber Cancer Institute and Weill Cornell Medicine
License: GNU GPL 2.0
"""
import logging

import pytest
from _pytest.logging import caplog as _caplog  # noqa: F401
from loguru import logger

from pathml._logging import PathMLLogger as pml
from pathml.utils import _test_log


@pytest.fixture
def caplog(_caplog):
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
