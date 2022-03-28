"""
Copyright 2021, Dana-Farber Cancer Institute and Weill Cornell Medicine
License: GNU GPL 2.0
"""
import os
import sys
from loguru import logger
import pytest
from pathml._logging import PathMLLogger as pml
from _pytest.logging import LogCaptureFixture


# inspiration from: https://loguru.readthedocs.io/en/stable/resources/migration.html#making-things-work-with-pytest-and-caplog
@pytest.fixture
def caplog_enable(caplog: LogCaptureFixture):
    handler_id = pml.toggle_logging(fmt="{message}", sink=caplog.handler)
    yield caplog
    logger.remove(handler_id)


@pytest.fixture
def caplog_disable(caplog: LogCaptureFixture):
    handler_id = pml.toggle_logging(enabled=False, sink=caplog.handler)
    yield caplog
    logger.remove(handler_id)


def test_logging_enables(caplog_enable):
    logger.info("Enabled Logging For PathML!")
    assert "Enabled Logging For PathML!" in caplog_enable.text


def test_logging_disables(caplog_disable):
    logger.info("Disabled Logging For PathML!")
    assert "Disabled Logging For PathML!" not in caplog_disable.text


def test_logging_trace(caplog_enable):
    logger.trace("trace log goes here")
    # due to handler that is set within pathml._logging to only collect logs that are above level DEBUG
    assert "trace log goes here" not in caplog_enable.text


def test_logging_debug(caplog_enable):

    logger.debug("debug log goes here")
    assert "debug log goes here" in caplog_enable.text


def test_logging_info(caplog_enable):
    logger.info("info log goes here")

    assert "info log goes here" in caplog_enable.text


def test_logging_success(caplog_enable):
    logger.success("success log goes here")

    assert "success log goes here" in caplog_enable.text


def test_logging_warning(caplog_enable):
    logger.warning("warning log goes here")

    assert "warning log goes here" in caplog_enable.text


def test_logging_error(caplog_enable):
    logger.error("error log goes here")

    assert "error log goes here" in caplog_enable.text


def test_logging_critical(caplog_enable):
    logger.critical("critical log goes here")

    assert "critical log goes here" in caplog_enable.text
