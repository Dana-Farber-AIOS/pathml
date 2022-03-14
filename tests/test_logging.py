from pathml._logging import enable_logging, logger


def test_logging_enables(caplog):
    enable_logging()

    assert "Enabled Logging For PathML!" in caplog.text


def test_logging_trace(caplog):
    logger.trace("trace log goes here")

    # due to handler that is set within pathml._logging to only collect logs that are above level DEBUG
    assert "trace log goes here" not in caplog.text


def test_logging_debug(caplog):
    logger.debug("debug log goes here")

    assert "debug log goes here" in caplog.text


def test_logging_info(caplog):
    logger.info("info log goes here")

    assert "info log goes here" in caplog.text


def test_logging_success(caplog):
    logger.success("success log goes here")

    assert "success log goes here" in caplog.text


def test_logging_warning(caplog):
    logger.warning("warning log goes here")

    assert "warning log goes here" in caplog.text


def test_logging_error(caplog):
    logger.error("error log goes here")

    assert "error log goes here" in caplog.text


def test_logging_critical(caplog):
    logger.critical("critical log goes here")

    assert "critical log goes here" in caplog.text
