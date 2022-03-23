"""
Copyright 2021, Dana-Farber Cancer Institute and Weill Cornell Medicine
License: GNU GPL 2.0
"""

from loguru import logger
import functools
import sys


class path_ml_logger:

    logger.disable("pathml")
    logger.disable(__name__)

    @staticmethod
    def toggle_logging(
        enabled=True,
        sink=sys.stderr,
        level="DEBUG",
        fmt="{time:HH:mm:ss} | {level:<8} | {module} | {function: ^15} | {line: >3} | {message}",
        **kwargs
    ):
        """
        Turn on or off logging for PathML

        Args:
            enabled(bool):
                Whether to save logs. Defaults to True.
            sink(str, <class '_io.TextIOWrapper'>):
                where the sink goes
            level(str):
                level of logs to capture
            fmt(str):
                formatting for the log message. default: '{time:HH:mm:ss} | {level:<8} | {module} | {function: ^15} | {line: >3} | {message}'
            **kwargs(dict, optional):
                additional options passed to configure logger. See: :_Example: 'https://loguru.readthedocs.io/en/stable/api/logger.html#loguru._logger.Logger.configure'

        Returns:
            logger(<class 'loguru._logger.Logger'>):
                The updated logger with the applicable user settings applied.
        """
        if enabled:
            logger.remove()
            handler_id = logger.add(sink=sink, level=level, format=fmt, **kwargs)
            logger.enable("pathml")
            logger.enable(__name__)
            logger.info("Enabled Logging For PathML!")
            return handler_id

        else:
            logger.disable("pathml")
            logger.disable(__name__)
            logger.info(
                "Disabled Logging For PathML! If you are seeing this, there is a problem"
            )

        logger.info("If you are seeing this, there is a problem")


# courtesy of the people at loguru
# https://loguru.readthedocs.io/en/stable/resources/recipes.html#:~:text=or%20fallback%20policy.-,Logging%20entry%20and%20exit%20of%20functions%20with%20a%20decorator,-%EF%83%81
def logger_wraps(*, entry=True, exit=True, level="DEBUG"):
    def wrapper(func):
        name = func.__name__

        @functools.wraps(func)
        def wrapped(*args, **kwargs):
            logger_ = logger.opt(depth=1)
            if entry:
                logger_.bind(enter_exit=True).log(
                    level, "Entering '{}' (args={}, kwargs={})", name, args, kwargs
                )
            result = func(*args, **kwargs)
            if exit:
                logger_.bind(enter_exit=True).log(
                    level, "Exiting '{}' (result={})", name, result
                )
            return result

        return wrapped

    return wrapper
