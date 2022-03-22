"""
Copyright 2021, Dana-Farber Cancer Institute and Weill Cornell Medicine
License: GNU GPL 2.0
"""

import os
import sys
from loguru import logger
from pathlib import Path
import typing
import functools


class path_ml_logger:

    pathml_log_activation = logger.disable("pathml")
    log_activation = logger.disable(__name__)

    @staticmethod
    def toggle_logging(
        enabled=True,
        sink="~/Documents/pathml_logs/log.log",
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
        handler_id = None
        if (
            enabled
        ):  # leaving this in the event where env variables are to be used os.getenv("ENABLE_PATHML_LOGS", 'False').lower() in ('true', '1', 't'):
            path_ml_logger.pathml_log_activation = logger.enable("pathml")
            path_ml_logger.log_activation = logger.enable(__name__)
            handler_id = logger.add(
                sink=sink, level=level, format="{message}", **kwargs
            )

        else:
            pathml_log_activation = logger.disable("pathml")
            log_activation = logger.disable(__name__)

            logger.info("Disabled Logging For PathML!")

        logger.info("Enabled Logging For PathML!")

        return handler_id


def logging_example():
    logger.info("PathML Logging Example")


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
