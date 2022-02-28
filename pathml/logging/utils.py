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

global log_activation = logger.disable("pathml")

# check to see if user has enabled pathml logs 
try:
    enable_logging = os.environ['ENABLE_PATHML_LOGS']
    if enable_logging:
        global log_activation = logger.enable("pathml")
except KeyError as e:
    pass


fmt = "{time:HH:mm:ss} | {level:<8} | {module} | {function: ^15} | {line: >3} | {message}"
config = {
    "handlers": [
        #dict(sink=sys.stderr, colorize=True, format=fmt, level="DEBUG", diagnose=True),
        dict(sink="./pathml_logs/trace.log", format=fmt, level="TRACE", diagnose=True),
        dict(sink="./pathml_logs/debug.log", format=fmt, level="DEBUG", diagnose=True),
        dict(sink="./pathml_logs/info.log", format=fmt, level="INFO", diagnose=True),
        dict(sink="./pathml_logs/success.log", format=fmt, level="SUCCESS", diagnose=True),
        dict(sink="./pathml_logs/warning.log", format=fmt, level="WARNING", diagnose=True, backtrace=True),
        dict(sink="./pathml_logs/error.log", format=fmt, level="ERROR", diagnose=True, backtrace=True),
        dict(sink="./pathml_logs/critical.log", format=fmt, level="CRITICAL", diagnose=True, backtrace=True),
        dict(sink="./pathml_logs/enter_exit.log", filter=lambda record: "enter_exit" in record["extra"], format=fmt),
        dict(sink="./pathml_logs/CORE.log", filter=lambda record: "core_specific" in record["extra"], format=fmt),
        dict(sink="./pathml_logs/DATASET.log", filter=lambda record: "dataset_specific" in record["extra"], format=fmt),
        dict(sink="./pathml_logs/ML.log", filter=lambda record: "ml_specific" in record["extra"], format=fmt),
        dict(sink="./pathml_logs/PREPROCESSING.log", filter=lambda record: "preprocessing_specific" in record["extra"], format=fmt),
        ]}

logger.configure(**config)

# courtesy of the people at loguru
# https://loguru.readthedocs.io/en/stable/resources/recipes.html#:~:text=or%20fallback%20policy.-,Logging%20entry%20and%20exit%20of%20functions%20with%20a%20decorator,-%EF%83%81
def logger_wraps(*, entry=True, exit=True, level="DEBUG"):

    def wrapper(func):
        name = func.__name__

        @functools.wraps(func)
        def wrapped(*args, **kwargs):
            logger_ = logger.opt(depth=1)
            if entry:
                logger_.bind(enter_exit=True).log(level, "Entering '{}' (args={}, kwargs={})", name, args, kwargs)
            result = func(*args, **kwargs)
            if exit:
                logger_.bind(enter_exit=True).log(level, "Exiting '{}' (result={})", name, result)
            return result

        return wrapped

    return wrapper
