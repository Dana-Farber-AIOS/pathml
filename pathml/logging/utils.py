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

def dev_mode(dev_bool=True):
    global log_activation
    if dev_bool:
        log_activation = logger.enable("pathml")
        log.info("Enabled logging into pathml.log file.")
    else:
        log_activation = logger.disable("pathml")
        #this should not be logged. If it has then something bad happened
        log.info("Disabled logging into pathml.log file.")
        log.info("Wait, you shouldnt be seeing this, something is broken if you are reading this")

config = {
    "handlers": [
        {
            "sink": sys.stdout, 
            "colorize": True,
            "format": "<green>{time:YYYY-MM-DD at HH:mm:ss}<green> | {level} | {message}"
        }]}

logger.configure(**config)

core_level = logger.level("CORE", no=21, color="<yellow>", icon='🫀')
dataset_level = logger.level("DATASET", no=22, color="<magenta>", icon='🗄️ ')
ml_level = logger.level("ML", no=23, color="<cyan>", icon='🧠')
preprocessing_level = logger.level("PREPROCESSING", no=24, color="<green>", icon='✨')

# courtesy of the people at loguru
# https://loguru.readthedocs.io/en/stable/resources/recipes.html#:~:text=or%20fallback%20policy.-,Logging%20entry%20and%20exit%20of%20functions%20with%20a%20decorator,-%EF%83%81
def logger_wraps(*, entry=True, exit=True, level="DEBUG"):

    def wrapper(func):
        name = func.__name__

        @functools.wraps(func)
        def wrapped(*args, **kwargs):
            logger_ = logger.opt(depth=1)
            if entry:
                logger_.log(level, "Entering '{}' (args={}, kwargs={})", name, args, kwargs)
            result = func(*args, **kwargs)
            if exit:
                logger_.log(level, "Exiting '{}' (result={})", name, result)
            return result

        return wrapped

    return wrapper
