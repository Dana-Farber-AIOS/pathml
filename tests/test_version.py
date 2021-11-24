"""
Copyright 2021, Dana-Farber Cancer Institute and Weill Cornell Medicine
License: GNU GPL 2.0
"""

import re
import pathml


def test_version_format():
    # semantic versioning format: major.minor.patch
    assert re.match("^[\d]+.[\d]+.[\d]+$", pathml.__version__)
