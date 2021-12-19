"""
Copyright 2021, Dana-Farber Cancer Institute and Weill Cornell Medicine
License: GNU GPL 2.0
"""

import pytest
import urllib.request


@pytest.mark.parametrize(
    "url",
    [
        "https://www.pathml.org",
        # Vignettes
        # "https://github.com/Dana-Farber-AIOS/pathml/tree/master/examples/vignettes/",
        # docs
        "https://pathml.readthedocs.io/en/latest/",
    ],
)
def test_urls(url):
    # Make sure that the urls linked in the manuscript are not broken!
    # This should be a complete list of all urls in the manuscript + supplemental materials
    r = urllib.request.urlopen(url)
    # HTTP status code 200 means "OK"
    assert r.getcode() == 200
