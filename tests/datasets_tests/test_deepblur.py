"""
Copyright 2021, Dana-Farber Cancer Institute and Weill Cornell Medicine
License: GNU GPL 2.0
"""

import shutil
import urllib
from pathlib import Path

import h5py
import numpy as np
import pytest

from pathml.core.utils import writedataframeh5
from pathml.datasets import DeepFocusDataModule


@pytest.fixture
def create_incomplete_deepfocus_data():
    """
    create fake deepfocus data simulating incomplete download
    """
    target_dir = Path("dftests")
    target_dir.mkdir(parents=True, exist_ok=True)
    f = h5py.File(target_dir / Path("outoffocus2017_patches5Classification.h5"), "w")
    X = np.random.randint(low=1, high=254, size=(1000, 64, 64, 3), dtype=np.uint8)
    writedataframeh5(f, "X", X)
    Y = np.random.randint(low=1, high=5, size=(204000,), dtype=np.uint8)
    writedataframeh5(f, "Y", Y)
    return f


def test_incomplete_fails(create_incomplete_deepfocus_data):
    f = create_incomplete_deepfocus_data
    target_dir = "dftests"
    with pytest.raises(AssertionError):
        DeepFocusDataModule(target_dir, download=False)
    shutil.rmtree(target_dir)


def check_deepfocus_data_urls():
    # make sure that the urls for the pannuke data are still valid!
    url = f"https://zenodo.org/record/1134848/files/outoffocus2017_patches5Classification.h5"
    r = urllib.request.urlopen(url)
    # HTTP status code 200 means "OK"
    assert r.getcode() == 200


def check_wrong_path_download_false_fails():
    with pytest.raises(AssertionError):
        deepfocus = DeepFocusDataModule(
            data_dir="wrong/path/to/pannuke", download=False
        )


# TODO: How to test datamodule arguments if checksum without downloading the full dataset?
