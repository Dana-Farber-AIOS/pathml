"""
Copyright 2021, Dana-Farber Cancer Institute and Weill Cornell Medicine
License: GNU GPL 2.0
"""

import pytest
import urllib
from pathlib import Path

from pathml.datasets import PesoDataModule

@pytest.fixture
def create_incomplete_peso_data():
    """
    create fake peso data simulating incomplete download
    """
    target_dir = Path("dftests")
    target_dir.mkdir(parents=True, exist_ok=True)
    f = h5py.File(target_dir / Path("fake.h5py"), "w")
    X = np.random.randint(low=1, high=254, size=(1000, 64, 64, 3), dtype=np.uint8)
    writedataframeh5(f, "X", X)
    Y = np.random.randint(low=1, high=5, size=(204000,), dtype=np.uint8)
    writedataframeh5(f, "Y", Y)
    return f


def test_incomplete_fails(create_incomplete_peso_data):
    f = create_incomplete_peso_data
    target_dir = "dftests"
    with pytest.raises(AssertionError):
        PesoDataModule(target_dir, download=False)
    shutil.rmtree(target_dir)


def check_peso_data_urls():
    # make sure that the urls for the pannuke data are still valid!
    url = f"https://zenodo.org/record/1485967/files/"
    r = urllib.request.urlopen(url)
    # HTTP status code 200 means "OK"
    assert r.getcode() == 200

def check_wrong_path_download_false_fails():
    with pytest.raises(AssertionError):
        peso = PesoDataModule(
            data_dir="wrong/path/to/peso", download=False
        )
