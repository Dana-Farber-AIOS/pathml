"""
Copyright 2021, Dana-Farber Cancer Institute and Weill Cornell Medicine
License: GNU GPL 2.0
"""

import shutil
import urllib
from pathlib import Path
from unittest.mock import MagicMock, patch

import h5py
import numpy as np
import pytest

from pathml.core.utils import writedataframeh5
from pathml.datasets.deepfocus import DeepFocusDataModule, DeepFocusDataset


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
    target_dir = "dftests"
    with pytest.raises(AssertionError):
        DeepFocusDataModule(target_dir, download=False)
    shutil.rmtree(target_dir)


def check_deepfocus_data_urls():
    # make sure that the urls for the pannuke data are still valid!
    url = "https://zenodo.org/record/1134848/files/outoffocus2017_patches5Classification.h5"
    r = urllib.request.urlopen(url)
    # HTTP status code 200 means "OK"
    assert r.getcode() == 200


def check_wrong_path_download_false_fails():
    with pytest.raises(AssertionError):
        DeepFocusDataModule(data_dir="wrong/path/to/pannuke", download=False)


# TODO: How to test datamodule arguments if checksum without downloading the full dataset?


def create_mock_h5py_file():
    """
    Create a mock h5py file with a smaller dataset.
    """
    mock_h5py_file = MagicMock()
    mock_X = np.random.rand(100, 224, 224, 3)  # Smaller image dimensions
    mock_Y = np.random.randint(0, 2, size=(100,))  # Binary labels

    # Mock the dataset and slicing
    mock_h5py_file.__getitem__.side_effect = lambda k: {"X": mock_X, "Y": mock_Y}[k]
    return mock_h5py_file


@pytest.mark.parametrize("fold_ix", [1, 2, 3, None])
def test_deepfocus_dataset(fold_ix):
    with patch("h5py.File", return_value=create_mock_h5py_file()):
        data_dir = Path("fake/path")  # Using pathlib.Path for fake data directory
        deepfocus_dataset = DeepFocusDataset(data_dir=data_dir, fold_ix=fold_ix)

        # Testing data retrieval
        img, label = deepfocus_dataset[0]
        assert img.shape == (224, 224, 3), "Image shape is incorrect"
        assert isinstance(label, np.integer), "Label type is incorrect"

        # Additional checks for specific folds
        if fold_ix == 1:
            # Check if data is from the training set
            assert len(deepfocus_dataset) == 100, "Training set size is incorrect"
        elif fold_ix == 2:
            # Check if data is from the validation set
            assert len(deepfocus_dataset) == 0, "Validation set size is incorrect"
        elif fold_ix == 3:
            # Check if data is from the test set
            assert len(deepfocus_dataset) == 0, "Test set size is incorrect"
        else:
            # If fold_ix is None, it should return the entire dataset
            assert (
                len(deepfocus_dataset) == 100
            ), "Dataset size is incorrect for the entire dataset"
