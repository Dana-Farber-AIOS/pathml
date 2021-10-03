"""
Copyright 2021, Dana-Farber Cancer Institute and Weill Cornell Medicine
License: GNU GPL 2.0
"""
import os
import shutil
import urllib

import anndata
import numpy as np
# define fixtures here, and use them throughout the other tests in core_tests/
import pytest
from pathml.core import SlideData, SlideDataset, Tile, VectraSlide, types


@pytest.fixture
def example_slide_data():
    labs = {
        "test_string_label": "testlabel",
        "test_array_label": np.array([2, 3, 4]),
        "test_int_label": 3,
        "test_float_label": 3.0,
        "test_bool_label": True,
    }
    wsi = SlideData(
        "tests/testdata/small_HE.svs", name=f"test", labels=labs, backend="openslide"
    )
    return wsi


@pytest.fixture
def example_slide_data_with_tiles(tile):
    labs = {
        "test_string_label": "testlabel",
        "test_array_label": np.array([2, 3, 4]),
        "test_int_label": 3,
        "test_float_label": 3.0,
        "test_bool_label": True,
    }
    adata = anndata.AnnData()
    wsi = SlideData(
        "tests/testdata/small_HE.svs",
        name=f"test",
        labels=labs,
        backend="openslide",
        tiles=[tile],
        counts=adata,
    )
    return wsi


@pytest.fixture()
def slide_dataset(example_slide_data_with_tiles):
    n = 4
    labs = {
        "test_string_label": "testlabel",
        "test_array_label": np.array([2, 3, 4]),
        "test_int_label": 3,
        "test_float_label": 3.0,
        "test_bool_label": True,
    }
    slide_list = [
        SlideData(
            "tests/testdata/small_HE.svs",
            name=f"slide{i}",
            labels=labs,
            backend="openslide",
        )
        for i in range(n)
    ]
    slide_dataset = SlideDataset(slide_list)
    return slide_dataset


@pytest.fixture()
def slide_dataset_with_tiles(tile, example_slide_data_with_tiles):
    n = 4
    labs = {
        "test_string_label": "testlabel",
        "test_array_label": np.array([2, 3, 4]),
        "test_int_label": 3,
        "test_float_label": 3.0,
        "test_bool_label": True,
    }
    slide_list = [
        SlideData(
            "tests/testdata/small_HE.svs",
            name=f"slide{i}",
            labels=labs,
            backend="openslide",
            tiles=[tile],
        )
        for i in range(n)
    ]
    slide_dataset = SlideDataset(slide_list)
    return slide_dataset


@pytest.fixture
def vectra_slide():
    temp_path = "tests/testdata/small_vectra.qptiff"
    vectra_slide = VectraSlide(temp_path, backend="bioformats", slide_type=types.Vectra)
    return vectra_slide


class Helpers:
    @staticmethod
    def download_from_url(url, download_dir, name=None):
        """
        Download a file from a url to destination directory.
        If the file already exists, does not download.

        Args:
            url (str): Url of file to download
            download_dir (str): Directory where file will be downloaded
            name (str, optional): Name of saved file. If ``None``, uses base name of url argument. Defaults to ``None``.

        See: https://stackoverflow.com/questions/7243750/download-file-from-web-in-python-3
        """
        if name is None:
            name = os.path.basename(url)

        path = os.path.join(download_dir, name)

        if os.path.exists(path):
            return
        else:
            os.makedirs(download_dir, exist_ok=True)

            # Download the file from `url` and save it locally under `file_name`:
            with urllib.request.urlopen(url) as response, open(path, "wb") as out_file:
                shutil.copyfileobj(response, out_file)


@pytest.fixture
def helpers():
    return Helpers
