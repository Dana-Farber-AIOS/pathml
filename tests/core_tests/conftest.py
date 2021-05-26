"""
Copyright 2021, Dana-Farber Cancer Institute and Weill Cornell Medicine
License: GNU GPL 2.0
"""

# define fixtures here, and use them throughout the other tests in core_tests/
import pytest
import numpy as np

from pathml.core import Tiles, SlideData, SlideDataset


@pytest.fixture
def example_slide_data():
    labs = {"test_string_label": "testlabel", "test_array_label": np.array([2, 3, 4]),
            "test_int_label": 3, "test_float_label": 3.0}
    wsi = SlideData("tests/testdata/small_HE.svs", name = f"test",
                    labels = labs, backend = "openslide")
    return wsi


@pytest.fixture
def example_slide_data_with_tiles(tile):
    labs = {"test_string_label": "testlabel", "test_array_label": np.array([2, 3, 4]),
            "test_int_label": 3, "test_float_label": 3.0}
    wsi = SlideData("tests/testdata/small_HE.svs", name = f"test",
                    labels = labs, backend = "openslide", tiles = [tile])
    return wsi


@pytest.fixture()
def slide_dataset(example_slide_data_with_tiles):
    n = 4
    labs = {"test_string_label": "testlabel", "test_array_label": np.array([2, 3, 4]),
            "test_int_label": 3, "test_float_label": 3.0}
    slide_list = [SlideData("tests/testdata/small_HE.svs", name = f"slide{i}",
                            labels = labs, backend = "openslide") for i in range(n)]
    slide_dataset = SlideDataset(slide_list)
    return slide_dataset


@pytest.fixture()
def slide_dataset_with_tiles(tile, example_slide_data_with_tiles):
    n = 4
    labs = {"test_string_label": "testlabel", "test_array_label": np.array([2, 3, 4]),
            "test_int_label": 3, "test_float_label": 3.0}
    slide_list = [SlideData("tests/testdata/small_HE.svs", name = f"slide{i}",
                            labels = labs, backend = "openslide", tiles = [tile]) for i in range(n)]
    slide_dataset = SlideDataset(slide_list)
    return slide_dataset
