"""
Copyright 2021, Dana-Farber Cancer Institute and Weill Cornell Medicine
License: GNU GPL 2.0
"""

# define fixtures here, and use them throughout the other tests in core_tests/
import pytest
import numpy as np

from pathml.core import Tiles, Masks, SlideData, HESlide, OpenSlideBackend, SlideDataset, Tile


@pytest.fixture
def tile_nomasks():
    testtile = Tile(np.random.randn(224, 224, 3), coords = (1, 3))
    return testtile


@pytest.fixture
def tile_nomasks():
    tile_shape = (224, 224, 3)
    labs = {"test_string_label": "testlabel", "test_array_label": np.array([2,3,4]), "test_int_label": 3, "test_float_label": 3.0}
    im = np.random.randint(low = 1, high = 255, dtype = np.uint8, size = tile_shape)
    return Tile(image = im, coords = (1, 3), labels = labs)


@pytest.fixture
def tile_withmasks(tile_nomasks):
    mask_size = tile_nomasks.image.shape[0:2]
    mask_dict = {f"mask{i}": np.random.choice([False, True], size = mask_size) for i in range(5)}
    masks = Masks(mask_dict)
    tile_withmasks = tile_nomasks
    tile_withmasks.masks = masks
    return tile_withmasks


@pytest.fixture
def example_slide_data():
    labs = {"test_string_label": "testlabel", "test_array_label": np.array([2,3,4]), "test_int_label": 3, "test_float_label": 3.0}
    wsi = SlideData("tests/testdata/small_HE.svs", name = f"test_array_in_labels",
                    labels = labs, slide_backend = OpenSlideBackend)
    return wsi


@pytest.fixture
def example_slide_data_with_tiles(tile_withmasks):
    tiles_dict = {(42, 42): tile_withmasks}
    tiles = Tiles(tiles_dict)
    # second label tests ordering
    # do not test np.ndarrays, these should be masks
    labs = {"test_string_label": "testlabel", "test_array_label": np.array([2,3,4]), "test_int_label": 3, "test_float_label": 3.0}
    wsi = SlideData("tests/testdata/small_HE.svs", name = f"test_array_in_labels",
                    labels = labs, slide_backend = OpenSlideBackend, tiles = tiles)
    return wsi


@pytest.fixture()
def slide_dataset(example_slide_data_with_tiles):
    n = 4
    labs = {"test_string_label": "testlabel", "test_array_label": np.array([2,3,4]), "test_int_label": 3, "test_float_label": 3.0}
    slide_list = [SlideData("tests/testdata/small_HE.svs",
                            name = f"slide{i}",
                            labels = labs,
                            slide_backend = OpenSlideBackend) for i in range(n)]
    slide_dataset = SlideDataset(slide_list)
    return slide_dataset


@pytest.fixture()
def slide_dataset_with_tiles(tile_withmasks, example_slide_data_with_tiles):
    n = 4
    tiles_dict = {(42, 42): tile_withmasks}
    tiles = Tiles(tiles_dict)
    labs = {"test_string_label": "testlabel", "test_array_label": np.array([2,3,4]), "test_int_label": 3, "test_float_label": 3.0}
    slide_list = [SlideData("tests/testdata/small_HE.svs",
                            name = f"slide{i}",
                            labels = labs,
                            slide_backend = OpenSlideBackend,
                            tiles = tiles) for i in range(n)]
    slide_dataset = SlideDataset(slide_list)
    return slide_dataset
