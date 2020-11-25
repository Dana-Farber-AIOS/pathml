import numpy as np
import pytest

import pathml.preprocessing.tiling as tiling


@pytest.fixture
def array_5_5_3():
    return np.arange(75).reshape((5, 5, 3))


@pytest.fixture
def mask_5_5():
    m = np.zeros((5, 5, 1))
    m[0:3, 0:3] = 1
    return m


@pytest.fixture
def mask_5_5_all_zeros():
    m = np.zeros((5, 5, 1))
    return m


# checks that the function doesn't work for incorrect inputs
@pytest.mark.parametrize("incorrect_input", [None, True, 5, [5, 4, 3], "string", {"dict": "testing"}])
def test_extract_tiles_array_incorrect_input(incorrect_input):
    with pytest.raises(AttributeError):
        tiling.extract_tiles_array(incorrect_input, tile_size = 4, stride = 1)


def test_extract_tiles_array(array_5_5_3):
    a = array_5_5_3
    b = tiling.extract_tiles_array(im = a, tile_size = 4, stride = 1)
    assert b.shape == (2, 2, 4, 4, 3)
    assert np.allclose(b[0, 0, ...], a[0:4, 0:4, :])


def test_extract_tiles(array_5_5_3):
    tiles = tiling.extract_tiles(im = array_5_5_3, tile_size = 4, stride = 1)
    assert np.all([isinstance(t, tiling.Tile) for t in tiles])
    assert len(tiles) == 4
    assert tiles[3].i == 1 and tiles[3].j == 1
    assert np.array_equal(tiles[3].array, array_5_5_3[1:, 1:, :])


def test_extract_tiles_with_mask(array_5_5_3, mask_5_5):
    tiles = tiling.extract_tiles_with_mask(im = array_5_5_3, tile_size = 4,
                                           stride = 1, mask = mask_5_5, mask_thresholds = 0.5)
    assert len(tiles) == 1
    assert tiles[0].i == 0 and tiles[0].j == 0


def test_extract_tiles_empty_mask(array_5_5_3, mask_5_5_all_zeros):
    tiles = tiling.extract_tiles_with_mask(im = array_5_5_3, tile_size = 4,
                                           stride = 1, mask = mask_5_5_all_zeros, mask_thresholds = 0.5)
    assert len(tiles) == 0


def test_tile_extractor(example_slide_data_with_mask):
    extractor1 = tiling.SimpleTileExtractor(tile_size = 25)
    extractor1.apply(example_slide_data_with_mask)
    assert len(example_slide_data_with_mask.tiles) == 4
    # test specifying a mask ix
    extractor2 = tiling.SimpleTileExtractor(tile_size = 25, mask_ix = 0)
    extractor2.apply(example_slide_data_with_mask)
    assert len(example_slide_data_with_mask.tiles) == 4
    extractor3 = tiling.SimpleTileExtractor(tile_size = 25, mask_ix = 1)
    extractor3.apply(example_slide_data_with_mask)
    assert len(example_slide_data_with_mask.tiles) == 16
    # test specifying single mask threshold
    extractor4 = tiling.SimpleTileExtractor(tile_size = 25, mask_thresholds = 0.01)
    extractor4.apply(example_slide_data_with_mask)
    assert len(example_slide_data_with_mask.tiles) == 9
    # test specifying multiple mask thresholds
    extractor5 = tiling.SimpleTileExtractor(tile_size = 25, mask_thresholds = [0.01, 0.99])
    extractor5.apply(example_slide_data_with_mask)
    assert len(example_slide_data_with_mask.tiles) == 9
    # specify mask ix and multiple thresholds
    extractor6 = tiling.SimpleTileExtractor(tile_size = 25, mask_thresholds = [0.01, 0.04], mask_ix = 1)
    extractor6.apply(example_slide_data_with_mask)
    assert len(example_slide_data_with_mask.tiles) == 16

