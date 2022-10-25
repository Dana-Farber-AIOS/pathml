"""
Copyright 2021, Dana-Farber Cancer Institute and Weill Cornell Medicine
License: GNU GPL 2.0
"""

import numpy as np
import pytest

from pathml.preprocessing.tiling import extract_tiles, extract_tiles_with_mask


@pytest.mark.parametrize("tile_size", [5, 20])
@pytest.mark.parametrize("stride", [None, 1, 5])
@pytest.mark.parametrize("n_channels", [1, 3, 11])
def test_extract_tiles(n_channels, stride, tile_size):
    # square
    arr_size = 100
    arr = np.arange(arr_size * arr_size * n_channels).reshape(
        (arr_size, arr_size, n_channels)
    )
    tiled = extract_tiles(arr, tile_size=tile_size, stride=stride)
    if stride is None:
        stride = tile_size
    n_tiles_expected = 1 + (arr_size - tile_size) / stride
    assert tiled.shape == (n_tiles_expected**2, tile_size, tile_size, n_channels)
    assert np.array_equal(tiled[0, ...], arr[0:tile_size, 0:tile_size, :])


@pytest.mark.parametrize("stride", [None, 5])
@pytest.mark.parametrize("n_channels_arr", [3])
@pytest.mark.parametrize("n_channels_mask", [5])
@pytest.mark.parametrize("tile_size", [5, 10, 25])
def test_extract_tiles_with_mask(n_channels_arr, n_channels_mask, stride, tile_size):
    arr_size = 100
    arr = np.arange(arr_size * arr_size * n_channels_arr).reshape(
        (arr_size, arr_size, n_channels_arr)
    )

    mask = np.zeros(shape=(arr_size, arr_size, n_channels_mask), dtype=np.uint8)
    mask[0:25, 0:25, ...] = 1

    tiled = extract_tiles_with_mask(
        arr, mask=mask, tile_size=tile_size, stride=stride, threshold=0.99
    )

    if stride is None:
        stride = tile_size

    # since the mask only has ones from [0:25, 0:25]
    # and we set a high threshold (almost 1)
    # n_expected should be the same as if we only tiled a (25 x 25) array
    n_tiles_expected = 1 + (25 - tile_size) // stride

    assert tiled.shape == (n_tiles_expected**2, tile_size, tile_size, n_channels_arr)
