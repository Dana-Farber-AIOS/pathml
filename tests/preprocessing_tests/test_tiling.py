import numpy as np
import pytest

from pathml.preprocessing.tiling import extract_tiles


@pytest.mark.parametrize("tile_size", [5, 20])
@pytest.mark.parametrize("stride", [None, 1, 5])
@pytest.mark.parametrize("n_channels", [1, 3, 11])
def test_extract_tiles(n_channels, stride, tile_size):
    arr_size = 100 # square
    arr = np.arange(arr_size*arr_size*n_channels).reshape((arr_size, arr_size, n_channels))
    tiled = extract_tiles(arr, tile_size = tile_size, stride = stride)
    if stride is None:
        stride = tile_size
    n_tiles_expected = 1 + (arr_size - tile_size) / stride
    assert tiled.shape == (n_tiles_expected**2, tile_size, tile_size, n_channels)
    assert np.array_equal(tiled[0, ...], arr[0:tile_size, 0:tile_size, :])