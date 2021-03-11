import pytest
import numpy as np
import openslide

from pathml.core.slide_backends import OpenSlideBackend, DICOMBackend, BioFormatsBackend
from pathml.core.tile import Tile


@pytest.fixture
def wsi_HE():
    wsi = OpenSlideBackend("tests/testdata/small_HE.svs")
    return wsi


@pytest.mark.parametrize("location", [(0, 0), (1000, 1000)])
@pytest.mark.parametrize("size", [100, (500, 200)])
@pytest.mark.parametrize("level", [None, 0])
def test_openslide_extract_tile(wsi_HE, location, size, level):
    region = wsi_HE.extract_region(location = location, size = size, level = level)
    assert isinstance(region, np.ndarray)
    assert region.dtype == np.uint8


@pytest.mark.parametrize("pad", [True, False])
def test_openslide_tile_generator(wsi_HE, pad):
    tiles = list(wsi_HE.generate_tiles(shape = 500, stride = 500, pad = pad, level = 0))
    # small_HE.svs has dimensions (2220, 2967)
    if pad:
        assert len(tiles) == 5 * 6
    else:
        assert len(tiles) == 4 * 5
    assert all([isinstance(tile, Tile) for tile in tiles])


def test_openslide_get_image_shape(wsi_HE):
    openslide_wsi = openslide.open_slide("tests/testdata/small_HE.svs")

    # need to flip the dims because openslide uses (x, y) convention but we use (i, j)
    assert wsi_HE.get_image_shape() == openslide_wsi.level_dimensions[0][::-1]
    assert wsi_HE.get_image_shape(level = 0) == openslide_wsi.level_dimensions[0][::-1]


def test_openslide_get_thumbnail(wsi_HE):
    thumbnail = wsi_HE.get_thumbnail(size = (500, 500))
    assert isinstance(thumbnail, np.ndarray)


def test_openslide_repr():
    backend = OpenSlideBackend("tests/testdata/small_HE.svs")
    repr(backend)
