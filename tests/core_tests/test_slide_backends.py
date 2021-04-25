import pytest
import numpy as np

from pathml.core.slide_backends import OpenSlideBackend, DICOMBackend, BioFormatsBackend
from pathml.core.tile import Tile


def openslide_backend():
    return OpenSlideBackend("tests/testdata/small_HE.svs")


def bioformats_backend():
    return BioFormatsBackend("tests/testdata/smalltif.tif")


def dicom_backend():
    return DICOMBackend("tests/testdata/small_dicom.dcm")


## test each method for each backend

@pytest.mark.parametrize("backend", [openslide_backend(), bioformats_backend(), dicom_backend()])
@pytest.mark.parametrize("location", [(0, 0), (1000, 1000)])
@pytest.mark.parametrize("size", [500, (500, 500)])
@pytest.mark.parametrize("level", [None, 0])
def test_openslide_extract_tile(backend, location, size, level):
    region = backend.extract_region(location = location, size = size, level = level)
    assert isinstance(region, np.ndarray)
    assert region.dtype == np.uint8


@pytest.mark.parametrize("backend,shape", [
    (openslide_backend(), (2220, 2967)),
    (bioformats_backend(), (640, 480)),
    (dicom_backend(), (None, None))
])
@pytest.mark.parametrize("pad", [True, False])
@pytest.mark.parametrize("tile_shape", [500, (500, 500)])
def test_openslide_tile_generator(backend, shape, tile_shape, pad):
    tiles = list(backend.generate_tiles(shape = tile_shape, stride = 500, pad = pad, level = 0))
    tile_shape = (tile_shape, tile_shape) if isinstance(tile_shape, int) else tile_shape
    if pad:
        assert len(tiles) == np.prod([shape[i] // tile_shape[i] for i in range(len(shape))])
    else:
        assert len(tiles) == np.prod([1 + shape[i] // tile_shape[i] for i in range(len(shape))])
    assert all([isinstance(tile, Tile) for tile in tiles])


@pytest.mark.parametrize("backend,shape", [
    (openslide_backend(), (2220, 2967)),
    (bioformats_backend(), (640, 480)),
    (dicom_backend(), (None, None))
])
def test_openslide_get_image_shape(backend, shape):
    assert backend.get_image_shape() == shape



@pytest.mark.parametrize("backend", [openslide_backend(), bioformats_backend(), dicom_backend()])
def test_get_thumbnail(backend):
    print(dir(backend))
    print(type(backend))
    thumbnail = backend.get_thumbnail(size = (500, 500))
    assert isinstance(thumbnail, np.ndarray)


@pytest.mark.parametrize("backend", [openslide_backend(), bioformats_backend(), dicom_backend()])
def test_repr(backend):
    # make sure there are no errors during repr
    print(backend)
