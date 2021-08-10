"""
Copyright 2021, Dana-Farber Cancer Institute and Weill Cornell Medicine
License: GNU GPL 2.0
"""

import pytest
import numpy as np

from pathml.core import OpenSlideBackend, DICOMBackend, BioFormatsBackend, Tile


def openslide_backend():
    return OpenSlideBackend("tests/testdata/small_HE.svs")


def bioformats_backend():
    return BioFormatsBackend("tests/testdata/smalltif.tif")


def bioformats_backend_qptiff():
    """Bioformats behaves differently for images that have shape XYZCT, so we need to test that separately"""
    return BioFormatsBackend("tests/testdata/small_vectra.qptiff")


def dicom_backend():
    return DICOMBackend("tests/testdata/small_dicom.dcm")


## test each method for each backend


@pytest.mark.parametrize(
    "backend", [openslide_backend(), bioformats_backend(), bioformats_backend_qptiff()]
)
@pytest.mark.parametrize("location", [(0, 0), (50, 100)])
@pytest.mark.parametrize("size", [50, (50, 100)])
@pytest.mark.parametrize("level", [None, 0])
def test_extract_region(backend, location, size, level):
    region = backend.extract_region(location=location, size=size, level=level)
    assert isinstance(region, np.ndarray)
    assert region.dtype == np.uint8


# separate dicom tests because dicom frame requires 500x500 tiles while bioformats has dim <500
@pytest.mark.parametrize("backend", [dicom_backend()])
@pytest.mark.parametrize("location", [(0, 0), (500, 500)])
@pytest.mark.parametrize("size", [500, (500, 500)])
@pytest.mark.parametrize("level", [None, 0])
def test_extract_region_dicom(backend, location, size, level):
    region = backend.extract_region(location=location, size=size, level=level)
    assert isinstance(region, np.ndarray)
    assert region.dtype == np.uint8


@pytest.mark.parametrize(
    "backend,shape",
    [
        (bioformats_backend(), (640, 480)),
        (dicom_backend(), (2638, 3236)),
        (bioformats_backend_qptiff(), (1920, 1440)),
    ],
)
@pytest.mark.parametrize("pad", [True, False])
@pytest.mark.parametrize("tile_shape", [500, (500, 500)])
def test_tile_generator(backend, shape, tile_shape, pad):
    tiles = list(backend.generate_tiles(shape=tile_shape, stride=500, pad=pad))
    tile_shape = (tile_shape, tile_shape) if isinstance(tile_shape, int) else tile_shape
    if not pad:
        assert len(tiles) == np.prod(
            [shape[i] // tile_shape[i] for i in range(len(shape))]
        )
    else:
        assert len(tiles) == np.prod(
            [1 + (shape[i] // tile_shape[i]) for i in range(len(shape))]
        )
    assert all([isinstance(tile, Tile) for tile in tiles])


@pytest.mark.parametrize("backend,shape", [(openslide_backend(), (2967, 2220))])
@pytest.mark.parametrize("pad", [True, False])
@pytest.mark.parametrize("tile_shape", [500, (500, 500)])
@pytest.mark.parametrize("level", [None, 0])
def test_tile_generator_with_level(backend, shape, tile_shape, pad, level):
    tiles = list(
        backend.generate_tiles(shape=tile_shape, stride=500, pad=pad, level=level)
    )
    tile_shape = (tile_shape, tile_shape) if isinstance(tile_shape, int) else tile_shape
    if not pad:
        assert len(tiles) == np.prod(
            [shape[i] // tile_shape[i] for i in range(len(shape))]
        )
    else:
        assert len(tiles) == np.prod(
            [1 + (shape[i] // tile_shape[i]) for i in range(len(shape))]
        )
    assert all([isinstance(tile, Tile) for tile in tiles])


@pytest.mark.parametrize(
    "backend,shape",
    [
        (openslide_backend(), (2967, 2220)),
        (bioformats_backend(), (640, 480)),
        (bioformats_backend_qptiff(), (1920, 1440)),
        (dicom_backend(), (2638, 3236)),
    ],
)
def test_get_image_shape(backend, shape):
    assert backend.get_image_shape() == shape


@pytest.mark.parametrize(
    "backend", [openslide_backend(), bioformats_backend(), bioformats_backend_qptiff()]
)
def test_get_thumbnail(backend):
    print(dir(backend))
    print(type(backend))
    thumbnail = backend.get_thumbnail(size=(500, 500))
    assert isinstance(thumbnail, np.ndarray)


@pytest.mark.parametrize(
    "backend",
    [
        openslide_backend(),
        bioformats_backend(),
        dicom_backend(),
        bioformats_backend_qptiff(),
    ],
)
def test_repr(backend):
    # make sure there are no errors during repr or str
    repr(backend)
    print(backend)


def test_dicom_coords_index_conversion():
    backend = dicom_backend()
    # shape of the dicom image: (2638, 3236)
    # frame size: (500, 500)
    check = {0: (0, 0), 1: (0, 500), 14: (1000, 0), 41: (2500, 3000)}
    for index, coords in check.items():
        assert backend._index_to_coords(index) == coords
        assert backend._coords_to_index(coords) == index


# this test takes a long time, so skip by running 'python -m pytest -m "not slow"'
@pytest.mark.slow
def test_bioformats_vm_handling(vectra_slide):
    tiles = vectra_slide.generate_tiles(shape=10)
    list_of_tiles = list(tiles)
