"""
Copyright 2021, Dana-Farber Cancer Institute and Weill Cornell Medicine
License: GNU GPL 2.0
"""

import numpy as np
import pytest

from pathml.core import BioFormatsBackend, DICOMBackend, OpenSlideBackend, Tile


def openslide_backend():
    return OpenSlideBackend("tests/testdata/small_HE.svs")


def bioformats_backend():
    return BioFormatsBackend("tests/testdata/smalltif.tif")


def bioformats_backend_qptiff():
    """Bioformats behaves differently for images that have shape XYZCT, so we need to test that separately"""
    return BioFormatsBackend("tests/testdata/small_vectra.qptiff")


def dicom_backend():
    return DICOMBackend("tests/testdata/small_dicom.dcm")


# test each method for each backend


@pytest.mark.parametrize(
    "backend",
    [
        openslide_backend(),
        bioformats_backend(),
        bioformats_backend_qptiff(),
    ],
)
@pytest.mark.parametrize("location", [(0, 0), (50, 60)])
@pytest.mark.parametrize("size", [50, (50, 100)])
@pytest.mark.parametrize("level", [None, 0])
def test_extract_region(backend, location, size, level):
    region = backend.extract_region(location=location, size=size, level=level)
    assert isinstance(region, np.ndarray)
    assert region.dtype == np.uint8


@pytest.mark.parametrize("backend", [bioformats_backend(), bioformats_backend_qptiff()])
@pytest.mark.parametrize("normalize", [True, False])
def test_extract_region_bioformats_no_normalize(backend, normalize):
    reg = backend.extract_region(location=(0, 0), size=10, normalize=normalize)
    if normalize:
        assert reg.dtype == np.dtype("uint8")
    else:
        assert reg.dtype == np.dtype("float64")


@pytest.mark.parametrize("shape", [500, (500, 250)])
def test_extract_region_openslide(example_slide_data, shape):
    """
    make sure that the coordinates for openslide backend are in correct order.
    Issue #181 caused by discrepancy between openslide (x, y) coord system and the rest of pathml which uses (i, j)
    """
    # get the array for the image
    # note that calling np.array() on the PIL image automatically takes care of flipping it to (i, j) coords!!
    raw_im_array = np.array(
        example_slide_data.slide.slide.read_region(
            location=(0, 0),
            size=example_slide_data.slide.slide.level_dimensions[0],
            level=0,
        )
    )
    raw_im_array = raw_im_array[:, :, 0:3]
    if isinstance(shape, int):
        shape = (shape, shape)
    h, w = shape
    for tile in example_slide_data.generate_tiles(shape=shape):
        i, j = tile.coords
        assert np.array_equal(tile.image, raw_im_array[i : i + h, j : j + w, :])
        assert tile.image.shape[0:2] == shape


def test_extract_region_levels_openslide():
    # testing bug when reading regions from levels above 0
    # see: https://github.com/Dana-Farber-AIOS/pathml/issues/240
    # this is caused because openslide.read_region expects coords in the level 0 reference coord system
    # but in pathml, we use coords relative to each level
    # so to convert, we need to stretch coords by the downsample factor to convert to level 0 system
    #   before passing them to the openslide API
    # this multilevel testing image is taken from the openslide test suite:
    # https://github.com/openslide/openslide-python/blob/main/tests/boxes.tiff
    wsi = OpenSlideBackend("tests/testdata/small_HE_levels.tiff")
    # at level zero, the tile at (100, 100) of size 100px is entirely blue, i.e. pixel values [0, 0, 255]
    # so this should be true as well for the corresponding regions in lower levels
    # level 0
    im_level0 = wsi.extract_region(location=(100, 100), size=100, level=0)
    assert np.array_equal(im_level0[:, :, 2], 255 * np.ones((100, 100)))
    # level 1
    im_level1 = wsi.extract_region(location=(50, 50), size=50, level=1)
    assert np.array_equal(im_level1[:, :, 2], 255 * np.ones((50, 50)))
    # level 2
    im_level2 = wsi.extract_region(location=(25, 25), size=25, level=2)
    assert np.array_equal(im_level2[:, :, 2], 255 * np.ones((25, 25)))


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
        (bioformats_backend(), (480, 640)),
        (dicom_backend(), (2638, 3236)),
        (bioformats_backend_qptiff(), (1440, 1920)),
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


@pytest.mark.parametrize("backend", [BioFormatsBackend, OpenSlideBackend])
@pytest.mark.parametrize("pad", [True, False])
def test_tile_generator_with_pad_evenly_divide(backend, pad):
    """When tile shape evenly divides slide shape, padding should make no difference"""
    slide = backend("tests/testdata/smalltif.tif")
    tile_shape = 160
    shape = (480, 640)
    tiles = list(slide.generate_tiles(shape=tile_shape, pad=pad))
    assert len(tiles) == np.prod([shape_i / tile_shape for shape_i in shape])


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


@pytest.mark.parametrize("backend", [bioformats_backend(), bioformats_backend_qptiff()])
@pytest.mark.parametrize("normalize", [True, False])
def test_generate_tiles_bioformats_no_normalize(backend, normalize):
    gen = backend.generate_tiles(shape=10, normalize=normalize)
    tile1 = next(gen)
    if normalize:
        assert tile1.image.dtype == np.dtype("uint8")
    else:
        assert tile1.image.dtype == np.dtype("float64")


@pytest.mark.parametrize(
    "backend,shape",
    [
        (openslide_backend(), (2967, 2220)),
        (bioformats_backend(), (480, 640)),
        (bioformats_backend_qptiff(), (1440, 1920)),
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
    vectra_slide.generate_tiles(shape=10)
