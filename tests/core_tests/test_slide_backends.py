import pytest
import numpy as np
import openslide

from pathml.core.slide_backends import OpenSlideBackend, DICOMBackend, BioFormatsBackend


@pytest.fixture
def wsi_HE():
    wsi = OpenSlideBackend("tests/testdata/small_HE.svs")
    return wsi


@pytest.fixture
def multiparametric_image():
    image = BioFormatsBackend("tests/testdata/smalltif.svs") 
    return image


@pytest.mark.parametrize("location", [(0, 0), (1000, 1000)])
@pytest.mark.parametrize("size", [100, (500, 200)])
@pytest.mark.parametrize("level", [None, 0])
def test_openslide_extract_tile(wsi_HE, location, size, level):
    region = wsi_HE.extract_region(location = location, size = size, level = level)
    assert isinstance(region, np.ndarray)
    assert region.dtype == np.uint8


@pytest.mark.parametrize("location", [(0, 0), (1000, 1000)])
@pytest.mark.parametrize("size", [100, (500, 200)])
def test_bioformats_extract_tile(multiparametric_image, location, size):
    region = multiparametric_image.extract_region(location = location, size = size)
    assert isinstance(region, np.ndarray)
    assert region.dtype == np.uint8


def test_openslide_get_image_shape(wsi_HE):
    openslide_wsi = openslide.open_slide("tests/testdata/small_HE.svs")
    # need to flip the dims because openslide uses (x, y) convention but we use (i, j)
    assert wsi_HE.get_image_shape() == openslide_wsi.level_dimensions[0][::-1]
    assert wsi_HE.get_image_shape(level = 0) == openslide_wsi.level_dimensions[0][::-1]


def test_openslide_get_thumbnail(wsi_HE):
    thumbnail = wsi_HE.get_thumbnail(size = (500, 500))
    assert isinstance(thumbnail, np.ndarray)


def test_bioformats_get_thumbnail(multiparametric_image):
    thumbnail = multiparametric_image.get_thumbnail(size = (500, 500))
    assert isinstance(thumbnail, np.ndarray)


def test_openslide_repr():
    backend = OpenSlideBackend("tests/testdata/small_HE.svs")
    repr(backend)


def test_bioformats_repr():
    backend = BioFormatsBackend("tests/testdata/smalltif.svs")
    repr(backend)
