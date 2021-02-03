import pytest
import numpy as np

from pathml.core.slide_backends import OpenSlideBackend, DICOMBackend, BioFormatsBackend


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


def test_openslide_get_level_shape(wsi_HE):
    pass


def test_openslide_get_thumbnail(wsi_HE):
    pass
