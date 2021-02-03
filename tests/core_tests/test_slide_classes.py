import pytest

from pathml.core.slide_classes import HESlide, RGBSlide
from pathml.core.slide_data import SlideData
from pathml.core.slide_backends import OpenSlideBackend
from pathml.core.tile import Tile


@pytest.fixture
def he_slide():
    wsi = HESlide("tests/testdata/small_HE.svs")
    return wsi


def test_he(he_slide):
    assert isinstance(he_slide, SlideData)
    assert isinstance(he_slide, RGBSlide)
    assert isinstance(he_slide.slide, OpenSlideBackend)


@pytest.mark.parametrize("shape", [500, (500, 400)])
@pytest.mark.parametrize("stride", [None, 1000])
@pytest.mark.parametrize("pad", [True, False])
@pytest.mark.parametrize("level", [0])
def test_generate_tiles_he(he_slide, shape, stride, pad, level):
    for tile in he_slide.generate_tiles(shape = shape, stride = stride, pad = pad, level = level):
        assert isinstance(tile, Tile)


@pytest.mark.parametrize("pad", [True, False])
def test_generate_tiles_padding(he_slide, pad):
    # calc by hand how many tiles we expect to get with/without padding for small_HE.svs with specified tile size
    pass


@pytest.mark.parametrize("slide", [HESlide, RGBSlide])
def test_repr(slide):
    s = slide("tests/testdata/small_HE.svs")
    repr(s)
