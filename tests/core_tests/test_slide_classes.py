import pytest

from pathml.core.slide_classes import HESlide, RGBSlide, MultiparametricSlide 
from pathml.core.slide_data import SlideData
from pathml.core.slide_backends import OpenSlideBackend
from pathml.core.tile import Tile


@pytest.fixture
def he_slide():
    wsi = HESlide("tests/testdata/small_HE.svs")
    return wsi


@pytest.fixture
def multiparametric_slide():
    wsi = MultiparametricSlide("tests/testdata/smalltif.svs") 
    return wsi


def test_he(he_slide):
    assert isinstance(he_slide, SlideData)
    assert isinstance(he_slide, RGBSlide)
    assert isinstance(he_slide.slide, OpenSlideBackend)


def test_multiparametric(multiparametric_slide):
    assert isinstance(multiparametric_slide, SlideData)
    assert isinstance(multiparametric_slide.slide, BioFormatsBackend)


@pytest.mark.parametrize("shape", [500, (500, 400)])
@pytest.mark.parametrize("stride", [None, 1000])
@pytest.mark.parametrize("pad", [True, False])
@pytest.mark.parametrize("level", [0])
def test_generate_tiles_he(he_slide, shape, stride, pad, level):
    for tile in he_slide.generate_tiles(shape = shape, stride = stride, pad = pad, level = level):
        assert isinstance(tile, Tile)


@pytest.mark.parametrize("shape", [500, (500, 400)])
@pytest.mark.parametrize("stride", [None, 1000])
@pytest.mark.parametrize("pad", [True, False])
def test_generate_tiles_multiparametric(multiparameteric_slide, shape, stride, pad):
    for tile in multiparametric_slide.generate_tiles(shape = shape, stride = stride, pad = pad):
        assert isinstance(tile, Tile)


@pytest.mark.parametrize("pad", [True, False])
def test_generate_tiles_padding(he_slide, pad):
    shape = 300
    stride = 300

    tiles = list(he_slide.generate_tiles(shape = shape, stride = stride, pad = pad))

    # he_slide.slide.get_image_shape() --> (2967, 2220)
    # if no padding, expect: 9*7 = 63 tiles
    # if padding, expect: 10*8 - 80 tiles

    if not pad:
        assert len(tiles) == 63
    else:
        assert len(tiles) == 80


@pytest.mark.parametrize("slide", [HESlide, RGBSlide])
def test_repr(slide):
    s = slide("tests/testdata/small_HE.svs")
    repr(s)
