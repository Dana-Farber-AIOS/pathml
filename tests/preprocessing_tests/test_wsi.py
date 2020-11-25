import openslide
import pytest
import numpy as np
import cv2

from pathml.preprocessing.base import BaseSlide, Slide2d, RGBSlide
from pathml.preprocessing.wsi import HESlide


@pytest.fixture
def openslide_example():
    im = openslide.open_slide("tests/testdata/CMU-1-Small-Region.svs")
    im_image = im.read_region(level = 0, location=(0,0), size=im.level_dimensions[0])
    im_np = np.asarray(im_image)
    im_np_rgb = cv2.cvtColor(im_np, cv2.COLOR_RGBA2RGB)
    return im_np_rgb


def test_HE_slide(openslide_example):
    wsi = HESlide(path = "tests/testdata/CMU-1-Small-Region.svs")
    assert wsi.name == "CMU-1-Small-Region"
    assert wsi.path == "tests/testdata/CMU-1-Small-Region.svs"
    slide_data = wsi.load_data(level = 0, location = (0, 0), size = None)
    assert np.array_equal(slide_data.image, openslide_example)
    slide_data2 = wsi.load_data(level = 0, location = (200, 200), size = (200, 200))
    assert np.array_equal(slide_data2.image, openslide_example[200:400, 200:400, :])

    # check that the hierarchy of the slide class structure is working properly
    assert isinstance(wsi, BaseSlide)
    assert isinstance(wsi, Slide2d)
    assert isinstance(wsi, RGBSlide)

    
@pytest.mark.parametrize("stride,size,n_expected", [(500, 500, 4*5), (100, 500, 18*25), (None, 500, 4*5), (500, 2000, 2)])
def test_HE_chunks(stride, size, n_expected):
    wsi = HESlide(path = "tests/testdata/CMU-1-Small-Region.svs")
    chunk_counter = 0
    for chunk in wsi.chunks(level = 0, size = size, stride = stride, pad = False):
        assert chunk.shape == (size, size, 3)
        chunk_counter += 1
    assert chunk_counter == n_expected


@pytest.mark.parametrize("stride,n_expected", [(500, 5*6), (100, 23*30), (None, 5*6)])
def test_HE_chunks_padding(stride, n_expected):
    wsi = HESlide(path = "tests/testdata/CMU-1-Small-Region.svs")
    chunk_counter = 0
    for chunk in wsi.chunks(level = 0, size = 500, stride = stride, pad = True):
        assert chunk.shape == (500, 500, 3)
        chunk_counter += 1
    assert chunk_counter == n_expected
