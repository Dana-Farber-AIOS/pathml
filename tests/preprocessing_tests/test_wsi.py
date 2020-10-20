import openslide
import pytest
import numpy as np
import cv2

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


@pytest.mark.parametrize("stride,n_expected", [(500, 4*5), (100, 22*29), (None, 4*5)])
def test_HE_chunks(stride, n_expected):
    wsi = HESlide(path = "tests/testdata/CMU-1-Small-Region.svs")
    chunk_counter = 0
    for chunk in wsi.chunks(level = 0, size = 500, stride = stride):
        assert chunk.shape == (500, 500, 3)
        chunk_counter += 1
    assert chunk_counter == n_expected
