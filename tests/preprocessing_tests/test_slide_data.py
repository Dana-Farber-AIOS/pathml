import pytest
import numpy as np
import openslide
import cv2

from pathml.preprocessing.slide_data import SlideData


@pytest.fixture(scope = "module")
def example_slide_data():
    path = "tests/testdata/CMU-1-Small-Region.svs"
    slide = openslide.open_slide(path)
    image_array_pil = slide.read_region(location = (900, 800), level = 0, size = (100, 100))
    # note that PIL uses (width, height) but when converting to numpy we get the correct (height, width) dims
    image_array_rgba = np.asarray(image_array_pil, dtype = np.uint8)
    image_array = cv2.cvtColor(image_array_rgba, cv2.COLOR_RGBA2RGB)
    out = SlideData(wsi = slide, image = image_array)
    return out


@pytest.fixture()
def example_mask_100_100():
    out = np.zeros((100, 100), dtype = np.uint8)
    out[25:75, 25:75] = 1
    out[45:55, 45:55] = 0
    return out


def test_slide_data(example_slide_data, example_mask_100_100):
    # test adding mask
    example_slide_data.mask = example_mask_100_100
    assert example_slide_data.mask.shape == (100, 100)
    assert example_slide_data.mask.dtype == np.uint8
    # add another mask, make sure it stacks
    example_slide_data.mask = example_mask_100_100
    assert example_slide_data.mask.shape == (100, 100, 2)
    assert example_slide_data.mask.dtype == np.uint8
