import pytest
import numpy as np
import cv2
import openslide

from pathml.core.slide_classes import HESlide
from pathml.core.tile import Tile
from pathml.core.masks import Masks


@pytest.fixture
def tileHE():
    """
    Example of pathml.core.Tile object
    """
    s = openslide.open_slide("tests/testdata/small_HE.svs")
    im_image = s.read_region(level = 0, location = (900, 800), size = (500, 500))
    im_np = np.asarray(im_image)
    im_np_rgb = cv2.cvtColor(im_np, cv2.COLOR_RGBA2RGB)

    # make mask object
    masks = np.random.randint(low = 1, high = 255, size = (im_np_rgb.shape[0], im_np_rgb.shape[1]), dtype = np.uint8)
    masks = Masks(masks = {"testmask" : masks})

    # labels dict
    labels = {"test_str_label": "stringlabel", "test_np_array_label": np.ones(shape = (2, 3, 3))}

    tile = Tile(image = im_np_rgb, coords = (0, 0), masks = masks, slidetype = HESlide, labels = labels)
    return tile
