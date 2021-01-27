import pytest
import numpy as np
import cv2
import openslide


from pathml.core.tile import Tile
from pathml.core.masks import Masks


@pytest.fixture
def tileHE():
    """ Example of pathml.core.Tile object """
    s = openslide.open_slide("tests/testdata/small_HE.svs")
    im_image = s.read_region(level = 0, location = (900, 800), size = (500, 500))
    im_np = np.asarray(im_image)
    im_np_rgb = cv2.cvtColor(im_np, cv2.COLOR_RGBA2RGB)

    mask = np.zeros((im_np_rgb.shape[0], im_np_rgb.shape[1]), dtype = np.uint8)
    center = np.ones((50, 50))
    center_circle = cv2.getStructuringElement(shape = cv2.MORPH_ELLIPSE, ksize = (25, 25))
    center[12:37, 12:37] -= center_circle
    mask[25:75, 25:75] = center

    mask[200:400, 200:300] = 1

    m = Masks(masks = {"testmask" : mask})
    tile = Tile(image = im_np_rgb, coords = (0, 0), masks = m, slidetype = "HE")
    return tile
