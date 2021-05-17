"""
Copyright 2021, Dana-Farber Cancer Institute and Weill Cornell Medicine
License: GNU GPL 2.0
"""

import pytest
import numpy as np
import cv2
import openslide
import javabridge

from pathml.core import HESlide, Tile, Masks


def pytest_sessionfinish(session, exitstatus):
    """
    Pytest will not terminate if javabridge is not killed.
    But if we terminate javabridge in BioFormatsBackend, we can not spawn another javabridge in the same thread.

    This Pytest sessionfinish hook runs automatically at the end of testing.
    """
    javabridge.kill_vm()


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

@pytest.fixture
def tileVectra():
    """
    Example of pathml.core.Tile representation of Vectra image
    """
    slidedata = read(path="tests/testdata/vectra.tif", backend = "bioformats")
    region = data.slide.extract_region(location=(0,0,0,0,0), size=(500,500,1,7,1))

    # make mask object
    masks = np.random.randint(low = 1, high = 255, size = (im_np_rgb.shape[0], im_np_rgb.shape[1]), dtype = np.uint8)
    masks = Masks(masks = {"testmask" : masks})

    # labels dict
    labels = {"test_str_label": "stringlabel", "test_np_array_label": np.ones(shape = (2, 3, 3))}

    tile = Tile(image = region, coords = (0,0), masks = None, slidetype = VectraSlide, labels = labels)
    return tile

@pytest.fixture
def tileCODEX():
    """
    Example of pathml.core.Tile representation of CODEX image
    """
    pass
