import numpy as np
import pytest

from pathml.preprocessing.masks import Masks
from pathml.preprocessing.wsi import HESlide

from collections import OrderedDict

@pytest.fixture
def openslide_example():
    im = openslide.open_slide("tests/testdata/CMU-1-Small-Region.svs")
    im_image = im.read_region(level = 0, location=(0,0), size=im.level_dimensions[0])
    im_np = np.asarray(im_image)
    im_np_rgb = cv2.cvtColor(im_np, cv2.COLOR_RGBA2RGB)
    return im_np_rgb

def test_mask_wsi(openslide_example):
    mask1 = np.full_like(openslide_example)
    mask2 = np.empty_like(openslide_example)
    mask3 = np.full_like(openslide_example)
    wsi = HESlide("tests/testdata/CMU-1-Small-Region.svs", masks = {'one':mask1,'two':mask2})
    wsi = wsi.load_data()
    wsi.masks.add('three', mask3)
    assert wsi.masks() == OrderedDict({'one':mask1,'two':mask2,'three':mask3}) 
    # TODO: test chunks
