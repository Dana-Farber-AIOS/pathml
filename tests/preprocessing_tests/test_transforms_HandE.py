import numpy as np
import pytest

from pathml.preprocessing.wsi import HESlide
from pathml.preprocessing.transforms_HandE import (
    TissueDetectionHE, BlackPenDetectionHE, BasicNucleusDetectionHE
)


@pytest.fixture(scope = "module")
def example_he_tile():
    wsi = HESlide(path = "tests/testdata/CMU-1-Small-Region.svs")
    slide_data = wsi.load_data(level = 0, location = (900, 800), size = (100, 100))
    return slide_data.image


@pytest.mark.parametrize('transform', [
    TissueDetectionHE(), BlackPenDetectionHE(), BasicNucleusDetectionHE()
])
def test_segmentation_transform(example_he_tile, transform):
    out = transform.apply(example_he_tile)
    assert out.shape == example_he_tile.shape[0:2]
    assert out.dtype == np.uint8
