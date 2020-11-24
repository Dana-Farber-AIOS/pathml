import numpy as np
import pytest

from pathml.preprocessing.transforms_HandE import (
    TissueDetectionHE, BlackPenDetectionHE, BasicNucleusDetectionHE
)


@pytest.mark.parametrize('transform', [
    TissueDetectionHE(), BlackPenDetectionHE(), BasicNucleusDetectionHE()
])
def test_segmentation_transform(example_he_tile, transform):
    out = transform.apply(example_he_tile)
    assert out.shape == example_he_tile.shape[0:2]
    assert out.dtype == np.uint8
