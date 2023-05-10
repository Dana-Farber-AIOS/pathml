"""
Copyright 2021, Dana-Farber Cancer Institute and Weill Cornell Medicine
License: GNU GPL 2.0
"""

import numpy as np
import pytest

from pathml.core import Tile


@pytest.mark.parametrize(
    "incorrect_input", ["string", True, 5, [5, 4, 3], {"dict": "testing"}]
)
@pytest.mark.parametrize(
    "incorrect_input_name", [True, 5, [5, 4, 3], {"dict": "testing"}]
)
def test_init_incorrect_input(incorrect_input, incorrect_input_name):
    with pytest.raises(AssertionError):
        Tile(incorrect_input, coords=(1, 3))
        Tile(np.random.randn((224, 224, 3)), coords=incorrect_input)
        Tile(np.random.randn((224, 224, 3)), coords=(1, 3), masks=incorrect_input)
        Tile(np.random.randn((224, 224, 3)), coords=(1, 3), name=incorrect_input_name)


def test_tile():
    # test init
    tile = Tile(np.ones((224, 224, 3)), coords=(1, 3))
    assert (tile.image).all() == (np.ones((224, 224, 3))).all()
    # test repr
    print(tile)


def test_tile_mask():
    # https://github.com/Dana-Farber-AIOS/pathml/issues/175
    tile = Tile(np.ones((224, 224, 3)), coords=(1, 3), masks={})
    print(tile)
