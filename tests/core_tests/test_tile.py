import pytest
import numpy as np
import string
import random

from pathml.core.tile import Tile

@pytest.fixture
def tile_nomasks():
    testtile = Tile(np.random.randn((224, 224, 3)), coords = (1,3))
    return testtile


@pytest.fixture
def tile_withmasks():
    if labeltype == str:
        letters = string.ascii_letters + string.digits
        maskdict = {}
        for i in range(50):
            randomkey = 'test' + ''.join(random.choice(letters) for _ in range(i))
            maskdict[randomkey] = np.random.randint(2, size = (224,224,3))
        masks = Masks(maskdict)
    return Tile(np.random.random_sample((224,224,3)), coords = (1,3), masks = masks)

@pytest.mark.parametrize("incorrect_input", ["string", True, 5, [5, 4, 3], {"dict": "testing"}])
@pytest.mark.parametrize("incorrect_input_name",  [True, 5, [5, 4, 3], {"dict": "testing"}])
def test_init_incorrect_input(incorrect_input):
    with pytest.raises(ValueError):
        testimage = Tile(incorrect_input, coords=(1,3))
        testcoords = Tile(np.random.randn((224,224,3)), coords=incorrect_input)
        testmasks = Tile(np.random.randn((224,224,3)), coords=(1,3), masks=incorrect_input)
        testname = Tile(np.random.randn((224,224,3)), coords=(1,3), name=incorrect_input_name)


def test_image():
    tile = Tile(np.ones((224,224,3)), coords=(1,3))
    assert (tile.image).all() = (np.ones((224,224,3))).all()


def test_repr(tile_withmasks):
    tile = tile_withmasks()
    print(tile)
