import pytest
import numpy as np
import string
import random

from pathml.core.tile import Tile

@pytest.fixture
def tile_nomasks(shape=(224, 224, 3), i=1, j=3):
    testtile = Tile(np.random.randn(*shape), coords = (i, j))
    return testtile


@pytest.fixture
def tile_withmasks(shape=(224, 224, 3), coords=(1, 3), stack=50, labeltype=str):
    if labeltype == str:
        letters = string.ascii_letters + string.digits
        maskdict = {}
        for i in range(stack):
            randomkey = 'test' + ''.join(random.choice(letters) for _ in range(i))
            maskdict[randomkey] = np.random.randint(2, size = shape)
        masks = Masks(maskdict)
    return Tile(np.random.random_sample(shape), coords = coords, masks = masks)


@pytest.mark.parametrize("incorrect_input", ["string", True, 5, [5, 4, 3], {"dict": "testing"}])
def test_init_incorrect_input(incorrect_input):
    with pytest.raises(ValueError):
        testimage = Tile(incorrect_input, coords=(1,3))
        testcoords = Tile(np.random.randn((224,224,3)), coords=incorrect_input)
        testmasks = Tile(np.random.randn((224,224,3)), coords=(1,3), masks=incorrect_input)


@pytest.mark.parametrize("incorrect_input",  [True, 5, [5, 4, 3], {"dict": "testing"}])
def test_init_name_incorrect_input(incorrect_input):
    with pytest.raises(ValueError):
        testname = Tile(np.random.randn((224,224,3)), coords=(1,3), name=incorrect_input)

def test_image():
    tile = Tile(np.ones((224,224,3)), coords=(1,3))
    assert (tile.image).all() = (np.ones((224,224,3))).all()

def test_repr(tile_withmasks):
    tile = tile_withmasks()
    print(tile)
