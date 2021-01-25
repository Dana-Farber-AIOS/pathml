import pytest
import numpy as np
import string
import random

from pathml.core.tiles import Tile, Tiles
from pathml.core.masks import Masks

@pytest.fixture
def emptytiles():
    return Tiles()

@pytest.fixture
def tile_nomasks(shape = (224,224,3), i=1, j=3)
    testtile = Tile(np.random.randn(shape), i=i, j=j)
    return testtile

@pytest.fixture
def tile_withmasks(shape = (224,224,3), i=1, j=3, stack = 50, labeltype = str)
    if labeltype == str:
        letters = string.ascii_letters + string.digits + string.punctuation
        maskdict = []
        for i in range(stack):
            randomkey = ''.join(random.choice(letters) for j in range(i)) 
            maskdict[randomkey] = np.randint(2, size=shape) 
        masks = Masks(maskdict)
    testtile = Tile(np.random.randn(shape), i=i, j=j, masks=masks)

@pytest.mark.parametrize()
def test_init_incorrect_input():
    raise NotImplementedError

def test_init():
    raise NotImplementedError

@pytest.mark.parametrize("incorrect_input", ["string", None, True, 5, [5,4,3], {"dict":"testing"}])
def test_add_incorrect_input(incorrect_input, emptytiles, tile):
    tiles = emptytiles()
    tile = tile()
    tiles.add((1,3), incorrect_input)
    with pytest.raises(ValueError):
        tiles.add(incorrect_input, tile)

def test_add_nomasks(emptytiles, tile_nomasks):
    tiles = emptytiles()
    tile = tile_nomasks()
    tiles.add((1,3), tile)
    assert tiles['(1, 3)'] == tile

def test_add_withmasks(emptytiles, tile_withmasks)
    tiles = emptytiles()
    tile = tile_withmasks()
    testmask = tile.masks[0]
    tiles.add((1,3), tile)
    # TODO: for this to pass get must return a Tile object
    assert tiles['(1, 3)'].masks[0] == testmask

@pytest.mark.parametrize("incorrect_input", ["string", None, True, 5, [5,4,3], {"dict":"testing"}])
def test_remove_incorrect_input(incorrect_input, emptytiles, tile_nomasks):
    tiles = emptytiles()
    tile = tile_nomasks()
    tiles.add((1,3), tile) 
    with pytest.raises(KeyError):
        tiles.remove(incorrect_input)

def test_remove_nomasks(emptytiles, tile_nomasks):
    tiles = emptytiles()
    tile = tile_nomasks()
    tiles.add((1,3,tile))
    tiles.remove('(1, 3)')
    with pytest.raises(Exception):
        triggerexception = tiles['(1, 3)']
