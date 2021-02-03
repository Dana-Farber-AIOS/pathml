import pytest
import numpy as np
import string
import random

from pathml.core.tiles import Tiles
from pathml.core.tile import Tile
from pathml.core.masks import Masks


@pytest.fixture
def emptytiles():
    return Tiles()


@pytest.fixture
def tile_nomasks():
    testtile = Tile(np.random.randn((224, 224, 3)), name='test', coords = (1, 3), slidetype=None)
    return testtile


@pytest.fixture
def tile_withmasks():
    letters = string.ascii_letters + string.digits
    maskdict = {}
    for i in range(50):
        randomkey = 'test' + ''.join(random.choice(letters) for _ in range(i))
        maskdict[randomkey] = np.random.randint(2, size = (224,224,3))
    masks = Masks(maskdict)
    return Tile(np.random.random_sample((224,224,3)), name='test', coords = (1,3), masks = masks, slidetype=None)


@pytest.mark.parametrize("incorrect_input", ["string", True, 5, [5, 4, 3], {"dict": "testing"}])
def test_init_incorrect_input(incorrect_input):
    with pytest.raises(ValueError):
        tiles = Tiles(incorrect_input)


def test_init(tile_withmasks):
    tilelist = [tile_withmasks(coords = (k, k)) for k in range(20)]
    tiledict = {(k, k): tile_withmasks(coords = (k, k)) for k in range(20)}
    tiles = Tiles(tilelist)
    tiles2 = Tiles(tiledict)
    assert tiles[str((0, 0))] == tilelist[0]
    assert tiles2[str((0, 0))] == tiledict[str((0, 0))]


@pytest.mark.parametrize("incorrect_input", ["string", None, True, 5, [5, 4, 3], {"dict": "testing"}])
def test_add_incorrect_input(incorrect_input, emptytiles, tile_nomasks):
    tiles = emptytiles
    tile = tile_nomasks
    with pytest.raises(ValueError):
        tiles.add(incorrect_input, tile)
        tiles.add((1, 3), incorrect_input)


def test_add_get_nomasks(emptytiles, tile_nomasks):
    tiles = emptytiles
    tile = tile_nomasks
    tiles.add((1, 3), tile)
    assert (tiles[(1, 3)].image == tile.image).all()
    assert tiles[(1, 3)].name = tile.name
    assert tiles[(1, 3)].coords = tile.coords
    assert tiles[(1, 3)].labels = tile.labels
    assert tiles[(1, 3)].slidetype = tile.slidetype
    assert (tiles[0].image == tile.image).all()


def test_add_get_withmasks(emptytiles, tile_withmasks):
    tiles = emptytiles
    test = tile_withmasks
    tile = test 
    tiles.add((1, 3), tile)
    for mask in test.masks:
        assert (tiles[(1, 3)].masks[mask] == test.masks[mask]).all()
        assert (tiles[0].masks[mask] == test.masks[mask]).all()


@pytest.mark.parametrize("incorrect_input", ["string", None, True, 5, [5, 4, 3], {"dict": "testing"}])
def test_remove_incorrect_input(incorrect_input, emptytiles, tile_nomasks):
    tiles = emptytiles
    tile = tile_nomasks
    tiles.add((1, 3), tile)
    with pytest.raises(KeyError):
        tiles.remove(incorrect_input)


def test_remove_nomasks(emptytiles, tile_nomasks):
    tiles = emptytiles
    tile = tile_nomasks
    tiles.add((1, 3), tile)
    tiles.remove((1, 3))
    with pytest.raises(Exception):
        triggerexception = tiles['(1, 3)']

def test_slice_nomasks(emptytiles, tile_nomasks):
    # slice one tile
    tiles = emptytiles
    tile = tile_nomasks
    tiles.add((1,3), tile)
    slices = [slice(2,5)]
    for s in tiles.slice(slices):
        print(s)

def test_slice_withmasks(emptytiles, tile_withmasks):
    tiles = emptytiles
    tile = tile_withmasks
    tiles.add((1,3), tile)
    slices = [slice(2,5)]
    for s in tiles.slice(slices):
        print(s)
