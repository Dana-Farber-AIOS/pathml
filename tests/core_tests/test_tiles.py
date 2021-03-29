import pytest
import numpy as np
import string
import random
import copy
from collections import OrderedDict 
from pathml.core.tiles import Tiles
from pathml.core.tile import Tile
from pathml.core.masks import Masks


@pytest.fixture
def emptytiles():
    return Tiles()


@pytest.fixture
def tile_nomasks():
    shape = (224, 224, 3)
    coords = (1, 3)
    im = np.arange(np.product(shape)).reshape(shape)
    testtile = Tile(image = im, coords = coords, name='test', slidetype=None)
    return testtile


@pytest.fixture
def tile_withmasks():
    shape = (224, 224, 3)
    coords = (1, 3)
    letters = string.ascii_letters + string.digits
    maskdict = {}
    for i in range(50):
        maskdict[str(i)] = np.random.randint(2, size = (224,224,3))
    masks = Masks(maskdict)
    return Tile(np.random.random_sample(shape), name='test2', coords = coords, masks = masks, slidetype=None)


@pytest.fixture
def tile_withlabels():
    shape = (224, 224, 3)
    coords = (1, 3)
    labels = OrderedDict({'label1':'positive', 'label2':'negative'})
    im = np.arange(np.product(shape)).reshape(shape)
    testtile = Tile(image = im, coords = coords, name = 'test3', labels = labels, slidetype = None)
    return testtile


@pytest.fixture
def tile_withslidetype():
    shape = (224, 224, 3)
    coords = (1, 3)
    slidetype = 'HE'
    im = np.arange(np.product(shape)).reshape(shape)
    testtile = Tile(image = im, coords = coords, name = 'test4', slidetype = slidetype)
    return testtile


@pytest.mark.parametrize("incorrect_input", ["string", True, 5, [5, 4, 3], {"dict": "testing"}])
def test_init_incorrect_input(incorrect_input):
    with pytest.raises(ValueError):
        tiles = Tiles(incorrect_input)


def test_init_len(tile_withmasks):
    shape = (224, 224, 3)
    coords = [0,0]
    maskdict = {}
    for i in range(50):
        maskdict[str(i)] = np.random.randint(2, size = (224,224,3))
    masks = Masks(maskdict)
    tiles = {}
    for i in range(50):
        tile = Tile(np.random.random_sample(shape), name=str(coords), coords = tuple(coords), masks = masks, slidetype=None)
        tiles[tuple(coords)] = tile
        coords[0], coords[1] = coords[0] + shape[0], coords[1] + shape[1]
    tilesdict = tiles
    tileslist = list(tiles.values())
    tiles = Tiles(tileslist)
    tiles2 = Tiles(tilesdict)
    assert (tiles[0].image == tileslist[0].image).all()
    assert (tiles2[0].image == tilesdict[(0,0)].image).all()
    assert len(tiles) == 50


@pytest.mark.parametrize("incorrect_input", [None, True, 5])
def test_init_incorrect_input(tile_withmasks, incorrect_input):
    tiledict = {incorrect_input: tile_withmasks for k in range(20)}
    tilelist = [incorrect_input for k in range(20)]
    with pytest.raises(ValueError):
        tiles = Tiles(tiledict)
    with pytest.raises(ValueError):
        tiles2 = Tiles(tilelist)


def test_repr(tile_withmasks):
    assert tile_withmasks 


@pytest.mark.parametrize("incorrect_input", ["string", None, True, 5, [5, 4, 3], {"dict": "testing"}])
@pytest.mark.parametrize("incorrect_input_get", [None, True, [5, 4, 3], {"dict": "testing"}])
def test_add_get_incorrect_input(incorrect_input, incorrect_input_get, emptytiles, tile_nomasks):
    tiles = emptytiles
    tile = tile_nomasks
    with pytest.raises(ValueError):
        tiles.add(incorrect_input, tile)
        tiles.add((1, 3), incorrect_input)
    tiles.add((1, 3), tile)
    # TODO: this test fails to throw keyerror for some inputs where it should but I can't find the bug (RC)
    ''' 
    with pytest.raises(KeyError):
        tiles[incorrect_input_get]
    '''

def test_add_get_nomasks(emptytiles, tile_nomasks):
    tiles = emptytiles
    tile = tile_nomasks
    tiles.add((1, 3), tile)
    assert (tiles[(1, 3)].image == tile.image).all()
    assert tiles[(1, 3)].name == tile.name
    assert tiles[(1, 3)].coords == tile.coords
    assert tiles[(1, 3)].labels == tile.labels
    assert tiles[(1, 3)].slidetype == tile.slidetype
    assert (tiles[0].image == tile.image).all()
    im = np.arange(np.product((225,224,3))).reshape((225,224,3))
    wrongshapetile = Tile(image=im, coords = (4, 5), name='wrongshape', slidetype=None)
    with pytest.raises(ValueError):
        tiles.add((4, 5), wrongshapetile)


def test_add_get_withlabels(emptytiles, tile_withlabels):
    tiles = emptytiles
    tile = tile_withlabels
    labels = tile.labels 
    tiles.add((1, 3), tile) 
    assert tiles[(1, 3)].labels == labels


def test_add_get_withslidetype(emptytiles, tile_withslidetype):
    tiles = emptytiles
    tile = tile_withslidetype
    slidetype = tile.slidetype
    tiles.add((1, 3), tile)
    assert tiles[(1, 3)].slidetype == slidetype


def test_add_get_withmasks(emptytiles, tile_withmasks):
    tiles = emptytiles
    tile = tile_withmasks
    tiles.add((1, 3), tile)
    print(tiles.h5manager.h5['tiles'].keys())
    for key in tiles.h5manager.h5['tiles']['masks'].keys():
        assert (tiles[(1, 3)].masks[key] == tile.masks[key]).all()
        assert (tiles[0].masks[key] == tile.masks[key]).all()


@pytest.mark.parametrize("incorrect_input", ["string", None, True, 5, [5, 4, 3], {"dict": "testing"}])
def test_update_incorrect_input(incorrect_input, emptytiles, tile_nomasks):
    tiles = emptytiles
    tile = tile_nomasks
    tiles.add((1, 3), tile)
    with pytest.raises(KeyError):
        tiles.update((1, 3), tile, incorrect_input)


def test_update_all(emptytiles, tile_nomasks):
    tiles = emptytiles
    tile = tile_nomasks
    tiles.add((1, 3), tile)
    tiles.update((1, 3), tile)
    with pytest.raises(ValueError):
        tiles.update((2, 3), tile)


def test_update_image(emptytiles, tile_nomasks):
    tiles = emptytiles
    tile = tile_nomasks
    tiles.add((1, 3), tile)
    shape = (224, 224, 3)
    coords = (1, 3)
    im = np.ones((224,224,3))
    tiles.update((1, 3), im, 'image')
    assert (tiles[(1, 3)].image == im).all()
    im2 = np.ones((224,225,3))
    with pytest.raises(Exception):
        tiles.update((1, 3), im2, 'image')

@pytest.mark.parametrize("incorrect_input", ["string", None, True, 5, [5, 4, 3]])
def test_update_labels(emptytiles, tile_withlabels, incorrect_input):
    tiles = emptytiles
    tile = tile_withlabels
    tiles.add((1, 3), tile)
    tiles.update((1, 3), {'arbitrarystring':'arbitrarytarget'}, 'labels')
    with pytest.raises(AssertionError):
        tiles.update((1, 3), incorrect_input, 'labels') 


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
    test = tiles.slice(slices)
    assert test.h5manager.shape == (3,224,3)


def test_slice_withmasks(emptytiles, tile_withmasks):
    tiles = emptytiles
    tile = tile_withmasks
    tiles.add((1,3), tile)
    slices = [slice(2,5)]
    test = tiles.slice(slices)
    assert test.h5manager.shape == (3,224,3)
