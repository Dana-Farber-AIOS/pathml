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
def tile():
    shape = (224, 224, 3)
    coords = (1, 3)
    slidetype = "<class 'pathml.core.slide_backend.OpenSlideBackend'>" 
    maskdict = {str(i) : np.random.randint(2, size = shape) for i in range(20)}
    masks = Masks(maskdict)
    labels = {'label1' : 'positive', 'label2' : 'negative'}
    image = np.random.random_sample(shape)
    return Tile(image = image, name = 'test', coords = coords, slidetype = slidetype, masks = masks, labels = labels)


@pytest.fixture
def tiles():
    """ 
    dict of adjacent tiles 
    """
    tiles = {}
    for i in range(2):
        for j in range(2):
            # create tile
            shape = (224, 224, 3)
            coords = (224*i, 224*j)
            slidetype = "<class 'pathml.core.slide_backend.OpenSlideBackend'>" 
            name = f"{i}_{j}" 
            maskdict = {str(k) : np.random.randint(2, size = shape) for k in range(20)}
            masks = Masks(maskdict)
            labels = {'label1' : 'tumor', 'label2' : 'stroma', 'label3' : 'cribriform'}
            image = np.random.random_sample(shape)
            tile = Tile(image = image, name = name, coords = coords, slidetype = slidetype, masks = masks, labels = labels)
            # add to dict
            tiles[coords] = tile
    return tiles 


@pytest.fixture
def tilesnonconsecutive():
    """
    dict of nonconsecutive tiles 
    """
    tiles = {}
    for i in range(2):
        for j in range(2):
            # create tile
            shape = (224, 224, 3)
            coords = (224*2*(i+1), 224*2*(j+2))
            slidetype = "<class 'pathml.core.slide_backend.OpenSlideBackend'>" 
            name = f"{i}_{j}" 
            maskdict = {str(k) : np.random.randint(2, size = shape) for k in range(20)}
            masks = Masks(maskdict)
            labels = {'label1' : 'tumor', 'label2' : 'stroma', 'label3' : 'cribriform'}
            image = np.random.random_sample(shape)
            tile = Tile(image = image, name = name, coords = coords, slidetype = slidetype, masks = masks, labels = labels)
            # add to dict
            tiles[coords] = tile
    return tiles 


@pytest.mark.parametrize("incorrect_input", ["string", True, 5, [5, 4, 3], {"dict": "testing"}])
def test_init(tiles, tilesnonconsecutive, incorrect_input):
    # init from dict
    tilesdict = tiles
    tiles1 = Tiles(tilesdict)
    assert (tiles1[0].image == tilesdict[(0,0)].image).all()
    # init from list
    tileslist = list(tilesdict.values())
    tiles2 = Tiles(tileslist)
    assert (tiles2[0].image == tileslist[0].image).all()
    # init len
    assert len(tiles1) == 4
    assert len(tiles2) == 4
    # incorrect input
    with pytest.raises(ValueError or KeyError):
        tiles = Tiles(incorrect_input)
    # nonconsecutive tiles
    tilesdict2 = tilesnonconsecutive
    tiles3 = Tiles(tilesdict2)
    assert (tiles3[0].image == tilesdict2[(224*2*1, 224*2*2)].image).all()


def test_repr(tiles):
    assert Tiles(tiles)


@pytest.mark.parametrize("incorrect_input", ["string", None, True, 5, [5, 4, 3], {"dict": "testing"}])
@pytest.mark.parametrize("incorrect_input2", [None, True, [5, 4, 3], {"dict": "testing"}])
def test_add_get(emptytiles, tile, incorrect_input, incorrect_input2):
    # add single tile
    tiles = emptytiles
    tile = tile
    tiles.add(tile)
    # get by coords and by index
    assert (tiles[(1, 3)].image == tile.image).all()
    assert (tiles[0].image == tile.image).all()
    assert tiles[(1, 3)].name == tile.name
    assert tiles[(1, 3)].coords == tile.coords
    assert tiles[(1, 3)].labels == tile.labels
    assert tiles[(1, 3)].slidetype == tile.slidetype
    # get masks
    for mask in tiles.h5manager.h5['tiles']['masks'].keys():
        # masks by coords and by index
        assert (tiles[(1, 3)].masks[mask] == tile.masks[mask]).all()
        assert (tiles[0].masks[mask] == tile.masks[mask]).all()
    # incorrect input
    with pytest.raises(ValueError or KeyError):
        tiles.add(incorrect_input)
    with pytest.raises(KeyError):
        tiles[incorrect_input2]
    # wrong shape
    im = np.arange(np.product((225,224,3))).reshape((225,224,3))
    wrongshapetile = Tile(image = im, coords = (4, 5), name = 'wrong')
    with pytest.raises(ValueError):
        tiles.add(wrongshapetile)


@pytest.mark.parametrize("incorrect_target", ["string", None, True, 5, [5, 4, 3], {"dict": "testing"}])
@pytest.mark.parametrize("incorrect_labels", ["string", None, True, 5, [5, 4, 3]])
def test_update(emptytiles, tile, incorrect_target, incorrect_labels):
    tiles = emptytiles
    tile = tile
    tiles.add(tile)
    # update all
    shape = (224, 224, 3)
    coords = (1, 3)
    slidetype = "<class 'pathml.core.slide_backend.OpenSlideBackend'>" 
    maskdict = {str(i) : np.ones(shape) for i in range(20)}
    masks = Masks(maskdict)
    labels = {'label1' : 'new1', 'label2' : 'new2'}
    img = np.ones(shape)
    newtile = Tile(image = img, name = 'new', coords = coords, slidetype = slidetype, masks = masks, labels = labels)
    tiles.update((1, 3), newtile, 'all') 
    assert (tiles[(1, 3)].image == np.ones(shape)).all()
    assert tiles[(1, 3)].name == 'new'
    assert tiles[(1, 3)].labels == labels 
    assert tiles[(1, 3)].slidetype == slidetype 
    # update image
    tiles = emptytiles
    tile = tile
    tiles.add(tile)
    tiles.update((1, 3), img, 'image')
    assert (tiles[(1, 3)].image == np.ones(shape)).all()
    # incorrect image 
    tiles = emptytiles
    tile = tile
    tiles.add(tile)
    im2 = np.ones((224,225,3))
    with pytest.raises(Exception):
        tiles.update((1, 3), im2, 'image')
    # update labels
    tiles = emptytiles
    tile = tile
    tiles.add(tile)
    newlabels = {'newkey1' : 'newvalue1', 'newkey2' : 'newalue2'}
    tiles.update((1, 3), newlabels, 'labels')
    assert tiles[(1, 3)].labels == newlabels 
    # incorrect labels
    with pytest.raises(AssertionError):
        tiles.update((1, 3), incorrect_labels, 'labels') 
    # incorrect target
    with pytest.raises(KeyError):
        tiles.update(tile.coords, tile, incorrect_target)


@pytest.mark.parametrize("incorrect_input", ["string", None, True, 5, [5, 4, 3], {"dict": "testing"}])
def test_remove(emptytiles, tile, incorrect_input):
    tiles = emptytiles
    tile = tile
    tiles.add(tile)
    tiles.remove((1, 3))
    with pytest.raises(Exception):
        triggerexception = tiles[(1, 3)]
    with pytest.raises(KeyError):
        tiles.remove((1, 3))
    # incorrect input 
    with pytest.raises(KeyError):
        tiles.remove(incorrect_input)


@pytest.mark.parametrize("incorrect_input", ["string", None, True, 5, {"dict": "testing"}])
def test_slice(emptytiles, tile, incorrect_input):
    tiles = emptytiles
    tile = tile 
    tiles.add(tile)
    slices = [slice(2,5)]
    test = tiles.slice(slices)
    assert test.h5manager.shape == (3, 224, 3)
    assert test[0].image.shape == (3, 224, 3)
    assert test[0].masks[0].shape == (3, 224, 3) 
    with pytest.raises(KeyError):
        test = tiles.slice(incorrect_input)

def test_reshape(tiles):
    tilesdict = tiles
    tiles1 = Tiles(tilesdict)
    tiles1.reshape(shape=(112, 112))
    assert tiles1[0].image.shape == (112, 112, 3)
    tiles1.reshape(shape=(225, 225))
    assert len(tiles1) == 1 
    assert tiles1[0].image.shape == (225, 225, 3)
    # centercrop
    tiles1.reshape(shape = (446, 446, 3), centercrop = True)
    assert tiles1[0].coords[0] == 1
