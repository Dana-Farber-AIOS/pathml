"""
Copyright 2021, Dana-Farber Cancer Institute and Weill Cornell Medicine
License: GNU GPL 2.0
"""

import pytest
import numpy as np

import pathml.core.h5managers
from pathml.core import Tiles, Tile, Masks, OpenSlideBackend, types, HESlide


@pytest.fixture
def emptytiles():
    slidedata = HESlide("tests/testdata/small_HE.svs")
    return slidedata.tiles


@pytest.fixture
def tiles():
    """
    dict of adjacent tiles
    """
    tiles = []
    for i in range(2):
        for j in range(2):
            # create tile
            shape = (224, 224, 3)
            coords = (224 * i, 224 * j)
            name = f"{i}_{j}"
            masks = {str(k): np.random.randint(2, size=shape) for k in range(2)}
            labs = {
                "test_string_label": "testlabel",
                "test_array_label": np.array([2, 3, 4]),
                "test_int_label": 3,
                "test_float_label": 3.0,
            }
            image = np.random.random_sample(shape)
            tile = Tile(
                image=image,
                name=name,
                coords=coords,
                slide_type=types.HE,
                masks=masks,
                labels=labs,
            )
            # add to dict
            tiles.append(tile)
    return tiles


@pytest.fixture
def tilesnonconsecutive():
    """
    dict of nonconsecutive tiles
    """
    tiles = []
    for i in range(2):
        for j in range(2):
            # create tile
            shape = (224, 224, 3)
            coords = (224 * 2 * (i + 1), 224 * 2 * (j + 2))
            name = f"{i}_{j}"
            masks = {str(k): np.random.randint(2, size=shape) for k in range(2)}
            labs = {
                "test_string_label": "testlabel",
                "test_array_label": np.array([2, 3, 4]),
                "test_int_label": 3,
                "test_float_label": 3.0,
            }
            image = np.random.random_sample(shape)
            tile = Tile(
                image=image,
                name=name,
                coords=coords,
                slide_type=types.HE,
                masks=masks,
                labels=labs,
            )
            # add to dict
            tiles.append(tile)
    return tiles


@pytest.mark.parametrize(
    "incorrect_input", ["string", True, 5, [5, 4, 3], {"dict": "testing"}]
)
def test_init(tiles, tilesnonconsecutive, incorrect_input):
    # init from dict
    slidedata = HESlide("tests/testdata/small_HE.svs", tiles=tiles)
    assert (slidedata.tiles[0].image == tiles[0].image).all()
    # init len
    assert len(slidedata.tiles) == 4
    # incorrect input
    with pytest.raises(AssertionError):
        # fix obj
        slidedata = HESlide("tests/testdata/small_HE.svs", tiles=incorrect_input)
    # nonconsecutive tiles
    slidedata = HESlide("tests/testdata/small_HE.svs", tiles=tilesnonconsecutive)
    np.testing.assert_array_equal(
        slidedata.tiles[(896, 1344)].image, tilesnonconsecutive[3].image
    )


def test_repr(tiles):
    slidedata = HESlide("tests/testdata/small_HE.svs", tiles=tiles)
    assert repr(slidedata.tiles)


@pytest.mark.parametrize(
    "incorrect_input", ["string", None, True, 5, [5, 4, 3], {"dict": "testing"}]
)
@pytest.mark.parametrize(
    "incorrect_input2", [None, True, [5, 4, 3], {"dict": "testing"}]
)
def test_add_get(emptytiles, tileHE, incorrect_input, incorrect_input2):
    # add single tile
    tiles = emptytiles
    tiles.add(tileHE)
    # get by coords and by index
    assert (tiles[(1, 3)].image == tileHE.image).all()
    assert (tiles[0].image == tileHE.image).all()
    assert tiles[(1, 3)].name == tileHE.name
    assert tiles[(1, 3)].coords == tileHE.coords
    for label in tiles[(1, 3)].labels:
        if isinstance(tiles[(1, 3)].labels[label], np.ndarray):
            np.testing.assert_array_equal(
                tiles[(1, 3)].labels[label], tileHE.labels[label]
            )
        else:
            assert tiles[(1, 3)].labels[label] == tileHE.labels[label]
    assert tiles[(1, 3)].slide_type == tileHE.slide_type
    # get masks
    for mask in tiles.h5manager.h5["masks"].keys():
        # masks by coords and by index
        assert (tiles[(1, 3)].masks[mask] == tileHE.masks[mask]).all()
        assert (tiles[0].masks[mask] == tileHE.masks[mask]).all()
    # incorrect input
    with pytest.raises(ValueError or KeyError):
        tiles.add(incorrect_input)
    with pytest.raises(KeyError or IndexError):
        tiles[incorrect_input2]
    # wrong shape
    im = np.arange(np.product((225, 224, 3))).reshape((225, 224, 3))
    wrongshapetile = Tile(image=im, coords=(4, 5), name="wrong")
    with pytest.raises(ValueError):
        tiles.add(wrongshapetile)


@pytest.mark.parametrize(
    "incorrect_target", ["string", None, True, 5, [5, 4, 3], {"dict": "testing"}]
)
@pytest.mark.parametrize("incorrect_labels", ["string", None, True, 5, [5, 4, 3]])
def test_update(emptytiles, tileHE, incorrect_target, incorrect_labels):
    # add single tile
    tiles = emptytiles
    tiles.add(tileHE)
    # update all
    shape = tileHE.image.shape
    coords = (1, 3)
    slide_type = types.HE
    masks = {str(i): np.ones(shape[0:2]) for i in range(2)}
    labs = {
        "test_string_label": "testlabel",
        "test_array_label": np.array([2, 3, 4]),
        "test_int_label": 3,
        "test_float_label": 3.0,
    }
    img = np.ones(shape)
    newtile = Tile(
        image=img,
        name="new",
        coords=coords,
        slide_type=types.HE,
        masks=masks,
        labels=labs,
    )
    tiles.update((1, 3), newtile, "all")
    assert (tiles[(1, 3)].image == np.ones(shape)).all()
    assert tiles[(1, 3)].name == "new"
    # labels may be numpy arrays, must check each
    for label in tiles[(1, 3)].labels:
        if isinstance(tiles[(1, 3)].labels[label], np.ndarray):
            np.testing.assert_array_equal(tiles[(1, 3)].labels[label], labs[label])
        else:
            assert tiles[(1, 3)].labels[label] == labs[label]
    assert tiles[(1, 3)].slide_type == slide_type
    # update image
    tiles = emptytiles
    tiles.add(tileHE)
    tiles.update((1, 3), img, "image")
    assert (tiles[(1, 3)].image == np.ones(shape)).all()
    # incorrect image
    tiles = emptytiles
    tiles.add(tileHE)
    im2 = np.ones((224, 225, 3))
    with pytest.raises(Exception):
        tiles.update((1, 3), im2, "image")
    # update labels
    tiles = emptytiles
    tiles.add(tileHE)
    newlabels = {"newkey1": "newvalue1", "newkey2": "newalue2"}
    tiles.update((1, 3), newlabels, "labels")
    assert set(newlabels.keys()).issubset(set(tiles[(1, 3)].labels.keys()))
    # incorrect labels
    with pytest.raises(AssertionError):
        tiles.update((1, 3), incorrect_labels, "labels")
    # incorrect target
    with pytest.raises(KeyError):
        tiles.update(tileHE.coords, tileHE, incorrect_target)


@pytest.mark.parametrize(
    "incorrect_input", ["string", None, True, 5, [5, 4, 3], {"dict": "testing"}]
)
def test_remove(emptytiles, tileHE, incorrect_input):
    tiles = emptytiles
    tiles.add(tileHE)
    tiles.remove((1, 3))
    with pytest.raises(Exception):
        triggerexception = tiles[(1, 3)]
    with pytest.raises(KeyError):
        tiles.remove((1, 3))
    # incorrect input
    with pytest.raises(KeyError):
        tiles.remove(incorrect_input)


@pytest.mark.parametrize(
    "incorrect_input", ["string", None, True, 5, {"dict": "testing"}]
)
def test_slice(emptytiles, tileHE, incorrect_input):
    tiles = emptytiles
    tiles.add(tileHE)
    slices = [slice(2, 5)]
    test = tiles.slice(slices)
    assert test[0].image.shape == tileHE.image[slices[0], ...].shape
    assert (
        next(iter(test[0].masks.items()))[1].shape
        == tileHE.image[slices[0], ...].shape[0:2]
    )
    with pytest.raises(KeyError):
        test = tiles.slice(incorrect_input)


def test_reshape(tiles, monkeypatch):
    slidedata = HESlide("tests/testdata/small_HE.svs", tiles=tiles)
    slidedata.tiles.reshape(shape=(112, 112))
    assert slidedata.tiles[0].image.shape == (112, 112, 3)
    # monkeypatch input to overwrite labels
    monkeypatch.setattr("builtins.input", lambda _: "y")
    slidedata.tiles.reshape(shape=(225, 225))
    assert len(slidedata.tiles) == 1
    assert slidedata.tiles[0].image.shape == (225, 225, 3)
    # centercrop
    slidedata.tiles.reshape(shape=(446, 446, 3), centercrop=True)
    assert slidedata.tiles[0].coords[0] == 1
