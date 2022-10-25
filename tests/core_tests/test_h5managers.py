"""
Copyright 2021, Dana-Farber Cancer Institute and Weill Cornell Medicine
License: GNU GPL 2.0
"""

import copy

import numpy as np

from pathml.core import HESlide
from pathml.preprocessing.pipeline import Pipeline


def test_h5manager(example_slide_data):
    """
    See issue #181.
    """
    pipe = Pipeline([])
    example_slide_data.run(pipe, distributed=False)
    for tile in example_slide_data.tiles:
        assert np.count_nonzero(tile.image) > 0


def test_h5manager2(tileHE):
    slidedata1 = HESlide("tests/testdata/small_HE.svs")
    slidedata2 = HESlide("tests/testdata/small_HE.svs")
    tiles1 = slidedata1.tiles
    tiles2 = slidedata2.tiles
    coordslist = [(0, 0), (0, 500), (0, 0)]
    for coord in coordslist[0:2]:
        tile = copy.deepcopy(tileHE)
        tile.coords = coord
        tiles1.add(tile)

    for coord in coordslist:
        tile = copy.deepcopy(tileHE)
        tile.coords = coord
        tiles2.add(tile)

    for tile1, tile2 in zip(tiles1, tiles2):
        np.testing.assert_array_equal(tile1.image, tile2.image)


def test_tile_dtype_HE(tileHE):
    """Test that tiles have default float16 dtype and masks are bool"""
    slidedata = HESlide("tests/testdata/small_HE.svs")
    slidedata.tiles.add(tileHE)
    tile_retrieved = slidedata.tiles[tileHE.coords]
    assert tile_retrieved.image.dtype == np.float16
    assert tile_retrieved.masks["testmask"].dtype == bool


def test_tile_dtype_HE_uint8(tileHE):
    """Test that tiles have modified uint8 dtype and masks are bool"""
    slidedata = HESlide("tests/testdata/small_HE.svs", dtype=np.dtype("uint8"))
    slidedata.tiles.add(tileHE)
    tile_retrieved = slidedata.tiles[tileHE.coords]
    assert tile_retrieved.image.dtype == np.uint8
    assert tile_retrieved.masks["testmask"].dtype == bool


def test_tile_dtype_IF(tileVectra, vectra_slide):
    """Test that tiles have float16 dtype and masks have bool dtype."""
    vectra_slide.tiles.add(tileVectra)
    tile_retrieved = vectra_slide.tiles[tileVectra.coords]
    assert tile_retrieved.image.dtype == np.float16
    assert tile_retrieved.masks["testmask"].dtype == bool
