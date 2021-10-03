"""
Copyright 2021, Dana-Farber Cancer Institute and Weill Cornell Medicine
License: GNU GPL 2.0
"""

import copy
from pathlib import Path

import numpy as np
import pytest
from memory_profiler import memory_usage
from pathml.core import HESlide
from pathml.core.h5managers import h5pathManager
from pathml.core.tiles import Tiles
from pathml.preprocessing.pipeline import Pipeline


def test_h5manager(example_slide_data):
    """
    See issue #181.
    """
    pipe = Pipeline([])
    example_slide_data.run(pipe, distributed=False, tile_size=200)
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


# this test takes a long time, so skip by running 'python -m pytest -m "not slow"'
@pytest.mark.slow
def test_h5_dataset_memory(tmp_path, helpers):
    # this test fails if h5.Group.create_dataset allocates
    # more memory than available on the test machine when declaring
    # the data array for very large slides
    # see https://github.com/Dana-Farber-AIOS/pathml/pull/200
    def memory_wrapper(path, helper):
        wsi_url = (
            "https://data.kitware.com/api/v1/file/5899dd6d8d777f07219fcb23/download"
        )
        helper.download_from_url(wsi_url, tmp_path, name="tempslide.svs")
        slidepath = str(Path(tmp_path) / Path("tempslide.svs"))
        slide = HESlide(slidepath)
        tile = next(slide.slide.generate_tiles(shape=(10, 10)))
        slide.h5manager.add_tile(tile)

    memory = memory_usage((memory_wrapper, {"path": tmp_path}, {"helper": helpers}))
    # use no more than 1GB memory
    assert max(memory) / 1000 < 1
