"""
Copyright 2021, Dana-Farber Cancer Institute and Weill Cornell Medicine
License: GNU GPL 2.0
"""

import numpy as np
import pytest

from pathml.core import SlideData
from pathml.preprocessing import Pipeline
from pathml.preprocessing.transforms import Transform
from pathml.ml import TileDataset


class TestingTransform(Transform):
    """Transform for testing dataset"""

    def apply(self, tile):
        tile.labels = {"testing_coords_label": tile.coords}
        tile.masks["test"] = np.ones(tile.image.shape[0:2]) * 5


@pytest.mark.parametrize(
    "im_path",
    [
        "tests/testdata/small_HE.svs",
        "tests/testdata/small_vectra.qptiff",
        "tests/testdata/small_dicom.dcm",
    ],
)
def test_dataset(tmp_path, im_path):
    # first create and run pipeline, and save h5path file
    labs = {
        "test_string_label": "testlabel",
        "test_array_label": np.array([2, 3, 4]),
        "test_int_label": 3,
        "test_float_label": 3.0,
        "test_bool_label": True,
    }
    wsi = SlideData(im_path, labels=labs)
    pipeline = Pipeline([TestingTransform()])
    wsi.run(pipeline, distributed=False, tile_size=500)
    save_path = str(tmp_path) + str(np.round(np.random.rand(), 8)) + "slide.h5"
    wsi.write(path=save_path)
    # load dataset from h5path, and compare to what we expect
    dataset = TileDataset(save_path)
    assert len(dataset) == len(wsi.tiles)

    im, mask, lab_tile, lab_slide = dataset[0]

    for k, v in lab_slide.items():
        if isinstance(v, np.ndarray):
            assert np.array_equal(v, labs[k])
        else:
            assert v == labs[k]

    if wsi.name == "small_vectra":
        # 5-dim images (XYZCT converted to TCZXY for batching)
        assert np.array_equal(im, wsi.tiles[0].image.transpose(4, 3, 2, 1, 0))
    else:
        assert np.array_equal(im, wsi.tiles[0].image.transpose(2, 0, 1))

    assert list(lab_tile.keys()) == ["testing_coords_label"]
    assert np.array_equal(
        lab_tile["testing_coords_label"], np.array(wsi.tiles[0].coords)
    )
    assert np.array_equal(mask, np.ones((1, 500, 500)) * 5)
