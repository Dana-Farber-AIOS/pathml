"""
Copyright 2021, Dana-Farber Cancer Institute and Weill Cornell Medicine
License: GNU GPL 2.0
"""

from pathlib import Path
import pytest
from dask.distributed import Client
import numpy as np
import h5py

import pathml
from pathml.core import (
    SlideData,
    HESlide,
    MultiparametricSlide,
    OpenSlideBackend,
    BioFormatsBackend,
    Tile,
)
from pathml.core.slide_data import get_file_ext
from pathml.preprocessing import Pipeline, BoxBlur


@pytest.mark.parametrize("slide", [SlideData, HESlide, MultiparametricSlide])
def test_repr(slide):
    s = slide("tests/testdata/small_HE.svs")
    repr(s)


@pytest.mark.parametrize(
    "path,ext",
    [
        ("/test/testing/test.txt", ".txt"),
        ("/test/testing/test.txt.gz", ".txt"),
        ("/test/testing/test.txt.bz2", ".txt"),
        ("/test/testing/test.qptiff", ".qptiff"),
        ("/test/testing/test.ext1.ext2", ".ext1.ext2"),
    ],
)
def test_get_file_ext(path, ext):
    result = get_file_ext(path)
    assert result == ext


def test_write_with_array_labels(tmp_path, example_slide_data):
    example_slide_data.write(tmp_path / "test_array_in_labels.h5path")
    assert Path(tmp_path / "test_array_in_labels.h5path").is_file()


def test_run_pipeline(example_slide_data):
    pipeline = Pipeline([BoxBlur(kernel_size=15)])
    # start the dask client
    client = Client()
    # run the pipeline
    example_slide_data.run(pipeline=pipeline, client=client, tile_size=50)
    # close the dask client
    client.close()


@pytest.mark.parametrize("overwrite_tiles", [True, False])
def test_run_existing_tiles(slide_dataset_with_tiles, overwrite_tiles):
    dataset = slide_dataset_with_tiles
    pipeline = Pipeline([BoxBlur(kernel_size=15)])
    if overwrite_tiles:
        dataset.run(pipeline, overwrite_existing_tiles=overwrite_tiles)
    else:
        with pytest.raises(Exception):
            dataset.run(pipeline, overwrite_existing_tiles=overwrite_tiles)


@pytest.fixture
def he_slide():
    wsi = HESlide("tests/testdata/small_HE.svs", backend="openslide")
    return wsi


@pytest.fixture
def multiparametric_slide():
    wsi = MultiparametricSlide("tests/testdata/smalltif.tif", backend="bioformats")
    return wsi


def test_multiparametric(multiparametric_slide):
    assert isinstance(multiparametric_slide, SlideData)
    assert multiparametric_slide.slide_type == pathml.types.IF


@pytest.mark.parametrize("shape", [500, (500, 400)])
@pytest.mark.parametrize("stride", [None, 1000])
@pytest.mark.parametrize("pad", [True, False])
@pytest.mark.parametrize("level", [0])
def test_generate_tiles_he(he_slide, shape, stride, pad, level):
    for tile in he_slide.generate_tiles(
        shape=shape, stride=stride, pad=pad, level=level
    ):
        assert isinstance(tile, Tile)


@pytest.mark.parametrize("shape", [100, (50, 100)])
@pytest.mark.parametrize("stride", [None, 100])
@pytest.mark.parametrize("pad", [True, False])
def test_generate_tiles_multiparametric(multiparametric_slide, shape, stride, pad):
    for tile in multiparametric_slide.generate_tiles(
        shape=shape, stride=stride, pad=pad
    ):
        assert isinstance(tile, Tile)


@pytest.mark.parametrize("pad", [True, False])
def test_generate_tiles_padding(he_slide, pad):
    shape = 300
    stride = 300
    tiles = list(he_slide.generate_tiles(shape=shape, stride=stride, pad=pad))
    # he_slide.slide.get_image_shape() --> (2967, 2220)
    # if no padding, expect: 9*7 = 63 tiles
    # if padding, expect: 10*8 - 80 tiles
    if not pad:
        assert len(tiles) == 63
    else:
        assert len(tiles) == 80


def test_read_write_heslide(tmp_path, example_slide_data_with_tiles):
    slidedata = example_slide_data_with_tiles
    path = tmp_path / "testhe.h5"
    slidedata.write(path)
    readslidedata = SlideData(path)
    assert readslidedata.name == slidedata.name
    np.testing.assert_equal(readslidedata.labels, slidedata.labels)
    if slidedata.masks is None:
        assert readslidedata.masks is None
    if slidedata.tiles is None:
        assert readslidedata.tiles is None
    assert scan_hdf5(readslidedata.h5manager.h5) == scan_hdf5(slidedata.h5manager.h5)
    if readslidedata.counts.obs.empty:
        assert slidedata.counts.obs.empty
    else:
        np.testing.assert_equal(readslidedata.counts.obs, slidedata.counts.obs)
    if readslidedata.counts.var.empty:
        assert slidedata.counts.var.empty
    else:
        np.testing.assert_equal(readslidedata.counts.var, slidedata.counts.var)


def scan_hdf5(f, recursive=True, tab_step=2):
    def scan_node(g, tabs=0):
        elems = []
        for k, v in g.items():
            if isinstance(v, h5py.Dataset):
                elems.append(v.name)
            elif isinstance(v, h5py.Group) and recursive:
                elems.append((v.name, scan_node(v, tabs=tabs + tab_step)))
        return elems

    return scan_node(f)


def compare_dict_ignore_order(d1, d2):
    """
    Compare two dictionaries, ignoring order of values
    """
    vals_a = list(d1.values()).sort()
    vals_b = list(d2.values()).sort()
    if vals_a != vals_b:
        return False
    for k1, k2 in zip(vals_a, vals_b):
        if d1[k1] != d2[k2]:
            return False
    return True
