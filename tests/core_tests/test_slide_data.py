"""
Copyright 2021, Dana-Farber Cancer Institute and Weill Cornell Medicine
License: GNU GPL 2.0
"""

import sys
from pathlib import Path

import h5py
import numpy as np
import pytest
from dask.distributed import Client

import pathml
from pathml.core import (
    CODEXSlide,
    HESlide,
    IHCSlide,
    MultiparametricSlide,
    SlideData,
    Tile,
)
from pathml.core.slide_data import infer_backend
from pathml.preprocessing import BoxBlur, Pipeline


@pytest.mark.parametrize("slide", [SlideData, HESlide, MultiparametricSlide])
def test_repr(slide):
    s = slide("tests/testdata/small_HE.svs")
    repr(s)


@pytest.mark.parametrize(
    "path,backend",
    [
        ("/test/testing/test.qptiff", "bioformats"),
        ("/test/dot.dot/space space space/File with.spaces and.dots.h5path", "h5path"),
        ("test.dcm", "dicom"),
        ("test.file.multiple.exts.jpg.qptiff.tiff.ome.tiff", "bioformats"),
    ],
)
def test_infer_backend(path, backend):
    assert infer_backend(path) == backend


def test_infer_backend_unsupported_extension():
    # Define a file path with an unsupported extension
    unsupported_path = "unsupported_file.xyz"

    # Use pytest.raises to verify that a ValueError is raised with the expected message
    with pytest.raises(ValueError) as excinfo:
        infer_backend(unsupported_path)

    # Check if the error message contains the expected content
    assert (
        f"input path {unsupported_path} doesn't match any supported file extensions"
        in str(excinfo.value)
    )


def test_write_with_array_labels(tmp_path, example_slide_data):
    example_slide_data.write(tmp_path / "test_array_in_labels.h5path")
    assert Path(tmp_path / "test_array_in_labels.h5path").is_file()


def test_run_pipeline(example_slide_data):
    if sys.platform.startswith("win"):
        pytest.skip(
            "dask distributed not available on windows", allow_module_level=False
        )

    pipeline = Pipeline([BoxBlur(kernel_size=15)])
    # start the dask client
    client = Client()
    # run the pipeline
    example_slide_data.run(pipeline=pipeline, client=client, tile_size=50)
    # close the dask client
    client.close()


@pytest.mark.parametrize("overwrite_tiles", [True, False])
def test_run_existing_tiles(slide_dataset_with_tiles, overwrite_tiles):

    # windows dask distributed incompatiblility
    if sys.platform.startswith("win"):
        dist = False
    else:
        dist = True
    dataset = slide_dataset_with_tiles
    pipeline = Pipeline([BoxBlur(kernel_size=15)])
    if overwrite_tiles:
        dataset.run(
            pipeline,
            overwrite_existing_tiles=overwrite_tiles,
            distributed=dist,
            tile_size=500,
        )
    else:
        with pytest.raises(Exception):
            dataset.run(
                pipeline,
                overwrite_existing_tiles=overwrite_tiles,
                distributed=dist,
                tile_size=500,
            )


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


@pytest.fixture
def ihc_slide_path():
    return "tests/testdata/small_HE.svs"


@pytest.fixture
def codex_slide_path():
    return "tests/testdata/small_vectra.qptiff"


def test_ihc_slide_creation(ihc_slide_path):
    slide = IHCSlide(ihc_slide_path)
    assert isinstance(slide, SlideData)
    assert slide.slide_type == pathml.types.IHC
    # Assuming 'backend' needs to be explicitly passed for IHCSlide, otherwise, test its default if applicable


def test_codex_slide_creation_with_default_backend(codex_slide_path):
    slide = CODEXSlide(codex_slide_path)
    assert isinstance(slide, SlideData)
    assert slide.slide_type == pathml.types.CODEX
    assert slide.backend == "bioformats"


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
    path = tmp_path / "testhe.test.test.dots space dots.h5"
    slidedata.write(path)
    readslidedata = SlideData(path)
    repr(readslidedata)
    assert readslidedata.name == slidedata.name
    assert readslidedata.shape == slidedata.shape
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


@pytest.mark.parametrize("write", [True, False])
def test_run_and_write(tmpdir, write):
    wsi = HESlide("tests/testdata/small_HE.svs", backend="openslide", name="testwrite")
    pipe = Pipeline()

    if write:
        write_dir_arg = tmpdir
    else:
        write_dir_arg = None

    wsi.run(pipe, tile_size=500, distributed=False, write_dir=write_dir_arg)

    written_path = tmpdir / "testwrite.h5path"

    if write:
        assert written_path.isfile()
    else:
        assert not written_path.isfile()
