from pathlib import Path
import pytest
from dask.distributed import Client

from pathml.core.slide_data import SlideData, HESlide, MultiparametricSlide, RGBSlide
from pathml.core.slide_backends import OpenSlideBackend, BioFormatsBackend
from pathml.preprocessing.pipeline import Pipeline
from pathml.preprocessing.transforms import BoxBlur
from pathml.core.tile import Tile


@pytest.mark.parametrize("slide", [SlideData, HESlide, RGBSlide, MultiparametricSlide])
def test_repr(slide):
    s = slide("tests/testdata/small_HE.svs")
    repr(s)


def test_write_with_array_labels(tmp_path, example_slide_data):
    example_slide_data.write(tmp_path / "test_array_in_labels.h5path")
    assert Path(tmp_path / "test_array_in_labels.h5path").is_file()


def test_run_pipeline(example_slide_data):
    pipeline = Pipeline([BoxBlur(kernel_size = 15)])
    # start the dask client
    client = Client()
    # run the pipeline
    example_slide_data.run(pipeline = pipeline, client = client, tile_size = 50)
    # close the dask client
    client.close()


@pytest.mark.parametrize("overwrite_tiles", [True, False])
def test_run_existing_tiles(slide_dataset_with_tiles, overwrite_tiles):
    dataset = slide_dataset_with_tiles
    pipeline = Pipeline([BoxBlur(kernel_size = 15)])
    if overwrite_tiles:
        dataset.run(pipeline, overwrite_existing_tiles = overwrite_tiles)
    else:
        with pytest.raises(Exception):
            dataset.run(pipeline, overwrite_existing_tiles = overwrite_tiles)


@pytest.fixture
def he_slide():
    wsi = HESlide("tests/testdata/small_HE.svs")
    return wsi


@pytest.fixture
def multiparametric_slide():
    wsi = MultiparametricSlide("tests/testdata/smalltif.tif")
    return wsi


def test_he(he_slide):
    assert isinstance(he_slide, SlideData)
    assert isinstance(he_slide, RGBSlide)
    assert isinstance(he_slide.slide, OpenSlideBackend)


def test_multiparametric(multiparametric_slide):
    assert isinstance(multiparametric_slide, SlideData)
    assert isinstance(multiparametric_slide.slide, BioFormatsBackend)


@pytest.mark.parametrize("shape", [500, (500, 400)])
@pytest.mark.parametrize("stride", [None, 1000])
@pytest.mark.parametrize("pad", [True, False])
@pytest.mark.parametrize("level", [0])
def test_generate_tiles_he(he_slide, shape, stride, pad, level):
    for tile in he_slide.generate_tiles(shape = shape, stride = stride, pad = pad, level = level):
        assert isinstance(tile, Tile)


@pytest.mark.parametrize("shape", [100, (50, 100)])
@pytest.mark.parametrize("stride", [None, 100])
@pytest.mark.parametrize("pad", [True, False])
def test_generate_tiles_multiparametric(multiparametric_slide, shape, stride, pad):
    for tile in multiparametric_slide.generate_tiles(shape = shape, stride = stride, pad = pad):
        assert isinstance(tile, Tile)


@pytest.mark.parametrize("pad", [True, False])
def test_generate_tiles_padding(he_slide, pad):
    shape = 300
    stride = 300
    tiles = list(he_slide.generate_tiles(shape = shape, stride = stride, pad = pad))
    # he_slide.slide.get_image_shape() --> (2967, 2220)
    # if no padding, expect: 9*7 = 63 tiles
    # if padding, expect: 10*8 - 80 tiles
    if not pad:
        assert len(tiles) == 63
    else:
        assert len(tiles) == 80
