"""
Copyright 2021, Dana-Farber Cancer Institute and Weill Cornell Medicine
License: GNU GPL 2.0
"""

from dask.distributed import Client
from pathlib import Path

from pathml.core import SlideData, Tile
from pathml.preprocessing import Pipeline, BoxBlur


def test_dataset_len_getitem(slide_dataset):
    # check len and getitem
    assert len(slide_dataset) == 4
    for i in range(len(slide_dataset)):
        assert isinstance(slide_dataset[i], SlideData)


def test_dataset_save(tmp_path, slide_dataset):
    slide_dataset.write(tmp_path)
    # now check each file
    for slide in slide_dataset:
        fname = Path(str(tmp_path / slide.name) + ".h5path")
        assert fname.is_file()


def test_run_pipeline_and_tile_dataset_and_reshape(slide_dataset):
    pipeline = Pipeline([BoxBlur(kernel_size=15)])
    # run the pipeline
    slide_dataset.run(pipeline=pipeline, distributed=False, tile_size=50)

    tile = slide_dataset[0].tiles[0]
    assert isinstance(tile, Tile)
    assert tile.image.shape == (50, 50, 3)

    slide_dataset.reshape(shape=(25, 25, 3))
    tile_after_reshape = slide_dataset[0].tiles[0]
    assert isinstance(tile_after_reshape, Tile)
    assert tile_after_reshape.image.shape == (25, 25, 3)
