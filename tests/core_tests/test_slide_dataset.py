"""
Copyright 2021, Dana-Farber Cancer Institute and Weill Cornell Medicine
License: GNU GPL 2.0
"""

from pathlib import Path

import pytest

from pathml.core import SlideData, Tile
from pathml.preprocessing import BoxBlur, Pipeline


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


# def test_run_pipeline_and_tile_dataset_and_reshape(slide_dataset):
#     for slide in slide_dataset.slides:
#         slide.tile_size = 50
#     pipeline = Pipeline([BoxBlur(kernel_size=15)])
#     # run the pipeline
#     slide_dataset.run(pipeline=pipeline, distributed=False)
#     tile = slide_dataset[0].tiles[0]
#     assert isinstance(tile, Tile)
#     assert tile.image.shape == (50, 50, 3)


@pytest.mark.parametrize("write", [True, False])
def test_run_and_write_dataset(tmpdir, write, slide_dataset):
    pipe = Pipeline()

    if write:
        write_dir_arg = tmpdir
    else:
        write_dir_arg = None

    slide_dataset.tile_size = 500
    slide_dataset.run(pipe, distributed=False, write_dir=write_dir_arg)

    for s in slide_dataset:
        written_path = tmpdir / f"{s.name}.h5path"
        if write:
            assert written_path.isfile()
        else:
            assert not written_path.isfile()
