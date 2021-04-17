import numpy as np
from pathlib import Path
import pytest
from dask.distributed import Client

from pathml.core.slide_data import SlideData
from pathml.core.slide_backends import OpenSlideBackend
from pathml.preprocessing.pipeline import Pipeline
from pathml.preprocessing.transforms import BoxBlur


def test_repr():
    s = SlideData()
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
