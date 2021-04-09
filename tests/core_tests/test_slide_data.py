import numpy as np
from pathlib import Path

from pathml.core.slide_data import SlideData
from pathml.core.slide_backends import OpenSlideBackend


def test_repr():
    s = SlideData()
    repr(s)


def test_write_with_array_labels(tmp_path):
    labs = {"testing": "testlabel", "test2": np.array([2, 3, 4])}
    wsi = SlideData("tests/testdata/small_HE.svs", name = f"test_array_in_labels",
                    labels = labs, slide_backend = OpenSlideBackend)
    wsi.write(tmp_path / "test_array_in_labels.h5path")
    assert Path(tmp_path / "test_array_in_labels.h5path").is_file()
