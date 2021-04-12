import numpy as np
from pathlib import Path
import pytest

from pathml.core.slide_data import SlideData
from pathml.core.slide_backends import OpenSlideBackend


def test_repr():
    s = SlideData()
    repr(s)


def test_write_with_array_labels(tmp_path, example_slide_data):
    example_slide_data.write(tmp_path / "test_array_in_labels.h5path")
    assert Path(tmp_path / "test_array_in_labels.h5path").is_file()
