import pytest
import numpy as np

from pathml.datasets.base import BaseSlideDataset
from pathml.preprocessing.tiling import Tile
from pathml.preprocessing.wsi import HESlide


@pytest.fixture
def example_he_tile():
    wsi = HESlide(path = "tests/testdata/CMU-1-Small-Region.svs")
    slide_data = wsi.load_data(level = 0, location = (900, 800), size = (100, 100))
    return slide_data.image


@pytest.fixture
def example_slide_data():
    path = "tests/testdata/CMU-1-Small-Region.svs"
    slide = HESlide(path)
    out = slide.load_data(level = 0, location = (900, 800), size = (100, 100))
    return out


@pytest.fixture
def example_slide_data_with_tiles(example_slide_data):
    tiles = []
    n_tiles = 8
    for ix in range(n_tiles):
        t = Tile(np.random.randint(0, 255, (20, 20, 3)), i = ix, j = ix)
        tiles.append(t)
    example_slide_data.tiles = tiles
    return example_slide_data


@pytest.fixture
def example_slide_data_with_mask():
    wsi = HESlide(path = "tests/testdata/CMU-1-Small-Region.svs")
    slide_data = wsi.load_data(level = 0, location = (900, 800), size = (100, 100))
    m = np.zeros((100, 100), dtype = np.uint8)
    m[0:55, 0:55] = 1
    slide_data.mask = m
    m2 = np.zeros((100, 100), dtype = np.uint8)
    m2[0:95, 0:95] = 1
    slide_data.mask = m2
    return slide_data


@pytest.fixture
def example_slide_dataset():

    class ExampleSlideDataset(BaseSlideDataset):
        """dummy dataset containing n copies of example_slide_data"""
        def __init__(self, n=4):
            path = "tests/testdata/CMU-1-Small-Region.svs"
            self._slides = [HESlide(path, name = f"slide_{ix}") for ix in range(n)]
            self._slides = [s.load_data(level = 0, location = (900, 800), size = (100, 100)) for s in self._slides]

        def __getitem__(self, ix):
            return self._slides[ix]

        def __len__(self):
            return len(self._slides)

    return ExampleSlideDataset(n=4)


