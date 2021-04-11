import pytest
import h5py
import numpy as np
import os

from pathml.core.masks import Masks
from pathml.core.slide_classes import HESlide
from pathml.core.h5path import read
from pathml.preprocessing.pipeline import Pipeline
from pathml.preprocessing.transforms import BoxBlur, TissueDetectionHE

@pytest.fixture
def he_slidedata():
    masks = Masks({'example' : np.ones((2967, 2220))})
    wsi = HESlide("tests/testdata/small_HE.svs", name = "test", masks = masks, labels={'testkey':'testval'})
    pipeline = Pipeline([
        BoxBlur(kernel_size=15),
        TissueDetectionHE(mask_name = "tissue", min_region_size = 500,
                          threshold = 30, outer_contours_only = True)
    ])
    wsi.run(pipeline, tile_size = 250)
    # add labels and name to test read/write from tilesdict
    for tile in wsi.tiles.h5manager.tilesdict:
        wsi.tiles.h5manager.tilesdict[tile]['labels'] = {'key1' : 'val1', 'key2' : 'val2'}
        wsi.tiles.h5manager.tilesdict[tile]['name'] = str(tile)
    return wsi

def test_read_write_heslide(tmp_path, he_slidedata):
    slidedata = he_slidedata
    path = tmp_path / 'testhe.h5'
    slidedata.write(path)
    readslidedata = read(path) 
    assert readslidedata.name == slidedata.name
    assert readslidedata.slide_backend == slidedata.slide_backend
    assert readslidedata.labels == slidedata.labels
    assert readslidedata.history == slidedata.history
    if slidedata.masks is None:
        assert readslidedata.masks is None
    if slidedata.masks is not None:
        assert scan_hdf5(readslidedata.masks.h5manager.h5) == scan_hdf5(slidedata.masks.h5manager.h5)
    if slidedata.tiles is None:
        assert readslidedata.tiles is None
    if slidedata.tiles is not None:
        assert scan_hdf5(readslidedata.tiles.h5manager.h5) == scan_hdf5(slidedata.tiles.h5manager.h5)
        assert readslidedata.tiles.h5manager.tilesdict == slidedata.tiles.h5manager.tilesdict

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


def test_write_to_existing_file_fails(tmp_path):
    # trying to write to an existing file should raise an exception
    wsi1 = HESlide("tests/testdata/small_HE.svs", name = "test1")
    wsi2 = HESlide("tests/testdata/small_HE.svs", name = "test2")
    wsi1.write(tmp_path / "testing.h5path")
    with pytest.raises(ValueError):
        wsi2.write(tmp_path / "testing.h5path")
