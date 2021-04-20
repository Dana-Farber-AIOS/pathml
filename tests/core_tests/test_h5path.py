import pytest
import h5py
import numpy as np

from pathml.core.slide_classes import HESlide
from pathml.core.h5path import read


def test_read_write_heslide(tmp_path, example_slide_data_with_tiles):
    slidedata = example_slide_data_with_tiles
    path = tmp_path / 'testhe.h5'
    slidedata.write(path)
    readslidedata = read(path) 
    assert readslidedata.name == slidedata.name
    assert readslidedata.slide_backend == slidedata.slide_backend
    np.testing.assert_equal(readslidedata.labels, slidedata.labels)
    assert readslidedata.history == slidedata.history
    if slidedata.masks is None:
        assert readslidedata.masks is None
    if slidedata.masks is not None:
        assert scan_hdf5(readslidedata.masks.h5manager.h5) == scan_hdf5(slidedata.masks.h5manager.h5)
    if slidedata.tiles is None:
        assert readslidedata.tiles is None
    if slidedata.tiles is not None:
        assert scan_hdf5(readslidedata.tiles.h5manager.h5) == scan_hdf5(slidedata.tiles.h5manager.h5)
        print(readslidedata.tiles.h5manager.tiles)
        print(slidedata.tiles.h5manager.tiles)
        np.testing.assert_equal(readslidedata.tiles.h5manager.tiles, slidedata.tiles.h5manager.tiles)


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
