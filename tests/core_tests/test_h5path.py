"""
Copyright 2021, Dana-Farber Cancer Institute and Weill Cornell Medicine
License: GNU GPL 2.0
"""

import h5py
import numpy as np
from pandas.testing import assert_frame_equal

from pathml.core import read


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
        np.testing.assert_equal(readslidedata.tiles.h5manager.tiles, slidedata.tiles.h5manager.tiles)
    if slidedata.counts is not None:
        assert_frame_equal(readslidedata.counts.obs, slidedata.counts.obs)


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
