import pytest

from pathml.core.slide_classes import HESlide
from pathml.core.h5path import read
from pathml.preprocessing.pipeline import Pipeline
from pathml.preprocessing.transforms import TissueDetectionHE

@pytest.fixture
def he_slidedata():
    wsi = HESlide("tests/testdata/small_HE.svs")
    pipeline = Pipeline([
        TissueDetectionHE(mask_name = "tissue", min_region_size = 500,
                          threshold = 30, outer_contours_only = True)
    ])
    wsi.run(pipeline)
    return wsi

def test_read_write(he_slidedata):
    slidedata = he_slidedata
    path = 'tests/testdata/testhe.h5path'
    slidedata.write(path)
    newslidedata = read(path) 
    assert newslidedata.name == slidedata.name
    assert newslidedata.slide_backend == slidedata.slide_backend
    assert newslidedata.labels == slidedata.labels
    assert newslidedata.history == slidedata.history
    # TODO: deeper check for masks and tiles?
    assert scan_hdf5(newslidedata.masks) == scan_hdf5(slidedata.masks)
    assert scan_hdf5(newslidedata.tiles) == scan_hdf5(slidedata.tiles)
    # how to do masks and tile?

def scan_hdf5(path, recursive=True, tab_step=2):
    def scan_node(g, tabs=0):
        elems = []
        for k, v in g.items():
            if isinstance(v, h5.Dataset):
                elems.append(v.name)
            elif isinstance(v, h5.Group) and recursive:
                elems.append((v.name, scan_node(v, tabs=tabs + tab_step)))
        return elems
    with h5.File(path, 'r') as f:
        return scan_node(f)

