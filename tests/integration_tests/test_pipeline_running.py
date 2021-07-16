"""
Copyright 2021, Dana-Farber Cancer Institute and Weill Cornell Medicine
License: GNU GPL 2.0
"""

import os

import h5py
import numpy as np
import pytest
from dask.distributed import Client, LocalCluster
from pathml.core import HESlide, SlideData, VectraSlide
from pathml.preprocessing import (BoxBlur, CollapseRunsVectra, Pipeline,
                                  QuantifyMIF, SegmentMIF, TissueDetectionHE)


# test HE pipelines with both DICOM and OpenSlide backends
@pytest.mark.parametrize(
    "im_path", ["tests/testdata/small_HE.svs", "tests/testdata/small_dicom.dcm"]
)
@pytest.mark.parametrize("dist", [False, True])
def test_pipeline_HE(tmp_path, im_path, dist):
    slide = HESlide(im_path)
    pipeline = Pipeline(
        [BoxBlur(kernel_size=15), TissueDetectionHE(mask_name="tissue")]
    )
    if dist:
        cluster = LocalCluster(n_workers=2)
        cli = Client(cluster)
    else:
        cli = None
    slide.run(pipeline, distributed=dist, client=cli, tile_size=500)
    slide.write(path=str(tmp_path) + str(np.round(np.random.rand(), 8)) + "HE_slide.h5")
    if dist:
        cli.shutdown()


# test pipelines with bioformats backends, both tiff and qptiff files
# need to test tif and qptiff because they can have different behaviors due to different shapes (HWC vs HWZCT)
@pytest.mark.parametrize("dist", [False, True])
@pytest.mark.parametrize("tile_size", [400, (640, 480)])
def test_pipeline_bioformats_tiff(tmp_path, dist, tile_size):
    slide = VectraSlide("tests/testdata/smalltif.tif")
    # use a passthru dummy pipeline
    pipeline = Pipeline([])
    if dist:
        cluster = LocalCluster(n_workers=2)
        cli = Client(cluster)
    else:
        cli = None
    slide.run(pipeline, distributed=dist, client=cli, tile_size=tile_size)
    slide.write(path=str(tmp_path) + "tifslide.h5")
    readslidedata = SlideData(str(tmp_path) + "tifslide.h5")
    assert readslidedata.name == slide.name
    print(readslidedata.labels)
    print(slide.labels)
    np.testing.assert_equal(readslidedata.labels, slide.labels)
    if slide.masks is None:
        assert readslidedata.masks is None
    if slide.tiles is None:
        assert readslidedata.tiles is None
    assert scan_hdf5(readslidedata.h5manager.h5) == scan_hdf5(slide.h5manager.h5)
    if readslidedata.counts.obs.empty:
        assert slide.counts.obs.empty
    else:
        np.testing.assert_equal(readslidedata.counts.obs, slide.counts.obs)
    if readslidedata.counts.var.empty:
        assert slide.counts.var.empty
    else:
        np.testing.assert_equal(readslidedata.counts.var, slide.counts.var)
    os.remove(str(tmp_path) + "tifslide.h5")
    if dist:
        cli.shutdown()


# currently can't use distributed with SegmentMIF, since mesmer tensorflow model isn't pickleable for dask
@pytest.mark.parametrize("dist", [False])
@pytest.mark.parametrize("tile_size", [1000, (1920, 1440)])
def test_pipeline_bioformats_vectra(tmp_path, dist, tile_size):
    slide = VectraSlide("tests/testdata/small_vectra.qptiff")
    pipeline = Pipeline(
        [
            CollapseRunsVectra(),
            SegmentMIF(
                model="mesmer",
                nuclear_channel=0,
                cytoplasm_channel=2,
                image_resolution=0.5,
            ),
            QuantifyMIF(segmentation_mask="cell_segmentation"),
        ]
    )
    if dist:
        cluster = LocalCluster(n_workers=2)
        cli = Client(cluster)
    else:
        cli = None
    slide.run(pipeline, distributed=dist, client=cli, tile_size=tile_size)
    slide.write(path=str(tmp_path) + "vectraslide.h5")
    os.remove(str(tmp_path) + "vectraslide.h5")
    if dist:
        cli.shutdown()


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
