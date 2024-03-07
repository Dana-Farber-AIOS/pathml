"""
Copyright 2021, Dana-Farber Cancer Institute and Weill Cornell Medicine
License: GNU GPL 2.0
"""

import os
import sys
import h5py
import numpy as np
import pytest
from dask.distributed import Client, LocalCluster

from pathml.core import HESlide, SlideData, VectraSlide
from pathml.ml import TileDataset
from pathml.preprocessing import (
    BoxBlur,
    CollapseRunsVectra,
    Pipeline,
    QuantifyMIF,
    TissueDetectionHE,
)
from pathml.preprocessing.transforms import Transform
from pathml.utils import pil_to_rgb


# test HE pipelines with both DICOM and OpenSlide backends
@pytest.mark.parametrize(
    "im_path", ["tests/testdata/small_HE.svs", "tests/testdata/small_dicom.dcm"]
)
@pytest.mark.parametrize("dist", [False, True])
def test_pipeline_HE(tmp_path, im_path, dist):

    if dist:
        if sys.platform.startswith("win"):
            pytest.skip("dask distributed not available on windows", allow_module_level=False)
    
    labs = {
        "test_string_label": "testlabel",
        "test_array_label": np.array([2, 3, 4]),
        "test_int_label": 3,
        "test_float_label": 3.0,
        "test_bool_label": True,
    }
    slide = HESlide(im_path, labels=labs)
    pipeline = Pipeline(
        [BoxBlur(kernel_size=15), TissueDetectionHE(mask_name="tissue")]
    )
    if dist:
        cluster = LocalCluster(n_workers=2)
        cli = Client(cluster)
    else:
        cli = None
    slide.run(pipeline, distributed=dist, client=cli, tile_size=500)
    save_path = str(tmp_path) + str(np.round(np.random.rand(), 8)) + "HE_slide.h5"
    slide.write(path=save_path)
    if dist:
        cli.shutdown()

    # test out the dataset
    dataset = TileDataset(save_path)
    assert len(dataset) == len(slide.tiles)

    im, mask, lab_tile, lab_slide = dataset[0]

    for k, v in lab_slide.items():
        if isinstance(v, np.ndarray):
            assert np.array_equal(v, labs[k])
        else:
            assert v == labs[k]
    assert np.array_equal(im, slide.tiles[0].image.transpose(2, 0, 1))


# test pipelines with bioformats backends, both tiff and qptiff files
# need to test tif and qptiff because they can have different behaviors due to different shapes (HWC vs HWZCT)
@pytest.mark.parametrize("dist", [False, True])
@pytest.mark.parametrize("tile_size", [256, (256, 256)])
def test_pipeline_bioformats_tiff(tmp_path, dist, tile_size):

    if dist:
        if sys.platform.startswith("win"):
            pytest.skip("dask distributed not available on windows", allow_module_level=False)
    
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


@pytest.mark.parametrize("dist", [False, True])
@pytest.mark.parametrize("tile_size", [256, (256, 256)])
def test_pipeline_bioformats_vectra(tmp_path, dist, tile_size):
    if dist:
        if sys.platform.startswith("win"):
            pytest.skip("dask distributed not available on windows", allow_module_level=False)
        
    from pathml.preprocessing.transforms import SegmentMIFRemote

    slide = VectraSlide("tests/testdata/small_vectra.qptiff")
    pipeline = Pipeline(
        [
            CollapseRunsVectra(),
            SegmentMIFRemote(
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


class AddMean(Transform):
    """Transform using global statistic for tile (average)"""

    def F(self, arr):
        return arr + np.mean(arr)

    def apply(self, tile):
        tile.image = self.F(tile.image)


@pytest.mark.parametrize("tile_size", [500])
@pytest.mark.parametrize("stride", [250, 500, 1000])
@pytest.mark.parametrize("pad", [True, False])
def test_pipeline_overlapping_tiles(tmp_path, stride, pad, tile_size):
    """test that we can run pipeline with overlapping tiles"""
    pipe = Pipeline([AddMean()])
    wsi = SlideData("tests/testdata/small_HE.svs")

    wsi.run(
        pipe, distributed=False, tile_size=tile_size, tile_stride=stride, tile_pad=pad
    )

    if pad:
        tile_count = [dim // stride + 1 for dim in wsi.shape]
    else:
        tile_count = [(dim - tile_size) // stride + 1 for dim in wsi.shape]

    # make sure that we have the correct number of tiles
    assert len(wsi.tiles) == np.prod(tile_count)

    path = tmp_path / "testhe.h5"
    wsi.write(path)
    readslidedata = SlideData(path)

    assert len(readslidedata.tiles) == np.prod(tile_count)

    # make sure that getting tiles works as expected
    # if overlapping tiles are not implemented correctly, this will fail because parts of the tile will
    # get overwritten by subsequent overlapping tiles, and because we are using a transform which is different
    # for each tile, we will be able to identify if this has happened
    im = pil_to_rgb(
        wsi.slide.slide.read_region(
            location=(1000, 1000), level=0, size=(tile_size, tile_size)
        )
    )
    expected = AddMean().F(im).astype(np.float16)
    np.testing.assert_equal(readslidedata.tiles[(1000, 1000)].image, expected)
