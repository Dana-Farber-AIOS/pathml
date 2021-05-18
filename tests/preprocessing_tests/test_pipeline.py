"""
Copyright 2021, Dana-Farber Cancer Institute and Weill Cornell Medicine
License: GNU GPL 2.0
"""

import pickle
import numpy as np
import pandas as pd
import copy

from pathml.preprocessing import Pipeline

from pathml.preprocessing import (
    MedianBlur, GaussianBlur, BoxBlur, BinaryThreshold,
    MorphOpen, MorphClose, ForegroundDetection, SuperpixelInterpolation,
    StainNormalizationHE, NucleusDetectionHE, TissueDetectionHE,
    SegmentMIF, QuantifyMIF, CollapseRunsVectra
)
from pathml.utils import RGB_to_GREY


# make an example pipeline
def test_pipeline_HE(tileHE):
    pipe = Pipeline([
        MedianBlur(),
        GaussianBlur(),
        BoxBlur(),
        BinaryThreshold(mask_name = "testing"),
        MorphOpen(mask_name = "testing"),
        MorphClose(mask_name = "testing")
    ])

    assert len(pipe) == 6

    orig_im = tileHE.image
    pipe.apply(tileHE)

    im = MedianBlur().F(orig_im)
    im = GaussianBlur().F(im)
    im = BoxBlur().F(im)
    m = BinaryThreshold().F(RGB_to_GREY(im))
    m = MorphOpen().F(m)
    m = MorphClose().F(m)

    assert np.array_equal(tileHE.image, im)
    assert np.array_equal(tileHE.masks["testing"], m)

# TODO: this segmentation model requires gpu 
def test_pipeline_mif(tileVectra):
    """
    Run MIF pipeline
    """
    orig_tile = copy.copy(tileVectra)
    pipe = Pipeline([
        CollapseRunsVectra(),
        SegmentMIF(model='mesmer', nuclear_channel=0, cytoplasm_channel=2, image_resolution=0.5),
        QuantifyMIF(segmentation_mask = 'cell_segmentation')
    ])
    
    assert len(pipe) == 3
    pipe.apply(tileVectra)

    collapsed_im = CollapseRunsVectra().F(orig_tile.image)
    cell_segmentation, nuclear_segmentation = SegmentMIF(model='mesmer', nuclear_channel=0, cytoplasm_channel=2, image_resolution=0.5).F(collapsed_im)
    orig_tile.image = collapsed_im
    orig_tile.masks["segmentation_mask"] = cell_segmentation
    adata = QuantifyMIF(segmentation_mask="segmentation_mask").F(orig_tile)

    assert np.array_equal(tileVectra.masks['cell_segmentation'], cell_segmentation)
    pd.testing.assert_frame_equal(adata.obs, tileVectra.counts.obs) 

    
def test_pipeline_save(tmp_path):
    # tmp_path is a temporary path used for testing
    fp = tmp_path / "test"

    pipeline = Pipeline([MedianBlur()])
    pipeline.save(fp)

    pipeline_loaded = pickle.load(open(fp, "rb"))

    assert repr(pipeline_loaded) == repr(pipeline)
    assert type(pipeline_loaded) == type(pipeline)
