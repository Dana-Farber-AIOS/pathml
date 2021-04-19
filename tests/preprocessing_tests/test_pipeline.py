import pytest
import pickle
import numpy as np

from pathml.preprocessing.pipeline import Pipeline

from pathml.preprocessing.transforms import (
    MedianBlur, GaussianBlur, BoxBlur, BinaryThreshold,
    MorphOpen, MorphClose, ForegroundDetection, SuperpixelInterpolation,
    StainNormalizationHE, NucleusDetectionHE, TissueDetectionHE,
    BackgroundSubtractCODEX, DriftCompensateMIF, DeconvolveMIF,
    SegmentMIF, QuantifyMIF
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


def test_pipeline_Vectra(tileIHC):
    """
    Vectra images are background subtracted during spectral deconvolution step.
    """
    pipe = Pipeline([
        DeconvolveMIF(),
        SegmentMIF(),
        QuantifyMIF()
    )]
    
    assert len(pipe) == 2
    pipe.apply(tileIHC)

    orig_im = tileIHC.image
    segmented_mask = SegmentMIF().F(orig_im)
    adata = QuantifyMIF().F(orig_im, segmented_mask)

    assert np.array_equal(tileIHC.masks['cell_segmentation'], segmented_mask)
    assert adata == tileIHC.counts 


def test_pipeline_CODEX(tileCODEX):
    pipe = Pipeline([
        BackgroundSubtractCODEX(),
        DriftCompensateMIF(),
        DeconvolveMIF(),
        SegmentMIF(),
        QuantifyMIF()
    ])

    assert len(pipe) == 5
    pipe.apply(tileCODEX)



def test_pipeline_save(tmp_path):
    # tmp_path is a temporary path used for testing
    fp = tmp_path / "test"

    pipeline = Pipeline([MedianBlur()])
    pipeline.save(fp)

    pipeline_loaded = pickle.load(open(fp, "rb"))

    assert repr(pipeline_loaded) == repr(pipeline)
    assert type(pipeline_loaded) == type(pipeline)
