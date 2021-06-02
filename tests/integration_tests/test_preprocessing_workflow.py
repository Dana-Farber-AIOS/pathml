"""
Copyright 2021, Dana-Farber Cancer Institute and Weill Cornell Medicine
License: GNU GPL 2.0
"""

from dask.distributed import Client


from pathml.core import HESlide, VectraSlide
from pathml.preprocessing import Pipeline, BoxBlur, TissueDetectionHE, SegmentMIF, QuantifyMIF, CollapseRunsVectra


def test_pipeline_1(tmp_path):
    slide = HESlide("tests/testdata/small_HE.svs")
    pipeline = Pipeline([
        BoxBlur(kernel_size = 15),
        TissueDetectionHE(mask_name = "tissue")
    ])

    slide.run(pipeline, distributed=False, tile_size = 250)

def test_vectra_pipeline(tmp_path):
    slide = VectraSlide("tests/testdata/small_vectra.qptiff")
    pipeline = Pipeline([
        CollapseRunsVectra(),
        SegmentMIF(model='mesmer', nuclear_channel=0, cytoplasm_channel=2, image_resolution=0.5),
        QuantifyMIF(segmentation_mask='cell_segmentation')
    ])

    slide.run(pipeline, distributed=False, tile_size = 250)

    slide.write(path = str(tmp_path) + "vectraslide.h5")
