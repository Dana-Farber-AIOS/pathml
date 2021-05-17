"""
Copyright 2021, Dana-Farber Cancer Institute and Weill Cornell Medicine
License: GNU GPL 2.0
"""

from dask.distributed import Client


from pathml.core import HESlide, VectraSlide
from pathml.preprocessing import Pipeline, BoxBlur, TissueDetectionHE, SegmentMIF, QuantifyMIF


def test_pipeline_1(tmp_path):
    slide = HESlide("tests/testdata/small_HE.svs")
    pipeline = Pipeline([
        BoxBlur(kernel_size = 15),
        TissueDetectionHE(mask_name = "tissue")
    ])

    client = Client()

    slide.run(pipeline, client = client, tile_size = 250)

    #shut down the client after running
    client.close()

def test_vectra_pipeline(tmp_path):
    slide = VectraSlide("tests/testdata/small_vectra.qptiff")
    pipeline = Pipeline([
        SegmentMIF(model='mesmer', nuclear_channel=0, cytoplasm_channel=6, image_resolution=0.5),
        QuantifyMIF()
    ])

    client = Client()

    slide.run(pipeline, client = client, tile_size = 250)

    slide.tiles.write(out_dir = tmp_path, filename = "vectratiles.h5")

    #shut down the client after running
    client.close()
