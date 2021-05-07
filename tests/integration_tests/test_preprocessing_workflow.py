"""
Copyright 2021, Dana-Farber Cancer Institute and Weill Cornell Medicine
License: GNU GPL 2.0
"""

from dask.distributed import Client

from pathml.core import HESlide
from pathml.preprocessing import Pipeline, BoxBlur, TissueDetectionHE


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
