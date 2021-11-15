"""
Copyright 2021, Dana-Farber Cancer Institute and Weill Cornell Medicine
License: GNU GPL 2.0
"""

# benchmark a simple H&E image pipeline with 10 workers on a local cluster

import cProfile
import pstats
import tempfile
from pstats import SortKey

import h5py
import matplotlib.pyplot as plt
import numpy as np
from dask.distributed import Client, LocalCluster
from pathml.core import HESlide
from pathml.preprocessing import BoxBlur, Pipeline, TissueDetectionHE
from pathml.utils import download_from_url

# cProfile insists that this benchmark is written directly in the main method
# fails if written in a separate method then called in main method
if __name__ == "__main__":
    download_from_url(
        "https://data.cytomine.coop/open/openslide/aperio-svs/CMU-1.svs",
        download_dir="testdata/",
    )
    wsi = HESlide("testdata/CMU-1.svs", name="example")
    pipeline = Pipeline(
        [
            BoxBlur(kernel_size=15),
            TissueDetectionHE(
                mask_name="tissue",
                min_region_size=500,
                threshold=30,
                outer_contours_only=True,
            ),
        ]
    )

    cluster = LocalCluster(n_workers=10)
    client = Client(cluster)

    # run cProfile for parallel pipeline
    cProfile.run(
        "wsi.run(pipeline, distributed=True, tile_size=256, client=client)",
        "benchmark_results",
    )
    p = pstats.Stats("benchmark_results")
    p.strip_dirs().sort_stats(-1).print_stats()
