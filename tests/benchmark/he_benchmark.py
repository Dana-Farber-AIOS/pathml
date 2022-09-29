"""
Copyright 2021, Dana-Farber Cancer Institute and Weill Cornell Medicine
License: GNU GPL 2.0
"""

# benchmark a simple H&E image pipeline with 10 workers on a local cluster
# usage: `python he_benchmark.py`

import argparse
import cProfile
import logging
import pstats
import tempfile
from pstats import SortKey

import h5py
import matplotlib.pyplot as plt
import numpy as np
from dask.distributed import Client, LocalCluster
from torch.utils.data import DataLoader

from pathml.core import HESlide
from pathml.ml import TileDataset
from pathml.preprocessing import BoxBlur, Pipeline, TissueDetectionHE
from pathml.utils import download_from_url

parser = argparse.ArgumentParser()
parser.add_argument(
    "-n", "--nworkers", help="number of workers", type=int, default=10, dest="n_workers"
)
args = parser.parse_args()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# cProfile insists that this benchmark is written directly in the main method
# fails if written in a separate method then called in main method
if __name__ == "__main__":
    logging.info("beginning file download")
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

    logging.info(f"spinning up LocalCluster with {args.n_workers} workers")
    cluster = LocalCluster(n_workers=args.n_workers)
    client = Client(cluster)

    logging.info("beginning pipeline run")
    # run cProfile for parallel pipeline
    cProfile.run(
        "wsi.run(pipeline, distributed=True, tile_size=256, client=client)",
        "benchmark_pipeline_running",
    )
    logging.info("shutting down dask client")

    client.shutdown()

    logging.info("writing to h5path")

    cProfile.run(
        "wsi.write('benchmark_he.h5path')",
        "benchmark_writing_to_h5",
    )

    logging.info("creating dataloader")
    dset = TileDataset("benchmark_he.h5path")
    dloader = DataLoader(dset, batch_size=16, shuffle=True, num_workers=4)
    cProfile.run(
        "for batch in dloader: pass",
        "benchmark_dataloader",
    )

    logging.info("printing benchmarking results")

    # sort profile by cumulative time in a function and print 10 most significant lines
    pipeline_stats = pstats.Stats("benchmark_pipeline_running")
    pipeline_stats.strip_dirs().sort_stats(SortKey.CUMULATIVE).print_stats(10)

    writing_h5_stats = pstats.Stats("benchmark_writing_to_h5")
    writing_h5_stats.strip_dirs().sort_stats(SortKey.CUMULATIVE).print_stats(10)

    dataloader_stats = pstats.Stats("benchmark_dataloader")
    dataloader_stats.strip_dirs().sort_stats(SortKey.CUMULATIVE).print_stats(10)
