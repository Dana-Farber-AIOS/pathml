import glob
import re
from os import listdir, path

import anndata as ad
from dask.distributed import Client, LocalCluster

from pathml.core import SlideDataset
from pathml.core.slide_data import CODEXSlide
from pathml.preprocessing.pipeline import Pipeline
from pathml.preprocessing.transforms import CollapseRunsCODEX, QuantifyMIF, SegmentMIF


def run_vectra_workflow(
    slide_dir,
    slide_ext="tif",
    nuclear_channel=0,
    cytoplasmic_channel=29,
    image_resolution=0.377442,
    use_parallel=True,
    n_cpus=10,
    tile_size=(1920, 1440),
    save_slidedata_file="./data/dataset_processed.h5",
    save_anndata_file="./data/adata_combined.h5ad",
):

    # assuming that all slides are in a single directory, all with .tif file extension
    for A, B in [listdir(slide_dir)]:
        vectra_list_A = [
            CODEXSlide(p, stain="IF")
            for p in glob.glob(path.join(slide_dir, A, f"*.{slide_ext}"))
        ]
        vectra_list_B = [
            CODEXSlide(p, stain="IF")
            for p in glob.glob(path.join(slide_dir, B, f"*.{slide_ext}"))
        ]
        # Fix the slide names and add origin labels (A, B)
        for slide_A, slide_B in zip(vectra_list_A, vectra_list_B):
            slide_A.name = re.sub("X.*", "A", slide_A.name)
            slide_B.name = re.sub("X.*", "B", slide_B.name)
        # Store all slides in a SlideDataSet object
        dataset = SlideDataset(vectra_list_A + vectra_list_B)

    pipe = Pipeline(
        [
            CollapseRunsCODEX(z=0),
            SegmentMIF(
                model="mesmer",
                nuclear_channel=nuclear_channel,
                cytoplasm_channel=cytoplasmic_channel,
                image_resolution=image_resolution,
            ),
            QuantifyMIF(segmentation_mask="cell_segmentation"),
        ]
    )

    if use_parallel:
        # Initialize a dask cluster using 10 workers. PathML pipelines can be run in distributed mode on
        # cloud compute or a cluster using dask.distributed.
        cluster = LocalCluster(n_workers=n_cpus, threads_per_worker=1, processes=True)
        client = Client(cluster)

        # Run the pipeline
        dataset.run(
            pipe, distributed=True, client=client, tile_size=tile_size, tile_pad=False
        )
    else:
        dataset.run(pipe, distributed=False, tile_size=tile_size, tile_pad=False)

    # Write the processed datasets to disk
    dataset.write(save_slidedata_file)

    # Combine the count matrices into a single adata object:
    adata = ad.concat(
        [x.counts for x in dataset.slides],
        join="outer",
        label="Region",
        index_unique="_",
    )
    # Fix and replace the regions names
    origin = adata.obs["Region"]
    origin = origin.astype(str).str.replace(r"[^a-zA-Z0-9 \n\.]", "")
    origin = origin.astype(str).str.replace("[\n]", "")
    origin = origin.str.replace("SlideDataname", "")
    adata.obs["Region"] = origin

    # save the adata object
    adata.write(filename=save_anndata_file)
