Preprocessing Quickstart
========================

Preprocessing pipelines define how raw images are transformed and prepared to be fed as inputs into models.
The ``pathml.preprocessing`` module provides tools for defining preprocessing pipelines for whole-slide images.

Getting started with premade pipelines
--------------------------------------

The general preprocessing workflow is:

.. image:: _static/images/preprocess_schematic_single.png

Get started by loading a WSI from disk and running a preprocessing pipeline in 10 lines of code:

.. code-block::

    from pathml.core.slide_classes import HESlide
    from pathml.preprocessing.pipeline import Pipeline
    from pathml.preprocessing.transforms import BoxBlur, TissueDetectionHE

    wsi = HESlide("../data/CMU-1.svs", name = "example")

    pipeline = Pipeline([
        BoxBlur(kernel_size=15),
        TissueDetectionHE(mask_name = "tissue", min_region_size=500,
                          threshold=30, outer_contours_only=True)
    ])

    wsi.run(pipeline)


Pipelines can also be run on entire datasets, with no change to the code:

.. image:: _static/images/preprocess_schematic_dataset.png

.. code-block::

    from pathml.datasets import PESO
    from pathml.preprocessing.pipelines import DefaultTilingPipeline

    peso = PESO(data_dir = "/path/to/data/", download = True)
    pipeline = DefaultTilingPipeline()
    pipeline.run(peso, output_dir = "/path/to/output/dir")

    # to be updated

When running a pipeline on a dataset, ``PathML`` will use multiprocessing by default to distribute the workload to
all available cores. This allows users to efficiently process large datasets by scaling up computational resources
(local cluster, cloud machines, etc.) without needing to make any changes to the code.

Currently available premade pipelines
-------------------------------------

+--------------------------------------------+------------------------------------------------------------------------+
| Pipeline name                              | Description                                                            |
+============================================+========================================================================+
| DefaultHEPipeline                          | Divides input wsi into tiles. Does not apply any tile-level processing.|
+--------------------------------------------+------------------------------------------------------------------------+

[implement a few default pipelines, and add here with links in the left column]
