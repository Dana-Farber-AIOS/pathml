Datasets Quickstart
===================

The ``pathml.datasets`` module provides easy access to common datasets for standardized model evaluation and comparison.

DataModules
--------------

``PathML`` uses ``DataModules`` to encapsulate datasets.
DataModule objects are responsible for downloading the data (if necessary) and formatting the data into ``DataSet`` s and
``DataLoader`` s for use in downstream tasks.
Keeping everything in a single object is easier for users and also facilitates reproducibility.

Inspired by `PyTorch Lightning <https://pytorch-lightning.readthedocs.io/en/latest/datamodules.html>`_.


Using public datasets
---------------------

PathML has built-in support for several public datasets:

.. list-table:: Datasets
   :widths: 20 50 10 20
   :header-rows: 1

   * - Dataset
     - Description
     - Image type
     - Size
   * - :class:`~pathml.datasets.pannuke.PanNukeDataModule`
     - Pixel-level nucleus classification, with 6 nucleus types and 19 tissue types.
       Images are 256px RGB. [PanNuke1]_ [PanNuke2]_
     - H&E
     - n=7901 (37.33 GB)
   * - :class:`~pathml.datasets.deepblur.DeepFocusDataModule`
     - Patch-level focus classification with 3 IHC and 1 H&E histologies. [DeepFocus]_
     - H&E, IHC
     - n=204k (10.0 GB)


References
----------

.. [PanNuke1] Gamper, J., Koohbanani, N.A., Benet, K., Khuram, A. and Rajpoot, N., 2019, April. PanNuke: an open pan-cancer
        histology dataset for nuclei instance segmentation and classification. In European Congress on Digital
        Pathology (pp. 11-19). Springer, Cham.
.. [PanNuke2] Gamper, J., Koohbanani, N.A., Graham, S., Jahanifar, M., Khurram, S.A., Azam, A., Hewitt, K. and Rajpoot, N.,
        2020. PanNuke Dataset Extension, Insights and Baselines. arXiv preprint arXiv:2003.10778.
.. [DeepFocus] Senaras, C., Niazi, M., Lozanski, G., Gurcan, M., 2018, October. Deepfocus: Detection of out-of-focus regions
        in whole slide digital images using deep learning. PLOS One 13(10): e0205387.
