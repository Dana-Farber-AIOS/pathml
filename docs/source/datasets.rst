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

.. table::
    :widths: 10 60 10 10 10

    +-----------------------------------------------------+--------------------------------------+-------------+-----------+---------------+
    | Dataset                                             | Description                          | Image type  | Size      | Reference     |
    +=====================================================+======================================+=============+===========+===============+
    | :class:`~pathml.datasets.pannuke.PanNukeDataModule` | Pixel-level nucleus classification,  | H&E         | | n=7901  | | [PanNuke1]_ |
    |                                                     | with 6 nucleus types and 19 tissue   |             | | 37.33 GB| | [PanNuke2]_ |
    |                                                     | types. Images are 256px RGB.         |             |           |               |
    +-----------------------------------------------------+--------------------------------------+-------------+-----------+---------------+


Using local datasets
--------------------

Using "in-house" data from the local filesystem is also supported.
Classes :class:`~pathml.datasets.base.BaseSlideDataset` and :class:`~pathml.datasets.base.BaseTileDataset` are thin
wrappers around ``torch.utils.data.Dataset``, used for creating datasets at the slide or tile level.
Using a local dataset is as simple as creating a new class and implementing two special class methods: ``__len__(self)`` and ``__getitem__(self, index)``.

- ``__len__(self)`` should return the size of the dataset (i.e. total number of slides or tiles)
- ``__getitem__(self, i)`` should return a tuple of (*i* th item, *i* th label)

Create a dataset object for in-house data by creating a class with the logic for the above two methods for the local data.

Example
^^^^^^^

In the following example for a dataset of H&E slides and corresponding annotation masks, we assume that the slides are
saved in a directory named ``slides/`` and the annotations are saved in a directory named ``masks/``:

.. code-block::

    from pathlib import Path
    import cv2
    from pathml.datasets.base import BaseSlideDataset
    from pathml.preprocessing.wsi import HESlide

    class MyDataset(BaseSlideDataset):
        def __init__(self, data_path):
            self.data_path = Path(data_path)
            # assumes identical order of files in slides/ and masks/
            self.slide_paths = list(self.data_path.glob("slides/*.svs"))
            self.mask_paths  = list(self.data_path.glob("masks/*.jpg"))

        def __len__(self):
            # return total number of slides in dataset
            return len(self.slide_paths)

        def __getitem__(self, i):
            # return (slide, label) pair for the ith slide
            slide = HESlide(self.slide_paths[i])
            mask = cv2.imread(self.mask_paths[i])
            return slide, mask


References
----------

.. [PanNuke1] Gamper, J., Koohbanani, N.A., Benet, K., Khuram, A. and Rajpoot, N., 2019, April. PanNuke: an open pan-cancer
        histology dataset for nuclei instance segmentation and classification. In European Congress on Digital
        Pathology (pp. 11-19). Springer, Cham.
.. [PanNuke2] Gamper, J., Koohbanani, N.A., Graham, S., Jahanifar, M., Khurram, S.A., Azam, A., Hewitt, K. and Rajpoot, N.,
        2020. PanNuke Dataset Extension, Insights and Baselines. arXiv preprint arXiv:2003.10778.