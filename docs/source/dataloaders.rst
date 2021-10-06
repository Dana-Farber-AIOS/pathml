DataLoaders
===========

After running a preprocessing pipeline and writing the resulting ``.h5path`` file to disk, the next step is to
create a DataLoader for feeding tiles into a machine learning model in PyTorch.

To do this, use the :class:`~pathml.ml.dataset.TileDataset` class and then wrap it in a PyTorch DataLoader:

.. code-block::

    dataset = TileDataset("/path/to/file.h5path")
    dataloader = torch.utils.data.DataLoader(dataset, batch_size = 16, shuffle = True, num_workers = 4)

.. note::

    Label dictionaries are not standardized, as users are free to store whatever labels they want.
    For that reason, PyTorch cannot automatically stack labels into batches.
    It may therefore be necessary to create a custom ``collate_fn`` to specify how to create batches of labels.
    See `here <https://discuss.pytorch.org/t/how-to-use-collate-fn/27181>`_.

This provides an interface between PathML and the broader ecosystem of machine learning tools built on PyTorch.
For more information on how to use Datasets and DataLoaders, please see the PyTorch
`documentation <https://pytorch.org/docs/stable/data.html>`_ and
`tutorials <https://pytorch.org/tutorials/beginner/basics/data_tutorial.html>`_.
