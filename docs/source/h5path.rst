HDF5 Integration
================

Overview
--------

A single whole-slide image may contain on the order of 10\ :superscript:`10` pixels, making it infeasible to
load entire images in RAM. ``PathML`` supports efficient manipulation of large-scale imaging data via
the **h5path** format, a hierarchical data structure which allows users to access small regions of the processed WSI
without loading the entire image. This feature reduces the RAM required to run a ``PathML`` workflow (pipelines can be
run on a consumer laptop), simplifies the reading and writing of processed WSIs, improves data exploration utilities,
and enables fast reading for downstream tasks (e.g. PyTorch Dataloaders). Since slides are managed on disk, your drive
must have sufficient storage. Performance will benefit from storage with fast read/write (SSD, NVMe). 
How it Works
------------

The internals of ``PathML`` as well as the **h5path** file format are based on the hierarchical data format
`HDF5 <https://en.wikipedia.org/wiki/Hierarchical_Data_Format>`_, implemented by
`h5py <https://docs.h5py.org/en/stable/>`_.
Each instantiation of :class:`~pathml.core.slide_data.SlideData` contains internal
references to temporary on-disk h5py objects. As tiles are extracted and passed to a preprocessing pipeline, the
processed tiles are then aggregated and stored in the slide's h5py object.
All interaction with h5py is automatically handled by ``PathML`` on the backend by
:class:`~pathml.core.h5manager._h5manager`. For example, ``slidedata.tiles[tile_key]`` returns the tile at
key ``tile_key`` from the h5py file on disk. Note that this command has syntax like an in-memory dict.
At the conclusion of preprocessing, the h5py object can optionally be
permanently written to disk in ``.h5path`` format via the :meth:`SlideData.write() <pathml.core.slide_data.SlideData.write>` method.

``.h5path`` File Format
------------------------

**h5path** utilizes a self-describing hierarchical file system that mirrors 
:class:`~pathml.core.slide_data`. This allows for simple reading and writing
of :class:`~pathml.core.slide_data` objects.

Here we examine the **h5path** file format:

::

    root/
    ├── fields/
    │   ├── name
    │   ├── labels
    │   ├── history
    │   └── slide_backend
    ├── array
    ├── masks/
    │   ├── arraymask1
    │   └── ...
    └── tiles/
        ├── tilesdict
        ├── tilemask1
        └── ...


Objects are saved to **h5path** if they are present in :class:`~pathml.core.slide_data`. 
The file system is organized through ``h5py.Groups``. ``/root/`` is a group, as are ``fields/``,
``masks/``, and ``tiles/``. Groups are container-like and can be queried like dictionaries:

.. code-block::

   import h5py
   root = h5py.File('pathtoh5.h5', 'r')
   masks = root['masks']

Within groups, array-like objects are stored as ``h5py.Dataset`` objects that when accessed return
``numpy.ndArray`` objects. All tiles are stitched together in a single ``h5py.Dataset`` at ``array/``.
A dictionary is maintained at ``tiles/tilesdict`` with coordinates and fields describing each tile.

.. important::

    To retrieve a ``numpy.ndArray`` object from ``h5py.Dataset`` you must slice the Dataset with
    NumPy fancy-indexing syntax: for example [...] to retrieve the full array, or [a:b, :] to
    return the array with first dimension sliced to the interval [a, b].

.. code-block::

   import h5py
   root = h5py.File('path/to/file.h5path', 'r')
   masks = root['masks']
   segmentationmask = masks['segmentationmask'][...]
   segmentationmaskslice = segmentationmask[2,:,:]

Attributes are small named fields attached to ``h5py.Dataset`` and ``h5py.Group`` objects. String,
tuple, and dict type objects are stored as attributes of the ``Group`` or ``Dataset`` they describe.
``name`` and ``labels`` are stored as attributes describing ``fields/``.

.. code-block::

   import h5py
   root = h5py.File('path/to/file.h5path', 'r')
   tile = root['tiles']['tile1']
   tilecoords = tile.attrs['coords']

Reading and Writing
-------------------

:class:`~pathml.core.slide_data.SlideData` objects are easily written to **h5path** format
by calling :meth:`SlideData.write() <pathml.core.slide_data.SlideData.write>`.
All files with ``.h5`` or ``.h5path`` extensions are loaded to :class:`~pathml.core.slide_data.SlideData` objects
automatically by calling :func:`~pathml.core.h5path.read`.
