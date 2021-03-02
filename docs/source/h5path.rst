h5path
======

Overview
--------

``PathML`` supports efficient on-disk manipulation and storage of imaging data. 

The **h5path** format, based on the hierarchical array store 
`HDF5 <https://en.wikipedia.org/wiki/Hierarchical_Data_Format>`_ and utilizing 
`h5py <https://docs.h5py.org/en/stable/>`_, enables NumPy-style manipulation of multi-terabyte 
numerical data stored on disk. This feature dramatically reduces the RAM required to run a 
``PathML`` workflow and simplifies the reading and writing of processed WSI data.

:class:`~pathml.core.slide_data.SlideData` array-like fields :class:`~pathml.core.tiles` and 
:class:`~pathml.core.masks` contain references to temporary on disk h5py objects that can be 
optionally saved or are automatically deleted when their reference is lost. 

Interaction with h5 backed masks and tiles is automatically handled through 
:class:`~pathml.core.h5manager._h5manager`, for example ``slidedata.masks[amask]`` returns the mask at 
key amask from h5py on disk, but has syntax like an in-memory dict.

Hierarchical File System
------------------------

**h5path** utilizes a self-describing hierarchical file system that mirrors 
:class:`~pathml.core.slide_data`. Here we examine an example **h5path** file:

* /root
    * /fields (slide-level labels)
        * ///name
        * ///slide_backend
        * ///history
    * /masks 
        * //mask1
        * //mask2
        * ...
    * /tiles
        * /tile1
            * //tile
            * ///coords
            * ///slidetype
            * /masks (tile-level masks)
                * //mask1
                * //mask2
                * ...

The file system is organized through h5py.Groups. /root is a group, as are /fields, 
/masks, and /tiles. Groups are container-like and can be queried like dictionaries.

.. code-block::

   import h5py
   root = h5py.File('pathtoh5.h5', 'r')
   masks = root['masks']

Within groups, array-like objects are stored as h5py.Datasets that when accessed return 
numpy.ndArray objects. Arrays representing masks (//mask1) and tiles (//tile) are Datasets.


.. code-block::

   import h5py
   root = h5py.File('pathtoh5.h5', 'r')
   masks = root['masks']
   segmentationmask = masks['segmentationmask'][...]
   segmentationmaskslice = segmentationmask[2,:,:]

Attributes are small named fields attached to h5py.Dataset and h5py.Group objects. String,
tuple, and dict type objects are stored as attributes of the Group or Dataset they describe.
///coords and ///slidetype are attributes describing each /tile.

.. code-block::

   import h5py
   root = h5py.File('pathtoh5.h5', 'r')
   tile = root['tiles']['tile1']
   tilecoords = tile.attrs['coords']

Reading and Writing
-------------------

:class:`~pathml.core.slide_data.SlideData` objects are easily written to **h5path** format
by calling :meth:`~pathml.core.slide_data.SlideData.write`.
All files with .h5 or .h5py extensions are loaded to SlideData objects automatically by calling
:func:`~pathml.core.h5path.read`.
