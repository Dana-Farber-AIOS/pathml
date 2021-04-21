Core API
========

SlideData
---------

The central class in ``PathML`` for representing a whole-slide image.

.. autoclass:: pathml.core.slide_data.SlideData
    :members:

RGBSlide
^^^^^^^^

.. autoclass:: pathml.core.slide_data.RGBSlide
    :members:

HESlide
^^^^^^^

.. autoclass:: pathml.core.slide_data.HESlide
    :members:

MultiparametricSlide
^^^^^^^^^^^^^^^^^^^^

.. autoclass:: pathml.core.slide_data.MultiparametricSlide
    :members:

Tile
----

.. automodule:: pathml.core.tile
    :members:

SlideDataset
------------

.. autoclass:: pathml.core.slide_dataset.SlideDataset
    :members:

Tiles and Masks helper classes
------------------------------

.. automodule:: pathml.core.tiles
    :members:

.. automodule:: pathml.core.masks
    :members:


Slide Backends
--------------

OpenslideBackend
^^^^^^^^^^^^^^^^

.. autoclass:: pathml.core.slide_backends.OpenSlideBackend
    :members:

BioFormatsBackend
^^^^^^^^^^^^^^^^^

.. autoclass:: pathml.core.slide_backends.BioFormatsBackend
    :members:

DICOMBackend
^^^^^^^^^^^^

.. autoclass:: pathml.core.slide_backends.DICOMBackend
    :members:

HDF5 Integration
----------------

.. automodule:: pathml.core.h5managers
    :members:

.. automodule:: pathml.core.h5path
    :members:
