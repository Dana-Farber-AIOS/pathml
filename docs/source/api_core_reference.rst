Core API
========

SlideData
---------

The central class in ``PathML`` for representing a whole-slide image.

.. autoclass:: pathml.core.SlideData

RGBSlide
^^^^^^^^

.. autoclass:: pathml.core.RGBSlide

HESlide
^^^^^^^

.. autoclass:: pathml.core.HESlide

MultiparametricSlide
^^^^^^^^^^^^^^^^^^^^

.. autoclass:: pathml.core.MultiparametricSlide

Tile
----

.. autoclass:: pathml.core.Tile

SlideDataset
------------

.. autoclass:: pathml.core.SlideDataset

Tiles and Masks helper classes
------------------------------

.. autoclass:: pathml.core.Tiles

.. autoclass:: pathml.core.Masks


Slide Backends
--------------

OpenslideBackend
^^^^^^^^^^^^^^^^

.. autoclass:: pathml.core.OpenSlideBackend

BioFormatsBackend
^^^^^^^^^^^^^^^^^

.. autoclass:: pathml.core.BioFormatsBackend

DICOMBackend
^^^^^^^^^^^^

.. autoclass:: pathml.core.DICOMBackend

Reading and Writing
-------------------

.. autofunction:: pathml.core.h5path.write_h5path
.. autofunction:: pathml.core.h5path.read
.. autofunction:: pathml.core.h5path.read_h5path
.. autofunction:: pathml.core.h5path.read_openslide
.. autofunction:: pathml.core.h5path.read_bioformats
.. autofunction:: pathml.core.h5path.read_dicom
.. autofunction:: pathml.core.h5path.is_valid_path
