Core API
========

SlideData
---------

The central class in ``PathML`` for representing a whole-slide image.

.. autoclass:: pathml.core.SlideData


Convenience SlideData Classes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: pathml.core.HESlide
.. autoclass:: pathml.core.VectraSlide
.. autoclass:: pathml.core.MultiparametricSlide
.. autoclass:: pathml.core.CODEXSlide


Slide Types
-----------

.. autoclass:: pathml.core.SlideType
    :exclude-members: tma, platform, rgb, stain, volumetric, time_series


We also provide instantiations of common slide types for convenience:

    =============================  =======  ========   ======= =======  ==========  ===========
    Type                           stain    platform   rgb     tma      volumetric  time_series
    =============================  =======  ========   ======= =======  ==========  ===========
    ``pathml.core.types.HE``       'HE'     None       True    False    False       False
    ``pathml.core.types.IHC``      'IHC'    None       True    False    False       False
    ``pathml.core.types.IF``       'Fluor'  None       False   False    False       False
    ``pathml.core.types.CODX``     'Fluor'  'CODEX'    False   False    False       False
    ``pathml.core.types.Vectra``   'Fluor'  'Vectra'   False   False    False       False
    =============================  =======  ========   ======= =======  ==========  ===========

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

h5pathManager
-------------

.. autoclass:: pathml.core.h5managers.h5pathManager
