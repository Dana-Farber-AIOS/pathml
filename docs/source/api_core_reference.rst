Core API
========

SlideData
---------

The central class in ``PathML`` for representing a whole-slide image.

.. autoapiclass:: pathml.core.SlideData


Convenience SlideData Classes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoapiclass:: pathml.core.HESlide
.. autoapiclass:: pathml.core.VectraSlide
.. autoapiclass:: pathml.core.MultiparametricSlide
.. autoapiclass:: pathml.core.CODEXSlide


Slide Types
-----------

.. autoapiclass:: pathml.core.SlideType
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

.. autoapiclass:: pathml.core.Tile

SlideDataset
------------

.. autoapiclass:: pathml.core.SlideDataset

Tiles and Masks helper classes
------------------------------

.. autoapiclass:: pathml.core.Tiles

.. autoapiclass:: pathml.core.Masks


Slide Backends
--------------

OpenslideBackend
^^^^^^^^^^^^^^^^

.. autoapiclass:: pathml.core.OpenSlideBackend

BioFormatsBackend
^^^^^^^^^^^^^^^^^

.. autoapiclass:: pathml.core.BioFormatsBackend

DICOMBackend
^^^^^^^^^^^^

.. autoapiclass:: pathml.core.DICOMBackend

h5pathManager
-------------

.. autoapiclass:: pathml.core.h5managers.h5pathManager
