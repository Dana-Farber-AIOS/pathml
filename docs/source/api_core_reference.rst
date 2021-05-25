Core API
========

SlideData
---------

The central class in ``PathML`` for representing a whole-slide image.

.. autoclass:: pathml.core.SlideData


HESlide
^^^^^^^

.. autoclass:: pathml.core.HESlide


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
    ``pathml.core.types.HE_TMA``   'HE'     None       True    True     False       False
    ``pathml.core.types.IHC_TMA``  'IHC'    None       True    True     False       False
    ``pathml.core.types.IF_TMA``   'Fluor'  None       False   True     False       False
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
