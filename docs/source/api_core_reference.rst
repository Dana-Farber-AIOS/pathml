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
    :exclude-members: tma, rgb, stain, volumetric, time_series


We also provide instantiations of common slide types for convenience:

    =============================  =======  ======= =======  ==========  ===========
    Type                           stain    rgb     tma      volumetric  time_series
    =============================  =======  ======= =======  ==========  ===========
    ``pathml.core.types.HE``       'HE'     True    False    False       False
    ``pathml.core.types.IHC``      'IHC'    True    False    False       False
    ``pathml.core.types.IF``       'Fluor'  False   False    False       False
    ``pathml.core.types.HE_TMA``   'HE'     True    True     False       False
    ``pathml.core.types.IHC_TMA``  'IHC'    True    True     False       False
    ``pathml.core.types.IF_TMA``   'Fluor'  False   True     False       False
    =============================  =======  ======= =======  ==========  ===========

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
