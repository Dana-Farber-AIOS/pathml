Preprocessing
=============

The ``pathml.preprocessing`` module provides tools for defining preprocessing pipelines for whole-slide images.
The general workflow is:

1. Load whole-slide image from disk
2. Perform slide-level preprocessing
3. Extract tiles
4. Perform tile-level preprocessing
5. Save tiles to disk

Preprocessing pipelines are defined using the :class:`~pathml.preprocessing.pipeline.Pipeline` class, and
are composed of :ref:`transforms-label`. We provide a set of pre-built Transforms as well as tools
to build custom-made Transforms to suit the needs of specific projects.

Pipelines operate by modifying the state of :class:`~pathml.preprocessing.slide_data.SlideData` objects.


Loading Images
--------------
.. autoclass:: pathml.preprocessing.wsi.HESlide
    :members:
.. autoclass:: pathml.preprocessing.multiparametricslide.MultiparametricSlide
    :members:

Tiling Images
-------------
.. automodule:: pathml.preprocessing.tiling
    :members:

Preprocessing Pipeline
----------------------
.. autoclass:: pathml.preprocessing.pipeline.Pipeline
    :members:

.. autoclass:: pathml.preprocessing.slide_data.SlideData
    :members:

.. _transforms-label:

Transforms
----------
General-Purpose Transforms
^^^^^^^^^^^^^^^^^^^^^^^^^^
.. automodule:: pathml.preprocessing.transforms
    :members:

H&E-Specific Transforms
^^^^^^^^^^^^^^^^^^^^^^^
.. automodule:: pathml.preprocessing.transforms_HandE
    :members:


Miscellaneous
^^^^^^^^^^^^^

H&E Colors and Stains
---------------------
.. automodule:: pathml.preprocessing.stains
    :members:



Utilities
---------

Image Utilities
^^^^^^^^^^^^^^^
.. autofunction:: pathml.preprocessing.utils.pil_to_rgb
.. autofunction:: pathml.preprocessing.utils.segmentation_lines
.. autofunction:: pathml.preprocessing.utils.plot_mask
.. autofunction:: pathml.preprocessing.utils.contour_centroid
.. autofunction:: pathml.preprocessing.utils.sort_points_clockwise

Color Utilities
^^^^^^^^^^^^^^^
.. autofunction:: pathml.preprocessing.utils.RGB_to_HSI
.. autofunction:: pathml.preprocessing.utils.RGB_to_OD
.. autofunction:: pathml.preprocessing.utils.RGB_to_HSV
.. autofunction:: pathml.preprocessing.utils.RGB_to_LAB
.. autofunction:: pathml.preprocessing.utils.RGB_to_GREY


General Utilities
^^^^^^^^^^^^^^^^^
.. autofunction:: pathml.preprocessing.utils.upsample_array
.. autofunction:: pathml.preprocessing.utils.pad_or_crop
.. autofunction:: pathml.preprocessing.utils.normalize_matrix_rows
.. autofunction:: pathml.preprocessing.utils.normalize_matrix_cols


Abstract Classes
----------------

These classes are not designed to be instantiated.
They are inherited by other classes, and are used when implementing new classes.
The features in this section are meant for developers and advanced users!

Base Classes for Loading Slides
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: pathml.preprocessing.wsi.BaseSlide
    :members:

Base Classes for Preprocessing Pipelines
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. automodule:: pathml.preprocessing.base_preprocessor
    :members:

Base Classes for Transforms
^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. automodule:: pathml.preprocessing.base_transforms
    :members:

