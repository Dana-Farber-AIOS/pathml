Utilities API
=============

Documentation for various utilities from all modules.

Core Utils
----------

.. autofunction:: pathml.core.utils.readtilesdicth5
.. autofunction:: pathml.core.utils.readtupleh5
.. autofunction:: pathml.core.utils.writedataframeh5
.. autofunction:: pathml.core.utils.writedicth5
.. autofunction:: pathml.core.utils.writestringh5
.. autofunction:: pathml.core.utils.writetilesdicth5
.. autofunction:: pathml.core.utils.writetupleh5

Datasets Utils
--------------

.. autofunction:: pathml.datasets.utils.download_from_url
.. autofunction:: pathml.datasets.utils.pannuke_multiclass_mask_to_nucleus_mask
.. autofunction:: pathml.datasets.utils.parse_file_size

ML Utils
--------

.. autofunction:: pathml.ml.utils.center_crop_im_batch
.. autofunction:: pathml.ml.utils.dice_loss
.. autofunction:: pathml.ml.utils.dice_score
.. autofunction:: pathml.ml.utils.get_sobel_kernels
.. autofunction:: pathml.ml.utils.wrap_transform_multichannel

Miscellaneous Utils
-------------------

.. autofunction:: pathml.utils.upsample_array
.. autofunction:: pathml.utils.pil_to_rgb
.. autofunction:: pathml.utils.segmentation_lines
.. autofunction:: pathml.utils.plot_mask
.. autofunction:: pathml.utils.contour_centroid
.. autofunction:: pathml.utils.sort_points_clockwise
.. autofunction:: pathml.utils.pad_or_crop
.. autofunction:: pathml.utils.RGB_to_HSI
.. autofunction:: pathml.utils.RGB_to_OD
.. autofunction:: pathml.utils.RGB_to_HSV
.. autofunction:: pathml.utils.RGB_to_LAB
.. autofunction:: pathml.utils.RGB_to_GREY
.. autofunction:: pathml.utils.normalize_matrix_rows
.. autofunction:: pathml.utils.normalize_matrix_cols
.. autofunction:: pathml.utils.label_artifact_tile_HE
.. autofunction:: pathml.utils.label_whitespace_HE
.. autofunction:: pathml.utils.plot_segmentation
