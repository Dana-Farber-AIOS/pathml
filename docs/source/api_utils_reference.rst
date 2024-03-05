Utilities API
=============

Documentation for various utilities from all modules.

Logging Utils
-------------

.. autoapiclass:: pathml.PathMLLogger

Core Utils
----------

.. autoapifunction:: pathml.core.utils.readtupleh5
.. autoapifunction:: pathml.core.utils.writedataframeh5
.. autoapifunction:: pathml.core.utils.writedicth5
.. autoapifunction:: pathml.core.utils.writestringh5
.. autoapifunction:: pathml.core.utils.writetupleh5
.. autoapifunction:: pathml.core.utils.readcounts
.. autoapifunction:: pathml.core.utils.writecounts

Graph Utils
--------------

.. autoapifunction:: pathml.graph.utils.Graph
.. autoapifunction:: pathml.graph.utils.HACTPairData
.. autoapifunction:: pathml.graph.utils.get_full_instance_map
.. autoapifunction:: pathml.graph.utils.build_assignment_matrix
.. autoapifunction:: pathml.graph.utils.two_hop
.. autoapifunction:: pathml.graph.utils.two_hop_no_sparse

Datasets Utils
--------------

.. autoapiclass:: pathml.datasets.utils.DeepPatchFeatureExtractor
.. autoapifunction:: pathml.datasets.utils.pannuke_multiclass_mask_to_nucleus_mask
.. autoapifunction:: pathml.datasets.utils._remove_modules

ML Utils
--------

.. autoapifunction:: pathml.ml.utils.center_crop_im_batch
.. autoapifunction:: pathml.ml.utils.dice_loss
.. autoapifunction:: pathml.ml.utils.dice_score
.. autoapifunction:: pathml.ml.utils.get_sobel_kernels
.. autoapifunction:: pathml.ml.utils.wrap_transform_multichannel
.. autoapifunction:: pathml.ml.utils.scatter_sum
.. autoapifunction:: pathml.ml.utils.broadcast
.. autoapifunction:: pathml.ml.utils.get_degree_histogram
.. autoapifunction:: pathml.ml.utils.get_class_weights

Miscellaneous Utils
-------------------

.. autoapifunction:: pathml.utils.upsample_array
.. autoapifunction:: pathml.utils.pil_to_rgb
.. autoapifunction:: pathml.utils.segmentation_lines
.. autoapifunction:: pathml.utils.plot_mask
.. autoapifunction:: pathml.utils.contour_centroid
.. autoapifunction:: pathml.utils.sort_points_clockwise
.. autoapifunction:: pathml.utils.pad_or_crop
.. autoapifunction:: pathml.utils.RGB_to_HSI
.. autoapifunction:: pathml.utils.RGB_to_OD
.. autoapifunction:: pathml.utils.RGB_to_HSV
.. autoapifunction:: pathml.utils.RGB_to_LAB
.. autoapifunction:: pathml.utils.RGB_to_GREY
.. autoapifunction:: pathml.utils.normalize_matrix_rows
.. autoapifunction:: pathml.utils.normalize_matrix_cols
.. autoapifunction:: pathml.utils.plot_segmentation
