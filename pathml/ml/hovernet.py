"""
Copyright 2021, Dana-Farber Cancer Institute and Weill Cornell Medicine
License: GNU GPL 2.0
"""

import warnings

from pathml.ml.models.hovernet import (
    HoVerNet,
    _BatchNormRelu,
    _ce_loss_nc_head,
    _ce_loss_np_head,
    _convert_multiclass_mask_to_binary,
    _dice_loss_nc_head,
    _dice_loss_np_head,
    _get_gradient_hv,
    _HoverNetDecoder,
    _HoVerNetDenseUnit,
    _HoVerNetEncoder,
    _HoVerNetResidualUnit,
    _loss_hv_grad,
    _loss_hv_mse,
    _make_HoVerNet_dense_block,
    _make_HoVerNet_residual_block,
    _post_process_single_hovernet,
    _vis_outputs_single,
    compute_hv_map,
    extract_nuclei_info,
    group_centroids_by_type,
    loss_hovernet,
    post_process_batch_hovernet,
    remove_small_objs,
)

# Issue a deprecation warning when someone imports from this old file
warnings.warn(
    "Importing from 'pathml.ml.hovernet' is deprecated and will be removed in a future version. "
    "Please use 'pathml.ml.models.hovernet' instead.",
    DeprecationWarning,
    stacklevel=2,
)

# re-export the imported items so that the old import paths still work
__all__ = [
    "_BatchNormRelu",
    "_HoVerNetResidualUnit",
    "_make_HoVerNet_residual_block",
    "_HoVerNetEncoder",
    "_HoVerNetDenseUnit",
    "_make_HoVerNet_dense_block",
    "_HoverNetDecoder",
    "HoVerNet",
    "_convert_multiclass_mask_to_binary",
    "_dice_loss_np_head",
    "_dice_loss_nc_head",
    "_ce_loss_nc_head",
    "_ce_loss_np_head",
    "compute_hv_map",
    "_get_gradient_hv",
    "_loss_hv_grad",
    "_loss_hv_mse",
    "loss_hovernet",
    "remove_small_objs",
    "_post_process_single_hovernet",
    "post_process_batch_hovernet",
    "_vis_outputs_single",
    "extract_nuclei_info",
    "group_centroids_by_type",
]
