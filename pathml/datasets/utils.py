"""
Copyright 2021, Dana-Farber Cancer Institute and Weill Cornell Medicine
License: GNU GPL 2.0
"""

import numpy as np
from loguru import logger


def pannuke_multiclass_mask_to_nucleus_mask(multiclass_mask):
    """
    Convert multiclass mask from PanNuke to a single channel nucleus mask.
    Assumes each pixel is assigned to one and only one class. Sums across channels, except the last mask channel
    which indicates background pixels in PanNuke.
    Operates on a single mask.

    Args:
        multiclass_mask (torch.Tensor): Mask from PanNuke, in classification setting. (i.e. ``nucleus_type_labels=True``).
            Tensor of shape (6, 256, 256).

    Returns:
        Tensor of shape (256, 256).
    """
    # verify shape of input
    assert (
        multiclass_mask.ndim == 3 and multiclass_mask.shape[0] == 6
    ), f"Expecting a mask with dims (6, 256, 256). Got input of shape {multiclass_mask.shape}"
    assert (
        multiclass_mask.shape[1] == 256 and multiclass_mask.shape[2] == 256
    ), f"Expecting a mask with dims (6, 256, 256). Got input of shape {multiclass_mask.shape}"
    # ignore last channel
    out = np.sum(multiclass_mask[:-1, :, :], axis=0)
    return out
