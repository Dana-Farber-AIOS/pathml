# Utilities for ML module
import torch
from torch.nn import functional as F
import numpy as np


def center_crop_im_batch(batch, dims, batch_order = "BCHW"):
    """
    Center crop images in a batch.

    Args:
        batch: The batch of images to be cropped
        dims: Amount to be cropped (tuple for H, W)
    """
    assert batch.ndim == 4, f"ERROR input shape is {batch.shape} - expecting a batch with 4 dimensions total"
    assert len(dims) == 2, f"ERROR input cropping dims is {dims} - expecting a tuple with 2 elements total"
    assert batch_order in {"BHCW", "BCHW"}, \
        f"ERROR input batch order {batch_order} not recognized. Must be one of 'BHCW' or 'BCHW'"

    if dims == (0, 0):
        # no cropping necessary in this case
        batch_cropped = batch
    else:
        crop_t = dims[0] // 2
        crop_b = dims[0] - crop_t
        crop_l = dims[1] // 2
        crop_r = dims[1] - crop_l

        if batch_order == "BHWC":
            batch_cropped = batch[:, crop_t:-crop_b, crop_l:-crop_r, :]
        elif batch_order == "BCHW":
            batch_cropped = batch[:, :, crop_t:-crop_b, crop_l:-crop_r]
        else:
            raise Exception("Input batch order not valid")

    return batch_cropped


def dice_loss(true, logits, eps=1e-3):
    """
    Computes the Sørensen–Dice loss.
    Note that PyTorch optimizers minimize a loss. In this
    case, we would like to maximize the dice loss so we
    return 1 - dice loss.
    From: https://github.com/kevinzakka/pytorch-goodies/blob/c039691f349be9f21527bb38b907a940bfc5e8f3/losses.py#L54

    Args:
        true: a tensor of shape [B, 1, H, W].
        logits: a tensor of shape [B, C, H, W]. Corresponds to
            the raw output or logits of the model.
        eps: added to the denominator for numerical stability.

    Returns:
        dice_loss: the Sørensen–Dice loss.
    """
    assert true.dtype == torch.long, f"Input 'true' is of type {true.type}. It should be a long."
    num_classes = logits.shape[1]
    if num_classes == 1:
        true_1_hot = torch.eye(num_classes + 1)[true.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        true_1_hot_f = true_1_hot[:, 0:1, :, :]
        true_1_hot_s = true_1_hot[:, 1:2, :, :]
        true_1_hot = torch.cat([true_1_hot_s, true_1_hot_f], dim=1)
        pos_prob = torch.sigmoid(logits)
        neg_prob = 1 - pos_prob
        probas = torch.cat([pos_prob, neg_prob], dim=1)
    else:
        true_1_hot = torch.eye(num_classes)[true.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        probas = F.softmax(logits, dim=1)
    true_1_hot = true_1_hot.type(logits.type())
    dims = (0,) + tuple(range(2, true.ndimension()))
    intersection = torch.sum(probas * true_1_hot, dims)
    cardinality = torch.sum(probas + true_1_hot, dims)
    loss = (2. * intersection / (cardinality + eps)).mean()
    loss = 1 - loss
    return loss


def dice_score(pred, truth, eps=1e-3):
    """
    Calculate dice score for two tensors of the same shape.
    If tensors are not already binary, they are converted to bool by zero/non-zero.
    
    Args:
        pred (np.ndarray): Predictions
        truth (np.ndarray): ground truth
        
    Returns:
        float: Dice score
    """    
    assert isinstance(truth, np.ndarray) and isinstance(pred, np.ndarray), "inputs must be torch.Tensor or np.ndarray"
    assert pred.shape == truth.shape
    # turn into binary if not already
    pred = pred != 0
    truth = truth != 0
    
    num = 2 * np.sum(pred.flatten() * truth.flatten())
    denom = np.sum(pred) + np.sum(truth) + eps
    return float(num / denom)


def get_sobel_kernels(size, dt=torch.float32):
    """
    Create horizontal and vertical Sobel kernels for approximating gradients
    Returned kernels will be of shape (size, size)
    """
    assert size % 2 == 1, "Size must be odd"

    h_range = torch.arange(-size // 2 + 1, size // 2 + 1, dtype = dt)
    v_range = torch.arange(-size // 2 + 1, size // 2 + 1, dtype = dt)
    h, v = torch.meshgrid([h_range, v_range])
    h, v = h.transpose(0, 1), v.transpose(0, 1)

    kernel_h = h / (h * h + v * v + 1e-5)
    kernel_v = v / (h * h + v * v + 1e-5)

    kernel_h = kernel_h.type(dt)
    kernel_v = kernel_v.type(dt)

    return kernel_h, kernel_v


def wrap_transform_multichannel(transform):
    """
    Wrapper to make albumentations transform compatible with a multichannel mask.
    Channel should be in first dimension, i.e. (n_mask_channels, H, W)
    
    Args:
        transform: Albumentations transform. Must have 'additional_targets' parameter specified with 
            a total of `n_channels` key,value pairs. All values must be 'mask' but the keys don't matter.
            e.g. for a mask with 3 channels, you could use: 
                `additional targets = {'mask1' : 'mask', 'mask2' : 'mask', 'pathml' : 'mask'}`
    
    Returns:
        function that can be called with a multichannel mask argument
    """
    targets = transform.additional_targets    
    n_targets = len(targets)
    
    # make sure that everything is correct so that transform is correctly applied
    assert all([v == "mask" for v in targets.values()]), \
        f"error all values in transform.additional_targets must be 'mask'." 
    
    def transform_out(*args, **kwargs):
        mask = kwargs.pop("mask")
        assert mask.ndim == 3, f"input mask shape {mask.shape} must be 3-dimensions ()"
        assert mask.shape[0] == n_targets, \
            f"input mask shape {mask.shape} doesn't match additional_targets {transform.additional_targets}"
        
        mask_to_dict = {key : mask[i, :, :] for i, key in enumerate(targets.keys())}
        kwargs.update(mask_to_dict)
        out = transform(*args, **kwargs)
        mask_out = np.stack([out.pop(key) for key in targets.keys()], axis=0)
        assert mask_out.shape == mask.shape
        out["mask"] = mask_out
        return out
    
    return transform_out