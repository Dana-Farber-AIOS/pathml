"""
Copyright 2021, Dana-Farber Cancer Institute and Weill Cornell Medicine
License: GNU GPL 2.0
"""

import numpy as np
from loguru import logger
from pathml._logging import *

@logger_wraps()
def extract_tiles(arr, tile_size, stride=None):
    """
    Extract tiles from an array. Allows user to specify stride between tiles.
    Based on original implementation in ``sklearn.feature_extraction.image_ref._extract_patches``
    Incomplete tiles on the edge are dropped (TO DO: fix this (zero padding?))

    Args:
        arr (np.ndarray): input array. Must be 3 dimensional (H, W, n_channels)
        tile_size (int): Dimension of extracted tiles. Each tile will be shape (tile_size, tile_size, n_channels)
        stride (int, optional): Stride length between tiles.
            If ``None``, uses ``stride = tile_size`` for non-overlapping tiles. Defaults to None

    Returns:
        np.ndarray: Array of extracted tiles of shape `(n_tiles, tile_size, tile_size, n_channels)`
    """
    assert arr.ndim == 3, f"Number of input dimensions {arr.ndim} must be 3"
    if stride is None:
        stride = tile_size
    i, j, n_channels = arr.shape
    if (i - tile_size) % stride != 0 or (j - tile_size) % stride != 0:
        raise NotImplementedError(
                logger.exception(f"Array of shape {arr.shape} is not perfectly tiled by tiles of size {tile_size} and stride {stride}.")
        )
    patch_strides = arr.strides
    patch_shape = (tile_size, tile_size, n_channels)
    extraction_step = (stride, stride, 1)
    slices = tuple(slice(None, None, st) for st in extraction_step)
    indexing_strides = arr[slices].strides
    patch_indices_shape = (
        (np.array(arr.shape) - np.array(patch_shape)) // np.array(extraction_step)
    ) + 1
    shape = tuple(list(patch_indices_shape) + list(patch_shape))
    strides = tuple(list(indexing_strides) + list(patch_strides))
    tiles = np.lib.stride_tricks.as_strided(arr, shape=shape, strides=strides)
    # squeeze out unnecessary axis
    tiles = tiles.squeeze(axis=2)
    tiles = tiles.reshape(-1, *tiles.shape[2:])
    return tiles

@logger_wraps()
def extract_tiles_with_mask(arr, mask, tile_size, stride=None, threshold=0.5):
    """
    Generate tiles and only keep tiles that overlap with the
    masks above some threshold.

    Args:
        arr (np.ndarray): input array. Must be 3 dimensional (H, W, n_channels)
        mask (np.ndarray): array of masks. Must be 3 dimensional (H, W, n_masks).
        tile_size (int): Dimension of extracted tiles. Each tile will be shape (tile_size, tile_size, n_channels)
        stride (int, optional): Stride length between tiles.
            If ``None``, uses ``stride = tile_size`` for non-overlapping tiles. Defaults to None
        threshold (float): for each tile, all values in the corresponding mask region will be averaged.
            `threshold` is the cutoff value, above which the tile will be kept and below which the tile
            will be discarded. Defaults to 0.5

    Returns:
        np.ndarray: Array of extracted tiles of shape `(n_tiles, tile_size, tile_size, n_channels)`
    """
    # check inputs here
    assert isinstance(
        mask, np.ndarray
    ), f"Input mask type {type(mask)} must be a numpy array"
    assert isinstance(
        arr, np.ndarray
    ), f"input array type {type(arr)} must be a numpy array"
    assert arr.ndim == 3, f"array of shape {arr.shape} must be 3 dimensional"
    assert mask.ndim == 3, f"mask of shape {mask.shape} must be 3 dimensional"
    assert (
        arr.shape[0:2] == mask.shape[0:2]
    ), f"Dims of input image_ref {arr.shape} and mask {mask.shape} must match"

    arr_tiles = extract_tiles(arr, tile_size=tile_size, stride=stride)
    mask_tiles = extract_tiles(mask, tile_size=tile_size, stride=stride)

    tile_mask_means = mask_tiles.mean(axis=tuple(range(1, mask_tiles.ndim)))

    out = arr_tiles[tile_mask_means >= threshold, ...]
    return out


# TODO: do we need to have coords for each tile?
