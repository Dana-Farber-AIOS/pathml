import numpy as np
import os
import cv2

from pathml.preprocessing.base import BasePreprocessor, BaseTileExtractor
from pathml.preprocessing.masks import Masks


class Tile:
    """
    Object for representing a tile extracted from an image. Holds the image for the tile, as well as the (i,j)
    coordinates of the top-left corner of the tile in the original image. The (i,j) coordinate system is based
    on labelling the top-leftmost pixel as (0, 0)

    :param array: image for tile
    :type array: np.ndarray
    :param masks: masks for tile
    :type masks: dict
    :param i: vertical coordinate of top-left corner of tile, in original image
    :type i: int
    :param j: horizontal coordinate of top-left corner of tile, in original image
    :type j: int
    """
    def __init__(self, array, masks=None, i=None, j=None):
        assert isinstance(array, np.ndarray), "Array must be a np.ndarray"
        self.array = array
        self.i = i  # i coordinate of top left corner pixel
        self.j = j  # j coordinate of top left corner pixel
        if masks: 
            for val in masks.values():
                if val.shape != self.array.shape[:2]:
                    raise ValueError(f"mask is of shape {val.shape} but must match tile shape {self.array.shape}")
            self.masks = Masks(masks)
        elif masks == None:
            self.masks = masks 

    def __repr__(self):  # pragma: no cover
        return f"Tile(array shape {self.array.shape}, " \
               f"i={self.i if self.i else 'None'}, " \
               f"j={self.j if self.j else 'None'})"

    def save(self, out_dir, filename):
        """
        Save tile to disk as jpeg file.

        :param out_dir: directory to write to.
        :type out_dir: str
        :param filename: File name of saved tile.
        :type filename: str
        """
        # create output directory if it doesn't already exist
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        # TODO: don't assume jpeg. What's the most flexible file format that can handle n channels? Maybe tif?
        # tile_path = os.path.join(out_dir, f"{self.wsi.name}_{self.i}_{self.j}.jpeg")
        tile_path = os.path.join(out_dir, filename)
        cv2.imwrite(tile_path, self.array)


def extract_tiles_array(im, tile_size, stride=None):
    """
    Extract tiles from an array. Allows user to specify stride between tiles.
    Based on original implementation in ``sklearn.feature_extraction.image_ref._extract_patches``

    :param im: input image
    :type im: np.ndarray
    :param tile_size: Dimension of extracted tiles
    :type tile_size: int
    :param stride: Stride length between tiles. If ``None``, uses ``stride = tile_size`` for non-overlapping tiles.
        Defaults to None
    :type stride: int, optional
    :return: Array of extracted tiles of shape `(M, N, tile_size, tile_size, n_channels)`
    :rtype: np.ndarray
    """
    assert im.ndim == 3, f"Number of input dimensions {im.ndim} must be 3"
    if stride is None:
        stride = tile_size
    n_channels = im.shape[2]
    patch_strides = im.strides
    patch_shape = (tile_size, tile_size, n_channels)
    extraction_step = (stride, stride, 1)
    slices = tuple(slice(None, None, st) for st in extraction_step)
    indexing_strides = im[slices].strides
    patch_indices_shape = ((np.array(im.shape) - np.array(patch_shape)) //
                           np.array(extraction_step)) + 1
    shape = tuple(list(patch_indices_shape) + list(patch_shape))
    strides = tuple(list(indexing_strides) + list(patch_strides))
    tiles = np.lib.stride_tricks.as_strided(im, shape = shape, strides = strides)
    # squeeze out unnecessary axis
    tiles = tiles.squeeze(axis = 2)
    return tiles


def extract_tiles(im, tile_size, stride=None):
    """
    Extract tiles from an array. Allows user to specify stride between tiles.

    :param im: input image
    :type im: np.ndarray
    :param tile_size: Dimension of extracted tiles
    :type tile_size: int
    :param stride: Stride length between tiles. If ``None``, uses ``stride = tile_size`` for non-overlapping tiles.
        Defaults to ``None``
    :type stride: int, optional
    :return: list of :class:`~pathml.preprocessing.tiling.Tile` objects
    :rtype: list
    """
    if stride is None:
        stride = tile_size
    tiles_array = extract_tiles_array(im, tile_size, stride)
    tiles = []
    n_tiles_i, n_tiles_j = tiles_array.shape[0:2]
    for i_ix in range(n_tiles_i):
        for j_ix in range(n_tiles_j):
            i_pixel, j_pixel = i_ix * stride, j_ix * stride
            tile = Tile(array = tiles_array[i_ix, j_ix, ...], i = i_pixel, j = j_pixel)
            tiles.append(tile)
    return tiles


def extract_tiles_with_mask(im, tile_size, stride=None, mask=None, mask_thresholds=0.5):
    """
    Generate tiles. Specify arbitrary number of binary masks and only keep tiles that overlap with the
    masks above some threshold.

    :param im: input image `(M x N x n_channels)`
    :type im: np.ndarray
    :param tile_size: Dimensions of output tiles. Output tiles are square.
    :type tile_size: int
    :param stride: Stride length between tiles. If None, uses ``stride = tile_size`` for non-overlapping tiles.
        Defaults to None
    :type stride: int, optional
    :param mask: binary masks `(M x N x n_masks)`
    :type mask: np.ndarray
    :param mask_thresholds: Proportion of tile pixels which must be True in corresponding mask.
        If a single float, applies same threshold for all masks. If a list of floats same length as `n_masks` (i.e.
        number of channels in ``mask`` argument), applies
        threshold individually for each corresponding mask. Must be between 0 and 1. Defaults to 0.5
    :type mask_thresholds: float or list of floats, optional
    :return: list of :class:`~pathml.preprocessing.tiling.Tile` objects satisfying mask threshold criteria
    """
    # check inputs here
    assert isinstance(mask, np.ndarray), f"Input mask type {type(im)} must be a numpy array"
    assert isinstance(im, np.ndarray), f"input image_ref type {type(im)} must be a numpy array"
    assert im.shape[0:2] == mask.shape[0:2], f"Dimensions of input image_ref {im.shape} and mask {mask.shape} must match"
    # if mask is 2D array, add a depth axis
    if mask.ndim == 2:
        mask = np.expand_dims(mask, axis = 2)
    if isinstance(mask_thresholds, float):
        mask_thresholds = np.repeat(mask_thresholds, repeats = mask.shape[2])
    else:
        assert len(mask_thresholds) == mask.shape[2], \
            f"number of mask_thresholds ({len(mask_thresholds)}) ≠ number of masks ({mask.shape[2]}) and is not a float"
        mask_thresholds = np.array(mask_thresholds)
    assert all([1.0 >= thresh >= 0.0 for thresh in mask_thresholds]), "all thresholds must be between 0 and 1"

    tiles = extract_tiles(im, tile_size, stride)
    good_tiles = []
    for tile in tiles:
        # check overlap
        mask_slice = mask[tile.i:tile.i + tile_size, tile.j:tile.j + tile_size, ...]
        mask_means = np.mean(mask_slice, axis = (0, 1))
        if np.greater_equal(mask_means, mask_thresholds).all():
            good_tiles.append(tile)

    return good_tiles


class SimpleTileExtractor(BaseTileExtractor):
    """
    A basic :class:`~pathml.preprocessing.base_preprocessor.BaseTileExtractor` which extracts tiles from
    :attr:`~pathml.preprocessing.slide_data.SlideData.image`
    using :func:`~pathml.preprocessing.tiling.extract_tiles_with_mask`
    and populates
    :attr:`~pathml.preprocessing.slide_data.SlideData.tiles`

    :param tile_size: dimensions of tiles. Defaults to 500.
    :param stride: stride between tiles. If ``None``, uses ``stride = tile_size`` for non-overlapping tiles.
        Defaults to ``None``
    :param mask_ix: Indices of masks to use for tiling. If ``None``, uses all masks. Defaults to ``None``.
    :type mask_ix: list of ints, optional
    :param mask_thresholds: Proportion of tile pixels which must be ``True`` in corresponding mask.
        If a single float, applies same threshold for all masks. If a list of floats same length as ``masks``, applies
        threshold individually for each corresponding mask. Must be between 0 and 1. Defaults to 0.5
    :type mask_thresholds: float or list of floats. Optional.
    """
    def __init__(self, tile_size=500, stride=None, mask_ix=None, mask_thresholds=0.5):
        self.tile_size = tile_size
        self.stride = stride
        self.mask_ix = np.array(mask_ix)
        self.mask_thresholds = mask_thresholds

    def apply(self, data):
        """
        Applies tile extraction to input data.

        :param data: Input data object.
        :type data: :class:`~pathml.preprocessing.slide_data.SlideData`
        """
        im = data.image
        mask_thresh = self.mask_thresholds
        if self.mask_ix:
            mask = data.mask[:, :, self.mask_ix]
            if isinstance(self.mask_thresholds, float):
                mask_thresh = self.mask_thresholds
            elif isinstance(self.mask_thresholds, list):
                mask_thresh = self.mask_thresholds[self.mask_ix]
                assert isinstance(mask_thresh, float), \
                    f"Mask threshold at index {self.mask_ix} is of type {type(mask_thresh)} but must be a float"
            else:
                Exception(f"Mask_thresholds of type {type(self.mask_thresholds)} must be a float or list of floats")
        else:
            mask = data.masks
        data.tiles = extract_tiles_with_mask(
            im = im,
            tile_size = self.tile_size,
            stride = self.stride,
            mask = mask,
            mask_thresholds = mask_thresh
        )
        return data
