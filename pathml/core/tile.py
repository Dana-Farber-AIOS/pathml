import numpy as np
from collections import OrderedDict

import pathml.core.masks


class Tile:
    """
    Object for representing a tile region extracted from an image. Holds the image for the tile, as well as the (i,j)
    coordinates of the top-left corner of the tile in the original image. The (i,j) coordinate system is based
    on labelling the top-leftmost pixel as (0, 0)

    Args:
        image (np.ndarray): image
    name (str): name of tile
    masks (pathml.core.Masks): corresponding masks for region
    coords (tuple): Coordinates of tile relative to maximum resolution of whole-slide image.
        The (i,j) coordinate system is based on labelling the top-leftmost pixel as (0, 0)
    slidetype: type of image (e.g. pathml.HESlide). Defaults to None.
    labels: labels for the tile
    """
    def __init__(self, image, name=None, coords=None, slidetype=None, masks=None, labels=None):
        assert isinstance(image, np.ndarray), f"image of type {type(image)} must be a np.ndarray"
        assert masks is None or isinstance(masks, (pathml.core.masks.Masks, dict)), \
            f"masks is of type {type(masks)} but must be of type pathml.core.masks.Masks or dict"
        assert isinstance(coords, tuple), "coords must be a tuple e.g. (i, j)"
        # labels are dicts of strings or None
        assert labels is None or isinstance(labels, dict), f"labels is of type {type(labels)} but must be of type str or None"
        assert name is None or isinstance(name, str), f"name is of type {type(name)} but must be of type str or None"
        self.image = image
        if isinstance(masks, pathml.core.masks.Masks):
            # move masks to dict so that Tile is in memory (must pass to dask client) 
            maskdict = OrderedDict()
            for mask in masks.h5manager.h5.keys():
                maskdict[mask] = masks[mask] 
            self.masks = maskdict
        if isinstance(masks, dict):
            for val in masks.values():
                if val.shape[:2] != self.image.shape[:2]:
                    raise ValueError(f"mask is of shape {val.shape} but must match tile shape {self.image.shape}")
            self.masks = masks 
        elif masks is None:
            self.masks = OrderedDict() 
        self.name = name
        self.coords = coords
        self.slidetype = slidetype
        self.labels = labels

    def __repr__(self):
        out = f"Tile(image shape {self.image.shape}, slidetype={self.slidetype}, " \
              f"masks={repr(self.masks) if self.masks is not None else None}, " \
              f"coords={self.coords}, " \
              f"labels={list(self.labels.keys()) if self.labels is not None else None})"
        return out
