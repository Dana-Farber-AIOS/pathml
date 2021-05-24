"""
Copyright 2021, Dana-Farber Cancer Institute and Weill Cornell Medicine
License: GNU GPL 2.0
"""

import numpy as np
from collections import OrderedDict
import matplotlib.pyplot as plt
import h5py

import pathml.core.masks


class Tile:
    """
    Object representing a tile extracted from an image. Holds the array for the tile, as well as the (i,j)
    coordinates of the top-left corner of the tile in the original image. The (i,j) coordinate system is based
    on labelling the top-leftmost pixel as (0, 0)

    Args:
        image (np.ndarray): Image array of tile
        coords (tuple): Coordinates of tile relative to the whole-slide image.
            The (i,j) coordinate system is based on labelling the top-leftmost pixel of the WSI as (0, 0).
        name (str, optional): Name of tile
        masks (dict or pathml.core.Masks): masks belonging to tile. If masks are supplied, all masks must be the
            same shape as the tile.
        labels: labels belonging to tile
        slide_type (pathml.core.SlideType, optional): slide type specification. Must be a
            :class:`~pathml.core.SlideType` object. Alternatively, slide type can be specified by using the
            parameters ``stain``, ``tma``, ``rgb``, ``volumetric``, and ``time_series``.
        stain (str, optional): Flag indicating type of slide stain. Must be one of [‘HE’, ‘IHC’, ‘Fluor’].
            Defaults to ``None``. Ignored if ``slide_type`` is specified.
        tma (bool, optional): Flag indicating whether the image is a tissue microarray (TMA).
            Defaults to ``False``. Ignored if ``slide_type`` is specified.
        rgb (bool, optional): Flag indicating whether the image is in RGB color.
            Defaults to ``None``. Ignored if ``slide_type`` is specified.
        volumetric (bool, optional): Flag indicating whether the image is volumetric.
            Defaults to ``None``. Ignored if ``slide_type`` is specified.
        time_series (bool, optional): Flag indicating whether the image is a time series.
            Defaults to ``None``. Ignored if ``slide_type`` is specified.
    """
    def __init__(self, image, coords, name=None, masks=None, labels=None, slide_type=None, stain=None,
                 tma=None, rgb=None, volumetric=None, time_series=None):
        # check inputs
        assert isinstance(image, np.ndarray), f"image of type {type(image)} must be a np.ndarray"
        assert masks is None or isinstance(masks, (pathml.core.masks.Masks, dict)), \
            f"masks is of type {type(masks)} but must be of type pathml.core.masks.Masks or dict"
        assert isinstance(coords, tuple), "coords must be a tuple e.g. (i, j)"
        assert labels is None or isinstance(labels, dict), \
            f"labels is of type {type(labels)} but must be of type dict or None"
        if labels:
            assert all([isinstance(key, str) for key in labels.keys()]),\
                f"Input label keys are of types {[type(k) for k in labels.keys()]}. All label keys must be of type str."
            assert all([isinstance(val, (str, np.ndarray)) or np.issubdtype(type(val), np.number) for val in labels.values()]), \
                f"Input label vals are of types {[type(v) for v in labels.values()]}. " \
                f"All label values must be of type str or np.ndarray or a number (i.e. a subdtype of np.number) "

        assert name is None or isinstance(name, str), f"name is of type {type(name)} but must be of type str or None"

        assert slide_type is None or isinstance(slide_type, (pathml.core.SlideType, h5py._hl.group.Group)), \
            f"slide_type is of type {type(slide_type)} but must be of type pathml.core.types.SlideType"

        # instantiate SlideType object if needed
        if not slide_type and any([stain, tma, rgb, volumetric, time_series]):
            stain_type_dict = {"stain": stain, "tma": tma, "rgb": rgb, "volumetric": volumetric,
                               "time_series": time_series}
            # remove any Nones
            stain_type_dict = {key: val for key, val in stain_type_dict.items() if val}
            if stain_type_dict:
                slide_type = pathml.core.types.SlideType(**stain_type_dict)

        if isinstance(masks, pathml.core.masks.Masks):
            # move masks to dict so that Tile is in memory (must pass to dask client) 
            maskdict = OrderedDict()
            for mask in masks.h5manager.h5.keys():
                maskdict[mask] = masks[mask] 
            masks = maskdict

        if masks:
            for val in masks.values():
                if val.shape[:2] != image.shape[:2]:
                    raise ValueError(f"mask is of shape {val.shape} but must match tile shape {self.image.shape}")
            self.masks = masks 
        elif masks is None:
            self.masks = OrderedDict()

        self.image = image
        self.name = name
        self.coords = coords
        self.slide_type = slide_type
        self.labels = labels

    def __repr__(self):
        out = f"Tile(image shape {self.image.shape}, slidetype={self.slide_type}, " \
              f"masks={repr(self.masks) if self.masks else None}, " \
              f"coords={self.coords}, " \
              f"labels={list(self.labels.keys()) if self.labels else None})"
        return out

    def plot(self):
        """
        View the tile image, using matplotlib.
        Only supports RGB images currently
        """
        if self.image.shape[2] != 3 or self.image.ndim != 3:
            raise NotImplementedError(f"Plotting not supported for tile with image of shape {self.image.shape}")
        else:
            plt.imshow(self.image)
            if self.name:
                plt.title(self.name)
            plt.axis("off")
            plt.show()

    @property
    def shape(self):
        """
        convenience method.
        Calling ``tile.shape`` is equivalent to calling ``tile.image.shape``
        """
        return self.image.shape
