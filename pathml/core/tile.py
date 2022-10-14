"""
Copyright 2021, Dana-Farber Cancer Institute and Weill Cornell Medicine
License: GNU GPL 2.0
"""

import numpy as np
import anndata
from collections import OrderedDict
import matplotlib.pyplot as plt
import h5py
import reprlib
from loguru import logger
import dask
from dask.delayed import Delayed

import pathml.core.masks


class Tile:
    """
    Object representing a tile extracted from an image. Holds the array for the tile, as well as the (i,j)
    coordinates of the top-left corner of the tile in the original image. The (i,j) coordinate system is based
    on labelling the top-leftmost pixel as (0, 0)

    Args:
        image (np.ndarray or dask.delayed.Delayed): Tile image or dask.delayed.Delayed object to load image
        coords (tuple): Coordinates of tile relative to the whole-slide image.
            The (i,j) coordinate system is based on labelling the top-leftmost pixel of the WSI as (0, 0).
        name (str, optional): Name of tile
        masks (dict): masks belonging to tile. If masks are supplied, all masks must be the
            same shape as the tile.
        labels: labels belonging to tile
        counts (AnnData): counts matrix for the tile.
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

    def __init__(
        self,
        image,
        coords,
        name=None,
        masks=None,
        labels=None,
        counts=None,
        slide_type=None,
        stain=None,
        tma=None,
        rgb=None,
        volumetric=None,
        time_series=None,
    ):
        # check inputs
        assert isinstance(image, Delayed) or isinstance(
            image, np.ndarray
        ), f"image of type {type(image)} must be a np.ndarray or a dask.delayed.Delayed object"
        assert masks is None or isinstance(
            masks, dict
        ), f"masks is of type {type(masks)} but must be of type dict"
        assert isinstance(coords, tuple), "coords must be a tuple e.g. (i, j)"
        assert labels is None or isinstance(
            labels, dict
        ), f"labels is of type {type(labels)} but must be of type dict or None"
        if labels:
            assert all(
                [isinstance(key, str) for key in labels.keys()]
            ), f"Input label keys are of types {[type(k) for k in labels.keys()]}. All label keys must be of type str."
            assert all(
                [
                    isinstance(val, (str, np.ndarray))
                    or np.issubdtype(type(val), np.number)
                    or np.issubdtype(type(val), np.bool_)
                    for val in labels.values()
                ]
            ), (
                f"Input label vals are of types {[type(v) for v in labels.values()]}. "
                f"All label values must be of type str or np.ndarray or a number (i.e. a subdtype of np.number) "
            )

        assert (
            name != "None" and name != 0
        ), "Cannot use values of '0' or 'None' as tile names"
        assert name is None or isinstance(
            name, str
        ), f"name is of type {type(name)} but must be of type str or None"

        assert slide_type is None or isinstance(
            slide_type, (pathml.core.SlideType, h5py._hl.group.Group)
        ), f"slide_type is of type {type(slide_type)} but must be of type pathml.core.types.SlideType"

        # instantiate SlideType object if needed
        if not slide_type and any([stain, tma, rgb, volumetric, time_series]):
            stain_type_dict = {
                "stain": stain,
                "tma": tma,
                "rgb": rgb,
                "volumetric": volumetric,
                "time_series": time_series,
            }
            # remove any Nones
            stain_type_dict = {key: val for key, val in stain_type_dict.items() if val}
            if stain_type_dict:
                slide_type = pathml.core.slide_types.SlideType(**stain_type_dict)

        assert counts is None or isinstance(
            counts, anndata.AnnData
        ), f"counts is of type {type(counts)} but must be of type anndata.AnnData or None"

        self._image = image
        self.masks = masks if masks else OrderedDict()
        self.name = name
        self.coords = coords
        self.slide_type = slide_type
        self.labels = labels
        self.counts = counts

    @property
    def image(self):
        if isinstance(self._image, Delayed):
            image = dask.compute(self._image, scheduler="single-threaded")
            if isinstance(image, tuple):
                image = image[0]
            assert isinstance(
                image, np.ndarray
            ), f"image of type {type(image)} must be a np.ndarray"
            for val in self.masks.values():
                if val.shape[:2] != image.shape[:2]:
                    raise ValueError(
                        f"mask is of shape {val.shape} but must match tile shape {image.shape}"
                    )
            self._image = image
        return self._image

    @image.setter
    def image(self, image):
        assert isinstance(
            image, np.ndarray
        ), f"image of type {type(image)} must be a np.ndarray"
        self._image = image

    def __repr__(self):
        out = []
        out.append(f"Tile(coords={self.coords}")
        out.append(f"name={self.name}")
        out.append(f"image shape: {self.image.shape}")
        out.append(f"slide_type={repr(self.slide_type)}")
        if self.labels:
            out.append(
                f"{len(self.labels)} labels: {reprlib.repr(list(self.labels.keys()))}"
            )
        else:
            out.append("labels=None")
        if self.masks:
            out.append(
                f"{len(self.masks)} masks: {reprlib.repr(list(self.masks.keys()))}"
            )
        else:
            out.append("masks=None")
        if self.counts:
            out.append(f"counts matrix of shape {self.counts.shape}")
        else:
            out.append(f"counts=None")
        out = ",\n\t".join(out)
        out += ")"
        return out

    def plot(self, ax=None):
        """
        View the tile image, using matplotlib.
        Only supports RGB images currently

        Args:
            ax: matplotlib axis object on which to plot the thumbnail. Optional.
        """
        if self.image.shape[2] != 3 or self.image.ndim != 3:
            raise NotImplementedError(
                f"Plotting not supported for tile with image of shape {self.image.shape}"
            )

        if ax is None:
            ax = plt.gca()

        ax.imshow(self.image)
        if self.name:
            ax.set_title(self.name)
        ax.axis("off")

    @property
    def shape(self):
        """
        convenience method.
        Calling ``tile.shape`` is equivalent to calling ``tile.image.shape``
        """
        return self.image.shape
