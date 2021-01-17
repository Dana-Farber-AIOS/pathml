import numpy as np
import os
import cv2
import shutil
from pathlib import Path

from pathml.core.masks import Masks


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
    def __init__(self, array, labels=None, masks=None, i=None, j=None):
        assert isinstance(array, np.ndarray), "Array must be a np.ndarray"
        self.array = array
        self.i = i  # i coordinate of top left corner pixel
        self.j = j  # j coordinate of top left corner pixel
        assert isinstance(masks, (None, Masks, dict)), f"masks is of type {type(masks)} but must be of type pathml.core.Masks or dict"
        if isinstance(masks, Masks):
            self.masks = Masks
        if isinstance(masks, dict): 
            for val in masks.values():
                if val.shape != self.array.shape[:2]:
                    raise ValueError(f"mask is of shape {val.shape} but must match tile shape {self.array.shape}")
            self.masks = Masks(masks)
        elif masks == None:
            self.masks = masks 
        assert isinstance(labels, (None, dict))
        self.labels = labels
        # initialize temp file
        fd, path = tempfile.mkstemp()
        # write tile, masks, labels to h5, then clear
        # core?
        f = h5py.File(fd, 'w')
        f.write
        self.h5 = path

    def __repr__(self):  # pragma: no cover
        return f"Tile(array shape {self.array.shape}, " \
               f"i={self.i if self.i is not None else 'None'}, " \
               f"j={self.j if self.j is not None else 'None'})"

    def loadtile(self):
        pass

    def save(self, out_dir, filename):
        """
        Save tile to disk as jpeg file.

        :param out_dir: directory to write to.
        :type out_dir: str
        :param filename: File name of saved tile.
        :type filename: str
        """
        savepath = Path(out_dir)+Path(filename)
        shutil.copy(self.h5, savepath)
