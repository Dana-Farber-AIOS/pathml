import numpy as np
import os
import cv2

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
               f"i={self.i if self.i is not None else 'None'}, " \
               f"j={self.j if self.j is not None else 'None'})"

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
