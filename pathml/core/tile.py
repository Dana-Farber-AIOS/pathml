import numpy as np

from pathml.core.masks import Masks


class Tile:
    """
    Object for representing a tile region extracted from an image.

    Args:
        image (np.ndarray): image
        masks (pathml.core.Masks): corresponding masks for region
        coords (tuple): Coordinates of tile relative to maximum resolution of whole-slide image.
            The (i,j) coordinate system is based on labelling the top-leftmost pixel as (0, 0)
        slidetype (str): type of image (e.g. "HE")
    """
    def __init__(self, image, coords, slidetype, masks=None):
        assert isinstance(image, np.ndarray), f"image of type {type(image)} must be a np.ndarray"
        if masks is not None:
            assert isinstance(masks, Masks), f"masks of type {type(masks)} must be pathml.core.Masks"
        assert isinstance(coords, tuple) and len(coords) == 2, "coords must be a tuple of (i, j)"

        self.image = image
        self.coords = coords
        self.masks = masks
        self.slidetype = slidetype

    def __repr__(self):
        return f"Tile(image shape {self.image.shape}, slidetype={self.slidetype}, " \
               f"mask={repr(self.masks)}, coords={self.coords})"
