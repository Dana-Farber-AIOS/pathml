import numpy as np
from pathml.preprocessing.masks import Masks


class SlideData:
    """
    Main class for holding data during preprocessing pipeline.
    Preprocessing pipelines change the state of this object.
    Attributes may be added or modified depending on needs of specific pipelines.

    :param wsi: WSI object from which this data object was generated
    :type wsi: :class:`~pathml.preprocessing.base.BaseSlide`
    :param image: image of slide
    :type image: np.ndarray
    :param mask: Array of masks generated for input image
    :type mask: np.ndarray
    :param tiles: list of :class:`~pathml.preprocessing.tiling.Tile` objects
    :type tiles: list
    """
    # TODO look at changing this to a dataclass
    def __init__(self, wsi=None, image=None, mask=None, tiles=None):
        self.wsi = wsi
        self.image = None if image is None else image.astype(np.uint8)
        self._mask = wsi.masks 
        if mask:
            mask(mask)
        self.tiles = tiles

    def __repr__(self):  # pragma: no cover
        out = f"SlideData(wsi={repr(self.wsi)}, "
        out += f"image shape: {self.image.shape}, "
        out += f"mask shape: {'None' if self._mask is None else repr(self.masks)}, "
        out += f"number of tiles: {'None' if self.tiles is None else len(self.tiles)})"
        return out

    @property
    def mask(self):
        return self._mask

    # TODO make this more intuitive, like use a method like .add_mask(). The setter isn't very clear as is
    @mask.setter
    def mask(self, key, new_mask):
        # use setter to handle initial None for mask to make mask updating easy
        if self._mask is None:
            self._mask = Masks({key, new_mask}) 
        else:
            self._mask.add(key, new_mask) 
