import numpy as np

from pathml.preprocessing.masks import Masks


class SlideData:
    """
    Main class representing a slide and its annotations. 
    Preprocessing pipelines change the state of this object.
    Declared by subclassing Slide

    :param name: name of slide
    :type name: str 
    :param size: total size of slide in pixels 
    :type size: int 
    :param slide: slide object
    :type slide: subclass of `~pathml.core.slide` 
    :param masks: object containing {key,mask} pairs
    :type masks: :class:`~pathml.core.masks.Masks` 
    :param tiles: object containing {coordinates,tile} pairs 
    :type tiles: :class:`~pathml.core.tiles.Tiles`
    :param labels: dictionary containing {key,label} pairs
    :type labels: collections.OrderedDict 
    :param history: the history of operations applied to the SlideData object
    :type history: list of __repr__'s from each method called on SlideData 
    """
    def __init__(self, slide=None, image=None, mask=None, tiles=None):
        self.slide = slide
        self.image = None if image is None else image.astype(np.uint8)
        self.masks = wsi.masks 
        if mask:
            masks(mask)
        self.tiles = tiles

    def __repr__(self):  # pragma: no cover
        out = f"SlideData(wsi={repr(self.wsi)}, "
        out += f"image shape: {self.image.shape}, "
        out += f"mask shape: {'None' if self._mask is None else repr(self.masks)}, "
        out += f"number of tiles: {'None' if self.tiles is None else len(self.tiles)})"
        return out

    @property
    def masks(self):
        return self.masks

    # TODO make this more intuitive, like use a method like .add_mask(). The setter isn't very clear as is
    @mask.setter
    def masks(self, key, new_mask):
        # use setter to handle initial None for mask to make mask updating easy
        if self.masks is None:
            self.masks = Masks({key, new_mask}) 
        else:
            self.masks.add(key, new_mask) 
