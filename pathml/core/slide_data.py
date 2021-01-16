import numpy as np

from pathml.core.masks import Masks
from pathml.core.tiles import Tiles


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
    def __init__(self, slide=None, masks=None, tiles=None, labels=None):
        self.slide = slide
        self.name = None if slide is None else slide.name
        # TODO: should size be a dict containing the sizes of slide?
        self.size = None if slide is None else slide.size
        assert isinstance(masks, Masks), f"mask are of type {type(masks)} but must be of type pathml.core.masks.Masks"
        self.masks = masks 
        assert isinstance(tiles, Tiles), f"tiles are of type {type(tiles)} but must be of type pathml.core.tiles.Tiles" 
        self.tiles = tiles
        assert isinstance(labels, ('int','str',Masks)), f"labels are of type {type(labels)} but must be of type int, str, or pathml.core.masks.Masks"
        self.labels = labels
        self.history = []

    def __repr__(self):  # pragma: no cover
        out = f"SlideData(slide={repr(self.slide)}, "
        out += f"slide: {self.slide.shape}, "
        out += f"masks: {'None' if self.masks is None else repr(self.masks)}, "
        out += f"tiles: {'None' if self.tiles is None else repr(self.tiles)})"
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
