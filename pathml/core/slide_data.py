import numpy as np

from pathml.core.masks import Masks
from pathml.core.tiles import Tiles
from pathml.preprocessing.transforms import Transform
from pathml.preprocessing.pipeline import Pipeline


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

    def __repr__(self): 
        out = f"SlideData(slide={repr(self.slide)}, "
        out += f"slide: {self.slide.shape}, "
        out += f"masks: {'None' if self.masks is None else repr(self.masks)}, "
        out += f"tiles: {'None' if self.tiles is None else repr(self.tiles)})"
        return out 

    def run(pipeline, **kwargs):
        assert isinstance(pipeline, Pipeline), f"pipeline is of type {type(pipeline)} but must be of type pathml.preprocessing.pipeline.Pipeline"
        # pop args required for chunks and check that they exist
        chunksize = kwargs.pop("chunksize", 3000)
        for chunk in self.chunks():
            pipeline(chunk, **kwargs)

    def chunks(self, level=None, shape, stride=shape, pad=False):
        """
        Generator over chunks.
        All pipelines must be composed of transforms acting on chunks.

        Args:
            level (int): level from which to extract chunks.
            shape (tuple(int)): chunk shape.
            stride (int): stride between chunks. If ``None``, uses ``stride = size`` for non-overlapping chunks.
                Defaults to ``None``.
            pad (bool): How to handle chunks on the edges. If ``True``, these edge chunks will be zero-padded
                symmetrically and yielded with the other chunks. If ``False``, incomplete edge chunks will be ignored.
                Defaults to ``False``.
        Yields:
            np.ndarray: Extracted chunk of dimension (size, size, 3)
        """
        # if shape is int
        # square chunks
        # else chunks of shape
        if isinstance(shape, int):
            shape = (shape, shape)
        if self.slide.backend == 'openslide': 
            if level == None:
                # TODO: is this the right default for openslide?
                level = 1
            j, i = self.slide.level_dimensions[level]

            if stride is None:
                stride_i = shape[0]
                stride_j = shape[1]

            n_chunk_i = (i-shape[0])// stride_i +1
            n_chunk_j = (j-shape[1])// stride_i +1

            if pad:
                n_chunk_i = i // stride_i +1
                n_chunk_j = j // stride_j +1

            for ix_i in range(n_chunk_i):
                for ix_j in range(n_chunk_j):
                    
                    region = self.slide.read_region(
                        location = (int(ix_j * stride_j), int(ix_i * stride_i)),
                        level = level, size = (shape[0], shape[1])
                    )
                    region_rgb = pil_to_rgb(region)
                    # TODO: test. switch i and j?
                    if self.masks is not None:
                        masks_chunk = self.masks.slice([int(ix_j*stride):int(ix_j*stride)+size,int(ix_i*stride):int(ix_i*stride)+size, ...])
                    yield region_rgb, masks_chunk
        elif self.slide.backend == 'bioformats':
            # TODO: implement
            pass

    def plot():
        pass 

    def save():
        pass
