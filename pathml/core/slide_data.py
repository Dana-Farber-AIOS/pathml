import numpy as np

from pathml.core.slide import Slide
from pathml.core.masks import Masks
from pathml.core.tiles import Tiles
from pathml.core.chunk import Chunk
from pathml.preprocessing.transforms import Transform
from pathml.preprocessing.pipeline import Pipeline


class SlideData:
    """
    Main class representing a slide and its annotations. 
    Preprocessing pipelines change the state of this object.
    Declared by subclassing Slide

    Attributes:
        name (str): name of slide
        size (int): total size of slide in pixels 
        slide (`~pathml.core.slide.Slide`): slide object
        masks (`~pathml.core.masks.Masks`, optional): object containing {key,mask} pairs
        tiles (`~pathml.core.tiles.Tiles`, optional): object containing {coordinates,tile} pairs 
        labels (collections.OrderedDict, optional): dictionary containing {key,label} pairs
        history (list): the history of operations applied to the SlideData object
    """
    def __init__(self, slide=None, masks=None, tiles=None, labels=None):
        assert issubclass(slide, Slide), f"slide is of type {type(slide)} but must be a subclass of pathml.core.slide.Slide"
        self.slide = slide
        self._slidetype = type(slide)
        self.name = None if slide is None else slide.name
        # TODO: should size be a dict containing the sizes of slide?
        self.size = None if slide is None else slide.size
        assert isinstance(masks, (None, Masks)), f"mask are of type {type(masks)} but must be of type pathml.core.masks.Masks"
        self.masks = masks 
        assert isinstance(tiles, (None, Tiles)), f"tiles are of type {type(tiles)} but must be of type pathml.core.tiles.Tiles" 
        self.tiles = tiles
        assert isinstance(labels, (None, 'int', 'str')), f"labels are of type {type(labels)} but must be of type int or string. array-like labels should be stored in masks."
        self.labels = labels
        self.history = []

    def __repr__(self): 
        out = f"SlideData(slide={repr(self.slide)}, "
        out += f"slide={self.slide.shape}, "
        out += f"masks={'None' if self.masks is None else repr(self.masks)}, "
        out += f"tiles={'None' if self.tiles is None else repr(self.tiles)}, "
        out += f"labels={self.labels}, "
        out += f"history={self.history})"
        return out 

    def run(pipeline, **kwargs):
        assert isinstance(pipeline, Pipeline), f"pipeline is of type {type(pipeline)} but must be of type pathml.preprocessing.pipeline.Pipeline"
        chunkshape = kwargs.pop("chunkshape", 3000)
        chunklevel = kwargs.pop("chunklevel", None)
        chunkstride = kwargs.pop("chunkstride", chunkshape)
        chunkpad = kwargs.pop("chunkpad", False)
        for chunk in self.chunks(level = chunklevel, shape = chunkshape, stride = chunkstride, pad = chunkpad):
            pipeline(chunk, **kwargs)

    def chunks(self, level=None, shape=3000, stride=shape, pad=False):
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
                    coords = (ix_i, ix_j)
                    if self.masks is not None:
                        # TODO: test this line
                        masks_chunk = self.masks.slice([int(ix_j*stride_j):int(ix_j*stride_j)+size,int(ix_i*stride_i):int(ix_i*stride_i)+size, ...])
                    yield Chunk(region_rgb, masks_chunk, coords)

        elif self.slide.backend == 'bioformats':
            # TODO: this is complicated because need to handle both chunking, allocating different 2GB java arrays, and managing java heap  
            pass

    def plot():
        pass 

    def save():
        # see https://github.com/theislab/anndata/blob/master/anndata/_core/anndata.py#L1834-L1889
        # TODO: combine slide, masks, tiles .h5 objects into a single .h5 object 
        # TODO: read method
        pass
