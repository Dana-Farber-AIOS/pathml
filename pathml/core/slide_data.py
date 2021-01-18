import numpy as np

from pathml.core.slide import Slide
from pathml.core.masks import Masks
from pathml.core.tiles import Tile,Tiles
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
    def __init__(self, slide=None, masks=None, tiles=None, labels=None, h5=None):
        assert issubclass(slide, Slide), f"slide is of type {type(slide)} but must be a subclass of pathml.core.slide.Slide"
        self.slide = slide
        self._slidetype = type(slide)
        self.name = slide.name
        self.shape = None if slide is None else slide.shape
        assert isinstance(masks, (None, Masks)), f"mask are of type {type(masks)} but must be of type pathml.core.masks.Masks"
        self.masks = masks 
        assert isinstance(tiles, (None, Tiles)), f"tiles are of type {type(tiles)} but must be of type pathml.core.tiles.Tiles" 
        self.tiles = tiles
        assert isinstance(labels, dict), f"labels are of type {type(labels)} but must be of type dict. array-like labels should be stored in masks."
        self.labels = labels
        self.history = []
        self.h5 = h5

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
        tileshape = kwargs.pop("tileshape", 3000)
        tilelevel = kwargs.pop("tilelevel", None)
        tilestride = kwargs.pop("tilestride", tileshape)
        tilepad = kwargs.pop("tilepad", False)
        for tile in self.tiles(level = tilelevel, shape = tileshape, stride = tilestride, pad = tilepad):
            pipeline(tile, **kwargs)

    def generate_tiles(self, level=None, shape=3000, stride=shape, pad=False):
        """
        Generator over tiles.
        All pipelines must be composed of transforms acting on tiles.

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
                    yield Tile(region_rgb, masks_chunk, coords)

        elif self.slide.backend == 'bioformats':
            # TODO: this is complicated because need to handle both chunking, allocating different 2GB java arrays, and managing java heap  
            pass

    def plot():
        """

        Args:
            location
            tile = True
            size
            downsample
            mask(str)
            save
        """
        pass 

    def write_h5(
        self,
        filename: Optional[PathLike] = None,
        compression: Optional[Literal["gzip", "lzf"]] = None,
        compression_opts: Union[int, Any] = None,
    ):
        # see https://github.com/theislab/anndata/blob/master/anndata/_core/anndata.py#L1834-L1889
        # see https://github.com/theislab/anndata/blob/f1d1e17ec61bea97d7db9ea1bb9ee88f883fc484/anndata/_io/h5ad.py#L54
        # TODO: aggregate slide, masks, tiles .h5s into a single .h5 
        # TODO: method to populate from h5
        slideh5 = self.slide.h5
        masksh5 = self.masks.h5
        tilesh5 = self.tiles.h5

