import openslide
import cv2
import numpy as np

from pathml.core.slide import Slide
from pathml.core.slide_data import SlideData
from pathml.utils import pil_to_rgb
from pathml.core.masks import Masks


class RGBSlide(Slide):
    pass


class HESlide(RGBSlide):
    """
    Class for H&E stained slides, based on ``OpenSlide``
    """

    def __init__(self, path, name=None, masks=None):
        super().__init__(path, name)
        self.slide = openslide.open_slide(path)
        self.masks = masks 

    def __repr__(self):
        return f"HESlide(path={self.path}, name={self.name})"

    def load_data(self, level=0, location=(0, 0), size=None):
        """
        Load slide using ``OpenSlide``, and initialize a :class:`~pathml.preprocessing.slide_data.SlideData` object

        :param level: level of interest. Defaults to 0 (highest resolution)
        :type level: int
        :param location: (x, y) tuple giving the top left pixel in the level 0 reference frame. Defaults to (0, 0)
        :type location: tuple
        :param size: (width, height) tuple giving the region size inpixels. If ``None``,
            uses entire image at specified level. Defaults to ``None``.
        :type size: tuple
        :return: :class:`~pathml.preprocessing.slide_data.SlideData` object
        """
        # TODO check that input file is of a type that can be loaded by OpenSlide
        # TODO check if dimensions are getting flipped during conversion, since PIL uses W,H,D and np uses H,W,D

        if size is None:
            size = self.slide.level_dimensions[level]
        image_array_pil = self.slide.read_region(location = location, level = level, size = size)
        # note that PIL uses (width, height) but when converting to numpy we get the correct (height, width) dims
        image_array_rgba = np.asarray(image_array_pil, dtype = np.uint8)
        image_array = cv2.cvtColor(image_array_rgba, cv2.COLOR_RGBA2RGB)
        out = SlideData(wsi = self, image = image_array)

        if self.masks:
            # supports 2d (birdseye), and 3d (birdseye by channel) masking 
            for val in self.masks.values():
                if len(val) == 2:
                    if val.shape != imagearray.shape[:2]:
                        raise ValueError(f"mask is of shape {val.shape} but must match slide shape {imagearray.shape[:2]}")
                if len(val) == 3:
                    if val.shape != imagearray.shape:
                        raise ValueError(f"mask is of shape {val.shape} but must match slide shape {imagearray.shape}")
                else: 
                    raise ValueError(f"mask must be of dimension 2 (birdseye) or dimension 3 (birdseye with channel masking) but received mask of dimension {len(val)}")
            self.masks = Masks(masks)
        return out

    def chunks(self, level, size, stride=None, pad=False):
        """Generator over chunks. Useful for processing the image in pieces, avoiding having to load the entire image
        at full-resolution.

        Args:
            level (int): Level from which to extract chunks.
            size (int): Chunk size.
            stride (int): Stride between chunks. If ``None``, uses ``stride = size`` for non-overlapping chunks.
                Defaults to ``None``.
            pad (bool): How to handle chunks on the edges. If ``True``, these edge chunks will be zero-padded
                symmetrically and yielded with the other chunks. If ``False``, incomplete edge chunks will be ignored.
                Defaults to ``False``.

        Yields:
            np.ndarray: Extracted RGB chunk of dimension (size, size, 3)
        """
        j, i = self.slide.level_dimensions[level]

        if stride is None:
            stride = size

        n_chunk_i = (i-size)// stride +1
        n_chunk_j = (j-size)// stride +1

        if pad:
            n_chunk_i = i // stride +1
            n_chunk_j = j // stride +1

        for ix_i in range(n_chunk_i):
            for ix_j in range(n_chunk_j):
                
                region = self.slide.read_region(
                    location = (int(ix_j * stride), int(ix_i * stride)),
                    level = level, size = (size, size)
                )
                region_rgb = pil_to_rgb(region)
                # TODO: test. switch i and j?
                if self.masks is not None:
                    masks_chunk = self.masks.slice([int(ix_j*stride):int(ix_j*stride)+size,int(ix_i*stride):int(ix_i*stride)+size, ...])
                yield region_rgb, masks_chunk
