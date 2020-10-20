import openslide
import cv2
import os
import numpy as np

from pathml.preprocessing.slide_data import SlideData
from pathml.preprocessing.utils import pil_to_rgb


class BaseSlide:  # pragma: no cover
    """
    Base class for slides.
    """
    def __init__(self, path, name=None):
        self.path = path
        if name:
            self.name = name
        else:
            basename = os.path.basename(path)
            name = os.path.splitext(basename)[0]
            self.name = name

    def load_data(self, **kwargs):
        """Initialize a :class:`~pathml.preprocessing.slide_data.SlideData` object"""
        raise NotImplementedError

    def chunks(self, **kwargs):
        """Iterator over chunks. Implement for each different backend"""
        raise NotImplementedError


class HESlide(BaseSlide):
    """
    Class for H&E stained slides, based on ``OpenSlide``
    """

    def __init__(self, path, name=None):
        super().__init__(path, name)
        self.slide = openslide.open_slide(path)

    def __repr__(self):  # pragma: no cover
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
        return out

    def chunks(self, level, size, stride=None):
        """Iterates over chunks. Useful for processing the image in pieces, avoiding having to load the entire image
        at full-resolution.

        Args:
            level (int): Level from which to extract chunks.
            size (int): Chunk size.
            stride (int): stride between chunks. If None, defaults to stride = size for non-overlapping chunks.
        """
        j, i = self.slide.level_dimensions[level]

        if stride is None:
            stride = size

        # TODO update to use padding or something to handle the edges better
        n_chunk_i = i // stride
        n_chunk_j = j // stride

        for ix_i in range(n_chunk_i):
            for ix_j in range(n_chunk_j):
                region = self.slide.read_region(
                    location = (ix_j * stride, ix_i * stride),
                    level = level, size = (size, size)
                )
                region_rgb = pil_to_rgb(region)
                yield region_rgb
