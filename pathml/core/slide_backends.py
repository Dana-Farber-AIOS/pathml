import openslide
from typing import Tuple, Union

from pathml.utils import pil_to_rgb


class SlideBackend:
    """base class for classes to interface with slides on disk"""
    def extract_tile(self, location, size, level):
        raise NotImplementedError

    def get_thumbnail(self, *args, **kwargs):
        raise NotImplementedError

    def get_image_shape(self, level):
        raise NotImplementedError


class OpenSlideBackend(SlideBackend):
    """
    Class for using OpenSlide to interface with image files

    Args:
        filename (str): path to image file on disk
    """
    def __init__(self, filename):
        self.filename = filename
        self.slide = openslide.OpenSlide(filename = filename)

    def extract_tile(self, location, size, level=None):
        """
        Extract a region of the image

        Args:
            location (Tuple[int, int]): Location of top-left corner of tile
            size (Union[int, Tuple[int, int]]): Size of each tile. May be a tuple of (height, width) or a
                single integer, in which case square tiles of that size are generated.
            level (int): level from which to extract chunks. Level 0 is highest resolution.

        Returns:
            np.ndarray: image at the specified region
        """
        # verify args
        if isinstance(size, int):
            size = (size, size)
        else:
            assert isinstance(size, tuple) and all([isinstance(a, int) for a in size]) and len(size) == 2, \
                f"Input size {size} not valid. Must be an integer or a tuple of two integers."
        if level is None:
            level = 0
        else:
            assert isinstance(level, int), f"level {level} must be an integer"
            assert level < self.slide.level_count, \
                f"input level {level} invalid for a slide with {self.slide.level_count} levels"

        region = self.slide.read_region(location = location, level = level, size = size)
        region_rgb = pil_to_rgb(region)
        return region_rgb

    def get_image_shape(self, level=0):
        """
        Get the shape of the image at specified level.

        Args:
            level (int): Which level to get shape from. Level 0 is highest resolution. Defaults to 0.

        Returns:
            Tuple[int, int]: Shape of image at target level.
        """
        j, i = self.slide.level_dimensions[level]
        return i, j

    def get_thumbnail(self, size):
        """
        Get a thumbnail of the slide.

        Args:
            size (Tuple[int, int]): the maximum size of the thumbnail

        Returns:
            np.ndarray: RGB thumbnail image
        """
        thumbnail = self.slide.get_thumbnail(size)
        thumbnail = pil_to_rgb(thumbnail)
        return thumbnail


class BioFormatsBackend(SlideBackend):
    """
    Class for using BioFormats to interface with image files

    Args:
        filename (str): path to image file on disk
    """
    def __init__(self, filename):
        self.filename = filename
        raise NotImplementedError


class DICOMBackend(SlideBackend):
    """
    Class for interfacing with DICOM images on disk
    """
    def __init__(self, filename):
        self.filename = filename
        raise NotImplementedError
