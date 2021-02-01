import openslide
from typing import Tuple, Union
import bioformats
import javabridge
from bioformats.metadatatools import createOMEXMLMetadata

from pathml.utils import pil_to_rgb


class SlideBackend:
    """base class for classes to interface with slides on disk"""
    def extract_tile(self, location, size, **kwargs):
        raise NotImplementedError

    def get_thumbnail(self, size, **kwargs):
        raise NotImplementedError

    def get_image_shape(self, **kwargs):
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

    def get_thumbnail(self, size, **kwargs):
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
    Class for using BioFormats to interface with image files.

    Built on `python-bioformats <https://github.com/CellProfiler/python-bioformats>`_ which wraps ome bioformats
    java library, parses pixel and metadata of proprietary formats, and
    converts all formats to OME-TIFF. Please cite: https://pubmed.ncbi.nlm.nih.gov/20513764/

    Args:
        filename (str): path to image file on disk
    """
    def __init__(self, filename):
        self.filename = filename
        # init java virtual machine
        javabridge.start_vm(class_path = bioformats.JARS)
        # java maximum array size of 2GB constrains image size
        ImageReader = bioformats.formatreader.make_image_reader_class()
        FormatTools = bioformats.formatreader.make_format_tools_class()
        reader = ImageReader()
        omeMeta = createOMEXMLMetadata()
        reader.setMetadataStore(omeMeta)
        reader.setId(self.filename)
        sizex, sizey, sizez, sizec = reader.getSizeX(), reader.getSizeY(), reader.getSizeZ(), reader.getSizeC()
        self.twodshape = (sizex, sizey)
        self.threedshape = (sizex, sizey, sizez)
        self.fourdshape = (sizex, sizey, sizez, sizec)
        self.imsize = sizex * sizey * sizez * sizec
        raise NotImplementedError

    def extract_tile(self, location, size, **kwargs):
        raise NotImplementedError

    def get_thumbnail(self, size, **kwargs):
        raise NotImplementedError



class DICOMBackend(SlideBackend):
    """
    Class for interfacing with DICOM images on disk
    """
    def __init__(self, filename):
        self.filename = filename
        raise NotImplementedError
