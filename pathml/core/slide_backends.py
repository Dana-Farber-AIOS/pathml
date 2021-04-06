import openslide
import numpy as np
from typing import Tuple, Union
import bioformats
import javabridge
from scipy.ndimage import zoom
from bioformats.metadatatools import createOMEXMLMetadata

from pathml.core.tile import Tile
from pathml.core.slide_classes import MultiparametricSlide 
from pathml.utils import pil_to_rgb


class SlideBackend:
    """base class for classes to interface with slides on disk"""
    def extract_region(self, location, size, **kwargs):
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

    def extract_region(self, location, size, level=None):
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
        assert isinstance(level, int), f"level {level} invalid. Must be an int."
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
        # FormatTools = bioformats.formatreader.make_format_tools_class()
        reader = ImageReader()
        omeMeta = createOMEXMLMetadata()
        reader.setMetadataStore(omeMeta)
        reader.setId(str(self.filename))
        sizex, sizey, sizez, sizec, sizet = reader.getSizeX(), reader.getSizeY(), reader.getSizeZ(), reader.getSizeC(), reader.getSizeT()
        self.shape = (sizex, sizey, sizez, sizec, sizet)

    def get_image_shape(self):
        """
        Get the shape of the image.

        Returns:
            Tuple[int, int]: Shape of image. 
        """
        return self.shape 

    def extract_region(self, location, size, **kwargs):
        """
        Extract a region of the image. All bioformats images have 5 dimensions representing
        (x, y, z, channel, time). If a tuple with len < 5 is passed, missing dimensions will be 
        retrieved in full.

        Args:
            location (Tuple[int, int]): Location of corner of extracted region closest to the origin.
            size (Tuple[int, int, ...]): Size of each region. Must be a tuple 

        Returns:
            np.ndarray: image at the specified region

        Example:
            Extract 2000x2000 x,y region from upper left corner of 7 channel, 2d fluorescent image.
            data.slide.extract_region(location = (0,0), size = (2000,2000))
            # plot single channel
            plt.figure()
            plt.imshow(region[:,:,0,0,0])

            Extract 2000x2000 x,y region of the first channel from upper left corner.
            region = data.slide.extract_region(location = (0,0,0,0,0), size = (2000,2000,1,1,1))
            # plot full region
            plt.figure()
            plt.imshow(region[:,:,0,:,0])
        """
        if self.shape[0]*self.shape[1]*self.shape[2]*self.shape[3]*self.shape[4] > 2147483647:
            raise Exception(f"Java arrays allocate maximum 32 bits (~2GB). Image size is {self.imsize}")

        javabridge.start_vm(class_path = bioformats.JARS)
        reader = bioformats.ImageReader(str(self.filename), perform_init=True)
        array = np.empty(self.shape)
        for z in range(self.shape[2]):
            for c in range(self.shape[3]):
                for t in range(self.shape[4]):
                    data = reader.read(z=z, t=t, series=c, rescale = False)
                    slice_array = np.asarray(data)
                    array[:,:,z,c,t] = np.transpose(slice_array)
        # TODO: read slices directly, rather than read then slice 
        slices = [slice(location[i],location[i]+size[i]) for i in range(len(size))] 
        array = array[tuple(slices)]
        coords = location + [0]*(len(array)-len(location)) 
        return Tile(image=array, coords=tuple(coords), slidetype=MultiparametricSlide) 

    def get_thumbnail(self, size=None, **kwargs):
        """
        Get a thumbnail of the image. Since there is no default thumbnail for multiparametric, volumetric
        images, this function supports downsampling of all image dimensions. 

        Args:
            size (Tuple[int, int]): thumbnail size 

        Returns:
            np.ndarray: RGB thumbnail image

        Example:
            Get 1000x1000 thumbnail of 7 channel fluorescent image.
            shape = data.slide.get_image_shape()
            thumb = data.slide.get_thumbnail(size=(1000,1000, shape[2], shape[3], shape[4]))
        """
        assert isinstance(size, (tuple, type(None))), f"Size must be a tuple of ints."
        if size is not None:
            assert len(size) == len(self.shape), f"image is of dimension {len(self.shape)} which does not match passed dimension of size {len(size)}"
        if self.shape[0]*self.shape[1]*self.shape[2]*self.shape[3] > 2147483647:
            raise Exception(f"Java arrays allocate maximum 32 bits (~2GB). Image size is {self.imsize}")
        javabridge.start_vm(class_path = bioformats.JARS)
        reader = bioformats.ImageReader(str(self.filename), perform_init=True)
        array = np.empty(self.shape)
        for z in range(self.shape[2]):
            for c in range(self.shape[3]):
                for t in range(self.shape[4]):
                    data = reader.read(z=z, t=t, series=c, rescale = False)
                    slice_array = np.asarray(data)
                    array[:,:,z,c,t] = np.transpose(slice_array)
        if size is not None:
            ratio = tuple([x/y for x,y in zip(size, self.shape)]) 
            print(ratio)
            image_array = zoom(array, ratio) 
        return image_array

class DICOMBackend(SlideBackend):
    """
    Class for interfacing with DICOM images on disk
    """
    def __init__(self, filename):
        self.filename = filename
        raise NotImplementedError
