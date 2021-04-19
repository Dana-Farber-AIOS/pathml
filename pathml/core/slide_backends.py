import openslide
import numpy as np
from typing import Tuple
import bioformats
import javabridge
from scipy.ndimage import zoom
from bioformats.metadatatools import createOMEXMLMetadata

from pathml.utils import pil_to_rgb

import pathml.core



class SlideBackend:
    """base class for backends that interface with slides on disk"""
    def extract_region(self, location, size, level, **kwargs):
        raise NotImplementedError

    def get_thumbnail(self, size, **kwargs):
        raise NotImplementedError

    def get_image_shape(self, **kwargs):
        raise NotImplementedError

    def generate_tiles(self, shape, stride, pad, **kwargs):
        raise NotImplementedError


class OpenSlideBackend(SlideBackend):
    """
    Use OpenSlide to interface with image files

    Depends on `openslide-python <https://openslide.org/api/python/>`_ which wraps the `openslide <https://openslide.org/>`_ C library.

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
        assert level < self.slide.level_count, \
            f"input level {level} invalid for slide with {self.slide.level_count} levels total"
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

    def generate_tiles(self, shape=3000, stride=None, pad=False, level=0):
        """
        Generator over tiles.

        Padding works as follows:
        If ``pad is False``, then the first tile will start flush with the edge of the image, and the tile locations
        will increment according to specified stride, stopping with the last tile that is fully contained in the image.
        If ``pad is True``, then the first tile will start flush with the edge of the image, and the tile locations
        will increment according to specified stride, stopping with the last tile which starts in the image. Regions
        outside the image will be padded with 0.
        For example, for a 5x5 image with a tile size of 3 and a stride of 2, tile generation with ``pad=False`` will
        create 4 tiles total, compared to 6 tiles if ``pad=True``.

        Args:
            shape (int or tuple(int)): Size of each tile. May be a tuple of (height, width) or a single integer,
                in which case square tiles of that size are generated.
            stride (int): stride between chunks. If ``None``, uses ``stride = size`` for non-overlapping chunks.
                Defaults to ``None``.
            pad (bool): How to handle tiles on the edges. If ``True``, these edge tiles will be zero-padded
                and yielded with the other chunks. If ``False``, incomplete edge chunks will be ignored.
                Defaults to ``False``.
            level (int, optional): For slides with multiple levels, which level to extract tiles from.
                Defaults to 0 (highest resolution).

        Yields:
            pathml.core.tile.Tile: Extracted Tile object
        """
        assert isinstance(shape, int) or (isinstance(shape, tuple) and len(shape) == 2), \
            f"input shape {shape} invalid. Must be a tuple of (H, W), or a single integer for square tiles"
        if isinstance(shape, int):
            shape = (shape, shape)
        assert stride is None or isinstance(stride, int) or (isinstance(stride, tuple) and len(stride) == 2), \
            f"input stride {stride} invalid. Must be a tuple of (stride_H, stride_W), or a single int"
        assert isinstance(level, int), f"level {level} invalid. Must be an int."
        assert level < self.slide.level_count, \
            f"input level {level} invalid for slide with {self.slide.level_count} levels total"

        if stride is None:
            stride = shape
        elif isinstance(stride, int):
            stride = (stride, stride)

        i, j = self.get_image_shape(level = level)

        stride_i, stride_j = stride

        if pad:
            n_chunk_i = i // stride_i + 1
            n_chunk_j = j // stride_j + 1

        else:
            n_chunk_i = (i - shape[0]) // stride_i + 1
            n_chunk_j = (j - shape[1]) // stride_j + 1

        for ix_i in range(n_chunk_i):
            for ix_j in range(n_chunk_j):
                coords = (int(ix_j * stride_j), int(ix_i * stride_i))
                # get image for tile
                tile_im = self.extract_region(location = coords, size = shape, level = level)
                yield pathml.core.tile.Tile(image = tile_im, coords = coords)


class BioFormatsBackend(SlideBackend):
    """
    Use BioFormats to interface with image files.

    Depends on `python-bioformats <https://github.com/CellProfiler/python-bioformats>`_ which wraps ome bioformats
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
            Tuple[int, int]: Shape of image (H, W)
        """
        return self.shape[:2] 

    def extract_region(self, location, size, level=None):
        """
        Extract a region of the image. All bioformats images have 5 dimensions representing
        (x, y, z, channel, time). If a tuple with len < 5 is passed, missing dimensions will be 
        retrieved in full.

        Args:
            location (Tuple[int, int]): Location of corner of extracted region closest to the origin.
            size (Tuple[int, int, ...]): Size of each region. If an integer is passed, will convert to a tuple of (H, W)
                and extract a square region. If a tuple with len < 5 is passed, missing
                dimensions will be retrieved in full.
            level (int): level from which to extract chunks. Level 0 is highest resolution. Must be 0 or None, since
                BioFormatsBackend does not support multiple levels.

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
        if level not in [None, 0]:
            raise ValueError("BioFormatsBackend does not support levels, please pass a level in [None, 0]")

        # if a single int is passed for size, convert to a tuple to get a square region
        if type(size) is int:
            size = (size, size)
        if not (isinstance(location, tuple) and len(location) == 2 and all([isinstance(x, int) for x in location])):
            raise ValueError(f"input location {location} invalid. Must be a tuple of (i, j) integer coordinates")

        if np.prod(self.shape) > 2147483647:
            raise Exception(f"Java arrays allocate maximum 32 bits (~2GB). Image size is {np.prod(self.shape)}")

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
        array = array.astype(np.uint8)
        return array

    def get_thumbnail(self, size=None):
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
            if len(size) != len(self.shape):
                size = size + self.shape[len(size):]
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
            assert ratio[3] == 1, f"cannot interpolate between fluor channels, resampling doesn't apply, fix size[3]"
            image_array = zoom(array, ratio) 
        return image_array

    def generate_tiles(self, shape=3000, stride=None, pad=False):
        """
        Generator over tiles.

        Padding works as follows:
        If ``pad is False``, then the first tile will start flush with the edge of the image, and the tile locations
        will increment according to specified stride, stopping with the last tile that is fully contained in the image.
        If ``pad is True``, then the first tile will start flush with the edge of the image, and the tile locations
        will increment according to specified stride, stopping with the last tile which starts in the image. Regions
        outside the image will be padded with 0.
        For example, for a 5x5 image with a tile size of 3 and a stride of 2, tile generation with ``pad=False`` will
        create 4 tiles total, compared to 6 tiles if ``pad=True``.

        Args:
            shape (int or tuple(int)): Size of each tile. May be a tuple of (height, width) or a single integer,
                in which case square tiles of that size are generated.
            stride (int): stride between chunks. If ``None``, uses ``stride = size`` for non-overlapping chunks.
                Defaults to ``None``.
            pad (bool): How to handle tiles on the edges. If ``True``, these edge tiles will be zero-padded
                and yielded with the other chunks. If ``False``, incomplete edge chunks will be ignored.
                Defaults to ``False``.

        Yields:
            pathml.core.tile.Tile: Extracted Tile object
        """
        assert isinstance(shape, int) or (isinstance(shape, tuple) and len(shape) == 2), \
            f"input shape {shape} invalid. Must be a tuple of (H, W), or a single integer for square tiles"
        if isinstance(shape, int):
            shape = (shape, shape)
        assert stride is None or isinstance(stride, int) or (isinstance(stride, tuple) and len(stride) == 2), \
            f"input stride {stride} invalid. Must be a tuple of (stride_H, stride_W), or a single int"

        if stride is None:
            stride = shape
        elif isinstance(stride, int):
            stride = (stride, stride)

        i, j = self.get_image_shape()

        stride_i, stride_j = stride

        if pad:
            n_chunk_i = i // stride_i + 1
            n_chunk_j = j // stride_j + 1

        else:
            n_chunk_i = (i - shape[0]) // stride_i + 1
            n_chunk_j = (j - shape[1]) // stride_j + 1

        for ix_i in range(n_chunk_i):
            for ix_j in range(n_chunk_j):
                coords = (int(ix_j * stride_j), int(ix_i * stride_i))
                # get image for tile
                tile_im = self.extract_region(location = coords, size = shape)
                yield pathml.core.tile.Tile(image = tile_im, coords = coords)


class DICOMBackend(SlideBackend):
    """
    Class for interfacing with DICOM images on disk
    """
    def __init__(self, filename):
        self.filename = filename
        raise NotImplementedError
