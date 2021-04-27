import openslide
import numpy as np
from typing import Tuple
import bioformats
import javabridge
from scipy.ndimage import zoom
from bioformats.metadatatools import createOMEXMLMetadata
from pydicom.filereader import dcmread, data_element_offset_to_value
from pydicom.uid import UID
from pydicom.encaps import get_frame_offsets
from pydicom.filebase import DicomFile
from pydicom.tag import TupleTag, SequenceDelimiterTag
from pydicom.dataset import Dataset
from io import BytesIO
from PIL import Image

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

    def generate_tiles(self, shape=3000, stride=None, pad=False, level=None):
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
            level (int): level from which to extract chunks. Level 0 is highest resolution.
                Must be 0 or None, since BioFormatsBackend does not support multiple levels.

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
                tile_im = self.extract_region(location = coords, size = shape, level = level)
                yield pathml.core.tile.Tile(image = tile_im, coords = coords)


class DICOMBackend(SlideBackend):
    """
    Interface with DICOM files on disk.
    Provides efficient access to individual Frame items contained in the
    Pixel Data element without loading the entire element into memory.
    Assumes that frames are non-overlapping.

    Args:
        filename (str): Path to the DICOM Part10 file on disk
    """
    def __init__(self, filename):
        self.filename = str(filename)
        # read metadata fields of interest from DICOM, without reading the entire PixelArray
        tags = ["NumberOfFrames", "Rows", "Columns",
                "TotalPixelMatrixRows", "TotalPixelMatrixColumns"]
        metadata = dcmread(filename, specific_tags = tags)

        # can use frame shape, total shape to map between frame index and coords
        self.frame_shape = (metadata.Rows, metadata.Columns)
        self.shape = (metadata.TotalPixelMatrixRows, metadata.TotalPixelMatrixColumns)
        self.n_frames = int(metadata.NumberOfFrames)
        # use ceiling division to account for padding (i.e. still count incomplete frames on edge)
        # ceiling division from: https://stackoverflow.com/a/17511341
        self.n_rows = -(-self.shape[0] // self.frame_shape[0])
        self.n_cols = -(-self.shape[1] // self.frame_shape[1])
        self.transfer_syntax_uid = UID(metadata.file_meta.TransferSyntaxUID)

        # actual file
        self.fp = DicomFile(self.filename, mode = 'rb')
        self.fp.is_little_endian = self.transfer_syntax_uid.is_little_endian
        self.fp.is_implicit_VR = self.transfer_syntax_uid.is_implicit_VR
        # need to do this to advance the file to the correct point, at the beginning of the pixels
        self.metadata = dcmread(self.fp, stop_before_pixels = True)
        self.pixel_data_offset = self.fp.tell()
        self.fp.seek(self.pixel_data_offset, 0)
        # note that reading this tag is necessary to advance the file to correct position
        _ = TupleTag(self.fp.read_tag())
        # get basic offset table, to enable reading individual frames without loading entire image
        self.bot = self.get_bot(self.fp)
        self.first_frame = self.fp.tell()

    @staticmethod
    def get_bot(fp):
        """
        Reads the value of the Basic Offset Table. This table is used to access individual frames
        without loading the entire file into memory

        Args:
            fp (pydicom.filebase.DicomFile): pydicom DicomFile object

        Returns:
            list: Offset of each Frame of the Pixel Data element following the Basic Offset Table
        """
        # Skip Pixel Data element header
        pixel_data_offset = data_element_offset_to_value(fp.is_implicit_VR, 'OB')
        fp.seek(pixel_data_offset - 4, 1)
        _, basic_offset_table = get_frame_offsets(fp)
        first_frame_offset = fp.tell()
        fp.seek(first_frame_offset, 0)
        return basic_offset_table

    def get_image_shape(self, **kwargs):
        """
        Get the shape of the image.

        Returns:
            Tuple[int, int]: Shape of image (H, W)
        """
        return self.shape

    def get_thumbnail(self, size, **kwargs):
        raise NotImplementedError("DICOMBackend does not support thumbnail")

    def _index_to_coords(self, index):
        """
        convert from frame index to frame coordinates using image shape, frame_shape.
        Frames start at 0, go across the rows sequentially, use padding at edges.
        Coordinates are for top-left corner of each Frame.

        e.g. if the image is 100x100, and each frame is 10x10, then the top row will contain frames 0 through 9,
        second row will contain frames 10 through 19, etc.

        Args:
            index (int): index of frame

        Returns:
            tuple: corresponding coordinates
        """
        frame_i, frame_j = self.frame_shape
        # get which row and column we are in
        row_ix = index // self.n_cols
        col_ix = index % self.n_cols
        return row_ix * frame_i, col_ix * frame_j

    def _coords_to_index(self, coords):
        """
        convert from frame coordinates to frame index using image shape, frame_shape.
        Frames start at 0, go across the rows sequentially, use zero padding at edges.
        Coordinates are for top-left corner of each Frame.

        e.g. if the image is 100x100, and each frame is 10x10, then coordinate (10, 10) corresponds to frame index 11
        (second row, second column).

        Args:
            tuple (tuple): coordinates

        Returns:
            int: frame index
        """
        i, j = coords
        frame_i, frame_j = self.frame_shape
        # frame size must evenly divide coords, otherwise we aren't on a frame corner
        if i % frame_i or j % frame_j:
            raise ValueError(f"coords {coords} are not evenly divided by frame size {(frame_i, frame_j)}. "
                             f"Must provide coords at upper left corner of Frame.")

        row_ix = i / frame_i
        col_ix = j / frame_j

        index = (row_ix * self.n_cols) + col_ix
        return int(index)

    def extract_region(self, location, size=None, level=None, **kwargs):
        """
        Extract a single frame from the DICOM image.

        Args:
            location (Union[int, Tuple[int, int]]): coordinate location of top-left corner of frame, or integer index
                of frame.
            size (Union[int, Tuple[int, int]]): Size of each tile. May be a tuple of (height, width) or a
                single integer, in which case square tiles of that size are generated.
                Must be the same as the frame size.
            level (int): level from which to extract chunks. Level 0 is highest resolution.
                Must be 0 or None, since DICOMBackend does not support multiple levels.

        Returns:
            np.ndarray: image at the specified region
        """
        # check inputs first
        # check location
        if isinstance(location, tuple):
            frame_ix = self._coords_to_index(location)
        elif isinstance(location, int):
            frame_ix = location
        else:
            raise ValueError(f"Invalid location: {location}. Must be an int frame index or tuple of (i, j) coordinates")
        if frame_ix > self.n_frames:
            raise ValueError(f"location {location} invalid. Exceeds total number of frames ({self.n_frames})")
        # check level
        if level not in [None, 0]:
            raise ValueError("DICOMBackend does not support levels, please pass a level in [None, 0]")
        # check size
        if size:
            if type(size) is int:
                size = (size, size)
            if size != self.frame_shape:
                raise ValueError(f"Input size {size} must equal frame shape in DICOM image {self.frame_shape}")

        return self._read_frame(frame_ix)

    def _read_frame(self, frame_ix):
        """
        Reads and decodes the pixel data of one frame.
        Based on implementation from highDICOM: https://github.com/MGHComputationalPathology/highdicom

        Args:
            frame_ix (int): zero-based index of the frame

        Returns:
            np.ndarray: pixel data of that frame
        """
        frame_offset = self.bot[int(frame_ix)]
        self.fp.seek(self.first_frame + frame_offset, 0)
        try:
            stop_at = self.bot[frame_ix + 1] - frame_offset
        except IndexError:
            stop_at = None
        n = 0
        # A frame may comprised of multiple chunks
        chunks = []
        while True:
            tag = TupleTag(self.fp.read_tag())
            if n == stop_at or int(tag) == SequenceDelimiterTag:
                break
            length = self.fp.read_UL()
            chunks.append(self.fp.read(length))
            n += 8 + length

        frame_bytes = b''.join(chunks)

        decoded_frame_array = self._decode_frame(
            value = frame_bytes,
            rows = self.metadata.Rows,
            columns = self.metadata.Columns,
            samples_per_pixel = self.metadata.SamplesPerPixel,
            transfer_syntax_uid = self.metadata.file_meta.TransferSyntaxUID,
        )

        return decoded_frame_array

    @staticmethod
    def _decode_frame(value, transfer_syntax_uid, rows, columns,
                      samples_per_pixel, photometric_interpretation='RGB'):
        """
        Decodes the data of an individual frame.
        The pydicom library does currently not support reading individual frames.
        This solution inspired from HighDICOM creates a small dataset for the individual frame which
        can then be decoded using pydicom API.

        Args:
            value (bytes): Pixel data of a frame
            transfer_syntax_uid (str): Transfer Syntax UID
            rows (int): Number of pixel rows in the frame
            columns (int): Number of pixel columns in the frame
            samples_per_pixel (int): Number of samples per pixel
            photometric_interpretation (str): Photometric interpretation --currently only supporting RGB

        Returns:
            np.ndarray: decoded pixel data
        """
        filemetadata = Dataset()
        filemetadata.TransferSyntaxUID = transfer_syntax_uid
        dataset = Dataset()
        dataset.file_meta = filemetadata
        dataset.Rows = rows
        dataset.Columns = columns
        dataset.SamplesPerPixel = samples_per_pixel
        dataset.PhotometricInterpretation = photometric_interpretation
        image = Image.open(BytesIO(value))
        return np.asarray(image)

    def generate_tiles(self, shape, stride, pad, **kwargs):
        """
        Generator over tiles.
        For DICOMBackend, each tile corresponds to a frame.

        Args:
            shape (int or tuple(int)): Size of each tile. May be a tuple of (height, width) or a single integer,
                in which case square tiles of that size are generated. Must match frame size.
            stride (int): Ignored for DICOMBackend. Frames are yielded individually.
            pad (bool): How to handle tiles on the edges. If ``True``, these edge tiles will be zero-padded
                and yielded with the other chunks. If ``False``, incomplete edge chunks will be ignored.
                Defaults to ``False``.

        Yields:
            pathml.core.tile.Tile: Extracted Tile object
        """
        for i in range(self.n_frames):

            if not pad:
                # skip frame if it's in the last column
                if i % self.n_cols == (self.n_cols - 1):
                    continue
                # skip frame if it's in the last row
                if i >= (self.n_frames - self.n_cols):
                    continue

            frame_im = self.extract_region(location = i)
            coords = self._index_to_coords(i)
            frame_tile = pathml.core.tile.Tile(image = frame_im, coords = coords)
            yield frame_tile
