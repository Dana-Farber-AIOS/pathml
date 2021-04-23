import cv2
import numpy as np
from io import BytesIO
from PIL import Image
from pathml.preprocessing.base import RGBSlide
from pathml.preprocessing.slide_data import SlideData
from pathml.preprocessing.utils import pil_to_rgb
import numpy as np
from pydicom.dataset import Dataset
from pydicom.encaps import get_frame_offsets
from pydicom.filebase import DicomFile
from pydicom.filereader import (
    data_element_offset_to_value,
    dcmread,
    read_file_meta_info,
)
from pydicom.uid import (
    ExplicitVRLittleEndian,
    ImplicitVRLittleEndian,
    JPEG2000Lossless,
    JPEGBaseline,
    UID,
    RLELossless,
)
from pydicom.pixel_data_handlers.numpy_handler import unpack_bits
from pydicom.tag import TupleTag, ItemTag, SequenceDelimiterTag


def decode_individual_frame(value, transfer_syntax_uid, rows, columns, samples_per_pixel, photometric_interpretation = 'RGB'):
    """Decodes  data of an individual frame.
    Parameters
    ----------
    value: bytes
        Pixel data of a frame 
    transfer_syntax_uid: str
        Transfer Syntax UID
    rows: int
        Number of pixel rows in the frame
    columns: int
        Number of pixel columns in the frame
    samples_per_pixel: int
        Number of samples per pixel
    photometric_interpretation: str
        Photometric interpretation --currently only supporting RGB

    Returns
    -------
    pixel data: numpy.ndarray

    """

    # The pydicom library does currently not support reading individual frames.
    # This solution inspired from HighDICOM creates a small dataset for the individual frame which can then be decoded using pydicom API.
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


def get_bot(fp: DicomFile, number_of_frames):
    """reads the value of the Basic Offset Table
    Parameters
    ----------
    fp: pydicom.filebase.DicomFile

    number_of_frames: int
        Number of frames contained in the Pixel Data element of the DICOM metadata
    Returns
    -------
    List
        Offset of each Frame of the Pixel Data
        element following the Basic Offset Table

    """
    tag = TupleTag(fp.read_tag())

    # Skip Pixel Data element header
    pixel_data_offset = data_element_offset_to_value(fp.is_implicit_VR, 'OB' )
    fp.seek(pixel_data_offset - 4, 1)
    _is_empty, basic_offset_table = get_frame_offsets(fp)
    

    first_frame_offset = fp.tell()

    fp.seek(first_frame_offset, 0)

    return basic_offset_table

class DICOMSlide(object):

    """Reader for DICOM datasets representing Image Information Entities.
    It provides efficient access to individual Frame items contained in the
    Pixel Data element without loading the entire element into memory.
    Attributes
    ----------
    filename: str
        Path to the DICOM Part10 file on disk
    Examples
    --------

    """

    def __init__(self, filename):
        """
        Parameters
        ----------
        filename: str
            Path to a DICOM file containing a dataset of a whole slide image
        """
        self.filename = filename
        file_meta = read_file_meta_info(self.filename)
        transfer_syntax_uid = UID(file_meta.TransferSyntaxUID)
        self.fp = DicomFile(str(self.filename), mode='rb')
        self.fp.is_little_endian = transfer_syntax_uid.is_little_endian
        self.fp.is_implicit_VR = transfer_syntax_uid.is_implicit_VR
        
        self.metadata = dcmread(self.fp, stop_before_pixels=True)
        self.pixel_data_offset = self.fp.tell()
        self.number_of_frames = int(self.metadata.NumberOfFrames)
        tag = TupleTag(self.fp.read_tag())

        
        self.fp.seek(self.pixel_data_offset, 0)

        
        transfer_syntax_uid = self.metadata.file_meta.TransferSyntaxUID
        self.bot = get_bot(self.fp, number_of_frames=self.number_of_frames)

        self.first_frame = self.fp.tell()       


    def read_individual_pixel(self, index):
        """Reads the raw pixel data of an individual frame item.
        Parameters
        ----------
        index: int
            Zero-based frame index
        Returns
        -------
        bytes
            Pixel data of a specific frame.

        """

        frame_offset = self.bot[index]
        self.fp.seek(self.first_frame + frame_offset, 0)
        stop_at = self.bot[index + 1] - frame_offset
        n = 0
            # A frame may comprised of multiple chunks
        chunks = []
        while True:
            tag = TupleTag(self.fp.read_tag())
            if n == stop_at or int(tag) == SequenceDelimiterTag:
                break
            length = self.fp.read_UL()
            chunks.append(self.fp.read(length))
            n += 4 + 4 + length
        individual_frame_data = b''.join(chunks)


        return individual_frame_data

    def read_frame(self, index):
        """Reads and decodes the pixel data of one frme
        Parameters
        ----------
        index: int
            Zero-based frame index
        Returns
        -------
        numpy.ndarray
            Array of decoded pixels of the frame with shape (Rows x Columns)


        """
        individual_frame_data = self.read_individual_pixel(index)

        decoded_frame_array = decode_individual_frame(
            individual_frame_data,
            rows=self.metadata.Rows,
            columns=self.metadata.Columns,
            samples_per_pixel=self.metadata.SamplesPerPixel,
            transfer_syntax_uid=self.metadata.file_meta.TransferSyntaxUID,
        )

        return decoded_frame_array

    def chunks(self):
        """Generator over chunks. Useful for processing the image in pieces, avoiding having to load the entire image
        at full-resolution.

        Yields:
            np.ndarray: Extracted RGB chunk of dimension (size, size, 3)
        """
        slide = dcmread(self.filename)
        pixel_data_frame = pydicom.encaps.generate_pixel_data_frame(slide.PixelData)
        for frame in pixel_data_frame:
            yield np.array(Image.open(BytesIO(frame))) 

    def get_thumbnail (self):
        """Read image from disk, using appropriate backend"""
        raise NotImplementedError

    def get_image_shape(self, shape_type="TotalPixelMatrix"):
        """Provides the shape of various elements within the DICOM image

        return:
            
        """
        if shape_type == "TotalPixelMatrix":
            return (self.metadata.TotalPixelMatrixRows, self.metadata.TotalPixelMatrixColumns)
        if shape_type == "Frame":
            return (self.metadata.Rows, self.metadata.Columns)