import openslide
import cv2
import os
import numpy as np

from pathml.preprocessing.slide_data import SlideData

# multiparametric imports
import javabridge
import bioformats.formatreader as biordr
from bioformats.formatreader import ImageReader
from bioformats.metadatatools import createOMEXMLMetadata


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

class MultiparametricSlide(BaseSlide):
    """
    Class for multiparametric if/ihc including: CODEX, Hyperion, Vectra Polaris    
    Depends on cellprofiler/python-bioformats https://github.com/CellProfiler/python-bioformats

    Dependencies:
    sudo apt-get install default-jdk
    pip install python-bioformats

    python-bioformats wraps ome bioformats java library
    parses pixel and metadata of proprietary formats
    converts all formats to OME-TIFF
    please cite: https://pubmed.ncbi.nlm.nih.gov/20513764/
<<<<<<< HEAD
=======
    java code is compiled one time into platform independent bite code, making this more distributable
    https://ilovesymposia.com/2014/08/10/read-microscopy-images-to-numpy-arrays-with-python-bioformats/
>>>>>>> aa65864ab84a290463e1200e6836e2b074be5c8a
    """

    def __init__(self, path, name=None):
        super().__init__(path, name)

        # this field is too specific to openslide
        self.slide = None 
        self.path = path
    
    def __sizeof__(self, name=None):
        # init java virtual machine
        javabridge.start_vm(class_path=bioformats.JARS)
        self.slide = self._read_bioformats(path) 

        # java maximum array size of 2GB constrains image size
        # we need to check if we need to allocate multiple arrays of 2GB
        # read image dimensions from metadata
        ImageReader = bioformats.formatreader.make_image_reader_class()
        FormatTools = bioformats.formatreader.make_format_tools_class()
        reader = ImageReader()
        omeMeta = createOMEXMLMetadata()
        reader.setMetadataStore(omeMeta)
        reader.setId(self.path)
        sizex, sizey, sizez, sizec = reader.getSizeX(), reader.getSizeY(), reader.getSizeZ(), reader.getSizeC()
        return((sizex,sizey,sizez,sizec))

    def __repr__(self):
        return f"MultiparametricSlide(path={self.path}, name={self.name})"

    def load_data(self):
        """
        Load slide using ``python-bioformats``, and initialize a :class:`~pathml.preprocessing.slide_data.SlideData` object
        
        """
        javabridge.start_vm(class_path=bioformats.JARS)

        # cast to ome-tiff
        ImageReader = bioformats.formatreader.make_image_reader_class()
        FormatTools = bioformats.formatreader.make_format_tools_class()
        reader = ImageReader()
        reader.setId(path)
        data = reader.openByes(0)
        data = bioformats.formatreader.load_using_bioformats(path, rescale=False)

        # init java virtual machine
        javabridge.start_vm(class_path=bioformats.JARS)

        # TODO: Handling for images > 2GB

        # load ome-tiff array
        data = bioformats.formatreader.load_using_bioformats(self.path, rescale=False)

        # ome-tiff array to ndarray
        image_array = np.asarray(data, dtype = np.uint8) 
        out = SlideData(wsi = self, image = image_array)
        return out 
        # ome-tiff to ndarray

        image_array = 
        out = SlideData(wsi = self, image = image_array)
        return 
