from warnings import warn
import sys
import os
import numpy as np

from pathml.preprocessing.slide_data import SlideData
from pathml.preprocessing.base import Slide2d

import bioformats
import javabridge
import bioformats.formatreader as biordr
from bioformats.formatreader import ImageReader
from bioformats.metadatatools import createOMEXMLMetadata

def check_mac_java_home():
    is_mac = sys.platform == 'darwin'
    if is_mac and "JAVA_HOME" not in os.environ:
        warn("""
            It looks like you are using a mac, and the $JAVA_HOME variable was not found in your environment.
            This means that the javabridge may not work correctly!

            Try these steps to resolve:
                1. Find the path to JAVA SDK 8: 
                    os.system('/usr/libexec/java_home -V')
                2. export that path to JAVA_HOME:
                    os.environ["JAVA_HOME"] = '/Library/Java/JavaVirtualMachines/jdk1.8.0_261.jdk/Contents/Home'
                    (the path on your machine may be different)
            """)

check_mac_java_home()


class MultiparametricSlide2d(Slide2d):
    """
    Represents multiparametric IF/IHC images. Backend based on ``bioformats``.

    `python-bioformats <https://github.com/CellProfiler/python-bioformats>`_ wraps ome bioformats java library,
    parses pixel and metadata of proprietary formats, and
    converts all formats to OME-TIFF. Please cite: https://pubmed.ncbi.nlm.nih.gov/20513764/
    """

    def __init__(self, path):
        super().__init__(path)
        self.path = path

        # init java virtual machine
        javabridge.start_vm(class_path=bioformats.JARS)
        # java maximum array size of 2GB constrains image size
        ImageReader = bioformats.formatreader.make_image_reader_class()
        FormatTools = bioformats.formatreader.make_format_tools_class()
        reader = ImageReader()
        omeMeta = createOMEXMLMetadata()
        reader.setMetadataStore(omeMeta)
        reader.setId(self.path)
        sizex, sizey, sizez, sizec = reader.getSizeX(), reader.getSizeY(), reader.getSizeZ(), reader.getSizeC()
        self.imsize = sizex*sizey*sizez*sizec
    
    def __repr__(self):
        return f"MultiparametricSlide2d(path={self.path}, name={self.name})"

    def load_data(self):
        """
        Load slide using ``python-bioformats``, and initialize a :class:`~pathml.preprocessing.slide_data.SlideData` object
        """
        if self.imsize > 2147483647:
            raise Exception(f"Java arrays allocate maximum 32 bits (~2GB). Image size is {self.imsize}")  

        # init java virtual machine
        javabridge.start_vm(class_path=bioformats.JARS)

        # TODO: Handling for images > 2GB

        # load ome-tiff array
        data = bioformats.formatreader.load_using_bioformats(self.path, rescale=False)

        # ome-tiff array to ndarray
        image_array = np.asarray(data, dtype = np.uint8) 
        out = SlideData(wsi = self, image = image_array)
        return out
