import os
import numpy as np

from pathml.preprocessing.slide_data import SlideData
from pathml.preprocessing.wsi import BaseSlide 

try:
    import bioformats
    import javabridge
    import bioformats.formatreader as biordr
    from bioformats.formatreader import ImageReader
    from bioformats.metadatatools import createOMEXMLMetadata
except ImportError:
    warn(
        """MultiparametricSlide requires a jvm to interface with java bioformats library that is not installed by default.
        
    To use MultiparametricSlide, install the following in your conda environment:
        
    https://pythonhosted.org/javabridge/installation.html
    sudo apt-get install default-jdk
    pip install javabridge
    pip install python-bioformats
    """
    )

class MultiparametricSlide(BaseSlide):
    """
    Represents multiparametric if/ihc images
    Depends on cellprofiler/python-bioformats https://github.com/CellProfiler/python-bioformats

    Dependencies:
    sudo apt-get install default-jdk
    pip install python-bioformats

    python-bioformats wraps ome bioformats java library
    parses pixel and metadata of proprietary formats
    converts all formats to OME-TIFF
    please cite: https://pubmed.ncbi.nlm.nih.gov/20513764/
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
        return f"MultiparametricSlide(path={self.path}, name={self.name})"

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
