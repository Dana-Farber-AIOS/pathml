"""
Copyright 2021, Dana-Farber Cancer Institute and Weill Cornell Medicine
License: GNU GPL 2.0
"""

from .masks import Masks
from .h5path import read, read_dicom, read_bioformats, read_openslide, read_h5path
from .slide_backends import OpenSlideBackend, BioFormatsBackend, DICOMBackend
from .slide_data import SlideData, HESlide
from .slide_dataset import SlideDataset
from .tile import Tile
from .tiles import Tiles
from .types import SlideType
from . import types
