"""
Copyright 2021, Dana-Farber Cancer Institute and Weill Cornell Medicine
License: GNU GPL 2.0
"""

from .masks import Masks
from .slide_backends import BioFormatsBackend, DICOMBackend, OpenSlideBackend
from .slide_data import (
    CODEXSlide,
    HESlide,
    IHCSlide,
    MultiparametricSlide,
    SlideData,
    VectraSlide,
)
from .slide_dataset import SlideDataset
from .slide_types import SlideType, types
from .tile import Tile
from .tiles import Tiles
from .h5managers import h5pathManager
