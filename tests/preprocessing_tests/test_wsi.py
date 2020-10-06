import openslide
import pytest
import numpy as np
import cv2

from pathml.preprocessing.wsi import HESlide, MultiparametricSlide

import javabridge
import bioformats.formatreader as biordr
from bioformats.formatreader import ImageReader


@pytest.fixture
def openslide_example():
    im = openslide.open_slide("tests/testdata/CMU-1-Small-Region.svs")
    im_image = im.read_region(level = 0, location=(0,0), size=im.level_dimensions[0])
    im_np = np.asarray(im_image)
    im_np_rgb = cv2.cvtColor(im_np, cv2.COLOR_RGBA2RGB)
    return im_np_rgb

# TODO: make test less depeendent on javabridge? want to test bioformats also
# filesize too large for git repo
def confocal_zseries_multichannel_example():
    path = "tests/testdata/zeissconfocal3channelzseries.czi"
    javabridge.start_vm(class_path=bioformats.JARS)
    data = bioformats.formatreader.load_using_bioformats(path, rescale=False)
    im_np = np.asarray(data, dtype = np.uint8) 
    return im_np 

def smalltif_example():
    path = "tests/testdata/smalltif.tif"
    javabridge.start_vm(class_path=bioformats.JARS)
    data = bioformats.formatreader.load_using_bioformats(path, rescale=False)
    im_np = np.asarray(data, dtype = np.uint8) 
    return im_np 

def test_HE_slide(openslide_example):
    wsi = HESlide(path = "tests/testdata/CMU-1-Small-Region.svs")
    assert wsi.name == "CMU-1-Small-Region"
    assert wsi.path == "tests/testdata/CMU-1-Small-Region.svs"
    slide_data = wsi.load_data(level = 0, location = (0, 0), size = None)
    assert np.array_equal(slide_data.image, openslide_example)
    slide_data2 = wsi.load_data(level = 0, location = (200, 200), size = (200, 200))
    assert np.array_equal(slide_data2.image, openslide_example[200:400, 200:400, :])

def test_multiparametric_slide(confocal_zseries_multichannel_example):
    wsi = MultiparametricSlide(path = "/tests/testdata/zeissconfocal3channelzseries.czi"
    assert wsi.name == "zeissconfocal3channelzseries"
    assert wsi.path == "/tests/testdata/zeissconfocal3channelzseries.czi"
    slide_data = wsi.load_data()
    assert np.array_equal(slide_data.image, confocal_zseries_multichannel_example)
