from warnings import warn
import numpy as np
import pytest

from pathml.preprocessing.multiparametricslide import MultiparametricSlide

try:
    import javabridge
    import bioformats
except ImportError:
    warn(
            """MultiparametricSlide requires a jvm to interface with java bioformats library.
            See: https://pythonhosted.org/javabridge/installation.html. You can install using:
                
                sudo apt-get install openjdk-8-jdk
                pip install javabridge
                pip install python-bioformats
            """
    )


@pytest.fixture
def smalltif_example():
    path = "tests/testdata/smalltif.tif"
    javabridge.start_vm(class_path=bioformats.JARS)
    data = bioformats.formatreader.load_using_bioformats(path, rescale=False)
    im_np = np.asarray(data, dtype = np.uint8)
    return im_np


def test_multiparametric_slide(smalltif_example):
    wsi = MultiparametricSlide(path = "tests/testdata/smalltif.tif")
    assert wsi.name == "smalltif"
    assert wsi.path == "tests/testdata/smalltif.tif"
    slide_data = wsi.load_data()
    assert np.array_equal(slide_data.image, smalltif_example)
