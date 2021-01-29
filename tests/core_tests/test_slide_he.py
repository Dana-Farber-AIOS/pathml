from pathml.core.slide_he import HESlide, RGBSlide
from pathml.core.slide_data import SlideData


def test_slide_he():
    slide = HESlide("./../testdata/small_HE.svs", name = "testing")
    assert type(slide) == SlideData
    raise NotImplementedError