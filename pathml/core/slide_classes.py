from pathml.core.slide_data import SlideData
from pathml.core.slide_backends import OpenSlideBackend


class RGBSlide(SlideData):
    """
    Class for any RGB slide. Uses OpenSlide backend.
    """
    def __init__(self, filepath, name=None, masks=None, labels=None, tiles=None, h5=None):
        super().__init__(
            filepath = filepath,
            name = name,
            slide = OpenSlideBackend(filepath),
            masks = masks,
            tiles = tiles,
            labels = labels,
            h5 = h5
        )


class HESlide(RGBSlide):
    """
    Class for H&E stained slides. Uses OpenSlide backend.
    """
    def __init__(self, filepath, name=None, masks=None, labels=None, tiles=None, h5=None):
        super().__init__(
            filepath = filepath,
            name = name,
            slide = OpenSlideBackend(filepath),
            masks = masks,
            tiles = tiles,
            labels = labels,
            h5 = h5
        )
