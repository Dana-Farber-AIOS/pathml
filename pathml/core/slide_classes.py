from pathml.core.slide_data import SlideData
from pathml.core.slide_backends import OpenSlideBackend, BioFormatsBackend


class RGBSlide(SlideData):
    """
    Class for any RGB slide. Uses OpenSlide backend.
    Refer to :class:`~pathml.core.slide_data.SlideData` for full documentation.
    """
    def __init__(self, *args, **kwargs):
        kwargs["slide_backend"] = OpenSlideBackend
        super().__init__(*args, **kwargs)


class HESlide(RGBSlide):
    """
    Class for any H&E slide. Uses OpenSlide backend.
    Refer to :class:`~pathml.core.slide_data.SlideData` for full documentation.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class MultiparametricSlide(SlideData):
    """
    Class for any multiparametric slide. Uses BioFormats backend.
    Refer to :class:`~pathml.core.slide_data.SlideData` for full documentation.
    """
    def __init__(self, *args, **kwargs):
        kwargs["slide_backend"] = BioFormatsBackend
        super().__init__(*args, **kwargs)
