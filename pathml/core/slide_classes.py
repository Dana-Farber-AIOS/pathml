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


class IHCSlide(RGBSlide):
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


class VectraSlide(MultiparametricSlide):
    """
    Class for data in Vectra (Polaris) format.

    This class enables transformations in * * 
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class CODEXSlide(MultiparametricSlide):
    """
    Class for data in Akoya bioscience CODEX format. 
    Expects the following filesystem:
        
    This class enables transforms in *link to location in docs*. 
    This class enables CODEX pipeline.

    # TODO:
        hierarchical biaxial gating (flow-style analysis) 
        KNN/FLANN/leiden clustering
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

