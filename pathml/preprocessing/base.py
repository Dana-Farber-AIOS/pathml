import os


class BasePreprocessor:
    """
    Base class for all preprocessors.
    """
    def apply(self, *args):
        """implement this method in each subclass"""
        raise NotImplementedError


class BaseSlideLoader(BasePreprocessor):
    """
    Loads slide from disk, returns :class:`~pathml.preprocessing.slide_data.SlideData` object
    """
    def apply(self, path):
        """
        Implement slide loader here.
        Can be basic wrapper that initializes slide class and calls ``load_data()``
        """
        raise NotImplementedError


class BaseSlidePreprocessor(BasePreprocessor):
    """
    Performs slide-level preprocessing.
    """
    def apply(self, data):
        """By default, does not do any transformation to input"""
        return data


class BaseTileExtractor(BasePreprocessor):
    """
    Extracts tiles from input image
    """
    def apply(self, data):
        """By default, does not do any transformation to input"""
        return data


class BaseTilePreprocessor(BasePreprocessor):
    """
    Performs tile-level preprocessing
    """
    def apply(self, data):
        """By default, does not do any transformations to input"""
        return data


class ImageTransform:
    """
    Transforms of the form image -> image
    """
    def apply(self, image):
        """Apply transform to input image"""
        raise NotImplementedError


class SegmentationTransform:
    """
    Transforms of the form image -> mask
    """
    def apply(self, image):
        """Apply transform to input image"""
        raise NotImplementedError


class MaskTransform:
    """
    Transforms of the form mask -> mask
    """
    def apply(self, mask):
        """Apply transform to input mask"""
        raise NotImplementedError


class BaseSlide:
    """
    Base class for slides.
    """
    def __init__(self, path, name=None):
        self.path = path
        if name:
            self.name = name
        else:
            basename = os.path.basename(path)
            name = os.path.splitext(basename)[0]
            self.name = name

    def read_image(self, path):
        """Read image from disk, using appropriate backend"""
        raise NotImplementedError

    def load_data(self, **kwargs):
        """Initialize a :class:`~pathml.preprocessing.slide_data.SlideData` object"""
        raise NotImplementedError

    def read_region(self, **kwargs):
        """Extract a region from the image. Implement for each slide type"""
        raise NotImplementedError

    def chunks(self, **kwargs):
        """Iterator over chunks. Implement for each slide type"""
        raise NotImplementedError


class Slide3d(BaseSlide):
    """Base class for volumetric slides"""
    pass


class Slide2d(BaseSlide):
    """Base class for 2-d slides"""
    pass


class RGBSlide(Slide2d):
    """Base class for RGB slides"""
    pass