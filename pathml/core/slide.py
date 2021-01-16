import os


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