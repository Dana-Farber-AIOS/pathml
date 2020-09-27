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
