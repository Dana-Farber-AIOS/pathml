import os
import concurrent.futures

from pathml.datasets.base import BaseDataset


# Base classes for slides

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


# Base classes for Pipelines and Preprocessors for composing Pipelines

class BasePipeline:
    """
    Base class for Pipeline objects
    """

    def run_single(self, slide, **kwargs):
        """
        Define pipeline here for a single BaseSlide object
        """
        raise NotImplementedError

    def run(self, target, n_jobs=-1, **kwargs):
        """
        Execute pipeline on a single input or an entire dataset.
        If running on a dataset, this will use multiprocessing to distribute computation
        across available cores.

        Args:
            target (BaseSlide or BaseDataset): Input on which to execute pipeline.
            n_jobs (int): number of processes to use. If 1 or None, then no multiprocessing is used.
                If -1, then all available cores are used. Defaults to -1.
            kwargs (dict): Additional arguments passed to each individual call of self.run_single(target)
        """
        if isinstance(target, BaseSlide):
            # only need to run on a single input slide
            self.run_single(slide = target, **kwargs)
        elif isinstance(target, BaseDataset):
            # run on all elements in the dataset
            if n_jobs == 1 or n_jobs is None:
                # don't use multiprocessing in this case
                for slide in target:
                    self.run_single(slide = slide, **kwargs)

            else:
                if n_jobs == -1:
                    n_jobs = os.cpu_count()
                else:
                    assert isinstance(n_jobs, int), f"Input n_jobs {n_jobs} not valid. Must be None or an int"

                with concurrent.futures.ProcessPoolExecutor(max_workers = n_jobs) as executor:
                    for slide in target:
                        executor.submit(self.run_single, slide = slide, **kwargs)

from pathml.preprocessing.masks import Masks


class BasePreprocessor:
    """
    Base class for all preprocessors.
    """

    def apply(self, *args):
        """implement this method in each subclass"""
        raise NotImplementedError


class BaseSlideLoader(BasePreprocessor):
    """
    Ingests a :class:`~pathml.preprocessing.base.BaseSlide` object and returns a
    :class:`~pathml.preprocessing.slide_data.SlideData` object
    """

    def apply(self, slide):
        """
        Implement slide loader here.
        Can be basic wrapper that calls ``load_data()``, or something more complex
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


# Base classes for Transforms

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
