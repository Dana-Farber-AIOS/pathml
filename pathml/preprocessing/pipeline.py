from pathml.preprocessing.base import BaseSlideLoader, BaseSlidePreprocessor, BaseTileExtractor, BaseTilePreprocessor, \
    BasePipeline, BaseSlide
import pickle

class Pipeline(BasePipeline):
    """
    Convenience object for running preprocessing pipelines that are based on four main steps:
        1. Ingesting a Slide object and loading into SlideData object
        2. Performing slide-level preprocessing
        3. Extracting tiles
        4. performing tile-level preprocessing

    Each step is performed by a Preprocessor object.
    Pipelines can be composed modularly from default and custom Preprocessors.

    This structure of pipeline is useful in many cases, but it is not strictly necessary to follow this four-step
    process for all preprocessing pipelines. For complete control over custom pipelines,
    inherit directly from :class:`~pathml.preprocessing.base.BasePipeline` and define the ``run_single()`` method.

    :param slide_loader: preprocessor which ingests a BaseSlide object
    :type slide_loader: :class:`~pathml.preprocessing.base.BaseSlideLoader`
    :param slide_preprocessor: preprocessor to apply on slide level
    :type slide_preprocessor: :class:`~pathml.preprocessing.base.BaseSlidePreprocessor`
    :param tile_extractor: preprocessor to extract tiles
    :type tile_extractor: :class:`~pathml.preprocessing.base.BaseTileExtractor`
    :param tile_preprocessor: preprocessor to run on each tile
    :type tile_preprocessor: :class:`~pathml.preprocessing.base.BaseTilePreprocessor`
    """

    def __init__(self, slide_loader, slide_preprocessor, tile_extractor, tile_preprocessor):
        assert isinstance(slide_loader, BaseSlideLoader), \
            f"slide_loader is of type {type(slide_loader)}. Must inherit from BaseSlideLoader"
        assert isinstance(slide_preprocessor, BaseSlidePreprocessor), \
            f"slide_preprocessor is of type {type(slide_preprocessor)}. Must inherit from BaseSlidePreprocessor"
        assert isinstance(tile_extractor, BaseTileExtractor), \
            f"tile_extractor is of type {type(tile_extractor)}. Must inherit from BaseTileExtractor"
        assert isinstance(tile_preprocessor, BaseTilePreprocessor), \
            f"tile_preprocessor is of type {type(tile_preprocessor)}. Must inherit from BaseTilePreprocessor"

        self.slide_loader = slide_loader
        self.slide_preprocessor = slide_preprocessor
        self.tile_extractor = tile_extractor
        self.tile_preprocessor = tile_preprocessor

    def load_slide(self, slide):
        """Run only the slide_loader component of the pipeline"""
        assert isinstance(slide, BaseSlide), f"Input slide type {type(slide)} invalid. Must inherit from BaseSlide"
        data = self.slide_loader.apply(slide)
        return data

    def run_slide_level(self, data):
        """Run only the slide_preprocessor component of the pipeline"""
        data = self.slide_preprocessor.apply(data)
        return data

    def extract_tiles(self, data):
        """Run only the tile_extractor component of the pipeline"""
        data = self.tile_extractor.apply(data)
        return data

    def run_tile_level(self, data):
        """Run only the tile_preprocessor component of the pipeline"""
        data = self.tile_preprocessor.apply(data)
        return data

    def run_single(self, slide):
        """
        Run full preprocessing pipeline

        :param slide: input Slide object
        :type slide: :class:`~pathml.preprocessing.base.BaseSlide`
        :return: :class:`~pathml.preprocessing.slide_data.SlideData` object resulting from running full pipeline on
            input image
        """
        data = self.load_slide(slide)
        data = self.run_slide_level(data)
        data = self.extract_tiles(data)
        data = self.run_tile_level(data)
        return data

    def save(self, filename):
        """
        save pipeline by writing them to disk
        :param filename: save path on disk
        :type path: str
        :return: string indicated file saved to above path
        """
        pickle.dump(self, open(filename, "wb"))
        return filename