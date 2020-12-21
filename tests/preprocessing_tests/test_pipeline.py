import pytest

from pathml.preprocessing.pipeline import Pipeline
from pathml.preprocessing.base import (
    BaseSlideLoader, BaseSlidePreprocessor, BaseTileExtractor, BaseTilePreprocessor, BasePipeline)


@pytest.fixture
def dummy_pipeline():
    class DummyPipeline(BasePipeline):
        """create a dummy pipeline for testing"""
        def __init__(self):
            self.record = {}

        def run_single(self, slide):
            # add an entry to record so we know the slide was touched by this pipeline
            self.record[slide.name] = True
            return slide
    return DummyPipeline()


def test_base_pipeline_single_slide(example_he_slide, dummy_pipeline):
    _ = dummy_pipeline.run(example_he_slide)
    assert dummy_pipeline.record[example_he_slide.name]


# @pytest.mark.parametrize("n_jobs", [-1, None, 1, 3, 20])
@pytest.mark.parametrize("n_jobs", [None, 1])
def test_base_pipeline_dataset(example_slide_dataset, dummy_pipeline, n_jobs):
    _ = dummy_pipeline.run(target = example_slide_dataset, n_jobs = n_jobs)
    for slide in example_slide_dataset:
        assert dummy_pipeline.record[slide.name]


def test_pipeline(example_he_slide):
    # define dummy preprocessors amd dummy data class for testing
    # we are testing the pipeline, not the preprocessors themselves
    class MySlideLoader(BaseSlideLoader):
        def apply(self, slide):
            data = {"slide_loaded": True}
            return data

    class MySlidePreprocessor(BaseSlidePreprocessor):
        def apply(self, data):
            data["slide_preprocessed"] = True
            return data

    class MyTileExtractor(BaseTileExtractor):
        def apply(self, data):
            data["tiles_extracted"] = True
            return data

    class MyTilePreprocessor(BaseTilePreprocessor):
        def apply(self, data):
            data["tiles_preprocessed"] = True
            return data

    # now create pipeline to string these together and check that it worked properly
    my_pipeline = Pipeline(
        slide_loader = MySlideLoader(),
        slide_preprocessor = MySlidePreprocessor(),
        tile_extractor = MyTileExtractor(),
        tile_preprocessor = MyTilePreprocessor()
    )

    test_data = my_pipeline.run_single(example_he_slide)

    assert test_data["slide_loaded"]
    assert test_data["slide_preprocessed"]
    assert test_data["tiles_extracted"]
    assert test_data["tiles_preprocessed"]
