from pathml.preprocessing.pipeline import Pipeline
from pathml.preprocessing.base import BaseSlideLoader, BaseSlidePreprocessor, BaseTileExtractor, BaseTilePreprocessor


def test_pipeline():
    # define dummy preprocessors amd dummy data class for testing
    # we are testing the pipeline, not the preprocessors themselves
    class MySlideLoader(BaseSlideLoader):
        def apply(self, path):
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

    test_data = my_pipeline.run(path = "dummy/path")

    assert test_data["slide_loaded"]
    assert test_data["slide_preprocessed"]
    assert test_data["tiles_extracted"]
    assert test_data["tiles_preprocessed"]
