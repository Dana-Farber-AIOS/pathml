from pathml.core.slide_he import HESlide
from pathml.preprocessing.pipeline import Pipeline
from pathml.preprocessing.transforms import GaussianBlur, TissueDetectionHE


def test_pipeline_1():
    slide = HESlide("../testdata/small_HE.svs")
    pipeline = Pipeline([
        GaussianBlur(),
        TissueDetectionHE()
    ])

    slide.run(pipeline)

    # check that the tiles were stored in h5

    raise NotImplementedError