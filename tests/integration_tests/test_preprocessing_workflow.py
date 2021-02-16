from pathml.core.slide_classes import HESlide
from pathml.preprocessing.pipeline import Pipeline
from pathml.preprocessing.transforms import BoxBlur, TissueDetectionHE


def test_pipeline_1(tmp_path):
    slide = HESlide("tests/testdata/small_HE.svs")
    pipeline = Pipeline([
        BoxBlur(kernel_size = 15),
        TissueDetectionHE(mask_name = "tissue")
    ])

    slide.run(pipeline)

    slide.tiles.write(out_dir = tmp_path, filename = "slidetiles.h5")
