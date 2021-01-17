import pytest

from pathml.preprocessing.transforms import StainNormalizationHE


@pytest.mark.parametrize("target", ["normalize", "eosin", "hematoxylin"])
def test_stain_normalization_he_target(example_he_tile, target):
    normalizer = StainNormalizationHE(target = target)
    out = normalizer.apply(example_he_tile)
    assert out.shape == example_he_tile.shape
    assert out.dtype == example_he_tile.dtype


@pytest.mark.parametrize("method", ["macenko", "vahadane"])
def test_stain_normalization_he_method(example_he_tile, method):
    normalizer = StainNormalizationHE(stain_estimation_method = method)
    out = normalizer.apply(example_he_tile)
    assert out.shape == example_he_tile.shape
    assert out.dtype == example_he_tile.dtype


def test_fitting_to_reference(example_he_tile):
    normalizer = StainNormalizationHE(stain_estimation_method = "macenko", target = "normalize")
    normalizer.fit_to_reference(example_he_tile)
    out = normalizer.apply(example_he_tile)
    assert out.shape == example_he_tile.shape
    assert out.dtype == example_he_tile.dtype
