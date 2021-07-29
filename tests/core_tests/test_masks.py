"""
Copyright 2021, Dana-Farber Cancer Institute and Weill Cornell Medicine
License: GNU GPL 2.0
"""

import pytest
import numpy as np

import pathml.core.h5managers
from pathml.core import Masks, HESlide


@pytest.fixture
def emptymasks():
    slidedata = HESlide("tests/testdata/small_HE.svs")
    return Masks(h5manager=pathml.core.h5managers.h5pathManager(slidedata=slidedata))


@pytest.fixture
def smallmasks():
    slidedata = HESlide("tests/testdata/small_HE.svs")
    shape = (224, 224, 3)
    im = np.arange(np.product(shape)).reshape(shape)
    testmasks = Masks(
        h5manager=pathml.core.h5managers.h5pathManager(slidedata=slidedata),
        masks={"mask1": im, "mask2": im},
    )
    return testmasks


@pytest.mark.parametrize(
    "incorrect_input", ["string", True, 5, [5, 4, 3], {"dict": "testing"}]
)
def test_init_incorrect_input(incorrect_input):
    slidedata = HESlide("tests/testdata/small_HE.svs")
    with pytest.raises(ValueError):
        masks = Masks(
            h5manager=pathml.core.h5managers.h5pathManager(slidedata=slidedata),
            masks=incorrect_input,
        )


@pytest.mark.parametrize(
    "incorrect_input", ["string", True, [5, 4, 3], {"dict": "testing"}]
)
def test_add_get_incorrect_input(emptymasks, smallmasks, incorrect_input):
    with pytest.raises(ValueError):
        emptymasks.add(
            incorrect_input, np.arange(np.product((224, 224, 3))).reshape((224, 224, 3))
        )
        emptymasks.add("newmask", incorrect_input)
    with pytest.raises(KeyError):
        mask = smallmasks[incorrect_input]


@pytest.mark.parametrize(
    "incorrect_input", ["string", True, [5, 4, 3], {"dict": "testing"}]
)
def test_slice(smallmasks, incorrect_input):
    masks = smallmasks
    slices = [slice(2, 5)]
    test = masks.slice(slices)
    assert test[list(test.keys())[0]].shape == (3, 224, 3)
    with pytest.raises(Exception):
        test = masks.slice(incorrect_input)


@pytest.mark.parametrize(
    "incorrect_input", ["string", True, 5, [5, 4, 3], {"dict": "testing"}]
)
def test_remove(smallmasks, incorrect_input):
    masks = smallmasks
    masks.remove("mask1")
    with pytest.raises(KeyError):
        mask = masks["mask1"]
        masks.remove(incorrect_input)
