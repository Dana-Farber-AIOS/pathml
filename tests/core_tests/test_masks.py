import pytest
import numpy as np

from pathml.core.masks import Masks


@pytest.fixture
def emptymasks():
    return Masks()


@pytest.fixture
def smallmasks():
    shape = (224, 224, 3)
    im = np.arange(np.product(shape)).reshape(shape)
    testmasks = Masks({'mask1':im, 'mask2':im})
    return testmasks


@pytest.mark.parametrize("incorrect_input", ["string", True, 5, [5, 4, 3], {"dict": "testing"}])
def test_init_incorrect_input(incorrect_input):
    with pytest.raises(ValueError):
        masks = Masks(incorrect_input)


@pytest.mark.parametrize("incorrect_input", ["string", True, 5, [5, 4, 3], {"dict": "testing"}])
def test_add_get_incorrect_input(emptymasks, smallmasks, incorrect_input):
    masks = emptymasks
    with pytest.raises(ValueError):
        masks.add(incorrect_input, np.arange(np.product((224,224,3))).reshape((224,224,3)))
        masks.add('newmask', incorrect_input)
    masks = smallmasks
    with pytest.raises(KeyError):
        mask = masks[incorrect_input]


@pytest.mark.parametrize("incorrect_input", ["string", True, [5, 4, 3], {"dict": "testing"}])
def test_slice(smallmasks, incorrect_input):
    masks = smallmasks
    slices = [slice(2,5)]
    test = masks.slice(slices)
    assert test.h5manager.shape == (3,224,3)
    with pytest.raises(Exception):
        test = masks.slice(incorrect_input)


@pytest.mark.parametrize("incorrect_input", ["string", True, 5, [5, 4, 3], {"dict": "testing"}])
def test_remove(smallmasks, incorrect_input):
    masks = smallmasks
    masks.remove('mask1')
    with pytest.raises(KeyError):
        mask = masks['mask1']
        masks.remove(incorrect_input)
