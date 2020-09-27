"""testing for all transforms"""
import cv2
import numpy as np
import pytest

from pathml.preprocessing.transforms import (
    MedianBlur, GaussianBlur, BoxBlur, BinaryThreshold,
    MorphOpen, MorphClose, ForegroundDetection
)


@pytest.fixture
def image():
    return np.random.randint(low = 0, high = 256, size = (100, 100, 3), dtype = np.uint8)


@pytest.mark.parametrize('transform', [
    MedianBlur(), GaussianBlur(), BoxBlur(), MorphOpen(), MorphClose()
])
def test_image_transform(image, transform):
    out = transform.apply(image)
    assert out.shape == image.shape
    assert out.dtype == image.dtype


@pytest.mark.parametrize('ksize', [3, 7, 21])
def test_median_blur_large_kernel(image, ksize):
    # for ksize > 5, dtype must be uint8
    out = MedianBlur(kernel_size = ksize).apply(image)
    assert out.shape == image.shape
    assert out.dtype == image.dtype


@pytest.mark.parametrize('transform', [
    BinaryThreshold()
])
def test_segmentation_output_shape(image, transform):
    """check Segmentation transforms to ensure that output is same (height, width) as input"""
    out = transform.apply(image)
    assert out.shape == image.shape[0:2]
    assert out.dtype == np.uint8


@pytest.fixture
def mask():
    """
    mask that has a central square of ones with a circle of zeroes inside it
    """
    out = np.zeros((100, 100), dtype = np.uint8)
    center = np.ones((50, 50))
    center_circle = cv2.getStructuringElement(shape = cv2.MORPH_ELLIPSE, ksize = (25, 25))
    center[12:37, 12:37] -= center_circle
    out[25:75, 25:75] = center
    return out


@pytest.mark.parametrize('transform', [
    ForegroundDetection()
])
def test_mask_transform_output_shape(mask, transform):
    """check MaskTransform to ensure that output is same shape as input"""
    out = transform.apply(mask)
    assert out.shape == mask.shape
    assert out.dtype == np.uint8


def test_foreground_detection_outer_contour_only(mask):
    out = ForegroundDetection(outer_contours_only = True).apply(mask)
    assert out.shape == mask.shape
    assert out.dtype == np.uint8
