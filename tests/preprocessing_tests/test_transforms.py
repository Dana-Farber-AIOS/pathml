"""testing for all transforms"""
import cv2
import numpy as np
import pytest
import openslide

from pathml.core.tile import Tile
from pathml.core.masks import Masks

from pathml.preprocessing.transforms import (
    MedianBlur, GaussianBlur, BoxBlur, BinaryThreshold,
    MorphOpen, MorphClose, ForegroundDetection, SuperpixelInterpolation,
    StainNormalizationHE, NucleusDetectionHE, TissueDetectionHE
)
from pathml.utils import RGB_to_GREY


@pytest.fixture
def tileHE():
    """ Example of pathml.Tile object """
    s = openslide.open_slide("tests/testdata/small_HE.svs")
    im_image = s.read_region(level = 0, location = (900, 800), size = (500, 500))
    im_np = np.asarray(im_image)
    im_np_rgb = cv2.cvtColor(im_np, cv2.COLOR_RGBA2RGB)

    mask = np.zeros((im_np_rgb.shape[0], im_np_rgb.shape[1]), dtype = np.uint8)
    center = np.ones((50, 50))
    center_circle = cv2.getStructuringElement(shape = cv2.MORPH_ELLIPSE, ksize = (25, 25))
    center[12:37, 12:37] -= center_circle
    mask[25:75, 25:75] = center

    m = Masks(masks = {"testmask" : mask})
    tile = Tile(image = im_np_rgb, coords = (0, 0), masks = m, slidetype = "HE")
    return tile


@pytest.mark.parametrize('ksize', [3, 7, 21])
@pytest.mark.parametrize('transform', [MedianBlur, BoxBlur])
def test_median_box_blur(tileHE, ksize, transform):
    t = transform(kernel_size = ksize)
    orig_im = tileHE.image
    t.apply(tileHE)
    assert np.array_equal(tileHE.image, t.F(orig_im))


@pytest.mark.parametrize('ksize', [3, 7, 21])
@pytest.mark.parametrize('sigma', [0.1, 3, 0.999])
def test_gaussian_blur(tileHE, ksize, sigma):
    t = GaussianBlur(kernel_size = ksize, sigma = sigma)
    orig_im = tileHE.image
    t.apply(tileHE)
    assert np.array_equal(tileHE.image, t.F(orig_im))


@pytest.mark.parametrize('thresh', [0, 0.5, 200])
@pytest.mark.parametrize('otsu', [True, False])
def test_binary_thresholding(tileHE, thresh, otsu):
    t = BinaryThreshold(use_otsu = otsu, threshold = thresh, mask_name = "testing")
    t.apply(tileHE)
    assert np.array_equal(tileHE.masks["testing"], t.F(RGB_to_GREY(tileHE.image)))


@pytest.mark.parametrize('kernel', [None, np.ones((4, 4), dtype = np.uint8)])
@pytest.mark.parametrize('n_iter', [1, 3, 7, 21])
@pytest.mark.parametrize('ksize', [3, 7, 21])
@pytest.mark.parametrize('transform', [MorphOpen, MorphClose])
def test_open_close(tileHE, transform, ksize, n_iter, kernel):
    t = transform(kernel_size = ksize, n_iterations = n_iter, custom_kernel = kernel, mask_name = "testmask")
    orig_mask = tileHE.masks["testmask"]
    t.apply(tileHE)
    assert np.array_equal(tileHE.masks["testmask"], t.F(orig_mask))


@pytest.mark.parametrize('min_reg_size', [0, 10])
@pytest.mark.parametrize('max_hole_size', [0, 10])
@pytest.mark.parametrize('outer_contours_only', [True, False])
def test_foreground_detection(tileHE, min_reg_size, max_hole_size, outer_contours_only):
    t = ForegroundDetection(min_region_size = min_reg_size, max_hole_size = max_hole_size,
                            outer_contours_only = outer_contours_only, mask_name = "testmask")
    orig_mask = tileHE.masks["testmask"]
    t.apply(tileHE)
    assert np.array_equal(tileHE.masks["testmask"], t.F(orig_mask))


@pytest.mark.parametrize('n_iter', [1, 30])
@pytest.mark.parametrize('region_size', [10, 20])
def test_superpix_interp(tileHE, region_size, n_iter):
    t = SuperpixelInterpolation(region_size = region_size, n_iter = n_iter)
    orig_im = tileHE.image
    t.apply(tileHE)
    assert np.array_equal(tileHE.image, t.F(orig_im))


@pytest.mark.parametrize('target', ["normalize", "hematoxylin", "eosin"])
@pytest.mark.parametrize('method', ["vahadane", "macenko"])
def test_stain_normalization_he(tileHE, method, target):
    t = StainNormalizationHE(target = target, stain_estimation_method = method)
    orig_im = tileHE.image
    t.apply(tileHE)
    assert np.allclose(tileHE.image, t.F(orig_im))


def test_nuc_detectionHE(tileHE):
    t = NucleusDetectionHE(mask_name = "testing")
    orig_im = tileHE.image
    t.apply(tileHE)
    assert np.array_equal(tileHE.masks["testing"], t.F(orig_im))


@pytest.mark.parametrize('use_saturation', [True, False])
@pytest.mark.parametrize('threshold', [None, 100])
def test_tissue_detectionHE(tileHE):
    t = TissueDetectionHE(mask_name = "testing")
    orig_im = tileHE.image
    t.apply(tileHE)
    assert np.array_equal(tileHE.masks["testing"], t.F(orig_im))
