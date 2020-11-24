import numpy as np
import pytest
import cv2
from matplotlib.testing.decorators import check_figures_equal
from matplotlib.patches import Rectangle

from pathml.preprocessing.tiling import Tile
from pathml.preprocessing.utils import (
    segmentation_lines, contour_centroid, sort_points_clockwise,
    pad_or_crop, _pad_or_crop_1d, upsample_array, plot_mask, plot_extracted_tiles,
    RGB_to_HSV, RGB_to_OD, RGB_to_HSI, RGB_to_GREY, RGB_to_LAB,
    normalize_matrix_cols, normalize_matrix_rows, label_artifact_tile_HE, label_whitespace_HE
)
from pathml.preprocessing.wsi import HESlide


@pytest.fixture(scope = "module")
def random_rgb():
    im = np.random.randint(low = 0, high = 255, size = (50, 50, 3), dtype = np.uint8)
    return im


@pytest.fixture(scope = "module")
def simple_mask():
    im = np.zeros((7, 7), dtype = np.uint8)
    im[2:5, 2:5] = 1
    return im


@pytest.fixture(scope = "module")
def example_contour(simple_mask):
    # contour will be at points (2, 2), (2, 4), (4, 4), (4, 2)
    cnt, _ = cv2.findContours(simple_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return cnt[0].astype(np.uint8)


@pytest.fixture(scope = "module")
def random_5_5_5_5():
    m = np.random.rand(5, 5, 5, 5)
    return m


@pytest.fixture(scope = "module")
def random_50_50():
    m = np.random.rand(50, 50)
    return m


# Tests

@pytest.mark.parametrize("conv", [RGB_to_HSV, RGB_to_OD, RGB_to_HSI, RGB_to_LAB])
def test_color_conversion_shape(conv, random_rgb):
    im_converted = conv(random_rgb)
    assert im_converted.shape == random_rgb.shape


def test_RGB_to_GREY(random_rgb):
    im_converted = RGB_to_GREY(random_rgb)
    assert im_converted.shape[0:2] == random_rgb.shape[0:2]


def test_segmentation_lines(simple_mask):
    x, y = segmentation_lines(simple_mask)
    for x, y in zip(x, y):
        assert (x, y) in [
            (1, 1), (1, 2), (1, 3), (1, 4), (1, 5),
            (2, 1), (2, 5),
            (3, 1), (3, 5),
            (4, 1), (4, 5),
            (5, 1), (5, 2), (5, 3), (5, 4), (5, 5)
        ]


@check_figures_equal(extensions = [".png"])
def test_plot_mask(fig_test, fig_ref):
    # dummy image
    im = np.random.randint(low = 1, high = 254, size = (6, 6, 3), dtype = np.uint8)
    # mask
    mask = np.zeros((6, 6), dtype = np.uint8)
    mask[2:4, 2:4] = 1
    # manual generation of points around mask
    x = [1, 1, 1, 1, 2, 2, 3, 3, 4, 4, 4, 4]
    y = [1, 2, 3, 4, 1, 4, 1, 4, 1, 2, 3, 4]
    # manually plot fig_ref
    fig_ref.gca().imshow(im)
    fig_ref.gca().scatter(x, y, color = "red", marker = ".", s = 1)
    fig_ref.gca().axis('off')
    # plot_mask on fig_test
    plot_mask(im = im, mask_in = mask, ax = fig_test.gca())


@check_figures_equal(extensions = [".png"])
def test_plot_mask_downsample(fig_test, fig_ref):
    # to test downsample by factor of two, we can use same code as above but upscale everything first
    # dummy image
    im = np.random.randint(low = 1, high = 254, size = (12, 12, 3), dtype = np.uint8)
    # mask
    mask = np.zeros((12, 12), dtype = np.uint8)
    mask[4:8, 4:8] = 1
    # manual generation of points around mask
    x = [1, 1, 1, 1, 2, 2, 3, 3, 4, 4, 4, 4]
    y = [1, 2, 3, 4, 1, 4, 1, 4, 1, 2, 3, 4]
    # manually plot fig_ref
    fig_ref.gca().imshow(im[::2, ::2, :])
    fig_ref.gca().scatter(x, y, color = "red", marker = ".", s = 1)
    fig_ref.gca().axis('off')
    # plot_mask on fig_test
    plot_mask(im = im, mask_in = mask, ax = fig_test.gca(), downsample_factor = 2)


def example_slide_data():
    wsi = HESlide(path = "tests/testdata/CMU-1-Small-Region.svs")
    slide_data = wsi.load_data(level = 0, location = (900, 800), size = (100, 100))
    slide_data.tiles = [
        Tile(i = 10, j = 10, array = slide_data.image[10:40, 10:40, :]),
        Tile(i = 50, j = 55, array = slide_data.image[50:80, 55:85, :]),
    ]
    return slide_data


@check_figures_equal(extensions = [".png"])
def test_plot_extracted_tiles(fig_test, fig_ref):
    # not using fixture as arg because check_figures_equal requires there to be only
    #   two arguments, named specifically fig_test and fig_ref
    data = example_slide_data()
    # manual plotting on fig_test
    fig_ref.gca().imshow(data.image)
    for t in data.tiles:
        fig_ref.gca().add_patch(Rectangle(xy = (t.j, t.i), width = 30, height = -30))

    plot_extracted_tiles(data, downsample_factor = 1, ax = fig_test.gca())


def test_upsample_array(simple_mask):
    factor = 11
    upsampled = upsample_array(simple_mask, factor = factor)
    upsampled_kron = np.kron(simple_mask, np.ones((factor, factor), simple_mask.dtype))
    assert np.array_equal(upsampled, upsampled_kron)


def test_sort_points_clockwise():
    points = np.array([
        [1, 1],
        [-1, 1],
        [1, -1],
        [-1, -1]])
    points_sorted_correct = np.array([
        [-1, -1],
        [-1, 1],
        [1, 1],
        [1, -1]])
    points_sorted = sort_points_clockwise(points)
    assert points.shape == points_sorted.shape
    assert np.array_equal(points_sorted, points_sorted_correct)


def test_contour_centroid(example_contour):
    i, j = contour_centroid(example_contour.astype(np.float32))
    assert np.allclose([i, j], [3, 3])


def test_pad_or_crop_1d(random_5_5_5_5):
    padded = _pad_or_crop_1d(random_5_5_5_5, axis = 2, target_dim = 10)
    assert padded.shape == (5, 5, 10, 5)
    cropped = _pad_or_crop_1d(random_5_5_5_5, axis = 1, target_dim = 2)
    assert cropped.shape == (5, 2, 5, 5)
    no_change = _pad_or_crop_1d(random_5_5_5_5, axis = 0, target_dim = 5)
    assert no_change.shape == (5, 5, 5, 5)


@pytest.mark.parametrize("target_shape", [(5, 5, 5, 5), (5, 6, 6, 5), (5, 4, 4, 5), (6, 4, 5, 5), (10, 11, 12, 13)])
def test_pad_or_crop(random_5_5_5_5, target_shape):
    assert pad_or_crop(random_5_5_5_5, target_shape).shape == target_shape, f"target shape: {target_shape}"


def test_normalize_matrix_rows(random_50_50):
    a = normalize_matrix_rows(random_50_50)
    assert np.all(np.isclose(np.linalg.norm(a, axis = 1), 1.))


def test_normalize_matrix_cols(random_50_50):
    a = normalize_matrix_cols(random_50_50)
    assert np.all(np.isclose(np.linalg.norm(a, axis = 0), 1.))


def test_label_artifact_tile(random_rgb):
    a = label_artifact_tile_HE(random_rgb)
    assert a in [True, False]


def test_label_whitespace_tile(random_rgb):
    a = label_whitespace_HE(random_rgb)
    assert a in [True, False]
