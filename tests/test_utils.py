"""
Copyright 2021, Dana-Farber Cancer Institute and Weill Cornell Medicine
License: GNU GPL 2.0
"""

import numpy as np
import pytest
import cv2
from matplotlib.testing.decorators import check_figures_equal

from pathml.utils import (
    segmentation_lines,
    contour_centroid,
    sort_points_clockwise,
    pad_or_crop,
    _pad_or_crop_1d,
    upsample_array,
    plot_mask,
    RGB_to_HSV,
    RGB_to_OD,
    RGB_to_HSI,
    RGB_to_GREY,
    RGB_to_LAB,
    normalize_matrix_cols,
    normalize_matrix_rows,
    parse_file_size,
    download_from_url
)

@pytest.mark.parametrize(
    "test_input,expected", [("10 gb", 10 ** 10), ("1.17 mB", 1.17e6), ("0.89 KB", 890)]
)
def test_parse_file_sizes(test_input, expected):
    assert parse_file_size(test_input) == expected


def test_download_from_url(tmp_path):
    url = "http://openslide.cs.cmu.edu/download/openslide-testdata/Aperio/index.yaml"
    d = tmp_path / "test"
    download_from_url(url=url, download_dir=d, name="testfile")
    file1 = open(d / "testfile", "r")
    assert file1.readline() == "format: Aperio SVS\n"


@pytest.fixture(scope="module")
def random_rgb():
    im = np.random.randint(low=0, high=255, size=(50, 50, 3), dtype=np.uint8)
    return im


@pytest.fixture(scope="module")
def simple_mask():
    im = np.zeros((7, 7), dtype=np.uint8)
    im[2:5, 2:5] = 1
    return im


@pytest.fixture(scope="module")
def example_contour(simple_mask):
    # contour will be at points (2, 2), (2, 4), (4, 4), (4, 2)
    cnt, _ = cv2.findContours(simple_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return cnt[0].astype(np.uint8)


@pytest.fixture(scope="module")
def random_5_5_5_5():
    m = np.random.rand(5, 5, 5, 5)
    return m


@pytest.fixture(scope="module")
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
            (1, 1),
            (1, 2),
            (1, 3),
            (1, 4),
            (1, 5),
            (2, 1),
            (2, 5),
            (3, 1),
            (3, 5),
            (4, 1),
            (4, 5),
            (5, 1),
            (5, 2),
            (5, 3),
            (5, 4),
            (5, 5),
        ]


@check_figures_equal(extensions=[".png"])
def test_plot_mask(fig_test, fig_ref):
    # dummy image
    im = np.random.randint(low=1, high=254, size=(6, 6, 3), dtype=np.uint8)
    # mask
    mask = np.zeros((6, 6), dtype=np.uint8)
    mask[2:4, 2:4] = 1
    # manual generation of points around mask
    x = [1, 1, 1, 1, 2, 2, 3, 3, 4, 4, 4, 4]
    y = [1, 2, 3, 4, 1, 4, 1, 4, 1, 2, 3, 4]
    # manually plot fig_ref
    fig_ref.gca().imshow(im)
    fig_ref.gca().scatter(x, y, color="red", marker=".", s=1)
    fig_ref.gca().axis("off")
    # plot_mask on fig_test
    plot_mask(im=im, mask_in=mask, ax=fig_test.gca())


@check_figures_equal(extensions=[".png"])
def test_plot_mask_downsample(fig_test, fig_ref):
    # to test downsample by factor of two, we can use same code as above but upscale everything first
    # dummy image
    im = np.random.randint(low=1, high=254, size=(12, 12, 3), dtype=np.uint8)
    # mask
    mask = np.zeros((12, 12), dtype=np.uint8)
    mask[4:8, 4:8] = 1
    # manual generation of points around mask
    x = [1, 1, 1, 1, 2, 2, 3, 3, 4, 4, 4, 4]
    y = [1, 2, 3, 4, 1, 4, 1, 4, 1, 2, 3, 4]
    # manually plot fig_ref
    fig_ref.gca().imshow(im[::2, ::2, :])
    fig_ref.gca().scatter(x, y, color="red", marker=".", s=1)
    fig_ref.gca().axis("off")
    # plot_mask on fig_test
    plot_mask(im=im, mask_in=mask, ax=fig_test.gca(), downsample_factor=2)


def test_upsample_array(simple_mask):
    factor = 11
    upsampled = upsample_array(simple_mask, factor=factor)
    upsampled_kron = np.kron(simple_mask, np.ones((factor, factor), simple_mask.dtype))
    assert np.array_equal(upsampled, upsampled_kron)


def test_sort_points_clockwise():
    points = np.array([[1, 1], [-1, 1], [1, -1], [-1, -1]])
    points_sorted_correct = np.array([[-1, -1], [-1, 1], [1, 1], [1, -1]])
    points_sorted = sort_points_clockwise(points)
    assert points.shape == points_sorted.shape
    assert np.array_equal(points_sorted, points_sorted_correct)


def test_contour_centroid(example_contour):
    i, j = contour_centroid(example_contour.astype(np.float32))
    assert np.allclose([i, j], [3, 3])


def test_pad_or_crop_1d(random_5_5_5_5):
    padded = _pad_or_crop_1d(random_5_5_5_5, axis=2, target_dim=10)
    assert padded.shape == (5, 5, 10, 5)
    cropped = _pad_or_crop_1d(random_5_5_5_5, axis=1, target_dim=2)
    assert cropped.shape == (5, 2, 5, 5)
    no_change = _pad_or_crop_1d(random_5_5_5_5, axis=0, target_dim=5)
    assert no_change.shape == (5, 5, 5, 5)


@pytest.mark.parametrize(
    "target_shape",
    [(5, 5, 5, 5), (5, 6, 6, 5), (5, 4, 4, 5), (6, 4, 5, 5), (10, 11, 12, 13)],
)
def test_pad_or_crop(random_5_5_5_5, target_shape):
    assert (
        pad_or_crop(random_5_5_5_5, target_shape).shape == target_shape
    ), f"target shape: {target_shape}"


def test_normalize_matrix_rows(random_50_50):
    a = normalize_matrix_rows(random_50_50)
    assert np.all(np.isclose(np.linalg.norm(a, axis=1), 1.0))


def test_normalize_matrix_cols(random_50_50):
    a = normalize_matrix_cols(random_50_50)
    assert np.all(np.isclose(np.linalg.norm(a, axis=0), 1.0))
