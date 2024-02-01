"""
Copyright 2021, Dana-Farber Cancer Institute and Weill Cornell Medicine
License: GNU GPL 2.0
"""

import os
import shutil
import tempfile
from pathlib import Path
from unittest.mock import mock_open, patch

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.testing.decorators import check_figures_equal

from pathml.utils import (
    RGB_to_GREY,
    RGB_to_HSI,
    RGB_to_HSV,
    RGB_to_LAB,
    RGB_to_OD,
    _pad_or_crop_1d,
    contour_centroid,
    download_from_url,
    find_qupath_home,
    normalize_matrix_cols,
    normalize_matrix_rows,
    pad_or_crop,
    parse_file_size,
    plot_mask,
    plot_segmentation,
    segmentation_lines,
    setup_qupath,
    sort_points_clockwise,
    upsample_array,
)


@pytest.mark.parametrize(
    "test_input,expected", [("10 gb", 10**10), ("1.17 mB", 1.17e6), ("0.89 KB", 890)]
)
def test_parse_file_sizes(test_input, expected):
    assert parse_file_size(test_input) == expected


def test_download_from_url(tmp_path):
    url = "http://openslide.cs.cmu.edu/download/openslide-testdata/Aperio/index.yaml"
    d = tmp_path / "test"
    download_from_url(url=url, download_dir=d, name="testfile")
    file1 = open(d / "testfile", "r")
    assert file1.readline() == "format: Aperio SVS\n"


# Test successful download
def test_successful_download(tmp_path):
    url = "http://example.com/testfile.txt"
    download_dir = tmp_path / "downloads"
    file_content = b"Sample file content"
    with patch(
        "urllib.request.urlopen", mock_open(read_data=file_content)
    ) as mocked_url_open, patch("builtins.open", mock_open()) as mocked_file:
        download_from_url(url, download_dir, "downloaded_file.txt")

        mocked_url_open.assert_called_with(url)
        mocked_file.assert_called_with(
            os.path.join(download_dir, "downloaded_file.txt"), "wb"
        )


# Test skipping download for existing file
def test_skip_existing_download(tmp_path):
    url = "http://example.com/testfile.txt"
    download_dir = tmp_path / "downloads"
    download_dir.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
    existing_file = download_dir / "existing_file.txt"
    existing_file.touch()  # Create an empty file

    with patch("urllib.request.urlopen", mock_open()) as mocked_url_open:
        download_from_url(url, download_dir, "existing_file.txt")

        mocked_url_open.assert_not_called()


# Test download with default filename
def test_download_default_filename(tmp_path):
    url = "http://example.com/testfile.txt"
    download_dir = tmp_path / "downloads"
    file_content = b"Sample file content for default"

    with patch(
        "urllib.request.urlopen", mock_open(read_data=file_content)
    ) as mocked_url_open, patch("builtins.open", mock_open()) as mocked_file:
        download_from_url(url, download_dir)

        mocked_url_open.assert_called_with(url)
        mocked_file.assert_called_with(os.path.join(download_dir, "testfile.txt"), "wb")


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


def test_plot_segmentation():
    ax = plt.gca()

    masks = np.zeros((3, 12, 12), dtype=np.uint8)
    masks[0, 4:8, 4:8] = 1
    masks[1, 1:3, 1:3] = 2
    masks[2, 8:11, 8:11] = 3

    palette = None
    markersize = 5

    plot_segmentation(ax, masks, palette, markersize)


def test_find_existing_qupath_home(tmp_path):
    # Create a mock QuPath directory structure
    qupath_dir = tmp_path / "qupath"
    qupath_dir.mkdir(parents=True)
    qupath_jar = qupath_dir / "qupath.jar"
    qupath_jar.touch()

    # Test if the function finds the QuPath home correctly
    qupath_home = find_qupath_home(str(tmp_path))
    assert qupath_home == str(qupath_dir.parent.parent)


def test_no_qupath_home_found(tmp_path):
    # Test with a directory without QuPath JAR
    qupath_home = find_qupath_home(str(tmp_path))
    assert qupath_home is None


def test_find_qupath_home():
    # Create a temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        # Case 1: QuPath jar is present
        os.makedirs(Path(temp_dir) / "qupath")
        open(Path(temp_dir) / "qupath/qupath.jar", "a").close()
        assert find_qupath_home(temp_dir) is not None

        # Cleanup
        shutil.rmtree(Path(temp_dir) / "qupath")

        # Case 2: QuPath jar is not present
        assert find_qupath_home(temp_dir) is None


@patch("builtins.print")  # To suppress print statements in the test
def test_setup_qupath(mock_print):
    with tempfile.TemporaryDirectory() as temp_dir:
        qupath_home = Path(temp_dir) / "qupath"

        # Simulate the environment before QuPath installation
        expected_path = (
            qupath_home / "QuPath"
        )  # Update according to the actual behavior of setup_qupath
        assert setup_qupath(str(qupath_home)) == str(expected_path)
        print(setup_qupath(str(qupath_home)))
        print(str(expected_path))
        # Test when QuPath is already installed
        assert setup_qupath(str(qupath_home)) == str(expected_path)
