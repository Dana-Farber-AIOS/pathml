"""
Copyright 2021, Dana-Farber Cancer Institute and Weill Cornell Medicine
License: GNU GPL 2.0
"""

import os
import shutil
import urllib

import cv2
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from matplotlib.colors import TABLEAU_COLORS


def download_from_url(url, download_dir, name=None):
    """
    Download a file from a url to destination directory.
    If the file already exists, does not download.

    Args:
        url (str): Url of file to download
        download_dir (str): Directory where file will be downloaded
        name (str, optional): Name of saved file. If ``None``, uses base name of url argument. Defaults to ``None``.

    See: https://stackoverflow.com/questions/7243750/download-file-from-web-in-python-3
    """
    if name is None:
        name = os.path.basename(url)

    path = os.path.join(download_dir, name)

    if os.path.exists(path):
        return
    else:
        os.makedirs(download_dir, exist_ok=True)

        # Download the file from `url` and save it locally under `file_name`:
        with urllib.request.urlopen(url) as response, open(path, "wb") as out_file:
            shutil.copyfileobj(response, out_file)


def parse_file_size(fs):
    """
    Parse a file size string into bytes.
    """
    units = {"B": 1, "KB": 10 ** 3, "MB": 10 ** 6, "GB": 10 ** 9, "TB": 10 ** 12}
    number, unit = [s.strip() for s in fs.split()]
    return int(float(number) * units[unit.upper()])


def upsample_array(arr, factor):
    """
    Upsample array by a factor. Each element in input array will become a CxC block in the upsampled array, where
    C is the constant upsampling factor. From https://stackoverflow.com/a/32848377

    :param arr: input array to be upsampled
    :type arr: np.ndarray
    :param factor: Upsampling factor
    :type factor: int
    :return: np.ndarray
    """
    r, c = arr.shape  # number of rows/columns
    rs, cs = arr.strides  # row/column strides
    x = np.lib.stride_tricks.as_strided(
        arr, (r, factor, c, factor), (rs, 0, cs, 0)
    )  # view a as larger 4D array
    return x.reshape(r * factor, c * factor)  # create new 2D array


def pil_to_rgb(image_array_pil):
    """
    Convert PIL RGBA Image to numpy RGB array
    """
    image_array_rgba = np.asarray(image_array_pil)
    image_array = cv2.cvtColor(image_array_rgba, cv2.COLOR_RGBA2RGB).astype(np.uint8)
    return image_array


def segmentation_lines(mask_in):
    """
    Generate coords of points bordering segmentations from a given mask.
    Useful for plotting results of tissue detection or other segmentation.
    """
    assert (
        mask_in.dtype == np.uint8
    ), f"Input mask dtype {mask_in.dtype} must be np.uint8"
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(mask_in, kernel)
    diff = np.logical_xor(dilated.astype(bool), mask_in.astype(bool))
    y, x = np.nonzero(diff)
    return x, y


def plot_mask(im, mask_in, ax=None, color="red", downsample_factor=None):
    """
    plot results of segmentation, overlaying on original image_ref

    :param im: Original RGB image_ref
    :type im: np.ndarray
    :param mask_in: Boolean array of segmentation mask, with True values for masked pixels. Must be same shape as im.
    :type mask_in: np.ndarray
    :param ax: Matplotlib axes object to plot on. If None, creates a new plot. Defaults to None.
    :param color: Color to plot outlines of mask. Defaults to "red". Must be recognized by matplotlib.
    :param downsample_factor: Downsample factor for image_ref and mask to speed up plotting for big images
    """
    if downsample_factor:
        mask_in = mask_in[::downsample_factor, ::downsample_factor]
        im = im[::downsample_factor, ::downsample_factor]

    x, y = segmentation_lines(mask_in)
    if ax is None:
        fig, ax = plt.subplots()
    ax.imshow(im)
    ax.scatter(x, y, color=color, marker=".", s=1)
    ax.axis("off")
    return ax


def contour_centroid(contour):
    """
    Return the centroid of a contour, calculated using moments.
    From `OpenCV implementation <https://docs.opencv.org/3.4/d0/d49/tutorial_moments.html>`_

    :param contour: Contour array as returned by cv2.findContours
    :type contour: np.array
    :return: (x, y) coordinates of centroid.
    :rtype: tuple
    """
    # from the docs: "Note that the numpy type for the input array should be either np.int32 or np.float32"
    assert contour.dtype == np.float32
    # get the moments
    mu = cv2.moments(contour)
    # get the centers of mass
    # add 1e-5 to avoid division by zero
    i, j = (mu["m10"] / (mu["m00"] + 1e-5), mu["m01"] / (mu["m00"] + 1e-5))
    return i, j


def sort_points_clockwise(points):
    """
    Sort a list of points into clockwise order around centroid, ordering by angle with centroid and x-axis.
    After sorting, we can pass the points to cv2 as a contour.
    Centroid is defined as center of bounding box around points.

    :param points: Array of points (N x 2)
    :type points: np.ndarray
    :return: Array of points, sorted in order by angle with centroid (N x 2)
    :rtype: np.ndarray

    Return sorted points
    """
    # identify centroid as point in center of box bounding all points
    x, y, w, h = cv2.boundingRect(points)
    centroid = (x + w // 2, y + h // 2)
    # get angle of vector between point and centroid
    diffs = [point - centroid for point in points]
    angles = [np.arctan2(d[0], d[1]) for d in diffs]
    # sort by angle to order points around the circle
    return points[np.argsort(angles)]


def _pad_or_crop_1d(array, axis, target_dim):
    """
    Modify shape of input array at target axis by zero-padding or cropping.

    :param array: Input array
    :type array: np.ndarray
    :param axis: Index of target axis
    :type axis: int
    :param target_dim: target size of specified axis
    :return: np.ndarray
    """
    in_dim = array.shape[axis]
    if in_dim == target_dim:
        # no action needed
        return array
    diff = target_dim - in_dim
    offset = (int(np.floor(abs(diff) / 2)), int(np.ceil(abs(diff) / 2)))
    if diff > 0:
        # pad
        n_pad = [(0, 0)] * array.ndim
        n_pad[axis] = offset
        return np.pad(array, pad_width=n_pad, mode="constant", constant_values=0)
    else:
        # crop
        # need to use slice(none) to access only target dimension
        slc = [slice(None)] * array.ndim
        slc[axis] = slice(offset[0], -offset[1])
        array = array[tuple(slc)]
        return array


def pad_or_crop(array, target_shape):
    """
    Make dimensions of input array match target shape by either zero-padding or cropping each axis.

    :param array: Input array
    :type array: np.ndarray
    :param target_shape: Target shape of output
    :type target_shape: tuple
    :return: Input array cropped/padded to match target_shape
    :rtype: np.ndarray
    """
    if array.shape == target_shape:
        # no need to do anything
        return array

    for axis, target in enumerate(target_shape):
        array = _pad_or_crop_1d(array, axis=axis, target_dim=target)
    return array


def RGB_to_HSI(imarr):
    """
    Convert imarr from RGB to HSI colorspace.

    :param imarr: numpy array of RGB image_ref (m, n, 3)
    :type imarr: np.ndarray
    :return: numpy array of HSI image_ref (m, n, 3)
    :rtype: np.ndarray

    References:
        http://eng.usf.edu/~hady/courses/cap5400/rgb-to-hsi.pdf
    """
    assert imarr.dtype == np.uint8, f"Input image dtype {imarr.dtype} must be np.uint8"
    R = imarr[:, :, 0]
    G = imarr[:, :, 1]
    B = imarr[:, :, 2]
    # add some noise to avoid divide by zero
    eps = 1e-6
    patch_sum = np.sum(imarr, axis=2) + eps
    r = R / patch_sum
    g = G / patch_sum
    b = B / patch_sum
    h = np.zeros_like(r, dtype=np.float32)
    # when R=G=B, we need to assign h=0 otherwise we get divide by 0
    h_0 = np.logical_and(R == G, G == B)
    num_h = 0.5 * ((r[~h_0] - g[~h_0]) + (r[~h_0] - b[~h_0]))
    denom_h = np.sqrt(
        (r[~h_0] - g[~h_0]) ** 2 + (r[~h_0] - b[~h_0]) * (g[~h_0] - b[~h_0])
    )
    h[~h_0] = np.arccos(num_h / denom_h)
    h[B > G] = 2 * np.pi - h[B > G]
    h = h / (2.0 * np.pi)
    patch_norm = np.stack([r, g, b], axis=2)
    s = 1 - 3 * np.amin(patch_norm, axis=2)
    patchsum = np.sum(imarr, axis=2)
    i = patchsum / (3 * 255)
    out = np.stack([h, s, i], axis=2)
    return out


def RGB_to_OD(imarr):
    """
    Convert input image from RGB space to optical density (OD) space.
    `OD = -log(I)`, where I is the input image in RGB space.

    :param imarr: Image array, RGB format
    :type imarr: numpy.ndarray
    :return: Image array, OD format
    :rtype: numpy.ndarray
    """
    assert imarr.dtype == np.uint8, f"Input image dtype {imarr.dtype} must be np.uint8"
    # need to account for possible zero values
    OD = -np.log((imarr.astype(np.float32) + 1) / 255.0)
    return OD


def RGB_to_HSV(imarr):
    """convert image from RGB to HSV"""
    assert imarr.dtype == np.uint8, f"Input image dtype {imarr.dtype} must be np.uint8"
    hsv = cv2.cvtColor(imarr, cv2.COLOR_RGB2HSV)
    return hsv


def RGB_to_LAB(imarr):
    """convert image from RGB to LAB color space"""
    assert imarr.dtype == np.uint8, f"Input image dtype {imarr.dtype} must be np.uint8"
    imarr_float32 = imarr.astype(np.float32) / 255
    lab = cv2.cvtColor(imarr_float32, cv2.COLOR_RGB2Lab)
    return lab


def RGB_to_GREY(imarr):
    """convert image_ref from RGB to HSV"""
    assert imarr.dtype == np.uint8, f"Input image dtype {imarr.dtype} must be np.uint8"
    grey = cv2.cvtColor(imarr, cv2.COLOR_RGB2GRAY)
    return grey


def normalize_matrix_rows(A):
    """
    Normalize the rows of an array.

    :param A: Input array.
    :type A: np.ndarray
    :return: Array with rows normalized.
    :rtype: np.ndarray
    """
    return A / np.linalg.norm(A, axis=1)[:, None]


def normalize_matrix_cols(A):
    """
    Normalize the columns of an array.

    :param A: An array
    :type A: np.ndarray
    :return: Array with columns normalized
    :rtype: np.ndarray
    """
    return A / np.linalg.norm(A, axis=0)[None, :]


def plot_segmentation(ax, masks, palette=None, markersize=5):
    """
    Plot segmentation contours. Supports multi-class masks.

    Args:
        ax: matplotlib axis
        masks (np.ndarray): Mask array of shape (n_masks, H, W). Zeroes are background pixels.
        palette: color palette to use. if None, defaults to matplotlib.colors.TABLEAU_COLORS
        markersize (int): Size of markers used on plot. Defaults to 5
    """
    assert masks.ndim == 3
    n_channels = masks.shape[0]

    if palette is None:
        palette = list(TABLEAU_COLORS.values())

    nucleus_labels = list(np.unique(masks))
    if 0 in nucleus_labels:
        nucleus_labels.remove(0)  # background
    # plot each individual nucleus
    for label in nucleus_labels:
        for i in range(n_channels):
            nuclei_mask = masks[i, ...] == label
            x, y = segmentation_lines(nuclei_mask.astype(np.uint8))
            ax.scatter(x, y, color=palette[i], marker=".", s=markersize)


def _test_log(msg):
    # passes thru message to pathml logger
    # used for testing logging
    logger.info(msg)
