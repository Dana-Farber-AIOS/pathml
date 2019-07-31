"""
Code for creating and working with tiles
"""

import skimage as ski
import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = None


def prep_crop_image(imarray, patch_size):
    """
    function to crop image so that image dimensions evenly divide patch dimensions.

    imarray: numpy array of image
    patch_size: dimension of patches to extract

    returns numpy array of cropped image
    """

    h = imarray.shape[0]
    w = imarray.shape[1]
    hcrop = (h % patch_size)
    wcrop = (w % patch_size)
    # handle odd numbers of pixels to crop
    if hcrop % 2 == 1:
        croptop = ((hcrop - 1) / 2) + 1
        cropbot = ((hcrop - 1) / 2)
    else:
        croptop = cropbot = hcrop / 2
    if wcrop % 2 == 1:
        cropleft = ((wcrop - 1) / 2) + 1
        cropright = ((wcrop - 1) / 2)
    else:
        cropleft = cropright = wcrop / 2
    # crop
    cropdims = ((croptop, cropbot), (cropleft, cropright), (0, 0))
    out = ski.util.crop(imarray, cropdims, copy=True)
    return out


def tile_image(imagepath, patchsize):
    """
    Divides image into non-overlapping patches

    imagepath: path to .jpg image (m, n, 3)
    patchsize: dimensions of output patches

    returns tensor of patches (m/patchsize, n/patchsize, 1, patchsize, patchsize, 3)
    """

    # divide image into non-overlapping tiles (aka blocks)
    im = Image.open(imagepath)
    imarray = np.array(im)
    im_cropped = prep_crop_image(imarray, patchsize)
    # divide into patches
    patches = ski.util.shape.view_as_blocks(im_cropped, block_shape=(patchsize, patchsize, 3))
    return patches


def flatten_patches(patches):
    """
    reshape patches array so that all patches are stacked along a single axis (1st axis)

    patches: tensor of RGB images (m/patchsize, n/patchsize, 1, patchsize, patchsize, 3)

    returns: tensor of RGB images (m*n/patchsize**2, patchsize, patchsize, 3)
    """

    s = patches.shape
    patches_flat = patches.reshape((-1, s[3], s[4], s[5]))
    return patches_flat, s


# noinspection PyPep8Naming,PyPep8Naming,PyPep8Naming
def rgb2hsi_v(patch):
    """
    Convert RGB image to HSI

    patch: (n, n, 3) numpy array of RGB image

    returns (n, n, 3) numpy array of HSI image
    """

    R = patch[:, :, 0]
    G = patch[:, :, 1]
    B = patch[:, :, 2]
    patch_sum = np.sum(patch, axis=2)
    r = R / patch_sum
    g = G / patch_sum
    b = B / patch_sum
    h = np.zeros_like(r, dtype=float)
    # when R=G=B, we need to assign h=0 otherwise we get divide by 0
    h_0 = np.logical_and(R == G, G == B)
    num_h = 0.5 * ((r[~h_0] - g[~h_0]) + (r[~h_0] - b[~h_0]))
    denom_h = (np.sqrt((r[~h_0] - g[~h_0]) ** 2 + (r[~h_0] - b[~h_0]) * (g[~h_0] - b[~h_0])))
    h[~h_0] = np.arccos(num_h / denom_h)
    h[B > G] = 2 * np.pi - h[B > G]
    h = h / (2. * np.pi)
    patch_norm = np.stack([r, g, b], axis=2)
    s = 1 - 3 * np.amin(patch_norm, axis=2)
    patchsum = np.sum(patch, axis=2)
    i = patchsum / (3 * 255)
    out = np.stack([h, s, i], axis=2)
    return out


def label_artifact_tile_v(patch):
    """
    vectorized function to apply criteria from Kothari et al. (2012) for labeling artifacts
    for quality control of tiles

    input: rgb patch

    returns artifact status (1 - contains artifacts; 0 - no artifacts)
    """

    hsi_patch = rgb2hsi_v(patch)
    h = hsi_patch[:, :, 0]
    s = hsi_patch[:, :, 1]
    i = hsi_patch[:, :, 2]
    whitespace = np.logical_and(i >= 0.1, s <= 0.1)
    p1 = np.logical_and(0.4 < h, 0.7 > h)
    p2 = np.logical_and(p1, s > 0.1)
    penmark = np.logical_or(p2, i < 0.1)
    tissue = ~np.logical_or(whitespace, penmark)
    mean_whitespace = np.mean(whitespace)
    mean_pen = np.mean(penmark)
    mean_tissue = np.mean(tissue)
    if (mean_whitespace >= 0.8) or (mean_pen >= 0.05) or (mean_tissue < 0.5):
        return 1
    else:
        return 0
