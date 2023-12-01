"""
Copyright 2021, Dana-Farber Cancer Institute and Weill Cornell Medicine
License: GNU GPL 2.0
"""

import numpy as np
import pytest
from skimage.draw import ellipse
from skimage.measure import label

from pathml.graph import ColorMergedSuperpixelExtractor
from pathml.graph.preprocessing import SLICSuperpixelExtractor


def make_fake_instance_maps(num, image_size, ellipse_height, ellipse_width):
    img = np.zeros(image_size)

    # Draw n ellipses
    for i in range(num):
        # Random center for each ellipse
        center_x = np.random.randint(ellipse_width, image_size[1] - ellipse_width)
        center_y = np.random.randint(ellipse_height, image_size[0] - ellipse_height)

        # Coordinates for the ellipse
        rr, cc = ellipse(
            center_y, center_x, ellipse_height, ellipse_width, shape=image_size
        )

        # Draw the ellipse
        img[rr, cc] = 1

    label_img = label(img.astype(int))

    return label_img


def make_fake_image(instance_map):
    image = instance_map[:, :, None]
    image[image > 0] = 1
    noised_image = (
        np.random.rand(instance_map.shape[0], instance_map.shape[1], 3) * 0.15 + image
    ) * 255

    return noised_image.astype("uint8")


@pytest.mark.parametrize("superpixel_size", [20, 200])
@pytest.mark.parametrize("compactness", [50, 100])
@pytest.mark.parametrize("blur_kernel_size", [0.2, 1])
@pytest.mark.parametrize("threshold", [0.1, 0.9])
@pytest.mark.parametrize("downsampling_factor", [4, 10])
@pytest.mark.parametrize(
    "extractor", [ColorMergedSuperpixelExtractor, SLICSuperpixelExtractor]
)
def test_tissue_extractors(
    superpixel_size,
    compactness,
    blur_kernel_size,
    threshold,
    downsampling_factor,
    extractor,
):
    image_size = (256, 256)

    instance_map = make_fake_instance_maps(
        num=30, image_size=image_size, ellipse_height=20, ellipse_width=8
    )
    image = make_fake_image(instance_map.copy())

    tissue_detector = extractor(
        superpixel_size=superpixel_size,
        compactness=compactness,
        blur_kernel_size=blur_kernel_size,
        threshold=threshold,
        downsampling_factor=downsampling_factor,
    )

    superpixels = tissue_detector.process(image)

    if isinstance(superpixels, tuple):
        superpixels = superpixels[0]

    assert superpixels.shape == image_size
