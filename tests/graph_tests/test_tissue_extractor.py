"""
Copyright 2021, Dana-Farber Cancer Institute and Weill Cornell Medicine
License: GNU GPL 2.0
"""

import numpy as np
import pytest
from skimage.draw import ellipse
from skimage.measure import label, regionprops

from pathml.datasets.utils import DeepPatchFeatureExtractor


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
        np.random.rand(instance_map.shape[0], instance_map.shape[1], 3) * image * 255
    )

    return noised_image.astype("uint8")


@pytest.mark.parametrize("patch_size", [1, 64, 128])
@pytest.mark.parametrize("entity", ["cell", "tissue"])
@pytest.mark.parametrize("threshold", [0, 0.05])
def test_feature_extractor(entity, patch_size, threshold):
    image_size = (256, 256)

    instance_map = make_fake_instance_maps(
        num=20, image_size=image_size, ellipse_height=20, ellipse_width=8
    )
    image = make_fake_image(instance_map.copy())
    regions = regionprops(instance_map)

    extractor = DeepPatchFeatureExtractor(
        patch_size=patch_size,
        batch_size=1,
        entity=entity,
        architecture="resnet34",
        fill_value=255,
        resize_size=224,
        threshold=threshold,
    )
    features = extractor.process(image, instance_map)

    if threshold == 0:
        assert features.shape == (len(regions), 512)
    else:
        assert features.shape[0] <= len(regions)
        assert features.shape[1] == 512
