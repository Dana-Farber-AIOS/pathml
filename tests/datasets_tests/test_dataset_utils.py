"""
Copyright 2021, Dana-Farber Cancer Institute and Weill Cornell Medicine
License: GNU GPL 2.0
"""
import importlib.util

import numpy as np
import pytest
import torch.nn as nn
import torch.nn.functional as F
from skimage.draw import ellipse
from skimage.measure import label, regionprops

from pathml.datasets.utils import DeepPatchFeatureExtractor


def requires_torchvision(func):
    """Decorator to skip tests that require torchvision."""
    torchvision_installed = importlib.util.find_spec("torchvision") is not None
    reason = "torchvision is required"
    return pytest.mark.skipif(not torchvision_installed, reason=reason)(func)


class SimpleCNN(nn.Module):
    def __init__(self, input_shape):
        super(SimpleCNN, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels=input_shape[0],
            out_channels=32,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        fc_input_size = (input_shape[1] // 4) * (input_shape[2] // 4) * 64

        self.fc1 = nn.Linear(fc_input_size, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))

        # Flatten the output for the fully connected layer
        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


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


@pytest.mark.parametrize("patch_size", [1, 64, 128])
@pytest.mark.parametrize("entity", ["cell", "tissue"])
@pytest.mark.parametrize("threshold", [0, 0.1, 0.8])
@pytest.mark.parametrize("with_instance_masking", [True, False])
@pytest.mark.parametrize("extraction_layer", [None, "fc1"])
def test_feature_extractor(
    entity, patch_size, threshold, with_instance_masking, extraction_layer
):

    image_size = (256, 256)

    instance_map = make_fake_instance_maps(
        num=20, image_size=image_size, ellipse_height=20, ellipse_width=8
    )
    image = make_fake_image(instance_map.copy())
    regions = regionprops(instance_map)

    model = SimpleCNN(input_shape=(3, 224, 224))

    extractor = DeepPatchFeatureExtractor(
        patch_size=patch_size,
        batch_size=1,
        entity=entity,
        architecture=model,
        fill_value=255,
        resize_size=224,
        threshold=threshold,
        with_instance_masking=with_instance_masking,
        extraction_layer=extraction_layer,
    )
    features = extractor.process(image, instance_map)

    if threshold == 0:
        assert features.shape[0] == len(regions)
    else:
        assert features.shape[0] <= len(regions)


@requires_torchvision
@pytest.mark.parametrize("patch_size", [1, 64, 128])
@pytest.mark.parametrize("entity", ["cell", "tissue"])
@pytest.mark.parametrize("threshold", [0, 0.1, 0.8])
@pytest.mark.parametrize("extraction_layer", [None, "fc"])
def test_feature_extractor_torchvision(entity, patch_size, threshold, extraction_layer):
    # pytest.importorskip("torchvision")

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
        extraction_layer=extraction_layer,
    )
    features = extractor.process(image, instance_map)

    if threshold == 0:
        assert features.shape[0] == len(regions)
    else:
        assert features.shape[0] <= len(regions)


@requires_torchvision
@pytest.mark.parametrize("patch_size", [64, 128])
@pytest.mark.parametrize("entity", ["cell", "tissue"])
@pytest.mark.parametrize("threshold", [0.8])
@pytest.mark.parametrize("extraction_layer", [None, "12"])
def test_feature_extractor_torchvision_no_resnet(
    entity, patch_size, threshold, extraction_layer
):
    # pytest.importorskip("torchvision")

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
        architecture="mobilenet_v3_small",
        fill_value=255,
        resize_size=224,
        threshold=threshold,
        extraction_layer=extraction_layer,
    )
    features = extractor.process(image, instance_map)

    if threshold == 0:
        assert features.shape[0] == len(regions)
    else:
        assert features.shape[0] <= len(regions)
