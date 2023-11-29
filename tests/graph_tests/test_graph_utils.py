"""
Copyright 2021, Dana-Farber Cancer Institute and Weill Cornell Medicine
License: GNU GPL 2.0
"""

import numpy as np
import pytest
from skimage.draw import ellipse
from skimage.measure import label

from pathml.graph import build_assignment_matrix


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


@pytest.mark.parametrize("matrix", [True, False])
def test_build_assignment_matrix(matrix):
    image_size = (1024, 2048)

    tissue_instance_map = make_fake_instance_maps(
        num=20, image_size=image_size, ellipse_height=20, ellipse_width=8
    )
    cell_centroids = np.random.rand(200, 2)

    assignment = build_assignment_matrix(
        cell_centroids, tissue_instance_map, matrix=matrix
    )

    if matrix:
        assert assignment.shape[0] == 200
    else:
        assert assignment.shape[1] == 200
