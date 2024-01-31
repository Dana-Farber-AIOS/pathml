"""
Copyright 2021, Dana-Farber Cancer Institute and Weill Cornell Medicine
License: GNU GPL 2.0
"""

import numpy as np
import pytest
import torch
from skimage.draw import ellipse
from skimage.measure import label
from torch_geometric.loader import DataLoader

import pathml
from pathml.core import SlideData
from pathml.graph import Graph, HACTPairData, build_assignment_matrix
from pathml.graph.utils import get_full_instance_map
from pathml.preprocessing import Pipeline
from pathml.preprocessing.transforms import Transform


@pytest.mark.parametrize("batch_size", [1, 8, 32])
@pytest.mark.parametrize("include_target", [True, False])
def test_pathml_graph(batch_size, include_target):

    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
    node_centroids = torch.randn(3, 2)
    node_features = torch.randn(3, 2)

    if include_target:
        target = torch.tensor([1])

    graph_obj = Graph(
        edge_index=edge_index,
        node_centroids=node_centroids,
        node_features=node_features,
        target=target if include_target else None,
    )
    loader = DataLoader([graph_obj] * batch_size, batch_size=batch_size)
    batch = next(iter(loader))

    assert batch.node_centroids.shape == (batch_size * 3, 2)
    assert batch.node_features.shape == (batch_size * 3, 2)
    assert batch.edge_index.shape == (2, batch_size * 4)
    assert batch.batch.shape == (batch_size * 3,)


@pytest.mark.parametrize("batch_size", [1, 8, 32])
def test_pathml_hactnet_graph(batch_size):

    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
    node_features = torch.randn(3, 2)

    x_cell = node_features
    edge_index_cell = edge_index
    x_tissue = node_features
    edge_index_tissue = edge_index
    assignment = edge_index
    target = torch.tensor([2])

    graph_obj = HACTPairData(
        x_cell=x_cell,
        edge_index_cell=edge_index_cell,
        x_tissue=x_tissue,
        edge_index_tissue=edge_index_tissue,
        assignment=assignment,
        target=target,
    )
    loader = DataLoader([graph_obj] * batch_size, batch_size=batch_size)
    batch = next(iter(loader))

    assert batch.x_cell.shape == (batch_size * 3, 2)
    assert batch.x_tissue.shape == (batch_size * 3, 2)

    assert batch.edge_index_cell.shape == (2, batch_size * 4)
    assert batch.edge_index_tissue.shape == (2, batch_size * 4)


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


class DummyTransform(Transform):
    def __init__(
        self,
        mask_name,
    ):
        self.mask_name = mask_name

    def F(self, image):
        return image

    def apply(self, tile):
        assert isinstance(
            tile, pathml.core.tile.Tile
        ), f"tile is type {type(tile)} but must be pathml.core.tile.Tile"

        nucleus_mask = self.F(tile.image)
        tile.masks[self.mask_name] = nucleus_mask


@pytest.mark.parametrize("mask_name", ["test"])
def test_instance_map(mask_name):
    image_path = "tests/testdata/small_HE.svs"
    wsi = SlideData(image_path, name=image_path, backend="openslide", stain="HE")

    pipeline = Pipeline([DummyTransform(mask_name)])

    wsi.run(
        pipeline,
        overwrite_existing_tiles=True,
        distributed=False,
        tile_pad=True,
        tile_size=1024,
    )

    image_norm, label_instance_map, instance_centroids = get_full_instance_map(
        wsi, patch_size=1024, mask_name="test"
    )

    assert image_norm.shape == (wsi.shape[0], wsi.shape[1], 3)
    assert label_instance_map.shape == (wsi.shape[0], wsi.shape[1])
    assert instance_centroids.shape[1] == 2
