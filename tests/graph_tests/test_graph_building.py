"""
Copyright 2021, Dana-Farber Cancer Institute and Weill Cornell Medicine
License: GNU GPL 2.0
"""

import numpy as np
import pytest
import torch
from skimage.draw import ellipse
from skimage.measure import label, regionprops
from sklearn.metrics import pairwise_distances

from pathml.graph import KNNGraphBuilder, RAGGraphBuilder
from pathml.graph.preprocessing import CentroidGraphBuilder


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


@pytest.mark.parametrize("k", [1, 10, 50])
@pytest.mark.parametrize("thresh", [0, 10, 200])
@pytest.mark.parametrize("add_loc_feats", [True, False])
@pytest.mark.parametrize("add_node_labels", [True, False])
def test_knn_graph_building(k, thresh, add_loc_feats, add_node_labels):
    image_size = (1024, 2048)

    instance_map = make_fake_instance_maps(
        num=100, image_size=image_size, ellipse_height=10, ellipse_width=8
    )
    regions = regionprops(instance_map)

    features = torch.randn(len(regions), 512)
    if add_node_labels:
        annotation = torch.randn(len(regions), 4)
    else:
        annotation = None

    graph_builder = KNNGraphBuilder(k=k, thresh=thresh, add_loc_feats=add_loc_feats)

    graph = graph_builder.process(
        instance_map, features, annotation=annotation, target=1
    )

    assert graph.node_centroids.shape == (len(regions), 2)
    assert graph.edge_index.shape[0] == 2
    if add_loc_feats:
        assert graph.node_features.shape == (len(regions), 512 + 2)
    else:
        assert graph.node_features.shape == (len(regions), 512)

    if add_node_labels:
        assert graph.node_labels.shape == (len(regions), 4)


@pytest.mark.parametrize("kernel_size", [1, 3, 10])
@pytest.mark.parametrize("hops", [1, 2, 5])
@pytest.mark.parametrize("add_loc_feats", [True, False])
@pytest.mark.parametrize("add_node_labels", [True, False])
def test_rag_graph_building(kernel_size, hops, add_loc_feats, add_node_labels):
    image_size = (1024, 2048)

    instance_map = make_fake_instance_maps(
        num=100, image_size=image_size, ellipse_height=10, ellipse_width=8
    )
    regions = regionprops(instance_map)

    features = torch.randn(len(regions), 512)
    if add_node_labels:
        annotation = torch.randn(len(regions), 4)
    else:
        annotation = None

    graph_builder = RAGGraphBuilder(
        kernel_size=kernel_size, hops=hops, add_loc_feats=add_loc_feats
    )

    graph = graph_builder.process(
        instance_map, features, annotation=annotation, target=1
    )

    assert graph.node_centroids.shape == (len(regions), 2)
    assert graph.edge_index.shape[0] == 2
    if add_loc_feats:
        assert graph.node_features.shape == (len(regions), 514)
    else:
        assert graph.node_features.shape == (len(regions), 512)

    if add_node_labels:
        assert graph.node_labels.shape == (len(regions), 4)


def test_centroid_graph_builder_initialization():
    centroids = np.array([[0, 0], [1, 1], [2, 2]])
    builder = CentroidGraphBuilder(centroids)
    assert np.array_equal(builder.centroids, centroids)


def test_knn_graph_construction():
    centroids = np.array([[0, 0], [1, 1], [2, 2]])
    builder = CentroidGraphBuilder(centroids)
    knn_graph = builder.build_knn_graph(k=2)
    assert len(knn_graph.nodes) == 3
    assert len(knn_graph.edges) == 3  # This depends on the value of k


def test_mst_graph_construction():
    centroids = np.array([[0, 0], [1, 1], [2, 2]])
    builder = CentroidGraphBuilder(centroids)
    mst_graph = builder.build_knn_mst_graph(k=2)
    assert len(mst_graph.nodes) == 3
    assert len(mst_graph.edges) <= len(builder.build_knn_graph(k=2).edges)


def test_edge_weights():
    centroids = np.array([[0, 0], [1, 1], [2, 2]])
    builder = CentroidGraphBuilder(centroids)
    knn_graph = builder.build_knn_graph(k=2)
    distances = pairwise_distances(centroids)

    # Ensure the indices in the graph correspond to those in the centroids array
    for u, v, data in knn_graph.edges(data=True):
        weight = data["weight"]
        assert np.isclose(weight, distances[int(u)][int(v)], atol=1e-6)
