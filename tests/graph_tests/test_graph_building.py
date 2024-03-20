"""
Copyright 2021, Dana-Farber Cancer Institute and Weill Cornell Medicine
License: GNU GPL 2.0
"""

import numpy as np
import pytest
import torch
from skimage.draw import ellipse
from skimage.measure import label, regionprops

from pathml.graph import KNNGraphBuilder, RAGGraphBuilder
from pathml.graph.preprocessing import MSTGraphBuilder


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
@pytest.mark.parametrize("return_networkx", [True, False])
@pytest.mark.parametrize("use_centroids", [True, False])
def test_knn_graph_building(
    k, thresh, add_loc_feats, add_node_labels, return_networkx, use_centroids
):

    if not use_centroids:
        image_size = (1024, 2048)

        instance_map = make_fake_instance_maps(
            num=100, image_size=image_size, ellipse_height=10, ellipse_width=8
        )
        regions = regionprops(instance_map)
        features = torch.randn(len(regions), 512)
        num_nodes = len(regions)

        if add_node_labels:
            annotation = torch.randn(len(regions), 4)
        else:
            annotation = None

        graph_builder = KNNGraphBuilder(
            k=k,
            thresh=thresh,
            add_loc_feats=add_loc_feats,
            return_networkx=return_networkx,
        )

        graph = graph_builder.process(
            instance_map, features, annotation=annotation, target=1
        )

    elif use_centroids:
        centroids = torch.randn(100, 2)
        features = torch.randn(100, 512)
        if add_node_labels:
            annotation = torch.randn(100, 4)
        else:
            annotation = None
        num_nodes = 100

        graph_builder = KNNGraphBuilder(
            k=k,
            thresh=thresh,
            add_loc_feats=add_loc_feats,
            return_networkx=return_networkx,
        )

        graph = graph_builder.process_with_centroids(
            centroids,
            features,
            annotation=annotation,
            image_size=(1000, 1000),
            target=1,
        )

    if return_networkx:
        assert len(graph.nodes) == num_nodes
        if add_loc_feats:
            assert len(graph.nodes[0]["node_features"]) == 514
        else:
            assert len(graph.nodes[0]["node_features"]) == 512
    else:
        assert graph.node_centroids.shape == (num_nodes, 2)
        assert graph.edge_index.shape[0] == 2
        if add_loc_feats:
            assert graph.node_features.shape == (num_nodes, 512 + 2)
        else:
            assert graph.node_features.shape == (num_nodes, 512)

        if add_node_labels:
            assert graph.node_labels.shape == (num_nodes, 4)


@pytest.mark.parametrize("kernel_size", [1, 3, 10])
@pytest.mark.parametrize("hops", [1, 2, 5])
@pytest.mark.parametrize("add_loc_feats", [True, False])
@pytest.mark.parametrize("add_node_labels", [True, False])
@pytest.mark.parametrize("return_networkx", [True, False])
def test_rag_graph_building(
    kernel_size, hops, add_loc_feats, add_node_labels, return_networkx
):
    image_size = (1024, 2048)

    instance_map = make_fake_instance_maps(
        num=100, image_size=image_size, ellipse_height=10, ellipse_width=8
    )
    regions = regionprops(instance_map)
    num_nodes = len(regions)
    features = torch.randn(len(regions), 512)
    if add_node_labels:
        annotation = torch.randn(len(regions), 4)
    else:
        annotation = None

    graph_builder = RAGGraphBuilder(
        kernel_size=kernel_size,
        hops=hops,
        add_loc_feats=add_loc_feats,
        return_networkx=return_networkx,
    )

    graph = graph_builder.process(
        instance_map, features, annotation=annotation, target=1
    )

    if return_networkx:
        assert len(graph.nodes) == num_nodes
        if add_loc_feats:
            assert len(graph.nodes[0]["node_features"]) == 514
        else:
            assert len(graph.nodes[0]["node_features"]) == 512
    else:
        assert graph.node_centroids.shape == (num_nodes, 2)
        assert graph.edge_index.shape[0] == 2
        if add_loc_feats:
            assert graph.node_features.shape == (num_nodes, 512 + 2)
        else:
            assert graph.node_features.shape == (num_nodes, 512)

        if add_node_labels:
            assert graph.node_labels.shape == (num_nodes, 4)


@pytest.mark.parametrize("k", [1, 10, 50])
@pytest.mark.parametrize("thresh", [0, 10, 200])
@pytest.mark.parametrize("add_loc_feats", [True, False])
@pytest.mark.parametrize("add_node_labels", [True, False])
@pytest.mark.parametrize("return_networkx", [True, False])
@pytest.mark.parametrize("use_centroids", [True, False])
def test_mst_graph_building(
    k, thresh, add_loc_feats, add_node_labels, return_networkx, use_centroids
):

    if not use_centroids:
        image_size = (1024, 2048)

        instance_map = make_fake_instance_maps(
            num=100, image_size=image_size, ellipse_height=10, ellipse_width=8
        )
        regions = regionprops(instance_map)
        features = torch.randn(len(regions), 512)
        num_nodes = len(regions)

        if add_node_labels:
            annotation = torch.randn(len(regions), 4)
        else:
            annotation = None

        graph_builder = MSTGraphBuilder(
            k=k,
            thresh=thresh,
            add_loc_feats=add_loc_feats,
            return_networkx=return_networkx,
        )

        graph = graph_builder.process(
            instance_map, features, annotation=annotation, target=1
        )

    elif use_centroids:
        centroids = torch.randn(100, 2)
        features = torch.randn(100, 512)
        if add_node_labels:
            annotation = torch.randn(100, 4)
        else:
            annotation = None
        num_nodes = 100

        graph_builder = KNNGraphBuilder(
            k=k,
            thresh=thresh,
            add_loc_feats=add_loc_feats,
            return_networkx=return_networkx,
        )

        graph = graph_builder.process_with_centroids(
            centroids,
            features,
            annotation=annotation,
            image_size=(1000, 1000),
            target=1,
        )

    if return_networkx:
        assert len(graph.nodes) == num_nodes
        if add_loc_feats:
            assert len(graph.nodes[0]["node_features"]) == 514
        else:
            assert len(graph.nodes[0]["node_features"]) == 512
    else:
        assert graph.node_centroids.shape == (num_nodes, 2)
        assert graph.edge_index.shape[0] == 2
        if add_loc_feats:
            assert graph.node_features.shape == (num_nodes, 512 + 2)
        else:
            assert graph.node_features.shape == (num_nodes, 512)

        if add_node_labels:
            assert graph.node_labels.shape == (num_nodes, 4)
