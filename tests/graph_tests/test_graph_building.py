"""
Copyright 2021, Dana-Farber Cancer Institute and Weill Cornell Medicine
License: GNU GPL 2.0
"""

import numpy as np
import pytest
import torch
from skimage.draw import ellipse
from skimage.measure import regionprops, label

from pathml.graph import KNNGraphBuilder, RAGGraphBuilder

def make_fake_instance_maps(num, image_size, ellipse_height, ellipse_width):
    img = np.zeros(image_size)
    
    # Draw n ellipses
    for i in range(num):
        # Random center for each ellipse
        center_x = np.random.randint(ellipse_width, image_size[1] - ellipse_width)
        center_y = np.random.randint(ellipse_height, image_size[0] - ellipse_height)

        # Coordinates for the ellipse
        rr, cc = ellipse(center_y, center_x, ellipse_height, ellipse_width, shape=image_size)
        
        # Draw the ellipse
        img[rr, cc] = 1

    label_img = label(img.astype(int))
    
    return label_img

@pytest.mark.parametrize("k", [1, 10, 50])
@pytest.mark.parametrize("thresh", [0, 10, 200])
@pytest.mark.parametrize("add_loc_feats", [True, False])
def test_knn_graph_building(k, thresh, add_loc_feats):
    image_size = (1024, 2048)
    
    instance_map = make_fake_instance_maps(num=100, image_size=image_size, ellipse_height=10, ellipse_width=8)
    regions = regionprops(instance_map)

    features = torch.randn(len(regions), 512)

    graph_builder = KNNGraphBuilder(k=k, thresh=thresh, add_loc_feats=add_loc_feats)
    
    graph = graph_builder.process(instance_map, features, target = 1)

    assert graph.node_centroids.shape == (len(regions), 2)
    assert graph.edge_index.shape[0] == 2
    if add_loc_feats:
        assert graph.node_features.shape == (len(regions), 514)
    else:
        assert graph.node_features.shape == (len(regions), 512)


@pytest.mark.parametrize("kernel_size", [1, 3, 10])
@pytest.mark.parametrize("hops", [1, 2, 5])
@pytest.mark.parametrize("add_loc_feats", [True, False])
def test_rag_graph_building(kernel_size, hops, add_loc_feats):
    image_size = (1024, 2048)
    
    instance_map = make_fake_instance_maps(num=100, image_size=image_size, ellipse_height=10, ellipse_width=8)
    regions = regionprops(instance_map)

    features = torch.randn(len(regions), 512)

    graph_builder = RAGGraphBuilder(kernel_size=kernel_size, hops=hops, add_loc_feats=add_loc_feats)
    
    graph = graph_builder.process(instance_map, features, target = 1)

    assert graph.node_centroids.shape == (len(regions), 2)
    assert graph.edge_index.shape[0] == 2
    if add_loc_feats:
        assert graph.node_features.shape == (len(regions), 514)
    else:
        assert graph.node_features.shape == (len(regions), 512)
