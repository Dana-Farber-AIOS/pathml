"""
Copyright 2021, Dana-Farber Cancer Institute and Weill Cornell Medicine
License: GNU GPL 2.0
"""

import networkx as nx
import pytest

from pathml.graph.preprocessing import GraphFeatureExtractor


@pytest.mark.parametrize("use_weight", [True, False])
@pytest.mark.parametrize("alpha", [0, 0.5, 0.95])
def test_graph_feature_extractor(use_weight, alpha):

    # Creating a simple graph
    G = nx.DiGraph()

    # Adding nodes
    G.add_weighted_edges_from([(1, 2, 1), (2, 3, 1), (3, 4, 1), (4, 5, 1), (5, 1, 1)])

    extractor = GraphFeatureExtractor(use_weight=use_weight, alpha=alpha)
    features = extractor.process(G)

    assert features
