"""
Copyright 2021, Dana-Farber Cancer Institute and Weill Cornell Medicine
License: GNU GPL 2.0
"""

import pytest
import torch
from torch_geometric.utils import erdos_renyi_graph

from pathml.graph.utils import HACTPairData
from pathml.ml import HACTNet


def fake_hactnet_inputs():
    """fake batch of input for HACTNet"""
    cell_features = torch.rand(200, 256)
    cell_edge_index = erdos_renyi_graph(200, 0.2, directed=False)
    tissue_features = torch.rand(20, 256)
    tissue_edge_index = erdos_renyi_graph(20, 0.2, directed=False)
    target = torch.tensor([1, 2])
    assignment = torch.randint(low=0, high=20, size=(200,)).long()
    data = HACTPairData(
        x_cell=cell_features,
        edge_index_cell=cell_edge_index,
        x_tissue=tissue_features,
        edge_index_tissue=tissue_edge_index,
        assignment=assignment,
        target=target,
    )
    return data


@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("readout_op", ["lstm", "concat", None])
def test_hovernet_forward_pass(batch_size, readout_op):
    """Make sure that dimensions of outputs are as expected from forward pass"""
    batch = fake_hactnet_inputs()
    batch["x_cell_batch"] = torch.zeros(200).long()
    batch["x_tissue_batch"] = torch.zeros(20).long()
    if batch_size > 1:
        batch["x_cell_batch"][-100:] = 1
        batch["x_tissue_batch"][-10:] = 1

    cell_deg = torch.randint(low=0, high=20000, size=(14,))
    tissue_deg = torch.randint(low=0, high=2000, size=(14,))

    kwargs_pna_cell = {
        "aggregators": ["mean", "max", "min", "std"],
        "scalers": ["identity", "amplification", "attenuation"],
        "deg": cell_deg,
    }
    kwargs_pna_tissue = {
        "aggregators": ["mean", "max", "min", "std"],
        "scalers": ["identity", "amplification", "attenuation"],
        "deg": tissue_deg,
    }

    cell_params = {
        "layer": "PNAConv",
        "in_channels": 256,
        "hidden_channels": 64,
        "num_layers": 3,
        "out_channels": 64,
        "readout_op": readout_op,
        "readout_type": "mean",
        "kwargs": kwargs_pna_cell,
    }

    tissue_params = {
        "layer": "PNAConv",
        "in_channels": 256,
        "hidden_channels": 64,
        "num_layers": 3,
        "out_channels": 64,
        "readout_op": readout_op,
        "readout_type": "mean",
        "kwargs": kwargs_pna_tissue,
    }

    classifier_params = {
        "in_channels": 128,
        "hidden_channels": 128,
        "out_channels": 7,
        "num_layers": 2,
        "norm": "batch_norm" if batch_size > 1 else "layer_norm",
    }

    model = HACTNet(cell_params, tissue_params, classifier_params)

    with torch.no_grad():
        outputs = model(batch)

    assert outputs.shape == (batch_size, 7)
