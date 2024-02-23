"""
Copyright 2021, Dana-Farber Cancer Institute and Weill Cornell Medicine
License: GNU GPL 2.0
"""

import torch
import torch.nn as nn
from torch_geometric.nn.models import MLP

from pathml.ml.layers import GNNLayer
from pathml.ml.utils import scatter_sum


class HACTNet(nn.Module):
    """
    Hierarchical cell-to-tissue model for supervised prediction using cell and tissue graphs.

    Args:
        cell_params (dict): Dictionary containing parameters for cell graph GNN.
        tissue_params (dict): Dictionary containing parameters for tissue graph GNN.
        classifier_params (dict): Dictionary containing parameters for prediction MLP.

    References:
        Pati, P., Jaume, G., Foncubierta-Rodriguez, A., Feroce, F., Anniciello, A.M., Scognamiglio, G., Brancati, N., Fiche, M.,
        Dubruc, E., Riccio, D. and Di Bonito, M., 2022.
        Hierarchical graph representations in digital pathology.
        Medical image analysis, 75, p.102264.
    """

    def __init__(self, cell_params, tissue_params, classifier_params):
        super().__init__()

        # Get cell and tissue graph readout operations
        self.cell_readout_op = cell_params["readout_op"]
        self.tissue_readout_op = tissue_params["readout_op"]

        # Modify tissue GNN parameters
        if self.cell_readout_op == "concat":
            tissue_params["in_channels"] = (
                tissue_params["in_channels"]
                + cell_params["out_channels"] * cell_params["num_layers"]
            )
        else:
            tissue_params["in_channels"] = (
                tissue_params["in_channels"] + cell_params["out_channels"]
            )

        # Main GNN model for cell and tissue graphs
        self.cell_gnn = GNNLayer(**cell_params)
        self.tissue_gnn = GNNLayer(**tissue_params)

        # Modify classifier parameters
        if self.tissue_readout_op == "concat":
            classifier_params["in_channels"] = (
                tissue_params["out_channels"] * tissue_params["num_layers"]
            )
        else:
            classifier_params["in_channels"] = tissue_params["out_channels"]

        # Main classifier head
        self.classifier = MLP(**classifier_params)

    def forward(self, batch):

        x_cell = batch.x_cell
        x_tissue = batch.x_tissue

        z_cell = self.cell_gnn(
            x_cell, batch.edge_index_cell, batch.x_cell_batch, with_readout=False
        )

        out = torch.zeros(
            (x_tissue.shape[0], z_cell.shape[1]),
            dtype=z_cell.dtype,
            device=z_cell.device,
        )

        z_cell_to_tissue = scatter_sum(z_cell, batch.assignment, dim=0, out=out)
        x_tissue = torch.cat((z_cell_to_tissue, x_tissue), dim=1)

        z_tissue = self.tissue_gnn(
            x_tissue, batch.edge_index_tissue, batch.x_tissue_batch
        )
        out = self.classifier(z_tissue)
        return out
