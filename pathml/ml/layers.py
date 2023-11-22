import torch.nn as nn
import importlib
import torch
from torch_geometric.nn.conv import PNAConv, GINConv, GCNConv
from torch_geometric.nn.pool import global_mean_pool


class GNNLayer(nn.Module):
    """
    GNN layer for processing graph structures. 

    Args:
        layer (str): Type of torch_geometric GNN layer to be used. See https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#convolutional-layers for all available options. 
        in_channels (int): Number of input features supplied to the model. 
        hidden_channels (int): Number of hidden channels used in each layer of the GNN model.
        num_layers (int): Number of message-passing layers in the model.
        out_channels (int): Number of output features returned by the model. 
        readout_op (str): Readout operation to summarize features from each layer. Supports 'lstm' and 'concat'.
        readout_type (str): Type of readout to aggregate node embeddings. Supports 'mean'. 
        kwargs (dict): Extra layer-specific arguments. Must have required keyword arguments of layer from https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#convolutional-layers.
    """
    
    def __init__(self, layer, in_channels, hidden_channels, num_layers, out_channels, readout_op, readout_type, kwargs):
        super().__init__()
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.readout_type = readout_type
        self.readout_op = readout_op
        
        conv_module = importlib.import_module('torch_geometric.nn.conv')
        module = getattr(conv_module, layer)

        self.convs.append(module(in_channels, hidden_channels, **kwargs))
        self.batch_norms.append(nn.BatchNorm1d(hidden_channels))
        for _ in range(1, num_layers-1):
            conv = module(hidden_channels, hidden_channels, **kwargs)
            self.convs.append(conv)
            self.batch_norms.append(nn.BatchNorm1d(hidden_channels))
        self.convs.append(module(hidden_channels, out_channels, **kwargs))
        self.batch_norms.append(nn.BatchNorm1d(out_channels))
        
        if readout_op == "lstm":
            self.lstm = nn.LSTM(
                out_channels, (num_layers * out_channels) // 2,
                bidirectional=True,
                batch_first=True)
            self.att = nn.Linear(2 * ((num_layers * out_channels) // 2), 1)
        
    def forward(self, x, edge_index, batch, with_readout=True):
        h = []
        x = x.float()
        for norm, conv in zip(self.batch_norms, self.convs):
            x = conv(x, edge_index)
            x = norm(x)
            h.append(x)
        if self.readout_op == "concat":
            out = torch.cat(h, dim=-1)
        elif self.readout_op == "lstm":
            x = torch.stack(h, dim=1)
            alpha, _ = self.lstm(x)
            alpha = self.att(alpha).squeeze(-1)  
            alpha = torch.softmax(alpha, dim=-1)
            out = (x * alpha.unsqueeze(-1)).sum(dim=1)
        else:
            out = h[-1]
        if with_readout:
            if self.readout_type == 'mean':
                out = global_mean_pool(out, batch)
        return out


